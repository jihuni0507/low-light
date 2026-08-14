import os
import struct
import numpy as np
import collections

# Structures to store data
CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "track", "point2D_idxs"])

CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=10),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
}
CAMERA_MODEL_IDS = {camera_model.model_id: camera_model for camera_model in CAMERA_MODELS}
CAMERA_MODEL_NAMES = {camera_model.model_name: camera_model for camera_model in CAMERA_MODELS}


def qvec2rotmat(qvec):
    qvec = np.asarray(qvec, dtype=np.float64)
    w, x, y, z = qvec
    return np.array([
        [1 - 2 * (y**2 + z**2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x**2 + y**2)],
    ], dtype=np.float64)


def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]],
        dtype=np.float64,
    ) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)


class ColmapLoader:
    def __init__(self, colmap_dir):
        self.colmap_dir = colmap_dir
        self.cameras = {}
        self.images = {}
        self.points3d = {}
        self._load_data()

    def _load_data(self):
        camera_bin = os.path.join(self.colmap_dir, "cameras.bin")
        image_bin = os.path.join(self.colmap_dir, "images.bin")
        points_bin = os.path.join(self.colmap_dir, "points3d.bin")

        if os.path.exists(camera_bin) and os.path.exists(image_bin) and os.path.exists(points_bin):
            self.cameras = self.read_cameras_binary(camera_bin)
            self.images = self.read_images_binary(image_bin)
            self.points3d = self.read_points3d_binary(points_bin)
            return

        self.cameras = self.read_cameras_text(os.path.join(self.colmap_dir, "cameras.txt"))
        self.images = self.read_images_text(os.path.join(self.colmap_dir, "images.txt"))
        self.points3d = self.read_points3d_text(os.path.join(self.colmap_dir, "points3d.txt"))

    def get_camera_poses(self):
        pose_data = []
        for img_id, img_data in self.images.items():
            R = qvec2rotmat(img_data.qvec)
            T = np.asarray(img_data.tvec, dtype=np.float64)
            cam_data = self.cameras[img_data.camera_id]
            pose_data.append({
                "id": img_id,
                "image_name": img_data.name,
                "R": R,
                "T": T,
                "width": cam_data.width,
                "height": cam_data.height,
                "params": np.asarray(cam_data.params, dtype=np.float64),
            })
        return pose_data

    def get_initial_point_cloud(self):
        xyz, rgb = [], []
        for point_id, point_data in self.points3d.items():
            xyz.append(point_data.xyz)
            rgb.append(point_data.rgb)
        if len(xyz) == 0:
            return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.uint8)
        return np.asarray(xyz, dtype=np.float64), np.asarray(rgb, dtype=np.uint8)

    @staticmethod
    def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
        data = fid.read(num_bytes)
        if len(data) != num_bytes:
            raise ValueError(f"Unexpected end of file while reading {num_bytes} bytes.")
        return struct.unpack(endian_character + format_char_sequence, data)

    def read_cameras_binary(self, path_to_model_file):
        cameras = {}
        with open(path_to_model_file, "rb") as fid:
            num_cameras = self.read_next_bytes(fid, 8, "Q")[0]
            for _ in range(num_cameras):
                camera_id, model_id = self.read_next_bytes(fid, 8, "i i")
                width = self.read_next_bytes(fid, 8, "Q")[0]
                height = self.read_next_bytes(fid, 8, "Q")[0]

                model = CAMERA_MODEL_IDS.get(model_id)
                if model is None:
                    raise NotImplementedError(f"Unsupported camera model id: {model_id}")

                params = self.read_next_bytes(fid, 8 * model.num_params, "d" * model.num_params)
                cameras[camera_id] = Camera(
                    id=camera_id,
                    model=model.model_name,
                    width=width,
                    height=height,
                    params=np.asarray(params, dtype=np.float64),
                )
        return cameras

    def read_images_binary(self, path_to_model_file):
        images = {}
        with open(path_to_model_file, "rb") as fid:
            num_reg_images = self.read_next_bytes(fid, 8, "Q")[0]
            for _ in range(num_reg_images):
                image_id = self.read_next_bytes(fid, 8, "Q")[0]
                qvec = np.asarray(self.read_next_bytes(fid, 32, "dddd"), dtype=np.float64)
                tvec = np.asarray(self.read_next_bytes(fid, 24, "ddd"), dtype=np.float64)
                camera_id = self.read_next_bytes(fid, 4, "i")[0]

                name_chars = []
                while True:
                    char = fid.read(1)
                    if not char or char == b"\x00":
                        break
                    name_chars.append(char.decode("utf-8"))
                image_name = "".join(name_chars)

                num_points2D = self.read_next_bytes(fid, 8, "Q")[0]
                xys = []
                point3D_ids = []
                for _ in range(num_points2D):
                    x, y = self.read_next_bytes(fid, 16, "dd")
                    point3D_id = self.read_next_bytes(fid, 8, "Q")[0]
                    xys.append((float(x), float(y)))
                    point3D_ids.append(point3D_id)

                images[image_id] = Image(
                    id=image_id,
                    qvec=qvec,
                    tvec=tvec,
                    camera_id=camera_id,
                    name=image_name,
                    xys=np.asarray(xys, dtype=np.float64) if xys else np.empty((0, 2), dtype=np.float64),
                    point3D_ids=np.asarray(point3D_ids, dtype=np.int64) if point3D_ids else np.empty((0,), dtype=np.int64),
                )
        return images

    def read_points3d_binary(self, path_to_model_file):
        points3d = {}
        with open(path_to_model_file, "rb") as fid:
            num_points = self.read_next_bytes(fid, 8, "Q")[0]
            for _ in range(num_points):
                point3D_id = self.read_next_bytes(fid, 8, "Q")[0]
                xyz = np.asarray(self.read_next_bytes(fid, 24, "ddd"), dtype=np.float64)
                rgb = np.asarray(self.read_next_bytes(fid, 3, "BBB"), dtype=np.uint8)
                error = self.read_next_bytes(fid, 8, "d")[0]
                track_length = self.read_next_bytes(fid, 8, "Q")[0]

                track = []
                point2D_idxs = []
                for _ in range(track_length):
                    image_id = self.read_next_bytes(fid, 8, "Q")[0]
                    point2D_idx = self.read_next_bytes(fid, 8, "Q")[0]
                    track.append((image_id, point2D_idx))
                    point2D_idxs.append(point2D_idx)

                points3d[point3D_id] = Point3D(
                    id=point3D_id,
                    xyz=xyz,
                    rgb=rgb,
                    error=error,
                    track=track,
                    point2D_idxs=np.asarray(point2D_idxs, dtype=np.int64) if point2D_idxs else np.empty((0,), dtype=np.int64),
                )
        return points3d

    def read_cameras_text(self, path_to_model_file):
        cameras = {}
        with open(path_to_model_file, "r", encoding="utf-8") as fid:
            for line in fid:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue
                camera_id = int(parts[0])
                model_name = parts[1]
                width = int(parts[2])
                height = int(parts[3])
                params = np.asarray([float(v) for v in parts[4:]], dtype=np.float64)
                cameras[camera_id] = Camera(
                    id=camera_id,
                    model=model_name,
                    width=width,
                    height=height,
                    params=params,
                )
        return cameras

    def read_images_text(self, path_to_model_file):
        images = {}
        with open(path_to_model_file, "r", encoding="utf-8") as fid:
            lines = [line.strip() for line in fid if line.strip() and not line.startswith("#")]

        idx = 0
        while idx < len(lines):
            parts = lines[idx].split()
            if len(parts) < 10:
                idx += 1
                continue

            image_id = int(parts[0])
            qvec = np.asarray(parts[1:5], dtype=np.float64)
            tvec = np.asarray(parts[5:8], dtype=np.float64)
            camera_id = int(parts[8])
            image_name = parts[9]

            xys = []
            point3D_ids = []
            idx += 1
            while idx < len(lines):
                next_parts = lines[idx].split()
                if len(next_parts) != 3:
                    break
                xys.append([float(next_parts[0]), float(next_parts[1])])
                point3D_ids.append(int(next_parts[2]))
                idx += 1

            images[image_id] = Image(
                id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=image_name,
                xys=np.asarray(xys, dtype=np.float64) if xys else np.empty((0, 2), dtype=np.float64),
                point3D_ids=np.asarray(point3D_ids, dtype=np.int64) if point3D_ids else np.empty((0,), dtype=np.int64),
            )
        return images

    def read_points3d_text(self, path_to_model_file):
        points3d = {}
        with open(path_to_model_file, "r", encoding="utf-8") as fid:
            for line in fid:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 9:
                    continue

                point3D_id = int(parts[0])
                xyz = np.asarray(parts[1:4], dtype=np.float64)
                rgb = np.asarray(parts[4:7], dtype=np.uint8)
                error = float(parts[7])
                track_length = int(parts[8])

                track = []
                point2D_idxs = []
                data = parts[9:]
                for i in range(track_length):
                    if 2 * i + 1 >= len(data):
                        break
                    image_id = int(data[2 * i])
                    point2D_idx = int(data[2 * i + 1])
                    track.append((image_id, point2D_idx))
                    point2D_idxs.append(point2D_idx)

                points3d[point3D_id] = Point3D(
                    id=point3D_id,
                    xyz=xyz,
                    rgb=rgb,
                    error=error,
                    track=track,
                    point2D_idxs=np.asarray(point2D_idxs, dtype=np.int64) if point2D_idxs else np.empty((0,), dtype=np.int64),
                )
        return points3d


__all__ = ["ColmapLoader", "CameraModel", "Camera", "Image", "Point3D", "qvec2rotmat", "rotmat2qvec"]