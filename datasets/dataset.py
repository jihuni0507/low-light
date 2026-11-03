import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset


def _parse_colmap_cameras(path: Path) -> dict[int, dict[str, Any]]:
    cameras = {}
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            values = line.split()
            if len(values) < 5:
                raise ValueError(f"Malformed COLMAP camera record at {path}:{line_number}")

            camera_id = int(values[0])
            model = values[1]
            width, height = int(values[2]), int(values[3])
            params = [float(value) for value in values[4:]]
            if model == "SIMPLE_PINHOLE" and len(params) == 3:
                fx, cx, cy = params
                fy = fx
            elif model == "PINHOLE" and len(params) == 4:
                fx, fy, cx, cy = params
            else:
                raise ValueError(
                    f"Unsupported COLMAP camera model {model!r} in {path}:{line_number}; "
                    "undistort the images and use PINHOLE or SIMPLE_PINHOLE."
                )
            if width <= 0 or height <= 0 or fx <= 0 or fy <= 0:
                raise ValueError(
                    f"Invalid COLMAP camera dimensions or focal lengths at "
                    f"{path}:{line_number}"
                )
            if camera_id in cameras:
                raise ValueError(f"Duplicate COLMAP camera id {camera_id} in {path}")
            cameras[camera_id] = {
                "width": width,
                "height": height,
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy,
            }
    if not cameras:
        raise ValueError(f"No COLMAP cameras found in {path}")
    return cameras


def _qvec_to_rotation(qvec: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = qvec
    return np.asarray(
        [
            [
                1 - 2 * (qy * qy + qz * qz),
                2 * (qx * qy - qw * qz),
                2 * (qx * qz + qw * qy),
            ],
            [
                2 * (qx * qy + qw * qz),
                1 - 2 * (qx * qx + qz * qz),
                2 * (qy * qz - qw * qx),
            ],
            [
                2 * (qx * qz - qw * qy),
                2 * (qy * qz + qw * qx),
                1 - 2 * (qx * qx + qy * qy),
            ],
        ],
        dtype=np.float64,
    )


def _parse_colmap_images(
    path: Path, cameras: dict[int, dict[str, Any]]
) -> dict[int, dict[str, Any]]:
    images = {}
    lines = path.read_text(encoding="utf-8").splitlines()
    line_index = 0
    while line_index < len(lines):
        line = lines[line_index].strip()
        if not line or line.startswith("#"):
            line_index += 1
            continue

        values = line.split(maxsplit=9)
        if len(values) != 10:
            raise ValueError(
                f"Malformed COLMAP image record at {path}:{line_index + 1}"
            )
        image_id = int(values[0])
        qvec = np.asarray([float(value) for value in values[1:5]], dtype=np.float64)
        tvec = np.asarray([float(value) for value in values[5:8]], dtype=np.float64)
        camera_id = int(values[8])
        image_name = values[9]
        if image_id in images:
            raise ValueError(f"Duplicate COLMAP image id {image_id} in {path}")
        if camera_id not in cameras:
            raise ValueError(
                f"COLMAP image {image_name!r} references missing camera id {camera_id}"
            )
        if not np.isfinite(qvec).all() or np.linalg.norm(qvec) <= 1e-12:
            raise ValueError(
                f"Invalid COLMAP quaternion at {path}:{line_index + 1}"
            )
        if not np.isfinite(tvec).all():
            raise ValueError(f"Invalid COLMAP translation at {path}:{line_index + 1}")

        rotation_w2c = _qvec_to_rotation(qvec / np.linalg.norm(qvec))
        rotation_c2w = rotation_w2c.T
        position = -rotation_c2w @ tvec
        images[image_id] = {
            "id": image_id,
            "image_name": image_name,
            **cameras[camera_id],
            "rotation": rotation_c2w.tolist(),
            "position": position.tolist(),
        }

        # Each image pose record is followed by its 2D point record.
        line_index += 2

    if not images:
        raise ValueError(f"No registered COLMAP images found in {path}")
    return images


def load_colmap_cameras(model_path: Path) -> dict[int, dict[str, Any]]:
    cameras_path = model_path / "cameras.txt"
    images_path = model_path / "images.txt"
    missing = [path for path in (cameras_path, images_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "COLMAP text model is missing required files: "
            + ", ".join(str(path) for path in missing)
        )
    return _parse_colmap_images(images_path, _parse_colmap_cameras(cameras_path))


class GaussianSceneDataset(Dataset):
    def __init__(self, dataset_yaml: str):
        self.dataset_yaml = Path(dataset_yaml).resolve()

        with self.dataset_yaml.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

        dataset_config = config["dataset"]
        self.root = Path(dataset_config.get("root", "."))
        if not self.root.is_absolute():
            self.root = (self.dataset_yaml.parent / self.root).resolve()

        self.samples = dataset_config["samples"]
        if not self.samples:
            raise ValueError("dataset.samples must contain at least one scene")
        self._camera_cache: dict[Path, dict[int, dict[str, Any]]] = {}

        for i, sample in enumerate(self.samples):
            if not sample.get("id"):
                raise ValueError(f"Scene at index {i} is missing 'id'")
            if not sample.get("source_gaussian_ply"):
                raise ValueError(f"Scene {sample['id']!r} is missing 'source_gaussian_ply'")
            if not sample.get("camera_json") and not sample.get("colmap_model"):
                raise ValueError(
                    f"Scene {sample['id']!r} must provide either 'camera_json' "
                    "or 'colmap_model'"
                )

            input_views = sample.get("input_views")
            if not isinstance(input_views, list) or len(input_views) != 2:
                raise ValueError(
                    f"Scene {sample['id']!r} must provide exactly two input_views"
                )
            if not isinstance(sample.get("render_view"), dict):
                raise ValueError(f"Scene {sample['id']!r} is missing render_view")

            required_paths = ["source_gaussian_ply"]
            if sample.get("camera_json"):
                required_paths.append("camera_json")
            if sample.get("colmap_model"):
                required_paths.append("colmap_model")
            for key in required_paths:
                path = self._resolve_path(sample[key])
                if key == "colmap_model":
                    if not path.is_dir():
                        raise FileNotFoundError(
                            f"Scene {sample['id']!r}: COLMAP model directory not found: {path}"
                        )
                    load_colmap_cameras(path)
                elif not path.is_file():
                    raise FileNotFoundError(f"Scene {sample['id']!r}: {path}")

            for view in input_views:
                image_value = view.get("image", view.get("input_image"))
                if not image_value or "camera_id" not in view:
                    raise ValueError(
                        f"Each input view in scene {sample['id']!r} needs image "
                        "and camera_id"
                    )
                image_path = self._resolve_path(image_value)
                if not image_path.is_file():
                    raise FileNotFoundError(
                        f"Scene {sample['id']!r}: image not found: {image_path}"
                    )
            if "camera_id" not in sample["render_view"]:
                raise ValueError(
                    f"Scene {sample['id']!r} render_view needs camera_id"
                )

    def _resolve_path(self, path: str) -> Path:
        path = Path(path)
        return path.resolve() if path.is_absolute() else (self.root / path).resolve()

    @staticmethod
    def _load_image(path: Path) -> torch.Tensor:
        with Image.open(path) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1).contiguous()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]

        camera_json = None
        if sample.get("camera_json"):
            camera_json = self._resolve_path(sample["camera_json"])
            with camera_json.open("r", encoding="utf-8") as file:
                camera_entries = json.load(file)
            camera_by_id = {int(camera["id"]): camera for camera in camera_entries}
        else:
            colmap_model = self._resolve_path(sample["colmap_model"])
            if colmap_model not in self._camera_cache:
                self._camera_cache[colmap_model] = load_colmap_cameras(colmap_model)
            camera_by_id = self._camera_cache[colmap_model]

        input_views = []

        for view_config in sample["input_views"]:
            camera_id = int(view_config["camera_id"])
            if camera_id not in camera_by_id:
                raise ValueError(
                    f"Camera id {camera_id} not found for scene {sample['id']!r}"
                )

            image_path = self._resolve_path(
                view_config.get("image", view_config.get("input_image"))
            )
            camera = camera_by_id[camera_id]
            image = self._load_image(image_path)

            if image.shape[-2:] != (int(camera["height"]), int(camera["width"])):
                raise ValueError(
                    f"Image size {tuple(image.shape[-2:])} does not match camera "
                    f"size {(camera['height'], camera['width'])}: {image_path}"
                )

            input_views.append({
                "image": image,
                "image_path": str(image_path),
                "camera_id": camera_id,
                "image_name": camera.get("image_name"),
                "camera": camera,
            })

        render_camera_id = int(sample["render_view"]["camera_id"])
        if render_camera_id not in camera_by_id:
            raise ValueError(
                f"Render camera id {render_camera_id} not found for scene "
                f"{sample['id']!r}"
            )

        return {
            "scene_id": str(sample["id"]),
            "ply_path": str(self._resolve_path(sample["source_gaussian_ply"])),
            "camera_json": str(camera_json) if camera_json else None,
            "colmap_model": (
                str(self._resolve_path(sample["colmap_model"]))
                if sample.get("colmap_model") else None
            ),
            "input_views": input_views,
            "render_view": {
                "camera_id": render_camera_id,
                "image_name": camera_by_id[render_camera_id].get("image_name"),
                "camera": camera_by_id[render_camera_id],
            },
        }


def scene_collate_fn(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Gaussian counts and view counts can differ between scenes.
    return batch