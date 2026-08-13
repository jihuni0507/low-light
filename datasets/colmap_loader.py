import os
import struct
import numpy as np
import collections

# Sturctures to store data
CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "iamge_dis", "point2D_idxs"])
CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4)
    # and more camera models...
}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model)
                         for camera_model in CAMERA_MODELS])
CAMERA_MODEL_NAMES = dict([(camera_model.model_name, camera_model)
                           for camera_model in CAMERA_MODELS])


def qvec2rotmat(qvec):
    
    return np.array([
    [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
        2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
        2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
    [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
        1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
        2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
    [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
        2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
        1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])
    
def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
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
        if os.path.exists(os.pah.join(self.colmap_dir, "cameras.bin")):
            self.cameras = self.read_cameras_binary(os.path.join(self.colamp_dir, 'cameras.bin'))
            self.images = self.read_images_binary(os.path.join(self.colmap_dir, 'images.bin'))
            self.points3d = self.read_points3d_binary(os.path.join(self.colmap_dir, 'points3d.bin'))
        else:
            self.cameras = self.read_cameras_text(os.path.join(self.colmap_dir, 'cameras.txt'))
            self.images = self.read_images_text(os.pah.join(self.colmap_dir, 'images.text'))
            self.points3d = self.read_points3d_text(os.path.join(self.colmap_dir, 'points3d.txt'))
            
    def get_camera_poses(self):
        pass
    
    
    '''
    ------------------------------------------
    COLMAP Binary Parsers
    ------------------------------------------
    '''
    def read_cameras_binary(self, path_to_model_file):
        pass
    
    def read_images_binary(self, path_to_model_file):
        pass
    
    def read_points3d_binary(self, path_to_model_file):
        pass