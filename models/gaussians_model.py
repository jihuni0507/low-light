''' gaussians_model.py

Container model to store each generated Gaussian primitives.

'''


import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict
import struct

class GaussianModel:
    def __init__(self, sh_degree):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        
        # Tensor Initialization - Geometry
        self._xyz = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        
        # Tensor Initialization - Color Features
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        
    def create_from_pcd(self, pcd, spatial_lr_scale):
        '''
        Generate base gaussian primitives from COLMAP point cloud data(pcd)
        
        xyz, color, scale -> nn.Parameters
        
        Args:
            pcd: Dict containing 'xyz' and 'rgb' arrays from COLMAP
            spatial_lr_scale: Learning rate scale for spatial parameters
        '''
        xyz = np.asarray(pcd['xyz'])
        rgb = np.asarray(pcd['rgb'])
        
        num_points = xyz.shape[0]
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Initialize geometry parameters
        self._xyz = nn.Parameter(
            torch.from_numpy(xyz).float().to(device)
        )
        
        # Initialize rotation as identity quaternion [w, x, y, z]
        self._rotation = nn.Parameter(
            torch.zeros((num_points, 4), dtype=torch.float32, device=device)
        )
        self._rotation.data[:, 0] = 1.0  # Identity quaternion
        
        # Initialize scaling (isotropic, in log space)
        self._scaling = nn.Parameter(
            torch.full((num_points, 3), -1.0, dtype=torch.float32, device=device)
        )
        
        # Initialize opacity to 1.0
        self._opacity = nn.Parameter(
            torch.ones((num_points, 1), dtype=torch.float32, device=device)
        )
        
        # Initialize color features from RGB
        # SH order 0 (DC component) from RGB
        srgb = rgb / 255.0 if rgb.max() > 1 else rgb
        self._features_dc = nn.Parameter(
            torch.from_numpy(srgb).float().to(device).unsqueeze(1)  # (N, 1, 3)
        )
        
        # Initialize SH rest (higher order terms) to zero
        n_rest_features = (self.max_sh_degree + 1) ** 2 - 1
        self._features_rest = nn.Parameter(
            torch.zeros((num_points, n_rest_features, 3), dtype=torch.float32, device=device)
        )
    
    def freeze_geometry(self):
        '''
        Freeze gaussain's geometry when training injection network
        '''
        self._xyz.requires_grad_(False)
        self._scaling.requires_grad_(False)
        self._rotation.requires_grad_(False)
        self._opacity.requires_grad_(False)
        
    def to(self, device):
        """Move all Gaussian tensors to the target device."""
        for attr in ['_xyz', '_features_dc', '_features_rest', '_scaling', '_rotation', '_opacity']:
            tensor = getattr(self, attr, None)
            if isinstance(tensor, torch.Tensor):
                setattr(self, attr, tensor.to(device))
        return self

    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self): # Input sh value for injection network
        
        return torch.cat((self._features_dc, self._features_rest), dim=1)
    
    def save_ply(self, path):
        '''Save trained base gaussians into .ply file'''
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Move tensors to CPU for numpy conversion
        xyz = self._xyz.data.cpu().numpy()
        rotation = self._rotation.data.cpu().numpy()
        scaling = self._scaling.data.cpu().numpy()
        opacity = self._opacity.data.cpu().numpy()
        features_dc = self._features_dc.data.cpu().numpy()
        features_rest = self._features_rest.data.cpu().numpy()
        
        num_points = xyz.shape[0]
        
        # Prepare vertex array
        rest_fields = [
            ('f_rest_{}'.format(i), 'f4')
            for i in range(features_rest.shape[1] * features_rest.shape[2])
        ]
        vertices = np.zeros(num_points, dtype=[
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
            *rest_fields,
            ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
            ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4'),
            ('opacity', 'f4'),
        ])
        
        # Fill vertex data
        vertices['x'] = xyz[:, 0]
        vertices['y'] = xyz[:, 1]
        vertices['z'] = xyz[:, 2]
        
        # Normal placeholder
        vertices['nx'] = 0
        vertices['ny'] = 0
        vertices['nz'] = 1
        
        # DC features
        vertices['f_dc_0'] = features_dc[:, 0, 0]
        vertices['f_dc_1'] = features_dc[:, 0, 1]
        vertices['f_dc_2'] = features_dc[:, 0, 2]
        
        rest_flat = features_rest.transpose(0, 2, 1).reshape(num_points, -1)
        for i, field_name in enumerate(name for name in vertices.dtype.names if name.startswith('f_rest_')):
            vertices[field_name] = rest_flat[:, i]
        
        # Geometry
        vertices['scale_0'] = scaling[:, 0]
        vertices['scale_1'] = scaling[:, 1]
        vertices['scale_2'] = scaling[:, 2]
        
        vertices['rot_0'] = rotation[:, 0]
        vertices['rot_1'] = rotation[:, 1]
        vertices['rot_2'] = rotation[:, 2]
        vertices['rot_3'] = rotation[:, 3]
        
        vertices['opacity'] = opacity[:, 0]
        # Write PLY file manually (simple binary PLY writer)
        self._write_ply(path, vertices)
    
    def _write_ply(self, path, vertices):
        '''Write vertices to PLY file in binary format'''
        with open(path, 'wb') as f:
            # Write header
            f.write(b'ply\n')
            f.write(b'format binary_little_endian 1.0\n')
            f.write(f'element vertex {len(vertices)}\n'.encode())
            
            for field_name in vertices.dtype.names:
                field_type = vertices.dtype.fields[field_name][0].str[1]
                if field_type == 'f':
                    ply_type = 'float'
                elif field_type == 'u':
                    ply_type = 'uchar'
                else:
                    ply_type = 'float'
                f.write(f'property {ply_type} {field_name}\n'.encode())
            
            f.write(b'end_header\n')
            
            # Write binary data
            f.write(vertices.tobytes())
    
    def load_ply(self, path):
        '''Load Gaussians from PLY file'''
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"PLY file not found: {path}")
        
        # Read PLY file
        vertices = self._read_ply(path)
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load geometry
        xyz = np.stack([vertices['x'], vertices['y'], vertices['z']], axis=1)
        self._xyz = nn.Parameter(torch.from_numpy(xyz).float().to(device))
        
        scaling = np.stack([vertices['scale_0'], vertices['scale_1'], vertices['scale_2']], axis=1)
        self._scaling = nn.Parameter(torch.from_numpy(scaling).float().to(device))
        
        rotation = np.stack([vertices['rot_0'], vertices['rot_1'], vertices['rot_2'], vertices['rot_3']], axis=1)
        self._rotation = nn.Parameter(torch.from_numpy(rotation).float().to(device))
        
        opacity = vertices['opacity'].reshape(-1, 1)
        self._opacity = nn.Parameter(torch.from_numpy(opacity).float().to(device))
        
        # Load features
        dc = np.stack([vertices['f_dc_0'], vertices['f_dc_1'], vertices['f_dc_2']], axis=1)
        self._features_dc = nn.Parameter(torch.from_numpy(dc).float().to(device).unsqueeze(1))
        
        n_rest = (self.max_sh_degree + 1) ** 2 - 1
        rest_names = [name for name in vertices.dtype.names if name.startswith('f_rest_')]
        rest_names.sort(key=lambda name: int(name.rsplit('_', 1)[1]))
        expected_rest = 3 * n_rest
        if len(rest_names) != expected_rest:
            raise ValueError(
                f'Expected {expected_rest} f_rest fields for SH degree {self.max_sh_degree}, '
                f'found {len(rest_names)}.'
            )
        rest = np.stack([vertices[name] for name in rest_names], axis=1)
        rest = rest.reshape(-1, 3, n_rest).transpose(0, 2, 1)
        self._features_rest = nn.Parameter(
            torch.from_numpy(rest).float().to(device)
        )
        
        if 'sh_degree' in vertices.dtype.names:
            self.active_sh_degree = int(vertices['sh_degree'][0])
    
    def _read_ply(self, path):
        '''Read PLY file and return numpy structured array'''
        with open(path, 'rb') as f:
            # Read header
            header_lines = []
            while True:
                line = f.readline().decode('ascii').strip()
                header_lines.append(line)
                if line == 'end_header':
                    break
            
            # Parse header
            n_vertices = 0
            dtype_list = []
            for line in header_lines:
                if line.startswith('element vertex'):
                    n_vertices = int(line.split()[-1])
                elif line.startswith('property'):
                    parts = line.split()
                    prop_type = parts[1]
                    prop_name = parts[2]
                    if prop_type == 'float':
                        dtype_list.append((prop_name, 'f4'))
                    elif prop_type == 'uchar':
                        dtype_list.append((prop_name, 'u1'))
            
            # Read binary data
            dtype = np.dtype(dtype_list)
            data = np.frombuffer(f.read(n_vertices * dtype.itemsize), dtype=dtype).copy()
            
        return data
    
    def get_geometry(self) -> Dict[str, torch.Tensor]:
        '''Get only geometry parameters (xyz, scale, rotation, opacity)'''
        return {
            'xyz': self._xyz,
            'scaling': self._scaling,
            'rotation': self._rotation,
            'opacity': self._opacity,
        }
    
    def set_features(self, features_dc, features_rest=None):
        '''Set SH features (for injection network)'''
        self._features_dc = nn.Parameter(features_dc)
        if features_rest is not None:
            self._features_rest = nn.Parameter(features_rest)