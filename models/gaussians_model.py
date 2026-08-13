''' gaussians_model.py

Container model to store each generated Gaussian primitives.

'''


import torch
import torch.nn as nn

class GaussianModel:
    def __init__(self, sh_degree):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        
        # Tensor Initialization
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        
    def create_from_pcd(self, pcd, spatial_lr_scale):
        '''
        Generate base gaussian primitives from COLMAP point cloud data(pcd)
        
        xyz, color, scale -> nn.Parameters
        '''
        pass
    
    def freeze_geometry(self):
        '''
        Freeze gaussain's geometry when training injection network
        '''
        self._xyz.requires_grad_(False)
        self._scaling.requires_grad_(False)
        self._rotation.requires_grad_(False)
        self._opacity.requires_grad_(False)
        
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self): # Input sh value for injection network
        
        return torch.cat((self._features_dc, self._features_rest), dim=1)
    
    def save_ply(self, path):   # Save trained base gaussians into .ply file
        pass