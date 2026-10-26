# 이미지 그리드 기반 TV loss는 그대로 사용하지 못함
# 가우시안 간 위치 관계에 대한 regularization 항과 위치 관계를 따로 정의할 필요가 있다

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.models.vgg import vgg16
import numpy as np

# Gaussian-Domain Loss (Regularization)
'''
gaussians_chunk_size=1024 와 같이 가우시안을 나눠서 학습할 때 
사용할 수 있는 손실로 설계해야함
'''

class L_reg(nn.Module):
    '''
    Enhance Curve Regularization Term
    '''
    
    def __init__(self):
        super(L_reg, self).__init__()
        
    def forward(self, x):
        '''
        x : Gaussian Primitives, (Batch Size, SH coefficinets, # of Gaussians)
        '''
        
        b, c, n = x.shape
        
        # Will be Implemented
        
        return 0
        
        
# Rendered Image Domain Loss

class L_color(nn.Module):
    def __init__(self):
        super(L_color, self).__init__()
        
    def forward(self, x):
        
        b, c, h, w = x.shape
        
        mean_rgb = torch.mean(x, [2, 3], keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr-mg, 2)
        Drb = torch.pow(mr-mb, 2)
        Dgb = torch.pow(mb-mg, 2)
        k = torch.pow(torch.pow(Drg, 2) + torch.pow(Drb, 2) + torch.pow(Dgb, 2), 0.5)
        
        return k
    
class L_exp(nn.Module):
    def __init__(self, patch_size, mean_val):
        super(L_exp, self).__init__()
        self.pool = nn.AvgPool2d(patch_size)
        self.mean_val = mean_val
    
    def forward(self, x):
        
        b, c, h, w = x.shape
        x = torch.mean(x, 1, keepdim=True)
        mean = self.pool(x)
        
        d = torch.mean(torch.pow(mean - torch.FloatTensor([self.mean_val]).cuda(), 2))
        return d
    
class L_spa(nn.Module):
    def __init__(self):
        super(L_spa, self).__init__()
        
        self.register_buffer(
            "weight_left",
            torch.tensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]], dtype=torch.float32)
            .unsqueeze(0).unsqueeze(0),
        )
        self.register_buffer(
            "weight_right",
            torch.tensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]], dtype=torch.float32)
            .unsqueeze(0).unsqueeze(0),
        )
        self.register_buffer(
            "weight_up",
            torch.tensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.float32)
            .unsqueeze(0).unsqueeze(0),
        )
        self.register_buffer(
            "weight_down",
            torch.tensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]], dtype=torch.float32)
            .unsqueeze(0).unsqueeze(0),
        )
        self.pool = nn.AvgPool2d(4)
        
    def forward(self, original, enhanced):
        b, c, h, w = original.shape
        
        original_mean = torch.mean(original, axis=1, keepdim=True)
        enhanced_mean = torch.mean(enhanced, axis=1, keepdim=True)
        original_pool = self.pool(original_mean)
        enhanced_pool = self.pool(enhanced_mean)
        
        D_original_left = F.conv2d(original_pool, self.weight_left, padding=1)
        D_original_right = F.conv2d(original_pool, self.weight_right, padding=1)
        D_original_up = F.conv2d(original_pool, self.weight_up, padding=1)
        D_original_down = F.conv2d(original_pool, self.weight_down, padding=1)
        
        D_enhanced_left = F.conv2d(enhanced_pool, self.weight_left, padding=1)
        D_enhanced_right = F.conv2d(enhanced_pool, self.weight_right, padding=1)
        D_enhanced_up = F.conv2d(enhanced_pool, self.weight_up, padding=1)
        D_enhanced_down = F.conv2d(enhanced_pool, self.weight_down, padding=1)
        
        D_left = torch.pow(D_original_left - D_enhanced_left, 2)
        D_right = torch.pow(D_original_right - D_enhanced_right, 2)
        D_up = torch.pow(D_original_up - D_enhanced_up, 2)
        D_down = torch.pow(D_original_down - D_enhanced_down, 2)
        
        E = (D_left + D_right + D_up + D_down)
        
        return E
        