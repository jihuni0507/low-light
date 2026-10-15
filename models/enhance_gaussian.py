import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import numpy as np

class gaussian_enhance_net(nn.Module):
    
    def __init__(self):
        super(gaussian_enhance_net, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        
        # nn.Conv1d requires tensor with shape of [B, C, L]
        # nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias)
        number_f = 32
        # in_channes = 3 for SH DC coefficients
        self.e_conv1 = nn.Conv1d(3, number_f, 1, 1, 0, bias=True)
        self.e_conv2 = nn.Conv1d(number_f, number_f, 1, 1, 0, bias=True)
        self.e_conv3 = nn.Conv1d(number_f, number_f, 1, 1, 0, bias=True)
        self.e_conv4 = nn.Conv1d(number_f, number_f, 1, 1, 0, bias=True)
        self.e_conv5 = nn.Conv1d(number_f*2, number_f, 1, 1, 0, bias=True)
        self.e_conv6 = nn.Conv1d(number_f*2, number_f, 1, 1, 0, bias=True)
        self.e_conv7 = nn.Conv1d(number_f*2, 24, 1, 1, 0, bias=True)
        

    def forward(self, x):
        # x.shape = [B, 3, N], 3 for SH DC coefficients
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        
        x_r = torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
        r1,r2,r3,r4,r5,r6,r7,r8 = torch.split(x_r, 3, dim=1)

        x = x + r1 * (torch.pow(x, 2) - x)
        x = x + r2 * (torch.pow(x, 2) - x)
        x = x + r3 * (torch.pow(x, 2) - x)
        enhanced_gaussians_1 = x + r4 * (torch.pow(x, 2) - x)
        x = enhanced_gaussians_1 + r5 * (torch.pow(enhanced_gaussians_1, 2) - enhanced_gaussians_1)
        x = x + r6 * (torch.pow(x, 2) - x)
        x = x + r7 * (torch.pow(x, 2) - x)
        enhanced_gaussians = x + r8 * (torch.pow(x, 2) - x)
        r = torch.cat([r1, r2, r3, r4, r5, r6, r7, r8], 1)
        
        return enhanced_gaussians_1, enhanced_gaussians, r
        
        
