import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader
import loss
import numpy as np

current_dir = os.path.dirname(os.pah.abspath(__file__))
parent_dir = os.path.dirname(current_dir) #low-light
sys.path.append(parent_dir)

import models.enhance_gaussian
import models.gaussians_model

from torchvision import transforms

'''
Inputs:
    i1, i2 : two low-light images
    G : PLY file of baseline gaussian primitives (geometry frozen)
    
Outputs:
    G' : PLY file Enhanced Gaussian primiitve
    ??? : curve parameters
'''


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def train(config):
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    GE_net = models.enhance_gaussian.gaussian_enhance_net().cuda()
    GE_net.apply(weights_init)
    if config.load_pretain == True:
        GE_net.load_state_dict(torch.load(config.pretrain_dir))
    # train_dataset = dataloader.gaussian_loader(config.lowlight_images_path)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)
    
    # Gaussian-wise loss term
    L_reg = loss.L_reg()
    # Rendered Loss
    L_color = loss.L_color()
    L_exp = loss.L_exp()
    
    optimizer = torch.optim.Adam(GE_net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    
    GE_net.train()
    
    for epoch in range(config.num_epochs):
        for iteration, lowlight_img1, lowlight_img2 in enumerate(train_loader):
            lowlight_img1 = lowlight_img1.cuda()
            lowlight_img2 = lowlight_img2.cuda()
            base_gaussians = None           
            
            enhanced_gaussians_1, enhanced_gaussians, r = GE_net(base_gaussians)
            rendered_img = None # Rendered based on given third camera position
            
            # loss_reg = torch.(L_reg(enhanced_gaussians))
            loss_col = 5 * torch.mean(L_color(rendered_img))
            loss_exp = 10 * torch.mean(L_color(rendered_img))
            
            # Final Loss
            loss = loss_col + loss_exp
            
            optimizer.zero_grad()
            loss.backward()
            

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--load_pretrain', type=str, default='', required=True, help='')
    parser.add_argument('--lr', type=float, default='', required=True, help='')
    parser.add_argument('--weight_decay', type=float, default='', required=True, help='')
    parser.add_argument('--num_epochs', type=int, default='', required=True, help='')
    
    config = parser.parse_args()
    if not os.path.exists(config.snapshots_folder):
        os.mkdir(config.snapshots_folder)
        
    train(config)