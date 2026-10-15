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
    # tran_loadr = ...
    
    L_color = loss.L_color()
    L_second = loss.L_second()
    # and more losses...
    
    optimizer = torch.optim.Adam(GE_net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    
    GE_net.train()
    
    for epoch in range(config.num_epochs):
        pass

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