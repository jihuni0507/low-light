import torch
import os
import sys
import argparse
import loss
import numpy as np

from plyfile import PlyData
import torch.nn.functional as F
import models.enhance_gaussian
import datasets.dataset
from initialize_gaussians import initialize_gaussians

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir) #low-light
sys.path.append(parent_dir)

C0 = 0.28209479177387814

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


######################################################
# Train Loop
######################################################


def train(config):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    GE_net = models.enhance_gaussian.gaussian_enhance_net().to(device)
    GE_net.apply(weights_init)
    if config.load_pretrain:
        GE_net.load_state_dict(
            torch.load(config.pretrain_dir, map_location=device)
        )
    ######################################################
    # Datasets
    ######################################################
    
    dataset = datasets.dataset.GaussianSceneDataset(config.dataset_yaml)
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0, # 에러 없이 잘 굴러가면 늘리기
        collate_fn=datasets.dataset.scene_collate_fn,
    )

    ######################################################
    # Loss
    ######################################################
    # Rendered loss
    L_color = loss.L_color()
    L_exp = loss.L_exp(16, 0.6)

    optimizer = torch.optim.Adam(GE_net.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    GE_net.train()

    scene_cache = {}

    for epoch in range(config.num_epochs):
        for scene_batch in train_loader:
            for sample in scene_batch:
                scene_id = sample["scene_id"]

                if scene_id not in scene_cache:
                    scene_cache[scene_id] = initialize_gaussians(sample, device)

                base_gaussians = scene_cache[scene_id]
                optimizer.zero_grad()
                enhanced_rgb_1, enhanced_rgb, r = GE_net(
                    base_gaussians["initial_rgb"]
                )
                enhanced_rgb = enhanced_rgb[0].transpose(0, 1)
                enhanced_rgb = torch.where(
                    base_gaussians["observed_mask"][:, None],
                    enhanced_rgb,
                    base_gaussians["base_rgb"],
                )

                enhanced_gaussians = dict(base_gaussians)
                enhanced_gaussians["features_dc"] = (
                    enhanced_rgb - 0.5
                ) / C0
                # The remaining geometry and SH coefficients are unchanged.
                rendered_img = None  # Render at sample["render_view"]["camera"].

                if rendered_img is None:
                    raise NotImplementedError(
                        "Connect the differentiable Gaussian renderer before training."
                    )

                loss_col = 5 * torch.mean(L_color(rendered_img))
                loss_exp = 10 * torch.mean(L_exp(rendered_img))
                loss = loss_col + loss_exp

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    GE_net.parameters(), config.grad_clip_norm
                )
                optimizer.step()
            

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--load_pretrain', action='store_true', help='')
    parser.add_argument('--pretrain_dir', type=str, default='')
    parser.add_argument('--dataset_yaml', type=str, required=True, default='')
    parser.add_argument('--train_batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--gaussian_batch_size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=0.0001, help='')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='')
    parser.add_argument('--num_epochs', type=int, default=200, help='')
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)
    parser.add_argument('--display_iter', type=int, default=10)
    parser.add_argument('--snapshot_iter', type=int, default=10)
    parser.add_argument('--snapshots_folder', type=str, default='snapshots')
    
    config = parser.parse_args()
    if config.load_pretrain and not config.pretrain_dir:
        parser.error("--pretrain_dir is required when --load_pretrain is set")
    os.makedirs(config.snapshots_folder, exist_ok=True)
        
    train(config)