import torch
import os
import sys
import argparse
import numpy as np

from plyfile import PlyData
import torch.nn.functional as F
import models.enhance_gaussian
import datasets.dataset
import render.renderer_splats
from initialize_gaussians import initialize_gaussians
from loss import L_color, L_exp, L_reg, L_spa

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
    color_loss = L_color()
    exposure_loss = L_exp(16, 0.6)
    spatial_loss = L_spa().to(device)

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
                rendered_img = render.renderer_splats.render_gaussians(
                    enhanced_gaussians,
                    sample["render_view"]["camera"],
                    background=(0.0, 0.0, 0.0),
                    device=device
                )
                if rendered_img is None:
                    raise TypeError(
                        "Got 'None' for rendered image"
                    )
                rendered_img = rendered_img.unsqueeze(0)

                loss_col = 5 * color_loss(rendered_img).mean()
                loss_exp = 10 * exposure_loss(rendered_img).mean()
                spatial_losses = []
                for input_view in sample["input_views"]:
                    rendered_input_view = render.renderer_splats.render_gaussians(
                        enhanced_gaussians,
                        input_view["camera"],
                        background=(0.0, 0.0, 0.0),
                        device=device,
                    )
                    if rendered_input_view is None:
                        raise TypeError("Got 'None' for rendered input view")

                    target_image = input_view["image"].to(
                        device=device, dtype=rendered_input_view.dtype
                    ).unsqueeze(0)
                    spatial_losses.append(
                        spatial_loss(
                            target_image, rendered_input_view.unsqueeze(0)
                        ).mean()
                    )
                loss_spa = torch.stack(spatial_losses).mean()
                loss_reg = 0    # L_reg(enhancement)
                loss = config.lambda_color * loss_col \
                    + config.lambda_exp * loss_exp \
                    + config.lambda_spa * loss_spa \
                    + config.lambda_curve * loss_reg
                        
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
    parser.add_argument('--lambda_exp', type=float, default=0.25)
    parser.add_argument('--lambda_color', type=float, default=0.25)
    parser.add_argument('--lambda_spa', type=float, default=0.25)
    parser.add_argument('--lambda_curve', type=float, default=0.25)
    parser.add_argument('--display_iter', type=int, default=10)
    parser.add_argument('--snapshot_iter', type=int, default=10)
    parser.add_argument('--snapshots_folder', type=str, default='snapshots')
    
    config = parser.parse_args()
    if config.load_pretrain and not config.pretrain_dir:
        parser.error("--pretrain_dir is required when --load_pretrain is set")
    os.makedirs(config.snapshots_folder, exist_ok=True)
        
    train(config)