import torch
import os
import sys
import argparse
import numpy as np
from tqdm import tqdm

from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from plyfile import PlyData
import torch.nn.functional as F
import models.enhance_gaussian
import datasets.dataset
import render.renderer_splats
from initialize_gaussians import initialize_gaussians
from loss import L_color, L_exp, L_reg, L_spa

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
        checkpoint = torch.load(config.pretrain_dir, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        GE_net.load_state_dict(state_dict)
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
    snapshots_folder = Path(config.snapshots_folder)
    snapshots_folder.mkdir(parents=True, exist_ok=True)

    lowest_loss = None
    
    for epoch in range(config.num_epochs):
        epoch_loss = 0.0
        epoch_steps = 0
        epoch_render_calls = 0
        epoch_scene_stats = {}
        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{config.num_epochs}",
            unit="batch",
        )
        for scene_batch in progress:
            batch_loss = 0.0
            batch_steps = 0
            for sample in scene_batch:
                scene_id = sample["scene_id"]

                if scene_id not in scene_cache:
                    scene_cache[scene_id] = initialize_gaussians(sample, device)

                base_gaussians = scene_cache[scene_id]
                gaussian_count = base_gaussians["xyz"].shape[0]
                scene_stats = epoch_scene_stats.setdefault(
                    scene_id,
                    {"gaussians": gaussian_count, "render_calls": 0},
                )
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
                scene_stats["render_calls"] += 1
                epoch_render_calls += 1
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
                    scene_stats["render_calls"] += 1
                    epoch_render_calls += 1
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
                batch_loss += loss.detach().item()
                batch_steps += 1

            if batch_steps:
                epoch_loss += batch_loss
                epoch_steps += batch_steps
                progress.set_postfix(
                    scene=scene_id,
                    gaussians=gaussian_count,
                    scene_renders=scene_stats["render_calls"],
                    epoch_renders=epoch_render_calls,
                    loss=f"{batch_loss / batch_steps:.4f}",
                    avg=f"{epoch_loss / epoch_steps:.4f}",
                )

        average_loss = epoch_loss / max(epoch_steps, 1)
        scene_render_summary = ", ".join(
            f"{scene_id}: {stats['gaussians']} Gaussians/"
            f"{stats['render_calls']} renders"
            for scene_id, stats in epoch_scene_stats.items()
        )
        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": GE_net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": average_loss,
            "render_calls": epoch_render_calls,
            "scene_stats": epoch_scene_stats,
        }
        
        if lowest_loss is None or average_loss < lowest_loss:
            lowest_loss = average_loss
            torch.save(
                checkpoint,
                snapshots_folder / f"best_{epoch + 1:04d}.pth",
            )
        torch.save(checkpoint, snapshots_folder / "latest.pth")
        if (
            (epoch + 1) % config.snapshot_iter == 0
            or epoch + 1 == config.num_epochs
        ):
            # torch.save(
            #     checkpoint,
            #     snapshots_folder / f"epoch_{epoch + 1:04d}.pth",
            # )
            print(
                f"Epoch {epoch + 1}/{config.num_epochs} "
                f"- average loss: {average_loss:.6f} "
                f"- rasterizer calls: {epoch_render_calls} "
                f"- scenes: {scene_render_summary}"
            )

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
    if config.num_epochs <= 0:
        parser.error("--num_epochs must be positive")
    if config.snapshot_iter <= 0:
        parser.error("--snapshot_iter must be positive")
    os.makedirs(config.snapshots_folder, exist_ok=True)
        
    train(config)