import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from models.clip_encoder import CLIPEncoder
from models.injection_network import ConditionedGaussianSHNet
from train.injection_dataset import build_injection_dataloader, encode_prompt_batch

STEREOGS_ROOT = Path(__file__).resolve().parents[2] / "StereoGS"
if str(STEREOGS_ROOT) not in sys.path:
    sys.path.insert(0, str(STEREOGS_ROOT))

from gaussian_renderer import render as stereogs_render
from scene.cameras import Camera
from scene.gaussian_model import GaussianModel as StereoGaussianModel


@dataclass
class TrainConfig:
    dataset_yaml: str = "dataset.yaml"
    sh_degree: int = 3
    batch_size: int = 1
    num_workers: int = 0
    clip_model: str = "openai/clip-vit-base-patch32"
    condition_dim: int = 512
    hidden_dim: int = 256
    epochs: int = 10
    learning_rate: float = 1e-4
    save_dir: str = "./output/injection"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    lambda_sh: float = 0.0
    lambda_image: float = 1.0
    lambda_ssim: float = 0.2


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train a Gaussian SH injection network conditioned on a CLIP embedding.")
    parser.add_argument("--dataset_yaml", type=str, default="dataset.yaml", help="Dataset YAML containing source Gaussian PLY, target GS PLYs, and prompts.")
    parser.add_argument("--sh_degree", type=int, default=3, help="SH degree used by the source and target PLY files.")
    parser.add_argument("--batch_size", type=int, default=1, help="Number of dataset samples per optimization step.")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader worker count.")
    parser.add_argument("--clip_model", type=str, default="openai/clip-vit-base-patch32", help="CLIP model name.")
    parser.add_argument("--condition_dim", type=int, default=512, help="Dimension of the CLIP embedding.")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension for the injection network.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--save_dir", type=str, default="./output/injection", help="Directory to save the trained injection network.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Training device, e.g. cuda or cpu.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--lambda_sh", type=float, default=0.0, help="Deprecated; pseudo-GT image training has no target SH loss.")
    parser.add_argument("--lambda_image", type=float, default=1.0, help="Weight of rendered-image L1 loss.")
    parser.add_argument("--lambda_ssim", type=float, default=0.2, help="Weight of rendered-image SSIM loss.")
    return TrainConfig(**vars(parser.parse_args()))


def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_distributed(config):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if config.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for rendered Gaussian training.")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        if distributed:
            raise ValueError("Distributed training requires a CUDA device.")
        device = torch.device(config.device)

    if distributed:
        dist.init_process_group(backend="nccl")

    return distributed, local_rank, device


def cleanup_distributed(distributed):
    if distributed and dist.is_initialized():
        dist.destroy_process_group()


def make_camera(camera_entry, device):
    """Rebuild a StereoGS camera from Scene-generated cameras.json metadata."""
    width = int(camera_entry["width"])
    height = int(camera_entry["height"])
    camera_to_world = torch.eye(4, dtype=torch.float64).numpy()
    camera_to_world[:3, :3] = camera_entry["rotation"]
    camera_to_world[:3, 3] = camera_entry["position"]
    world_to_camera = torch.linalg.inv(torch.from_numpy(camera_to_world)).numpy()
    fovx = 2.0 * math.atan(width / (2.0 * float(camera_entry["fx"])))
    fovy = 2.0 * math.atan(height / (2.0 * float(camera_entry["fy"])))
    return Camera(
        colmap_id=int(camera_entry.get("id", 0)),
        R=world_to_camera[:3, :3].T,
        T=world_to_camera[:3, 3],
        FoVx=fovx,
        FoVy=fovy,
        image=torch.zeros((3, height, width), device=device),
        image_name=camera_entry.get("img_name"),
        uid=int(camera_entry.get("id", 0)),
        data_device=str(device),
    )


def load_render_cameras(camera_json, camera_ids, device):
    with open(camera_json, "r", encoding="utf-8") as file:
        entries = json.load(file)
    by_id = {int(entry["id"]): entry for entry in entries}
    selected_ids = camera_ids if camera_ids is not None else sorted(by_id)[:5]
    if len(selected_ids) != 5:
        raise ValueError(
            f"Expected exactly 5 cameras for a sample, got {len(selected_ids)}"
        )
    missing = [camera_id for camera_id in selected_ids if int(camera_id) not in by_id]
    if missing:
        raise ValueError(f"Camera ids not found in {camera_json}: {missing}")
    return [make_camera(by_id[int(camera_id)], device) for camera_id in selected_ids]


def ssim_loss(predicted, target, window_size=11):
    """Differentiable SSIM approximation used alongside rendered L1."""
    padding = window_size // 2
    mean_pred = F.avg_pool2d(predicted, window_size, stride=1, padding=padding)
    mean_target = F.avg_pool2d(target, window_size, stride=1, padding=padding)
    variance_pred = F.avg_pool2d(predicted * predicted, window_size, stride=1, padding=padding) - mean_pred * mean_pred
    variance_target = F.avg_pool2d(target * target, window_size, stride=1, padding=padding) - mean_target * mean_target
    covariance = F.avg_pool2d(predicted * target, window_size, stride=1, padding=padding) - mean_pred * mean_target
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    numerator = (2 * mean_pred * mean_target + c1) * (2 * covariance + c2)
    denominator = (mean_pred * mean_pred + mean_target * mean_target + c1) * (variance_pred + variance_target + c2)
    return 1.0 - (numerator / (denominator + 1e-8)).mean()


def render_image_loss(model, cameras, target_images, pipeline, background):
    terms = []
    for camera, target_image in zip(cameras, target_images):
        predicted_image = stereogs_render(camera, model, pipeline, background, train=False)["render"].unsqueeze(0)
        target_image = target_image.unsqueeze(0)
        terms.append((F.l1_loss(predicted_image, target_image), ssim_loss(predicted_image, target_image)))
    l1 = torch.stack([term[0] for term in terms]).mean()
    ssim = torch.stack([term[1] for term in terms]).mean()
    return l1, ssim


def load_render_sample(dataset_sample, sh_degree, device):
    source_model = StereoGaussianModel(sh_degree)
    source_model.load_ply(dataset_sample["source_gaussian_ply"])
    camera_json = dataset_sample["camera_json"]
    if not camera_json:
        raise ValueError(
            f"Sample {dataset_sample['id']!r} requires camera_json for rendered loss."
        )

    cameras = load_render_cameras(
        camera_json, dataset_sample["camera_ids"], device
    )
    target_images = [image.to(device) for image in dataset_sample["target_images"]]
    for camera, target_image in zip(cameras, target_images):
        if target_image.shape[-2:] != (camera.image_height, camera.image_width):
            raise ValueError(
                f"Pseudo-GT image shape {tuple(target_image.shape[-2:])} does not match "
                f"camera shape {(camera.image_height, camera.image_width)} for sample "
                f"{dataset_sample['id']!r}"
            )
    return source_model, cameras, target_images


def main():
    config = parse_args()
    set_seed(config.seed)
    distributed, rank, device = setup_distributed(config)
    if config.lambda_sh != 0:
        raise ValueError("--lambda_sh must be 0: pseudo-GT image training has no target Gaussian SH.")
    if config.lambda_image <= 0 and config.lambda_ssim <= 0:
        raise ValueError("At least one image loss weight must be greater than 0.")

    encoder = CLIPEncoder(model_name=config.clip_model, device=str(device))
    dataloader = build_injection_dataloader(
        dataset_yaml=config.dataset_yaml,
        sh_degree=config.sh_degree,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        device="cpu",
    )
    sampler = DistributedSampler(dataloader.dataset, shuffle=True) if distributed else None
    if sampler is not None:
        dataloader = torch.utils.data.DataLoader(
            dataloader.dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=config.num_workers,
            collate_fn=dataloader.collate_fn,
            pin_memory=False,
        )

    condition_dim = encoder.feature_dim
    if config.condition_dim != condition_dim:
        print(
            f"Warning: --condition_dim={config.condition_dim} does not match "
            f"the CLIP encoder dimension {condition_dim}; using {condition_dim}."
        )

    net = ConditionedGaussianSHNet(
        condition_dim=condition_dim,
        sh_channels=3,
        hidden_dim=config.hidden_dim,
    ).to(device)
    if distributed:
        net = DDP(net, device_ids=[rank], output_device=rank)

    optimizer = torch.optim.Adam(net.parameters(), lr=config.learning_rate)

    save_dir = Path(config.save_dir)
    if not distributed or rank == 0:
        save_dir.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()

    pipeline = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    background = torch.zeros(3, device=device)
    if config.lambda_image > 0 or config.lambda_ssim > 0:
        if device.type != "cuda":
            raise ValueError("StereoGS rendered hybrid loss requires a CUDA device.")
        print("Rendering is enabled; scenes will be loaded one at a time.")

    for epoch in range(config.epochs):
        net.train()
        epoch_loss = 0.0
        if sampler is not None:
            sampler.set_epoch(epoch)

        for batch in dataloader:
            conditions = encode_prompt_batch(batch, encoder, device=str(device))

            for sample, condition in zip(batch, conditions):
                optimizer.zero_grad()
                source_features = sample["source_features"].to(device)
                predicted = net(source_features, condition)
                source_model, cameras, target_images = load_render_sample(
                    sample, config.sh_degree, device
                )
                source_model._features_dc = predicted[:, :1, :]
                source_model._features_rest = predicted[:, 1:, :]
                image_l1, image_ssim = render_image_loss(
                    source_model, cameras, target_images, pipeline, background
                )
                total_loss = config.lambda_image * image_l1
                if config.lambda_ssim > 0:
                    total_loss = total_loss + config.lambda_ssim * image_ssim
                total_loss.backward()
                optimizer.step()
                epoch_loss += total_loss.item()

                del total_loss, image_l1, image_ssim
                del predicted, source_features, source_model, cameras, target_images
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        mean_loss = epoch_loss / len(dataloader.dataset)
        print(f"epoch={epoch + 1}/{config.epochs} loss={mean_loss:.6f}")

    if not distributed or rank == 0:
        checkpoint_path = save_dir / "injection_network.pt"
        state_dict = net.module.state_dict() if distributed else net.state_dict()
        torch.save(
            {
                "model_state_dict": state_dict,
                "condition_dim": condition_dim,
                "hidden_dim": config.hidden_dim,
                "sh_degree": config.sh_degree,
            },
            checkpoint_path,
        )
        print(f"Saved injection network to: {checkpoint_path}")

    cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
