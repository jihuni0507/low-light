import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

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
    lambda_sh: float = 1.0
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
    parser.add_argument("--lambda_sh", type=float, default=1.0, help="Weight of SH feature MSE.")
    parser.add_argument("--lambda_image", type=float, default=1.0, help="Weight of rendered-image L1 loss.")
    parser.add_argument("--lambda_ssim", type=float, default=0.2, help="Weight of rendered-image SSIM loss.")
    return TrainConfig(**vars(parser.parse_args()))


def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    selected_ids = camera_ids if camera_ids is not None else sorted(by_id)[:3]
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


def render_teacher_images(model, cameras, pipeline, background):
    with torch.no_grad():
        return [
            stereogs_render(camera, model, pipeline, background, train=False)["render"].detach()
            for camera in cameras
        ]


def render_image_loss(model, cameras, target_images, pipeline, background):
    terms = []
    for camera, target_image in zip(cameras, target_images):
        predicted_image = stereogs_render(camera, model, pipeline, background, train=False)["render"].unsqueeze(0)
        target_image = target_image.unsqueeze(0)
        terms.append((F.l1_loss(predicted_image, target_image), ssim_loss(predicted_image, target_image)))
    l1 = torch.stack([term[0] for term in terms]).mean()
    ssim = torch.stack([term[1] for term in terms]).mean()
    return l1, ssim


def main():
    config = parse_args()
    set_seed(config.seed)
    device = torch.device(config.device)

    encoder = CLIPEncoder(model_name=config.clip_model, device=str(device))
    dataloader = build_injection_dataloader(
        dataset_yaml=config.dataset_yaml,
        sh_degree=config.sh_degree,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        device="cpu",
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

    optimizer = torch.optim.Adam(net.parameters(), lr=config.learning_rate)

    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    render_cache = {}
    source_model = None
    pipeline = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    background = torch.zeros(3, device=device)
    if config.lambda_image > 0:
        if device.type != "cuda":
            raise ValueError("StereoGS rendered hybrid loss requires a CUDA device.")
        source_model = StereoGaussianModel(config.sh_degree)
        source_model.load_ply(dataloader.dataset.source_gaussian_ply)
        for dataset_index, dataset_sample in enumerate(dataloader.dataset):
            camera_json = dataset_sample["camera_json"]
            if not camera_json:
                raise ValueError(
                    f"Sample {dataset_sample['id']!r} requires camera_json for rendered loss."
                )
            cameras = load_render_cameras(
                camera_json, dataset_sample["camera_ids"], device
            )
            target_model = StereoGaussianModel(config.sh_degree)
            target_model.load_ply(dataset_sample["target_gs"])
            render_cache[dataset_sample["id"]] = (
                cameras,
                render_teacher_images(target_model, cameras, pipeline, background),
            )
            del target_model

    for epoch in range(config.epochs):
        net.train()
        epoch_loss = 0.0

        for batch in dataloader:
            optimizer.zero_grad()
            conditions = encode_prompt_batch(batch, encoder, device=str(device))
            batch_loss = torch.zeros((), device=device)

            for sample, condition in zip(batch, conditions):
                source_features = sample["source_features"].to(device)
                target_features = sample["target_features"].to(device)
                predicted = net(source_features, condition)
                total_loss = config.lambda_sh * F.mse_loss(predicted, target_features)
                if config.lambda_image > 0:
                    cameras, target_images = render_cache[sample["id"]]
                    source_model._features_dc = predicted[:, :1, :]
                    source_model._features_rest = predicted[:, 1:, :]
                    image_l1, image_ssim = render_image_loss(
                        source_model, cameras, target_images, pipeline, background
                    )
                    total_loss = total_loss + config.lambda_image * image_l1
                    total_loss = total_loss + config.lambda_ssim * image_ssim
                batch_loss = batch_loss + total_loss

            batch_loss = batch_loss / len(batch)
            batch_loss.backward()
            optimizer.step()
            epoch_loss += batch_loss.item()

        mean_loss = epoch_loss / len(dataloader)
        print(f"epoch={epoch + 1}/{config.epochs} loss={mean_loss:.6f}")

    checkpoint_path = save_dir / "injection_network.pt"
    torch.save(
        {
            "model_state_dict": net.state_dict(),
            "condition_dim": condition_dim,
            "hidden_dim": config.hidden_dim,
            "sh_degree": config.sh_degree,
        },
        checkpoint_path,
    )
    print(f"Saved injection network to: {checkpoint_path}")


if __name__ == "__main__":
    main()
