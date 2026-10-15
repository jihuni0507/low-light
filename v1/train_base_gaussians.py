''' train_base_gaussians.py

Train a base Gaussian model initialized from a COLMAP reconstruction.
This script follows a four-stage structure:
1) config
2) data
3) model
4) training loop
'''

import argparse
import importlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from datasets.colmap_loader import ColmapLoader


@dataclass
class TrainConfig:
    colmap_dir: str = "data/colmap"
    data_dir: Optional[str] = None
    output_dir: str = "./output"
    max_train_steps: int = 1000
    epochs: int = 1
    learning_rate: float = 1e-3
    batch_size: int = 1
    num_workers: int = 0
    device: str = "cuda"
    seed: int = 43
    load_pretrained_model: bool = False
    model_name: str = "GSRecon"


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train base gaussian primitives from COLMAP reconstruction.")
    parser.add_argument("--colmap_dir", type=str, default="data/colmap", help="Path to the COLMAP sparse folder.")
    parser.add_argument("--data_dir", type=str, default=None, help="Optional image or dataset root directory.")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory for checkpoints and outputs.")
    parser.add_argument("--max_train_steps", type=int, default=1000, help="Maximum training steps to run.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=1, help="Mini-batch size.")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader worker count.")
    parser.add_argument("--device", type=str, default="cuda", help="Training device, e.g. cuda or cpu.")
    parser.add_argument("--seed", type=int, default=43, help="Random seed.")
    parser.add_argument("--load_pretrained_model", action="store_true", help="Load a pretrained model checkpoint if available.")
    parser.add_argument("--model_name", type=str, default="GSRecon", help="Model name to instantiate.")
    return TrainConfig(**vars(parser.parse_args()))


def _load_gsrecon_class():
    """Try to import the actual GSRecon implementation if it exists in the project."""
    candidates = [
        "src.models",
        "src.models.gsrecon",
        "src.models.GSRecon",
        "low_light.models.gaussians_model",
        "models.gaussians_model",
    ]

    for module_name in candidates:
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue

        if hasattr(module, "GSRecon"):
            return getattr(module, "GSRecon")
        if hasattr(module, "GaussianModel"):
            return getattr(module, "GaussianModel")
    return None


class ColmapPointCloudDataset(Dataset):
    """Simple dataset built from the initial COLMAP point cloud."""

    def __init__(self, xyz, rgb, camera_poses):
        self.xyz = torch.as_tensor(xyz, dtype=torch.float32)
        self.rgb = torch.as_tensor(rgb, dtype=torch.float32)
        self.camera_poses = camera_poses

    def __len__(self):
        return self.xyz.shape[0]

    def __getitem__(self, idx):
        return {
            "xyz": self.xyz[idx],
            "rgb": self.rgb[idx],
            "camera_pose": self.camera_poses[idx % len(self.camera_poses)] if self.camera_poses else None,
        }


class PlaceholderGaussianModel(nn.Module):
    """Fallback model used when GSRecon is not available yet."""

    def __init__(self, num_points: int):
        super().__init__()
        self.num_points = num_points
        self.pred_offset = nn.Parameter(torch.zeros(num_points, 3, dtype=torch.float32))

    def forward(self, batch):
        xyz = batch["xyz"]
        return xyz + self.pred_offset[: xyz.shape[0]]


def load_colmap_data(colmap_dir: str):
    """Stage 2: load reconstruction data and convert to training-ready tensors."""
    if not os.path.exists(colmap_dir):
        raise FileNotFoundError(f"COLMAP directory not found: {colmap_dir}")

    colmap_loader = ColmapLoader(colmap_dir)
    camera_poses = colmap_loader.get_camera_poses()
    xyz, rgb = colmap_loader.get_initial_point_cloud()

    if xyz.size == 0:
        raise ValueError(f"No 3D points were loaded from COLMAP directory: {colmap_dir}")

    return {
        "camera_poses": camera_poses,
        "xyz": xyz,
        "rgb": rgb,
    }


def build_dataloader(config: TrainConfig, recon_data: Dict[str, Any]):
    """Stage 2: create a simple DataLoader from point cloud data."""
    dataset = ColmapPointCloudDataset(recon_data["xyz"], recon_data["rgb"], recon_data["camera_poses"])
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def build_model(config: TrainConfig, recon_data: Dict[str, Any]):
    """Stage 3: instantiate the actual model if available, otherwise use a placeholder."""
    GSRecon = _load_gsrecon_class()

    if GSRecon is not None:
        try:
            model = GSRecon(**{"num_points": recon_data["xyz"].shape[0], "device": config.device})
        except TypeError:
            try:
                model = GSRecon()
            except TypeError:
                model = GSRecon
                if not isinstance(model, nn.Module):
                    raise TypeError("GSRecon is not a valid nn.Module-like class.")
        return model

    return PlaceholderGaussianModel(recon_data["xyz"].shape[0])


def compute_loss(prediction, target):
    """Stage 4: real loss computation placeholder for Gaussian optimization."""
    if isinstance(target, dict):
        target_xyz = target["xyz"]
    else:
        target_xyz = target

    return F.mse_loss(prediction, target_xyz)


def train_loop(config: TrainConfig, model, train_loader, accelerator: Accelerator):
    """Stage 4: training loop with Accelerator support."""
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

    logger = logging.getLogger(__name__)
    logger.info(f"Starting training with learning rate={config.learning_rate}, epochs={config.epochs}, max_train_steps={config.max_train_steps}")

    global_step = 0
    progress = tqdm(range(config.max_train_steps), desc="Training", ncols=100)

    for epoch in range(config.epochs):
        model.train()
        for batch in train_loader:
            if global_step >= config.max_train_steps:
                progress.close()
                logger.info("Training finished due to max_train_steps.")
                return

            optimizer.zero_grad()
            prediction = model(batch)
            loss = compute_loss(prediction, batch["xyz"])
            accelerator.backward(loss)
            optimizer.step()

            progress.set_postfix({"epoch": epoch, "loss": round(float(loss.detach().cpu()), 6)})
            progress.update(1)
            global_step += 1

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "base_gaussian_last.pt"
    accelerator.save_state(checkpoint_path)
    logger.info(f"Saved checkpoint to {checkpoint_path}")


def main():
    config = parse_args()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )

    recon_data = load_colmap_data(config.colmap_dir)
    train_loader = build_dataloader(config, recon_data)
    model = build_model(config, recon_data)

    if config.load_pretrained_model:
        checkpoint_path = Path(config.output_dir) / "base_gaussian_last.pt"
        if checkpoint_path.exists():
            state = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(state)
        else:
            logging.warning(f"Requested pretrained model, but no checkpoint was found at {checkpoint_path}")

    accelerator = Accelerator(device_placement=True, mixed_precision="no")
    train_loop(config, model, train_loader, accelerator)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())