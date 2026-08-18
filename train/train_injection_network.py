import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from models.clip_encoder import CLIPEncoder
from models.gaussians_model import GaussianModel
from models.injection_network import ConditionedGaussianSHNet


@dataclass
class TrainConfig:
    gaussian_ply: str = ""
    target_ply: str = ""
    text: str = ""
    image: str = ""
    image_dir: str = ""
    condition_mode: str = "text"
    clip_model: str = "openai/clip-vit-base-patch32"
    condition_dim: int = 512
    hidden_dim: int = 256
    epochs: int = 10
    learning_rate: float = 1e-4
    save_dir: str = "./output/injection"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train a Gaussian SH injection network conditioned on a CLIP embedding.")
    parser.add_argument("--gaussian_ply", type=str, default="", help="Path to the reconstructed Gaussian .ply file.")
    parser.add_argument("--target_ply", type=str, default="", help="Optional .ply file with target SH values for supervised training.")
    parser.add_argument("--text", type=str, default="", help="Text prompt for CLIP conditioning.")
    parser.add_argument("--image", type=str, default="", help="Single image path for CLIP conditioning.")
    parser.add_argument("--image_dir", type=str, default="", help="Directory containing images; one condition per image.")
    parser.add_argument("--condition_mode", type=str, choices=["text", "image"], default="text", help="Source of the CLIP condition vector.")
    parser.add_argument("--clip_model", type=str, default="openai/clip-vit-base-patch32", help="CLIP model name.")
    parser.add_argument("--condition_dim", type=int, default=512, help="Dimension of the CLIP embedding.")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension for the injection network.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--save_dir", type=str, default="./output/injection", help="Directory to save the trained injection network.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Training device, e.g. cuda or cpu.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return TrainConfig(**vars(parser.parse_args()))


def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_gaussian_model(path: str, device: str) -> GaussianModel:
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Gaussian PLY not found: {path}")

    model = GaussianModel(sh_degree=3)
    model.load_ply(path)
    model.to(device)
    model.freeze_geometry()
    return model


def build_condition_vector(
    encoder: CLIPEncoder,
    gaussian_model: GaussianModel,
    config: TrainConfig,
) -> torch.Tensor:
    if config.condition_mode == "text":
        if not config.text:
            raise ValueError("--text must be provided when --condition_mode text is used.")
        cond = encoder.encode_text(config.text)
    else:
        if config.image:
            cond = encoder.encode_image(config.image)
        elif config.image_dir:
            # Use the first image in the directory if multiple images are not provided.
            files = sorted(
                [os.path.join(config.image_dir, f) for f in os.listdir(config.image_dir) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp"))]
            )
            if not files:
                raise FileNotFoundError(f"No image files found in directory: {config.image_dir}")
            cond = encoder.encode_image(files[0])
        else:
            raise ValueError("Either --image or --image_dir must be provided when --condition_mode image is used.")

    cond = cond.to(device=config.device)
    if cond.dim() == 1:
        cond = cond.unsqueeze(0)

    num_gaussians = gaussian_model.get_features.shape[0]
    if cond.shape[0] == 1 and num_gaussians > 1:
        cond = cond.expand(num_gaussians, -1)

    if cond.shape[0] != num_gaussians:
        cond = cond[:num_gaussians]

    return cond


def get_target_features(model: GaussianModel, target_path: Optional[str], device: str) -> torch.Tensor:
    if target_path and os.path.exists(target_path):
        target_model = GaussianModel(sh_degree=3)
        target_model.load_ply(target_path)
        target_model.to(device)
        target_model.freeze_geometry()
        return target_model.get_features.detach().to(device)
    return model.get_features.detach().to(device)


def main():
    config = parse_args()
    set_seed(config.seed)
    device = torch.device(config.device)

    if not config.gaussian_ply:
        raise ValueError("--gaussian_ply is required. Provide a reconstructed Gaussian .ply file.")

    gaussian_model = load_gaussian_model(config.gaussian_ply, str(device))
    source_features = gaussian_model.get_features.detach().to(device)
    target_features = get_target_features(gaussian_model, config.target_ply, str(device))

    encoder = CLIPEncoder(model_name=config.clip_model, device=str(device))
    condition = build_condition_vector(encoder, gaussian_model, config)

    net = ConditionedGaussianSHNet(
        condition_dim=config.condition_dim,
        sh_channels=3,
        hidden_dim=config.hidden_dim,
    ).to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=config.learning_rate)

    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(config.epochs):
        net.train()
        optimizer.zero_grad()

        predicted = net(source_features, condition)
        loss = F.mse_loss(predicted, target_features)

        loss.backward()
        optimizer.step()

        print(f"epoch={epoch + 1}/{config.epochs} loss={loss.item():.6f}")

    checkpoint_path = save_dir / "injection_network.pt"
    torch.save(net.state_dict(), checkpoint_path)
    print(f"Saved injection network to: {checkpoint_path}")


if __name__ == "__main__":
    main()
