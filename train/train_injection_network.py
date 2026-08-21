import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from models.clip_encoder import CLIPEncoder
from models.injection_network import ConditionedGaussianSHNet
from train.injection_dataset import build_injection_dataloader, encode_prompt_batch


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
    return TrainConfig(**vars(parser.parse_args()))


def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
                batch_loss = batch_loss + F.mse_loss(predicted, target_features)

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
