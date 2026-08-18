import argparse
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from accelerate import Accelerator

from models.clip_encoder import CLIPEncoder
from models.gaussians_model import GaussianModel
from models.injection_network import ConditionedGaussianSHNet
from train.train_base_gaussians import TrainConfig as BaseTrainConfig
from train.train_base_gaussians import build_dataloader, build_model, load_colmap_data, train_loop


def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    return config


def resolve_path(path_value, root_dir):
    if not path_value:
        return ""
    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    return str((root_dir / path).resolve())


def run_base_training(config, root_dir):
    train_cfg = config.get("train", {})
    base_cfg = {
        "colmap_dir": resolve_path(train_cfg.get("colmap_dir", "datasets/colmap"), root_dir),
        "data_dir": resolve_path(train_cfg.get("data_dir", ""), root_dir),
        "output_dir": resolve_path(train_cfg.get("output_dir", "./output"), root_dir),
        "max_train_steps": int(train_cfg.get("max_train_steps", 1000)),
        "epochs": int(train_cfg.get("epochs", 1)),
        "learning_rate": float(train_cfg.get("learning_rate", 1e-3)),
        "batch_size": int(train_cfg.get("batch_size", 1)),
        "num_workers": int(train_cfg.get("num_workers", 0)),
        "device": train_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
        "seed": int(train_cfg.get("seed", 43)),
        "load_pretrained_model": bool(train_cfg.get("load_pretrained_model", False)),
        "model_name": train_cfg.get("model_name", "GSRecon"),
    }

    base_config = BaseTrainConfig(**base_cfg)
    recon_data = load_colmap_data(base_config.colmap_dir)
    loader = build_dataloader(base_config, recon_data)
    model = build_model(base_config, recon_data)
    accelerator = Accelerator(device_placement=True, mixed_precision="no")
    train_loop(base_config, model, loader, accelerator)
    return model


def encode_clip_condition(config, root_dir):
    clip_cfg = config.get("clip", {})
    device = config.get("global", {}).get("device", "cuda" if torch.cuda.is_available() else "cpu")
    encoder = CLIPEncoder(model_name=clip_cfg.get("model_name", "openai/clip-vit-base-patch32"), device=device)

    text_prompt = clip_cfg.get("text_prompt", "")
    image_path = resolve_path(clip_cfg.get("image_path", ""), root_dir)

    if image_path and os.path.exists(image_path):
        cond = encoder.encode_image(image_path)
    elif text_prompt:
        cond = encoder.encode_text(text_prompt)
    else:
        raise ValueError("CLIP config requires either text_prompt or image_path.")

    return cond.squeeze(0).to(device)


def run_injection_training(config, root_dir, condition_vec):
    injection_cfg = config.get("injection", {})
    train_cfg = config.get("train", {})
    device = config.get("global", {}).get("device", "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    sh_degree = int(train_cfg.get("sh_degree", 3))

    gaussian_ply = resolve_path(injection_cfg.get("gaussian_ply", "output/base_gaussian_last.pt"), root_dir)
    if gaussian_ply.endswith(".pt"):
        raise ValueError("Injection pipeline expects a Gaussian .ply file, not a checkpoint .pt file.")
    if not os.path.exists(gaussian_ply):
        raise FileNotFoundError(f"Gaussian PLY file not found: {gaussian_ply}")

    model = GaussianModel(sh_degree=sh_degree)
    model.load_ply(gaussian_ply)
    model.to(device)
    model.freeze_geometry()

    source_features = model.get_features.to(device)
    target_path = resolve_path(injection_cfg.get("target_ply", ""), root_dir)
    if target_path and os.path.exists(target_path):
        target_model = GaussianModel(sh_degree=sh_degree)
        target_model.load_ply(target_path)
        target_model.to(device)
        target_features = target_model.get_features.to(device)
    else:
        target_features = source_features.clone()

    net = ConditionedGaussianSHNet(
        condition_dim=condition_vec.shape[-1],
        sh_channels=3,
        hidden_dim=int(injection_cfg.get("hidden_dim", 256)),
    ).to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=float(injection_cfg.get("learning_rate", 1e-4)))

    for epoch in range(int(injection_cfg.get("epochs", 10))):
        net.train()
        optimizer.zero_grad()
        predicted = net(source_features, condition_vec.unsqueeze(0).expand(source_features.shape[0], -1))
        loss = F.mse_loss(predicted, target_features)
        loss.backward()
        optimizer.step()
        print(f"epoch={epoch + 1}/{injection_cfg.get('epochs', 10)} loss={loss.item():.6f}")

    updated = net(source_features, condition_vec.unsqueeze(0).expand(source_features.shape[0], -1))
    model.set_features(updated[:, :1, :], updated[:, 1:, :])

    output_ply = resolve_path(injection_cfg.get("output_ply", "output/injection_updated_gaussians.ply"), root_dir)
    model.save_ply(output_ply)
    print(f"Updated Gaussian SH features saved to: {output_ply}")

    return model, net


def build_parser():
    parser = argparse.ArgumentParser(description="YAML-driven pipeline orchestrator for Gaussian reconstruction and CLIP-conditioned SH injection.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the pipeline YAML file.")
    parser.add_argument("--stage", type=str, default="all", choices=["all", "base", "clip", "injection"], help="Pipeline stage to run.")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    root_dir = Path(__file__).resolve().parent
    config_path = root_dir / args.config

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = load_config(str(config_path))
    pipeline_cfg = config.get("pipeline", {})

    if args.stage in ("all", "base") and pipeline_cfg.get("run_train_base", False):
        print("[1/3] Running base Gaussian training...")
        run_base_training(config, root_dir)

    if args.stage in ("all", "clip") and pipeline_cfg.get("run_clip_encode", False):
        print("[2/3] Encoding CLIP condition vector...")
        condition_vec = encode_clip_condition(config, root_dir)
        print("CLIP condition shape:", tuple(condition_vec.shape))
    else:
        condition_vec = None

    if args.stage in ("all", "injection") and pipeline_cfg.get("run_injection", False):
        if condition_vec is None:
            condition_vec = encode_clip_condition(config, root_dir)
        print("[3/3] Running SH injection training...")
        run_injection_training(config, root_dir, condition_vec)

    print("Pipeline complete.")


if __name__ == "__main__":
    main()