''' train_base_gaussians.py

Evaluate rendered 'base gaussians' with input images by L1 Loss, SSIM
and optimize gaussians.

CLI entry point for later extension.
'''

import argparse
import importlib
import os
import sys
import logging
import torch

from tqdm import tqdm
from accelerate import Accelerator
from datasets.colmap_loader import ColmapLoader


def _load_gsrecon_class():
    """Load GSRecon from src.models if available, otherwise fall back to local model."""
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
            if hasattr(module, "GSRecon"):
                return getattr(module, "GSRecon")
            if hasattr(module, "GaussianModel"):
                return getattr(module, "GaussianModel")
        except ModuleNotFoundError:
            continue

    return None


def parse_args():
    parser = argparse.ArgumentParser(description="Train base gaussians from a COLMAP reconstruction.")

    # Basic placeholders, to be extended gradually
    parser.add_argument("--colmap_dir", type=str, default="data/colmap", help="Path to COLMAP sparse folder.")
    parser.add_argument("--data_dir", type=str, default=None, help="Dataset root or image directory.")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory for checkpoint/output.")
    parser.add_argument("--max_train_steps", type=int, default=1000, help="Maximum training steps")
    parser.add_argument("--epochs", type=int, default=1, help="Trainng epochs (placeholder).")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate (placeholder).")
    parser.add_argument("--device", type=str, default="cuda", help="Training device, e.g. cuda or cpu.")
    parser.add_argument("--seed", type=int, default=43, help="Random seed.")
    parser.add_argument("--load_pretrained_model", action="store_true", help="Load pretrained model if available.")

    return parser.parse_args()


def main():
    args = parse_args()
    accelerator = Accelerator()
    
    train_loader = None  # Placeholder for future dataset loader
    val_loader = None    # Placeholder for future validation dataset loader
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO
    )
    logger = logging.getLogger(__name__)

    colmap_dir = args.colmap_dir
    if not os.path.exists(colmap_dir):
        raise FileNotFoundError(f"COLMAP directory not found: {colmap_dir}")

    # Load reconstruction data
    colmap_loader = ColmapLoader(colmap_dir)
    camera_poses = colmap_loader.get_camera_poses()
    xyz, rgb = colmap_loader.get_initial_point_cloud()

    print(f"Loaded {len(camera_poses)} camera poses from {colmap_dir}")
    print(f"Point cloud shape: {xyz.shape}, rgb shape: {rgb.shape}")

    # GSRecon import and model creation (placeholder API)
    GSRecon = _load_gsrecon_class()
    if GSRecon is not None:
        try:
            model = GSRecon()
            print(f"Initialized model: {type(model).__name__}")
        except TypeError:
            model = GSRecon
            print(f"Loaded GSRecon class: {GSRecon.__name__}")
    else:
        # Keep the script runnable even before the upstream model is integrated.
        print("GSRecon not found yet. Using a minimal placeholder path.")
        model = None
        
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate) if model is not None else None

    # Placeholder training entry; this will be filled in progressively.
    # Example future logic:
    # model = GSRecon(...)
    if args.load_pretrained_model is not None:
        # Load pretrained model here
        logger.info(f"Loading GSRecon checkpoint from [{args.load_pretrained_model}]")
        model = model.load_from_checkpoint(args.load_pretrained_model)
    
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )
    
    logger.info(f"Starting training for {args.epochs} epochs with learning rate {args.learning_rate}")
    logger.info(f"Maximum training steps: {args.max_train_steps}")
    logger.info(f"Training device: {args.device}, Random seed: {args.seed}")
    
    # Start training
    global_update_step = 0
    logger.logger.propagate = False
    progress_bar = tqdm(
        range(args.max_train_steps),
        initial=global_update_step,
        desc="Training",
        ncols=100,
        disable=False
    )
    
    def compute_loss(outputs, batch):
        # Placeholder for loss computation logic
        # This should compute the loss between model outputs and ground truth from batch
        return torch.tensor(0.0, requires_grad=True)  # Dummy loss for placeholder 
    
    for epoch in range(args.epochs):
        
        if global_update_step >= args.max_train_steps:
            progress_bar.close()
            logger.logger.propagate = True
            logger.info("Training completed.\n")
            return
    
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            outputs=model(batch)
            loss = compute_loss(outputs, batch) # Placeholder for loss computation
            accelerator.backward(loss)
            optimizer.step()
            # lr_scheduler.step()  # Placeholder for learning rate scheduler if used
            optimizer.zero_grad()
            
            global_update_step += 1
            progress_bar.update(1)           
        
        
        
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())