from pathlib import Path
from typing import Any, Dict, List
import json

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
import yaml

from models.gaussians_model import GaussianModel


NUM_VIEWS = 5


class GaussianInjectionDataset(Dataset):
    """Dataset where exactly five input views define one reconstruction scene."""

    def __init__(
        self,
        dataset_yaml: str,
        sh_degree: int = 3,
        device: str = "cpu",
        validate_paths: bool = True,
    ):
        self.dataset_yaml = Path(dataset_yaml).resolve()
        self.device = torch.device(device)
        self.sh_degree = sh_degree

        with self.dataset_yaml.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file) or {}

        dataset_config = config.get("dataset", {})
        configured_num_views = int(dataset_config.get("num_views", NUM_VIEWS))
        if configured_num_views != NUM_VIEWS:
            raise ValueError(
                f"This dataset requires exactly {NUM_VIEWS} views per scene, "
                f"got num_views={configured_num_views}"
            )

        root = Path(dataset_config.get("root", "."))
        if not root.is_absolute():
            root = self.dataset_yaml.parent / root
        self.root = root.resolve()

        captions_value = dataset_config.get("captions_json", "")
        self.captions = {}
        if captions_value:
            captions_path = self._resolve_path(captions_value)
            if not captions_path.is_file():
                raise FileNotFoundError(f"Captions JSON not found: {captions_path}")
            with captions_path.open("r", encoding="utf-8") as file:
                self.captions = json.load(file)
            if not isinstance(self.captions, dict):
                raise ValueError("dataset.captions_json must contain a JSON object")

        samples = dataset_config.get("samples", [])
        if not samples:
            raise ValueError("dataset.samples must contain at least one sample")
        self.samples = samples

        validation = dataset_config.get("validation", {})
        self._validate_sample_schema(validation)
        if validate_paths:
            self._validate_paths()

        self.sample_source_features = [
            self._load_features(self._resolve_path(sample["source_gaussian_ply"]))
            for sample in self.samples
        ]

    def _resolve_path(self, value: str) -> Path:
        path = Path(value)
        if not path.is_absolute():
            path = self.root / path
        if not path.exists() and str(path).startswith("/home/student_2/"):
            container_path = Path("/workspace") / path.relative_to("/home/student_2")
            if container_path.exists():
                path = container_path
        return path.resolve()

    def _validate_sample_schema(self, validation: Dict[str, Any]) -> None:
        for index, sample in enumerate(self.samples):
            if not isinstance(sample, dict):
                raise ValueError(f"Dataset sample {index} must be a mapping")
            views = sample.get("views")
            if not isinstance(views, list) or len(views) != NUM_VIEWS:
                raise ValueError(
                    f"Dataset sample {sample.get('id', index)!r} must contain exactly "
                    f"{NUM_VIEWS} views"
                )
            if not sample.get("source_gaussian_ply"):
                raise ValueError(
                    f"Dataset sample {sample.get('id', index)!r} must provide "
                    "source_gaussian_ply"
                )
            camera_ids = sample.get("camera_ids")
            if camera_ids is not None and len(camera_ids) != NUM_VIEWS:
                raise ValueError(
                    f"Dataset sample {sample.get('id', index)!r} must provide exactly "
                    f"{NUM_VIEWS} camera_ids when camera_ids is specified"
                )
            missing = []
            if validation.get("require_prompt", validation.get("require_caption", False)):
                prompt = sample.get("prompt")
                prompt_key = sample.get("prompt_key")
                if not prompt and prompt_key:
                    prompt = self.captions.get(prompt_key)
                if not isinstance(prompt, str) or not prompt.strip():
                    missing.append("prompt")
            if missing:
                raise ValueError(
                    f"Dataset sample {sample.get('id', index)!r} is missing: {', '.join(missing)}"
                )
            for view_index, view in enumerate(views):
                if not isinstance(view, dict) or not view.get("input_image"):
                    raise ValueError(
                        f"Dataset sample {sample.get('id', index)!r} view {view_index} "
                        "must provide input_image"
                    )

    def _validate_paths(self) -> None:
        paths = []
        for index, sample in enumerate(self.samples):
            sample_id = sample.get("id", index)
            paths.append(
                (
                    f"sample {sample_id} source_gaussian_ply",
                    self._resolve_path(sample["source_gaussian_ply"]),
                )
            )
            for view_index, view in enumerate(sample["views"]):
                paths.append(
                    (
                        f"sample {sample_id} view {view_index} input_image",
                        self._resolve_path(view["input_image"]),
                    )
                )
            if sample.get("camera_json"):
                paths.append((f"sample {sample_id} camera_json", self._resolve_path(sample["camera_json"])))

        missing = [f"{name}: {path}" for name, path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError("Dataset files not found:\n" + "\n".join(missing))

    def _load_features(self, path: Path) -> torch.Tensor:
        model = GaussianModel(sh_degree=self.sh_degree)
        model.load_ply(path)
        features = model.get_features.detach().to(device=self.device, dtype=torch.float32)
        if features.ndim != 3 or features.shape[1:] != ((self.sh_degree + 1) ** 2, 3):
            raise ValueError(
                f"Unexpected SH feature shape in {path}: {tuple(features.shape)}"
            )
        if not torch.isfinite(features).all():
            raise ValueError(f"SH features contain NaN or infinity: {path}")
        return features.cpu()

    def _load_target_image(self, path: Path) -> torch.Tensor:
        with Image.open(path) as image:
            image_rgb = image.convert("RGB")
            image_tensor = torch.from_numpy(
                np.array(image_rgb, dtype="float32")
            ).permute(2, 0, 1).contiguous() / 255.0
        if not torch.isfinite(image_tensor).all():
            raise ValueError(f"Pseudo-GT image contains NaN or infinity: {path}")
        return image_tensor

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.samples[index]
        prompt = sample.get("prompt")
        prompt_key = sample.get("prompt_key")
        if not prompt and prompt_key:
            prompt = self.captions.get(prompt_key)
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"Sample {sample.get('id', index)!r} has no text prompt")
        input_images = [
            str(self._resolve_path(view["input_image"])) for view in sample["views"]
        ]
        target_images = [
            self._load_target_image(Path(image_path))
            for image_path in input_images
        ]
        return {
            "id": sample.get("id", str(index)),
            "source_gaussian_ply": str(
                self._resolve_path(sample["source_gaussian_ply"])
            ),
            "source_features": self.sample_source_features[index],
            "views": input_images,
            "input_images": input_images,
            "input_image": input_images[0],
            "target_images": target_images,
            "camera_json": (
                str(self._resolve_path(sample["camera_json"]))
                if sample.get("camera_json") else None
            ),
            "camera_ids": sample.get("camera_ids"),
            "prompt": prompt.strip(),
        }


def injection_collate_fn(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep samples as a list because Gaussian counts may differ between samples."""
    return batch


def build_injection_dataloader(
    dataset_yaml: str,
    sh_degree: int = 3,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 0,
    device: str = "cpu",
    validate_paths: bool = True,
) -> DataLoader:
    dataset = GaussianInjectionDataset(
        dataset_yaml=dataset_yaml,
        sh_degree=sh_degree,
        device=device,
        validate_paths=validate_paths,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=injection_collate_fn,
        pin_memory=device.startswith("cuda"),
    )


def encode_prompt_batch(batch: List[Dict[str, Any]], encoder, device: str) -> torch.Tensor:
    """Encode one text transformation prompt per scene."""
    prompts = [sample["prompt"] for sample in batch]
    return encoder.encode_text(prompts).to(device=device, dtype=torch.float32)
