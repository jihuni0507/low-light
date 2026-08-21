from pathlib import Path
from typing import Any, Dict, List
import json

import torch
from torch.utils.data import DataLoader, Dataset
import yaml

from models.gaussians_model import GaussianModel


NUM_VIEWS = 3


class GaussianInjectionDataset(Dataset):
    """Dataset where exactly three input views define one reconstruction scene."""

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

        source_value = dataset_config.get("source_gaussian_ply", "")
        if not source_value:
            raise ValueError("dataset.source_gaussian_ply must be provided")
        self.source_gaussian_ply = self._resolve_path(source_value)

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

        self.source_features = self._load_features(self.source_gaussian_ply)
        self.source_geometry = self._load_geometry(self.source_gaussian_ply)
        self._target_features: Dict[int, torch.Tensor] = {}
        for index, sample in enumerate(self.samples):
            target_path = self._resolve_path(sample["target_gs"])
            target_features = self._load_features(target_path)
            if target_features.shape != self.source_features.shape:
                raise ValueError(
                    f"Sample {sample.get('id', index)!r} has incompatible SH shape: "
                    f"source={tuple(self.source_features.shape)}, "
                    f"target={tuple(target_features.shape)}"
                )
            target_geometry = self._load_geometry(target_path)
            for name in self.source_geometry:
                if not torch.allclose(self.source_geometry[name], target_geometry[name], atol=1e-5, rtol=1e-5):
                    raise ValueError(
                        f"Sample {sample.get('id', index)!r} target geometry does not match "
                        f"source geometry in '{name}'"
                    )
            self._target_features[index] = target_features

    def _resolve_path(self, value: str) -> Path:
        path = Path(value)
        if not path.is_absolute():
            path = self.root / path
        return path.resolve()

    def _validate_sample_schema(self, validation: Dict[str, Any]) -> None:
        required_fields = []
        if validation.get("require_target_gs", True):
            required_fields.append("target_gs")

        for index, sample in enumerate(self.samples):
            if not isinstance(sample, dict):
                raise ValueError(f"Dataset sample {index} must be a mapping")
            missing = [field for field in required_fields if not sample.get(field)]
            if validation.get("require_caption", False):
                caption_key = sample.get("caption_key", sample.get("id", index))
                if caption_key not in self.captions and not sample.get("prompt"):
                    missing.append("caption_key")
            if missing:
                raise ValueError(
                    f"Dataset sample {sample.get('id', index)!r} is missing: {', '.join(missing)}"
                )
            views = sample.get("views")
            if not isinstance(views, list) or len(views) != NUM_VIEWS:
                raise ValueError(
                    f"Dataset sample {sample.get('id', index)!r} must contain exactly "
                    f"{NUM_VIEWS} views"
                )
            for view_index, view in enumerate(views):
                if not isinstance(view, dict) or not view.get("input_image"):
                    raise ValueError(
                        f"Dataset sample {sample.get('id', index)!r} view {view_index} "
                        "must provide input_image"
                    )

    def _validate_paths(self) -> None:
        paths = [("source_gaussian_ply", self.source_gaussian_ply)]
        for index, sample in enumerate(self.samples):
            sample_id = sample.get("id", index)
            for view_index, view in enumerate(sample["views"]):
                paths.append(
                    (
                        f"sample {sample_id} view {view_index} input_image",
                        self._resolve_path(view["input_image"]),
                    )
                )
            paths.append((f"sample {sample_id} target_gs", self._resolve_path(sample["target_gs"])))
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

    def _load_geometry(self, path: Path) -> Dict[str, torch.Tensor]:
        model = GaussianModel(sh_degree=self.sh_degree)
        model.load_ply(path)
        return {
            name: value.detach().cpu().clone()
            for name, value in model.get_geometry().items()
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.samples[index]
        caption_key = sample.get("caption_key", sample.get("id", str(index)))
        prompt = self.captions.get(caption_key, sample.get("prompt"))
        if not prompt:
            raise ValueError(
                f"Sample {sample.get('id', index)!r} has no caption for key {caption_key!r}"
            )
        input_images = [
            str(self._resolve_path(view["input_image"])) for view in sample["views"]
        ]
        return {
            "id": sample.get("id", str(index)),
            "source_features": self.source_features,
            "target_features": self._target_features[index],
            "views": input_images,
            "input_images": input_images,
            "input_image": input_images[0],
            "target_gs": str(self._resolve_path(sample["target_gs"])),
            "camera_json": (
                str(self._resolve_path(sample["camera_json"]))
                if sample.get("camera_json") else None
            ),
            "camera_ids": sample.get("camera_ids"),
            "prompt": prompt,
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
    """Encode all prompts in one DataLoader batch with the shared CLIP encoder."""
    prompts = [sample["prompt"] for sample in batch]
    return encoder.encode_text(prompts).to(device=device, dtype=torch.float32)
