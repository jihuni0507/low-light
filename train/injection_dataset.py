from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader, Dataset
import yaml

from models.gaussians_model import GaussianModel


class GaussianInjectionDataset(Dataset):
    """Dataset of source Gaussian features, target Gaussian features, and prompts."""

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
        root = Path(dataset_config.get("root", "."))
        if not root.is_absolute():
            root = self.dataset_yaml.parent / root
        self.root = root.resolve()

        source_value = dataset_config.get("source_gaussian_ply", "")
        if not source_value:
            raise ValueError("dataset.source_gaussian_ply must be provided")
        self.source_gaussian_ply = self._resolve_path(source_value)

        samples = dataset_config.get("samples", [])
        if not samples:
            raise ValueError("dataset.samples must contain at least one sample")
        self.samples = samples

        validation = dataset_config.get("validation", {})
        self._validate_sample_schema(validation)
        if validate_paths:
            self._validate_paths()

        self.source_features = self._load_features(self.source_gaussian_ply)
        self._target_features: Dict[int, torch.Tensor] = {}
        for index, sample in enumerate(self.samples):
            target_features = self._load_features(self._resolve_path(sample["target_gs"]))
            if target_features.shape != self.source_features.shape:
                raise ValueError(
                    f"Sample {sample.get('id', index)!r} has incompatible SH shape: "
                    f"source={tuple(self.source_features.shape)}, "
                    f"target={tuple(target_features.shape)}"
                )
            self._target_features[index] = target_features

    def _resolve_path(self, value: str) -> Path:
        path = Path(value)
        if not path.is_absolute():
            path = self.root / path
        return path.resolve()

    def _validate_sample_schema(self, validation: Dict[str, Any]) -> None:
        required_fields = []
        if validation.get("require_input_image", True):
            required_fields.append("input_image")
        if validation.get("require_target_gs", True):
            required_fields.append("target_gs")
        if validation.get("require_prompt", True):
            required_fields.append("prompt")

        for index, sample in enumerate(self.samples):
            if not isinstance(sample, dict):
                raise ValueError(f"Dataset sample {index} must be a mapping")
            missing = [field for field in required_fields if not sample.get(field)]
            if missing:
                raise ValueError(
                    f"Dataset sample {sample.get('id', index)!r} is missing: {', '.join(missing)}"
                )

    def _validate_paths(self) -> None:
        paths = [("source_gaussian_ply", self.source_gaussian_ply)]
        for index, sample in enumerate(self.samples):
            sample_id = sample.get("id", index)
            paths.append((f"sample {sample_id} input_image", self._resolve_path(sample["input_image"])))
            paths.append((f"sample {sample_id} target_gs", self._resolve_path(sample["target_gs"])))

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

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.samples[index]
        return {
            "id": sample.get("id", str(index)),
            "source_features": self.source_features,
            "target_features": self._target_features[index],
            "input_image": str(self._resolve_path(sample["input_image"])),
            "target_gs": str(self._resolve_path(sample["target_gs"])),
            "prompt": sample["prompt"],
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
