from __future__ import annotations

import os
from typing import Iterable, List, Optional, Sequence, Union

import torch
from PIL import Image

try:
    import open_clip
except Exception:  # pragma: no cover - optional dependency
    open_clip = None

try:
    from transformers import CLIPModel, CLIPProcessor
except Exception:  # pragma: no cover - optional dependency
    CLIPModel = None
    CLIPProcessor = None


class CLIPEncoder:
    """Thin wrapper for a CLIP text/image encoder.

    It tries to use `open_clip` first and falls back to Hugging Face `transformers`.
    The returned embedding is a normalized torch tensor of shape (B, D) for text or image.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: Optional[str] = None,
        backend: str = "auto",
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.backend = self._resolve_backend(backend)

        if self.backend == "open_clip":
            if open_clip is None:
                raise ImportError(
                    "open_clip is not installed. Run: pip install open_clip_torch"
                )
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name,
                device=self.device,
            )
            self.model.to(self.device)
            self.model.eval()
            self.tokenizer = open_clip.get_tokenizer(model_name)
            self.feature_dim = getattr(self.model, "text_projection", None)
            if self.feature_dim is not None:
                self.feature_dim = self.feature_dim.shape[-1]
            else:
                self.feature_dim = 512

        elif self.backend == "transformers":
            if CLIPModel is None or CLIPProcessor is None:
                raise ImportError(
                    "transformers is not installed. Run: pip install transformers pillow"
                )
            self.model = CLIPModel.from_pretrained(model_name)
            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            self.tokenizer = self.processor.tokenizer
            self.feature_dim = self.model.config.projection_dim

        else:
            raise ImportError(
                "No supported CLIP backend is installed. Install either `open_clip_torch` or `transformers`."
            )

    def _resolve_backend(self, backend: str) -> str:
        if backend != "auto":
            return backend

        if open_clip is not None:
            return "open_clip"
        if CLIPModel is not None and CLIPProcessor is not None:
            return "transformers"
        return "none"

    @torch.no_grad()
    def encode_text(self, texts: Union[str, Sequence[str]]) -> torch.Tensor:
        if isinstance(texts, str):
            texts = [texts]

        if self.backend == "open_clip":
            tokens = self.tokenizer(list(texts)).to(self.device)
            embeddings = self.model.encode_text(tokens)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            return embeddings

        text_inputs = self.processor(text=list(texts), return_tensors="pt", padding=True)
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
        embeddings = self.model.get_text_features(**text_inputs)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        return embeddings

    @torch.no_grad()
    def encode_image(self, image: Union[str, Image.Image, torch.Tensor]) -> torch.Tensor:
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        if isinstance(image, torch.Tensor):
            if image.dim() == 3:
                image = image.unsqueeze(0)
            if image.dtype != torch.float32:
                image = image.float()
            image = image.to(self.device)
            if self.backend == "open_clip":
                embeddings = self.model.encode_image(image)
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
                return embeddings
            raise NotImplementedError("Tensor image input is supported only for open_clip backend.")

        if self.backend == "open_clip":
            processed = self.preprocess(image).unsqueeze(0).to(self.device)
            embeddings = self.model.encode_image(processed)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            return embeddings

        processed = self.processor(images=image, return_tensors="pt")
        processed = {k: v.to(self.device) for k, v in processed.items()}
        embeddings = self.model.get_image_features(**processed)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        return embeddings


def encode_text(texts: Union[str, Sequence[str]], model_name: str = "openai/clip-vit-base-patch32") -> torch.Tensor:
    encoder = CLIPEncoder(model_name=model_name)
    return encoder.encode_text(texts)


def encode_image(image: Union[str, Image.Image, torch.Tensor], model_name: str = "openai/clip-vit-base-patch32") -> torch.Tensor:
    encoder = CLIPEncoder(model_name=model_name)
    return encoder.encode_image(image)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Encode text or an image with a CLIP model.")
    parser.add_argument("--text", nargs="*", default=None, help="Text prompt(s) to encode.")
    parser.add_argument("--image", type=str, default=None, help="Path to an input image.")
    parser.add_argument("--model", default="openai/clip-vit-base-patch32", help="CLIP model name.")
    args = parser.parse_args()

    encoder = CLIPEncoder(model_name=args.model)
    if args.text:
        emb = encoder.encode_text(args.text)
        print(emb.shape)
    if args.image:
        emb = encoder.encode_image(args.image)
        print(emb.shape)
