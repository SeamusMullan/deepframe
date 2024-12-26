"""Depth Anything V2 model wrapper."""

from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .base import DepthEstimator, ModelInfo, normalize_depth


class DepthAnythingEstimator(DepthEstimator):
    """Depth Anything V2 depth estimation model."""

    VARIANTS = {
        "vits": {
            "encoder": "vits",
            "display_name": "Depth Anything V2 ViT-S",
            "description": "Small model. Fast with good quality.",
            "memory_mb": 400,
            "input_size": (518, 518),
            "features": 64,
            "out_channels": [48, 96, 192, 384],
            "hf_repo": "depth-anything/Depth-Anything-V2-Small",
            "hf_filename": "depth_anything_v2_vits.pth",
        },
        "vitb": {
            "encoder": "vitb",
            "display_name": "Depth Anything V2 ViT-B",
            "description": "Base model. Better quality.",
            "memory_mb": 800,
            "input_size": (518, 518),
            "features": 128,
            "out_channels": [96, 192, 384, 768],
            "hf_repo": "depth-anything/Depth-Anything-V2-Base",
            "hf_filename": "depth_anything_v2_vitb.pth",
        },
        "vitl": {
            "encoder": "vitl",
            "display_name": "Depth Anything V2 ViT-L",
            "description": "Large model. Best quality.",
            "memory_mb": 1500,
            "input_size": (518, 518),
            "features": 256,
            "out_channels": [256, 512, 1024, 1024],
            "hf_repo": "depth-anything/Depth-Anything-V2-Large",
            "hf_filename": "depth_anything_v2_vitl.pth",
        },
    }

    def __init__(
        self,
        variant: Literal["vits", "vitb", "vitl"] = "vits",
        device: torch.device | None = None,
    ) -> None:
        super().__init__(device)
        self.variant = variant
        self._variant_info = self.VARIANTS[variant]

    @property
    def model_info(self) -> ModelInfo:
        info = self._variant_info
        return ModelInfo(
            name=f"depth_anything_{self.variant}",
            display_name=info["display_name"],
            description=info["description"],
            memory_mb=info["memory_mb"],
            supports_batching=True,
            default_input_size=info["input_size"],
        )

    def load(self) -> None:
        """Load the Depth Anything V2 model."""
        if self._loaded:
            return

        # Build model architecture
        self._model = self._build_model()

        # Download and load weights
        weights_path = self._download_weights()

        state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)
        self._model.load_state_dict(state_dict)

        self._model.to(self.device)
        self._model.eval()
        self._loaded = True

    def _build_model(self):
        """Build the Depth Anything V2 model architecture."""
        from .dpt import DepthAnythingV2

        info = self._variant_info
        return DepthAnythingV2(
            encoder=info["encoder"],
            features=info["features"],
            out_channels=info["out_channels"],
        )

    def _download_weights(self) -> Path:
        """Download model weights from HuggingFace Hub."""
        from huggingface_hub import hf_hub_download

        info = self._variant_info
        cache_dir = Path.home() / ".cache" / "deepframe" / "models"
        cache_dir.mkdir(parents=True, exist_ok=True)

        weights_path = cache_dir / info["hf_filename"]

        if not weights_path.exists():
            downloaded = hf_hub_download(
                repo_id=info["hf_repo"],
                filename=info["hf_filename"],
                local_dir=cache_dir,
            )
            weights_path = Path(downloaded)

        return weights_path

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
        self._loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def estimate(self, frame: np.ndarray) -> np.ndarray:
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        original_h, original_w = frame.shape[:2]
        input_tensor = self._preprocess(frame)

        with torch.no_grad():
            depth = self._model(input_tensor)
            depth = F.interpolate(
                depth.unsqueeze(1),
                size=(original_h, original_w),
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_np = depth.cpu().numpy()
        return normalize_depth(depth_np, invert=False, clip_percentile=1.0)

    def estimate_batch(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        if len(frames) == 0:
            return []

        original_h, original_w = frames[0].shape[:2]
        input_tensors = [self._preprocess(frame) for frame in frames]
        input_batch = torch.cat(input_tensors, dim=0)

        with torch.no_grad():
            depths = self._model(input_batch)
            depths = F.interpolate(
                depths.unsqueeze(1),
                size=(original_h, original_w),
                mode="bicubic",
                align_corners=False,
            ).squeeze(1)

        results = []
        for i in range(len(frames)):
            depth_np = depths[i].cpu().numpy()
            results.append(normalize_depth(depth_np, invert=False, clip_percentile=1.0))

        return results

    def _preprocess(self, frame: np.ndarray) -> torch.Tensor:
        input_size = self._variant_info["input_size"]
        resized = cv2.resize(frame, input_size, interpolation=cv2.INTER_LINEAR)

        tensor = torch.from_numpy(resized).float() / 255.0
        tensor = tensor.permute(2, 0, 1)

        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor - mean) / std

        return tensor.unsqueeze(0).to(self.device)
