"""MiDaS depth estimation model wrapper."""

from typing import Literal

import cv2
import numpy as np
import torch

from .base import DepthEstimator, ModelInfo, normalize_depth


class MiDaSEstimator(DepthEstimator):
    """MiDaS depth estimation using PyTorch Hub."""

    # Available MiDaS model variants
    VARIANTS = {
        "small": {
            "hub_name": "MiDaS_small",
            "transform_type": "small",
            "display_name": "MiDaS Small",
            "description": "Fast, lower quality. Good for real-time preview.",
            "memory_mb": 200,
            "input_size": (256, 256),
        },
        "large": {
            "hub_name": "DPT_Large",
            "transform_type": "dpt",
            "display_name": "MiDaS DPT-Large",
            "description": "High quality depth estimation.",
            "memory_mb": 1200,
            "input_size": (384, 384),
        },
        "hybrid": {
            "hub_name": "DPT_Hybrid",
            "transform_type": "dpt",
            "display_name": "MiDaS DPT-Hybrid",
            "description": "Balanced quality and speed.",
            "memory_mb": 600,
            "input_size": (384, 384),
        },
    }

    def __init__(
        self,
        variant: Literal["small", "large", "hybrid"] = "small",
        device: torch.device | None = None,
    ) -> None:
        """
        Initialize MiDaS estimator.

        Args:
            variant: Model variant to use.
            device: PyTorch device.
        """
        super().__init__(device)
        self.variant = variant
        self._variant_info = self.VARIANTS[variant]
        self._transform = None

    @property
    def model_info(self) -> ModelInfo:
        """Get model information."""
        info = self._variant_info
        return ModelInfo(
            name=f"midas_{self.variant}",
            display_name=info["display_name"],
            description=info["description"],
            memory_mb=info["memory_mb"],
            supports_batching=True,
            default_input_size=info["input_size"],
        )

    def load(self) -> None:
        """Load the MiDaS model from PyTorch Hub."""
        if self._loaded:
            return

        hub_name = self._variant_info["hub_name"]

        # Load model from torch hub
        self._model = torch.hub.load(
            "intel-isl/MiDaS",
            hub_name,
            pretrained=True,
            trust_repo=True,
        )
        self._model.to(self.device)
        self._model.eval()

        # Load transforms
        midas_transforms = torch.hub.load(
            "intel-isl/MiDaS",
            "transforms",
            trust_repo=True,
        )

        if self._variant_info["transform_type"] == "dpt":
            self._transform = midas_transforms.dpt_transform
        else:
            self._transform = midas_transforms.small_transform

        self._loaded = True

    def unload(self) -> None:
        """Unload the model to free memory."""
        if self._model is not None:
            del self._model
            self._model = None

        if self._transform is not None:
            del self._transform
            self._transform = None

        self._loaded = False

        # Free GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def estimate(self, frame: np.ndarray) -> np.ndarray:
        """
        Estimate depth for a single frame.

        Args:
            frame: RGB image (H, W, 3), uint8.

        Returns:
            Normalized depth map (H, W), float32, 0-1 range.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        original_h, original_w = frame.shape[:2]

        # Apply MiDaS transform
        input_batch = self._transform(frame).to(self.device)

        with torch.no_grad():
            prediction = self._model(input_batch)

            # Resize to original dimensions
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(original_h, original_w),
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth = prediction.cpu().numpy()

        # MiDaS outputs inverse depth (near = high values)
        # Normalize and invert to get conventional depth (near = 0, far = 1)
        return normalize_depth(depth, invert=True, clip_percentile=1.0)

    def estimate_batch(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        """
        Estimate depth for a batch of frames.

        Args:
            frames: List of RGB images.

        Returns:
            List of normalized depth maps.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        if len(frames) == 0:
            return []

        # Get original dimensions (assuming all frames are same size)
        original_h, original_w = frames[0].shape[:2]

        # Transform all frames
        input_tensors = [self._transform(frame) for frame in frames]
        input_batch = torch.cat(input_tensors, dim=0).to(self.device)

        with torch.no_grad():
            predictions = self._model(input_batch)

            # Resize all predictions to original dimensions
            predictions = torch.nn.functional.interpolate(
                predictions.unsqueeze(1),
                size=(original_h, original_w),
                mode="bicubic",
                align_corners=False,
            ).squeeze(1)

        # Convert to numpy and normalize
        results = []
        for i in range(len(frames)):
            depth = predictions[i].cpu().numpy()
            results.append(normalize_depth(depth, invert=True, clip_percentile=1.0))

        return results
