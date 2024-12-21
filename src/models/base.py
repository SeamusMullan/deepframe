"""Base class for depth estimation models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch


@dataclass
class ModelInfo:
    """Information about a depth model."""

    name: str
    display_name: str
    description: str
    memory_mb: int  # Approximate GPU memory usage
    supports_batching: bool
    default_input_size: tuple[int, int]  # (height, width)


class DepthEstimator(ABC):
    """Abstract base class for depth estimation models."""

    def __init__(self, device: torch.device | None = None) -> None:
        """
        Initialize the depth estimator.

        Args:
            device: PyTorch device to use. If None, will auto-detect.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self._model: torch.nn.Module | None = None
        self._loaded = False

    @property
    @abstractmethod
    def model_info(self) -> ModelInfo:
        """Get information about this model."""
        pass

    @abstractmethod
    def load(self) -> None:
        """Load the model weights. Must be called before estimate()."""
        pass

    @abstractmethod
    def unload(self) -> None:
        """Unload the model to free memory."""
        pass

    @abstractmethod
    def estimate(self, frame: np.ndarray) -> np.ndarray:
        """
        Estimate depth for a single frame.

        Args:
            frame: Input RGB image as numpy array (H, W, 3), uint8 0-255.

        Returns:
            Normalized depth map (H, W) as float32, range 0-1.
            0 = near, 1 = far (or vice versa depending on model).
        """
        pass

    def estimate_batch(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        """
        Estimate depth for a batch of frames.

        Default implementation processes frames one at a time.
        Override for models that support efficient batching.

        Args:
            frames: List of RGB images as numpy arrays.

        Returns:
            List of normalized depth maps.
        """
        return [self.estimate(frame) for frame in frames]

    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._loaded

    def __enter__(self) -> "DepthEstimator":
        """Context manager entry."""
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.unload()


def normalize_depth(
    depth: np.ndarray,
    invert: bool = False,
    clip_percentile: float = 0.0,
) -> np.ndarray:
    """
    Normalize depth map to 0-1 range.

    Args:
        depth: Raw depth values.
        invert: If True, invert the depth (near becomes 1, far becomes 0).
        clip_percentile: Clip outliers at this percentile (0 = no clipping).

    Returns:
        Normalized depth map in 0-1 range.
    """
    if clip_percentile > 0:
        low = np.percentile(depth, clip_percentile)
        high = np.percentile(depth, 100 - clip_percentile)
        depth = np.clip(depth, low, high)

    depth_min = depth.min()
    depth_max = depth.max()

    if depth_max - depth_min < 1e-8:
        return np.zeros_like(depth, dtype=np.float32)

    normalized = (depth - depth_min) / (depth_max - depth_min)

    if invert:
        normalized = 1.0 - normalized

    return normalized.astype(np.float32)
