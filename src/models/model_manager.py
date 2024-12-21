"""Model manager for loading and caching depth estimation models."""

from pathlib import Path
from typing import TYPE_CHECKING

import torch

from ..utils.config import DepthModel
from .base import DepthEstimator, ModelInfo

if TYPE_CHECKING:
    from .midas import MiDaSEstimator
    from .depth_anything import DepthAnythingEstimator


class ModelManager:
    """Manages loading, caching, and switching between depth models."""

    def __init__(self, device: torch.device | None = None) -> None:
        """
        Initialize the model manager.

        Args:
            device: PyTorch device to use.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self._current_model: DepthEstimator | None = None
        self._current_model_type: DepthModel | None = None
        self._cache_dir = self._get_cache_dir()

    def _get_cache_dir(self) -> Path:
        """Get the model cache directory."""
        cache_dir = Path.home() / ".cache" / "deepframe" / "models"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    def get_model(self, model_type: DepthModel) -> DepthEstimator:
        """
        Get a depth estimation model.

        Loads the model if not already loaded, or returns cached instance.

        Args:
            model_type: Type of model to load.

        Returns:
            Loaded depth estimator.
        """
        # Return cached model if same type
        if self._current_model is not None and self._current_model_type == model_type:
            return self._current_model

        # Unload previous model to free memory
        self.unload_current()

        # Create new model
        self._current_model = self._create_model(model_type)
        self._current_model_type = model_type

        # Load model weights
        self._current_model.load()

        return self._current_model

    def _create_model(self, model_type: DepthModel) -> DepthEstimator:
        """Create a model instance without loading weights."""
        if model_type == DepthModel.MIDAS_SMALL:
            from .midas import MiDaSEstimator

            return MiDaSEstimator(variant="small", device=self.device)

        elif model_type == DepthModel.MIDAS_LARGE:
            from .midas import MiDaSEstimator

            return MiDaSEstimator(variant="large", device=self.device)

        elif model_type == DepthModel.MIDAS_HYBRID:
            from .midas import MiDaSEstimator

            return MiDaSEstimator(variant="hybrid", device=self.device)

        elif model_type in (
            DepthModel.DEPTH_ANYTHING_VITS,
            DepthModel.DEPTH_ANYTHING_VITB,
            DepthModel.DEPTH_ANYTHING_VITL,
        ):
            from .depth_anything import DepthAnythingEstimator

            variant_map = {
                DepthModel.DEPTH_ANYTHING_VITS: "vits",
                DepthModel.DEPTH_ANYTHING_VITB: "vitb",
                DepthModel.DEPTH_ANYTHING_VITL: "vitl",
            }
            return DepthAnythingEstimator(
                variant=variant_map[model_type],
                device=self.device,
            )

        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def unload_current(self) -> None:
        """Unload the currently loaded model."""
        if self._current_model is not None:
            self._current_model.unload()
            self._current_model = None
            self._current_model_type = None

            # Free GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def get_model_info(self, model_type: DepthModel) -> ModelInfo:
        """Get information about a model without loading it."""
        model = self._create_model(model_type)
        info = model.model_info
        del model
        return info

    @staticmethod
    def get_available_models() -> list[tuple[DepthModel, str, str]]:
        """
        Get list of available models with display info.

        Returns:
            List of (model_type, display_name, description) tuples.
        """
        return [
            (DepthModel.MIDAS_SMALL, "MiDaS Small", "Fast, lower quality"),
            (DepthModel.MIDAS_HYBRID, "MiDaS Hybrid", "Balanced speed/quality"),
            (DepthModel.MIDAS_LARGE, "MiDaS DPT-Large", "High quality"),
            (DepthModel.DEPTH_ANYTHING_VITS, "Depth Anything ViT-S", "Fast, good quality"),
            (DepthModel.DEPTH_ANYTHING_VITB, "Depth Anything ViT-B", "Better quality"),
            (DepthModel.DEPTH_ANYTHING_VITL, "Depth Anything ViT-L", "Best quality"),
        ]

    @property
    def current_model(self) -> DepthEstimator | None:
        """Get the currently loaded model."""
        return self._current_model

    @property
    def current_model_type(self) -> DepthModel | None:
        """Get the type of currently loaded model."""
        return self._current_model_type

    def __del__(self) -> None:
        """Clean up on deletion."""
        self.unload_current()
