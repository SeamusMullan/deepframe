"""Frame processor - orchestrates depth estimation and SBS generation."""

from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..models.base import DepthEstimator
from ..models.model_manager import ModelManager
from ..utils.config import Config, DepthModel, FillMode, SBSLayout
from .sbs_generator import SBSGenerator, SBSSettings


@dataclass
class ProcessingResult:
    """Result of processing a single frame."""

    original: np.ndarray
    depth: np.ndarray
    sbs: np.ndarray
    anaglyph: np.ndarray | None = None


class FrameProcessor:
    """Orchestrates depth estimation and SBS generation for video frames."""

    def __init__(self, config: Config) -> None:
        """
        Initialize the frame processor.

        Args:
            config: Application configuration.
        """
        self.config = config
        self._model_manager = ModelManager()
        self._sbs_generator = SBSGenerator()
        self._current_model: DepthEstimator | None = None

    def load_model(self, model_type: DepthModel | None = None) -> None:
        """
        Load the depth estimation model.

        Args:
            model_type: Model to load. Uses config default if None.
        """
        if model_type is None:
            model_type = self.config.depth.model

        self._current_model = self._model_manager.get_model(model_type)

    def unload_model(self) -> None:
        """Unload the current model to free memory."""
        self._model_manager.unload_current()
        self._current_model = None

    def get_sbs_settings(self) -> SBSSettings:
        """Get SBS settings from config."""
        return SBSSettings(
            depth_strength=self.config.depth.depth_strength,
            eye_separation=self.config.depth.eye_separation,
            depth_focus=self.config.depth.depth_focus,
            fill_mode=FillMode(self.config.depth.fill_mode.value),
            layout=SBSLayout(self.config.depth.output_layout.value),
        )

    def process_frame(
        self,
        frame: np.ndarray,
        generate_anaglyph: bool = False,
    ) -> ProcessingResult:
        """
        Process a single frame.

        Args:
            frame: RGB input frame (H, W, 3), uint8.
            generate_anaglyph: Whether to also generate anaglyph preview.

        Returns:
            ProcessingResult with depth map and SBS output.
        """
        if self._current_model is None:
            raise RuntimeError("No model loaded. Call load_model() first.")

        # Estimate depth
        depth = self._current_model.estimate(frame)

        # Get SBS settings from config
        sbs_settings = self.get_sbs_settings()

        # Generate SBS
        sbs = self._sbs_generator.generate(frame, depth, sbs_settings)

        # Optionally generate anaglyph
        anaglyph = None
        if generate_anaglyph:
            anaglyph = self._sbs_generator.generate_anaglyph(frame, depth, sbs_settings)

        return ProcessingResult(
            original=frame,
            depth=depth,
            sbs=sbs,
            anaglyph=anaglyph,
        )

    def process_batch(
        self,
        frames: list[np.ndarray],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[ProcessingResult]:
        """
        Process a batch of frames.

        Args:
            frames: List of RGB frames.
            progress_callback: Optional callback(current, total) for progress updates.

        Returns:
            List of ProcessingResults.
        """
        if self._current_model is None:
            raise RuntimeError("No model loaded. Call load_model() first.")

        # Batch depth estimation
        depths = self._current_model.estimate_batch(frames)

        # Generate SBS for each frame
        sbs_settings = self.get_sbs_settings()
        results = []

        for i, (frame, depth) in enumerate(zip(frames, depths)):
            sbs = self._sbs_generator.generate(frame, depth, sbs_settings)
            results.append(ProcessingResult(
                original=frame,
                depth=depth,
                sbs=sbs,
            ))

            if progress_callback:
                progress_callback(i + 1, len(frames))

        return results

    def estimate_depth(self, frame: np.ndarray) -> np.ndarray:
        """
        Estimate depth for a single frame.

        Args:
            frame: RGB input frame.

        Returns:
            Normalized depth map.
        """
        if self._current_model is None:
            raise RuntimeError("No model loaded. Call load_model() first.")

        return self._current_model.estimate(frame)

    def generate_sbs(
        self,
        frame: np.ndarray,
        depth: np.ndarray,
    ) -> np.ndarray:
        """
        Generate SBS output from frame and depth.

        Args:
            frame: RGB input frame.
            depth: Normalized depth map.

        Returns:
            SBS output image.
        """
        sbs_settings = self.get_sbs_settings()
        return self._sbs_generator.generate(frame, depth, sbs_settings)

    def generate_anaglyph(
        self,
        frame: np.ndarray,
        depth: np.ndarray,
    ) -> np.ndarray:
        """
        Generate anaglyph preview from frame and depth.

        Args:
            frame: RGB input frame.
            depth: Normalized depth map.

        Returns:
            Anaglyph RGB image.
        """
        sbs_settings = self.get_sbs_settings()
        return self._sbs_generator.generate_anaglyph(frame, depth, sbs_settings)

    @property
    def is_model_loaded(self) -> bool:
        """Check if a model is currently loaded."""
        return self._current_model is not None

    @property
    def current_model_type(self) -> DepthModel | None:
        """Get the currently loaded model type."""
        return self._model_manager.current_model_type
