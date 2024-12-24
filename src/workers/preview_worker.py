"""Preview worker for real-time depth estimation and SBS preview."""

from typing import TYPE_CHECKING

import numpy as np
from PyQt6.QtCore import QObject, QThread, pyqtSignal

if TYPE_CHECKING:
    from ..core.frame_processor import FrameProcessor


class PreviewWorker(QObject):
    """Worker for processing preview frames in a background thread."""

    # Signals
    result_ready = pyqtSignal(object)  # ProcessingResult
    error_occurred = pyqtSignal(str)
    model_loaded = pyqtSignal()
    model_loading = pyqtSignal(str)  # model name

    def __init__(self, processor: "FrameProcessor") -> None:
        """
        Initialize the preview worker.

        Args:
            processor: Frame processor instance.
        """
        super().__init__()
        self._processor = processor
        self._pending_frame: np.ndarray | None = None
        self._should_stop = False

    def request_process(self, frame: np.ndarray) -> None:
        """
        Request processing of a frame.

        Only the most recent frame is kept if multiple requests come in
        before processing completes.

        Args:
            frame: RGB frame to process.
        """
        self._pending_frame = frame.copy()

    def process_pending(self) -> None:
        """Process the pending frame if any."""
        if self._pending_frame is None or self._should_stop:
            return

        frame = self._pending_frame
        self._pending_frame = None

        try:
            # Ensure model is loaded
            if not self._processor.is_model_loaded:
                self.model_loading.emit("Loading depth model...")
                self._processor.load_model()
                self.model_loaded.emit()

            # Process frame
            result = self._processor.process_frame(frame, generate_anaglyph=True)
            self.result_ready.emit(result)

        except Exception as e:
            self.error_occurred.emit(str(e))

    def stop(self) -> None:
        """Stop the worker."""
        self._should_stop = True


class PreviewThread(QThread):
    """Thread for running preview worker."""

    result_ready = pyqtSignal(object)
    error_occurred = pyqtSignal(str)
    model_loaded = pyqtSignal()
    model_loading = pyqtSignal(str)

    def __init__(self, processor: "FrameProcessor", parent=None) -> None:
        """
        Initialize preview thread.

        Args:
            processor: Frame processor instance.
            parent: Parent QObject.
        """
        super().__init__(parent)
        self._processor = processor
        self._worker: PreviewWorker | None = None
        self._pending_frame: np.ndarray | None = None
        self._should_stop = False
        self._frame_requested = False

    def request_process(self, frame: np.ndarray) -> None:
        """
        Request processing of a frame.

        Args:
            frame: RGB frame to process.
        """
        self._pending_frame = frame.copy()
        self._frame_requested = True

    def _ensure_correct_model(self) -> None:
        """Ensure the correct model is loaded based on config."""
        config_model = self._processor.config.depth.model
        current_model = self._processor.current_model_type

        if current_model != config_model:
            self.model_loading.emit(f"Loading {config_model.value}...")
            self._processor.unload_model()
            self._processor.load_model(config_model)
            self.model_loaded.emit()
        elif not self._processor.is_model_loaded:
            self.model_loading.emit(f"Loading {config_model.value}...")
            self._processor.load_model(config_model)
            self.model_loaded.emit()

    def run(self) -> None:
        """Thread main loop."""
        while not self._should_stop:
            if self._frame_requested and self._pending_frame is not None:
                self._frame_requested = False
                frame = self._pending_frame
                self._pending_frame = None

                try:
                    # Ensure correct model is loaded
                    self._ensure_correct_model()

                    # Process frame
                    result = self._processor.process_frame(frame, generate_anaglyph=True)
                    self.result_ready.emit(result)

                except Exception as e:
                    self.error_occurred.emit(str(e))

            # Small sleep to prevent busy waiting
            self.msleep(10)

    def stop(self) -> None:
        """Stop the thread."""
        self._should_stop = True
        self.wait()
