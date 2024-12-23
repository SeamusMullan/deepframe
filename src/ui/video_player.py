"""Video player widget with preview modes."""

from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
from PyQt6.QtCore import QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from ..utils.config import Config

if TYPE_CHECKING:
    from ..core.frame_processor import ProcessingResult


class VideoPlayerWidget(QWidget):
    """Widget for video playback and preview."""

    frame_changed = pyqtSignal(int)
    video_loaded = pyqtSignal(str)
    process_frame_requested = pyqtSignal(object)  # np.ndarray

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config

        self._video_path: str | None = None
        self._capture: cv2.VideoCapture | None = None
        self._frame_count: int = 0
        self._fps: float = 30.0
        self._current_frame: int = 0
        self._is_playing: bool = False
        self._preview_mode: str = config.app.preview_mode
        self._processing: bool = False

        # Current frame data
        self._current_image: np.ndarray | None = None
        self._current_depth: np.ndarray | None = None
        self._current_sbs: np.ndarray | None = None
        self._current_anaglyph: np.ndarray | None = None

        # Playback timer
        self._play_timer = QTimer()
        self._play_timer.timeout.connect(self._on_play_tick)

        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the video player UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Video display area
        self.display_frame = QFrame()
        self.display_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        self.display_frame.setStyleSheet("background-color: #1a1a1a;")
        self.display_frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        display_layout = QVBoxLayout(self.display_frame)
        display_layout.setContentsMargins(0, 0, 0, 0)

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setText("Drop video here or use File → Open")
        self.video_label.setStyleSheet("color: #666; font-size: 14px;")
        display_layout.addWidget(self.video_label)

        layout.addWidget(self.display_frame, stretch=1)

        # Controls row
        controls_widget = QWidget()
        controls_layout = QHBoxLayout(controls_widget)
        controls_layout.setContentsMargins(4, 4, 4, 4)
        controls_layout.setSpacing(8)

        # Play/Pause button
        self.play_btn = QPushButton("▶")
        self.play_btn.setFixedWidth(40)
        self.play_btn.clicked.connect(self._toggle_play)
        self.play_btn.setEnabled(False)
        controls_layout.addWidget(self.play_btn)

        # Previous frame
        self.prev_btn = QPushButton("◀")
        self.prev_btn.setFixedWidth(40)
        self.prev_btn.clicked.connect(self._prev_frame)
        self.prev_btn.setEnabled(False)
        controls_layout.addWidget(self.prev_btn)

        # Next frame
        self.next_btn = QPushButton("▶")
        self.next_btn.setFixedWidth(40)
        self.next_btn.clicked.connect(self._next_frame)
        self.next_btn.setEnabled(False)
        controls_layout.addWidget(self.next_btn)

        # Timeline slider
        self.timeline = QSlider(Qt.Orientation.Horizontal)
        self.timeline.setEnabled(False)
        self.timeline.valueChanged.connect(self._on_timeline_changed)
        controls_layout.addWidget(self.timeline, stretch=1)

        # Time label
        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setFixedWidth(100)
        controls_layout.addWidget(self.time_label)

        layout.addWidget(controls_widget)

        # Preview mode tabs
        mode_widget = QWidget()
        mode_layout = QHBoxLayout(mode_widget)
        mode_layout.setContentsMargins(4, 0, 4, 4)
        mode_layout.setSpacing(4)

        self._mode_buttons = {}
        for mode, label in [
            ("original", "Original"),
            ("depth", "Depth"),
            ("sbs", "SBS"),
            ("anaglyph", "Anaglyph"),
        ]:
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setChecked(mode == self._preview_mode)
            btn.clicked.connect(lambda checked, m=mode: self._on_mode_clicked(m))
            mode_layout.addWidget(btn)
            self._mode_buttons[mode] = btn

        mode_layout.addStretch()
        layout.addWidget(mode_widget)

    def load_video(self, path: str) -> bool:
        """Load a video file."""
        if self._capture is not None:
            self._capture.release()

        self._capture = cv2.VideoCapture(path)
        if not self._capture.isOpened():
            self._video_path = None
            self._capture = None
            self.video_label.setText("Failed to load video")
            return False

        self._video_path = path
        self._frame_count = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self._fps = self._capture.get(cv2.CAP_PROP_FPS) or 30.0
        self._current_frame = 0

        # Enable controls
        self.play_btn.setEnabled(True)
        self.prev_btn.setEnabled(True)
        self.next_btn.setEnabled(True)
        self.timeline.setEnabled(True)
        self.timeline.setMaximum(max(0, self._frame_count - 1))
        self.timeline.setValue(0)

        # Load first frame
        self._seek_frame(0)
        self._update_time_label()

        self.video_loaded.emit(path)
        return True

    def _seek_frame(self, frame_num: int) -> None:
        """Seek to a specific frame and display it."""
        if self._capture is None:
            return

        frame_num = max(0, min(frame_num, self._frame_count - 1))
        self._capture.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = self._capture.read()

        if ret:
            self._current_frame = frame_num
            self._current_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Clear processed data - will be recomputed
            self._current_depth = None
            self._current_sbs = None
            self._current_anaglyph = None
            self._display_frame()
            self.frame_changed.emit(frame_num)
            # Request processing for non-original modes
            if self._preview_mode != "original" and not self._processing:
                self._request_processing()

    def _display_frame(self) -> None:
        """Display the current frame according to preview mode."""
        if self._current_image is None:
            return

        display_image = self._get_preview_image()
        if display_image is None:
            return

        # Convert to QPixmap
        h, w = display_image.shape[:2]
        if len(display_image.shape) == 2:
            # Grayscale
            q_image = QImage(display_image.data, w, h, w, QImage.Format.Format_Grayscale8)
        else:
            # RGB
            bytes_per_line = 3 * w
            q_image = QImage(display_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

        pixmap = QPixmap.fromImage(q_image)

        # Scale to fit the display area while maintaining aspect ratio
        scaled = pixmap.scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.video_label.setPixmap(scaled)

    def _get_preview_image(self) -> np.ndarray | None:
        """Get the image for the current preview mode."""
        if self._current_image is None:
            return None

        if self._preview_mode == "original":
            return self._current_image

        elif self._preview_mode == "depth":
            return self._get_depth_preview()

        elif self._preview_mode == "sbs":
            return self._get_sbs_preview()

        elif self._preview_mode == "anaglyph":
            return self._get_anaglyph_preview()

        return self._current_image

    def _get_depth_preview(self) -> np.ndarray:
        """Get depth map visualization."""
        if self._current_depth is None:
            # Placeholder - show "Processing..." indicator
            h, w = self._current_image.shape[:2]
            depth = np.tile(np.linspace(0, 255, w, dtype=np.uint8), (h, 1))
            colored = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
            return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

        # Apply colormap to depth and convert BGR->RGB
        depth_vis = (self._current_depth * 255).astype(np.uint8)
        colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
        return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    def _get_sbs_preview(self) -> np.ndarray:
        """Get side-by-side preview."""
        if self._current_sbs is not None:
            return self._current_sbs

        # Placeholder while processing
        h, w = self._current_image.shape[:2]
        half_w = w // 2
        left = cv2.resize(self._current_image, (half_w, h))
        right = cv2.resize(self._current_image, (half_w, h))
        return np.hstack([left, right])

    def _get_anaglyph_preview(self) -> np.ndarray:
        """Get red/cyan anaglyph preview."""
        if self._current_anaglyph is not None:
            return self._current_anaglyph

        # Placeholder while processing
        h, w = self._current_image.shape[:2]
        offset = 20
        left = self._current_image.copy()
        right = np.roll(self._current_image, offset, axis=1)
        anaglyph = np.zeros_like(self._current_image)
        anaglyph[:, :, 0] = left[:, :, 0]
        anaglyph[:, :, 1] = right[:, :, 1]
        anaglyph[:, :, 2] = right[:, :, 2]
        return anaglyph

    def _request_processing(self) -> None:
        """Request depth processing for current frame."""
        if self._current_image is not None:
            self._processing = True
            self.process_frame_requested.emit(self._current_image.copy())

    def on_processing_result(self, result: "ProcessingResult") -> None:
        """Handle processing result from worker."""
        self._processing = False
        self._current_depth = result.depth
        self._current_sbs = result.sbs
        self._current_anaglyph = result.anaglyph
        self._display_frame()

    def _toggle_play(self) -> None:
        """Toggle play/pause."""
        if self._is_playing:
            self._pause()
        else:
            self._play()

    def _play(self) -> None:
        """Start playback."""
        if self._capture is None:
            return

        self._is_playing = True
        self.play_btn.setText("❚❚")
        interval = int(1000 / self._fps)
        self._play_timer.start(interval)

    def _pause(self) -> None:
        """Pause playback."""
        self._is_playing = False
        self.play_btn.setText("▶")
        self._play_timer.stop()

    def _on_play_tick(self) -> None:
        """Handle playback timer tick."""
        if self._current_frame >= self._frame_count - 1:
            self._pause()
            return

        self._seek_frame(self._current_frame + 1)
        self.timeline.blockSignals(True)
        self.timeline.setValue(self._current_frame)
        self.timeline.blockSignals(False)
        self._update_time_label()

    def _prev_frame(self) -> None:
        """Go to previous frame."""
        if self._current_frame > 0:
            self._seek_frame(self._current_frame - 1)
            self.timeline.setValue(self._current_frame)
            self._update_time_label()

    def _next_frame(self) -> None:
        """Go to next frame."""
        if self._current_frame < self._frame_count - 1:
            self._seek_frame(self._current_frame + 1)
            self.timeline.setValue(self._current_frame)
            self._update_time_label()

    def _on_timeline_changed(self, value: int) -> None:
        """Handle timeline slider change."""
        if self._is_playing:
            return
        self._seek_frame(value)
        self._update_time_label()

    def _update_time_label(self) -> None:
        """Update the time display label."""
        current_sec = self._current_frame / self._fps if self._fps > 0 else 0
        total_sec = self._frame_count / self._fps if self._fps > 0 else 0

        current_str = f"{int(current_sec // 60):02d}:{int(current_sec % 60):02d}"
        total_str = f"{int(total_sec // 60):02d}:{int(total_sec % 60):02d}"
        self.time_label.setText(f"{current_str} / {total_str}")

    def _on_mode_clicked(self, mode: str) -> None:
        """Handle preview mode button click."""
        self.set_preview_mode(mode)

    def set_preview_mode(self, mode: str) -> None:
        """Set the preview mode."""
        old_mode = self._preview_mode
        self._preview_mode = mode
        self.config.app.preview_mode = mode

        # Update button states
        for m, btn in self._mode_buttons.items():
            btn.setChecked(m == mode)

        # Request processing if switching to a mode that needs it
        if mode != "original" and old_mode == "original":
            if self._current_depth is None and not self._processing:
                self._request_processing()

        # Refresh display
        self._display_frame()

    def set_depth_map(self, depth: np.ndarray) -> None:
        """Set the current depth map (from depth estimator)."""
        self._current_depth = depth
        if self._preview_mode in ("depth", "sbs", "anaglyph"):
            self._display_frame()

    def refresh_preview(self) -> None:
        """Refresh the preview display and reprocess if needed."""
        # Clear processed data to force reprocessing with new settings
        self._current_depth = None
        self._current_sbs = None
        self._current_anaglyph = None
        # Always allow new processing request on refresh (cancel previous)
        self._processing = False
        if self._preview_mode != "original":
            self._request_processing()
        self._display_frame()

    def get_current_frame(self) -> np.ndarray | None:
        """Get the current frame image."""
        return self._current_image

    def cleanup(self) -> None:
        """Clean up resources."""
        self._play_timer.stop()
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def resizeEvent(self, event) -> None:
        """Handle resize to update displayed frame."""
        super().resizeEvent(event)
        self._display_frame()
