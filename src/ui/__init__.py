"""UI components for DeepFrame."""

from .main_window import MainWindow
from .model_dialog import ModelLoadDialog, ensure_model_loaded
from .queue_panel import QueuePanel
from .settings_panel import SettingsPanel
from .video_player import VideoPlayerWidget

__all__ = [
    "MainWindow",
    "ModelLoadDialog",
    "QueuePanel",
    "SettingsPanel",
    "VideoPlayerWidget",
    "ensure_model_loaded",
]
