"""Model download and loading progress dialog."""

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)


class ModelLoadThread(QThread):
    """Thread for loading model in background."""

    progress = pyqtSignal(str)  # status message
    finished_ok = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, processor, parent=None):
        super().__init__(parent)
        self._processor = processor

    def run(self):
        try:
            self.progress.emit("Downloading model (first time only)...")
            self._processor.load_model()
            self.finished_ok.emit()
        except Exception as e:
            self.error.emit(str(e))


class ModelLoadDialog(QDialog):
    """Dialog shown while loading/downloading a model."""

    def __init__(self, processor, parent=None):
        super().__init__(parent)
        self._processor = processor
        self._thread: ModelLoadThread | None = None
        self._success = False

        self.setWindowTitle("Loading Model")
        self.setFixedSize(400, 150)
        self.setModal(True)
        self.setWindowFlags(
            self.windowFlags() & ~Qt.WindowType.WindowCloseButtonHint
        )

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)

        self.title_label = QLabel("Loading Depth Model")
        self.title_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(self.title_label)

        self.status_label = QLabel("Initializing...")
        layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # Indeterminate
        layout.addWidget(self.progress_bar)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self._on_cancel)
        layout.addWidget(self.cancel_btn, alignment=Qt.AlignmentFlag.AlignRight)

    def start_loading(self):
        """Start loading the model."""
        self._thread = ModelLoadThread(self._processor, self)
        self._thread.progress.connect(self._on_progress)
        self._thread.finished_ok.connect(self._on_finished)
        self._thread.error.connect(self._on_error)
        self._thread.start()

    def _on_progress(self, message: str):
        self.status_label.setText(message)

    def _on_finished(self):
        self._success = True
        self.accept()

    def _on_error(self, error: str):
        self.status_label.setText(f"Error: {error}")
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.cancel_btn.setText("Close")

    def _on_cancel(self):
        if self._thread and self._thread.isRunning():
            self._thread.terminate()
            self._thread.wait()
        self.reject()

    @property
    def was_successful(self) -> bool:
        return self._success

    def closeEvent(self, event):
        if self._thread and self._thread.isRunning():
            self._thread.terminate()
            self._thread.wait()
        super().closeEvent(event)


def ensure_model_loaded(processor, parent=None) -> bool:
    """
    Ensure the model is loaded, showing a dialog if needed.

    Args:
        processor: The FrameProcessor instance.
        parent: Parent widget for the dialog.

    Returns:
        True if model is loaded successfully.
    """
    if processor.is_model_loaded:
        return True

    dialog = ModelLoadDialog(processor, parent)
    dialog.start_loading()
    result = dialog.exec()

    return dialog.was_successful
