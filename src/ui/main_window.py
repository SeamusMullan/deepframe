"""Main application window for DeepFrame."""

from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal, QThread
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QProgressDialog,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from ..core.frame_processor import FrameProcessor
from ..utils import get_gpu_info, select_device
from ..utils.config import Config
from ..workers.export_worker import ExportWorker, ExportJob
from ..workers.preview_worker import PreviewThread
from .queue_panel import QueuePanel, QueueStatus
from .settings_panel import SettingsPanel
from .video_player import VideoPlayerWidget


class MainWindow(QMainWindow):
    """Main application window."""

    video_loaded = pyqtSignal(str)
    export_requested = pyqtSignal()

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config

        # Initialize frame processor
        device = select_device(config.app.device_preference)
        self._processor = FrameProcessor(config)

        # Preview thread (will be started on first use)
        self._preview_thread: PreviewThread | None = None

        # Export worker
        self._export_worker: ExportWorker | None = None

        self._setup_ui()
        self._setup_menu()
        self._setup_statusbar()
        self._connect_signals()
        self._setup_preview_thread()

        # Restore window geometry
        self.setWindowTitle("DeepFrame - 2D to VR SBS Converter")
        self.resize(1400, 900)

    def _setup_preview_thread(self) -> None:
        """Set up the background preview thread."""
        self._preview_thread = PreviewThread(self._processor, self)
        self._preview_thread.result_ready.connect(self._on_preview_result)
        self._preview_thread.error_occurred.connect(self._on_preview_error)
        self._preview_thread.model_loading.connect(self._on_model_loading)
        self._preview_thread.model_loaded.connect(self._on_model_loaded)
        self._preview_thread.start()

    def _setup_ui(self) -> None:
        """Set up the main UI layout."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # Main splitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left side: Video preview + Queue
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

        # Video player widget
        self.video_player = VideoPlayerWidget(self.config)
        left_layout.addWidget(self.video_player, stretch=3)

        # Queue panel
        self.queue_panel = QueuePanel()
        left_layout.addWidget(self.queue_panel, stretch=1)

        splitter.addWidget(left_widget)

        # Right side: Settings panel
        self.settings_panel = SettingsPanel(self.config)
        splitter.addWidget(self.settings_panel)

        # Set initial splitter sizes (70% left, 30% right)
        splitter.setSizes([980, 420])
        splitter.setStretchFactor(0, 7)
        splitter.setStretchFactor(1, 3)

        main_layout.addWidget(splitter)

    def _setup_menu(self) -> None:
        """Set up the menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        open_action = QAction("&Open Video...", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self._open_video)
        file_menu.addAction(open_action)

        add_to_queue_action = QAction("&Add to Queue...", self)
        add_to_queue_action.setShortcut(QKeySequence("Ctrl+Shift+O"))
        add_to_queue_action.triggered.connect(self._add_to_queue)
        file_menu.addAction(add_to_queue_action)

        file_menu.addSeparator()

        export_action = QAction("&Export Selected...", self)
        export_action.setShortcut(QKeySequence("Ctrl+E"))
        export_action.triggered.connect(self._export_selected)
        file_menu.addAction(export_action)

        export_all_action = QAction("Export &All...", self)
        export_all_action.setShortcut(QKeySequence("Ctrl+Shift+E"))
        export_all_action.triggered.connect(self._export_all)
        file_menu.addAction(export_all_action)

        file_menu.addSeparator()

        quit_action = QAction("&Quit", self)
        quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        # View menu
        view_menu = menubar.addMenu("&View")

        preview_group_label = QAction("Preview Mode:", self)
        preview_group_label.setEnabled(False)
        view_menu.addAction(preview_group_label)

        self._preview_actions = {}
        for mode, label in [
            ("original", "Original"),
            ("depth", "Depth Map"),
            ("sbs", "SBS Preview"),
            ("anaglyph", "Anaglyph (Red/Cyan)"),
        ]:
            action = QAction(f"  {label}", self)
            action.setCheckable(True)
            action.setChecked(mode == self.config.app.preview_mode)
            action.triggered.connect(lambda checked, m=mode: self._set_preview_mode(m))
            view_menu.addAction(action)
            self._preview_actions[mode] = action

        # Settings menu
        settings_menu = menubar.addMenu("&Settings")

        device_label = QAction("Device:", self)
        device_label.setEnabled(False)
        settings_menu.addAction(device_label)

        self._device_actions = {}
        for device, label in [("auto", "Auto"), ("cuda", "CUDA (GPU)"), ("cpu", "CPU")]:
            action = QAction(f"  {label}", self)
            action.setCheckable(True)
            action.setChecked(device == self.config.app.device_preference)
            action.triggered.connect(lambda checked, d=device: self._set_device(d))
            settings_menu.addAction(action)
            self._device_actions[device] = action

        settings_menu.addSeparator()

        save_settings_action = QAction("&Save Settings", self)
        save_settings_action.setShortcut(QKeySequence("Ctrl+S"))
        save_settings_action.triggered.connect(self._save_settings)
        settings_menu.addAction(save_settings_action)

        # Help menu
        help_menu = menubar.addMenu("&Help")

        about_action = QAction("&About DeepFrame", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _setup_statusbar(self) -> None:
        """Set up the status bar."""
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)

        # Status message
        self.status_label = QLabel("Ready")
        self.statusbar.addWidget(self.status_label, stretch=1)

        # Progress bar (hidden by default)
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setVisible(False)
        self.statusbar.addWidget(self.progress_bar)

        # GPU info
        gpu_info = get_gpu_info()
        if gpu_info.available:
            gpu_text = f"GPU: {gpu_info.name}"
        else:
            gpu_text = "GPU: None (CPU mode)"
        self.gpu_label = QLabel(gpu_text)
        self.statusbar.addPermanentWidget(self.gpu_label)

    def _connect_signals(self) -> None:
        """Connect signals between components."""
        # Queue panel signals
        self.queue_panel.item_selected.connect(self._on_queue_item_selected)
        self.queue_panel.export_requested.connect(self._export_selected)

        # Settings panel signals
        self.settings_panel.settings_changed.connect(self._on_settings_changed)
        self.settings_panel.export_clicked.connect(self._export_selected)

        # Video player signals
        self.video_player.process_frame_requested.connect(self._on_process_frame_requested)

    def _on_process_frame_requested(self, frame) -> None:
        """Handle frame processing request from video player."""
        if self._preview_thread is not None:
            self._preview_thread.request_process(frame)

    def _on_preview_result(self, result) -> None:
        """Handle preview processing result."""
        self.video_player.on_processing_result(result)
        self.status_label.setText("Ready")

    def _on_preview_error(self, error: str) -> None:
        """Handle preview processing error."""
        self.status_label.setText(f"Error: {error}")

    def _on_model_loading(self, message: str) -> None:
        """Handle model loading notification."""
        self.status_label.setText(message)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate

    def _on_model_loaded(self) -> None:
        """Handle model loaded notification."""
        self.status_label.setText("Model loaded")
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 100)

    def _open_video(self) -> None:
        """Open a video file for preview."""
        start_dir = self.config.app.last_input_dir or str(Path.home())

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Video",
            start_dir,
            "Video Files (*.mp4 *.mkv *.avi *.mov *.webm *.wmv *.flv);;All Files (*)",
        )

        if file_path:
            self.config.app.last_input_dir = str(Path(file_path).parent)
            self._load_video(file_path)

    def _add_to_queue(self) -> None:
        """Add video files to the processing queue."""
        start_dir = self.config.app.last_input_dir or str(Path.home())

        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Add Videos to Queue",
            start_dir,
            "Video Files (*.mp4 *.mkv *.avi *.mov *.webm *.wmv *.flv);;All Files (*)",
        )

        if file_paths:
            self.config.app.last_input_dir = str(Path(file_paths[0]).parent)
            for path in file_paths:
                self.queue_panel.add_item(path)

    def _load_video(self, path: str) -> None:
        """Load a video file for preview."""
        self.video_player.load_video(path)
        self.status_label.setText(f"Loaded: {Path(path).name}")
        self.video_loaded.emit(path)

        # Also add to queue if not already there
        self.queue_panel.add_item(path, select=True)

    def _on_queue_item_selected(self, path: str) -> None:
        """Handle queue item selection."""
        if path:
            self._load_video(path)

    def _on_settings_changed(self) -> None:
        """Handle settings changes."""
        # Trigger preview refresh
        self.video_player.refresh_preview()

    def _set_preview_mode(self, mode: str) -> None:
        """Set the preview mode."""
        # Update checkmarks
        for m, action in self._preview_actions.items():
            action.setChecked(m == mode)

        self.config.app.preview_mode = mode
        self.video_player.set_preview_mode(mode)

    def _set_device(self, device: str) -> None:
        """Set the processing device."""
        # Update checkmarks
        for d, action in self._device_actions.items():
            action.setChecked(d == device)

        self.config.app.device_preference = device
        self.status_label.setText(f"Device set to: {device}")

        # Unload current model to force reload on new device
        self._processor.unload_model()

    def _export_selected(self) -> None:
        """Export selected queue items."""
        selected = self.queue_panel.get_selected_items()
        if not selected:
            QMessageBox.information(
                self, "No Selection", "Please select videos from the queue to export."
            )
            return

        self._start_export(selected)

    def _export_all(self) -> None:
        """Export all queue items."""
        all_items = self.queue_panel.get_all_items()
        if not all_items:
            QMessageBox.information(self, "Empty Queue", "Please add videos to the queue first.")
            return

        self._start_export(all_items)

    def _start_export(self, paths: list[str]) -> None:
        """Start exporting the given video paths."""
        # Get output directory
        start_dir = self.config.app.last_output_dir or str(Path.home())

        output_dir = QFileDialog.getExistingDirectory(
            self, "Select Output Directory", start_dir
        )

        if not output_dir:
            return

        self.config.app.last_output_dir = output_dir

        # Create export jobs
        jobs = []
        for input_path in paths:
            input_name = Path(input_path).stem
            output_path = str(Path(output_dir) / f"{input_name}_SBS.mp4")
            jobs.append(ExportJob(
                input_path=input_path,
                output_path=output_path,
                config=self.config,
            ))

        # Create and start export worker
        self._export_worker = ExportWorker(self._processor, jobs, self)
        self._export_worker.progress.connect(self._on_export_progress)
        self._export_worker.job_started.connect(self._on_export_job_started)
        self._export_worker.job_completed.connect(self._on_export_job_completed)
        self._export_worker.job_failed.connect(self._on_export_job_failed)
        self._export_worker.all_completed.connect(self._on_export_completed)

        # Show progress
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText(f"Exporting {len(paths)} video(s)...")

        # Disable export buttons during export
        self.settings_panel.export_btn.setEnabled(False)

        self._export_worker.start()

    def _on_export_progress(self, progress) -> None:
        """Handle export progress update."""
        self.progress_bar.setValue(int(progress.percent))
        self.status_label.setText(
            f"Exporting {progress.current_file}: {progress.percent:.1f}% "
            f"(Frame {progress.frame_index}/{progress.total_frames})"
        )

    def _on_export_job_started(self, filename: str) -> None:
        """Handle export job started."""
        # Find and update queue item status
        for path in self.queue_panel.get_all_items():
            if Path(path).name == filename:
                self.queue_panel.set_item_status(path, QueueStatus.PROCESSING)
                break

    def _on_export_job_completed(self, filename: str) -> None:
        """Handle export job completed."""
        # Find and update queue item status
        for path in self.queue_panel.get_all_items():
            if Path(path).name == filename:
                self.queue_panel.set_item_status(path, QueueStatus.COMPLETED)
                break

    def _on_export_job_failed(self, filename: str, error: str) -> None:
        """Handle export job failed."""
        # Find and update queue item status
        for path in self.queue_panel.get_all_items():
            if Path(path).name == filename:
                self.queue_panel.set_item_status(path, QueueStatus.ERROR)
                break

        QMessageBox.warning(
            self,
            "Export Failed",
            f"Failed to export {filename}:\n{error}",
        )

    def _on_export_completed(self) -> None:
        """Handle all exports completed."""
        self.progress_bar.setVisible(False)
        self.status_label.setText("Export completed")
        self.settings_panel.export_btn.setEnabled(True)

        QMessageBox.information(
            self,
            "Export Complete",
            "All videos have been exported successfully.",
        )

    def _save_settings(self) -> None:
        """Save current settings to config file."""
        self.config.save()
        self.status_label.setText("Settings saved")

    def _show_about(self) -> None:
        """Show the about dialog."""
        QMessageBox.about(
            self,
            "About DeepFrame",
            "<h2>DeepFrame</h2>"
            "<p>Version 0.1.0</p>"
            "<p>Convert 2D videos to VR-compatible side-by-side 3D "
            "using AI depth estimation.</p>"
            "<p>100% local processing - your files never leave your computer.</p>",
        )

    def closeEvent(self, event) -> None:
        """Handle window close event."""
        # Stop preview thread
        if self._preview_thread is not None:
            self._preview_thread.stop()

        # Cancel any running export
        if self._export_worker is not None and self._export_worker.isRunning():
            self._export_worker.cancel()
            self._export_worker.wait()

        # Unload model
        self._processor.unload_model()

        # Save settings on exit
        self.config.save()

        # Clean up video player
        self.video_player.cleanup()

        event.accept()
