"""Export worker for batch video processing."""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from PyQt6.QtCore import QThread, pyqtSignal

from ..core.video_reader import VideoReader
from ..core.video_writer import VideoWriter, OutputSettings
from ..utils.config import Config, SBSLayout

if TYPE_CHECKING:
    from ..core.frame_processor import FrameProcessor


@dataclass
class ExportJob:
    """Represents a single export job."""

    input_path: str
    output_path: str
    config: Config


@dataclass
class ExportProgress:
    """Progress information for export."""

    job_index: int
    total_jobs: int
    current_file: str
    frame_index: int
    total_frames: int
    percent: float


class ExportWorker(QThread):
    """Worker thread for exporting videos."""

    # Signals
    progress = pyqtSignal(object)  # ExportProgress
    job_started = pyqtSignal(str)  # filename
    job_completed = pyqtSignal(str)  # filename
    job_failed = pyqtSignal(str, str)  # filename, error
    all_completed = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(
        self,
        processor: "FrameProcessor",
        jobs: list[ExportJob],
        parent=None,
    ) -> None:
        """
        Initialize export worker.

        Args:
            processor: Frame processor instance.
            jobs: List of export jobs.
            parent: Parent QObject.
        """
        super().__init__(parent)
        self._processor = processor
        self._jobs = jobs
        self._should_cancel = False
        self._is_paused = False

    def run(self) -> None:
        """Process all export jobs."""
        for job_idx, job in enumerate(self._jobs):
            if self._should_cancel:
                break

            filename = Path(job.input_path).name
            self.job_started.emit(filename)

            try:
                self._process_job(job, job_idx)
                self.job_completed.emit(filename)
            except Exception as e:
                self.job_failed.emit(filename, str(e))

        self.all_completed.emit()

    def _process_job(self, job: ExportJob, job_index: int) -> None:
        """
        Process a single export job.

        Args:
            job: Export job to process.
            job_index: Index of this job in the queue.
        """
        # Update processor config
        self._processor.config = job.config

        # Ensure model is loaded
        if not self._processor.is_model_loaded:
            self._processor.load_model()

        # Open input video
        reader = VideoReader(job.input_path)
        if not reader.open():
            raise RuntimeError(f"Failed to open input video: {job.input_path}")

        try:
            info = reader.info
            if info is None:
                raise RuntimeError("Failed to get video info")

            # Determine output dimensions based on SBS layout
            if job.config.depth.output_layout == SBSLayout.FULL:
                output_width = info.width * 2
            else:
                output_width = info.width
            output_height = info.height

            # Create output settings
            output_settings = OutputSettings(
                codec=job.config.output.codec.value,
                quality=job.config.output.quality,
            )

            # Open output writer
            writer = VideoWriter(
                job.output_path,
                output_width,
                output_height,
                info.fps,
                output_settings,
                audio_source=job.input_path if info.has_audio else None,
            )

            if not writer.open():
                raise RuntimeError(f"Failed to open output video: {job.output_path}")

            try:
                # Process frames in batches
                batch_size = 4  # Adjust based on GPU memory

                for batch in reader.iter_batches(batch_size):
                    if self._should_cancel:
                        break

                    # Wait while paused
                    while self._is_paused and not self._should_cancel:
                        self.msleep(100)

                    # Extract frames from batch
                    frame_nums = [fn for fn, _ in batch]
                    frames = [f for _, f in batch]

                    # Process batch
                    results = self._processor.process_batch(frames)

                    # Write SBS frames
                    for result in results:
                        writer.write_frame(result.sbs)

                    # Emit progress
                    current_frame = frame_nums[-1] if frame_nums else 0
                    progress = ExportProgress(
                        job_index=job_index,
                        total_jobs=len(self._jobs),
                        current_file=Path(job.input_path).name,
                        frame_index=current_frame,
                        total_frames=info.frame_count,
                        percent=(current_frame / info.frame_count) * 100,
                    )
                    self.progress.emit(progress)

            finally:
                writer.close()

        finally:
            reader.close()

    def cancel(self) -> None:
        """Cancel the export."""
        self._should_cancel = True

    def pause(self) -> None:
        """Pause the export."""
        self._is_paused = True

    def resume(self) -> None:
        """Resume the export."""
        self._is_paused = False

    @property
    def is_paused(self) -> bool:
        """Check if export is paused."""
        return self._is_paused

    @property
    def is_cancelled(self) -> bool:
        """Check if export was cancelled."""
        return self._should_cancel
