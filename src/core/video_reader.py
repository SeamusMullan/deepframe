"""FFmpeg-based video reader with seeking support."""

from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Iterator

import cv2
import numpy as np

try:
    import ffmpeg
except ImportError:
    ffmpeg = None


@dataclass
class VideoInfo:
    """Information about a video file."""

    path: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration: float  # seconds
    codec: str
    has_audio: bool


class VideoReader:
    """Read video frames using OpenCV (with FFmpeg backend)."""

    def __init__(self, path: str) -> None:
        """
        Initialize video reader.

        Args:
            path: Path to video file.
        """
        self.path = path
        self._capture: cv2.VideoCapture | None = None
        self._info: VideoInfo | None = None

    def open(self) -> bool:
        """
        Open the video file.

        Returns:
            True if successful, False otherwise.
        """
        self._capture = cv2.VideoCapture(self.path)
        if not self._capture.isOpened():
            self._capture = None
            return False

        # Get video info
        width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self._capture.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))
        codec_int = int(self._capture.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((codec_int >> 8 * i) & 0xFF) for i in range(4)])

        # Check for audio using ffprobe if available
        has_audio = self._check_audio()

        self._info = VideoInfo(
            path=self.path,
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            duration=frame_count / fps if fps > 0 else 0,
            codec=codec,
            has_audio=has_audio,
        )

        return True

    def _check_audio(self) -> bool:
        """Check if video has audio stream."""
        if ffmpeg is None:
            return False

        try:
            probe = ffmpeg.probe(self.path)
            audio_streams = [s for s in probe["streams"] if s["codec_type"] == "audio"]
            return len(audio_streams) > 0
        except Exception:
            return False

    def close(self) -> None:
        """Close the video file."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    @property
    def info(self) -> VideoInfo | None:
        """Get video information."""
        return self._info

    @property
    def is_open(self) -> bool:
        """Check if video is open."""
        return self._capture is not None and self._capture.isOpened()

    def seek(self, frame_number: int) -> bool:
        """
        Seek to a specific frame.

        Args:
            frame_number: Frame number to seek to.

        Returns:
            True if successful.
        """
        if self._capture is None:
            return False

        self._capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        return True

    def read_frame(self) -> tuple[bool, np.ndarray | None]:
        """
        Read the next frame.

        Returns:
            Tuple of (success, frame). Frame is RGB uint8.
        """
        if self._capture is None:
            return False, None

        ret, frame = self._capture.read()
        if ret:
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return True, frame

        return False, None

    def read_frame_at(self, frame_number: int) -> np.ndarray | None:
        """
        Read a specific frame.

        Args:
            frame_number: Frame number to read.

        Returns:
            RGB frame or None if failed.
        """
        if not self.seek(frame_number):
            return None

        ret, frame = self.read_frame()
        return frame if ret else None

    def iter_frames(
        self,
        start: int = 0,
        end: int | None = None,
        step: int = 1,
    ) -> Generator[tuple[int, np.ndarray], None, None]:
        """
        Iterate over frames in the video.

        Args:
            start: Starting frame number.
            end: Ending frame number (exclusive). None = end of video.
            step: Frame step (1 = every frame, 2 = every other, etc.)

        Yields:
            Tuple of (frame_number, frame).
        """
        if self._info is None:
            return

        if end is None:
            end = self._info.frame_count

        self.seek(start)
        current = start

        while current < end:
            ret, frame = self.read_frame()
            if not ret:
                break

            yield current, frame

            # Skip frames if step > 1
            if step > 1:
                current += step
                self.seek(current)
            else:
                current += 1

    def iter_batches(
        self,
        batch_size: int,
        start: int = 0,
        end: int | None = None,
    ) -> Generator[list[tuple[int, np.ndarray]], None, None]:
        """
        Iterate over batches of frames.

        Args:
            batch_size: Number of frames per batch.
            start: Starting frame number.
            end: Ending frame number (exclusive).

        Yields:
            List of (frame_number, frame) tuples.
        """
        batch = []

        for frame_num, frame in self.iter_frames(start, end):
            batch.append((frame_num, frame))

            if len(batch) >= batch_size:
                yield batch
                batch = []

        # Yield remaining frames
        if batch:
            yield batch

    def get_current_position(self) -> int:
        """Get current frame position."""
        if self._capture is None:
            return 0
        return int(self._capture.get(cv2.CAP_PROP_POS_FRAMES))

    def __enter__(self) -> "VideoReader":
        """Context manager entry."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()
