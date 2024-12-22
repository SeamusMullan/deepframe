"""FFmpeg-based video writer with audio support."""

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

try:
    import ffmpeg
except ImportError:
    ffmpeg = None


@dataclass
class OutputSettings:
    """Settings for video output encoding."""

    codec: str = "libx264"  # libx264, libx265, libvpx-vp9
    quality: int = 23  # CRF value (lower = better quality)
    preset: str = "medium"  # ultrafast, fast, medium, slow, slower
    pixel_format: str = "yuv420p"


class VideoWriter:
    """Write video frames to file using FFmpeg."""

    def __init__(
        self,
        output_path: str,
        width: int,
        height: int,
        fps: float,
        settings: OutputSettings | None = None,
        audio_source: str | None = None,
    ) -> None:
        """
        Initialize video writer.

        Args:
            output_path: Path for output video.
            width: Frame width.
            height: Frame height.
            fps: Frames per second.
            settings: Encoding settings.
            audio_source: Path to source video for audio passthrough.
        """
        self.output_path = output_path
        self.width = width
        self.height = height
        self.fps = fps
        self.settings = settings or OutputSettings()
        self.audio_source = audio_source

        self._process: subprocess.Popen | None = None
        self._frame_count = 0
        self._temp_video: str | None = None

    def open(self) -> bool:
        """
        Open the video writer.

        Returns:
            True if successful.
        """
        if ffmpeg is None:
            raise RuntimeError("ffmpeg-python not installed")

        # If we have audio source, write to temp file first, then mux
        if self.audio_source:
            self._temp_video = tempfile.mktemp(suffix=".mp4")
            output = self._temp_video
        else:
            output = self.output_path

        # Build FFmpeg command
        try:
            self._process = (
                ffmpeg.input(
                    "pipe:",
                    format="rawvideo",
                    pix_fmt="rgb24",
                    s=f"{self.width}x{self.height}",
                    r=self.fps,
                )
                .output(
                    output,
                    vcodec=self.settings.codec,
                    crf=self.settings.quality,
                    preset=self.settings.preset,
                    pix_fmt=self.settings.pixel_format,
                )
                .overwrite_output()
                .run_async(pipe_stdin=True, pipe_stderr=subprocess.PIPE)
            )
            return True
        except Exception as e:
            print(f"Failed to open video writer: {e}")
            return False

    def write_frame(self, frame: np.ndarray) -> bool:
        """
        Write a frame to the video.

        Args:
            frame: RGB frame (H, W, 3), uint8.

        Returns:
            True if successful.
        """
        if self._process is None:
            return False

        try:
            # Ensure frame is correct format
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)

            # Write raw bytes
            self._process.stdin.write(frame.tobytes())
            self._frame_count += 1
            return True
        except Exception as e:
            print(f"Failed to write frame: {e}")
            return False

    def close(self) -> bool:
        """
        Close the video writer and finalize the file.

        Returns:
            True if successful.
        """
        if self._process is None:
            return False

        try:
            # Close stdin and wait for process
            self._process.stdin.close()
            self._process.wait()

            # Check for errors
            if self._process.returncode != 0:
                stderr = self._process.stderr.read().decode()
                print(f"FFmpeg error: {stderr}")
                return False

            # Mux audio if needed
            if self._temp_video and self.audio_source:
                success = self._mux_audio()
                # Clean up temp file
                try:
                    Path(self._temp_video).unlink()
                except Exception:
                    pass
                return success

            return True

        except Exception as e:
            print(f"Failed to close video writer: {e}")
            return False
        finally:
            self._process = None

    def _mux_audio(self) -> bool:
        """
        Mux audio from source into output video.

        Returns:
            True if successful.
        """
        if ffmpeg is None or self._temp_video is None:
            return False

        try:
            # Get video stream from temp file
            video = ffmpeg.input(self._temp_video)

            # Get audio stream from source
            audio = ffmpeg.input(self.audio_source).audio

            # Mux together
            (
                ffmpeg.output(
                    video,
                    audio,
                    self.output_path,
                    vcodec="copy",  # Copy video, don't re-encode
                    acodec="aac",  # Re-encode audio to AAC for compatibility
                    strict="experimental",
                )
                .overwrite_output()
                .run(capture_stderr=True)
            )
            return True

        except ffmpeg.Error as e:
            print(f"Audio mux error: {e.stderr.decode()}")
            return False
        except Exception as e:
            print(f"Audio mux error: {e}")
            return False

    @property
    def frames_written(self) -> int:
        """Get number of frames written."""
        return self._frame_count

    def __enter__(self) -> "VideoWriter":
        """Context manager entry."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()


def extract_audio(input_path: str, output_path: str) -> bool:
    """
    Extract audio from video to separate file.

    Args:
        input_path: Source video path.
        output_path: Output audio path.

    Returns:
        True if successful.
    """
    if ffmpeg is None:
        return False

    try:
        (
            ffmpeg.input(input_path)
            .output(output_path, acodec="copy", vn=None)
            .overwrite_output()
            .run(capture_stderr=True)
        )
        return True
    except Exception:
        return False
