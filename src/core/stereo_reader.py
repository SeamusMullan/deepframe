"""Utility for reading paired stereo video streams.
+
+Provides :class:`StereoVideoReader`, a thin wrapper around two OpenCV
+``VideoCapture`` objects that yields ``(left_frame, right_frame)`` tuples.
+It assumes both videos have the same frame count and FPS.
+"""

from pathlib import Path
from typing import Iterator, Tuple

import cv2


class StereoVideoReader:
    """Read two synchronized video files and produce paired frames.
    +
    +    Parameters
    +    ----------
    +    left_path: str | Path
    +        Path to the left‑eye video.
    +    right_path: str | Path
    +        Path to the right‑eye video.
    +"""

    def __init__(self, left_path: str | Path, right_path: str | Path) -> None:
        self.left_cap = cv2.VideoCapture(str(left_path))
        self.right_cap = cv2.VideoCapture(str(right_path))
        if not self.left_cap.isOpened() or not self.right_cap.isOpened():
            raise RuntimeError("Could not open one or both stereo video files")

    def __iter__(self) -> Iterator[Tuple[bytes, bytes]]:
        return self

    def __next__(self) -> Tuple[bytes, bytes]:
        ret_l, frame_l = self.left_cap.read()
        ret_r, frame_r = self.right_cap.read()
        if not ret_l or not ret_r:
            self.left_cap.release()
            self.right_cap.release()
            raise StopIteration
        return frame_l, frame_r
