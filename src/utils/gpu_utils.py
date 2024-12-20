"""GPU detection and device management utilities."""

from dataclasses import dataclass
from typing import Literal

import torch


@dataclass
class GPUInfo:
    """Information about the available GPU."""

    available: bool
    name: str
    memory_total_mb: int
    memory_free_mb: int
    cuda_version: str | None


def is_cuda_available() -> bool:
    """Check if CUDA is available for PyTorch."""
    return torch.cuda.is_available()


def get_device() -> torch.device:
    """Get the best available device (CUDA if available, else CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_gpu_info() -> GPUInfo:
    """Get detailed information about the available GPU."""
    if not torch.cuda.is_available():
        return GPUInfo(
            available=False,
            name="CPU Only",
            memory_total_mb=0,
            memory_free_mb=0,
            cuda_version=None,
        )

    device_idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device_idx)
    memory_total = props.total_memory // (1024 * 1024)
    memory_free = (props.total_memory - torch.cuda.memory_allocated(device_idx)) // (1024 * 1024)

    return GPUInfo(
        available=True,
        name=props.name,
        memory_total_mb=memory_total,
        memory_free_mb=memory_free,
        cuda_version=torch.version.cuda,
    )


def select_device(preference: Literal["auto", "cuda", "cpu"] = "auto") -> torch.device:
    """
    Select device based on preference.

    Args:
        preference: 'auto' uses CUDA if available, 'cuda' requires CUDA,
                   'cpu' forces CPU usage.

    Returns:
        torch.device for the selected device.

    Raises:
        RuntimeError: If 'cuda' is requested but not available.
    """
    if preference == "cpu":
        return torch.device("cpu")

    if preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")

    # Auto mode
    return get_device()


def get_optimal_batch_size(frame_width: int, frame_height: int, model_memory_mb: int = 500) -> int:
    """
    Estimate optimal batch size based on available GPU memory.

    Args:
        frame_width: Width of video frames.
        frame_height: Height of video frames.
        model_memory_mb: Estimated memory used by the model.

    Returns:
        Recommended batch size (minimum 1).
    """
    if not torch.cuda.is_available():
        return 1

    gpu_info = get_gpu_info()
    available_mb = gpu_info.memory_free_mb - model_memory_mb - 500  # Safety margin

    if available_mb <= 0:
        return 1

    # Estimate memory per frame (RGB float32 + depth float32 + working space)
    bytes_per_pixel = 4 * 4  # 4 channels * 4 bytes (float32)
    frame_memory_mb = (frame_width * frame_height * bytes_per_pixel) / (1024 * 1024)
    # Account for intermediate tensors (roughly 3x the frame size)
    frame_memory_mb *= 3

    if frame_memory_mb <= 0:
        return 1

    batch_size = int(available_mb / frame_memory_mb)
    return max(1, min(batch_size, 16))  # Cap at 16 for reasonable memory usage
