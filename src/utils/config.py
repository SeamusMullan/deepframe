"""Configuration management for DeepFrame."""

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class FillMode(str, Enum):
    """Mode for filling disoccluded regions in SBS output."""

    INPAINT = "inpaint"
    STRETCH = "stretch"
    BLACK = "black"


class SBSLayout(str, Enum):
    """Side-by-side output layout."""

    FULL = "full"  # Left|Right at full resolution (2x width)
    HALF = "half"  # Left|Right squeezed to original width


class DepthModel(str, Enum):
    """Available depth estimation models."""

    MIDAS_SMALL = "midas_small"
    MIDAS_LARGE = "midas_large"
    MIDAS_HYBRID = "midas_hybrid"
    DEPTH_ANYTHING_VITS = "depth_anything_vits"
    DEPTH_ANYTHING_VITB = "depth_anything_vitb"
    DEPTH_ANYTHING_VITL = "depth_anything_vitl"


class VideoCodec(str, Enum):
    """Supported output video codecs."""

    H264 = "libx264"
    H265 = "libx265"
    VP9 = "libvpx-vp9"


@dataclass
class DepthSettings:
    """Settings for depth estimation and SBS generation."""

    model: DepthModel = DepthModel.MIDAS_SMALL
    depth_strength: float = 0.5  # 0.0 - 1.0
    eye_separation: int = 63  # Pixels (typical IPD ~63mm scaled)
    depth_focus: float = 0.5  # 0.0 (near) - 1.0 (far)
    fill_mode: FillMode = FillMode.INPAINT
    output_layout: SBSLayout = SBSLayout.HALF


@dataclass
class OutputSettings:
    """Settings for video output."""

    codec: VideoCodec = VideoCodec.H264
    quality: int = 23  # CRF value (lower = better, 18-28 typical)
    preserve_resolution: bool = True
    custom_width: int | None = None
    custom_height: int | None = None


@dataclass
class AppSettings:
    """Application-wide settings."""

    device_preference: str = "auto"  # auto, cuda, cpu
    last_input_dir: str = ""
    last_output_dir: str = ""
    dark_theme: bool = True
    preview_mode: str = "sbs"  # original, depth, sbs, anaglyph


@dataclass
class Config:
    """Main configuration container."""

    depth: DepthSettings = field(default_factory=DepthSettings)
    output: OutputSettings = field(default_factory=OutputSettings)
    app: AppSettings = field(default_factory=AppSettings)

    _config_path: Path | None = field(default=None, repr=False)

    @classmethod
    def get_config_path(cls) -> Path:
        """Get the default configuration file path."""
        config_dir = Path.home() / ".config" / "deepframe"
        config_dir.mkdir(parents=True, exist_ok=True)
        return config_dir / "config.json"

    @classmethod
    def load(cls, path: Path | None = None) -> "Config":
        """Load configuration from file."""
        config_path = path or cls.get_config_path()

        if not config_path.exists():
            config = cls()
            config._config_path = config_path
            return config

        try:
            with open(config_path, "r") as f:
                data = json.load(f)

            config = cls(
                depth=DepthSettings(
                    model=DepthModel(data.get("depth", {}).get("model", "midas_small")),
                    depth_strength=data.get("depth", {}).get("depth_strength", 0.5),
                    eye_separation=data.get("depth", {}).get("eye_separation", 63),
                    depth_focus=data.get("depth", {}).get("depth_focus", 0.5),
                    fill_mode=FillMode(data.get("depth", {}).get("fill_mode", "inpaint")),
                    output_layout=SBSLayout(data.get("depth", {}).get("output_layout", "half")),
                ),
                output=OutputSettings(
                    codec=VideoCodec(data.get("output", {}).get("codec", "libx264")),
                    quality=data.get("output", {}).get("quality", 23),
                    preserve_resolution=data.get("output", {}).get("preserve_resolution", True),
                    custom_width=data.get("output", {}).get("custom_width"),
                    custom_height=data.get("output", {}).get("custom_height"),
                ),
                app=AppSettings(
                    device_preference=data.get("app", {}).get("device_preference", "auto"),
                    last_input_dir=data.get("app", {}).get("last_input_dir", ""),
                    last_output_dir=data.get("app", {}).get("last_output_dir", ""),
                    dark_theme=data.get("app", {}).get("dark_theme", True),
                    preview_mode=data.get("app", {}).get("preview_mode", "sbs"),
                ),
            )
            config._config_path = config_path
            return config

        except (json.JSONDecodeError, KeyError, ValueError):
            # Return default config on any error
            config = cls()
            config._config_path = config_path
            return config

    def save(self, path: Path | None = None) -> None:
        """Save configuration to file."""
        config_path = path or self._config_path or self.get_config_path()

        data = {
            "depth": {
                "model": self.depth.model.value,
                "depth_strength": self.depth.depth_strength,
                "eye_separation": self.depth.eye_separation,
                "depth_focus": self.depth.depth_focus,
                "fill_mode": self.depth.fill_mode.value,
                "output_layout": self.depth.output_layout.value,
            },
            "output": {
                "codec": self.output.codec.value,
                "quality": self.output.quality,
                "preserve_resolution": self.output.preserve_resolution,
                "custom_width": self.output.custom_width,
                "custom_height": self.output.custom_height,
            },
            "app": {
                "device_preference": self.app.device_preference,
                "last_input_dir": self.app.last_input_dir,
                "last_output_dir": self.app.last_output_dir,
                "dark_theme": self.app.dark_theme,
                "preview_mode": self.app.preview_mode,
            },
        }

        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(data, f, indent=2)

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "depth": asdict(self.depth),
            "output": asdict(self.output),
            "app": asdict(self.app),
        }
