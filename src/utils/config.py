"""Configuration management for DeepFrame.

Provides XML (de)serialization for preset handling with versioning.
"""

import json
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
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


CURRENT_PRESET_VERSION = "1.0"


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

    # ---------------------------------------------------------------------
    # XML preset persistence (versioned)
    # ---------------------------------------------------------------------
    @staticmethod
    def _bool_to_str(v: bool) -> str:
        return "true" if v else "false"

    @staticmethod
    def _str_to_bool(s: str) -> bool:
        return s.lower() in ("true", "1", "yes")

    def to_xml(self) -> str:
        """Serialize the current configuration to a versioned XML string.

        Returns:
            str: Pretty‑printed XML document.
        """
        root = ET.Element("Preset", {"version": CURRENT_PRESET_VERSION})

        # Depth block
        depth_el = ET.SubElement(root, "Depth")
        ET.SubElement(depth_el, "Model").text = self.depth.model.value
        ET.SubElement(depth_el, "Strength").text = str(self.depth.depth_strength)
        ET.SubElement(depth_el, "EyeSeparation").text = str(self.depth.eye_separation)
        ET.SubElement(depth_el, "Focus").text = str(self.depth.depth_focus)
        ET.SubElement(depth_el, "FillMode").text = self.depth.fill_mode.value
        ET.SubElement(depth_el, "Layout").text = self.depth.output_layout.value

        # Output block
        out_el = ET.SubElement(root, "Output")
        ET.SubElement(out_el, "Codec").text = self.output.codec.value
        ET.SubElement(out_el, "Quality").text = str(self.output.quality)
        ET.SubElement(out_el, "PreserveResolution").text = self._bool_to_str(
            self.output.preserve_resolution
        )
        ET.SubElement(out_el, "CustomWidth").text = (
            "" if self.output.custom_width is None else str(self.output.custom_width)
        )
        ET.SubElement(out_el, "CustomHeight").text = (
            "" if self.output.custom_height is None else str(self.output.custom_height)
        )

        # App block
        app_el = ET.SubElement(root, "App")
        ET.SubElement(app_el, "DevicePreference").text = self.app.device_preference
        ET.SubElement(app_el, "DarkTheme").text = self._bool_to_str(self.app.dark_theme)
        ET.SubElement(app_el, "PreviewMode").text = self.app.preview_mode

        # Pretty‑print using minidom
        rough = ET.tostring(root, "utf-8")
        reparsed = minidom.parseString(rough)
        return reparsed.toprettyxml(indent="  ")

    @classmethod
    def from_xml(cls, xml_str: str) -> "Config":
        """Parse a preset XML string and return a Config instance.

        Raises:
            ValueError: If the preset version does not match ``CURRENT_PRESET_VERSION``.
        """
        root = ET.fromstring(xml_str)
        file_version = root.attrib.get("version", "0.0")
        if file_version != CURRENT_PRESET_VERSION:
            raise ValueError(
                f"Preset version {file_version} does not match supported version {CURRENT_PRESET_VERSION}"
            )

        # Helper for optional int conversion
        def _opt_int(text: str | None) -> int | None:
            if text is None or text.strip() == "":
                return None
            return int(text)

        # Depth
        depth_el = root.find("Depth")
        if depth_el is None:
            raise ValueError("Missing <Depth> element in preset XML")
        depth = DepthSettings(
            model=DepthModel(depth_el.findtext("Model", "midas_small")),
            depth_strength=float(depth_el.findtext("Strength", "0.5")),
            eye_separation=int(depth_el.findtext("EyeSeparation", "63")),
            depth_focus=float(depth_el.findtext("Focus", "0.5")),
            fill_mode=FillMode(depth_el.findtext("FillMode", "inpaint")),
            output_layout=SBSLayout(depth_el.findtext("Layout", "half")),
        )

        # Output
        out_el = root.find("Output")
        if out_el is None:
            raise ValueError("Missing <Output> element in preset XML")
        output = OutputSettings(
            codec=VideoCodec(out_el.findtext("Codec", "libx264")),
            quality=int(out_el.findtext("Quality", "23")),
            preserve_resolution=cls()._str_to_bool(out_el.findtext("PreserveResolution", "true")),
            custom_width=_opt_int(out_el.findtext("CustomWidth")),
            custom_height=_opt_int(out_el.findtext("CustomHeight")),
        )

        # App
        app_el = root.find("App")
        app = AppSettings(
            device_preference=app_el.findtext("DevicePreference", "auto"),
            last_input_dir=app_el.findtext("LastInputDir", ""),
            last_output_dir=app_el.findtext("LastOutputDir", ""),
            dark_theme=cls()._str_to_bool(app_el.findtext("DarkTheme", "true")),
            preview_mode=app_el.findtext("PreviewMode", "sbs"),
        )

        cfg = cls(depth=depth, output=output, app=app)
        cfg._config_path = None
        return cfg
