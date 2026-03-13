"""Settings panel for depth and output configuration."""

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..utils.config import (
    Config,
    DepthModel,
    FillMode,
    SBSLayout,
    VideoCodec,
)


class LabeledSlider(QWidget):
    """Slider with a value label."""

    valueChanged = pyqtSignal(int)

    def __init__(
        self,
        min_val: int = 0,
        max_val: int = 100,
        initial: int = 50,
        suffix: str = "",
        divisor: int = 1,
    ) -> None:
        super().__init__()
        self._divisor = divisor
        self._suffix = suffix

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self.slider = QSlider()
        self.slider.setOrientation(Qt.Orientation.Horizontal)
        self.slider.setMinimum(min_val)
        self.slider.setMaximum(max_val)
        self.slider.setValue(initial)
        self.slider.valueChanged.connect(self._on_value_changed)

        self.label = QLabel(self._format_value(initial))
        self.label.setMinimumWidth(50)

        row = QWidget()
        row_layout = QVBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.addWidget(self.slider)

        layout.addWidget(row)
        layout.addWidget(self.label)

    def _format_value(self, value: int) -> str:
        if self._divisor > 1:
            return f"{value / self._divisor:.2f}{self._suffix}"
        return f"{value}{self._suffix}"

    def _on_value_changed(self, value: int) -> None:
        self.label.setText(self._format_value(value))
        self.valueChanged.emit(value)

    def value(self) -> int:
        return self.slider.value()

    def setValue(self, value: int) -> None:
        self.slider.setValue(value)


class SettingsPanel(QWidget):
    """Panel for configuring depth and output settings."""

    settings_changed = pyqtSignal()
    export_clicked = pyqtSignal()

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self._setup_ui()
        self._load_from_config()

    def _setup_ui(self) -> None:
        """Set up the settings panel UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Depth Settings Group
        depth_group = QGroupBox("Depth Settings")
        depth_layout = QFormLayout(depth_group)

        # Model selection
        self.model_combo = QComboBox()
        model_options = [
            ("MiDaS Small (Fast)", DepthModel.MIDAS_SMALL),
            ("MiDaS Large", DepthModel.MIDAS_LARGE),
            ("MiDaS Hybrid", DepthModel.MIDAS_HYBRID),
            ("Depth Anything ViT-S", DepthModel.DEPTH_ANYTHING_VITS),
            ("Depth Anything ViT-B", DepthModel.DEPTH_ANYTHING_VITB),
            ("Depth Anything ViT-L (Best)", DepthModel.DEPTH_ANYTHING_VITL),
        ]
        for label, value in model_options:
            self.model_combo.addItem(label, value)
        self.model_combo.currentIndexChanged.connect(self._on_setting_changed)
        depth_layout.addRow("Model:", self.model_combo)

        # Depth strength
        self.strength_slider = LabeledSlider(0, 100, 50, "", 100)
        self.strength_slider.valueChanged.connect(self._on_setting_changed)
        depth_layout.addRow("Depth Strength:", self.strength_slider)

        # Eye separation
        self.eye_sep_spin = QSpinBox()
        self.eye_sep_spin.setRange(20, 150)
        self.eye_sep_spin.setValue(63)
        self.eye_sep_spin.setSuffix(" px")
        self.eye_sep_spin.valueChanged.connect(self._on_setting_changed)
        depth_layout.addRow("Eye Separation:", self.eye_sep_spin)

        # Depth focus
        self.focus_slider = LabeledSlider(0, 100, 50, "", 100)
        self.focus_slider.valueChanged.connect(self._on_setting_changed)
        depth_layout.addRow("Depth Focus:", self.focus_slider)

        # Fill mode
        self.fill_combo = QComboBox()
        fill_options = [
            ("Inpaint (Recommended)", FillMode.INPAINT),
            ("Stretch Edges", FillMode.STRETCH),
            ("Black Fill", FillMode.BLACK),
        ]
        for label, value in fill_options:
            self.fill_combo.addItem(label, value)
        self.fill_combo.currentIndexChanged.connect(self._on_setting_changed)
        depth_layout.addRow("Fill Mode:", self.fill_combo)

        layout.addWidget(depth_group)

        # Output Settings Group
        output_group = QGroupBox("Output Settings")
        output_layout = QFormLayout(output_group)

        # SBS Layout
        self.layout_combo = QComboBox()
        layout_options = [
            ("Half SBS (VR Compatible)", SBSLayout.HALF),
            ("Full SBS (2x Width)", SBSLayout.FULL),
        ]
        for label, value in layout_options:
            self.layout_combo.addItem(label, value)
        self.layout_combo.currentIndexChanged.connect(self._on_setting_changed)
        output_layout.addRow("SBS Layout:", self.layout_combo)

        # Codec
        self.codec_combo = QComboBox()
        codec_options = [
            ("H.264 (Most Compatible)", VideoCodec.H264),
            ("H.265/HEVC (Smaller Files)", VideoCodec.H265),
            ("VP9 (WebM)", VideoCodec.VP9),
        ]
        for label, value in codec_options:
            self.codec_combo.addItem(label, value)
        self.codec_combo.currentIndexChanged.connect(self._on_setting_changed)
        output_layout.addRow("Codec:", self.codec_combo)

        # Quality (CRF)
        self.quality_slider = LabeledSlider(15, 35, 23, "")
        self.quality_slider.valueChanged.connect(self._on_setting_changed)
        quality_help = QLabel("Lower = better quality, larger file")
        quality_help.setStyleSheet("color: #888; font-size: 11px;")
        output_layout.addRow("Quality (CRF):", self.quality_slider)
        output_layout.addRow("", quality_help)

        layout.addWidget(output_group)

        # Apply to all button
        self.apply_all_btn = QPushButton("Apply to All in Queue")
        self.apply_all_btn.clicked.connect(self._on_apply_all)
        layout.addWidget(self.apply_all_btn)

        # Export button
        self.export_btn = QPushButton("Export Selected")
        self.export_btn.setStyleSheet(
            "QPushButton { background-color: #2d5a27; font-weight: bold; padding: 8px; }"
            "QPushButton:hover { background-color: #3d7a37; }"
        )
        self.export_btn.clicked.connect(self.export_clicked.emit)
        layout.addWidget(self.export_btn)

        layout.addStretch()

    def load_from_config(self) -> None:
        """Public wrapper to load settings from the attached Config.
        This method is called by MainWindow when applying a preset.
        """
        self._load_from_config()

    def _load_from_config(self) -> None:
        """Load settings from config."""
        # Depth settings
        model_idx = self.model_combo.findData(self.config.depth.model)
        if model_idx >= 0:
            self.model_combo.setCurrentIndex(model_idx)

        self.strength_slider.setValue(int(self.config.depth.depth_strength * 100))
        self.eye_sep_spin.setValue(self.config.depth.eye_separation)
        self.focus_slider.setValue(int(self.config.depth.depth_focus * 100))

        fill_idx = self.fill_combo.findData(self.config.depth.fill_mode)
        if fill_idx >= 0:
            self.fill_combo.setCurrentIndex(fill_idx)

        layout_idx = self.layout_combo.findData(self.config.depth.output_layout)
        if layout_idx >= 0:
            self.layout_combo.setCurrentIndex(layout_idx)

        # Output settings
        codec_idx = self.codec_combo.findData(self.config.output.codec)
        if codec_idx >= 0:
            self.codec_combo.setCurrentIndex(codec_idx)

        self.quality_slider.setValue(self.config.output.quality)

    def _save_to_config(self) -> None:
        """Save settings to config."""
        # Depth settings
        self.config.depth.model = self.model_combo.currentData()
        self.config.depth.depth_strength = self.strength_slider.value() / 100
        self.config.depth.eye_separation = self.eye_sep_spin.value()
        self.config.depth.depth_focus = self.focus_slider.value() / 100
        self.config.depth.fill_mode = self.fill_combo.currentData()
        self.config.depth.output_layout = self.layout_combo.currentData()

        # Output settings
        self.config.output.codec = self.codec_combo.currentData()
        self.config.output.quality = self.quality_slider.value()

    def _on_setting_changed(self) -> None:
        """Handle any setting change."""
        self._save_to_config()
        self.settings_changed.emit()

    def _on_apply_all(self) -> None:
        """Apply current settings to all queue items."""
        self._save_to_config()
        # Settings are shared, so just emit the signal
        self.settings_changed.emit()
