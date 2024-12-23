"""Main application entry point for DeepFrame."""

import sys
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

from .ui.main_window import MainWindow
from .utils.config import Config


def load_stylesheet() -> str:
    """Load the application stylesheet."""
    style_path = Path(__file__).parent / "ui" / "styles.qss"
    if style_path.exists():
        return style_path.read_text()
    return ""


def main() -> int:
    """Main entry point for DeepFrame application."""
    # Enable high DPI scaling
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("DeepFrame")
    app.setApplicationVersion("0.1.0")
    app.setOrganizationName("DeepFrame")

    # Load configuration
    config = Config.load()

    # Apply stylesheet
    stylesheet = load_stylesheet()
    if stylesheet:
        app.setStyleSheet(stylesheet)

    # Create and show main window
    window = MainWindow(config)
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
