#!/usr/bin/env python3
"""DeepFrame - 2D to VR SBS Video Converter.

Main entry point for running the application.
"""

import sys


def main() -> int:
    """Run the DeepFrame application."""
    from src.app import main as app_main

    return app_main()


if __name__ == "__main__":
    sys.exit(main())
