# deepframe

Convert 2D videos to VR-ready side-by-side 3D using AI depth estimation.

## Features
- MiDaS and Depth Anything V2 models
- Real-time preview (original, depth, SBS, anaglyph)
- Batch processing with queue
- GPU acceleration (CUDA) with CPU fallback
- Adjustable depth strength, eye separation, focus plane

## Install
```
uv sync
uv run python main.py
```

## Usage
Drop a video file into the window or use File > Open. Adjust depth settings in the right panel, preview with the mode buttons, then export.
