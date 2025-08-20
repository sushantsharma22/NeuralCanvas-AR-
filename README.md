
# NeuralCanvas AR

A real-time augmented reality drawing application that enables users to create digital artwork using hand gestures and finger movements. The system combines computer vision, machine learning, and advanced gesture recognition to provide an intuitive and responsive drawing experience.

## Overview

NeuralCanvas AR transforms any webcam-enabled device into a virtual canvas where users can draw, erase, and create digital art using natural hand movements. The application features ultra-smooth line rendering, intelligent gesture recognition, voice commands, and real-time performance optimization.

## Features

### Core Functionality
- **Real-time Hand Tracking**: Advanced MediaPipe-based hand detection and tracking
- **Gesture Recognition**: Intelligent finger pattern recognition for various drawing modes
- **Ultra-smooth Drawing**: Professional-grade line smoothing and interpolation algorithms
- **Multi-modal Input**: Support for hand gestures, voice commands, and keyboard shortcuts
- **Real-time Performance**: Optimized for 60 FPS operation with minimal latency

### Drawing Capabilities
- **Precision Drawing**: Sub-pixel accuracy with pressure sensitivity simulation
- **Brush System**: Variable brush sizes with anti-aliased rendering
- **Color Management**: Dynamic color selection and palette management
- **Stroke Management**: Intelligent stroke start/stop detection with coordinate jump handling
- **Eraser Tool**: Precision erasing with adjustable thickness

### Advanced Features
- **Emotion-based Coloring**: (optional) Facial expression analysis for automatic color selection — currently disabled in this build
- **Voice Control**: Removed in this build
- **3D/2D Mode Toggle**: Support for both 2D drawing and 3D space interaction
- **Session Recording**: Capture and replay drawing sessions
# NeuralCanvas AR

A real-time augmented reality drawing application that enables users to create digital artwork using hand gestures and finger movements. The system combines computer vision, machine learning, and gesture recognition to provide an intuitive and responsive drawing experience.

## Overview

NeuralCanvas AR turns a webcam-enabled device into a virtual canvas. Use natural hand movements to draw, erase, change colors, save, and export artwork. This README focuses on how to set up and use the project.

## Key Features

- Real-time hand tracking and gesture recognition (MediaPipe-based)
- Ultra-smooth drawing with velocity-aware smoothing
- Color palette and quick color-change gesture
- Save and export artwork (PNG/JPEG)
- Session recording and replay

> Note: Voice control and emotion-based coloring are not enabled in this build.

## System Requirements

- Webcam (720p recommended)
- Python 3.8 or higher
- 4GB RAM minimum, 8GB recommended

See `requirements.txt` for full dependency details.

## Installation

Automatic:

```bash
chmod +x INSTALL.sh
./INSTALL.sh
```

Manual:

```bash
git clone https://github.com/yourusername/NeuralCanvas-AR.git
cd NeuralCanvas-AR
python -m venv neuralcanvas_env
source neuralcanvas_env/bin/activate
pip install -r requirements.txt
```

## Usage

Activate the environment and run the app:

```bash
source neuralcanvas_env/bin/activate
python main.py
```

### Gesture Controls (summary)

- Index finger only: Draw
- Index + Middle finger: Navigate
- All fingers extended (4): Clear canvas
- Three fingers: Save artwork
- Thumb + Index finger (tap): Cycle color palette
- Closed fist: Idle / stop drawing

### Keyboard Shortcuts

- `q` — Quit
- `c` — Clear canvas
- `s` — Save artwork
- `r` — Toggle recording
- `3` — Toggle 3D/2D mode
- `+` / `-` — Adjust brush size
- `u` — Undo last stroke

## Project Structure (high level)

```
NeuralCanvas-AR/
├── main.py
├── requirements.txt
├── INSTALL.sh
├── config/
├── core/
├── modules/
├── ui/
├── utils/
├── assets/
└── exports/
```

## Configuration

Edit `config/settings.py` for camera, drawing, and gesture thresholds (camera width/height, brush sizes, prediction thresholds, etc.).

## API Reference (brief)

- `NeuralCanvasAR` — main application class coordinating components
- `AdvancedGestureEngine` — gesture classification
- `NeuralDrawingEngine` — drawing and canvas management
- `AdvancedHandTracker` — hand detection and landmark extraction

## Testing

Run the included tests:

```bash
python test_drawing.py
python test_smooth_drawing.py
python test_closed_hand.py
python comprehensive_test.py
```

## Troubleshooting

- Camera not detected: check permissions and camera index in `config/settings.py`.
- Poor gesture recognition: improve lighting and keep hand in view; tune thresholds in config.
- Performance issues: reduce camera resolution and/or disable performance logging.

## License

This project is licensed under the MIT License — see `LICENSE` for details.

## Version History

- v1.3.0 — Latest: performance and stability improvements
- v1.2.0 — Ultra-smooth drawing engine and closed-hand detection
- v1.1.0 — Gesture recognition added
- v1.0.0 — Initial release

---

NeuralCanvas AR — Real-time AR drawing with hand gestures.
