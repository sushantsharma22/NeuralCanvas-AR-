
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
- **Emotion-based Coloring**: Facial expression analysis for automatic color selection
- **Voice Control**: Natural language commands for tool selection and canvas management
- **3D/2D Mode Toggle**: Support for both 2D drawing and 3D space interaction
- **Session Recording**: Capture and replay drawing sessions
- **Export Options**: Save artwork in multiple formats (PNG, JPEG)

## System Requirements

### Hardware
- Webcam (minimum 720p resolution recommended)
- 4GB RAM minimum, 8GB recommended
- Modern CPU with support for OpenCV operations
- Optional: Microphone for voice commands

### Software
- Python 3.8 or higher
- OpenCV 4.8+
- MediaPipe 0.10+
- TensorFlow 2.13+
- See `requirements.txt` for complete dependency list

## Installation

### Automatic Installation
```bash
chmod +x INSTALL.sh
./INSTALL.sh
```

### Manual Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/NeuralCanvas-AR.git
cd NeuralCanvas-AR
```

2. Create and activate virtual environment:
```bash
python -m venv neuralcanvas_env
source neuralcanvas_env/bin/activate  # On Windows: neuralcanvas_env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Operation
```bash
# Activate virtual environment
source neuralcanvas_env/bin/activate

# Start NeuralCanvas AR
python main.py
```

### Gesture Controls

| Gesture | Action | Description |
|---------|--------|-------------|
| Index finger only | Draw | Primary drawing mode |
| Index + Middle finger | Navigate | Move without drawing |
| All fingers extended | Erase | Eraser tool |
| Thumb + Index finger | Color Change | Cycle through color palette |
| Index + Pinky finger | Shape Mode | Toggle shape recognition |
| Thumb + Pinky finger | Voice Control | Activate/deactivate voice commands |
| Three fingers | Save | Save current artwork |
| Four fingers | Clear | Clear entire canvas |
| Closed fist | Idle | Stop all drawing operations |

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `q` | Quit application |
| `c` | Clear canvas |
| `s` | Save artwork |
| `r` | Toggle recording |
| `e` | Toggle emotion coloring |
| `3` | Toggle 3D/2D mode |
| `+/-` | Adjust brush size |
| `u` | Undo last stroke |

### Voice Commands
- "draw" - Switch to drawing mode
- "erase" - Switch to eraser mode
- "red/blue/green" - Change color
- "clear" - Clear canvas
- "save" - Save artwork
- "bigger/smaller" - Adjust brush size

## Project Structure

```
NeuralCanvas-AR/
├── main.py                    # Main application entry point
├── requirements.txt           # Python dependencies
├── INSTALL.sh                # Automated installation script
├── config/
│   ├── __init__.py
│   └── settings.py           # Configuration parameters
├── core/
│   ├── __init__.py
│   ├── ai_predictor.py       # AI-based prediction algorithms
│   ├── ar_renderer.py        # Augmented reality rendering
│   ├── drawing_engine.py     # Core drawing functionality
│   └── gesture_engine.py     # Hand gesture recognition
├── modules/
│   ├── __init__.py
│   ├── background_processor.py # Background image processing
│   ├── face_analyzer.py      # Facial expression analysis
│   ├── hand_tracker.py       # Hand tracking algorithms
│   ├── session_recorder.py   # Session recording functionality
│   ├── shape_recognizer.py   # Geometric shape recognition
│   └── voice_controller.py   # Voice command processing
├── ui/
│   ├── __init__.py
│   ├── color_palette.py      # Color management interface
│   ├── interface_manager.py  # UI coordination
│   └── toolbar.py           # Drawing tools interface
├── utils/
│   ├── __init__.py
│   ├── file_manager.py       # File I/O operations
│   ├── math_helpers.py       # Mathematical utilities
│   └── performance_monitor.py # Performance tracking
├── assets/
│   ├── models/              # AI model files
│   ├── sounds/              # Audio feedback files
│   └── textures/            # UI textures and images
└── exports/
    ├── images/              # Saved artwork
    ├── models/              # 3D model exports
    └── videos/              # Recorded sessions
```

## Configuration

The application can be configured through `config/settings.py`:

```python
# Camera settings
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
FPS_TARGET = 60

# Drawing parameters
BRUSH_SIZE_DEFAULT = 8
BRUSH_SIZE_MIN = 2
BRUSH_SIZE_MAX = 50

# Gesture recognition
PREDICTION_THRESHOLD = 0.5
MAX_HANDS = 2

# Performance
ENABLE_PERFORMANCE_LOGGING = True
```

## API Reference

### Core Classes

#### NeuralCanvasAR
Main application class that coordinates all components.

#### AdvancedGestureEngine
Handles gesture recognition and classification.
- `recognize_gesture(landmarks)` - Classify hand gesture
- `get_confidence_score()` - Get recognition confidence

#### NeuralDrawingEngine
Manages drawing operations and canvas state.
- `start_stroke(x, y, pressure, z)` - Begin new drawing stroke
- `draw_point(x, y, pressure, z)` - Add point to current stroke
- `end_stroke()` - Complete current stroke

#### AdvancedHandTracker
Provides hand detection and tracking capabilities.
- `process_frame(frame)` - Process video frame for hands
- `get_landmarks()` - Extract hand landmark coordinates

## Performance Optimization

The application includes several optimization techniques:

- **Frame Buffering**: Intelligent frame management for consistent performance
- **Coordinate Smoothing**: Velocity-based smoothing for natural drawing
- **Memory Management**: Efficient canvas and stroke data handling
- **Threading**: Background processing for non-critical operations

## Testing

The project includes comprehensive test suites:

```bash
# Test drawing functionality
python test_drawing.py

# Test smooth drawing algorithms
python test_smooth_drawing.py

# Test gesture recognition
python test_closed_hand.py

# Comprehensive system test
python comprehensive_test.py
```

## Troubleshooting

### Common Issues

**Camera not detected:**
- Ensure webcam is connected and not in use by other applications
- Try changing camera index in settings
- Check camera permissions

**Poor gesture recognition:**
- Ensure adequate lighting
- Position hand clearly in camera view
- Calibrate gesture thresholds in settings

**Performance issues:**
- Lower camera resolution in settings
- Disable performance logging
- Close other resource-intensive applications

**Dependencies not installing:**
- Ensure Python 3.8+ is installed
- Use virtual environment
- Update pip: `pip install --upgrade pip`

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add comprehensive docstrings
- Include unit tests for new features
- Update documentation as needed

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- MediaPipe team for hand tracking technology
- OpenCV community for computer vision tools
- TensorFlow team for machine learning framework
- Contributors and testers who helped improve the application

## Version History

- **v1.0.0** - Initial release with basic drawing functionality
- **v1.1.0** - Added gesture recognition and voice control
- **v1.2.0** - Ultra-smooth drawing engine with closed-hand detection
- **v1.3.0** - Enhanced performance optimization and stability improvements

## Support

For support, bug reports, or feature requests, please open an issue on the GitHub repository or contact the development team.

---

*NeuralCanvas AR - Transforming digital art creation through advanced computer vision and gesture recognition technology.*
