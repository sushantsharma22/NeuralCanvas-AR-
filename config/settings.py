"""
Configuration settings for NeuralCanvas AR
"""
import numpy as np


class Config:
    # Camera settings
    CAMERA_WIDTH = 1280
    CAMERA_HEIGHT = 720
    FPS_TARGET = 60

    # Hand tracking
    MAX_HANDS = 2
    DETECTION_CONFIDENCE = 0.8
    TRACKING_CONFIDENCE = 0.7

    # Drawing settings
    DEFAULT_BRUSH_SIZE = 8
    MAX_BRUSH_SIZE = 100
    MIN_BRUSH_SIZE = 2
    SMOOTHING_FACTOR = 0.8

    # 3D drawing
    DEPTH_LAYERS = 10
    Z_SENSITIVITY = 0.5

    # AI prediction
    PREDICTION_BUFFER_SIZE = 20
    PREDICTION_THRESHOLD = 0.5  # Lower threshold for better gesture detection

    # Colors (BGR format)
    COLORS = {
        "neon_blue": (255, 100, 0),
        "electric_pink": (180, 0, 255),
        "laser_green": (0, 255, 100),
        "plasma_orange": (0, 150, 255),
        "cyber_purple": (255, 0, 150),
        "atomic_yellow": (0, 255, 255),
        "quantum_red": (0, 0, 255),
        "hologram_white": (255, 255, 255),
    }

    # Gesture commands
    GESTURES = {
        "DRAW": [0, 1, 0, 0, 0],          # Index finger
        "NAVIGATE": [0, 1, 1, 0, 0],      # Index + Middle
        "ERASE": [1, 1, 1, 1, 1],         # All fingers
        "COLOR_CHANGE": [1, 1, 0, 0, 0],  # Thumb + Index
        "SAVE": [0, 1, 1, 1, 0],          # Peace sign + middle
        "CLEAR": [1, 1, 1, 1, 0],          # Four fingers
    }
