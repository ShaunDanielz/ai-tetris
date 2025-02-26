# config.py
"""
Configuration settings for the Tetris AI trainer.

This module contains global constants, color definitions, and tetromino shapes
that are used throughout the application. Centralizing these values makes it easier
to modify game parameters without changing multiple files.
"""

import numpy as np
import random

# Check if tensorflowjs is available
try:
    import tensorflowjs as tfjs
    TFJS_AVAILABLE = True
    print("TensorFlow.js converter available - Web export enabled")
except ImportError:
    TFJS_AVAILABLE = False
    print("TensorFlow.js converter not available. To enable web export, install with: pip install tensorflowjs")

# Define colorblind-friendly colors
COLORBLIND_COLORS = [
    (0, 114, 178),    # Blue
    (230, 159, 0),    # Orange/Amber
    (0, 158, 115)     # Green
]

# More vibrant versions for UI
VIBRANT_COLORS = [
    (0, 144, 238),    # Bright Blue
    (255, 179, 0),    # Bright Orange
    (0, 198, 145)     # Bright Green
]

# UI Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)
GRID_COLOR = (50, 50, 50)

# Tetromino shapes
TETROMINOS = {
    'I': {
        'shape': np.array([
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])
    },
    'J': {
        'shape': np.array([
            [0, 0, 0],
            [2, 2, 2],
            [0, 0, 2]
        ])
    },
    'L': {
        'shape': np.array([
            [0, 0, 0],
            [3, 3, 3],
            [3, 0, 0]
        ])
    },
    'O': {
        'shape': np.array([
            [4, 4],
            [4, 4]
        ])
    },
    'S': {
        'shape': np.array([
            [0, 0, 0],
            [0, 5, 5],
            [5, 5, 0]
        ])
    },
    'T': {
        'shape': np.array([
            [0, 0, 0],
            [6, 6, 6],
            [0, 6, 0]
        ])
    },
    'Z': {
        'shape': np.array([
            [0, 0, 0],
            [7, 7, 0],
            [0, 7, 7]
        ])
    }
}
