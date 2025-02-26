# main.py
"""
Entry point for the Tetris AI Training Application.

This script serves as the main entry point for starting the Tetris
AI training visualization. It initializes the required components
and launches the visual training interface.
"""

import os
import sys
from utils import setup_gpu
from trainer_visualization import VisualTetrisDQNTrainer

if __name__ == "__main__":
    # Initialize GPU/CPU configuration
    device_info = setup_gpu()
    
    # Create and run the visualizer
    visualizer = VisualTetrisDQNTrainer()
    visualizer.run()
