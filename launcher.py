#!/usr/bin/env python3
"""
Tetris DQN Trainer Launcher

This script sets up and runs the Tetris DQN trainer with web export capability.
Run this script to start the application.

Requirements:
- tensorflow
- pygame
- numpy
- tensorflowjs (optional, for web export)

Install with:
pip install tensorflow pygame numpy
pip install tensorflowjs  # Optional, for web export
"""

import os
import sys

def check_dependencies():
    """Check if all required dependencies are installed"""
    try:
        import tensorflow as tf
        import pygame
        import numpy as np
        
        # Check for tensorflowjs
        try:
            import tensorflowjs
            print("[✓] tensorflowjs is installed - Web export will be available")
        except ImportError:
            print("[!] tensorflowjs is not installed - Web export will not be available")
            print("    Install with: pip install tensorflowjs")
        
        print("[✓] All required dependencies are installed")
        return True
        
    except ImportError as e:
        print(f"[✗] Missing dependency: {e}")
        print("\nPlease install all required dependencies:")
        print("pip install tensorflow pygame numpy")
        print("pip install tensorflowjs  # Optional, for web export")
        return False

def create_directories():
    """Create necessary directories for models and exports"""
    directories = [
        "models",
        "models/tetris_dqn",
        "models/tetris_dqn/milestones",
        "models/tetris_dqn/web_model_latest"
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"[✓] Created directory: {directory}")

def main():
    """Main function to run the DQN trainer"""
    # Check dependencies
    if not check_dependencies():
        return
    
    # Create necessary directories
    create_directories()
    
    # Import the trainer after checks
    print("[*] Starting Tetris DQN Trainer...")
    
    # Import only after checks to avoid errors if dependencies are missing
    from ai_trainer_full import VisualTetrisDQNTrainer
    
    # Create and run the visualizer
    visualizer = VisualTetrisDQNTrainer()
    visualizer.run()

if __name__ == "__main__":
    main()