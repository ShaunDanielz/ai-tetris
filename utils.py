# utils.py
"""
Utility functions for the Tetris AI trainer.

This module provides helper functions for GPU setup and model conversion 
that are used by other components of the application.
"""

import os
import tensorflow as tf
from config import TFJS_AVAILABLE

def setup_gpu():
    """Configure GPU usage and return information about available devices"""
    # Check for available GPUs
    gpus = tf.config.list_physical_devices('GPU')
    cpus = tf.config.list_physical_devices('CPU')
    
    device_info = {
        'gpus_available': len(gpus),
        'gpu_names': [],
        'using_gpu': False,
        'memory_growth_enabled': False
    }
    
    if gpus:
        try:
            # Get GPU details
            for gpu in gpus:
                device_info['gpu_names'].append(gpu.name)
            
            # Enable memory growth to avoid allocating all GPU memory at once
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            device_info['memory_growth_enabled'] = True
            
            # By default, TensorFlow will use GPU if available, so we just need to confirm
            device_info['using_gpu'] = True
            
            print(f"GPU acceleration enabled. Found {len(gpus)} GPU(s):")
            for i, gpu_name in enumerate(device_info['gpu_names']):
                print(f"  {i+1}. {gpu_name}")
                
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
            print("Falling back to CPU")
            # Disable all GPUs if there was an error
            tf.config.set_visible_devices([], 'GPU')
            device_info['using_gpu'] = False
    else:
        print("No GPU found. Using CPU for training (slower).")
        print("Available CPUs:", len(cpus))
    
    return device_info

def convert_model_to_tfjs(h5_path, output_folder):
    """
    Convert a saved Keras/TensorFlow model to TensorFlow.js format.
    
    Args:
        h5_path: Path to the saved .h5 model file
        output_folder: Folder where the converted model will be saved
    
    Returns:
        bool: True if conversion was successful, False otherwise
    """
    if not TFJS_AVAILABLE:
        print("Cannot convert model: TensorFlow.js package not installed")
        print("Install with: pip install tensorflowjs")
        return False
    
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    try:
        # Convert the model
        import tensorflowjs as tfjs
        tfjs.converters.save_keras_model(tf.keras.models.load_model(h5_path), output_folder)
        print(f"Model successfully converted to TensorFlow.js format")
        print(f"Saved to: {output_folder}")
        return True
    except Exception as e:
        print(f"Error converting model: {e}")
        return False
