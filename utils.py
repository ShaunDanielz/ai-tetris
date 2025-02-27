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
    import time  # Add this import
    
    if not TFJS_AVAILABLE:
        print("Cannot convert model: TensorFlow.js package not installed")
        print("Install with: pip install tensorflowjs")
        return False
    
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    try:
        # Load the model
        model = tf.keras.models.load_model(h5_path)
        
        # Print model information for debugging
        print(f"\nModel Information for TensorFlow.js Export:")
        print(f"Input shape: {model.input_shape}")
        print(f"Output shape: {model.output_shape}")
        
        # Add additional metadata to help with loading
        metadata = {
            "modelType": "tetris_dqn",
            "inputShape": list(map(lambda x: int(x) if x is not None else None, model.input_shape)),
            "outputShape": list(map(lambda x: int(x) if x is not None else None, model.output_shape)),
            "stateSize": int(model.input_shape[1]),
            "actionSize": int(model.output_shape[1]),
            "version": "1.0.0"
        }
        
        # Convert the model with options that your version supports
        import tensorflowjs as tfjs
        
        # Get the version of tensorflowjs to adapt parameters
        tfjs_version = getattr(tfjs, "__version__", "unknown")
        print(f"TensorFlow.js Converter version: {tfjs_version}")
        
        # Adapt conversion parameters based on available options
        try:
            # Try using the weight_shard_size_bytes parameter (newer versions)
            tfjs.converters.save_keras_model(
                model, 
                output_folder,
                weight_shard_size_bytes=1024 * 1024 * 4,  # 4MB per shard
                metadata=metadata
            )
        except TypeError:
            # Fall back to basic conversion if the above fails
            tfjs.converters.save_keras_model(
                model, 
                output_folder,
                metadata=metadata
            )
        
        # Save a separate JSON file with additional metadata
        import json
        import datetime  # Alternative to time if needed
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(os.path.join(output_folder, "model_info.json"), "w") as f:
            json.dump({
                "metadata": metadata,
                "stateSize": int(model.input_shape[1]),
                "actionSize": int(model.output_shape[1]),
                "modelVersion": "1.0.0",
                "exportDate": current_time,
                "loadingInstructions": "Load both model.json and weights files"
            }, f, indent=2)
        
        print(f"\nModel successfully converted to TensorFlow.js format")
        print(f"Saved to: {output_folder}")
        print(f"Model files:")
        for file in os.listdir(output_folder):
            print(f" - {file} ({os.path.getsize(os.path.join(output_folder, file)) / 1024:.1f} KB)")
        
        return True
    except Exception as e:
        print(f"Error converting model: {e}")
        import traceback
        traceback.print_exc()
        return False