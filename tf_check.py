import sys
import importlib.util
import subprocess
import os

def check_tensorflow():
    """Check TensorFlow installation and provide helpful information"""
    print("Checking Python environment...")
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    
    # Check if TensorFlow is installed
    tf_spec = importlib.util.find_spec("tensorflow")
    if tf_spec is None:
        print("\n[!] TensorFlow is not installed")
        print("    Try installing with: pip install tensorflow")
        return False
    
    # Import TensorFlow and check version
    try:
        import tensorflow as tf
        print(f"\nTensorFlow version: {tf.__version__}")
        print(f"TensorFlow path: {tf.__file__}")
        
        # Check for GPU support
        if hasattr(tf, 'config'):
            # TF 2.x
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                print(f"GPU detected: {len(gpus)} available")
                for i, gpu in enumerate(gpus):
                    print(f"  GPU {i+1}: {gpu}")
            else:
                print("No GPU detected, running on CPU only")
        else:
            # TF 1.x
            from tensorflow.python.client import device_lib
            devices = device_lib.list_local_devices()
            gpus = [d for d in devices if d.device_type == 'GPU']
            if gpus:
                print(f"GPU detected: {len(gpus)} available")
                for i, gpu in enumerate(gpus):
                    print(f"  GPU {i+1}: {gpu.name}")
            else:
                print("No GPU detected, running on CPU only")
        
        # Check CUDA and cuDNN for GPU versions
        if hasattr(tf, 'sysconfig'):
            print("\nTensorFlow build information:")
            build_info = tf.sysconfig.get_build_info()
            for key, value in build_info.items():
                if 'cuda' in key.lower() or 'cudnn' in key.lower():
                    print(f"  {key}: {value}")
        
        # Check TensorFlow.js availability
        try:
            import tensorflowjs
            print(f"\nTensorFlow.js version: {tensorflowjs.__version__}")
            print(f"TensorFlow.js path: {tensorflowjs.__file__}")
        except ImportError:
            print("\nTensorFlow.js is not installed")
            print("Install with: pip install tensorflowjs")
        
        return True
    
    except Exception as e:
        print(f"\n[!] Error importing TensorFlow: {e}")
        print("    There might be an issue with your TensorFlow installation")
        return False

if __name__ == "__main__":
    print("TensorFlow Installation Check Tool")
    print("=================================")
    check_tensorflow()
    
    print("\nPIP List (TensorFlow related packages):")
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "list"], 
                               capture_output=True, text=True, check=True)
        packages = result.stdout.splitlines()
        filtered_packages = [p for p in packages if any(name in p.lower() 
                              for name in ['tensorflow', 'keras', 'cuda', 'cudnn', 'gpu'])]
        for package in filtered_packages:
            print(f"  {package}")
    except Exception as e:
        print(f"Error running pip list: {e}")