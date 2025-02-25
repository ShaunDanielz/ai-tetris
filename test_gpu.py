# test_gpu.py

# Check if GPU is available via tensorflow

import tensorflow as tf
import tensorflowjs as tfjs


print(f"TensorFlow version: {tf.__version__}")
print(f"TensorFlow.js version: {tfjs.__version__}")

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))