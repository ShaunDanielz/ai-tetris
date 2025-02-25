# test_gpu.py

# Check if GPU is available via tensorflow

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))