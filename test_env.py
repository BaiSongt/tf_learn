import tensorflow as tf
print(f"TensorFlow: {tf.__version__}")
print("Devices:", tf.config.list_physical_devices())

import torch
print("torch GPU: ", torch.cuda.is_available() )
