import tensorflow as tf
print(f"TensorFlow: {tf.__version__}")
print("Devices:", tf.config.list_physical_devices())

import torch
print("torch GPU: ", torch.cuda.is_available() )


# 检查是否可以使用 GPU
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# 打印 GPU 信息
if tf.config.list_physical_devices('GPU'):
    print("GPU is available")
    print("GPU Device: ", tf.config.list_physical_devices('GPU'))
else:
    print("GPU is not available")
