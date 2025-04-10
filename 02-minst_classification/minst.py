import tensorflow as tf
from keras import layers, models
# from keras.src.datasets import mnist
# from keras.src.utils import to_categorical
from keras.datasets import mnist
from keras.utils import to_categorical

# Keras 是一个高级神经网络 API，运行在 TensorFlow 之上，提供了快速构建和训练深度学习模型的能力。
# 常用模块介绍：
# 1. layers: 提供了各种神经网络层，例如 Dense、Conv2D、MaxPooling2D 等。
# 2. models: 用于构建和管理模型，包括 Sequential 和 Functional API。
# 3. datasets: 提供了常用数据集的加载接口，例如 MNIST、CIFAR-10 等。
# 4. utils: 提供了辅助工具，例如 one-hot 编码、模型保存与加载等。

# 加载MNIST数据集（手写数字数据集）
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 数据预处理
# 将训练集和测试集的图像数据调整为 (28, 28, 1) 的形状，并归一化到 [0, 1] 范围
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1)).astype('float32') / 255
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1)).astype('float32') / 255
# 将标签转换为独热编码格式，例如标签3会变成 [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 构建卷积神经网络模型
model = models.Sequential([
  # 第一层卷积层，使用32个3x3的卷积核，激活函数为ReLU，输入形状为28x28x1
  layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
  # 第一层池化层，使用2x2的最大池化
  layers.MaxPooling2D((2, 2)),
  # 第二层卷积层，使用64个3x3的卷积核，激活函数为ReLU
  layers.Conv2D(64, (3, 3), activation='relu'),
  # 第二层池化层，使用2x2的最大池化
  layers.MaxPooling2D((2, 2)),
  # 第三层卷积层，使用64个3x3的卷积核，激活函数为ReLU
  layers.Conv2D(64, (3, 3), activation='relu'),
  # 将多维特征图展平为一维向量
  layers.Flatten(),
  # 全连接层，包含64个神经元，激活函数为ReLU
  layers.Dense(64, activation='relu'),
  # 输出层，包含10个神经元（对应10个类别），激活函数为softmax，用于多分类
  layers.Dense(10, activation='softmax')
])

# 编译模型
# 优化器使用Adam，损失函数为交叉熵，评估指标为准确率
model.compile(optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'])

# 训练模型
# 使用训练数据训练模型，训练5个周期，每批次大小为64，保留10%的数据用于验证
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1)

# 评估模型
# 使用测试数据评估模型的性能，输出测试集的损失和准确率
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc}")  # 打印测试集的准确率
