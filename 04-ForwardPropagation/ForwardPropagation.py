import tensorflow as tf
import keras
from keras import datasets
import os

# 设置环境变量，减少 TensorFlow 的日志输出级别，避免打印过多无关信息
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# 加载 MNIST 数据集
# MNIST 是一个手写数字数据集，包含 60000 张训练图片和 10000 张测试图片
# 每张图片是 28x28 的灰度图像，标签是对应的数字（0-9）
(x, y), _ = datasets.mnist.load_data()

# 将 numpy 数组转换为 TensorFlow 的 tensor 数据类型
# x 转换为 float32 类型的张量，表示图像数据
# y 转换为 float32 类型的张量，表示标签数据
x = tf.convert_to_tensor(x, dtype=tf.float32)
y = tf.convert_to_tensor(y, dtype=tf.float32)

# 打印张量的形状和数据类型，检查数据是否加载正确
print(f"x: {x.shape},  {x.dtype}")  # 输出 x 的形状和数据类型
print(f"y: {y.shape},  {y.dtype}")  # 输出 y 的形状和数据类型

# 打印 x 和 y 的最小值和最大值，检查数据范围是否符合预期
print(f"x: min {tf.reduce_min(x)},   max {tf.reduce_max(x)}")
print(f"y: min {tf.reduce_min(y)},   max {tf.reduce_max(y)}")

# 对图像数据进行归一化处理
# 将像素值从 [0, 255] 缩放到 [0, 1] 范围，便于后续训练
x = x / 255

# 创建训练数据集
# 使用 tf.data.Dataset 将 (x, y) 数据对切片，并按批次大小 128 进行分组
# 这样可以提高训练效率，并便于在 GPU 上并行处理
train_db = tf.data.Dataset.from_tensor_slices((x, y)).batch(128)

# 创建一个迭代器，用于从数据集中获取批次数据
train_iter = iter(train_db)

# 获取第一个批次的数据
sample = next(train_iter)

# 打印批次数据的形状，确保数据分组正确
# sample[0] 是图像数据，形状为 [128, 28, 28]
# sample[1] 是标签数据，形状为 [128]
print(f"batch : {sample[0].shape}, {sample[1].shape}")

# 定义三层神经网络的权重和偏置
# 第一层：输入层的特征数为 784（28x28 的图像展平为一维向量），输出层的神经元数为 256
# 使用截断正态分布（truncated normal distribution）初始化权重，标准差为 0.1
# 截断正态分布会将随机值限制在均值两倍标准差范围内，避免出现过大的权重值

# 第一层权重矩阵，形状为 [784, 256]
w1 = tf.random.truncated_normal([784, 256], stddev=0.1)
b1 = tf.zeros([256])  # 第一层偏置向量，初始化为 0，形状为 [256]

# 第二层：输入层的特征数为 256，输出层的神经元数为 128
# 同样使用截断正态分布初始化权重，标准差为 0.1
# 第二层权重矩阵，形状为 [256, 128]
w2 = tf.random.truncated_normal([256, 128], stddev=0.1)
b2 = tf.zeros([128])  # 第二层偏置向量，初始化为 0，形状为 [128]

# 第三层：输入层的特征数为 128，输出层的神经元数为 10（对应 10 个分类）
# 使用截断正态分布初始化权重，标准差为 0.1
# 第三层权重矩阵，形状为 [128, 10]
w3 = tf.random.truncated_normal([128, 10], stddev=0.1)
b3 = tf.zeros([10])  # 第三层偏置向量，初始化为 0，形状为 [10]

# 将权重和偏置转换为可训练的变量
# tf.Variable 表示这些变量在训练过程中会被更新
w1 = tf.Variable(w1)
b1 = tf.Variable(b1)
w2 = tf.Variable(w2)
b2 = tf.Variable(b2)
w3 = tf.Variable(w3)
b3 = tf.Variable(b3)

# 设置学习率
# 学习率决定了每次参数更新的步长，值过大会导致训练不稳定，值过小会导致收敛速度慢
lr = 1e-3

# 开始训练循环
# 训练 10 个 epoch，每个 epoch 遍历一次整个训练数据集
for epoch in range(10):
    # 遍历训练数据集
    for step, (x, y) in enumerate(train_db):
        # x 的形状为 [128, 28, 28]，y 的形状为 [128]
        # 将图像数据展平为一维向量
        # [b, 28, 28] -> [b, 28*28]，即 [128, 784]
        x = tf.reshape(x, [-1, 28 * 28])

        # 使用 tf.GradientTape 记录梯度信息
        with tf.GradientTape() as tape:

            # 第一层前向传播
            # h1 = x @ w1 + b1
            # [b, 784] @ [784, 256] + [256] -> [b, 256]
            h1 = x @ w1 + tf.broadcast_to(
                b1, [x.shape[0], 256]
            )  # 广播偏置 b1 到 [b, 256]
            h1 = tf.nn.relu(h1)  # 应用 ReLU 激活函数

            # 第二层前向传播
            # h2 = h1 @ w2 + b2
            # [b, 256] @ [256, 128] + [128] -> [b, 128]
            h2 = h1 @ w2 + b2
            h2 = tf.nn.relu(h2)  # 应用 ReLU 激活函数

            # 第三层前向传播
            # out = h2 @ w3 + b3
            # [b, 128] @ [128, 10] + [10] -> [b, 10]
            out = h2 @ w3 + b3

            # 计算损失
            # 将标签 y 转换为 one-hot 编码
            # y 的形状为 [b]，转换后 y_one_hot 的形状为 [b, 10]
            y_one_hot = tf.one_hot(tf.cast(y, tf.int32), depth=10)

            # 使用均方误差 (MSE) 作为损失函数
            # 计算预测值与真实值的平方差
            loss = tf.square(y_one_hot - out)
            # 对所有样本的损失取平均值
            loss = tf.reduce_mean(loss)

        # 计算梯度
        grads = tape.gradient(loss, [w1, b1, w2, b2, w3, b3])

        # 更新权重和偏置
        # 使用梯度下降法更新每一层的权重和偏置
        # assign_sub :   -=
        # assign_sub :   +=
        # w1.assign_sub(lr * grads[0]) 表示 w1 = w1 - lr * grads[0]
        # 其中 lr 是学习率，grads[0] 是损失函数对 w1 的梯度
        w1.assign_sub(lr * grads[0])  # 更新第一层的权重
        b1.assign_sub(lr * grads[1])  # 更新第一层的偏置
        w2.assign_sub(lr * grads[2])  # 更新第二层的权重
        b2.assign_sub(lr * grads[3])  # 更新第二层的偏置
        w3.assign_sub(lr * grads[4])  # 更新第三层的权重
        b3.assign_sub(lr * grads[5])  # 更新第三层的偏置

        # 每 100 步打印一次损失值
        if step % 100 == 0:
            print("epoch: ", epoch, " step:", step, "  loss: ", float(loss))
