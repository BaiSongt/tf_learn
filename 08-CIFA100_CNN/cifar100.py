'''
网络结构:
输入层
卷积层 64通道
最大池化
卷积层 128通道
最大池化
卷积层 256通道
最大池化
卷积层 512通道
最大池化
卷积层 512通道
最大池化
'''
# 导入所需的库
import tensorflow as tf
from   keras import datasets, layers, models, optimizers, Sequential
from   tensorflow import keras

# 设置日志级别
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 设置随机种子,保证结果可复现
tf.random.set_seed(2345)

# 定义卷积网络结构,包含5个卷积单元,每个单元包含2个卷积层和1个最大池化层
conv_layers = [
# 第1个单元 - 64个通道
layers.Conv2D(64, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),  # 第一个卷积层,64个3x3卷积核
layers.Conv2D(64, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),  # 第二个卷积层,64个3x3卷积核
layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),  # 2x2最大池化,步长为2

# 第2个单元 - 128个通道
layers.Conv2D(128, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),  # 第一个卷积层,128个3x3卷积核
layers.Conv2D(128, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),  # 第二个卷积层,128个3x3卷积核
layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),  # 2x2最大池化,步长为2

# 第3个单元 - 256个通道
layers.Conv2D(256, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),  # 第一个卷积层,256个3x3卷积核
layers.Conv2D(256, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),  # 第二个卷积层,256个3x3卷积核
layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),  # 2x2最大池化,步长为2

# 第4个单元 - 512个通道
layers.Conv2D(512, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),  # 第一个卷积层,512个3x3卷积核
layers.Conv2D(512, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),  # 第二个卷积层,512个3x3卷积核
layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),  # 2x2最大池化,步长为2

# 第5个单元 - 512个通道
layers.Conv2D(512, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),  # 第一个卷积层,512个3x3卷积核
layers.Conv2D(512, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),  # 第二个卷积层,512个3x3卷积核
layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),  # 2x2最大池化,步长为2

]


def preprocess(x, y):
  # 数据预处理函数
  x = tf.cast(x, dtype=tf.float32) / 255.  # 将输入图像归一化到[0,1]范围
  y = tf.cast(y, dtype=tf.int32)  # 将标签转换为int32类型
  return x, y


# 加载CIFAR-100数据集
(x, y), (x_test, y_test) = datasets.cifar100.load_data()
y = tf.squeeze(y, axis=1)  # 压缩训练集标签的维度
y_test = tf.squeeze(y_test, axis=1)  # 压缩测试集标签的维度

print(x.shape, y.shape, x_test.shape, y_test.shape)  # 打印数据集形状

# 创建训练数据集
train_db = tf.data.Dataset.from_tensor_slices((x, y))
train_db = train_db.map(preprocess).shuffle(10000).batch(128)  # 预处理,打乱,分批

# 创建测试数据集
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_db = test_db.map(preprocess).shuffle(10000).batch(128)  # 预处理,打乱,分批



def main():
  # 创建卷积网络模型
  conv_set = Sequential(conv_layers)
  # 构建模型,指定输入形状为32x32x3的图像
  conv_set.build(input_shape=[None, 32, 32, 3])

  # 创建全连接网络
  fc_net = Sequential([
    layers.Dense(256, activation=tf.nn.relu),  # 第一个全连接层,256个神经元
    layers.Dense(128, activation=tf.nn.relu),  # 第二个全连接层,128个神经元
    layers.Dense(100, activation=None),  # 输出层,100个类别
  ])

  # 构建两个网络并打印结构
  conv_set.build(input_shape=[None, 32, 32, 3])
  conv_set.summary()
  fc_net.build(input_shape=[None, 512])
  fc_net.summary()

  # 创建Adam优化器
  optimizer = optimizers.Adam(learning_rate=1e-4)

  # 训练模型
  for epoch in range(5):  # 训练5个epoch
    for step, (x, y) in enumerate(train_db):
      with tf.GradientTape() as tape:
        # 前向传播
        out = conv_set(x)  # 通过卷积网络
        out = tf.reshape(out, [-1, 512])  # 展平特征图

        logits = fc_net(out)  # 通过全连接网络
        # 计算损失
        y_onehot = tf.one_hot(y, depth=100)  # 将标签转换为one-hot编码
        loss = tf.reduce_mean(tf.losses.categorical_crossentropy(y_onehot, logits))  # 计算交叉熵损失

        # 反向传播
        grads = tape.gradient(loss, conv_set.trainable_variables + fc_net.trainable_variables)  # 计算梯度
        optimizer.apply_gradients(zip(grads, conv_set.trainable_variables + fc_net.trainable_variables))  # 更新参数

        if step % 100 == 0:  # 每100步打印一次损失
          print(epoch, step, 'loss: ', float(loss))

  # 测试模型
  total_correct, total_num = 0, 0
  for x, y in test_db:
    out = conv_set(x)  # 前向传播
    out = tf.reshape(out, [-1, 512])
    logits = fc_net(out)
    pred = tf.argmax(logits, axis=1)  # 获取预测类别
    pred = tf.cast(pred, dtype=tf.int32)
    correct = tf.cast(tf.equal(pred, y), dtype=tf.int32)  # 计算正确预测数
    correct = tf.reduce_sum(correct)
    total_correct += int(correct)  # 累加正确预测数
    total_num += x.shape[0]  # 累加总样本数
  acc = total_correct / total_num  # 计算准确率
  print(epoch, 'acc: ', acc)

if __name__ == '__main__':
  main()

  """
  代码流程总结:
  1. 数据准备:
     - 加载CIFAR100数据集
     - 数据预处理:归一化、打包成batch

  2. 模型构建:
     - 构建卷积网络(conv_set):
       * Conv2D(64, 3) -> BatchNorm -> ReLU -> MaxPool2D
       * Conv2D(128, 3) -> BatchNorm -> ReLU -> MaxPool2D
       * Conv2D(256, 3) -> BatchNorm -> ReLU -> MaxPool2D
       * Conv2D(512, 3) -> BatchNorm -> ReLU -> MaxPool2D
       作用:提取图像特征,降维

     - 构建全连接网络(fc_net):
       * Flatten层:将特征图展平
       * Dense(256) -> ReLU
       * Dense(128) -> ReLU
       * Dense(100):输出层,对应100个类别
       作用:特征分类

  3. 训练过程:
     - 使用Adam优化器,学习率1e-4
     - 训练5个epoch
     - 每个batch:
       * 前向传播:conv_set->reshape->fc_net
       * 计算交叉熵损失
       * 反向传播更新参数
     - 每100步打印loss

  4. 测试评估:
     - 在测试集上进行预测
     - 计算分类准确率
  """
