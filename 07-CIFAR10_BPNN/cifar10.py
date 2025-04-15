import tensorflow as tf
from   keras import datasets, layers, models, optimizers
from   tensorflow import keras

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 加载数据
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
print(f"x_train shape: {x_train.shape}", f" y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape}", f" y_test shape: {y_test.shape}")
# x_train shape: (50000, 32, 32, 3)  y_train shape: (50000, 1)
# x_test shape: (10000, 32, 32, 3)   y_test shape: (10000, 1)
y_train = tf.squeeze(y_train)
y_test = tf.squeeze(y_test)
# x_train shape: (50000, 32, 32, 3)  y_train shape: (50000,)
# x_test shape: (10000, 32, 32, 3)   y_test shape: (10000,)
y_train = tf.one_hot(y_train, depth=10)
y_test = tf.one_hot(y_test, depth=10)
print('Datasets: ',  x_train.shape, y_train.shape, x_test.shape, y_test.shape, x_train.min(), x_train.max())

# 数据预处理
# 将图像数据归一化到[-1, 1]之间
def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255.0 - 1.0
    y = tf.cast(y, dtype=tf.int32)
    return x, y

# 创建训练数据集
batch_size = 128
db = tf.data.Dataset.from_tensor_slices((x_train, y_train))
db = db.map(preprocess).shuffle(10000).batch(batch_size)

# 创建测试数据集
db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db_test = db_test.map(preprocess).batch(batch_size)


# 自定义层
class MyDense(layers.Layer):
  '''
  自定义层 to_replace standard layers.Dense()
  '''
  def __init__(self, inout_shape) -> None:
    super(MyDense, self).__init__()
    self.inp_dim, self.outp_dim = inout_shape
    self.kernel = self.add_weight(name='w', shape=[self.inp_dim, self.outp_dim])
    self.bias = self.add_weight(name='b', shape=[self.outp_dim])

  def call(self, inputs, training=None):
    out = inputs @ self.kernel + self.bias
    return out


# 自定义模型
class MyNetWork(models.Model):
    '''
    自定义神经网络模型
    包含5个全连接层，用于CIFAR10图像分类
    网络结构：
    输入层: 32*32*3 -> 256
    第一层: 256 -> 128
    第二层: 128 -> 64
    第三层: 64 -> 32
    输出层: 32 -> 10
    '''
    def __init__(self) -> None:
        super(MyNetWork, self).__init__()
        self.fc1 = MyDense(inout_shape=[32*32*3, 256])
        self.fc2 = MyDense(inout_shape=[256, 128])
        self.fc3 = MyDense(inout_shape=[128, 64])
        self.fc4 = MyDense(inout_shape=[64, 32])
        self.fc5 = MyDense(inout_shape=[32, 10])

    def call(self, inputs, training=None):
        # 将输入图像展平为一维向量
        x = tf.reshape(inputs, [-1, 32*32*3])
        # 通过全连接层和ReLU激活函数
        x = tf.nn.relu(self.fc1(x))
        x = tf.nn.relu(self.fc2(x))
        x = tf.nn.relu(self.fc3(x))
        x = tf.nn.relu(self.fc4(x))
        # 最后一层不使用激活函数，直接输出logits
        x = self.fc5(x)
        return x

# 创建优化器
opt = optimizers.Adam(learning_rate=1e-3)

# 创建模型
network = MyNetWork()
# 编译模型
# optimizer: 使用前面定义的Adam优化器,学习率为1e-3
# loss: 使用分类交叉熵损失函数
#   - from_logits=True 表示模型的输出是原始logits,而不是经过softmax的概率分布
#   - 损失函数会在内部进行softmax转换
# metrics: 评估指标为准确率(accuracy)
#   - 用于在训练过程中监控模型性能
network.compile(optimizer=opt,
                loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

# 训练模型
# 训练模型
# fit()方法参数说明:
# - db: 训练数据集,包含训练样本和标签
# - epochs: 训练轮数,这里设置为5轮
# - validation_data: 验证数据集,用于评估模型性能,这里使用db_test测试集
# - validation_freq: 验证频率,每2个epoch验证一次模型性能
#   例如:epoch=1时不验证,epoch=2时验证,epoch=3时不验证,epoch=4时验证,epoch=5时验证
network.fit(db, epochs=5, validation_data=db_test, validation_freq=1)
print(network.summary())

# 创建保存模型的目录
import os
os.makedirs('models', exist_ok=True)

# 保存权重模型
print("保存权重模型", "=" * 50)
network.save_weights('models/cifar10.weights.h5')
print("权重模型已保存")

# 保存完整模型
print("保存完整模型", "=" * 50)
keras.saving.save_model(network, 'my_model.keras')
# network.save('models/cifar10_model.h5')
print("完整模型已保存")

# 测试保存的模型
print("测试保存的模型", "=" * 50)
test_loss, test_acc = network.evaluate(db_test)
print(f"原始模型测试准确率: {test_acc:.4f}")

# 加载权重模型
print("加载权重模型", "=" * 50)
weights_model = MyNetWork()
weights_model.build(input_shape=(None, 32, 32, 3))
weights_model.compile(optimizer=opt,
                     loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                     metrics=['accuracy'])
weights_model.load_weights('models/cifar10.weights.h5')
weights_loss, weights_acc = weights_model.evaluate(db_test)
print(f"权重模型测试准确率: {weights_acc:.4f}")

# 加载完整模型
print("加载完整模型", "=" * 50)
full_model = keras.models.load_model('models/my_model.keras')
full_loss, full_acc = full_model.evaluate(db_test)
print(f"完整模型测试准确率: {full_acc:.4f}")
print("=" * 80)
