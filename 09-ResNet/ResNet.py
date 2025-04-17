# 导入所需的库
import tensorflow as tf
from   keras import datasets, layers, models, optimizers, Sequential
from   tensorflow import keras

# 设置日志级别
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class BasicBlock(tf.keras.layers.Layer):
                    # ⌐-identity downsample---↘︎
  # ResNet 基本层，->  X -> 卷积层1 -> 卷积层2 -> Add -> 激活 -> output
#
  def __init__(self, filter_num, stride = 1):
    super(BasicBlock, self).__init__()

    # 卷积层 1
    self.conv1 = layers.Conv2D(filters=filter_num, kernel_size=[3, 3], strides=stride, padding='same')
    self.bn1 = layers.BatchNormalization()
    self.relu = layers.Activation(activation='relu')

    # 卷积层 2
    self.conv2 = layers.Conv2D(filters=filter_num, kernel_size=[3,3], strides=1, padding='same')
    self.bn2 = layers.BatchNormalization()

    # identity downsample
    self.id_downsample = lambda x : x  # 直接连接 默认
    if stride != 1: # 插入 identity 层，确保输入和输出可以相加
      self.id_downsample = Sequential()
      self.id_downsample.add(layers.Conv2D(filter_num, (1, 1), stride))


  def call(self ,inputs, training = True):
    # 卷积层 1
    out = self.conv1(inputs)
    out = self.bn1(out)
    out = self.relu(out)

    # 卷积层 2
    out = self.conv2(out)
    out = self.bn2(out)

    # identity 变换
    identity = self.id_downsample(inputs)

    # fx + x
    output = layers.add([out, identity])

    # 外部激活
    output = tf.nn.relu(output)
    return output

class ResNet(tf.keras.Model):
  def __init__(self, layer_dim, class_num = 100 ):
    super(ResNet, self).__init__()

    # 1 个预处理层
    self.preprocess = Sequential([
      layers.Conv2D(64, (3,3), (1, 1), 'same'),
      layers.BatchNormalization(),
      layers.Activation('relu'),
      layers.MaxPool2D(pool_size=(2,2), strides=(1,1), padding='same')
      ])

    # res 【2,2,2,2】
    self.layer1 =  self.build_resblock(64,  layer_dim[0])
    self.layer2 =  self.build_resblock(128, layer_dim[0], 2)
    self.layer3 =  self.build_resblock(256, layer_dim[0], 2)
    self.layer4 =  self.build_resblock(512, layer_dim[0], 2)

    # 全连接层 output [b, h , w , 512]
    self.avgpool = layers.GlobalAveragePooling2D()
    self.fc = layers.Dense(class_num,
                           kernel_regularizer=tf.keras.regularizers.l2(0.01))

  def build(self, input_shape):
        # 这个方法会在模型第一次调用时自动执行
        # 在这里初始化所有层的权重
        self.preprocess.build(input_shape)

        # 计算每一层的输出形状
        x = self.preprocess.compute_output_shape(input_shape)
        self.layer1.build(x)
        x = self.layer1.compute_output_shape(x)
        self.layer2.build(x)
        x = self.layer2.compute_output_shape(x)
        self.layer3.build(x)
        x = self.layer3.compute_output_shape(x)
        self.layer4.build(x)
        x = self.layer4.compute_output_shape(x)

        # 构建全连接层
        x = self.avgpool.compute_output_shape(x)
        self.fc.build(x)

        # 标记模型为已构建
        self.built = True


  def call(self, inputs):
    x = self.preprocess(inputs)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.avgpool(x)
    out = self.fc(x)
    return out

  # 一个 res block 至少由 2个 basic_block
  def build_resblock(self, filter, blocks, stride = 1):
    resblock = Sequential()
    # may downsample
    resblock.add(BasicBlock(filter_num=filter, stride=stride))

    for _ in range(1, blocks):
      resblock.add(BasicBlock(filter, 1))

    return resblock

def build_resnet18():
  return ResNet([2,2,2,2]) # 8 * 2 + 2

def build_resnet34():
  return ResNet([3,4,6,3]) # 16 * 2 + 2

def build_resnet54():
  pass
