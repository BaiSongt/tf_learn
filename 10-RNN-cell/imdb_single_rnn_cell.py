import os
import tensorflow as tf
import numpy as np
from tensorflow import keras

# IMDB数据集说明:
# 1. 数据集简介
#    IMDB是一个电影评论数据集,包含来自互联网电影数据库(IMDB)的50000条电影评论
#    其中25000条用于训练,25000条用于测试
#    每条评论都被标记为正面(1)或负面(0)评价,是一个二分类任务

# 2. 数据预处理
#    - 只保留最常见的10000个单词,降低模型复杂度
#    - 将每条评论截断或填充到固定长度80个单词
#    - 评论被转换为单词索引序列,每个单词用一个整数表示

# 3. 数据格式
#    x_train/x_test: 形状为[num_samples, 80]的二维数组
#                    每行代表一条评论,包含80个单词的索引
#    y_train/y_test: 形状为[num_samples]的一维数组
#                    包含0(负面)或1(正面)的标签

# 4. 数据集特点
#    - 平衡的数据集:正面和负面评论数量相等
#    - 序列数据:评论是单词序列,适合用RNN等序列模型处理
#    - 实际应用:代表了真实世界的情感分析任务


tf.random.set_seed(22)
np.random.seed(22)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')

batch_size = 128
epochs = 4
embedding_dim = 100

# 加载IMDB数据集,只保留最常见的10000个单词
# num_words=10000表示我们只保留训练数据中最频繁出现的10000个单词
# 这样可以减少词汇表大小,降低模型复杂度
total_words = 1000
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=total_words)

# 设置评论的最大长度为80个单词
# 如果评论长度超过80,则截断
# 如果评论长度不足80,则在序列前面补0
# 这样可以让所有评论序列等长,便于批量训练
max_review_length = 80
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_review_length)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_review_length)

# x_train: [b, 80, 1]  y_train: [b]
# x_test: [b, 80, 1]   y_test: [b]

# 创建数据集
# drop_remainder=True 表示丢弃最后一个不完整的batch
# 这样可以保证每个batch的大小都是相同的
# 虽然会丢失一些数据,但可以避免因batch大小不一致导致的训练问题
db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
db_train = db_train.shuffle(1000).batch(batch_size, drop_remainder=True)
db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db_test = db_test.batch(batch_size, drop_remainder=True)

print('x_train.shape:', x_train.shape, 'y_train.shape:', y_train.shape)
print('x_test.shape:', x_test.shape, 'y_test.shape:', y_test.shape)
print('y max and min:', y_train.max(), y_train.min(), '好评 和 差评， 二分类任务')


# max_review_length = 80 表示每条评论的最大长度为80个单词
# 在前面的数据预处理中,我们使用pad_sequences将所有评论长度统一为80:
# - 如果评论长度超过80,则截断保留前80个单词
# - 如果评论长度不足80,则在序列前面补0
# 这样做的目的是:
# 1. 让所有评论序列等长,便于批量训练
# 2. 控制序列长度,避免序列过长导致计算开销过大
# 3. 保留评论的主要信息,因为通常评论的关键信息都在前面部分

# 因此在MyRNN类的注释中:
# [b, 80] 表示输入shape为[batch_size, max_review_length]
# [b, 80, 100] 表示经过词嵌入后的shape为[batch_size, max_review_length, embedding_dim]
# 其中80就是我们预设的max_review_length

class MyRNN(keras.Model):
    '''
    实现单个RNN单元的情感分类模型

    参数:
        units: RNN隐藏状态维度,默认64维
    '''
    def __init__(self, units):
        super(MyRNN, self).__init__()

        # 1. 词嵌入层
        # 将单词索引映射为稠密向量表示
        # 输入: [batch_size, max_review_length] -> [b, 80]
        # 输出: [batch_size, max_review_length, embedding_dim] -> [b, 80, 100]
        self.embedding = keras.layers.Embedding(
            input_dim=total_words,  # 词汇表大小
            output_dim=embedding_dim,  # 词嵌入维度
        )

        # 2. RNN层
        # 处理序列数据,提取时序特征
        # 输入: [batch_size, timesteps, embedding_dim] -> [b, 80, 100]
        # 输出: [batch_size, units] -> [b, 64]
        self.rnn = keras.layers.SimpleRNN(
            units=units,  # RNN单元维度
            dropout=0.2,  # Dropout正则化
            return_sequences=False  # 只返回最后一个时间步的输出
        )

        # 3. 全连接输出层
        # 将RNN输出映射为二分类概率
        # 输入: [batch_size, units] -> [b, 64]
        # 输出: [batch_size, 1] -> [b, 1]
        self.out_layer = keras.layers.Dense(units=1)

        self.built = False

    def build(self, input_shape):
        """构建模型各层"""
        # 1. 构建词嵌入层
        self.embedding.build(input_shape)
        x_shape = self.embedding.compute_output_shape(input_shape)

        # 2. 构建RNN层
        self.rnn.build(x_shape)

        # 3. 构建输出层
        self.out_layer.build([x_shape[0], self.rnn.units])

        self.built = True
        super(MyRNN, self).build(input_shape)

    def call(self, inputs, training=None):
        """
        前向传播过程

        参数:
            inputs: 输入数据 shape=[batch_size, max_review_length]
            training: 是否为训练模式
        返回:
            prob: 预测概率 shape=[batch_size, 1]
        """
        # 1. 词嵌入
        x = self.embedding(inputs)  # [b, 80] -> [b, 80, 100]

        # 2. 处理输入维度
        # 检查输入张量的维度是否为2维
        # 如果是2维 [batch_size, features]
        # 则在中间插入时间步维度,扩展为3维 [batch_size, timesteps=1, features]
        # RNN层要求输入必须是3维张量 [batch_size, timesteps, features]
        if len(x.shape) == 2:
            x = tf.expand_dims(x, axis=1)  # 在axis=1处插入维度

        # 3. RNN处理
        out = self.rnn(x, training=training)  # [b, 80, 100] -> [b, 64]

        # 4. 全连接层分类
        logits = self.out_layer(out)  # [b, 64] -> [b, 1]
        prob = tf.sigmoid(logits)  # 二分类激活函数

        return prob

def main():
  units = 64
  epochs = 4
  model = MyRNN(units)
  model.compile(optimizer=keras.optimizers.Adam(0.001), # 优化器
                loss=keras.losses.BinaryCrossentropy(), # 二分类任务
                metrics=['accuracy']) # 准确率
  model.build(input_shape=(batch_size, max_review_length))
  model.summary()
  model.fit(db_train, epochs=epochs, validation_data = db_test)

if __name__ == '__main__':
  main()
