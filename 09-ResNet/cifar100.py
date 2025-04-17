# 导入所需的库
import tensorflow as tf
from   tensorflow import keras
from   keras import datasets, layers, models, optimizers, Sequential
from ResNet import build_resnet18

# 设置日志级别
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 设置随机种子,保证结果可复现
tf.random.set_seed(2345)

def preprocess(x, y):
  # 数据预处理函数
  # x = tf.cast(x, dtype=tf.float32) / 255.  # 将输入图像归一化到[0,1]范围
  x = 2 * tf.cast(x, dtype=tf.float32) / 255. - 1  # 将输入图像归一化到[0,1]范围
  y = tf.cast(y, dtype=tf.int32)  # 将标签转换为int32类型
  return x, y


# 加载CIFAR-100数据集
(x, y), (x_test, y_test) = datasets.cifar100.load_data()
y = tf.squeeze(y, axis=1)  # 压缩训练集标签的维度
y_test = tf.squeeze(y_test, axis=1)  # 压缩测试集标签的维度

print(x.shape, y.shape, x_test.shape, y_test.shape)  # 打印数据集形状

# 创建训练数据集
train_db = tf.data.Dataset.from_tensor_slices((x, y))
train_db = train_db.map(preprocess).shuffle(10000).batch(128).repeat(10)  # 预处理,打乱,分批

# 创建测试数据集
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_db = test_db.map(preprocess).shuffle(10000).batch(128)  # 预处理,打乱,分批


def main():

  resnet18_model = build_resnet18()

  # [b, 32, 32, 3] -> [b, 1 ,1 ,512]
  # 构建模型,指定输入形状为32x32x3的图像
  resnet18_model.build(input_shape=[None, 32, 32, 3])
  resnet18_model.summary()

  # # 创建Adam优化器
  # optimizer = optimizers.Adam(learning_rate=1e-3)

  # # 训练模型
  # for epoch in range(5):  # 训练5个epoch
  #   for step, (x, y) in enumerate(train_db):
  #     with tf.GradientTape() as tape:
  #       # 前向传播
  #       logits = resnet18_model(x)

  #       # 计算损失
  #       y_onehot = tf.one_hot(y, depth=100)  # 将标签转换为one-hot编码
  #       loss = tf.reduce_mean(tf.losses.categorical_crossentropy(y_onehot, logits))  # 计算交叉熵损失

  #       # 反向传播
  #       grads = tape.gradient(loss, resnet18_model.trainable_variables)  # 计算梯度
  #       optimizer.apply_gradients(zip(grads, resnet18_model.trainable_variables))  # 更新参数

  #       if step % 10 == 0:  # 每10步打印一次损失
  #         print(epoch, step, 'loss: ', float(loss))

  # # 测试模型
  # total_correct, total_num = 0, 0
  # for x, y in test_db:
  #   logits = resnet18_model(x)
  #   pred = tf.argmax(logits, axis=1)  # 获取预测类别
  #   pred = tf.cast(pred, dtype=tf.int32)
  #   correct = tf.cast(tf.equal(pred, y), dtype=tf.int32)  # 计算正确预测数
  #   correct = tf.reduce_sum(correct)
  #   total_correct += int(correct)  # 累加正确预测数
  #   total_num += x.shape[0]  # 累加总样本数
  # acc = total_correct / total_num  # 计算准确率
  # print(epoch, 'acc: ', acc)

if __name__ == '__main__':
  main()
