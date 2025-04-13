'''
Fashion MNIST 项目详解:

Step 1: 数据准备
- 使用tensorflow.keras.datasets加载Fashion MNIST数据集
- 数据集包含70000张28x28的灰度图像,分为10个类别
- 60000张训练图像,10000张测试图像
- 通过preprocess函数将像素值归一化到[0,1]区间
- 将数据打包成batch,方便训练

Step 2: 模型构建
- 使用Sequential API构建5层全连接神经网络
- 输入层:784个神经元(28*28)
- 隐藏层1:256个神经元,ReLU激活
- 隐藏层2:128个神经元,ReLU激活
- 隐藏层3:64个神经元,ReLU激活
- 隐藏层4:32个神经元,ReLU激活
- 输出层:10个神经元(对应10个类别)

Step 3: 模型训练
- 使用Adam优化器,学习率0.001
- 训练30个epoch
- 使用交叉熵损失函数
- 同时监控MSE损失
- 使用GradientTape记录梯度
- 每100步打印一次损失值

Step 4: 模型评估
- 在测试集上评估模型性能
- 计算预测准确率
- 使用softmax将输出转换为概率分布
- 选择概率最大的类别作为预测结果
'''


# Fashion Minst 时尚服饰图像分类
import tensorflow as tf
from keras import datasets,layers,models,optimizers,Sequential,metrics

def preprocess(x, y):
  # 数据预处理函数:将输入图像归一化到[0,1]区间,标签转换为整型
  x = tf.cast(x, dtype=tf.float32)/255
  y = tf.cast(y, dtype=tf.int32)
  return (x, y)

def data_predeal():
  # 加载Fashion MNIST数据集并进行预处理
  (x, y),(x_test, y_test) = datasets.fashion_mnist.load_data()
  print(x.shape, y.shape)
  print(x_test.shape, y_test.shape)

  batch_size = 128
  # 构建训练数据集
  db = tf.data.Dataset.from_tensor_slices((x, y))
  db = db.map(preprocess).shuffle(10000).batch(batch_size)

  # 构建测试数据集
  db_test = tf.data.Dataset.from_tensor_slices((x, y))
  db_test = db_test.map(preprocess).batch(batch_size)

  return db, db_test

def model_build():
  # 构建神经网络模型
  model = Sequential([
    layers.Dense(256, activation=tf.nn.relu), # 输入层到第一隐藏层 [784, 256]
    layers.Dense(128, activation=tf.nn.relu), # 第一隐藏层到第二隐藏层 [256, 128]
    layers.Dense( 64, activation=tf.nn.relu), # 第二隐藏层到第三隐藏层 [128, 64]
    layers.Dense( 32, activation=tf.nn.relu), # 第三隐藏层到第四隐藏层 [64, 32]
    layers.Dense( 10), # 输出层,10个类别 [32, 10]
  ])

  model.build(input_shape=[None, 28*28])
  model.summary()
  return model

def main(db, model):

  # 获取一个batch的样本数据
  db_iter = iter(db)
  sample = next(db_iter)
  print('batch: ', sample[0].shape, sample[1].shape)

  # 定义优化器
  optimizer = optimizers.Adam(lr=0.001)

  # 训练循环
  for epoch in range(30):
    for step, (x ,y) in enumerate(db):

      # 将输入图像展平: [b, 28, 28] -> [b, 784]
      x = tf.reshape(x, [-1, 28*28])

      with tf.GradientTape() as tape:
        # 前向传播: [b, 784] -> [b, 10]
        logits = model(x)
        # 将标签转换为one-hot编码
        y_onehot = tf.one_hot(y, depth=10)
        # 计算MSE损失
        loss_mes = tf.reduce_mean(tf.losses.MSE(y_onehot, logits))
        # 计算交叉熵损失
        loss_ce = tf.reduce_mean(tf.losses.categorical_crossentropy(y_onehot, logits, from_logits=True))

      # 计算梯度并更新模型参数
      grads = tape.gradient(loss_ce, model.trainable_variables)
      optimizer.apply_gradients(zip(grads, model.trainable_variables))

      if step % 100 == 0:
        print(epoch, step, 'loss: ', float(loss_ce), float(loss_mes))
        test(db_test, model)

def test(db_test, model):
  # 在测试集上评估模型性能
  total_correct = 0
  total_num = 0

  for x, y in  db_test:
    # 将输入图像展平
    x = tf.reshape(x, [-1, 28*28])
    # 前向传播得到预测结果
    logits = model(x)

    # 计算概率分布
    prob = tf.nn.softmax(logits, axis=1)
    # 获取最大概率对应的类别
    pred = tf.cast(tf.argmax(prob, axis=1), dtype=tf.int32)

    # 计算正确预测的样本数
    correct = tf.equal(pred, y)
    correct = tf.reduce_sum(tf.cast(correct, dtype=tf.int32))
    total_correct += int(correct)
    total_num += x.shape[0]

  # 计算准确率
  acc = total_correct / total_num
  print('acc: ', acc)


if __name__ == "__main__":
  db, db_test = data_predeal()
  model = model_build()
  main(db, model)

