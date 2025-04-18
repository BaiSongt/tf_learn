# MNIST分类：理论与解释

## 简介
MNIST数据集是机器学习和计算机视觉中的一个基准数据集。它包含70,000张手写数字（0-9）的灰度图像，每张图像的大小为28x28像素。数据集分为60,000张训练图像和10,000张测试图像。

## 问题定义
MNIST分类的目标是构建一个模型，能够准确地将输入图像分类为10个数字类别（0-9）之一。这是一个监督学习问题，其中输入是图像，输出是对应的数字标签。

## 关键概念

### 1. 神经网络
神经网络通常用于MNIST分类。一个典型的架构包括：
- **输入层**：28x28 = 784个神经元，用于像素值。
- **隐藏层**：全连接层或卷积层，用于特征提取。
- **输出层**：10个神经元（每个数字类别一个），使用softmax激活函数。

### 1. 神经网络

数学公式：
1. **前向传播**：
  对于第 $l$ 层的神经元：
  $$
  z^{(l)} = W^{(l)}a^{(l-1)} + b^{(l)}
  $$
  $$
  a^{(l)} = \sigma(z^{(l)})
  $$
  其中，$W^{(l)}$ 是权重矩阵，$b^{(l)}$ 是偏置向量，$\sigma$ 是激活函数。

2. **Softmax激活函数**：
  输出层的激活函数为：
  $$
  a_j = \frac{e^{z_j}}{\sum_{k=1}^{10} e^{z_k}}
  $$
  其中，$z_j$ 是第 $j$ 个输出神经元的输入。

### 2. 损失函数

交叉熵损失用于衡量预测概率与真实标签之间的差异。

数学公式：
$$
L = -\frac{1}{m} \sum_{i=1}^m \sum_{j=1}^{10} y_{ij} \log(a_{ij})
$$
其中，$m$ 是样本数量，$y_{ij}$ 是样本 $i$ 的真实标签（one-hot编码），$a_{ij}$ 是模型预测的概率。

### 3. 优化

梯度下降及其变体（如SGD、Adam）用于最小化损失函数并更新模型权重。

数学公式：
1. **梯度下降**：
  $$
  W^{(l)} := W^{(l)} - \eta \frac{\partial L}{\partial W^{(l)}}
  $$
  $$
  b^{(l)} := b^{(l)} - \eta \frac{\partial L}{\partial b^{(l)}}
  $$
  其中，$\eta$ 是学习率。

2. **Adam优化器**：
  $$
  m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
  $$
  $$
  v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
  $$
  $$
  \hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
  $$
  $$
  \theta_t = \theta_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
  $$
  其中，$g_t$ 是梯度，$\beta_1, \beta_2$ 是动量参数。

### 4. 评估指标

- **准确率**：
  $$
  \text{Accuracy} = \frac{\text{正确分类的样本数}}{\text{总样本数}}
  $$

- **混淆矩阵**：
  混淆矩阵是一个 $10 \times 10$ 的矩阵，其中第 $i$ 行第 $j$ 列表示真实类别为 $i$ 且预测类别为 $j$ 的样本数量。

### 5. 数据增强

应用变换（如旋转、缩放）以增加训练数据的多样性。

数学公式：
1. **旋转**：
  $$
  \begin{bmatrix}
  x' \\
  y'
  \end{bmatrix}
  =
  \begin{bmatrix}
  \cos\theta & -\sin\theta \\
  \sin\theta & \cos\theta
  \end{bmatrix}
  \begin{bmatrix}
  x \\
  y
  \end{bmatrix}
  $$
  其中，$\theta$ 是旋转角度。

2. **缩放**：
  $$
  \begin{bmatrix}
  x' \\
  y'
  \end{bmatrix}
  =
  \begin{bmatrix}
  s_x & 0 \\
  0 & s_y
  \end{bmatrix}
  \begin{bmatrix}
  x \\
  y
  \end{bmatrix}
  $$
  其中，$s_x, s_y$ 是缩放因子。

### 4. 数据增强
应用变换（如旋转、缩放）以增加训练数据的多样性。

## 结论
MNIST分类是机器学习中的基础任务，为实验各种模型和技术提供了平台。掌握MNIST分类为更复杂的图像分类挑战奠定了基础。
