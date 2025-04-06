#!/opt/homebrew/Caskroom/miniconda/base/envs/tf/bin/python python3
# -*- coding: utf-8 -*-

import numpy as np
import csv
import matplotlib.pyplot as plt


# 第一步：计算损失函数
# 损失函数公式：loss = 1/m * ( y - ( w*x + b ) ) ^ 2
def compute_loss_error(b, w, points):
  totalError = 0
  for i in range(0, len(points)):
    x = points[i][0]
    y = points[i][1]
    # 计算均方误差 (mean_squared_error)
    totalError += (y - (w * x + b)) ** 2
  # 返回每个点的平均损失
  return totalError / float(len(points))


# 第二步：计算梯度
def gradient_wb(b_current, w_current, points, learningRate):
  b_gradient = np.float64(0)  # b 的梯度
  w_gradient = np.float64(0)  # w 的梯度
  N = float(len(points))  # 数据点的数量
  for i in range(0, len(points)):
    x = points[i][0]
    y = points[i][1]
    # 计算 b 的梯度：grad_b = 2(w*x + b - y)
    b_gradient += (2 / N) * ((w_current * x + b_current) - y)
    # 计算 w 的梯度：grad_w = 2(wx + b - y) * x
    w_gradient += (2 / N) * ((w_current * x + b_current) - y) * x

  # 更新 b 和 w 的值
  new_b = b_current - (learningRate * b_gradient)
  new_w = w_current - (learningRate * w_gradient)
  return [new_b, new_w]


# 第三步：迭代优化
def gradient_descent_runner(
  points, starting_b, starting_w, learning_rate, num_iterations
):
  b = starting_b  # 初始化 b
  w = starting_w  # 初始化 w
  loss_history = []  # 用于记录每次迭代的损失值

  # 初始化实时绘图
  plt.ion()
  fig, ax = plt.subplots()
  ax.set_title("Loss Over Iterations")  # 图表标题
  ax.set_xlabel("Iteration")  # x 轴标签
  ax.set_ylabel("Loss")  # y 轴标签
  line, = ax.plot([], [], label="Loss")  # 绘制损失曲线
  ax.legend()

  # 进行多次迭代
  for i in range(num_iterations):
    b, w = gradient_wb(b, w, points, learning_rate)  # 更新 b 和 w
    loss = compute_loss_error(b, w, points)  # 计算当前损失
    loss_history.append(loss)  # 记录损失值

    # 更新绘图
    line.set_xdata(range(len(loss_history)))
    line.set_ydata(loss_history)
    ax.relim()
    ax.autoscale_view()
    plt.pause(0.01)

  plt.ioff()
  plt.show()

  return [b, w]


# 主函数
def runner(learning_rate, max_iterations, data_path: str):
  # 读取数据文件
  with open(data_path) as f:
    points = np.array(list(csv.reader(f))[1:], dtype=np.float64)

  # 初始化 b 和 w
  init_b = 0
  init_w = 0
  print(
    "Starting gradient descent at b = {0}, w = {1}, error = {2}".format(
      init_b, init_w, compute_loss_error(init_b, init_w, points)
    )
  )
  print("Running ... ")
  # 调用梯度下降函数
  [b, w] = gradient_descent_runner(
    points=points,
    starting_b=init_b,
    starting_w=init_w,
    learning_rate=learning_rate,
    num_iterations=max_iterations,
  )
  print(
    "After {0} iterations b = {1}, w = {2}, error = {3}".format(
      max_iterations, b, w, compute_loss_error(b, w, points)
    )
  )


if __name__ == "__main__":
  # 设置学习率、最大迭代次数和数据路径
  runner(
    learning_rate=0.000001,
    max_iterations=2000,
    data_path="01-linear_regression/data.csv",
  )
  print("Answer is : w = 2.5, b = 4.0 ")
