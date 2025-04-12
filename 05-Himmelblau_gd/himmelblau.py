import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


# Himmelblau function
def himmelblau(x):
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2


# function plot
def plotHfunction():
    x = np.arange(-6, 6, 0.1)
    y = np.arange(-6, 6, 0.1)
    print(f"x, y range: {x.shape, y.shape}")
    X, Y = np.meshgrid(x, y)
    print(f"X, Y maps: {X.shape, Y.shape}")
    Z = himmelblau([X, Y])

    fig = plt.figure("Himmelblau")
    ax = fig.add_subplot(111, projection="3d")
    # ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z)
    ax.view_init(60, -30)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")  # 添加z轴标签
    plt.title("Himmelblau Function")
    plt.show()


def find_min_by_gd(x_range: list):
    # 初始化x范围
    x = tf.constant(x_range, dtype=tf.float32)

    for step in range(100):
        with tf.GradientTape() as tape:
            tape.watch([x])
            y = himmelblau(x)

        # 计算梯度
        grads = tape.gradient(y, [x])
        x -= 0.01 * grads[0]

        if step % 10 == 0:
            print(f"step {step}: x = {x.numpy()} y = {y.numpy()} ")


if __name__ == "__main__":
    print("*" * 16, " -4 , 0 ", "*" * 16)
    find_min_by_gd([-4, 0])
    print("=" * 16, "=======", "=" * 16)
    print("*" * 16, " 0 , 4 ", "*" * 16)
    find_min_by_gd([0, 4])
    print("=" * 16, "=======", "=" * 16)
    print("*" * 16, " -4 , 4 ", "*" * 16)
    find_min_by_gd([-4, 4])


"""
****************  -4 , 0  ****************
step 0: x = [-2.98       -0.09999999] y = 146.0
step 10: x = [-3.0984046  -0.29872793] y = 103.52823638916016
step 20: x = [-3.6890159 -3.1276689] y = 6.054703235626221
step 30: x = [-3.7793097 -3.2831852] y = 4.94765117764473e-10
step 40: x = [-3.7793102 -3.283186 ] y = 0.0
step 50: x = [-3.7793102 -3.283186 ] y = 0.0
step 60: x = [-3.7793102 -3.283186 ] y = 0.0
step 70: x = [-3.7793102 -3.283186 ] y = 0.0
step 80: x = [-3.7793102 -3.283186 ] y = 0.0
step 90: x = [-3.7793102 -3.283186 ] y = 0.0
================ ======= ================
****************  0 , 4  ****************
step 0: x = [-0.17999999  2.7       ] y = 130.0
step 10: x = [-2.4416342  3.1055622] y = 10.931074142456055
step 20: x = [-2.8051    3.131311] y = 8.577626431360841e-08
step 30: x = [-2.805118   3.1313124] y = 2.2737367544323206e-13
step 40: x = [-2.805118   3.1313124] y = 2.2737367544323206e-13
step 50: x = [-2.805118   3.1313124] y = 2.2737367544323206e-13
step 60: x = [-2.805118   3.1313124] y = 2.2737367544323206e-13
step 70: x = [-2.805118   3.1313124] y = 2.2737367544323206e-13
step 80: x = [-2.805118   3.1313124] y = 2.2737367544323206e-13
step 90: x = [-2.805118   3.1313124] y = 2.2737367544323206e-13
================ ======= ================
****************  -4 , 4  ****************
step 0: x = [-2.66  3.02] y = 106.0
step 10: x = [-2.8051126  3.1313121] y = 8.094502845779061e-09
step 20: x = [-2.805118   3.1313124] y = 2.2737367544323206e-13
step 30: x = [-2.805118   3.1313124] y = 2.2737367544323206e-13
step 40: x = [-2.805118   3.1313124] y = 2.2737367544323206e-13
step 50: x = [-2.805118   3.1313124] y = 2.2737367544323206e-13
step 60: x = [-2.805118   3.1313124] y = 2.2737367544323206e-13
step 70: x = [-2.805118   3.1313124] y = 2.2737367544323206e-13
step 80: x = [-2.805118   3.1313124] y = 2.2737367544323206e-13
step 90: x = [-2.805118   3.1313124] y = 2.2737367544323206e-13
"""
