#######################################
# 程序名称：Linear_Regression_gradient_descent
# 程序功能：简单线性回归案例，梯度下降法 （2.4节案例）
#######################################
import numpy as np
import matplotlib.pyplot as plt

# 定义z向量
z = np.array([183, 175, 187, 185, 176, 176, 185, 191, 195, 185, 174, 180, 178, 170, 184]).reshape(-1, 1)
# 定义x向量
x = np.array([75, 71, 83, 74, 73, 67, 79, 73, 88, 80, 81, 78, 73, 68, 71]).reshape(-1, 1)
# 扩展x向量
x = np.hstack((np.ones_like(x), x))
# 定义y向量
y = np.array([120, 1]).reshape(-1, 1)

# 定义学习率
alpha = np.array([[0.001, 0], [0, 0.00001]])
# 定义停止条件阈值，用于判断系统是否到达稳态
tol = 1e-4
# 初始化函数变化
diff = np.inf
# 定义迭代次数
i = 0
# 定义最大迭代次数，用于限制程序运行时间
max_iter = 100000

f_y_pre = z.T @ z - 2 * z.T @ x @ y + y.T @ x.T @ x @ y

while diff > tol:
    # 更新y
    gradient = x.T @ (x @ y - z)
    y = y - alpha.dot(gradient)
    # 计算当前f_y
    f_y = z.T.dot(z) - 2 * z.T.dot(x).dot(y) + y.T.dot(x.T).dot(x).dot(y)
    # 计算两次迭代后y的变化
    diff = np.abs(f_y - f_y_pre)
    # 保存上一次f_y
    f_y_pre = f_y
    # 更新迭代次数
    i += 1
    # 如程序超过预设最大迭代步，则报错
    if i > max_iter:
        raise ValueError('Maximum Number of Iterations Exceeded')

    # 输出进度（可选）
    if i % 1000 == 0:
        print(f'Iteration {i}, function difference: {diff[0,0]}')

# 结果的可视化
plt.figure(figsize=(10, 4))
# 定义横坐标
x_draw = np.arange(65, 90, 0.1)
# 散点图
plt.scatter(x[:, 1], z.ravel(), 80, color="r")
# 线型图
plt.plot(x_draw, y[1] * x_draw + y[0])
plt.grid(True)
plt.show()
