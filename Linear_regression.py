#######################################
# 程序名称：Linear_Regression
# 程序功能：简单线性回归案例，解析解 （2.4节案例）
#######################################
import numpy as np
import matplotlib.pyplot as plt

# 定义z向量
z = np.array([183, 175, 187, 185, 176, 176, 185, 191, 195, 185, 174, 180, 178, 170, 184])
# 定义x向量
x = np.array([75, 71, 83, 74, 73, 67, 79, 73, 88, 80, 81, 78, 73, 68, 71])
# 扩展x向量
x = np.column_stack((np.ones(len(x)), x))

# 求解最优y
y = np.linalg.inv(x.T.dot(x)).dot(x.T.dot(z))

# 结果的可视化
plt.figure(figsize=(10, 4))
# 定义横坐标
x_draw = np.arange(65, 90, 0.1)
# 散点图
plt.scatter(x[:, 1], z, 80, color="r")
# 线型图
plt.plot(x_draw, y[1]*x_draw + y[0])
plt.grid(True)
plt.show()
