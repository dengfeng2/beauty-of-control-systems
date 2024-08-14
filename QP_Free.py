#######################################
# 程序名称：QP_Free
# 程序功能：无约束二次规划示例
#######################################
import numpy as np
import matplotlib.pyplot as plt

# 设置matplotlib支持中文的字体
plt.rcParams['font.sans-serif'] = ['Songti SC']   # 'Songti SC'是一款支持中文的字体
plt.rcParams['axes.unicode_minus'] = False     # 正确显示负号

# 定义二次规划问题的H和f
H = np.array([[1, 0],
              [0, 1]])
f = np.array([1, 1])

# 求解二次规划问题
u = -np.linalg.inv(H) @ f

# 创建一个图窗口并定义大小
plt.figure(1, figsize=(15, 5))

# 绘制二次规划问题的可行域和最优解点（3D图）
U1, U2 = np.meshgrid(np.arange(-2, 0.1, 0.1), np.arange(-2, 0.1, 0.1))
J = 0.5 * (U1**2 + U2**2) + U1 + U2

# 子图1: 3D曲面图
ax1 = plt.subplot(1, 2, 1, projection='3d')
ax1.plot_surface(U1, U2, J, alpha=0.1)
ax1.scatter(u[0], u[1], 0.5 * (u[0]**2 + u[1]**2) + u[0] + u[1], color='red', s=200, marker='^')
ax1.set_xlabel('u1')
ax1.set_ylabel('u2')
ax1.set_zlabel('J(u1,u2)')
ax1.set_xlim([-2, 0])
ax1.set_ylim([-2, 0])
ax1.set_zlim([-1.05, 0])
ax1.tick_params(labelsize=20)

# 子图2: 等高线图
ax2 = plt.subplot(1, 2, 2)
ax2.contour(U1, U2, J, 30)
ax2.plot(u[0], u[1], 'r*', markersize=10)
ax2.set_xlabel('u1')
ax2.set_ylabel('u2')
ax2.tick_params(labelsize=20)

# 设置整体标题
plt.suptitle('二次规划问题', fontsize=20)

# 显示图形
plt.show()
