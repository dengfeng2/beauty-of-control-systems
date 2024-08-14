#######################################
# 程序名称：QP_EQconstraint
# 程序功能：等式约束二次规划示例
#######################################
import numpy as np
import matplotlib.pyplot as plt

# 设置matplotlib支持中文的字体
plt.rcParams['font.sans-serif'] = ['Songti SC']   # 'Songti SC'是一款支持中文的字体
plt.rcParams['axes.unicode_minus'] = False     # 正确显示负号

# 定义二次规划问题的H, f, Meq
H = np.array([[1, 0],
              [0, 1]])
f = np.array([[1],
              [1]])
n = H.shape[0]

# 定义等式约束的Meq和beq
Meq = np.array([[1, -1]])
beq = np.array([[1]])
m = Meq.shape[0]

# 初始化二次规划问题的u和lamda
u = np.zeros((n, 1))
lamda = np.zeros((m, 1))

# 求解二次规划问题
KKT_matrix = np.block([[H, Meq.T],
                       [Meq, np.zeros((m, m))]])
rhs = np.vstack((-f, beq))
u_lamda = np.linalg.inv(KKT_matrix) @ rhs
u = u_lamda[:n]

# 创建一个图窗口并定义大小
plt.figure(1, figsize=(15, 5))

# 绘制二次规划问题的可行域和最优解点（3D图）
U1, U2 = np.meshgrid(np.arange(-2, 0.1, 0.1), np.arange(-2, 0.1, 0.1))
J = 0.5 * (U1**2 + U2**2) + U1 + U2

# 子图1: 3D曲面图
ax1 = plt.subplot(1, 2, 1, projection='3d')
ax1.plot_surface(U1, U2, J, alpha=0.1)
u1_proj = np.arange(-2, 0.1, 0.1)
u2_proj = u1_proj - 1
J_proj = 0.5 * (u1_proj ** 2 + u2_proj ** 2) + u1_proj + u2_proj
ax1.plot(u1_proj, u2_proj, J_proj, 'b', linewidth=5)
ax1.scatter(u[0], u[1], 0.5 * (u[0]**2 + u[1]**2) + u[0] + u[1], color='red', s=200, marker='^')
U1_proj, J_proj_mesh = np.meshgrid(u1_proj, np.arange(-1.05, 0.1, 0.1))
U2_proj = U1_proj - 1
ax1.plot_surface(U1_proj, U2_proj, J_proj_mesh, color='blue', alpha=0.2, edgecolor='none')
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
u1_con = np.arange(-1.0, 0.1, 0.1)
u2_con = u1_con - 1
ax2.plot(u1_con, u2_con, 'k', linewidth=2)
ax2.set_xlabel('u1')
ax2.set_ylabel('u2')
ax2.tick_params(labelsize=20)

# 设置整体标题
plt.suptitle('二次规划问题', fontsize=20)

# 显示图形
plt.show()