#######################################
# 程序名称：QP_nonEQconstraint
# 程序功能：不等式约束二次规划示例
#######################################
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# 设置matplotlib支持中文的字体
plt.rcParams['font.sans-serif'] = ['Songti SC']   # 'Songti SC'是一款支持中文的字体
plt.rcParams['axes.unicode_minus'] = False     # 正确显示负号

# 定义二次规划问题的H和f
H = np.array([[1, 0], [0, 1]])
f = np.array([1, 1])

# 定义不等式约束的A和b
A = np.array([[-1, 1], [1, 1]])
b = np.array([1, 2])

# 定义变量的边界条件
bounds = [(0, 1), (0, 2)]

# 转换为minimize函数所需的形式
def objective(u):
    return 0.5 * u.T @ H @ u + f.T @ u

def constraint1(u):
    return A[0, :] @ u - b[0]

def constraint2(u):
    return A[1, :] @ u - b[1]

constraints = ({'type': 'ineq', 'fun': constraint1},
               {'type': 'ineq', 'fun': constraint2})

# 求解二次规划问题
result = minimize(objective, np.zeros(2), method='SLSQP', bounds=bounds, constraints=constraints)
u = result.x
J = result.fun

# 绘制等高线图和可行域
plt.figure(figsize=(8,6))

# 绘制等高线图
U1, U2 = np.meshgrid(np.arange(-1, 2, 0.1), np.arange(-1, 2, 0.1))
J_grid = 0.5 * (U1**2 + U2**2) + U1 + U2
plt.contour(U1, U2, J_grid, 60)

# 绘制可行域
plt.plot([-0.5, 1], [0.5, 2], 'k', linewidth=1.5)
plt.plot([0, 1.5], [2, 0.5], 'k', linewidth=1.5)
plt.plot([0, 0], [-1, 2], 'k', linewidth=1.5)
plt.plot([-0.5, 1.5], [0, 0], 'k', linewidth=1.5)
plt.plot([1, 1], [-1, 2], 'k', linewidth=1.5)
plt.fill([0, 0.5, 1], [1, 1.5, 1], 'green', alpha=0.2, edgecolor='none')
plt.fill([0, 0, 1, 1], [0, 1, 1, 0], 'green', alpha=0.2, edgecolor='none')

# 绘制最优解点
plt.plot(u[0], u[1], 'r^', markersize=20, markerfacecolor='red')

# 添加坐标轴标签和图标题
plt.xlabel('u1')
plt.ylabel('u2')
plt.xlim([-0.5, 1.5])
plt.ylim([-0.5, 2])
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

# 显示图形
plt.show()
