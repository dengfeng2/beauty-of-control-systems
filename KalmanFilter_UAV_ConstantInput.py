#######################################
# 程序名称：KalmanFilter_UAV_ConstantInput
# 程序功能：线性卡尔曼滤波器案例（6.4.2节，无人机高度预测）
# 所用模块：
#      [F8]线性卡尔曼滤波器
#######################################
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import cont2discrete
from F8_LinearKalmanFilter import F8_LinearKalmanFilter
# 设置matplotlib支持中文的字体
plt.rcParams['font.sans-serif'] = ['Songti SC']   # 'Songti SC'是一款支持中文的字体
plt.rcParams['axes.unicode_minus'] = False     # 正确显示负号

# 系统定义
A = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
n = A.shape[0]
B = np.array([[0], [1], [0]])
p = B.shape[1]
H_m = np.eye(n)
g = 10

# 系统离散化
Ts = 0.1
sys_d = cont2discrete((A, B, np.eye(n), np.zeros((n, p))), Ts)
A, B = sys_d[0], sys_d[1]

# 参数设计
Q_c = np.array([[0.01, 0, 0], [0, 0.01, 0], [0, 0, 0]])
R_c = np.eye(n)

# 系统初始化
x0 = np.array([0, 1, -10])
x = x0
u0 = np.array([g])
u = u0
x_hat0 = np.array([0, 1, -10])
x_hat = x_hat0
P0 = np.eye(n)
P = P0

# 运行步数
k_steps = 100
# 历史数据
x_history = np.zeros((n, k_steps))
u_history = np.zeros((p, k_steps))
x_hat_history = np.zeros((n, k_steps))
x_hat_minus_history = np.zeros((n, k_steps))
z_history = np.zeros((n, k_steps))

# 读取噪声数据
w = pd.read_csv('NoiseData_W.csv', index_col=None).iloc[0:3, :].values
v = pd.read_csv('NoiseData_V.csv', index_col=None).iloc[0:3, :].values

# 仿真循环
for k in range(k_steps):
    x = A.dot(x) + B.dot(u) + w[:, k]
    z = H_m.dot(x) + v[:, k]
    x_hat, x_hat_minus, P = F8_LinearKalmanFilter(A, B, Q_c, R_c, H_m, z, x_hat, P, u)

    # 保存历史数据
    x_history[:, k] = x
    z_history[:, k] = z
    x_hat_minus_history[:, k] = x_hat_minus
    x_hat_history[:, k] = x_hat

# 结果显示
plt.figure(figsize=(15, 5))
plt.plot(x_history[0, :], '--', linewidth=2, label='真实值')
plt.plot(z_history[0, :], '*', markersize=8, label='测量值')
plt.plot(x_hat_minus_history[0, :], 'o', markersize=8, label='先验估计值')
plt.plot(x_hat_history[0, :], linewidth=2, label='后验估计值')
plt.legend(loc='lower right', fontsize=20)
plt.grid(True)
plt.ylim([-2, 12])
plt.show()

# 计算均方误差
Ems = np.sum((x_hat_history[0, :] - x_history[0, :]) ** 2) / k_steps
