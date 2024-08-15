#######################################
# 程序名称：MPC_Kalman_UAV
# 程序功能：无人机高度速度模型预测控制与卡尔曼滤波器结合示例(6.4.3节案例)
# 所用模块：
#        [F2]稳态非零控制矩阵转化模块
#        [F4]性能指标矩阵转换模块
#        [F6]约束条件矩阵转换模块
#        [F7]含约束二次规划求解模块
#        [F8]含约束二次规划求解模块
#######################################
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import cont2discrete

from F2_InputAugmentMatrix_SS_U import F2_InputAugmentMatrix_SS_U
from F4_MPC_Matrices_PM import F4_MPC_Matrices_PM
from F6_MPC_Matrices_Constraints import F6_MPC_Matrices_Constraints
from F7_MPC_Controller_withConstriants import F7_MPC_Controller_withConstriants
from F8_LinearKalmanFilter import F8_LinearKalmanFilter
# 设置matplotlib支持中文的字体
plt.rcParams['font.sans-serif'] = ['Songti SC']   # 'Songti SC'是一款支持中文的字体
plt.rcParams['axes.unicode_minus'] = False     # 正确显示负号

# 定义系统参数
m = 1.  # 无人机质量
g = 10.  # 重力加速度常数

# 系统定义
A = np.array([[0., 1., 0.], [0., 0., 1.], [0., 0., 0.]])
n = A.shape[0]
B = np.array([[0.], [1./m], [0.]])
p = B.shape[1]
H_m = np.eye(n)

# 系统离散化
Ts = 0.1
sys_d = cont2discrete((A, B, np.zeros((n, n)), np.zeros((n, p))), Ts)
A, B = sys_d[0], sys_d[1]

# 权重设计
Q = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 0.]])
S = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 0.]])
R = np.array([[0.1]])

# 系统参考值
xd = np.array([10., 0., -g])

# 卡尔曼滤波器参数设计
Q_c = np.array([[0.01, 0., 0.], [0., 0.01, 0.], [0., 0., 0.]])
R_c = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 0.]])

# 系统初始化
x0 = np.array([0., 0., -g])
x = x0
xa = np.concatenate((x, xd))
u0 = np.array([0])
u = u0
x_hat0 = np.array([0., 1., -g])
x_hat = x_hat0
P0 = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 0.]])
P = P0

# 系统约束定义
u_low = np.array([[-3.]])
u_high = np.array([[2.]])
x_low = np.array([[0.], [-4.], [-10.]])
x_high = np.array([[12.], [5.], [-10.]])
xa_low = np.vstack((x_low, (-np.inf * np.ones(3)).reshape((-1, 1))))
xa_high = np.vstack((x_low, (np.inf * np.ones(3)).reshape((-1, 1))))

# 系统运行步数
k_steps = 100
# 历史数据
x_history = np.zeros((n, k_steps))
u_history = np.zeros((p, k_steps))
x_hat_history = np.zeros((n, k_steps))
x_hat_minus_history = np.zeros((n, k_steps))
z_history = np.zeros((n, k_steps))

# 仿真环境
w = pd.read_csv('NoiseData_W.csv', index_col=None).iloc[0:3, :].values
v = pd.read_csv('NoiseData_V.csv', index_col=None).iloc[0:3, :].values

# 预测区间
N_P = 20

# 调用相关函数计算所需矩阵（此部分假设函数已提前定义）
Aa, Ba, Qa, Sa, R, ud = F2_InputAugmentMatrix_SS_U(A, B, Q, R, S, xd)
Phi, Gamma, Omega, Psi, F, H = F4_MPC_Matrices_PM(Aa, Ba, Qa, R, Sa, N_P)
M, Beta_bar, b = F6_MPC_Matrices_Constraints(xa_low, xa_high, u_low, u_high, N_P, Phi, Gamma)

# 仿真循环
for k in range(k_steps):
    print(k)
    delta_U, delta_u = F7_MPC_Controller_withConstriants(xa, F, H, M, Beta_bar, b, p)
    u = delta_u + ud
    x = A.dot(x) + B.dot(u) + w[:, k]
    z = H_m.dot(x) + v[:, k]
    x_hat, x_hat_minus, P = F8_LinearKalmanFilter(A, B, Q_c, R_c, H_m, z, x_hat, P, u)
    xa = np.concatenate((x_hat, xd))
    x_history[:, k] = x
    u_history[:, k] = u
    z_history[:, k] = z
    x_hat_minus_history[:, k] = x_hat_minus
    x_hat_history[:, k] = x_hat

# 结果显示
plt.figure(figsize=(15, 9))
plt.subplot(2, 1, 1)
plt.plot(x_history[0, :], '--', linewidth=2, label='真实值')
plt.plot(z_history[0, :], '*', markersize=8, label='测量值')
plt.plot(x_hat_minus_history[0, :], 'o', markersize=8, label='先验估计值')
plt.plot(x_hat_history[0, :], linewidth=2, label='后验估计值')
plt.grid(True)
plt.legend(loc='lower right', fontsize=20)
plt.ylim([-2, 14])

plt.subplot(2, 1, 2)
plt.step(range(k_steps), u_history[0, :], where='post', label="u")
plt.legend()
plt.grid(True)
plt.xlim([0, k_steps])

plt.show()

# 计算均方误差
Ems = np.sum((x_hat_history[0, :] - x_history[0, :]) ** 2) / k_steps