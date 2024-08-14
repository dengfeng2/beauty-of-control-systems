import numpy as np
from scipy.signal import cont2discrete
import matplotlib.pyplot as plt
from F2_InputAugmentMatrix_SS_U import F2_InputAugmentMatrix_SS_U
from F4_MPC_Matrices_PM import F4_MPC_Matrices_PM
from F6_MPC_Matrices_Constraints import F6_MPC_Matrices_Constraints
from F7_MPC_Controller_withConstriants import F7_MPC_Controller_withConstriants

# 定义系统参数
# 定义无人机质量
m = 1
# 定义重力加速度常数
g = 10

# 系统定义
# 构建系统矩阵A，n x n
A = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
# 计算A矩阵维度
n = A.shape[0]
# 构建输入矩阵B，n x p
B = np.array([[0], [1 / m], [0]])
# 计算输入矩阵维度
p = B.shape[1]

# 系统离散
# 离散时间步长
Ts = 0.1
# 连续系统转离散系统
sys_d = cont2discrete((A, B, np.zeros((n, 1)), np.zeros((1, 1))), Ts)
# 提取离散系统A矩阵
A = sys_d[0]
# 提取离散系统B矩阵
B = sys_d[1]

# 权重设计
# 设计状态权重系数矩阵, n x n
Q = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
# 设计终值权重系数矩阵, n x n
S = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
# 设计输入权重系数矩阵, p x p
R = 0.1

# 系统参考值
# 系统状态参考值
xd = np.array([[10], [0], [-g]])
# 构建目标转移矩阵
AD = np.eye(n)

# 系统初始化
# 初始化系统状态
x0 = np.array([[0], [0], [-g]])
x = x0
# 初始化增广状态矩阵
xa = np.vstack((x, xd))
# 初始化系统输入
u0 = np.array([[0]])
u = u0

# 系统约束定义
# 输入下限
u_low = np.array([[-3]])
# 输入上限
u_high = np.array([[2]])
# 状态下限
x_low = np.array([[0], [0], [-g]])
# 状态上限
x_high = np.array([[10], [3], [-g]])
# 增广状态下限
xa_low = np.vstack((x_low, -np.inf * np.ones((n, 1))))
# 增广状态上限
xa_high = np.vstack((x_high, np.inf * np.ones((n, 1))))

# 定义系统运行步数
k_steps = 100
# 定义x_history零矩阵，用于储存系统状态结果，维度n x k_step
x_history = np.zeros((n, k_steps + 1))
# 定义u_history零矩阵，用于储存系统输入结果，维度p x k_step
u_history = np.zeros((p, k_steps))

# 定义预测区间
N_P = 20

Aa, Ba, Qa, Sa, R, ud = F2_InputAugmentMatrix_SS_U(A, B, Q, R, S, xd)

Phi, Gamma, Omega, Psi, F, H = F4_MPC_Matrices_PM(Aa, Ba, Qa, R, Sa, N_P)

M, Beta_bar, b = F6_MPC_Matrices_Constraints(xa_low, xa_high, u_low, u_high, N_P, Phi, Gamma)

# for循环开始仿真
for k in range(k_steps):
    delta_U, delta_u = F7_MPC_Controller_withConstriants(xa, F, H, M, Beta_bar, b, p)

    # 更新状态和输入（依赖于前面的调用）
    u = delta_u + ud
    x = A @ x + B @ u

    # 更新增广矩阵xa
    xa = np.vstack((x, xd))

    # 保存系统状态和输入（仅供示例，需要使用真正的值替换）
    x_history[:, k + 1] = x.flatten()
    u_history[:, k] = u.flatten()

# 结果绘制 图1
plt.figure(figsize=(10, 10))
# 状态变量结果图
plt.subplot(2, 1, 1)
plt.plot(x_history[0, :], label="x1")
plt.plot(x_history[1, :], '--', label="x2")
plt.grid(True)
plt.legend()
plt.xlim([0, k_steps])
plt.ylim([0, 10.2])

# 系统输入
plt.subplot(2, 1, 2)
plt.step(range(k_steps), u_history[0, :], where='post', label="u")
plt.legend()
plt.grid(True)
plt.xlim([0, k_steps])
plt.ylim([-3, 3])  # 假定输入范围在[-3, 3]之间

plt.show()
