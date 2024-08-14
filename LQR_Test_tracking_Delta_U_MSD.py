#######################################
# 程序名称：LQR_Test_tracking_Delta_U_MSD
# 程序功能：弹簧质量阻尼系统 - 输入增量控制 （4.5.4节）
# 所用模块：
#        [F1]反馈矩阵求解模块
#        [F3]输入增量控制矩阵转换模块
#######################################
import numpy as np
from scipy.signal import cont2discrete
import matplotlib.pyplot as plt
from F3_InputAugmentMatrix_Delta_U import F3_InputAugmentMatrix_Delta_U
from F1_LQR_Gain import F1_LQR_Gain

# 定义系统参数
m_sys = 1  # 质量块质量
b_sys = 0.5  # 阻尼系数
k_sys = 1  # 弹簧弹性系数

# 系统定义
A = np.array([[0, 1], [-k_sys / m_sys, -b_sys / m_sys]])
n = A.shape[0]  # 计算A矩阵维度
B = np.array([[0], [1 / m_sys]])
p = B.shape[1]  # 计算输入矩阵维度

# 系统离散化
Ts = 0.1  # 离散时间步长
sys_d = cont2discrete((A, B, [[0]], [[0]]), Ts)
A = sys_d[0]
B = sys_d[1]

# 权重设计
Q = np.array([[1, 0], [0, 1]])
S = np.array([[1, 0], [0, 1]])
R = np.array([[0.1]])

# 系统参考值
xd = np.array([[1], [0]])
AD = np.eye(n)  # 构建目标转移矩阵

# 系统初始化
x0 = np.array([[0], [0]])  # 初始化系统状态
x = x0
u0 = np.array([[0]])  # 系统输入初始化
u = u0
xa = np.concatenate((x, xd, u))  # 构建初始化增广状态矩阵

# 定义系统运行步数
k_steps = 100
# 初始化x_history和u_history
x_history = np.zeros((n, k_steps + 1))
u_history = np.zeros((p, k_steps))

# 假设F3_InputAugmentMatrix_Delta_U和F1_LQR_Gain函数已经定义
[Aa, Ba, Qa, Sa, R] = F3_InputAugmentMatrix_Delta_U(A,B,Q,R,S,AD)
F = F1_LQR_Gain(Aa, Ba, Qa, R, Sa)

# 仿真开始
for k in range(k_steps):
    Delta_u = -F @ xa
    u = Delta_u + u
    x = A @ x + B * u
    xd = AD @ xd
    xa = np.concatenate((x, xd, u))
    x_history[:, k+1] = x.flatten()  # 将x保存到历史记录
    u_history[:, k] = u  # 将u保存到历史记录

# 结果可视化
plt.figure(figsize=(10, 10))
# 状态变量结果图
plt.subplot(2, 1, 1)
plt.plot(x_history[0, :], label="x1")
plt.plot(x_history[1, :], '--', label="x2")
plt.grid(True)
plt.legend()
plt.xlim([0, k_steps])
plt.ylim([-0.2, 1.2])

# 系统输入结果图
plt.subplot(2, 1, 2)
plt.step(range(k_steps), u_history[0, :], label="u")
plt.legend()
plt.grid(True)
plt.xlim([0, k_steps])
plt.ylim([0, 3])

plt.show()
