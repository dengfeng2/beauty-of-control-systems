#######################################
# 程序名称：LQR_UAV_tracking_SS_U_InputConstraints
# 程序功能：无人机高度追踪控制-包含饱和函数 （4.6节案例）
# 所用模块：
#        [F1]反馈矩阵求解模块
#        [F2]稳态非零控制矩阵转化模块
#######################################
import numpy as np
from scipy.signal import cont2discrete
import matplotlib.pyplot as plt
from F2_InputAugmentMatrix_SS_U import F2_InputAugmentMatrix_SS_U
from F1_LQR_Gain import F1_LQR_Gain

# 系统参数定义
m = 1.0  # 无人机质量
g = 10.0  # 重力加速度常数

# 系统矩阵构建
A = np.array([[0, 1, 0],
              [0, 0, 1],
              [0, 0, 0]])
n = A.shape[0]  # A矩阵维度
B = np.array([[0],
              [1 / m],
              [0]])
p = B.shape[1]  # B矩阵维度

# 系统离散化
Ts = 0.1  # 离散时间步长
sys_d = cont2discrete((A, B, np.zeros((n, 1)), np.zeros((1, 1))), Ts)
A = sys_d[0]
B = sys_d[1]

# 权重矩阵设计
Q = np.array([[1, 0, 0],
              [0, 1, 0],
              [0, 0, 0]])
S = np.array([[1, 0, 0],
              [0, 1, 0],
              [0, 0, 0]])
R = np.array([[0.1]])

# 系统参考值
xd = np.array([[10],
               [0],
               [-10]])
AD = np.eye(n)

# 系统初始化
x0 = np.array([[0],
               [0],
               [-10]])
x = x0
xa = np.concatenate((x, xd))
u0 = np.array([[0.0]])
u = u0

# 系统运行步数定义
k_steps = 100
x_history = np.zeros((n, k_steps + 1))
u_history = np.zeros((p, k_steps))

# 假设F2_InputAugmentMatrix_SS_U和F1_LQR_Gain函数已实现，并返回以下示例值
Aa, Ba, Qa, Sa, R, ud = F2_InputAugmentMatrix_SS_U(A, B, Q, R, S, xd)
F = F1_LQR_Gain(Aa, Ba, Qa, R, Sa)

# 系统输入限制
u_max = 12
u_min = 7

# 仿真开始
for k in range(k_steps):
    u = -F @ xa + ud
    u = np.clip(u, u_min, u_max)  # 施加系统输入限制（基于饱和函数的硬限制）
    x = A @ x + B * u
    xa = np.concatenate((x, xd))
    x_history[:, k + 1] = x.flatten()
    u_history[:, k] = u

# 结果可视化
plt.figure(figsize=(6, 6))

# 状态变量结果图
plt.subplot(2, 1, 1)
plt.plot(x_history[0, :], label="x1")
plt.plot(x_history[1, :], '--', label="x2")
plt.grid(True)
plt.legend(loc='upper right')
plt.xlim([0, k_steps])
plt.ylim([0, 10.2])

# 系统输入结果图
plt.subplot(2, 1, 2)
plt.step(range(k_steps), u_history[0, :], where='post', label="u")
plt.legend(loc='upper right')
plt.grid(True)
plt.xlim([0, k_steps])
plt.ylim([u_min, u_max])

plt.tight_layout()  # 自动调整子图间距
plt.show()  # 显示绘图窗口
