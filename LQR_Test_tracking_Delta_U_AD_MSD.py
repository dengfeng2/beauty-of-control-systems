#######################################
# 程序名称：LQR_Test_tracking_Delta_U_AD_MSD
# 程序功能：弹簧质量阻尼系统 -输入增量非常数目标（4.5.4节案例）
# 所用模块：
#        [F1]反馈矩阵求解模块
#        [F3]输入增量控制矩阵转换模块
#######################################
import numpy as np
from scipy.signal import cont2discrete
import matplotlib.pyplot as plt
from F3_InputAugmentMatrix_Delta_U import F3_InputAugmentMatrix_Delta_U
from F1_LQR_Gain import F1_LQR_Gain

# 系统参数定义
m_sys = 1.0  # 质量块质量
b_sys = 0.5  # 阻尼系数
k_sys = 1.0  # 弹簧弹性系数

# 系统矩阵构建
A = np.array([[0, 1], [-k_sys / m_sys, -b_sys / m_sys]])
n = A.shape[0]  # A矩阵维度
B = np.array([[0], [1 / m_sys]])
p = B.shape[1]  # B矩阵维度

# 系统离散化
Ts = 0.1  # 离散时间步长
sys_d = cont2discrete((A, B, np.zeros((n, 1)), np.zeros((1, 1))), Ts, method='zoh')
A = sys_d[0]
B = sys_d[1]

# 权重矩阵设计
Q = np.array([[1, 0], [0, 1]])
S = np.array([[1, 0], [0, 1]])
R = np.array([[1]])

# 系统参考值
xd = np.array([[0], [0.2]])

# 目标转移矩阵
AD = cont2discrete((np.array([[0, 1], [0, 0]]), np.array([[0], [0]]), np.zeros((2, 1)), np.zeros((1, 1))), Ts, method='zoh')[0]

# 系统初始化
x0 = np.array([[0], [0]])
x = x0
u0 = np.array([[0]])
u = u0
xa = np.concatenate((x, xd, u))  # 增广状态矩阵初始化

# 系统运行步数定义
k_steps = 200
x_history = np.zeros((n, k_steps))
u_history = np.zeros((p, k_steps))
xd_history = np.zeros((n, k_steps))

# 假设F3_InputAugmentMatrix_Delta_U和F1_LQR_Gain函数已实现
[Aa, Ba, Qa, Sa, R] = F3_InputAugmentMatrix_Delta_U(A, B, Q, R, S, AD)
F = F1_LQR_Gain(Aa, Ba, Qa, R, Sa)

# 仿真开始
for k in range(1, k_steps + 1):
    if k == 50:
        xd = np.array([[xd[0, 0]], [-0.2]])
    elif k == 100:
        xd = np.array([[xd[0, 0]], [0.2]])
    elif k == 150:
        xd = np.array([[xd[0, 0]], [-0.2]])
    elif k == 200:
        xd = np.array([[xd[0, 0]], [0.2]])

    # 输入增量计算
    Delta_u = -F @ xa
    u = Delta_u + u

    # 系统响应计算
    x = A @ x + B * u

    # 更新增广状态
    xd = AD @ xd
    xa = np.concatenate((x, xd, u))

    # 保存历史数据
    x_history[:, k - 1] = x.flatten()
    u_history[:, k - 1] = u
    xd_history[:, k - 1] = xd.flatten()

# 结果可视化
plt.figure(figsize=(6, 10))

# 系统状态x1结果图
plt.subplot(3, 1, 1)
plt.plot(x_history[0, :], 'b', label="x1", linewidth=1)
plt.plot(xd_history[0, :], 'r--', label="x1d", linewidth=2)
plt.grid(True)
plt.legend(loc='upper right')
plt.xlim([0, k_steps])
plt.ylim([-0.2, 1.2])

# 系统状态x2结果图
plt.subplot(3, 1, 2)
plt.plot(x_history[1, :], 'b', label="x2", linewidth=1)
plt.plot(xd_history[1, :], 'r--', label="x2d", linewidth=2)
plt.grid(True)
plt.legend(loc='upper right')
plt.xlim([0, k_steps])
plt.ylim([-0.5, 0.5])

# 系统输入结果图
plt.subplot(3, 1, 3)
plt.step(range(k_steps), u_history[0, :], 'g', label="u", where='post')
plt.legend(loc='upper right')
plt.grid(True)
plt.xlim([0, k_steps])
plt.ylim([-0.5, 1.4])

plt.tight_layout()  # 自动调整子图间距
plt.show()  # 显示绘图窗口