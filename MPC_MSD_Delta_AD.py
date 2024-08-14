#######################################
# 程序名称：MPC_MSD_Delta_AD
# 程序功能：弹簧质量阻尼系统模型预测控制-输入增量非常数目标 （5.4.2节案例）
# 所用模块：
#        [F3]输入增量控制矩阵转换模块
#        [F4]性能指标矩阵转换模块
#        [F5]无约束二次规划求解模块
#######################################
import numpy as np
from scipy.signal import cont2discrete
import matplotlib.pyplot as plt
from F3_InputAugmentMatrix_Delta_U import F3_InputAugmentMatrix_Delta_U
from F4_MPC_Matrices_PM import F4_MPC_Matrices_PM
from F5_MPC_Controller_noConstraints import F5_MPC_Controller_noConstraints

# System parameters
m_sys = 1
b_sys = 0.5
k_sys = 1

# System matrices A and B
A = np.array([[0, 1], [-k_sys / m_sys, -b_sys / m_sys]])
n = A.shape[0]
B = np.array([[0], [1 / m_sys]])
p = B.shape[1]

# Discretization of the system
Ts = 0.1
sys_d = cont2discrete((A, B, np.eye(n), np.zeros((n, p))), Ts)
A, B = sys_d[0], sys_d[1]

# Weighting matrices Q, S, and R
Q = np.eye(n)
S = np.eye(n)
R = 0.1

# Reference state xd and transition matrix AD
xd = np.array([[0], [0.2]])
AD = cont2discrete((np.array([[0, 1], [0, 0]]), np.zeros((2, 1)), np.eye(2), np.zeros((2, 1))), Ts)[0]

# Initial conditions
x0 = np.array([[0], [0]])
x = x0
u0 = 0
u = u0
xa = np.vstack([x, xd, u])

# Simulation settings
k_steps = 200
x_history = np.zeros((n, k_steps))
u_history = np.zeros((p, k_steps))
xd_history = np.zeros((n, k_steps))

# Prediction horizon
N_P = 20

# Calling F3 to compute augmented matrices
Aa, Ba, Qa, Sa, R = F3_InputAugmentMatrix_Delta_U(A, B, Q, R, S, AD)

# MPC matrices (you will need to define or import this function)
Phi, Gamma, Omega, Psi, F, H = F4_MPC_Matrices_PM(Aa, Ba, Qa, R, Sa, N_P)  # Use the corresponding Python implementation

# Main simulation loop with variable reference values
for k in range(k_steps):
    if k == 49:  # Adjust index since Python is 0-based while MATLAB is 1-based
        xd = np.array([xd[0], [-0.2]])
    elif k == 99:
        xd = np.array([xd[0], [0.2]])
    elif k == 149:
        xd = np.array([xd[0], [-0.2]])
    elif k == 199:
        xd = np.array([xd[0], [0.2]])

    # Compute control using F5 (MPC Controller)
    Delta_U, Delta_u = F5_MPC_Controller_noConstraints(xa, F, H, p)
    u += Delta_u
    x = A.dot(x) + B.dot(np.array([u]))

    # Update state and input histories
    xd = AD.dot(xd)
    xa = np.vstack([x, xd, u])
    x_history[:, k] = x.flatten()
    u_history[:, k] = u.flatten()
    xd_history[:, k] = xd.flatten()

# Plotting results
plt.figure(figsize=(10, 13.5))

# Plot x1 with reference
plt.subplot(3, 1, 1)
plt.plot(x_history[0, :], linewidth=1)
plt.plot(xd_history[0, :], '--', linewidth=2)
plt.grid(True)
plt.legend(["x1", "x1d"])
plt.xlim([0, k_steps - 1])
plt.ylim([-0.2, 1.2])

# Plot x2 with reference
plt.subplot(3, 1, 2)
plt.plot(x_history[1, :], linewidth=1)
plt.plot(xd_history[1, :], '--', linewidth=2)
plt.grid(True)
plt.legend(["x2", "x2d"])
plt.xlim([0, k_steps - 1])
plt.ylim([-0.5, 0.5])

# Plot system input
plt.subplot(3, 1, 3)
plt.step(range(k_steps), u_history[0, :], where='post')
plt.grid(True)
plt.legend(["u"])
plt.xlim([0, k_steps - 1])
plt.ylim([-0.5, 1.4])

plt.show()
