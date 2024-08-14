#######################################
# 程序名称：MPC_MSD_Delta_U
# 程序功能：弹簧质量阻尼系统模型预测控制-输入增量 （5.4.2节案例）
# 所用模块：
#        [F3]输入增量控制矩阵转换模块
#        [F4]性能指标矩阵转换模块
#        [F5]无约束二次规划求解模块
#######################################
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import cont2discrete
from F3_InputAugmentMatrix_Delta_U import F3_InputAugmentMatrix_Delta_U
from F4_MPC_Matrices_PM import F4_MPC_Matrices_PM
from F5_MPC_Controller_noConstraints import F5_MPC_Controller_noConstraints

# System parameters definition
m_sys = 1
b_sys = 0.5
k_sys = 1

# System matrices construction
A = np.array([[0, 1], [-k_sys/m_sys, -b_sys/m_sys]])
n = A.shape[0]
B = np.array([[0], [1/m_sys]])
p = B.shape[1]

# Discretization of the system
Ts = 0.1
sys_d = cont2discrete((A, B, np.eye(n), np.zeros((n, p))), Ts)
A, B = sys_d[0], sys_d[1]

# Weighting matrix design
Q = np.eye(n)
S = np.eye(n)
R = 0.1

# System reference values
xd = np.array([[1], [0]])
AD = np.eye(n)

# System initialization
x0 = np.array([[0], [0]])
x = x0
u0 = 0
u = u0
xa = np.vstack([x, xd, u])

# Simulation settings
k_steps = 100
x_history = np.zeros((n, k_steps+1))
u_history = np.zeros((p, k_steps))

# Prediction horizon
N_P = 20

# Calculate augmented matrices using F3 module
Aa, Ba, Qa, Sa, R = F3_InputAugmentMatrix_Delta_U(A, B, Q, R, S, AD)

# Calculate matrices for quadratic programming using F4 module
Phi, Gamma, Omega, Psi, F, H = F4_MPC_Matrices_PM(Aa, Ba, Qa, R, Sa, N_P)

# Main simulation loop
for k in range(k_steps):
    Delta_U, Delta_u = F5_MPC_Controller_noConstraints(xa, F, H, p)
    u += Delta_u
    x = A.dot(x) + B.dot(np.array([u]))
    xa = np.vstack([x, xd, u])
    x_history[:, k+1] = x.flatten()
    u_history[:, k] = u.flatten()

# Plotting results
plt.figure(figsize=(10, 10))

# Plot state variables
plt.subplot(2, 1, 1)
plt.plot(x_history[0, :], label="x1")
plt.plot(x_history[1, :], '--', label="x2")
plt.grid(True)
plt.legend()
plt.xlim([0, k_steps])
plt.ylim([-0.2, 1.2])

# Plot system input
plt.subplot(2, 1, 2)
plt.step(range(k_steps+1), np.hstack([u0, u_history[0, :]]), where='post', label="u")
plt.grid(True)
plt.legend()
plt.xlim([0, k_steps])
plt.ylim([0, 3])

plt.show()