#######################################
# 程序名称：MPC_MSD_SS_U
# 程序功能：弹簧质量阻尼系统模型预测控制-稳态输入 （5.4.1节案例）
# 所用模块：
#        [F2]稳态非零控制矩阵转化模块
#        [F4]性能指标矩阵转换模块
#        [F5]无约束二次规划求解模块
#######################################
import numpy as np
from scipy.signal import lti
import matplotlib.pyplot as plt
from F2_InputAugmentMatrix_SS_U import F2_InputAugmentMatrix_SS_U
from F4_MPC_Matrices_PM import F4_MPC_Matrices_PM
from F5_MPC_Controller_noConstraints import F5_MPC_Controller_noConstraints

# System parameter definition
m_sys = 1  # Mass of the block
b_sys = 0.5  # Damping coefficient
k_sys = 1  # Spring constant

# System definition
A = np.array([[0, 1], [-k_sys / m_sys, -b_sys / m_sys]])
n = A.shape[0]  # System state dimension
B = np.array([[0], [1 / m_sys]])
p = B.shape[1]  # Input matrix dimension

C = np.array([[1, 0]])  # Assuming we are interested in the position 'x'
D = np.array([[0]])

# System discretization
Ts = 0.1
sys_c = lti(A, B, C, D)
sys_d = sys_c.to_discrete(Ts)
A = sys_d.A
B = sys_d.B

# Weighting design
Q = np.eye(n)
S = np.eye(n)
R = 1

# System reference values
xd = np.array([[1], [0]])
AD = np.eye(n)

# System initialization
x0 = np.array([[0], [0]])
x = x0
xa = np.vstack((x, xd))
u = 0

# Simulation settings
k_steps = 100
x_history = np.zeros((n, k_steps + 1))
u_history = np.zeros((p, k_steps))
delta_u = 0

# Prediction horizon
N_P = 20

# Calculate augmented matrices
Aa, Ba, Qa, Sa, R, ud = F2_InputAugmentMatrix_SS_U(A, B, Q, R, S, xd)

# Calculate matrices for quadratic programming
Phi, Gamma, Omega, Psi, F, H = F4_MPC_Matrices_PM(Aa, Ba, Qa, R, Sa, N_P)

# Main simulation loop
for k in range(k_steps):
    delta_U, delta_u = F5_MPC_Controller_noConstraints(xa, F, H, p)
    x = A.dot(x) + B.dot(delta_u + ud)
    xa = np.vstack((x, xd))
    x_history[:, k + 1] = x.flatten()
    u_history[:, k] = delta_u + ud

# Plotting results
plt.figure(figsize=(10, 10))

# Plot state variables
plt.subplot(2, 1, 1)
plt.plot(range(k_steps + 1), x_history[0, :])
plt.plot(range(k_steps + 1), x_history[1, :], '--')
plt.grid(True)
plt.legend(['x1', 'x2'])
plt.xlim([0, k_steps])
plt.ylim([-0.2, 1.2])

# Plot system input
plt.subplot(2, 1, 2)
plt.step(range(k_steps), u_history[0, :], where='post')
plt.grid(True)
plt.legend(['u'])
plt.xlim([0, k_steps])
plt.ylim([0, 3])

plt.show()
