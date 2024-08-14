#######################################
# 程序名称：MPC_MSD_SS_U_withConstraints
# 程序功能：弹簧质量阻尼系统模型预测控制示例-用约束限制超调量
# 所用模块：
#        [F2]稳态非零控制矩阵转化模块
#        [F4]性能指标矩阵转换模块
#        [F5]无约束二次规划求解模块
#        [F6]约束条件矩阵转换模块
#        [F7]含约束二次规划求解模块
#######################################
import numpy as np
from scipy.signal import cont2discrete, dlti
import matplotlib.pyplot as plt
from F2_InputAugmentMatrix_SS_U import F2_InputAugmentMatrix_SS_U
from F4_MPC_Matrices_PM import F4_MPC_Matrices_PM
from F5_MPC_Controller_noConstraints import F5_MPC_Controller_noConstraints
from F6_MPC_Matrices_Constraints import F6_MPC_Matrices_Constraints
from F7_MPC_Controller_withConstriants import F7_MPC_Controller_withConstriants

# Define system parameters
m_sys = 1.0  # Mass of the block
b_sys = 0.5  # Damping coefficient
k_sys = 1.0  # Spring constant

# System matrix A, n x n
A = np.array([[0, 1], [-k_sys/m_sys, -b_sys/m_sys]])
n = A.shape[0]  # Size of A matrix
# Input matrix B, n x p
B = np.array([[0], [1/m_sys]])
p = B.shape[1]  # Size of B matrix

# Discretization of the system
Ts = 0.1
sys_d = cont2discrete((A, B, np.eye(n), np.zeros((n, p))), Ts)
A, B = sys_d[0], sys_d[1]

# Weighting matrices design
Q = np.eye(n)       # State weight matrix
S = np.eye(n)       # Terminal weight matrix
R = np.array([[1]]) # Input weight matrix

# Reference values for the system state
xd = np.array([[1], [0]])
AD = np.eye(n)

# Initialize system state
x0 = np.zeros((n, 1))
x = x0
# Initialize augmented state matrix
xa = np.vstack([x, xd])
# Initialize system input
u0 = np.zeros((p, 1))
u = u0

# System constraints
u_low = np.array([[-np.inf]])
u_high = np.array([[np.inf]])
x_low = np.array([[0], [-np.inf]])
x_high = np.array([[1], [np.inf]])
xa_low = np.vstack([x_low, -np.inf * np.ones((n, 1))])
xa_high = np.vstack([x_high, np.inf * np.ones((n, 1))])

# Define the number of simulation steps
k_steps = 100
# Initialize history matrices
x_history = np.zeros((n, k_steps + 1))
u_history = np.zeros((p, k_steps))
x_history_noconstraint = np.zeros((n, k_steps + 1))
u_history_noconstraint = np.zeros((p, k_steps))
N_P = 20  # Prediction horizon

# Compute augmented matrices for MPC
Aa, Ba, Qa, Sa, R, ud = F2_InputAugmentMatrix_SS_U(A, B, Q, R, S, xd)

# Compute quadratic programming matrices for MPC
Phi, Gamma, Omega, Psi, F, H = F4_MPC_Matrices_PM(Aa, Ba, Qa, R, Sa, N_P)

# Compute constrained quadratic programming matrices for MPC
M, Beta_bar, b = F6_MPC_Matrices_Constraints(xa_low, xa_high, u_low, u_high, N_P, Phi, Gamma)

# Simulation loop with constraints
for k in range(k_steps):
    # Compute control input increment with constraints
    delta_U, delta_u = F7_MPC_Controller_withConstriants(xa, F, H, M, Beta_bar, b, p)
    # Update system input based on increment
    u = delta_u + ud
    # Calculate system response
    x = A @ x + B @ u
    # Update augmented state
    xa = np.vstack([x, xd])
    # Store system state and input history
    x_history[:, k + 1] = x.flatten()
    u_history[:, k] = u.flatten()

# Reset system state for unconstrained simulation
x = x0
xa = np.vstack([x, xd])
u = u0

# Simulation loop without constraints
for k in range(k_steps):
    # Compute control input increment without constraints
    delta_U, delta_u = F5_MPC_Controller_noConstraints(xa, F, H, p)
    # Calculate system response
    x = A @ x + B @ (delta_u + ud)
    # Update augmented state
    xa = np.vstack([x, xd])
    # Store unconstrained system state and input history
    x_history_noconstraint[:, k + 1] = x.flatten()
    u_history_noconstraint[:, k] = (delta_u + ud).flatten()

# Plotting results
plt.figure(figsize=(15, 6))

# Plot state variable results
plt.subplot(2, 2, 1)
plt.plot(x_history[0, :], label='x1')
plt.plot(x_history[1, :], label='x2', linestyle='--')

plt.grid(True)
plt.legend()
plt.xlim([0, k_steps])
plt.ylim([-0.2, 1.2])

plt.show()