#######################################
# 程序名称：MPC_2D
# 程序功能：模型预测控制二维系统示例
# 所用模块：
#        [F4]性能指标矩阵转换模块
#        [F6]约束条件矩阵转换模块
#        [F7]含约束二次规划求解模块
#######################################
import numpy as np
import matplotlib.pyplot as plt
from F4_MPC_Matrices_PM import F4_MPC_Matrices_PM
from F6_MPC_Matrices_Constraints import F6_MPC_Matrices_Constraints
from F7_MPC_Controller_withConstriants import F7_MPC_Controller_withConstriants


# System definition
A = np.array([[1., 0.1], [0, -2.]])
n = A.shape[0]
B = np.array([[0., 0.2], [-0.1, 0.5]])
p = B.shape[1]

# Weight design
Q = np.eye(n)
S = np.eye(n)
R = np.array([[0.1, 0], [0, 0.1]])

# System initialization
x_0 = np.array([[1.], [-1.]])
x = x_0

# System constraints definition
x_low = np.array([[-np.inf], [-np.inf]])
x_high = np.array([[np.inf], [0]])
u_low = np.array([[-np.inf], [-3]])
u_high = np.array([[np.inf], [np.inf]])

# Simulation settings
k_steps = 10
x_history = np.zeros((n, k_steps+1))
x_history[:, 0] = x[:, 0]
u_history = np.zeros((p, k_steps))

# Prediction horizon
N_P = 2

# Compute quadratic programming matrices using F4
Phi, Gamma, Omega, Psi, F, H = F4_MPC_Matrices_PM(A, B, Q, R, S, N_P)

# Compute constrained quadratic programming matrices using F6
M, Beta_bar, b = F6_MPC_Matrices_Constraints(x_low, x_high, u_low, u_high, N_P, Phi, Gamma)

# Main simulation loop
for k in range(k_steps):
    # Compute system control (input) using F7
    U, u = F7_MPC_Controller_withConstriants(x, F, H, M, Beta_bar, b, p)
    # Apply control input and compute system response
    x = A @ x + B @ (u.reshape((-1, 1)))
    # Save system state and input to predefined matrices
    x_history[:, k+1] = x[:, 0]
    u_history[:, k] = u

# Plot results
plt.figure(figsize=(10, 10))
# Plot system state results
plt.subplot(2, 1, 1)
plt.plot(x_history[0, :], label='x1')
plt.plot(x_history[1, :], '--', label='x2')
plt.legend()
plt.grid(True)
plt.xlim([0, k_steps - 1])
plt.ylim([-1, 1])
# Plot system input results
plt.subplot(2, 1, 2)
plt.step(range(k_steps), u_history[0, :], where='post', label='u1')
plt.step(range(k_steps), u_history[1, :], where='post', linestyle='--', label='u2')
plt.legend()
plt.grid(True)
plt.xlim([0, k_steps - 1])
plt.ylim([-4, 6])
plt.show()