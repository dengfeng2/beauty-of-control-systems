#######################################
# 程序名称：LQR_Test_1D
# 程序功能：离散型一维案例分析 - LQR方法 （4.4.2节案例）
#######################################
import numpy as np
import matplotlib.pyplot as plt
from F1_LQR_Gain import F1_LQR_Gain


# System definition
A = np.array([[1]])
n = A.shape[0]  # Dimension of the system matrix A
B = np.array([[1]])
p = B.shape[1]  # Dimension of the input matrix B

# Weighting matrices design
Q = np.array([[1]])
S = np.array([[1]])
R = np.array([[1]])

# System reference value
xd = 0

# System initialization
x0 = 1  # Initial state
x = x0
x = np.array([x])
u0 = 0  # Initial input
u = u0
u = np.array([u])

# Simulation parameters
k_steps = 20
x_history = np.zeros((n, k_steps + 1))
x_history[:, 0] = x
u_history = np.zeros((p, k_steps))
u_history[:, 0] = u
N = k_steps  # Control interval

# Calculate system feedback gain F
F = F1_LQR_Gain(A, B, Q, R, S)

# Simulation loop
for k in range(k_steps):
    # Compute system input
    u = -F @ x  # Using the @ operator for matrix multiplication
    # Update system response
    x = A @ x + B @ u
    # Store results
    x_history[:, k+1] = x
    u_history[:, k] = u

# Results visualization
# System state vs. number of steps
plt.subplot(2, 1, 1)
plt.plot(x_history[0, :])
plt.legend(["x1"])
plt.grid(True)

# System input vs. number of steps
plt.subplot(2, 1, 2)
plt.plot(u_history[0, :])
plt.legend(["u1"])
plt.grid(True)

plt.show()
