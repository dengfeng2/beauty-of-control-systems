#######################################
# 程序名称：LQR_InvertedPed_Pendulum
# 程序功能：平衡车控制 - 连续系统案例分析 （4.4.5节案例）
#######################################
import numpy as np
from scipy.linalg import solve_continuous_are
from scipy.signal import StateSpace, lsim
import matplotlib.pyplot as plt

# Define system parameters
g = 10
d = 1

# System definition
A = np.array([[0, 1], [g/d, 0]])
n = A.shape[0]
B = np.array([[0], [1]])
p = B.shape[1]
C = np.array([[1, 0]])
D = 0

# Weighting matrices design for different scenarios
q1 = np.array([[100, 0], [0, 1]])
q2 = np.array([[1, 0], [0, 100]])
q3 = np.array([[1, 0], [0, 1]])
r1 = r2 = r3 = np.array([[1]])

# Initial state
x0 = np.array([np.pi / 20, 0])

# Solve the Algebraic Riccati Equation (ARE) for different scenarios
P1 = solve_continuous_are(A, B, q1, r1)
K1 = np.linalg.inv(r1) @ B.T @ P1

P2 = solve_continuous_are(A, B, q2, r2)
K2 = np.linalg.inv(r2) @ B.T @ P2

P3 = solve_continuous_are(A, B, q3, r3)
K3 = np.linalg.inv(r3) * B.T @ P3

# Closed-loop systems
sys_cl1 = StateSpace(A - B @ K1, [[0], [0]], C, D)
sys_cl2 = StateSpace(A - B @ K2, [[0], [0]], C, D)
sys_cl3 = StateSpace(A - B @ K3, [[0], [0]], C, D)

# Simulation
t_span = 0.01
t = np.arange(0, 5 + t_span, t_span)

# Initial response of closed-loop systems
_, y1, x1 = lsim(sys_cl1, U=np.zeros(len(t)), T=t, X0=x0)
_, y2, x2 = lsim(sys_cl2, U=np.zeros(len(t)), T=t, X0=x0)
_, y3, x3 = lsim(sys_cl3, U=np.zeros(len(t)), T=t, X0=x0)

# Results visualization
plt.figure(figsize=(10, 12))

# Plot the response of the first state (angle)
plt.subplot(3, 1, 1)
plt.plot(t, x1[:, 0], linewidth=2)
plt.plot(t, x2[:, 0], '--', linewidth=2)
plt.plot(t, x3[:, 0], '-.', linewidth=2)
plt.legend(['Test 1', 'Test 2', 'Test 3'], fontsize=14)
plt.xlim([0, 4])
plt.grid(True)

# Plot the response of the second state (angular velocity)
plt.subplot(3, 1, 2)
plt.plot(t, x1[:, 1], linewidth=2)
plt.plot(t, x2[:, 1], '--', linewidth=2)
plt.plot(t, x3[:, 1], '-.', linewidth=2)
plt.legend(['Test 1', 'Test 2', 'Test 3'], fontsize=14)
plt.xlim([0, 4])
plt.grid(True)

# Plot the input (acceleration)
plt.subplot(3, 1, 3)
plt.plot(t, (-K1 @ x1.T).T, linewidth=2)
plt.plot(t, (-K2 @ x2.T).T, '--', linewidth=2)
plt.plot(t, (-K3 @ x3.T).T, '-.', linewidth=2)
plt.legend(['Test 1', 'Test 2', 'Test 3'], fontsize=14)
plt.xlim([0, 4])
plt.grid(True)

plt.tight_layout()
plt.show()
