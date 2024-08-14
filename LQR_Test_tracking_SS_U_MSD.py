#######################################
# 程序名称：LQR_Test_tracking_SS_U_MSD
# 程序功能：弹簧质量阻尼系统 - 稳态非零参考值控制 （4.5.3节案例）
# 所用模块：
#        [F1]反馈矩阵求解模块
#        [F2]稳态非零控制矩阵转化模块
#######################################
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lti
from F2_InputAugmentMatrix_SS_U import F2_InputAugmentMatrix_SS_U
from F1_LQR_Gain import F1_LQR_Gain

# Define system parameters
m_sys = 1
b_sys = 0.5
k_sys = 1

# Define system matrices A, B
A = np.array([[0, 1], [-k_sys/m_sys, -b_sys/m_sys]])
B = np.array([[0], [1/m_sys]])
C = np.array([[1, 0]])  # Assuming we are interested in the position 'x'
D = np.array([[0]])

# Discretize the system
Ts = 0.1
sys_c = lti(A, B, C, D)
sys_d = sys_c.to_discrete(Ts)
A = sys_d.A
B = sys_d.B

# Define weighting matrices Q, R, S
Q = np.eye(2)
R = np.array([[0.1]])
S = np.eye(2)

# Define system reference value xd
xd = np.array([[1], [0]])

# Initialize state x and augmented state xa
x0 = np.array([[0], [0]])
x = x0
xa = np.vstack((x, xd))

# Initialize input u
u = 0

# Define number of steps
k_steps = 100
x_history = np.zeros((2, k_steps + 1))
u_history = np.zeros((1, k_steps))

# Example function calls (assuming these functions are implemented)
Aa, Ba, Qa, Sa, R, ud = F2_InputAugmentMatrix_SS_U(A, B, Q, R, S, xd)
F = F1_LQR_Gain(Aa, Ba, Qa, R, Sa)

# Simulation loop
for k in range(k_steps):
    # Compute input
    u = -F @ xa + ud
    # Update system states
    x = A @ x + B @ u
    # Update augmented state
    xa = np.vstack((x, xd))
    # Store states and inputs
    x_history[:, k + 1] = x.flatten()
    u_history[:, k] = u.flatten()

# Plot results
plt.figure(figsize=(6, 6))

# Plot state variables
plt.subplot(2, 1, 1)
plt.plot(x_history[0, :], label='x1')
plt.plot(x_history[1, :], '--', label='x2')
plt.grid(True)
plt.legend()
plt.xlim([0, k_steps])
plt.ylim([-0.2, 1.2])

# Plot input variable
plt.subplot(2, 1, 2)
plt.step(range(k_steps), u_history[0, :], where='post', label='u')
plt.grid(True)
plt.legend()
plt.xlim([0, k_steps])
plt.ylim([0, 3])

plt.tight_layout()
plt.show()
