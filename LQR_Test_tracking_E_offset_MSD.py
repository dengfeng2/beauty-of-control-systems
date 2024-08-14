#######################################
# 程序名称：LQR_Test_tracking_E_offset_MSD
# 程序功能：弹簧质量阻尼系统非零参考点分析-引入控制目标误差 （4.5.2节）
#######################################
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lti
from F1_LQR_Gain import F1_LQR_Gain

# Define system parameters
m_sys = 1.0
b_sys = 0.5
k_sys = 1.0

# System definition
A = np.array([[0., 1.], [-k_sys/m_sys, -b_sys/m_sys]])
n = A.shape[0]
B = np.array([[0.], [1./m_sys]])
p = B.shape[1]
C = np.array([[1., 0.]])  # Assuming we are interested in the position 'x'
D = np.array([[0.]])

# System discretization
Ts = 0.1

sys_c = lti(A, B, C, D)
sys_d = sys_c.to_discrete(Ts)
A_d = sys_d.A
B_d = sys_d.B

# Weighting matrix design
Q = np.eye(n)
S = np.eye(n)
R = np.array([[1.]])

# System reference value
xd = np.array([[1.], [0.]])
AD = np.eye(n)

# System initialization
x0 = np.array([[0.], [0.]])
x = x0
xa = np.vstack((x, xd))
u = np.array([[0.]])

# Run steps definition
k_steps = 100
x_history = np.zeros((n, k_steps + 1))
u_history = np.zeros((p, k_steps))

# Augmented matrices
Ca = np.hstack((np.eye(n), -np.eye(n)))
Aa = np.block([[A_d, np.zeros((n, n))],
               [np.zeros((n, n)), AD]])
Ba = np.vstack((B_d, np.zeros((n, p))))
Sa = Ca.T @ S @ Ca
Qa = Ca.T @ Q @ Ca

# Calculate feedback gain F
F = F1_LQR_Gain(Aa, Ba, Qa, R, Sa)

# Simulation loop
for k in range(k_steps):
    u = -F @ xa
    x = A_d @ x + B_d @ u
    xa = np.vstack((x, xd))
    x_history[:, k + 1] = x.flatten()
    u_history[:, k] = u.flatten()

# Results visualization
plt.figure(figsize=(6, 6))

# State variables plot
plt.subplot(2, 1, 1)
plt.plot(x_history[0, :], label='x1')
plt.plot(x_history[1, :], '--', label='x2')
plt.grid(True)
plt.legend()
plt.xlim([0, k_steps])
plt.ylim([-0.2, 1.2])

# Input variables plot
plt.subplot(2, 1, 2)
plt.step(range(k_steps), u_history[0, :], where='post', label='u')
plt.grid(True)
plt.legend()
plt.xlim([0, k_steps])
plt.ylim([0, 3])

plt.tight_layout()
plt.show()
