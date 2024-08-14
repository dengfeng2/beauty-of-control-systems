import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import cont2discrete
import time
from F2_InputAugmentMatrix_SS_U import F2_InputAugmentMatrix_SS_U
from F4_MPC_Matrices_PM import F4_MPC_Matrices_PM
from F6_MPC_Matrices_Constraints import F6_MPC_Matrices_Constraints
from F7_MPC_Controller_withConstriants import F7_MPC_Controller_withConstriants

# Define system parameters
# Define drone mass
m = 1
# Define gravitational acceleration constant
g = 10

# System definition
# Construct system matrix A, n x n
A = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
# Calculate dimension of A matrix
n = A.shape[0]
# Construct input matrix B, n x p
B = np.array([[0], [1 / m], [0]])
# Calculate dimension of input matrix
p = B.shape[1]

# System discretization
# Discrete time step (sampling time affects computation speed)
Ts = 0.1
# Convert continuous system to discrete system
sys_d = cont2discrete((A, B, np.zeros((n, 1)), np.zeros((1, 1))), Ts)
# Extract discrete system A matrix
A = sys_d[0]
# Extract discrete system B matrix
B = sys_d[1]

# Weight design
# Design state weight coefficient matrix, n x n
Q = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
# Design terminal weight coefficient matrix, n x n
S = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
# Design input weight coefficient matrix, p x p
R = np.array([[0.1]])

# System reference values
# System state reference value
xd = np.array([[10], [0], [-g]])
# Construct target transition matrix
AD = np.eye(n)

# System initialization
# Initialize system state
x0 = np.array([[0], [0], [-g]])
x = x0
# Initialize augmented state matrix
xa = np.vstack((x, xd))
# Initialize system input
u0 = 0
u = 0

# System constraints definition
# Input lower limit
u_low = np.array([[-3]])
# Input upper limit
u_high = np.array([[2]])
# State lower limit
x_low = np.array([[0], [0], [-10]])
# State upper limit
x_high = np.array([[10], [3], [-10]])
# Augmented state lower limit
xa_low = np.vstack((x_low, -np.inf * np.ones((n, 1))))
# Augmented state upper limit
xa_high = np.vstack((x_high, np.inf * np.ones((n, 1))))

# Define the number of steps the system runs (the computation steps need to be adjusted according to the sampling time)
k_steps = 100
# Define x_history zero matrix to store system state results, dimension n x k_step
x_history = np.zeros((n, k_steps + 1))
# Define u_history zero matrix to store system input results, dimension p x k_step
u_history = np.zeros((p, k_steps))

# Define prediction interval, please adjust according to sampling time
N_P = 100

# Define program runtime vector to store computation time and consider feasibility of real-time control
elapsed_time_history = np.zeros(k_steps)

Aa, Ba, Qa, Sa, R, ud = F2_InputAugmentMatrix_SS_U(A, B, Q, R, S, xd)

Phi, Gamma, Omega, Psi, F, H = F4_MPC_Matrices_PM(Aa, Ba, Qa, R, Sa, N_P)

M, Beta_bar, b = F6_MPC_Matrices_Constraints(xa_low, xa_high, u_low, u_high, N_P, Phi, Gamma)

# Start simulation loop
for k in range(k_steps):
    # Start timer
    start_time = time.time()

    delta_U, delta_u = F7_MPC_Controller_withConstriants(xa, F, H, M, Beta_bar, b, p)

    # Based on the increment calculate the system input
    u = delta_u + ud
    # System input into the system equation to calculate system response
    x = A.dot(x) + B.dot(u)
    # Update augmented matrix xa
    xa = np.vstack((x, xd))
    # Save the system state to the corresponding position in the predefined matrix
    x_history[:, k + 1] = x.flatten()
    # Save the system input to the corresponding position in the predefined matrix
    u_history[:, k] = u.flatten()
    # Record runtime
    elapsed_time = time.time() - start_time
    # Store the running time in the vector at the corresponding position
    elapsed_time_history[k] = elapsed_time

# Results display Figure 1
plt.figure(figsize=(10, 16))
# Plot state variable results
plt.subplot(3, 1, 1)
plt.plot(x_history[0, :], label="x1")
plt.plot(x_history[1, :], '--', label="x2")
plt.grid(True)
plt.legend()
plt.xlim([0, k_steps])
plt.ylim([-2, 11])

# Plot system input
plt.subplot(3, 1, 2)
plt.step(range(k_steps + 1), u_history[0, :], where='post', label="u")
plt.legend()
plt.grid(True)
plt.xlim([0, k_steps])
plt.ylim([6, 12])

# Plot computation time for each iteration step
plt.subplot(3, 1, 3)
plt.plot(elapsed_time_history, linewidth=2)
plt.grid(True)
plt.xlim([0, k_steps])

plt.show()