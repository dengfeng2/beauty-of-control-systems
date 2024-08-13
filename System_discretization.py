#######################################
# 程序名称：System_discretization
# 程序功能：系统离散化与比较
#######################################
import numpy as np
from scipy.linalg import expm, solve
from scipy.signal import StateSpace, dstep, step, cont2discrete
import matplotlib.pyplot as plt

# 构建系统矩阵A
A = np.array([[0., 1.],
              [-2., -3.]])
# 构建输入矩阵B
B = np.array([[0.],
              [1.]])

C = np.array([[1., 0.]])
D = np.array([[0.]])
# 定义两组采样时间
Ts_1 = 0.2
Ts_2 = 1

# 根据公式计算；
Fd_1 = expm(A * Ts_1)
Gd_1 = solve(A, (Fd_1 - np.eye(A.shape[0]))).dot(B)

Fd_2 = expm(A * Ts_2)
Gd_2 = solve(A, (Fd_2 - np.eye(A.shape[0]))).dot(B)

# 连续系统的单位阶跃响应
sys_c = StateSpace(A, B, C, D)
t_c, y_c = step(sys_c)

# 连续系统转离散系统并计算阶跃响应
sys_d_1 = cont2discrete((A, B, C, D), Ts_1)
t_d_1, y_d_1 = dstep(sys_d_1)
y_d_1 = y_d_1[0].squeeze()

# 连续系统转离散系统并计算阶跃响应
sys_d_2 = cont2discrete((A, B, C, D), Ts_2)
t_d_2, y_d_2 = dstep(sys_d_2)
y_d_2 = y_d_2[0].squeeze()

plt.plot(t_c, y_c, 'r', label='Continuous System')
plt.step(t_d_1[:50], y_d_1[:50], 'b', label=f'Discrete System (Ts={Ts_1})', where='post')
plt.step(t_d_2[:10], y_d_2[:10], label=f'Discrete System (Ts={Ts_2})', where='post')

plt.xlabel('Time (s)')
plt.ylabel('Response')
plt.title('Step Response of Systems')
plt.legend()
plt.show()
