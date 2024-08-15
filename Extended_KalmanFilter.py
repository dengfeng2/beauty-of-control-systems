#######################################
# 程序名称：Extended_KalmanFilter
# 程序功能：扩展卡尔曼滤波器案例（6.5.2节案例）
#######################################
import numpy as np
import matplotlib.pyplot as plt
# 设置matplotlib支持中文的字体
plt.rcParams['font.sans-serif'] = ['Songti SC']   # 'Songti SC'是一款支持中文的字体
plt.rcParams['axes.unicode_minus'] = False     # 正确显示负号

# 离散时间步长
Ts = 0.01

# 参数设计
Q = np.array([[0.01, 0], [0, 0.01]])
R_c = np.array([[0.1, 0], [0, 0.1]])
g = 10  # 重力加速度
l = 0.5  # 连杆长度

# 系统初始化
x0 = np.array([[np.pi / 4.], [0.]])
x = x0
z0 = np.array([[0.], [0.]])
z = z0
x_hat_minus0 = np.array([[0.], [0.]])
x_hat_minus = x_hat_minus0
x_hat0 = np.array([[np.pi / 4.], [0.]])
x_hat = x_hat0
P0 = np.eye(2)
P = P0
n = x.shape[0]

# 定义系统运行步数
k_steps = 200
x_history = np.zeros((n, k_steps + 1))
x_hat_history = np.zeros((n, k_steps + 1))
x_hat_minus_history = np.zeros((n, k_steps + 1))
z_history = np.zeros((n, k_steps + 1))

# 生成过程与测量噪声
Q_a = np.array([[0.01, 0], [0, 0.01]])
R_a = np.array([[0.1, 0], [0, 0.1]])
w = np.dot(np.linalg.cholesky(Q_a), np.random.randn(2, k_steps))
v = np.dot(np.sqrt(R_a), np.random.randn(2, k_steps))

# Kalman Filter
for k in range(k_steps):
    # 系统状态空间方程，计算实际状态变量x1和x2
    x[0] = x[0] + x[1] * Ts + w[0, k]
    x[1] = x[1] - (g / l) * np.sin(x[0]) * Ts + w[1, k]
    # 计算实际测量值
    z[0] = x[0] + v[0, k]
    z[1] = x[1] + v[1, k]

    # 扩展卡尔曼滤波器
    x_hat_minus[0] = x_hat[0] + x_hat[1] * Ts
    x_hat_minus[1] = x_hat[1] - (g / l) * np.sin(x_hat[0]) * Ts
    A = np.array([[1, Ts], [-(g / l) * np.cos(x_hat[0][0]) * Ts, 1]])
    W = np.eye(n)
    H_m = np.eye(n)
    V = np.eye(n)
    P_minus = A @ P @ A.T + W @ Q @ W.T
    K = P_minus @ H_m.T @ np.linalg.inv(H_m @ P_minus @ H_m.T + V @ R_c @ V.T)
    x_hat = x_hat_minus + K @ (z - x_hat_minus)
    P = (np.eye(n) - K @ H_m) @ P_minus

    # 保存数据
    aaa = x.ravel()
    x_history[:, k + 1] = x.ravel()
    z_history[:, k + 1] = z.ravel()
    x_hat_minus_history[:, k + 1] = x_hat_minus.ravel()
    x_hat_history[:, k + 1] = x_hat.ravel()

# 结果显示
plt.figure(figsize=(15, 5))
plt.plot(x_history[0, :], '--', linewidth=2, label='真实值')
plt.plot(z_history[0, :], '*', markersize=8, label='测量值')
plt.plot(x_hat_minus_history[0, :], 'o', markersize=8, label='先验估计值')
plt.plot(x_hat_history[0, :], linewidth=2, label='后验估计值')
plt.ylim([-3, 3])
plt.legend(loc='lower right', fontsize=20)
plt.grid(True)
plt.show()

# 计算均方误差
Ems = np.sum((x_hat_history[0, :] - x_history[0, :]) ** 2) / k_steps
print("均方误差 Ems =", Ems)