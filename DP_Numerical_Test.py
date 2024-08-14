#######################################
# 程序名称：DP_Numerical_Test
# 程序功能：无人机上升目标高度最短用时控制-动态规划数值方法 （4.2节案例）
#######################################
import numpy as np
import matplotlib.pyplot as plt

# 无人机高度初始化和速度初始化
h_init = 0.
v_init = 0.

# 无人机终点高度和速度
h_final = 10.
v_final = 0.

# 边界条件定义
h_min = 0.     # 高度下限
h_max = 10.    # 高度上限
N_h = 100     # 高度离散度
v_min = 0.     # 速度下限
v_max = 3.     # 速度上限
N_v = 500     # 速度离散度

# 创建离散向量
Hd = np.linspace(h_min, h_max, N_h + 1)
Vd = np.linspace(v_min, v_max, N_v + 1)

# 无人机加速度上下限设置
u_min = -3.
u_max = 2.

# 定义初始剩余代价矩阵
J_costTogo = np.zeros((N_h + 1, N_v + 1))
# 定义系统输入矩阵
Input_acc = np.zeros((N_h + 1, N_v + 1))

# 计算最后一级的情况
v_avg = 0.5 * (v_final + Vd)
T_delta = np.divide(h_max - h_min, N_h * v_avg)
acc = (v_final - Vd) / T_delta
acc_overLimit_indices = np.where(np.logical_or(acc <= u_min, acc >= u_max))
T_delta[acc_overLimit_indices] = np.inf
J_costTogo[1, :] = T_delta
Input_acc[1, :] = acc

# 倒数第二级至第二级的情况
Vd_x, Vd_y = np.meshgrid(Vd, Vd)
v_avg = 0.5 * (Vd_x + Vd_y)
T_delta = np.divide(h_max - h_min, N_h * v_avg)
acc = (Vd_y - Vd_x) / T_delta
acc_overLimit_indices = np.where(np.logical_or(acc <= u_min, acc >= u_max))
T_delta[acc_overLimit_indices] = np.inf
for k in range(2, N_h):
    J_temp = T_delta + np.tile(J_costTogo[k - 1, :], (N_v + 1, 1)).T
    J_costTogo[k, :], l = np.min(J_temp, axis=0), np.argmin(J_temp, axis=0)
    Input_acc[k, :] = acc[l, range(len(l))]

# 第二级至第一级的情况
v_avg = 0.5 * (Vd + v_init)
T_delta = np.divide(h_max - h_min, N_h * v_avg)
acc = (Vd - v_init) / T_delta
acc_overLimit_indices = np.where((acc <= u_min) | (acc >= u_max))
T_delta[acc_overLimit_indices] = np.inf
J_temp = T_delta + J_costTogo[N_h-1, :]
J_costTogo[N_h, 0], l = np.min(J_temp), np.argmin(J_temp)
Input_acc[N_h, 0] = acc[l]

# 结果（画图）
# 初始化参数
h_plot = np.zeros(N_h + 1)
v_plot = np.zeros(N_h + 1)
t_plot = np.zeros(N_h + 1)
acc_plot = np.zeros(N_h + 1)

# 初始条件
h_plot[0] = h_init
v_plot[0] = v_init
t_plot[0] = 0

# 查表确定最优路线
for k in range(N_h):
    h_plot_index = np.argmin(np.abs(h_plot[k] - Hd))
    v_plot_index = np.argmin(np.abs(v_plot[k] - Vd))
    acc_index = np.ravel_multi_index(((N_h + 1) - h_plot_index - 1, v_plot_index), Input_acc.shape)
    acc_plot[k] = Input_acc.flat[acc_index]
    v_2 = 2 * (h_max - h_min) / N_h * acc_plot[k] + v_plot[k]**2
    if 0 > v_2 > -1e-10:  # 浮点运算的误差可能导致sqrt失败
        v_plot[k + 1] = 0
    else:
        v_plot[k + 1] = np.sqrt(v_2)
    h_plot[k + 1] = h_plot[k] + (h_max - h_min) / N_h
    t_plot[k + 1] = t_plot[k] + 2 * (h_plot[k + 1] - h_plot[k]) / (v_plot[k + 1] + v_plot[k])

# 绘制视图
plt.subplot(3, 2, 1)
plt.plot(v_plot, h_plot, '--o')
plt.grid(True)
plt.xlabel('v(m/s)')
plt.ylabel('h(m)')

plt.subplot(3, 2, 2)
plt.plot(acc_plot[:-1], h_plot[:-1], '--o')  # Ignore the last point which is out of bound
plt.grid(True)
plt.xlabel('a(m/s^2)')
plt.ylabel('h(m)')

plt.subplot(3, 2, 3)
plt.plot(t_plot, v_plot, '--o')
plt.grid(True)
plt.xlabel('t(s)')
plt.ylabel('v(m/s)')

plt.subplot(3, 2, 4)
plt.plot(t_plot, h_plot, '--o')
plt.grid(True)
plt.xlabel('t(s)')
plt.ylabel('h(m)')

plt.subplot(3, 2, 5)
plt.plot(t_plot[:-1], acc_plot[:-1], '--o')  # Ignore the last point which is out of bound
plt.grid(True)
plt.xlabel('t(s)')
plt.ylabel('a(m/s^2)')

plt.tight_layout()
plt.show()