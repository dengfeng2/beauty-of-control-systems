#######################################
# 程序名称：MPC_1D
# 程序功能：模型预测控制一维示例（5.3.5节案例）
# 所用模块：
#        [F4]性能指标矩阵转换模块
#        [F5]无约束二次规划求解模块
#######################################
import numpy as np
import matplotlib.pyplot as plt
from F4_MPC_Matrices_PM import F4_MPC_Matrices_PM
from F5_MPC_Controller_noConstraints import F5_MPC_Controller_noConstraints
# 设置matplotlib支持中文的字体
plt.rcParams['font.sans-serif'] = ['Songti SC']   # 'Songti SC'是一款支持中文的字体
plt.rcParams['axes.unicode_minus'] = False     # 正确显示负号

# 定义系统参数
A = np.array([[1]])
B = np.array([[1]])
Q = np.array([[1]])
S = np.array([[1]])
R = np.array([[1]])
k_steps = 5
N_P = 5  # 预测区间

# 初始化变量
n = 1  # 状态维度
p = 1  # 输入维度
x_0 = np.array([1])
x = x_0
x2 = x_0

# 初始化历史记录数组
x_history = np.zeros((n, k_steps + 1))
x_history_2 = np.zeros((n, k_steps + 1))
u_history = np.zeros((p, k_steps))
u_history_2 = np.zeros((p, k_steps))

# 设置初始状态
x_history[:, 0] = x
x_history_2[:, 0] = x2

# 计算二次规划所需矩阵
Phi, Gamma, Omega, Psi, F, H = F4_MPC_Matrices_PM(A, B, Q, R, S, N_P)
# 计算离线控制输入
U_offline, u_offline = F5_MPC_Controller_noConstraints(x, F, H, p)

# 运行在线模拟
for k in range(k_steps):
    U, u = F5_MPC_Controller_noConstraints(x, F, H, p)
    x = A * x + B * u
    if k == 1:
        x += 0.2
    x_history[:, k + 1] = x
    u_history[:, k] = u

    # 离线控制输入
    u_offline = U_offline[k]
    x2 = A * x2 + B * u_offline
    if k == 1:
        x2 += 0.2
    x_history_2[:, k + 1] = x2
    u_history_2[:, k] = u_offline

# 系统状态结果视图 在线vs.离线
plt.subplot(2, 1, 1)
plt.plot(x_history[0], label="在线")
plt.plot(x_history_2[0], '--', label="离线")
plt.grid(True)
plt.legend()
plt.xlim([0, k_steps])
plt.title('系统状态')

# 系统输入结果视图 在线vs.离线
plt.subplot(2, 1, 2)
plt.step(range(k_steps), u_history[0], where='post', label="在线")
plt.step(range(k_steps), u_history_2[0], '--', where='post', label="离线")
plt.legend(loc='lower right')
plt.grid(True)
plt.xlim([0, k_steps])
plt.title('系统输入')

plt.show()
