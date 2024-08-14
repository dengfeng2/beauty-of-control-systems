#######################################
# 程序名称：F5_MPC_Controller_noConstraints %% [F5]无约束二次规划求解模块
# 模块功能：利用二次规划求解模型预测控制中的系统控制量
#######################################
import numpy as np
from scipy.optimize import minimize


def F5_MPC_Controller_noConstraints(x, F, H, p):
    # 目标函数
    def objective(U):
        return 0.5 * U.T @ H @ U + U.T @ (F @ x)

    # 初始控制输入猜测
    U0 = np.zeros(F.shape[0])

    # 边界条件（这里设置为空，因为没有约束条件）
    bnds = [(None, None) for _ in range(len(U0))]

    # 优化结果
    result = minimize(objective, U0, method='SLSQP', bounds=bnds, options={'maxiter': 200})

    # 如果优化成功，提取结果；否则，返回空值或错误处理
    if result.success:
        U = result.x
        u = U[:p]  # 取U向量的前p个元素
    else:
        raise ValueError("Optimization failed:", result.message)

    return U, u

