#######################################
# 程序名称：F7_MPC_Controller_withConstraints %% [F7]含约束二次规划求解模块
# 模块功能：利用二次规划求解模型预测控制中的系统控制量-含约束
#######################################
import numpy as np
from scipy.optimize import minimize


# 输入: 二次规划矩阵 F, H；系统控制量维度 p；系统状态：x 约束条件矩阵 M, Beta_bar, b
# 输出: 系统控制（输入） U, u
def F7_MPC_Controller_withConstriants(x, F, H, M, Beta_bar, b, p):
    # 利用二次规划求解系统控制（输入）
    # 目标函数
    def objective(U):
        return 0.5 * U.T @ H @ U + (F @ x).T @ U

    # 初始控制输入猜测
    U0 = np.zeros(F.shape[0])

    cons_num = M.shape[0]
    cons = [{'type': 'ineq', 'fun': lambda U: ((Beta_bar + b @ x).flatten()[i] - M[i, :] @ U)} for i in range(cons_num)]

    # 优化结果
    result = minimize(objective, U0, method='SLSQP', constraints=tuple(cons), options={'maxiter': 2000})

    # 如果优化成功，提取结果；否则，返回空值或错误处理
    if result.success:
        U = result.x
        u = U[:p]  # 取U向量的前p个元素
    else:
        raise ValueError("Optimization failed:", result.message)

    return U, u