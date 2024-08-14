#######################################
# 程序名称：F4_MPC_Matrices_PM %% [F4]性能指标矩阵转换模块
# 模块功能：
#        求解模型预测控制中二次规划所需矩阵F，H
#        求解模型预测控制一系列中间过程矩阵Phi，Gamma，Omega，Psi
#######################################
import numpy as np
from scipy.linalg import block_diag


def F4_MPC_Matrices_PM(A, B, Q, R, S, N_P):
    # 计算系统矩阵维度，n
    n = A.shape[0]
    # 计算输入矩阵维度，p
    p = B.shape[1]

    # 初始化Phi矩阵并定义维度
    Phi = np.zeros((N_P*n, n))
    # 初始化Gamma矩阵并定义维度
    Gamma = np.zeros((N_P*n, N_P*p))

    # for循环，用于构建Phi和Gamma矩阵
    for i in range(N_P):
        # 构建Phi矩阵
        Phi[i*n:(i+1)*n, :] = np.linalg.matrix_power(A, i+1)
        # 构建Gamma矩阵
        for j in range(i + 1):  # 计算Gamma的每一块
            Gamma[i * n:(i + 1) * n, j * p:(j + 1) * p] = np.linalg.matrix_power(A, i - j) @ B

    # 构建Omega矩阵，包含Q矩阵的部分
    Omega = np.kron(np.eye(N_P-1), Q)
    # 构建最终Omega矩阵，包含S矩阵
    Omega = block_diag(Omega, S)
    # 构建Psi矩阵，其为R矩阵组成的对角阵
    Psi = np.kron(np.eye(N_P), R)

    # 计算二次规划矩阵F
    F = Gamma.T @ Omega @ Phi
    # 计算二次规划矩阵H
    H = Psi + Gamma.T @ Omega @ Gamma

    return Phi, Gamma, Omega, Psi, F, H
