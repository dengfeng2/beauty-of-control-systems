#######################################
# 程序名称：F6_MPC_Matrices_Constraints %% [F6]约束条件矩阵转换模块
# 模块功能：生成MPC控制器所需的约束矩阵
#######################################
import numpy as np


def F6_MPC_Matrices_Constraints(x_low, x_high, u_low, u_high, N_P, Phi, Gamma):
    # Compute dimensions of system state and input
    n = len(x_low)
    p = len(u_low)

    # Construct M matrix
    M = np.vstack((np.zeros((p, n)), np.zeros((p, n)), -np.eye(n), np.eye(n)))
    # Construct F matrix
    F = np.vstack((-np.eye(p), np.eye(p), np.zeros((n, p)), np.zeros((n, p))))
    # Construct Beta matrix
    Beta = np.vstack((-u_low, u_high, -x_low, x_high))

    # Construct M_Np matrix
    M_Np = np.vstack((-np.eye(n), np.eye(n)))
    # Construct Beta_N matrix
    Beta_N = np.vstack((-x_low, x_high))

    # Construct M_bar matrix
    M_bar = np.zeros(((2 * n + 2 * p) * N_P + 2 * n, n))
    M_bar[:(2 * n + 2 * p), :] = M
    # Construct Beta_bar matrix (output of the module)
    Beta_bar = np.tile(Beta, (N_P, 1))
    Beta_bar = np.vstack((Beta_bar, Beta_N))

    # Initialize M_2bar and F_2bar matrices
    M_2bar = M
    F_2bar = F

    # Loop to create M_2bar and F_2bar matrices
    for i in range(1, N_P-1):
        M_2bar = np.block([
            [M_2bar, np.zeros((M_2bar.shape[0], M.shape[1]))],
            [np.zeros((M.shape[0], M_2bar.shape[1])), M]
        ])
        F_2bar = np.block([
            [F_2bar, np.zeros((F_2bar.shape[0], F.shape[1]))],
            [np.zeros((F.shape[0], F_2bar.shape[1])), F]
        ])

    M_2bar = np.block([
        [M_2bar, np.zeros((M_2bar.shape[0], M_Np.shape[1]))],
        [np.zeros((M_Np.shape[0], M_2bar.shape[1])), M_Np]
    ])
    # Final form of M_2bar matrix with zero row at the top
    M_2bar = np.vstack((np.zeros((2 * n + 2 * p, n * N_P)), M_2bar))
    F_2bar = np.block([
        [F_2bar, np.zeros((F_2bar.shape[0], F.shape[1]))],
        [np.zeros((F.shape[0], F_2bar.shape[1])), F]
    ])
    # Final form of F_2bar matrix with zero row at the bottom
    F_2bar = np.vstack((F_2bar, np.zeros((2 * n, p * N_P))))

    # Construct b matrix
    b = -(M_bar + M_2bar @ Phi)
    # Construct M matrix (final M as it's used outside)
    M = M_2bar @ Gamma + F_2bar

    return M, Beta_bar, b