#######################################
# 程序名称：F2_InputAugmentMatrix_SS_U %% [F2]稳态非零控制矩阵转化模块
# 模块功能：计算系统增广矩阵Aa，Ba，Qa，Sa，R以及稳态控制输入ud
#######################################
import numpy as np


def F2_InputAugmentMatrix_SS_U(A, B, Q, R, S, xd):
    # Calculate dimensions of system matrix A and input matrix B
    n = A.shape[0]
    p = B.shape[1]

    # Construct augmented matrix Ca
    Ca = np.block([np.eye(n), -np.eye(n)])

    # Construct augmented matrix Aa
    Aa = np.block([
        [A, np.eye(n) - A],
        [np.zeros((n, n)), np.eye(n)]
    ])

    # Construct augmented matrix Ba
    Ba = np.block([
        [B],
        [np.zeros((n, p))]
    ])

    # Construct augmented matrix Qa
    Qa = Ca.T @ Q @ Ca

    # Construct augmented matrix Sa
    Sa = Ca.T @ S @ Ca

    # Compute the steady-state control input ud
    # Here we use the pseudo-inverse since B may not be square or invertible
    ud = np.linalg.pinv(B) @ ((np.eye(n) - A) @ xd)

    return Aa, Ba, Qa, Sa, R, ud
