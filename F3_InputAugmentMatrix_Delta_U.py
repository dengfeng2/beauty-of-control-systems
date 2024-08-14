#######################################
# 程序名称：F3_InputAugmentMatrix_Delta_U %% [F3]输入增量控制矩阵转换模块
# 模块功能：计算系统增广矩阵Aa，Ba，Qa，Sa，R
#######################################
import numpy as np

def F3_InputAugmentMatrix_Delta_U(A, B, Q, R, S, AD):
    # 计算系统矩阵维度 n
    n = A.shape[0]
    # 计算输入矩阵维度 p
    p = B.shape[1]
    # 构建增广矩阵Ca，参考式（4.5.25）
    Ca = np.block([np.eye(n), -np.eye(n), np.zeros((n, p))])
    # 构建增广矩阵Aa，参考式（4.5.24b）
    Aa = np.block([
        [A, np.zeros((n, n)), B],
        [np.zeros((n, n)), AD, np.zeros((n, p))],
        [np.zeros((p, n)), np.zeros((p, n)), np.eye(p)]
    ])
    # 构建增广矩阵Ba，参考式（4.5.24b）
    Ba = np.block([
        [B],
        [np.zeros((n, p))],
        [np.eye(p)]
    ])
    # 构建增广矩阵Qa，参考式（4.5.26）
    Qa = Ca.T @ Q @ Ca
    # 构建增广矩阵Sa，参考式（4.5.27）
    Sa = Ca.T @ S @ Ca

    return Aa, Ba, Qa, Sa, R
