#######################################
# 程序名称：F8_KalmanFilter %% [F8]线性卡尔曼滤波器
# 模块功能：求解卡尔曼滤波最优估计值
#######################################
import numpy as np


def F8_LinearKalmanFilter(A, B, Q_c, R_c, H, z, x_hat, P, u):
    """
    线性卡尔曼滤波器函数

    参数:
    A: 系统矩阵
    B: 控制输入矩阵
    Q_c: 过程噪声协方差矩阵
    R_c: 观测噪声协方差矩阵
    H: 观测矩阵
    z: 测量值
    x_hat: 上一次的后验估计值
    P: 上一次的后验估计误差协方差矩阵
    u: 控制输入

    返回:
    x_hat: 当前后验估计值
    x_hat_minus: 先验估计值
    P: 当前后验估计误差协方差矩阵
    """

    # 计算先验状态估计
    x_hat_minus = A @ x_hat + B @ u

    # 计算先验估计误差协方差矩阵
    P_minus = A @ P @ A.T + Q_c

    # 计算卡尔曼增益
    K = P_minus @ H.T @ (np.linalg.pinv(H @ P_minus @ H.T + R_c))

    # 更新后验估计
    x_hat = x_hat_minus + K.dot(z - H.dot(x_hat_minus))

    # 后验估计误差协方差矩阵
    P = (np.eye(A.shape[0]) - K.dot(H)).dot(P_minus)

    return x_hat, x_hat_minus, P
