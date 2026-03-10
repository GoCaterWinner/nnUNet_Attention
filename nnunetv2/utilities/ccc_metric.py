"""
CCC (Concordance Correlation Coefficient) 一致性相关系数计算工具。

CCC 衡量预测值与真实值之间的一致性，值域 [-1, 1]，越接近 1 表示一致性越好。
公式：CCC = 2 * ρ * σ_x * σ_y / (σ_x² + σ_y² + (μ_x - μ_y)²)

在脂肪分割任务中，CCC 用于评估预测体积（体素数）与真实体积的一致性，
临床建议 CCC > 0.9 表示体积预测可接受。
"""
import numpy as np


def compute_ccc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算一致性相关系数 (CCC)。

    参数:
        y_true: 真实值数组 (1-D)
        y_pred: 预测值数组 (1-D)

    返回:
        CCC 值 (float)，范围 [-1, 1]
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

    if len(y_true) < 2:
        return float('nan')

    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    covariance = np.mean((y_true - mean_true) * (y_pred - mean_pred))

    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2
    if denominator < 1e-12:
        return float('nan')

    ccc = (2.0 * covariance) / denominator
    return float(ccc)
