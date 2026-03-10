"""
CCC（一致性相关系数 / Concordance Correlation Coefficient）体积指标工具模块
用于脂肪分割任务中预测体积 vs. 真实体积的准确性评估

CCC 公式:
    CCC = 2 * ρ * σ_x * σ_y / (σ_x² + σ_y² + (μ_x - μ_y)²)

CCC 取值范围 [-1, 1]：
    - 1.0  = 预测体积与真实体积完全吻合（理想）
    - >0.9 = 临床上可接受的一致性
    - <0.8 = 一致性较差，需要检查分割质量
"""

from typing import Optional
import numpy as np


def compute_ccc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算一致性相关系数（CCC）。

    Args:
        y_true: 真实体积数组，shape (N,)，N 为样本数量（必须 >= 2）
        y_pred: 预测体积数组，shape (N,)

    Returns:
        float: CCC 值，范围 [-1, 1]。若样本数 < 2 或方差为零，返回 nan。

    Example:
        >>> y_true = np.array([100., 200., 300., 400.])
        >>> y_pred = np.array([105., 195., 305., 395.])
        >>> ccc = compute_ccc(y_true, y_pred)  # 接近 1.0
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    assert y_true.shape == y_pred.shape, \
        f"y_true 和 y_pred 形状不匹配: {y_true.shape} vs {y_pred.shape}"

    n = len(y_true)
    if n < 2:
        return float('nan')

    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)

    var_true = np.var(y_true)  # population variance
    var_pred = np.var(y_pred)

    # 防止全为同一值时除以 0
    if (var_true + var_pred) == 0:
        return float('nan')

    # Pearson 协方差（population）
    covariance = np.mean((y_true - mean_true) * (y_pred - mean_pred))

    ccc = (2.0 * covariance) / (var_true + var_pred + (mean_true - mean_pred) ** 2)
    return float(ccc)


def compute_volume_voxels(mask: np.ndarray) -> int:
    """
    计算二值分割掩码的体素数量（即体积，单位：体素）。

    如果需要 mm³，请用返回值乘以每个体素的物理体积（spacing_x * spacing_y * spacing_z）。

    Args:
        mask: 二值 numpy 数组，True / 1 代表前景（分割区域），任意形状

    Returns:
        int: 前景体素数量
    """
    return int(np.sum(mask > 0))


def compute_ccc_from_segmentations(
    seg_ref_list: list,
    seg_pred_list: list,
    label: int,
    voxel_volume_mm3: Optional[float] = None
) -> dict:
    """
    给定多个样本的参考分割和预测分割，计算某个标签类别的 CCC。

    Args:
        seg_ref_list:  list of np.ndarray，参考（GT）分割图，每个为 (H, W, D) 整数数组
        seg_pred_list: list of np.ndarray，预测分割图，每个为 (H, W, D) 整数数组
        label:         目标分割标签（如 1 代表脂肪）
        voxel_volume_mm3: 可选，每个体素的物理体积（mm³），若提供则输出 mm³ 体积

    Returns:
        dict 包含:
            - 'CCC': float，一致性相关系数
            - 'volumes_ref': list[float]，每个样本的参考体积
            - 'volumes_pred': list[float]，每个样本的预测体积
            - 'unit': 'voxels' 或 'mm3'
    """
    assert len(seg_ref_list) == len(seg_pred_list), \
        "参考分割和预测分割样本数不匹配"

    volumes_ref = []
    volumes_pred = []

    for seg_ref, seg_pred in zip(seg_ref_list, seg_pred_list):
        vol_ref = compute_volume_voxels(seg_ref == label)
        vol_pred = compute_volume_voxels(seg_pred == label)

        if voxel_volume_mm3 is not None:
            vol_ref *= voxel_volume_mm3
            vol_pred *= voxel_volume_mm3

        volumes_ref.append(float(vol_ref))
        volumes_pred.append(float(vol_pred))

    ccc_value = compute_ccc(np.array(volumes_ref), np.array(volumes_pred))

    return {
        'CCC': ccc_value,
        'volumes_ref': volumes_ref,
        'volumes_pred': volumes_pred,
        'unit': 'mm3' if voxel_volume_mm3 is not None else 'voxels'
    }
