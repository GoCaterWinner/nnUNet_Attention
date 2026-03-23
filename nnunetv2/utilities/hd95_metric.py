from typing import Sequence

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure


def _normalize_mask_and_spacing(mask: np.ndarray, spacing: Sequence[float]) -> tuple[np.ndarray, tuple[float, ...]]:
    mask = np.asarray(mask, dtype=bool)
    spacing = tuple(float(i) for i in spacing)

    # read_seg returns segmentations with a leading channel axis. Remove it if present.
    if mask.ndim >= 4 and mask.shape[0] == 1:
        mask = mask[0]

    # 2D nnU-Net segmentations often keep a dummy singleton axis like (1, H, W).
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]
        spacing = spacing[-2:]
    elif len(spacing) > mask.ndim:
        spacing = spacing[-mask.ndim:]

    if mask.ndim not in (2, 3):
        raise ValueError(f"compute_hd95 only supports 2D/3D masks after normalization, got shape {mask.shape}")

    if len(spacing) != mask.ndim:
        raise ValueError(f"Spacing length {len(spacing)} does not match mask ndim {mask.ndim}")

    return mask, spacing


def _physical_diagonal(shape: Sequence[int], spacing: Sequence[float]) -> float:
    return float(np.sqrt(np.sum((np.asarray(shape, dtype=np.float64) * np.asarray(spacing, dtype=np.float64)) ** 2)))


def compute_hd95(mask_ref: np.ndarray, mask_pred: np.ndarray, spacing: Sequence[float]) -> float:
    """
    Compute the 95th percentile Hausdorff distance (HD95) in physical units.

    Parameters
    ----------
    mask_ref:
        Reference binary mask. Leading singleton channel axes are tolerated.

    mask_pred:
        Predicted binary mask. Leading singleton channel axes are tolerated.

    spacing:
        Physical voxel spacing that matches the spatial axis order of the mask.
        For 2D nnU-Net segmentations, `(999, sy, sx)` and `(1, H, W)` style dummy axes are handled automatically.

    Returns
    -------
    float:
        HD95 in physical units. Uses the following stable empty-mask policy:
        - both masks empty -> 0.0
        - exactly one mask empty -> physical diagonal length of the sample
    """
    mask_ref, spacing = _normalize_mask_and_spacing(mask_ref, spacing)
    mask_pred, _ = _normalize_mask_and_spacing(mask_pred, spacing)

    if mask_ref.shape != mask_pred.shape:
        raise ValueError(f"mask_ref and mask_pred shape mismatch: {mask_ref.shape} vs {mask_pred.shape}")

    ref_empty = not np.any(mask_ref)
    pred_empty = not np.any(mask_pred)
    if ref_empty and pred_empty:
        return 0.0
    if ref_empty or pred_empty:
        return _physical_diagonal(mask_ref.shape, spacing)

    structure = generate_binary_structure(mask_ref.ndim, 1)
    ref_surface = np.logical_xor(mask_ref, binary_erosion(mask_ref, structure=structure, border_value=0))
    pred_surface = np.logical_xor(mask_pred, binary_erosion(mask_pred, structure=structure, border_value=0))

    if not np.any(ref_surface):
        ref_surface = mask_ref
    if not np.any(pred_surface):
        pred_surface = mask_pred

    distances_to_ref = distance_transform_edt(~ref_surface, sampling=spacing)[pred_surface]
    distances_to_pred = distance_transform_edt(~pred_surface, sampling=spacing)[ref_surface]
    all_surface_distances = np.concatenate((distances_to_ref, distances_to_pred), axis=0)
    return float(np.percentile(all_surface_distances, 95))
