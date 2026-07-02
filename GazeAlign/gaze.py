"""
Gaze / scanpath utilities.

A "scanpath" here is the sequence of fixations a viewer made while looking
at an image, encoded as a tensor of shape [T, 3] holding normalized
(x, y, t) triples: pixel coordinates scaled to [0, 1] by image width/height,
and timestamps scaled to [0, 1] by the scanpath's own duration.

This module covers two CSV schemas:

1. MIMIC-style (used during training): a single dataframe with columns
   ``DICOM_ID, X_ORIGINAL, Y_ORIGINAL, Time (in secs)`` holding fixations
   for *every* image, indexed by DICOM_ID.

2. Single-image CSV (used at inference time, see ``predict_single.py``):
   the same four columns, but the file may contain fixations for one or
   more images -- the caller selects which DICOM_ID / image to use.

Both are read with :func:`load_fixation_csv`, and both feed into
:func:`get_scanpath` and :func:`fixation_heatmap`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from .constants import ID_COL, TIME_COL, X_COL, Y_COL


def load_fixation_csv(path: str) -> pd.DataFrame:
    """Load a fixations CSV.

    Expected columns: ``DICOM_ID, X_ORIGINAL, Y_ORIGINAL, Time (in secs)``.
    Extra columns are ignored. Raises a clear error if required columns
    are missing, rather than failing deep inside a tensor op later.
    """
    df = pd.read_csv(path)
    required = {ID_COL, X_COL, Y_COL, TIME_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Fixation CSV '{path}' is missing required column(s): {sorted(missing)}. "
            f"Expected columns: {sorted(required)}."
        )
    return df


def get_scanpath(
    df: pd.DataFrame,
    dicom_id: str,
    img_height: int,
    img_width: int,
    time_col: str = TIME_COL,
) -> torch.Tensor | None:
    """Extract and normalize a single image's scanpath from a fixations dataframe.

    Returns a ``[T, 3]`` float tensor of (x, y, t), all normalized to
    [0, 1], sorted by time. Returns ``None`` if no fixations exist for
    ``dicom_id``.
    """
    df_img = df[df[ID_COL] == dicom_id]
    if len(df_img) == 0:
        return None

    df_img = df_img.sort_values(time_col)

    x = df_img[X_COL].values / img_width
    y = df_img[Y_COL].values / img_height
    t = df_img[time_col].values.astype(np.float64)

    t = t - t.min()
    if t.max() > 0:
        t = t / t.max()

    seq = np.stack([x, y, t], axis=1)
    return torch.tensor(seq, dtype=torch.float32)


def fixation_heatmap(
    fix_df: pd.DataFrame,
    height: int,
    width: int,
    orig_height: int,
    orig_width: int,
    time_col: str = TIME_COL,
    x_col: str = X_COL,
    y_col: str = Y_COL,
    sigma: float = 35.0,
    weighted: bool = True,
) -> np.ndarray:
    """Render fixations as a 2D Gaussian-splatted heatmap at (height, width).

    If ``weighted`` is True, each fixation's contribution is scaled by the
    dwell time elapsed since the previous fixation (longer dwell -> brighter
    blob). If False, every fixation contributes equally -- useful for a
    quick "where did they look at all" view.

    This is the *raw gaze prior* (point-wise Gaussian splats), used for
    visualization. It is intentionally simple, in contrast to the
    learned, model-produced attention used for classification.
    """
    heatmap = np.zeros((height, width), dtype=np.float32)

    if fix_df is None or len(fix_df) == 0:
        return heatmap

    scale_x = width / orig_width
    scale_y = height / orig_height

    fix_df = fix_df.sort_values(time_col).reset_index(drop=True)

    xs = fix_df[x_col].values * scale_x
    ys = fix_df[y_col].values * scale_y
    times = fix_df[time_col].values

    for i in range(len(fix_df)):
        if np.isnan(xs[i]) or np.isnan(ys[i]):
            continue

        x = int(xs[i])
        y = int(ys[i])
        if not (0 <= x < width and 0 <= y < height):
            continue

        dt = 1.0
        if weighted and i > 0:
            dt = max(times[i] - times[i - 1], 0.0)

        patch_size = max(int(sigma * 3), 1)
        x_min = max(x - patch_size, 0)
        x_max = min(x + patch_size + 1, width)
        y_min = max(y - patch_size, 0)
        y_max = min(y + patch_size + 1, height)

        xv, yv = np.meshgrid(np.arange(x_min, x_max), np.arange(y_min, y_max))
        gaussian = np.exp(-((xv - x) ** 2 + (yv - y) ** 2) / (2 * sigma**2))
        heatmap[y_min:y_max, x_min:x_max] += gaussian * dt

    if heatmap.max() > 0:
        heatmap /= heatmap.max()

    return heatmap


def scanpath_patch_heatmap(
    points: np.ndarray,
    orig_h: int,
    orig_w: int,
    target_h: int = 518,
    target_w: int = 518,
    grid_size: int = 37,
) -> np.ndarray:
    """Bin fixations into a coarse ``grid_size x grid_size`` patch grid.

    Counts how many fixations fall in each ViT patch, normalizes to
    [0, 1], and nearest-neighbor-upsamples to ``(target_h, target_w)``
    for display alongside pixel-resolution heatmaps.
    """
    import cv2

    patch_map = np.zeros((grid_size, grid_size), dtype=np.float32)
    patch_h = orig_h / grid_size
    patch_w = orig_w / grid_size

    for x, y in points:
        if np.isnan(x) or np.isnan(y):
            continue
        px = int(np.clip(x // patch_w, 0, grid_size - 1))
        py = int(np.clip(y // patch_h, 0, grid_size - 1))
        patch_map[py, px] += 1

    if patch_map.max() > 0:
        patch_map /= patch_map.max()

    return cv2.resize(patch_map, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
