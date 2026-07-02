"""
Visualization helpers: training-time debug panels and inference-time overlays.
"""

from __future__ import annotations

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


def patch_to_image(heatmap: np.ndarray, height: int = 518, width: int = 518) -> np.ndarray:
    """Upsample a coarse ``[grid, grid]`` heatmap to pixel resolution and
    normalize to [0, 1].
    """
    if heatmap.ndim == 1:
        grid = int(round(len(heatmap) ** 0.5))
        heatmap = heatmap.reshape(grid, grid)
    hm = cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_NEAREST)
    hm = hm - hm.min()
    return hm / (hm.max() + 1e-8)


def overlay_saccades(img: np.ndarray, fix_df, color=(0, 255, 0), thickness: int = 2) -> np.ndarray:
    """Draw lines connecting consecutive fixations (the saccade path) on ``img``."""
    img_copy = img.copy()
    points = fix_df[["X_ORIGINAL", "Y_ORIGINAL"]].values
    for i in range(len(points) - 1):
        x1, y1 = int(points[i][0]), int(points[i][1])
        x2, y2 = int(points[i + 1][0]), int(points[i + 1][1])
        if 0 <= x1 < img.shape[1] and 0 <= y1 < img.shape[0] and 0 <= x2 < img.shape[1] and 0 <= y2 < img.shape[0]:
            cv2.line(img_copy, (x1, y1), (x2, y2), color=color, thickness=thickness)
    return img_copy


def plot_epoch_debug(
    img: torch.Tensor,
    patch_hm: np.ndarray,
    fix_weighted: np.ndarray,
    fix_unweighted: np.ndarray,
    gen_mask: torch.Tensor,
    saccade_patch_hm: np.ndarray,
    save_path: str,
):
    """6-panel debug figure saved once per epoch: original image, the
    model's learned patch attention, weighted/unweighted gaze priors, the
    binary attention mask, and the coarse saccade-density heatmap.
    """
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    img_np = img.cpu().numpy().transpose(1, 2, 0)
    axs[0, 0].imshow(img_np)
    axs[0, 0].set_title("Original Image")
    axs[0, 0].axis("off")

    axs[0, 1].imshow(patch_hm, cmap="hot")
    axs[0, 1].set_title("Patch Heatmap (DINOv3)")
    axs[0, 1].axis("off")

    axs[0, 2].imshow(fix_weighted, cmap="jet")
    axs[0, 2].set_title("Weighted Fixation Heatmap")
    axs[0, 2].axis("off")

    axs[1, 0].imshow(fix_unweighted, cmap="jet")
    axs[1, 0].set_title("Unweighted Fixation Heatmap")
    axs[1, 0].axis("off")

    axs[1, 1].imshow(gen_mask.squeeze(), cmap="gray")
    axs[1, 1].set_title("Binary Mask")
    axs[1, 1].axis("off")

    axs[1, 2].imshow(saccade_patch_hm, cmap="hot")
    axs[1, 2].set_title("Saccade Patch Heatmap")
    axs[1, 2].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def make_overlay(image_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45, colormap: int = cv2.COLORMAP_JET) -> Image.Image:
    """Blend a normalized ``[0, 1]`` heatmap over an RGB image as a colour overlay.

    Used by ``scripts/predict_single.py --save_overlay`` to produce a
    human-readable "where the model is looking + what it predicted" image.
    """
    heatmap_u8 = np.uint8(255 * np.clip(heatmap, 0, 1))
    heatmap_color = cv2.applyColorMap(heatmap_u8, colormap)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    if image_rgb.shape[:2] != heatmap_color.shape[:2]:
        heatmap_color = cv2.resize(heatmap_color, (image_rgb.shape[1], image_rgb.shape[0]))

    blended = (alpha * heatmap_color + (1 - alpha) * image_rgb).astype(np.uint8)
    return Image.fromarray(blended)


def heatmap_to_image(heatmap: np.ndarray, colormap: int = cv2.COLORMAP_JET) -> Image.Image:
    """Render a normalized ``[0, 1]`` heatmap as a standalone colour image
    (the "gaze-prior PNG" produced alongside the overlay).
    """
    heatmap_u8 = np.uint8(255 * np.clip(heatmap, 0, 1))
    heatmap_color = cv2.applyColorMap(heatmap_u8, colormap)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    return Image.fromarray(heatmap_color)
