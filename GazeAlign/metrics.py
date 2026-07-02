"""
Evaluation metrics for GazeAlign.

Two distinct things get measured during training/validation:

1. **Classification performance** (CHF/Normal/Pneumonia or any other class
   set): accuracy, balanced accuracy, macro precision/recall/F1, per-class
   and macro AUC. Standard multi-class metrics, computed once per epoch
   over all accumulated predictions.

2. **Fixation discrimination**: a sanity-check metric for the contrastive
   objective. The model scores the matching scanpath plus K mismatched
   ("negative") scanpaths against an image, and we check whether it
   correctly ranks the *true* scanpath highest. This isn't a metric for
   the paper's headline numbers -- it's a diagnostic for whether the
   image/scanpath embedding space is actually learning to align.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


def compute_fixation_metrics(logits: torch.Tensor):
    """Given similarity logits ``[B, 1+K]`` (index 0 = the true scanpath,
    indices 1..K = mismatched scanpaths), compute how reliably the model
    ranks the true scanpath above the mismatched ones.

    Returns ``(accuracy, auc, precision, recall)``.
    """
    probs = F.softmax(logits, dim=1)
    B, C = probs.shape

    labels = torch.zeros_like(probs)
    labels[:, 0] = 1

    preds = probs.argmax(dim=1)
    acc = (preds == 0).float().mean().item()

    y_true = labels.cpu().numpy().reshape(-1)
    y_scores = probs.detach().cpu().numpy().reshape(-1)

    y_pred_binary = np.zeros_like(y_true)
    pred_indices = preds.cpu().numpy()
    for i in range(B):
        y_pred_binary[i * C + pred_indices[i]] = 1

    try:
        auc = roc_auc_score(y_true, y_scores)
    except ValueError:
        auc = float("nan")
    try:
        precision = precision_score(y_true, y_pred_binary, zero_division=0)
    except ValueError:
        precision = float("nan")
    try:
        recall = recall_score(y_true, y_pred_binary, zero_division=0)
    except ValueError:
        recall = float("nan")

    return acc, auc, precision, recall


def compute_classification_metrics(all_logits: torch.Tensor, all_labels: torch.Tensor, num_classes: int) -> dict:
    """Full multi-class metric suite computed once over an epoch's accumulated
    predictions. Returns a flat dict so it's trivial to log / pretty-print.
    """
    all_preds = all_logits.argmax(dim=1)
    acc = (all_preds == all_labels).float().mean().item()

    labels_np = all_labels.cpu().numpy()
    preds_np = all_preds.cpu().numpy()

    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        labels_np, preds_np, average="macro", zero_division=0
    )
    bal_acc = balanced_accuracy_score(labels_np, preds_np)

    probs = torch.softmax(all_logits, dim=1).cpu().numpy()
    labels_bin = label_binarize(labels_np, classes=list(range(num_classes)))

    auc_per_class = []
    for c in range(num_classes):
        try:
            auc_per_class.append(roc_auc_score(labels_bin[:, c], probs[:, c]))
        except ValueError:
            auc_per_class.append(float("nan"))

    try:
        auc_macro = roc_auc_score(labels_bin, probs, average="macro", multi_class="ovr")
    except ValueError:
        auc_macro = float("nan")

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_macro": f1_macro,
        "auc_macro": auc_macro,
        "auc_per_class": auc_per_class,
    }


def dice_score(pred_mask: np.ndarray, gt_mask: np.ndarray, eps: float = 1e-6) -> float:
    """Dice coefficient between a predicted and ground-truth binary mask.

    Included for parity with segmentation-style evaluation (e.g. if you
    extend GazeAlign with pixel-level gaze-derived masks against an
    annotated ROI), even though the default classification pipeline does
    not require pixel-level ground truth.
    """
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    return float((2 * intersection + eps) / (pred.sum() + gt.sum() + eps))


def iou_score(pred_mask: np.ndarray, gt_mask: np.ndarray, eps: float = 1e-6) -> float:
    """Intersection-over-Union between a predicted and ground-truth binary mask."""
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float((intersection + eps) / (union + eps))
