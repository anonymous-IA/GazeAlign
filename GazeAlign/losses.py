"""
Loss functions for GazeAlign.

The total training objective combines five terms (see
``gazealign.train.compute_losses`` / ``configs/*.yaml`` for the weights):

1. ``scan_loss``        (multi_match_loss)   -- scanpath reconstruction
2. ``cls_loss``          (BCE, two views)     -- classification from gaze-masked features
3. ``inv_mask_loss``     (class_loss_inv)     -- attention-mining: the *unattended* region
                                                  should NOT be predictive of the class
4. ``mask_loss``         (mask_consistency_loss) -- the learned mask should differ from
                                                  masks generated for *mismatched* scanpaths
5. ``contrast_loss``     (contrastive_loss)   -- image/scanpath embeddings: pull matching
                                                  pairs together, push mismatched pairs apart
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def contrastive_loss(anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    """Triplet loss between an image embedding (anchor), its matching
    scanpath embedding (positive), and embeddings from K mismatched
    scanpaths (negative), normalized to [0, 1].

    anchor, positive: ``[B, 1, D]`` normalized embeddings.
    negative: ``[B, K, D]`` normalized embeddings.
    """
    loss_fn = nn.TripletMarginLoss(margin=margin, p=2)
    loss = loss_fn(anchor, positive, negative)
    # Max possible distance for normalized vectors is ~sqrt(2) + margin; clamp to [0, 1].
    return torch.clamp(loss / (margin + 2**0.5), 0.0, 1.0)


def multi_match_loss(
    pred: torch.Tensor,
    target_list: list[torch.Tensor],
    true_len: torch.Tensor,
    img_size: tuple[int, int] = (518, 518),
    max_time: float = 1.0,
) -> torch.Tensor:
    """Scanpath reconstruction loss with Hungarian (optimal bipartite) matching.

    Because fixation *order* in the reconstruction is not guaranteed to
    line up with the ground-truth order, each sample's predicted and
    target fixations are matched optimally (by L1 distance in
    pixel/time space) before computing the spatial + temporal error.
    This is the same idea as DETR's matching loss, applied to gaze
    points instead of bounding boxes.

    ``pred``: ``[B, T_pred, 3]`` model output (normalized x, y, t).
    ``target_list``: list of ``[T_i, 3]`` ground-truth scanpaths (normalized).
    ``true_len``: ``[B]`` number of *real* (non-padded) fixations per sample.
    """
    device = pred.device
    B = pred.shape[0]
    W, H = img_size

    total_loss = 0.0
    for b in range(B):
        T = true_len[b]
        pred_seq = pred[b][:T]
        target_seq = target_list[b].to(device)[:T]

        pred_scaled = pred_seq.clone()
        target_scaled = target_seq.clone()

        pred_scaled[:, 0] *= W
        pred_scaled[:, 1] *= H
        pred_scaled[:, 2] /= max_time

        target_scaled[:, 0] *= W
        target_scaled[:, 1] *= H
        target_scaled[:, 2] /= max_time

        cost = torch.cdist(pred_scaled, target_scaled, p=1)
        row_ind, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())

        matched_pred = pred_scaled[row_ind]
        matched_gt = target_scaled[col_ind]

        spatial_norm = torch.abs(matched_pred[:, :2] / W - matched_gt[:, :2] / H).mean()
        temporal = torch.abs(matched_pred[:, 2] - matched_gt[:, 2]).mean()

        loss = 0.7 * spatial_norm + 0.3 * temporal
        total_loss = total_loss + loss

    return total_loss / B


def class_loss_inv(log_probs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Attention-mining / invariance loss.

    Given log-probabilities predicted from the *unattended* (1 - mask)
    region of the image, this returns the mean predicted probability of
    the true class. Minimizing it during training pushes the model to
    move all class-discriminative evidence into the gaze-attended region,
    so the learned mask becomes more faithful to what's actually relevant.
    """
    probs = log_probs.exp()
    true_class_prob = probs.gather(1, labels.view(-1, 1)).squeeze(1)
    return true_class_prob.mean()


def mask_consistency_loss(
    pos_mask: torch.Tensor,
    neg_masks: torch.Tensor,
) -> torch.Tensor:
    """Penalize the learned mask for resembling masks generated from
    *mismatched* scanpaths (soft IoU + a KL-divergence dissimilarity term).

    ``pos_mask``: ``[B, 1, H, W]`` mask from the sample's own scanpath.
    ``neg_masks``: ``[B, K, 1, H, W]`` masks from K mismatched scanpaths.
    Returns a scalar in roughly [0, 1] -- lower is better (less overlap
    with mismatched-scanpath masks).
    """
    B, K = neg_masks.shape[0], neg_masks.shape[1]

    pos_s = pos_mask.unsqueeze(1)  # [B, 1, 1, H, W]

    intersection = (pos_s * neg_masks).sum(dim=(2, 3, 4))
    union = pos_s.sum(dim=(2, 3, 4)) + neg_masks.sum(dim=(2, 3, 4)) + 1e-6
    soft_iou = intersection / union

    p = F.softmax(pos_mask.view(B, -1), dim=1).unsqueeze(1)  # [B, 1, H*W]
    q = F.softmax(neg_masks.view(B, K, -1), dim=2)  # [B, K, H*W]

    kl = F.kl_div(q.log(), p.expand(-1, K, -1), reduction="none").mean(2)
    kl_dissim = 1 - torch.exp(-kl)

    return ((soft_iou.mean(dim=1) + kl_dissim.mean(dim=1)) / 2).mean()
