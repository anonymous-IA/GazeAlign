"""
Shared forward-pass / loss-computation logic used by both the training and
validation loops in ``scripts/train.py``.

The original research script duplicated this ~150-line block almost
verbatim between its train and val sections (the only differences being
``.train()``/``.eval()``, ``torch.no_grad()``, and whether ``.backward()``
was called). Factoring it into ``GazeAlignModel.forward_batch`` removes
that duplication and makes the two loops trivial to keep in sync.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from .losses import class_loss_inv, contrastive_loss, mask_consistency_loss, multi_match_loss
from .metrics import compute_fixation_metrics


@dataclass
class BatchOutputs:
    """Everything a training/validation step needs to log or visualize."""

    loss: torch.Tensor
    scan_loss: torch.Tensor
    cls_loss_main: torch.Tensor
    cls_loss_aux: torch.Tensor
    inv_mask_loss: torch.Tensor
    mask_loss: torch.Tensor
    contrast_loss: torch.Tensor
    sim_pos: torch.Tensor
    sim_neg: torch.Tensor
    class_logits: torch.Tensor          # [B, num_classes] from gaze-masked features
    aux_logits: torch.Tensor            # [B, num_classes] from raw [CLS] token
    fixation_metrics: tuple             # (acc, auc, precision, recall)
    patch_mask: torch.Tensor            # [B, grid, grid] learned attention mask (post-sigmoid)
    cls_to_patch_sim: torch.Tensor      # [B, grid, grid] cosine-sim(patch, cls) map


class GazeAlignModel(nn.Module):
    """Bundles the five GazeAlign sub-modules and exposes a single
    ``forward_batch`` entry point that runs the full pipeline: image
    encoding, scanpath encoding, negative sampling, mask generation, and
    every loss term -- shared verbatim between training and validation.
    """

    def __init__(
        self,
        image_encoder: nn.Module,
        scanpath_encoder: nn.Module,
        mask_generator: nn.Module,
        classifier: nn.Module,
        num_classes: int,
        grid_size: int,
        num_negatives: int = 5,
        loss_weights: dict | None = None,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.scanpath_encoder = scanpath_encoder
        self.mask_generator = mask_generator
        self.classifier = classifier

        self.num_classes = num_classes
        self.grid_size = grid_size
        self.num_negatives = num_negatives

        self.loss_weights = loss_weights or {
            "scan_loss": 0.5,
            "cls_loss_main": 0.8,
            "cls_loss_aux": 0.2,
            "inv_mask_loss": 0.3,
            "mask_consistency_loss": 0.3,
            "contrast_loss": 0.4,
        }

        self.bce = nn.BCEWithLogitsLoss()

    def sample_negatives(self, scanpath_pool: list[dict], dicom_ids: list[str], k: int, device) -> torch.Tensor:
        """For each sample in the batch, draw ``k`` scanpaths belonging to
        *different* images from ``scanpath_pool``, encode them, and stack
        into ``[B, k, D]``.
        """
        neg_emb_list = []
        for did in dicom_ids:
            candidates = [s for s in scanpath_pool if s["dicom_id"] != did]
            selected = random.sample(candidates, k)
            neg_scanpaths = [s["scanpath"][:200].to(device) for s in selected]
            _, emb_neg, _ = self.scanpath_encoder(neg_scanpaths)
            neg_emb_list.append(emb_neg)
        return torch.stack(neg_emb_list)

    def forward_batch(
        self,
        imgs: torch.Tensor,
        scanpaths: list[torch.Tensor],
        labels: torch.Tensor,
        dicom_ids: list[str],
        negative_pool: list[dict],
    ) -> BatchOutputs:
        device = imgs.device
        B = imgs.size(0)
        K = self.num_negatives

        labels_onehot = F.one_hot(labels.long(), num_classes=self.num_classes).float()

        # ---- Image encoding ----
        img_emb, patch_tokens, cls_token = self.image_encoder(imgs)

        # ---- Scanpath encoding (positive + negatives) ----
        pos_rec, sp_pos_emb, true_len = self.scanpath_encoder(scanpaths)
        neg_emb = self.sample_negatives(negative_pool, dicom_ids, K, device)

        # ---- cls-token vs patch-token similarity map (encoder's own saliency) ----
        N = patch_tokens.shape[1]
        cls_exp = cls_token.unsqueeze(1).expand(-1, N, -1)
        sim = F.cosine_similarity(F.normalize(patch_tokens, dim=-1), F.normalize(cls_exp, dim=-1), dim=2)
        cls_to_patch_sim = ((sim + 1) / 2).view(B, self.grid_size, self.grid_size)

        # ---- Gaze-conditioned attention mask ----
        patch_mask = torch.sigmoid(self.mask_generator(sp_pos_emb))  # [B, grid, grid]

        neg_masks = torch.stack(
            [torch.sigmoid(self.mask_generator(neg_emb[:, k, :])) for k in range(K)], dim=1
        ).unsqueeze(2)  # [B, K, 1, grid, grid]

        mask_loss = mask_consistency_loss(patch_mask.unsqueeze(1), neg_masks)

        # ---- Gaze-masked classification (main signal) ----
        weights_pos = patch_mask.view(B, -1, 1)
        weights_inv = (1.0 - patch_mask).view(B, -1, 1)

        feat_attended = (patch_tokens * weights_pos).mean(dim=1)
        feat_unattended = (patch_tokens * weights_inv).mean(dim=1)

        class_logits = self.classifier(feat_attended)
        aux_unattended_logits = F.log_softmax(self.classifier(feat_unattended), dim=1)
        inv_mask_loss = class_loss_inv(aux_unattended_logits, labels)

        # ---- Raw [CLS]-token classification (auxiliary signal) ----
        aux_logits = self.classifier(cls_token)

        cls_loss_main = self.bce(class_logits, labels_onehot)
        cls_loss_aux = self.bce(aux_logits, labels_onehot)

        # ---- Image <-> scanpath contrastive alignment ----
        img_emb_norm = F.normalize(img_emb, dim=-1)
        sim_pos = ((F.cosine_similarity(img_emb_norm, sp_pos_emb) + 1) / 2)
        sim_neg = ((F.cosine_similarity(img_emb_norm.unsqueeze(1), neg_emb) + 1) / 2).mean(dim=1)

        contrast_loss = contrastive_loss(img_emb_norm.unsqueeze(1), sp_pos_emb.unsqueeze(1), neg_emb)

        # ---- Scanpath reconstruction ----
        scan_loss = multi_match_loss(pos_rec, scanpaths, true_len)

        # ---- Fixation-discrimination diagnostic ----
        candidates = torch.cat([sp_pos_emb.unsqueeze(1), neg_emb], dim=1)  # [B, 1+K, D]
        fixation_logits = torch.sum(img_emb_norm.unsqueeze(1) * candidates, dim=-1)
        fixation_metrics = compute_fixation_metrics(fixation_logits)

        w = self.loss_weights
        loss = (
            w["scan_loss"] * scan_loss
            + w["cls_loss_main"] * cls_loss_main
            + w["cls_loss_aux"] * cls_loss_aux
            + w["inv_mask_loss"] * inv_mask_loss
            + w["mask_consistency_loss"] * mask_loss
            + w["contrast_loss"] * contrast_loss
        )

        return BatchOutputs(
            loss=loss,
            scan_loss=scan_loss,
            cls_loss_main=cls_loss_main,
            cls_loss_aux=cls_loss_aux,
            inv_mask_loss=inv_mask_loss,
            mask_loss=mask_loss,
            contrast_loss=contrast_loss,
            sim_pos=sim_pos,
            sim_neg=sim_neg,
            class_logits=class_logits,
            aux_logits=aux_logits,
            fixation_metrics=fixation_metrics,
            patch_mask=patch_mask,
            cls_to_patch_sim=cls_to_patch_sim,
        )


def build_scanpath_pool(loader) -> list[dict]:
    """Materialize every scanpath in a dataloader into a flat list of
    ``{scanpath, dicom_id}`` records, used as the source pool for
    negative sampling.
    """
    pool = []
    for _, _, scanpaths, _, dicom_ids, _ in loader:
        for sp, did in zip(scanpaths, dicom_ids):
            pool.append({"scanpath": sp, "dicom_id": did})
    return pool
