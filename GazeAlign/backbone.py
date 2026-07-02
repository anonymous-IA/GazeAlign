"""
Image backbone: a frozen RAD-DINO / DINOv3 ViT with one trainable extra
transformer block stacked on top.

Why a frozen backbone + one extra block? Fully fine-tuning a large
pretrained ViT on a small gaze-annotated medical dataset overfits fast.
Freezing the backbone keeps its general visual features intact, while the
single trainable block (an exact clone of the backbone's last layer, with
its MLP removed) gives the model just enough capacity to adapt attention
patterns to the gaze-alignment objective.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
from transformers import AutoModel

from .constants import BACKBONE_HIDDEN_DIM, EMBED_DIM, GRID_SIZE


class DINOv3Encoder(nn.Module):
    """Frozen ViT backbone + one trainable extra block + projection head.

    Forward returns a triple:
      - ``proj``: ``[B, EMBED_DIM]`` pooled image embedding (cls + mean + max, projected)
      - ``patch_tokens``: ``[B, N, hidden_dim]`` per-patch features (N = GRID_SIZE**2)
      - ``cls_token``: ``[B, hidden_dim]`` the [CLS] token after the extra block
    """

    def __init__(self, backbone_name: str = "microsoft/rad-dino-maira-2"):
        super().__init__()

        self.backbone = AutoModel.from_pretrained(backbone_name)
        hidden_dim = self.backbone.config.hidden_size  # 768 for RAD-DINO / DINOv3-B

        # Freeze the entire pretrained backbone.
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Add one trainable ViT block, cloned from the backbone's last layer
        # so it starts from a sensible initialization rather than random
        # weights. Its MLP is dropped (kept attention-only) to limit the
        # number of new trainable parameters.
        last_block = self.backbone.encoder.layer[-1]
        self.extra_block = copy.deepcopy(last_block)
        self.extra_block.mlp = nn.Identity()
        for p in self.extra_block.parameters():
            p.requires_grad = True

        self.drop = nn.Dropout(p=0.1)
        self.proj = nn.Linear(3 * hidden_dim, EMBED_DIM)

        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor):
        with torch.no_grad():
            outputs = self.backbone(x).last_hidden_state  # [B, 1+N, D]

        outputs = self.drop(outputs)
        outputs = self.extra_block(outputs)
        if isinstance(outputs, tuple):  # some HF blocks return a tuple
            outputs = outputs[0]

        patch_tokens = outputs[:, 1:]
        cls_token = outputs[:, 0]

        mean_pool = patch_tokens.mean(dim=1)
        max_pool = patch_tokens.max(dim=1).values

        fused = torch.cat([cls_token, mean_pool, max_pool], dim=-1)
        proj = self.proj(fused)

        return proj, patch_tokens, cls_token

    def unfreeze_last_block(self):
        """Unfreeze the backbone's own encoder stack (optional fine-tuning stage)."""
        for p in self.backbone.encoder.parameters():
            p.requires_grad = True
