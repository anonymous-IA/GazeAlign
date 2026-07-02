"""
Core GazeAlign modules: scanpath encoding, gaze-conditioned spatial masking,
and classification heads.

The pipeline at a glance:

    image  -> DINOv3Encoder        -> img_emb, patch_tokens, cls_token
    fixations -> ScanpathTransformer -> reconstructed scanpath, sp_embedding
    sp_embedding -> ViTMaskGenerator  -> learned [37x37] spatial attention mask
    (patch_tokens * mask).mean()      -> classifier -> class logits

Image and scanpath embeddings share a 512-d space and are pulled together
(matching image-scanpath pairs) / pushed apart (mismatched pairs) by a
contrastive loss, so the spatial mask the model learns to predict is
grounded in *where an expert actually looked*, not an arbitrary saliency
prior.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from .constants import EMBED_DIM, GRID_SIZE, MAX_SCANPATH_LEN, NUM_PATCHES, SCANPATH_DIM


# -----------------------------------------------------------------------------
# Positional encoding (shared by the scanpath transformer and mask generator)
# -----------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding, added to a sequence of embeddings."""

    def __init__(self, d_model: int, max_len: int = 200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


# -----------------------------------------------------------------------------
# Scanpath transformer: encodes a variable-length fixation sequence into a
# fixed-size embedding, and decodes it back out (autoencoder-style), so the
# embedding is trained to actually retain scanpath information rather than
# collapsing to a shortcut.
# -----------------------------------------------------------------------------
class ScanpathTransformer(nn.Module):
    """Encoder-decoder over (x, y, t) fixation sequences.

    ``forward(sequences)`` accepts a *list* of variable-length ``[T_i, 3]``
    tensors (one per sample in the batch) and returns:
      - ``outputs``: ``[B, max_len, 3]`` reconstructed scanpath (padded to ``max_len``)
      - ``latent``: ``[B, d_model]`` normalized scanpath embedding
      - ``true_len``: ``[B]`` original (unpadded) lengths, needed by the
        reconstruction loss to ignore padded positions
    """

    def __init__(self, input_dim: int = SCANPATH_DIM, d_model: int = EMBED_DIM, max_len: int = MAX_SCANPATH_LEN):
        super().__init__()
        self.max_len = max_len
        self.d_model = d_model

        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len + 1)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=2)

        self.latent_proj = nn.Linear(d_model * 3, d_model)

        dec_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=4, batch_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=2)

        self.query_embed = nn.Embedding(max_len, d_model)
        self.out_proj = nn.Linear(d_model, input_dim)

    def forward(self, sequences: list[torch.Tensor]):
        device = sequences[0].device
        B = len(sequences)

        true_len = torch.tensor([s.size(0) for s in sequences], device=device)
        padded = pad_sequence(sequences, batch_first=True)

        # ---- Encoder ----
        x = self.embedding(padded)
        x = self.pos_enc(x)

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)

        src_mask = torch.arange(padded.size(1), device=device)[None, :] >= true_len[:, None]
        src_mask = torch.cat([torch.zeros(B, 1, device=device, dtype=torch.bool), src_mask], dim=1)

        memory = self.encoder(x, src_key_padding_mask=src_mask)

        # ---- Pool encoder memory into a single latent vector ----
        mem_cls = memory[:, 0]
        mem_tokens = memory[:, 1:]
        token_mask = src_mask[:, 1:]

        mem_mean = mem_tokens.masked_fill(token_mask.unsqueeze(-1), 0).sum(1) / true_len[:, None]
        mem_max = mem_tokens.masked_fill(token_mask.unsqueeze(-1), float("-inf")).max(1).values

        proj = self.latent_proj(torch.cat([mem_cls, mem_mean, mem_max], dim=-1))
        latent = F.normalize(proj, dim=-1)

        # ---- Decoder: reconstruct the scanpath from the latent ----
        T = self.max_len
        queries = self.query_embed.weight[:T].unsqueeze(0).expand(B, -1, -1)
        queries = queries + latent.unsqueeze(1)
        queries = self.pos_enc(queries)

        memory_dec = latent.unsqueeze(1)
        dec_out = self.decoder(queries, memory_dec)
        outputs = self.out_proj(dec_out)

        return outputs, latent, true_len


# -----------------------------------------------------------------------------
# Gaze-conditioned spatial mask generator
# -----------------------------------------------------------------------------
class ViTMaskGenerator(nn.Module):
    """Cross-attention decoder that turns a scanpath embedding into a
    ``[GRID_SIZE, GRID_SIZE]`` spatial logit map (one value per ViT patch).

    A bank of learnable per-patch queries cross-attends to the (single-token)
    scanpath embedding, producing one logit per patch. Apply ``sigmoid`` to
    get an attention mask in [0, 1] that can directly re-weight ``patch_tokens``
    from :class:`~gazealign.backbone.DINOv3Encoder`.
    """

    def __init__(self, emb_dim: int = EMBED_DIM, num_patches: int = NUM_PATCHES, num_heads: int = 4, depth: int = 1):
        super().__init__()
        self.num_patches = num_patches
        self.emb_dim = emb_dim
        self.grid_size = int(round(num_patches**0.5))

        self.patch_queries = nn.Parameter(torch.randn(1, num_patches, emb_dim))
        self.pos_enc = PositionalEncoding(emb_dim, max_len=num_patches)

        self.layers = nn.ModuleList(
            [
                nn.TransformerDecoderLayer(
                    d_model=emb_dim, nhead=num_heads, dim_feedforward=emb_dim * 2, batch_first=True
                )
                for _ in range(depth)
            ]
        )
        self.head = nn.Linear(emb_dim, 1)

    def forward(self, scanpath_emb: torch.Tensor) -> torch.Tensor:
        """``scanpath_emb``: ``[B, emb_dim]`` -> returns ``[B, grid, grid]`` logits (pre-sigmoid)."""
        B = scanpath_emb.size(0)
        memory = scanpath_emb.unsqueeze(1)  # [B, 1, D]

        queries = self.patch_queries.expand(B, -1, -1)
        queries = self.pos_enc(queries)

        x = queries
        for layer in self.layers:
            x = layer(tgt=x, memory=memory)

        logits = self.head(x).squeeze(-1)  # [B, num_patches]
        return logits.view(B, self.grid_size, self.grid_size)


# -----------------------------------------------------------------------------
# Classification head
# -----------------------------------------------------------------------------
class GazeClassifier(nn.Module):
    """Shared trunk + one independent linear head per class.

    Each class gets its own binary logit (rather than a single softmax
    layer), trained with ``BCEWithLogitsLoss`` against one-hot labels.
    This mirrors a multi-label setup and is more forgiving for imbalanced
    medical classes, while still being read out with softmax/argmax at
    inference time for single-label prediction.
    """

    def __init__(self, num_classes: int, in_dim: int = 768, hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.class_heads = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(num_classes)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.head(x)
        return torch.cat([head(out) for head in self.class_heads], dim=-1)


# Backwards-compatible alias matching the original research-script name.
classifier = GazeClassifier


# -----------------------------------------------------------------------------
# Gaze-prior heatmap (Gaussian-splat visualization, not used in the loss)
# -----------------------------------------------------------------------------
class FixationMap(nn.Module):
    """Scatter normalized (x, y, t) fixations onto a pixel-resolution binary map."""

    def __init__(self, img_size: int):
        super().__init__()
        self.img_size = img_size

    def forward(self, scanpath: torch.Tensor) -> torch.Tensor:
        B, T, _ = scanpath.shape
        device = scanpath.device

        x, y, t = scanpath[..., 0], scanpath[..., 1], scanpath[..., 2]
        valid = torch.isfinite(x) & torch.isfinite(y) & torch.isfinite(t) & (t >= 0)

        x = (x * self.img_size).long().clamp(0, self.img_size - 1)
        y = (y * self.img_size).long().clamp(0, self.img_size - 1)

        fix_map = torch.zeros(B, 1, self.img_size, self.img_size, device=device)
        for b in range(B):
            xv, yv = x[b][valid[b]], y[b][valid[b]]
            if xv.numel() > 0:
                fix_map[b, 0, yv, xv] = 1.0
        return fix_map


class LearnableGaussian(nn.Module):
    """A 2D Gaussian blur whose bandwidth (sigma) is a learned parameter."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.kernel_size = kernel_size
        self.log_sigma = nn.Parameter(torch.tensor(2.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sigma = torch.exp(self.log_sigma)
        coords = torch.arange(self.kernel_size, device=x.device) - self.kernel_size // 2
        grid = coords[None, :] ** 2 + coords[:, None] ** 2
        kernel = torch.exp(-grid / (2 * sigma**2))
        kernel = (kernel / kernel.sum()).view(1, 1, self.kernel_size, self.kernel_size)
        return F.conv2d(x, kernel, padding=self.kernel_size // 2)


def patchify(map_full_res: torch.Tensor, patch_size: int = 14) -> torch.Tensor:
    """Average-pool a full-resolution map down to the ViT patch grid, then sigmoid."""
    patches = F.unfold(map_full_res, kernel_size=patch_size, stride=patch_size)
    patches = patches.mean(dim=1)
    grid = int(round((map_full_res.shape[-1] / patch_size)))
    heatmap = patches.view(map_full_res.shape[0], grid, grid)
    return torch.sigmoid(heatmap)


class GazePriorHeatmap(nn.Module):
    """Fixations -> Gaussian splat -> patch-grid -> Gaussian smooth.

    This is the model's *visualization-only* gaze prior: a smooth heatmap
    derived purely from fixation geometry, with no learned attention from
    the image encoder. Useful as a sanity-check overlay alongside the
    learned :class:`ViTMaskGenerator` output.
    """

    def __init__(self, img_size: int = 518, patch_size: int = 14):
        super().__init__()
        self.fixmap = FixationMap(img_size=img_size)
        self.gauss_full = LearnableGaussian(kernel_size=15)
        self.gauss_patch = LearnableGaussian(kernel_size=3)
        self.patch_size = patch_size

    def forward(self, scanpath: torch.Tensor) -> torch.Tensor:
        pos_map = self.fixmap(scanpath)
        heatmap = self.gauss_full(pos_map)
        heatmap = patchify(heatmap, patch_size=self.patch_size)
        heatmap = self.gauss_patch(heatmap.unsqueeze(1))
        B = heatmap.shape[0]
        grid = heatmap.shape[-1]
        return heatmap.view(B, grid, grid)


class FixationPixelHeatmap(nn.Module):
    """Fixations -> Gaussian splat at full pixel resolution (no patch downsampling).

    This is the map used for the human-readable overlay in
    ``predict_single.py`` -- a smooth, full-resolution "where did they look"
    visualization, as opposed to the coarse 37x37 grid used internally by
    the model.
    """

    def __init__(self, img_size: int = 518, kernel_size: int = 15):
        super().__init__()
        self.fixmap = FixationMap(img_size=img_size)
        self.gauss = LearnableGaussian(kernel_size=kernel_size)

    def forward(self, scanpath: torch.Tensor) -> torch.Tensor:
        pos_map = self.fixmap(scanpath)
        return self.gauss(pos_map)


# Backwards-compatible aliases matching the original research-script names.
ScanpathHeatmap = GazePriorHeatmap
ScanpathHeatmap1 = FixationPixelHeatmap
