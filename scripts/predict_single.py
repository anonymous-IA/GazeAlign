"""
predict_single.py — GazeAlign single-image inference CLI + Python API

Usage (CLI):
    python scripts/predict_single.py \
        --image  examples/images/0c1c6a70-a96f5b27-5d042944-6b49b3b3-fd6a8293.jpg \
        --fixations examples/fixations/fixations.csv \
        --output outputs/output_mask.png \
        --preset cxr

Usage (Python API):
    from scripts.predict_single import GazeAlignPredictor

    predictor = GazeAlignPredictor.from_preset("cxr", presets_path="configs/presets.yaml")
    result = predictor.predict(
        "examples/images/0c1c6a70-a96f5b27-5d042944-6b49b3b3-fd6a8293.jpg",
        "examples/fixations/fixations.csv",
    )
    print(result.predicted_class)
    print(result.class_probs)
"""

from __future__ import annotations

import sys
import os

# Make the repo root importable so "import GazeAlign" works without pip install
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import yaml


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class PredictionResult:
    """Holds all outputs from a single GazeAlign inference pass."""

    predicted_class: str
    """The top-1 class label string."""

    class_probs: Dict[str, float]
    """Softmax probability for every class, keyed by label string."""

    attention_mask: np.ndarray
    """Learned gaze-conditioned attention mask, shape [H, W], values in [0, 1]."""

    gaze_prior: np.ndarray
    """Raw fixation Gaussian-splat heatmap, shape [H, W], values in [0, 1]."""


# ---------------------------------------------------------------------------
# Predictor class
# ---------------------------------------------------------------------------

class GazeAlignPredictor:
    """
    High-level wrapper for GazeAlign single-image inference.

    Instantiate via :meth:`from_preset` (recommended) or :meth:`__init__`
    when you already have a loaded model and class list.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        classes: list[str],
        device: str = "cpu",
        img_size: int = 518,
    ):
        from GazeAlign.datasets import build_transform  # noqa: PLC0415

        self.model     = model.eval().to(device)
        self.classes   = classes
        self.device    = device
        self.img_size  = img_size
        # Same preprocessing used during training (resize + grayscale->3ch),
        # so single-image inference matches how the checkpoint was trained.
        self.transform = build_transform(img_size)

    # Expose the underlying sub-modules directly (used by the HF Space demo
    # and any code that wants to run the pipeline stage-by-stage).
    @property
    def image_encoder(self):
        return self.model.image_encoder

    @property
    def scanpath_encoder(self):
        return self.model.scanpath_encoder

    @property
    def mask_generator(self):
        return self.model.mask_generator

    @property
    def classifier(self):
        return self.model.classifier

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_preset(
        cls,
        preset_name: str,
        presets_path: str | Path = "configs/presets.yaml",
        device: Optional[str] = None,
    ) -> "GazeAlignPredictor":
        """
        Build a predictor from a named preset in *presets_path*.

        The YAML file must have a top-level key matching *preset_name*,
        with sub-keys ``checkpoint`` (path to .pth) and ``classes`` (list).

        Parameters
        ----------
        preset_name:
            Key in presets.yaml, e.g. ``"cxr"``.
        presets_path:
            Path to ``configs/presets.yaml``.
        device:
            ``"cuda"``, ``"cpu"``, or ``None`` (auto-detect).
        """
        presets_path = Path(presets_path)
        if not presets_path.exists():
            raise FileNotFoundError(f"Presets file not found: {presets_path}")

        with open(presets_path) as f:
            raw = yaml.safe_load(f)

        # Support both flat { cxr: {...} } and nested { presets: { cxr: {...} } }
        presets = raw.get("presets", raw)

        if preset_name not in presets:
            available = list(presets.keys())
            raise KeyError(
                f"Preset '{preset_name}' not found in {presets_path}. "
                f"Available presets: {available}"
            )

        cfg           = presets[preset_name]
        ckpt_path     = cfg["checkpoint"]
        classes       = cfg["classes"]
        img_size      = cfg.get("img_size", 518)
        grid_size     = cfg.get("grid_size", 37)
        backbone_name = cfg.get("backbone_name", "microsoft/rad-dino-maira-2")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        model = cls._load_model(
            ckpt_path,
            num_classes=len(classes),
            grid_size=grid_size,
            backbone_name=backbone_name,
            device=device,
        )
        return cls(model=model, classes=classes, device=device, img_size=img_size)

    @staticmethod
    def _load_model(
        ckpt_path: str | Path,
        num_classes: int,
        grid_size: int,
        backbone_name: str,
        device: str,
    ) -> torch.nn.Module:
        """Rebuild the five GazeAlign sub-modules and load a training checkpoint.

        Checkpoints written by ``scripts/train.py`` are wrapped dicts with
        one state-dict per sub-module (``image_encoder_state`` etc.), so we
        construct each module first and load them individually — mirroring
        ``scripts/run_eval.py``.
        """
        # Lazy imports so the module works even before GazeAlign is installed
        from GazeAlign.engine import GazeAlignModel  # noqa: PLC0415
        from GazeAlign.backbone import DINOv3Encoder  # noqa: PLC0415
        from GazeAlign.model import (  # noqa: PLC0415
            ScanpathTransformer,
            ViTMaskGenerator,
            classifier,
        )
        from GazeAlign.utils import (  # noqa: PLC0415
            load_gaze_checkpoint,
            load_submodule_state,
        )

        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt_path}\n"
                f"Download or train a model first (see scripts/train.py)."
            )

        image_encoder    = DINOv3Encoder(backbone_name=backbone_name)
        scanpath_encoder = ScanpathTransformer()
        mask_generator   = ViTMaskGenerator(num_patches=grid_size ** 2)
        clf              = classifier(num_classes=num_classes)

        # Normalizes both refactored and original research-script checkpoints.
        ckpt = load_gaze_checkpoint(ckpt_path, map_location=device)
        load_submodule_state(image_encoder, ckpt["image_encoder"])
        load_submodule_state(scanpath_encoder, ckpt["scanpath_encoder"])
        load_submodule_state(mask_generator, ckpt["mask_generator"])
        load_submodule_state(clf, ckpt["classifier"])
        if "epoch" in ckpt:
            print(f"[GazeAlign] Loaded checkpoint from epoch {ckpt['epoch']}")

        model = GazeAlignModel(
            image_encoder=image_encoder,
            scanpath_encoder=scanpath_encoder,
            mask_generator=mask_generator,
            classifier=clf,
            num_classes=num_classes,
            grid_size=grid_size,
        )
        return model

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        image_path: str | Path,
        fixations_csv: str | Path,
        dicom_id: Optional[str] = None,
    ) -> PredictionResult:
        """
        Run a full GazeAlign forward pass on one image + its fixation CSV.

        Parameters
        ----------
        image_path:
            Path to the input image (JPEG, PNG, …).
        fixations_csv:
            Path to a MIMIC-style fixations CSV (columns ``DICOM_ID``,
            ``X_ORIGINAL``, ``Y_ORIGINAL``, ``Time (in secs)``).
        dicom_id:
            Which image's scanpath to use from the CSV. Defaults to the
            stem of *image_path*; if that id is absent and the CSV holds a
            single ``DICOM_ID``, that one is used automatically.

        Returns
        -------
        PredictionResult
        """
        from GazeAlign.constants import ID_COL  # noqa: PLC0415
        from GazeAlign.gaze import (  # noqa: PLC0415
            fixation_heatmap,
            get_scanpath,
            load_fixation_csv,
        )

        image_path    = Path(image_path)
        fixations_csv = Path(fixations_csv)

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        if not fixations_csv.exists():
            raise FileNotFoundError(f"Fixations CSV not found: {fixations_csv}")

        # 1. Load + preprocess image (also gives us original pixel dims for
        #    normalizing the fixation coordinates the way training did).
        img_tensor, orig_hw = self._load_image(image_path)
        orig_h, orig_w = orig_hw

        # 2. Resolve which scanpath (DICOM_ID) to use.
        df = load_fixation_csv(str(fixations_csv))
        ids = set(df[ID_COL].astype(str))
        if dicom_id is None:
            dicom_id = image_path.stem
        if dicom_id not in ids:
            if len(ids) == 1:
                dicom_id = next(iter(ids))
            else:
                raise ValueError(
                    f"DICOM_ID '{dicom_id}' not found in {fixations_csv}. "
                    f"Available ids: {sorted(ids)}. "
                    f"Pass --dicom_id to select one."
                )

        # 3. Build the normalized [T, 3] scanpath the model was trained on.
        scanpath = get_scanpath(df, dicom_id, img_height=orig_h, img_width=orig_w)
        if scanpath is None or scanpath.numel() == 0:
            raise ValueError(f"No fixations for DICOM_ID '{dicom_id}' in {fixations_csv}.")
        scanpath = scanpath[:200].to(self.device)  # match training truncation

        # 4. Raw gaze-prior heatmap (visualization only, not used by the model).
        fix_df = df[df[ID_COL].astype(str) == str(dicom_id)]
        gaze_prior = fixation_heatmap(
            fix_df, orig_h, orig_w, orig_h, orig_w
        )  # np.ndarray [H, W] in [0, 1]

        # 5. Forward pass through the sub-modules (single-image inference: no
        #    negatives / contrastive terms — just image + scanpath -> mask ->
        #    gaze-masked classification, mirroring engine.forward_batch).
        self.model.eval()
        with torch.no_grad():
            _, patch_tokens, _ = self.model.image_encoder(
                img_tensor.unsqueeze(0).to(self.device)
            )
            _, sp_emb, _ = self.model.scanpath_encoder([scanpath])
            patch_mask = torch.sigmoid(self.model.mask_generator(sp_emb))  # [1, g, g]

            weights_pos = patch_mask.view(1, -1, 1)
            feat_attended = (patch_tokens * weights_pos).mean(dim=1)
            logits = self.model.classifier(feat_attended)  # [1, num_classes]

        probs = torch.softmax(logits[0], dim=-1).cpu().numpy()
        top_idx = int(probs.argmax())

        attention_mask = patch_mask[0].cpu().numpy()  # [grid, grid]
        # Normalise to [0, 1] for display
        mn, mx = attention_mask.min(), attention_mask.max()
        if mx > mn:
            attention_mask = (attention_mask - mn) / (mx - mn)

        return PredictionResult(
            predicted_class=self.classes[top_idx],
            class_probs={cls: float(p) for cls, p in zip(self.classes, probs)},
            attention_mask=attention_mask,
            gaze_prior=gaze_prior,
        )

    # ------------------------------------------------------------------
    # Image loading
    # ------------------------------------------------------------------

    def _load_image(self, image_path: Path):
        """Return (preprocessed tensor [3, H, W], (orig_h, orig_w)).

        Uses the shared ``build_transform`` preprocessing (resize +
        grayscale-replicated-to-3-channel) so inference matches training.
        """
        from PIL import Image

        img = Image.open(image_path).convert("RGB")
        orig_hw = (img.height, img.width)
        return self.transform(img), orig_hw


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _save_mask(mask: np.ndarray, path: Path) -> None:
    from PIL import Image
    arr = (mask * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(path)


def _save_overlay(
    image_path: Path,
    mask: np.ndarray,
    out_path: Path,
    alpha: float = 0.5,
    colormap: str = "jet",
) -> None:
    import matplotlib.cm as cm
    from PIL import Image

    img  = Image.open(image_path).convert("RGB")
    mask_resized = np.array(
        Image.fromarray((mask * 255).astype(np.uint8)).resize(
            (img.width, img.height), Image.BILINEAR
        )
    ) / 255.0

    cmap   = cm.get_cmap(colormap)
    heatmap = (cmap(mask_resized)[:, :, :3] * 255).astype(np.uint8)
    blended = Image.blend(img, Image.fromarray(heatmap), alpha=alpha)
    blended.save(out_path)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GazeAlign — single-image gaze-conditioned classification"
    )
    p.add_argument(
        "--image",
        default="examples/images/0c1c6a70-a96f5b27-5d042944-6b49b3b3-fd6a8293.jpg",
        help="Path to the input chest X-ray (or other medical image).",
    )
    p.add_argument(
        "--fixations",
        default="examples/fixations/fixations.csv",
        help=(
            "Path to a MIMIC-style fixations CSV with columns: "
            "DICOM_ID, X_ORIGINAL, Y_ORIGINAL, Time (in secs). "
            "The scanpath is selected by --dicom_id (default: stem of --image)."
        ),
    )
    p.add_argument(
        "--dicom_id",
        default=None,
        help=(
            "DICOM_ID whose scanpath to use from the fixations CSV. "
            "Defaults to the stem of --image; falls back to the sole id if "
            "the CSV contains only one."
        ),
    )
    p.add_argument(
        "--output",
        default="outputs/output_mask.png",
        help="Where to save the learned gaze-conditioned attention mask.",
    )
    p.add_argument(
        "--preset",
        default="cxr",
        help="Preset name in configs/presets.yaml (checkpoint + class list).",
    )
    p.add_argument(
        "--presets_path",
        default="configs/presets.yaml",
        help="Path to the presets YAML file.",
    )
    p.add_argument(
        "--device",
        default=None,
        help="'cuda' or 'cpu'. Defaults to cuda if available.",
    )
    p.add_argument(
        "--save_overlay",
        action="store_true",
        help=(
            "Also save a colour overlay (output_mask_overlay.png) "
            "and the raw gaze prior (output_mask_gaze_prior.png)."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[GazeAlign] Loading preset '{args.preset}' from {args.presets_path} …")
    predictor = GazeAlignPredictor.from_preset(
        args.preset,
        presets_path=args.presets_path,
        device=args.device,
    )

    image_stem = Path(args.image).stem
    print(f"[GazeAlign] Running inference on '{image_stem}' …")
    result = predictor.predict(args.image, args.fixations, dicom_id=args.dicom_id)

    # --- Print results ---
    print(f"\n  Predicted class : {result.predicted_class}")
    print("  Class probabilities:")
    for cls, prob in sorted(result.class_probs.items(), key=lambda x: -x[1]):
        bar = "█" * int(prob * 40)
        print(f"    {cls:>12s}  {prob:.4f}  {bar}")

    # --- Save attention mask ---
    _save_mask(result.attention_mask, out_path)
    print(f"\n  Saved attention mask → {out_path}")

    if args.save_overlay:
        overlay_path    = out_path.with_name(out_path.stem + "_overlay" + out_path.suffix)
        gaze_prior_path = out_path.with_name(out_path.stem + "_gaze_prior" + out_path.suffix)

        _save_overlay(args.image, result.attention_mask, overlay_path)
        _save_mask(result.gaze_prior, gaze_prior_path)

        print(f"  Saved colour overlay  → {overlay_path}")
        print(f"  Saved gaze prior      → {gaze_prior_path}")

    print("\n[GazeAlign] Done.")


if __name__ == "__main__":
    main()
