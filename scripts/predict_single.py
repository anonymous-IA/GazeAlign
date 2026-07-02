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
# CSV loading helpers
# ---------------------------------------------------------------------------

# Column names expected in fixations.csv  (edit here if your CSV differs)
_COL_IMAGE_ID = "image_id"
_COL_X        = "x_pixel"
_COL_Y        = "y_pixel"
_COL_TIME     = "timestamp"
_COL_DUR      = "duration"

_REQUIRED_COLS = {_COL_IMAGE_ID, _COL_X, _COL_Y, _COL_TIME}


def _load_fixations(csv_path: str | Path, image_stem: str) -> np.ndarray:
    """
    Load fixation rows for *image_stem* from *csv_path*.

    Parameters
    ----------
    csv_path:
        Path to fixations.csv.  Must have at minimum the columns defined
        in _REQUIRED_COLS.  Extra columns are silently ignored.
    image_stem:
        The bare filename of the target image **without** its extension,
        e.g. ``"0c1c6a70-a96f5b27-5d042944-6b49b3b3-fd6a8293"``.

    Returns
    -------
    fixations : np.ndarray, shape [N, 3]
        Each row is ``[x_pixel, y_pixel, timestamp_ms]``.
        Sorted by timestamp ascending.

    Raises
    ------
    ValueError
        If the CSV is missing required columns or no rows match *image_stem*.
    """
    import pandas as pd

    df = pd.read_csv(csv_path)

    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"fixations.csv is missing required column(s): {missing}\n"
            f"Found columns: {list(df.columns)}"
        )

    # Filter to this image
    mask = df[_COL_IMAGE_ID].astype(str) == image_stem
    sub = df[mask].copy()

    if sub.empty:
        raise ValueError(
            f"No fixation rows found for image_id='{image_stem}' in {csv_path}.\n"
            f"Check that the image_id column matches the image filename stem exactly."
        )

    sub = sub.sort_values(_COL_TIME)
    fixations = sub[[_COL_X, _COL_Y, _COL_TIME]].to_numpy(dtype=np.float32)
    return fixations


# ---------------------------------------------------------------------------
# Predictor class
# ---------------------------------------------------------------------------

class GazeAlignPredictor:
    """
    High-level wrapper for GazeAlign single-image inference.

    Instantiate via :meth:`from_preset` (recommended) or :meth:`__init__`
    when you already have a loaded model and class list.
    """

    def __init__(self, model: torch.nn.Module, classes: list[str], device: str = "cpu"):
        self.model   = model.eval().to(device)
        self.classes = classes
        self.device  = device

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

        cfg        = presets[preset_name]
        ckpt_path  = cfg["checkpoint"]
        classes    = cfg["classes"]

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        model = cls._load_model(ckpt_path, num_classes=len(classes), device=device)
        return cls(model=model, classes=classes, device=device)

    @staticmethod
    def _load_model(
        ckpt_path: str | Path, num_classes: int, device: str
    ) -> torch.nn.Module:
        """Load GazeAlignModel from a checkpoint file."""
        # Lazy import so the module works even before GazeAlign is installed
        from GazeAlign.engine import GazeAlignModel  # noqa: PLC0415

        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt_path}\n"
                f"Download or train a model first (see scripts/train.py)."
            )

        model = GazeAlignModel(num_classes=num_classes)
        state = torch.load(ckpt_path, map_location=device)
        # Support checkpoints saved as plain state-dicts or wrapped dicts
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)
        return model

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        image_path: str | Path,
        fixations_csv: str | Path,
    ) -> PredictionResult:
        """
        Run a full GazeAlign forward pass on one image + its fixation CSV.

        Parameters
        ----------
        image_path:
            Path to the input image (JPEG, PNG, …).
        fixations_csv:
            Path to a fixations CSV.  Rows are filtered to the stem of
            *image_path* via the ``image_id`` column.

        Returns
        -------
        PredictionResult
        """
        from GazeAlign.gaze import build_gaze_prior  # noqa: PLC0415

        image_path    = Path(image_path)
        fixations_csv = Path(fixations_csv)

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        if not fixations_csv.exists():
            raise FileNotFoundError(f"Fixations CSV not found: {fixations_csv}")

        image_stem = image_path.stem  # e.g. "0c1c6a70-a96f5b27-5d042944-6b49b3b3-fd6a8293"

        # 1. Load + preprocess image
        img_tensor, orig_hw = self._load_image(image_path)

        # 2. Load fixations filtered to this image
        fixations = _load_fixations(fixations_csv, image_stem)  # [N, 3]

        # 3. Build raw gaze prior heatmap (for visualisation / comparison)
        gaze_prior = build_gaze_prior(
            fixations, orig_hw, sigma_px=30
        )  # np.ndarray [H, W] in [0, 1]

        # 4. Prepare fixation tensor for the model
        fix_tensor = torch.from_numpy(fixations).unsqueeze(0).to(self.device)  # [1, N, 3]

        # 5. Forward pass
        with torch.no_grad():
            logits, attn_mask = self.model(
                img_tensor.unsqueeze(0).to(self.device),
                fix_tensor,
            )

        probs = torch.softmax(logits[0], dim=-1).cpu().numpy()
        top_idx = int(probs.argmax())

        attention_mask = attn_mask[0].cpu().numpy()  # [H, W]
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
        """Return (tensor [3, H, W] normalised, (orig_h, orig_w))."""
        from PIL import Image
        from torchvision import transforms

        img = Image.open(image_path).convert("RGB")
        orig_hw = (img.height, img.width)

        tf = transforms.Compose([
            transforms.Resize((518, 518)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        return tf(img), orig_hw


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
            "Path to a fixations CSV with columns: "
            f"{_COL_IMAGE_ID}, {_COL_X}, {_COL_Y}, {_COL_TIME}. "
            "Rows are filtered by the stem of --image."
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
    result = predictor.predict(args.image, args.fixations)

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
