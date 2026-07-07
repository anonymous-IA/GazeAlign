"""
Gradio demo for GazeAlign — gaze-supervised medical image classification.

Workflow
--------
1. Upload an image (JPG / PNG / BMP / TIFF / WEBP / DICOM).
2. Provide a radiologist-style scanpath in **either** of two ways:
     • click on the image to drop fixation points, or
     • upload a fixation table (.csv / .xlsx / .xls) and map its columns.
3. Run the model to get the predicted class (+ per-class probabilities)
   and the learned gaze-conditioned attention mask.

Run locally with:  python app.py
Deployed as a HuggingFace Space, this file is the entry point.
"""

from __future__ import annotations

import sys
import types
import os
from pathlib import Path

# ── 1. audioop shim (Python 3.13 removed audioop; some deps import it) ────────
if sys.version_info >= (3, 13):
    for _mod in ("audioop", "pyaudioop"):
        if _mod not in sys.modules:
            sys.modules[_mod] = types.ModuleType(_mod)

# ── 2. Patch starlette Jinja2Templates.TemplateResponse (old/new signature) ──
import starlette.templating as _st

_orig_TR = _st.Jinja2Templates.TemplateResponse


def _compat_TR(self, *args, **kwargs):
    if args and isinstance(args[0], str) and len(args) >= 2 and isinstance(args[1], dict):
        name = args[0]
        context = args[1]
        status_code = args[2] if len(args) > 2 else kwargs.get("status_code", 200)
        headers = kwargs.get("headers")
        media_type = kwargs.get("media_type")
        background = kwargs.get("background")
        template = self.get_template(name)
        return _st._TemplateResponse(
            template, context,
            status_code=status_code,
            headers=headers,
            media_type=media_type,
            background=background,
        )
    return _orig_TR(self, *args, **kwargs)


_st.Jinja2Templates.TemplateResponse = _compat_TR  # type: ignore[method-assign]

# ── 3. huggingface_hub HfFolder shim — MUST run *before* `import gradio` ──────
# Newer huggingface_hub versions removed `HfFolder`, but `gradio.oauth` does
# `from huggingface_hub import HfFolder, whoami` at import time, so importing
# gradio blows up unless we put a compatible `HfFolder` back first.
import huggingface_hub as _hfh

if not hasattr(_hfh, "HfFolder"):
    class _FakeHfFolder:
        @staticmethod
        def get_token():
            try:
                from huggingface_hub import get_token as _gt

                return _gt()
            except Exception:
                return None

        @staticmethod
        def save_token(token):
            return None

    _hfh.HfFolder = _FakeHfFolder  # type: ignore[attr-defined]
    sys.modules["huggingface_hub"].HfFolder = _FakeHfFolder  # type: ignore[assignment]

import gradio as gr

# ── 4. gradio_client schema shim (guards against bad additionalProperties) ───
try:
    import gradio_client.utils as _gcu

    _orig_inner = _gcu._json_schema_to_python_type

    def _safe_inner(schema, defs=None):
        if not isinstance(schema, dict):
            return "Any"
        if not isinstance(schema.get("additionalProperties"), dict):
            schema = {k: v for k, v in schema.items() if k != "additionalProperties"}
        return _orig_inner(schema, defs)

    _gcu._json_schema_to_python_type = _safe_inner
except Exception:
    pass

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw

# ── 5. Path setup — make the repo root importable ────────────────────────────
_here = Path(__file__).resolve().parent
for _candidate in [_here] + list(_here.parents):
    _s = str(_candidate)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from GazeAlign import get_device, get_scanpath  # noqa: E402
from GazeAlign.visualize import heatmap_to_image, make_overlay, patch_to_image  # noqa: E402
from scripts.predict_single import GazeAlignPredictor  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

PRESETS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "presets.yaml")
# Friendly label → preset key in configs/presets.yaml. Add rows here as you
# train GazeAlign on new modalities.
PRESETS = {
    "Chest X-ray — CHF / Normal / Pneumonia": "cxr",
}
POINT_COLORS = ["#ff3b30", "#ff9500", "#ffcc00", "#34c759", "#5ac8fa", "#007aff", "#af52de"]
_NO_COL = "— none —"

_UPLOAD_LABEL = "Drop / click to load  .jpg  .png  .bmp  .tif  .tiff  .webp  .dcm"
_FIXATION_LABEL = "Click to place fixations"
_FIXFILE_LABEL = "Upload fixation file (.csv / .xlsx / .xls) — optional"

_DEVICE = get_device()
_PREDICTORS: dict[str, GazeAlignPredictor] = {}


def get_predictor(preset_key: str) -> GazeAlignPredictor:
    """Lazily build & cache one predictor per preset."""
    if preset_key not in _PREDICTORS:
        _PREDICTORS[preset_key] = GazeAlignPredictor.from_preset(
            preset_key, presets_path=PRESETS_PATH, device=str(_DEVICE)
        )
    return _PREDICTORS[preset_key]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def dcm_to_pil(dcm_path: str) -> Image.Image:
    """Load a DICOM file and return an RGB PIL image."""
    import pydicom

    dcm = pydicom.dcmread(dcm_path)
    arr = dcm.pixel_array.astype(np.float32)
    arr = arr - arr.min()
    arr = arr / (arr.max() + 1e-8)
    arr = (arr * 255).astype(np.uint8)
    if arr.ndim == 2:
        return Image.fromarray(arr, mode="L").convert("RGB")
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):  # (C, H, W) → (H, W, C)
        arr = arr.transpose(1, 2, 0)
    return Image.fromarray(arr).convert("RGB")


def draw_points(image: Image.Image, points: list) -> Image.Image:
    """Overlay fixation circles + connecting saccades on a copy of `image`.

    `points`: list of (x_px, y_px, weight) in original-image pixel coords,
    `weight` in [0, 1] (relative dwell / recency, controls circle size).
    """
    if image is None:
        return None
    vis = image.convert("RGB").copy()
    draw = ImageDraw.Draw(vis)
    w, h = vis.size
    r = max(6, min(w, h) // 80)
    prev = None
    for (x_px, y_px, _weight) in points:
        color = POINT_COLORS[0]
        if prev is not None:
            draw.line([prev, (x_px, y_px)], fill=color, width=2)
        prev = (x_px, y_px)
    for i, (x_px, y_px, weight) in enumerate(points):
        color = POINT_COLORS[i % len(POINT_COLORS)]
        rad = r * (0.6 + 0.8 * float(weight))
        draw.ellipse(
            [x_px - rad, y_px - rad, x_px + rad, y_px + rad],
            outline=color, width=3,
        )
        draw.text((x_px + rad + 2, y_px - rad), str(i + 1), fill=color)
    return vis


def read_table(path: str) -> pd.DataFrame:
    """Load a .csv / .xlsx / .xls fixation file into a DataFrame."""
    ext = Path(path).suffix.lower()
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    # Sniff delimiter — eye-tracker exports are sometimes tab-separated
    # even with a .csv extension.
    return pd.read_csv(path, sep=None, engine="python")


def normalize_xy(x_vals: np.ndarray, y_vals: np.ndarray, img_w: int, img_h: int):
    """Convert X/Y column values to pixel coords for the given image size.

    Values already in [0, 1] (with rounding slack) are treated as
    normalised; otherwise they are assumed to be raw pixels and clamped to
    the image bounds.
    """
    looks_normalized = (
        np.nanmax(x_vals) <= 1.05 and np.nanmax(y_vals) <= 1.05
        and np.nanmin(x_vals) >= -0.05 and np.nanmin(y_vals) >= -0.05
    )
    if looks_normalized:
        x_px = np.clip(x_vals, 0, 1) * img_w
        y_px = np.clip(y_vals, 0, 1) * img_h
    else:
        x_px = np.clip(x_vals, 0, img_w)
        y_px = np.clip(y_vals, 0, img_h)
    return x_px, y_px


def _resolve_path(file_obj):
    """Extract a filesystem path from whatever gr.File passes."""
    if isinstance(file_obj, str):
        return file_obj
    if isinstance(file_obj, dict):
        return file_obj.get("name") or file_obj.get("path") or file_obj.get("tmp_path") or ""
    if hasattr(file_obj, "name"):
        return file_obj.name
    return ""


def _status(points: list) -> str:
    """Small feedback line so it's obvious when fixations register."""
    n = len(points) if points else 0
    if not n:
        return "_No fixations yet — click the image, or load a fixation file below._"
    return f"**{n}** fixation(s) placed."


def gaze_duration_heatmap(points, height, width, img_w, img_h, sigma=None):
    """Gaussian-splat heatmap of the fixations, each blob weighted by its
    dwell / duration (the 3rd component of each point), rendered at
    (height, width). This is the *observed* gaze heatmap — not the model's
    learned attention.

    points: list of (x_px, y_px, weight) in original-image pixels.
    """
    hm = np.zeros((height, width), dtype=np.float32)
    if not points:
        return hm
    if sigma is None:
        sigma = max(height, width) / 22.0
    sx, sy = width / max(img_w, 1), height / max(img_h, 1)
    rad = max(int(sigma * 3), 1)
    for x_px, y_px, wgt in points:
        cx, cy = int(round(x_px * sx)), int(round(y_px * sy))
        if not (0 <= cx < width and 0 <= cy < height):
            continue
        x0, x1 = max(cx - rad, 0), min(cx + rad + 1, width)
        y0, y1 = max(cy - rad, 0), min(cy + rad + 1, height)
        xv, yv = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
        g = np.exp(-((xv - cx) ** 2 + (yv - cy) ** 2) / (2.0 * sigma ** 2))
        # +0.15 floor so short-dwell fixations still register a little.
        hm[y0:y1, x0:x1] += g.astype(np.float32) * (0.15 + float(wgt))
    if hm.max() > 0:
        hm /= hm.max()
    return hm


# ─────────────────────────────────────────────────────────────────────────────
# Event handlers — image
# ─────────────────────────────────────────────────────────────────────────────


def on_file_upload(file_obj):
    """Load any image or DICOM and switch the panel to fixation-click mode."""
    _no_change = (None, [], "", gr.update(), gr.update(), gr.update(), gr.update())
    if file_obj is None:
        return _no_change

    path = _resolve_path(file_obj)
    if not path:
        gr.Warning("Could not resolve file path.")
        return _no_change

    image_name = Path(path).name
    ext = Path(path).suffix.lower()
    try:
        pil = dcm_to_pil(path) if ext == ".dcm" else Image.open(path).convert("RGB")
    except Exception as e:  # noqa: BLE001
        gr.Warning(f"Could not load file: {e}")
        return _no_change

    return (
        pil,                                                        # orig_image_state
        [],                                                         # points_state
        image_name,                                                 # image_name_state
        gr.update(visible=False),                                   # upload_zone → hide
        gr.update(value=pil, visible=True, label=_FIXATION_LABEL),  # image_panel → show
        gr.update(visible=True),                                    # delete_btn → show
        _status([]),                                                # fix_status
    )


def on_select(orig_image: Image.Image, points: list, weight: float, evt: gr.SelectData):
    """Record a fixation click in original-image pixel coords."""
    if orig_image is None:
        gr.Warning("Upload an image first.")
        return points, gr.update(), _status(points)
    x_px, y_px = float(evt.index[0]), float(evt.index[1])
    new_points = points + [(x_px, y_px, float(weight))]
    return new_points, draw_points(orig_image, new_points), _status(new_points)


def on_clear(orig_image):
    """Remove all fixations but keep the current image."""
    if orig_image is None:
        return [], gr.update(), _status([])
    return [], gr.update(value=orig_image), _status([])


def on_delete():
    """Delete the current image and return to upload mode."""
    return (
        None,                                    # orig_image_state
        [],                                      # points_state
        "",                                      # image_name_state
        gr.update(value=None, visible=True),     # upload_zone → show (reset)
        gr.update(value=None, visible=False),    # image_panel → hide
        gr.update(visible=False),                # delete_btn → hide
        _status([]),                             # fix_status
    )


# ─────────────────────────────────────────────────────────────────────────────
# Event handlers — fixation file
# ─────────────────────────────────────────────────────────────────────────────


def on_fixfile_upload(file_obj):
    """Load the fixation table and populate the column-mapping dropdowns."""
    _hide = (
        None, gr.update(visible=False),
        gr.update(choices=[], value=None), gr.update(choices=[], value=None),
        gr.update(choices=[], value=None), gr.update(choices=[], value=None),
        gr.update(visible=False),
    )
    if file_obj is None:
        return _hide

    path = _resolve_path(file_obj)
    if not path:
        gr.Warning("Could not resolve fixation file path.")
        return _hide

    try:
        df = read_table(path)
    except Exception as e:  # noqa: BLE001
        gr.Warning(f"Could not read fixation file: {e}")
        return _hide

    if df.empty or len(df.columns) == 0:
        gr.Warning("Fixation file appears to be empty.")
        return _hide

    cols = [str(c) for c in df.columns]

    def _guess(*keywords, fallback=None):
        # Priority-ordered: try each keyword across ALL columns before moving
        # to the next, so e.g. "dicom" wins over a stray "id" in "SESSION_ID".
        for k in keywords:
            for c in cols:
                if k in c.lower():
                    return c
        return fallback if fallback is not None else cols[0]

    guess_id = _guess("dicom", "image", "id", "name", "file", fallback=cols[0])
    guess_x = _guess("x_original", "x_orig", "fix_x", "pos_x", "gaze_x", "x_pixel", fallback=None)
    if guess_x is None:
        guess_x = next(
            (c for c in cols if c.lower().rstrip("_").endswith("x") and "index" not in c.lower()),
            cols[0],
        )
    guess_y = _guess("y_original", "y_orig", "fix_y", "pos_y", "gaze_y", "y_pixel", fallback=None)
    if guess_y is None:
        guess_y = next(
            (c for c in cols if c.lower().rstrip("_").endswith("y") and "index" not in c.lower()),
            cols[0],
        )

    time_choices = [_NO_COL] + cols
    guess_time = _guess("time", "secs", "duration", "dur", "timestamp", fallback=_NO_COL)

    return (
        df.to_json(),                                       # fixfile_df_state
        gr.update(visible=True),                            # mapping_row → show
        gr.update(choices=cols, value=guess_id),            # id_col_dd
        gr.update(choices=cols, value=guess_x),             # x_col_dd
        gr.update(choices=cols, value=guess_y),             # y_col_dd
        gr.update(choices=time_choices, value=guess_time),  # time_col_dd
        gr.update(visible=True),                            # apply_fix_btn → show
    )


def on_apply_fixfile(fixfile_json, id_col, x_col, y_col, time_col, orig_image, image_name):
    """Match rows to the loaded image (by filename) and load them as
    fixation points, replacing whatever points are currently set.

    If rows can't be matched by filename but the file holds a single image's
    worth of fixations, all rows are used (handy for single-image CSVs whose
    ID column doesn't match the uploaded filename)."""
    if orig_image is None:
        gr.Warning("Load an image first, then apply the fixation file.")
        return gr.update(), gr.update(), gr.update()
    if not fixfile_json:
        gr.Warning("Upload a fixation file first.")
        return gr.update(), gr.update(), gr.update()
    if not x_col or not y_col:
        gr.Warning("Pick the X and Y columns first.")
        return gr.update(), gr.update(), gr.update()

    df = pd.read_json(fixfile_json)

    sub = df
    if id_col and image_name:
        mask = df[id_col].astype(str) == image_name
        if not mask.any():
            stem = Path(image_name).stem
            mask = df[id_col].astype(str).apply(lambda v: Path(str(v)).stem) == stem
        if mask.any():
            sub = df[mask]
        elif df[id_col].nunique() > 1:
            gr.Warning(
                f"No rows match the loaded image ('{image_name}') and the file "
                f"has several ids — using ALL rows. Check the ID column."
            )

    if sub.empty:
        gr.Warning("No usable fixation rows found.")
        return gr.update(), gr.update(), gr.update()

    w, h = orig_image.size
    x_vals = sub[x_col].astype(float).to_numpy()
    y_vals = sub[y_col].astype(float).to_numpy()
    x_px, y_px = normalize_xy(x_vals, y_vals, w, h)

    if time_col and time_col != _NO_COL and time_col in sub.columns:
        t_raw = sub[time_col].astype(float).to_numpy()
        order = np.argsort(t_raw)  # chronological order
        x_px, y_px, t_raw = x_px[order], y_px[order], t_raw[order]
        # Per-fixation dwell = gap to the next fixation (last one gets the
        # median gap); normalised to [0,1] so it weights the duration heatmap.
        if len(t_raw) > 1:
            dwell = np.diff(t_raw, append=t_raw[-1] + np.median(np.diff(t_raw)))
            dwell = np.clip(dwell, 0, None)
            dmax = float(dwell.max())
            weight = dwell / dmax if dmax > 0 else np.ones_like(dwell)
        else:
            weight = np.ones(1)
    else:
        weight = np.ones(len(sub))

    new_points = [(float(xp), float(yp), float(wt)) for xp, yp, wt in zip(x_px, y_px, weight)]
    return new_points, draw_points(orig_image, new_points), _status(new_points)


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────


def run(orig_image: Image.Image, points: list, preset_name: str):
    import traceback

    if orig_image is None:
        gr.Warning("Upload an image first.")
        return None, "", None
    if not points or len(points) < 2:
        gr.Warning("Provide at least 2 fixations (click the image or load a fixation file).")
        return None, "", None

    preset_key = PRESETS[preset_name]
    try:
        predictor = get_predictor(preset_key)
    except FileNotFoundError as e:
        gr.Warning(str(e))
        return None, f"**Checkpoint not found** for preset `{preset_key}`.", None
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        gr.Warning(f"Could not load model: {e}")
        return None, "", None

    w, h = orig_image.size
    # Build a MIMIC-style scanpath dataframe. The 3rd component (weight)
    # drives a monotonically increasing time axis for the scanpath encoder.
    weights = np.asarray([p[2] for p in points], dtype=float)
    times = np.cumsum(np.clip(weights, 1e-3, None))
    df = pd.DataFrame(
        {
            "DICOM_ID": ["webdemo"] * len(points),
            "X_ORIGINAL": [p[0] for p in points],
            "Y_ORIGINAL": [p[1] for p in points],
            "Time (in secs)": times,
        }
    )

    scanpath = get_scanpath(df, "webdemo", img_height=h, img_width=w)
    if scanpath is None or scanpath.numel() == 0:
        gr.Warning("Could not build a scanpath from the fixations.")
        return None, "", None
    scanpath = scanpath[:200].to(predictor.device)

    img_tensor = predictor.transform(np.array(orig_image)).unsqueeze(0).to(predictor.device)
    try:
        with torch.no_grad():
            _, patch_tokens, _ = predictor.image_encoder(img_tensor)
            _, sp_emb, _ = predictor.scanpath_encoder([scanpath])
            patch_mask = torch.sigmoid(predictor.mask_generator(sp_emb))  # [1, g, g]

            B, N, D = patch_tokens.shape
            feat_attended = (patch_tokens * patch_mask.view(B, N, 1)).mean(dim=1)
            logits = predictor.classifier(feat_attended)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        gr.Warning(f"Prediction failed: {e}")
        return None, "", None

    class_probs = {c: float(p) for c, p in zip(predictor.classes, probs)}
    predicted_class = max(class_probs, key=class_probs.get)

    # Output visual: the *observed* gaze-fixation heatmap, each fixation
    # weighted by its dwell/duration — overlaid on the image and shown raw.
    img_size = predictor.img_size
    display_img = np.array(orig_image.resize((img_size, img_size)))
    gaze_hm = gaze_duration_heatmap(points, img_size, img_size, w, h)
    overlay = make_overlay(display_img, gaze_hm)

    prob_lines = "\n".join(
        f"- **{c}**: {p:.3f}" for c, p in sorted(class_probs.items(), key=lambda kv: -kv[1])
    )
    summary = f"### Predicted: **{predicted_class}**\n\n{prob_lines}"

    return class_probs, summary, overlay


# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────

_CSS = """
#run-btn {font-weight: 600;}
#upload-zone {min-height: 240px;}
#upload-zone .center {min-height: 220px;}
.footer-note {opacity: 0.7; font-size: 0.85rem;}
"""

with gr.Blocks(title="GazeAlign", css=_CSS) as demo:
    gr.Markdown(
        """
        # 👁️ GazeAlign — Gaze-Supervised Medical Image Classification

        **1.** Upload an image · **2.** Add fixations by *clicking* the image
        **or** *uploading a fixation table (.csv / .xlsx)* · **3.** Run the model.
        """
    )

    orig_image_state = gr.State(None)
    points_state = gr.State([])
    image_name_state = gr.State("")
    fixfile_df_state = gr.State(None)

    with gr.Row():
        # ── Left: image + fixations ──────────────────────────────────────────
        with gr.Column(scale=1):
            upload_zone = gr.File(
                label=_UPLOAD_LABEL,
                file_types=[".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".dcm"],
                type="filepath",
                elem_id="upload-zone",
            )
            # interactive=False → a pure display surface that reports click
            # coordinates via .select (an interactive Image opens an editor
            # instead and never fires reliable pixel coords).
            image_panel = gr.Image(
                label=_FIXATION_LABEL, type="pil", interactive=False,
                visible=False, height=440,
            )
            delete_btn = gr.Button("🗑  Delete image / load another", visible=False)
            fix_status = gr.Markdown("")

            # Load-from-file menu — collapsed; click the header to reveal.
            with gr.Accordion("📄  Load fixations from file", open=False):
                fixfile = gr.File(
                    label=_FIXFILE_LABEL, file_types=[".csv", ".xlsx", ".xls"], type="filepath"
                )
                with gr.Row(visible=False) as mapping_row:
                    id_col_dd = gr.Dropdown(label="ID column", choices=[])
                    x_col_dd = gr.Dropdown(label="X column", choices=[])
                    y_col_dd = gr.Dropdown(label="Y column", choices=[])
                    time_col_dd = gr.Dropdown(label="Time column (optional)", choices=[])
                apply_fix_btn = gr.Button("Apply fixation file", visible=False)

            # Duration weight + clear on one row (screenshot layout).
            with gr.Row():
                weight_slider = gr.Slider(
                    0.0, 1.0, value=1.0, step=0.05,
                    label="Fixation duration weight", scale=3,
                )
                clear_btn = gr.Button("✕ Clear fixations", scale=2)

            preset_dd = gr.Dropdown(
                choices=list(PRESETS.keys()),
                value=list(PRESETS.keys())[0],
                label="Model / modality preset",
            )
            run_btn = gr.Button("Run GazeAlign", variant="primary", elem_id="run-btn")

        # ── Right: results ───────────────────────────────────────────────────
        with gr.Column(scale=1):
            label_output = gr.Label(label="Predicted class (probabilities)", num_top_classes=5)
            summary_output = gr.Markdown()
            overlay_output = gr.Image(label="Gaze-fixation heatmap (dwell-weighted) — overlay")

    gr.Markdown(
        "<div class='footer-note'>See the "
        "<a href='https://github.com/anonymous-IA/GazeAlign'>GitHub repo</a> "
        "for training and evaluation code.</div>"
    )

    # ── wiring ──
    upload_zone.upload(
        on_file_upload,
        [upload_zone],
        [orig_image_state, points_state, image_name_state, upload_zone, image_panel, delete_btn, fix_status],
    )
    image_panel.select(
        on_select,
        [orig_image_state, points_state, weight_slider],
        [points_state, image_panel, fix_status],
    )
    clear_btn.click(on_clear, [orig_image_state], [points_state, image_panel, fix_status])
    delete_btn.click(
        on_delete,
        None,
        [orig_image_state, points_state, image_name_state, upload_zone, image_panel, delete_btn, fix_status],
    )

    fixfile.upload(
        on_fixfile_upload,
        [fixfile],
        [fixfile_df_state, mapping_row, id_col_dd, x_col_dd, y_col_dd, time_col_dd, apply_fix_btn],
    )
    apply_fix_btn.click(
        on_apply_fixfile,
        [fixfile_df_state, id_col_dd, x_col_dd, y_col_dd, time_col_dd, orig_image_state, image_name_state],
        [points_state, image_panel, fix_status],
    )

    run_btn.click(
        run,
        [orig_image_state, points_state, preset_dd],
        [label_output, summary_output, overlay_output],
    )


if __name__ == "__main__":
    demo.launch()
