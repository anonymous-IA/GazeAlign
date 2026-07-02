"""
Gradio demo for GazeAlign.

Upload a chest X-ray (or other supported modality) and click on the image
to lay down fixation points (simulating a radiologist's scanpath), then
run the model to see the predicted class and the learned gaze-conditioned
attention mask.

Run locally with:
    python app.py

Deployed as a HuggingFace Space, this file is the entry point.
"""

import os
import sys
import time

import gradio as gr
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gazealign import build_transform, get_device, get_scanpath
from gazealign.visualize import heatmap_to_image, make_overlay, patch_to_image
from scripts.predict_single import GazeAlignPredictor

PRESET_NAME = os.environ.get("GAZEALIGN_PRESET", "cxr")
PRESETS_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "presets.yaml")

_predictor = None


def get_predictor():
    global _predictor
    if _predictor is None:
        _predictor = GazeAlignPredictor.from_preset(PRESET_NAME, presets_path=PRESETS_PATH, device=get_device())
    return _predictor


def add_fixation(image, click_evt: gr.SelectData, fixations_state):
    """Record a click as a new fixation point, with a synthetic timestamp."""
    if image is None:
        return image, fixations_state, "Upload an image first."

    x, y = click_evt.index
    t = 0.0 if not fixations_state else fixations_state[-1]["t"] + 0.4
    fixations_state = fixations_state + [{"x": x, "y": y, "t": t}]

    # Draw all fixations + connecting lines as a quick visual confirmation.
    import cv2

    vis = np.array(image).copy()
    pts = [(f["x"], f["y"]) for f in fixations_state]
    for i, (px, py) in enumerate(pts):
        cv2.circle(vis, (px, py), 6, (255, 0, 0), -1)
        if i > 0:
            cv2.line(vis, pts[i - 1], pts[i], (255, 0, 0), 2)

    return vis, fixations_state, f"{len(fixations_state)} fixation(s) recorded."


def reset_fixations(image):
    return image, [], "Fixations cleared."


def run_prediction(image, fixations_state):
    if image is None:
        return None, None, None, "Please upload an image first."
    if len(fixations_state) < 2:
        return None, None, None, "Click at least 2 points on the image to form a scanpath."

    predictor = get_predictor()
    img_rgb = np.array(image)
    h, w = img_rgb.shape[:2]

    df = pd.DataFrame(
        {
            "DICOM_ID": ["webdemo"] * len(fixations_state),
            "X_ORIGINAL": [f["x"] for f in fixations_state],
            "Y_ORIGINAL": [f["y"] for f in fixations_state],
            "Time (in secs)": [f["t"] for f in fixations_state],
        }
    )

    scanpath = get_scanpath(df, "webdemo", img_height=h, img_width=w)[:200].to(predictor.device)

    img_tensor = predictor.transform(img_rgb).unsqueeze(0).to(predictor.device)
    with torch.no_grad():
        img_emb, patch_tokens, cls_token = predictor.image_encoder(img_tensor)
        _, sp_emb, _ = predictor.scanpath_encoder([scanpath])
        patch_mask = torch.sigmoid(predictor.mask_generator(sp_emb))

        B, N, D = patch_tokens.shape
        feat_attended = (patch_tokens * patch_mask.view(B, N, 1)).mean(dim=1)
        logits = predictor.classifier(feat_attended)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    class_probs = {c: float(p) for c, p in zip(predictor.classes, probs)}
    predicted_class = max(class_probs, key=class_probs.get)

    attention_mask_full = patch_to_image(patch_mask[0].cpu().numpy(), predictor.img_size, predictor.img_size)
    display_img = np.array(image.resize((predictor.img_size, predictor.img_size)))

    overlay = make_overlay(display_img, attention_mask_full)
    mask_img = heatmap_to_image(attention_mask_full)

    label_str = "\n".join(f"{c}: {p:.3f}" for c, p in class_probs.items())
    summary = f"Predicted: **{predicted_class}**\n\n{label_str}"

    return overlay, mask_img, class_probs, summary


with gr.Blocks(title="GazeAlign") as demo:
    gr.Markdown(
        """
        # GazeAlign — Gaze-Supervised Medical Image Classification

        Upload an image, click on it to simulate a radiologist's scanpath
        (each click = one fixation, in order), then run the model.
        """
    )

    fixations_state = gr.State([])

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="Upload image, then click to add fixations", type="pil", interactive=True)
            with gr.Row():
                clear_btn = gr.Button("Clear fixations")
                run_btn = gr.Button("Run GazeAlign", variant="primary")
            status = gr.Markdown()

        with gr.Column():
            overlay_output = gr.Image(label="Attention overlay")
            mask_output = gr.Image(label="Gaze-conditioned mask")
            label_output = gr.Label(label="Class probabilities")
            summary_output = gr.Markdown()

    image_input.select(add_fixation, [image_input, fixations_state], [image_input, fixations_state, status])
    clear_btn.click(reset_fixations, [image_input], [image_input, fixations_state])
    run_btn.click(
        run_prediction,
        [image_input, fixations_state],
        [overlay_output, mask_output, label_output, summary_output],
    )

if __name__ == "__main__":
    demo.launch()
