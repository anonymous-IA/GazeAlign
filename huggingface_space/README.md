---
title: GazeAlign
emoji: 👁️
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
license: mit
---

# GazeAlign

Gaze-supervised medical image classification.

**How to use**

1. **Upload an image** — JPG / PNG / BMP / TIFF / WEBP or a **DICOM** (`.dcm`).
2. **Add fixations** (a radiologist-style scanpath) in either of two ways:
   - **Click** on the image to drop fixation points, or
   - **Upload a fixation table** (`.csv` / `.xlsx` / `.xls`) and map its
     `ID / X / Y / Time` columns — X/Y may be raw pixels or normalised `[0,1]`.
3. **Run** to get the **predicted class** with per-class probabilities,
   plus the learned gaze-conditioned attention mask/overlay.

**Model weights**

The demo loads the checkpoint declared by the `cxr` preset in
`configs/presets.yaml` (default `checkpoints/best_model_CXR.pth`). The
weights are not committed to the GitHub repo (too large); add them to this
Space — e.g. track `checkpoints/*.pth` with Git LFS, or download them in a
startup step — so `checkpoints/best_model_CXR.pth` exists at launch.

See the [GitHub repository](https://github.com/MohammedOussamaBEN/GazeAlign)
for training code, evaluation scripts, and the paper.
