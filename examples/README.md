# Examples

This folder holds everything needed to run `scripts/predict_single.py` without
training anything yourself.

```
examples/
├── images/        # sample images to test against
└── fixations/     # sample fixation CSVs, one per modality/dataset
```

## Fixation CSV format

GazeAlign expects fixation sequences in the same format used during
training (MIMIC-style eye-tracking exports):

| Column            | Type  | Meaning                                              |
|--------------------|-------|-------------------------------------------------------|
| `DICOM_ID`         | str   | Identifier of the image this fixation belongs to       |
| `X_ORIGINAL`       | float | Fixation x-coordinate, in **original image pixels**   |
| `Y_ORIGINAL`       | float | Fixation y-coordinate, in **original image pixels**   |
| `Time (in secs)`   | float | Timestamp of the fixation, in seconds                 |

A single CSV can contain fixations for one image or many (one row per
fixation, grouped by `DICOM_ID`). Rows are sorted internally by time before
being normalized, so input order doesn't matter.

If your CSV only contains one `DICOM_ID`, `predict_single.py` will pick it
up automatically. If it contains several, pass `--dicom_id <id>` to select
which scanpath to use.

### Minimal example

```csv
DICOM_ID,X_ORIGINAL,Y_ORIGINAL,Time (in secs)
sample_cxr_001,512,340,0.00
sample_cxr_001,540,310,0.42
sample_cxr_001,610,355,0.95
```

## Usage

```bash
python scripts/predict_single.py \
    --image examples/images/sample_cxr_001.jpg \
    --fixations examples/fixations/cxr_samples.csv \
    --output outputs/output_mask.png \
    --preset cxr \
    --save_overlay
```

This produces:
- `outputs/output_mask.png` — the learned gaze-conditioned attention mask
- `outputs/output_mask_overlay.png` — the mask blended over the original image
- `outputs/output_mask_gaze_prior.png` — the raw fixation Gaussian-splat (no learned attention)

The predicted class and per-class probabilities are printed to stdout.
