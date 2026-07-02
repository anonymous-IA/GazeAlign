#!/usr/bin/env python
"""
Evaluate a trained GazeAlign checkpoint on a full dataset split.

Usage
-----
    python scripts/run_eval.py --config configs/mimic_cxr.yaml \
        --checkpoint checkpoints/best_model.pth --split test
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gazealign import (
    DINOv3Encoder,
    GazeAlignModel,
    ScanpathTransformer,
    ViTMaskGenerator,
    build_scanpath_pool,
    classifier,
    collect_images,
    compute_classification_metrics,
    gaze_collate_fn,
    get_device,
    load_config,
)
from gazealign.datasets import MimicGazeDataset
from gazealign.gaze import load_fixation_csv


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a GazeAlign checkpoint")
    p.add_argument("--config", type=str, default="configs/mimic_cxr.yaml")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--split", type=str, default="test", choices=["train", "test"])
    p.add_argument("--output", type=str, default=None, help="Optional path to dump metrics as JSON")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = get_device()

    data_cfg = cfg["data"]
    classes = data_cfg["classes"]
    num_classes = len(classes)
    img_size = data_cfg.get("img_size", 518)

    base = data_cfg["base_path"]
    split_dir = data_cfg["train_dir"] if args.split == "train" else data_cfg["test_dir"]
    gaze_path = os.path.join(base, data_cfg["gaze_dir"])
    fixations_df = load_fixation_csv(os.path.join(gaze_path, data_cfg["fixations_csv"]))

    samples = collect_images(os.path.join(base, split_dir), classes)
    dataset = MimicGazeDataset(samples, fixations_df, classes, img_size=img_size)
    loader = DataLoader(dataset, batch_size=cfg["eval"]["batch_size"], shuffle=False, collate_fn=gaze_collate_fn)

    model_cfg = cfg["model"]
    image_encoder = DINOv3Encoder(backbone_name=model_cfg["backbone_name"]).to(device)
    scanpath_encoder = ScanpathTransformer(d_model=model_cfg["scanpath_d_model"]).to(device)
    mask_generator = ViTMaskGenerator(
        emb_dim=model_cfg["embed_dim"], num_patches=model_cfg["grid_size"] ** 2,
        num_heads=model_cfg["mask_heads"], depth=model_cfg["mask_depth"],
    ).to(device)
    clf = classifier(num_classes=num_classes).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    image_encoder.load_state_dict(ckpt["image_encoder_state"])
    scanpath_encoder.load_state_dict(ckpt["scanpath_encoder_state"])
    mask_generator.load_state_dict(ckpt["mask_generator_state"])
    clf.load_state_dict(ckpt["classifier_state"])
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

    gaze_align = GazeAlignModel(
        image_encoder=image_encoder, scanpath_encoder=scanpath_encoder,
        mask_generator=mask_generator, classifier=clf,
        num_classes=num_classes, grid_size=model_cfg["grid_size"],
        num_negatives=cfg["training"]["num_negatives"],
        loss_weights=cfg["training"]["loss_weights"],
    ).to(device)
    gaze_align.eval()

    negative_pool = build_scanpath_pool(loader)

    all_logits, all_labels = [], []
    total_loss, n_batches = 0.0, 0

    with torch.no_grad():
        for imgs_vit, _, scanpaths, labels, dicom_ids, _ in loader:
            imgs_vit = imgs_vit.to(device)
            labels = labels.to(device)
            scanpaths = [sp[:200].to(device) for sp in scanpaths]

            out = gaze_align.forward_batch(imgs_vit, scanpaths, labels, dicom_ids, negative_pool)

            all_logits.append(out.class_logits)
            all_labels.append(labels)
            total_loss += out.loss.item()
            n_batches += 1

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    metrics = compute_classification_metrics(all_logits, all_labels, num_classes)
    metrics["loss"] = total_loss / n_batches
    metrics["split"] = args.split
    metrics["num_samples"] = len(dataset)

    print(f"\n=== Evaluation on '{args.split}' split ({len(dataset)} samples) ===")
    print(f"Loss: {metrics['loss']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}  Balanced Acc: {metrics['balanced_accuracy']:.4f}")
    print(f"Precision (macro): {metrics['precision_macro']:.4f}  Recall (macro): {metrics['recall_macro']:.4f}")
    print(f"F1 (macro): {metrics['f1_macro']:.4f}  AUC (macro): {metrics['auc_macro']:.4f}")
    for c, auc in enumerate(metrics["auc_per_class"]):
        print(f"  -> AUC class {c} ({classes[c]}): {auc:.4f}")

    if args.output:
        def _sanitize(obj):
            if isinstance(obj, float) and obj != obj:  # NaN check without importing math
                return None
            if isinstance(obj, list):
                return [_sanitize(v) for v in obj]
            if isinstance(obj, dict):
                return {k: _sanitize(v) for k, v in obj.items()}
            return obj

        with open(args.output, "w") as f:
            json.dump(_sanitize(metrics), f, indent=2)
        print(f"\nMetrics written to {args.output}")


if __name__ == "__main__":
    main()
