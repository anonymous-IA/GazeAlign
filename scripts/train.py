#!/usr/bin/env python
"""
Train GazeAlign on a gaze-annotated medical image classification dataset.

Usage
-----
    python scripts/train.py --config configs/mimic_cxr.yaml

This reproduces the original research script's training loop, but with the
duplicated train/val forward-pass logic factored into
``gazealign.engine.GazeAlignModel.forward_batch`` (see that module for
details on every loss term).
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from GazeAlign import (
    DINOv3Encoder,
    GazeAlignModel,
    GazePriorHeatmap,
    ScanpathTransformer,
    ViTMaskGenerator,
    build_scanpath_pool,
    classifier,
    collect_images,
    compute_classification_metrics,
    gaze_collate_fn,
    get_device,
    load_config,
    set_seed,
)
from GazeAlign.datasets import MimicGazeDataset
from GazeAlign.gaze import fixation_heatmap, load_fixation_csv, scanpath_patch_heatmap
from GazeAlign.visualize import patch_to_image, plot_epoch_debug
import cv2


def parse_args():
    p = argparse.ArgumentParser(description="Train GazeAlign")
    p.add_argument("--config", type=str, default="configs/mimic_cxr.yaml")
    p.add_argument("--resume", type=str, default=None, help="Path to a checkpoint to resume from")
    return p.parse_args()


def build_loaders(cfg):
    data_cfg = cfg["data"]
    base = data_cfg["base_path"]
    classes = data_cfg["classes"]
    img_size = data_cfg.get("img_size", 518)

    train_path = os.path.join(base, data_cfg["train_dir"])
    test_path = os.path.join(base, data_cfg["test_dir"])
    gaze_path = os.path.join(base, data_cfg["gaze_dir"])

    fixations_df = load_fixation_csv(os.path.join(gaze_path, data_cfg["fixations_csv"]))
    eye_gaze_df = load_fixation_csv(os.path.join(gaze_path, data_cfg["eye_gaze_csv"]))

    train_samples = collect_images(train_path, classes)
    test_samples = collect_images(test_path, classes)

    train_ds = MimicGazeDataset(train_samples, fixations_df, classes, img_size=img_size)
    test_ds = MimicGazeDataset(test_samples, fixations_df, classes, img_size=img_size)

    bs = cfg["training"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, collate_fn=gaze_collate_fn)
    test_loader = DataLoader(test_ds, batch_size=cfg["eval"]["batch_size"], shuffle=False, collate_fn=gaze_collate_fn)

    return train_loader, test_loader, eye_gaze_df, fixations_df


def build_model(cfg, num_classes, device):
    model_cfg = cfg["model"]

    image_encoder = DINOv3Encoder(backbone_name=model_cfg["backbone_name"])
    scanpath_encoder = ScanpathTransformer(d_model=model_cfg["scanpath_d_model"])
    mask_generator = ViTMaskGenerator(
        emb_dim=model_cfg["embed_dim"],
        num_patches=model_cfg["grid_size"] ** 2,
        num_heads=model_cfg["mask_heads"],
        depth=model_cfg["mask_depth"],
    )
    clf = classifier(num_classes=num_classes)
    gaze_prior = GazePriorHeatmap(img_size=cfg["data"].get("img_size", 518))

    if torch.cuda.device_count() > 1:
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
        image_encoder = nn.DataParallel(image_encoder)
        mask_generator = nn.DataParallel(mask_generator)
        clf = nn.DataParallel(clf)

    image_encoder, scanpath_encoder = image_encoder.to(device), scanpath_encoder.to(device)
    mask_generator, clf = mask_generator.to(device), clf.to(device)
    gaze_prior = gaze_prior.to(device)

    gaze_align = GazeAlignModel(
        image_encoder=image_encoder,
        scanpath_encoder=scanpath_encoder,
        mask_generator=mask_generator,
        classifier=clf,
        num_classes=num_classes,
        grid_size=model_cfg["grid_size"],
        num_negatives=cfg["training"]["num_negatives"],
        loss_weights=cfg["training"]["loss_weights"],
    ).to(device)

    return gaze_align, gaze_prior


@torch.no_grad()
def visualize_one_sample(gaze_align, gaze_prior, loader, eye_gaze_df, fixations_df, epoch, vis_dir, img_size, device):
    gaze_align.eval()

    batch = next(iter(loader))
    imgs_vit, _, scanpaths, labels, dicom_ids, paths = batch
    imgs_vit, labels = imgs_vit.to(device), labels.to(device)
    scanpaths = [sp[:200].to(device) for sp in scanpaths]

    _, sp_emb, _ = gaze_align.scanpath_encoder(scanpaths)
    patch_mask = torch.sigmoid(gaze_align.mask_generator(sp_emb))
    gaze_heatmap = gaze_prior(torch.stack([F.pad(s, (0, 0, 0, 200 - s.size(0))) for s in scanpaths]))

    dicom_id, path = dicom_ids[0], paths[0]
    img_orig = cv2.imread(path)
    orig_h, orig_w = img_orig.shape[:2]

    df_img = eye_gaze_df[eye_gaze_df["DICOM_ID"] == dicom_id]
    df_fix = fixations_df[fixations_df["DICOM_ID"] == dicom_id]

    fix_weighted = fixation_heatmap(df_fix, img_size, img_size, orig_h, orig_w, sigma=35, weighted=True)
    fix_unweighted = fixation_heatmap(df_fix, img_size, img_size, orig_h, orig_w, sigma=35, weighted=False)
    saccade_hm = scanpath_patch_heatmap(
        df_img[["X_ORIGINAL", "Y_ORIGINAL"]].values, orig_h, orig_w, img_size, img_size, grid_size=37
    )

    save_path = os.path.join(vis_dir, f"epoch_{epoch:03d}", f"sample_0_label_{labels[0].item()}.png")
    plot_epoch_debug(
        img=imgs_vit[0].cpu(),
        patch_hm=patch_to_image(patch_mask[0].cpu().numpy(), img_size, img_size),
        fix_weighted=fix_weighted,
        fix_unweighted=fix_unweighted,
        gen_mask=gaze_heatmap[0].detach().cpu(),
        saccade_patch_hm=saccade_hm,
        save_path=save_path,
    )
    print(f"[epoch {epoch}] saved debug visualization -> {save_path}")


def run_epoch(gaze_align, loader, optimizer, scaler, device, num_classes, train: bool):
    gaze_align.train(train)

    negative_pool = build_scanpath_pool(loader)

    totals = {
        "loss": 0.0, "sim_pos": 0.0, "sim_neg": 0.0, "scan_loss": 0.0,
        "fix_acc": 0.0, "fix_auc": 0.0, "fix_precision": 0.0, "fix_recall": 0.0,
    }
    all_logits, all_labels = [], []
    n_batches = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for imgs_vit, _, scanpaths, labels, dicom_ids, _ in loader:
            imgs_vit = imgs_vit.to(device)
            labels = labels.to(device)
            scanpaths = [sp[:200].to(device) for sp in scanpaths]

            if train:
                optimizer.zero_grad()
                with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                    out = gaze_align.forward_batch(imgs_vit, scanpaths, labels, dicom_ids, negative_pool)
                scaler.scale(out.loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out = gaze_align.forward_batch(imgs_vit, scanpaths, labels, dicom_ids, negative_pool)

            totals["loss"] += out.loss.item()
            totals["sim_pos"] += out.sim_pos.mean().item()
            totals["sim_neg"] += out.sim_neg.mean().item()
            totals["scan_loss"] += (1 - out.scan_loss).item()
            acc, auc, prec, rec = out.fixation_metrics
            totals["fix_acc"] += acc
            totals["fix_auc"] += auc
            totals["fix_precision"] += prec
            totals["fix_recall"] += rec

            all_logits.append(out.class_logits.detach())
            all_labels.append(labels)
            n_batches += 1

    for k in totals:
        totals[k] /= max(n_batches, 1)

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    cls_metrics = compute_classification_metrics(all_logits, all_labels, num_classes)

    return totals, cls_metrics


def main():
    args = parse_args()
    cfg = load_config(args.config)

    set_seed(cfg["training"].get("seed", 42))
    device = get_device()

    classes = cfg["data"]["classes"]
    num_classes = len(classes)

    train_loader, test_loader, eye_gaze_df, fixations_df = build_loaders(cfg)
    gaze_align, gaze_prior = build_model(cfg, num_classes, device)

    optimizer = torch.optim.AdamW(gaze_align.parameters(), lr=cfg["training"]["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        eta_min=cfg["training"]["lr"] * cfg["training"]["lr_min_factor"],
        T_max=cfg["training"]["epochs"],
    )
    scaler = torch.amp.GradScaler(enabled=cfg["training"]["use_amp"] and device.type == "cuda")

    save_dir = cfg["training"]["save_dir"]
    vis_dir = cfg["training"]["vis_dir"]
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    start_epoch = 0
    best_val_loss = float("inf")
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        gaze_align.image_encoder.load_state_dict(ckpt["image_encoder_state"])
        gaze_align.scanpath_encoder.load_state_dict(ckpt["scanpath_encoder_state"])
        gaze_align.mask_generator.load_state_dict(ckpt["mask_generator_state"])
        gaze_align.classifier.load_state_dict(ckpt["classifier_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt.get("epoch", 0)
        best_val_loss = ckpt.get("best_val_loss", best_val_loss)
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    img_size = cfg["data"].get("img_size", 518)

    for epoch in range(start_epoch, cfg["training"]["epochs"]):
        if cfg["training"].get("vis_every_epoch", True):
            try:
                visualize_one_sample(gaze_align, gaze_prior, test_loader, eye_gaze_df, fixations_df, epoch, vis_dir, img_size, device)
            except Exception as e:  # visualization is best-effort, never blocks training
                print(f"[epoch {epoch}] visualization skipped ({e})")

        train_totals, train_cls = run_epoch(gaze_align, train_loader, optimizer, scaler, device, num_classes, train=True)
        val_totals, val_cls = run_epoch(gaze_align, test_loader, optimizer, scaler, device, num_classes, train=False)

        print(
            f"Epoch [{epoch + 1}/{cfg['training']['epochs']}] "
            f"Train Loss: {train_totals['loss']:.4f} Acc: {train_cls['accuracy']:.4f} "
            f"FixAcc: {train_totals['fix_acc']:.4f} | "
            f"Val Loss: {val_totals['loss']:.4f} Acc: {val_cls['accuracy']:.4f} "
            f"BalAcc: {val_cls['balanced_accuracy']:.4f} F1: {val_cls['f1_macro']:.4f} "
            f"AUC(macro): {val_cls['auc_macro']:.4f} FixAcc: {val_totals['fix_acc']:.4f}"
        )
        for c, auc in enumerate(val_cls["auc_per_class"]):
            print(f"  -> Val AUC class {c} ({classes[c]}): {auc:.4f}")

        if val_totals["loss"] < best_val_loss:
            best_val_loss = val_totals["loss"]
            torch.save(
                {
                    "epoch": epoch + 1,
                    "best_val_loss": best_val_loss,
                    "image_encoder_state": gaze_align.image_encoder.state_dict(),
                    "scanpath_encoder_state": gaze_align.scanpath_encoder.state_dict(),
                    "mask_generator_state": gaze_align.mask_generator.state_dict(),
                    "classifier_state": gaze_align.classifier.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "classes": classes,
                    "config": cfg,
                },
                os.path.join(save_dir, "best_model.pth"),
            )
            print(f"Saved new best model at epoch {epoch + 1} (val_loss={best_val_loss:.4f})")

        scheduler.step()


if __name__ == "__main__":
    main()
