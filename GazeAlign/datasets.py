"""
Dataset loaders for GazeAlign.

The primary dataset is MIMIC-Eye / MIMIC-CXR-style: chest X-rays organized
into per-class folders, paired with a CSV of radiologist eye-tracking
fixations keyed by DICOM ID. ``MimicGazeDataset`` is the loader used during
training (see ``configs/mimic_cxr.yaml``).

A second, lighter-weight loader is provided for inference on a single
image (``examples/predict_single.py``), where there is no train/test split
and no class folder structure -- just one image and one fixations CSV.
"""

from __future__ import annotations

import os

import cv2
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset

from .constants import IMG_SIZE
from .gaze import get_scanpath


def build_transform(img_size: int = IMG_SIZE) -> T.Compose:
    """Shared preprocessing: tensor -> resize -> grayscale-replicated-to-3-channel.

    Chest X-rays are single-channel; the backbone expects 3-channel input,
    so grayscale is replicated across channels rather than treated as RGB.
    """
    return T.Compose(
        [
            T.ToTensor(),
            T.Resize((img_size, img_size)),
            T.Grayscale(num_output_channels=3),
        ]
    )


def collect_images(split_path: str, classes: list[str]) -> list[dict]:
    """Walk a ``split_path`` directory laid out as ``split_path/<class>/*.jpg``
    and return a flat list of ``{dicom_id, class, path}`` records.
    """
    samples = []
    for cls in classes:
        cls_path = os.path.join(split_path, cls)
        if not os.path.isdir(cls_path):
            continue
        for f in sorted(os.listdir(cls_path)):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                samples.append(
                    {
                        "dicom_id": os.path.splitext(f)[0],
                        "class": cls,
                        "path": os.path.join(cls_path, f),
                    }
                )
    return samples


class MimicGazeDataset(Dataset):
    """Image + scanpath + class-label dataset.

    Each item returns:
      - ``img_vit``  : ``[3, H, W]`` preprocessed image tensor
      - ``img_unet`` : ``[3, H, W]`` preprocessed image tensor (kept separate
        from ``img_vit`` in case the two consumers ever need different
        preprocessing -- currently identical)
      - ``scanpath`` : ``[T, 3]`` normalized (x, y, t) fixation sequence,
        or a single zero row if no fixations exist for this image
      - ``label``    : int class index
      - ``dicom_id`` : str
      - ``path``     : str, original image path
    """

    def __init__(self, samples: list[dict], fixations_df, classes: list[str], img_size: int = IMG_SIZE):
        self.samples = samples
        self.fixations_df = fixations_df
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.transform = build_transform(img_size)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]

        img = cv2.imread(s["path"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        img_vit = self.transform(img)
        img_unet = self.transform(img)

        scanpath = get_scanpath(self.fixations_df, s["dicom_id"], img_height=h, img_width=w)
        if scanpath is None:
            scanpath = torch.zeros((1, 3))

        label = self.class_to_idx[s["class"]]
        return img_vit, img_unet, scanpath, label, s["dicom_id"], s["path"]


def gaze_collate_fn(batch):
    """Collate function that keeps variable-length scanpaths as a list
    (rather than stacking them), since ``ScanpathTransformer`` pads them
    internally per-batch.
    """
    imgs_vit, imgs_unet, scanpaths, labels, dicom_ids, paths = zip(*batch)
    return (
        torch.stack(imgs_vit),
        torch.stack(imgs_unet),
        list(scanpaths),
        torch.tensor(labels),
        list(dicom_ids),
        list(paths),
    )


# -----------------------------------------------------------------------------
# Dataset registry, so new datasets can be added without touching training
# code -- just register a loader and reference its name from a config.
# -----------------------------------------------------------------------------
DATASET_REGISTRY = {
    "mimic_cxr": MimicGazeDataset,
}


def build_dataset(name: str, *args, **kwargs) -> Dataset:
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list(DATASET_REGISTRY)}")
    return DATASET_REGISTRY[name](*args, **kwargs)
