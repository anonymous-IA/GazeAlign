"""Small shared utilities: config loading, seeding, checkpoint I/O."""

from __future__ import annotations

import random

import numpy as np
import torch
import yaml


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def unwrap(model):
    """Return the underlying module if wrapped in DataParallel/DDP."""
    return model.module if hasattr(model, "module") else model
