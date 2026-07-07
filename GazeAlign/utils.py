"""Small shared utilities: config loading, seeding, checkpoint I/O."""

from __future__ import annotations

import random
import re

import numpy as np
import torch
import yaml


# Canonical sub-module name -> (refactored key, legacy research-script key)
# in a saved checkpoint. ``scripts/train.py``/``run_eval.py`` use the
# refactored ``*_state`` names; older research checkpoints use the second.
_CKPT_STATE_KEYS = {
    "image_encoder":    ("image_encoder_state",    "img_proj_state"),
    "scanpath_encoder": ("scanpath_encoder_state", "scan_proj_state"),
    "mask_generator":   ("mask_generator_state",   "gen_mask_state"),
    "classifier":       ("classifier_state",       "classy_state"),
}


def _strip_module_prefix(state: dict) -> dict:
    """Drop the ``module.`` prefix left by ``nn.DataParallel``/DDP."""
    return {
        (k[len("module."):] if k.startswith("module.") else k): v
        for k, v in state.items()
    }


def _remap_classifier_keys(state: dict) -> dict:
    """Rename legacy per-class heads ``class1/class2/...`` to the refactored
    ``GazeClassifier.class_heads.{0,1,...}`` (1-indexed -> 0-indexed)."""
    out = {}
    for k, v in state.items():
        m = re.match(r"class(\d+)\.(weight|bias)$", k)
        if m:
            out[f"class_heads.{int(m.group(1)) - 1}.{m.group(2)}"] = v
        else:
            out[k] = v
    return out


def load_gaze_checkpoint(path_or_ckpt, map_location="cpu") -> dict:
    """Load a GazeAlign checkpoint and normalize it to the current module API.

    Handles both the refactored checkpoint format (``image_encoder_state``
    etc.) and the original research script's format (``img_proj_state``,
    ``module.``-prefixed keys, ``classN`` heads). Returns a dict with one
    state-dict per canonical sub-module name — ``image_encoder``,
    ``scanpath_encoder``, ``mask_generator``, ``classifier`` — plus any
    ``epoch`` present, ready for ``module.load_state_dict(...)``.
    """
    ckpt = (
        path_or_ckpt
        if isinstance(path_or_ckpt, dict)
        else torch.load(path_or_ckpt, map_location=map_location)
    )

    out: dict = {}
    for name, (new_key, legacy_key) in _CKPT_STATE_KEYS.items():
        if new_key in ckpt:
            state = ckpt[new_key]
        elif legacy_key in ckpt:
            state = ckpt[legacy_key]
        else:
            raise KeyError(
                f"Checkpoint is missing state for '{name}' "
                f"(looked for '{new_key}' or '{legacy_key}'). "
                f"Found top-level keys: {sorted(ckpt.keys())}"
            )

        state = _strip_module_prefix(state)
        if name == "classifier":
            state = _remap_classifier_keys(state)
        out[name] = state

    if "epoch" in ckpt:
        out["epoch"] = ckpt["epoch"]
    return out


def load_submodule_state(module, state: dict, ignore_suffixes=("pos_enc.pe",)) -> None:
    """``module.load_state_dict(state)`` tolerant of deterministic buffers.

    Sinusoidal positional-encoding buffers (``pos_enc.pe``) are regenerated
    at construction time and may differ in length between checkpoint and
    model (e.g. a scanpath ``max_len`` of 200 vs 201). Those are dropped
    from *state* and left at the module's own values; any *other* missing
    or unexpected key still raises, so genuine architecture mismatches are
    not silently ignored.
    """
    filtered = {
        k: v for k, v in state.items()
        if not any(k.endswith(suf) for suf in ignore_suffixes)
    }
    missing, unexpected = module.load_state_dict(filtered, strict=False)

    real_missing = [k for k in missing if not any(k.endswith(s) for s in ignore_suffixes)]
    if real_missing or unexpected:
        raise RuntimeError(
            f"State-dict mismatch loading {type(module).__name__}: "
            f"missing={real_missing}, unexpected={list(unexpected)}"
        )


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
