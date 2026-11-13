from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer,
    step: int,
    model_config: dict[str, Any],
    tokenizer_metadata: dict[str, Any],
) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "model_config": model_config,
        "tokenizer_metadata": tokenizer_metadata,
    }
    torch.save(checkpoint, checkpoint_path)


def _torch_load(path: str | Path, map_location: str | torch.device, weights_only: bool) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=weights_only)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_checkpoint(
    path: str | Path,
    model: nn.Module | None = None,
    optimizer: Optimizer | None = None,
    map_location: str | torch.device = "cpu",
    trusted: bool = False,
    load_optimizer: bool = False,
) -> dict[str, Any]:
    checkpoint = _torch_load(path, map_location=map_location, weights_only=not trusted and not load_optimizer)

    if model is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and load_optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint


def find_latest_checkpoint(path: str | Path) -> Path | None:
    checkpoint_path = Path(path)
    if checkpoint_path.exists():
        return checkpoint_path
    return None