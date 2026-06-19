import os
import random
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from src.utils.logging import get_logger

logger = get_logger(__name__)


def load_config(config_path: str = "src/config", config_name: str = "config") -> DictConfig:
    if not Path(config_path).is_absolute():
        config_path = str(Path.cwd() / config_path)
    with hydra.initialize_config_dir(config_dir=config_path):
        cfg = hydra.compose(config_name=config_name)
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")


def save_checkpoint(state: dict, filepath: str | Path) -> None:
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, str(path))
    logger.info(f"Checkpoint saved to {path}")


def load_checkpoint(filepath: str | Path, device: str | None = None) -> dict[str, Any]:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(str(path), map_location=device)
    logger.info(f"Checkpoint loaded from {path}")
    return state
