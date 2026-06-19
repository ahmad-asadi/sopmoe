#!/usr/bin/env python3
import hydra
from omegaconf import DictConfig, OmegaConf

from src.utils import get_logger, set_seed

logger = get_logger(__name__)


@hydra.main(version_base=None, config_path="src/config", config_name="config")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    logger.info("Configuration loaded successfully")
    logger.info(f"Experiment config:\n{OmegaConf.to_yaml(cfg)}")
    print("\n✓ Pipeline setup verified — config, logging, and imports all working.\n")


if __name__ == "__main__":
    main()
