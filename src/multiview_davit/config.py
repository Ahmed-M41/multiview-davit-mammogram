from pathlib import Path
from omegaconf import OmegaConf, DictConfig


def load_config(path: str | Path) -> DictConfig:
    return OmegaConf.load(path)


def merge_configs(*paths: str | Path) -> DictConfig:
    configs = [OmegaConf.load(p) for p in paths]
    return OmegaConf.merge(*configs)
