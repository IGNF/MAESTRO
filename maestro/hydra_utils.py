"""Hydra utils module."""

import uuid
from pathlib import Path

from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything


def pre_call_seed(cfg: OmegaConf) -> None:
    """Seed everything if run is reproducible."""
    if cfg.run.reproducible:
        seed_everything(cfg.run.seed, workers=True)


def pre_call_resolve(cfg: OmegaConf) -> None:
    """Resolve hydra config."""
    cfg.run.exp_uuid = uuid.uuid4()  # set uuid
    if cfg.run.load_name:
        load_root = Path(cfg.run.exp_dir) / cfg.run.load_name
        ckpt_path = sorted(
            str(ckpt_path)
            for ckpt_path in load_root.rglob("*/checkpoints/pretrain-epoch=*.ckpt")
        ).pop()
        load_cfg_path = (
            Path(ckpt_path).parent.parent / ".hydra" / "config_resolved.yaml"
        )
        load_cfg = OmegaConf.load(load_cfg_path)

        if cfg.model != load_cfg.model:
            msg = "Config is not consistent with loaded config for model."
            raise ValueError(msg)

        cfg.run.load_ckpt_path = ckpt_path
        cfg.run.load_uuid = load_cfg.run.exp_uuid

    hydra_dir = get_hydra_dir()
    cfg_path = hydra_dir / ".hydra" / "config_resolved.yaml"
    OmegaConf.save(cfg, cfg_path)


def get_hydra_dir() -> Path:
    """Get hydra dir."""
    hydra_cfg = HydraConfig.get()
    return Path(hydra_cfg.runtime.output_dir)


def get_hydra_timestamp() -> str:
    """Get hydra timestamp, specifying log folder version."""
    hydra_dir = get_hydra_dir()
    return hydra_dir.stem
