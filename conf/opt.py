"""Optimizer options conf."""

from dataclasses import dataclass

from hydra_zen import store


@dataclass
class OptConfig:  # noqa: D101
    b1: float = 0.9
    b2: float = 0.99
    wd: float = 0.01
    accumulate_grad_batches: int = 1


@dataclass
class OptPretrainConfig(OptConfig):  # noqa: D101
    base_lr: float = 3e-5
    epochs: int = 20
    batch_size: int = 32


pretrain_store = store(group="opt_pretrain")
pretrain_store(OptPretrainConfig, name="default_pretrain")


@dataclass
class OptProbeConfig(OptConfig):  # noqa: D101
    base_lr: float = 1e-5
    epochs: int = 10
    batch_size: int = 32


probe_store = store(group="opt_probe")
probe_store(OptProbeConfig, name="default_probe")


@dataclass
class OptFinetuneConfig(OptConfig):
    """Finetuning optimizer config.

    Use:
        - monitor=treesat_mlc_thresh/weighted_f1_val for TreeSatAI-TS
        - monitor=pastis_seg/average_iou_val for PASTIS-HD
        - monitor=cosia/average_iou_val for FLAIR
    """

    base_lr: float = 1e-5
    epochs: int = 20
    batch_size: int = 32
    lw_decay: float | None = None
    final_factor: float = 2
    monitor: str | None = None


finetune_store = store(group="opt_finetune")
finetune_store(OptFinetuneConfig, name="default_finetune")
