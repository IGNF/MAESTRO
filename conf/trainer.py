"""Trainer conf."""

from dataclasses import dataclass

from hydra_zen import store


@dataclass
class TrainerConfig:  # noqa: D101
    accelerator: str = "auto"
    devices: str = "auto"
    strategy: str = "ddp_find_unused_parameters_true"
    precision: str = "16-mixed"
    num_nodes: int = 1


trainer_store = store(group="trainer")
trainer_store(TrainerConfig, name="default_trainer")
