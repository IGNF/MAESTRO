"""Data conf."""

from dataclasses import dataclass

from hydra_zen import store


@dataclass
class DataConfig:  # noqa: D101
    use_transform: bool = True
    random_dates: bool = True
    random_crop: bool = True
    num_workers: int = 12


data_store = store(group="data")
data_store(DataConfig, name="default_data")
