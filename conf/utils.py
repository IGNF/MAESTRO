"""Utils to convert between configs and dicts."""

import inspect
from dataclasses import asdict

from dacite.core import from_dict

from conf.dataset.utils import DatasetConfig
from conf.datasets import DatasetsConfig
from conf.mask import MaskConfig

DATASETS_TYPES = {
    name_arg: val_arg.annotation
    for name_arg, val_arg in inspect.signature(
        DatasetsConfig.__init__,
    ).parameters.items()
    if val_arg.annotation != inspect._empty  # noqa: SLF001
}


def datasets_from_dict(datasets_dict: dict) -> DatasetsConfig:
    """Convert dict to datasets config."""
    kwargs = {}
    for name_arg, type_arg in DATASETS_TYPES.items():
        if issubclass(type_arg, DatasetConfig):
            kwargs[name_arg] = from_dict(
                data_class=type_arg,
                data=datasets_dict[name_arg],
            )
        else:
            kwargs[name_arg] = datasets_dict[name_arg]

    return DatasetsConfig(**kwargs)


def datasets_to_dict(datasets: DatasetsConfig) -> dict:
    """Convert datasets config to dict."""
    datasets_dict = {}
    for name_arg, type_arg in DATASETS_TYPES.items():
        attr = getattr(datasets, name_arg)
        if issubclass(type_arg, DatasetConfig):
            datasets_dict[name_arg] = asdict(attr)
        else:
            datasets_dict[name_arg] = attr
    return datasets_dict


def mask_from_dict(mask_dict: dict) -> MaskConfig:
    """Convert dict to mask config."""
    return from_dict(data_class=MaskConfig, data=mask_dict)


def mask_to_dict(mask: MaskConfig) -> dict:
    """Convert mask config to dict."""
    return asdict(mask)
