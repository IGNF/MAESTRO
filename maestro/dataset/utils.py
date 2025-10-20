"""Utils module."""

import json
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd


def dates_numpy(dates: list[datetime]) -> np.ndarray:
    """Convert list of dates to (year, day of year, hour) numpy array."""
    return np.array(
        [[date.year, date.timetuple().tm_yday, date.hour] for date in dates],
        dtype=np.int16,
    )


def strs_datetimes(date_strs: list[str], fmt: str = "%Y-%m-%d") -> np.ndarray:
    """Convert list of strs to list of datetimes."""
    date_strs = [
        date_str[:-2] + "01" if date_str[-2:] == "00" else date_str
        for date_str in date_strs
    ]
    datetimes = [
        datetime.strptime(date_str, fmt)  # noqa: DTZ007
        for date_str in date_strs
    ]
    return dates_numpy(datetimes)


def products_datetimes(products: list[str] | list[bytes], idx: int) -> np.ndarray:
    """Convert list of products to list of datetimes."""
    if isinstance(products[0], str):
        datetimes = [
            datetime.strptime(  # noqa: DTZ007
                product.split("_")[-idx][:8],
                "%Y%m%d",
            )
            for product in products
        ]
    else:
        datetimes = [
            datetime.strptime(  # noqa: DTZ007
                product.decode().split("_")[-idx][:8],
                "%Y%m%d",
            )
            for product in products
        ]
    return dates_numpy(datetimes)


def naip_datetimes(datetime_str: str) -> np.ndarray:
    """Convert naip string to list of datetime."""
    datetimes = [
        datetime.strptime(  # noqa: DTZ007
            datetime_str.split("_")[-1][:8],
            "%Y%m%d",
        ),
    ]
    return dates_numpy(datetimes)


def dict_datetimes(datetime_dict: dict | str, start: int = 0) -> np.ndarray:
    """Convert list of datetime strings to list of datetimes."""
    if not isinstance(datetime_dict, dict):
        datetime_dict = json.loads(datetime_dict)
    datetimes = [
        datetime.strptime(  # noqa: DTZ007
            str(datetime_dict[str(idx)]),
            "%Y%m%d",
        )
        for idx in range(start, len(datetime_dict) + start)
    ]
    return dates_numpy(datetimes)


def read_csv(
    csv_dir: Path,
    stage: Literal["train", "val", "test"],
    ssl_phase: Literal["pretrain", "probe", "finetune"],
    version: str | None = None,
    filter_percent: int | None = None,
    fold: int | None = None,
    val_pretrain: bool = False,
    test_pretrain: bool = False,
    **kwargs,  # noqa: ANN003
) -> Path:
    """Read dataset csv."""
    csv_name = []
    if version:
        csv_name += [version]
    if filter_percent:
        csv_name += [f"filtered_{filter_percent}"]
    if fold:
        csv_name += [f"fold_{fold}"]

    stages = [stage]
    if stage == "train" and ssl_phase == "pretrain":
        if val_pretrain:
            stages.append("val")
        if test_pretrain:
            stages.append("test")

    return pd.concat(
        [
            pd.read_csv(csv_dir / f"{'_'.join([stage, *csv_name])}.csv", **kwargs)
            for stage in stages
        ],
    )
