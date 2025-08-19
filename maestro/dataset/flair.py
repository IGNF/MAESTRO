"""FLAIR dataset module."""

import json
from pathlib import Path
from typing import Literal

import geopandas as gpd
import numpy as np

from conf.dataset.utils import DatasetConfig
from maestro.dataset.dataset import GenericDataset
from maestro.dataset.utils import dict_datetimes, read_csv, strs_datetimes


class FLAIRDataset(GenericDataset):
    """FLAIR dataset."""

    def __init__(
        self,
        dataset: DatasetConfig,
        root_dir: Path,
        stage: Literal["train", "val", "test"],
        use_transform: bool,
        random_dates: bool,
        random_crop: bool,
        ssl_phase: Literal["pretrain", "probe", "finetune"],
        **kwargs,  # noqa: ARG002, ANN003
    ) -> None:
        super().__init__(
            dataset=dataset,
            stage=stage,
            use_transform=use_transform,
            random_dates=random_dates,
            random_crop=random_crop,
        )

        csv_dir = Path(dataset.csv_dir) if dataset.csv_dir else root_dir
        csv_data = read_csv(
            csv_dir=csv_dir,
            stage=stage,
            ssl_phase=ssl_phase,
            version=dataset.version,
            balance_pretrain=dataset.balance_pretrain,
            val_pretrain=dataset.val_pretrain,
            filter_percent=dataset.filter_percent,
        )

        self.root_dir = root_dir
        self.patch_ids = csv_data["patch_id"].tolist()
        self.base_length = len(self.patch_ids)
        self.repeats = dataset.repeats

        self.mod_mapping = {
            "aerial": "aerial_rgbi",
            "aerial_rlt": "aerial-rlt_pan",
            "dem": "dem_elev",
            "spot": "spot_rgbi",
            "s2": "sentinel2_ts",
            "s2_mask": "sentinel2_msk-sc",
            "s1_asc": "sentinel1-asc_ts",
            "s1_des": "sentinel1-desc_ts",
            "cosia": "aerial_label-cosia",
            "lpis": "all_label-lpis",
        }

        self.dates_str = {}
        for name_mod in ("aerial", "aerial_rlt", "spot"):
            name_flair = self.mod_mapping[name_mod].split("_")[0].upper()
            dates_path = (
                self.root_dir / "GLOBAL_ALL_MTD" / f"GLOBAL_{name_flair}_MTD_DATES.gpkg"
            )
            dates_mod = gpd.read_file(dates_path, engine="pyogrio", use_arrow=True)
            self.dates_str[name_mod] = {
                row["patch_id"]: row["date"] for _, row in dates_mod.iterrows()
            }

        self.dates_dict = {}
        for name_mod in ("s2", "s1_asc", "s1_des"):
            name_flair = self.mod_mapping[name_mod].split("_")[0].upper()
            dates_path = (
                self.root_dir / "GLOBAL_ALL_MTD" / f"GLOBAL_{name_flair}_MTD_DATES.gpkg"
            )
            dates_mod = gpd.read_file(dates_path, engine="pyogrio", use_arrow=True)
            self.dates_dict[name_mod] = {
                "_".join(row["patch_id"].split("_")[:2]): row["acquisition_dates"]
                for _, row in dates_mod.iterrows()
            }

        self.cosia_ignore = (15, 16, 17, 18)
        self.cosia_missing = dataset.cosia.missing_val

        self.lpis_ignore = (0,)
        self.lpis_missing = dataset.lpis.missing_val

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        """Get dataset item."""
        idx, start_gcd = self.sample_gcd(
            idx,
            base_length=self.base_length,
            repeats=self.repeats,
        )

        patch_id = self.patch_ids[idx]
        domain, area, pos = patch_id.split("_")
        zone_id = f"{domain}_{area}"

        meta = {}
        for name_mod in self.mod_mapping:
            name_flair = self.mod_mapping[name_mod].upper()
            meta[f"{name_mod}_path" if name_mod != "s2_mask" else "s2_mask"] = (
                self.root_dir
                / f"{domain}_{name_flair}"
                / area
                / f"{domain}_{name_flair}_{area}_{pos}.tif"
            )

        for name_mod in ("aerial", "aerial_rlt", "spot"):
            meta[f"{name_mod}_dates"] = strs_datetimes(
                [self.dates_str[name_mod][patch_id]],
                fmt="%Y%m%d",
            )

        for name_mod in ("s2", "s1_asc", "s1_des"):
            meta[f"{name_mod}_dates"] = dict_datetimes(
                json.loads(self.dates_dict[name_mod][zone_id]),
                start=1,
            )
        meta["s2_mask_bands"] = [0, 1]

        meta["dem_dates"] = meta["aerial_dates"]
        meta["lpis_dates"] = meta["aerial_dates"]
        meta["cosia_dates"] = meta["aerial_dates"]

        inputs = self.preprocess_rasters(meta, start_gcd=start_gcd)

        for name_mod, labels_ignore, label_missing in zip(
            ["cosia", "lpis"],
            [self.cosia_ignore, self.lpis_ignore],
            [self.cosia_missing, self.lpis_missing],
        ):
            if name_mod not in inputs:
                continue
            inputs[name_mod][np.isin(inputs[name_mod], labels_ignore)] = label_missing

        if "dem" in inputs:
            inputs["dem"][:, 0] = (inputs["dem"][:, 0] - inputs["dem"][:, 1]) * 30
        inputs["ref_date"] = meta["aerial_dates"]

        return self.transform_rasters(inputs)

    def __len__(self) -> int:
        """Return length of dataset."""
        return self.base_length * self.repeats**2
