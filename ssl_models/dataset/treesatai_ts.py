"""TreeSatAI-TS dataset."""

from pathlib import Path
from typing import Literal

import h5py
import numpy as np

from conf.dataset.utils import DatasetConfig
from ssl_models.dataset.dataset import GenericDataset
from ssl_models.dataset.utils import dates_numpy, products_datetimes, read_csv


class TreeSatAITSDataset(GenericDataset):
    """TreeSatAI-TS dataset."""

    def __init__(
        self,
        dataset: DatasetConfig,
        root_dir: Path,
        stage: Literal["train", "val", "test"],
        use_transform: bool,
        random_dates: bool,
        ssl_phase: Literal["pretrain", "probe", "finetune"],
        **kwargs,  # noqa: ARG002, ANN003
    ) -> None:
        super().__init__(
            dataset=dataset,
            stage=stage,
            use_transform=use_transform,
            random_dates=random_dates,
        )

        csv_data = read_csv(
            csv_dir=root_dir,
            stage=stage,
            ssl_phase=ssl_phase,
            balance_pretrain=dataset.balance_pretrain,
            val_pretrain=dataset.val_pretrain,
            filter_percent=dataset.filter_percent,
            parse_dates=["aerial_date"],
        )
        target_cols = [
            col
            for col in csv_data.columns
            if col not in ("aerial_name", "aerial_date", "sen_name")
        ]

        aerial_dates = [
            dates_numpy([aerial_date])
            for aerial_date in csv_data["aerial_date"].tolist()
        ]
        aerial_names = csv_data["aerial_name"].tolist()
        sen_names = csv_data["sen_name"].tolist()

        self.mlc_thresh = 0.07
        self.root_dir = root_dir

        self.aerial_names = aerial_names
        self.aerial_dates = aerial_dates
        self.sen_names = sen_names
        self.targets = csv_data[target_cols].to_numpy()

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        """Get dataset item."""
        sen_path = self.root_dir / "sentinel-ts" / self.sen_names[idx]
        aerial_path = self.root_dir / "aerial" / self.aerial_names[idx]
        aerial_date = self.aerial_dates[idx]
        target = self.targets[idx]

        meta = {}
        meta["aerial_path"] = aerial_path
        meta["aerial_shift"] = 2  # aerial images are 304x304 instead of 300x300
        meta["aerial_dates"] = aerial_date

        meta["s2_path"] = sen_path
        meta["s2_h5_name"] = "sen-2-data"
        meta["s2_h5_mask"] = "sen-2-masks"
        meta["s1_asc_path"] = sen_path
        meta["s1_asc_h5_name"] = "sen-1-asc-data"
        meta["s1_des_path"] = sen_path
        meta["s1_des_h5_name"] = "sen-1-des-data"

        with h5py.File(sen_path, "r") as sen_file:
            meta["s2_dates"] = products_datetimes(sen_file["sen-2-products"][:], 5)
            meta["s1_asc_dates"] = products_datetimes(
                sen_file["sen-1-asc-products"][:],
                5,
            )
            meta["s1_des_dates"] = products_datetimes(
                sen_file["sen-1-des-products"][:],
                5,
            )

        inputs = self.preprocess_rasters(meta)

        inputs["treesat_mlc"] = (target > 0).astype(int)
        inputs["treesat_mlc_dates"] = aerial_date
        inputs["treesat_mlc_thresh"] = (target > self.mlc_thresh).astype(int)
        inputs["treesat_mlc_thresh_dates"] = aerial_date
        inputs["ref_date"] = aerial_date

        return self.transform_rasters(inputs)

    def __len__(self) -> int:
        """Return length of dataset."""
        return len(self.aerial_names)
