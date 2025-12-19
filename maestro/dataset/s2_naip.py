"""S2-NAIP dataset module."""

from pathlib import Path
from typing import Literal

import numpy as np

from maestro.conf.dataset.utils import DatasetConfig
from maestro.dataset.dataset import GenericDataset
from maestro.dataset.utils import naip_datetimes, products_datetimes, read_csv


class S2NAIPDataset(GenericDataset):
    """S2-NAIP dataset."""

    def __init__(
        self,
        dataset: DatasetConfig,
        root_dir: Path,
        stage: Literal["train", "val", "test"],
        use_transform: bool,
        random_dates: bool,
        ssl_phase: Literal["pretrain", "probe", "finetune"],
        random_crop: bool,
        **kwargs,  # noqa: ARG002, ANN003
    ) -> None:
        super().__init__(
            dataset=dataset,
            stage=stage,
            use_transform=use_transform,
            random_dates=random_dates,
            random_crop=random_crop,
        )

        csv_data = read_csv(
            csv_dir=root_dir,
            stage=stage,
            ssl_phase=ssl_phase,
            val_pretrain=dataset.val_pretrain,
            test_pretrain=dataset.test_pretrain,
        )

        self.root_dir = root_dir
        self.image_ids = csv_data["name"].to_list()
        self.base_length = len(self.image_ids)
        self.repeats = dataset.repeats

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        """Get dataset item."""
        idx, start_gcd = self.sample_gcd(
            idx,
            base_length=self.base_length,
            repeats=self.repeats,
        )

        image_id = self.image_ids[idx]
        ref_date = naip_datetimes(
            str(
                np.loadtxt(
                    self.root_dir / "dates" / "naip" / f"{image_id}.txt",
                    dtype="str",
                ),
            ),
        )

        meta = {}
        meta["aerial_path"] = self.root_dir / "naip" / f"{image_id}.png"
        meta["aerial_dates"] = ref_date

        meta["spot_path"] = self.root_dir / "naip" / f"{image_id}.png"
        meta["spot_dates"] = ref_date

        meta["landsat_path"] = self.root_dir / "landsat" / f"{image_id}_stacked.tif"
        meta["landsat_dates"] = products_datetimes(
            np.loadtxt(
                self.root_dir / "dates" / "landsat" / f"{image_id}.txt",
                dtype="str",
            ),
            4,
        )

        meta["s2_path"] = self.root_dir / "sentinel2" / f"{image_id}_stacked.tif"
        meta["s2_dates"] = products_datetimes(
            np.loadtxt(
                self.root_dir / "dates" / "s2" / f"{image_id}.txt",
                dtype="str",
            ),
            5,
        )

        meta["s1_path"] = meta["s1_asc_path"] = (
            self.root_dir / "sentinel1" / f"{image_id}.tif"
        )
        meta["s1_dates"] = products_datetimes(
            np.loadtxt(self.root_dir / "dates" / "s1" / f"{image_id}.txt", dtype="str"),
            5,
        )

        meta["osm_seg_path"] = self.root_dir / "openstreetmap" / f"{image_id}.geojson"
        meta["osm_seg_dates"] = ref_date

        inputs = self.preprocess_rasters(meta, start_gcd=start_gcd)

        inputs["ref_date"] = ref_date

        return self.transform_rasters(inputs)

    def __len__(self) -> int:
        """Return length of dataset."""
        return self.base_length * self.repeats**2
