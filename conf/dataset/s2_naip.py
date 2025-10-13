"""S2-NAIP config."""

from dataclasses import dataclass, field

from conf.dataset.utils import (DatasetConfig, InputRasterConfig,
                                PatchSizeConfig, TargetRasterConfig)
from maestro.dataset.s2_naip import S2NAIPDataset


@dataclass
class S2NAIPConfig(DatasetConfig):  # noqa: D101
    rel_dir: str = "s2-naip-urban"
    val_pretrain: bool = True
    test_pretrain: bool = True
    repeats: int = 5
    crop_meters: float = 120

    ref_input: str | None = None  # defines grid for raster targets
    log_inputs: list[str] = field(
        default_factory=lambda: ["aerial", "spot"],
    )  # logged input modalities
    filter_inputs: list[str] = field(
        default_factory=lambda: [
            "aerial",
            "spot",
            "s2",
            "s1",
        ],
    )  # selected input modalities
    filter_targets: list[str] = field(
        default_factory=list,
    )  # selected target modalities

    aerial: InputRasterConfig = field(
        default_factory=lambda: InputRasterConfig(
            image_size=384,
            patch_size=PatchSizeConfig(mae=16),
            bands=[[3, 0, 1, 2]],
            norm_bands=[1, 3],
            norm_fac=255.0,
        ),
    )
    spot: InputRasterConfig = field(
        default_factory=lambda: InputRasterConfig(
            image_size=128,
            patch_size=PatchSizeConfig(mae=16),
            bands=3,
            norm_fac=255.0,
        ),
    )
    landsat: InputRasterConfig = field(
        default_factory=lambda: InputRasterConfig(
            image_size=12,
            patch_size=PatchSizeConfig(mae=2),
            bands=11,
            num_dates=16,
            norm_fac=5000.0,
        ),
    )
    s2: InputRasterConfig = field(
        default_factory=lambda: InputRasterConfig(
            image_size=12,
            patch_size=PatchSizeConfig(mae=2),
            bands=10,
            norm_bands=[4, 4, 2],
            num_dates=16,
            norm_fac=5000.0,
        ),
    )
    s1: InputRasterConfig = field(
        default_factory=lambda: InputRasterConfig(
            image_size=12,
            patch_size=PatchSizeConfig(mae=2),
            bands=2,
            norm_bands=[1, 1],
            num_dates=4,
            norm_fac=20.0,
            name_group="sen1",
        ),
    )

    def __post_init__(self) -> None:
        """Define non configurable attributes."""
        self.dataset_class = S2NAIPDataset
        self.total_meters = 512 * 1.25

        self.osm_seg = TargetRasterConfig(
            type_target="segment",
            num_classes=6,
            missing_val=-1,
        )
        super().__post_init__(
            resolutions_meters={
                "osm_seg": 1.25,
                "aerial": 1.25,
                "spot": 1.25,
                "landsat": 10.0,
                "s2": 10.0,
                "s1": 10.0,
            },
        )
