"""FLAIR config."""

from dataclasses import dataclass, field

from conf.dataset.utils import (
    DatasetConfig,
    InputRasterConfig,
    PatchSizeConfig,
    TargetRasterConfig,
)
from maestro.dataset.flair import FLAIRDataset


@dataclass
class FLAIRConfig(DatasetConfig):  # noqa: D101
    rel_dir: str = "FLAIR-HUB"
    csv_dir: str | None = None
    version: str | None = None
    val_pretrain: bool = True
    filter_percent: int | None = None
    repeats: int = 1
    crop_meters: float = 102.4

    ref_input: str | None = "aerial"  # defines grid for raster targets
    log_inputs: list[str] = field(
        default_factory=lambda: [
            "aerial",
        ],
    )  # logged input modalities

    filter_inputs: list[str] = field(
        default_factory=lambda: [
            "aerial",
            "dem",
            "s2",
            "s1_asc",
            "s1_des",
        ],
    )  # selected input modalities
    filter_targets: list[str] = field(
        default_factory=lambda: ["cosia"],
    )  # selected target modalities

    aerial: InputRasterConfig = field(
        default_factory=lambda: InputRasterConfig(
            image_size=512,
            patch_size=PatchSizeConfig(mae=16),
            bands=[[3, 0, 1, 2]],
            norm_bands=[1, 3],
            norm_fac=255.0,
        ),
    )
    dem: InputRasterConfig = field(
        default_factory=lambda: InputRasterConfig(
            image_size=512,
            patch_size=PatchSizeConfig(mae=32),
            bands=2,
            norm_fac=1000.0,
        ),
    )
    spot: InputRasterConfig = field(
        default_factory=lambda: InputRasterConfig(
            image_size=64,
            patch_size=PatchSizeConfig(mae=4),
            bands=4,
            norm_fac=2000.0,
        ),
    )
    s2: InputRasterConfig = field(
        default_factory=lambda: InputRasterConfig(
            image_size=10,
            patch_size=PatchSizeConfig(mae=2),
            bands=10,
            norm_bands=[4, 4, 2],
            num_dates=16,
            mask_threshold=0.0,
            norm_fac=5000.0,
        ),
    )
    s1_asc: InputRasterConfig = field(
        default_factory=lambda: InputRasterConfig(
            image_size=10,
            patch_size=PatchSizeConfig(mae=2),
            bands=2,
            norm_bands=[1, 1],
            num_dates=4,
            norm_fac=5.0,
            log_scale=True,
            name_group="sen1",
        ),
    )
    s1_des: InputRasterConfig = field(
        default_factory=lambda: InputRasterConfig(
            image_size=10,
            patch_size=PatchSizeConfig(mae=2),
            bands=2,
            norm_bands=[1, 1],
            num_dates=4,
            norm_fac=5.0,
            log_scale=True,
            name_group="sen1",
        ),
    )

    def __post_init__(self) -> None:
        """Define non configurable attributes."""
        self.dataset_class = FLAIRDataset
        self.total_meters = 102.4

        self.cosia = TargetRasterConfig(
            type_target="segment",
            num_classes=15,
            missing_val=-1,
            bands=1,
        )

        self.lpis = TargetRasterConfig(
            type_target="segment",
            num_classes=74,
            missing_val=-1,
            bands=1,
        )

        super().__post_init__(
            resolutions_meters={
                "cosia": 0.2,
                "lpis": 0.2,
                "aerial": 0.2,
                "dem": 0.2,
                "spot": 1.6,
                "s2": 10.24,
                "s1_asc": 10.24,
                "s1_des": 10.24,
            },
        )
