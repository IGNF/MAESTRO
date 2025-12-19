"""TreeSatAI-TS config."""

from dataclasses import dataclass, field

from maestro.conf.dataset.utils import (
    DatasetConfig,
    InputRasterConfig,
    PatchSizeConfig,
    TargetConfig,
)
from maestro.dataset.treesatai_ts import TreeSatAITSDataset


@dataclass
class TreeSatAITSConfig(DatasetConfig):  # noqa: D101
    rel_dir: str = "TreeSatAI-TS"
    val_pretrain: bool = True
    filter_percent: int | None = None
    crop_meters: float = 60.0
    grid_pos_enc: int | None = 96

    ref_input: str | None = None
    log_inputs: list[str] = field(
        default_factory=lambda: ["aerial"],
    )  # logged input modalities
    filter_inputs: list[str] = field(
        default_factory=lambda: ["aerial", "s2", "s1_asc", "s1_des"],
    )  # selected input modalities
    filter_targets: list[str] = field(
        default_factory=lambda: ["treesat_mlc_thresh"],
    )  # selected target modalities

    aerial: InputRasterConfig = field(
        default_factory=lambda: InputRasterConfig(
            image_size=300,
            patch_size=PatchSizeConfig(mae=20),
            bands=4,
            norm_bands=[1, 3],
            norm_fac=255.0,
        ),
    )
    s2: InputRasterConfig = field(
        default_factory=lambda: InputRasterConfig(
            image_size=6,
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
            image_size=6,
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
            image_size=6,
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
        self.dataset_class = TreeSatAITSDataset
        self.total_meters = 60.0

        self.treesat_mlc = TargetConfig(
            type_target="multilabel_classif",
            num_classes=15,
            missing_val=-1,
        )
        self.treesat_mlc_thresh = TargetConfig(
            type_target="multilabel_classif",
            num_classes=15,
            missing_val=-1,
        )
        super().__post_init__(
            resolutions_meters={
                "aerial": 0.2,
                "s2": 10.0,
                "s1_asc": 10.0,
                "s1_des": 10.0,
            },
        )
