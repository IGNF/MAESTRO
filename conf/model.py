"""Model conf."""

from dataclasses import dataclass

from hydra_zen import store


@dataclass
class ModelConfig:  # noqa: D101
    multimodal: str = "group"
    allmods_depth: int = 0  # nb of blocks with all mods
    model: str = "mae"
    model_size: str = "tiny"
    type_head: str = "attentive"
    loss: str = "l1_norm"
    unpool_dim: int | None = None
    use_date_enc: bool = True
    batch_repeats: int | None = None
    use_ema: bool = True


@dataclass
class BaselineConfig:  # noqa: D101
    multimodal: str = "shared"
    model: str = "dinov2"
    model_size: str = "small"
    type_head: str = "linear"
    loss: str = "l2_norm"
    freeze: bool = False
    weight_source: str = "imagenat"
    pretrained_path: str = None
    keep_norm: bool = True
    add_date_enc: bool = True
    use_ema: bool = True


model_store = store(group="model")
model_store(ModelConfig, name="default_model")
model_store(BaselineConfig, name="baseline_model")
