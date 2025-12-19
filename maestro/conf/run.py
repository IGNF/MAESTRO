"""Run conf."""  # noqa: EXE002

from dataclasses import dataclass, field

from hydra_zen import MISSING, store


@dataclass
class RunConfig:  # noqa: D101
    exp_dir: str = MISSING
    exp_name: str = MISSING
    exp_uuid: str | None = None
    load_name: str | None = None
    load_phase: str = "pretrain"
    load_uuid: str | None = None
    load_ckpt_path: str | None = None
    fit_name: str | None = None
    fit_phase: str = "pretrain"
    fit_uuid: str | None = None
    fit_ckpt_path: str | None = None
    reproducible: bool = True
    seed: int = 42
    logged_images_per_epoch: int = 5
    use_clearml: bool = False
    clearml_project_name: str = "ssl"
    clearml_tags: list[str] = field(default_factory=lambda: ["multimodal", "hydra"])
    clearml_offline_mode: bool = False


run_store = store(group="run")
run_store(
    RunConfig,
    exp_uuid=None,
    load_uuid=None,
    load_ckpt_path=None,
    name="default_run",
)
