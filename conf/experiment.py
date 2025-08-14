"""Experiment conf."""  # noqa: EXE002

from hydra_zen import MISSING, builds, store

from ssl_models.run_experiment import run_experiment

Experiment = builds(
    run_experiment,
    populate_full_signature=True,
    hydra_defaults=[
        "_self_",
        {"run": "default_run"},
        {"opt_pretrain": "default_pretrain"},
        {"opt_probe": "default_probe"},
        {"opt_finetune": "default_finetune"},
        {"datasets": "default_datasets"},
        {"data": "default_data"},
        {"mask": "default_mask"},
        {"model": "default_model"},
        {"trainer": "default_trainer"},
    ],
    run=MISSING,
    opt_pretrain=MISSING,
    opt_probe=MISSING,
    opt_finetune=MISSING,
    datasets=MISSING,
    data=MISSING,
    mask=MISSING,
    model=MISSING,
    trainer=MISSING,
)
store(Experiment, name="base_ssl_experiment")
