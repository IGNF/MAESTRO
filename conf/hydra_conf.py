"""Hydra conf."""  # noqa: EXE002

from hydra.conf import HydraConf, JobConf, RunDir, RuntimeConf
from hydra_zen import store

store(
    HydraConf(
        job=JobConf(chdir=False),
        run=RunDir(dir="${run.exp_dir}/${run.exp_name}/${now:%Y-%m-%d_%H-%M-%S}"),
        runtime=RuntimeConf(
            output_dir="${run.exp_dir}/${run.exp_name}/${now:%Y-%m-%d_%H-%M-%S}",
        ),
    ),
)
store.add_to_hydra_store()
