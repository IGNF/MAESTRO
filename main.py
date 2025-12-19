"""Command line interface to run experiments.

From the project root directory, run:

    poetry run python main.py group.field=value
"""

from hydra_zen import store, zen

import maestro.conf.data
import maestro.conf.dataset
import maestro.conf.experiment
import maestro.conf.hydra_conf
import maestro.conf.model
import maestro.conf.opt
import maestro.conf.run
import maestro.conf.trainer  # noqa: F401
from maestro.hydra_utils import pre_call_resolve, pre_call_seed
from maestro.run_experiment import run_experiment

# %%
if __name__ == "__main__":
    store.add_to_hydra_store(overwrite_ok=True)
    zen_f = zen(run_experiment, pre_call=[pre_call_seed, pre_call_resolve])
    zen_f.hydra_main(config_name="base_ssl_experiment", version_base="1.3")
