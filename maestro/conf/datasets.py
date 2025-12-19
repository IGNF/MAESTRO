"""Dataset class."""

from hydra_zen import MISSING, builds, store

from maestro.conf.dataset.flair import FLAIRConfig
from maestro.conf.dataset.pastis_hd import PASTISHDConfig
from maestro.conf.dataset.s2_naip import S2NAIPConfig
from maestro.conf.dataset.treesatai_ts import TreeSatAITSConfig


# It does not seems possible to have nested dataclasses at all config levels,
# so we do not use a dataclass at the top level.
class DatasetsConfig:
    """Dataset config."""

    def __init__(
        self,
        root_dir: str,
        name_dataset: str,
        treesatai_ts: TreeSatAITSConfig,
        pastis_hd: PASTISHDConfig,
        flair: FLAIRConfig,
        s2_naip: S2NAIPConfig,
    ) -> None:
        """Assign args to config attributes."""
        self.__dict__ = {
            name_arg: val_arg
            for name_arg, val_arg in locals().items()
            if val_arg is not self
        }
        self.__post_init__()

    def __post_init__(self) -> None:
        """Check validity of dataset names and further define non configurable attrs."""
        if self.name_dataset not in self.__dict__:
            msg = f"Invalid dataset name {self.name_dataset}. Not an attribute."
            raise ValueError(msg)

        dataset = getattr(self, self.name_dataset)
        self.dataset_class = dataset.dataset_class
        self.dataset = dataset


dataset_build = builds(
    DatasetsConfig,
    root_dir=MISSING,
    name_dataset=MISSING,
    treesatai_ts=TreeSatAITSConfig(),
    pastis_hd=PASTISHDConfig(),
    flair=FLAIRConfig(),
    s2_naip=S2NAIPConfig(),
)

dataset_store = store(group="datasets")
dataset_store(dataset_build, name="default_datasets")
