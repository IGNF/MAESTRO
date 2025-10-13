"""Dataset class."""

from hydra_zen import MISSING, builds, store

from conf.dataset.flair import FLAIRConfig
from conf.dataset.pastis_hd import PASTISHDConfig
from conf.dataset.s2_naip import S2NAIPConfig
from conf.dataset.treesatai_ts import TreeSatAITSConfig


# It does not seems possible to have nested dataclasses at all config levels,
# so we do not use a dataclass at the top level.
class DatasetsConfig:
    """Dataset config."""

    def __init__(
        self,
        root_dir: str,
        filter_pretrain: list[str],
        filter_finetune: list[str],
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
        for name_dataset in self.filter_pretrain + self.filter_finetune:
            if name_dataset not in self.__dict__:
                msg = f"Invalid dataset name {name_dataset}. Not an attribute."
                raise ValueError(msg)

        self.pretrain = {
            name_dataset: getattr(self, name_dataset)
            for name_dataset in self.filter_pretrain
        }
        self.finetune = {
            name_dataset: getattr(self, name_dataset)
            for name_dataset in self.filter_finetune
        }

        # temporary restriction to single datasets
        names_dataset = self.filter_pretrain
        name_dataset = self.filter_pretrain[0]
        dataset = self.pretrain[name_dataset]

        if len(names_dataset) > 1:
            msg = "Multi-dataset pretraining/finetuning not implemented yet."
            raise NotImplementedError(msg)

        self.dataset_class = dataset.dataset_class
        self.dataset = dataset


dataset_build = builds(
    DatasetsConfig,
    root_dir=MISSING,
    filter_pretrain=MISSING,
    filter_finetune=MISSING,
    treesatai_ts=TreeSatAITSConfig(),
    pastis_hd=PASTISHDConfig(),
    flair=FLAIRConfig(),
    s2_naip=S2NAIPConfig(),
)

dataset_store = store(group="datasets")
dataset_store(dataset_build, name="default_datasets")
