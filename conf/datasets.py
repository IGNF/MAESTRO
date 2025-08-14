"""Dataset class."""

from hydra_zen import MISSING, builds, store

from conf.dataset.flair import FLAIRConfig
from conf.dataset.pastis_hd import PASTISHDConfig
from conf.dataset.treesatai_ts import TreeSatAITSConfig

ALL_DATASETS = [
    "flair",
    "pastis_hd",
    "treesatai_ts",
]


# It does not seems possible to have nested dataclasses at all config levels,
# so we do not use a dataclass at the top level.
class DatasetsConfig:
    """Dataset config."""

    def __init__(
        self,
        **kwargs,  # noqa: ANN003
    ) -> None:
        """Assign args to config attributes."""
        self.__dict__ = kwargs
        self.__post_init__()

    def __post_init__(self) -> None:
        """Check validity of dataset names and further define non configurable attrs."""
        for name_dataset in self.filter_pretrain + self.filter_finetune:
            if name_dataset not in self.__dict__:
                msg = f"Invalid dataset name {name_dataset}. Not an attribute."
                raise ValueError(msg)
        self.allowed = {
            name_dataset: getattr(self, name_dataset)
            for name_dataset in self.filter_allowed
        }
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
    filter_allowed=ALL_DATASETS,
    filter_pretrain=ALL_DATASETS,
    filter_finetune=ALL_DATASETS,
    flair=FLAIRConfig(),
    pastis_hd=PASTISHDConfig(),
    treesatai_ts=TreeSatAITSConfig(),
)

dataset_store = store(group="datasets")
dataset_store(dataset_build, name="default_datasets")
