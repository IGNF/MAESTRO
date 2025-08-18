import pytest

from conf.dataset.treesatai_ts import TreeSatAITSConfig
from conf.datasets import DatasetsConfig
from conf.mask import MaskConfig
from conf.opt import OptFinetuneConfig
from tests import TEST_DATA_DIR


@pytest.fixture()
def datasets_treesat():
    return DatasetsConfig(
        root_dir=TEST_DATA_DIR,
        treesatai_ts=TreeSatAITSConfig(rel_dir=""),
        filter_allowed=["treesatai_ts"],
        filter_pretrain=["treesatai_ts"],
        filter_finetune=["treesatai_ts"],
    )


@pytest.fixture()
def mask():
    return MaskConfig()


@pytest.fixture()
def opt_finetune():
    return OptFinetuneConfig()
