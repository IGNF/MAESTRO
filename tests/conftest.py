import pytest

from conf.dataset.flair import FLAIRConfig
from conf.dataset.pastis_hd import PASTISHDConfig
from conf.dataset.s2_naip import S2NAIPConfig
from conf.dataset.treesatai_ts import TreeSatAITSConfig
from conf.datasets import DatasetsConfig
from conf.mask import MaskConfig
from conf.opt import OptFinetuneConfig
from tests import TEST_DATA_DIR


@pytest.fixture()
def datasets_treesat():
    return DatasetsConfig(
        root_dir=TEST_DATA_DIR,
        name_dataset="treesatai_ts",
        treesatai_ts=TreeSatAITSConfig(rel_dir=""),
        pastis_hd=PASTISHDConfig(),
        flair=FLAIRConfig(),
        s2_naip=S2NAIPConfig(),
    )


@pytest.fixture()
def mask():
    return MaskConfig()


@pytest.fixture()
def opt_finetune():
    return OptFinetuneConfig()
