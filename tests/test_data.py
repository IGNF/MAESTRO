import itertools

import pytest

from maestro import LOGGER
from maestro.train.data import SSLDataModule


@pytest.mark.parametrize(
    "stage, ssl_phase, use_transform, random_dates, random_crop",
    itertools.product(
        ("fit", "validate"),
        ("pretrain", "probe", "finetune"),
        (True, False),
        (True, False),
        (True, False),
    ),
)
def test_data(
    datasets_treesat,
    opt_finetune,
    stage,
    ssl_phase,
    use_transform,
    random_dates,
    random_crop,
):
    data_module = SSLDataModule(
        datasets=datasets_treesat,
        ssl_phase=ssl_phase,
        opt=opt_finetune,
        num_workers=1,
        use_transform=use_transform,
        random_dates=random_dates,
        random_crop=random_crop,
    )
    data_module.setup(stage=stage)
    data_loader = data_module.val_dataloader()
    num_epochs = 2
    for _ in range(num_epochs):
        for idx, batch in enumerate(data_loader):
            LOGGER.info(idx)
            LOGGER.info(f"id: {idx}, batch: {batch.keys()}")
