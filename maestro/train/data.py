"""SSL data module."""

from pathlib import Path
from typing import Literal

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from maestro.conf.datasets import DatasetsConfig
from maestro.conf.opt import OptFinetuneConfig, OptPretrainConfig, OptProbeConfig


class SSLDataModule(LightningDataModule):
    """SSL data module."""

    def __init__(
        self,
        datasets: DatasetsConfig,
        ssl_phase: Literal["pretrain", "probe", "finetune"],
        opt: OptPretrainConfig | OptProbeConfig | OptFinetuneConfig,
        num_workers: int,
        use_transform: bool,
        random_dates: bool,
        random_crop: bool,
    ) -> None:
        super().__init__()

        root_dir = Path(datasets.root_dir) / datasets.dataset.rel_dir
        train_dataset = datasets.dataset_class(
            dataset=datasets.dataset,
            root_dir=root_dir,
            stage="train",
            ssl_phase=ssl_phase,
            use_transform=use_transform,
            random_dates=random_dates,
            random_crop=random_crop,
        )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=opt.batch_size,
            num_workers=num_workers,
            shuffle=True,
            drop_last=True,
        )

        val_dataset = datasets.dataset_class(
            dataset=datasets.dataset,
            root_dir=root_dir,
            stage="val",
            ssl_phase=ssl_phase,
            use_transform=use_transform,
            random_dates=random_dates,
            random_crop=random_crop,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=opt.batch_size,
            num_workers=num_workers,
            shuffle=True,
            drop_last=True,
        )

        test_dataset = datasets.dataset_class(
            dataset=datasets.dataset,
            root_dir=root_dir,
            stage="test",
            ssl_phase=ssl_phase,
            use_transform=use_transform,
            random_dates=random_dates,
            random_crop=random_crop,
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=opt.batch_size,
            num_workers=num_workers,
            shuffle=True,
            drop_last=True,
        )

    def train_dataloader(self) -> DataLoader:
        """Return train data loader."""
        return self.train_loader

    def val_dataloader(self) -> DataLoader:
        """Return val data loader."""
        return self.val_loader

    def test_dataloader(self) -> DataLoader:
        """Return test data loader."""
        return self.test_loader
