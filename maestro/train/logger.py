"""Logger module."""

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from torch import Tensor

RGB_BANDS = 3


class TensorBoardLogger(TensorBoardLogger):
    """Custom tensorboard logger."""

    def __init__(
        self,
        ssl_phase: Literal["pretrain", "probe", "finetune"],
        save_dir: str,
        name: str,
        version: str,
        default_hp_metric: bool,
    ) -> None:
        self.ssl_phase = ssl_phase
        super().__init__(
            save_dir=save_dir,
            name=name,
            version=version,
            default_hp_metric=default_hp_metric,
        )

    def add_image(self, tag: str, image: Tensor, epoch: int) -> None:
        """Add image to tensorboard logger."""
        self._experiment.add_image(tag, image, epoch)

    @rank_zero_only
    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log metrics, appending ssl phase info to epoch logging."""
        if "epoch" in metrics:
            metrics[f"{self.ssl_phase}_epoch"] = metrics["epoch"]
            metrics.pop("epoch", None)
        super().log_metrics(metrics, step)


class ImageLogger(Callback):
    """Log masked images, predicted images, and target images."""

    @staticmethod
    def to_numpy(x: Tensor) -> np.ndarray:
        """Move to CPU and convert to numpy."""
        x_npy = x.detach().cpu().numpy().astype(np.float32)
        return (
            x_npy[:3]
            if x_npy.shape[0] >= RGB_BANDS
            else x_npy.mean(axis=0, keepdims=True)
        )

    @rank_zero_only
    def on_batch_end(
        self,
        trainer: Trainer,
        outputs: dict[str, dict],
        batch: dict[str, Tensor],  # noqa: ARG002
        batch_idx: int,
        stage: Literal["train", "val", "test"],
    ) -> None:
        """Log images."""
        match stage:
            case "train":
                num_batches_per_epoch = len(trainer.train_dataloader)
            case "val":
                num_batches_per_epoch = len(trainer.val_dataloaders)
            case "test":
                num_batches_per_epoch = len(trainer.test_dataloaders)
        log_every_n_batches = round(
            num_batches_per_epoch / trainer.logged_images_per_epoch,
        )
        if batch_idx % log_every_n_batches:
            return
        epoch = trainer.current_epoch
        prefix_idx = batch_idx // log_every_n_batches

        for log_title, log_img in outputs["log_inputs"].items():
            trainer.tb_logger.add_image(
                log_title.replace("/", f"/{prefix_idx}"),
                ImageLogger.to_numpy(log_img),
                epoch,
            )
        for log_title, log_img in outputs["log_preds"].items():
            trainer.tb_logger.add_image(
                log_title.replace("/", f"/{prefix_idx}"),
                ImageLogger.to_numpy(log_img),
                epoch,
            )
        for log_title, log_img in outputs["log_targets"].items():
            trainer.tb_logger.add_image(
                log_title.replace("/", f"/{prefix_idx}"),
                ImageLogger.to_numpy(log_img),
                epoch,
            )

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,  # noqa: ARG002
        outputs: dict[str, Tensor],
        batch: dict[str, Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,  # noqa: ARG002
    ) -> None:
        """Log train images."""
        self.on_batch_end(
            trainer,
            outputs,
            batch,
            batch_idx,
            stage="train",
        )

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,  # noqa: ARG002
        outputs: dict[str, Tensor],
        batch: dict[str, Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,  # noqa: ARG002
    ) -> None:
        """Log validation images."""
        self.on_batch_end(
            trainer,
            outputs,
            batch,
            batch_idx,
            stage="val",
        )


class MetricsLogger(Callback):
    """Log metrics."""

    @staticmethod
    def cm2arr(cm: np.ndarray, normalized: bool = False) -> np.ndarray:
        if cm.shape[0] <= 10:  # noqa: PLR2004
            figsize = (10, 7)
        elif cm.shape[0] > 10 and cm.shape[0] <= 16:  # noqa: PLR2004
            figsize = (12, 9)
        elif cm.shape[0] > 16 and cm.shape[0] <= 20:  # noqa: PLR2004
            figsize = (16, 11)
        else:
            figsize = (21, 16)

        match normalized:
            case False:
                cm_df = pd.DataFrame(cm)
                fmt = ".0f"
            case True:
                cm_df = pd.DataFrame(
                    np.divide(
                        cm.astype(np.float32),
                        np.sum(cm, axis=1)[:, np.newaxis],
                        out=np.zeros_like(cm.astype(np.float32)),
                        where=(np.sum(cm, axis=1)[:, np.newaxis] != 0),
                    ),
                )
                fmt = ".3f"

        fig, _ = plt.subplots(figsize=figsize)
        sns.heatmap(cm_df, annot=True, fmt=fmt)
        rgba_buf = fig.canvas.buffer_rgba()
        (w, h) = fig.canvas.get_width_height()
        arr = (
            np.frombuffer(rgba_buf, dtype=np.uint8)
            .reshape((h, w, 4))
            .transpose((2, 0, 1))[:3]
        )
        plt.close()
        return arr

    def on_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        stage: Literal["train", "val", "test"],
    ) -> None:
        """Log epoch metrics for training/val/test."""
        match trainer.ssl_phase:
            case "pretrain":
                self.log_metric(
                    pl_module=pl_module,
                    name=f"{trainer.ssl_phase}_loss_rec/{stage}",
                    value=pl_module.metrics[f"loss_rec_{stage}"].compute(),
                )
                pl_module.metrics[f"loss_rec_{stage}"].reset()
            case "probe" | "finetune":
                self.log_metric(
                    pl_module=pl_module,
                    name=f"{trainer.ssl_phase}_loss_pred/{stage}",
                    value=pl_module.metrics[f"loss_pred_{stage}"].compute(),
                )
                pl_module.metrics[f"loss_pred_{stage}"].reset()

                for name_target in pl_module.dataset.targets:
                    metrics = pl_module.metrics[f"{name_target}_{stage}"].compute()
                    for name_metric, value in metrics.items():
                        if name_metric == "confusion_matrix":
                            cm = value.cpu().numpy()
                            self.log_confmat(
                                trainer=trainer,
                                name=f"confmat_{trainer.ssl_phase}_{name_target}_{stage}/confmat",
                                arr=MetricsLogger.cm2arr(cm, normalized=False),
                                cm=cm,
                            )
                            self.log_confmat(
                                trainer=trainer,
                                name=f"confmat_{trainer.ssl_phase}_{name_target}_{stage}/confmat_norm",
                                arr=MetricsLogger.cm2arr(cm, normalized=True),
                            )
                        else:
                            self.log_metric(
                                pl_module=pl_module,
                                name=f"{trainer.ssl_phase}_{name_target}/{name_metric}_{stage}",
                                value=value,
                            )
                    pl_module.metrics[f"{name_target}_{stage}"].reset()

    def on_train_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        self.on_epoch_end(trainer=trainer, pl_module=pl_module, stage="train")

    def on_validation_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        self.on_epoch_end(trainer=trainer, pl_module=pl_module, stage="val")

    def on_test_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        self.on_epoch_end(trainer=trainer, pl_module=pl_module, stage="test")

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: dict[str, Tensor],
        batch: dict[str, Tensor],  # noqa: ARG002
        batch_idx: int,  # noqa: ARG002
        dataloader_idx: int = 0,  # noqa: ARG002
    ) -> None:
        """Log train loss."""
        match trainer.ssl_phase:
            case "pretrain":
                name = "loss_rec"
            case "probe" | "finetune":
                name = "loss_pred"

        pl_module.log(
            name=f"{trainer.ssl_phase}_{name}/step_train",
            value=outputs["loss"] * trainer.accumulate_grad_batches,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

    @rank_zero_only
    def log_confmat(
        self,
        trainer: Trainer,
        name: str,
        arr: np.ndarray,
        cm: np.ndarray | None = None,
    ) -> None:
        trainer.tb_logger.add_image(name, arr, trainer.current_epoch)
        if cm is not None:
            np.save(f"{trainer.tb_logger.log_dir}/{name.split('/')[0]}.npy", cm)

    def log_metric(
        self,
        pl_module: LightningModule,
        name: str,
        value: Tensor,
    ) -> None:
        """Log accuracy metric."""
        pl_module.log(
            name=name,
            value=value,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
