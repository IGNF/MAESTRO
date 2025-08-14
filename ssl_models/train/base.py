"""Code adapted from Clay Foundation Model: https://github.com/Clay-foundation/model."""

from abc import ABC
from typing import Literal

import torch
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import Module
from torch.nn import functional as F  # noqa: N812

from conf.datasets import DatasetsConfig
from ssl_models.ssl.mae import MAE

NDIM_RASTER = 5  # (batch, dates, channels, height, width)


class BaseModule(LightningModule, ABC):
    """Lightning module skeleton."""

    def __init__(
        self,
        # shared args
        datasets: DatasetsConfig,
        model: Module,
        loss: Module,
    ) -> None:
        super().__init__()
        self.datasets = datasets
        self.model = model
        self.loss = loss

    def configure_optimizers(self) -> dict:
        """Configure optimizer."""
        total_steps = self.trainer.estimated_stepping_batches
        lr = (
            # use sqrt scaling rule with Adam
            self.trainer.base_lr * self.trainer.train_dataloader.batch_size**0.5
        )
        match self.trainer.lw_decay:
            case None:
                grouped_params = self.model.parameters()
                max_lr = lr
            case _:
                if not isinstance(self.model, MAE):
                    msg = "Layer-wise decay only implemented for MAE."
                    raise NotImplementedError(msg)

                encoder = next(iter(self.model.encoder.values()))
                num_layers = len(encoder.layers)

                grouped_params = []
                grouped_params.append(
                    {
                        "params": self.model.patch_embed.parameters(),
                        "lr": lr * self.trainer.lw_decay ** (num_layers + 1),
                        "name": "layer_0_embed",
                    },
                )
                for idx in range(num_layers):
                    grouped_params.append(
                        {
                            "params": [
                                param
                                for encoder in self.model.encoder.values()
                                for param in encoder.layers[idx].parameters()
                            ],
                            "lr": lr * self.trainer.lw_decay ** (num_layers - idx),
                            "name": f"layer_{idx+1}",
                        },
                    )
                grouped_params.append(
                    {
                        "params": self.model.heads.parameters(),
                        "lr": lr,
                        "name": f"layer_{num_layers+1}_head",
                    },
                )
                max_lr = [param["lr"] for param in grouped_params]

        optimizer = torch.optim.AdamW(
            grouped_params,
            lr=lr,
            weight_decay=self.trainer.wd,
            betas=(self.trainer.b1, self.trainer.b2),
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=0.2,
            cycle_momentum=False,
            div_factor=1000,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "name": f"{self.trainer.ssl_phase}_AdamW_lr",
            },
        }

    def log_metric(
        self,
        name: str,
        value: Tensor,
    ) -> None:
        """Log accuracy metric."""
        super().log(
            name=name,
            value=value,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

    def log_step(
        self,
        name: str,
        value: Tensor,
        ssl_phase: Literal["pretrain", "probe", "finetune"],
        stage: Literal["train", "val", "test"],
    ) -> None:
        """Log metric."""
        if stage != "train":
            return
        super().log(
            name=f"{ssl_phase}_{name}/step_{stage}",
            value=value,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

    def probe_or_finetune_step(
        self,
        batch: dict[str, torch.Tensor],
        stage: Literal["train", "val", "test"],
    ) -> dict[str, torch.Tensor]:
        """Probing/finetuning step."""
        if stage == "train" or self.ema_model is None:
            model = self.model
        else:
            model = self.ema_model
        _, log_inputs, log_preds, log_targets, loss_pred = model(
            batch,
            stage=stage,
            ssl_phase=self.trainer.ssl_phase,
        )

        return {
            "loss": loss_pred,
            "log_inputs": log_inputs,
            "log_preds": log_preds,
            "log_targets": log_targets,
        }

    def shared_step(
        self,
        batch: dict[str, torch.Tensor],
        stage: Literal["train", "val", "test"],
    ) -> dict[str, torch.Tensor]:
        """Shared step for training/validation."""
        # resize
        for name_mod, mod in self.dataset.inputs.items():
            batch[name_mod] = F.interpolate(
                batch[name_mod].flatten(0, 1),
                size=(mod.image_size,) * 2,
                mode="nearest",
            ).unflatten(0, (-1, mod.num_dates))
            batch[f"{name_mod}_dates"] = (
                F.interpolate(
                    batch[f"{name_mod}_dates"].float().flatten(0, 1),
                    size=(mod.image_size,) * 2,
                    mode="nearest",
                ).unflatten(0, (-1, mod.num_dates))
                if batch[f"{name_mod}_dates"].ndim == NDIM_RASTER
                else batch[f"{name_mod}_dates"].float()
            )

        match self.trainer.ssl_phase:
            case "pretrain":
                if self.batch_repeats:
                    batch = {
                        name_mod: torch.repeat_interleave(
                            batch[name_mod],
                            self.batch_repeats,
                            dim=0,
                        )
                        for name_mod in batch
                    }
                return self.pretrain_step(batch, stage)
            case "probe" | "finetune":
                return self.probe_or_finetune_step(batch, stage)
            case _:
                msg = (
                    f"Invalid ssl phase {self.trainer.ssl_phase}."
                    " Expected 'pretrain' or 'probe' or 'finetune'"
                )
                raise ValueError(msg)

    def training_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,  # noqa: ARG002
    ) -> dict[str, torch.Tensor]:
        return self.shared_step(batch, stage="train")

    def validation_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,  # noqa: ARG002
    ) -> dict[str, torch.Tensor]:
        return self.shared_step(batch, stage="val")

    def test_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,  # noqa: ARG002
    ) -> dict[str, torch.Tensor]:
        return self.shared_step(batch, stage="test")

    def on_train_epoch_end(self) -> None:
        if self.ema_model is not None:
            self.update_ema()

    def update_ema(self) -> None:
        """Update EMA model."""
        momentum = 1 - 1 / (self.trainer.max_epochs * 0.2)
        for param, param_ema in zip(
            self.model.parameters(),
            self.ema_model.parameters(),
        ):
            param_ema.data.mul_(momentum).add_((1.0 - momentum) * param.detach().data)
