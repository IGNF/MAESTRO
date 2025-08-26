"""Code adapted from Clay Foundation Model: https://github.com/Clay-foundation/model."""

import copy
from typing import Literal

import torch

from conf.datasets import DatasetsConfig
from conf.mask import MaskConfig
from maestro.ssl.mae import mae_large, mae_medium, mae_small, mae_tiny
from maestro.train.base import BaseModule

torch.set_float32_matmul_precision(precision="medium")


class SSLModule(BaseModule):
    """SSL module."""

    def __init__(
        self,
        # shared args
        datasets: DatasetsConfig,
        mask: MaskConfig,
        fusion_mode: Literal["msgfm", "shared", "monotemp", "mod", "group"],
        inter_depth: int,
        model: Literal["mae"],
        model_size: Literal["tiny", "small", "base", "medium", "large", "huge"],
        type_head: Literal["linear", "attentive"] = "linear",
        loss: Literal["l1", "l2", "l1_norm", "l2_norm"] = "l2_norm",
        use_date_enc: bool = True,
        use_ema: bool = False,
    ) -> None:
        super().__init__(datasets, model, loss)

        match loss:
            case "l1":
                norm_pix_loss = False
                loss_fn = torch.abs
            case "l2":
                norm_pix_loss = False
                loss_fn = torch.square
            case "l1_norm":
                norm_pix_loss = True
                loss_fn = torch.abs
            case "l2_norm":
                norm_pix_loss = True
                loss_fn = torch.square
            case _:
                msg = f"Invalid loss {loss}."
                raise ValueError(msg)

        match model:
            case "mae":
                num_levels = 1
                fac_abs_enc = 1.0  # abs pos encodings are necessary for MAE
                fac_date_enc = 1.0 if use_date_enc else 0.0

                model_map = {
                    "tiny": mae_tiny,
                    "small": mae_small,
                    "medium": mae_medium,
                    "large": mae_large,
                }
            case _:
                msg = f"Invalid model name {model}. Not implemented"
                raise ValueError(msg)

        if inter_depth and fusion_mode not in ("mod", "group"):
            msg = (
                "Simultaneous encoding of all mods"
                f" not yet compatible with fusion mode: {fusion_mode}."
            )
            raise NotImplementedError(msg)

        if model_size in model_map:
            model_args = {
                "datasets": datasets,
                "mask": mask,
                "fusion_mode": fusion_mode,
                "inter_depth": inter_depth,
                "model": model,
                "num_levels": num_levels,
                "type_head": type_head,
                "loss_fn": loss_fn,
                "norm_pix_loss": norm_pix_loss,
                "fac_abs_enc": fac_abs_enc,
                "fac_date_enc": fac_date_enc,
            }
            self.model = model_map[model_size](**model_args)
        else:
            msg = f"Invalid model size {model_size}. Expected one of {model_map.keys()}"
            raise ValueError(msg)

        if use_ema:
            self.ema_model = copy.deepcopy(self.model).to("cpu")
            for param in self.ema_model.parameters():
                param.requires_grad = False
        else:
            self.ema_model = None

        self.dataset = datasets.dataset
        self.save_hyperparameters(ignore=["datasets"])

    def configure_optimizers(self) -> dict:
        """Configure optimizer."""
        total_steps = self.trainer.estimated_stepping_batches
        total_batch_size = (
            self.trainer.train_dataloader.batch_size
            * self.trainer.accumulate_grad_batches
            * self.trainer.num_nodes
            * self.trainer.num_devices
            / 3.0  # remain iso with past runs
        )
        lr = (
            # use sqrt scaling rule with Adam
            self.trainer.base_lr * total_batch_size**0.5
        )

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=self.trainer.wd,
            betas=(self.trainer.b1, self.trainer.b2),
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            total_steps=total_steps,
            pct_start=0.2,
            cycle_momentum=False,
            div_factor=1000,
            final_div_factor=self.trainer.final_factor / 1000.0,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "name": f"{self.trainer.ssl_phase}_AdamW_lr",
            },
        }

    def pretrain_step(
        self,
        batch: dict[str, torch.Tensor],
        stage: Literal["train", "val", "test"],
    ) -> dict[str, torch.Tensor]:
        """Pretraining step."""
        loss_rec, log_inputs, log_preds, log_targets, _ = self.model(
            batch,
            ssl_phase=self.trainer.ssl_phase,
            stage=stage,
        )

        return {
            "loss": loss_rec,
            "log_inputs": log_inputs,
            "log_preds": log_preds,
            "log_targets": log_targets,
        }
