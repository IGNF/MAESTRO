"""Code adapted from Clay Foundation Model: https://github.com/Clay-foundation/model."""

import copy
from typing import Literal

import torch

from conf.datasets import DatasetsConfig
from maestro.baselines.croma import CROMABaseline
from maestro.baselines.dinov2 import Dinov2Baseline
from maestro.baselines.dofa import DOFABaseline
from maestro.train.base import BaseModule

torch.set_float32_matmul_precision(precision="medium")

NDIM_RASTER = 5  # (batch, dates, channels, height, width)


class BaselineModule(BaseModule):
    """Baseline module for fine-tuning."""

    def __init__(
        self,
        # shared args
        datasets: DatasetsConfig,
        model: Literal["dinov2", "dofa", "croma"],
        model_size: Literal["small", "base", "large"],
        type_head: Literal["linear", "attentive"] = "linear",
        loss: Literal["l1", "l2", "l1_norm", "l2_norm"] = "l2_norm",
        weight_source: str = "imagenat",
        pretrained_path: str | None = None,
        freeze: bool = False,
        use_ema: bool = False,
        multimodal: Literal["shared", "monotemp", "croma-intergroup"] = "shared",
        keep_norm: bool = True,
        add_date_enc: bool = True,
    ) -> None:
        super().__init__(datasets, model, loss)

        match model:
            case "dinov2":
                unpool_dim = None

                if len(datasets.filter_finetune) > 1:
                    msg = "Too many datasets given."
                    raise NotImplementedError(msg)

                model_args = {
                    "datasets": datasets,
                    "type_head": type_head,
                    "backbone_size": model_size,
                    "unpool_dim": unpool_dim,
                    "weight_source": weight_source,
                    "pretrained_path": pretrained_path,
                    "freeze": freeze,
                    "multimodal": multimodal,
                    "keep_norm": keep_norm,
                    "add_date_enc": add_date_enc,
                }

                self.model = Dinov2Baseline(**model_args)

            case "dofa" | "croma":
                unpool_dim = None

                if len(datasets.filter_finetune) > 1:
                    msg = "Too many datasets given."
                    raise NotImplementedError(msg)

                model_args = {
                    "datasets": datasets,
                    "type_head": type_head,
                    "backbone_size": model_size,
                    "unpool_dim": unpool_dim,
                    "freeze": freeze,
                    "pretrained_path": pretrained_path,
                    "multimodal": multimodal,
                    "keep_norm": keep_norm,
                    "add_date_enc": add_date_enc,
                }

                model_dict = {"dofa": DOFABaseline, "croma": CROMABaseline}
                self.model = model_dict[model](**model_args)

            case _:
                msg = f"Invalid model name {model}. Not implemented"
                raise ValueError(msg)

        if use_ema:
            self.ema_model = copy.deepcopy(self.model).to("cpu")
            for param in self.ema_model.parameters():
                param.requires_grad = False
        else:
            self.ema_model = None

        self.dataset = datasets.dataset
        self.tb_logger = None
        self.save_hyperparameters(ignore=["datasets"])

    def configure_optimizers(self) -> dict:
        """Configure optimizer."""
        total_steps = self.trainer.estimated_stepping_batches
        total_batch_size = (
            self.trainer.train_dataloader.batch_size
            * self.trainer.accumulate_grad_batches
            * self.trainer.num_nodes
            * self.trainer.num_devices
            / 3.0  # remain iso with existing runs on Jean Zellou
        )

        lr = (
            # use sqrt scaling rule with Adam
            self.trainer.base_lr * total_batch_size**0.5
        )
        match self.trainer.lw_decay:
            case None:
                grouped_params = self.model.parameters()
                max_lr = lr
            case _:
                grouped_params = self.model.grouped_parameters(
                    lr,
                    rate_decay=self.trainer.lw_decay,
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
