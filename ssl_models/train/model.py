"""Code adapted from Clay Foundation Model: https://github.com/Clay-foundation/model."""

import copy
from typing import Literal

import torch

from conf.datasets import DatasetsConfig
from conf.mask import MaskConfig
from ssl_models.ssl.mae import MAE, mae_large, mae_medium, mae_small, mae_tiny
from ssl_models.train.base import BaseModule

torch.set_float32_matmul_precision(precision="medium")

NDIM_RASTER = 5  # (batch, dates, channels, height, width)


class SSLModule(BaseModule):
    """SSL module."""

    def __init__(  # noqa: PLR0913
        self,
        # shared args
        datasets: DatasetsConfig,
        mask: MaskConfig,
        multimodal: Literal["msgfm", "shared", "monotemp", "mod", "group"],
        allmods_depth: int,
        model: Literal["mae"],
        model_size: Literal["tiny", "small", "base", "medium", "large", "huge"],
        type_head: Literal["linear", "attentive"] = "linear",
        loss: Literal["l1", "l2", "l1_norm", "l2_norm"] = "l2_norm",
        unpool_dim: int | None = None,
        use_date_enc: bool = True,
        batch_repeats: int | None = None,
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

        if allmods_depth and multimodal not in ("mod", "group"):
            msg = (
                "Simultaneous encoding of all mods"
                f" not yet compatible with multimodal choice: {multimodal}."
            )
            raise NotImplementedError(msg)

        if model_size in model_map:
            model_args = {
                "datasets": datasets,
                "mask": mask,
                "multimodal": multimodal,
                "allmods_depth": allmods_depth,
                "model": model,
                "unpool_dim": unpool_dim,
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
        self.batch_repeats = batch_repeats
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
                if not isinstance(self.model, MAE):
                    msg = "Layer-wise decay only implemented for MAE."
                    raise NotImplementedError(msg)

                layer_parameters = {}
                for name_mod, encoder in self.model.encoder.items():
                    if name_mod == "all":
                        continue
                    for idx, layer in enumerate(encoder.layers):
                        if idx not in layer_parameters:
                            layer_parameters[idx] = []
                        layer_parameters[idx].extend(layer.parameters())

                if "all" in self.model.encoder:
                    for layer in self.model.encoder["all"].layers:
                        layer_parameters[len(layer_parameters)] = list(
                            layer.parameters(),
                        )

                num_layers = len(layer_parameters)
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
                            "params": layer_parameters[idx],
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
        if stage == "train" or self.ema_model is None:
            model = self.model
        else:
            model = self.ema_model
        loss_rec, log_inputs, log_preds, log_targets, _ = model(
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
