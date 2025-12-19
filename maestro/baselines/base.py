"""Base module for baseline FMs."""

from abc import ABC, abstractmethod
from functools import partial
from typing import Literal

import torch
import torch.nn.functional as F  # noqa: N812
from einops import rearrange
from torch import Tensor
from torch.nn import Module, ModuleDict

from maestro.conf.dataset.utils import RasterConfig
from maestro.conf.datasets import DatasetsConfig
from maestro.layers.head import ClassificationHead, PixelifyHead
from maestro.layers.utils import encode_dates, group_mods, ungroup_mods


class BaseModule(Module, ABC):
    """Base Module for baselines."""

    def __init__(
        self,
        datasets: DatasetsConfig,
        fusion_mode: Literal["shared", "monotemp", "mod", "late-croma", "inter-croma"],
        patch_size: int,
        embed_dim: int,
        type_head: Literal["linear", "attentive"] = "attentive",
        interpolate: Literal["nearest", "bilinear", "bicubic"] = "nearest",
        add_date_enc: bool = True,
        fac_date_enc: float = 1.0,
        date_dim: int = 8,
        keep_norm: bool = True,
        **kwargs,  # noqa: ANN003
    ) -> None:
        super().__init__(**kwargs)

        self.dataset = datasets.dataset
        self.interpolate = interpolate
        self.type_head = type_head
        self.patch_size = patch_size
        self.fusion_mode = fusion_mode
        self.keep_norm = keep_norm
        self.add_date_enc = add_date_enc

        self.num_bands = {
            name_mod: (
                [mod.bands] if isinstance(mod.bands, int) else list(map(len, mod.bands))
            )
            for name_mod, mod in datasets.dataset.inputs.items()
        }

        self.len_bands = {
            name_mod: len(num_bands) for name_mod, num_bands in self.num_bands.items()
        }
        self.grid_size = {}

        for name_mod, mod in self.dataset.inputs.items():
            self.grid_size[name_mod] = mod.image_size // self.patch_size

        self.num_dates = {
            name_mod: mod.num_dates * self.len_bands[name_mod]
            for name_mod, mod in self.dataset.inputs.items()
        }
        # heads
        self.heads = ModuleDict()
        for name_target, target in self.dataset.targets.items():
            if isinstance(target, RasterConfig):
                if self.dataset.ref_input is None:
                    msg = f"Ref input must be provided for raster target {name_target}"
                    raise ValueError(msg)
                target_image_size = round(
                    self.dataset.crop_meters / target.resolution_meters,
                )
                ref_grid_size = self.grid_size[self.dataset.ref_input]
                if target_image_size % ref_grid_size:
                    msg = (
                        f"Target image size {target_image_size} "
                        f"is not a multiple of ref input grid {ref_grid_size}"
                    )
                    raise ValueError(msg)
                self.heads[name_target] = PixelifyHead(
                    type_head,
                    embed_dim,
                    target.num_classes,
                    target_image_size // ref_grid_size,
                )
            else:
                self.heads[name_target] = ClassificationHead(
                    type_head,
                    embed_dim,
                    target.num_classes,
                )

        # flattening/unflattening of date dimensions
        self.group = partial(
            group_mods,
            fusion_mode=self.fusion_mode,
            groups=self.dataset.groups,
        )
        self.ungroup = partial(
            ungroup_mods,
            fusion_mode=self.fusion_mode,
            groups=self.dataset.groups,
            num_dates=self.num_dates,
            grid_size=self.grid_size,
        )

        self.enc_date_encoding = {
            name_mod: partial(
                encode_dates,
                dim=embed_dim,
                date_dim=date_dim,
                fac_date_enc=fac_date_enc,
                grid_size=self.grid_size[name_mod],
                len_bands=self.len_bands[name_mod],
            )
            for name_mod in self.dataset.inputs
        }

    @abstractmethod
    def _build_backbone(self) -> None:
        return

    @abstractmethod
    def freeze_backbone(self) -> None:
        return

    @abstractmethod
    def grouped_parameters(self, base_lr: float, rate_decay: float = 0.75) -> None:
        return

    def _add_date_encodings(
        self,
        x_enc: dict[str, Tensor],
        dates: dict[str, Tensor],
        ref_date: Tensor,
    ) -> dict[str, Tensor]:
        """Add positional and date encodings before encoder."""
        x_enc = self.ungroup(x_enc)
        dates_k = set(dates.keys())
        x_k = set(x_enc.keys())

        for name_mod in dates_k.intersection(x_k):
            date_encoding = self.enc_date_encoding[name_mod](
                dates=dates[name_mod],
                ref_date=ref_date,
            )
            x_enc[name_mod] = x_enc[name_mod] + date_encoding

        return self.group(x_enc)

    def compute_logits(
        self,
        x_enc: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        """Compute logits."""
        x_enc = self.ungroup(x_enc)

        ref_input = self.dataset.ref_input
        if ref_input is not None:
            x_ref = {}
            for name_mod in x_enc:
                D = x_enc[name_mod].shape[1]  # noqa: N806
                H = self.grid_size[name_mod]  # noqa: N806
                x_ref[name_mod] = rearrange(
                    x_enc[name_mod],
                    "b d (h w) c -> (b d) c h w",
                    h=H,
                )
                x_ref[name_mod] = F.interpolate(
                    x_ref[name_mod],
                    (self.grid_size[ref_input],) * 2,
                    mode="bilinear",
                )
                x_ref[name_mod] = rearrange(
                    x_ref[name_mod],
                    "(b d) c h w -> b d (h w) c",
                    d=D,
                )
            x_ref = torch.cat(
                [x_ref[name_mod] for name_mod in x_ref],
                dim=1,
            )

        x_enc = torch.cat(
            [x_enc[name_mod].flatten(1, 2) for name_mod in x_enc],
            dim=1,
        )
        logits = {}
        for name_target, target in self.dataset.targets.items():
            match target.type_target:
                case "segment":
                    logits[name_target] = self.heads[name_target](
                        x_ref,
                        ssl_phase="finetune",
                    )
                case "multilabel_classif" | "classif":
                    logits[name_target] = self.heads[name_target](
                        x_enc,
                        ssl_phase="finetune",
                    )
        return logits

    def resize_and_rescale(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Resize and/or rescale input rasters based on dataset config."""
        for name_mod, mod in self.dataset.inputs.items():
            batch[name_mod] = F.interpolate(
                batch[name_mod].flatten(0, 1),
                size=(mod.image_size,) * 2,
                mode=self.interpolate,
            ).unflatten(0, (-1, mod.num_dates))
            if mod.rescale_elev:
                batch[name_mod][:, :, 1:] = 30 * (
                    batch[name_mod][:, :, :1] - batch[name_mod][:, :, 1:]
                )
        return batch
