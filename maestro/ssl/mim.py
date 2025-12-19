"""Generic MIM base module, inherited in specialized MIM modules."""

from abc import ABC, abstractmethod
from functools import partial, reduce
from math import gcd
from typing import Literal

import torch
import torch.nn.functional as F  # noqa: N812
from einops import rearrange
from torch import Tensor, nn

from maestro.conf.dataset.utils import RasterConfig
from maestro.conf.datasets import DatasetsConfig
from maestro.layers.embed import Patchify, Pixelify
from maestro.layers.head import ClassificationHead, PixelifyHead
from maestro.layers.utils import (
    encode_dates,
    group_mods,
    posemb_sincos_2d,
    reshape_encoding,
    ungroup_mods,
)


class BaseMIM(nn.Module, ABC):
    """Masked Auto Encoder (MAE)."""

    def __init__(
        self,
        datasets: DatasetsConfig,
        interpolate: Literal["nearest", "bilinear", "bicubic"],
        fusion_mode: Literal["shared", "monotemp", "mod", "group"],
        model: Literal["mae"],  # noqa: ARG002
        num_levels: Literal[1],
        embed_dim: int,
        decoder_dim: int,
        type_head: Literal["linear", "attentive"] = "attentive",
        fac_abs_enc: float = 1.0,
        fac_date_enc: float = 1.0,
        date_dim: int = 8,
    ) -> None:
        super().__init__()
        self.dataset = datasets.dataset
        self.stride = 2 ** (num_levels - 1)
        self.interpolate = interpolate

        # patchify and pixelify
        self.num_bands = {
            name_mod: (
                [mod.bands] if isinstance(mod.bands, int) else list(map(len, mod.bands))
            )
            for name_mod, mod in datasets.dataset.inputs.items()
        }
        self.len_bands = {
            name_mod: len(num_bands) for name_mod, num_bands in self.num_bands.items()
        }

        self.mod_embed = {}
        self.grid_size, self.out_grid_size = {}, {}
        self.patch_embed, self.embed_to_rec = nn.ModuleDict(), nn.ModuleDict()
        for name_mod, mod in self.dataset.inputs.items():
            name_embed = mod.name_embed if mod.name_embed else name_mod
            self.mod_embed[name_mod] = name_embed
            patch_size = mod.patch_size.mae
            self.grid_size[name_mod] = mod.image_size // patch_size
            self.out_grid_size[name_mod] = mod.image_size // (patch_size * self.stride)
            if name_embed in self.patch_embed:
                continue
            self.patch_embed[name_embed] = Patchify(
                mod.bands,
                embed_dim,
                patch_size,
            )
            self.embed_to_rec[name_embed] = Pixelify(
                decoder_dim,
                mod.bands,
                patch_size * self.stride,
            )

        # Fix the position encodings to sine & cosine functions
        if self.dataset.grid_pos_enc is not None:
            grid_pos_enc = self.dataset.grid_pos_enc
        else:
            grid_pos_enc = reduce(
                lambda a, b: a * b // gcd(a, b),
                self.grid_size.values(),
            )  # smallest common multiple
        self.register_buffer(
            name="enc_pos_encoding",
            tensor=posemb_sincos_2d(
                h=grid_pos_enc,
                w=grid_pos_enc,
                dim=embed_dim,
                date_dim=date_dim,
            ).float()
            * fac_abs_enc,
            persistent=False,
        )
        self.register_buffer(
            name="dec_pos_encoding",
            tensor=posemb_sincos_2d(
                h=grid_pos_enc,
                w=grid_pos_enc,
                dim=decoder_dim,
                date_dim=date_dim,
            ).float(),
            persistent=False,
        )
        # Freeze weights of positional encoding
        self.enc_pos_encoding = self.enc_pos_encoding.requires_grad_(
            requires_grad=False,
        )
        self.dec_pos_encoding = self.dec_pos_encoding.requires_grad_(
            requires_grad=False,
        )

        # functions to construct date encodings
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
        self.dec_date_encoding = {
            name_mod: partial(
                encode_dates,
                dim=decoder_dim,
                date_dim=date_dim,
                fac_date_enc=fac_date_enc,
                grid_size=self.out_grid_size[name_mod],
                len_bands=self.len_bands[name_mod],
            )
            for name_mod in self.dataset.inputs
        }

        # flattening/unflattening of date dimensions
        self.group = partial(
            group_mods,
            fusion_mode=fusion_mode,
            groups=self.dataset.groups,
        )
        self.ungroup = partial(
            ungroup_mods,
            fusion_mode=fusion_mode,
            groups=self.dataset.groups,
            num_dates={
                name_mod: mod.num_dates * self.len_bands[name_mod]
                for name_mod, mod in self.dataset.inputs.items()
            },
            grid_size=self.grid_size,
        )

        # mask tokens
        self.mask_token = nn.ParameterDict(
            {
                name_mod: nn.Parameter(
                    torch.randn(1, len_bands, 1, 1, decoder_dim),
                )
                for name_mod, len_bands in self.len_bands.items()
            },
        )

        # heads
        self.heads = nn.ModuleDict()
        for name_target, target in self.dataset.targets.items():
            if isinstance(target, RasterConfig):
                if self.dataset.ref_input is None:
                    msg = f"Ref input must be provided for raster target {name_target}"
                    raise ValueError(msg)
                target_image_size = round(
                    self.dataset.crop_meters / target.resolution_meters,
                )
                ref_grid_size = self.out_grid_size[self.dataset.ref_input]
                if target_image_size % ref_grid_size:
                    msg = (
                        f"Target image size {target_image_size} "
                        f"is not a multiple of ref input grid {ref_grid_size}"
                    )
                    raise ValueError(msg)
                self.heads[name_target] = PixelifyHead(
                    type_head,
                    embed_dim * self.stride,
                    target.num_classes,
                    target_image_size // ref_grid_size,
                )
            else:
                self.heads[name_target] = ClassificationHead(
                    type_head,
                    embed_dim * self.stride,
                    target.num_classes,
                )

    def embed(
        self,
        batch: dict[str, Tensor],
    ) -> tuple[
        dict[str, Tensor],
        dict[str, Tensor],
        dict[str, Tensor],
        Tensor,
    ]:
        """Embed patches and fetch dates."""
        x_enc, mask_token, dates = {}, {}, {}
        for name_mod in self.dataset.inputs:
            name_embed = self.mod_embed[name_mod]
            x_enc[name_mod] = self.patch_embed[name_embed](
                batch[name_mod],
            )
            B = x_enc[name_mod].shape[0]  # noqa: N806
            D = x_enc[name_mod].shape[1]  # noqa: N806
            G = self.len_bands[name_mod]  # noqa: N806
            L_out = x_enc[name_mod].shape[2] // self.stride**2  # noqa: N806
            mask_token[name_mod] = self.mask_token[name_mod].to(x_enc[name_mod].dtype)
            mask_token[name_mod] = (
                mask_token[name_mod].expand((B, -1, D // G, L_out, -1)).flatten(1, 2)
            )
            dates[name_mod] = batch[f"{name_mod}_dates"]

        return (
            self.group(x_enc),
            self.group(mask_token),
            dates,
            batch["ref_date"],
        )

    def enc_add_encodings(
        self,
        x_enc: dict[str, Tensor],
        dates: dict[str, Tensor],
        ref_date: Tensor,
    ) -> dict[str, Tensor]:
        """Add positional and date encodings before encoder."""
        x_enc = self.ungroup(x_enc)

        for name_mod in x_enc:
            pos_encoding = reshape_encoding(
                self.enc_pos_encoding,
                self.grid_size[name_mod],
            )
            date_encoding = self.enc_date_encoding[name_mod](
                dates=dates[name_mod],
                ref_date=ref_date,
            )
            x_enc[name_mod] = x_enc[name_mod] + pos_encoding + date_encoding

        return self.group(x_enc)

    def dec_add_encodings(
        self,
        x_dec: dict[str, Tensor],
        dates: dict[str, Tensor],
        ref_date: Tensor,
    ) -> dict[str, Tensor]:
        """Add positional and date encodings before decoder."""
        x_dec = self.ungroup(x_dec)

        for name_mod in x_dec:
            pos_encoding = reshape_encoding(
                self.dec_pos_encoding,
                self.out_grid_size[name_mod],
            )
            date_encoding = self.dec_date_encoding[name_mod](
                dates=dates[name_mod],
                ref_date=ref_date,
            )
            x_dec[name_mod] = x_dec[name_mod] + pos_encoding + date_encoding

        return self.group(x_dec)

    def mask(
        self,
        x: dict[str, Tensor],
        mask: dict[str, Tensor],
        ssl_phase: Literal["pretrain", "probe", "finetune"],
    ) -> tuple[
        dict[str, Tensor],
        dict[str, Tensor | None],
        dict[str, Tensor | None],
        dict[str, Tensor | None],
    ]:
        """Mask."""
        if ssl_phase == "pretrain":
            mask_rec = self.mask_struct(x)

            x_enc, mask_token, mask_attn = {}, {}, {}
            for name_group in x:
                (
                    x_enc[name_group],
                    mask_token[name_group],
                    mask_attn[name_group],
                    mask_rec[name_group],
                ) = self.mask_seq(
                    x[name_group],
                    mask[name_group],
                    mask_rec[name_group],
                    name_group,
                )
            return x_enc, mask_token, mask_attn, mask_rec

        # else, probe or finetune
        mask_token, mask_attn, mask_rec = ({name_group: None for name_group in x},) * 3
        return x, mask_token, mask_attn, mask_rec

    def unmask(
        self,
        x: dict[str, Tensor],
        mask: dict[str, Tensor],
        mask_rec: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        """Unmask."""
        x_dec = {}
        for name_group in x:
            x_dec[name_group] = self.unmask_seq(
                x[name_group],
                mask[name_group],
                mask_rec[name_group],
            )
        return x_dec

    def rec_pixels(
        self,
        x_dec: dict[str, Tensor],
        mask_rec: dict[str, Tensor],
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """Reconstruct pixels from embeddings."""
        x_dec, mask_rec = self.ungroup(x_dec), self.ungroup(mask_rec)

        pixels_rec = {}
        for name_mod in x_dec:
            name_embed = self.mod_embed[name_mod]
            pixels_rec[name_mod], mask_rec[name_mod] = self.embed_to_rec[name_embed](
                x_dec[name_mod],
                mask_rec[name_mod],
            )
        return pixels_rec, mask_rec

    def compute_logits(
        self,
        x_enc: dict[str, Tensor],
        ssl_phase: Literal["pretrain", "probe", "finetune"],
    ) -> dict[str, Tensor]:
        """Compute logits."""
        x_enc = self.ungroup(x_enc)

        ref_input = self.dataset.ref_input
        if ref_input is not None:
            x_ref = {}
            for name_mod in x_enc:
                D = x_enc[name_mod].shape[1]  # noqa: N806
                H = self.out_grid_size[name_mod]  # noqa: N806
                x_ref[name_mod] = rearrange(
                    x_enc[name_mod],
                    "b d (h w) c -> (b d) c h w",
                    h=H,
                )
                x_ref[name_mod] = F.interpolate(
                    x_ref[name_mod],
                    (self.out_grid_size[ref_input],) * 2,
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
                        ssl_phase=ssl_phase,
                    )
                case "multilabel_classif" | "classif":
                    logits[name_target] = self.heads[name_target](
                        x_enc,
                        ssl_phase=ssl_phase,
                    )
        return logits

    def encode_or_decode(
        self,
        x: dict[str, Tensor],
        model: nn.Module,
    ) -> dict[str, Tensor]:
        """Apply encoder or decoder."""
        for name_group in x:
            model_group = model[name_group] if name_group in model else model["shared"]
            x[name_group] = model_group(x[name_group])

        return x

    def encode_or_decode_all(
        self,
        x: dict[str, Tensor],
        model: nn.Module,
    ) -> dict[str, Tensor]:
        """Apply encoder or decoder on all modalities."""
        name_groups = list(x.keys())
        split_groups = tuple(x[name_group].shape[1] for name_group in name_groups)

        x_groups = torch.cat([x[name_group] for name_group in name_groups], dim=1)
        x_groups = model(x_groups)
        x_groups = x_groups.split(split_groups, dim=1)
        for name_group, x_group in zip(name_groups, x_groups):
            x[name_group] = x_group

        return x

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

    @abstractmethod
    def mask_struct(self, x: Tensor) -> dict[str, Tensor | None]:
        """Abstract method for structural masking."""

    @abstractmethod
    def mask_seq(
        self,
        x: Tensor,
        mask_token: Tensor,
        mask_rec: Tensor | None,
        name_group: str,
    ) -> tuple[Tensor, Tensor, Tensor | None, Tensor]:
        """Abstract method for masking of sequence."""

    @abstractmethod
    def unmask_seq(self, x: Tensor, mask_token: Tensor, mask_rec: Tensor) -> Tensor:
        """Abstract method for unmasking of sequence."""

    @abstractmethod
    def encode(
        self,
        x: dict[str, Tensor],
        mask_attn: dict[str, Tensor | None],
    ) -> dict[str, Tensor]:
        """Abstract method for encoding."""

    @abstractmethod
    def encoder_to_decoder(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        """Abstract method for encoder to decoder step."""

    @abstractmethod
    def decode(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        """Abstract method for decoding."""

    def forward(
        self,
        batch: dict[str, Tensor],
        ssl_phase: Literal["pretrain", "probe", "finetune"],
    ) -> tuple[
        dict[str, Tensor] | None,
        dict[str, Tensor] | None,
        dict[str, Tensor] | None,
        dict[str, Tensor] | None,
    ]:
        """Shared forward pass for all MIM modules."""
        batch = self.resize_and_rescale(batch)

        x_enc, mask_token, dates, ref_date = self.embed(batch)
        x_enc = self.enc_add_encodings(x_enc, dates, ref_date)

        x_enc, mask_token, mask_attn, mask_rec = self.mask(x_enc, mask_token, ssl_phase)

        x_enc = self.encode(x_enc, mask_attn)

        if ssl_phase == "pretrain":
            x_dec = self.encoder_to_decoder(x_enc)
            x_dec = self.unmask(x_dec, mask_token, mask_rec)
            x_dec = self.dec_add_encodings(x_dec, dates, ref_date)

            x_dec = self.decode(x_dec)

            pixels_rec, mask_rec = self.rec_pixels(x_dec, mask_rec)
            return batch, pixels_rec, mask_rec, None

        # else, probe or finetune
        logits = self.compute_logits(x_enc, ssl_phase)
        return batch, None, None, logits
