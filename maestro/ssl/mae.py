"""MAE module. Adapted from clay foundation model: https://github.com/Clay-foundation/model."""

from typing import Literal

import torch
from einops import rearrange
from torch import Tensor, nn
from vit_pytorch.vit import Transformer

from conf.datasets import DatasetsConfig
from conf.mask import MaskConfig
from maestro.ssl.mim import BaseMIM


class MAE(BaseMIM):
    """Masked Auto Encoder (MAE)."""

    def __init__(  # noqa: PLR0913
        self,
        datasets: DatasetsConfig,
        mask: MaskConfig,
        multimodal: Literal["msgfm", "shared", "monotemp", "mod", "group"],
        allmods_depth: tuple[int],
        model: Literal["mae"],
        num_levels: Literal[1],
        unpool_dim: int | None,
        embed_dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_ratio: float,
        decoder_dim: int,
        decoder_depth: int,
        decoder_heads: int,
        decoder_dim_head: int,
        decoder_mlp_ratio: float,
        type_head: Literal["linear", "attentive"] = "linear",
        loss_fn: Literal[torch.abs, torch.square] = torch.square,
        norm_pix_loss: bool = True,
        fac_abs_enc: float = 1.0,
        fac_date_enc: float = 1.0,
        **kwargs,  # noqa: ANN003, ARG002
    ) -> None:
        super().__init__(
            datasets=datasets,
            multimodal=multimodal,
            model=model,
            num_levels=num_levels,
            unpool_dim=unpool_dim,
            embed_dim=embed_dim,
            decoder_dim=decoder_dim,
            type_head=type_head,
            loss_fn=loss_fn,
            norm_pix_loss=norm_pix_loss,
            fac_abs_enc=fac_abs_enc,
            fac_date_enc=fac_date_enc,
        )

        self.len_bands = {
            name_mod: 1 if isinstance(mod.bands, int) else len(mod.bands)
            for name_mod, mod in datasets.dataset.inputs.items()
        }

        num_dates_mod, num_dates_group = {}, {}
        for name_mod, name_group in datasets.dataset.groups:
            mod = datasets.dataset.inputs[name_mod]
            num_dates = mod.num_dates * self.len_bands[name_mod]
            if name_mod not in num_dates_mod:
                num_dates_mod[name_mod] = 0
            if name_group not in num_dates_group:
                num_dates_group[name_group] = 0
            num_dates_mod[name_mod] += num_dates
            num_dates_group[name_group] += num_dates

        (
            self.mask_ratio,
            self.mask_mod,
            self.mask_bands,
            self.mask_dates,
            self.mask_loc,
        ) = (
            {},
            {},
            {},
            {},
            {},
        )
        match multimodal:
            case "msgfm" | "shared" | "monotemp":
                name_models = (
                    list(num_dates_mod.keys())
                    if multimodal == "monotemp"
                    else ["shared"]
                )
                for name_mod in num_dates_mod:
                    self.mask_ratio[name_mod] = mask.mask_ratio
                    self.mask_mod[name_mod] = None
                    self.mask_bands[name_mod] = None
                    self.mask_dates[name_mod] = None
                    self.mask_loc[name_mod] = None
            case "mod" | "group":
                name_models = (
                    list(num_dates_group.keys())
                    if multimodal == "group"
                    else list(num_dates_mod.keys())
                )
                for name_mod, name_group in datasets.dataset.groups:
                    mod = datasets.dataset.inputs[name_mod]
                    if multimodal == "group":
                        scale_fac = num_dates_group[name_group] ** mask.mask_scale
                        self.mask_ratio[name_group] = (
                            1 - (1 - mask.mask_ratio) / scale_fac
                        )
                        self.mask_mod[name_mod] = (
                            mask.mask_mod
                            if num_dates_mod[name_mod] != num_dates_group[name_group]
                            else None
                        )
                    else:
                        scale_fac = num_dates_mod[name_mod] ** mask.mask_scale
                        self.mask_ratio[name_mod] = (
                            1 - (1 - mask.mask_ratio) / scale_fac
                        )
                        self.mask_mod[name_mod] = None

                    self.mask_bands[name_mod] = (
                        mask.mask_bands if self.len_bands[name_mod] > 1 else None
                    )
                    self.mask_dates[name_mod] = (
                        mask.mask_dates if mod.num_dates > 1 else None
                    )
                    self.mask_loc[name_mod] = mask.mask_loc
            case _:
                msg = f"Invalid multimodal mode {multimodal}."
                raise ValueError(msg)

        self.encoder = nn.ModuleDict(
            {
                name_mod: Transformer(
                    dim=embed_dim,
                    depth=depth - allmods_depth,
                    heads=heads,
                    dim_head=dim_head,
                    mlp_dim=embed_dim * mlp_ratio,
                )
                for name_mod in name_models
            },
        )
        self.enc_to_dec = nn.ModuleDict(
            {
                name_mod: (
                    nn.Linear(embed_dim, decoder_dim)
                    if embed_dim != decoder_dim
                    else nn.Identity()
                )
                for name_mod in name_models
            },
        )
        self.decoder = nn.ModuleDict(
            {
                name_mod: Transformer(
                    dim=decoder_dim,
                    depth=decoder_depth,
                    heads=decoder_heads,
                    dim_head=decoder_dim_head,
                    mlp_dim=embed_dim * decoder_mlp_ratio,
                )
                for name_mod in name_models
            },
        )
        if allmods_depth:
            self.encoder_all = Transformer(
                dim=embed_dim,
                depth=allmods_depth,
                heads=heads,
                dim_head=dim_head,
                mlp_dim=embed_dim * mlp_ratio,
            )
        else:
            self.encoder_all = None

    def mask_struct(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        """Structural masking of modalities, band groups, dates."""
        mask_group, mask_mod = {}, {}
        for name_group in x:
            B, L, _ = x[name_group].shape  # noqa: N806
            mask_group[name_group] = torch.ones((B, L, 1), dtype=torch.bool)
            mask_mod[name_group] = torch.ones((B, L, 1), dtype=torch.bool)

        mask_mod = self.ungroup(mask_mod)

        while any(
            mask_group[name_group].all(dim=(1, 2)).any(dim=0)
            for name_group in mask_group
        ):
            mask = {}
            for name_mod, len_bands in self.len_bands.items():
                B, D, L, _ = mask_mod[name_mod].shape  # noqa: N806
                G = len_bands  # noqa: N806
                mask[name_mod] = torch.zeros(
                    (B, G, D // G, L),
                    dtype=torch.bool,
                )
                if self.mask_mod[name_mod]:
                    mask[name_mod] += torch.rand((B, 1, 1, 1)) < self.mask_mod[name_mod]
                if self.mask_bands[name_mod]:
                    mask[name_mod] += (
                        torch.rand((B, G, 1, 1)) < self.mask_bands[name_mod]
                    )
                if self.mask_dates[name_mod]:
                    mask[name_mod] += (
                        torch.rand((B, 1, D // G, 1)) < self.mask_dates[name_mod]
                    )
                if self.mask_loc[name_mod]:
                    mask[name_mod] += torch.rand((B, 1, 1, L)) < self.mask_loc[name_mod]
                mask[name_mod] = mask[name_mod].reshape((B, D, L, 1))

            mask = self.group(mask)

            for name_group in mask_group:
                mask_group[name_group] = torch.where(
                    mask_group[name_group].all(dim=1, keepdim=True),
                    mask[name_group],
                    mask_group[name_group],
                )

        return {
            name_group: mask_group[name_group].to(x[name_group].device)
            for name_group in mask_group
        }

    def mask_seq(
        self,
        x: Tensor,
        mask_token: Tensor,
        mask_rec: Tensor,
        name_group: str,
    ) -> tuple[Tensor, Tensor, None, Tensor]:
        """Mask sequence."""
        B, L, _ = x.shape  # noqa: N806

        # sample masked tokens
        noise = torch.rand((B, L), device=x.device)  # [B L]
        noise *= 1 - mask_rec.reshape((B, L)).float()
        random_indices = torch.argsort(noise, dim=-1)  # [B L]
        reverse_indices = torch.argsort(random_indices, dim=-1)

        num_masked_patches = round(
            self.mask_ratio[name_group] * L,
        )  # Number of patches to mask
        mask_rec = torch.zeros((B, L), dtype=torch.bool, device=x.device)
        mask_rec[:, :num_masked_patches] = True
        mask_rec = torch.gather(
            mask_rec,
            dim=1,
            index=reverse_indices,
        )
        mask_rec = mask_rec[:, :, None]

        # encode only unmasked tokens
        batch_indices = rearrange(torch.arange(B, device=x.device), "b -> b 1")
        masked_indices, _ = random_indices[:, :num_masked_patches].sort(dim=1)
        unmasked_indices, _ = random_indices[:, num_masked_patches:].sort(dim=1)

        x_enc = x[batch_indices, unmasked_indices, :]  # [B L:(1 - mask_ratio) C]
        mask_token = mask_token[batch_indices, masked_indices, :]  # [B L: mask_ratio C]

        return x_enc, mask_token, None, mask_rec

    def unmask_seq(self, x: Tensor, mask_token: Tensor, mask_rec: Tensor) -> Tensor:
        """Unmask sequence, filling with mask token."""
        B, L_enc, C = x.shape  # noqa: N806
        _, L, _ = mask_rec.shape  # noqa: N806

        # find masked and unmasked tokens
        mask_rec = mask_rec.float()
        mask_rec = mask_rec.squeeze(dim=2)
        mask_rec = mask_rec.argsort(dim=1, descending=True)
        masked_indices, unmasked_indices = torch.split(
            mask_rec,
            [L - L_enc, L_enc],
            dim=1,
        )
        unmasked_indices, _ = unmasked_indices.sort(dim=1)

        # collate encoded patches and mask tokens
        batch_indices = rearrange(torch.arange(B, device=x.device), "b -> b 1")
        x_dec = torch.zeros((B, L, C), dtype=x.dtype, device=x.device)
        x_dec[batch_indices, masked_indices, :] = mask_token.to(x.dtype)
        x_dec[batch_indices, unmasked_indices, :] = x
        return x_dec

    def encode(
        self,
        x: dict[str, Tensor],
        mask_attn: dict[str, None],  # noqa: ARG002
    ) -> dict[str, Tensor]:
        """Apply MAE encoder."""
        x = self.encode_or_decode(x, model=self.encoder)
        if self.encoder_all:
            x = self.encode_or_decode_all(x, model=self.encoder_all)
        return x

    def encoder_to_decoder(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        """Apply MAE encoder-to-decoder."""
        return self.encode_or_decode(x, model=self.enc_to_dec)

    def decode(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        """Apply MAE decoder."""
        return self.encode_or_decode(x, model=self.decoder)


def mae_tiny(**kwargs) -> MAE:  # noqa: ANN003
    """Construct MAE tiny."""
    args = {
        "embed_dim": 192,
        "depth": 12,
        "heads": 3,
        "dim_head": 64,
        "mlp_ratio": 2,
        "decoder_dim": 512,
        "decoder_depth": 1,
        "decoder_heads": 16,
        "decoder_dim_head": 32,
        "decoder_mlp_ratio": 4,
    }
    args.update(kwargs)
    return MAE(**args)


def mae_small(**kwargs) -> MAE:  # noqa: ANN003
    """Construct MAE small."""
    args = {
        "embed_dim": 384,
        "depth": 12,
        "heads": 6,
        "dim_head": 64,
        "mlp_ratio": 2,
        "decoder_dim": 512,
        "decoder_depth": 2,
        "decoder_heads": 16,
        "decoder_dim_head": 32,
        "decoder_mlp_ratio": 4,
    }
    args.update(kwargs)
    return MAE(**args)


def mae_medium(**kwargs) -> MAE:  # noqa: ANN003
    """Construct MAE base/medium."""
    args = {
        "embed_dim": 768,
        "depth": 12,
        "heads": 12,
        "dim_head": 64,
        "mlp_ratio": 4,
        "decoder_dim": 512,
        "decoder_depth": 3,
        "decoder_heads": 16,
        "decoder_dim_head": 32,
        "decoder_mlp_ratio": 4,
    }
    args.update(kwargs)
    return MAE(**args)


def mae_large(**kwargs) -> MAE:  # noqa: ANN003
    """Construct MAE large."""
    args = {
        "embed_dim": 1024,
        "depth": 24,
        "heads": 16,
        "dim_head": 64,
        "mlp_ratio": 4,
        "decoder_dim": 512,
        "decoder_depth": 4,
        "decoder_heads": 16,
        "decoder_dim_head": 32,
        "decoder_mlp_ratio": 4,
    }
    args.update(kwargs)
    return MAE(**args)
