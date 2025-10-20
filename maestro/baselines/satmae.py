"""SatMAE module.

References.
MAE: https://github.com/facebookresearch/mae
timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
DeiT: https://github.com/facebookresearch/deit
"""

from functools import partial
from typing import Literal

import numpy as np
import torch
from timm.models.vision_transformer import Block, PatchEmbed
from torch import Tensor, nn

from conf.datasets import DatasetsConfig
from maestro import LOGGER
from maestro.baselines.base import BaseModule
from maestro.baselines.utils import interpolate_pos_encoding

BASE_IMAGE_SIZE = 224
ORIG_BANDS = (0, 1, 2)  # S2 bands included in original model


class SatMAEBaseline(BaseModule):
    """Baseline model based on SatMAE."""

    def __init__(
        self,
        datasets: DatasetsConfig,
        backbone_size: str = "base",
        freeze: bool = False,
        pretrained_path: str | None = None,
        type_head: Literal["linear", "attentive"] = "attentive",
        interpolate: Literal["nearest", "bilinear", "bicubic"] = "nearest",
        fusion_mode: Literal["mod"] = "mod",
        add_date_enc: bool = True,
        fac_date_enc: float = 1.0,
        date_dim: int = 8,
        keep_norm: bool = True,
        **kwargs,  # noqa: ANN003
    ) -> None:
        """Create the SatMAE encoder model.

        Parameters
        ----------
        datasets: DatasetsConfig
            The dataset config used in the probing/finetuning phase.
        backbone_size: str
            Backbone size to use. To choose in "base", "large".
        freeze: bool
            Whether to freeze the backbone.
        pretrained_path: str
            Path to the location of the pretrained weights.
        type_head: str
            Segmentation head to use. Either "linear" or "attentive".
        interpolate: str
            Interpolation used in the image resizing before patchification.
        fusion_mode: str
            Fusion strategy. Necessarily "mod".
        add_date_enc: bool
            Whether to add the date encodings to the embeddings.
        fac_date_enc: float
            Factor used to compute the date encodings.
        date_dim: int
            Dimension of the date embeddings.
        keep_norm: bool
            Whether to keep the final layernorm layer.
        kwargs:
            Arguments to pass to `BaseModel` constructor.

        """
        self.backbone_size = backbone_size
        self.freeze = freeze
        self.type_head = type_head
        self.interpolate = interpolate
        self.fusion_mode = fusion_mode

        self.pretrained_path = pretrained_path
        self.add_date_enc = add_date_enc
        self.fac_date_enc = fac_date_enc
        self.date_dim = date_dim
        self.keep_norm = keep_norm

        self.patch_size = 16
        match backbone_size:
            case "base":
                self.encoder_dim = 768
            case "large":
                self.encoder_dim = 1024
            case _:
                msg = "Backbone's size should be `base` or `large`"
                raise ValueError(msg)

        self.std = 0.01
        bands = datasets.dataset.inputs["s2"].bands
        self.bands = (
            list(range(bands))
            if isinstance(bands, int)
            else [idx for band in bands for idx in band]
        )
        self.num_channels = len(self.bands)
        self.image_size = datasets.dataset.inputs["s2"].image_size

        super().__init__(
            datasets,
            self.fusion_mode,
            self.patch_size,
            self.encoder_dim,
            self.type_head,
            self.interpolate,
            self.add_date_enc,
            self.fac_date_enc,
            self.date_dim,
            self.keep_norm,
            **kwargs,
        )

        self._build_backbone(backbone_size)
        if pretrained_path is not None:
            self._load_weights(pretrained_path=pretrained_path)

        self._transfer_patch_embedding_to_more_channels()

        if freeze:
            self.freeze_backbone()

    def _load_weights(self, pretrained_path: str) -> None:
        state_dict = torch.load(
            pretrained_path,
            map_location="cpu",
            weights_only=False,
        )["model"]

        if self.image_size != BASE_IMAGE_SIZE:
            interpolated_pos_embed = interpolate_pos_encoding(
                state_dict["pos_embed"],
                self.image_size,
                self.image_size,
                self.patch_size,
            )
            interpolated_decoder_pos_embed = interpolate_pos_encoding(
                state_dict["decoder_pos_embed"],
                self.image_size,
                self.image_size,
                self.patch_size,
            )
            state_dict["decoder_pos_embed"] = torch.cat(
                interpolated_decoder_pos_embed,
                dim=1,
            )  # fuze cls and pos embed
            state_dict["pos_embed"] = torch.cat(
                interpolated_pos_embed,
                dim=1,
            )  # fuze cls and pos embed

        self.encoder.load_state_dict(state_dict, strict=True)
        LOGGER.info(f"Loaded pretrained weights from {pretrained_path}")
        del state_dict
        torch.cuda.empty_cache()

    def _build_backbone(self, backbone_size: str) -> None:
        if backbone_size == "base":
            self.encoder = mae_vit_base_patch16_dec512d8b(img_size=self.image_size)
        elif backbone_size == "large":
            self.encoder = mae_vit_large_patch16_dec512d8b(img_size=self.image_size)
        else:
            msg = f"Backbone size {backbone_size} not recognized."
            raise ValueError(msg)

    def _transfer_patch_embedding_to_more_channels(self) -> None:
        orig_weight = self.encoder.patch_embed.proj.weight
        new_weight = torch.zeros(
            (orig_weight.shape[0], self.num_channels, *orig_weight.shape[2:]),
        )
        nn.init.normal_(new_weight, std=self.std)
        orig_bands = [idx for idx, band in enumerate(ORIG_BANDS) if band in self.bands]
        new_bands = [self.bands.index(ORIG_BANDS[idx]) for idx in orig_bands]
        new_weight[:, new_bands] = orig_weight[:, orig_bands]

        # Replace
        new_proj = nn.Conv2d(
            in_channels=self.num_channels,
            out_channels=orig_weight.shape[0],
            kernel_size=self.encoder.patch_embed.proj.kernel_size,
            stride=self.encoder.patch_embed.proj.stride,
            padding=self.encoder.patch_embed.proj.padding,
            bias=True,
        )
        new_proj.weight.data = new_weight
        new_proj.bias.data = self.encoder.patch_embed.proj.bias.data.clone()
        self.encoder.patch_embed.proj = new_proj

    def freeze_backbone(self) -> None:
        """Freeze the backbone's parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False

    def grouped_parameters(
        self,
        **kwargs,  # noqa: ANN003
    ) -> list[dict]:
        """Skip implem."""

    def format_dates(self, dates: Tensor) -> Tensor:
        year = dates[:, :, 0:1]
        month = dates[:, :, 1:2]
        hour = dates[:, :, 2:3]

        return torch.stack([year, month, hour], dim=-1)

    def forward(
        self,
        batch: dict[str, Tensor],
        ssl_phase: Literal["pretrain", "probe", "finetune"],  # noqa: ARG002
    ) -> tuple[
        dict[str, Tensor] | None,
        None,
        None,
        dict[str, Tensor] | None,
    ]:
        """SatMAE forward pass.

        Parameters
        ----------
        batch: dict
            Dictionary containing the data.
        ssl_phase: str
            SSL phase (pretrain, probe, finetune). Only probing and finetuning
            are needed for baselines.

        Returns
        -------
        tuple: SatMAE's outputs.

        """
        batch = self.resize_and_rescale(batch)

        output_satmae = self.encoder.forward_encoder(
            batch["s2"],
            self.format_dates(batch["s2_dates"]),
        )

        x_enc = {}
        x_enc["s2"] = output_satmae

        logits = self.compute_logits(x_enc)
        return batch, None, None, logits


class MaskedAutoencoderViT(nn.Module):
    """Masked Autoencoder with VisionTransformer backbone."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
        mlp_ratio: float = 4.0,
        norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()

        # MAE encoder specifics
        self.patch_embed = PatchEmbed(
            (img_size,) * 2,
            patch_size,
            in_chans,
            embed_dim,
        )
        self.num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim - 384),
            requires_grad=False,
        )  # fixed sin-cos embedding

        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for _ in range(depth)
            ],
        )
        self.norm = norm_layer(embed_dim)

        # MAE decoder specifics
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, decoder_embed_dim - 192),
            requires_grad=False,
        )  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for _ in range(decoder_depth)
            ],
        )

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim,
            patch_size**2 * in_chans,
            bias=True,
        )  # decoder to patch

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.patch_embed.num_patches**0.5),
            cls_token=True,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            int(self.patch_embed.num_patches**0.5),
            cls_token=True,
        )
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0),
        )

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02)
        # as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_encoder(
        self,
        x: Tensor,
        timestamps: Tensor,
    ) -> Tensor:
        # embed patches
        B, D = x.shape[:2]  # noqa: N806
        x = torch.cat([self.patch_embed(x[:, idx]) for idx in range(D)], dim=1)

        ts_embed = torch.cat(
            [
                get_1d_sincos_pos_embed_from_grid_torch(
                    128,
                    timestamps.reshape(-1, 3)[:, 0].float(),
                ),
                get_1d_sincos_pos_embed_from_grid_torch(
                    128,
                    timestamps.reshape(-1, 3)[:, 1].float(),
                ),
                get_1d_sincos_pos_embed_from_grid_torch(
                    128,
                    timestamps.reshape(-1, 3)[:, 2].float(),
                ),
            ],
            dim=1,
        ).float()
        ts_embed = ts_embed.reshape(B, D, 1, ts_embed.shape[-1]).expand(
            -1,
            -1,
            self.num_patches,
            -1,
        )
        pos_embed = self.pos_embed[:, None, 1:, :].expand(B, D, -1, -1)

        # add pos embed w/o cls token
        x = x + torch.cat([pos_embed, ts_embed], dim=-1).flatten(1, 2)

        # append cls token
        cls_token = self.cls_token  # + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x[:, 1:, :]


def mae_vit_base_patch16_dec512d8b(**kwargs) -> MaskedAutoencoderViT:  # noqa: ANN003
    """Construct SatMAE Base."""
    return MaskedAutoencoderViT(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


def mae_vit_large_patch16_dec512d8b(**kwargs) -> MaskedAutoencoderViT:  # noqa: ANN003
    """Construct SatMAE Large."""
    return MaskedAutoencoderViT(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


def get_1d_sincos_pos_embed_from_grid_torch(embed_dim: int, pos: Tensor) -> Tensor:
    """Construct 1-d positional encodings."""
    if embed_dim % 2:
        msg = f"Embed dim {embed_dim} is not a multiple of 2"
        raise ValueError(msg)
    omega = torch.arange(embed_dim // 2, dtype=float, device=pos.device)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb.double()


def get_2d_sincos_pos_embed(
    embed_dim: int,
    grid_size: int,
    cls_token: bool = False,
) -> np.ndarray:
    """Construct 2-d positional encodings."""
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: tuple[int]) -> np.ndarray:
    """Construct 2-d positional encodings from grid."""
    if embed_dim % 2:
        msg = f"Embed dim {embed_dim} is not a multiple of 2"
        raise ValueError(msg)

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    return np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    """Construct 1-d positional encodings from grid."""
    if embed_dim % 2:
        msg = f"Embed dim {embed_dim} is not a multiple of 2"
        raise ValueError(msg)
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    return np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
