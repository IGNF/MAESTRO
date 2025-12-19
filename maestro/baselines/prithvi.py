"""Prithvi module."""

from typing import Literal

import torch
from terratorch import BACKBONE_REGISTRY
from torch import Tensor, nn
from torch.nn import Module

from maestro.baselines.base import BaseModule
from maestro.conf.datasets import DatasetsConfig

ORIG_BANDS = (0, 1, 2, 6, 8, 9)  # S2 bands included in original model


class PrithviBaseline(BaseModule):
    """Baseline model based on Prithvi."""

    def __init__(
        self,
        datasets: DatasetsConfig,
        backbone_size: Literal["base", "large"] = "large",
        version: str = "v2",
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
        """Create the Prithvi encoder model.

        Parameters
        ----------
        datasets: DatasetsConfig
            The dataset config used in the probing/finetuning phase.
        backbone_size: str
            Backbone size to use. To choose in "base", "large".
        version: str
            Prithvi version.
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
        self.version = version
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

        match self.backbone_size:
            case "base":
                self.encoder_dim = 768
                self.encoder_depth = 12
                self.num_heads = 16
                self.patch_size = 16
            case "large":
                self.encoder_dim = 1024
                self.encoder_depth = 24
                self.num_heads = 16
                self.patch_size = 16
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

        if not self._check_model_version_and_size():
            msg = (
                f"Invalid model version {self.version} and size {self.backbone_size}. "
                "Supported versions are: 'v1' for base and 'v2' for large."
            )
            raise ValueError(msg)

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

        self._build_backbone()
        self._transfer_patch_embedding_to_more_channels()

        if self.freeze:
            self.freeze_backbone()

    def _check_model_version_and_size(self) -> bool:
        model_assoc = [
            ("large", "v2", False),
            ("large", "v2", True),
            ("base", "v1", False),
        ]

        return (self.backbone_size, self.version, self.add_date_enc) in model_assoc

    def _build_backbone(self) -> Module:
        model_dict = {
            ("base", "v1", False): "terratorch_prithvi_eo_v1_100",
            ("large", "v2", False): "terratorch_prithvi_eo_v2_300",
            ("large", "v2", True): "terratorch_prithvi_eo_v2_300_tl",
        }

        self.encoder = BACKBONE_REGISTRY.build(
            model_dict[self.backbone_size, self.version, self.add_date_enc],
            pretrained=True,
            ckpt_path=self.pretrained_path,
            num_frames=self.num_dates["s2"],
        )

    def _transfer_patch_embedding_to_more_channels(self) -> None:
        orig_proj = self.encoder.patch_embed.proj
        orig_weight = orig_proj.weight  # shape: (out_dim, in_channels, 1, 16, 16)

        # Create new weight tensor
        new_weight = torch.zeros(
            (orig_weight.shape[0], self.num_channels, *orig_weight.shape[2:]),
        )

        # Copy existing channels
        nn.init.normal_(new_weight, std=self.std)
        orig_bands = [idx for idx, band in enumerate(ORIG_BANDS) if band in self.bands]
        new_bands = [self.bands.index(ORIG_BANDS[idx]) for idx in orig_bands]
        new_weight[:, new_bands] = orig_weight[:, orig_bands]

        # Build new conv3d
        new_proj = nn.Conv3d(
            in_channels=self.num_channels,
            out_channels=orig_weight.shape[0],
            kernel_size=orig_proj.kernel_size,
            stride=orig_proj.stride,
            padding=orig_proj.padding,
            bias=True,
        )

        # Assign weights & bias
        new_proj.weight.data = new_weight
        new_proj.bias.data = self.encoder.patch_embed.proj.bias.data.clone()
        self.encoder.patch_embed.proj = new_proj

    def freeze_backbone(self) -> None:
        """Freeze the backbone's parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False

    def grouped_parameters(
        self,
        base_lr: float,
        rate_decay: float = 0.75,
    ) -> list[dict]:
        """Skip implem."""

    def format_dates(self, dates: Tensor) -> Tensor:
        year = dates[:, :, 0]
        doy = dates[:, :, 1] / 365.25

        return torch.stack([year, doy], dim=-1)

    def _add_date_encodings(self, x: Tensor, temporal_coords: Tensor) -> Tensor:
        """Add date encodings to the embeddings."""
        if self.encoder.temporal_encoding and temporal_coords is not None:
            num_tokens_per_frame = x.shape[1] // self.encoder.num_frames

            temporal_encoding = self.encoder.temporal_embed_enc(
                temporal_coords,
                num_tokens_per_frame,
            )

            x = x + temporal_encoding

        return x

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
        """Prithvi v1/v2 forward pass.

        Parameters
        ----------
        batch: dict
            Dictionary containing the data.
        ssl_phase: str
            SSL phase (pretrain, probe, finetune). Only probing and finetuning
            are needed for baselines.

        Returns
        -------
        tuple: Prithvi's outputs.

        """
        batch = self.resize_and_rescale(batch)

        s2_input = batch["s2"].transpose(1, 2)  # (B, D, C, H, W) -> (B, C, D, H, W)

        if self.add_date_enc:
            features_prithvi = self.encoder(
                x=s2_input,
                temporal_coords=self.format_dates(batch["s2_dates"]),
            )
        else:
            features_prithvi = self.encoder(x=s2_input)

        output_prithvi = features_prithvi[-1]
        output_prithvi = output_prithvi[:, 1:, :]  # ignore cls token

        x_enc = {}
        x_enc["s2"] = output_prithvi

        logits = self.compute_logits(x_enc)
        return batch, None, None, logits
