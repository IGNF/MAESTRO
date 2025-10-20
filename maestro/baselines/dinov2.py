"""DinoV2 Module.

The DinoV2 framework: https://arxiv.org/abs/2304.07193
Pretrained weights('sat' in the code below) from an adaptation
on high resolution satellite images: https://arxiv.org/abs/2304.07213
"""

from typing import Literal

import torch
from torch import Tensor
from torch.nn import LayerNorm, Module, ModuleDict
from transformers import AutoConfig, Dinov2Config
from transformers.models.dinov2.modeling_dinov2 import Dinov2Embeddings, Dinov2Encoder

from conf.datasets import DatasetsConfig
from maestro.baselines.base import BaseModule
from maestro.baselines.utils import (
    filter_dict,
    get_imagenat_dinov2_state_dict,
    get_sat_dinov2_state_dict,
)

RGB_BANDS = 3


class Dinov2Baseline(BaseModule):
    """Baseline model based on DinoV2."""

    def __init__(
        self,
        datasets: DatasetsConfig,
        backbone_size: Literal["small", "base", "large", "huge"],
        freeze: bool = False,
        pretrained_path: str | None = None,
        weight_source: Literal["imagenat", "sat"] = "imagenat",
        type_head: Literal["linear", "attentive"] = "attentive",
        interpolate: Literal["nearest", "bilinear", "bicubic"] = "nearest",
        fusion_mode: Literal["shared", "monotemp"] = "shared",
        add_date_enc: bool = True,
        fac_date_enc: float = 1.0,
        date_dim: int = 8,
        keep_norm: bool = True,
        **kwargs,  # noqa: ANN003
    ) -> None:
        """Create the DINOv2 encoder model using transformers.

        Parameters
        ----------
        datasets: DatasetsConfig
            The dataset config used in the probing/finetuning phase.
        backbone_size: str
            Backbone size to use. To choose in "small", "base", "large", "huge".
        freeze: bool
            Whether to freeze the backbone.
        pretrained_path: str
            Path to the location of the pretrained weights.
        weight_source: str
            Either "imagenat" (default) or "sat".
        type_head: str
            Segmentation head to use. Either "linear" or "attentive".
        interpolate: str
            Interpolation used in the image resizing before patchification.
        fusion_mode: str
            Fusion strategy. Either "shared" or "monotemp".
        add_date_enc: bool
            Whether to add date encodings to the embeddings.
        fac_date_enc: float
            Factor used to compute the date encodings.
        date_dim: int
            Dimension of the date embeddings.
        keep_norm: bool
            Whether to keep the final layernorm layer.
        kwargs:
            Arguments to pass to `BaseModel` constructor.

        """
        self.dataset = datasets.dataset
        self.type_head = type_head
        self.interpolate = interpolate
        self.fusion_mode = fusion_mode
        self.weight_source = weight_source
        self.pretrained_path = pretrained_path
        self.backbone_size = backbone_size
        self.std = 0.01
        self.keep_norm = keep_norm
        self.add_date_enc = add_date_enc
        self.fac_date_enc = fac_date_enc
        self.date_dim = date_dim

        if self.weight_source == "imagenat":
            self.patch_size = 14
        else:
            self.patch_size = 16

        self.embed_dim_by_size = {
            "small": 384,
            "base": 768,
            "large": 1024,
            "huge": 1280,
        }

        self.depth_by_size = {
            "small": 12,
            "base": 12,
            "large": 24,
            "huge": 32,
        }

        self.embed_dim = self.embed_dim_by_size[self.backbone_size]
        self.depth = self.depth_by_size[self.backbone_size]

        if self.weight_source == "sat" and backbone_size != "large":
            msg = f"Weight source {weight_source} is not compatible \
                with backbone_size {backbone_size}"

            raise ValueError(msg)

        if self.weight_source == "sat" and self.pretrained_path is None:
            msg = "A weight path should be given."

            raise ValueError(msg)

        super().__init__(
            datasets,
            self.fusion_mode,
            self.patch_size,
            self.embed_dim,
            self.type_head,
            self.interpolate,
            self.add_date_enc,
            self.fac_date_enc,
            self.date_dim,
            self.keep_norm,
            **kwargs,
        )

        self._build_backbone(backbone_size, weight_source)

        if freeze:
            self.freeze_backbone()

        self.rgb = {
            name_mod: self.len_bands[name_mod] <= RGB_BANDS
            for name_mod in self.len_bands
        }

    def _get_base_config(self, backbone_size: str) -> dict[str, Tensor]:
        original_config = AutoConfig.from_pretrained(f"facebook/dinov2-{backbone_size}")
        config_dict = original_config.to_dict()
        config_dict["patch_size"] = self.patch_size
        config_dict["image_size"] = 224 if self.weight_source == "sat" else 518

        return config_dict

    def _build_patch_embed(
        self,
        config_dict: dict[str, Tensor],
        state_dict: dict[str, Tensor] | None = None,
    ) -> ModuleDict:
        patch_embed = ModuleDict()

        for name_mod in self.dataset.inputs:
            config_dict["num_channels"] = self.num_bands[name_mod][0]
            mod_config = Dinov2Config(**config_dict)

            patch_embed[name_mod] = Dinov2Embeddings(mod_config)

            if state_dict is not None:
                self._transfer(
                    state_dict,
                    patch_embed[name_mod],
                    config_dict["num_channels"],
                    "embeddings",
                )

        return patch_embed

    def _build_encoder(
        self,
        config_dict: dict[str, Tensor],
        state_dict: dict[str, Tensor] | None = None,
    ) -> ModuleDict:
        encoder = ModuleDict()

        model_names = (
            self.dataset.inputs if self.fusion_mode != "shared" else ["shared"]
        )
        model_config = Dinov2Config(**config_dict)

        for name_mod in model_names:
            encoder[name_mod] = Dinov2Encoder(model_config)

            if state_dict is not None:
                self._transfer(state_dict, encoder[name_mod], RGB_BANDS, "encoder")

        return encoder

    def _build_layer_norm(
        self,
        config_dict: dict[str, Tensor],
        state_dict: dict[str, Tensor] | None = None,
    ) -> LayerNorm:
        layernorm = LayerNorm(
            config_dict["hidden_size"],
            eps=config_dict["layer_norm_eps"],
        )

        if state_dict is not None:
            self._transfer(state_dict, layernorm, RGB_BANDS, "layernorm")

        return layernorm

    def _transfer(
        self,
        state_dict: dict[str, Tensor],
        module: Module,
        num_channels: int,
        headname: str,
    ) -> None:
        if num_channels > RGB_BANDS:
            adapted_state_dict = self._transfer_patch_embedding_to_more_channels(
                state_dict,
                num_channels,
                self.std,
            )
            module.load_state_dict(
                filter_dict(adapted_state_dict, headname),
                strict=False,
            )
        else:
            module.load_state_dict(filter_dict(state_dict, headname), strict=False)

    def _build_backbone(
        self,
        backbone_size: Literal["small", "base", "large", "huge"],
        weight_source: Literal["imagenat", "sat", "random"],
    ) -> None:
        # config dict
        config_dict = self._get_base_config(backbone_size)

        # state dict
        if weight_source == "imagenat":
            state_dict = get_imagenat_dinov2_state_dict(backbone_size)
        elif weight_source == "sat":
            state_dict = get_sat_dinov2_state_dict(self.pretrained_path)
        else:  # random initialization
            state_dict = None

        # patch embedding layers (per modality)
        self.patch_embed = self._build_patch_embed(config_dict, state_dict=state_dict)
        # encoder
        self.encoder = self._build_encoder(config_dict, state_dict=state_dict)
        # layer norm
        if self.keep_norm:
            self.layernorm = self._build_layer_norm(config_dict, state_dict=state_dict)

    def _transfer_patch_embedding_to_more_channels(
        self,
        state_dict: dict[str, Tensor],
        num_channels: int,
        std: float,
    ) -> dict[str, Tensor]:
        weight_key = "embeddings.patch_embeddings.projection.weight"

        with torch.no_grad():
            new_state_dict = state_dict.copy()
            old_weights = new_state_dict[weight_key]
            bs, _, hs, ws = old_weights.shape

            means = torch.zeros((bs, num_channels - RGB_BANDS, hs, ws))
            stds = torch.ones((bs, num_channels - RGB_BANDS, hs, ws)) * std

            pad_tensor = torch.normal(means, stds)
            new_weights = torch.cat((old_weights, pad_tensor), dim=1)
            new_state_dict[weight_key] = new_weights

        return new_state_dict

    def freeze_backbone(
        self,
        ssl_phase: Literal["probe", "finetune"] = "finetune",
    ) -> None:
        """Freeze the backbone's parameters.

        Parameters
        ----------
        ssl_phase: str
            SSL phase (pretrain, probe, finetune). Only probing and finetuning
            are needed for baselines.

        Returns
        -------
        None

        """
        if ssl_phase == "finetune":
            for name_mod in self.patch_embed:
                # Necessary to unfreeze patchify layer if there are more than 3 channels
                if self.rgb[name_mod]:
                    for param in self.patch_embed[name_mod].parameters():
                        param.requires_grad = False

        for name_mod in self.encoder:
            for param in self.encoder[name_mod].parameters():
                param.requires_grad = False

        if self.keep_norm:
            for param in self.layernorm.parameters():
                param.requires_grad = False

    def grouped_parameters(
        self,
        base_lr: float,
        rate_decay: float = 0.75,
    ) -> list[dict]:
        """Return the grouped parameters dictionnary with layerwise learning rate.

        Parameters
        ----------
        base_lr: float
            Base learning rate.
        rate_decay: float
            Input's key.

        Returns
        -------
        List: A list of dictionnaries, containing "params" and "lr" keys, in which
        parameters are grouped.

        """
        lr_map = {}

        for name_mod in self.patch_embed:
            if self.rgb[name_mod]:  # RGB
                lr_map[f"patch_embed.{name_mod}"] = base_lr * rate_decay ** (
                    self.depth + 1
                )
            else:
                lr_map[f"patch_embed.{name_mod}"] = base_lr

        for name_mod in self.encoder:
            for i in range(self.depth):
                lr_map[f"encoder.{name_mod}.{i}"] = base_lr * rate_decay ** (
                    self.depth - i
                )

        lr_map["heads"] = base_lr

        params_dict = {}

        for name_mod in self.patch_embed:
            params_dict[f"patch_embed.{name_mod}"] = self.patch_embed[
                name_mod
            ].parameters()

        for name_mod in self.encoder:
            for i in range(self.depth):
                params_dict[f"encoder.{name_mod}.{i}"] = (
                    self.encoder[name_mod].layer[i].parameters()
                )

        if self.keep_norm:
            lr_map["layernorm"] = base_lr
            params_dict["layernorm"] = self.layernorm.parameters()

        params_dict["heads"] = self.heads.parameters()

        grouped_parameters = []
        for k, lr in lr_map.items():
            grouped_parameters.append({"params": params_dict[k], "lr": lr, "name": k})

        return grouped_parameters

    def forward(
        self,
        batch: dict[str, Tensor],
        ssl_phase: Literal["probe", "finetune"],  # noqa: ARG002
    ) -> tuple[
        dict[str, Tensor] | None,
        None,
        None,
        dict[str, Tensor] | None,
    ]:
        """Dinov2 forward pass.

        Parameters
        ----------
        batch: dict
            Dictionary containing the data.
        ssl_phase: str
            SSL phase (pretrain, probe, finetune). Only probing and finetuning
            are needed for baselines.

        Returns
        -------
        tuple: DinoV2's outputs.

        """
        batch = self.resize_and_rescale(batch)

        group_batch = self.group(batch)

        x_enc, dates = {}, {}
        for name_mod in self.dataset.inputs:
            encoder_mod = "shared" if self.fusion_mode == "shared" else name_mod

            x_tokenized = self.patch_embed[name_mod](group_batch[name_mod])
            x_mod = self.encoder[encoder_mod](x_tokenized)["last_hidden_state"]

            if self.keep_norm:
                x_mod = self.layernorm(x_mod)

            x_mod = x_mod[:, 1:]  # ignore CLS token

            x_enc[name_mod] = x_mod
            dates[name_mod] = batch[f"{name_mod}_dates"]

        if self.add_date_enc:
            ref_date = batch["ref_date"]
            x_enc = self._add_date_encodings(x_enc, dates, ref_date)

        logits = self.compute_logits(x_enc)
        return batch, None, None, logits
