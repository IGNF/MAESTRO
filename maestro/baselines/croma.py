"""CROMA Module.

Model from https://papers.neurips.cc/paper_files/paper/2023/file/11822e84689e631615199db3b75cd0e4-Paper-Conference.pdf
"""

import itertools
import math
from typing import Literal

import torch
from einops import rearrange
from torch import Tensor, einsum, nn
from torch.nn import Module

from maestro.conf.datasets import DatasetsConfig
from maestro.baselines.base import BaseModule


class CROMABaseline(BaseModule):
    """Baseline model based on CROMA."""

    def __init__(
        self,
        datasets: DatasetsConfig,
        backbone_size: Literal["base", "large"],
        freeze: bool = False,
        pretrained_path: str | None = None,
        type_head: Literal["linear", "attentive"] = "attentive",
        interpolate: Literal["nearest", "bilinear", "bicubic"] = "nearest",
        fusion_mode: Literal["late-croma", "inter-croma"] = "inter-croma",
        add_date_enc: bool = True,
        fac_date_enc: float = 1.0,
        date_dim: int = 8,
        keep_norm: bool = True,
        **kwargs,  # noqa: ANN003
    ) -> None:
        """Create the CROMA encoder model.

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
            Fusion strategy. Either "late-croma" or "inter-croma".
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
                self.patch_size = 8
            case "large":
                self.encoder_dim = 1024
                self.encoder_depth = 24
                self.num_heads = 16
                self.patch_size = 8
            case _:
                msg = "Backbone's size should be `base` or `large`"
                raise ValueError(msg)

        self.image_size = self._check_image_size()

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

        if self.freeze:
            self.freeze_backbone()

        if self.fusion_mode == "inter-croma":
            ref_mod = "s2" if "s2" in self.dataset.inputs else "s1_asc"
            self.num_dates["joint"] = self.num_dates[ref_mod]
            self.grid_size["joint"] = self.grid_size[ref_mod]

    def _build_backbone(self) -> Module:
        modalities = self.dataset.inputs.keys()

        if "s2" in modalities and "s1_asc" in modalities:
            self.croma_modality = "both"
        elif "s2" in modalities:
            self.croma_modality = "optical"
        elif "s1_asc" in modalities:
            self.croma_modality = "SAR"
        else:
            raise NotImplementedError

        self.encoder = PretrainedCROMA(
            encoder_dim=self.encoder_dim,
            encoder_depth=self.encoder_depth,
            num_heads=self.num_heads,
            patch_size=self.patch_size,
            pretrained_path=self.pretrained_path,
            modality=self.croma_modality,
            image_resolution=self.image_size,
        )

    def _check_image_size(self) -> int:
        all_image_size = None
        used_mods = ["s2", "s1_des", "s1_asc"]

        for name_mod in self.dataset.inputs:
            if name_mod in used_mods:
                mod_image_size = self.dataset.inputs[name_mod].image_size

                if all_image_size is None:
                    all_image_size = mod_image_size

                if mod_image_size != all_image_size:
                    msg = "All modalities should have the same image size."
                    raise ValueError(msg)

        return all_image_size

    def freeze_backbone(self) -> None:
        """Freeze the backbone's parameters."""
        for param in self.encoder.parameters():
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
        params_dict = {}

        s2_depth = self.encoder_depth
        s1_depth = s2_depth // 2

        lr_0 = base_lr * rate_decay ** (s2_depth + 1)
        lr_map["s1_encoder.linear_input"] = lr_0
        lr_map["s2_encoder.linear_input"] = lr_0

        params_dict["s1_encoder.linear_input"] = (
            self.encoder.s1_encoder.linear_input.parameters()
        )

        params_dict["s2_encoder.linear_input"] = (
            self.encoder.s2_encoder.linear_input.parameters()
        )

        for i in range(s1_depth):
            s1_key = f"s1_encoder.base_transformer.{i}"

            lr_i = base_lr * rate_decay ** (s2_depth - i)
            lr_map[s1_key] = lr_i

            params_dict[s1_key] = self.encoder.s1_encoder.transformer.layers[
                i
            ].parameters()

        for i in range(s2_depth):
            s2_key = f"s2_encoder.base_transformer.{i}"

            lr_i = base_lr * rate_decay ** (s2_depth - i)
            lr_map[s2_key] = lr_i

            params_dict[s2_key] = self.encoder.s2_encoder.transformer.layers[
                i
            ].parameters()

        for i in range(s1_depth):
            ce_key = f"cross_encoder.{i}"
            lr_i = base_lr * rate_decay ** (s1_depth - i)

            lr_map[ce_key] = lr_i
            params_dict[ce_key] = self.encoder.cross_encoder.layers[i].parameters()

        lr_map["heads"] = base_lr
        params_dict["heads"] = self.heads.parameters()

        grouped_parameters = []
        for k, lr in lr_map.items():
            grouped_parameters.append({"params": params_dict[k], "lr": lr, "name": k})

        return grouped_parameters

    def _expand_channels(
        self,
        tensor_to_expand: Tensor,
        channels_to_copy: list[int],
    ) -> Tensor:
        s = tensor_to_expand[:, channels_to_copy, :, :]
        return torch.cat((tensor_to_expand, s), dim=1)

    def _ungroup_s1(self, outputs: Tensor) -> tuple[Tensor, Tensor]:
        n_asc, n_des = self.num_dates["s1_asc"], self.num_dates["s1_des"]
        outputs = outputs.unflatten(0, (-1, n_asc + n_des))
        x_s1 = {}
        x_s1["s1_asc"] = outputs[:, :n_asc]
        x_s1["s1_des"] = outputs[:, n_asc : n_asc + n_des]
        x_s1 = self.group(x_s1)

        return x_s1["s1_asc"], x_s1["s1_des"]

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
        """CROMA forward pass.

        Parameters
        ----------
        batch: dict
            Dictionary containing the data.
        ssl_phase: str
            SSL phase (pretrain, probe, finetune). Only probing and finetuning
            are needed for baselines.

        Returns
        -------
        tuple: CROMA's outputs.

        """
        batch = self.resize_and_rescale(batch)

        batch["s1"] = torch.concat((batch["s1_asc"], batch["s1_des"]), dim=1)

        group_batch = self.group(batch)

        sar_input = group_batch["s1"]
        optical_input = self._expand_channels(group_batch["s2"], [9, 9])

        output_croma = self.encoder(sar_images=sar_input, optical_images=optical_input)

        x_enc = {}
        if self.croma_modality in ("both", "optical"):
            x_enc["s2"] = output_croma["optical_encodings"]
        if self.croma_modality in ("both", "SAR"):
            x_enc["s1_asc"], x_enc["s1_des"] = self._ungroup_s1(
                output_croma["SAR_encodings"],
            )
        if self.fusion_mode == "inter-croma":
            x_enc["joint"] = output_croma["joint_encodings"]

        if self.add_date_enc:
            dates = {}
            for name_mod in self.dataset.inputs:
                dates[name_mod] = batch[f"{name_mod}_dates"]

            ref_date = batch["ref_date"]
            x_enc = self._add_date_encodings(x_enc, dates, ref_date)

        logits = self.compute_logits(x_enc)
        return batch, None, None, logits


# The following lines are adapted from https://github.com/antofuller/CROMA
class PretrainedCROMA(nn.Module):
    """PretrainedCROMA class."""

    def __init__(  # noqa: C901
        self,
        encoder_dim: int,
        encoder_depth: int,
        num_heads: int,
        patch_size: int,
        pretrained_path: str | None,
        modality: str,
        image_resolution: int,
    ) -> None:
        """Create the pretrained CROMA model.

        NOTE: image_resolution is not the spatial, spectral, or temporal resolution.
        It is the height and width of the image, in pixels. E.g., CROMA was pretrained
        on 120x120px images, hence image_resolution is 120 by default
        """
        super().__init__()

        if not isinstance(modality, str):
            msg = f"modality must be a string, not {type(modality)}"
            raise TypeError(msg)

        if not isinstance(image_resolution, int):
            msg = f"image_resolution must be an int, not {type(image_resolution)}"
            raise TypeError(msg)

        if image_resolution % 8 != 0:
            msg = f"image_resolution must be a multiple of 8, not {image_resolution}"
            raise ValueError(msg)

        if modality not in ("both", "SAR", "optical"):
            msg = f"modality must be either both, SAR, or optical, not {modality}"
            raise ValueError(msg)

        self.encoder_dim = encoder_dim
        self.encoder_depth = encoder_depth
        self.num_heads = num_heads
        self.patch_size = patch_size

        self.modality = modality
        self.num_patches = (image_resolution // 8) ** 2
        self.s1_channels = 2  # fixed at 2 SAR backscatter channels
        self.s2_channels = 12  # fixed at 12 multispectral optical channels
        self.attn_bias = get_2dalibi(
            num_heads=self.num_heads,
            num_patches=self.num_patches,
        )

        if modality in ("SAR", "both"):
            self.s1_encoder = ViT(
                dim=self.encoder_dim,
                depth=self.encoder_depth // 2,
                in_channels=self.s1_channels,
            )

            self.GAP_FFN_s1 = nn.Sequential(
                nn.LayerNorm(self.encoder_dim),
                nn.Linear(
                    self.encoder_dim,
                    4 * self.encoder_dim,
                ),  # (BSZ, inner_dim)
                nn.GELU(),  # (BSZ, inner_dim)
                nn.Linear(4 * self.encoder_dim, self.encoder_dim),  # (BSZ, dim)
            )

            if pretrained_path:
                # load weights
                self.s1_encoder.load_state_dict(
                    torch.load(pretrained_path)["s1_encoder"],
                )
                self.GAP_FFN_s1.load_state_dict(
                    torch.load(pretrained_path)["s1_GAP_FFN"],
                )

        if modality in ("optical", "both"):
            self.s2_encoder = ViT(
                dim=self.encoder_dim,
                depth=self.encoder_depth,
                in_channels=self.s2_channels,
            )
            self.GAP_FFN_s2 = nn.Sequential(
                nn.LayerNorm(self.encoder_dim),
                nn.Linear(
                    self.encoder_dim,
                    4 * self.encoder_dim,
                ),  # (BSZ, inner_dim)
                nn.GELU(),  # (BSZ, inner_dim)
                nn.Linear(4 * self.encoder_dim, self.encoder_dim),  # (BSZ, dim)
            )

            if pretrained_path:
                # load weights
                self.s2_encoder.load_state_dict(
                    torch.load(pretrained_path)["s2_encoder"],
                )
                self.GAP_FFN_s2.load_state_dict(
                    torch.load(pretrained_path)["s2_GAP_FFN"],
                )

        if modality == "both":
            self.cross_encoder = BaseTransformerCrossAttn(
                dim=self.encoder_dim,
                depth=self.encoder_depth // 2,
                num_heads=self.num_heads,
            )

            if pretrained_path:
                # load weights
                self.cross_encoder.load_state_dict(
                    torch.load(pretrained_path)["joint_encoder"],
                )

    def forward(
        self,
        sar_images: Tensor | None = None,
        optical_images: Tensor | None = None,
    ) -> dict[str, Tensor]:
        return_dict = {}
        if self.modality in ("SAR", "both"):
            if sar_images is None:
                msg = f"Modality is set to {self.modality}, but sar_images are None"
                raise ValueError(msg)

            sar_encodings = self.s1_encoder(
                imgs=sar_images,
                attn_bias=self.attn_bias.to(sar_images.device),
            )  # (bsz, num_patches, encoder_dim)
            sar_gap = self.GAP_FFN_s1(sar_encodings.mean(dim=1))  # (bsz, encoder_dim)
            return_dict["SAR_encodings"] = sar_encodings
            return_dict["SAR_GAP"] = sar_gap

        if self.modality in ("optical", "both"):
            if optical_images is None:
                msg = f"Modality is set to {self.modality}, but optical_images are None"
                raise ValueError(msg)

            optical_encodings = self.s2_encoder(
                imgs=optical_images,
                attn_bias=self.attn_bias.to(optical_images.device),
            )  # (bsz, num_patches, encoder_dim)

            optical_gap = self.GAP_FFN_s2(
                optical_encodings.mean(dim=1),
            )  # (bsz, encoder_dim)
            return_dict["optical_encodings"] = optical_encodings
            return_dict["optical_GAP"] = optical_gap

        if self.modality == "both":
            joint_encodings = self.cross_encoder(
                x=sar_encodings,
                context=optical_encodings,
                relative_position_bias=self.attn_bias.to(optical_images.device),
            )  # (bsz, num_patches, encoder_dim)

            joint_gap = joint_encodings.mean(dim=1)  # (bsz, encoder_dim)
            return_dict["joint_encodings"] = joint_encodings
            return_dict["joint_GAP"] = joint_gap

        return return_dict


def get_2dalibi(num_heads: int, num_patches: int) -> Tensor:
    """Inspired by: https://github.com/ofirpress/attention_with_linear_biases."""
    points = list(
        itertools.product(
            range(int(num_patches**0.5)),
            range(int(num_patches**0.5)),
        ),
    )

    def get_slopes(n: int) -> list[float]:
        def get_slopes_power_of_2(n: int) -> list[float]:
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio**i for i in range(n)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)

        closest_power_of_2 = 2 ** math.floor(math.log2(n))
        return (
            get_slopes_power_of_2(closest_power_of_2)
            + get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
        )

    slopes = torch.Tensor(get_slopes(num_heads)).unsqueeze(1)
    idxs = []
    for p1 in points:
        for p2 in points:
            dist = ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5
            idxs.append(dist * slopes * -1)
    all_bias = torch.cat(idxs, dim=1)
    return all_bias.view(1, num_heads, num_patches, num_patches)


class FFN(nn.Module):
    """Implement a feedforward neural network."""

    def __init__(
        self,
        dim: int,
        mult: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        inner_dim = dim * mult

        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),  # (BSZ, num_patches, inner_dim)
            nn.GELU(),  # (BSZ, num_patches, inner_dim)
            nn.Dropout(dropout),  # (BSZ, num_patches, inner_dim)
            nn.Linear(inner_dim, dim),  # (BSZ, num_patches, dim)
        )
        self.input_norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.input_norm(x)  # (BSZ, num_patches, dim)
        return self.net(x)  # (BSZ, num_patches, dim)


class Attention(nn.Module):
    """Implement a Self-Attention layer."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        if dim % num_heads != 0:
            msg = "dim must be evenly divisible by num_heads"
            raise ValueError(msg)

        dim_head = dim // num_heads
        self.scale = dim_head**-0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)
        self.input_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, relative_position_bias: Tensor) -> Tensor:
        x = self.input_norm(x)  # (BSZ, num_patches, dim)
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)  # (BSZ, num_patches, dim)
        q, k, v = (
            rearrange(t, "b n (h d) -> b h n d", h=self.num_heads) for t in (q, k, v)
        )  # (BSZ, num_heads, num_patches, dim_head)

        attention_scores = (
            einsum(
                "b h i d, b h j d -> b h i j",
                q,
                k,
            )
            * self.scale
        )  # (BSZ, num_heads, num_patches, num_patches)

        attention_scores = attention_scores + relative_position_bias

        attn = attention_scores.softmax(dim=-1)
        attn = self.dropout(attn)  # (BSZ, num_heads, num_patches, num_patches)

        out = einsum(
            "b h i j, b h j d -> b h i d",
            attn,
            v,
        )  # (BSZ, num_heads, num_patches, dim_head)

        out = rearrange(out, "b h n d -> b n (h d)")  # (BSZ, num_patches, dim)
        return self.to_out(out)  # (BSZ, num_patches, dim)


class CrossAttention(nn.Module):
    """Implement a Cross-Attention layer."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        if dim % num_heads != 0:
            msg = "dim must be evenly divisible by num_heads"
            raise ValueError(msg)

        dim_head = dim // num_heads
        self.scale = dim_head**-0.5

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

        self.to_out = nn.Linear(dim, dim)
        self.input_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        context: Tensor,
        relative_position_bias: Tensor,
    ) -> Tensor:
        x = self.input_norm(x)  # (BSZ, num_patches, dim)
        context = self.input_norm(context)  # (BSZ, num_patches, dim)

        q = self.to_q(x)  # (BSZ, num_patches, dim)
        k = self.to_k(context)  # (BSZ, num_patches, dim)
        v = self.to_v(context)  # (BSZ, num_patches, dim)

        q, k, v = (
            rearrange(t, "b n (h d) -> b h n d", h=self.num_heads) for t in (q, k, v)
        )  # (BSZ, num_heads, num_patches, dim_head)

        attention_scores = (
            einsum(
                "b h i d, b h j d -> b h i j",
                q,
                k,
            )
            * self.scale
        )  # (BSZ, num_heads, num_patches, num_patches)

        attention_scores = attention_scores + relative_position_bias

        attn = attention_scores.softmax(dim=-1)
        attn = self.dropout(attn)  # (BSZ, num_heads, num_patches, num_patches)

        out = einsum(
            "b h i j, b h j d -> b h i d",
            attn,
            v,
        )  # (BSZ, num_heads, num_patches, dim_head)

        out = rearrange(out, "b h n d -> b n (h d)")  # (BSZ, num_patches, dim)
        return self.to_out(out)  # (BSZ, num_patches, dim)


class BaseTransformer(nn.Module):
    """Implement a Transformer architecture."""

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int = 8,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        ff_mult: int = 4,
        final_norm: bool = True,
    ) -> None:
        super().__init__()
        self.final_norm = final_norm
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(dim=dim, num_heads=num_heads, dropout=attn_dropout),
                        FFN(dim=dim, mult=ff_mult, dropout=ff_dropout),
                    ],
                ),
            )

        if self.final_norm:
            self.norm_out = nn.LayerNorm(dim)

    def forward(self, x: Tensor, relative_position_bias: Tensor) -> Tensor:
        for self_attn, ffn in self.layers:
            x = self_attn(x, relative_position_bias) + x  # (BSZ, num_patches, dim)
            x = ffn(x) + x  # (BSZ, num_patches, dim)

        if self.final_norm:
            return self.norm_out(x)

        return x


class BaseTransformerCrossAttn(nn.Module):
    """Implement a Transformer architecture with cross-attention."""

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int = 8,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        ff_mult: int = 4,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(dim=dim, num_heads=num_heads, dropout=attn_dropout),
                        CrossAttention(
                            dim=dim,
                            num_heads=num_heads,
                            dropout=attn_dropout,
                        ),
                        FFN(dim=dim, mult=ff_mult, dropout=ff_dropout),
                    ],
                ),
            )

        self.norm_out = nn.LayerNorm(dim)

    def forward(
        self,
        x: Tensor,
        context: Tensor,
        relative_position_bias: Tensor,
    ) -> Tensor:
        for self_attn, cross_attn, ffn in self.layers:
            x = self_attn(x, relative_position_bias) + x  # (BSZ, num_patches, dim)
            x = cross_attn(x, context, relative_position_bias) + x
            x = ffn(x) + x  # (BSZ, num_patches, dim)

        return self.norm_out(x)  # (BSZ, num_patches, dim)


class ViT(nn.Module):
    """Implement a Vision Transformer."""

    def __init__(self, dim: int, depth: int, in_channels: int) -> None:
        super().__init__()
        self.depth = depth
        self.in_channels = in_channels
        self.dim = dim
        self.num_heads = 16  # always 16, for base and large models
        self.patch_size = 8  # always 8, for base and large models

        pixels_per_patch = self.patch_size * self.patch_size * in_channels
        self.linear_input = nn.Linear(pixels_per_patch, self.dim)
        self.transformer = BaseTransformer(
            dim=self.dim,
            depth=self.depth,
            num_heads=self.num_heads,
        )

    def forward(self, imgs: Tensor, attn_bias: Tensor) -> Tensor:
        x = rearrange(
            imgs,
            "b c (h i) (w j) -> b (h w) (c i j)",
            i=self.patch_size,
            j=self.patch_size,
        )
        # x is shape -> (bsz,num_patches,self.channels*self.patch_size*self.patch_size)

        x = self.linear_input(x)  # (bsz, num_patches, dim)

        return self.transformer(x, relative_position_bias=attn_bias)
