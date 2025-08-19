"""Some utils functions (mainly about state dict loading) for baselines."""

from collections import OrderedDict
from typing import Literal

import torch
import torch.nn.functional as F  # noqa: N812
from einops import rearrange
from huggingface_hub import hf_hub_download
from torch import Tensor


def replace_if_needed(k: str, dict_association: dict[str, str]) -> str:
    """If k is a key in dict_association, then return the corresponding value.

    Otherwise, return k.
    """
    if k in dict_association:
        return dict_association[k]

    return k


def rename_weights(ckpt: dict[str, Tensor]) -> OrderedDict[str, Tensor]:
    """Rename weight names in the checkout."""
    dict_association = {
        "cls_token": "embeddings.cls_token",
        "mask_token": "embeddings.mask_token",
        "patch_embed": "embeddings.patch_embeddings",
        "pos_embed": "embeddings.position_embeddings",
        "proj": "projection",
        "blocks": "encoder.layer",
        "ls1": "layer_scale1",
        "ls2": "layer_scale2",
        "attn": "attention",
        "qkv": "attention.qkv",
        "gamma": "lambda1",
    }

    backbone_ckpt = {}
    for k, v in ckpt.items():
        if "backbone" in k:
            modules = k.split(".")[1:]
            modules = [replace_if_needed(m, dict_association) for m in modules]
            new_k = ".".join(modules)
            new_k = new_k.replace("attention.projection", "attention.output.dense")

            if "qkv.weight" in new_k:
                new_size = v.shape[0] // 3
                k_query = new_k.replace("qkv.weight", "query.weight")
                backbone_ckpt[k_query] = v[:new_size, :]

                k_key = new_k.replace("qkv.weight", "key.weight")
                backbone_ckpt[k_key] = v[new_size : 2 * new_size, :]

                k_value = new_k.replace("qkv.weight", "value.weight")
                backbone_ckpt[k_value] = v[2 * new_size : 3 * new_size, :]

            elif "qkv.bias" in new_k:
                new_size = v.shape[0] // 3
                k_query = new_k.replace("qkv.bias", "query.bias")
                backbone_ckpt[k_query] = v[:new_size]

                k_key = new_k.replace("qkv.bias", "key.bias")
                backbone_ckpt[k_key] = v[new_size : 2 * new_size]

                k_value = new_k.replace("qkv.bias", "value.bias")
                backbone_ckpt[k_value] = v[2 * new_size : 3 * new_size]

            elif ".scale" in new_k or ".zero_point" in new_k:
                continue
            elif "norm.weight" in new_k or "norm.bias" in new_k:
                new_k = new_k.replace("norm", "layernorm")
                backbone_ckpt[new_k] = v
            else:
                backbone_ckpt[new_k] = v

    return OrderedDict(backbone_ckpt)


def load_and_dequantize(path_compressed: str) -> OrderedDict[str, Tensor]:
    """Load the compressed state dict located at path_compressed and dequantize it."""
    state_dict = torch.load(path_compressed, weights_only=True)
    deq_state_dict = []

    for key, value in state_dict.items():
        if "_packed_params._packed_params" in key:
            layer_name = ".".join(key.split(".")[:-2])

            deq_weight = torch.dequantize(value[0])
            deq_bias = torch.dequantize(value[1])

            deq_state_dict.append(("{}.{}".format(layer_name, "weight"), deq_weight))
            deq_state_dict.append(("{}.{}".format(layer_name, "bias"), deq_bias))
        elif "_packed_params" not in key:
            deq_state_dict.append((key, value))

    return OrderedDict(deq_state_dict)


def get_sat_dinov2_state_dict(pretrained_path: str) -> dict[str, Tensor]:
    """Load weights from the paper.

        "Very high resolution sat height maps from RGB imagery"
    (https://arxiv.org/abs/2304.07213)

    Parameters
    ----------
    pretrained_path: str
        Path to the weights.
    num_channels: int
        Number of input channels.

    """
    if "compressed" in pretrained_path:
        state_dict = load_and_dequantize(pretrained_path)
    else:
        state_dict = torch.load(pretrained_path, weights_only=True)

    # Rename and load
    return rename_weights(state_dict)


def get_imagenat_dinov2_state_dict(
    model_size: Literal["small", "base", "large", "huge"] = "large",
) -> dict[str, Tensor]:
    """Load the raw state_dict without initializing the whole model."""
    # Déterminer le nom du repo selon la taille
    repo_id = f"facebook/dinov2-{model_size}"

    # Télécharger le fichier de poids directement
    model_path = hf_hub_download(
        repo_id=repo_id,
        filename="pytorch_model.bin",
        cache_dir="./cache",
    )

    # Charger le state_dict
    return torch.load(model_path, weights_only=True)


def filter_dict(dictionnary: dict, head: str) -> dict:
    """Filter dictionnary by selecting keys beginning with head."""
    prefix = f"{head}."
    return {k[len(prefix) :]: v for k, v in dictionnary.items() if k.startswith(prefix)}


def interpolate_pos_encoding(
    pretrained_pos_encoding: torch.Tensor,
    height: int,
    width: int,
    patch_size: int,
) -> torch.Tensor:
    """Allow to interpolate the pre-trained position encodings.

    Position encodings interpolation allows to be able to use the model on higher resolution images.
    This method is also adapted to support torch.jit tracing and interpolation at torch.float32 precision.

    Adapted from:
    - https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174-L194, and
    - https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/models/vision_transformer.py#L179-L211
    """  # noqa: E501
    grid_height = height // patch_size
    grid_width = width // patch_size

    num_patches = grid_height * grid_width
    num_positions = pretrained_pos_encoding.shape[1] - 1

    # always interpolate when tracing to ensure the exported model works
    # for dynamic input shapes
    if not torch.jit.is_tracing() and num_patches == num_positions and height == width:
        return pretrained_pos_encoding

    class_pos_enc = pretrained_pos_encoding[:, :1]
    patch_pos_enc = pretrained_pos_encoding[:, 1:]

    sqrt_num_positions = int(num_positions**0.5)
    patch_pos_enc = rearrange(
        patch_pos_enc,
        "b (pw ph) d -> b d pw ph",
        pw=sqrt_num_positions,
    )

    target_dtype = patch_pos_enc.dtype
    patch_pos_enc = F.interpolate(
        patch_pos_enc.to(torch.float32),
        size=(grid_height, grid_width),
        mode="bicubic",
        align_corners=False,
    ).to(dtype=target_dtype)

    patch_pos_enc = rearrange(
        patch_pos_enc,
        "b d gw gh -> b (gw gh) d",
    )
    return class_pos_enc, patch_pos_enc
