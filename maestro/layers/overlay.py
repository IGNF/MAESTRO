"""Overlay module."""

import torch
import torch.nn.functional as F  # noqa: N812
from matplotlib import cm
from matplotlib.colors import to_hex
from torch import Tensor
from torchvision.utils import draw_segmentation_masks


def onehot_pred_from_logits(logits: Tensor, num_classes: int) -> Tensor:
    """Convert prediction logits to onehot for segmentation task."""
    return (
        F.one_hot(
            torch.argmax(
                logits,
                dim=0,
            ),
            num_classes,
        )
        .type(torch.bool)
        .permute(2, 0, 1)
    )


def onehot_target_from_batch(
    batch_target: Tensor,
    num_classes: int,
    missing_val: int,
) -> Tensor:
    """Convert batch target mask to one hot."""
    batch_target = torch.where(batch_target == missing_val, num_classes, batch_target)
    target_onehot = F.one_hot(
        batch_target.type(torch.int64),
        num_classes + 1,
    )
    target_onehot = target_onehot[:, :, :num_classes]
    return target_onehot.permute(2, 0, 1).type(torch.bool)


def create_overlay(img: Tensor, msk: Tensor, num_classes: int) -> Tensor:
    """Create overlayed image with input image and mask."""
    colormap = list(
        map(
            to_hex,
            cm.get_cmap("plasma", num_classes).colors,
        ),
    )
    img = img.unsqueeze(0)
    img = F.interpolate(img, msk.shape[-2:], mode="bilinear")
    img = img.squeeze(0)
    return draw_segmentation_masks(img, msk, 0.4, colormap)
