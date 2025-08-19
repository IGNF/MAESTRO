"""Utils module."""

import torch
import torch.nn.functional as F  # noqa: N812
from matplotlib import cm
from matplotlib.colors import to_hex
from torch import Tensor
from torchvision.utils import draw_segmentation_masks


def get_segment_mask_from_logits(logits: Tensor, num_classes: int) -> Tensor:
    """Convert prediction logits to mask for segmentation task."""
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


def get_cd_mask_from_logits(logits: Tensor) -> Tensor:
    """Convert prediction logits to mask for change detection task."""
    return (torch.sigmoid(logits) > 0.5).long()[0, 0].type(torch.bool)  # noqa: PLR2004


def get_target_mask_from_batch(
    batch_target: Tensor,
    num_classes: int,
    missing_val: int,
) -> Tensor:
    """Convert batch target mask to one hot mask."""
    batch_target = torch.where(batch_target == missing_val, num_classes, batch_target)
    target_msk = F.one_hot(
        batch_target.type(torch.int64),
        num_classes + 1,
    )
    target_msk = target_msk[:, :, :num_classes]
    return target_msk.permute(2, 0, 1).type(torch.bool)


def create_masked_image(img: Tensor, msk: Tensor, num_classes: int) -> Tensor:
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
