"""Head layers."""

from abc import ABC
from functools import partial
from typing import Literal

import torch
from einops import rearrange
from torch import Tensor, nn

from maestro.layers.embed import PixelifyBands


class Head(nn.Module, ABC):
    """Abstract head class."""

    def maybe_detach_features(
        self,
        x: Tensor,
        ssl_phase: Literal["probe", "finetune"],
    ) -> Tensor:
        """Detach encoder features in probing phase."""
        if ssl_phase == "probe":
            x = x.detach()
        return x


class AttentiveReduce(nn.Module):
    """Attentive reduction with self-attention."""

    def __init__(
        self,
        dim: int,
        heads: int = 8,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.norm_fc = nn.LayerNorm(dim)
        self.softmax = nn.Softmax(dim=-1)

        self.to_kv = nn.Linear(dim, dim * 2, bias=False)
        self.query = nn.Parameter(torch.randn(dim))

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        out = self.norm(x)

        q = rearrange(self.query, "(h d) -> 1 h 1 d", h=self.heads)
        k, v = self.to_kv(out).chunk(2, dim=-1)
        k = rearrange(k, "b l (h d) -> b h l d", h=self.heads)
        v = rearrange(v, "b l (h d) -> b h l d", h=self.heads)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.softmax(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h 1 d -> b (h d)")
        return self.norm_fc(out)


class ClassificationHead(Head):
    """Classification head."""

    def __init__(
        self,
        type_head: Literal["mean", "attentive"],
        dim: int,
        num_classes: int,
        heads: int = 8,
    ) -> None:
        super().__init__()

        match type_head:
            case "linear":
                self.reduce = partial(torch.mean, dim=1)
            case "attentive":
                self.reduce = AttentiveReduce(dim=dim, heads=heads)

        self.linear = nn.Linear(dim, num_classes)

    def forward(
        self,
        x: Tensor,
        ssl_phase: Literal["probe", "finetune"],
    ) -> Tensor:
        out = self.maybe_detach_features(x, ssl_phase)
        out = self.reduce(out)
        return self.linear(out)


class PixelifyHead(PixelifyBands, Head):
    """Pixelify head."""

    def __init__(
        self,
        type_head: Literal["mean", "attentive"],
        dim: int,
        out_chans: int,
        patch_size: int,
        heads: int = 8,
    ) -> None:
        super().__init__(  # method used is from the first listed inherited class
            embed_dim=dim,
            out_chans=out_chans,
            patch_size=patch_size,
        )

        match type_head:
            case "linear":
                self.reduce = partial(torch.mean, dim=1)
            case "attentive":
                self.reduce = AttentiveReduce(dim=dim, heads=heads)

    def forward(
        self,
        x: Tensor,
        ssl_phase: Literal["probe", "finetune"],
    ) -> Tensor:
        out = self.maybe_detach_features(x, ssl_phase)
        out = rearrange(out, "b d l c -> (b l) d c")
        out = self.reduce(out)
        out = rearrange(out, "(b l) c -> b 1 l c", l=x.shape[2])
        return super().forward(  # method used is from the first listed inherited class
            out,
        )
