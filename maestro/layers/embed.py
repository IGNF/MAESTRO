"""Patchify/pixelify layers."""

import torch
from einops import rearrange, repeat
from torch import Tensor, nn


class Patchify(nn.Module):
    """Patchify the input cube & create embeddings."""

    def __init__(
        self,
        bands: int | list[list[int]],
        embed_dim: int,
        patch_size: int,
    ) -> None:
        super().__init__()
        self.num_bands = [bands] if isinstance(bands, int) else list(map(len, bands))

        self.patchify_bands = nn.ModuleList(
            [
                PatchifyBands(
                    embed_dim=embed_dim,
                    in_chans=in_chans,
                    patch_size=patch_size,
                )
                for in_chans in self.num_bands
            ],
        )

    def forward(self, x: Tensor) -> tuple[Tensor, list[list[Tensor]]]:
        x = torch.split(x, self.num_bands, dim=2)
        x = [self.patchify_bands[idx](x[idx]) for idx, _ in enumerate(x)]
        return torch.cat(x, dim=1)


class PatchifyBands(nn.Module):
    """Patchify the input cube & create embeddings per patch/bands."""

    def __init__(
        self,
        in_chans: int,
        embed_dim: int,
        patch_size: int,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size

        self.conv = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.norm = nn.GroupNorm(1, embed_dim)

    def forward(self, x: Tensor) -> tuple[Tensor, list[Tensor]]:
        _, D, _, H, W = x.shape  # noqa: N806
        x = rearrange(x, "b d c hp1 wp2 -> (b d) c hp1 wp2")
        x = self.conv(x)
        x = self.norm(x)
        return rearrange(
            x,
            "(b d) c h w -> b d (h w) c",
            d=D,
        )


class Pixelify(nn.Module):
    """Pixelify per patch."""

    def __init__(
        self,
        embed_dim: int,
        bands: int | list[list[int]],
        patch_size: int,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.num_bands = [bands] if isinstance(bands, int) else list(map(len, bands))

        self.pixelify_bands = nn.ModuleList(
            [
                PixelifyBands(
                    embed_dim=embed_dim,
                    out_chans=out_chans,
                    patch_size=patch_size,
                )
                for out_chans in self.num_bands
            ],
        )

    def forward(
        self,
        x: Tensor,
        mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        x = rearrange(x, "b (g d) l c -> g b d l c", g=len(self.num_bands))
        mask = repeat(
            mask,
            "b (g d) (h w) c -> g b d c (h p1) (w p2)",
            g=len(self.num_bands),
            p1=self.patch_size,
            p2=self.patch_size,
            h=round(mask.shape[2] ** 0.5),
        )

        x_rec, mask_rec = [], []
        for idx, pixelify in enumerate(self.pixelify_bands):
            x_rec.append(pixelify(x[idx]))
            mask_rec.append(mask[idx].expand((-1, -1, self.num_bands[idx], -1, -1)))

        if len(self.num_bands) > 1:
            x_rec = torch.cat(x_rec, dim=2)
            mask_rec = torch.cat(mask_rec, dim=2)
        else:
            x_rec = x_rec.pop()
            mask_rec = mask_rec.pop()

        return x_rec, mask_rec


class PixelifyBands(nn.Module):
    """Pixelify per patch/bands."""

    def __init__(
        self,
        embed_dim: int,
        out_chans: int,
        patch_size: int,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size

        self.conv = nn.Conv2d(
            embed_dim,
            out_chans * self.patch_size**2,
            kernel_size=1,
        )

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        _, D, L, _ = x.shape  # noqa: N806
        H = round(L**0.5)  # noqa: N806

        x = rearrange(
            x,
            "b d (h w) c -> (b d) c h w",
            h=H,
        )
        x = self.conv(x)
        return rearrange(
            x,
            "(b d) (p1 p2 c) h w -> b d c (h p1) (w p2)",
            d=D,
            p1=self.patch_size,
            p2=self.patch_size,
        )
