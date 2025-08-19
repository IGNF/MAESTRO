"""Patchify/pixelify layers."""

import torch
from einops import rearrange, repeat
from torch import Tensor, nn

from maestro.layers.utils import downsamplings_from_patch_size


class Patchify(nn.Module):
    """Patchify the input cube & create embeddings."""

    def __init__(
        self,
        bands: int | list[list[int]],
        embed_dim: int,
        patch_size: int,
        unpool_dim: int | None,
    ) -> None:
        super().__init__()
        self.num_bands = [bands] if isinstance(bands, int) else list(map(len, bands))
        self.patchify_bands = nn.ModuleList(
            [
                PatchifyBands(
                    embed_dim=embed_dim,
                    in_chans=in_chans,
                    patch_size=patch_size,
                    unpool_dim=unpool_dim,
                )
                for in_chans in self.num_bands
            ],
        )

    def forward(self, x: Tensor) -> tuple[Tensor, list[list[Tensor]]]:
        x = torch.split(x, self.num_bands, dim=2)
        x, inds = zip(
            *[self.patchify_bands[idx](x[idx]) for idx, _ in enumerate(x)],
        )
        return torch.cat(x, dim=1), list(inds)


class PatchifyBands(nn.Module):
    """Patchify the input cube & create embeddings per patch/bands."""

    def __init__(
        self,
        in_chans: int,
        embed_dim: int,
        patch_size: int,
        unpool_dim: int | None,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.unpool_dim = unpool_dim
        self.downsamplings = downsamplings_from_patch_size(unpool_dim, patch_size)

        self.conv_skip = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.norm_skip = nn.GroupNorm(1, embed_dim)
        if self.unpool_dim and self.downsamplings:
            channels = []
            for idx, _ in enumerate(self.downsamplings):
                pow_in = len(self.downsamplings) - idx
                pow_out = len(self.downsamplings) - (idx + 1)
                channels_in = self.unpool_dim // 2**pow_in
                channels_out = self.unpool_dim // 2**pow_out
                channels.append((channels_in, channels_out))

            self.conv = nn.Conv2d(
                in_chans,
                channels[0][0],
                kernel_size=1,
            )
            self.norms = nn.ModuleDict(
                {
                    f"{branch}_{idx}": nn.GroupNorm(1, channels_in)
                    for branch in ("skip", "res")
                    for idx, (channels_in, _) in enumerate(channels)
                },
            )
            self.convs = nn.ModuleDict(
                {
                    f"{branch}_{idx}": nn.Conv2d(
                        channels_in,
                        channels_out,
                        kernel_size=self.downsamplings[idx],
                        stride=self.downsamplings[idx],
                        padding_mode="reflect",
                    )
                    for branch in ("skip", "res")
                    for idx, (channels_in, channels_out) in enumerate(channels)
                },
            )
            self.norm = nn.GroupNorm(1, channels[-1][1])
            self.fc = nn.Conv2d(
                channels[-1][1],
                embed_dim,
                kernel_size=1,
            )
            self.norm_fc = nn.GroupNorm(1, embed_dim)
            self.act = nn.SiLU()

    def forward(self, x: Tensor) -> tuple[Tensor, list[Tensor]]:
        _, D, _, H, W = x.shape  # noqa: N806
        x = rearrange(x, "b d c hp1 wp2 -> (b d) c hp1 wp2")
        skip = self.conv_skip(x)
        skip = self.norm_skip(skip)
        skip = rearrange(
            skip,
            "bd c h w -> (bd h w) c 1 1",
        )

        inds = []
        if self.unpool_dim and self.downsamplings:
            x = rearrange(
                x,
                "bd c (h p1) (w p2) -> (bd h w) c p1 p2",
                p1=self.patch_size,
                p2=self.patch_size,
            )
            x = self.conv(x)
            for idx, _ in enumerate(self.downsamplings):
                x_res, x_skip = x, x

                x_res = self.norms[f"res_{idx}"](x_res)
                x_res = self.act(x_res)
                x_res = self.convs[f"res_{idx}"](x_res)

                x_skip = self.norms[f"skip_{idx}"](x_skip)
                x_skip = self.convs[f"skip_{idx}"](x_skip)

                x = x_res + x_skip

            x = self.norm(x)
            x = self.fc(x)
            x = self.norm_fc(x)
            x = x + skip
        else:
            x = skip
        x = rearrange(
            x,
            "(b d h w) c 1 1 -> b d (h w) c",
            d=D,
            h=H // self.patch_size,
            w=W // self.patch_size,
        )
        return x, inds


class Pixelify(nn.Module):
    """Pixelify per patch."""

    def __init__(
        self,
        embed_dim: int,
        bands: int | list[list[int]],
        patch_size: int,
        unpool_dim: int | None,
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
                    unpool_dim=unpool_dim,
                )
                for out_chans in self.num_bands
            ],
        )

    def forward(
        self,
        x: Tensor,
        mask: Tensor,
        inds: list[list[Tensor]],  # noqa: ARG002
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
            x_rec.append(pixelify(x[idx], inds=[]))
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
        unpool_dim: int | None,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.unpool_dim = unpool_dim
        self.downsamplings = downsamplings_from_patch_size(unpool_dim, patch_size)

        self.conv_skip = nn.Conv2d(
            embed_dim,
            out_chans * self.patch_size**2,
            kernel_size=1,
        )
        if self.unpool_dim and self.downsamplings:
            channels = []
            for idx, _ in enumerate(self.downsamplings):
                pow_in = idx
                pow_out = idx + 1
                channels_in = self.unpool_dim // 2**pow_in
                channels_out = self.unpool_dim // 2**pow_out
                channels.append((channels_in, channels_out))

            self.conv = nn.Conv2d(
                embed_dim,
                channels[0][0],
                kernel_size=1,
            )
            self.norms = nn.ModuleDict(
                {
                    f"{branch}_{idx}": nn.GroupNorm(1, channels_in)
                    for branch in ("skip", "res")
                    for idx, (channels_in, _) in enumerate(channels)
                },
            )
            self.convs = nn.ModuleDict(
                {
                    f"{branch}_{idx}": nn.ConvTranspose2d(
                        channels_in,
                        channels_out,
                        kernel_size=self.downsamplings[::-1][idx],
                        stride=self.downsamplings[::-1][idx],
                    )
                    for branch in ("skip", "res")
                    for idx, (channels_in, channels_out) in enumerate(channels)
                },
            )
            self.norm = nn.GroupNorm(1, channels[-1][1])
            self.fc = nn.Conv2d(
                channels[-1][1],
                out_chans,
                kernel_size=1,
            )
            self.act = nn.SiLU()

    def forward(
        self,
        x: Tensor,
        inds: list[Tensor],  # noqa: ARG002
    ) -> Tensor:
        _, D, L, _ = x.shape  # noqa: N806
        H = round(L**0.5)  # noqa: N806

        x = rearrange(
            x,
            "b d (h w) c -> (b d) c h w",
            h=H,
        )
        skip = self.conv_skip(x)
        skip = rearrange(
            skip,
            "bd (p1 p2 c) h w -> bd c (h p1) (w p2)",
            p1=self.patch_size,
            p2=self.patch_size,
        )
        if self.unpool_dim and self.downsamplings:
            x = self.conv(x)
            for idx, _ in enumerate(self.downsamplings):
                x_res, x_skip = x, x

                x_res = self.norms[f"res_{idx}"](x_res)
                x_res = self.act(x_res)
                x_res = self.convs[f"res_{idx}"](x_res)

                x_skip = self.norms[f"skip_{idx}"](x_skip)
                x_skip = self.convs[f"skip_{idx}"](x_skip)

                x = x_res + x_skip

            x = self.norm(x)
            x = self.fc(x)
            x = x + skip
        else:
            x = skip

        return rearrange(
            x,
            "(b d) c h w -> b d c h w",
            d=D,
        )
