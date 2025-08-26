"""Generic dataset module."""

from abc import ABC
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
import rasterio
from rasterio.windows import Window
from torch.utils.data import Dataset

from conf.dataset.utils import DatasetConfig


class GenericDataset(Dataset, ABC):
    """Generic dataset."""

    def __init__(
        self,
        dataset: DatasetConfig,
        stage: Literal["train", "val", "test"],
        use_transform: bool,
        random_dates: bool = False,
        random_crop: bool = False,
    ) -> None:
        self.rng = np.random.default_rng(seed=42)
        self.use_transform = use_transform
        self.dataset = dataset
        self.random_dates = random_dates and stage == "train"
        self.random_crop = random_crop and stage == "train"

    @staticmethod
    def unflatten(x: np.ndarray, axis: int, reshape_axis: tuple) -> np.ndarray:
        """Unflatten specific axes in numpy array."""
        if axis == -1:
            axis = x.ndim - 1
        shape = (*x.shape[:axis], *reshape_axis, *x.shape[axis + 1 :])
        return x.reshape(shape)

    def sample_gcd(
        self,
        idx: int,
        base_length: int,
        repeats: int,
    ) -> tuple[int, np.ndarray]:
        """Sample crop part in image based on index of repetition."""
        if not self.random_crop:
            idx_repeat = idx // base_length
            idx_repeat = (idx_repeat // repeats, idx_repeat % repeats)
            start_gcd = np.array(
                (
                    idx_repeat[0] * self.dataset.size_gcd // repeats,
                    idx_repeat[1] * self.dataset.size_gcd // repeats,
                ),
            )
        else:
            start_gcd = None

        return (idx % base_length), start_gcd

    def preprocess_rasters(
        self,
        meta: dict[str, Path | str | np.ndarray],
        start_gcd: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Read raster inputs."""
        if any(
            f"{name_mod}_path" not in meta or f"{name_mod}_dates" not in meta
            for name_mod in self.dataset.rasters
        ):
            msg = "A modality is missing in meta data."
            raise ValueError(msg)

        sizes_date = {
            name_mod: len(meta[f"{name_mod}_dates"])
            for name_mod in self.dataset.rasters
        }
        if start_gcd is None:
            start_gcd = self.rng.integers(
                low=0,
                high=self.dataset.size_gcd - self.dataset.crop_gcd + 1,
                size=2,
            )

        start, end = {}, {}
        for name_mod, mod in self.dataset.rasters.items():
            start_mod = start_gcd * (
                self.dataset.sizes[name_mod] // self.dataset.size_gcd
            )
            start_mod += meta.get(f"{name_mod}_shift", 0)  # align TreeSatAI-TS aerial
            start_mod_date = self.rng.integers(
                low=0,
                high=sizes_date[name_mod] % mod.num_dates + 1,
            )
            start[name_mod] = (*start_mod, start_mod_date)

            end_mod = (start_gcd + self.dataset.crop_gcd) * (
                self.dataset.sizes[name_mod] // self.dataset.size_gcd
            )
            end_mod += meta.get(f"{name_mod}_shift", 0)  # align TreeSatAI-TS aerial
            end_mod_date = start_mod_date + mod.num_dates * (
                sizes_date[name_mod] // mod.num_dates
            )
            end[name_mod] = (*end_mod, end_mod_date)

        # read raster slices
        inputs = {}
        for name_mod, mod in self.dataset.rasters.items():
            input_mod, dates_mod = self.preprocess_raster(
                meta[f"{name_mod}_path"],
                meta[f"{name_mod}_dates"],
                meta.get(f"{name_mod}_mask", None),
                meta.get(f"{name_mod}_mask_bands", None),
                meta.get(f"{name_mod}_h5_name", None),
                meta.get(f"{name_mod}_h5_mask", None),
                start[name_mod],
                end[name_mod],
                **vars(mod),
            )
            inputs[name_mod] = input_mod
            inputs[f"{name_mod}_dates"] = dates_mod

        return inputs

    def preprocess_raster(  # noqa: PLR0915, C901
        self,
        path_mod: Path,
        dates_mod: np.ndarray,
        mask: Path | None,
        mask_bands: list[list[int]],
        h5_name: str | None,
        h5_mask: str | None,
        start: tuple[int],
        end: tuple[int],
        bands: int | list[list[int]],
        num_dates: int,
        mask_threshold: float,  # threshold on mask percentage cover
        norm_fac: float | None,
        log_scale: bool,
        **kwargs,  # noqa: ARG002, ANN003
    ) -> tuple[np.ndarray, np.ndarray]:
        """Read raster modality."""
        slices = (
            slice(start[0], end[0]),
            slice(start[1], end[1]),
            slice(start[2], end[2]),
        )
        bands = (
            range(bands)
            if isinstance(bands, int)
            else [idx for band in bands for idx in band]
        )
        use_mask = (mask_threshold / 100.0) < 1.0
        mask_mod = None  # by default, no mask is used

        match path_mod.suffix:
            case ".tif" | ".png" | ".jpg" | ".jpeg":
                window = Window(
                    start[1],
                    start[0],
                    end[1] - start[1],
                    end[0] - start[0],
                )
                with rasterio.open(path_mod) as src:
                    input_mod = src.read(window=window)
                    input_mod = self.unflatten(input_mod, 0, (len(dates_mod), -1))
                    input_mod = input_mod[:, bands]
                if use_mask and mask is not None:
                    bands = mask_bands if mask_bands else bands
                    with rasterio.open(mask) as src:
                        mask_mod = src.read(window=window)
                        mask_mod = self.unflatten(mask_mod, 0, (len(dates_mod), -1))
                        mask_mod = mask_mod[:, bands, :, :]
            case ".npy":
                npy_file = np.load(path_mod, mmap_mode="r")
                if npy_file.ndim < 4:  # noqa: PLR2004
                    npy_file = np.expand_dims(npy_file, axis=0)
                input_mod = npy_file[:, :, slices[0], slices[1]][:, bands]
                if use_mask and mask is not None:
                    mask_file = np.load(mask, mmap_mode="r")
                    mask_mod = mask_file[:, slices[0], slices[1]]
            case ".h5":
                with h5py.File(path_mod, "r") as h5_file:
                    input_mod = h5_file[h5_name][:, :, slices[0], slices[1]][:, bands]
                    if use_mask and h5_mask is not None:
                        mask_mod = h5_file[h5_mask][:, :, slices[0], slices[1]]
            case _:
                msg = f"File format {path_mod.suffix} not supported."
                raise NotImplementedError(msg)

        if len(dates_mod) != num_dates:
            input_mod = input_mod[slices[2]]
            input_mod = self.unflatten(input_mod, 0, (num_dates, -1))
            dates_mod = dates_mod[slices[2], :, None, None]
            dates_mod = self.unflatten(dates_mod, 0, (num_dates, -1))
            if mask_mod is not None:
                mask_mod = mask_mod[slices[2]]
                mask_mod = self.unflatten(mask_mod, 0, (num_dates, -1))
                mask_mod = (mask_mod > mask_threshold).any(axis=2, keepdims=True)
                mask_mod = mask_mod & ~(
                    mask_mod.any(axis=(3, 4), keepdims=True).all(axis=1, keepdims=True)
                )
                input_mod = np.where(mask_mod, np.nan, input_mod)

            diff_mod = input_mod - np.nanmedian(input_mod, axis=1, keepdims=True)
            diff_mod = np.abs(diff_mod)
            if self.random_dates:
                diff_mod = 0 * diff_mod  # keep NaNs
                diff_mod += self.rng.random(diff_mod.shape).astype(diff_mod.dtype)

            diff_mod = np.mean(diff_mod, axis=(2, 3, 4), keepdims=True)
            median_inds = np.nanargmin(diff_mod, axis=1, keepdims=True)
            input_mod = np.take_along_axis(input_mod, median_inds, axis=1).squeeze(1)
            dates_mod = np.take_along_axis(dates_mod, median_inds, axis=1).squeeze(1)

        input_mod = input_mod.astype(np.float32)
        if log_scale:
            input_mod = np.log(np.maximum(input_mod, 1e-10))

        if norm_fac is not None:
            input_mod /= norm_fac

        return input_mod, dates_mod

    def transform_rasters(
        self,
        inputs: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Transform model rasters."""
        if self.use_transform:
            if self.rng.choice([True, False]):
                inputs.update(
                    {
                        name_mod: np.flip(inputs[name_mod], axis=2)
                        for name_mod in self.dataset.rasters
                    },
                )
            if self.rng.choice([True, False]):
                inputs.update(
                    {
                        name_mod: np.flip(inputs[name_mod], axis=3)
                        for name_mod in self.dataset.rasters
                    },
                )
            if self.rng.choice([True, False]):
                inputs.update(
                    {
                        name_mod: np.swapaxes(inputs[name_mod], 2, 3)
                        for name_mod in self.dataset.rasters
                    },
                )
            inputs.update(
                {
                    name_mod: np.ascontiguousarray(inputs[name_mod])
                    for name_mod in self.dataset.rasters
                },
            )
        return inputs
