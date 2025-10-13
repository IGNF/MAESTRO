import itertools

import pytest
import torch

from maestro.ssl.mae import mae_large, mae_medium, mae_small, mae_tiny


@pytest.fixture()
def model_map():
    return {
        "tiny": mae_tiny,
        "small": mae_small,
        "medium": mae_medium,
        "large": mae_large,
    }


@pytest.mark.parametrize(
    "model_size, fusion_mode",
    itertools.product(
        ("tiny",),
        ("shared", "monotemp", "mod", "group"),
    ),
)
def test_mae(datasets_treesat, mask, model_map, model_size, fusion_mode):
    model_args = {
        "datasets": datasets_treesat,
        "mask": mask,
        "interpolate": "nearest",
        "fusion_mode": fusion_mode,
        "model": "mae",
        "inter_depth": 0,
        "num_levels": 1,
        "type_head": "attentive",
        "fac_abs_enc": 1.0,
        "fac_date_enc": 1.0,
    }
    model_map[model_size](**model_args)


@pytest.mark.parametrize(
    "model_size, interpolate, type_head, fac_abs_enc, fac_date_enc",
    itertools.product(
        ("tiny",),
        ("nearest", "bilinear"),
        ("attentive", "linear"),
        (1.0, 0.0),
        (1.0, 0.0),
    ),
)
def test_mae_architecture(
    datasets_treesat,
    mask,
    model_map,
    model_size,
    interpolate,
    type_head,
    fac_abs_enc,
    fac_date_enc,
):
    model_args = {
        "datasets": datasets_treesat,
        "mask": mask,
        "interpolate": interpolate,
        "fusion_mode": "group",
        "model": "mae",
        "inter_depth": 0,
        "num_levels": 1,
        "type_head": type_head,
        "fac_abs_enc": fac_abs_enc,
        "fac_date_enc": fac_date_enc,
    }
    model_map[model_size](**model_args)
