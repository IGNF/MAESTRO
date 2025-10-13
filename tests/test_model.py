import itertools

import pytest

from maestro.train.model import SSLModule


@pytest.mark.parametrize(
    "model_size, fusion_mode, inter_depth, stage",
    itertools.product(
        ("tiny",),
        ("shared", "monotemp", "mod", "group"),
        (0,),
        ("train", "val", "test"),
    ),
)
def test_ssl_mae(
    datasets_treesat,
    mask,
    model_size,
    fusion_mode,
    inter_depth,
    stage,
):
    ssl_module = SSLModule(
        datasets=datasets_treesat,
        mask=mask,
        interpolate="nearest",
        fusion_mode=fusion_mode,
        inter_depth=inter_depth,
        model="mae",
        model_size=model_size,
    )
    ssl_module.setup(stage=stage)


@pytest.mark.parametrize(
    "model_size, fusion_mode, inter_depth, stage",
    itertools.product(
        ("tiny",),
        ("mod", "group"),
        (3,),
        ("train", "val", "test"),
    ),
)
def test_ssl_inter_mae(
    datasets_treesat,
    mask,
    model_size,
    fusion_mode,
    inter_depth,
    stage,
):
    ssl_module = SSLModule(
        datasets=datasets_treesat,
        mask=mask,
        interpolate="nearest",
        fusion_mode=fusion_mode,
        inter_depth=inter_depth,
        model="mae",
        model_size=model_size,
    )
    ssl_module.setup(stage=stage)


@pytest.mark.parametrize(
    "model_size, interpolate, fusion_mode, inter_depth, stage, type_head, loss",
    itertools.product(
        ("tiny",),
        ("nearest", "bilinear"),
        ("group",),
        (0,),
        ("train", "val", "test"),
        ("attentive", "linear"),
        ("l1_norm", "l1"),
    ),
)
def test_ssl_architecture(
    datasets_treesat,
    mask,
    model_size,
    interpolate,
    fusion_mode,
    inter_depth,
    stage,
    type_head,
    loss,
):
    ssl_module = SSLModule(
        datasets=datasets_treesat,
        mask=mask,
        interpolate=interpolate,
        fusion_mode=fusion_mode,
        inter_depth=inter_depth,
        model="mae",
        model_size=model_size,
        type_head=type_head,
        loss=loss,
    )
    ssl_module.setup(stage=stage)


@pytest.mark.parametrize(
    "model_size, fusion_mode, inter_depth, stage, loss",
    itertools.product(
        ("tiny",),
        ("group",),
        (0,),
        ("train", "val", "test"),
        ("l1_norm", "l1"),
    ),
)
def test_ssl_loss(
    datasets_treesat,
    mask,
    model_size,
    fusion_mode,
    inter_depth,
    stage,
    loss,
):
    ssl_module = SSLModule(
        datasets=datasets_treesat,
        mask=mask,
        interpolate="nearest",
        fusion_mode=fusion_mode,
        inter_depth=inter_depth,
        model="mae",
        model_size=model_size,
        loss=loss,
    )
    ssl_module.setup(stage=stage)


@pytest.mark.xfail
def test_model_should_fail(datasets_treesat, mask):
    _ = SSLModule(
        datasets=datasets_treesat,
        mask=mask,
        fusion_mode="group",
        inter_depth=0,
        model="MAE",
        model_size="tiny",
    )


@pytest.mark.xfail
def test_missing_args_should_fail():
    _ = SSLModule(model="mae", model_size="tiny")
