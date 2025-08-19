import itertools

import pytest

from maestro.train.model import SSLModule


@pytest.mark.parametrize(
    "multimodal, allmods_depth, stage, model_size",
    itertools.product(
        ("shared", "monotemp", "mod", "group"),
        (0,),
        ("train", "val", "test"),
        ("tiny",),
    ),
)
def test_ssl_mae(
    datasets_treesat,
    mask,
    multimodal,
    allmods_depth,
    stage,
    model_size,
):
    ssl_module = SSLModule(
        datasets=datasets_treesat,
        mask=mask,
        multimodal=multimodal,
        allmods_depth=allmods_depth,
        model="mae",
        model_size=model_size,
    )
    ssl_module.setup(stage=stage)


@pytest.mark.parametrize(
    "multimodal, allmods_depth, stage, model_size",
    itertools.product(
        ("mod", "group"),
        (3,),
        ("train", "val", "test"),
        ("tiny",),
    ),
)
def test_ssl_allmods_mae(
    datasets_treesat,
    mask,
    multimodal,
    allmods_depth,
    stage,
    model_size,
):
    ssl_module = SSLModule(
        datasets=datasets_treesat,
        mask=mask,
        multimodal=multimodal,
        allmods_depth=allmods_depth,
        model="mae",
        model_size=model_size,
    )
    ssl_module.setup(stage=stage)


@pytest.mark.parametrize(
    "multimodal, allmods_depth, stage, model_size, type_head, loss",
    itertools.product(
        ("group",),
        (0,),
        ("train", "val", "test"),
        ("tiny",),
        ("attentive", "linear"),
        ("l1_norm", "l1"),
    ),
)
def test_ssl_head_loss(
    datasets_treesat,
    mask,
    multimodal,
    allmods_depth,
    stage,
    model_size,
    type_head,
    loss,
):
    ssl_module = SSLModule(
        datasets=datasets_treesat,
        mask=mask,
        multimodal=multimodal,
        allmods_depth=allmods_depth,
        model="mae",
        model_size=model_size,
        type_head=type_head,
        loss=loss,
    )
    ssl_module.setup(stage=stage)


@pytest.mark.xfail
def test_model_should_fail(datasets_treesat, mask):
    _ = SSLModule(
        datasets=datasets_treesat,
        mask=mask,
        multimodal="group",
        allmods_depth=0,
        model="MAE",
        model_size="tiny",
    )


@pytest.mark.xfail
def test_missing_args_should_fail():
    _ = SSLModule(model="mae", model_size="tiny")
