"""WDSR build options for cheaper training steps: bfloat16 residual blocks
(``mixed_precision``) and weight norms initialised at build
(``eager_weight_norm_init``), plus the step benchmark's variant names."""
from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from euclid_polish.training.augmentation import asinh_stretch_multi_knee
from euclid_polish.training.models.wdsr import (
    MIXED_POLICY,
    PrecisionWeightNorm,
    initialize_weight_norms,
    wdsr,
)
from euclid_polish.training.trainer import seed_everything
from scripts.benchmark_train_step import isolated_order, parse_feed, parse_variant

KNEES = (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)
ARCH = {"scale": 2, "num_res_blocks": 2, "nchan_in": 24, "nchan_out": 4, "input_knees": 6,
        "learned_output_knee": 10.0}


def _stretched_input(seed=0):
    lr = np.random.default_rng(seed).gamma(0.5, 50.0, size=(2, 24, 24, 4)).astype(np.float32)
    return asinh_stretch_multi_knee(tf.constant(lr), KNEES)


def _trunk_and_rest(net):
    """The weight-normalised convs of the residual blocks, and every other one."""
    convs = [layer for layer in net.layers if hasattr(layer, "layer") and hasattr(layer, "v")]
    n_blocks = ARCH["num_res_blocks"]
    entry, blocks, rest = convs[0], convs[1:1 + 3 * n_blocks], convs[1 + 3 * n_blocks:]
    return blocks, [entry] + rest


def test_eager_weight_norm_init_starts_where_the_stock_build_does():
    x = _stretched_input()
    seed_everything(1)
    stock = wdsr(**ARCH)
    seed_everything(1)
    eager = wdsr(**ARCH, eager_weight_norm_init=True)
    # The stock build sets g = |v| on its first call, the eager one at build.
    np.testing.assert_array_equal(eager(x).numpy(), stock(x).numpy())


def test_eager_weight_norm_init_computes_the_same_values_from_the_same_weights():
    x = _stretched_input()
    stock = wdsr(**ARCH)
    expected = stock(x).numpy()
    eager = wdsr(**ARCH, eager_weight_norm_init=True)
    eager.set_weights(stock.get_weights())
    np.testing.assert_array_equal(eager(x).numpy(), expected)
    assert all(isinstance(layer, PrecisionWeightNorm) and layer.eager_init
               for layer in eager.layers if hasattr(layer, "v"))


def test_mixed_precision_runs_only_the_residual_blocks_in_bfloat16():
    net = wdsr(**ARCH, mixed_precision=True)
    blocks, rest = _trunk_and_rest(net)
    assert len(blocks) == 3 * ARCH["num_res_blocks"]
    assert {layer.layer.dtype_policy.name for layer in blocks} == {MIXED_POLICY}
    assert {layer.layer.dtype_policy.name for layer in rest} == {"float32"}
    # Every variable stays float32, so either build loads the other's weights.
    assert {v.dtype for v in net.trainable_variables} == {tf.float32}
    out = net(_stretched_input())
    assert out.dtype == tf.float32


def test_mixed_precision_output_stays_close_to_the_float32_build():
    x = _stretched_input()
    stock = wdsr(**ARCH)
    expected = stock(x).numpy()
    mixed = wdsr(**ARCH, mixed_precision=True)
    mixed.set_weights(stock.get_weights())
    got = mixed(x).numpy()
    # Compare where the model works, in asinh space at its knee.
    err = np.abs(np.arcsinh(got / 10.0) - np.arcsinh(expected / 10.0))
    assert np.median(err) < 1e-2
    assert not np.array_equal(got, expected)       # it really ran in bfloat16


@pytest.mark.parametrize("options", [{"mixed_precision": True}, {"eager_weight_norm_init": True}])
def test_a_float32_checkpoint_restores_into_either_build(tmp_path, options):
    x = _stretched_input()
    stock = wdsr(**ARCH)
    stock(x)                                        # initialise the weight norms
    tf.train.Checkpoint(model=stock).save(str(tmp_path / "ckpt"))
    twin = wdsr(**ARCH, **options)
    tf.train.Checkpoint(model=twin).restore(
        tf.train.latest_checkpoint(str(tmp_path))).assert_existing_objects_matched()
    for got, want in zip(twin.get_weights(), stock.get_weights(), strict=True):
        np.testing.assert_array_equal(got, want)


def test_initialize_weight_norms_does_what_the_first_call_does():
    x = _stretched_input()
    seed_everything(2)
    called = wdsr(**ARCH)
    expected = called(x).numpy()                     # g = |v| on the first call
    seed_everything(2)
    initialised = wdsr(**ARCH)
    initialize_weight_norms(initialised)
    eager = wdsr(**ARCH, eager_weight_norm_init=True)
    eager.set_weights(initialised.get_weights())
    np.testing.assert_array_equal(eager(x).numpy(), expected)
    initialize_weight_norms(called)                  # already initialised: unchanged
    np.testing.assert_array_equal(called(x).numpy(), expected)


def test_stock_build_keeps_tfps_weight_norm():
    net = wdsr(**ARCH)
    assert not any(isinstance(layer, PrecisionWeightNorm) for layer in net.layers)


def test_benchmark_variant_names():
    assert parse_variant("control") == frozenset()
    assert parse_variant("xla+bf16") == {"xla", "bf16"}
    assert parse_variant("eagerwn") == {"eagerwn"}
    for bad in ("fp16", "xla+", "control+xla", ""):
        with pytest.raises(ValueError):
            parse_variant(bad)


def test_benchmark_feed_names():
    assert parse_feed("spc1") == (1, False, False)
    assert parse_feed("spc25+prefetch") == (25, True, False)
    assert parse_feed("spc25+prefetch+fixed") == (25, True, True)
    for bad in ("spc0", "spc", "25", "spc4+xla", "spc4+prefetch+x", "spc4+fused",
                "spc4+fixed+fixed", ""):
        with pytest.raises(ValueError):
            parse_feed(bad)


def test_benchmark_isolated_order_is_abba():
    assert isolated_order(["a", "b"], 2) == ["a", "b", "b", "a"]
    assert isolated_order(["a", "b", "c"], 3) == ["a", "b", "c", "c", "b", "a", "a", "b", "c"]
    assert isolated_order(["a"], 1) == ["a"]
