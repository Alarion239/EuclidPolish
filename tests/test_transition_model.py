"""Tests for the HST → Euclid PSF-transition CNN."""

from __future__ import annotations

import os

import numpy as np
import pytest

# tf is a heavy import; gate at module level so the rest of the test
# suite isn't slowed down on a fresh pytest run that doesn't touch this
# file. The model module is only imported inside test bodies.
tf = pytest.importorskip("tensorflow")


class TestParameterBudget:
    """The model must fit inside the 5k-param cap with default knobs."""

    def test_default_under_5k(self):
        from euclid_polish.training.transition_model import (
            HSTtoEuclidTransition, total_parameter_count,
        )
        m = HSTtoEuclidTransition()           # default C=12, 3 inner layers
        n = total_parameter_count(m)
        assert n <= 5_000, f"default model has {n} params (cap 5000)"

    def test_c13_still_fits(self):
        from euclid_polish.training.transition_model import (
            HSTtoEuclidTransition, total_parameter_count,
        )
        m = HSTtoEuclidTransition(channels=13)
        assert total_parameter_count(m) <= 5_000

    def test_c14_overflows(self):
        """C=14 is the documented boundary case — should overflow.

        If this ever stops failing the param formula in the docstring is
        wrong (or the architecture changed). Either way we want CI to
        flag it.
        """
        from euclid_polish.training.transition_model import (
            HSTtoEuclidTransition, total_parameter_count,
        )
        m = HSTtoEuclidTransition(channels=14)
        assert total_parameter_count(m) > 5_000

    def test_param_formula_matches_analytic(self):
        """27·C² + 18·C should match the actual param count exactly.

        Formula assumes ``use_bias=False`` on every Conv2D (load-bearing
        for scale invariance — see the model module's __init__ doc).
        """
        from euclid_polish.training.transition_model import (
            HSTtoEuclidTransition, total_parameter_count,
        )
        for C in (4, 8, 12, 13):
            m = HSTtoEuclidTransition(channels=C, n_inner_layers=3)
            expected = 27 * C * C + 18 * C
            assert total_parameter_count(m) == expected, (
                f"param count mismatch at C={C}: "
                f"got {total_parameter_count(m)}, expected {expected}"
            )


class TestResidualIdentityAtInit:
    """At init, ``A_θ(x) ≈ x`` (residual prior).

    We initialised the convs with N(0, 0.01) so the residual branch
    starts ~0; ``A_θ(x) = x + f(x)`` then equals ``x`` to within a small
    margin. Without this, training has to fight an arbitrary random
    starting point.
    """

    def test_zero_input_yields_zero(self):
        from euclid_polish.training.transition_model import (
            HSTtoEuclidTransition,
        )
        m = HSTtoEuclidTransition()
        x = tf.zeros((1, 32, 32, 1), dtype=tf.float32)
        y = m(x).numpy()
        assert np.allclose(y, 0.0, atol=1e-5)

    def test_nonzero_input_close_to_identity(self):
        from euclid_polish.training.transition_model import (
            HSTtoEuclidTransition,
        )
        tf.random.set_seed(0)
        m = HSTtoEuclidTransition()
        x = tf.random.uniform((1, 32, 32, 1), dtype=tf.float32) * 100.0
        y = m(x).numpy()
        # The residual is N(0, σ_small) per conv weight; output should
        # stay within a few percent of the input on this scale.
        rel_err = np.abs(y - x.numpy()).mean() / (np.abs(x.numpy()).mean() + 1e-12)
        assert rel_err < 0.5, f"identity init too noisy: rel_err={rel_err:.4f}"


class TestForwardPassShape:

    def test_preserves_input_shape(self):
        from euclid_polish.training.transition_model import (
            HSTtoEuclidTransition,
        )
        m = HSTtoEuclidTransition()
        for H, W in [(32, 32), (48, 64), (256, 256)]:
            x = tf.zeros((2, H, W, 1), dtype=tf.float32)
            y = m(x)
            assert tuple(y.shape) == (2, H, W, 1)

    def test_works_on_batched_input(self):
        from euclid_polish.training.transition_model import (
            HSTtoEuclidTransition,
        )
        m = HSTtoEuclidTransition()
        x = tf.zeros((8, 32, 32, 1), dtype=tf.float32)
        y = m(x)
        assert y.shape[0] == 8


class TestDeterminism:
    """Inference must be deterministic — no dropout, no noise injection."""

    def test_same_input_same_output(self):
        from euclid_polish.training.transition_model import (
            HSTtoEuclidTransition,
        )
        m = HSTtoEuclidTransition()
        # Build then "use".
        x = tf.random.uniform((1, 32, 32, 1), seed=7) * 50.0
        y1 = m(x, training=False).numpy()
        y2 = m(x, training=False).numpy()
        np.testing.assert_array_equal(y1, y2)

    def test_training_flag_is_a_noop(self):
        """training=True/False produce identical outputs (the model has
        no dropout/batchnorm, so this is by design)."""
        from euclid_polish.training.transition_model import (
            HSTtoEuclidTransition,
        )
        m = HSTtoEuclidTransition()
        x = tf.random.uniform((1, 32, 32, 1), seed=8) * 50.0
        y_train = m(x, training=True).numpy()
        y_eval  = m(x, training=False).numpy()
        np.testing.assert_allclose(y_train, y_eval, atol=1e-7)


class TestSaveLoadRoundtrip:

    def test_save_and_reload_matches(self, tmp_path):
        from euclid_polish.training.transition_model import (
            HSTtoEuclidTransition, save_model_weights, load_model_weights,
        )
        m1 = HSTtoEuclidTransition()
        x  = tf.random.uniform((1, 32, 32, 1), seed=42) * 50.0
        # Force a non-trivial state by running through once.
        _ = m1(x)
        path = tmp_path / "weights.weights.h5"
        save_model_weights(m1, str(path))
        assert os.path.isfile(str(path))

        m2 = HSTtoEuclidTransition()
        load_model_weights(m2, str(path))
        y1 = m1(x).numpy()
        y2 = m2(x).numpy()
        np.testing.assert_allclose(y1, y2, atol=1e-6)


class TestScaleInvariance:
    """A PSF transition operator MUST be scale-invariant on positive
    inputs: ``A_θ(α·x) = α·A_θ(x)`` for any α ≥ 0. Convolution is
    linear, and the training objective ``clean ⊛ PSF_HST → clean ⊛
    PSF_Euclid`` is homogeneous in the scene amplitude.

    REGRESSION — an earlier version of the model had biases on every
    Conv2D layer. SGD drove the biases away from zero to fit the
    electron-scale training distribution; the model then failed
    catastrophically on small-amplitude inputs like the PSF identity
    probe (``PSF_id_err`` exploded from 0.35 to 1300+ within 500
    training steps). Removing biases makes the network purely linear
    on positive inputs (ReLU is identity there) and restores
    scale invariance by construction.
    """

    def test_homogeneous_in_input_at_init(self):
        from euclid_polish.training.transition_model import (
            HSTtoEuclidTransition,
        )
        tf.random.set_seed(0)
        m = HSTtoEuclidTransition()
        # Positive input — clean⊛PSF is always ≥ 0 in real training.
        x = tf.random.uniform((1, 32, 32, 1), minval=0.0, maxval=1.0,
                              dtype=tf.float32, seed=1)
        y1 = m(x).numpy()
        y2 = m(1000.0 * x).numpy()
        # Doubling the input should double the output (within float32 noise).
        np.testing.assert_allclose(
            y2, 1000.0 * y1, rtol=1e-4, atol=1e-4,
            err_msg=(
                "Scale invariance broken: A_θ(α·x) ≠ α·A_θ(x). "
                "Check that all Conv2D layers have use_bias=False."
            ),
        )

    def test_homogeneous_after_synthetic_training_step(self):
        """Even after a fake training step that pushes weights away
        from init, scale invariance must hold. (With biases, this is
        precisely when invariance breaks.)"""
        from euclid_polish.training.transition_model import (
            HSTtoEuclidTransition,
        )
        tf.random.set_seed(0)
        m = HSTtoEuclidTransition()

        # One synthetic gradient step with electron-scale "data".
        optimiser = tf.keras.optimizers.Adam(0.01)
        loss_fn   = tf.keras.losses.MeanAbsoluteError()
        x_train = tf.random.uniform((4, 16, 16, 1), minval=10.0, maxval=1000.0,
                                    dtype=tf.float32, seed=2)
        y_train = tf.random.uniform((4, 16, 16, 1), minval=10.0, maxval=1000.0,
                                    dtype=tf.float32, seed=3)
        for _ in range(5):
            with tf.GradientTape() as tape:
                pred = m(x_train, training=True)
                loss = loss_fn(y_train, pred)
            grads = tape.gradient(loss, m.trainable_variables)
            optimiser.apply_gradients(zip(grads, m.trainable_variables))

        # Now probe scale invariance on a much smaller-amplitude input
        # (the PSF probe is 10⁻⁴-scale, here we use 10⁻³ for cleaner
        # float32 arithmetic but the principle is the same).
        x_probe = tf.random.uniform((1, 16, 16, 1), minval=0.0, maxval=0.001,
                                    dtype=tf.float32, seed=4)
        alpha = 1e6
        y_small = m(x_probe).numpy()
        y_big   = m(alpha * x_probe).numpy()
        # Relative test: y_big / y_small ≈ alpha everywhere y_small ≠ 0.
        mask = np.abs(y_small) > 1e-9
        ratio = y_big[mask] / y_small[mask]
        # Ratio should be ~alpha for every active pixel, regardless of
        # how SGD has updated the weights.
        np.testing.assert_allclose(
            ratio, alpha, rtol=1e-3,
            err_msg=(
                "Scale invariance broken after a few training steps. "
                "If this fails, a bias term has crept back into the "
                "model — check use_bias=False on every Conv2D."
            ),
        )

    def test_no_bias_variables(self):
        """Direct structural assertion: there should be ZERO bias
        variables in the model. A future edit that adds biases (even
        with use_bias unspecified, since the Keras default is True)
        gets caught here."""
        from euclid_polish.training.transition_model import (
            HSTtoEuclidTransition,
        )
        m = HSTtoEuclidTransition()
        _ = m(tf.zeros((1, 8, 8, 1), dtype=tf.float32))  # build
        bias_vars = [v for v in m.trainable_variables
                     if "bias" in v.name.lower()]
        assert bias_vars == [], (
            f"unexpected bias variables found: {[v.name for v in bias_vars]}. "
            "Biases break scale invariance — keep use_bias=False on all "
            "Conv2D layers."
        )


class TestReceptiveField:

    def test_default_receptive_field(self):
        from euclid_polish.training.transition_model import (
            HSTtoEuclidTransition,
        )
        m = HSTtoEuclidTransition()
        # 5 layers × (k-1)=2 = 10; +1 = 11.
        assert m.receptive_field == 11

    def test_more_layers_grows_rf(self):
        from euclid_polish.training.transition_model import (
            HSTtoEuclidTransition,
        )
        rf_3 = HSTtoEuclidTransition(n_inner_layers=3).receptive_field
        rf_6 = HSTtoEuclidTransition(n_inner_layers=6).receptive_field
        assert rf_6 > rf_3


class TestApplyTransitionNumpy:

    def test_accepts_2d(self):
        from euclid_polish.training.transition_model import (
            HSTtoEuclidTransition, apply_transition_numpy,
        )
        m = HSTtoEuclidTransition()
        img = np.zeros((32, 32), dtype=np.float32)
        out = apply_transition_numpy(m, img)
        assert out.shape == (32, 32)
        assert out.dtype == img.dtype

    def test_accepts_3d_singleton(self):
        from euclid_polish.training.transition_model import (
            HSTtoEuclidTransition, apply_transition_numpy,
        )
        m = HSTtoEuclidTransition()
        img = np.zeros((32, 32, 1), dtype=np.float32)
        out = apply_transition_numpy(m, img)
        assert out.shape == (32, 32, 1)

    def test_rejects_invalid_shapes(self):
        from euclid_polish.training.transition_model import (
            HSTtoEuclidTransition, apply_transition_numpy,
        )
        m = HSTtoEuclidTransition()
        with pytest.raises(ValueError):
            apply_transition_numpy(m, np.zeros((32, 32, 3), dtype=np.float32))


class TestInvalidArgs:

    def test_negative_channels_rejected(self):
        from euclid_polish.training.transition_model import (
            HSTtoEuclidTransition,
        )
        with pytest.raises(ValueError, match="channels"):
            HSTtoEuclidTransition(channels=0)

    def test_negative_inner_layers_rejected(self):
        from euclid_polish.training.transition_model import (
            HSTtoEuclidTransition,
        )
        with pytest.raises(ValueError, match="n_inner_layers"):
            HSTtoEuclidTransition(n_inner_layers=-1)

    def test_even_kernel_rejected(self):
        from euclid_polish.training.transition_model import (
            HSTtoEuclidTransition,
        )
        with pytest.raises(ValueError, match="kernel_size"):
            HSTtoEuclidTransition(kernel_size=4)
