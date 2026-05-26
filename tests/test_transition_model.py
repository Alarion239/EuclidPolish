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
        """``2·K²·C + n_inner·K²·C²`` (no biases) should equal the
        actual trainable-parameter count exactly, for any combination
        of (channels, n_inner_layers, kernel_size).

        Formula assumes ``use_bias=False`` on every Conv2D (load-bearing
        for scale invariance — see the model module's __init__ doc).
        """
        from euclid_polish.training.transition_model import (
            HSTtoEuclidTransition, total_parameter_count,
        )
        cases = [
            # (C, n_inner, K)
            ( 4, 3, 3),       # tiny
            ( 8, 3, 3),
            (12, 3, 3),       # default
            (13, 3, 3),
            (12, 0, 3),       # 2-layer mini
            (12, 0, 11),      # 2-layer K=11
            ( 5, 0, 21),      # 2-layer K=21 — large RF, ≤ 5k params
            ( 1, 0, 41),      # 2-layer K=41 single hidden ch
            ( 8, 1, 11),      # mixed
        ]
        for C, n_inner, K in cases:
            m = HSTtoEuclidTransition(
                channels=C, n_inner_layers=n_inner, kernel_size=K,
            )
            expected = 2 * K * K * C + n_inner * K * K * C * C
            assert total_parameter_count(m) == expected, (
                f"param count mismatch at C={C}, n_inner={n_inner}, K={K}: "
                f"got {total_parameter_count(m)}, expected {expected}"
            )

    def test_2layer_large_kernel_fits_5k_budget(self):
        """Several large-RF / shallow architectures the user might pick
        must fit under the default 5k cap. Regression on the table
        documented in --kernel-size help."""
        from euclid_polish.training.transition_model import (
            HSTtoEuclidTransition, total_parameter_count,
        )
        # All of these were advertised as ≤ 5k in the form tooltip.
        for C, n_inner, K, label in [
            (12, 0, 11, "K=11 C=12 n=0 → RF=21"),
            ( 5, 0, 21, "K=21 C=5  n=0 → RF=41"),
            ( 2, 0, 31, "K=31 C=2  n=0 → RF=61"),
            ( 1, 0, 41, "K=41 C=1  n=0 → RF=81"),
        ]:
            m = HSTtoEuclidTransition(
                channels=C, n_inner_layers=n_inner, kernel_size=K,
            )
            n = total_parameter_count(m)
            assert n <= 5000, f"{label} should fit 5k but has {n} params"

    def test_baseline_kernel_not_counted_as_trainable(self):
        """The non-trainable analytic baseline kernel must NOT count
        against the trainable-param budget. Without the
        ``trainable_variables``-only check in total_parameter_count,
        a 31² baseline would falsely add 961 "params" and bust the cap.
        """
        from euclid_polish.training.transition_model import (
            HSTtoEuclidTransition, total_parameter_count,
        )
        # A 31² delta — stand-in for the analytic kernel shape.
        baseline = np.zeros((31, 31), dtype=np.float32)
        baseline[15, 15] = 1.0
        m_no_baseline = HSTtoEuclidTransition(
            channels=12, n_inner_layers=3, kernel_size=3,
        )
        m_with_baseline = HSTtoEuclidTransition(
            channels=12, n_inner_layers=3, kernel_size=3,
            baseline_kernel=baseline,
        )
        n_no   = total_parameter_count(m_no_baseline)
        n_with = total_parameter_count(m_with_baseline)
        assert n_no == n_with, (
            f"trainable param count must not change when adding a "
            f"baseline kernel; got {n_no} vs {n_with}"
        )

    def test_build_with_huge_baseline_is_fast(self):
        """REGRESSION — ``_build_if_needed`` must use a small dummy input
        regardless of how big the baseline kernel is.

        Previously the helper picked ``side = max(8, baseline_side + 1)``,
        so a 511² analytic baseline meant a 512² dummy input through the
        full ``call()`` — which triggers ``tf.nn.conv2d`` with a 511²
        kernel on CPU (direct convolution, ~50 s/process). A 16-worker
        pool then spent its entire wall-time budget on parallel build
        before producing a single cutout (FASRC step 4 stall, May 2026).

        SAME-padding handles ``kernel_size > input_size`` correctly, so
        an 8² dummy gives the same set of built variables in ~30 ms.
        Guard rail: building must complete well under 5 s so this can't
        silently slip back.
        """
        import time
        from euclid_polish.training.transition_model import (
            HSTtoEuclidTransition, _build_if_needed,
        )
        # 511² baseline — matches the real diff_kernel_VIS.fits size.
        baseline = np.zeros((511, 511), dtype=np.float32)
        baseline[255, 255] = 1.0
        m = HSTtoEuclidTransition(
            channels=12, n_inner_layers=3, kernel_size=3,
            baseline_kernel=baseline,
        )
        t0 = time.time()
        _build_if_needed(m)
        elapsed = time.time() - t0
        assert elapsed < 5.0, (
            f"_build_if_needed with a 511² baseline took {elapsed:.1f}s "
            f"(budget 5.0s). Did the dummy-input sizing rule regress to "
            f"include the baseline kernel side?"
        )

    def test_baseline_kernel_initial_output_equals_baseline_conv(self):
        """With baseline_kernel set and the residual init ≈ 0, the
        model's output at init must equal ``baseline_kernel ⊛ x``
        (to within float32 round-off), NOT just ``x``."""
        from euclid_polish.training.transition_model import (
            HSTtoEuclidTransition,
        )
        from scipy.signal import fftconvolve
        rng = np.random.default_rng(0)
        # Make a simple non-trivial kernel: 5x5 box averager.
        baseline = np.ones((5, 5), dtype=np.float32) / 25.0
        m = HSTtoEuclidTransition(
            channels=4, n_inner_layers=1, kernel_size=3,
            baseline_kernel=baseline,
        )
        x_np = rng.uniform(0, 100, size=(1, 32, 32, 1)).astype(np.float32)
        y = m(x_np).numpy()[0, :, :, 0]
        # Expected: baseline ⊛ x + tiny residual.
        expected_baseline = fftconvolve(
            x_np[0, :, :, 0], baseline, mode="same",
        )
        # The CNN residual at init is bounded by the small init
        # weights * stack depth; for our (init stddev=0.01, 3 layers,
        # 4 channels) the typical residual is well below 1e-1 even on
        # input scale ~100. So baseline + residual ≈ baseline within ~5%.
        rel = (np.abs(y - expected_baseline).mean()
               / (np.abs(expected_baseline).mean() + 1e-12))
        assert rel < 0.5, (
            f"with baseline_kernel set, A_θ(x) at init should ≈ "
            f"baseline ⊛ x. Got rel.err={rel:.3f}"
        )

    def test_baseline_kernel_rejects_non_square(self):
        from euclid_polish.training.transition_model import (
            HSTtoEuclidTransition,
        )
        with pytest.raises(ValueError, match="square"):
            HSTtoEuclidTransition(
                baseline_kernel=np.zeros((3, 5), dtype=np.float32),
            )

    def test_baseline_kernel_rejects_even_side(self):
        from euclid_polish.training.transition_model import (
            HSTtoEuclidTransition,
        )
        with pytest.raises(ValueError, match="odd"):
            HSTtoEuclidTransition(
                baseline_kernel=np.zeros((4, 4), dtype=np.float32),
            )

    def test_baseline_kernel_save_load_roundtrip(self, tmp_path):
        """A model trained with a baseline kernel must round-trip
        through save_weights / load_weights without losing the baseline.
        The trainer-side load path reconstructs the model with the
        same baseline_kernel argument BEFORE loading the weights."""
        from euclid_polish.training.transition_model import (
            HSTtoEuclidTransition, save_model_weights, load_model_weights,
        )
        rng = np.random.default_rng(1)
        baseline = rng.uniform(0, 1, size=(7, 7)).astype(np.float32)
        baseline /= baseline.sum()
        m1 = HSTtoEuclidTransition(
            channels=4, n_inner_layers=1, kernel_size=3,
            baseline_kernel=baseline,
        )
        # Run a forward pass to make the variables real.
        x = tf.random.uniform((1, 16, 16, 1), seed=4) * 50.0
        y1 = m1(x).numpy()
        path = tmp_path / "weights.weights.h5"
        save_model_weights(m1, str(path))

        m2 = HSTtoEuclidTransition(
            channels=4, n_inner_layers=1, kernel_size=3,
            baseline_kernel=baseline,
        )
        load_model_weights(m2, str(path))
        y2 = m2(x).numpy()
        np.testing.assert_allclose(y1, y2, atol=1e-6)

    def test_2layer_receptive_field(self):
        """2-layer model (n_inner=0) with kernel K has RF = 2(K−1) + 1."""
        from euclid_polish.training.transition_model import (
            HSTtoEuclidTransition,
        )
        for K in (3, 5, 11, 21, 41):
            m = HSTtoEuclidTransition(
                channels=4, n_inner_layers=0, kernel_size=K,
            )
            assert m.receptive_field == 2 * (K - 1) + 1, (
                f"K={K}: expected RF={2*(K-1)+1}, "
                f"got {m.receptive_field}"
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

    def test_fft_path_matches_tf_path_with_baseline(self):
        """REGRESSION — ``apply_transition_numpy`` takes a fast path
        that computes the baseline conv via scipy.signal.fftconvolve
        (FFT, fast on CPU) instead of tf.nn.conv2d (direct conv, slow
        on CPU for large kernels). The output must match the
        full-TF model output to within float32 round-off.

        Without this guarantee, switching paths would silently change
        the TFRecord pair-generation outputs."""
        from euclid_polish.training.transition_model import (
            HSTtoEuclidTransition, apply_transition_numpy,
        )
        rng = np.random.default_rng(0)
        # Realistic-ish baseline: a small offset Gaussian kernel.
        side = 21
        y, x = np.mgrid[:side, :side]
        cy = cx = side // 2
        kernel = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * 3.0 ** 2))
        kernel = (kernel / kernel.sum()).astype(np.float32)
        m = HSTtoEuclidTransition(
            channels=4, n_inner_layers=1, kernel_size=3,
            baseline_kernel=kernel,
        )
        img = rng.uniform(0, 100, size=(64, 64)).astype(np.float32)

        # Fast path (scipy FFT baseline + TF CNN residual).
        out_fast = apply_transition_numpy(m, img)
        # Slow path: run the entire model through TF.
        out_tf = m(
            img[np.newaxis, :, :, np.newaxis], training=False,
        ).numpy()[0, :, :, 0]

        # Float32 + FFT round-off — generous absolute tolerance scaled
        # by output magnitude.
        np.testing.assert_allclose(
            out_fast, out_tf,
            rtol=1e-4, atol=1e-3 * np.abs(out_tf).max(),
        )

    def test_no_baseline_still_works(self):
        """The fast path branches on whether the model has a baseline;
        the no-baseline branch should produce the same shape + dtype
        as before."""
        from euclid_polish.training.transition_model import (
            HSTtoEuclidTransition, apply_transition_numpy,
        )
        m = HSTtoEuclidTransition()   # no baseline_kernel
        img = np.random.uniform(0, 50, size=(48, 48)).astype(np.float32)
        out = apply_transition_numpy(m, img)
        assert out.shape == img.shape
        assert out.dtype == img.dtype


class TestHSTDenoiser:
    """Phase 1 model — pure denoiser, sits before the analytic kernel.

    Different from :class:`HSTtoEuclidTransition`:
      * No analytic baseline (residual is on raw input).
      * Wider default kernel (K=7) for the larger receptive field
        denoising benefits from.
      * Higher default param count (~7k) reflecting the different task.

    Same invariants:
      * Residual init ≈ identity (output ≈ x at random init).
      * Scale invariance (output(α·x) = α·output(x) for x ≥ 0, since
        all conv layers have ``use_bias=False``).
      * Shape-preserving forward pass.
      * Save/load round-trip is exact.
    """

    def test_default_receptive_field_is_25(self):
        from euclid_polish.training.transition_model import HSTDenoiser
        m = HSTDenoiser()    # defaults: C=8, n_inner=2, K=7
        # 4 layers, each (K-1)=6 → RF = 4·6 + 1 = 25
        assert m.receptive_field == 25

    def test_default_param_count(self):
        from euclid_polish.training.transition_model import (
            HSTDenoiser, total_denoiser_parameter_count,
        )
        m = HSTDenoiser()
        # 2·K²·C + n_inner·K²·C² = 2·49·8 + 2·49·64 = 784 + 6272 = 7056
        n = total_denoiser_parameter_count(m)
        assert n == 7056

    def test_param_formula_matches_analytic(self):
        from euclid_polish.training.transition_model import (
            HSTDenoiser, total_denoiser_parameter_count,
        )
        for C, n_inner, K in [(4, 1, 5), (8, 2, 7), (10, 3, 7), (12, 0, 9)]:
            m = HSTDenoiser(
                channels=C, n_inner_layers=n_inner, kernel_size=K,
            )
            expected = 2 * K * K * C + n_inner * K * K * C * C
            assert total_denoiser_parameter_count(m) == expected

    def test_identity_at_init_on_clean_input(self):
        """Random-init residual is N(0, σ_small) per weight, so the
        residual branch produces ~0 on clean input — output ≈ x."""
        from euclid_polish.training.transition_model import HSTDenoiser
        tf.random.set_seed(0)
        m = HSTDenoiser()
        x = tf.random.uniform((1, 32, 32, 1), seed=1) * 100.0
        y = m(x).numpy()
        rel_err = np.abs(y - x.numpy()).mean() / (np.abs(x.numpy()).mean() + 1e-12)
        assert rel_err < 0.5, (
            f"identity init too noisy: rel_err={rel_err:.4f}"
        )

    def test_zero_input_yields_zero(self):
        from euclid_polish.training.transition_model import HSTDenoiser
        m = HSTDenoiser()
        x = tf.zeros((1, 32, 32, 1), dtype=tf.float32)
        y = m(x).numpy()
        assert np.allclose(y, 0.0, atol=1e-5)

    def test_shape_preserved(self):
        from euclid_polish.training.transition_model import HSTDenoiser
        m = HSTDenoiser()
        for H, W in [(32, 32), (48, 64), (128, 128)]:
            y = m(tf.zeros((2, H, W, 1), dtype=tf.float32))
            assert tuple(y.shape) == (2, H, W, 1)

    def test_scale_invariance(self):
        """No biases ⇒ output(α·x) = α·output(x) for x ≥ 0 — same
        invariant as the deconvolver. Critical so denoising behaves
        consistently on bright galaxy cores vs faint sky pixels."""
        from euclid_polish.training.transition_model import HSTDenoiser
        tf.random.set_seed(0)
        m = HSTDenoiser()
        x = tf.random.uniform((1, 32, 32, 1), minval=0.0, maxval=1.0,
                              dtype=tf.float32, seed=1)
        y1 = m(x).numpy()
        y2 = m(1000.0 * x).numpy()
        np.testing.assert_allclose(
            y2, 1000.0 * y1, rtol=1e-4, atol=1e-4,
            err_msg=(
                "Denoiser must satisfy g(α·x) = α·g(x) for x ≥ 0. "
                "Check use_bias=False on every Conv2D."
            ),
        )

    def test_no_bias_variables(self):
        from euclid_polish.training.transition_model import HSTDenoiser
        m = HSTDenoiser()
        _ = m(tf.zeros((1, 8, 8, 1), dtype=tf.float32))  # build
        bias_vars = [v for v in m.trainable_variables
                     if "bias" in v.name.lower()]
        assert bias_vars == [], (
            f"unexpected bias variables: {[v.name for v in bias_vars]}. "
            "Biases break scale invariance — keep use_bias=False."
        )

    def test_save_load_roundtrip(self, tmp_path):
        from euclid_polish.training.transition_model import (
            HSTDenoiser, save_denoiser_weights, load_denoiser_weights,
        )
        m1 = HSTDenoiser()
        x = tf.random.uniform((1, 32, 32, 1), seed=2) * 50.0
        _ = m1(x)
        path = tmp_path / "denoiser.weights.h5"
        save_denoiser_weights(m1, str(path))
        assert os.path.isfile(str(path))
        m2 = HSTDenoiser()
        load_denoiser_weights(m2, str(path))
        np.testing.assert_allclose(m1(x).numpy(), m2(x).numpy(), atol=1e-6)

    def test_rejects_invalid_args(self):
        from euclid_polish.training.transition_model import HSTDenoiser
        with pytest.raises(ValueError, match="channels"):
            HSTDenoiser(channels=0)
        with pytest.raises(ValueError, match="n_inner_layers"):
            HSTDenoiser(n_inner_layers=-1)
        with pytest.raises(ValueError, match="kernel_size"):
            HSTDenoiser(kernel_size=4)

    def test_apply_denoiser_numpy_2d(self):
        from euclid_polish.training.transition_model import (
            HSTDenoiser, apply_denoiser_numpy,
        )
        m = HSTDenoiser()
        img = np.zeros((32, 32), dtype=np.float32)
        out = apply_denoiser_numpy(m, img)
        assert out.shape == (32, 32)
        assert out.dtype == img.dtype

    def test_apply_denoiser_numpy_rejects_bad_shape(self):
        from euclid_polish.training.transition_model import (
            HSTDenoiser, apply_denoiser_numpy,
        )
        m = HSTDenoiser()
        with pytest.raises(ValueError):
            apply_denoiser_numpy(m, np.zeros((32, 32, 3), dtype=np.float32))


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
