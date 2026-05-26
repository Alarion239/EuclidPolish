"""Tests for the transition-model training-time augmentations."""

from __future__ import annotations

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")


# ---------------------------------------------------------------------------
# add_hlsp_noise — Phase 1 (denoiser) noise model
# ---------------------------------------------------------------------------

class TestAddHlspNoise:
    """The noise helper has to produce the right variance at every
    pixel intensity AND stay well-defined on sky-subtracted negative
    pixels — the literal-Poisson path that ``rng.poisson(λ)`` would
    take fails on the latter."""

    def test_sky_only_matches_sigma_floor(self):
        """On all-zero input the only noise source should be the
        sky/read floor; measured σ should match within √(2/N) at
        large N."""
        from euclid_polish.training.transition_augmentations import (
            add_hlsp_noise,
        )
        rng = np.random.default_rng(0)
        clean = np.zeros((512, 512), dtype=np.float32)
        noisy = add_hlsp_noise(
            clean, alpha=0.8, sigma_floor=12.0, rng=rng,
        )
        measured_sigma = float(noisy.std())
        # 512² samples → 1-σ uncertainty on the std estimate is ~σ/√(2N)
        # ≈ 0.013 → 5σ tolerance is ~5%.
        assert abs(measured_sigma - 12.0) / 12.0 < 0.03, (
            f"sky-only σ should be {12.0}, got {measured_sigma:.3f}"
        )

    def test_shot_variance_scales_linearly_with_signal(self):
        """At zero floor, σ² should equal ``α · signal``. Test on a
        flat patch at a few signal levels."""
        from euclid_polish.training.transition_augmentations import (
            add_hlsp_noise,
        )
        alpha = 0.8
        rng = np.random.default_rng(1)
        for level in (100.0, 1000.0, 10000.0):
            clean = np.full((512, 512), level, dtype=np.float32)
            noisy = add_hlsp_noise(
                clean, alpha=alpha, sigma_floor=0.0, rng=rng,
            )
            measured_var = float(noisy.var())
            expected_var = alpha * level
            # 5% tolerance (large-N estimate of variance).
            assert abs(measured_var - expected_var) / expected_var < 0.05, (
                f"shot variance at signal={level} should be {expected_var}, "
                f"got {measured_var:.2f}"
            )

    def test_negative_pixels_get_only_sky_floor(self):
        """REGRESSION — sky-subtracted HLSP cutouts have negative
        pixels near sky. ``max(clean, 0)`` in the shot-variance term
        means negative pixels contribute zero shot noise; only the
        sky floor σ applies. Otherwise we'd get a NaN from
        ``√(α · negative)``."""
        from euclid_polish.training.transition_augmentations import (
            add_hlsp_noise,
        )
        rng = np.random.default_rng(2)
        clean = -100.0 * np.ones((512, 512), dtype=np.float32)
        noisy = add_hlsp_noise(
            clean, alpha=0.8, sigma_floor=10.0, rng=rng,
        )
        assert np.all(np.isfinite(noisy)), (
            "negative input must not produce NaN/inf — shot-variance "
            "term should clip to ``max(clean, 0)``"
        )
        # Variance should match just the sky floor.
        measured_sigma = float(noisy.std())
        assert abs(measured_sigma - 10.0) / 10.0 < 0.05

    def test_rejects_negative_alpha(self):
        from euclid_polish.training.transition_augmentations import (
            add_hlsp_noise,
        )
        with pytest.raises(ValueError, match="alpha"):
            add_hlsp_noise(np.zeros((4, 4)), alpha=-0.1, sigma_floor=5.0)

    def test_rejects_negative_sigma_floor(self):
        from euclid_polish.training.transition_augmentations import (
            add_hlsp_noise,
        )
        with pytest.raises(ValueError, match="sigma_floor"):
            add_hlsp_noise(np.zeros((4, 4)), alpha=0.5, sigma_floor=-1.0)

    def test_seeded_rng_is_reproducible(self):
        from euclid_polish.training.transition_augmentations import (
            add_hlsp_noise,
        )
        clean = np.ones((16, 16), dtype=np.float32) * 100.0
        a = add_hlsp_noise(
            clean, alpha=0.8, sigma_floor=10.0,
            rng=np.random.default_rng(42),
        )
        b = add_hlsp_noise(
            clean, alpha=0.8, sigma_floor=10.0,
            rng=np.random.default_rng(42),
        )
        np.testing.assert_array_equal(a, b)


class TestSampleHlspNoiseParams:

    def test_samples_within_range(self):
        from euclid_polish.training.transition_augmentations import (
            sample_hlsp_noise_params,
        )
        rng = np.random.default_rng(0)
        for _ in range(500):
            a, s = sample_hlsp_noise_params(rng)
            assert 0.5 <= a <= 1.0
            assert 8.0 <= s <= 22.0


# ---------------------------------------------------------------------------
# ensure_unit_sum — defensive PSF normalisation at the convolution site
# ---------------------------------------------------------------------------

class TestEnsureUnitSum:
    """The helper is a no-op for already-normalised inputs and rescales
    otherwise. The convolution sites call it right before fftconvolve so
    a PSF FITS saved un-normalised (e.g. raw EPSFBuilder output with
    sum≈3) can't silently rescale the convolution amplitude."""

    def test_already_unit_sum_passes_through(self):
        from euclid_polish.training.transition_augmentations import (
            ensure_unit_sum,
        )
        psf = np.zeros((5, 5), dtype=np.float32)
        psf[2, 2] = 1.0   # sum already = 1
        out = ensure_unit_sum(psf)
        assert out is psf, "no-op path should return the same array (no copy)"

    def test_un_normalised_gets_rescaled(self):
        from euclid_polish.training.transition_augmentations import (
            ensure_unit_sum,
        )
        psf = np.zeros((5, 5), dtype=np.float32)
        psf[2, 2] = 3.0   # sum = 3, matches the real Euclid VIS FITS case
        out = ensure_unit_sum(psf)
        assert abs(float(out.sum()) - 1.0) < 1e-6, (
            f"expected sum=1 after normalisation, got {out.sum()}"
        )
        # Shape preserved.
        assert out.shape == psf.shape
        # Dtype preserved (no silent upcast to float64).
        assert out.dtype == psf.dtype

    def test_zero_sum_is_no_op(self):
        """Degenerate: dividing by zero would NaN out. Helper must
        return the original array unchanged in that case so the
        caller's existing error-handling path triggers."""
        from euclid_polish.training.transition_augmentations import (
            ensure_unit_sum,
        )
        psf = np.zeros((5, 5), dtype=np.float32)
        out = ensure_unit_sum(psf)
        assert np.array_equal(out, psf)
        assert not np.isnan(out).any()

    def test_convolution_amplitude_unaffected_by_input_normalisation(self):
        """REGRESSION — the whole point of ``ensure_unit_sum`` is that a
        scene convolved with an un-normalised PSF should produce the same
        result as a scene convolved with the same PSF normalised to sum=1.
        Without the helper, a PSF with sum=3 would silently scale the
        output by 3 and shift the photometric L1 loss baseline."""
        from euclid_polish.training.transition_augmentations import (
            ensure_unit_sum,
        )
        from scipy.signal import fftconvolve
        # Two PSFs with identical shape, different total flux.
        rng = np.random.default_rng(0)
        psf_unit = rng.uniform(0.1, 1.0, size=(7, 7)).astype(np.float32)
        psf_unit /= psf_unit.sum()           # sum = 1
        psf_inflated = psf_unit * 3.0        # sum = 3, same shape
        scene = rng.uniform(0, 100, size=(32, 32)).astype(np.float32)
        out_unit     = fftconvolve(scene, ensure_unit_sum(psf_unit),     mode="same")
        out_inflated = fftconvolve(scene, ensure_unit_sum(psf_inflated), mode="same")
        np.testing.assert_allclose(
            out_unit, out_inflated, atol=1e-4, rtol=1e-4,
            err_msg=(
                "ensure_unit_sum should make a sum=3 PSF give the same "
                "convolution result as the sum=1 version — otherwise the "
                "Euclid-vs-HST normalisation mismatch propagates into "
                "training data amplitudes."
            ),
        )


# ---------------------------------------------------------------------------
# Fixtures: tiny Gaussian PSFs on the HR grid
# ---------------------------------------------------------------------------

def _gauss_kernel(side: int, sigma: float) -> np.ndarray:
    """Unit-flux Gaussian on an odd-side grid."""
    y, x = np.mgrid[:side, :side]
    cy = cx = (side - 1) / 2.0
    g = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * sigma ** 2))
    return (g / g.sum()).astype(np.float32)


@pytest.fixture
def psfs():
    """A tight (HST-like) PSF and a broader (Euclid-like) PSF on the
    same grid, both unit-flux. Sigma values are chosen to give visibly
    different convolutions at typical star positions."""
    psf_hst    = _gauss_kernel(31, sigma=2.0)
    psf_euclid = _gauss_kernel(31, sigma=4.0)
    return psf_hst, psf_euclid


# ---------------------------------------------------------------------------
# make_star_field — numpy core
# ---------------------------------------------------------------------------

class TestMakeStarField:

    def test_shape_and_dtype_default_rebin(self, psfs):
        """At default rebin_factor=2, input is HR and target is LR
        (half the side per axis)."""
        from euclid_polish.training.transition_augmentations import (
            make_star_field,
        )
        psf_h, psf_e = psfs
        inp, tgt = make_star_field(
            psf_h, psf_e,
            image_size=128, n_stars=5,
            rng=np.random.default_rng(0),
        )
        assert inp.shape == (128, 128, 1)
        assert tgt.shape == (64, 64, 1)    # 128 / rebin_factor=2 = 64
        assert inp.dtype == np.float32
        assert tgt.dtype == np.float32

    def test_shape_with_rebin_one_keeps_target_at_hr(self, psfs):
        """rebin_factor=1 is the legacy mode (target stays at HR).
        Verify both sides have the same shape in that case."""
        from euclid_polish.training.transition_augmentations import (
            make_star_field,
        )
        psf_h, psf_e = psfs
        inp, tgt = make_star_field(
            psf_h, psf_e,
            image_size=64, n_stars=2, rebin_factor=1,
            rng=np.random.default_rng(0),
        )
        assert inp.shape == (64, 64, 1)
        assert tgt.shape == (64, 64, 1)

    def test_flux_conserved_by_unit_psf(self, psfs):
        """Sum of input ≈ sum of target (both pass through unit-flux
        PSFs, and sum-rebin is photometric so it preserves total
        flux). Edge effects: stars near the boundary lose ~1-2σ of
        flux off the array — with sigma ≤ 4 px and 128² grid the
        loss is well under 5%. Sum-rebin doesn't change this — it
        bins what's already in the array."""
        from euclid_polish.training.transition_augmentations import (
            make_star_field,
        )
        psf_h, psf_e = psfs
        rng = np.random.default_rng(0)
        inp, tgt = make_star_field(
            psf_h, psf_e,
            image_size=128, n_stars=3, amp_min=100.0, amp_max=300.0,
            rng=rng,
        )
        assert inp.sum() > 200.0
        assert tgt.sum() > 200.0
        # Both sides carry essentially the same total flux: sum-rebin
        # preserves it exactly, and the two unit-flux PSFs preserve it
        # modulo edge clipping.
        rel_diff = abs(inp.sum() - tgt.sum()) / max(inp.sum(), tgt.sum())
        assert rel_diff < 0.10, (
            f"input and target totals should match within ~10% (unit-flux "
            f"PSFs + photometric rebin); got input={inp.sum():.2f} vs "
            f"target={tgt.sum():.2f}"
        )

    def test_target_is_sum_rebin_of_target_at_hr(self, psfs):
        """Direct invariant: target_LR == sum_rebin(scene ⊛ PSF_Euclid).

        Construct a single-star scene by hand and verify the target
        side of make_star_field equals the explicit sum-rebin of the
        convolved scene.
        """
        from euclid_polish.training.transition_augmentations import (
            make_star_field, sum_rebin_2d,
        )
        from scipy import signal as scipy_signal
        psf_h, psf_e = psfs

        # Deterministic single-star delta: known position + amplitude.
        rng = np.random.default_rng(2024)
        inp, tgt = make_star_field(
            psf_h, psf_e, image_size=64, n_stars=1,
            amp_min=200.0, amp_max=200.0,        # fix amp
            rebin_factor=2, rng=rng,
        )
        # Reverse-engineer the delta image from input: deconvolve isn't
        # easy here, so instead recompute by constructing the same
        # field with the same RNG sequence the helper uses internally.
        rng2 = np.random.default_rng(2024)
        deltas = np.zeros((64, 64), dtype=np.float64)
        # _gen does: cy, cx = integers(0, image_size); amp = uniform(amp_min, amp_max)
        cy = int(rng2.integers(0, 64))
        cx = int(rng2.integers(0, 64))
        amp = float(rng2.uniform(200.0, 200.0))
        deltas[cy, cx] = amp
        tgt_hr_expected = scipy_signal.fftconvolve(
            deltas, psf_e.astype(np.float64), mode="same",
        ).astype(np.float32)
        tgt_lr_expected = sum_rebin_2d(tgt_hr_expected, 2)[..., np.newaxis]
        np.testing.assert_allclose(tgt, tgt_lr_expected, atol=1e-5)

    def test_input_and_target_match_when_psfs_match_with_rebin(self, psfs):
        """When both PSFs are equal AND rebin_factor=1, input == target.

        With rebin_factor=2 they're equal only up to the rebin
        operation, so we compare ``sum_rebin(input)`` to ``target``."""
        from euclid_polish.training.transition_augmentations import (
            make_star_field, sum_rebin_2d,
        )
        psf_h, _ = psfs
        # rebin=1: literal equality.
        inp1, tgt1 = make_star_field(
            psf_h, psf_h, image_size=64, n_stars=3, rebin_factor=1,
            rng=np.random.default_rng(7),
        )
        np.testing.assert_allclose(inp1, tgt1, atol=1e-6)
        # rebin=2: equality after rebin of input.
        inp2, tgt2 = make_star_field(
            psf_h, psf_h, image_size=64, n_stars=3, rebin_factor=2,
            rng=np.random.default_rng(7),
        )
        inp2_lr = sum_rebin_2d(inp2[..., 0], 2)[..., np.newaxis]
        np.testing.assert_allclose(tgt2, inp2_lr, atol=1e-6)

    def test_zero_stars_yields_zero_field(self, psfs):
        from euclid_polish.training.transition_augmentations import (
            make_star_field,
        )
        psf_h, psf_e = psfs
        inp, tgt = make_star_field(
            psf_h, psf_e,
            image_size=64, n_stars=0,
            rng=np.random.default_rng(0),
        )
        assert inp.sum() == 0.0
        assert tgt.sum() == 0.0

    def test_reproducible_with_seed(self, psfs):
        from euclid_polish.training.transition_augmentations import (
            make_star_field,
        )
        psf_h, psf_e = psfs
        inp1, tgt1 = make_star_field(
            psf_h, psf_e, image_size=64, n_stars=3,
            rng=np.random.default_rng(42),
        )
        inp2, tgt2 = make_star_field(
            psf_h, psf_e, image_size=64, n_stars=3,
            rng=np.random.default_rng(42),
        )
        np.testing.assert_array_equal(inp1, inp2)
        np.testing.assert_array_equal(tgt1, tgt2)

    def test_mismatched_psf_shapes_rejected(self):
        from euclid_polish.training.transition_augmentations import (
            make_star_field,
        )
        a = _gauss_kernel(31, 2.0)
        b = _gauss_kernel(33, 4.0)
        with pytest.raises(ValueError, match="share shape"):
            make_star_field(a, b, image_size=64, n_stars=1)

    def test_image_size_must_be_divisible_by_rebin(self, psfs):
        from euclid_polish.training.transition_augmentations import (
            make_star_field,
        )
        psf_h, psf_e = psfs
        with pytest.raises(ValueError, match="divisible by"):
            make_star_field(psf_h, psf_e, image_size=65, n_stars=1,
                            rebin_factor=2)


class TestSumRebin2d:
    """Unit tests for the photometric sum-rebin helper."""

    def test_sums_inside_block(self):
        from euclid_polish.training.transition_augmentations import (
            sum_rebin_2d,
        )
        a = np.array([[1, 2, 3, 4],
                      [5, 6, 7, 8],
                      [9, 10, 11, 12],
                      [13, 14, 15, 16]], dtype=np.float32)
        out = sum_rebin_2d(a, 2)
        # Top-left 2×2 sums to 1+2+5+6=14; etc.
        expected = np.array([[14, 22],
                             [46, 54]], dtype=np.float32)
        np.testing.assert_array_equal(out, expected)

    def test_factor_1_is_copy(self):
        from euclid_polish.training.transition_augmentations import (
            sum_rebin_2d,
        )
        a = np.arange(16, dtype=np.float32).reshape(4, 4)
        out = sum_rebin_2d(a, 1)
        np.testing.assert_array_equal(out, a)
        # Must be a copy, not the same buffer.
        out[0, 0] = -1
        assert a[0, 0] == 0

    def test_rejects_non_divisible(self):
        from euclid_polish.training.transition_augmentations import (
            sum_rebin_2d,
        )
        with pytest.raises(ValueError, match="not divisible"):
            sum_rebin_2d(np.zeros((5, 4), dtype=np.float32), 2)

    def test_handles_channel_axis(self):
        from euclid_polish.training.transition_augmentations import (
            sum_rebin_2d,
        )
        a = np.ones((4, 4, 3), dtype=np.float32)
        out = sum_rebin_2d(a, 2)
        assert out.shape == (2, 2, 3)
        np.testing.assert_array_equal(out, 4.0 * np.ones((2, 2, 3)))

    def test_preserves_total_flux(self):
        """Photometric invariant: sum_rebin preserves the total."""
        from euclid_polish.training.transition_augmentations import (
            sum_rebin_2d,
        )
        rng = np.random.default_rng(0)
        a = rng.uniform(0, 100, size=(8, 8)).astype(np.float32)
        out = sum_rebin_2d(a, 4)
        np.testing.assert_allclose(out.sum(), a.sum(), rtol=1e-5)


# ---------------------------------------------------------------------------
# build_star_pair_dataset — tf.data producer
# ---------------------------------------------------------------------------

class TestStarPairDataset:

    def test_emits_correct_shapes_with_rebin(self, psfs):
        """Default rebin_factor=2 → target side = image_size / 2."""
        from euclid_polish.training.transition_augmentations import (
            build_star_pair_dataset,
        )
        psf_h, psf_e = psfs
        ds = build_star_pair_dataset(
            psf_h, psf_e, image_size=64,
            n_stars_min=1, n_stars_max=4, seed=1,
        )
        for inp, tgt in ds.take(3):
            assert tuple(inp.shape) == (64, 64, 1)
            assert tuple(tgt.shape) == (32, 32, 1)   # rebin_factor=2 default

    def test_emits_correct_shapes_with_rebin_one(self, psfs):
        """rebin_factor=1 keeps target at HR (legacy mode)."""
        from euclid_polish.training.transition_augmentations import (
            build_star_pair_dataset,
        )
        psf_h, psf_e = psfs
        ds = build_star_pair_dataset(
            psf_h, psf_e, image_size=64, rebin_factor=1,
            n_stars_min=1, n_stars_max=4, seed=1,
        )
        for inp, tgt in ds.take(2):
            assert tuple(inp.shape) == (64, 64, 1)
            assert tuple(tgt.shape) == (64, 64, 1)

    def test_emits_unique_samples_per_draw(self, psfs):
        """Two consecutive draws should differ — the generator must
        not be caching."""
        from euclid_polish.training.transition_augmentations import (
            build_star_pair_dataset,
        )
        psf_h, psf_e = psfs
        ds = build_star_pair_dataset(
            psf_h, psf_e, image_size=64,
            n_stars_min=2, n_stars_max=6, seed=1,
        )
        samples = list(ds.take(3))
        # No pair of consecutive samples should be identical (3 stars
        # with random positions has effectively zero collision rate).
        for i in range(len(samples) - 1):
            inp_i, _ = samples[i]
            inp_j, _ = samples[i + 1]
            assert not np.array_equal(inp_i.numpy(), inp_j.numpy())

    def test_rejects_inverted_min_max(self, psfs):
        from euclid_polish.training.transition_augmentations import (
            build_star_pair_dataset,
        )
        psf_h, psf_e = psfs
        with pytest.raises(ValueError, match="n_stars_max"):
            build_star_pair_dataset(
                psf_h, psf_e, image_size=64,
                n_stars_min=10, n_stars_max=5,
            )


# ---------------------------------------------------------------------------
# embed_in_canvas + make_psf_identity_pair (validation seed)
# ---------------------------------------------------------------------------

class TestEmbedInCanvas:

    def test_identity_when_sizes_match(self):
        from euclid_polish.training.transition_augmentations import (
            embed_in_canvas,
        )
        a = np.arange(25.0, dtype=np.float32).reshape(5, 5)
        out = embed_in_canvas(a, 5)
        np.testing.assert_array_equal(out, a)

    def test_centre_pad_when_smaller(self):
        from euclid_polish.training.transition_augmentations import (
            embed_in_canvas,
        )
        # 3×3 hot pixel pattern centred in a 7×7 canvas → centre is (3, 3).
        a = np.ones((3, 3), dtype=np.float32) * 7.0
        out = embed_in_canvas(a, 7)
        assert out.shape == (7, 7)
        # Pad zone is zeros.
        assert np.all(out[:2, :] == 0)
        assert np.all(out[-2:, :] == 0)
        # Inner 3×3 block matches the source.
        np.testing.assert_array_equal(out[2:5, 2:5], a)

    def test_centre_crop_when_larger(self):
        from euclid_polish.training.transition_augmentations import (
            embed_in_canvas,
        )
        # 7×7 with a centre pixel marked; cropping to 3×3 keeps the centre.
        a = np.zeros((7, 7), dtype=np.float32)
        a[3, 3] = 42.0
        out = embed_in_canvas(a, 3)
        assert out.shape == (3, 3)
        assert out[1, 1] == 42.0

    def test_preserves_centroid_on_pad(self):
        """Centroid must end up on the canvas's geometric centre after
        padding — otherwise the PSF identity sample would land off-centre
        and confuse the diagnostic."""
        from euclid_polish.training.transition_augmentations import (
            embed_in_canvas,
        )
        # Gaussian-ish 5×5 → embed in 9×9.
        a = np.zeros((5, 5), dtype=np.float32)
        a[2, 2] = 1.0   # centre pixel
        out = embed_in_canvas(a, 9)
        # The hot pixel should now be at (4, 4) — the canvas centre.
        assert out[4, 4] == 1.0
        assert out.sum() == 1.0

    def test_rejects_non_square(self):
        from euclid_polish.training.transition_augmentations import (
            embed_in_canvas,
        )
        with pytest.raises(ValueError, match="square"):
            embed_in_canvas(np.zeros((3, 5), dtype=np.float32), 7)

    def test_rejects_non_2d(self):
        from euclid_polish.training.transition_augmentations import (
            embed_in_canvas,
        )
        with pytest.raises(ValueError, match="2-D"):
            embed_in_canvas(np.zeros((3, 3, 3), dtype=np.float32), 7)


class TestMakePsfIdentityPair:

    def test_shape_and_dtype_default_rebin(self, psfs):
        """input HR (64²), target LR (32² at rebin=2)."""
        from euclid_polish.training.transition_augmentations import (
            make_psf_identity_pair,
        )
        psf_h, psf_e = psfs
        inp, tgt = make_psf_identity_pair(psf_h, psf_e, image_size=64)
        assert inp.shape == (64, 64, 1)
        assert tgt.shape == (32, 32, 1)
        assert inp.dtype == np.float32

    def test_shape_with_rebin_one(self, psfs):
        """Legacy mode: input and target both HR."""
        from euclid_polish.training.transition_augmentations import (
            make_psf_identity_pair,
        )
        psf_h, psf_e = psfs
        inp, tgt = make_psf_identity_pair(
            psf_h, psf_e, image_size=64, rebin_factor=1,
        )
        assert inp.shape == (64, 64, 1)
        assert tgt.shape == (64, 64, 1)

    def test_input_is_hst_target_is_rebinned_euclid(self, psfs):
        """The pair must be ordered (input=HST_HR, target=rebin(Euclid)).
        Otherwise the model would have to learn to invert the transition
        direction, which is the wrong thing."""
        from euclid_polish.training.transition_augmentations import (
            embed_in_canvas, make_psf_identity_pair, sum_rebin_2d,
        )
        psf_h, psf_e = psfs
        inp, tgt = make_psf_identity_pair(psf_h, psf_e, image_size=64)
        np.testing.assert_array_equal(
            inp[..., 0], embed_in_canvas(psf_h, 64),
        )
        np.testing.assert_array_equal(
            tgt[..., 0],
            sum_rebin_2d(embed_in_canvas(psf_e, 64), 2),
        )

    def test_unit_flux_preserved(self, psfs):
        """Both PSFs sum to 1 by construction; embedding in a larger
        canvas zero-pads, and sum-rebin is photometric — so the totals
        must still be ~1."""
        from euclid_polish.training.transition_augmentations import (
            make_psf_identity_pair,
        )
        psf_h, psf_e = psfs
        inp, tgt = make_psf_identity_pair(psf_h, psf_e, image_size=64)
        assert abs(inp.sum() - 1.0) < 1e-4
        assert abs(tgt.sum() - 1.0) < 1e-4

    def test_mismatched_psf_shapes_rejected(self):
        from euclid_polish.training.transition_augmentations import (
            make_psf_identity_pair,
        )
        a = _gauss_kernel(31, 2.0)
        b = _gauss_kernel(33, 4.0)
        with pytest.raises(ValueError, match="share shape"):
            make_psf_identity_pair(a, b, image_size=64)


# ---------------------------------------------------------------------------
# apply_linear_combo_augmentation — tf.data transformer
# ---------------------------------------------------------------------------

class TestLinearComboAugmentation:

    def _make_fixed_ds(self, value: float, n: int = 16):
        """A tf.data.Dataset emitting ``n`` copies of a constant pair.

        Using a constant value lets us reason about the linear
        combination output algebraically: if A == B == c, then
        α·A + β·B = (α + β)·c.
        """
        x = np.full((1, 8, 8, 1), value, dtype=np.float32)
        y = np.full((1, 8, 8, 1), value * 2, dtype=np.float32)
        ds = tf.data.Dataset.from_tensor_slices((x, y))
        ds = ds.unbatch().repeat(n)
        # Unbatch operates on the leading axis; result yields one (H,W,1)
        # tensor pair per element.
        return ds

    def test_fraction_zero_returns_source_unchanged(self):
        from euclid_polish.training.transition_augmentations import (
            apply_linear_combo_augmentation,
        )
        ds = self._make_fixed_ds(1.0)
        ds2 = apply_linear_combo_augmentation(ds, fraction=0.0, seed=42)
        # Identity: returned dataset is exactly the source.
        assert ds2 is ds

    def test_fraction_one_always_mixes(self):
        """At fraction=1, every example is a linear combination. With
        a constant-valued source, the result must equal (α+β)·c for
        per-example α, β within the documented ranges."""
        from euclid_polish.training.transition_augmentations import (
            apply_linear_combo_augmentation,
        )
        # Same value in A and B means (α·A + β·B) = (α+β)·A.
        ds = self._make_fixed_ds(1.0, n=32)
        ds2 = apply_linear_combo_augmentation(ds, fraction=1.0, seed=42)
        # Default α ∈ [0.3, 1.5], β ∈ [-0.5, 0.5] → α+β ∈ [-0.2, 2.0].
        for inp, tgt in ds2.take(8):
            v_inp = float(inp.numpy().mean())
            v_tgt = float(tgt.numpy().mean())
            # input was a constant 1.0; target was constant 2.0.
            # The scaled values should equal each other (since input ==
            # target factor of 2) and lie in the [-0.2 * 2, 2.0 * 2] = [-0.4, 4.0] range.
            np.testing.assert_allclose(v_tgt, 2.0 * v_inp, atol=1e-5)
            assert -0.5 < v_inp < 2.1

    def test_input_and_target_scaled_identically(self, psfs):
        """The same α, β are applied to input and target. So
        ``target / input`` stays at the source ratio, regardless of
        the random scaling."""
        from euclid_polish.training.transition_augmentations import (
            apply_linear_combo_augmentation,
        )
        # Source has target = 3 × input. Linearity must preserve that.
        x = np.full((1, 8, 8, 1), 1.0, dtype=np.float32)
        y = np.full((1, 8, 8, 1), 3.0, dtype=np.float32)
        ds = tf.data.Dataset.from_tensor_slices((x, y)).unbatch().repeat(64)
        ds2 = apply_linear_combo_augmentation(ds, fraction=1.0, seed=11)
        for inp, tgt in ds2.take(8):
            v_inp = float(inp.numpy().mean())
            v_tgt = float(tgt.numpy().mean())
            if abs(v_inp) > 1e-3:
                np.testing.assert_allclose(
                    v_tgt / v_inp, 3.0, rtol=1e-5,
                    err_msg=(
                        f"target/input ratio drifted under augmentation: "
                        f"input={v_inp}, target={v_tgt}"
                    ),
                )

    def test_invalid_fraction_rejected(self):
        from euclid_polish.training.transition_augmentations import (
            apply_linear_combo_augmentation,
        )
        ds = self._make_fixed_ds(1.0)
        with pytest.raises(ValueError, match="fraction"):
            apply_linear_combo_augmentation(ds, fraction=1.5, seed=0)
        with pytest.raises(ValueError, match="fraction"):
            apply_linear_combo_augmentation(ds, fraction=-0.1, seed=0)
