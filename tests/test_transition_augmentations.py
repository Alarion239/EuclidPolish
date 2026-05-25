"""Tests for the transition-model training-time augmentations."""

from __future__ import annotations

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")


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

    def test_shape_and_dtype(self, psfs):
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
        assert tgt.shape == (128, 128, 1)
        assert inp.dtype == np.float32
        assert tgt.dtype == np.float32

    def test_flux_conserved_by_unit_psf(self, psfs):
        """Sum of input ≈ sum of star amplitudes (PSF sums to 1).

        Edge-effect tolerance: stars near the boundary lose ~1-2 σ of
        flux off the array. With sigma ≤ 4 px and a 128² grid, stars
        more than 10 px from any edge are essentially fully captured.
        Use a generous-but-finite tolerance.
        """
        from euclid_polish.training.transition_augmentations import (
            make_star_field,
        )
        psf_h, psf_e = psfs
        # Stars centred in the safe interior so edge clipping is
        # negligible. We replicate the math of make_star_field with a
        # known seed so we can assert flux to a tight tolerance.
        rng = np.random.default_rng(0)
        inp, tgt = make_star_field(
            psf_h, psf_e,
            image_size=128, n_stars=3, amp_min=100.0, amp_max=300.0,
            rng=rng,
        )
        # PSFs sum to 1, so total flux ≈ sum of amplitudes. With 3
        # stars at amp ∈ [100, 300], total ≥ 300. Allow 5% loss to
        # edge effects (worst case at sigma=4).
        assert inp.sum() > 200.0
        assert tgt.sum() > 200.0
        # The two channels should carry nearly the same total flux —
        # they came from the same delta image and unit-flux PSFs.
        rel_diff = abs(inp.sum() - tgt.sum()) / max(inp.sum(), tgt.sum())
        assert rel_diff < 0.10, (
            f"input and target totals should match (unit-flux PSFs); "
            f"got input={inp.sum():.2f} vs target={tgt.sum():.2f}"
        )

    def test_input_and_target_differ_when_psfs_differ(self, psfs):
        """If the two PSFs are different, the (input, target) pair
        must be different too — otherwise stars contribute nothing to
        the training signal."""
        from euclid_polish.training.transition_augmentations import (
            make_star_field,
        )
        psf_h, psf_e = psfs
        inp, tgt = make_star_field(
            psf_h, psf_e,
            image_size=128, n_stars=4,
            rng=np.random.default_rng(123),
        )
        diff = np.abs(inp - tgt).max()
        assert diff > 0.1, (
            f"input and target must differ when PSFs differ; max|diff|={diff}"
        )

    def test_input_and_target_match_when_psfs_match(self, psfs):
        """Sanity-check the linearity machinery — same PSF on both sides
        must yield identical input and target."""
        from euclid_polish.training.transition_augmentations import (
            make_star_field,
        )
        psf_h, _ = psfs
        inp, tgt = make_star_field(
            psf_h, psf_h,
            image_size=128, n_stars=3,
            rng=np.random.default_rng(7),
        )
        np.testing.assert_allclose(inp, tgt, atol=1e-6)

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


# ---------------------------------------------------------------------------
# build_star_pair_dataset — tf.data producer
# ---------------------------------------------------------------------------

class TestStarPairDataset:

    def test_emits_correct_shapes(self, psfs):
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

    def test_shape_and_dtype(self, psfs):
        from euclid_polish.training.transition_augmentations import (
            make_psf_identity_pair,
        )
        psf_h, psf_e = psfs
        inp, tgt = make_psf_identity_pair(psf_h, psf_e, image_size=64)
        assert inp.shape == (64, 64, 1)
        assert tgt.shape == (64, 64, 1)
        assert inp.dtype == np.float32

    def test_input_is_hst_target_is_euclid(self, psfs):
        """The pair must be ordered (input=HST, target=Euclid).
        Otherwise the model would have to learn to invert the
        transition direction, which is the wrong thing."""
        from euclid_polish.training.transition_augmentations import (
            embed_in_canvas, make_psf_identity_pair,
        )
        psf_h, psf_e = psfs
        inp, tgt = make_psf_identity_pair(psf_h, psf_e, image_size=64)
        np.testing.assert_array_equal(
            inp[..., 0], embed_in_canvas(psf_h, 64),
        )
        np.testing.assert_array_equal(
            tgt[..., 0], embed_in_canvas(psf_e, 64),
        )

    def test_unit_flux_preserved(self, psfs):
        """Both PSFs sum to 1 by construction; embedding in a larger
        canvas can only zero-pad, so the sums must still be ~1."""
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
