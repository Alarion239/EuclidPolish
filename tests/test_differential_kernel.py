"""Tests for the differential-PSF kernel solver."""

from __future__ import annotations

import os

import numpy as np
import pytest

from euclid_polish.sky.differential_kernel import (
    DifferentialKernel,
    apply_kernel,
    compute_differential_kernel,
)


def _gauss2d(side: int, fwhm_pix: float) -> np.ndarray:
    sigma = fwhm_pix / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    y, x = np.mgrid[:side, :side]
    cy = cx = (side - 1) / 2.0
    g = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * sigma ** 2))
    return (g / g.sum()).astype(np.float64)


class TestSolverShapeAndNorm:

    def test_rejects_mismatched_shapes(self):
        a = np.zeros((10, 10))
        b = np.zeros((12, 12))
        with pytest.raises(ValueError, match="shapes must match"):
            compute_differential_kernel(a, b)

    def test_rejects_non_2d(self):
        a = np.zeros((10,))
        b = np.zeros((10,))
        with pytest.raises(ValueError, match="2-D"):
            compute_differential_kernel(a, b)

    def test_output_shape_matches_input(self):
        e = _gauss2d(31, 5.0)
        h = _gauss2d(31, 3.0)
        a = compute_differential_kernel(e, h)
        assert a.shape == e.shape

    def test_dc_gain_is_unity(self):
        """A unit-flux convolution chain should preserve total flux."""
        e = _gauss2d(63, 6.0)
        h = _gauss2d(63, 3.0)
        a = compute_differential_kernel(e, h, regularisation=1e-3)
        assert a.sum() == pytest.approx(1.0, abs=1e-2)


class TestRoundTrip:

    def _radial_profile(self, img: np.ndarray) -> np.ndarray:
        """Azimuthally averaged radial profile from the image centre."""
        H, W = img.shape
        cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
        y, x = np.mgrid[:H, :W]
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        r_int = r.astype(int)
        max_r = min(H, W) // 2
        out = np.zeros(max_r, dtype=np.float64)
        for k in range(max_r):
            mask = r_int == k
            if mask.any():
                out[k] = float(img[mask].mean())
        return out

    def _fwhm_from_profile(self, profile: np.ndarray) -> float:
        """Half-maximum radius from a centred radial profile (pixels)."""
        peak = profile[0]
        if peak <= 0:
            return float("nan")
        half = peak / 2.0
        for r in range(1, len(profile)):
            if profile[r] <= half:
                # Linear interp between r-1 and r.
                t = (profile[r - 1] - half) / (profile[r - 1] - profile[r] + 1e-30)
                return 2.0 * ((r - 1) + t)
        return float("nan")

    def test_a_conv_h_recovers_e_fwhm(self):
        """A ⊛ H should recover E's FWHM to within ~10 %.

        Wiener regularisation is biased slightly low-pass (overestimates
        FWHM by a few %) and the kernel's outer wings are truncated by
        the input-shape crop; 10 % tolerance comfortably covers both
        effects without masking a broken solver.
        """
        side = 127
        e = _gauss2d(side, 8.0)        # Euclid FWHM 8 pix
        h = _gauss2d(side, 4.0)        # Hubble FWHM 4 pix (sharper)
        a = compute_differential_kernel(e, h, regularisation=1e-4)
        recovered = apply_kernel(h, a)
        # Radial profiles avoid per-pixel boundary noise from cropping A's
        # wings; compare azimuthally averaged FWHM instead.
        fwhm_e   = self._fwhm_from_profile(self._radial_profile(e))
        fwhm_rec = self._fwhm_from_profile(self._radial_profile(recovered))
        assert fwhm_rec == pytest.approx(fwhm_e, rel=0.10)

    def test_a_conv_h_recovers_e_total_flux(self):
        """Flux conservation: integrated A ⊛ H ≈ integrated E."""
        side = 127
        e = _gauss2d(side, 8.0)
        h = _gauss2d(side, 4.0)
        a = compute_differential_kernel(e, h, regularisation=1e-4)
        recovered = apply_kernel(h, a)
        assert float(recovered.sum()) == pytest.approx(float(e.sum()), rel=2e-2)

    def test_kernel_is_low_pass_when_e_broader_than_h(self):
        """A should suppress high frequencies (Euclid is broader)."""
        e = _gauss2d(63, 6.0)
        h = _gauss2d(63, 3.0)
        a = compute_differential_kernel(e, h)
        # In Fourier space, |A_hat| should monotonically (approximately)
        # decrease with |k|. Check: average |A_hat| at high-k < at low-k.
        a_hat = np.abs(np.fft.fftshift(np.fft.fft2(a)))
        c = a.shape[0] // 2
        low_k_avg  = float(a_hat[c-2:c+3, c-2:c+3].mean())
        high_k_avg = float(a_hat[:3, :3].mean())
        assert high_k_avg < low_k_avg


class TestNoisePropagation:

    def test_noise_variance_reduces(self):
        """Applying A (a low-pass) to white noise should reduce variance.

        This is the operational claim we lean on for the HST-template
        path: A(sH) has lower per-pixel variance than sH itself.
        """
        rng = np.random.default_rng(0)
        e = _gauss2d(63, 6.0)
        h = _gauss2d(63, 3.0)
        a = compute_differential_kernel(e, h)

        noise = rng.normal(0, 1.0, size=(256, 256)).astype(np.float32)
        filtered = apply_kernel(noise, a)
        # Variance must drop by at least 2x for these PSF widths.
        assert float(filtered.var()) < 0.5 * float(noise.var())


def _load_script_module():
    """Import the differential-kernel script as a module so we can
    unit-test its helpers without going through the CLI."""
    import importlib.util, os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, "scripts",
                        "fasrc_compute_differential_kernel.py")
    spec = importlib.util.spec_from_file_location("_fdk", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestBgSubtract:
    """The script-level ``--bg-subtract`` cleanup that runs right
    before the FFT. Lives in ``scripts/fasrc_compute_differential_kernel``
    so we import it directly here — keeps the cleanup logic next to its
    only caller while still being unit-testable."""

    def _load_helper(self):
        return _load_script_module()._bg_subtract_and_clip

    def test_subtracts_bg_offset_from_psf_pixels(self):
        """Plain offset on a clean Gaussian: after cleanup, the PSF
        wings should sit at zero, not at the original bg level."""
        bg_subtract = self._load_helper()
        psf = _gauss2d(31, 4.0) + 0.01      # additive 0.01 background
        cleaned = bg_subtract(psf)
        # No pixel should be negative (positivity floor).
        assert cleaned.min() >= 0.0
        # The wings (far from centre) should be at zero now — the bg
        # offset has been removed, not just clipped under.
        corner = cleaned[:5, :5]
        assert corner.max() < 1e-6
        # And the kernel is still unit flux.
        assert cleaned.sum() == pytest.approx(1.0, abs=1e-6)

    def test_kills_negative_half_of_bg_noise(self):
        """Noisy bg: negative excursions go to 0; positive stay."""
        rng = np.random.default_rng(42)
        bg_subtract = self._load_helper()
        psf = _gauss2d(63, 5.0)
        # Inject white-noise background at 1 % of peak.
        psf = psf + rng.normal(0, 0.01 * psf.max(), size=psf.shape)
        cleaned = bg_subtract(psf)
        assert cleaned.min() >= 0.0
        # At least 30 % of pixels should be exactly zero (the previously-
        # negative half of bg noise + the lowest of the positive half
        # that fell below the median).
        zero_frac = float((cleaned == 0).mean())
        assert zero_frac > 0.30

    def test_renormalises_to_unit_flux(self):
        """DC gain of the kernel relies on both PSFs summing to 1."""
        bg_subtract = self._load_helper()
        psf = _gauss2d(31, 4.0) + 0.02     # bg high enough to clip a lot
        cleaned = bg_subtract(psf)
        assert cleaned.sum() == pytest.approx(1.0, abs=1e-6)

    def test_preserves_peak_position(self):
        """Cleanup must not move the PSF centroid — the FFT phase
        reference is the centre pixel, a shift would silently introduce
        a linear phase term in Â."""
        bg_subtract = self._load_helper()
        psf = _gauss2d(31, 4.0) + 0.005
        cleaned = bg_subtract(psf)
        # argmax in flat indexing; both should be the centre pixel.
        assert np.argmax(cleaned) == np.argmax(psf)


class TestZeroBorders:
    """The script-level ``--border-pixels`` cleanup that zeros the
    outer N rows/cols of each PSF before the FFT. Same loader pattern
    as TestBgSubtract — the helper lives in the script."""

    def _load_helper(self):
        return _load_script_module()._zero_borders

    def test_zero_returns_input_unchanged(self):
        """``border_pixels=0`` short-circuits — no copy, no renorm."""
        zb = self._load_helper()
        psf = _gauss2d(31, 4.0)
        out = zb(psf, border_pixels=0)
        # Identity — same object, since the helper bails early.
        assert out is psf

    def test_zeros_correct_pixels_and_only_those(self):
        zb = self._load_helper()
        psf = _gauss2d(31, 4.0)
        out = zb(psf, border_pixels=3)
        # All four borders are zero.
        assert np.all(out[:3, :]  == 0)
        assert np.all(out[-3:, :] == 0)
        assert np.all(out[:, :3]  == 0)
        assert np.all(out[:, -3:] == 0)
        # Interior is non-zero (Gaussian core sits in the middle).
        assert np.any(out[3:-3, 3:-3] > 0)

    def test_renormalises_to_unit_flux(self):
        """Renormalisation is essential — without it the kernel's DC
        gain drifts whenever we clip any non-trivial border flux."""
        zb = self._load_helper()
        psf = _gauss2d(31, 4.0)
        out = zb(psf, border_pixels=5)
        assert out.sum() == pytest.approx(1.0, abs=1e-6)

    def test_raises_when_border_consumes_whole_stamp(self):
        """Guard against silent zero-array output if the caller passes
        an absurd width — easier to spot at config time than after."""
        zb = self._load_helper()
        psf = _gauss2d(21, 4.0)
        with pytest.raises(ValueError, match="too large"):
            zb(psf, border_pixels=11)        # 2*11 == 22 >= 21

    def test_preserves_centroid(self):
        """Symmetric mask → centroid unchanged. Important because the
        FFT phase reference is the centre pixel."""
        zb = self._load_helper()
        psf = _gauss2d(31, 4.0)
        out = zb(psf, border_pixels=5)
        assert np.argmax(out) == np.argmax(psf)


class TestCommonSideCrop:
    """``--common-side`` is the headline fix for the noisy-kernel bug
    where one PSF (1023²) got zero-padded around a much smaller one
    (~154²). The previous max()-based grid choice put a sharp
    content→zero boundary into the FFT, whose sinc-like ringing
    dominated Â. Now both PSFs get cropped/padded to a sensible
    common grid before the FFT — verify the helper that does the
    cropping actually preserves flux when the input fits and trims
    only the outer regions when it doesn't."""

    def test_centre_crop_preserves_flux_when_target_is_larger(self):
        """Pad case: smaller-than-target PSF gets zero-padded — every
        bit of flux survives."""
        mod = _load_script_module()
        psf = _gauss2d(51, 5.0)
        out = mod._centre_crop_to(psf, 101)
        assert out.shape == (101, 101)
        assert out.sum() == pytest.approx(psf.sum(), abs=1e-9)
        # The Gaussian core sits at the centre of the padded array.
        assert np.argmax(out) == np.argmax(_gauss2d(101, 5.0))

    def test_centre_crop_trims_outer_when_target_is_smaller(self):
        """Crop case: larger-than-target PSF is sliced down — for a
        Gaussian narrow enough to fit, almost all flux survives."""
        mod = _load_script_module()
        psf = _gauss2d(101, 5.0)         # σ ≈ 2.1 px — well inside 21²
        out = mod._centre_crop_to(psf, 21)
        assert out.shape == (21, 21)
        # >99 % of a Gaussian with σ=2.1 fits in a centred 21×21 window.
        assert out.sum() / psf.sum() > 0.99

    def test_centre_crop_centroid_preserved(self):
        """The FFT phase reference is the centre pixel — a crop that
        shifts the peak by even one pixel would silently introduce a
        linear phase term in Ĥ. Test both pad-larger and trim-smaller."""
        mod = _load_script_module()
        for src_side, tgt_side in [(31, 71), (71, 31)]:
            psf = _gauss2d(src_side, 4.0)
            out = mod._centre_crop_to(psf, tgt_side)
            # argmax in linear-indexed form; for a centred Gaussian on
            # an odd-sided grid, the centre pixel is (side//2, side//2).
            ay, ax = np.unravel_index(int(np.argmax(out)), out.shape)
            assert (ay, ax) == (tgt_side // 2, tgt_side // 2), (
                f"crop {src_side}→{tgt_side} moved the peak to "
                f"({ay},{ax}) instead of centre"
            )


class TestSaveLoadProvenance:

    def test_round_trip_through_fits(self, tmp_path):
        e = _gauss2d(31, 5.0)
        h = _gauss2d(31, 3.0)
        a = compute_differential_kernel(e, h)
        dk = DifferentialKernel(
            data=a, pixel_scale_arcsec=0.05,
            euclid_band="VIS", hst_filter="F814W",
            regularisation=1e-3,
        )
        path = str(tmp_path / "diff_kernel.fits")
        dk.save(path)
        assert os.path.isfile(path)
        loaded = DifferentialKernel.from_fits(path)
        assert loaded.euclid_band == "VIS"
        assert loaded.hst_filter == "F814W"
        assert loaded.pixel_scale_arcsec == pytest.approx(0.05)
        assert loaded.regularisation == pytest.approx(1e-3)
        np.testing.assert_allclose(loaded.data, a, rtol=1e-5)

    def test_dc_gain_property(self):
        a = _gauss2d(31, 5.0)
        dk = DifferentialKernel(
            data=a, pixel_scale_arcsec=0.05,
            euclid_band="VIS", hst_filter="F814W",
            regularisation=1e-3,
        )
        assert dk.dc_gain == pytest.approx(1.0, abs=1e-3)
