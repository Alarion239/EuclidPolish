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
