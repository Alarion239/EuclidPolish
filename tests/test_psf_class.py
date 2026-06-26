"""Tests for :mod:`euclid_polish.psf` — the central PSF package.

Coverage:
  * The PSF class methods (normalise, resample, crop, recentre,
    convolve, save/load).
  * The standalone measurement helpers (FWHM, centroid).
  * The instrument loaders (sum=1, on HR grid, optionally recentred).
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from astropy.io import fits

from euclid_polish.psf import (
    PSF,
    DEFAULT_HR_PIXEL_SCALE,
    flux_centroid,
    fwhm_pixels_radial,
    fwhm_pixels_area_equivalent,
    estimate_fwhm_pixels_1d,
    peak_offset_from_centre,
    radial_profile,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _gauss(side: int, sigma_pix: float, *, centre_offset=(0.0, 0.0)) -> np.ndarray:
    """Unit-flux Gaussian, optionally with a fractional-pixel centroid
    offset so we can test recentring."""
    half = (side - 1) / 2.0
    yy, xx = np.indices((side, side), dtype=np.float64)
    cy = half + centre_offset[0]
    cx = half + centre_offset[1]
    r2 = (yy - cy) ** 2 + (xx - cx) ** 2
    g = np.exp(-r2 / (2 * sigma_pix ** 2))
    return (g / g.sum()).astype(np.float32)


@pytest.fixture
def gauss_psf():
    """Default test PSF: 31×31, σ=2 px, sum=1, peak at geom centre."""
    return PSF(
        data=_gauss(31, 2.0),
        pixel_scale=0.05,
        fwhm_arcsec=None,
        oversampling=1,
    )


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------

class TestMeasurements:

    def test_estimate_fwhm_pixels_1d_on_gaussian(self):
        side, sigma = 51, 3.0
        row = _gauss(side, sigma)[side // 2]
        # Gaussian FWHM = 2·sqrt(2·ln 2)·σ ≈ 2.3548·σ
        expected = 2 * np.sqrt(2 * np.log(2)) * sigma
        got = estimate_fwhm_pixels_1d(row)
        assert abs(got - expected) < 1.5    # integer-pixel-counting tolerance

    def test_fwhm_pixels_radial_subpixel(self):
        sigma = 3.0
        got = fwhm_pixels_radial(_gauss(51, sigma))
        expected = 2 * np.sqrt(2 * np.log(2)) * sigma
        # Sub-pixel interpolation, should be more accurate than 1-D.
        assert abs(got - expected) < 0.2

    def test_fwhm_pixels_area_equivalent_on_gaussian(self):
        """Area-equiv FWHM should also recover the Gaussian σ within
        sampling tolerance."""
        sigma = 5.0
        got = fwhm_pixels_area_equivalent(_gauss(61, sigma))
        expected = 2 * np.sqrt(2 * np.log(2)) * sigma
        assert abs(got - expected) < 1.5

    def test_flux_centroid_on_offset_gaussian(self):
        # Offset the Gaussian by 0.5 pixel in y, 0.3 in x.
        side, sigma = 31, 2.0
        offset = (0.5, 0.3)
        d = _gauss(side, sigma, centre_offset=offset)
        cy, cx = flux_centroid(d)
        geom_y = geom_x = (side - 1) / 2.0
        assert abs(cy - (geom_y + offset[0])) < 0.05
        assert abs(cx - (geom_x + offset[1])) < 0.05

    def test_flux_centroid_handles_negative(self):
        """Sky-subtracted PSF can have tiny negative residuals. The
        centroid mustn't be biased toward them — and definitely
        mustn't NaN."""
        d = _gauss(31, 2.0)
        d -= 1e-3   # uniform negative offset
        cy, cx = flux_centroid(d)
        assert np.isfinite(cy) and np.isfinite(cx)
        # Centre still recoverable to high accuracy.
        assert abs(cy - 15.0) < 0.05
        assert abs(cx - 15.0) < 0.05

    def test_peak_offset_zero_for_centred_gaussian(self):
        d = _gauss(31, 2.0)
        dy, dx = peak_offset_from_centre(d)
        assert dy == 0.0
        assert dx == 0.0

    def test_peak_offset_detects_known_shift(self):
        # Shift the peak by exactly 2 pixels in y, -3 in x.
        d = _gauss(31, 1.5)
        shifted = np.roll(d, shift=(2, -3), axis=(0, 1))
        dy, dx = peak_offset_from_centre(shifted)
        assert dy == 2.0
        assert dx == -3.0

    def test_radial_profile_monotone_decreasing(self):
        d = _gauss(51, 5.0)
        radii, prof = radial_profile(d)
        # Should be monotone-decreasing (or non-increasing) from the
        # peak outward for a non-pathological Gaussian.
        diffs = np.diff(prof[:25])
        assert (diffs <= 1e-12).all(), f"profile not non-increasing: {diffs}"


# ---------------------------------------------------------------------------
# PSF class — operations
# ---------------------------------------------------------------------------

class TestPSFOperations:

    def test_with_unit_sum_idempotent_on_normalised(self, gauss_psf):
        out = gauss_psf.with_unit_sum()
        assert out is gauss_psf   # no-op returns same instance

    def test_with_unit_sum_rescales(self):
        psf = PSF(data=3.0 * _gauss(15, 1.0), pixel_scale=0.05)
        assert psf.total_flux == pytest.approx(3.0, abs=1e-5)
        out = psf.with_unit_sum()
        assert out.total_flux == pytest.approx(1.0, abs=1e-5)
        # Shape + dtype unchanged.
        assert out.data.shape == psf.data.shape
        assert out.data.dtype == psf.data.dtype

    def test_with_unit_sum_safe_on_zero_sum(self):
        psf = PSF(data=np.zeros((9, 9), dtype=np.float32), pixel_scale=0.05)
        out = psf.with_unit_sum()
        assert out is psf
        assert not np.isnan(out.data).any()

    def test_resampled_to_changes_grid_preserves_flux(self):
        # 31² @ 0.025"/pix → resample to 0.05"/pix. Should ~halve the
        # side; sum stays 1 (renormalised); flux distribution
        # roughly preserved.
        psf = PSF(data=_gauss(31, 3.0), pixel_scale=0.025)
        out = psf.resampled_to(0.05)
        assert out.pixel_scale == pytest.approx(0.05)
        assert out.shape[0] < psf.shape[0]    # downsampled
        assert out.total_flux == pytest.approx(1.0, abs=1e-5)
        # FWHM in arcsec should be preserved (within a few %).
        old_fwhm_as = psf.fwhm_pixels("radial") * psf.pixel_scale
        new_fwhm_as = out.fwhm_pixels("radial") * out.pixel_scale
        assert abs(new_fwhm_as - old_fwhm_as) / old_fwhm_as < 0.10

    def test_resampled_to_noop_when_scales_match(self, gauss_psf):
        out = gauss_psf.resampled_to(gauss_psf.pixel_scale)
        assert out is gauss_psf

    def test_centre_cropped_to_smaller(self):
        psf = PSF(data=_gauss(51, 3.0), pixel_scale=0.05)
        out = psf.centre_cropped_to(21)
        assert out.shape == (21, 21)
        # Peak stays at the centre of the new stamp.
        cy, cx = np.unravel_index(int(np.argmax(out.data)), out.shape)
        assert (cy, cx) == (10, 10)
        # Renormalised — sum=1.
        assert out.total_flux == pytest.approx(1.0, abs=1e-5)

    def test_centre_cropped_to_pads_when_input_smaller(self):
        psf = PSF(data=_gauss(11, 1.0), pixel_scale=0.05)
        out = psf.centre_cropped_to(21)
        assert out.shape == (21, 21)
        cy, cx = np.unravel_index(int(np.argmax(out.data)), out.shape)
        assert (cy, cx) == (10, 10)

    def test_centre_cropped_to_rejects_even_side(self, gauss_psf):
        with pytest.raises(ValueError, match="odd"):
            gauss_psf.centre_cropped_to(20)

    def test_recentred_peak_fixes_integer_offset(self):
        """REGRESSION — the Euclid VIS ePSF has its peak at (510, 511)
        on a 1023² stamp = 1-pixel y-offset. ``recentred()`` must
        move it back to the geometric centre."""
        d = np.roll(_gauss(31, 2.0), shift=(1, -2), axis=(0, 1))
        psf = PSF(data=d, pixel_scale=0.05)
        # Confirm the offset is there before recentring.
        dy_before, dx_before = psf.peak_offset_from_centre()
        assert (dy_before, dx_before) == (1.0, -2.0)
        out = psf.recentred(on="peak", subpixel=False)
        dy_after, dx_after = out.peak_offset_from_centre()
        assert (dy_after, dx_after) == (0.0, 0.0)
        # Sum preserved.
        assert out.total_flux == pytest.approx(1.0, abs=1e-5)

    def test_recentred_subpixel_centroid(self):
        """Sub-pixel centroid recentring (used by the Euclid loader)
        moves a fractional-offset PSF to within < 0.1 px of centre."""
        d = _gauss(41, 2.5, centre_offset=(0.4, -0.3))
        psf = PSF(data=d, pixel_scale=0.05)
        out = psf.recentred(on="centroid", subpixel=True)
        cy, cx = out.flux_centroid()
        H = out.shape[0]
        geom = (H - 1) / 2.0
        assert abs(cy - geom) < 0.1
        assert abs(cx - geom) < 0.1

    def test_convolved_with_preserves_flux(self):
        psf = PSF(data=_gauss(21, 2.0), pixel_scale=0.05)
        # Delta source → convolution returns the kernel.
        scene = np.zeros((64, 64), dtype=np.float32)
        scene[32, 32] = 100.0
        out = psf.convolved_with(scene)
        # Flux preserved (sum=1 kernel * 100 e- source = 100 e- output).
        assert out.sum() == pytest.approx(100.0, rel=1e-3)

    def test_methods_return_new_instances(self, gauss_psf):
        """Operations must be non-mutating — same PSF instance can be
        reused across multiple call sites without surprise side
        effects."""
        original_data = gauss_psf.data.copy()
        _ = gauss_psf.with_unit_sum()
        _ = gauss_psf.centre_cropped_to(15)
        _ = gauss_psf.recentred()
        np.testing.assert_array_equal(gauss_psf.data, original_data)


# ---------------------------------------------------------------------------
# PSF class — I/O
# ---------------------------------------------------------------------------

class TestPSFIO:

    def test_save_then_from_fits_roundtrips(self, tmp_path, gauss_psf):
        out_path = gauss_psf.save(str(tmp_path), filename="psf.fits")
        assert os.path.isfile(out_path)
        loaded = PSF.from_fits(out_path)
        assert loaded.pixel_scale == pytest.approx(gauss_psf.pixel_scale)
        assert loaded.shape == gauss_psf.shape
        np.testing.assert_allclose(loaded.data, gauss_psf.data, atol=1e-6)

    def test_save_writes_sum_one_even_from_unnormalised_input(self, tmp_path):
        """REGRESSION — the Euclid VIS PSF on disk used to ship with
        sum=3 because its save path didn't enforce normalisation.
        :meth:`PSF.save` must always write a sum=1 kernel regardless
        of the in-memory PSF's normalisation state."""
        psf = PSF(data=3.0 * _gauss(11, 1.0), pixel_scale=0.05)
        assert psf.total_flux == pytest.approx(3.0)
        out_path = psf.save(str(tmp_path), filename="psf_unnorm.fits")
        with fits.open(out_path) as h:
            assert h[0].data.sum() == pytest.approx(1.0, abs=1e-5)

    def test_from_fits_reads_pixscale_or_pixscale_fallback(self, tmp_path):
        """The HST extractor writes ``PIXSCALE``; ours writes
        ``PXSCALE``. The loader must accept either."""
        # Write a fake HST-style file with PIXSCALE only.
        path = tmp_path / "hst.fits"
        hdu = fits.PrimaryHDU(_gauss(11, 1.0))
        hdu.header["PIXSCALE"] = 0.0500004
        hdu.writeto(str(path), overwrite=True)
        psf = PSF.from_fits(str(path))
        assert psf.pixel_scale == pytest.approx(0.0500004, rel=1e-6)


# ---------------------------------------------------------------------------
# Rotation (telescope-roll modelling)
# ---------------------------------------------------------------------------

class TestRotation:

    def _spiked(self, side: int = 21) -> PSF:
        """Centred Gaussian + a bright horizontal 'spike' row (asymmetric)."""
        a = _gauss(side, 2.0).astype(np.float64)
        a[side // 2, :] += 0.02              # horizontal ray through centre
        return PSF(data=(a / a.sum()).astype(np.float32), pixel_scale=0.05)

    def test_zero_is_identity(self):
        p = self._spiked()
        np.testing.assert_array_equal(p.rotated(0).data, p.data)

    def test_360_is_identity(self):
        p = self._spiked()
        np.testing.assert_allclose(p.rotated(360).data, p.data, rtol=1e-6)

    def test_90_is_exact_rot90_direction(self):
        """The 90° fast path is exact np.rot90(k=1) — direction matches
        scipy.ndimage.rotate (verified at implementation time)."""
        p = self._spiked()
        expected = np.rot90(p.data, k=1)
        expected = expected / expected.sum()
        np.testing.assert_allclose(p.rotated(90).data, expected, rtol=1e-6)

    def test_90_turns_horizontal_spike_vertical(self):
        c = 21 // 2
        rot = self._spiked(21).rotated(90)
        # The bright row becomes a bright column.
        assert rot.data[:, c].sum() > rot.data[c, :].sum()

    def test_sum_preserved_and_nonneg_for_arbitrary_angle(self):
        rot = self._spiked().rotated(37.0)
        assert rot.data.sum() == pytest.approx(1.0, abs=1e-6)
        assert float(rot.data.min()) >= 0.0          # clip_negative default

    def test_peak_stays_centred(self):
        c = 21 // 2
        rot = self._spiked(21).rotated(45.0)
        pky, pkx = np.unravel_index(int(np.argmax(rot.data)), rot.data.shape)
        assert (pky, pkx) == (c, c)

    def test_roundtrip_approx_identity_for_smooth_psf(self):
        """A smooth PSF rotated +θ then −θ returns almost exactly (no
        systematic shift; interpolation is gentle on smooth content)."""
        p = PSF(data=_gauss(31, 3.0), pixel_scale=0.05)
        back = p.rotated(30.0).rotated(-30.0)
        np.testing.assert_allclose(back.data, p.data, atol=1e-4)

    def test_sharp_feature_degrades_but_bounded(self):
        """A sharp spike loses contrast through arbitrary-angle interpolation
        (the cost of a non-90° rotation) — bounded, and avoided entirely on
        the exact quarter-turn path."""
        p = self._spiked()
        back = p.rotated(30.0).rotated(-30.0)
        peak = float(p.data.max())
        assert float(np.abs(back.data - p.data).max()) < 0.4 * peak  # smeared, not destroyed
        # Exact 90° round-trip is loss-free (no interpolation).
        exact = p.rotated(90.0).rotated(-90.0)
        np.testing.assert_allclose(exact.data, p.data, rtol=1e-6)

    def test_negative_clip_can_be_disabled(self):
        """Cubic spline rings slightly negative on the sharp ray when the
        clip is off — confirms the default clip is doing something."""
        rot = self._spiked().rotated(37.0, clip_negative=False, order=3)
        assert float(rot.data.min()) < 0.0

    def test_pixel_scale_and_oversampling_preserved(self):
        p = PSF(data=_gauss(21, 2.0), pixel_scale=0.025, oversampling=4)
        rot = p.rotated(23.0)
        assert rot.pixel_scale == 0.025
        assert rot.oversampling == 4


# ---------------------------------------------------------------------------
# Background cleaning (noise floor + radial taper)
# ---------------------------------------------------------------------------

def _noisy_gaussian_kernel(n=257, sigma=3.0, pedestal=1e-6):
    yy, xx = np.indices((n, n), dtype=np.float64)
    c = (n - 1) / 2.0
    g = np.exp(-((yy - c) ** 2 + (xx - c) ** 2) / (2 * sigma ** 2))
    return (g / g.sum() + pedestal)


def test_background_cleaned_strips_floor_and_corners():
    psf = PSF(data=_noisy_gaussian_kernel(),
              pixel_scale=0.05).with_unit_sum()
    out = psf.background_cleaned()
    # Square boundary gone: corners exactly zero, kernel still unit-sum,
    # centre untouched.
    assert out.data[:20, :20].max() == 0.0
    assert out.data[-20:, -20:].max() == 0.0
    assert out.total_flux == pytest.approx(1.0, abs=1e-9)
    c = (out.shape[0] - 1) // 2
    assert np.unravel_index(np.argmax(out.data), out.shape) == (c, c)
    # The uniform pedestal dominates the stamp, so it IS the 90th
    # percentile -> wiped outside the core even inside the taper radius.
    assert out.data[c, c + 60] == 0.0


def test_background_cleaned_noop_on_clean_gaussian_core():
    # An analytic kernel (median 0 outside the core) keeps its core shape.
    yy, xx = np.indices((257, 257), dtype=np.float64)
    c = 128.0
    g = np.exp(-((yy - c) ** 2 + (xx - c) ** 2) / (2 * 3.0 ** 2))
    psf = PSF(data=g / g.sum(), pixel_scale=0.05)
    out = psf.background_cleaned()
    np.testing.assert_allclose(out.data[118:139, 118:139],
                               psf.data[118:139, 118:139], rtol=1e-6)


def test_background_cleaned_disable_flags():
    psf = PSF(data=_noisy_gaussian_kernel(),
              pixel_scale=0.05).with_unit_sum()
    out = psf.background_cleaned(floor_percentile=0.0,
                                 taper_inner_frac=0.0,
                                 taper_outer_frac=0.0)
    np.testing.assert_allclose(out.data, psf.data, rtol=1e-12)
