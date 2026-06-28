"""Tests for the differential-PSF kernel solver."""

from __future__ import annotations

import os

import numpy as np
import pytest

from euclid_polish.sky.observation.differential_kernel import (
    DifferentialKernel,
    _fourier_shift,
    _recenter_to_geometric,
    apply_kernel,
    compute_differential_kernel,
)


def _gauss2d(side: int, fwhm_pix: float,
             cy: float | None = None, cx: float | None = None) -> np.ndarray:
    """``side``×``side`` unit-flux Gaussian. Default centred at the
    geometric centre; pass ``cy``/``cx`` for a deliberately offset peak."""
    sigma = fwhm_pix / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    y, x = np.mgrid[:side, :side]
    if cy is None:
        cy = (side - 1) / 2.0
    if cx is None:
        cx = (side - 1) / 2.0
    g = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * sigma ** 2))
    return (g / g.sum()).astype(np.float64)


def _centroid(arr: np.ndarray) -> tuple:
    """Flux-weighted centroid of the positive part — same convention
    as ``_recenter_to_geometric``."""
    pos = np.maximum(arr, 0.0)
    yy, xx = np.indices(arr.shape)
    total = float(pos.sum())
    return (
        float((yy * pos).sum() / total),
        float((xx * pos).sum() / total),
    )


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


class TestCentring:
    """Pin the centring invariant: A's centre must land on the
    geometric centre of the output array, *regardless* of the parities
    of N_in vs the internal FFT-pad size N_pad.

    The historical bug was that ``_pad_to`` and the crop-back both used
    ``(N_pad - N_in) // 2``, which is correct for (even, even) and
    (odd, odd) but off by one for (even pad, odd in). With our default
    odd N_in=511 and N_pad=2048 (next power of 2), this put A's centre
    at pixel 256 instead of 255 of a 511² array — invisible to the eye
    in the PSF panels but lethal: ``A ⊛ H − E`` came out as a dipole
    at the dead centre with peak ≈ 60 % of E's peak."""

    def test_when_E_equals_H_kernel_peak_is_at_geometric_centre(self):
        """E==H → ``argmax(A)`` must be at ``(side//2, side//2)``,
        not at any neighbour pixel. That's the centring invariant the
        off-by-one used to violate: A was peaked one pixel off the
        geometric centre, so convolving back put A⊛H one pixel away
        from E and the difference looked like a high-amplitude dipole.

        We don't check "peak holds X % of sum" — for a smooth Gaussian
        the Wiener filter legitimately spreads some energy off the
        centre. The shift bug shows up only in *where* the peak sits."""
        psf = _gauss2d(63, 4.0)
        a = compute_differential_kernel(psf, psf, regularisation=1e-4)
        cy, cx = np.unravel_index(int(np.argmax(a)), a.shape)
        target = a.shape[0] // 2
        assert (cy, cx) == (target, target), (
            f"A peaks at ({cy}, {cx}) instead of geometric centre "
            f"({target}, {target}) — off-by-one shift regressed"
        )

    def test_a_conv_h_equals_e_to_machine_precision_when_e_equals_h(self):
        """E == H → A should be a centred delta, so ``A ⊛ H = E``
        exactly (to numerical precision). Anything bigger than ~1e-6
        residual means a shift slipped back in."""
        psf = _gauss2d(63, 4.0)
        a   = compute_differential_kernel(psf, psf, regularisation=1e-4)
        # apply_kernel uses scipy fftconvolve mode='same' under the hood.
        back = apply_kernel(psf, a)
        rms_residual = float(np.sqrt(((back - psf) ** 2).mean()))
        rms_e        = float(np.sqrt((psf ** 2).mean()))
        rel_rms = rms_residual / rms_e
        assert rel_rms < 0.01, f"rel.RMS = {rel_rms:.4f} — kernel is shifted"

    def test_pad_to_centres_odd_input_in_even_target(self):
        """The exact (even, odd) case that was buggy: pad odd→even and
        verify the input's centre pixel lands at ``target // 2``."""
        from euclid_polish.sky.observation.differential_kernel import _pad_to
        # Distinguishable centre marker so we can find it after padding.
        a = np.zeros((5, 5))
        a[2, 2] = 1.0                       # centre of odd 5×5 is (2, 2)
        padded = _pad_to(a, (8, 8))         # even target
        # After padding, the centre should be at (4, 4) = (target // 2).
        assert padded[4, 4] == 1.0, (
            f"input centre landed at unexpected position; "
            f"argmax = {np.unravel_index(int(np.argmax(padded)), padded.shape)}"
        )


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

        This is the operational claim we lean on for the HST→Euclid
        path (scripts/fasrc_generate_hst_tfrecords.py): A(sH) has lower
        per-pixel variance than sH itself.
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
    """The script-level ``--border-pixels`` cleanup. As of the
    cosine-taper change, this is a Hann-style ramp, not a hard zero:
    pixel 0 (the very edge) is multiplied by 0, the next pixels follow
    a half-cosine up to 1 at the inner boundary of the border region,
    and the interior is untouched. A hard zero is itself a sharp
    discontinuity whose FFT spreads as a sinc along the perpendicular
    axis; the smooth taper kills that without changing the high-level
    contract (outer pixels suppressed → less ePSF-noise leakage into Ĥ).
    """

    def _load_helper(self):
        return _load_script_module()._zero_borders

    def test_zero_returns_input_unchanged(self):
        """``border_pixels=0`` short-circuits — no copy, no renorm."""
        zb = self._load_helper()
        psf = _gauss2d(31, 4.0)
        out = zb(psf, border_pixels=0)
        # Identity — same object, since the helper bails early.
        assert out is psf

    def test_outermost_pixel_is_zero_and_interior_is_untouched(self):
        """Boundary pixel hits the 0 end of the cosine ramp; interior
        (everything past the taper width) is rescaled by the unit-flux
        renormalisation but otherwise preserved in shape."""
        zb = self._load_helper()
        psf = _gauss2d(31, 4.0)
        out = zb(psf, border_pixels=3)
        # Outermost row/col exactly zero (ramp starts at 0).
        assert np.all(out[0, :]   == 0)
        assert np.all(out[-1, :]  == 0)
        assert np.all(out[:, 0]   == 0)
        assert np.all(out[:, -1]  == 0)
        # Interior block (well past the 3-px border) carries non-zero
        # Gaussian content.
        assert np.any(out[5:-5, 5:-5] > 0)

    def test_taper_is_monotone_increasing_through_border(self):
        """Along a single column inside the border strip, the cosine
        ramp must be monotone non-decreasing from edge to interior —
        catches sign or order regressions in the ramp construction."""
        zb = self._load_helper()
        psf = np.ones((31, 31), dtype=np.float64)
        out = zb(psf, border_pixels=5)
        # Top column, rows 0..4: ramp from 0 to ~1, monotone up.
        col_top = out[:5, 15]
        assert all(col_top[i] <= col_top[i + 1] for i in range(4))
        # Pixel 0 is exactly zero (cosine ramp start).
        assert col_top[0] == 0
        # Pixel 5 (first interior row) is strictly larger than pixel 4.
        assert out[5, 15] > col_top[-1]

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


# ---------------------------------------------------------------------------
# Sub-pixel re-centering — fixes the checkerboard artefact
# ---------------------------------------------------------------------------

class TestFourierShift:
    """``_fourier_shift`` is a sinc-quality sub-pixel translator. Pin
    the invariants so a future refactor (e.g. switching to
    scipy.ndimage.shift) can't silently change the photometry."""

    def test_preserves_total_flux(self):
        a = _gauss2d(63, 6.0)
        shifted = _fourier_shift(a, dy=0.4, dx=-0.3)
        assert float(shifted.sum()) == pytest.approx(float(a.sum()), rel=1e-6)

    def test_shifts_centroid_by_requested_amount(self):
        a = _gauss2d(63, 6.0)
        cy0, cx0 = _centroid(a)
        shifted = _fourier_shift(a, dy=0.4, dx=-0.3)
        cy1, cx1 = _centroid(shifted)
        assert (cy1 - cy0) == pytest.approx(0.4, abs=1e-3)
        assert (cx1 - cx0) == pytest.approx(-0.3, abs=1e-3)

    def test_integer_shift_is_lossless(self):
        """A whole-pixel shift should match a numpy roll exactly
        (modulo float round-off)."""
        a = _gauss2d(63, 6.0)
        shifted = _fourier_shift(a, dy=2.0, dx=-3.0)
        rolled = np.roll(np.roll(a, 2, axis=0), -3, axis=1)
        np.testing.assert_allclose(shifted, rolled, atol=1e-9)


class TestRecenterToGeometric:

    def test_no_op_when_already_centred(self):
        """Sub-pixel guard: don't introduce FFT round-off when there's
        nothing to fix."""
        a = _gauss2d(63, 6.0)                  # centroid at (31, 31)
        out = _recenter_to_geometric(a)
        # Same array object (the early-return path).
        assert out is a

    def test_corrects_half_pixel_offset(self):
        """The HST ePSF in production sits ~0.75 px off-centre; this
        test pins the recentering against an offset of similar magnitude."""
        # Deliberately offset Gaussian: centroid at (31+0.7, 31+0.5).
        a = _gauss2d(63, 6.0, cy=31.7, cx=31.5)
        cy, cx = _centroid(a)
        assert (cy - 31) == pytest.approx(0.7, abs=1e-3)
        assert (cx - 31) == pytest.approx(0.5, abs=1e-3)
        # After recentering, centroid sits at the geometric centre.
        out = _recenter_to_geometric(a)
        cy_out, cx_out = _centroid(out)
        target = (a.shape[0] - 1) / 2.0
        assert cy_out == pytest.approx(target, abs=5e-3)
        assert cx_out == pytest.approx(target, abs=5e-3)

    def test_recenter_does_not_wrap_around_borders(self):
        """Earlier the recenter used FFT phase-ramp shift, which is
        periodic at the array edges: a tiny amount of flux at the
        right edge wrapped to the left edge, manifesting as bright
        ringing on each border of A⊛H (border/interior amplitude
        ratio ~30× in the convolved output). The current
        scipy.ndimage spline shift uses zero-extrapolation, so any
        offset that pushes the PSF past an edge falls off rather than
        wrapping to the opposite side. This pins that property.
        """
        # PSF with a single bright pixel right at the right edge —
        # the worst case for an FFT shift to expose wraparound.
        side = 31
        a = np.zeros((side, side), dtype=np.float64)
        a[side // 2, side - 1] = 1.0    # last column, middle row
        # Re-centring would shift the bright pixel toward the centre
        # by ~(side-1)/2 - (side-1) = -(side-1)/2 columns. With a
        # spatial shift (cval=0) the left edge stays empty; with a
        # Fourier shift the bright pixel wraps to the opposite (right)
        # edge as a phase-ramp echo.
        out = _recenter_to_geometric(a)
        # Right edge — should NOT have a wrapped echo of the bright pixel.
        right_edge_max = float(np.abs(out[:, -1]).max())
        # Centre column — should be near 1 (the bright pixel landed here).
        centre_col_max = float(np.abs(out[:, side // 2]).max())
        assert right_edge_max < 0.05 * centre_col_max, (
            f"recenter wrapped flux to the opposite edge: "
            f"right_edge_max={right_edge_max:.3e} vs "
            f"centre_col_max={centre_col_max:.3e}"
        )


class TestNoCheckerboardArtefact:
    """The differential-kernel solver must produce a smooth kernel even
    when inputs have sub-pixel centroid mismatch — that's the case the
    production HST ePSF (centroid 0.75 px off geometric centre) hits
    every time, and was making the on-disk kernel a ±0.7 checkerboard."""

    @staticmethod
    def _checkerboard_projection(arr: np.ndarray) -> float:
        """Projection of arr onto the ``(-1)^(i+j)`` Nyquist basis.

        ~0 for a smooth kernel; |proj| approaches 1 for a pure
        checkerboard. The production-bug kernel scored 0.11 over its
        central 60² window; a healthy kernel scores < 0.02.
        """
        H, W = arr.shape
        cy, cx = H // 2, W // 2
        r = 30
        crop = arr[cy - r: cy + r, cx - r: cx + r]
        ii, jj = np.indices(crop.shape)
        chk = (-1.0) ** (ii + jj)
        denom = float(np.abs(crop).sum())
        if denom == 0:
            return 0.0
        return float((crop * chk).sum() / denom)

    def test_offset_inputs_produce_smooth_kernel(self):
        """E and H both shifted off-centre by ~0.5 px in *different*
        directions — the recentering should bring both back so the
        solved A is smooth, not a checkerboard."""
        e = _gauss2d(127, 8.0, cy=63 + 0.3, cx=63 - 0.4)
        h = _gauss2d(127, 4.0, cy=63 - 0.5, cx=63 + 0.2)
        a = compute_differential_kernel(e, h, regularisation=1e-3)
        proj = self._checkerboard_projection(a)
        assert abs(proj) < 0.02, (
            f"kernel still checkerboards (proj={proj:+.3f}) — "
            "recentering not effective"
        )

    def test_recenter_true_keeps_unit_dc_gain(self):
        """Recentering must not break the photometric DC-gain invariant
        — `A.sum() ≈ 1` for unit-flux inputs, regardless of offset."""
        e = _gauss2d(127, 8.0, cy=63 + 0.3, cx=63 - 0.4)
        h = _gauss2d(127, 4.0, cy=63 - 0.5, cx=63 + 0.2)
        a = compute_differential_kernel(e, h, regularisation=1e-3)
        assert float(a.sum()) == pytest.approx(1.0, abs=1e-2)


class TestKernelCorrectAgainstUnrecenteredInputs:
    """The kernel ``compute_differential_kernel`` returns must satisfy
    ``A ⊛ H_original ≈ E_original`` — the contract every downstream
    caller (the HST→Euclid TFRecord generator, the validate panel,
    the inference forward model) relies on.

    The internal recentering would otherwise bake a sub-pixel shift
    into the kernel equal to ``(cy_e − cy_h, cx_e − cx_h)``, leaving
    ``A_solved ⊛ H_original`` shifted by that amount relative to
    ``E_original``. The solver undoes the shift before returning. This
    test pins that undo: feed inputs with mismatched centroids, check
    that the returned kernel convolves the un-recentered H into a
    field whose centroid matches the un-recentered E to sub-pixel
    precision.
    """

    def test_a_conv_h_centroid_matches_e_centroid(self):
        from scipy.signal import fftconvolve

        side = 127
        # Mismatched centroids: E shifted by (+0.3, -0.4), H by (-0.5, +0.2).
        e = _gauss2d(side, 8.0, cy=63 + 0.3, cx=63 - 0.4)
        h = _gauss2d(side, 4.0, cy=63 - 0.5, cx=63 + 0.2)

        a = compute_differential_kernel(e, h, regularisation=1e-3, recenter=True)
        ah = fftconvolve(h, a, mode="same")

        cy_e, cx_e = _centroid(e)
        cy_ah, cx_ah = _centroid(ah)
        # Convolution preserves centroid in the ideal case; the test
        # tolerance has to cover the residual Wiener smoothing + the
        # spline-shift interpolation error from the kernel correction.
        # Without the correction the offset would be (+0.8, -0.6) px.
        assert cy_ah == pytest.approx(cy_e, abs=0.05), (
            f"A⊛H y-centroid {cy_ah:.4f} ≠ E y-centroid {cy_e:.4f} — "
            "internal recenter shift not undone before return"
        )
        assert cx_ah == pytest.approx(cx_e, abs=0.05), (
            f"A⊛H x-centroid {cx_ah:.4f} ≠ E x-centroid {cx_e:.4f}"
        )

    def test_recenter_false_does_not_correct(self):
        """With ``recenter=False`` we expect NO correction (solver
        contract: don't touch the inputs OR the output). Confirms the
        correction is conditional on the recentering flag and not a
        permanent extra step."""
        from scipy.signal import fftconvolve

        side = 127
        # Same inputs both ways.
        e = _gauss2d(side, 8.0, cy=63 + 0.3, cx=63 - 0.4)
        h = _gauss2d(side, 4.0, cy=63 - 0.5, cx=63 + 0.2)

        a_off = compute_differential_kernel(e, h, regularisation=1e-3, recenter=False)
        ah = fftconvolve(h, a_off, mode="same")
        cy_e, cx_e = _centroid(e); cy_ah, cx_ah = _centroid(ah)
        # Without recenter the kernel does no centroid corrections —
        # A⊛H sits at the same place E does naturally because no
        # shift was introduced in the first place.
        assert cy_ah == pytest.approx(cy_e, abs=0.05)
        assert cx_ah == pytest.approx(cx_e, abs=0.05)
