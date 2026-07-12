"""The :class:`PSF` class — central object for every PSF operation.

A ``PSF`` carries the kernel data and its physical metadata
(pixel scale, optional measured FWHM, optional oversampling), and
exposes every spatial operation the rest of the codebase needs:

  * normalisation (sum=1)
  * resampling to a target pixel scale (cubic spline + flux preserve)
  * centre-crop / zero-pad to an odd square
  * sub-pixel re-centring on the peak / flux centroid
  * convolution with an image
  * FWHM (1-D, radial, 2-D area-equivalent), flux centroid,
    peak-offset measurements
  * FITS save / FITS load

Every operation that changes the data returns a NEW ``PSF`` — the
class is treated as immutable so consumers can chain operations
``psf.resampled_to(0.05).centre_cropped_to(421).with_unit_sum()``
without worrying about which step mutates what.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, replace
from typing import ClassVar

import numpy as np
from astropy.io import fits
from scipy.ndimage import rotate as ndi_rotate
from scipy.ndimage import shift as ndi_shift
from scipy.ndimage import zoom
from scipy.signal import fftconvolve

from euclid_polish.config import Config
from euclid_polish.provenance.fits import read_stamp_cards, write_stamp_cards
from euclid_polish.provenance.persistable import StampCarrier
from euclid_polish.provenance.records import Format
from euclid_polish.psf import measurements as _meas

# Canonical HR pixel scale (arcsec / pixel). Everything in the project
# resamples PSFs to this scale before convolution. Exposed here so
# loaders / scripts share one constant instead of hard-coding 0.05.
DEFAULT_HR_PIXEL_SCALE: float = float(Config.DEFAULT_PIXEL_SCALE)


@dataclass
class PSF(StampCarrier):
    """A PSF kernel + its physical metadata, with operations.

    Parameters
    ----------
    data
        2-D float array, the PSF stamp. Convention: ``sum=1`` for a
        normalised kernel, but the class doesn't enforce it on
        construction so loaders can read raw FITS and choose when to
        normalise.
    pixel_scale
        Arcsec per pixel of the kernel's grid. Required so consumers
        know whether they need to resample before convolution.
    fwhm_arcsec
        Optional cached FWHM measurement (from the FITS header, or
        computed from :meth:`fwhm_pixels` and stored).
    oversampling
        Optional EPSFBuilder oversampling factor (integer ≥ 1).
        Purely informational — it tells you "this kernel was built
        at N× the native instrument pixel scale", which doesn't
        affect convolution behaviour but is useful in headers.

    Immutability
    ------------
    Every operation method returns a NEW ``PSF`` instance. The
    underlying ``data`` array of an existing ``PSF`` is never
    written to in place. This keeps the class safe to share across
    threads / multiple convolution sites.
    """

    data: np.ndarray
    pixel_scale: float
    fwhm_arcsec: float | None = None
    oversampling: int | None = None

    PROV_FORMAT: ClassVar[Format] = Format.FITS

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.data.shape)   # type: ignore[return-value]

    @property
    def total_flux(self) -> float:
        return float(np.asarray(self.data).sum())

    @property
    def peak(self) -> float:
        return float(np.asarray(self.data).max())

    # ------------------------------------------------------------------
    # Measurements (delegate to the standalone helpers so the same
    # code path runs for a PSF object and for a raw array)
    # ------------------------------------------------------------------

    def central_profile(self) -> np.ndarray:
        """Single-row slice through the brightest pixel."""
        cy = int(np.unravel_index(int(np.argmax(self.data)), self.data.shape)[0])
        return np.asarray(self.data[cy, :])

    def fwhm_pixels(self, method: str = "radial") -> float:
        """FWHM in pixels.

        ``method`` ∈
          * ``"row"``     — single row through the peak (legacy fallback).
          * ``"radial"``  — azimuthally-averaged profile (default; best
                            for spike-bearing PSFs).
          * ``"area_eq"`` — 2× equivalent radius of the > peak/2 area.
        """
        d = np.asarray(self.data)
        if method == "row":
            return _meas.estimate_fwhm_pixels_1d(self.central_profile())
        if method == "radial":
            return _meas.fwhm_pixels_radial(d)
        if method == "area_eq":
            return _meas.fwhm_pixels_area_equivalent(d)
        raise ValueError(
            f"unknown FWHM method {method!r}; "
            f"choose row | radial | area_eq"
        )

    def flux_centroid(self) -> tuple[float, float]:
        return _meas.flux_centroid(self.data)

    def peak_offset_from_centre(self) -> tuple[float, float]:
        return _meas.peak_offset_from_centre(self.data)

    def radial_profile(self) -> tuple[np.ndarray, np.ndarray]:
        return _meas.radial_profile(self.data)

    # ------------------------------------------------------------------
    # Operations — each returns a NEW PSF
    # ------------------------------------------------------------------

    def with_unit_sum(self) -> PSF:
        """Return a new PSF with ``data / data.sum()`` (defensive).

        No-op when the input is already sum=1 within float tolerance.
        Returns the same instance if the sum is ≤ 0 (degenerate;
        caller's problem to handle).
        """
        s = self.total_flux
        if s <= 0 or abs(s - 1.0) < 1e-7:
            return self
        return replace(
            self,
            data=(self.data / s).astype(self.data.dtype, copy=False),
        )

    def background_cleaned(
        self,
        *,
        floor_percentile: float = Config.PSF_FLOOR_PERCENTILE,
        taper_inner_frac: float = Config.PSF_TAPER_INNER_FRAC,
        taper_outer_frac: float = Config.PSF_TAPER_OUTER_FRAC,
    ) -> PSF:
        """Strip the empirical kernel's noise floor and square edges.

        An EPSFBuilder kernel carries residual stacking noise across the
        whole square stamp; convolved with a bright star it imprints a
        visible square far above the Euclid noise. Two cuts (each ≤ 0
        disables it):

        * **floor**: pixels at or below the stamp's ``floor_percentile``-th
          percentile are zeroed — the bulk of a large stamp is noise, and
          real PSF flux sits orders of magnitude above it;
        * **radial taper**: a cosine roll-off takes the kernel smoothly to
          zero between ``taper_inner_frac`` and ``taper_outer_frac`` of the
          half-side, so no square boundary survives.

        The result is re-normalised to sum=1 (flux conservation in the
        forward convolution). Analytic Gaussian kernels pass through
        essentially unchanged (most of their stamp underflows to exactly 0,
        so the percentile threshold is 0 and the guard skips the cut).
        """
        data = np.asarray(self.data, dtype=np.float64).copy()
        if floor_percentile > 0:
            thr = float(np.percentile(data, floor_percentile))
            if thr > 0:
                data[data <= thr] = 0.0
        if 0 < taper_inner_frac < taper_outer_frac:
            H, W = data.shape
            half = min(H, W) / 2.0
            r_in, r_out = taper_inner_frac * half, taper_outer_frac * half
            yy, xx = np.indices(data.shape, dtype=np.float64)
            rr = np.hypot(yy - (H - 1) / 2.0, xx - (W - 1) / 2.0)
            w = 0.5 * (1.0 + np.cos(np.pi * np.clip(
                (rr - r_in) / max(r_out - r_in, 1e-9), 0.0, 1.0)))
            data *= w
        s = data.sum()
        if s > 0:
            data /= s
        return replace(self, data=data.astype(self.data.dtype, copy=False))

    def resampled_to(self, target_pixel_scale: float) -> PSF:
        """Resample onto a new pixel grid via cubic-spline ``zoom``.

        Preserves the PSF's physical extent in arcsec — the new
        kernel has ``pixel_scale = target_pixel_scale`` and a
        proportionally rescaled shape. Renormalises to sum=1 to
        absorb any tiny interpolation drift.

        No-op when the source and target scales already match (within
        1e-6 arcsec).

        Cubic spline (``order=3``) was chosen over Lanczos-3 in
        ``psf_library`` because Lanczos has negative side lobes that
        ring on the PSF wings — those create small negative pixels
        that shrink under the sum=1 renormalisation. Cubic spline is
        monotone-preserving in practice for Gaussian-like PSFs.
        """
        target = float(target_pixel_scale)
        if abs(self.pixel_scale - target) < 1e-6:
            return self
        factor = float(self.pixel_scale) / target
        out = zoom(
            self.data.astype(np.float64), zoom=factor,
            order=3, mode="constant", grid_mode=False,
        )
        s = float(out.sum())
        if s > 0:
            out = out / s
        return replace(
            self,
            data=out.astype(np.float32),
            pixel_scale=target,
        )

    def centre_cropped_to(self, side: int, *, renormalise: bool = True) -> PSF:
        """Return a ``(side, side)`` PSF centred on the geometric
        centre, padding with zeros if the input is smaller.

        ``side`` must be odd so the centre pixel is unambiguous.
        ``renormalise=True`` (default) re-divides by the new sum
        so the cropped kernel still integrates to 1.
        """
        if side % 2 == 0:
            raise ValueError(f"side must be odd, got {side}")
        psf = np.asarray(self.data)
        H, W = psf.shape
        if side > H or side > W:
            pad_h = max(0, (side - H + 1) // 2)
            pad_w = max(0, (side - W + 1) // 2)
            psf = np.pad(psf, ((pad_h, pad_h), (pad_w, pad_w)),
                         mode="constant")
            H, W = psf.shape
        i0 = (H - side) // 2
        j0 = (W - side) // 2
        out = psf[i0 : i0 + side, j0 : j0 + side]
        if renormalise:
            s = float(out.sum())
            if s > 0:
                out = out / s
        return replace(self, data=out.astype(self.data.dtype, copy=False))

    def recentred(self, *, on: str = "peak", subpixel: bool = True) -> PSF:
        """Shift the kernel so its peak / flux centroid lands on the
        geometric centre of the stamp.

        ``on`` ∈ {"peak", "centroid"}:
          * ``peak``     — integer-pixel argmax to centre (fast, exact).
          * ``centroid`` — flux-weighted (sub-pixel).

        ``subpixel=True`` enables fractional-pixel ``scipy.ndimage.shift``
        with cubic spline (order=3). Set to False to round to the
        nearest integer pixel (cheaper, but leaves up to 0.5 px of
        residual offset).

        Critical regression site: the Euclid VIS ePSF FITS file has
        its peak at (510, 511) of a 1023² stamp — 1 pixel y-offset
        from the geometric centre (511, 511). Before this method
        existed every loader path silently propagated that offset
        into the forward operator (and thus the trained model's PSF).
        Call ``.recentred()`` on load and the offset is gone.
        """
        H, W = self.data.shape
        cy_geo = (H - 1) / 2.0
        cx_geo = (W - 1) / 2.0
        if on == "peak":
            pky, pkx = np.unravel_index(
                int(np.argmax(self.data)), self.data.shape,
            )
            dy = cy_geo - pky
            dx = cx_geo - pkx
        elif on == "centroid":
            cy, cx = _meas.flux_centroid(self.data)
            if not np.isfinite(cy) or not np.isfinite(cx):
                return self
            dy = cy_geo - cy
            dx = cx_geo - cx
        else:
            raise ValueError(
                f"unknown re-centring anchor {on!r}; choose peak | centroid"
            )
        if not subpixel:
            dy = round(dy)
            dx = round(dx)
        if dy == 0 and dx == 0:
            return self
        shifted = ndi_shift(
            self.data.astype(np.float64), shift=(dy, dx),
            order=3, mode="constant", cval=0.0,
        )
        s = float(shifted.sum())
        if s > 0:
            shifted = shifted / s
        return replace(
            self,
            data=shifted.astype(self.data.dtype, copy=False),
        )

    def rotated(
        self,
        angle_deg: float,
        *,
        order: int = 3,
        clip_negative: bool = True,
    ) -> PSF:
        """Rotate the kernel ``angle_deg`` degrees about its centre.

        Models the telescope ROLL: the diffraction spikes are fixed in the
        instrument frame, but each Euclid pointing has a different roll, so
        the PSF appears rotated by that angle in the North-up MER mosaic.
        Convolving a North-up scene with a roll-θ-rotated kernel reproduces
        a given roll, so per-scene random rotation lets the model see the
        full roll family without ever blending two orientations into an
        unphysical multi-spike PSF.

        Direction matches ``scipy.ndimage.rotate`` (verified: a whole
        quarter-turn equals ``np.rot90(k=round(angle/90))``).

        Whole multiples of 90° take the **exact** ``np.rot90`` path — no
        interpolation, no flux loss, no ringing. Any other angle resamples
        onto the rotated grid with a spline (``order``: 3 = cubic, 1 =
        bilinear). The kernel is odd-sided and (after ``.recentred()``)
        centred, so rotation is about the true centre pixel and the peak
        stays put.

        ``clip_negative`` (default True) zeroes the small negative pixels a
        cubic spline rings into the sharp spikes; the result is then
        renormalised to sum=1 (interpolation does not conserve flux
        exactly). ``mode="constant"`` fills the rotated-in corners with 0
        (sky); rotation preserves radius-from-centre, so all energy within
        radius ``N/2`` stays in frame at every angle and the corner-clipped
        wing flux (typically <~1% at our kernel sizes) is absorbed by the
        renormalisation.
        """
        angle = float(angle_deg) % 360.0
        if angle == 0.0:
            return self
        if angle % 90.0 == 0.0:
            # Exact quarter-turn: index remap, no interpolation.
            out = np.rot90(np.asarray(self.data), k=int(round(angle / 90.0)) % 4)
        else:
            out = ndi_rotate(
                self.data.astype(np.float64), angle,
                reshape=False, order=int(order),
                mode="constant", cval=0.0,
            )
            if clip_negative:
                out = np.clip(out, 0.0, None)
        out = np.ascontiguousarray(out, dtype=np.float64)
        s = float(out.sum())
        if s > 0:
            out = out / s
        return replace(
            self,
            data=out.astype(self.data.dtype, copy=False),
        )

    def convolved_with(self, image: np.ndarray) -> np.ndarray:
        """``scipy.signal.fftconvolve(image, self.data, mode='same')``.

        Returns a numpy array, NOT a ``PSF`` — the result is an image
        rendered through this PSF, not another PSF. The kernel is
        defensively normalised to sum=1 here (so a convolution can't
        silently rescale flux), matching the contract every
        downstream consumer (pair-gen, trainer augmentations,
        differential-kernel script) already relies on.
        """
        k = self.with_unit_sum().data
        return fftconvolve(
            np.asarray(image, dtype=np.float32),
            k,
            mode="same",
        ).astype(np.float32)

    # ------------------------------------------------------------------
    # I/O — FITS save / load
    # ------------------------------------------------------------------

    def save(
        self,
        output_dir: str,
        filename: str = Config.DEFAULT_PSF_FITS_FILENAME,
    ) -> str:
        """Write the PSF to a FITS file with headers describing the
        physical metadata. Path returned. Always writes
        ``data / data.sum()`` so the on-disk kernel is sum=1
        regardless of how the in-memory PSF was constructed —
        prevents the historical "Euclid file is sum=3" mismatch."""
        os.makedirs(output_dir, exist_ok=True)
        fits_path = os.path.join(output_dir, filename)
        unit = self.with_unit_sum()
        hdu = fits.PrimaryHDU(data=unit.data.astype(np.float32))
        hdu.header["PXSCALE"] = (
            float(unit.pixel_scale), "Pixel scale (arcsec/pixel)",
        )
        if unit.oversampling is not None:
            hdu.header["OVERSAMP"] = (
                int(unit.oversampling), "EPSFBuilder oversampling factor",
            )
        if unit.fwhm_arcsec is not None:
            hdu.header["FWHM"] = (
                float(unit.fwhm_arcsec), "Measured FWHM (arcsec)",
            )
        # FITS header values are strict 7-bit ASCII — no em-dashes, no
        # smart quotes. Stick to plain ASCII text.
        hdu.header["COMMENT"] = (
            "Written by euclid_polish.psf.PSF.save -- sum=1 normalised."
        )
        if self.stamp is not None:
            write_stamp_cards(hdu.header, self.stamp)
        fits.HDUList([hdu]).writeto(fits_path, overwrite=True)
        return fits_path

    @classmethod
    def from_fits(cls, fits_path: str, *, normalise: bool = True) -> PSF:
        """Load a PSF from a FITS file.

        Reads ``PXSCALE`` (default 0.0), ``OVERSAMP`` (optional),
        ``FWHM`` (optional). When the file uses ``PIXSCALE`` instead
        of ``PXSCALE`` (HST extractor convention), that key is read
        as a fallback.

        ``normalise=True`` (default) returns a sum=1 PSF — what every
        downstream convolution expects. Set ``False`` only if you
        deliberately want the raw pixel values (e.g. for diagnostic
        purposes).
        """
        with fits.open(fits_path) as hdul:
            header = hdul[0].header
            data = np.asarray(hdul[0].data, dtype=np.float32)
        stamp = read_stamp_cards(header)
        # Two historical FITS conventions — PXSCALE (our save) and
        # PIXSCALE (HST extractor + HLSP WCS). Accept either.
        pix = header.get("PXSCALE", None)
        if pix is None:
            pix = header.get("PIXSCALE", 0.0)
        pixel_scale = float(pix)
        if pixel_scale == 0.0:
            warnings.warn(
                f"PSF loaded from {fits_path!r} has pixel_scale=0.0 "
                "(PXSCALE/PIXSCALE missing from header); "
                "resampling and FWHM computations will be incorrect.",
                UserWarning,
                stacklevel=2,
            )
        oversampling = int(header["OVERSAMP"]) if "OVERSAMP" in header else None
        fwhm_arcsec  = float(header["FWHM"])   if "FWHM"    in header else None
        psf = cls(
            data=data,
            pixel_scale=pixel_scale,
            fwhm_arcsec=fwhm_arcsec,
            oversampling=oversampling,
        )
        if normalise:
            psf = psf.with_unit_sum()
        if stamp is not None:
            psf = psf.with_stamp(stamp)
        return psf
