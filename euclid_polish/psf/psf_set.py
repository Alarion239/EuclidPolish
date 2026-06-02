"""The :class:`PSFSet` — an ordered ensemble of PSFs for one band.

The Euclid PSF varies across the focal plane. Rather than collapse a
band's star cutouts into one average ePSF, we cluster the stars
spatially and build **one ePSF per cluster**, giving K kernels that
sample the field's PSF variation. :class:`PSFSet` holds those K
:class:`PSF` kernels (all on the same grid, same shape, each sum=1)
and exposes:

  * :meth:`mean`   — the field-averaged PSF (a single :class:`PSF`).
  * :meth:`sample` — a random *convex* (Dirichlet) blend of the K
    kernels, used at generation time so each synthetic scene sees a
    different-but-physical PSF drawn from the ensemble.
  * grid ops (:meth:`resampled_to`, :meth:`centre_cropped_to`,
    :meth:`recentred`) that map over the members and return a new set.
  * FITS I/O — a multi-extension file whose **PrimaryHDU is the mean
    PSF** (so every existing single-PSF consumer that reads HDU[0]
    keeps working unchanged) and whose ImageHDUs 1..K are the cluster
    kernels.

``PSF`` stays a clean single-kernel value object; ``PSFSet`` only
*composes* it, so no ``PSF`` operation has to learn about stacks.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from astropy.io import fits

from euclid_polish.psf.core import PSF


@dataclass
class PSFSet:
    """An ordered ensemble of K :class:`PSF` kernels for one band.

    Parameters
    ----------
    psfs
        The K cluster PSFs. All must share ``pixel_scale`` and shape
        and be sum=1 normalised (the constructor does not enforce it;
        :meth:`from_psfs` / :meth:`from_fits` do).
    pixel_scale
        Arcsec/pixel of every member's grid (they all match).
    centroids
        Optional per-PSF ``(ra, dec)`` field centroid of the star
        cluster the kernel was built from — informational, written to
        the FITS headers for provenance.
    oversampling
        Optional EPSFBuilder oversampling factor (shared by members).

    Immutability mirrors :class:`PSF`: every operation returns a NEW
    ``PSFSet``; members are never mutated in place.
    """

    psfs: List[PSF]
    pixel_scale: float
    centroids: Optional[List[Tuple[float, float]]] = None
    oversampling: Optional[int] = None

    def __post_init__(self) -> None:
        if not self.psfs:
            raise ValueError("PSFSet needs at least one PSF")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n(self) -> int:
        return len(self.psfs)

    @property
    def shape(self) -> Tuple[int, int]:
        return self.psfs[0].shape

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_psfs(
        cls,
        psfs: List[PSF],
        *,
        centroids: Optional[List[Tuple[float, float]]] = None,
    ) -> PSFSet:
        """Build a set from a list of PSFs, normalising each to sum=1.

        Requires all members to share pixel scale and shape — the
        ensemble convex blend (and the mean) is only meaningful on a
        common grid.
        """
        if not psfs:
            raise ValueError("from_psfs needs at least one PSF")
        unit = [p.with_unit_sum() for p in psfs]
        scale = float(unit[0].pixel_scale)
        shape = unit[0].shape
        for p in unit[1:]:
            if abs(float(p.pixel_scale) - scale) > 1e-6:
                raise ValueError(
                    "all PSFs in a set must share pixel_scale "
                    f"({p.pixel_scale} != {scale})"
                )
            if p.shape != shape:
                raise ValueError(
                    f"all PSFs in a set must share shape ({p.shape} != {shape})"
                )
        return cls(
            psfs=unit,
            pixel_scale=scale,
            centroids=list(centroids) if centroids is not None else None,
            oversampling=unit[0].oversampling,
        )

    # ------------------------------------------------------------------
    # Reductions / sampling
    # ------------------------------------------------------------------

    def mean(self) -> PSF:
        """Field-averaged PSF (plain mean of the members; sum=1).

        Valid because every member is sum=1 → the unweighted mean is
        also sum=1. This is the single-PSF representative the legacy
        consumers (differential kernel, inference forward op, viz) get
        from ``HDU[0]``.
        """
        stack = np.mean([np.asarray(p.data, dtype=np.float64)
                         for p in self.psfs], axis=0)
        return PSF(
            data=stack.astype(np.float32),
            pixel_scale=self.pixel_scale,
            oversampling=self.oversampling,
        ).with_unit_sum()

    def sample(self, rng: np.random.Generator, *, alpha: float = 1.0) -> PSF:
        """Return a random convex blend of the K kernels.

        Weights ``w ~ Dirichlet(alpha · 1_K)`` (so ``Σw = 1`` and the
        blend is flux-preserving). ``alpha = 1`` is uniform over the
        simplex; ``alpha < 1`` concentrates near a single cluster PSF;
        ``alpha > 1`` concentrates near the mean. A 1-element set
        returns its only member (deterministic), so a single-PSF band
        reproduces the old behaviour exactly.
        """
        if self.n == 1:
            return self.psfs[0]
        w = rng.dirichlet([float(alpha)] * self.n)
        blended = np.zeros(self.shape, dtype=np.float64)
        for wi, p in zip(w, self.psfs):
            blended += float(wi) * np.asarray(p.data, dtype=np.float64)
        return PSF(
            data=blended.astype(np.float32),
            pixel_scale=self.pixel_scale,
            oversampling=self.oversampling,
        ).with_unit_sum()

    # ------------------------------------------------------------------
    # Grid operations — map over members, return a new PSFSet
    # ------------------------------------------------------------------

    def resampled_to(self, target_pixel_scale: float) -> PSFSet:
        out = [p.resampled_to(target_pixel_scale) for p in self.psfs]
        return PSFSet(
            psfs=out,
            pixel_scale=float(target_pixel_scale),
            centroids=self.centroids,
            oversampling=self.oversampling,
        )

    def centre_cropped_to(self, side: int, *, renormalise: bool = True) -> PSFSet:
        out = [p.centre_cropped_to(side, renormalise=renormalise)
               for p in self.psfs]
        return PSFSet(
            psfs=out,
            pixel_scale=self.pixel_scale,
            centroids=self.centroids,
            oversampling=self.oversampling,
        )

    def recentred(self, *, on: str = "peak", subpixel: bool = True) -> PSFSet:
        out = [p.recentred(on=on, subpixel=subpixel) for p in self.psfs]
        return PSFSet(
            psfs=out,
            pixel_scale=self.pixel_scale,
            centroids=self.centroids,
            oversampling=self.oversampling,
        )

    # ------------------------------------------------------------------
    # I/O — multi-extension FITS (PrimaryHDU = mean, ImageHDUs = clusters)
    # ------------------------------------------------------------------

    def save(self, output_dir: str, filename: str) -> str:
        """Write the set as a multi-extension FITS file.

        ``HDU[0]`` (Primary) is the **mean** PSF, sum=1 — so every
        existing reader that does ``PSF.from_fits`` / ``load_band_psf``
        transparently gets a single representative PSF. ``HDU[1..K]``
        are the cluster kernels (sum=1) with provenance headers.
        """
        os.makedirs(output_dir, exist_ok=True)
        fits_path = os.path.join(output_dir, filename)

        mean = self.mean()
        primary = fits.PrimaryHDU(data=mean.data.astype(np.float32))
        primary.header["PXSCALE"] = (
            float(self.pixel_scale), "Pixel scale (arcsec/pixel)")
        primary.header["NPSF"] = (int(self.n), "Number of cluster PSFs")
        if self.oversampling is not None:
            primary.header["OVERSAMP"] = (
                int(self.oversampling), "EPSFBuilder oversampling factor")
        if mean.fwhm_arcsec is not None:
            primary.header["FWHM"] = (
                float(mean.fwhm_arcsec), "Mean PSF FWHM (arcsec)")
        primary.header["COMMENT"] = (
            "PSFSet: HDU0=mean (sum=1), HDU1..N=cluster PSFs (sum=1).")

        hdus = [primary]
        for i, psf in enumerate(self.psfs):
            unit = psf.with_unit_sum()
            hdu = fits.ImageHDU(data=unit.data.astype(np.float32),
                                name=f"PSF{i:03d}")
            hdu.header["PXSCALE"] = (
                float(unit.pixel_scale), "Pixel scale (arcsec/pixel)")
            if unit.oversampling is not None:
                hdu.header["OVERSAMP"] = (int(unit.oversampling),
                                          "EPSFBuilder oversampling factor")
            if unit.fwhm_arcsec is not None:
                hdu.header["FWHM"] = (float(unit.fwhm_arcsec),
                                      "Measured FWHM (arcsec)")
            if self.centroids is not None and i < len(self.centroids):
                ra, dec = self.centroids[i]
                hdu.header["RA"] = (float(ra), "Cluster centroid RA (deg)")
                hdu.header["DEC"] = (float(dec), "Cluster centroid Dec (deg)")
            hdus.append(hdu)

        fits.HDUList(hdus).writeto(fits_path, overwrite=True)
        return fits_path

    @classmethod
    def from_fits(cls, fits_path: str, *, normalise: bool = True) -> PSFSet:
        """Load a :class:`PSFSet` from a FITS file.

        Reads the cluster kernels from ``HDU[1..K]`` when present. A
        **single-HDU legacy file** (the old one-PSF format, or any FITS
        with only a PrimaryHDU) loads as a 1-element set built from the
        primary — so the new loader works on both formats.
        """
        members: List[PSF] = []
        centroids: List[Tuple[float, float]] = []
        have_centroids = False
        with fits.open(fits_path) as hdul:
            image_hdus = [h for h in hdul
                          if getattr(h, "data", None) is not None]
            cluster_hdus = image_hdus[1:] if len(image_hdus) > 1 else image_hdus
            for h in cluster_hdus:
                data = np.asarray(h.data, dtype=np.float32)
                pix = h.header.get("PXSCALE", h.header.get("PIXSCALE", 0.0))
                oversamp = int(h.header["OVERSAMP"]) if "OVERSAMP" in h.header else None
                fwhm = float(h.header["FWHM"]) if "FWHM" in h.header else None
                members.append(PSF(data=data, pixel_scale=float(pix),
                                   fwhm_arcsec=fwhm, oversampling=oversamp))
                if "RA" in h.header and "DEC" in h.header:
                    centroids.append((float(h.header["RA"]),
                                      float(h.header["DEC"])))
                    have_centroids = True
                else:
                    centroids.append((float("nan"), float("nan")))
        if normalise:
            members = [p.with_unit_sum() for p in members]
        return cls(
            psfs=members,
            pixel_scale=float(members[0].pixel_scale),
            centroids=centroids if have_centroids else None,
            oversampling=members[0].oversampling,
        )
