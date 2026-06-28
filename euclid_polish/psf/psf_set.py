"""The :class:`PSFSet` — an ordered ensemble of PSFs for one band.

The Euclid PSF varies across the focal plane. Rather than collapse a
band's star cutouts into one average ePSF, we cluster the stars
spatially and build **one ePSF per cluster**, giving K kernels that
sample the field's PSF variation. :class:`PSFSet` holds those K
:class:`PSF` kernels (all on the same grid, same shape, each sum=1)
and exposes:

  * :meth:`mean`   — the field-averaged PSF (a single :class:`PSF`).
  * :meth:`sample_for_generation` — draw one PSF for a synthetic scene:
    pick a cluster weighted by its star count (few-star, noisier PSFs
    appear rarely), then with a configurable probability rotate it by a
    random roll angle (modelling the per-pointing telescope roll). We do
    NOT blend two cluster PSFs — blending rolls would superimpose
    diffraction spikes into an unphysical multi-spike PSF.
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
from typing import ClassVar

import numpy as np
from astropy.io import fits

from euclid_polish.provenance.fits import read_stamp_cards, write_stamp_cards
from euclid_polish.provenance.persistable import StampCarrier
from euclid_polish.provenance.records import Format
from euclid_polish.psf.core import PSF


@dataclass(frozen=True)
class PSFSample:
    """One scene's PSF draw, shared across all bands so the synthetic data is
    physically consistent (one pointing → one field position + one roll).

    ``index`` selects the cluster PSF (the band PSFSets share a common
    clustering + numbering, so the same index is the same field region in
    every band). ``angle`` is the telescope roll in degrees, or ``None`` for
    no rotation. The same :class:`PSFSample` is applied to every band's set."""

    index: int
    angle: float | None = None


@dataclass
class PSFSet(StampCarrier):
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
    n_stars
        Optional per-PSF count of the stars its ePSF was built from.
        Used as the sampling weight (a PSF from few stars is noisier, so
        it should appear rarely). ``None`` → uniform sampling.
    oversampling
        Optional EPSFBuilder oversampling factor (shared by members).

    Immutability mirrors :class:`PSF`: every operation returns a NEW
    ``PSFSet``; members are never mutated in place.
    """

    psfs: list[PSF]
    pixel_scale: float
    centroids: list[tuple[float, float]] | None = None
    n_stars: list[int] | None = None
    oversampling: int | None = None

    PROV_FORMAT: ClassVar[Format] = Format.FITS

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
    def shape(self) -> tuple[int, int]:
        return self.psfs[0].shape

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_psfs(
        cls,
        psfs: list[PSF],
        *,
        centroids: list[tuple[float, float]] | None = None,
        n_stars: list[int] | None = None,
    ) -> PSFSet:
        """Build a set from a list of PSFs, normalising each to sum=1.

        Requires all members to share pixel scale and shape — the mean
        and the sampling weights are only meaningful on a common grid.
        ``n_stars`` (per-PSF star counts) becomes the sampling weight.
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
            n_stars=[int(c) for c in n_stars] if n_stars is not None else None,
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

    def _pick_weights(self) -> np.ndarray | None:
        """Per-PSF sampling probabilities ∝ star count, or ``None`` (→
        uniform) when no usable counts are present."""
        if (self.n_stars is not None and len(self.n_stars) == self.n
                and sum(self.n_stars) > 0):
            w = np.asarray(self.n_stars, dtype=np.float64)
            return w / w.sum()
        return None

    def draw_sample(
        self,
        rng: np.random.Generator,
        *,
        use_unrotated_prob: float = 0.3,
        angle_min: int = 1,
        angle_max: int = 359,
    ) -> PSFSample:
        """Draw a :class:`PSFSample` (cluster index + roll) for one scene.

        1. Pick a cluster with probability **proportional to its star count**
           (``n_stars``); few-star, noisier PSFs appear rarely. Uniform when
           no counts are present.
        2. With probability ``use_unrotated_prob`` (default 0.3) no rotation.
        3. Otherwise a random integer roll in ``[angle_min, angle_max]``
           (default 1..359 — 0/360 is the identity, covered by step 2).

        Drawn ONCE per scene from a reference set and applied to every band
        via :meth:`apply_sample`, so all bands share the field position + roll.
        """
        idx = int(rng.choice(self.n, p=self._pick_weights()))
        if rng.random() < float(use_unrotated_prob):
            return PSFSample(index=idx, angle=None)
        return PSFSample(index=idx,
                         angle=float(rng.integers(int(angle_min),
                                                  int(angle_max) + 1)))

    def apply_sample(self, sample: PSFSample, *, rotation_order: int = 3) -> PSF:
        """Realise a :class:`PSFSample` against THIS band's set: take cluster
        ``sample.index`` (clamped if this set has fewer members — e.g. a
        Gaussian-fallback band) and rotate by the shared roll. No blending,
        no cropping — one real single-roll kernel."""
        psf = self.psfs[min(int(sample.index), self.n - 1)]
        if sample.angle is None:
            return psf
        return psf.rotated(float(sample.angle), order=int(rotation_order))

    def sample_for_generation(
        self,
        rng: np.random.Generator,
        *,
        use_unrotated_prob: float = 0.3,
        angle_min: int = 1,
        angle_max: int = 359,
        rotation_order: int = 3,
    ) -> PSF:
        """Single-set convenience: ``apply_sample(draw_sample(rng))``. The
        multi-band generator instead draws one :class:`PSFSample` and applies
        it across all bands so the roll + field position are shared."""
        spec = self.draw_sample(rng, use_unrotated_prob=use_unrotated_prob,
                                angle_min=angle_min, angle_max=angle_max)
        return self.apply_sample(spec, rotation_order=rotation_order)

    # ------------------------------------------------------------------
    # Grid operations — map over members, return a new PSFSet
    # ------------------------------------------------------------------

    def _with_psfs(
        self, psfs: list[PSF], pixel_scale: float | None = None
    ) -> PSFSet:
        """Return a new PSFSet with replaced ``psfs``, copying all sidecar fields.

        ``pixel_scale`` overrides the stored value (use for :meth:`resampled_to`
        where the scale changes); omit to keep the current scale.
        """
        result = PSFSet(
            psfs=psfs,
            pixel_scale=float(pixel_scale) if pixel_scale is not None
                        else self.pixel_scale,
            centroids=self.centroids,
            n_stars=self.n_stars,
            oversampling=self.oversampling,
        )
        if self.stamp is not None:
            result = result.with_stamp(self.stamp)
        return result

    def resampled_to(self, target_pixel_scale: float) -> PSFSet:
        out = [p.resampled_to(target_pixel_scale) for p in self.psfs]
        return self._with_psfs(out, pixel_scale=target_pixel_scale)

    def background_cleaned(self) -> PSFSet:
        """Apply :meth:`PSF.background_cleaned` to every member (noise-floor
        cut + radial taper, Config-tuned)."""
        return self._with_psfs([p.background_cleaned() for p in self.psfs])

    def centre_cropped_to(self, side: int, *, renormalise: bool = True) -> PSFSet:
        out = [p.centre_cropped_to(side, renormalise=renormalise)
               for p in self.psfs]
        return self._with_psfs(out)

    def recentred(self, *, on: str = "peak", subpixel: bool = True) -> PSFSet:
        out = [p.recentred(on=on, subpixel=subpixel) for p in self.psfs]
        return self._with_psfs(out)

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
        if self.stamp is not None:
            write_stamp_cards(primary.header, self.stamp)

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
            if self.n_stars is not None and i < len(self.n_stars):
                hdu.header["NSTARS"] = (int(self.n_stars[i]),
                                        "Stars the ePSF was built from")
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
        members: list[PSF] = []
        centroids: list[tuple[float, float]] = []
        star_counts: list[int] = []
        have_centroids = False
        have_counts = False
        stamp = None
        with fits.open(fits_path) as hdul:
            stamp = read_stamp_cards(hdul[0].header)
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
                if "NSTARS" in h.header:
                    star_counts.append(int(h.header["NSTARS"]))
                    have_counts = True
                else:
                    star_counts.append(0)
        if normalise:
            members = [p.with_unit_sum() for p in members]
        result = cls(
            psfs=members,
            pixel_scale=float(members[0].pixel_scale),
            centroids=centroids if have_centroids else None,
            n_stars=star_counts if have_counts else None,
            oversampling=members[0].oversampling,
        )
        if stamp is not None:
            result = result.with_stamp(stamp)
        return result
