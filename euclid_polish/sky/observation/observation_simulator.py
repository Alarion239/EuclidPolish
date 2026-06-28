"""
Multi-band forward model: HR clean (4 channels) → LR dirty (4 channels) + HR clean target (4 channels).

All four bands sit on a common 0.10″/pix LR grid, so every band is modelled the
same way. Pipeline per band on the HR canvas (0.05″ HR pixel scale, electrons):

  VIS / Y_E / J_E / H_E:
    HR (0.05″, e⁻)
      → fftconvolve with the band PSF sample (real ePSF / Gaussian fallback)
      → sum-rebin round(0.10 / 0.05) = 2× → 0.10″
      → Poisson(sky + signal) − sky + read_noise · √N_exp  (+ optional artifacts)
      → LR (0.10″, e⁻)

A Lanczos-3 resample-to-VIS-LR stage (sky/resample.py) maps each band onto the
shared LR grid; under the uniform 0.10″ band configuration its factor is 1, so
it is a no-op. It activates when a band's LR pixel scale differs (e.g. 0.30″).
Bright-star saturation is then applied to the assembled dirty LR stack.

The HR clean target keeps all four channels: the model super-resolves VIS and
the NISP bands jointly; band k of the target is band k of the LR input.

The output of :meth:`ObservationSimulator.process_hr_to_lr` is a pair of
``Image`` objects:

  * ``lr``  : (H_lr, W_lr, 4), pixel scale 0.10″, dirty (Poisson+read), e⁻
  * ``hr``  : (H_hr, W_hr, 4), pixel scale 0.05″, clean (all bands), e⁻
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from euclid_polish.config import BandConfig, Config
from euclid_polish.psf import PSF
from euclid_polish.psf.psf_set import PSFSet, PSFSample
# Re-export the canonical noise function (defined in sky.observation.noise).
from euclid_polish.sky.observation.noise import apply_band_noise   # noqa: F401
from euclid_polish.sky.observation.saturation import (
    StarSaturationModel, apply_star_saturation,
)
from euclid_polish.psf.psf_library import (
    make_gaussian_psf, psf_side_pixels_for_band,
)
from euclid_polish.sky.observation.resample import upsample as resample_upsample
from euclid_polish.image import Image, Role
from euclid_polish.provenance.defaults import mint_id
from euclid_polish.provenance.records import Stamp


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ObservationSimulatorConfig:
    add_noise: bool = True
    add_artifacts: bool = True       # cosmic rays + hot pixels
    add_saturation: bool = True      # bright-star detector saturation (per band)
    nisp_resample_kernel: str = Config.NISP_RESAMPLE_KERNEL  # "lanczos3" or "cubic"
    hr_pixel_scale: float = Config.DEFAULT_PIXEL_SCALE        # 0.05 arcsec
    artifact_config: Optional["ArtifactConfig"] = None        # type: ignore[name-defined]
    # Position-dependent PSF: when ``randomize_psf`` is on, each scene draws one
    # PSF — a star-count-weighted cluster pick, then with probability
    # (1 - psf_unrotated_prob) a random roll rotation (per-pointing telescope
    # roll). No blending. With ``randomize_psf`` off, the deterministic
    # field-mean PSF is used.
    # With psf_unrotated_prob=1.0, draw_sample always returns angle=None, so
    # apply_sample never rotates: each scene draws a random (star-count-weighted)
    # cluster PSF and applies it without roll rotation. Set randomize_psf=False
    # for the deterministic field-mean PSF; lower psf_unrotated_prob (<1.0) to
    # add roll-rotation augmentation.
    randomize_psf: bool = True
    psf_unrotated_prob: float = 1.0

    def __post_init__(self) -> None:
        if self.nisp_resample_kernel not in ("lanczos3", "cubic"):
            raise ValueError(
                f"nisp_resample_kernel must be 'lanczos3' or 'cubic'; "
                f"got {self.nisp_resample_kernel!r}"
            )
        if self.hr_pixel_scale <= 0:
            raise ValueError("hr_pixel_scale must be positive")


# ---------------------------------------------------------------------------
# PSF helpers
# ---------------------------------------------------------------------------

def default_psf_for_band(band: BandConfig, hr_pixel_scale: float) -> PSF:
    """Construct a Gaussian PSF for ``band`` at the HR pixel scale.

    The stamp side is derived from the band's FWHM via
    :func:`psf_side_pixels_for_band`, so each band gets a kernel sized
    to its own resolution (a wide H_E PSF is wider than a narrow VIS
    PSF). For VIS the caller usually supplies the empirical ePSF
    instead; this Gaussian is the fallback when no PSF is provided.
    """
    side = psf_side_pixels_for_band(band, hr_pixel_scale)
    return make_gaussian_psf(band.psf_fwhm_arcsec, hr_pixel_scale, size=side)


# ---------------------------------------------------------------------------
# Forward model
# ---------------------------------------------------------------------------

# ``apply_band_noise`` is defined in :mod:`euclid_polish.sky.noise` and
# re-exported at the top of this module (see the import block above).


class ObservationSimulator:
    """Apply per-band PSF + noise + (NISP→VIS-LR resample) to a 4-band HR field."""

    def __init__(
        self,
        psfs_by_band: Optional[Dict[str, PSF]] = None,
        config: Optional[ObservationSimulatorConfig] = None,
        *,
        psf_sets_by_band: Optional[Dict[str, PSFSet]] = None,
    ):
        """
        Parameters
        ----------
        psfs_by_band : dict mapping band name → PSF, all on the HR grid.
                       Each is wrapped as a 1-element :class:`PSFSet` (so a
                       single PSF behaves deterministically). Missing entries
                       are filled with a Gaussian fallback.
        psf_sets_by_band : dict mapping band name → :class:`PSFSet` (the
                       position-dependent ensemble). Takes priority over
                       ``psfs_by_band`` for any band present in both. Enables
                       the per-scene random cluster pick (one kernel per scene;
                       no blending).
        config       : :class:`ObservationSimulatorConfig`.
        """
        self.config = config or ObservationSimulatorConfig()
        # Unify on PSFSets internally; a 1-element set is a single fixed PSF.
        sets: Dict[str, PSFSet] = (
            dict(psf_sets_by_band) if psf_sets_by_band is not None else {})
        if psfs_by_band is not None:
            for name, psf in psfs_by_band.items():
                sets.setdefault(name, PSFSet.from_psfs([psf]))
        # Fill missing bands with Gaussian fallbacks (1-element set) at HR scale.
        for band in Config.BANDS:
            if band.name not in sets:
                sets[band.name] = PSFSet.from_psfs(
                    [default_psf_for_band(band, self.config.hr_pixel_scale)])
        self._psf_sets = sets
        # Bright-star saturation model (per-band well depths precomputed once).
        self._sat_model = (StarSaturationModel()
                           if self.config.add_saturation else None)
        # Sanity-check PSF pixel scales (every member of every set).
        for band_name, pset in self._psf_sets.items():
            for psf in pset.psfs:
                if abs(psf.pixel_scale - self.config.hr_pixel_scale) > 1e-4:
                    raise ValueError(
                        f"PSF for band {band_name} has pixel_scale={psf.pixel_scale}; "
                        f"forward model expects HR pixel scale {self.config.hr_pixel_scale}"
                    )

    # ------------------------------------------------------------------ #
    @staticmethod
    def sum_rebin(arr_2d: np.ndarray, factor: int) -> np.ndarray:
        """Photometric sum-rebin with trailing-row trim.

        Wrapper around :meth:`euclid_polish.image.Image.rebin_array`.
        Trailing rows / cols that don't fit a whole bin are trimmed
        (``trim_remainder=True``).
        """
        return Image.rebin_array(
            arr_2d, int(factor), trim_remainder=True,
        )

    # ------------------------------------------------------------------ #
    @property
    def target_lr_pixel_scale_arcsec(self) -> float:
        """Pixel scale of the unified LR grid every channel ends up on.

        Anchored to the VIS LR grid. Non-VIS bands are resampled into this
        grid after their own native rebin + noise stage.
        """
        return Config.BAND_VIS.pixel_scale_lr_arcsec

    def _draw_psf_sample(self, rng: np.random.Generator) -> "PSFSample":
        """Draw ONE :class:`PSFSample` (cluster index + roll) for the whole
        scene, from the band with the most cluster PSFs (the reference for the
        common clustering). Applied to every band so all four share the field
        position and the telescope roll — physically one pointing."""
        ref = max(self._psf_sets.values(), key=lambda p: p.n)
        return ref.draw_sample(
            rng, use_unrotated_prob=self.config.psf_unrotated_prob)

    def _process_one_band(
        self,
        hr_channel: np.ndarray,
        band: BandConfig,
        rng: np.random.Generator,
        psf_spec: "Optional[PSFSample]" = None,
    ) -> np.ndarray:
        """HR (0.05″) → LR-on-the-shared-grid (= VIS LR) for one channel."""
        # Realise the scene's shared PSF sample against this band's set: cluster
        # ``psf_spec.index`` rotated by the shared roll (no blending).
        # ``psf_spec=None`` → the field-mean PSF.
        pset = self._psf_sets[band.name]
        psf = pset.apply_sample(psf_spec) if psf_spec is not None else pset.mean()
        target_scale = self.target_lr_pixel_scale_arcsec

        # 1. PSF convolution on HR plane via PSF.convolved_with (sum=1-normalises
        #    the kernel and runs fftconvolve mode="same" + float32 cast).
        hr_e = psf.convolved_with(hr_channel)

        # 2. Sum-rebin to the band's LR scale (preserves photon shot noise
        #    statistics at the right pixel size for the per-band noise step).
        rebin_factor = int(round(band.pixel_scale_lr_arcsec / self.config.hr_pixel_scale))
        lr_signal_e = self.sum_rebin(hr_e, rebin_factor)

        # 3. Apply per-band noise on the LR grid.
        if self.config.add_noise:
            lr_e = apply_band_noise(
                lr_signal_e, band, rng,
                add_artifacts=self.config.add_artifacts,
                artifact_config=self.config.artifact_config,
            )
        else:
            lr_e = lr_signal_e.astype(np.float32, copy=False)

        # 4. Resample to the shared LR grid; VIS factor is 1.
        upsample_factor = int(round(band.pixel_scale_lr_arcsec / target_scale))
        if upsample_factor > 1:
            lr_e = resample_upsample(
                lr_e, factor=upsample_factor, kernel=self.config.nisp_resample_kernel,
            )
        return lr_e.astype(np.float32, copy=False)

    # ------------------------------------------------------------------ #
    def apply(self, hr: Image, rng=None, *, store=None) -> Image:
        """Forward-model a clean HR :class:`Image` into a dirty LR Image (role ``'lr'``).

        The OO operator verb over :meth:`process`: PSF-convolve, rebin, add
        per-band noise + artifacts, returning the dirty LR input the model
        super-resolves. The LR's provenance parent is ``hr``'s id.

            lr = forward.apply(hr)
        """
        if rng is None:
            rng = np.random.default_rng()
        lr_img, _hr_out = self.process(hr, rng)
        parent = hr.stamp.id if hr.stamp is not None else None
        parents = tuple(p for p in (parent,) if p is not None)
        return lr_img.with_role(Role.LR).with_stamp(
            Stamp(id=mint_id(store), parents=parents, schema_version=3,
                  subset=hr.subset))

    def process(
        self,
        hr_4ch: Image,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[Image, Image]:
        """Run the full forward model on one HR clean 4-channel field.

        Parameters
        ----------
        hr_4ch : :class:`Image` with shape (H, W, 4),
                 ``pixel_scale_arcsec == hr_pixel_scale``, band order
                 ``Config.LR_INPUT_BAND_NAMES``.
        rng    : reproducible noise source; ``np.random.default_rng()`` if None.

        Returns
        -------
        lr_4ch : :class:`Image` shape (H_lr, W_lr, 4) at 0.10″/pix,
                 e⁻ (can be negative after sky subtraction).
        hr_4ch_out : :class:`Image` shape (H, W, 4) at 0.05″/pix,
                 e⁻, clean (no noise) — the network's 4-band HR target.
        """
        if abs(hr_4ch.pixel_scale_arcsec - self.config.hr_pixel_scale) > 1e-4:
            raise ValueError(
                f"hr_4ch.pixel_scale_arcsec={hr_4ch.pixel_scale_arcsec} does not "
                f"match forward.hr_pixel_scale={self.config.hr_pixel_scale}"
            )
        if hr_4ch.band_names != Config.LR_INPUT_BAND_NAMES:
            raise ValueError(
                f"hr_4ch.band_names={hr_4ch.band_names} must equal "
                f"Config.LR_INPUT_BAND_NAMES={Config.LR_INPUT_BAND_NAMES}"
            )
        if rng is None:
            rng = np.random.default_rng()

        # HR canvas must be divisible by every band's rebin factor so all four
        # LR channels land on the same grid after the band-specific rebin (and
        # NISP upsample) chain. Trim to the largest such multiple — a few
        # pixels at the edge are not science-relevant.
        max_rebin = max(
            int(round(b.pixel_scale_lr_arcsec / self.config.hr_pixel_scale))
            for b in Config.BANDS
        )
        H_full, W_full = hr_4ch.data.shape[:2]
        H_trim = (H_full // max_rebin) * max_rebin
        W_trim = (W_full // max_rebin) * max_rebin
        hr_data_trim = hr_4ch.data[:H_trim, :W_trim, :]

        # Draw ONE PSF sample (cluster index + roll) for the whole scene so all
        # four bands share the field position and the telescope roll — one
        # physical pointing. ``None`` (randomisation off) → each band's mean.
        psf_spec = self._draw_psf_sample(rng) if self.config.randomize_psf else None

        # Process each channel; the four LR channels are stacked at the end.
        lr_channels = []
        for k, band_name in enumerate(Config.LR_INPUT_BAND_NAMES):
            band = Config.get_band(band_name)
            lr_channels.append(self._process_one_band(
                hr_data_trim[..., k], band, rng, psf_spec=psf_spec,
            ))
        # All channels must end on the same grid (the VIS LR grid).
        target_shape = lr_channels[0].shape
        for k, ch in enumerate(lr_channels):
            if ch.shape != target_shape:
                raise RuntimeError(
                    f"Channel {k} has shape {ch.shape}; expected {target_shape}. "
                    "Check rebin / resample factors."
                )
        lr_stack = np.stack(lr_channels, axis=-1)

        # Bright-star detector saturation: clip a blocky region per saturating
        # (star, band) onto the dirty LR image. Independent per band; the clean
        # HR target is untouched. The native well depth is the clip level on
        # this shared 0.10″ grid.
        if self._sat_model is not None:
            stars = (getattr(hr_4ch, "metadata", None) or {}).get("stars", [])
            if stars:
                hr_to_lr = self.config.hr_pixel_scale / self.target_lr_pixel_scale_arcsec
                apply_star_saturation(
                    lr_stack, stars, self._sat_model, rng,
                    hr_to_lr_scale=hr_to_lr,
                    band_names=Config.LR_INPUT_BAND_NAMES)

        # HR target: all four bands (clean, no noise applied), trimmed to the
        # same spatial extent the LR pipeline saw. Band k of the target is
        # band k of the LR input — the model super-resolves VIS+NISP jointly.
        hr_clean = hr_data_trim.astype(np.float32, copy=True)

        lr_img = Image(
            data=lr_stack,
            pixel_scale_arcsec=self.target_lr_pixel_scale_arcsec,
            band_names=Config.LR_INPUT_BAND_NAMES,
            is_clean=False,
            index=hr_4ch.index,
            subset=hr_4ch.subset,
        )
        hr_img = Image(
            data=hr_clean,
            pixel_scale_arcsec=hr_4ch.pixel_scale_arcsec,
            band_names=Config.HR_TARGET_BAND_NAMES,
            is_clean=True,
            index=hr_4ch.index,
            subset=hr_4ch.subset,
            metadata=hr_4ch.metadata,
        )
        return lr_img, hr_img
