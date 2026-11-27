"""
Multi-band forward model: HR clean (4 channels) → LR dirty (4 channels) + HR clean target (4 channels).

All four bands reach the network on a common 0.10″/pix LR grid. The optical
signal is modelled there because the empirical MER ePSFs already contain the
detector sampling and mosaic interpolation. Noise follows the detector:

  VIS / Y_E / J_E / H_E:
    HR (0.05″, e⁻)
      → fftconvolve with the band PSF sample (real ePSF / Gaussian fallback)
      → sum-rebin round(0.10 / 0.05) = 2× → 0.10″
      → MER noise, every band alike: Euclid's MER noise level plus source
        photon noise, with the pixel correlation of the scene's exposures
        (four at the median level) shifted, bilinearly resampled onto 0.10″
        (from 0.10″ VIS, 0.30″ NISP detector pixels) and median-combined,
        then sparse artifacts
      → LR (0.10″, e⁻)

Bright-star saturation is then applied to the assembled dirty LR stack.

The HR clean target keeps all four channels: the model super-resolves VIS and
the NISP bands jointly; band k of the target is band k of the LR input.

The output of :meth:`ObservationSimulator.process_hr_to_lr` is a pair of
``Image`` objects:

  * ``lr``  : (H_lr, W_lr, 4), pixel scale 0.10″, dirty (MER noise), e⁻
  * ``hr``  : (H_hr, W_hr, 4), pixel scale 0.05″, clean (all bands), e⁻
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

import numpy as np

from euclid_polish.config import BandConfig, Config
from euclid_polish.image import Image, Role
from euclid_polish.photometry import ab_mag_to_electrons, electrons_to_ab_mag
from euclid_polish.provenance.defaults import mint_id
from euclid_polish.provenance.records import Stamp
from euclid_polish.psf import PSF
from euclid_polish.psf.psf_library import (
    make_gaussian_psf,
    psf_side_pixels_for_band,
)
from euclid_polish.psf.psf_set import PSFSample, PSFSet
from euclid_polish.sky.observation.blank_pixels import BAND_NAMES as BLANK_MASK_BAND_NAMES
from euclid_polish.sky.observation.blank_pixels import BlankPixelModel
from euclid_polish.sky.observation.bright_stars import (
    FarFieldParams,
    OffFieldPrior,
    TrailParams,
    draw_jitter,
    draw_off_field_stars,
    draw_trail_masks,
    far_field_flux,
    join_radii,
    load_far_field_params,
    load_off_field_prior,
    load_trail_params,
    render_far_field,
    ridge_reach_arcsec,
    spike_ridges,
    trail_axis,
)
from euclid_polish.sky.observation.field_variations import (
    DistantStarWing,
    add_distant_star_wings,
    draw_distant_star_wings,
    draw_noise_scale_map,
)
from euclid_polish.sky.observation.mer_noise_levels import load_mer_noise_levels
from euclid_polish.sky.observation.noise import NoiseTextureJitter, apply_archive_noise
from euclid_polish.sky.observation.resample import upsample as resample_upsample
from euclid_polish.sky.observation.saturation import (
    StarSaturationModel,
    apply_saturation_masking,
)

if TYPE_CHECKING:
    from euclid_polish.sky.observation.artifacts import ArtifactConfig

_ResampleKernel = Literal["bilinear", "cubic"]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ObservationSimulatorConfig:
    add_noise: bool = True
    add_artifacts: bool = True       # CRs, hot/dead pixels, streaks, distant-star wings
    add_saturation: bool = True      # bright-star detector saturation (per band)
    nisp_resample_kernel: str = Config.NISP_RESAMPLE_KERNEL  # "bilinear" or "cubic"
    hr_pixel_scale: float = Config.DEFAULT_PIXEL_SCALE        # 0.05 arcsec
    artifact_config: ArtifactConfig | None = None
    # Identity of the delivered-MER noise model, not a knob: it rides along in
    # every provenance snapshot of this config so training can refuse records
    # generated under a different noise model.
    noise_model: str = Config.NOISE_MODEL
    # Draw each scene's four-band sky noise level from the measured Q1
    # distribution (one real sky position per scene). Off uses every band's
    # median level.
    draw_mer_noise_levels: bool = True
    # Randomise the noise texture per scene and band (live training only):
    # the exposure-count range of the dithered stack, and the largest white
    # share of its per-pixel variance. ``None`` keeps the fixed texture
    # every record was generated with.
    noise_texture_exposures: tuple[int, int] | None = None
    noise_texture_white_max: float = 0.0
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
    # Optional elastic observation-PSF augmentation. One deformed PSF sample
    # is shared by the ordinary scene and the separately carried star plane,
    # matching a single physical exposure. The generic operator defaults OFF;
    # generation and OnTheFlyForward configure it persistently.
    psf_warp_prob: float = 0.0
    psf_warp_alpha_max: float = 0.0
    psf_warp_sigma: float = 3.0
    # Probability that an above-well connected source is converted into the
    # MER-style rectangular blackout. A value of one keeps the legacy
    # always-mask operator; train/validate/test and on-the-fly configs pass the
    # lower, real-field-calibrated training default explicitly.
    saturation_mask_prob: float = 1.0
    # Blackout probability of the brightest sources. A source's probability
    # rises in log(peak/well) from ``saturation_mask_prob`` at the first well
    # ratio to this value at the second, as in real MER fields where
    # marginally saturated cores usually survive but the brightest are almost
    # always masked. It never drops below ``saturation_mask_prob``, so the
    # always-mask default is unchanged.
    saturation_mask_prob_bright: float = Config.SATURATION_MASK_PROB_BRIGHT
    saturation_mask_ramp_well_ratios: tuple[float, float] = (
        Config.SATURATION_MASK_RAMP_WELL_RATIOS
    )
    # Blank (exactly zero) pixels where every exposure of the stack is
    # invalid — persistence masks, cosmic rays, bad pixels and, with
    # ``add_saturation``, the cores of bright stars — at the rates of this
    # release ("dr1" or "q1", see ``blank_pixels.py``). It then replaces the
    # rectangular saturation blackout. ``None`` keeps that legacy step, as
    # the generated record sets were made; live training defaults to
    # ``Config.TRAIN_BLANK_PIXELS``.
    blank_pixels: str | None = None
    # A very bright star thousands of pixels beyond the generated field can
    # cast a nearly straight, field-spanning PSF wing through the LR cutout.
    add_distant_star_wings: bool = True
    distant_star_wing_probability: float = 0.20
    distant_star_wing_amplitude_sigma_min: float = 2.5
    distant_star_wing_amplitude_sigma_max: float = 8.0
    distant_star_wing_width_min_lr_pix: float = 0.8
    distant_star_wing_width_max_lr_pix: float = 2.0
    distant_star_wing_fade_length_min_lr_pix: float = 60.0
    distant_star_wing_fade_length_max_lr_pix: float = 220.0
    # Far-field light of stars beyond the ePSF stamp (``bright_stars.py``):
    # the measured DR1 halo and six spike ridges, at the scene kernel's own
    # spike angles, joined to the kernel in the ``bright_star_join_arcsec``
    # annulus (moved inside a stamp's own edge taper when the stamp is
    # smaller). Star kernels are scaled so kernel + far field keep the
    # star's flux; galaxy light is scaled by the same 1 - f_far (its far
    # field lies far below the noise and is not drawn). Off reproduces the
    # stamp-only scene bit for bit.
    bright_star_far_field: bool = False
    # Taper the NISP (Y/J/H) ePSFs at the join annulus instead of at their
    # 21-25″ stamp edge. Removes their flat outer pedestal (7-9% of the flux
    # at 10-26″ against ~0.9% in real H stars) and puts that light back into
    # the core, so NISP photometry of every source changes. Off by default.
    nisp_psf_taper: bool = False
    # The ePSFs run too bright beyond ~7″ (VIS halo 3.5× the real one at 9″,
    # NISP 2× at 9″ and 5× at 14″); real and kernel agree at 3-6″.
    bright_star_join_arcsec: tuple[float, float] = (4.5, 6.5)
    # Stars whose far field never reaches this many electrons per pixel are
    # not rendered (cost only; far below any noise level).
    bright_star_min_peak_e: float = 0.05
    # Bright stars OUTSIDE the field (needs ``bright_star_far_field``): drawn
    # at the DR1 Wide counts (VIS 6.5-16) around the field out to where their
    # brightest ridge falls below ``off_field_min_peak_sigma`` of the median
    # sky noise; their kernel spill and far field (halo + spikes at the
    # scene's spike angles) reach into the field, while the target stays
    # without them. Replaces ``distant_star_wings``. Off by default.
    off_field_stars: bool = False
    off_field_min_peak_sigma: float = 0.05
    # Blank trails of bright stars (needs ``blank_pixels``): the VIS bleed
    # trail (VIS <~ 15, 1-50″ long, ~1″ wide) and the ~150″ bowed NISP
    # persistence trail (VIS <~ 12), along the scene's detector columns
    # (strongest spike axis + 9.45°), for in-field and off-field stars.
    # Drawn after the other blanks. Off by default.
    bright_star_trails: bool = False
    # A tiny per-scene depth jitter plus an occasional per-band pointing
    # seam. Both scale distributions are centred on one.
    # The jitter stays small on purpose: the drawn Q1 level already carries
    # the real field-to-field spread (sd of log level 19% in VIS, 14-15% in
    # NISP), so a wide knob both inflates that spread and, being a single
    # common factor across the bands, lifts the measured VIS-NISP level
    # correlation (0.30 -> 0.53 at +-20%). It exists only to keep scenes off
    # the 294 discrete table values; the table's own levels are medians over
    # thousands of noise-map pixels, so they are precise to well under 1%.
    add_noise_variation: bool = True
    noise_global_scale_min: float = 0.99
    noise_global_scale_max: float = 1.01
    # The strip stands for a within-field pointing seam. Measured on the 294
    # scene-sized Q1 noise cutouts (2026-09-19): a clean straight-line step
    # appears in 7-12% of fields, and per band independently — of 35 fields
    # with a clean VIS seam only one also had a clean J seam and none shared
    # its geometry, so every band draws its own.
    noise_region_probability: float = 0.10
    noise_region_fraction_min: float = 0.25
    noise_region_fraction_max: float = 0.50
    # A seam is one side of the field getting a different number of exposures,
    # so its depth step is the square root of a small integer ratio:
    # sqrt(4/3) = 1.15, sqrt(3/2) = 1.22, sqrt(2) = 1.41. The measured clean
    # steps cluster right there (p50 1.20, p90 1.36, max 2.03), far short of
    # the 0.5-1.5 this used to guess. Applied deeper or shallower alike.
    noise_region_step_min: float = 1.10
    noise_region_step_max: float = 1.45

    def __post_init__(self) -> None:
        if self.nisp_resample_kernel not in ("bilinear", "cubic"):
            raise ValueError(
                f"nisp_resample_kernel must be 'bilinear' or 'cubic'; "
                f"got {self.nisp_resample_kernel!r}"
            )
        if self.hr_pixel_scale <= 0:
            raise ValueError("hr_pixel_scale must be positive")
        if self.noise_model != Config.NOISE_MODEL:
            raise ValueError(
                f"noise_model must be {Config.NOISE_MODEL!r}; it identifies the "
                f"implemented noise model and cannot be chosen (got "
                f"{self.noise_model!r})"
            )
        if not 0.0 <= float(self.psf_warp_prob) <= 1.0:
            raise ValueError("psf_warp_prob must be in [0, 1]")
        if float(self.psf_warp_alpha_max) < 0.0:
            raise ValueError("psf_warp_alpha_max must be >= 0")
        if float(self.psf_warp_sigma) <= 0.0:
            raise ValueError("psf_warp_sigma must be > 0")
        if not 0.0 <= float(self.saturation_mask_prob) <= 1.0:
            raise ValueError("saturation_mask_prob must be in [0, 1]")
        if not 0.0 <= float(self.saturation_mask_prob_bright) <= 1.0:
            raise ValueError("saturation_mask_prob_bright must be in [0, 1]")
        ratios = tuple(
            float(value) for value in self.saturation_mask_ramp_well_ratios
        )
        if len(ratios) != 2 or not 0.0 < ratios[0] < ratios[1]:
            raise ValueError(
                "saturation_mask_ramp_well_ratios must be (start, full) "
                "with 0 < start < full"
            )
        self.saturation_mask_ramp_well_ratios = (ratios[0], ratios[1])
        if not 0.0 <= float(self.distant_star_wing_probability) <= 1.0:
            raise ValueError("distant_star_wing_probability must be in [0, 1]")
        if not (
            0.0 < float(self.distant_star_wing_amplitude_sigma_min)
            <= float(self.distant_star_wing_amplitude_sigma_max)
        ):
            raise ValueError(
                "distant-star wing amplitudes must satisfy 0 < min <= max"
            )
        if not (
            0.0 < float(self.distant_star_wing_width_min_lr_pix)
            <= float(self.distant_star_wing_width_max_lr_pix)
        ):
            raise ValueError(
                "distant-star wing widths must satisfy 0 < min <= max"
            )
        if not (
            0.0 < float(self.distant_star_wing_fade_length_min_lr_pix)
            <= float(self.distant_star_wing_fade_length_max_lr_pix)
        ):
            raise ValueError(
                "distant-star wing fade lengths must satisfy 0 < min <= max"
            )
        join = tuple(float(value) for value in self.bright_star_join_arcsec)
        if len(join) != 2 or not 0.0 < join[0] < join[1]:
            raise ValueError(
                "bright_star_join_arcsec must be (r_in, r_out) with 0 < r_in < r_out"
            )
        self.bright_star_join_arcsec = (join[0], join[1])
        if float(self.bright_star_min_peak_e) < 0.0:
            raise ValueError("bright_star_min_peak_e must be >= 0")
        if self.off_field_stars and not self.bright_star_far_field:
            raise ValueError("off_field_stars needs bright_star_far_field")
        if not float(self.off_field_min_peak_sigma) > 0.0:
            raise ValueError("off_field_min_peak_sigma must be > 0")
        if self.bright_star_trails and not self.blank_pixels:
            raise ValueError("bright_star_trails needs blank_pixels (a release)")
        if not (
            0.0 < float(self.noise_global_scale_min)
            <= float(self.noise_global_scale_max)
        ):
            raise ValueError(
                "global noise scales must satisfy 0 < min <= max"
            )
        if not 0.0 <= float(self.noise_region_probability) <= 1.0:
            raise ValueError("noise_region_probability must be in [0, 1]")
        if not (
            0.0 < float(self.noise_region_fraction_min)
            <= float(self.noise_region_fraction_max) < 1.0
        ):
            raise ValueError(
                "noise region fractions must satisfy 0 < min <= max < 1"
            )
        if not (
            1.0 <= float(self.noise_region_step_min)
            <= float(self.noise_region_step_max)
        ):
            raise ValueError(
                "regional depth steps must satisfy 1 <= min <= max"
            )


# ---------------------------------------------------------------------------
# PSF helpers
# ---------------------------------------------------------------------------

def default_psf_for_band(band: BandConfig, hr_pixel_scale: float) -> PSF:
    """Construct a Gaussian PSF for ``band`` at the HR pixel scale.

    The stamp side is derived from the band's FWHM via
    :func:`psf_side_pixels_for_band`, so each band gets a kernel sized
    to its own resolution (a wide H_E PSF is wider than a narrow VIS
    PSF). Callers usually supply an empirical ePSF set for every band
    (:func:`load_all_band_psf_sets`); this Gaussian is the fallback for a
    band with no PSF provided.
    """
    side = psf_side_pixels_for_band(band, hr_pixel_scale)
    return make_gaussian_psf(band.psf_fwhm_arcsec, hr_pixel_scale, size=side)


# ---------------------------------------------------------------------------
# Forward model
# ---------------------------------------------------------------------------

class ObservationSimulator:
    """Apply per-band PSF plus detector/MER noise to a 4-band HR field."""

    def __init__(
        self,
        psfs_by_band: dict[str, PSF] | None = None,
        config: ObservationSimulatorConfig | None = None,
        *,
        psf_sets_by_band: dict[str, PSFSet] | None = None,
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
        sets: dict[str, PSFSet] = (
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
        # Bright-star far field: per band the join annulus, the kernels
        # tapered there, and the far field's expected flux share.
        self._far_params: FarFieldParams | None = None
        self._far_join: dict[str, tuple[float, float]] = {}
        self._far_flux: dict[str, float] = {}
        self._far_ridges: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
        self._off_prior: OffFieldPrior | None = None
        self._off_reach = np.zeros(0)
        self._off_floor_e: dict[str, float] = {}
        self._trail_params: TrailParams | None = (
            load_trail_params() if self.config.bright_star_trails else None)
        if self.config.bright_star_far_field or self.config.nisp_psf_taper:
            self._setup_far_field()
        if self.config.off_field_stars:
            self._setup_off_field()
        # Bright-star saturation model (per-band well depths precomputed once).
        self._sat_model = (StarSaturationModel()
                           if self.config.add_saturation else None)
        # Blank-pixel model (an unknown release is an error here, not mid-run).
        self._blank_model = (BlankPixelModel(self.config.blank_pixels)
                             if self.config.blank_pixels else None)
        # Sanity-check PSF pixel scales (every member of every set).
        for band_name, pset in self._psf_sets.items():
            for psf in pset.psfs:
                if abs(psf.pixel_scale - self.config.hr_pixel_scale) > 1e-4:
                    raise ValueError(
                        f"PSF for band {band_name} has pixel_scale={psf.pixel_scale}; "
                        f"forward model expects HR pixel scale {self.config.hr_pixel_scale}"
                    )

    def _setup_far_field(self) -> None:
        """Taper the kernels at their join annulus and price the far field.

        VIS (and NISP with ``nisp_psf_taper``) join at
        ``bright_star_join_arcsec``; NISP without it keeps its stamp and joins
        just inside the stamp's own edge taper. A band only changes when the
        far field is on, or when it is NISP and ``nisp_psf_taper`` is on.
        """
        cfg = self.config
        if cfg.bright_star_far_field:
            self._far_params = load_far_field_params()
        for band in Config.BANDS:
            nisp = band.name != Config.BAND_VIS.name
            taper_nisp = nisp and cfg.nisp_psf_taper
            if not (cfg.bright_star_far_field or taper_nisp):
                continue
            pset = self._psf_sets[band.name]
            half = 0.5 * min(pset.shape) * pset.pixel_scale
            requested = cfg.bright_star_join_arcsec
            if nisp and not taper_nisp:
                # Far field only: join where the stamp's own taper begins.
                edge = Config.PSF_TAPER_INNER_FRAC * half
                requested = (edge - (requested[1] - requested[0]), edge)
            r_in, r_out = join_radii(half, requested, Config.PSF_TAPER_INNER_FRAC)
            self._psf_sets[band.name] = pset.radially_tapered(r_in, r_out)
            self._far_join[band.name] = (r_in, r_out)
            if self._far_params is not None:
                self._far_flux[band.name] = far_field_flux(
                    self._far_params.bands[band.name], self._far_params, r_in, r_out)

    def _ridges_for(self, band: BandConfig, psf: PSF,
                    psf_spec: PSFSample | None) -> tuple[np.ndarray, np.ndarray]:
        """Spike ridge angles + heights of a scene kernel, cached per
        (band, cluster, roll); the elastic warp barely moves them."""
        key = (band.name,
               None if psf_spec is None else int(psf_spec.index),
               None if psf_spec is None else psf_spec.angle)
        hit = self._far_ridges.get(key)
        if hit is None:
            hit = spike_ridges(psf.data, psf.pixel_scale, self._n_spikes)
            self._far_ridges[key] = hit
        return hit

    @property
    def _n_spikes(self) -> int:
        return self._far_params.n_spikes if self._far_params is not None else 6

    def _vis_ridges(self, psf_spec: PSFSample | None) -> tuple[np.ndarray, np.ndarray]:
        """The scene's VIS spike ridges (unwarped kernel), cached."""
        band = Config.BAND_VIS
        key = (band.name, None if psf_spec is None else int(psf_spec.index),
               None if psf_spec is None else psf_spec.angle)
        hit = self._far_ridges.get(key)
        if hit is None:
            pset = self._psf_sets[band.name]
            if psf_spec is None:
                psf = pset.mean()
            else:
                psf = pset.psfs[min(int(psf_spec.index), pset.n - 1)]
                if psf_spec.angle is not None:
                    psf = psf.rotated(float(psf_spec.angle), order=3)
            hit = self._ridges_for(band, psf, psf_spec)
        return hit

    def _apply_trails(
        self,
        lr_stack: np.ndarray,
        rng: np.random.Generator,
        psf_spec: PSFSample | None,
        star_data: np.ndarray | None,
        off_field_stars: tuple[np.ndarray, ...] | None,
    ) -> None:
        """Zero the blank trails of the scene's bright stars in place."""
        assert self._trail_params is not None
        k_vis = Config.LR_INPUT_BAND_NAMES.index(Config.BAND_VIS.name)
        ys, xs, flux = [], [], []
        if star_data is not None:
            sy, sx = np.nonzero(star_data[..., k_vis])
            ys.append(sy); xs.append(sx); flux.append(star_data[sy, sx, k_vis])
        if off_field_stars is not None:
            ys.append(off_field_stars[0]); xs.append(off_field_stars[1])
            flux.append(off_field_stars[2][:, k_vis])
        if not ys:
            return
        hy, hx = np.concatenate(ys), np.concatenate(xs)
        f_vis = np.asarray(np.concatenate(flux), np.float64)
        keep = f_vis > 0.0
        mags = np.asarray(electrons_to_ab_mag(f_vis[keep], Config.BAND_VIS), np.float64).reshape(-1)
        limit = max(self._trail_params.vis_mag_max, self._trail_params.nisp_mag_max)
        if not np.any(mags < limit):
            return
        f = float(round(self.target_lr_pixel_scale_arcsec / self.config.hr_pixel_scale))
        angles, rel = self._vis_ridges(psf_spec)
        axis = trail_axis(angles, rel, self._trail_params.axis_offset_deg)
        mask = draw_trail_masks(
            rng, self._trail_params, lr_stack.shape[:2],
            (hy[keep] + 0.5) / f - 0.5, (hx[keep] + 0.5) / f - 0.5, mags, axis,
            band_names=Config.LR_INPUT_BAND_NAMES,
            pixel_scale_arcsec=self.target_lr_pixel_scale_arcsec,
        )
        lr_stack[mask] = 0.0

    def _setup_off_field(self) -> None:
        """Load the off-field prior and, per magnitude bin, how far beyond
        the field a star of the bin's bright edge can still reach in."""
        assert self._far_params is not None
        prior = load_off_field_prior()
        table = load_mer_noise_levels()
        sigma = {b: float(np.median(table.levels_e[:, i])) for i, b in enumerate(table.bands)}
        ratio = {b: float(np.percentile(prior.colour_ratios[:, j], 95))
                 for j, b in enumerate(prior.colour_bands)}
        ratio[Config.BAND_VIS.name] = 1.0
        frac = float(self.config.off_field_min_peak_sigma)
        reach = []
        for mag in prior.mag_lo:
            vis_e = float(ab_mag_to_electrons(float(mag), Config.BAND_VIS))
            reach.append(max(
                ridge_reach_arcsec(self._far_params.bands[b], self._far_params,
                                   vis_e * ratio[b], frac * sigma[b])
                for b in Config.LR_INPUT_BAND_NAMES))
        self._off_prior = prior
        self._off_reach = np.asarray(reach)
        # Off-field stars render only where they beat this (per band).
        self._off_floor_e = {b: frac * sigma[b] for b in Config.LR_INPUT_BAND_NAMES}

    def _draw_off_field(
        self, field_shape: tuple[int, int], rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        """One scene's off-field stars: integer HR positions (outside the
        field), four-band electrons ``(n, 4)`` and amplitude factors."""
        assert self._off_prior is not None and self._far_params is not None
        prior = self._off_prior
        ys, xs, mags, colour = draw_off_field_stars(
            rng, prior, self._off_reach, field_shape, self.config.hr_pixel_scale)
        if ys.size == 0:
            return None
        flux = np.empty((ys.size, len(Config.LR_INPUT_BAND_NAMES)))
        vis_e = np.asarray(ab_mag_to_electrons(mags, Config.BAND_VIS), np.float64)
        flux[:, Config.LR_INPUT_BAND_NAMES.index(Config.BAND_VIS.name)] = vis_e
        for j, name in enumerate(prior.colour_bands):
            flux[:, Config.LR_INPUT_BAND_NAMES.index(name)] = vis_e * prior.colour_ratios[colour, j]
        halo_jitter, spike_jitter = draw_jitter(int(ys.size), self._far_params, rng)
        return (np.round(ys).astype(np.int64), np.round(xs).astype(np.int64), flux,
                halo_jitter, spike_jitter)

    @staticmethod
    def _stamp_kernel_at(shape: tuple[int, int], ys: np.ndarray, xs: np.ndarray,
                         values: np.ndarray, psf: PSF) -> np.ndarray:
        """Stamp ``values * psf`` at integer HR positions that may lie
        outside ``shape``; only the overlapping part lands."""
        out = np.zeros(shape, np.float32)
        kernel = np.asarray(psf.data, np.float32)
        kh, kw = kernel.shape
        cy, cx = (kh - 1) // 2, (kw - 1) // 2
        height, width = shape
        for y, x, v in zip(ys, xs, values, strict=True):
            oy0, ox0 = max(0, y - cy), max(0, x - cx)
            oy1, ox1 = min(height, y - cy + kh), min(width, x - cx + kw)
            if oy1 <= oy0 or ox1 <= ox0:
                continue
            ky0, kx0 = oy0 - (y - cy), ox0 - (x - cx)
            out[oy0:oy1, ox0:ox1] += np.float32(v) * kernel[
                ky0:ky0 + (oy1 - oy0), kx0:kx0 + (ox1 - ox0)]
        return out

    def _far_field_lr(
        self,
        band: BandConfig,
        stars: tuple[np.ndarray, ...],
        psf: PSF,
        psf_spec: PSFSample | None,
        lr_shape: tuple[int, int],
        rebin_factor: int,
    ) -> np.ndarray:
        """Far-field light on this band's LR grid of stars at HR positions
        ``ys, xs`` with this band's electrons and amplitude factors."""
        assert self._far_params is not None
        ys, xs, fluxes, halo_jitter, spike_jitter, floor_e = stars
        angles, rel = self._ridges_for(band, psf, psf_spec)
        r_in, r_out = self._far_join[band.name]
        f = float(rebin_factor)
        return render_far_field(
            lr_shape,
            (ys + 0.5) / f - 0.5, (xs + 0.5) / f - 0.5,
            np.asarray(fluxes, np.float64),
            self._far_params.bands[band.name], self._far_params,
            pixel_scale_arcsec=self.config.hr_pixel_scale * f,
            angles=angles, rel=rel, r_in=r_in, r_out=r_out,
            halo_jitter=halo_jitter, spike_jitter=spike_jitter,
            min_peak_e=floor_e,
        )

    def warm_up(self) -> None:
        """Compile the elastic warp's numba kernel for these PSFs' dtype.

        A generation parent calls this once before forking its workers, which
        then inherit the compiled kernel instead of each compiling it on
        their first field. Pure: the warped kernel is discarded.
        """
        psf = next(iter(self._psf_sets.values())).psfs[0]
        psf.elastic_warp(1.0, 1.0, seed=0)

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

        Anchored to the VIS LR grid. Every band's archive scale is 0.10″, so
        the resample step of :meth:`_process_one_band` is skipped for all four
        bands; it acts only on a band whose LR scale is coarser.
        """
        return Config.BAND_VIS.pixel_scale_lr_arcsec

    def _draw_psf_sample(
        self,
        rng: np.random.Generator,
        *,
        allow_warp: bool = True,
    ) -> PSFSample:
        """Draw ONE :class:`PSFSample` (cluster index + roll) for the whole
        scene, from the band with the most cluster PSFs (the reference for the
        common clustering). Applied to every band so all four share the field
        position and the telescope roll — physically one pointing."""
        ref = max(self._psf_sets.values(), key=lambda p: p.n)
        return ref.draw_sample(
            rng, use_unrotated_prob=self.config.psf_unrotated_prob,
            warp_prob=(self.config.psf_warp_prob if allow_warp else 0.0),
            warp_alpha_max=(self.config.psf_warp_alpha_max
                            if allow_warp else 0.0),
            warp_sigma=self.config.psf_warp_sigma,
        )

    def _psf_for_sample(
        self,
        band: BandConfig,
        psf_spec: PSFSample | None,
        warp_displacements: dict[
            tuple[int, int], tuple[np.ndarray, np.ndarray]
        ] | None,
    ) -> PSF:
        """Realise one shared scene/star PSF sample for ``band``."""
        pset = self._psf_sets[band.name]
        if psf_spec is None:
            return pset.mean()
        displacement = None
        if (warp_displacements is not None
                and psf_spec.warp_seed is not None
                and psf_spec.warp_alpha > 0.0):
            shape = (int(pset.shape[0]), int(pset.shape[1]))
            displacement = warp_displacements.get(shape)
            if displacement is None:
                displacement = PSF.elastic_displacement(
                    shape,
                    psf_spec.warp_alpha,
                    psf_spec.warp_sigma,
                    seed=int(psf_spec.warp_seed),
                )
                warp_displacements[shape] = displacement
        return pset.apply_sample(
            psf_spec, warp_displacement=displacement,
        )

    @staticmethod
    def _convolve_sparse_deltas(image: np.ndarray, psf: PSF) -> np.ndarray:
        """Convolve a sparse HR star plane by stamping ``psf`` at its deltas.

        Star injection deposits one value at each integer HR coordinate.  A
        direct sparse stamp is exactly the same linear convolution as
        ``fftconvolve(..., mode="same")`` for that representation, without a
        second full-field FFT merely to render a handful of stars.
        """
        source = np.asarray(image, np.float32)
        out = np.zeros_like(source)
        ys, xs = np.nonzero(source)
        if ys.size == 0:
            return out
        kernel = np.asarray(psf.data, np.float32)
        kh, kw = kernel.shape
        cy, cx = (kh - 1) // 2, (kw - 1) // 2
        height, width = source.shape
        for y, x in zip(ys, xs, strict=False):
            oy0, ox0 = max(0, y - cy), max(0, x - cx)
            oy1 = min(height, y - cy + kh)
            ox1 = min(width, x - cx + kw)
            ky0, kx0 = oy0 - (y - cy), ox0 - (x - cx)
            ky1, kx1 = ky0 + (oy1 - oy0), kx0 + (ox1 - ox0)
            out[oy0:oy1, ox0:ox1] += (
                source[y, x] * kernel[ky0:ky1, kx0:kx1]
            )
        return out

    def _process_one_band(
        self,
        hr_channel: np.ndarray,
        band: BandConfig,
        rng: np.random.Generator,
        psf_spec: PSFSample | None = None,
        star_hr_channel: np.ndarray | None = None,
        star_psf_spec: PSFSample | None = None,
        distant_star_wings: tuple[DistantStarWing, ...] = (),
        noise_scale_map: np.ndarray | None = None,
        sky_rms_e: float | None = None,
        warp_displacements: dict[
            tuple[int, int], tuple[np.ndarray, np.ndarray]
        ] | None = None,
        far_field_stars: tuple[
            np.ndarray, np.ndarray, np.ndarray, np.ndarray
        ] | None = None,
        off_field_stars: tuple[
            np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
        ] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Return final dirty LR, its pre-noise optical trigger plane, and the
        star plane's share of that trigger (``None`` without stars)."""
        # The ordinary scene and sparse star plane share the same sampled,
        # optionally elastically deformed observation PSF.
        psf = self._psf_for_sample(band, psf_spec, warp_displacements)
        target_scale = self.target_lr_pixel_scale_arcsec
        resample_kernel = cast(
            _ResampleKernel, self.config.nisp_resample_kernel,
        )

        # 1. PSF convolution on HR plane via PSF.convolved_with (sum=1-normalises
        #    the kernel and runs fftconvolve mode="same" + float32 cast).
        hr_e = psf.convolved_with(hr_channel)
        far_share = self._far_flux.get(band.name, 0.0)
        if far_share > 0.0:
            # The share of every source's light that lands in its far field.
            hr_e *= np.float32(1.0 - far_share)
        star_e = None
        star_psf = psf
        if star_hr_channel is not None and np.any(star_hr_channel):
            # process() gives the star plane the scene's own sample: reuse
            # its realised (warped) kernel rather than warping it again.
            star_psf = (psf if star_psf_spec is psf_spec else self._psf_for_sample(
                band, star_psf_spec, warp_displacements,
            ))
            if far_share > 0.0:
                # Kernel + far field together carry the star's flux.
                star_e = self._convolve_sparse_deltas(
                    star_hr_channel * np.float32(1.0 - far_share), star_psf)
            else:
                star_e = self._convolve_sparse_deltas(star_hr_channel, star_psf)
            hr_e += star_e
        band_k = Config.LR_INPUT_BAND_NAMES.index(band.name)
        if off_field_stars is not None:
            # Off-field stars: only the part of their kernel that overlaps
            # the field lands (their far field follows on the LR grid).
            oys, oxs, oflux = off_field_stars[:3]
            spill = self._stamp_kernel_at(
                hr_e.shape, oys, oxs, oflux[:, band_k] * (1.0 - far_share), star_psf)
            hr_e += spill
            star_e = spill if star_e is None else star_e + spill

        # 2. Sum-rebin to the band's LR scale (preserves photon shot noise
        #    statistics at the right pixel size for the per-band noise step).
        rebin_factor = int(round(band.pixel_scale_lr_arcsec / self.config.hr_pixel_scale))
        lr_signal_e = self.sum_rebin(hr_e, rebin_factor)
        star_signal_e = (self.sum_rebin(star_e, rebin_factor)
                         if star_e is not None else None)
        far_sets = []
        if (far_field_stars is not None and star_hr_channel is not None
                and np.any(star_hr_channel)):
            ys, xs, hj, sj = far_field_stars
            far_sets.append((ys, xs, star_hr_channel[ys, xs], hj, sj,
                             np.full(ys.size, self.config.bright_star_min_peak_e)))
        if off_field_stars is not None:
            oys, oxs, oflux, ohj, osj = off_field_stars
            far_sets.append((oys, oxs, oflux[:, band_k], ohj, osj,
                             np.full(oys.size, self._off_floor_e[band.name])))
        if far_sets and star_signal_e is not None and band.name in self._far_flux:
            # Far-field halo + spike ridges, before noise: photon noise and
            # the saturation trigger see them like any starlight.
            stars = tuple(np.concatenate([fs[i] for fs in far_sets]) for i in range(6))
            far_e = self._far_field_lr(
                band, stars, star_psf, star_psf_spec, lr_signal_e.shape, rebin_factor,
            )
            lr_signal_e = lr_signal_e + far_e
            star_signal_e = star_signal_e + far_e

        # 3. Apply delivered-MER noise: Euclid's noise level with the pixel
        #    correlation of a dithered, bilinearly resampled, median-combined
        #    stack, identical for all bands.
        if self.config.add_noise:
            texture = (
                NoiseTextureJitter(
                    exposures=tuple(self.config.noise_texture_exposures),
                    white_max=float(self.config.noise_texture_white_max),
                )
                if self.config.noise_texture_exposures is not None else None
            )
            lr_e = apply_archive_noise(
                lr_signal_e, band, rng,
                add_artifacts=self.config.add_artifacts,
                artifact_config=self.config.artifact_config,
                noise_scale_map=noise_scale_map,
                sky_rms_e=sky_rms_e,
                noise_texture=texture,
            )
        else:
            lr_e = lr_signal_e.astype(np.float32, copy=False)

        # 4. Resample to the shared LR grid; VIS factor is 1.
        upsample_factor = int(round(band.pixel_scale_lr_arcsec / target_scale))
        if upsample_factor > 1:
            lr_e = resample_upsample(
                lr_e, factor=upsample_factor, kernel=resample_kernel,
            )
            lr_signal_e = resample_upsample(
                lr_signal_e,
                factor=upsample_factor,
                kernel=resample_kernel,
            )
            if star_signal_e is not None:
                star_signal_e = resample_upsample(
                    star_signal_e,
                    factor=upsample_factor,
                    kernel=resample_kernel,
                )
        if distant_star_wings:
            residual = lr_e - lr_signal_e
            residual_median = float(np.median(residual))
            local_sigma_e = 1.4826 * float(np.median(
                np.abs(residual - residual_median)
            ))
            lr_e = add_distant_star_wings(
                lr_e,
                distant_star_wings,
                local_sigma_e=local_sigma_e,
            )
        return (
            lr_e.astype(np.float32, copy=False),
            lr_signal_e.astype(np.float32, copy=False),
            (star_signal_e.astype(np.float32, copy=False)
             if star_signal_e is not None else None),
        )

    # ------------------------------------------------------------------ #
    def apply_saturation(
        self,
        lr_stack: np.ndarray,
        trigger_stack: np.ndarray,
        rng: np.random.Generator,
        star_trigger_stack: np.ndarray | None = None,
    ) -> None:
        """Black out saturated sources of ``lr_stack`` in place.

        ``trigger_stack`` is the pre-noise optical signal that decides which
        sources saturate; ``star_trigger_stack`` is the star plane's share of
        it, which lets galaxy light saturate at the extended well (see
        :func:`apply_saturation_masking`). Uses this simulator's well depths
        and blackout probabilities, and does nothing when saturation is off.
        """
        if self._sat_model is None:
            return
        apply_saturation_masking(
            lr_stack, self._sat_model, rng,
            band_names=Config.LR_INPUT_BAND_NAMES,
            trigger_4ch=trigger_stack,
            star_trigger_4ch=star_trigger_stack,
            mask_probability=self.config.saturation_mask_prob,
            bright_mask_probability=self.config.saturation_mask_prob_bright,
            bright_well_ratios=self.config.saturation_mask_ramp_well_ratios,
        )

    def apply_blank_pixels(
        self,
        lr_stack: np.ndarray,
        rng: np.random.Generator,
        sky_levels: dict[str, float] | None = None,
        trigger_stack: np.ndarray | None = None,
        star_trigger_stack: np.ndarray | None = None,
    ) -> None:
        """Zero the pixels of ``lr_stack`` that no exposure measured.

        With ``add_saturation`` and the scene's pre-noise ``trigger_stack``
        (and the stars' share of it) the cores of bright stars are among
        them, in all four bands; the other blanks fall on NISP only.

        Fewer exposures leave more blanks, so the scene's exposure count
        follows its depth: where its H sky noise level (``sky_levels``) sits
        among the measured sky positions, from the shallowest (0) to the
        deepest (1). Without a level the depth is drawn at random. Does
        nothing unless ``config.blank_pixels`` names a release. The draws
        come last from ``rng``, so the rest of the field is the same with
        and without blanks.
        """
        if self._blank_model is None:
            return
        level = (sky_levels or {}).get("H_E")
        if level is None:
            depth = float(rng.random())
        else:
            table = load_mer_noise_levels()
            measured = table.levels_e[:, table.bands.index("H_E")]
            depth = float((np.sum(measured > level) + rng.random()) / (len(measured) + 1))
        cores = self._sat_model is not None and trigger_stack is not None
        blank = self._blank_model.draw(
            lr_stack.shape[:2], depth, rng,
            trigger=trigger_stack if cores else None,
            star_trigger=star_trigger_stack if cores else None,
        )
        for k, name in enumerate(BLANK_MASK_BAND_NAMES):
            lr_stack[..., Config.LR_INPUT_BAND_NAMES.index(name)][blank[..., k]] = 0.0

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
        rng: np.random.Generator | None = None,
        *,
        star_hr_4ch: np.ndarray | None = None,
    ) -> tuple[Image, Image]:
        """Run the full forward model on one HR clean 4-channel field.

        Parameters
        ----------
        hr_4ch : :class:`Image` with shape (H, W, 4),
                 ``pixel_scale_arcsec == hr_pixel_scale``, band order
                 ``Config.LR_INPUT_BAND_NAMES``.
        rng    : reproducible noise source; ``np.random.default_rng()`` if None.
        star_hr_4ch : optional sparse star-only HR plane. It is forwarded
                 separately so the caller can choose a starless or starfull
                 target and so saturation can tell star cores from galaxy
                 light; it shares the scene's observation PSF. The
                 returned HR target contains ``hr_4ch + star_hr_4ch``.

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
        star_data = None
        if star_hr_4ch is not None:
            star_data = np.asarray(star_hr_4ch, np.float32)
            if star_data.shape != hr_4ch.data.shape:
                raise ValueError(
                    f"star_hr_4ch shape {star_data.shape} must match HR scene "
                    f"shape {hr_4ch.data.shape}"
                )

        # HR canvas must be divisible by every archive-grid rebin factor so all
        # four delivered channels have the same shape. The NISP detector-noise
        # helper pads its private 0.30" workspace when this archive shape is
        # not divisible by 3, then crops back without changing public shapes.
        max_rebin = max(
            int(round(b.pixel_scale_lr_arcsec / self.config.hr_pixel_scale))
            for b in Config.BANDS
        )
        H_full, W_full = hr_4ch.data.shape[:2]
        H_trim = (H_full // max_rebin) * max_rebin
        W_trim = (W_full // max_rebin) * max_rebin
        hr_data_trim = hr_4ch.data[:H_trim, :W_trim, :]
        star_data_trim = (
            star_data[:H_trim, :W_trim, :] if star_data is not None else None
        )
        target_rebin = int(round(
            self.target_lr_pixel_scale_arcsec / self.config.hr_pixel_scale
        ))
        target_lr_shape = (H_trim // target_rebin, W_trim // target_rebin)

        distant_star_wings: tuple[DistantStarWing, ...] = ()
        if (
            self.config.add_noise
            and self.config.add_artifacts
            and self.config.add_distant_star_wings
            and not self.config.off_field_stars
        ):
            distant_star_wings = draw_distant_star_wings(
                target_lr_shape,
                rng,
                probability=self.config.distant_star_wing_probability,
                amplitude_sigma_min=(
                    self.config.distant_star_wing_amplitude_sigma_min
                ),
                amplitude_sigma_max=(
                    self.config.distant_star_wing_amplitude_sigma_max
                ),
                width_min_lr_pix=(
                    self.config.distant_star_wing_width_min_lr_pix
                ),
                width_max_lr_pix=(
                    self.config.distant_star_wing_width_max_lr_pix
                ),
                fade_length_min_lr_pix=(
                    self.config.distant_star_wing_fade_length_min_lr_pix
                ),
                fade_length_max_lr_pix=(
                    self.config.distant_star_wing_fade_length_max_lr_pix
                ),
            )

        # One field depth for the whole scene — the bands' depths already move
        # together through the drawn level row — but a seam per band: real VIS
        # and NISP mosaics do not share their pointing boundaries.
        noise_scale_maps: dict[str, np.ndarray] = {}
        if self.config.add_noise and self.config.add_noise_variation:
            field_scale = float(rng.uniform(
                self.config.noise_global_scale_min,
                self.config.noise_global_scale_max,
            ))
            for band_name in Config.LR_INPUT_BAND_NAMES:
                noise_scale_maps[band_name] = draw_noise_scale_map(
                    target_lr_shape,
                    rng,
                    global_scale=field_scale,
                    global_scale_min=self.config.noise_global_scale_min,
                    global_scale_max=self.config.noise_global_scale_max,
                    region_probability=self.config.noise_region_probability,
                    region_fraction_min=self.config.noise_region_fraction_min,
                    region_fraction_max=self.config.noise_region_fraction_max,
                    region_step_min=self.config.noise_region_step_min,
                    region_step_max=self.config.noise_region_step_max,
                )
        # One real Q1 sky position per scene sets all four bands' sky noise.
        sky_levels: dict[str, float] = {}
        if self.config.add_noise and self.config.draw_mer_noise_levels:
            sky_levels = load_mer_noise_levels().draw(rng)

        # Draw ONE PSF sample (cluster index + roll) for the whole scene so all
        # four bands share the field position and the telescope roll — one
        # physical pointing. ``None`` (randomisation off) → each band's mean.
        psf_spec = (
            self._draw_psf_sample(rng)
            if self.config.randomize_psf else None
        )
        # The star plane is separate for target semantics, not a second
        # optical exposure: every source shares the same deformed PSF.
        star_psf_spec = psf_spec
        # Far field: every star's position (one HR delta in all bands) and
        # its halo/spike amplitude factors, shared by the four bands.
        far_field_stars = None
        if self._far_params is not None and star_data_trim is not None:
            ys, xs = np.nonzero(np.any(star_data_trim != 0.0, axis=-1))
            if ys.size:
                halo_jitter, spike_jitter = draw_jitter(
                    int(ys.size), self._far_params, rng)
                far_field_stars = (ys, xs, halo_jitter, spike_jitter)
        off_field_stars = (self._draw_off_field((H_trim, W_trim), rng)
                           if self._off_prior is not None else None)

        # Process each channel; the four LR channels are stacked at the end.
        lr_channels = []
        saturation_trigger_channels = []
        star_trigger_channels = []
        # Same-shaped band kernels share one displacement field for this
        # physical exposure. Coordinate generation is the expensive part of
        # an elastic warp; reuse keeps live augmentation cheap without
        # coupling separate exposures or retaining a global cache.
        warp_displacements: dict[
            tuple[int, int], tuple[np.ndarray, np.ndarray]
        ] = {}
        for k, band_name in enumerate(Config.LR_INPUT_BAND_NAMES):
            band = Config.get_band(band_name)
            lr_channel, trigger_channel, star_channel = self._process_one_band(
                hr_data_trim[..., k], band, rng, psf_spec=psf_spec,
                star_hr_channel=(star_data_trim[..., k]
                                 if star_data_trim is not None else None),
                star_psf_spec=star_psf_spec,
                distant_star_wings=distant_star_wings,
                noise_scale_map=noise_scale_maps.get(band_name),
                sky_rms_e=sky_levels.get(band.name),
                warp_displacements=warp_displacements,
                far_field_stars=far_field_stars,
                off_field_stars=off_field_stars,
            )
            lr_channels.append(lr_channel)
            saturation_trigger_channels.append(trigger_channel)
            star_trigger_channels.append(
                star_channel if star_channel is not None
                else np.zeros_like(trigger_channel))
        # All channels must end on the same grid (the VIS LR grid).
        target_shape = lr_channels[0].shape
        for k, ch in enumerate(lr_channels):
            if ch.shape != target_shape:
                raise RuntimeError(
                    f"Channel {k} has shape {ch.shape}; expected {target_shape}. "
                    "Check rebin / resample factors."
                )
        lr_stack = np.stack(lr_channels, axis=-1)
        saturation_trigger_stack = np.stack(
            saturation_trigger_channels, axis=-1,
        )
        # With a star plane the masking can tell star cores from galaxy light.
        star_trigger_stack = (
            np.stack(star_trigger_channels, axis=-1)
            if star_data_trim is not None or off_field_stars is not None else None
        )

        # Detector saturation masking, mirroring the MER pipeline — NOT
        # clipped to the well. Star cores past the band well are masked to ~0
        # over a blocky rectangular patch; galaxy light only loses its pixels
        # past the extended well. Per band; the clean HR target is untouched.
        if self._blank_model is None:
            self.apply_saturation(lr_stack, saturation_trigger_stack, rng,
                                  star_trigger_stack)
        else:
            # Pixels every exposure flagged invalid are blank in the mosaic:
            # bright cores as Euclid loses them (round, brightness-dependent)
            # instead of the rectangular blackout, plus the sky's own blanks.
            self.apply_blank_pixels(lr_stack, rng, sky_levels,
                                    saturation_trigger_stack, star_trigger_stack)
            if self._trail_params is not None:
                self._apply_trails(lr_stack, rng, psf_spec, star_data_trim,
                                   off_field_stars)

        # HR target: all four bands (clean, no noise applied), trimmed to the
        # same spatial extent the LR pipeline saw. Band k of the target is
        # band k of the LR input — the model super-resolves VIS+NISP jointly.
        hr_clean = hr_data_trim.astype(np.float32, copy=True)
        if star_data_trim is not None:
            hr_clean += star_data_trim

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
