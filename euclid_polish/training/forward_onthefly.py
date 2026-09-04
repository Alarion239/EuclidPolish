"""On-the-fly forward model for training: clean field → dirty LR, per visit.

Instead of training on the noise/PSF realization baked into the dirty
TFRecords, each visit of a clean field runs the FULL generation forward model
live — PSF draw (from the member's bagged pre-rotated pool), per-band
convolution, 2× sum-rebin, Poisson + read noise, cosmic-ray/hot-pixel
artifacts and bright-star saturation — then cuts ``crops_per_field`` aligned
LR/HR training crops from the one forward-modelled field.

Design points (why full-field, why multiple crops):

* The forward model runs on the WHOLE field, never on a padded crop — a
  bright star OUTSIDE the crop scatters PSF-wing flux INTO it, which the
  FASRC benchmark measured at up to ~50–84% of the read noise for truncated
  crop-local convolution. Full-field pays the wings exactly.
* One full-field forward (~350 ms CPU) is amortised over K crops. The default
  K=8 returns eight random 256² HR crops from each 510² field; the downstream
  shuffle spreads those siblings across batches.
  Crops from one field share that visit's PSF draw + noise realization
  (they are one exposure — physically consistent), while every EPOCH
  re-draws both; the pipeline's crop-level shuffle spreads the siblings
  across batches.

PSFs come from :func:`member_psf_sets`: the member's seeded random subset of
the pre-rotated kernel pools (PSF bagging — each ensemble member trains
against its own instrument-response sub-population), falling back to the
plain unrotated cluster sets when no pool has been built.

Thread-safety: tf.data invokes the numpy hook from several threads;
per-call RNGs are spawned from one lock-guarded ``SeedSequence`` so draws
are never shared and the heavy work runs lock-free.
"""

from __future__ import annotations

import threading

import numpy as np

from euclid_polish.config import Config
from euclid_polish.image import Image
from euclid_polish.psf.psf_library import load_all_band_psf_sets
from euclid_polish.psf.psf_set import PSFSet
from euclid_polish.psf.rotpool import load_all_band_rotpools
from euclid_polish.sky.generation.sky_simulator import inject_random_stars
from euclid_polish.sky.generation.stellar_sed import EmpiricalStellarPrior
from euclid_polish.sky.observation.noise_calibration import VISNoiseCalibration
from euclid_polish.sky.observation.observation_simulator import (
    ObservationSimulator,
    ObservationSimulatorConfig,
)
from euclid_polish.training.target_blur import (
    blur_target_array,
    validate_target_fwhm_arcsec,
)

#: Default PSF-bagging subset (source clusters per member) when a rotation
#: pool is available. Bounded so a member never loads the whole multi-GB
#: pool; ~64 clusters × (rolls+1) kernels is plenty of diversity.
DEFAULT_PSF_SUBSET = 64

#: Training defaults for the on-the-fly forward. A 256px HR crop is the nearest
#: scale-aligned half-side crop of a 510px field (255 is not divisible by 2).
#: Eight random positions reuse each expensive field forward while the crop
#: shuffle still mixes exposures before batching.
DEFAULT_CROPS_PER_FIELD = 8
DEFAULT_ONTHEFLY_HR_CROP_SIZE = 256


def member_psf_sets(*, seed: int | None, psf_subset: int | None = None,
                    psf_dir: str = Config.EUCLID_PSF_DIR,
                    ) -> tuple[dict[str, PSFSet], str]:
    """The PSF sets a member's on-the-fly forward should draw from.

    Prefers the pre-rotated pools with a seeded cluster-subset bag
    (``psf_subset`` clusters, default :data:`DEFAULT_PSF_SUBSET`; the seed is
    the member's training seed, so the bag is reproducible and differs across
    members). Falls back to the plain cluster ePSF sets (unrotated — exactly
    what generation uses) when no pool is on disk. Returns ``(sets, note)``
    where ``note`` describes the choice for the training log.
    """
    subset = int(psf_subset) if psf_subset else DEFAULT_PSF_SUBSET
    pools = load_all_band_rotpools(psf_dir=psf_dir, subset_clusters=subset,
                                   subset_seed=seed)
    if pools is not None:
        n = pools[Config.BAND_VIS.name].n
        return pools, (f"pre-rotated pool, bagged to {subset} clusters "
                       f"({n} kernels/band, seed={seed})")
    sets = load_all_band_psf_sets(psf_dir=psf_dir,
                                  target_pixel_scale=Config.DEFAULT_PIXEL_SCALE)
    return sets, ("no rotation pool on disk — full unrotated cluster sets "
                  "(build one with scripts/pregenerate_psf_rotations.py "
                  "for roll diversity + PSF bagging)")


class OnTheFlyForward:
    """Callable forward-model hook for the training input pipeline.

    ``crops(field)`` forward-models one clean HR field (``(H, W, 4)``
    electrons, 0.05″ grid) and returns ``crops_per_field`` aligned crops:
    ``(lr_crops, hr_crops)`` with shapes ``(K, c/2, c/2, 4)`` /
    ``(K, c, c, 4)`` where ``c`` is the HR crop size — raw electrons, the
    caller applies the asinh stretch (and any further augmentation).
    """

    def __init__(
        self,
        psf_sets: dict[str, PSFSet],
        *,
        seed: int | None = None,
        crops_per_field: int = DEFAULT_CROPS_PER_FIELD,
        hr_crop_size: int = DEFAULT_ONTHEFLY_HR_CROP_SIZE,
        scale: int = Config.DEFAULT_REBIN_FACTOR,
        add_noise: bool = True,
        add_artifacts: bool = True,
        add_saturation: bool = True,
        starless: bool = True,
        inject_stars: bool = True,
        star_density_arcmin2: float = Config.DEFAULT_STAR_DENSITY_ARCMIN2,
        star_prior_payload: dict | None = None,
        vis_noise_calibration_payload: dict | None = None,
        pixel_scale_arcsec: float = Config.DEFAULT_PIXEL_SCALE,
        psf_warp_prob: float = Config.TRAIN_PSF_WARP_PROB,
        psf_warp_alpha_max: float = Config.TRAIN_PSF_WARP_ALPHA_MAX,
        psf_warp_sigma: float = Config.TRAIN_PSF_WARP_SIGMA,
        saturation_mask_prob: float = Config.TRAIN_SATURATION_MASK_PROB,
        target_fwhm_arcsec: float = Config.TARGET_PSF_FWHM_ARCSEC,
    ) -> None:
        self.crops_per_field = int(crops_per_field)
        self.hr_crop_size = int(hr_crop_size)
        self.scale = int(scale)
        # Stars-as-artifacts: a FRESH star realization is drawn and deposited
        # (HR deltas, before the PSF) on every visit in BOTH regimes. ``starless``
        # picks the TARGET — the starless scene (erase the injected stars) vs the
        # with-stars scene (reconstruct them). Density mirrors generation; the
        # activated stellar prior owns the magnitude law and colour locus.
        self.starless = bool(starless)
        self.inject_stars = bool(inject_stars)
        self.star_density_arcmin2 = float(star_density_arcmin2)
        self.stellar_prior = (
            EmpiricalStellarPrior.from_payload(star_prior_payload)
            if star_prior_payload else None
        )
        vis_noise_calibration = (
            VISNoiseCalibration.from_payload(vis_noise_calibration_payload)
            if vis_noise_calibration_payload else None
        )
        if (
            self.inject_stars
            and self.star_density_arcmin2 > 0.0
            and self.stellar_prior is None
        ):
            raise ValueError(
                "star injection requires an active empirical stellar prior"
            )
        self.pixel_scale_arcsec = float(pixel_scale_arcsec)
        self.target_fwhm_arcsec = validate_target_fwhm_arcsec(
            target_fwhm_arcsec)
        if self.crops_per_field < 1:
            raise ValueError("crops_per_field must be >= 1")
        if self.hr_crop_size < self.scale:
            raise ValueError("hr_crop_size must be positive and >= scale")
        if self.hr_crop_size % self.scale:
            raise ValueError("hr_crop_size must be divisible by the scale")
        if not 0.0 <= float(saturation_mask_prob) <= \
                Config.TRAIN_SATURATION_MASK_PROB_MAX:
            raise ValueError(
                "saturation_mask_prob must be in [0, "
                f"{Config.TRAIN_SATURATION_MASK_PROB_MAX:g}] for training"
            )
        self._sim = ObservationSimulator(
            psf_sets_by_band=psf_sets,
            config=ObservationSimulatorConfig(
                add_noise=add_noise, add_artifacts=add_artifacts,
                add_saturation=add_saturation,
                # Pool members are PRE-rotated: pick one at random per scene
                # (star-count weighted), never rotate at train time.
                randomize_psf=True, psf_unrotated_prob=1.0,
                # Learn a PSF distribution (polish-pub-style elastic warp).
                # Training gets a fresh draw per visit; generated validation
                # and test records use their own fixed, seeded draws.
                psf_warp_prob=float(psf_warp_prob),
                psf_warp_alpha_max=float(psf_warp_alpha_max),
                psf_warp_sigma=float(psf_warp_sigma),
                saturation_mask_prob=float(saturation_mask_prob),
                vis_noise_calibration=vis_noise_calibration,
            ))
        # tf.data calls from several threads → per-call child RNGs, spawned
        # under a lock (SeedSequence.spawn is not thread-safe), used lock-free.
        self._seq = np.random.SeedSequence(seed)
        self._seq_lock = threading.Lock()

    def _rng(self) -> np.random.Generator:
        with self._seq_lock:
            child = self._seq.spawn(1)[0]
        return np.random.default_rng(child)

    def _inject_stars(self, canvas: np.ndarray, rng: np.random.Generator) -> None:
        """Deposit a fresh random star realization onto ``canvas`` in place."""
        if not self.inject_stars or self.star_density_arcmin2 <= 0.0:
            return
        side_arcmin = canvas.shape[0] * self.pixel_scale_arcsec / 60.0
        n_stars = int(rng.poisson(self.star_density_arcmin2 * side_arcmin ** 2))
        assert self.stellar_prior is not None
        inject_random_stars(
            canvas, rng, n_stars=n_stars,
            stellar_prior=self.stellar_prior)

    def crops(self, field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        field = np.asarray(field, np.float32)
        if field.ndim != 3 or field.shape[-1] != len(Config.LR_INPUT_BAND_NAMES):
            raise ValueError(f"expected (H, W, 4) clean field, got {field.shape}")
        c, s = self.hr_crop_size, self.scale
        if field.shape[0] < c or field.shape[1] < c:
            raise ValueError(
                f"field {field.shape[:2]} smaller than the HR crop {c}")
        rng = self._rng()

        # Keep stars on a separate sparse plane for target bookkeeping. They
        # share the scene's sampled, elastically deformed observation PSF. The
        # TARGET is what differs: starless → the ORIGINAL field; starfull →
        # field + star deltas.
        stars = np.zeros_like(field)
        self._inject_stars(stars, rng)
        hr_img = Image(data=field,
                       pixel_scale_arcsec=Config.DEFAULT_PIXEL_SCALE,
                       band_names=Config.LR_INPUT_BAND_NAMES, is_clean=True)
        lr_img, hr_out = self._sim.process(
            hr_img, rng, star_hr_4ch=stars,
        )
        lr = np.asarray(lr_img.data, np.float32)      # (H/s, W/s, 4) w/ stars
        # Target, trimmed exactly as process trimmed its HR output so LR/HR stay
        # block-aligned: starless = the original field, starfull = process's
        # (with-stars) HR.
        ht, wt = hr_out.data.shape[:2]
        target = (field[:ht, :wt, :] if self.starless
                  else np.asarray(hr_out.data, np.float32))
        # Blur the complete target before selecting crops so the boundary of
        # an individual crop never becomes an artificial PSF edge.
        hr = blur_target_array(
            target, self.target_fwhm_arcsec,
            pixel_scale_arcsec=self.pixel_scale_arcsec,
        )

        lr_crops = np.empty((self.crops_per_field, c // s, c // s, lr.shape[-1]),
                            np.float32)
        hr_crops = np.empty((self.crops_per_field, c, c, hr.shape[-1]),
                            np.float32)
        max_x = (hr.shape[0] - c) // s * s
        max_y = (hr.shape[1] - c) // s * s
        for k in range(self.crops_per_field):
            # Block-aligned offsets — same alignment law as the record-mode
            # crop (_augment_multiband), so LR pixel (i,j) ↔ HR block.
            x = int(rng.integers(0, max_x + 1)) // s * s
            y = int(rng.integers(0, max_y + 1)) // s * s
            hr_crops[k] = hr[x: x + c, y: y + c, :]
            lr_crops[k] = lr[x // s: (x + c) // s, y // s: (y + c) // s, :]
        return lr_crops, hr_crops
