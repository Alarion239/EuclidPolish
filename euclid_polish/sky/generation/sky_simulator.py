"""
Multi-band clean-HR scene generator.

Every field galaxy uses a resolved TNG50 SKIRT morphology. The active Euclid
prior supplies a VIS Sérsic half-light radius, a jointly conditioned VIS 2FWHM
aperture brightness, and a PHZ redshift conditioned on that brightness. The
observed radius stays fixed while the TNG band proportions receive the existing
redshift photometry; one final shared scale sets the exact VIS aperture target.
The same brightness draw supplies the MER photometric FWHM that defines the
aperture radius and target PSF used by the normalization.

The output of :meth:`SkySimulator.simulate_field` is a single :class:`Image`
with ``data`` of shape ``(H, W, 4)`` in **raw electrons** on the 0.05″ HR
grid, one channel per band ordered as :attr:`Config.LR_INPUT_BAND_NAMES`
(``VIS, Y_E, J_E, H_E``).
"""

from __future__ import annotations

import hashlib
import math
import os
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import cast

import numpy as np

from euclid_polish.config import Config
from euclid_polish.image import Image, Role
from euclid_polish.photometry import ab_mag_to_electrons
from euclid_polish.provenance.defaults import mint_id
from euclid_polish.provenance.records import Stamp
from euclid_polish.psf.psf_library import make_gaussian_psf
from euclid_polish.psf.psf_set import PSFSet
from euclid_polish.sky.generation.compositing import composite_stamp
from euclid_polish.sky.generation.cosmos_tng_prior import (
    MORPHOLOGY_ACTIVITY_THRESHOLD_LOGSSFR,
    MORPHOLOGY_BALANCE_POWER,
    MORPHOLOGY_MIN_EFFECTIVE_DONORS,
    CosmosTngPrior,
    JointGalaxyPopulationPrior,
    conditional_mass_quantiles,
    conditional_ssfr_quantiles,
    cross_validated_mass_bandwidth,
    joint_quantile_transport_weights,
)
from euclid_polish.sky.generation.lens_population import (
    render_lens_to_multiband_canvas,
    sample_lens_geometry,
)
from euclid_polish.sky.generation.stellar_sed import (
    EmpiricalStellarPrior,
    sample_stellar_sed,
)
from euclid_polish.tng import TNGAtlas, TNGGalaxy, TNGRenderer
from euclid_polish.tng.redshift import (
    compactness_factor,
    physical_pc_to_arcsec,
    sigma_v_from_stellar_mass,
)
from euclid_polish.tng.renderer import (
    TNG_RADIUS_RENDERER_FINGERPRINT,
    TNG_RADIUS_RENDERING,
)
from euclid_polish.tng.types import N_ORIENTATIONS, TNG_NATIVE_PC_PER_PIXEL


class _NoRenderableTNGDonorError(ValueError):
    """A sampled field-galaxy geometry exceeds the available donor support."""


_MER_APERTURE_GAUSSIAN_SIGMA_SUPPORT = 5.0
_OFF_FIELD_GALAXY_RE_SUPPORT = 4.0
_OFF_FIELD_GALAXY_SEED_TAG = 0x4F464647  # ``OFFG``
_GALAXY_REDSHIFT_SEED_TAG = 0x5245445A  # ``REDZ``


def _derived_rng_from_state(
    rng: np.random.Generator, tag: int,
) -> np.random.Generator:
    """Derive a deterministic stream without consuming the source stream."""
    digest = hashlib.sha256(repr(rng.bit_generator.state).encode()).digest()
    words = np.frombuffer(digest, dtype="<u4").astype(np.uint64)
    return np.random.default_rng(np.random.SeedSequence([
        *(int(word) for word in words), int(tag),
    ]))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SkySimulatorConfig:
    """Field-level config for the multi-band simulator.

    ``galaxy_density_arcmin2`` controls a single Euclid-conditioned TNG
    population. There is no analytic Sérsic field-galaxy branch.
    """
    image_size:               int   = Config.DEFAULT_IMAGE_SIZE
    pixel_scale:              float = Config.DEFAULT_PIXEL_SCALE     # arcsec/pix
    galaxy_density_arcmin2:   float = Config.GALAXY_DENSITY_ARCMIN2
    # Galaxy centres outside the saved field are proposed out to half the
    # 32-block WDSR receptive field (69 LR pixels = 3.4 arcsec = 68 HR
    # pixels).  A source-specific 4 R_e reach test rejects irrelevant
    # proposals before donor selection or TNG rendering.
    galaxy_off_field_padding_hr_pix: int = 68
    # Calibration-only master density. With a shared field seed, lower-density
    # runs are exact nested thinnings of the same master source proposals.
    galaxy_thinning_max_density_arcmin2: float | None = None
    tng_galaxy_dir:           str   = Config.TNG_SKIRT_DIR
    tng_properties_csv:       str   = ""
    tng_radius_manifest_path: str   = os.path.join(
        Config.DATA_DIR, "_tng_infographics", "tng_radius_manifest.json"
    )
    # Frozen into generation provenance; not a user-selectable setting.
    tng_radius_rendering: str = field(
        default=TNG_RADIUS_RENDERING, init=False,
    )
    tng_radius_renderer_fingerprint: str = field(
        default=TNG_RADIUS_RENDERER_FINGERPRINT, init=False,
    )
    strict_population_artifacts: bool = False
    # Stars
    star_density_arcmin2:     float = Config.DEFAULT_STAR_DENSITY_ARCMIN2
    star_prior_payload:       dict | None = None
    # Lenses
    lens_density_arcmin2:     float = Config.LENS_DENSITY_ARCMIN2
    lens_sigma_v_min_kms:     float = Config.LENS_SIGMA_V_MIN_KMS
    lens_sigma_v_max_kms:     float = Config.LENS_SIGMA_V_MAX_KMS
    lens_theta_e_min_re_ratio: float = Config.LENS_THETA_E_MIN_RE_RATIO
    lens_require_showable:    bool  = False

    def validate(self) -> tuple[bool, str | None]:
        if self.image_size <= 0:
            return False, "image_size must be positive"
        if self.pixel_scale <= 0:
            return False, "pixel_scale must be positive"
        if min(self.galaxy_density_arcmin2,
               self.star_density_arcmin2, self.lens_density_arcmin2) < 0:
            return False, "densities must be non-negative"
        if (
            isinstance(self.galaxy_off_field_padding_hr_pix, bool)
            or not isinstance(
                self.galaxy_off_field_padding_hr_pix, (int, np.integer)
            )
            or self.galaxy_off_field_padding_hr_pix < 0
        ):
            return False, "galaxy_off_field_padding_hr_pix must be a non-negative integer"
        if (
            self.galaxy_thinning_max_density_arcmin2 is not None
            and self.galaxy_thinning_max_density_arcmin2
            < self.galaxy_density_arcmin2
        ):
            return False, (
                "galaxy_thinning_max_density_arcmin2 must be >= "
                "galaxy_density_arcmin2"
            )
        if not (0.0 < self.lens_sigma_v_min_kms < self.lens_sigma_v_max_kms):
            return False, ("lens_sigma_v_min_kms must be in "
                           "(0, lens_sigma_v_max_kms)")
        if self.lens_theta_e_min_re_ratio <= 0.0:
            return False, "lens_theta_e_min_re_ratio must be > 0"
        return True, None


# ---------------------------------------------------------------------------
# Stars (point sources on a correlated stellar-colour locus)
# ---------------------------------------------------------------------------

_STAR_MAG_KEYS = {
    "VIS": "mag_vis",
    "Y_E": "mag_y_e",
    "J_E": "mag_j_e",
    "H_E": "mag_h_e",
}


def star_band_magnitudes_from_record(star: dict) -> dict[str, float]:
    """Read the current persisted four-band stellar magnitudes."""
    return {
        band_name: float(star[_STAR_MAG_KEYS[band_name]])
        for band_name in Config.LR_INPUT_BAND_NAMES
    }


def _deposit_star(
    canvas_4ch: np.ndarray,
    x_pix: float,
    y_pix: float,
    mag_vis: float,
    *,
    band_magnitudes: dict[str, float],
) -> None:
    """Drop a point source at the nearest HR pixel in all four bands."""
    H, W, C = canvas_4ch.shape
    ix = int(round(x_pix))
    iy = int(round(y_pix))
    if not (0 <= ix < W and 0 <= iy < H):
        return
    for k, band_name in enumerate(Config.LR_INPUT_BAND_NAMES):
        band = Config.get_band(band_name)
        mag_k = float(band_magnitudes[band_name])
        canvas_4ch[iy, ix, k] += np.float32(ab_mag_to_electrons(mag_k, band))


def inject_random_stars(
    canvas_4ch: np.ndarray, rng: np.random.Generator, *,
    n_stars: int, stellar_prior: EmpiricalStellarPrior,
) -> list[dict]:
    """Draw ``n_stars`` random point sources and DEPOSIT them onto ``canvas_4ch``.

    The shared star primitive: generation uses it to place a field's fixed
    stars (validate/test), and the on-the-fly forward calls it to inject a
    FRESH star realization per visit — stars are HR deltas added *before* the
    PSF/rebin, so the forward gives them realistic shape and the model learns
    to erase them (the target stays starless). Returns the per-star metadata
    (position + four-band magnitudes), so a caller can persist it to the
    source CSV.
    """
    N = canvas_4ch.shape[0]
    stars: list[dict] = []
    for _ in range(int(n_stars)):
        x_pix = float(rng.uniform(0.0, N - 1))
        y_pix = float(rng.uniform(0.0, N - 1))
        mag = stellar_prior.sample_magnitude(rng)
        sed = sample_stellar_sed(rng, mag, stellar_prior)
        band_mags = sed.magnitudes
        _deposit_star(
            canvas_4ch, x_pix, y_pix, mag, band_magnitudes=band_mags,
        )
        stars.append({
            "type": "star", "x_pix": x_pix, "y_pix": y_pix,
            **{_STAR_MAG_KEYS[name]: value for name, value in band_mags.items()},
            "temperature_k": sed.temperature_k,
            "extinction_av": sed.extinction_av,
        })
    return stars


# ---------------------------------------------------------------------------
# Multi-band simulator
# ---------------------------------------------------------------------------

class SkySimulator:
    """Generates ``(H, W, 4)`` HR clean fields in electrons.

    COSMOS supplies the joint population draw; TNG supplies morphology.
    """

    def __init__(
        self,
        population_prior: CosmosTngPrior | JointGalaxyPopulationPrior | None,
        config: SkySimulatorConfig | None = None,
        *,
        vis_psf_set: PSFSet | None = None,
    ):
        self.population_prior = population_prior
        self.config  = config or SkySimulatorConfig()
        self.vis_psf_set = vis_psf_set
        ok, why = self.config.validate()
        if not ok:
            raise ValueError(f"Invalid generator config: {why}")
        prior_density = getattr(
            population_prior, "surface_density_arcmin2", None
        )
        if prior_density is not None:
            maximum_density = float(prior_density)
            requested_densities = [self.config.galaxy_density_arcmin2]
            if self.config.galaxy_thinning_max_density_arcmin2 is not None:
                requested_densities.append(
                    self.config.galaxy_thinning_max_density_arcmin2
                )
            if max(requested_densities) > maximum_density + 1e-9:
                raise ValueError(
                    "galaxy density exceeds the activated magnitude-law "
                    f"population limit of {maximum_density:g} arcmin^-2"
                )
        response_radius = self._maximum_aperture_response_radius_pixels()
        self._tng_max_output_side = int(
            2 * max(self.config.image_size, response_radius + 2) + 1
        )
        self.tng_renderer = TNGRenderer(
            pixel_scale_arcsec=self.config.pixel_scale,
            max_output_side=self._tng_max_output_side,
        )
        self.stellar_prior = (
            EmpiricalStellarPrior.from_payload(self.config.star_prior_payload)
            if self.config.star_prior_payload else None
        )
        if (
            population_prior is None
            and self.config.galaxy_density_arcmin2 > 0.0
        ):
            raise ValueError(
                "population_prior=None requires galaxy_density_arcmin2=0"
            )

        if self.config.strict_population_artifacts:
            if self.config.star_density_arcmin2 > 0.0 and self.stellar_prior is None:
                raise ValueError(
                    "strict population generation requires an active empirical "
                    "stellar prior"
                )
            if (
                population_prior is None
                and (
                    self.config.galaxy_density_arcmin2 > 0.0
                    or self.config.lens_density_arcmin2 > 0.0
                )
            ):
                raise ValueError(
                    "strict population generation requires a galaxy "
                    "population prior"
                )

        # Opening the facade is read-only: the parent pipeline repairs the
        # manifest before workers start, while every worker validates the same
        # inventory/catalog/radius snapshot before rendering anything.
        needs_tng = (
            population_prior is not None
            or self.config.galaxy_density_arcmin2 > 0.0
            or self.config.lens_density_arcmin2 > 0.0
        )
        self.tng_atlas: TNGAtlas | None = None
        if needs_tng:
            try:
                self.tng_atlas = TNGAtlas.open(
                    self.config.tng_galaxy_dir,
                    properties_path=self.config.tng_properties_csv or None,
                    manifest_path=self.config.tng_radius_manifest_path or None,
                )
            except ValueError as exc:
                raise ValueError(
                    f"TNG atlas is not submit-ready: {exc}"
                ) from exc

        if self.config.galaxy_density_arcmin2 > 0.0 and not self.tng_atlas:
            # HARD failure, not a warning: with a TNG population requested and
            # zero usable stamps, every field silently renders star-only — a
            # buried stderr line shipped 200 galaxy-free validate/test fields
            # when the netscratch purge deleted the SKIRT atlas (2026-07-06).
            raise RuntimeError(
                f"galaxy_density_arcmin2={self.config.galaxy_density_arcmin2:g} but "
                f"ZERO usable TNG galaxies under "
                f"{self.config.tng_galaxy_dir!r} (a galaxy needs its .done "
                "marker + complete four-band views — an empty dir usually means the "
                "atlas was purged from netscratch). Re-download it via the "
                "TNG atlas page's download step, or set "
                "galaxy_density_arcmin2=0 for star-only fields.")

        # TNG properties for mass → σ_v mapping (redshift mode).
        self._atlas_logm: np.ndarray | None = None
        self._atlas_sfr: np.ndarray | None = None
        self._atlas_logssfr: np.ndarray | None = None
        self._atlas_zero_sfr: np.ndarray | None = None
        self._atlas_activity_class: np.ndarray | None = None
        self._atlas_mass_quantile: np.ndarray | None = None
        self._atlas_ssfr_quantile: np.ndarray | None = None
        self._mass_kernel_bandwidth_by_class: dict[str, float] = {}
        self._ssfr_kernel_bandwidth_by_class: dict[str, float] = {}
        self._morphology_use_counts = np.zeros(
            len(self.tng_atlas) if self.tng_atlas is not None else 0,
            dtype=np.int64,
        )
        if self.tng_atlas is not None:
            atlas = self.tng_atlas
            properties = tuple(
                atlas.properties.get(galaxy.subhalo_id)
                for galaxy in atlas.galaxies
            )
            self._atlas_logm = np.asarray([
                row.log_stellar_mass if row is not None else float("nan")
                for row in properties
            ], dtype=np.float64)
            sfr = np.array([
                row.sfr_msun_yr if row is not None else float("nan")
                for row in properties
            ], dtype=np.float64)
            if self.config.strict_population_artifacts and (
                not atlas.properties
                or not np.isfinite(self._atlas_logm).all()
                or not np.isfinite(sfr).all()
                or np.any(sfr < 0.0)
            ):
                raise ValueError(
                    "strict population generation requires finite mass_stars "
                    "and non-negative SFR properties for every TNG galaxy"
                )
            if not np.isfinite(self._atlas_logm).any():
                self._atlas_logm = None
            elif np.isfinite(self._atlas_logm).all() and np.isfinite(sfr).all():
                atlas_logm = self._atlas_logm
                if atlas_logm is None:  # narrowed above; keeps type explicit
                    raise RuntimeError("TNG mass array disappeared during setup")
                self._atlas_sfr = sfr.astype(np.float64)
                atlas_zero_sfr = np.asarray([
                    row.zero_sfr if row is not None else False
                    for row in properties
                ], dtype=bool)
                self._atlas_zero_sfr = atlas_zero_sfr
                atlas_logssfr = np.asarray([
                    row.log_ssfr if row is not None else float("nan")
                    for row in properties
                ], dtype=np.float64)
                self._atlas_logssfr = atlas_logssfr
                self._atlas_activity_class = np.where(
                    atlas_zero_sfr | (
                        atlas_logssfr
                        < MORPHOLOGY_ACTIVITY_THRESHOLD_LOGSSFR
                    ),
                    "quenched", "star_forming",
                )
                self._atlas_mass_quantile = conditional_mass_quantiles(
                    atlas_logm, self._atlas_activity_class,
                )
                self._atlas_ssfr_quantile = conditional_ssfr_quantiles(
                    atlas_logssfr,
                    self._atlas_activity_class,
                    zero_sfr=self._atlas_zero_sfr,
                )
                for label in np.unique(self._atlas_activity_class):
                    class_quantiles = self._atlas_mass_quantile[
                        self._atlas_activity_class == label
                    ]
                    class_ssfr_quantiles = self._atlas_ssfr_quantile[
                        self._atlas_activity_class == label
                    ]
                    self._mass_kernel_bandwidth_by_class[str(label)] = float(
                        cross_validated_mass_bandwidth(class_quantiles)
                    )
                    self._ssfr_kernel_bandwidth_by_class[str(label)] = float(
                        cross_validated_mass_bandwidth(class_ssfr_quantiles)
                    )

    @staticmethod
    def _psf_fwhm_arcsec(psf) -> float:
        return (
            float(psf.fwhm_arcsec)
            if psf.fwhm_arcsec is not None
            else float(psf.fwhm_pixels("radial") * psf.pixel_scale)
        )

    def _maximum_aperture_response_radius_pixels(self) -> int:
        """Conservative aperture-plus-PSF support for bounded TNG renders."""
        if isinstance(self.population_prior, JointGalaxyPopulationPrior):
            fwhm = (
                self.population_prior.aperture_fwhm_distribution.maximum_arcsec
            )
            sigma_pixels = (
                fwhm
                / self.config.pixel_scale
                / (2.0 * math.sqrt(2.0 * math.log(2.0)))
            )
            kernel_half_support = int(math.ceil(
                _MER_APERTURE_GAUSSIAN_SIGMA_SUPPORT * sigma_pixels
            ))
            return (
                int(math.ceil(fwhm / self.config.pixel_scale))
                + kernel_half_support
            )
        if self.vis_psf_set is None:
            psfs = [make_gaussian_psf(
                Config.get_band("VIS").psf_fwhm_arcsec,
                self.config.pixel_scale,
            )]
        else:
            psfs = self.vis_psf_set.psfs
        radius = 0
        for psf in psfs:
            fwhm = self._psf_fwhm_arcsec(psf)
            if not np.isfinite(fwhm) or fwhm <= 0.0:
                raise ValueError("VIS PSF set contains a non-positive FWHM")
            kernel_half_support = max(
                int(math.ceil((psf.shape[0] - 1) / 2.0)),
                int(math.ceil((psf.shape[1] - 1) / 2.0)),
            )
            radius = max(
                radius,
                int(math.ceil(fwhm / self.config.pixel_scale))
                + kernel_half_support,
            )
        return radius

    def _build_mer_aperture_psf(
        self, fwhm_arcsec: float,
    ) -> tuple[np.ndarray, float, str]:
        """Build the circular target PSF represented by the MER FWHM."""
        fwhm = float(fwhm_arcsec)
        if not np.isfinite(fwhm) or fwhm <= 0.0:
            raise ValueError("MER aperture FWHM must be finite and positive")
        sigma_pixels = (
            fwhm
            / self.config.pixel_scale
            / (2.0 * math.sqrt(2.0 * math.log(2.0)))
        )
        half_support = max(1, int(math.ceil(
            _MER_APERTURE_GAUSSIAN_SIGMA_SUPPORT * sigma_pixels
        )))
        psf = make_gaussian_psf(
            fwhm,
            self.config.pixel_scale,
            size=2 * half_support + 1,
        )
        return (
            np.asarray(psf.data, dtype=np.float32),
            fwhm,
            "q1_mer_photometric_fwhm_gaussian",
        )

    # ------------------------------------------------------------------ #
    def _field_area_arcmin2(self) -> float:
        side_arcmin = self.config.image_size * self.config.pixel_scale / 60.0
        return side_arcmin ** 2

    def _random_pix(self, rng: np.random.Generator) -> tuple[float, float]:
        N = self.config.image_size
        return float(rng.uniform(0.0, N - 1)), float(rng.uniform(0.0, N - 1))

    def _off_field_area_arcmin2(self) -> float:
        """Area of the bounded exterior proposal frame."""
        padding = int(self.config.galaxy_off_field_padding_hr_pix)
        if padding <= 0:
            return 0.0
        pixel_area_arcmin2 = (self.config.pixel_scale / 60.0) ** 2
        side = self.config.image_size
        return float(((side + 2 * padding) ** 2 - side**2) * pixel_area_arcmin2)

    def _random_off_field_pix(
        self, rng: np.random.Generator,
    ) -> tuple[float, float]:
        """Draw uniformly from the exterior frame, with no rejection loop."""
        side = float(self.config.image_size)
        padding = float(self.config.galaxy_off_field_padding_hr_pix)
        if padding <= 0.0:
            raise ValueError("off-field galaxy padding is disabled")

        # Top/bottom own the four corners; left/right span only the field
        # height.  The four rectangles are disjoint and exactly tile the
        # expanded square minus the saved field.
        horizontal_area = padding * (side + 2.0 * padding)
        vertical_area = padding * side
        selector = float(rng.uniform(
            0.0, 2.0 * horizontal_area + 2.0 * vertical_area,
        ))
        if selector < horizontal_area:
            return (
                float(rng.uniform(-padding, side + padding)),
                float(rng.uniform(-padding, 0.0)),
            )
        selector -= horizontal_area
        if selector < horizontal_area:
            return (
                float(rng.uniform(-padding, side + padding)),
                float(rng.uniform(side, side + padding)),
            )
        selector -= horizontal_area
        if selector < vertical_area:
            return (
                float(rng.uniform(-padding, 0.0)),
                float(rng.uniform(0.0, side)),
            )
        return (
            float(rng.uniform(side, side + padding)),
            float(rng.uniform(0.0, side)),
        )

    def _off_field_galaxy_reaches_canvas(
        self,
        x_pix: float,
        y_pix: float,
        re_arcsec: float,
    ) -> bool:
        """Whether the source's bounded 4-R_e support reaches the field."""
        side = float(self.config.image_size)
        dx = max(0.0, -float(x_pix), float(x_pix) - side)
        dy = max(0.0, -float(y_pix), float(y_pix) - side)
        distance = math.hypot(dx, dy)
        support = min(
            float(self.config.galaxy_off_field_padding_hr_pix),
            _OFF_FIELD_GALAXY_RE_SUPPORT
            * float(re_arcsec)
            / self.config.pixel_scale,
        )
        return bool(np.isfinite(support) and support > 0.0 and distance <= support)

    def _eligible_morphology_indices(
        self,
        target_re_arcsec: float,
    ) -> np.ndarray:
        """Return donors with at least one orientation large enough to shrink."""
        atlas = self.tng_atlas
        if (
            atlas is None
            or not atlas
            or not np.isfinite(target_re_arcsec)
            or target_re_arcsec <= 0.0
        ):
            raise ValueError("TNG shrink-only donor selection is unavailable")
        eligible_galaxies = atlas.eligible_galaxies(
            target_re_arcsec,
            self.config.pixel_scale,
        )
        eligible_ids = {galaxy.subhalo_id for galaxy in eligible_galaxies}
        eligible = np.asarray(
            [
                index
                for index, galaxy in enumerate(atlas.galaxies)
                if galaxy.subhalo_id in eligible_ids
            ],
            dtype=np.int64,
        )
        if not eligible.size:
            maximum_arcsec = float(
                max(atlas.max_native_re_px(galaxy) for galaxy in atlas)
                * self.config.pixel_scale
            )
            raise _NoRenderableTNGDonorError(
                f"no TNG donor can render R_e={target_re_arcsec:g} arcsec "
                f"without enlargement; atlas maximum is {maximum_arcsec:g} "
                "arcsec"
            )
        return eligible

    # ------------------------------------------------------------------ #
    def _pick_field_galaxy(
        self,
        rng: np.random.Generator,
        target_mass_quantile: float,
        target_ssfr_quantile: float,
        activity_class: str,
        target_re_arcsec: float,
    ) -> tuple[TNGGalaxy, dict[str, float | int | str]]:
        """Choose a TNG donor by conditional mass-sSFR rank transport."""
        if (
            self._atlas_logm is None
            or self._atlas_activity_class is None
            or self._atlas_mass_quantile is None
            or self._atlas_ssfr_quantile is None
            or self._atlas_sfr is None
            or self._atlas_logssfr is None
            or self._atlas_zero_sfr is None
        ):
            raise ValueError(
                "TNG morphology quantile transport requires mass and SFR "
                "properties for every donor"
            )
        target_mass = float(target_mass_quantile)
        target_ssfr = float(target_ssfr_quantile)
        label = str(activity_class)
        if not np.isfinite(target_mass) or not 0.0 <= target_mass <= 1.0:
            raise ValueError("COSMOS morphology target mass quantile is invalid")
        if not np.isfinite(target_ssfr) or not 0.0 <= target_ssfr <= 1.0:
            raise ValueError("COSMOS morphology target sSFR quantile is invalid")
        renderable = self._eligible_morphology_indices(target_re_arcsec)
        candidates = np.intersect1d(
            np.flatnonzero(self._atlas_activity_class == label),
            renderable,
            assume_unique=True,
        )
        if not candidates.size:
            raise _NoRenderableTNGDonorError(
                f"TNG atlas has no shrink-only {label!r} morphology donor "
                f"for R_e={target_re_arcsec:g} arcsec"
            )
        mass_bandwidth = self._mass_kernel_bandwidth_by_class.get(label)
        ssfr_bandwidth = self._ssfr_kernel_bandwidth_by_class.get(label)
        if mass_bandwidth is None or ssfr_bandwidth is None:
            raise ValueError(f"TNG atlas lacks a {label!r} transport bandwidth")
        balance = np.power(
            1.0 + self._morphology_use_counts[candidates].astype(np.float64),
            -MORPHOLOGY_BALANCE_POWER,
        )
        (
            probabilities, used_mass_bandwidth, used_ssfr_bandwidth,
            effective_donors,
        ) = joint_quantile_transport_weights(
            self._atlas_mass_quantile[candidates],
            self._atlas_ssfr_quantile[candidates],
            target_mass,
            target_ssfr,
            mass_bandwidth=mass_bandwidth,
            ssfr_bandwidth=ssfr_bandwidth,
            minimum_effective_donors=MORPHOLOGY_MIN_EFFECTIVE_DONORS,
            balance_weights=balance,
        )
        local_index = int(rng.choice(candidates.size, p=probabilities))
        selected = int(candidates[local_index])
        use_count = int(self._morphology_use_counts[selected]) + 1
        self._morphology_use_counts[selected] = use_count
        donor_mass_quantile = float(self._atlas_mass_quantile[selected])
        donor_ssfr_quantile = float(self._atlas_ssfr_quantile[selected])
        proxy_logmass_method = getattr(
            self.population_prior, "proxy_logmass", None
        )
        proxy_logmass = (
            float(
                cast(Callable[[float, str], float], proxy_logmass_method)(
                    donor_mass_quantile, label
                )
            )
            if callable(proxy_logmass_method)
            else float("nan")
        )
        atlas = self.tng_atlas
        if atlas is None:
            raise ValueError("TNG atlas is unavailable")
        return atlas.galaxies[selected], {
            "activity_class": label,
            "target_mass_quantile": target_mass,
            "target_ssfr_quantile": target_ssfr,
            "tng_mass_quantile": donor_mass_quantile,
            "tng_ssfr_quantile": donor_ssfr_quantile,
            "native_tng_logmass": float(self._atlas_logm[selected]),
            "native_tng_sfr": float(self._atlas_sfr[selected]),
            "native_tng_logssfr": float(self._atlas_logssfr[selected]),
            "native_tng_zero_sfr": bool(self._atlas_zero_sfr[selected]),
            "morphology_proxy_logmass": proxy_logmass,
            "mass_quantile_delta": donor_mass_quantile - target_mass,
            "ssfr_quantile_delta": donor_ssfr_quantile - target_ssfr,
            "selection_probability": float(probabilities[local_index]),
            "effective_donors": float(effective_donors),
            "kernel_bandwidth_quantile": float(used_mass_bandwidth),
            "mass_kernel_bandwidth_quantile": float(used_mass_bandwidth),
            "ssfr_kernel_bandwidth_quantile": float(used_ssfr_bandwidth),
            "worker_donor_use_count": use_count,
        }

    def _pick_random_field_galaxy(
        self,
        rng: np.random.Generator,
        target_re_arcsec: float,
    ) -> tuple[TNGGalaxy, dict[str, float | int | str]]:
        """Choose an explicitly unconditioned, diversity-balanced TNG donor."""
        if (
            self._atlas_logm is None
            or self._atlas_activity_class is None
            or self._atlas_mass_quantile is None
            or self._atlas_ssfr_quantile is None
            or self._atlas_sfr is None
            or self._atlas_logssfr is None
            or self._atlas_zero_sfr is None
        ):
            raise ValueError(
                "random TNG morphology assignment requires complete mass and "
                "SFR properties for every donor"
            )
        candidates = self._eligible_morphology_indices(target_re_arcsec)
        balance = np.power(
            1.0 + self._morphology_use_counts[candidates].astype(np.float64),
            -MORPHOLOGY_BALANCE_POWER,
        )
        probabilities = balance / np.sum(balance)
        local_index = int(rng.choice(candidates.size, p=probabilities))
        selected = int(candidates[local_index])
        use_count = int(self._morphology_use_counts[selected]) + 1
        self._morphology_use_counts[selected] = use_count

        atlas = self.tng_atlas
        if atlas is None:
            raise ValueError("TNG atlas is unavailable")
        return atlas.galaxies[selected], {
            "activity_class": str(self._atlas_activity_class[selected]),
            "target_mass_quantile": float("nan"),
            "target_ssfr_quantile": float("nan"),
            "tng_mass_quantile": float(self._atlas_mass_quantile[selected]),
            "tng_ssfr_quantile": float(self._atlas_ssfr_quantile[selected]),
            "native_tng_logmass": float(self._atlas_logm[selected]),
            "native_tng_sfr": float(self._atlas_sfr[selected]),
            "native_tng_logssfr": float(self._atlas_logssfr[selected]),
            "native_tng_zero_sfr": bool(self._atlas_zero_sfr[selected]),
            "morphology_proxy_logmass": float("nan"),
            "mass_quantile_delta": float("nan"),
            "ssfr_quantile_delta": float("nan"),
            "selection_probability": float(probabilities[local_index]),
            "effective_donors": float(
                1.0 / np.sum(probabilities * probabilities)
            ),
            "kernel_bandwidth_quantile": float("nan"),
            "mass_kernel_bandwidth_quantile": float("nan"),
            "ssfr_kernel_bandwidth_quantile": float("nan"),
            "worker_donor_use_count": use_count,
        }

    # ------------------------------------------------------------------ #
    def _add_tng_galaxy(
        self, canvas_4ch: np.ndarray, rng: np.random.Generator,
        *, position: tuple[float, float] | None = None,
        off_field: bool = False,
        redshift_rng: np.random.Generator | None = None,
        _attempt: int = 0,
    ) -> dict | None:
        """Resolve TNG geometry/donor before drawing independent brightness."""
        if self.population_prior is None:
            return None
        prior = self.population_prior
        staged = isinstance(prior, JointGalaxyPopulationPrior)
        if staged and redshift_rng is None:
            redshift_rng = _derived_rng_from_state(
                rng, _GALAXY_REDSHIFT_SEED_TAG,
            )
        if isinstance(prior, JointGalaxyPopulationPrior):
            draw = prior.sample_geometry(rng)
        else:
            draw = prior.sample(rng)
        if off_field:
            if position is None:
                raise ValueError("an off-field galaxy requires an exterior position")
            if not self._off_field_galaxy_reaches_canvas(
                position[0], position[1], draw.re_arcsec,
            ):
                return None
        try:
            if getattr(prior, "morphology_mode", "") == (
                "balanced_random_tng_atlas"
            ):
                galaxy, morphology = self._pick_random_field_galaxy(
                    rng, draw.re_arcsec,
                )
            else:
                galaxy, morphology = self._pick_field_galaxy(
                    rng,
                    draw.mass_quantile,
                    draw.ssfr_quantile,
                    draw.activity_class,
                    draw.re_arcsec,
                )
        except _NoRenderableTNGDonorError as exc:
            if _attempt < 31:
                return self._add_tng_galaxy(
                    canvas_4ch, rng, position=position, off_field=off_field,
                    redshift_rng=redshift_rng,
                    _attempt=_attempt + 1,
                )
            raise RuntimeError(
                "TNG population produced 32 geometries outside the "
                "shrink-only donor support"
            ) from exc
        atlas = self.tng_atlas
        if atlas is None:
            raise ValueError("TNG atlas is unavailable")
        views = atlas.eligible_views(
            galaxy,
            draw.re_arcsec,
            self.config.pixel_scale,
        )
        if not views:
            raise RuntimeError("selected TNG donor has no shrink-only view")
        view = views[int(rng.integers(0, len(views)))]
        if staged:
            rendered = self.tng_renderer.render_observed_radius(
                view,
                draw.re_arcsec,
                rng=rng,
                target_vis_flux_e=None,
            )
            draw = prior.complete_draw(
                draw,
                rng,
                redshift_rng=redshift_rng,
            )
            rendered = self.tng_renderer.apply_redshift_photometry(
                rendered,
                draw.z,
                rng=redshift_rng,
            )
        elif np.isfinite(draw.z):
            rendered = self.tng_renderer.render_observed_radius_at_redshift(
                view,
                draw.re_arcsec,
                draw.z,
                rng=rng,
                target_vis_flux_e=(
                    None if staged else draw.target_vis_flux_e
                ),
            )
        else:
            rendered = self.tng_renderer.render_observed_radius(
                view,
                draw.re_arcsec,
                rng=rng,
                target_vis_flux_e=(
                    None if staged else draw.target_vis_flux_e
                ),
            )
        if isinstance(prior, JointGalaxyPopulationPrior):
            magnitude = draw.target_vis_mag
            target_aperture_flux = draw.target_vis_flux_e
            aperture_fwhm = prior.sample_aperture_fwhm(
                rng, magnitude=magnitude,
            )
            psf_kernel, psf_fwhm, psf_source = (
                self._build_mer_aperture_psf(aperture_fwhm)
            )
            try:
                rendered = self.tng_renderer.normalize_vis_2fwhm(
                    rendered,
                    target_flux_e=target_aperture_flux,
                    psf_kernel=psf_kernel,
                    psf_fwhm_arcsec=psf_fwhm,
                    psf_identity=psf_source,
                )
            except ValueError as exc:
                if _attempt < 31:
                    return self._add_tng_galaxy(
                        canvas_4ch, rng, position=position,
                        off_field=off_field,
                        redshift_rng=redshift_rng,
                        _attempt=_attempt + 1,
                    )
                raise RuntimeError(
                    "TNG population produced 32 unusable 2FWHM stamps"
                ) from exc
        x_pix, y_pix = (
            position if position is not None else self._random_pix(rng)
        )
        intersects = composite_stamp(
            canvas_4ch, rendered.data, x_pix, y_pix,
        )
        if off_field and not intersects:
            return None
        tmeta = rendered.record_fields()
        return {
            "type": "galaxy",
            "render": "tng",
            "off_field": bool(off_field),
            "x_pix": float(x_pix),
            "y_pix": float(y_pix),
            "subhalo_id":   tmeta["subhalo_id"],
            "orientation":  tmeta["orientation"],
            "rebin_factor": tmeta["rebin_factor"],
            "rot_angle": tmeta["rot_angle"],
            "arbitrary_rotation": tmeta["arbitrary_rotation"],
            "catalog_id":   draw.catalog_id,
            "z":            draw.z,
            "mass_scale":   1.0,
            "native_tng_logmass": morphology["native_tng_logmass"],
            "native_tng_sfr": morphology["native_tng_sfr"],
            "native_tng_logssfr": morphology["native_tng_logssfr"],
            "native_tng_zero_sfr": morphology["native_tng_zero_sfr"],
            "morphology_proxy_logmass": morphology["morphology_proxy_logmass"],
            "target_mass_quantile": morphology["target_mass_quantile"],
            "target_ssfr_quantile": morphology["target_ssfr_quantile"],
            "target_logmass": draw.logmass,
            "target_logssfr": draw.logssfr,
            "logmass": draw.logmass,
            "physical_model_fingerprint": draw.physical_model_fingerprint,
            "tng_mass_quantile": morphology["tng_mass_quantile"],
            "tng_ssfr_quantile": morphology["tng_ssfr_quantile"],
            "morphology_mass_quantile_delta": morphology[
                "mass_quantile_delta"
            ],
            "morphology_ssfr_quantile_delta": morphology[
                "ssfr_quantile_delta"
            ],
            "morphology_selection_probability": morphology[
                "selection_probability"
            ],
            "morphology_effective_donors": morphology["effective_donors"],
            "morphology_kernel_bandwidth_quantile": morphology[
                "kernel_bandwidth_quantile"
            ],
            "morphology_mass_kernel_bandwidth_quantile": morphology[
                "mass_kernel_bandwidth_quantile"
            ],
            "morphology_ssfr_kernel_bandwidth_quantile": morphology[
                "ssfr_kernel_bandwidth_quantile"
            ],
            "morphology_worker_use_count": morphology[
                "worker_donor_use_count"
            ],
            "morphology_activity_class": morphology["activity_class"],
            "galaxy_density_arcmin2": float(self.config.galaxy_density_arcmin2),
            "galaxy_prior_density_arcmin2": float(
                getattr(
                    self.population_prior,
                    "surface_density_arcmin2",
                    self.config.galaxy_density_arcmin2,
                )
            ),
            "galaxy_vis_magnitude_max": float(
                getattr(
                    getattr(self.population_prior, "magnitude_law", None),
                    "mag_faint",
                    float("nan"),
                )
            ),
            "galaxy_magnitude_break": float(
                getattr(
                    getattr(self.population_prior, "magnitude_law", None),
                    "break_magnitude",
                    float("nan"),
                )
            ),
            "galaxy_faint_density_cap_arcmin2_mag": float(
                getattr(
                    getattr(self.population_prior, "magnitude_law", None),
                    "density_cap_arcmin2_mag",
                    float("nan"),
                )
            ),
            "population_prior": getattr(
                self.population_prior,
                "population_label",
                (
                    "joint_analytical_staged_2fwhm_v3"
                    if getattr(
                        self.population_prior, "_physical_conditionals", None
                    ) is not None
                    else (
                        "joint_analytical_staged_v3"
                        if isinstance(
                            self.population_prior, JointGalaxyPopulationPrior,
                        )
                        else "cosmos2025_joint"
                    )
                ),
            ),
            "mag_hst_f814w": draw.mag_hst_f814w,
            "target_vis_mag": draw.target_vis_mag,
            "magnitude_fit_fingerprint": str(
                getattr(self.population_prior, "fingerprint", "")
            ),
            "target_vis_2fwhm_mag": float(
                tmeta.get("target_vis_2fwhm_mag", float("nan"))
            ),
            "target_vis_2fwhm_flux_e": float(
                tmeta.get("target_vis_2fwhm_flux_e", float("nan"))
            ),
            "achieved_vis_2fwhm_mag": float(
                tmeta.get("achieved_vis_2fwhm_mag", float("nan"))
            ),
            "achieved_vis_2fwhm_flux_e": float(
                tmeta.get("achieved_vis_2fwhm_flux_e", float("nan"))
            ),
            "mer_photometric_fwhm_arcsec": float(
                tmeta.get("mer_photometric_fwhm_arcsec", float("nan"))
            ),
            "aperture_radius_arcsec": float(
                tmeta.get("aperture_radius_arcsec", float("nan"))
            ),
            "aperture_diameter_arcsec": float(
                tmeta.get("aperture_diameter_arcsec", float("nan"))
            ),
            "aperture_psf_fwhm_arcsec": float(
                tmeta.get("aperture_psf_fwhm_arcsec", float("nan"))
            ),
            "aperture_psf_source": str(tmeta.get("aperture_psf_source", "")),
            "brightness_transfer": draw.brightness_transfer,
            "brightness_scale": float(
                tmeta.get("brightness_scale", float("nan"))
            ),
            "drift_eps":    float(tmeta.get("drift_eps", float("nan"))),
            "target_re_arcsec":   float(tmeta.get("target_re_arcsec", float("nan"))),
            "apparent_re_arcsec": float(tmeta.get("apparent_re_arcsec", float("nan"))),
            "achieved_re_arcsec": float(tmeta.get("achieved_re_arcsec", float("nan"))),
            "native_halflight_px": float(
                tmeta.get("native_halflight_px", float("nan"))
            ),
            "radius_scale_factor": float(
                tmeta.get("radius_scale_factor", float("nan"))
            ),
            "radius_rendering": str(tmeta.get("radius_rendering", "")),
            "radius_renderer_fingerprint": str(
                tmeta.get("radius_renderer_fingerprint", "")
            ),
            "radius_manifest_fingerprint": str(
                tmeta.get("radius_manifest_fingerprint", "")
            ),
            "render_support_clipped": bool(
                tmeta.get("render_support_clipped", False)
            ),
            "tng_render_trace": tmeta,
            # Unified half-light radius + log stellar mass persisted to the
            # source catalog for later analysis.
            "re_arcsec":    draw.re_arcsec,
            "imputed_size": draw.imputed_size,
            "flux_e_per_band": [float(tmeta["flux_e_per_band"][b])
                                for b in Config.LR_INPUT_BAND_NAMES],
        }

    def _draw_star(self, rng: np.random.Generator) -> dict:
        """Draw one star's position + four-band colour — WITHOUT depositing it.

        Deposition is deferred: at generation the base scene stays starless
        (stars are re-added in the forward op — fresh per visit on-the-fly, or
        from this recorded metadata for the fixed validate/test fields)."""
        x_pix, y_pix = self._random_pix(rng)
        if self.stellar_prior is None:
            raise ValueError("an active empirical stellar prior is required")
        mag = self.stellar_prior.sample_magnitude(rng)
        sed = sample_stellar_sed(rng, mag, self.stellar_prior)
        band_mags = sed.magnitudes
        return {
            "type": "star",
            "x_pix": float(x_pix),
            "y_pix": float(y_pix),
            **{_STAR_MAG_KEYS[name]: value for name, value in band_mags.items()},
            "temperature_k": sed.temperature_k,
            "extinction_av": sed.extinction_av,
        }

    def _add_lens_pure(
        self, canvas_4ch: np.ndarray, rng: np.random.Generator,
        *, max_tries: int = 32,
    ) -> dict | None:
        """Catalog-free lens system: TNG deflector + TNG source.

        Used when no COSMOS catalog is available (TNG-only mode). Per try:
        pick a subhalo, derive σ_v from its stellar mass, draw geometry via
        :func:`sample_lens_geometry`, accept when θ_E ≥ κ × apparent R_e.
        """
        cfg = self.config
        kappa = cfg.lens_theta_e_min_re_ratio
        atlas = self.tng_atlas
        if atlas is None:
            raise ValueError("TNG lens rendering requires an open atlas")
        for _ in range(max_tries):
            galaxy = atlas.galaxies[int(rng.integers(0, len(atlas)))]
            gid = galaxy.subhalo_id
            orientation = int(rng.integers(1, N_ORIENTATIONS + 1))
            props = atlas.properties.get(gid)
            mstar = (
                props.stellar_mass_msun
                if props is not None else float("nan")
            )
            sigma_v = sigma_v_from_stellar_mass(mstar, rng)
            if not math.isfinite(sigma_v):
                if cfg.strict_population_artifacts:
                    raise ValueError(
                        f"TNG stellar mass is invalid for strict lens {gid}"
                    )
                sigma_v = float(rng.uniform(cfg.lens_sigma_v_min_kms,
                                            cfg.lens_sigma_v_max_kms))
            lp = sample_lens_geometry(rng, sigma_v)
            if lp is None:
                continue
            lens_view = atlas.view(galaxy, orientation)
            re_px = lens_view.native_re_px
            re_app = physical_pc_to_arcsec(
                re_px * TNG_NATIVE_PC_PER_PIXEL,
                lp.z_lens) / compactness_factor(lp.z_lens)
            if lp.theta_E_arcsec < kappa * re_app:
                continue
            source_galaxy = atlas.galaxies[
                int(rng.integers(0, len(atlas)))
            ]
            sori = int(rng.integers(1, N_ORIENTATIONS + 1))
            source_view = atlas.view(source_galaxy, sori)
            if cfg.lens_require_showable:
                r_vis = self.tng_renderer.predict_visible_radius_arcsec(
                    lens_view, lp.z_lens
                )
                if (lp.theta_E_arcsec
                        < Config.LENS_SHOWABLE_THETA_E_FRAC * r_vis):
                    continue
                if (self.tng_renderer.predict_vis_flux_e(
                        source_view, lp.z_source)
                        < Config.LENS_SHOWABLE_MIN_SRC_VIS_E):
                    continue
            try:
                lens_light_stamp = (
                    self.tng_renderer.render_physical_at_redshift(
                        lens_view, lp.z_lens, rng=rng
                    )
                )
                source_stamp = self.tng_renderer.render_physical_at_redshift(
                    source_view, lp.z_source, rng=rng
                )
            except (OSError, TypeError, ValueError):
                continue
            x_pix, y_pix = self._random_pix(rng)
            lp = replace(lp, centre_x_pix=x_pix, centre_y_pix=y_pix)
            render_lens_to_multiband_canvas(
                canvas_4ch, params=lp, pixel_scale=cfg.pixel_scale,
                lens_light_stamp=lens_light_stamp,
                source_stamp=source_stamp,
            )
            lens_trace = lens_light_stamp.record_fields()
            source_trace = source_stamp.record_fields()
            return {
                "type": "lens",
                "x_pix": float(x_pix),
                "y_pix": float(y_pix),
                "z_lens": float(lp.z_lens),
                "z_source": float(lp.z_source),
                "theta_E_arcsec": float(lp.theta_E_arcsec),
                "sie_axis_ratio": float(lp.lens_q),
                "sigma_v_kms": float(sigma_v),
                "lens_mstar_msun": float(mstar),
                "lens_apparent_re_arcsec": float(re_app),
                "lens_light_render": "tng",
                "lens_light_re_arcsec": float(re_app),
                "source_render": "tng",
                "lens_subhalo_id": str(gid),
                "lens_orientation": lens_trace["orientation"],
                "source_subhalo_id": source_trace["subhalo_id"],
                "source_orientation": source_trace["orientation"],
                "radius_manifest_fingerprint": atlas.fingerprint,
                "lens_tng_trace": lens_trace,
                "source_tng_trace": source_trace,
                "lens_visible_r_arcsec": float(
                    lens_light_stamp.shape[0] * cfg.pixel_scale / 2.0),
                "source_flux_vis_e": source_stamp.flux_e("VIS"),
            }
        return None

    def _add_lens(
        self, canvas_4ch: np.ndarray, rng: np.random.Generator,
    ) -> dict | None:
        """Render one pure-TNG strong-lens system onto the canvas."""
        if self.tng_atlas is None or not self.tng_atlas:
            return None
        return self._add_lens_pure(canvas_4ch, rng)

    # ------------------------------------------------------------------ #
    def generate(self, rng=None, *, store=None, **kwargs) -> Image:
        """Generate one clean HR field as a stamped :class:`Image` (role ``'hr'``)."""
        if rng is None:
            rng = np.random.default_rng()
        sky, _meta = self.simulate_field(rng, **kwargs)
        return sky.with_role(Role.HR).with_stamp(
            Stamp(id=mint_id(store), schema_version=3))

    def simulate_field(
        self,
        rng: np.random.Generator,
        *,
        n_galaxies: int | None = None,
        n_stars:   int | None = None,
        n_lenses:  int | None = None,
        deposit_stars: bool = False,
    ) -> tuple[Image, dict]:
        """Render one clean HR field in 4 bands.

        The rendered scene is STARLESS by default (galaxies + lenses only) —
        the network's target. Stars are still DRAWN and returned in
        ``metadata["stars"]`` (positions + four-band magnitudes) so the forward op can
        re-inject them: fresh per visit for on-the-fly training, or from this
        record for the fixed validate/test fields. Pass ``deposit_stars=True``
        to also stamp them onto the canvas (the with-stars scene, e.g. for
        inspection); ``n_stars=0`` draws none (the training split).

        Returns
        -------
        sky_image : :class:`Image` with shape ``(image_size, image_size, 4)``
                    in raw electrons (starless unless ``deposit_stars``).
        metadata  : dict with per-source parameter records.
        """
        cfg  = self.config
        N    = cfg.image_size
        area = self._field_area_arcmin2()
        sample_population = n_galaxies is None

        # Component seeds are consumed once per field. Galaxy-count changes can
        # therefore never shift the star, lens, or forward-model RNG streams.
        component_seeds = rng.integers(
            0, np.iinfo(np.uint64).max, size=3, dtype=np.uint64,
        )
        galaxy_rng = np.random.default_rng(component_seeds[0])
        star_rng = np.random.default_rng(component_seeds[1])
        lens_rng = np.random.default_rng(component_seeds[2])

        # Derive an independent deterministic stream without consuming another
        # value from the caller's RNG.  This preserves the in-field galaxy,
        # star, lens, and subsequent forward-model streams exactly.
        off_field_rng = np.random.default_rng(np.random.SeedSequence([
            int(component_seeds[0]), _OFF_FIELD_GALAXY_SEED_TAG,
        ]))

        galaxy_seeds: list[int]
        if n_galaxies is None:
            master_density = float(
                cfg.galaxy_thinning_max_density_arcmin2
                if cfg.galaxy_thinning_max_density_arcmin2 is not None
                else cfg.galaxy_density_arcmin2
            )
            master_count = int(galaxy_rng.poisson(master_density * area))
            keep_probability = (
                cfg.galaxy_density_arcmin2 / master_density
                if master_density > 0 else 0.0
            )
            keep = galaxy_rng.random(master_count) < keep_probability
            proposals = galaxy_rng.integers(
                0, np.iinfo(np.uint64).max,
                size=master_count, dtype=np.uint64,
            )
            galaxy_seeds = [
                int(seed) for seed, retained
                in zip(proposals, keep, strict=True) if retained
            ]
            n_galaxies = len(galaxy_seeds)
        else:
            galaxy_seeds = [
                int(value) for value in galaxy_rng.integers(
                    0, np.iinfo(np.uint64).max,
                    size=int(n_galaxies), dtype=np.uint64,
                )
            ]
        off_field_galaxy_seeds: list[int] = []
        if (
            sample_population
            and cfg.galaxy_density_arcmin2 > 0.0
            and cfg.galaxy_off_field_padding_hr_pix > 0
        ):
            off_field_master_count = int(off_field_rng.poisson(
                master_density * self._off_field_area_arcmin2()
            ))
            off_field_keep = (
                off_field_rng.random(off_field_master_count) < keep_probability
            )
            off_field_proposals = off_field_rng.integers(
                0, np.iinfo(np.uint64).max,
                size=off_field_master_count, dtype=np.uint64,
            )
            off_field_galaxy_seeds = [
                int(seed) for seed, retained
                in zip(off_field_proposals, off_field_keep, strict=True)
                if retained
            ]
        if n_stars is None:
            n_stars = int(star_rng.poisson(cfg.star_density_arcmin2 * area))
        if n_lenses is None:
            n_lenses = int(lens_rng.poisson(cfg.lens_density_arcmin2 * area))

        canvas = np.zeros((N, N, Config.NUM_LR_CHANNELS), dtype=np.float32)
        galaxies, off_field_galaxies, stars, lenses = [], [], [], []

        for source_seed in galaxy_seeds:
            rec = self._add_tng_galaxy(
                canvas,
                np.random.default_rng(source_seed),
            )
            if rec is not None:
                galaxies.append(rec)

        for source_seed in off_field_galaxy_seeds:
            source_rng = np.random.default_rng(source_seed)
            position = self._random_off_field_pix(source_rng)
            rec = self._add_tng_galaxy(
                canvas,
                source_rng,
                position=position,
                off_field=True,
            )
            if rec is not None:
                off_field_galaxies.append(rec)

        # Stars are DRAWN (recorded) but not deposited — the base stays
        # starless; the forward op re-injects them (see inject_random_stars).
        for _ in range(n_stars):
            star = self._draw_star(star_rng)
            if deposit_stars:
                _deposit_star(
                    canvas,
                    star["x_pix"],
                    star["y_pix"],
                    star["mag_vis"],
                    band_magnitudes=star_band_magnitudes_from_record(star),
                )
            stars.append(star)

        for _ in range(n_lenses):
            rec = self._add_lens(canvas, lens_rng)
            if rec is not None:
                lenses.append(rec)

        meta = {
            "field_area_arcmin2":      float(area),
            "galaxy_density_arcmin2":  float(cfg.galaxy_density_arcmin2),
            "galaxy_prior_density_arcmin2": float(
                getattr(
                    self.population_prior,
                    "surface_density_arcmin2",
                    cfg.galaxy_density_arcmin2,
                )
            ),
            "galaxy_vis_magnitude_max": float(
                getattr(
                    getattr(self.population_prior, "magnitude_law", None),
                    "mag_faint",
                    float("nan"),
                )
            ),
            "galaxy_magnitude_break": float(
                getattr(
                    getattr(self.population_prior, "magnitude_law", None),
                    "break_magnitude",
                    float("nan"),
                )
            ),
            "galaxy_faint_density_cap_arcmin2_mag": float(
                getattr(
                    getattr(self.population_prior, "magnitude_law", None),
                    "density_cap_arcmin2_mag",
                    float("nan"),
                )
            ),
            "galaxy_thinning_max_density_arcmin2": (
                float(cfg.galaxy_thinning_max_density_arcmin2)
                if cfg.galaxy_thinning_max_density_arcmin2 is not None
                else None
            ),
            "population_prior": getattr(
                self.population_prior, "population_label", "cosmos2025_joint",
            ) if self.population_prior is not None else "none",
            "star_density_arcmin2":    float(cfg.star_density_arcmin2),
            "star_population_fingerprint": (
                str(cfg.star_prior_payload.get("fingerprint", ""))
                if cfg.star_prior_payload else "none"
            ),
            "lens_density_arcmin2":    float(cfg.lens_density_arcmin2),
            "n_galaxies": len(galaxies),
            "n_off_field_galaxy_proposals": len(off_field_galaxy_seeds),
            "n_off_field_galaxies": len(off_field_galaxies),
            "n_stars":    len(stars),
            "n_lenses":   len(lenses),
            "galaxies":   galaxies,
            "off_field_galaxies": off_field_galaxies,
            "stars":      stars,
            "lenses":     lenses,
        }
        sky = Image(
            data=canvas,
            pixel_scale_arcsec=cfg.pixel_scale,
            band_names=Config.LR_INPUT_BAND_NAMES,
            is_clean=True,
            metadata=meta,
        )
        return sky, meta
