"""COSMOS2025 physical prior for TNG morphology draws.

COSMOS supplies redshift, stellar mass, apparent size, and an HST/ACS F814W
brightness anchor. TNG supplies the Euclid VIS/NISP morphology and colours.
The F814W anchor is mapped to VIS by the fitted observation transfer, then one
shared scalar normalizes all four TNG channels so their ratios are preserved.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from euclid_polish.config import Config
from euclid_polish.photometry import ab_mag_to_electrons
from euclid_polish.population.euclid_galaxy_prior import (
    JOINT_EUCLID_GALAXY_VERSION,
    ConditionalRadiusLaw,
    joint_density_grid,
)
from euclid_polish.population.magnitude_law import StraightMagnitudeLaw

# This is an explicit project calibration choice, not a TNG mass correction.
# It is used only to keep quenched and star-forming morphology donors separate
# during empirical rank transport.
MORPHOLOGY_ACTIVITY_THRESHOLD_LOGSSFR = -11.0
MORPHOLOGY_MIN_EFFECTIVE_DONORS = 64
MORPHOLOGY_BALANCE_POWER = 0.5


@dataclass(frozen=True)
class CosmosTngDraw:
    catalog_id: str
    mag_hst_f814w: float
    target_vis_mag: float
    target_vis_flux_e: float
    z: float
    logmass: float
    re_arcsec: float
    imputed_size: bool
    brightness_transfer: str
    mass_quantile: float = float("nan")
    ssfr_quantile: float = float("nan")
    activity_class: str = "unknown"
    logssfr: float = float("nan")
    physical_model_fingerprint: str = ""


@dataclass(frozen=True)
class F814WToVisTransfer:
    offset_mag: float = 0.0
    magnitude_slope: float = 1.0
    scatter_mag: float = 0.0
    source: str = "embedded_fit"
    fingerprint: str = ""

    def sample_vis_mag(
        self, mag_hst_f814w: float, rng: np.random.Generator,
    ) -> float:
        pivot = 24.0
        mean = (
            pivot
            + self.magnitude_slope * (float(mag_hst_f814w) - pivot)
            + self.offset_mag
        )
        return float(mean + rng.normal(0.0, self.scatter_mag))


def cross_validated_mass_bandwidth(logmass: np.ndarray) -> float:
    """Choose a Gaussian morphology-kernel bandwidth by leave-one-out CV."""
    values = np.asarray(logmass, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return 0.25
    spread = float(np.std(values)) or 0.25
    scale = float(np.median(np.abs(values - np.median(values)))) * 1.4826
    scale = max(scale, spread / max(values.size ** 0.2, 1.0), 0.05)
    grid = np.geomspace(
        max(0.03, scale / 4.0), min(1.0, max(0.08, scale * 4.0)), 24,
    )
    best_h, best_score = float(grid[0]), -float("inf")
    for h in grid:
        diff = (values[:, None] - values[None, :]) / h
        kernels = np.exp(-0.5 * diff * diff)
        np.fill_diagonal(kernels, 0.0)
        denom = kernels.sum(axis=1)
        density = denom / (
            max(values.size - 1, 1) * h * np.sqrt(2.0 * np.pi)
        )
        score = float(np.log(np.maximum(density, 1e-300)).sum())
        if score > best_score:
            best_h, best_score = float(h), score
    return best_h


def empirical_mid_quantiles(values: np.ndarray) -> np.ndarray:
    """Return tie-aware empirical CDF midpoints in ``(0, 1)``.

    Rank transport deliberately discards the absolute mass scale.  Equal
    masses receive the same midpoint so file ordering cannot create a false
    morphology distinction.
    """
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not array.size or not np.isfinite(array).all():
        raise ValueError("empirical quantiles require finite one-dimensional data")
    ordered = np.sort(array, kind="mergesort")
    left = np.searchsorted(ordered, array, side="left")
    right = np.searchsorted(ordered, array, side="right")
    return (left + right).astype(np.float64) / (2.0 * array.size)


def conditional_mass_quantiles(
    logmass: np.ndarray, activity_class: np.ndarray,
) -> np.ndarray:
    """Mass ranks computed independently within each activity population."""
    mass = np.asarray(logmass, dtype=float)
    classes = np.asarray(activity_class).astype(str)
    if mass.ndim != 1 or classes.shape != mass.shape:
        raise ValueError("mass and activity arrays must be aligned")
    result = np.full(mass.shape, np.nan, dtype=np.float64)
    for label in np.unique(classes):
        indices = np.flatnonzero(classes == label)
        result[indices] = empirical_mid_quantiles(mass[indices])
    return result


def conditional_ssfr_quantiles(
    logssfr: np.ndarray,
    activity_class: np.ndarray,
    *,
    zero_sfr: np.ndarray | None = None,
) -> np.ndarray:
    """Within-class sSFR ranks with exact-zero SFR treated as censored low.

    TNG records genuinely zero SFR for some donors.  Those objects have no
    finite logarithmic sSFR, so they share the midpoint of a lowest-rank point
    mass instead of receiving an invented logarithmic value.
    """
    ssfr = np.asarray(logssfr, dtype=float)
    classes = np.asarray(activity_class).astype(str)
    zeros = (
        np.zeros(ssfr.shape, dtype=bool)
        if zero_sfr is None else np.asarray(zero_sfr, dtype=bool)
    )
    if ssfr.ndim != 1 or classes.shape != ssfr.shape or zeros.shape != ssfr.shape:
        raise ValueError("sSFR, activity, and zero-SFR arrays must be aligned")
    if np.any(~zeros & ~np.isfinite(ssfr)):
        raise ValueError("non-zero SFR donors require finite logarithmic sSFR")
    result = np.full(ssfr.shape, np.nan, dtype=np.float64)
    for label in np.unique(classes):
        indices = np.flatnonzero(classes == label)
        class_zeros = zeros[indices]
        n_total = indices.size
        n_zero = int(np.sum(class_zeros))
        if n_zero:
            result[indices[class_zeros]] = n_zero / (2.0 * n_total)
        finite_indices = indices[~class_zeros]
        if finite_indices.size:
            values = ssfr[finite_indices]
            ordered = np.sort(values, kind="mergesort")
            left = np.searchsorted(ordered, values, side="left")
            right = np.searchsorted(ordered, values, side="right")
            result[finite_indices] = (
                2.0 * n_zero + left + right
            ) / (2.0 * n_total)
    return result


def effective_sample_size(weights: np.ndarray) -> float:
    """Effective categorical donor count, ``1 / sum(p_j**2)``."""
    array = np.asarray(weights, dtype=float)
    total = float(np.sum(array))
    if array.ndim != 1 or not np.isfinite(array).all() or total <= 0.0:
        return 0.0
    probabilities = array / total
    return float(1.0 / np.sum(probabilities * probabilities))


def quantile_transport_weights(
    donor_quantiles: np.ndarray,
    target_quantile: float,
    *,
    bandwidth: float,
    minimum_effective_donors: int = MORPHOLOGY_MIN_EFFECTIVE_DONORS,
    balance_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, float, float]:
    """Gaussian rank-transport probabilities with an adaptive diversity floor.

    The cross-validated bandwidth is widened only when an edge draw would have
    too few effective donors.  Optional balance weights downweight donors that
    have already been used frequently by the current generator worker.
    """
    quantiles = np.asarray(donor_quantiles, dtype=float)
    target = float(target_quantile)
    initial = float(bandwidth)
    if (
        quantiles.ndim != 1 or not quantiles.size
        or not np.isfinite(quantiles).all()
        or not np.isfinite(target) or not 0.0 <= target <= 1.0
        or not np.isfinite(initial) or initial <= 0.0
    ):
        raise ValueError("invalid morphology quantile-transport inputs")
    if balance_weights is None:
        balance = np.ones(quantiles.size, dtype=np.float64)
    else:
        balance = np.asarray(balance_weights, dtype=float)
        if (
            balance.shape != quantiles.shape or not np.isfinite(balance).all()
            or np.any(balance <= 0.0)
        ):
            raise ValueError("invalid morphology donor balance weights")

    required = min(max(1, int(minimum_effective_donors)), quantiles.size)
    selected_bandwidth = initial
    for _ in range(64):
        distance = (quantiles - target) / selected_bandwidth
        weights = np.exp(-0.5 * distance * distance) * balance
        effective = effective_sample_size(weights)
        if effective >= required - 1e-9 or selected_bandwidth >= 4.0:
            break
        selected_bandwidth = min(4.0, selected_bandwidth * 1.25)
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("morphology quantile transport has zero probability")
    probabilities = weights / total
    return probabilities, float(selected_bandwidth), effective_sample_size(
        probabilities
    )


def joint_quantile_transport_weights(
    donor_mass_quantiles: np.ndarray,
    donor_ssfr_quantiles: np.ndarray,
    target_mass_quantile: float,
    target_ssfr_quantile: float,
    *,
    mass_bandwidth: float,
    ssfr_bandwidth: float,
    minimum_effective_donors: int = MORPHOLOGY_MIN_EFFECTIVE_DONORS,
    balance_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, float, float, float]:
    """Two-dimensional mass-sSFR rank transport with a diversity floor."""
    mass = np.asarray(donor_mass_quantiles, dtype=float)
    ssfr = np.asarray(donor_ssfr_quantiles, dtype=float)
    target_mass = float(target_mass_quantile)
    target_ssfr = float(target_ssfr_quantile)
    initial_mass = float(mass_bandwidth)
    initial_ssfr = float(ssfr_bandwidth)
    if (
        mass.ndim != 1 or not mass.size or ssfr.shape != mass.shape
        or not np.isfinite(mass).all() or not np.isfinite(ssfr).all()
        or not np.isfinite(target_mass) or not 0.0 <= target_mass <= 1.0
        or not np.isfinite(target_ssfr) or not 0.0 <= target_ssfr <= 1.0
        or not np.isfinite(initial_mass) or initial_mass <= 0.0
        or not np.isfinite(initial_ssfr) or initial_ssfr <= 0.0
    ):
        raise ValueError("invalid joint morphology quantile-transport inputs")
    if balance_weights is None:
        balance = np.ones(mass.size, dtype=np.float64)
    else:
        balance = np.asarray(balance_weights, dtype=float)
        if (
            balance.shape != mass.shape or not np.isfinite(balance).all()
            or np.any(balance <= 0.0)
        ):
            raise ValueError("invalid morphology donor balance weights")

    required = min(max(1, int(minimum_effective_donors)), mass.size)
    scale = 1.0
    for _ in range(64):
        used_mass = min(4.0, initial_mass * scale)
        used_ssfr = min(4.0, initial_ssfr * scale)
        distance2 = (
            ((mass - target_mass) / used_mass) ** 2
            + ((ssfr - target_ssfr) / used_ssfr) ** 2
        )
        weights = np.exp(-0.5 * distance2) * balance
        effective = effective_sample_size(weights)
        if effective >= required - 1e-9 or (
            used_mass >= 4.0 and used_ssfr >= 4.0
        ):
            break
        scale *= 1.25
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("joint morphology quantile transport has zero probability")
    probabilities = weights / total
    return (
        probabilities,
        float(used_mass),
        float(used_ssfr),
        effective_sample_size(probabilities),
    )


def _transfer_fingerprint(payload: dict, fit: dict) -> str:
    """Stable identity for coefficients plus the Euclid cone selection."""
    inputs = payload.get("inputs") or {}
    identity = {
        "version": 3,
        "fit_kind": "fixed_normalization",
        "fit": {
            key: fit.get(key)
            for key in (
                "vis_minus_f814w_mag", "magnitude_slope", "scatter_mag",
                "completeness_m50", "completeness_width_mag",
                "poisson_deviance", "dof",
            )
        },
        "euclid_cones": inputs.get("euclid_cones"),
        "euclid_cone_count": inputs.get("euclid_cone_count"),
        "euclid_area_arcmin2": inputs.get("euclid_area_arcmin2"),
        "classification_weighting": inputs.get("classification_weighting"),
    }
    encoded = json.dumps(
        identity, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def brightness_transfer_payload(path: str | Path) -> dict | None:
    """Return the fixed-normalization transfer candidate and quality flags."""
    fit_path = Path(path)
    try:
        payload = json.loads(fit_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    fit = payload.get("fit") or {}
    try:
        coefficients = {
            "offset_mag": float(fit["vis_minus_f814w_mag"]),
            "magnitude_slope": float(fit["magnitude_slope"]),
            "scatter_mag": max(0.0, float(fit["scatter_mag"])),
        }
    except (KeyError, TypeError, ValueError):
        return None
    warnings: list[str] = []
    if coefficients["scatter_mag"] >= 0.999:
        warnings.append("scatter reached the observation-fit upper bound")
    dof = max(1, int(fit.get("dof", 1)))
    reduced_deviance = float(fit.get("poisson_deviance", 0.0)) / dof
    if reduced_deviance > 5.0:
        warnings.append("fixed-normalization fit has high Poisson deviance")
    fingerprint = _transfer_fingerprint(payload, fit)
    return {
        "version": 3,
        "kind": "fixed_normalization",
        "valid": not warnings,
        "fingerprint": fingerprint,
        "coefficients": coefficients,
        "fit_quality": {
            "poisson_deviance": float(fit.get("poisson_deviance", 0.0)),
            "dof": dof,
            "reduced_poisson_deviance": reduced_deviance,
            "warnings": warnings,
            "valid": not warnings,
        },
        "observation_model": {
            "completeness_m50": float(fit.get("completeness_m50", 0.0)),
            "completeness_width_mag": float(
                fit.get("completeness_width_mag", 0.0)
            ),
        },
        "inputs": {
            key: (payload.get("inputs") or {}).get(key)
            for key in (
                "euclid_cone_count", "euclid_area_arcmin2", "euclid_cones",
                "classification_weighting",
            )
        },
        "source_fit": str(fit_path),
    }


def load_brightness_transfer(path: str | Path) -> F814WToVisTransfer:
    """Read the preferred fitted F814W→VIS transfer from an analysis artifact.

    The returned object is self-contained so callers submitting work to a
    different machine can serialize its coefficients instead of relying on
    the artifact being present at the same path remotely.
    """
    transfer = brightness_transfer_payload(path)
    if transfer is None:
        raise ValueError(
            f"missing or malformed fitted F814W→VIS transfer: {path}"
        )
    coefficients = transfer["coefficients"]
    try:
        return F814WToVisTransfer(
            offset_mag=float(coefficients["offset_mag"]),
            magnitude_slope=float(coefficients["magnitude_slope"]),
            scatter_mag=float(coefficients["scatter_mag"]),
            source=(
                "fixed_normalization_fit:"
                f"{transfer['fingerprint']}:{Path(path)}"
            ),
            fingerprint=str(transfer["fingerprint"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid fitted F814W→VIS transfer: {path}") from exc


class CosmosTngPrior:
    """Memory-resident COSMOS physical/brightness sampler."""

    def __init__(
        self,
        path: str | Path,
        *,
        photometric_fit_path: str | Path = Config.COSMOS_EUCLID_FIT_PATH,
        photometric_transfer: F814WToVisTransfer | None = None,
        mag_min: float | None = None,
        mag_max: float | None = None,
    ):
        self.path = str(path)
        with np.load(self.path, allow_pickle=False) as data:
            keys = set(data.files)

            def take(*names: str) -> np.ndarray:
                for name in names:
                    if name in keys:
                        return np.asarray(data[name])
                raise KeyError(f"{self.path} has none of {names!r}")

            catalog_id = take("catalog_id")
            # Old artifacts are accepted for replay, but all newly extracted
            # files use the physical filter name.
            f814w = take("mag_hst_f814w", "mag_vis", "mag_VIS")
            z = take("z_phot")
            mass = take("logmass_lephare", "logmass")
            logssfr = take("logssfr_lephare")
            re = take("re_combined_arcsec", "disk_re_arcsec")
            if "generator_ready" not in keys:
                raise ValueError(
                    f"{self.path} predates the strict generator-ready schema"
                )
            generator_ready = np.asarray(data["generator_ready"], dtype=bool)

        valid = (
            np.isfinite(f814w)
            & np.isfinite(z) & (z > 0.01) & (z < 6.0)
            & np.isfinite(mass) & (mass > 4.0) & (mass < 13.0)
            & np.isfinite(logssfr)
            & generator_ready
            & np.isfinite(re) & (re > 0.01) & (re < 20.0)
        )
        if mag_min is not None:
            valid &= f814w >= float(mag_min)
        if mag_max is not None:
            valid &= f814w < float(mag_max)
        if not np.any(valid):
            raise ValueError(f"No usable COSMOS physical rows in {self.path}")
        self.catalog_id = catalog_id[valid].astype(str)
        self.f814w = f814w[valid].astype(np.float32)
        self.z = z[valid].astype(np.float32)
        self.mass = mass[valid].astype(np.float32)
        self.logssfr = logssfr[valid].astype(np.float32)
        self.re = re[valid].astype(np.float32)
        if not len(self.re):
            raise ValueError(f"COSMOS prior lacks valid generator-ready sizes: {self.path}")
        self.brightness_transfer = (
            photometric_transfer
            if photometric_transfer is not None
            else load_brightness_transfer(photometric_fit_path)
        )
        self.activity_class = np.where(
            self.logssfr < MORPHOLOGY_ACTIVITY_THRESHOLD_LOGSSFR,
            "quenched", "star_forming",
        )
        self.mass_quantile = conditional_mass_quantiles(
            self.mass, self.activity_class,
        )
        self.ssfr_quantile = conditional_ssfr_quantiles(
            self.logssfr, self.activity_class,
        )

    def __len__(self) -> int:
        return len(self.f814w)

    def mass_support_indices(
        self, lower_logmass: float, upper_logmass: float,
    ) -> np.ndarray:
        """Indices whose stellar masses are supported by the local atlas."""
        lower = float(lower_logmass)
        upper = float(upper_logmass)
        if not (np.isfinite(lower) and np.isfinite(upper) and lower <= upper):
            raise ValueError("TNG morphology mass support is invalid")
        indices = np.flatnonzero((self.mass >= lower) & (self.mass <= upper))
        if not indices.size:
            raise ValueError(
                "COSMOS prior has no rows inside the TNG morphology mass support"
            )
        return indices

    def sample(
        self,
        rng: np.random.Generator,
        *,
        eligible_indices: np.ndarray | None = None,
    ) -> CosmosTngDraw:
        if eligible_indices is None:
            i = int(rng.integers(0, len(self)))
        else:
            eligible = np.asarray(eligible_indices, dtype=np.int64)
            if eligible.ndim != 1 or not eligible.size:
                raise ValueError("COSMOS eligible-index pool is empty")
            i = int(eligible[int(rng.integers(0, eligible.size))])
            if i < 0 or i >= len(self):
                raise ValueError("COSMOS eligible-index pool is out of bounds")
        re = float(self.re[i])
        mag_hst_f814w = float(self.f814w[i])
        target_vis_mag = self.brightness_transfer.sample_vis_mag(
            mag_hst_f814w, rng
        )
        return CosmosTngDraw(
            catalog_id=str(self.catalog_id[i]),
            mag_hst_f814w=mag_hst_f814w,
            target_vis_mag=target_vis_mag,
            target_vis_flux_e=float(ab_mag_to_electrons(
                target_vis_mag, Config.get_band("VIS")
            )),
            z=float(self.z[i]),
            logmass=float(self.mass[i]),
            re_arcsec=re,
            imputed_size=False,
            brightness_transfer=self.brightness_transfer.source,
            mass_quantile=float(self.mass_quantile[i]),
            ssfr_quantile=float(self.ssfr_quantile[i]),
            activity_class=str(self.activity_class[i]),
            logssfr=float(self.logssfr[i]),
        )

    def proxy_logmass(self, quantile: float, activity_class: str) -> float:
        """Map a donor rank onto the corresponding COSMOS conditional mass."""
        values = self.mass[self.activity_class == str(activity_class)]
        if not values.size:
            raise ValueError(
                f"COSMOS prior has no {activity_class!r} morphology population"
            )
        return float(np.quantile(values.astype(np.float64), float(quantile)))


class JointGalaxyPopulationPrior:
    """Minimal Euclid joint prior: radius first, brightness given radius."""

    morphology_mode = "balanced_random_tng_atlas"
    population_label = "euclid_vis2fwhm_sersic_re_joint_v4"

    def __init__(self, payload: dict):
        if payload.get("version") != JOINT_EUCLID_GALAXY_VERSION:
            raise ValueError("joint galaxy population has an unsupported version")
        if payload.get("kind") != "euclid_vis2fwhm_sersic_re_joint":
            raise ValueError("joint galaxy population has the wrong kind")
        if not payload.get("active") or not payload.get("valid"):
            raise ValueError("joint galaxy population is not active and valid")
        self.fingerprint = str(payload.get("fingerprint") or "")
        if len(self.fingerprint) != 64:
            raise ValueError("joint galaxy population fingerprint is invalid")
        try:
            self.magnitude_law = StraightMagnitudeLaw.from_payload(
                payload["magnitude_law"]
            )
            self.radius_law = ConditionalRadiusLaw.from_payload(
                payload["radius_law"]
            )
            expected = float(payload["generation"]["surface_density_arcmin2"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("joint galaxy population model is incomplete") from exc
        if not np.isclose(self.magnitude_law.integrated_density(), expected):
            raise ValueError("joint galaxy population density does not match activation")
        grid = joint_density_grid(self.magnitude_law, self.radius_law)
        self._density = np.asarray(grid["density"], dtype=np.float64)
        self._magnitude_edges = np.asarray(grid["magnitude_edges"])
        self._log_radius_edges = np.asarray(grid["log_radius_edges"])
        radius_weight = np.sum(self._density, axis=0)
        self._radius_cdf = np.cumsum(radius_weight) / np.sum(radius_weight)
        self.surface_density_arcmin2 = self.magnitude_law.integrated_density()

    def proxy_logmass(self, quantile: float, activity_class: str) -> float:
        return float("nan")

    def __len__(self) -> int:
        return int(self._density.size)

    def sample_geometry(self, rng: np.random.Generator) -> CosmosTngDraw:
        """Draw the Euclid Sérsic-radius marginal; brightness remains unset."""
        ri = min(
            int(np.searchsorted(self._radius_cdf, rng.random(), side="right")),
            self._radius_cdf.size - 1,
        )
        log_radius = float(rng.uniform(
            self._log_radius_edges[ri], self._log_radius_edges[ri + 1],
        ))
        return CosmosTngDraw(
            catalog_id=f"euclid-joint:{self.fingerprint[:12]}:radius:{ri}",
            mag_hst_f814w=float("nan"),
            target_vis_mag=float("nan"),
            target_vis_flux_e=float("nan"),
            z=float("nan"),
            logmass=float("nan"),
            re_arcsec=float(10.0 ** log_radius),
            imputed_size=False,
            brightness_transfer=f"euclid_joint:{self.fingerprint}:activated",
            mass_quantile=float("nan"),
            ssfr_quantile=float("nan"),
            activity_class="unconditioned",
            logssfr=float("nan"),
            physical_model_fingerprint="",
        )

    def sample_brightness(
        self, rng: np.random.Generator, *, radius_arcsec: float | None = None,
    ) -> tuple[float, float]:
        """Draw VIS 2FWHM brightness conditional on the selected radius."""
        if radius_arcsec is None:
            magnitude = self.magnitude_law.sample(rng)
        else:
            ri = int(np.clip(
                np.searchsorted(
                    self._log_radius_edges, np.log10(radius_arcsec),
                    side="right",
                ) - 1,
                0, self._density.shape[1] - 1,
            ))
            probability = self._density[:, ri].copy()
            probability /= np.sum(probability)
            mi = int(rng.choice(probability.size, p=probability))
            magnitude = float(rng.uniform(
                self._magnitude_edges[mi], self._magnitude_edges[mi + 1],
            ))
        return magnitude, float(ab_mag_to_electrons(
            magnitude, Config.get_band("VIS"),
        ))

    def sample(self, rng: np.random.Generator) -> CosmosTngDraw:
        geometry = self.sample_geometry(rng)
        magnitude, flux = self.sample_brightness(
            rng, radius_arcsec=geometry.re_arcsec,
        )
        return replace(
            geometry, target_vis_mag=magnitude, target_vis_flux_e=flux,
        )
