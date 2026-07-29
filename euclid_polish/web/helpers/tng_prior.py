"""Data-derived normalization diagnostics for the synthetic TNG population."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from euclid_polish.config import Config

DETECTION_SIGMA = 4.0
DETECTION_NPIXELS = 8
DETECTION_BOX_SIZE = 32
DETECTION_DEBLEND_LEVELS = 24
DETECTION_DEBLEND_CONTRAST = 0.003
TRUTH_MATCH_RADIUS_LR_PIX = 4.0
CATALOG_MAG_BIN = 0.5
BOOTSTRAPS = 1000


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _segment_centroids(
    plane: np.ndarray,
) -> tuple[list[tuple[float, float]], int]:
    """Detect positive sources and matched-significance negative islands."""
    from astropy.stats import SigmaClip
    from photutils.background import Background2D, MedianBackground
    from photutils.segmentation import (
        SourceCatalog,
        deblend_sources,
        detect_sources,
    )

    data = np.asarray(plane, dtype=np.float64)
    background = Background2D(
        data,
        (DETECTION_BOX_SIZE, DETECTION_BOX_SIZE),
        filter_size=(3, 3),
        sigma_clip=SigmaClip(sigma=3.0, maxiters=5),
        bkg_estimator=MedianBackground(),
    )
    residual = data - background.background
    threshold = DETECTION_SIGMA * background.background_rms

    def segments(sign: int):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            segmentation = detect_sources(
                sign * residual,
                threshold,
                npixels=DETECTION_NPIXELS,
            )
            if segmentation is None:
                return None
            return deblend_sources(
                sign * residual,
                segmentation,
                npixels=DETECTION_NPIXELS,
                nlevels=DETECTION_DEBLEND_LEVELS,
                contrast=DETECTION_DEBLEND_CONTRAST,
                progress_bar=False,
            )

    positive = segments(1)
    negative = segments(-1)
    if positive is None:
        return [], 0 if negative is None else int(negative.nlabels)
    catalog = SourceCatalog(residual, positive)
    centroids = [
        (float(x), float(y))
        for x, y in zip(catalog.xcentroid, catalog.ycentroid, strict=True)
        if np.isfinite(x) and np.isfinite(y)
    ]
    return centroids, 0 if negative is None else int(negative.nlabels)


def _match_truth(
    centroids: list[tuple[float, float]],
    truth: list[dict[str, Any]],
    shape: tuple[int, int],
) -> tuple[int, int, int]:
    """Greedily match unique detections to unique HR source positions."""
    scale = float(Config.DEFAULT_PIXEL_SCALE / Config.VIS_PIXEL_SCALE_ARCSEC)
    height, width = shape
    candidates: list[tuple[float, int, int]] = []
    valid_truth: list[dict[str, Any]] = []
    for row in truth:
        x = _finite(row.get("x_pix"))
        y = _finite(row.get("y_pix"))
        if x is None or y is None:
            continue
        x *= scale
        y *= scale
        if not (0 <= x < width and 0 <= y < height):
            continue
        truth_index = len(valid_truth)
        valid_truth.append(row)
        for detection_index, (det_x, det_y) in enumerate(centroids):
            distance = float(np.hypot(det_x - x, det_y - y))
            if distance <= TRUTH_MATCH_RADIUS_LR_PIX:
                candidates.append((distance, detection_index, truth_index))

    used_detections: set[int] = set()
    used_truth: set[int] = set()
    matched_galaxies = 0
    matched_stars = 0
    for _distance, detection_index, truth_index in sorted(candidates):
        if detection_index in used_detections or truth_index in used_truth:
            continue
        used_detections.add(detection_index)
        used_truth.add(truth_index)
        kind = str(valid_truth[truth_index].get("type", "unknown"))
        if kind == "star":
            matched_stars += 1
        elif kind in {"galaxy", "lens"}:
            matched_galaxies += 1
    truth_galaxies = sum(
        str(row.get("type", "unknown")) in {"galaxy", "lens"}
        for row in valid_truth
    )
    return matched_galaxies, matched_stars, truth_galaxies


@dataclass
class DetectionAccumulator:
    """Streaming per-field VIS detections, optionally matched to truth."""

    positive: list[int] = field(default_factory=list)
    negative: list[int] = field(default_factory=list)
    matched_galaxies: list[int] = field(default_factory=list)
    matched_stars: list[int] = field(default_factory=list)
    truth_galaxies: list[int] = field(default_factory=list)

    def add(
        self,
        vis_plane: np.ndarray,
        truth: list[dict[str, Any]] | None = None,
    ) -> None:
        centroids, negative = _segment_centroids(vis_plane)
        self.positive.append(len(centroids))
        self.negative.append(negative)
        if truth is None:
            self.matched_galaxies.append(0)
            self.matched_stars.append(0)
            self.truth_galaxies.append(0)
            return
        galaxies, stars, truth_count = _match_truth(
            centroids, truth, np.asarray(vis_plane).shape
        )
        self.matched_galaxies.append(galaxies)
        self.matched_stars.append(stars)
        self.truth_galaxies.append(truth_count)

    def payload(self) -> dict[str, Any]:
        return {
            "positive": self.positive,
            "negative": self.negative,
            "matched_galaxies": self.matched_galaxies,
            "matched_stars": self.matched_stars,
            "truth_galaxies": self.truth_galaxies,
        }


def detection_payload(
    synthetic: DetectionAccumulator,
    real: DetectionAccumulator,
) -> dict[str, Any]:
    return {
        "settings": {
            "band": "VIS",
            "threshold_sigma": DETECTION_SIGMA,
            "minimum_connected_pixels": DETECTION_NPIXELS,
            "background_box_pixels": DETECTION_BOX_SIZE,
            "deblend_levels": DETECTION_DEBLEND_LEVELS,
            "deblend_contrast": DETECTION_DEBLEND_CONTRAST,
            "negative_image_correction": True,
            "truth_match_radius_lr_pixels": TRUTH_MATCH_RADIUS_LR_PIX,
        },
        "synthetic": synthetic.payload(),
        "real": real.payload(),
    }


def _interval(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    return {
        "median": float(np.median(finite)),
        "p16": float(np.percentile(finite, 16)),
        "p84": float(np.percentile(finite, 84)),
    }


def _poisson_deviance(observed: np.ndarray, expected: np.ndarray) -> float:
    valid = expected > 0
    observed = observed[valid].astype(np.float64)
    expected = expected[valid].astype(np.float64)
    terms = expected - observed
    positive = observed > 0
    terms[positive] += observed[positive] * np.log(
        observed[positive] / expected[positive]
    )
    return float(2.0 * np.sum(terms))


def catalog_prior_estimate(
    synthetic_rows: list[dict[str, Any]],
    euclid_rows: list[dict[str, Any]],
    synthetic_area_arcmin2: float,
    euclid_area_arcmin2: float,
    *,
    current_prior: float = Config.TNG_GAL_DENSITY_ARCMIN2,
    seed: int = 71029,
) -> dict[str, Any] | None:
    """Fit a scalar prior over the empirical Euclid count-turnover range."""
    synthetic_mags = np.asarray([
        value for row in synthetic_rows
        if str(row.get("type")) == "galaxy"
        if (value := _finite(row.get("mag_vis"))) is not None
    ])
    euclid_mags = np.asarray([
        value for row in euclid_rows
        if str(row.get("type")) != "star"
        if (value := _finite(row.get("mag_vis"))) is not None
    ])
    if (
        not synthetic_mags.size or not euclid_mags.size
        or synthetic_area_arcmin2 <= 0 or euclid_area_arcmin2 <= 0
    ):
        return None

    lower = math.floor(min(synthetic_mags.min(), euclid_mags.min()) * 2) / 2
    upper = math.ceil(max(synthetic_mags.max(), euclid_mags.max()) * 2) / 2
    edges = np.arange(lower, upper + CATALOG_MAG_BIN, CATALOG_MAG_BIN)
    centers = (edges[:-1] + edges[1:]) / 2
    synthetic_counts, _ = np.histogram(synthetic_mags, bins=edges)
    euclid_counts, _ = np.histogram(euclid_mags, bins=edges)
    if euclid_counts.size < 3:
        return None
    smoothed_real = np.convolve(
        euclid_counts.astype(float), [0.25, 0.5, 0.25], mode="same"
    )
    peak_index = int(np.argmax(smoothed_real))
    turnover_limit = float(edges[peak_index + 1])
    selected = (
        (centers < turnover_limit)
        & (synthetic_counts >= 10)
        & (euclid_counts >= 5)
    )
    if np.count_nonzero(selected) < 2:
        return None

    synthetic_density = synthetic_counts / synthetic_area_arcmin2
    euclid_density = euclid_counts / euclid_area_arcmin2
    scale = (
        float(np.sum(euclid_counts[selected]) / euclid_area_arcmin2)
        / float(np.sum(synthetic_counts[selected]) / synthetic_area_arcmin2)
    )
    fitted_prior = current_prior * scale
    expected_real = (
        scale * synthetic_density[selected] * euclid_area_arcmin2
    )
    deviance = _poisson_deviance(euclid_counts[selected], expected_real)
    degrees_of_freedom = max(int(np.count_nonzero(selected)) - 1, 1)
    reduced_deviance = deviance / degrees_of_freedom

    per_bin_prior = np.full_like(centers, np.nan, dtype=float)
    ratio_valid = selected & (synthetic_density > 0)
    per_bin_prior[ratio_valid] = (
        current_prior
        * euclid_density[ratio_valid]
        / synthetic_density[ratio_valid]
    )
    slope = float(np.polyfit(
        centers[ratio_valid],
        np.log10(per_bin_prior[ratio_valid]),
        1,
    )[0])

    rng = np.random.default_rng(seed)
    boot = np.full(BOOTSTRAPS, np.nan)
    for index in range(BOOTSTRAPS):
        synthetic_draw = rng.poisson(synthetic_counts[selected])
        euclid_draw = rng.poisson(euclid_counts[selected])
        synthetic_total = int(np.sum(synthetic_draw))
        if synthetic_total > 0:
            boot[index] = current_prior * (
                (np.sum(euclid_draw) / euclid_area_arcmin2)
                / (synthetic_total / synthetic_area_arcmin2)
            )

    return {
        "method": "shared VIS catalog counts before empirical Euclid turnover",
        "current_prior_arcmin2": float(current_prior),
        "fitted_prior_arcmin2": float(fitted_prior),
        "interval_arcmin2": _interval(boot),
        "turnover_limit_mag": turnover_limit,
        "selected_bin_count": int(np.count_nonzero(selected)),
        "synthetic_selected_count": int(np.sum(synthetic_counts[selected])),
        "euclid_selected_count": int(np.sum(euclid_counts[selected])),
        "reduced_poisson_deviance": float(reduced_deviance),
        "log10_prior_slope_per_mag": slope,
        "single_scalar_adequate": bool(
            reduced_deviance < 2.0 and abs(slope) < 0.1
        ),
        "curve": {
            "mag": centers[ratio_valid].astype(float).tolist(),
            "prior_arcmin2": per_bin_prior[ratio_valid].astype(float).tolist(),
            "synthetic_density": synthetic_density[ratio_valid].astype(float).tolist(),
            "euclid_density": euclid_density[ratio_valid].astype(float).tolist(),
        },
        "uncertainty_note": (
            "16–84% interval is Poisson-only; one Euclid cone does not measure "
            "field-to-field cosmic variance"
        ),
    }


def visible_prior_estimate(
    source_detection: dict[str, Any] | None,
    euclid_rows: list[dict[str, Any]],
    euclid_area_arcmin2: float,
    field_area_arcmin2: float,
    *,
    current_prior: float = Config.TNG_GAL_DENSITY_ARCMIN2,
    seed: int = 71030,
) -> dict[str, Any] | None:
    """Infer the prior from common positive-minus-negative VIS detections."""
    if source_detection is None or euclid_area_arcmin2 <= 0:
        return None
    synthetic = source_detection.get("synthetic", {})
    real = source_detection.get("real", {})
    synthetic_positive = np.asarray(synthetic.get("positive", []), dtype=float)
    synthetic_negative = np.asarray(synthetic.get("negative", []), dtype=float)
    synthetic_stars = np.asarray(
        synthetic.get("matched_stars", []), dtype=float
    )
    real_positive = np.asarray(real.get("positive", []), dtype=float)
    real_negative = np.asarray(real.get("negative", []), dtype=float)
    if not synthetic_positive.size or not real_positive.size:
        return None
    synthetic_net = np.maximum(
        synthetic_positive - synthetic_negative - synthetic_stars, 0.0
    )
    real_net = np.maximum(real_positive - real_negative, 0.0)
    star_count = sum(str(row.get("type")) == "star" for row in euclid_rows)
    real_star_density = star_count / euclid_area_arcmin2
    synthetic_density = float(np.mean(synthetic_net) / field_area_arcmin2)
    real_density = max(
        float(np.mean(real_net) / field_area_arcmin2) - real_star_density,
        0.0,
    )
    if synthetic_density <= 0:
        return None
    fitted_prior = current_prior * real_density / synthetic_density

    rng = np.random.default_rng(seed)
    boot = np.full(BOOTSTRAPS, np.nan)
    for index in range(BOOTSTRAPS):
        synthetic_draw = rng.choice(
            synthetic_net, size=synthetic_net.size, replace=True
        )
        real_draw = rng.choice(real_net, size=real_net.size, replace=True)
        sampled_star_density = (
            rng.poisson(star_count) / euclid_area_arcmin2
        )
        synthetic_boot_density = float(
            np.mean(synthetic_draw) / field_area_arcmin2
        )
        real_boot_density = max(
            float(np.mean(real_draw) / field_area_arcmin2)
            - sampled_star_density,
            0.0,
        )
        if synthetic_boot_density > 0:
            boot[index] = (
                current_prior * real_boot_density / synthetic_boot_density
            )

    truth_galaxies = np.asarray(
        synthetic.get("truth_galaxies", []), dtype=float
    )
    matched_galaxies = np.asarray(
        synthetic.get("matched_galaxies", []), dtype=float
    )
    truth_total = float(np.sum(truth_galaxies))
    truth_density = float(np.mean(truth_galaxies) / field_area_arcmin2)
    return {
        "method": (
            "same 4σ, 8-connected-pixel VIS detector; negative-image islands "
            "and catalog star density removed"
        ),
        "current_prior_arcmin2": float(current_prior),
        "fitted_prior_arcmin2": float(fitted_prior),
        "interval_arcmin2": _interval(boot),
        "synthetic_detected_density_arcmin2": synthetic_density,
        "real_detected_density_arcmin2": real_density,
        "synthetic_retained_truth_density_arcmin2": truth_density,
        "matched_truth_fraction": (
            float(np.sum(matched_galaxies) / truth_total)
            if truth_total > 0 else math.nan
        ),
        "synthetic_fields": int(synthetic_net.size),
        "real_fields": int(real_net.size),
        "settings": source_detection.get("settings", {}),
        "caveat": (
            "The ratio assumes detected density scales linearly with the raw "
            "draw prior; crowding and deblending require a small prior sweep."
        ),
    }


def tng_prior_payload(
    synthetic_rows: list[dict[str, Any]],
    euclid_rows: list[dict[str, Any]],
    synthetic_area_arcmin2: float,
    euclid_area_arcmin2: float,
    field_area_arcmin2: float,
    source_detection: dict[str, Any] | None,
    *,
    dataset_prior: float,
    configured_prior: float = Config.TNG_GAL_DENSITY_ARCMIN2,
) -> dict[str, Any] | None:
    catalog = catalog_prior_estimate(
        synthetic_rows,
        euclid_rows,
        synthetic_area_arcmin2,
        euclid_area_arcmin2,
        current_prior=dataset_prior,
    )
    visible = visible_prior_estimate(
        source_detection,
        euclid_rows,
        euclid_area_arcmin2,
        field_area_arcmin2,
        current_prior=dataset_prior,
    )
    if catalog is None and visible is None:
        return None

    estimates = [
        estimate["fitted_prior_arcmin2"]
        for estimate in (catalog, visible)
        if estimate is not None
    ]
    pilot_grid: list[int] = []
    if estimates:
        low = min(estimates)
        high = max(estimates)
        pilot_grid = sorted({
            int(round(low / 20.0) * 20),
            int(round(((low + high) / 2.0) / 20.0) * 20),
            int(round(high / 20.0) * 20),
        })
    scalar_adequate = bool(
        catalog is not None and catalog["single_scalar_adequate"]
    )
    return {
        "catalog": catalog,
        "visible": visible,
        "dataset_prior_arcmin2": float(dataset_prior),
        "configured_prior_arcmin2": float(configured_prior),
        "configured_mf_alpha": float(Config.TNG_MF_ALPHA),
        "single_scalar_adequate": scalar_adequate,
        "pilot_grid_arcmin2": pilot_grid,
        "recommendation": (
            "Fit the luminosity/redshift population as well as normalization; "
            "the magnitude-bin ratios reject one global density scalar."
            if not scalar_adequate
            else "A single normalization is consistent over the selected bins."
        ),
    }
