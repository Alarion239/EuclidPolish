"""Compact, plot-ready galaxy population marginals for the WebUI."""

from __future__ import annotations

import copy
import csv
import hashlib
import heapq
import json
import math
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import contourpy
import numpy as np
from scipy.ndimage import gaussian_filter

from euclid_polish.config import Config
from euclid_polish.photometry import electrons_to_ab_mag, uJy_to_ab_mag
from euclid_polish.population.euclid_galaxy_prior import (
    ConditionalRadiusLaw,
    joint_density_grid,
)
from euclid_polish.population.joint_galaxy import (
    COSMOS_AREA_ARCMIN2,
    COSMOS_FIT_MAG_MIN,
    COSMOS_FIT_Z_MAX,
    COSMOS_FIT_Z_MIN,
    read_cosmos_population,
)
from euclid_polish.population.magnitude_law import (
    ContinuousBrightBridgeFaintCappedMagnitudeLaw,
)
from euclid_polish.web.helpers.population_calibration import (
    joint_galaxy_candidate,
    joint_galaxy_candidate_path,
    joint_galaxy_state,
)
from euclid_polish.web.helpers.population_comparison import (
    FIELD_AREA_ARCMIN2,
    _synthetic_paths,
    euclid_catalog_meta_path,
    euclid_catalog_path,
    euclid_phz_pdf_path,
    read_phz_pdf_cache,
)
from euclid_polish.web.helpers.q1_galaxy_counts import (
    q1_galaxy_counts_path,
    q1_galaxy_fit_path,
    read_q1_galaxy_aperture_counts,
    read_q1_galaxy_aperture_fit,
)
from euclid_polish.web.helpers.q1_galaxy_radius_statistics import (
    q1_galaxy_radius_statistics_path,
    read_q1_galaxy_radius_statistics,
)

ARTIFACT_VERSION = 19
MAG_EDGES = np.arange(14.0, 30.0001, 0.25)
RADIUS_MAX_VIS_PIXELS = 100.0
RADIUS_MAX_ARCSEC = RADIUS_MAX_VIS_PIXELS * float(Config.VIS_PIXEL_SCALE_ARCSEC)
LOG_RADIUS_EDGES = np.arange(
    -2.4, np.log10(RADIUS_MAX_ARCSEC) + 0.0001, 0.10,
)
MASS_EDGES = np.arange(7.0, 13.0001, 0.20)
SSFR_EDGES = np.arange(-14.0, -8.1999, 0.20)
APERTURE_SIZE_EDGES = np.asarray([
    0.0, 0.25, 0.50, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0,
])
APERTURE_DELTA_MAG_EDGES = np.linspace(-1.0, 4.0, 251)
APERTURE_SCATTER_MAG_EDGES = np.arange(14.0, 30.0001, 0.5)
APERTURE_SCATTER_PER_MAG_BIN = 250
JOINT_CONTOUR_MASS_FRACTIONS = (
    0.995, 0.99, 0.95, 0.80, 0.50, 0.25, 0.10,
)

MER_BRIGHTNESS_SERIES = {
    "mer_vis_1fwhm": (
        "flux_vis_1fwhm_aper_uJy", "VIS · 1 FWHM", "1-FWHM diameter aperture",
    ),
    "mer_vis_2fwhm": (
        "flux_vis_2fwhm_aper_uJy", "VIS · 2 FWHM", "2-FWHM diameter aperture",
    ),
    "mer_vis_3fwhm": (
        "flux_vis_3fwhm_aper_uJy", "VIS · 3 FWHM", "3-FWHM diameter aperture",
    ),
    "mer_vis_4fwhm": (
        "flux_vis_4fwhm_aper_uJy", "VIS · 4 FWHM", "4-FWHM diameter aperture",
    ),
    "mer_vis_kron": (
        "flux_detection_total_uJy", "VIS detection · Kron", "Kron total aperture",
    ),
    "mer_vis_sersic": (
        "flux_vis_sersic_uJy", "VIS · S\u00e9rsic", "S\u00e9rsic-model total",
    ),
}

COSMOS_APERTURE_DIAMETERS_ARCSEC = {
    "native": (0.1, 0.25, 0.5, 1.0, 1.5),
    "homogenized": (0.2, 0.3, 0.5, 0.75, 1.0),
}


def artifact_path() -> Path:
    return Path(Config.DATA_DIR) / "population_comparison" / "galaxy_distributions.json"


def _signature(path: Path) -> dict[str, int] | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return {"size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def _inputs() -> dict[str, Any]:
    _records, synthetic_sources = _synthetic_paths()
    _, synthetic_sources_with_training = _synthetic_paths(
        include_training=True,
    )
    training_sources = [
        path for path in synthetic_sources_with_training
        if path.stem == "sources_train"
    ]
    synthetic_clean = [
        path.with_name(path.name.replace("sources_", "clean_").replace(".csv", ".tfrecord"))
        for path in synthetic_sources
    ]
    return {
        "euclid_csv": _signature(euclid_catalog_path()),
        "euclid_meta": _signature(euclid_catalog_meta_path()),
        "euclid_phz_pdf": _signature(euclid_phz_pdf_path()),
        "q1_galaxy_counts": _signature(q1_galaxy_counts_path()),
        "q1_galaxy_fit": _signature(q1_galaxy_fit_path()),
        "q1_galaxy_radius": _signature(q1_galaxy_radius_statistics_path()),
        "joint_galaxy_candidate": _signature(joint_galaxy_candidate_path()),
        "cosmos": _signature(Path(Config.COSMOS_POPULATION_PRIOR_PATH)),
        "fit": _signature(Path(Config.JOINT_GALAXY_POPULATION_FIT_PATH)),
        "synthetic_sources": {
            path.name: _signature(path) for path in synthetic_sources
        },
        "synthetic_training_sources": {
            path.name: _signature(path) for path in training_sources
        },
        "synthetic_clean": {
            path.name: _signature(path) for path in synthetic_clean
        },
    }


def _json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _curve(edges: np.ndarray, counts: np.ndarray, area: float, definition: str) -> dict[str, Any]:
    width = np.diff(edges)
    density = np.divide(
        np.asarray(counts, dtype=np.float64),
        area * width,
        out=np.zeros_like(width),
        where=width > 0,
    )
    return {
        "x": (0.5 * (edges[:-1] + edges[1:])).tolist(),
        "density": density.tolist(),
        "weighted_count": float(np.sum(counts)),
        "definition": definition,
    }


def _normalized_density(
    density: np.ndarray, bin_width: np.ndarray,
) -> np.ndarray:
    """Normalize a non-negative per-unit density to unit integrated mass."""
    values = np.asarray(density, dtype=np.float64)
    width = np.asarray(bin_width, dtype=np.float64)
    if (
        values.ndim != 1
        or values.shape != width.shape
        or not np.all(np.isfinite(values) & (values >= 0.0))
        or not np.all(np.isfinite(width) & (width > 0.0))
    ):
        raise ValueError("radius density cannot be normalized")
    total = float(np.sum(values * width))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("radius density cannot be normalized")
    return values / total


def _mass_contour_thresholds(
    density: np.ndarray,
    cell_mass: np.ndarray,
) -> list[tuple[float, float]]:
    """Density thresholds enclosing fixed fractions of joint population mass."""
    values = np.asarray(density, dtype=np.float64).ravel()
    mass = np.asarray(cell_mass, dtype=np.float64).ravel()
    keep = (
        np.isfinite(values) & np.isfinite(mass)
        & (values > 0.0) & (mass > 0.0)
    )
    values, mass = values[keep], mass[keep]
    if values.size < 2 or float(np.sum(mass)) <= 0.0:
        return []
    order = np.argsort(values)[::-1]
    cumulative = np.cumsum(mass[order]) / np.sum(mass)
    return [
        (
            fraction,
            float(values[order[min(
                int(np.searchsorted(cumulative, fraction)),
                len(order) - 1,
            )]]),
        )
        for fraction in JOINT_CONTOUR_MASS_FRACTIONS
    ]


def _joint_contours(
    density: np.ndarray,
    cell_mass: np.ndarray,
    magnitude_center: np.ndarray,
    log_radius_center: np.ndarray,
) -> list[dict[str, Any]]:
    """Trace plot-ready 10/25/50/80/95/99/99.5-percent mass contours."""
    generator = contourpy.contour_generator(
        x=np.asarray(magnitude_center, dtype=np.float64),
        y=np.asarray(log_radius_center, dtype=np.float64),
        z=np.asarray(density, dtype=np.float64).T,
        corner_mask=True,
    )
    contours = []
    seen: set[float] = set()
    for mass_fraction, level in _mass_contour_thresholds(density, cell_mass):
        # Sparse histograms can assign more than one enclosed-mass fraction to
        # the same density threshold.  One geometric line is sufficient; its
        # label retains every represented fraction.
        rounded = round(level, 12)
        if rounded in seen:
            continue
        seen.add(rounded)
        paths = []
        for vertices in generator.lines(level):
            if vertices.shape[0] < 2:
                continue
            paths.append({
                "x": vertices[:, 0].astype(float).tolist(),
                "y": vertices[:, 1].astype(float).tolist(),
            })
        if paths:
            contours.append({
                "mass_fraction": mass_fraction,
                "level": level,
                "paths": paths,
            })
    return contours


def _joint_map(
    *,
    key: str,
    label: str,
    detail: str,
    color: str,
    magnitude_edges: np.ndarray,
    log_radius_edges: np.ndarray,
    cell_mass_arcmin2: np.ndarray,
    rows: int | None = None,
    contour_smoothing_sigma_bins: float = 0.0,
) -> dict[str, Any]:
    """Convert one joint population histogram to a common map contract."""
    mass = np.asarray(cell_mass_arcmin2, dtype=np.float64)
    expected_shape = (
        magnitude_edges.size - 1,
        log_radius_edges.size - 1,
    )
    if mass.shape != expected_shape or not np.all(
        np.isfinite(mass) & (mass >= 0.0)
    ):
        raise ValueError(f"{key} joint magnitude-radius grid is malformed")
    magnitude_width = np.diff(magnitude_edges)
    log_radius_width = np.diff(log_radius_edges)
    density = np.divide(
        mass,
        magnitude_width[:, None] * log_radius_width[None, :],
        out=np.zeros_like(mass),
        where=(
            magnitude_width[:, None] * log_radius_width[None, :]
        ) > 0.0,
    )
    contour_mass = mass
    contour_density = density
    if contour_smoothing_sigma_bins > 0.0:
        contour_mass = gaussian_filter(
            mass,
            sigma=float(contour_smoothing_sigma_bins),
            mode="constant",
            cval=0.0,
        )
        smoothed_total = float(np.sum(contour_mass))
        if smoothed_total <= 0.0 or not np.isfinite(smoothed_total):
            raise ValueError(f"{key} smoothed contour grid is malformed")
        contour_mass *= float(np.sum(mass)) / smoothed_total
        contour_density = np.divide(
            contour_mass,
            magnitude_width[:, None] * log_radius_width[None, :],
            out=np.zeros_like(contour_mass),
            where=(
                magnitude_width[:, None] * log_radius_width[None, :]
            ) > 0.0,
        )
    magnitude_center = 0.5 * (magnitude_edges[:-1] + magnitude_edges[1:])
    log_radius_center = 0.5 * (
        log_radius_edges[:-1] + log_radius_edges[1:]
    )
    return {
        "key": key,
        "label": label,
        "detail": detail,
        "color": color,
        "density": density.tolist(),
        "surface_density_arcmin2": float(np.sum(mass)),
        "rows": rows,
        "contour_smoothing_sigma_bins": contour_smoothing_sigma_bins,
        "contours": _joint_contours(
            contour_density,
            contour_mass,
            magnitude_center,
            log_radius_center,
        ),
    }


def _joint_magnitude_radius_maps(
    synthetic: dict[str, Any],
) -> dict[str, Any]:
    """Build Q1, current-field, and active-model maps on the native Q1 grid."""
    try:
        q1 = read_q1_galaxy_radius_statistics()
        magnitude_edges = np.asarray(q1["magnitude_edges"], dtype=np.float64)
        radius_edges = np.asarray(q1["radius_edges_arcsec"], dtype=np.float64)
        area_arcmin2 = float(q1["footprint_area_arcmin2"])
        joint_bins = q1["joint_bins"]
    except (KeyError, TypeError, ValueError) as exc:
        return {
            "available": False,
            "detail": f"Q1 joint magnitude-radius aggregate unavailable: {exc}",
        }
    log_radius_edges = np.log10(radius_edges)
    shape = (magnitude_edges.size - 1, radius_edges.size - 1)
    q1_mass = np.zeros(shape, dtype=np.float64)
    try:
        for item in joint_bins:
            q1_mass[
                int(item["magnitude_bin"]), int(item["radius_bin"]),
            ] += float(item["expected_radii"]) / area_arcmin2
    except (IndexError, KeyError, TypeError, ValueError) as exc:
        return {
            "available": False,
            "detail": f"Q1 joint magnitude-radius aggregate is malformed: {exc}",
        }

    maps = [
        _joint_map(
            key="q1",
            label="Q1 MER + PHZ",
            detail=(
                "Aggregate PHZ-weighted VIS 2FWHM magnitude × clean "
                "circularized VIS Sérsic Rₑ brackets"
            ),
            color="#737373",
            magnitude_edges=magnitude_edges,
            log_radius_edges=log_radius_edges,
            cell_mass_arcmin2=q1_mass,
        ),
    ]

    synthetic_magnitude = np.asarray(
        synthetic.pop("_joint_vis_2fwhm_mag", []), dtype=np.float64,
    )
    synthetic_radius = np.asarray(
        synthetic.pop("_joint_re_arcsec", []), dtype=np.float64,
    )
    synthetic_area = float(
        synthetic.get("_joint_area_arcmin2")
        or synthetic.get("area_arcmin2")
        or 0.0
    )
    valid = (
        np.isfinite(synthetic_magnitude)
        & np.isfinite(synthetic_radius)
        & (synthetic_radius > 0.0)
    )
    if synthetic_area > 0.0 and np.any(valid):
        synthetic_count = np.histogram2d(
            synthetic_magnitude[valid],
            np.log10(synthetic_radius[valid]),
            bins=(magnitude_edges, log_radius_edges),
        )[0]
        maps.append(_joint_map(
            key="synthetic",
            label=str(
                synthetic.get("_joint_label")
                or "Current generated galaxies"
            ),
            detail=(
                str(
                    synthetic.get("_joint_detail")
                    or (
                        "Actual test + validation VIS 2FWHM source-record "
                        "magnitude × requested circularized Sérsic Rₑ draws"
                    )
                )
                + "; contours use one-bin smoothing to resolve the sparse "
                "empirical outskirts"
            ),
            color="#0072b2",
            magnitude_edges=magnitude_edges,
            log_radius_edges=log_radius_edges,
            cell_mass_arcmin2=synthetic_count / synthetic_area,
            rows=int(np.count_nonzero(valid)),
            contour_smoothing_sigma_bins=1.0,
        ))

    candidate = joint_galaxy_candidate()
    if candidate:
        try:
            magnitude_law = (
                ContinuousBrightBridgeFaintCappedMagnitudeLaw.from_payload(
                    candidate["magnitude_law"],
                )
            )
            radius_law = ConditionalRadiusLaw.from_payload(
                candidate["radius_law"],
            )
            model_mass = joint_density_grid(
                magnitude_law,
                radius_law,
                magnitude_edges=magnitude_edges,
                log_radius_edges=log_radius_edges,
            )["density"]
            maps.append(_joint_map(
                key="model",
                label="Active generation law",
                detail=(
                    "Analytical VIS 2FWHM count law × conditional "
                    "circularized Sérsic Rₑ probability"
                ),
                color="#d55e00",
                magnitude_edges=magnitude_edges,
                log_radius_edges=log_radius_edges,
                cell_mass_arcmin2=model_mass,
            ))
        except (KeyError, TypeError, ValueError):
            pass

    positive = np.concatenate([
        np.asarray(item["density"], dtype=np.float64).ravel()
        for item in maps
    ])
    positive = positive[np.isfinite(positive) & (positive > 0.0)]
    return {
        "available": len(maps) >= 2,
        "magnitude_edges": magnitude_edges.tolist(),
        "log_radius_edges": log_radius_edges.tolist(),
        "density_unit": "objects arcmin⁻² mag⁻¹ dex⁻¹",
        "contour_mass_fractions": list(JOINT_CONTOUR_MASS_FRACTIONS),
        "shared_density_max": float(np.max(positive)) if positive.size else 1.0,
        "maps": maps,
        "detail": (
            "All maps use the Q1 bin grid; contours enclose 10%, 25%, 50%, "
            "80%, 95%, 99%, and 99.5% of each map's own surface-density "
            "mass."
        ),
    }


def _brightness_curve(
    values: np.ndarray,
    area: float,
    *,
    label: str,
    survey: str,
    band: str,
    estimator: str,
    selection: str,
    weights: np.ndarray | None = None,
    default_on: bool = False,
) -> dict[str, Any]:
    """One directly measured AB-magnitude number-count curve."""
    magnitudes = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(magnitudes) & (magnitudes >= MAG_EDGES[0]) & (magnitudes < MAG_EDGES[-1])
    curve_weights = None if weights is None else np.asarray(weights, dtype=np.float64)[valid]
    curve = _curve(
        MAG_EDGES,
        np.histogram(magnitudes[valid], MAG_EDGES, weights=curve_weights)[0],
        area,
        estimator,
    )
    return {
        **curve,
        "label": label,
        "survey": survey,
        "band": band,
        "estimator": estimator,
        "selection": selection,
        "default_on": default_on,
    }


def _radius_curve(
    values_arcsec: np.ndarray,
    area: float,
    *,
    label: str,
    source: str,
    radius_type: str,
    definition: str,
    weights: np.ndarray | None = None,
    default_on: bool = False,
) -> dict[str, Any]:
    """One directly measured or analytically predicted radius distribution."""
    values = np.asarray(values_arcsec, dtype=np.float64)
    valid = np.isfinite(values) & (values > 0.0)
    curve_weights = None if weights is None else np.asarray(weights, dtype=np.float64)[valid]
    log_values = np.log10(values[valid])
    curve = _curve(
        LOG_RADIUS_EDGES,
        np.histogram(log_values, LOG_RADIUS_EDGES, weights=curve_weights)[0],
        area,
        definition,
    )
    return {
        **curve,
        "label": label,
        "source": source,
        "radius_type": radius_type,
        "units": "arcsec",
        "normalization": "surface_density",
        "default_on": default_on,
    }


def _valid_ab_magnitude(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.where(
        np.isfinite(values) & (values > 5.0) & (values < 50.0),
        values,
        np.nan,
    )


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _measure_field_half_light_radii(
    vis_image: np.ndarray,
    rows: list[dict[str, Any]],
    *,
    pixel_scale_arcsec: float,
) -> dict[int, float]:
    """Measure rendered circular half-light radii in one clean VIS scene.

    The clean record is starless and noiseless but contains every galaxy in
    the field.  Catalogue total flux supplies the otherwise ambiguous
    curve-of-growth endpoint.  We reject sources whose half-light aperture
    could contain more than ten per cent catalogue flux from neighbouring
    galaxies, so this achieved-size diagnostic is deliberately an isolated
    subset rather than a deblender measurement.
    """
    image = np.asarray(vis_image, dtype=np.float64)
    if image.ndim != 2 or pixel_scale_arcsec <= 0.0:
        return {}
    usable: list[tuple[int, float, float, float, float]] = []
    for index, row in enumerate(rows):
        if str(row.get("type", "")).lower() not in {"galaxy", "lens"}:
            continue
        x = _finite_number(row.get("x_pix"))
        y = _finite_number(row.get("y_pix"))
        flux = _finite_number(row.get("flux_vis_e"))
        radius = _finite_number(row.get("re_arcsec"))
        if x is None or y is None or flux is None or flux <= 0.0:
            continue
        usable.append((index, round(x), round(y), flux, radius or pixel_scale_arcsec))
    if not usable:
        return {}

    positions = np.asarray([(item[1], item[2]) for item in usable], dtype=np.float64)
    fluxes = np.asarray([item[3] for item in usable], dtype=np.float64)
    nominal_radii = np.asarray([item[4] for item in usable], dtype=np.float64)
    measured: dict[int, float] = {}
    height, width = image.shape
    for local_index, (row_index, x, y, flux, nominal_radius) in enumerate(usable):
        edge_radius = min(x, y, width - 1 - x, height - 1 - y)
        maximum_radius = min(
            edge_radius,
            max(12.0, 8.0 * nominal_radius / pixel_scale_arcsec),
            120.0,
        )
        if maximum_radius < 2.0:
            continue
        x_lo = max(0, int(math.floor(x - maximum_radius)))
        x_hi = min(width, int(math.ceil(x + maximum_radius)) + 1)
        y_lo = max(0, int(math.floor(y - maximum_radius)))
        y_hi = min(height, int(math.ceil(y + maximum_radius)) + 1)
        yy, xx = np.indices((y_hi - y_lo, x_hi - x_lo), dtype=np.float64)
        radius_pixels = np.hypot(xx + x_lo - x, yy + y_lo - y)
        inside = radius_pixels <= maximum_radius
        radii = radius_pixels[inside]
        values = np.maximum(image[y_lo:y_hi, x_lo:x_hi][inside], 0.0)
        order = np.argsort(radii)
        radii = radii[order]
        cumulative = np.cumsum(values[order])
        crossing = int(np.searchsorted(cumulative, 0.5 * flux))
        if crossing >= radii.size:
            continue
        achieved = float(radii[crossing] * pixel_scale_arcsec)

        distances = np.hypot(
            positions[:, 0] - x,
            positions[:, 1] - y,
        ) * pixel_scale_arcsec
        neighbours = np.arange(len(usable)) != local_index
        # Four nominal radii is a conservative light-support proxy.  The
        # ten-per-cent cut removes blends without discarding a bright, large
        # source merely because many much fainter objects share its field.
        overlaps = neighbours & (
            distances - 4.0 * np.maximum(nominal_radii, pixel_scale_arcsec)
            <= achieved
        )
        contamination_bound = float(np.sum(fluxes[overlaps])) / flux
        if contamination_bound > 0.10:
            continue
        measured[row_index] = max(achieved, pixel_scale_arcsec)
    return measured


def _read_synthetic(
    parameters: dict[str, Any],
    progress: Callable[[int, int, str], None] | None = None,
    *,
    include_training: bool = False,
    measure_clean_images: bool = True,
) -> dict[str, Any]:
    """Add generated draws, optionally including catalog-only training rows.

    Training image records are never read here.  The optional training layer
    comes exclusively from ``sources_train.csv``; quantities missing from that
    legacy catalogue retain the area of the splits that actually provide them.
    """
    _records, source_paths = (
        _synthetic_paths(include_training=True)
        if include_training else _synthetic_paths()
    )
    if not source_paths:
        return {
            "available": False,
            "detail": "No generated source catalogues are cached.",
        }

    galaxies: list[dict[str, Any]] = []
    fields = 0
    split_rows: dict[str, dict[int, list[dict[str, Any]]]] = {}
    split_field_counts: dict[str, int] = {}
    for path in source_paths:
        split = path.stem.removeprefix("sources_")
        by_field: dict[int, list[dict[str, Any]]] = {}
        with path.open(newline="", encoding="utf-8") as handle:
            for raw in csv.DictReader(handle):
                field_index = int(raw["field_index"])
                row = dict(raw)
                row["_split"] = split
                row["_field_index"] = field_index
                row["_field_row"] = len(by_field.setdefault(field_index, []))
                by_field[field_index].append(row)
                if str(raw.get("type", "")).lower() == "galaxy":
                    galaxies.append(row)
        split_field_counts[split] = len(by_field)
        fields += split_field_counts[split]
        split_rows[split] = by_field

    area = fields * FIELD_AREA_ARCMIN2
    if not galaxies or area <= 0.0:
        return {
            "available": False,
            "detail": "Generated source catalogues contain no galaxies.",
        }
    training_fields = int(split_field_counts.get("train", 0))
    training_included = bool(include_training and training_fields)
    catalogue_scope = (
        "training/test/validation" if training_included else "test/validation"
    )

    measured_by_identity: dict[tuple[str, int, int], float] = {}
    measured_fields = 0
    if measure_clean_images:
        for source_path in source_paths:
            split = source_path.stem.removeprefix("sources_")
            # Training is intentionally catalogue-only even if a training
            # TFRecord happens to be present in a developer cache.
            if split == "train":
                continue
            clean_path = source_path.with_name(f"clean_{split}.tfrecord")
            if not clean_path.is_file():
                continue
            import tensorflow as tf

            from euclid_polish.image.core import Image

            for field_index, raw_record in enumerate(
                tf.data.TFRecordDataset([str(clean_path)])
            ):
                image = Image.from_tfrecord(raw_record)
                field_rows = split_rows[split].get(field_index, [])
                field_measurements = _measure_field_half_light_radii(
                    np.asarray(image.data)[..., 0],
                    field_rows,
                    pixel_scale_arcsec=float(image.pixel_scale_arcsec),
                )
                for row_index, radius in field_measurements.items():
                    measured_by_identity[(split, field_index, row_index)] = radius
                measured_fields += 1
                if progress:
                    progress(
                        4, 6,
                        "measure clean generated galaxies · "
                        f"{measured_fields}/{fields}",
                    )

    def values(*keys: str) -> np.ndarray:
        output = []
        for row in galaxies:
            value = next(
                (
                    candidate
                    for key in keys
                    if (candidate := _finite_number(row.get(key))) is not None
                ),
                math.nan,
            )
            output.append(value)
        return np.asarray(output, dtype=np.float64)

    def ordered_splits(names: set[str]) -> list[str]:
        preferred = [
            split for split in ("train", "test", "validate")
            if split in names
        ]
        return preferred + sorted(names.difference(preferred))

    def contributing_splits(valid: np.ndarray) -> list[str]:
        mask = np.asarray(valid, dtype=bool)
        return ordered_splits({
            str(row["_split"])
            for row, keep in zip(galaxies, mask, strict=True)
            if keep
        })

    def effective_area(valid: np.ndarray) -> tuple[float, list[str]]:
        splits = contributing_splits(valid)
        contributor_fields = sum(split_field_counts[split] for split in splits)
        return contributor_fields * FIELD_AREA_ARCMIN2, splits

    redshift = values("z")
    mass = values("target_logmass", "logmass")
    ssfr = values("target_logssfr", "native_tng_logssfr")
    requested_radius = values("re_arcsec", "target_re_arcsec")
    achieved_f2 = values("achieved_vis_2fwhm_mag", "target_vis_2fwhm_mag")
    redshift_edges = np.linspace(0.0, 6.0, 49)
    parameter_coverage: dict[str, Any] = {}
    for key, data, edges, definition in (
        (
            "redshift", redshift, redshift_edges,
            f"actual generated {catalogue_scope} galaxy redshift draws",
        ),
        (
            "stellar_mass", mass, MASS_EDGES,
            f"actual generated {catalogue_scope} target stellar-mass draws",
        ),
        (
            "specific_sfr", ssfr, SSFR_EDGES,
            f"actual generated {catalogue_scope} target specific-SFR draws",
        ),
    ):
        valid = np.isfinite(data)
        if key == "specific_sfr":
            valid &= data < -8.2
        parameter_area, contributor_splits = effective_area(valid)
        parameters[key]["series"]["synthetic"] = _curve(
            edges,
            np.histogram(data[valid], edges)[0],
            parameter_area or area,
            definition,
        )
        parameter_coverage[key] = {
            "splits": contributor_splits,
            "area_arcmin2": parameter_area,
        }

    photometry = parameters["magnitude"]["photometry_series"]
    achieved_f2_valid = np.isfinite(achieved_f2)
    achieved_f2_area, achieved_f2_splits = effective_area(achieved_f2_valid)
    photometry["synthetic_vis_2fwhm"] = _brightness_curve(
        achieved_f2,
        achieved_f2_area or area,
        label="Generated fields · VIS 2FWHM",
        survey="synthetic",
        band="Euclid VIS",
        estimator="achieved 2FWHM-diameter aperture magnitude from source record",
        selection=(
            "galaxies in source catalogues that store an exact VIS 2FWHM "
            "magnitude; legacy training rows are excluded when unavailable"
        ),
        default_on=True,
    )
    parameter_coverage["vis_2fwhm"] = {
        "splits": achieved_f2_splits,
        "area_arcmin2": achieved_f2_area,
    }
    for band, flux_key, magnitude_key, label in (
        ("VIS", "flux_vis_e", "mag_vis", "VIS total stamp"),
        ("Y_E", "flux_y_e", "mag_y_e", "Y total stamp"),
        ("J_E", "flux_j_e", "mag_j_e", "J total stamp"),
        ("H_E", "flux_h_e", "mag_h_e", "H total stamp"),
    ):
        flux = values(flux_key)
        magnitudes = values(magnitude_key)
        flux_valid = np.isfinite(flux) & (flux > 0.0)
        magnitudes[flux_valid] = [
            electrons_to_ab_mag(item, Config.get_band(band))
            for item in flux[flux_valid]
        ]
        valid = np.isfinite(magnitudes)
        magnitude_area, magnitude_splits = effective_area(valid)
        photometry[f"synthetic_{band.lower()}_total"] = _brightness_curve(
            magnitudes,
            magnitude_area or area,
            label=f"Generated fields · {label}",
            survey="synthetic",
            band=f"Euclid {band.replace('_E', '')}",
            estimator=(
                "total rendered-stamp flux or its stored AB magnitude in the "
                "source catalogue"
            ),
            selection=(
                "all galaxies in the selected source-catalogue splits with "
                "this band available"
            ),
            default_on=band == "VIS",
        )
        parameter_coverage[f"{band.lower()}_total"] = {
            "splits": magnitude_splits,
            "area_arcmin2": magnitude_area,
        }

    radius_series = parameters["radius"]["radius_series"]
    requested_valid = np.isfinite(requested_radius) & (requested_radius > 0.0)
    requested_area, requested_splits = effective_area(requested_valid)
    radius_series["synthetic_requested_re"] = _radius_curve(
        requested_radius,
        requested_area or area,
        label="Generated fields · requested Sérsic Rₑ",
        source="synthetic",
        radius_type="half_light",
        definition=(
            "actual nominal circularized Sérsic Rₑ draws stored in the source CSV; "
            "continuous-space generator geometry, not an image measurement"
        ),
        default_on=True,
    )
    parameter_coverage["requested_re"] = {
        "splits": requested_splits,
        "area_arcmin2": requested_area,
    }
    achieved_radius = np.asarray([
        measured_by_identity.get(
            (str(row["_split"]), int(row["_field_index"]), int(row["_field_row"])),
            math.nan,
        )
        for row in galaxies
    ], dtype=np.float64)
    if measure_clean_images:
        achieved_valid = np.isfinite(achieved_radius)
        achieved_area, achieved_splits = effective_area(achieved_valid)
        radius_series["synthetic_clean_half_light"] = _radius_curve(
            achieved_radius,
            achieved_area or area,
            label="Generated fields · clean-image half-light radius",
            source="synthetic",
            radius_type="rendered_half_light",
            definition=(
                "circular curve-of-growth half-light radius measured directly "
                "on starless clean VIS images; catalogue total flux sets the "
                "endpoint; neighbour-contamination bound <= 10%; resolution "
                "floor is one 0.05 arcsec clean-image pixel"
            ),
            default_on=True,
        )
        parameter_coverage["clean_half_light"] = {
            "splits": achieved_splits,
            "area_arcmin2": achieved_area,
        }
    measured_count = int(np.count_nonzero(np.isfinite(achieved_radius)))
    joint_valid = achieved_f2_valid & requested_valid
    joint_area, joint_splits = effective_area(joint_valid)
    if training_included and "train" not in joint_splits:
        joint_detail = (
            "Exact test + validation VIS 2FWHM source-record magnitude × "
            "requested circularized Sérsic Rₑ draws. The legacy training "
            "catalogue has no 2FWHM magnitude, so it is not substituted with "
            "total VIS in this estimator-matched contour."
        )
        joint_label = "Current generated galaxies · exact 2FWHM subset"
    elif training_included:
        joint_detail = (
            "Training + test + validation VIS 2FWHM source-record magnitude × "
            "requested circularized Sérsic Rₑ draws"
        )
        joint_label = "Generated galaxies · all catalogued splits"
    else:
        joint_detail = (
            "Actual test + validation VIS 2FWHM source-record magnitude × "
            "requested circularized Sérsic Rₑ draws"
        )
        joint_label = "Current generated galaxies"
    return {
        "available": True,
        "rows": len(galaxies),
        "fields": fields,
        "area_arcmin2": area,
        "splits": ordered_splits(set(split_rows)),
        "training_included": training_included,
        "training_catalog_only": training_included,
        "training_fields": training_fields,
        "parameter_coverage": parameter_coverage,
        "measured_radius_rows": measured_count,
        "measured_radius_fraction": measured_count / len(galaxies),
        "detail": (
            (
                f"Training + test + validation source catalogues ({fields:,} "
                "fields); training is catalogue-only and no training image "
                "records are local or read"
            ) if training_included else (
                f"Current regenerated test + validation draws; "
                f"{measured_count:,} isolated galaxy radii measured directly "
                "on clean VIS images"
            )
        ),
        # Consumed by ``build_galaxy_distributions`` to make the joint map,
        # then removed before the compact source ledger is serialized.
        "_joint_vis_2fwhm_mag": achieved_f2.tolist(),
        "_joint_re_arcsec": requested_radius.tolist(),
        "_joint_area_arcmin2": joint_area,
        "_joint_label": joint_label,
        "_joint_detail": joint_detail,
    }


def _aperture_columns(values: np.ndarray, count: int) -> np.ndarray | None:
    """Normalize a FITS vector column saved in the compact NPZ to (row, aperture)."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        return None
    if array.shape[1] == count:
        return array
    if array.shape[0] == count:
        return array.T
    return None


def _histogram_quantile(
    histogram: np.ndarray, edges: np.ndarray, probability: float,
) -> list[float | None]:
    """Approximate row-wise quantiles from a compact weighted histogram."""
    centers = 0.5 * (edges[:-1] + edges[1:])
    output: list[float | None] = []
    for row in np.asarray(histogram, dtype=np.float64):
        total = float(np.sum(row))
        if total <= 0.0:
            output.append(None)
            continue
        cumulative = np.cumsum(row)
        output.append(float(np.interp(probability * total, cumulative, centers)))
    return output


def _aperture_growth_payload(
    histograms: dict[str, np.ndarray], weighted_count: dict[str, np.ndarray],
) -> dict[str, Any]:
    curves = {}
    for key, histogram in histograms.items():
        curves[key] = {
            "p16": _histogram_quantile(histogram, APERTURE_DELTA_MAG_EDGES, 0.16),
            "median": _histogram_quantile(histogram, APERTURE_DELTA_MAG_EDGES, 0.50),
            "p84": _histogram_quantile(histogram, APERTURE_DELTA_MAG_EDGES, 0.84),
            "weighted_count": weighted_count[key].tolist(),
        }
    return {
        "x": (0.5 * (APERTURE_SIZE_EDGES[:-1] + APERTURE_SIZE_EDGES[1:])).tolist(),
        "x_edges": APERTURE_SIZE_EDGES.tolist(),
        "x_label": "circularized MER radius / local PSF FWHM",
        "y_label": "smaller/reference aperture magnitude difference",
        "selection": (
            "probability-weighted nonstellar sources with VIS 4-FWHM S/N >= 10; "
            "Kron comparison additionally requires VIS detection and "
            "DET_QUALITY_FLAG = 0"
        ),
        "interpretation": (
            "positive values mean the smaller aperture is fainter and misses "
            "flux relative to the named reference; neither Kron nor Sérsic is "
            "treated as exact ground truth"
        ),
        "curves": curves,
    }


def _aperture_scatter_payload(
    samples: list[tuple[float, str, tuple[float, ...]]],
) -> dict[str, Any]:
    """Return compact column arrays for the interactive aperture ladder."""
    ordered = [sample for _priority, _object_id, sample in sorted(
        samples, key=lambda item: (item[2][3], item[0], item[1]),
    )]
    array = np.asarray(ordered, dtype=np.float64)
    if array.size == 0:
        array = np.empty((0, 7), dtype=np.float64)
    return {
        "count": int(len(array)),
        "magnitudes": {
            f"f{index + 1}": array[:, index].tolist()
            for index in range(4)
        },
        "growth": {
            f"g{index + 1}": array[:, index + 4].tolist()
            for index in range(3)
        },
        "definitions": {
            "g1": "m1 - m4 = -2.5 log10(F1 / F4)",
            "g2": "m2 - m4 = -2.5 log10(F2 / F4)",
            "g3": "m3 - m4 = -2.5 log10(F3 / F4)",
        },
        "selection": (
            "MER EXTENDED_FLAG galaxies plus otherwise unclassified sources "
            "with PHZ_GAL_PROB >= 0.5; SPURIOUS_PROB <= 0.5, positive VIS "
            "F1-F4 fluxes, and VIS F4 S/N >= 10; deterministic sample "
            f"capped at {APERTURE_SCATTER_PER_MAG_BIN} objects per 0.5-mag "
            "F4 bin"
        ),
    }


def _empty_parameters() -> dict[str, dict[str, Any]]:
    return {
        "redshift": {
            "label": "Redshift",
            "x_label": "Redshift z",
            "density_unit": "objects / arcmin² / redshift",
            "note": (
                "Euclid uses probability-weighted PHZ PDFs; COSMOS uses "
                "catalogue photo-z; the fit is the corrected latent draw model; "
                "generated points are the actual test/validation draws."
            ),
            "series": {},
        },
        "magnitude": {
            "label": "Apparent brightness",
            "x_label": "catalogue AB magnitude (native estimator)",
            "density_unit": "objects / arcmin² / mag",
            "note": (
                "Direct catalogue measurements only. Curves retain their native "
                "VIS or HST/F814W passband, PSF treatment, and flux estimator; "
                "no F814W-to-VIS transfer is drawn. Generated curves come from "
                "the current source records, not the analytic law."
            ),
            "series": {},
            "photometry_series": {},
            "photometry_missing": [],
        },
        "radius": {
            "label": "Angular size",
            "x_label": "log₁₀ radius (arcsec)",
            "x_domain": [
                float(LOG_RADIUS_EDGES[0]),
                float(np.log10(RADIUS_MAX_ARCSEC)),
            ],
            "density_unit": "objects / arcmin² / dex",
            "note": (
                "The fitted quantity is the science-clean PHZ/MER "
                "circularized VIS Sérsic effective radius, "
                "Rₑ,circ = Rₑ,major√q. Detection, Kron, and COSMOS curves are diagnostics "
                "only and do not enter the generator fit. "
                "Generated Rₑ is nominal continuous-space geometry over "
                "0.03–10 arcsec (up to 100 native VIS pixels), including "
                "values below one 0.05 arcsec HR pixel. The separately labelled "
                "clean-image half-light curve is measured after rendering and "
                "has a one-pixel resolution floor. Normalized shape "
                "controls are unit-integral probability densities per dex "
                "and remain separate from catalogue sky-density controls."
            ),
            "series": {},
            "radius_series": {},
            "radius_missing": [],
        },
        "stellar_mass": {
            "label": "Stellar mass",
            "x_label": "log₁₀ stellar mass (M☉)",
            "density_unit": "objects / arcmin² / dex",
            "note": (
                "PHZ and COSMOS values are posterior/catalogue estimates. The fit "
                "curve requires a PHZ-enhanced physical conditional model."
            ),
            "series": {},
        },
        "specific_sfr": {
            "label": "Specific star-formation rate",
            "x_label": "log₁₀ sSFR (yr⁻¹)",
            "density_unit": "objects / arcmin² / dex",
            "note": (
                "The documented pathological tail at log₁₀ sSFR ≥ -8.2 is "
                "excluded from PHZ constraints and the fitted model."
            ),
            "series": {},
        },
    }


def _read_q1_bright_counts(parameters: dict[str, Any]) -> dict[str, Any]:
    """Add progressive Q1 MER+PHZ aperture counts to brightness curves."""
    try:
        payload = read_q1_galaxy_aperture_counts()
    except ValueError:
        return {
            "available": False,
            "detail": "Query Q1 MER + PHZ aperture counts across VIS 14–28.",
        }

    selection = str(payload["selection"])
    for key, aperture in payload["apertures"].items():
        bins = aperture["bins"]
        if not bins:
            continue
        parameters["magnitude"]["photometry_series"][f"q1_vis_{key}"] = {
            "x": [
                0.5 * (float(item["mag_lo"]) + float(item["mag_hi"]))
                for item in bins
            ],
            "density": [float(item["density_arcmin2_mag"]) for item in bins],
            "weighted_count": float(aperture["expected_galaxies"]),
            "definition": "PHZ-probability-weighted hard-MER galaxy counts",
            "label": f"Q1 MER + PHZ galaxies · {aperture['label']}",
            "survey": "euclid",
            "band": "Euclid VIS",
            "estimator": aperture["estimator"],
            "selection": selection,
            "default_on": key == "f2",
        }
    fitted_apertures: set[str] = set()
    try:
        fitted = read_q1_galaxy_aperture_fit()
    except ValueError:
        fitted = None
    if fitted:
        for key, curve in fitted["apertures"].items():
            fitted_apertures.add(key)
            parameters["magnitude"]["photometry_series"][
                f"q1_fit_vis_{key}"
            ] = {
                "x": [float(value) for value in curve["x"]],
                "density": [float(value) for value in curve["density"]],
                "weighted_count": float(
                    payload["apertures"][key]["expected_galaxies"]
                ),
                "definition": (
                    "straight log-density fit to the automatically selected "
                    "Q1 MER + PHZ 2FWHM count region"
                ),
                "label": f"Q1 fitted · {curve['label']}",
                "survey": "fit",
                "band": "Euclid VIS",
                "estimator": (
                    f"{curve['estimator']}; straight log-density law"
                ),
                "selection": str(fitted["scope"]),
                "default_on": key == "f2",
                "fit_interval": [
                    float(curve["law"]["fit_bright"]),
                    float(curve["law"]["fit_faint"]),
                ],
                "sampling_interval": [
                    float(curve["law"]["mag_bright"]),
                    float(curve["law"]["mag_faint"]),
                ],
                "extrapolated_interval": [
                    float(value)
                    for value in curve["extrapolated_faint_interval"]
                ],
            }
    return {
        "available": True,
        "footprint_area_deg2": float(payload["footprint_area_deg2"]),
        "bright": float(payload["bright"]),
        "faint": float(payload["faint"]),
        "bin_width": float(payload["bin_width"]),
        "bins": int(len(payload["edges"]) - 1),
        "query_count": int(payload["query_count"]),
        "completed_queries": int(payload["completed_queries"]),
        "total_queries": int(payload["total_queries"]),
        "complete": bool(payload["complete"]),
        "phases_completed": int(payload["phases_completed"]),
        "phase_count": int(payload["phase_count"]),
        "fit_available": bool(fitted_apertures),
        "fitted_apertures": sorted(fitted_apertures),
        "selection": selection,
        "aperture_counts": {
            key: float(aperture["expected_galaxies"])
            for key, aperture in payload["apertures"].items()
        },
        "detail": "progressive Q1 MER + PHZ bright-galaxy aperture counts",
    }


def _read_q1_radius_statistics(parameters: dict[str, Any]) -> dict[str, Any]:
    """Use only aggregate Q1 brackets for the fitted circularized radius."""
    radius_parameter = parameters["radius"]
    radius_series = radius_parameter["radius_series"]
    # An old field/cone catalogue may still provide detection and Kron
    # diagnostics, but it must never masquerade as the fitted Sersic sample.
    radius_series.pop("euclid_sersic_re", None)
    radius_series.pop("euclid_sersic_re_shape", None)
    radius_parameter["radius_missing"] = [
        message for message in radius_parameter["radius_missing"]
        if "Sersic" not in message and "Sérsic" not in message
    ]
    try:
        payload = read_q1_galaxy_radius_statistics()
    except ValueError:
        radius_parameter["radius_missing"].append(
            "Press Query MER + PHZ to cache aggregate VIS 2FWHM x "
            "circularized-Sersic-R_e brackets. Random population cones are "
            "not used."
        )
        return {
            "available": False,
            "detail": "Query aggregate Q1 circularized-radius brackets.",
        }
    bins = payload["radius_bins"]
    x = [
        0.5 * (
            math.log10(float(item["radius_lo_arcsec"]))
            + math.log10(float(item["radius_hi_arcsec"]))
        )
        for item in bins
    ]
    density = np.asarray([
        float(item["density_arcmin2_dex"]) for item in bins
    ], dtype=np.float64)
    log_radius_width = np.asarray([
        math.log10(float(item["radius_hi_arcsec"]))
        - math.log10(float(item["radius_lo_arcsec"]))
        for item in bins
    ], dtype=np.float64)
    normalized_density = _normalized_density(density, log_radius_width)
    weighted_count = float(sum(
        float(item["expected_radii"]) for item in bins
    ))
    radius_series["euclid_sersic_re"] = {
        "x": x,
        "density": density.tolist(),
        "weighted_count": weighted_count,
        "definition": (
            "aggregate science-clean Q1 MER morphology circularized VIS "
            "Sersic effective-radius brackets, R_e,circ = R_e,major sqrt(q); "
            "weighted by PHZ_GAL_PROB"
        ),
        "label": "Q1 PHZ/MER · circularized VIS Sersic R_e",
        "source": "euclid",
        "radius_type": "half_light",
        "units": "arcsec",
        "normalization": "surface_density",
        "default_on": False,
    }
    radius_series["euclid_sersic_re_shape"] = {
        "x": x,
        "density": normalized_density.tolist(),
        "weighted_count": weighted_count,
        "definition": (
            "science-clean Q1 circularized VIS Sérsic Rₑ marginal, "
            "normalized to unit probability over log-radius"
        ),
        "label": "Q1 clean · normalized circularized Sérsic Rₑ shape",
        "source": "euclid",
        "radius_type": "half_light_shape",
        "units": "arcsec",
        "normalization": "probability_density",
        "default_on": True,
    }
    return {
        "available": True,
        "complete": bool(payload["complete"]),
        "completed_queries": int(payload["completed_queries"]),
        "total_queries": int(payload["total_queries"]),
        "magnitude_brackets": len(payload["magnitude_bins"]),
        "radius_brackets": len(bins),
        "footprint_area_deg2": float(payload["footprint_area_deg2"]),
        "selection": str(payload["selection"]),
        "acquisition": str(payload["acquisition"]),
        "detail": (
            "aggregate Q1 VIS 2FWHM x cleaned circularized-Sersic-R_e "
            "statistics"
        ),
    }


def _read_euclid(parameters: dict[str, Any], progress: Callable[[int, int, str], None]) -> dict[str, Any]:
    path = euclid_catalog_path()
    meta = _json(euclid_catalog_meta_path()) or {}
    area = float(meta.get("area_arcmin2") or 0.0)
    if not path.is_file() or area <= 0:
        return {"available": False, "detail": "Query Euclid MER + PHZ to create the catalogue cache."}

    pdf_by_id: dict[str, np.ndarray] = {}
    pdf_edges: np.ndarray | None = None
    try:
        pdf = read_phz_pdf_cache()
        pdf_by_id = dict(zip(np.asarray(pdf["object_id"]).astype(str), pdf["probability"], strict=True))
        pdf_edges = np.asarray(pdf["z_edges"], dtype=np.float64)
    except (OSError, KeyError, ValueError):
        pass

    mag = np.zeros(len(MAG_EDGES) - 1)
    radius = np.zeros(len(LOG_RADIUS_EDGES) - 1)
    mass = np.zeros(len(MASS_EDGES) - 1)
    ssfr = np.zeros(len(SSFR_EDGES) - 1)
    redshift = np.zeros(len(pdf_edges) - 1) if pdf_edges is not None else None
    aperture_keys = (
        "1fwhm_minus_4fwhm", "2fwhm_minus_4fwhm",
        "3fwhm_minus_4fwhm", "4fwhm_minus_kron",
        "4fwhm_minus_sersic",
    )
    aperture_histograms = {
        key: np.zeros((
            len(APERTURE_SIZE_EDGES) - 1,
            len(APERTURE_DELTA_MAG_EDGES) - 1,
        ))
        for key in aperture_keys
    }
    aperture_counts = {
        key: np.zeros(len(APERTURE_SIZE_EDGES) - 1)
        for key in aperture_keys
    }
    aperture_scatter_heaps: list[list[tuple[int, str, tuple[float, ...]]]] = [
        [] for _ in range(len(APERTURE_SCATTER_MAG_EDGES) - 1)
    ]
    brightness_magnitudes = {key: [] for key in MER_BRIGHTNESS_SERIES}
    brightness_weights = {key: [] for key in MER_BRIGHTNESS_SERIES}
    radius_values = {
        key: []
        for key in (
            "euclid_detection", "euclid_kron", "euclid_sersic_re",
        )
    }
    radius_weights = {key: [] for key in radius_values}
    rows = phz_rows = physical_rows = 0
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        has_sersic_re = {
            "morph_sersic_vis_radius_arcsec",
            "morph_sersic_visnir_flags",
        }.issubset(reader.fieldnames or ())
        for row in reader:
            rows += 1
            try:
                point = float(row.get("point_like_prob", "nan"))
                spurious = float(row.get("spurious_prob", "nan"))
                magnitude = float(row.get("mag_vis", "nan"))
            except ValueError:
                continue
            if not (np.isfinite(point) and 0 <= point <= 1 and np.isfinite(spurious) and spurious <= 0.5):
                continue
            mer_weight = 1.0 - point
            try:
                gal_weight = float(row.get("phz_gal_prob", "nan"))
            except ValueError:
                gal_weight = np.nan
            if np.isfinite(magnitude):
                mag += np.histogram([magnitude], MAG_EDGES, weights=[mer_weight])[0]
            for brightness_key, (column, _label, _estimator) in MER_BRIGHTNESS_SERIES.items():
                if brightness_key == "mer_vis_kron":
                    try:
                        vis_detection = float(row.get("vis_det", "nan"))
                        detection_quality = float(row.get("det_quality_flag", "nan"))
                    except ValueError:
                        continue
                    if vis_detection != 1.0 or detection_quality != 0.0:
                        continue
                try:
                    direct_flux = float(row.get(column, "nan"))
                except ValueError:
                    continue
                if np.isfinite(direct_flux) and direct_flux > 0.0:
                    direct_magnitude = float(uJy_to_ab_mag(direct_flux))
                    if np.isfinite(direct_magnitude):
                        brightness_magnitudes[brightness_key].append(direct_magnitude)
                        brightness_weights[brightness_key].append(mer_weight)
            try:
                semimajor = float(row.get("semimajor_axis", "nan"))
                ellipticity = float(row.get("ellipticity", "nan"))
            except ValueError:
                semimajor = ellipticity = np.nan
            circularized = 0.1 * semimajor * np.sqrt(max(0.0, 1.0 - ellipticity))
            if np.isfinite(circularized) and circularized > 0:
                radius += np.histogram([np.log10(circularized)], LOG_RADIUS_EDGES, weights=[mer_weight])[0]
            detection_radius = 0.1 * semimajor
            if np.isfinite(detection_radius) and detection_radius > 0.0:
                radius_values["euclid_detection"].append(detection_radius)
                radius_weights["euclid_detection"].append(mer_weight)
            try:
                kron_radius = float(row.get("kron_radius", "nan"))
            except ValueError:
                kron_radius = np.nan
            kron_radius_arcsec = 0.1 * kron_radius
            if np.isfinite(kron_radius_arcsec) and kron_radius_arcsec > 0.0:
                radius_values["euclid_kron"].append(kron_radius_arcsec)
                radius_weights["euclid_kron"].append(mer_weight)
            try:
                sersic_re_arcsec = float(row.get(
                    "morph_sersic_vis_radius_arcsec", "nan",
                ))
                sersic_flags = float(row.get(
                    "morph_sersic_visnir_flags", "nan",
                ))
            except ValueError:
                sersic_re_arcsec = sersic_flags = np.nan
            if (
                sersic_flags == 0.0
                and np.isfinite(sersic_re_arcsec)
                and sersic_re_arcsec > 0.0
                and np.isfinite(gal_weight)
                and 0.0 < gal_weight <= 1.0
            ):
                radius_values["euclid_sersic_re"].append(sersic_re_arcsec)
                radius_weights["euclid_sersic_re"].append(gal_weight)

            try:
                fwhm = float(row.get("fwhm", "nan"))
                flux4 = float(row.get("flux_vis_4fwhm_aper_uJy", "nan"))
                error4 = float(row.get("fluxerr_vis_4fwhm_aper_uJy", "nan"))
            except ValueError:
                fwhm = flux4 = error4 = np.nan
            try:
                phz_galaxy_probability = float(row.get("phz_gal_prob", "nan"))
            except ValueError:
                phz_galaxy_probability = np.nan
            is_galaxy = row.get("type") == "galaxy" or (
                row.get("type") == "unknown"
                and np.isfinite(phz_galaxy_probability)
                and phz_galaxy_probability >= 0.5
            )
            if (
                is_galaxy
                and np.isfinite(spurious) and spurious <= 0.5
                and np.isfinite(flux4) and flux4 > 0.0
                and np.isfinite(error4) and error4 > 0.0
                and flux4 / error4 >= 10.0
            ):
                try:
                    fluxes = np.asarray([
                        float(row.get(
                            f"flux_vis_{multiple}fwhm_aper_uJy", "nan",
                        ))
                        for multiple in range(1, 5)
                    ])
                except ValueError:
                    fluxes = np.full(4, np.nan)
                if np.all(np.isfinite(fluxes)) and np.all(fluxes > 0.0):
                    magnitudes = np.asarray(uJy_to_ab_mag(fluxes))
                    growth = -2.5 * np.log10(fluxes[:3] / fluxes[3])
                    mag_bin = int(np.searchsorted(
                        APERTURE_SCATTER_MAG_EDGES,
                        magnitudes[3],
                        side="right",
                    ) - 1)
                    if 0 <= mag_bin < len(aperture_scatter_heaps):
                        object_id = str(row.get("object_id", ""))
                        priority = int.from_bytes(hashlib.blake2b(
                            object_id.encode("utf-8"),
                            digest_size=8,
                            person=b"aper-gro",
                        ).digest())
                        sample = tuple(np.concatenate(
                            (magnitudes, growth),
                        ).tolist())
                        item = (-priority, object_id, sample)
                        heap = aperture_scatter_heaps[mag_bin]
                        if len(heap) < APERTURE_SCATTER_PER_MAG_BIN:
                            heapq.heappush(heap, item)
                        elif priority < -heap[0][0]:
                            heapq.heapreplace(heap, item)
            size_ratio = circularized / fwhm if fwhm > 0 else np.nan
            size_bin = int(np.searchsorted(
                APERTURE_SIZE_EDGES, size_ratio, side="right",
            ) - 1)
            if (
                0 <= size_bin < len(APERTURE_SIZE_EDGES) - 1
                and np.isfinite(flux4) and flux4 > 0.0
                and np.isfinite(error4) and error4 > 0.0
                and flux4 / error4 >= 10.0
            ):
                references = {
                    "1fwhm_minus_4fwhm": row.get("flux_vis_1fwhm_aper_uJy"),
                    "2fwhm_minus_4fwhm": row.get("flux_vis_2fwhm_aper_uJy"),
                    "3fwhm_minus_4fwhm": row.get("flux_vis_3fwhm_aper_uJy"),
                    "4fwhm_minus_sersic": row.get("flux_vis_sersic_uJy"),
                }
                try:
                    vis_det = float(row.get("vis_det", "nan"))
                    quality = float(row.get("det_quality_flag", "nan"))
                except ValueError:
                    vis_det = quality = np.nan
                if vis_det == 1.0 and quality == 0.0:
                    references["4fwhm_minus_kron"] = row.get(
                        "flux_detection_total_uJy",
                    )
                for key, raw_reference in references.items():
                    try:
                        reference = float(raw_reference or "nan")
                    except ValueError:
                        continue
                    if not np.isfinite(reference) or reference <= 0.0:
                        continue
                    if key.startswith("4fwhm"):
                        delta = -2.5 * np.log10(flux4 / reference)
                    else:
                        delta = -2.5 * np.log10(reference / flux4)
                    delta = float(np.clip(
                        delta,
                        APERTURE_DELTA_MAG_EDGES[0] + 1e-9,
                        APERTURE_DELTA_MAG_EDGES[-1] - 1e-9,
                    ))
                    delta_bin = int(np.searchsorted(
                        APERTURE_DELTA_MAG_EDGES, delta, side="right",
                    ) - 1)
                    aperture_histograms[key][size_bin, delta_bin] += mer_weight
                    aperture_counts[key][size_bin] += mer_weight

            object_pdf = pdf_by_id.get(str(row.get("object_id", "")))
            if (
                object_pdf is not None
                and np.isfinite(gal_weight)
                and 0 <= gal_weight <= 1
                and magnitude < 24.5
            ):
                redshift += gal_weight * object_pdf  # type: ignore[operator]
                phz_rows += 1
            try:
                flags = float(row.get("phz_phys_flags", "nan"))
                quality = float(row.get("phz_phys_quality_flag", "nan"))
                logmass = float(row.get("phz_pp_median_stellarmass", "nan"))
                logsfr = float(row.get("phz_pp_median_sfr", "nan"))
            except ValueError:
                continue
            logssfr = logsfr - logmass
            if (
                flags == 0
                and quality == 0
                and np.isfinite(logmass)
                and np.isfinite(logssfr)
                and logssfr < -8.2
                and np.isfinite(gal_weight)
            ):
                mass += np.histogram([logmass], MASS_EDGES, weights=[gal_weight])[0]
                ssfr += np.histogram([logssfr], SSFR_EDGES, weights=[gal_weight])[0]
                physical_rows += 1
            if rows % 50000 == 0:
                progress(rows, int(meta.get("rows") or rows), "stream Euclid MER + PHZ")

    parameters["magnitude"]["series"]["euclid"] = _curve(MAG_EDGES, mag, area, "MER 1 − POINT_LIKE_PROB")
    for key, (_column, label, estimator) in MER_BRIGHTNESS_SERIES.items():
        selection = (
            "SPURIOUS_PROB <= 0.5; probability weight 1 − POINT_LIKE_PROB; "
            "positive reported flux"
        )
        if key == "mer_vis_kron":
            selection += "; VIS_DET = 1 and DET_QUALITY_FLAG = 0"
        parameters["magnitude"]["photometry_series"][key] = _brightness_curve(
            np.asarray(brightness_magnitudes[key]),
            area,
            label=label,
            survey="euclid",
            band="Euclid VIS",
            estimator=estimator,
            selection=selection,
            weights=np.asarray(brightness_weights[key]),
            default_on=key in {"mer_vis_1fwhm", "mer_vis_4fwhm"},
        )
    parameters["radius"]["series"]["euclid"] = _curve(
        LOG_RADIUS_EDGES, radius, area, "MER circularized size proxy, galaxy-weighted"
    )
    parameters["radius"]["radius_series"]["euclid_detection"] = _radius_curve(
        np.asarray(radius_values["euclid_detection"]),
        area,
        label="Euclid · detection a",
        source="euclid",
        radius_type="detection",
        definition="VIS detection/deblender semi-major axis, 0.1 arcsec/pixel",
        weights=np.asarray(radius_weights["euclid_detection"]),
        default_on=False,
    )
    parameters["radius"]["radius_series"]["euclid_kron"] = _radius_curve(
        np.asarray(radius_values["euclid_kron"]),
        area,
        label="Euclid · Kron radius",
        source="euclid",
        radius_type="kron",
        definition="VIS Kron radius reported by the MER photometry catalogue, 0.1 arcsec/pixel",
        weights=np.asarray(radius_weights["euclid_kron"]),
        default_on=False,
    )
    if radius_values["euclid_sersic_re"]:
        parameters["radius"]["radius_series"]["euclid_sersic_re"] = (
            _radius_curve(
                np.asarray(radius_values["euclid_sersic_re"]),
                area,
                label="Euclid PHZ/MER · VIS Sérsic Rₑ",
                source="euclid",
                radius_type="half_light",
                definition=(
                    "MER morphology VIS Sérsic effective radius in arcsec; "
                    "SERSIC_VISNIR_FLAGS = 0; weighted by PHZ_GAL_PROB"
                ),
                weights=np.asarray(radius_weights["euclid_sersic_re"]),
                default_on=True,
            )
        )
    elif not has_sersic_re:
        parameters["radius"]["radius_missing"].append(
            "Euclid PHZ/MER VIS Sérsic Rₑ requires a version-7 MER + "
            "morphology cache; re-query the catalogue to add it."
        )
    else:
        parameters["radius"]["radius_missing"].append(
            "The MER morphology cache contains no clean positive VIS Sérsic "
            "Rₑ values with PHZ_GAL_PROB > 0."
        )
    if redshift is not None and pdf_edges is not None:
        pdf_source = str(meta.get("phz_pdf_source") or "archive_full_pdf")
        definition = (
            "PHZ_GAL_PROB × PDF reconstructed from cached modes"
            if pdf_source == "summary_reconstruction"
            else "PHZ_GAL_PROB × rebinned archive PHZ PDF"
        )
        parameters["redshift"]["series"]["euclid"] = _curve(
            pdf_edges, redshift, area, definition,
        )
    if physical_rows:
        parameters["stellar_mass"]["series"]["euclid"] = _curve(
            MASS_EDGES, mass, area, "valid PHZ physical posterior, galaxy-weighted"
        )
        parameters["specific_sfr"]["series"]["euclid"] = _curve(
            SSFR_EDGES, ssfr, area, "PHZ log SFR − log mass, galaxy-weighted"
        )
    coverage = meta.get("phz_coverage") or {}
    return {
        "available": True,
        "rows": rows,
        "area_arcmin2": area,
        "schema_version": meta.get("catalog_version"),
        "phz_pdf_rows": phz_rows,
        "physical_rows": physical_rows,
        "phz_pdf_source": meta.get("phz_pdf_source"),
        "phz_pdf_activation_eligible": meta.get("phz_pdf_activation_eligible"),
        "phz_coverage": coverage,
        "aperture_growth": _aperture_growth_payload(
            aperture_histograms, aperture_counts,
        ),
        "aperture_scatter": _aperture_scatter_payload([
            (-negative_priority, object_id, sample)
            for heap in aperture_scatter_heaps
            for negative_priority, object_id, sample in heap
        ]),
        "detail": (
            "MER cache ready with locally reconstructed PHZ summaries"
            if redshift is not None and meta.get("phz_pdf_source") == "summary_reconstruction"
            else "MER cache ready with archive PHZ PDFs"
            if redshift is not None
            else "MER cache ready; PHZ sidecar missing"
        ),
    }


def _read_cosmos(parameters: dict[str, Any]) -> dict[str, Any]:
    path = Path(Config.COSMOS_POPULATION_PRIOR_PATH)
    if not path.is_file():
        return {"available": False, "detail": "COSMOS2025 population prior is missing."}
    cosmos = read_cosmos_population(path)
    definitions = {
        "redshift": (cosmos["redshift"], np.linspace(0.05, 5.5, 45), "COSMOS2025 LePhare photo-z"),
        "magnitude": (
            cosmos["magnitude"], MAG_EDGES,
            "HST F814W single-Sérsic total AB magnitude",
        ),
        "radius": (
            np.log10(cosmos["radius_arcsec"][cosmos["has_radius"]]),
            LOG_RADIUS_EDGES,
            "combined half-light radius",
        ),
        "stellar_mass": (cosmos["logmass"], MASS_EDGES, "LePhare stellar mass"),
        "specific_sfr": (cosmos["logssfr"], SSFR_EDGES, "LePhare specific SFR"),
    }
    for key, (values, edges, definition) in definitions.items():
        finite = np.isfinite(values)
        if key == "specific_sfr":
            finite &= values < -8.2
        parameters[key]["series"]["cosmos"] = _curve(
            edges, np.histogram(values[finite], edges)[0], COSMOS_AREA_ARCMIN2, definition
        )
    cosmos_radius = np.asarray(cosmos["radius_arcsec"], dtype=np.float64)
    cosmos_radius_valid = np.isfinite(cosmos_radius) & (cosmos_radius > 0.0)
    parameters["radius"]["radius_series"]["cosmos_re"] = _radius_curve(
        cosmos_radius[cosmos_radius_valid],
        COSMOS_AREA_ARCMIN2,
        label="COSMOS · combined Rₑ",
        source="cosmos",
        radius_type="half_light",
        definition="COSMOS2025 combined circularized bulge+disk half-light radius",
        default_on=False,
    )
    with np.load(path, allow_pickle=False) as prior:
        available = set(prior.files)
        reference_magnitude = np.asarray(prior["mag_hst_f814w"], dtype=np.float64)
        redshift = np.asarray(prior["z_phot"], dtype=np.float64)
        population = (
            np.isfinite(reference_magnitude)
            & np.isfinite(redshift)
            & (reference_magnitude >= COSMOS_FIT_MAG_MIN)
            & (reference_magnitude < MAG_EDGES[-1])
            & (redshift >= COSMOS_FIT_Z_MIN)
            & (redshift < COSMOS_FIT_Z_MAX)
        )
        selection = (
            "COSMOS2025 population galaxies: LePHARE TYPE = 0, FLAG_STAR = 0, "
            f"{COSMOS_FIT_Z_MIN:g} <= z < {COSMOS_FIT_Z_MAX:g}; finite native magnitude"
        )
        scalar_series = (
            (
                "cosmos_f814w_model", "mag_hst_f814w",
                "F814W · profile total", "SE++ profile-model total", True,
            ),
            (
                "cosmos_f814w_auto", "mag_auto_hst_f814w",
                "F814W · AUTO", "Kron-like AUTO with aperture and PSF correction", False,
            ),
            (
                "cosmos_f814w_bd_total", "mag_bd_hst_f814w",
                "F814W · B+D total", "SE++ bulge+disk-model total", True,
            ),
            (
                "cosmos_f814w_bulge", "mag_bulge_hst_f814w",
                "F814W · bulge", "SE++ bulge-component model", False,
            ),
            (
                "cosmos_f814w_disk", "mag_disk_hst_f814w",
                "F814W · disk", "SE++ disk-component model", False,
            ),
        )
        for key, array_key, label, estimator, default_on in scalar_series:
            if array_key not in available:
                continue
            values = _valid_ab_magnitude(np.asarray(prior[array_key], dtype=np.float64))
            parameters["magnitude"]["photometry_series"][key] = _brightness_curve(
                values[population],
                COSMOS_AREA_ARCMIN2,
                label=label,
                survey="cosmos",
                band="HST/ACS F814W",
                estimator=estimator,
                selection=selection,
                default_on=default_on,
            )

        aperture_arrays = (
            (
                "homogenized", "mag_aper_hst_f814w", "PSF-homogenized circular aperture",
            ),
            (
                "native", "mag_native_aper_hst_f814w", "native-PSF circular aperture",
            ),
        )
        for family, array_key, estimator in aperture_arrays:
            if array_key not in available:
                parameters["magnitude"]["photometry_missing"].append(
                    f"COSMOS {family}-PSF F814W aperture vectors are absent from the compact prior; "
                    "rerun the COSMOS2025 extraction against the master FITS"
                )
                continue
            diameters = COSMOS_APERTURE_DIAMETERS_ARCSEC[family]
            aperture_values = _aperture_columns(prior[array_key], len(diameters))
            if aperture_values is None or aperture_values.shape[0] != population.size:
                continue
            for index, diameter in enumerate(diameters):
                key = f"cosmos_f814w_{family}_aper_{index + 1}"
                parameters["magnitude"]["photometry_series"][key] = _brightness_curve(
                    _valid_ab_magnitude(aperture_values[:, index])[population],
                    COSMOS_AREA_ARCMIN2,
                    label=f"F814W · {diameter:g}″ diameter",
                    survey="cosmos",
                    band="HST/ACS F814W",
                    estimator=estimator,
                    selection=selection,
                    default_on=False,
                )
    return {
        "available": True,
        "rows": int(len(cosmos["magnitude"])),
        "area_arcmin2": COSMOS_AREA_ARCMIN2,
        "measured_size_rows": int(np.sum(cosmos["has_radius"])),
        "detail": "COSMOS2025 diagnostic only; excluded from all fitting",
    }


def _read_fit(parameters: dict[str, Any]) -> dict[str, Any]:
    photometry_series = parameters["magnitude"]["photometry_series"]
    photometry_series.pop("generator_vis_f2", None)
    radius_series = parameters["radius"]["radius_series"]
    for key in [
        item for item, curve in radius_series.items()
        if item.startswith("fit_re") or curve.get("source") == "fit"
    ]:
        del radius_series[key]
    source = joint_galaxy_candidate()
    if not source:
        return {
            "available": False,
            "detail": "Fit the Euclid VIS 2FWHM × Sérsic Rₑ model first.",
        }
    try:
        radius_plot = source["plots"]["radius"]
        radius_x = np.asarray(radius_plot["x"], dtype=np.float64)
        radius_density = np.asarray(radius_plot["density"], dtype=np.float64)
        q1_weighted_radius_density = np.asarray(
            radius_plot["q1_weighted_density"], dtype=np.float64,
        )
        radius_law = source["radius_law"]
        radius_min = float(radius_law["log_radius_min"])
        radius_max = float(radius_law["log_radius_max"])
        generation_plot = source["magnitude_plot"]["generation_law"]
        generation_x = np.asarray(generation_plot["x"], dtype=np.float64)
        generation_density = np.asarray(
            generation_plot["density"], dtype=np.float64,
        )
        generation = source["generation"]
        generation_interval = [
            float(generation["vis_magnitude_min"]),
            float(generation["vis_magnitude_max"]),
        ]
        density_cap = float(
            generation["differential_density_cap_arcmin2_mag"]
        )
        break_magnitude = float(generation["break_magnitude"])
        magnitude_law = source["magnitude_law"]
        if magnitude_law.get("kind") != (
            "continuous_three_slope_bright_bridge_main_flat_faint_counts"
        ):
            raise ValueError("candidate does not use the v11 magnitude law")
        bright_join_magnitudes = [
            float(value) for value in magnitude_law["bright_join_magnitudes"]
        ]
        bright_slopes = [
            float(value) for value in magnitude_law["bright_slopes"]
        ]
        if len(bright_join_magnitudes) != 3 or len(bright_slopes) != 3:
            raise ValueError("candidate bright bridge must have three segments")
        main_slope = float(magnitude_law["straight_law"]["slope"])
    except (KeyError, TypeError, ValueError) as exc:
        return {"available": False, "detail": f"Fit artifact cannot be reconstructed: {exc}"}
    if (
        radius_x.ndim != 1
        or radius_x.size < 2
        or radius_density.shape != radius_x.shape
        or q1_weighted_radius_density.shape != radius_x.shape
        or not np.all(np.diff(radius_x) > 0.0)
        or not radius_min < radius_x[0] < radius_x[-1] < radius_max
    ):
        return {
            "available": False,
            "detail": "Fit artifact cannot be reconstructed: radius grid is malformed",
        }
    radius_edges = np.empty(radius_x.size + 1, dtype=np.float64)
    radius_edges[0] = radius_min
    radius_edges[-1] = radius_max
    radius_edges[1:-1] = 0.5 * (radius_x[:-1] + radius_x[1:])
    radius_width = np.diff(radius_edges)
    try:
        full_radius_shape = _normalized_density(radius_density, radius_width)
        q1_weighted_radius_shape = _normalized_density(
            q1_weighted_radius_density, radius_width,
        )
    except ValueError as exc:
        return {
            "available": False,
            "detail": f"Fit artifact cannot be reconstructed: {exc}",
        }
    photometry_series["generator_vis_f2"] = {
        "x": generation_x.tolist(),
        "density": generation_density.tolist(),
        "weighted_count": float(generation["surface_density_arcmin2"]),
        "definition": (
            "generation law: three fitted continuous bright-bridge segments "
            "ending at fixed joins, followed by the main Q1 VIS 2FWHM line "
            "and constant differential density through VIS 29"
        ),
        "label": "Generator · three-segment bright bridge + main + flat",
        "survey": "generation",
        "band": "Euclid VIS",
        "estimator": "2FWHM aperture magnitude; generated count law",
        "selection": (
            "Q1 MER + PHZ galaxy fit; faint differential density capped at "
            f"{density_cap:g} objects / arcmin2 / mag"
        ),
        "default_on": True,
        "fit_interval": [
            float(source["fitted_magnitude_law"]["fit_bright"]),
            float(source["fitted_magnitude_law"]["fit_faint"]),
        ],
        "generation_interval": generation_interval,
        "generation_bright_join_magnitudes": bright_join_magnitudes,
        "generation_bright_slopes": bright_slopes,
        "generation_main_slope": main_slope,
        "generation_break_magnitude": break_magnitude,
        "generation_density_cap_arcmin2_mag": density_cap,
    }
    radius_series["fit_re"] = {
        "x": radius_x.tolist(),
        "density": radius_density.tolist(),
        "weighted_count": float(source["generation"]["surface_density_arcmin2"]),
        "definition": (
            "nominal continuous-space circularized Euclid Sérsic Rₑ generated "
            "by the joint fit marginalized over VIS 2FWHM brightness; not "
            "remeasured after TNG pixel rendering"
        ),
        "label": "Generator · circularized Euclid Sérsic Rₑ",
        "source": "fit",
        "radius_type": "half_light",
        "units": "arcsec",
        "normalization": "surface_density",
        "default_on": False,
    }
    radius_series["fit_re_q1_weighted_shape"] = {
        "x": radius_x.tolist(),
        "density": q1_weighted_radius_shape.tolist(),
        "weighted_count": float(np.sum(
            q1_weighted_radius_density * radius_width
        )),
        "definition": (
            "candidate straight truncated-Gaussian circularized Sérsic Rₑ "
            "law weighted by the clean Q1 VIS 2FWHM magnitude brackets, "
            "then normalized to unit probability"
        ),
        "label": (
            "Candidate · Q1-magnitude-weighted circularized Sérsic Rₑ"
        ),
        "source": "fit",
        "radius_type": "half_light_shape",
        "units": "arcsec",
        "normalization": "probability_density",
        "default_on": True,
    }
    radius_series["fit_re_full_generation_shape"] = {
        "x": radius_x.tolist(),
        "density": full_radius_shape.tolist(),
        "weighted_count": float(source["generation"]["surface_density_arcmin2"]),
        "definition": (
            "candidate straight truncated-Gaussian circularized Sérsic Rₑ "
            "law marginalized over the complete VIS 14-29 generation law, "
            "including the flat faint extension, then normalized to unit "
            "probability"
        ),
        "label": (
            "Candidate · full-generation circularized Sérsic Rₑ "
            "(faint extension)"
        ),
        "source": "fit",
        "radius_type": "half_light_shape",
        "units": "arcsec",
        "normalization": "probability_density",
        "default_on": True,
    }
    state = joint_galaxy_state()
    return {
        "available": True,
        "version": source.get("version"),
        "fingerprint": source.get("fingerprint"),
        "validated": bool(source.get("validated")),
        "is_active": bool(state.get("is_active")),
        "active_fingerprint": ((state.get("active") or {}).get("fingerprint")),
        "detail": (
            "Euclid continuous three-segment bright bridge/main/flat "
            "brightness × one straight truncated-Gaussian "
            "circularized-Sérsic-radius fit with no radius tail or break; "
            f"counts flatten at VIS {break_magnitude:.2f}"
        ),
    }


def _training_variant(
    parameters: dict[str, Any],
    current_synthetic: dict[str, Any],
) -> dict[str, Any] | None:
    """Build the catalogue-only all-split overlay without training images."""
    _, source_paths = _synthetic_paths(include_training=True)
    if not any(path.stem == "sources_train" for path in source_paths):
        return None
    training_parameters = copy.deepcopy(parameters)
    synthetic = _read_synthetic(
        training_parameters,
        include_training=True,
        measure_clean_images=False,
    )
    if not synthetic.get("training_included"):
        return None
    # Direct clean-image radii are deliberately kept from test+validation.
    # They remain a valid curve with their own area; the all-split catalogue
    # pass must not erase or renormalize them over the training footprint.
    for key in (
        "measured_radius_rows", "measured_radius_fraction",
    ):
        synthetic[key] = current_synthetic.get(key, 0)
    coverage = synthetic.setdefault("parameter_coverage", {})
    current_coverage = current_synthetic.get("parameter_coverage", {})
    if "clean_half_light" in current_coverage:
        coverage["clean_half_light"] = current_coverage["clean_half_light"]
    joint_maps = _joint_magnitude_radius_maps(synthetic)
    return {
        "sources": {"synthetic": synthetic},
        "parameters": training_parameters,
        "joint_maps": joint_maps,
    }


def build_galaxy_distributions(progress: Callable[[int, int, str], None] | None = None) -> dict[str, Any]:
    tick = progress or (lambda _done, _total, _label: None)
    parameters = _empty_parameters()
    tick(0, 6, "read Euclid MER + PHZ")
    euclid = _read_euclid(parameters, tick)
    tick(2, 6, "read progressive Q1 MER + PHZ bright-galaxy counts")
    q1_counts = _read_q1_bright_counts(parameters)
    q1_radius = _read_q1_radius_statistics(parameters)
    tick(3, 6, "read COSMOS2025")
    cosmos = _read_cosmos(parameters)
    tick(4, 6, "measure current generated fields")
    synthetic = _read_synthetic(parameters, tick)
    tick(5, 6, "reconstruct fitted distribution")
    fit = _read_fit(parameters)
    joint_maps = _joint_magnitude_radius_maps(synthetic)
    training_variant = _training_variant(parameters, synthetic)
    payload = {
        "version": ARTIFACT_VERSION,
        "inputs": _inputs(),
        "sources": {
            "euclid": euclid, "synthetic": synthetic,
            "cosmos": cosmos, "fit": fit,
            "q1_radius": q1_radius,
        },
        "q1_counts": q1_counts,
        "q1_radius": q1_radius,
        "joint_maps": joint_maps,
        "parameters": parameters,
        "training_variant": training_variant,
    }
    path = artifact_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, separators=(",", ":")))
    os.replace(temporary, path)
    tick(6, 6, "galaxy-distribution plots ready")
    return payload


def read_galaxy_distributions(
    *, include_training: bool = False,
) -> dict[str, Any]:
    payload = _json(artifact_path())
    stale = not payload or payload.get("version") != ARTIFACT_VERSION or payload.get("inputs") != _inputs()
    if not payload:
        payload = {
            "version": ARTIFACT_VERSION,
            "sources": {},
            "q1_counts": {"available": False},
            "q1_radius": {"available": False},
            "joint_maps": {"available": False},
            "parameters": _empty_parameters(),
        }
    training_variant = payload.pop("training_variant", None)
    training_available = isinstance(training_variant, dict)
    if include_training and training_available:
        variant_sources = training_variant.get("sources")
        if isinstance(variant_sources, dict):
            payload.setdefault("sources", {}).update(variant_sources)
        if isinstance(training_variant.get("parameters"), dict):
            payload["parameters"] = training_variant["parameters"]
        if isinstance(training_variant.get("joint_maps"), dict):
            payload["joint_maps"] = training_variant["joint_maps"]
    payload["training_included"] = bool(include_training and training_available)
    payload["training_variant_available"] = training_available

    # The archive job checkpoints after every aperture/bin request. Overlay
    # that small cache at read time so the plot can advance while the longer
    # query is still running, without rebuilding the other population layers.
    parameters = payload.get("parameters")
    if isinstance(parameters, dict):
        magnitude = parameters.get("magnitude")
        series = magnitude.get("photometry_series") if isinstance(magnitude, dict) else None
        if isinstance(series, dict):
            for key in [item for item in series if item.startswith("q1_vis_")]:
                del series[key]
        payload["q1_counts"] = _read_q1_bright_counts(parameters)
        payload["q1_radius"] = _read_q1_radius_statistics(parameters)
        sources = payload.setdefault("sources", {})
        if isinstance(sources, dict):
            sources["fit"] = _read_fit(parameters)
    return {**payload, "stale": stale, "artifact_path": str(artifact_path())}
