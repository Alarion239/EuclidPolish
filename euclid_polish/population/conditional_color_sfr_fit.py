"""Fit the conditional colour+SFR forest from the cached Euclid Q1 sample.

This is the ONLY module in the population package that imports scikit-learn:
the forest is grown here at calibration time and exported as plain arrays
(no pickle) into the joint-galaxy artifact, so the generation path stays pure
NumPy (:mod:`euclid_polish.population.conditional_color_sfr`).

Row selection deliberately keeps the faint end. MER NISP photometry is forced
at the VIS position, so negative NISP aperture fluxes are unbiased noisy
measurements and are KEPT — filtering them would truncation-bias the faint
colour distribution. Only a non-positive VIS flux (the detection band) drops a
row. Colours live in NISP/VIS flux-ratio space with per-row noise variances
propagated from the catalogue flux errors; the sampler deconvolves them at
draw time.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

from euclid_polish.photometry import uJy_to_ab_mag
from euclid_polish.population._codec import array_sha256, encode_array
from euclid_polish.population.conditional_color_sfr import (
    COLOR_SFR_ACTIVITY_THRESHOLD_LOGSSFR,
    COLOR_SFR_ASINH_SOFTENING,
    COLOR_SFR_FEATURE_NAMES,
    COLOR_SFR_FIT_SEED,
    COLOR_SFR_FOREST_TREES,
    COLOR_SFR_MIN_LEAF_WEIGHT,
    COLOR_SFR_MODEL_KIND,
    COLOR_SFR_MODEL_VERSION,
    COLOR_SFR_PATHOLOGICAL_LOGSSFR,
    COLOR_SFR_RATIO_FLOOR,
    COLOR_SFR_VIS_SNR_FLOOR,
    SFR_CLASS_QUENCHED,
    SFR_CLASS_STAR_FORMING,
    SFR_CLASS_UNKNOWN,
    ConditionalColorSFRSampler,
    weighted_mid_quantiles,
    weighted_quantile,
)
from euclid_polish.population.euclid_galaxy_prior import ConditionalRadiusLaw

_RATIO_COLUMNS = ("y", "j", "h")
_SELECTION = (
    "cached Q1 MER+PHZ rows with VIS_DET = 1, DET_QUALITY_FLAG < 4, "
    "non-spurious retained flags/probability, POINT_LIKE_PROB < 0.5, "
    "positive FLUX_VIS_2FWHM_APER with S/N >= "
    f"{COLOR_SFR_VIS_SNR_FLOOR:g} (the VIS flux sits in the ratio "
    "denominator, so below its own noise the measured colours are "
    "meaningless), and finite NISP 2FWHM aperture fluxes with positive "
    "reported errors; negative NISP fluxes are kept (forced photometry); "
    "no NISP magnitude cutoff and no PHZ_GAL_PROB gate"
)
_WEIGHT_POLICY = "1 - POINT_LIKE_PROB"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _optional_float(row: dict[str, str], key: str) -> float | None:
    text = str(row.get(key) or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return float("nan")


class ColorSFRRows(NamedTuple):
    """The colour-model row selection, one array entry per selected row.

    ``resolved`` marks a usable circularized Sérsic radius; unresolved rows
    carry the radius-law median in ``log_radius``. Rows without a usable PHZ
    physical fit carry NaN ``log_sfr`` and class ``SFR_CLASS_UNKNOWN``.
    """

    magnitude: np.ndarray
    log_radius: np.ndarray
    resolved: np.ndarray
    ratio: np.ndarray
    ratio_var: np.ndarray
    weight: np.ndarray
    log_sfr: np.ndarray
    sfr_valid: np.ndarray
    sfr_class: np.ndarray
    catalog_rows: int


def read_color_sfr_rows(
    catalog_path: str | Path,
    *,
    radius_law: ConditionalRadiusLaw,
) -> ColorSFRRows:
    """Stream the cached catalogue and apply the colour-model row selection."""
    catalog_path = Path(catalog_path)
    if not catalog_path.is_file():
        raise ValueError("A cached Euclid population catalogue is required")

    magnitude: list[float] = []
    log_radius: list[float] = []
    resolved: list[bool] = []
    ratio: list[tuple[float, float, float]] = []
    ratio_var: list[tuple[float, float, float]] = []
    weight: list[float] = []
    log_sfr: list[float] = []
    sfr_valid: list[bool] = []
    sfr_class: list[int] = []
    catalog_rows = 0
    with catalog_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            catalog_rows += 1
            try:
                flux_vis = float(row["flux_vis_2fwhm_aper_uJy"])
                fluxerr_vis = float(row["fluxerr_vis_2fwhm_aper_uJy"])
                vis_det = float(row["vis_det"])
                det_quality_flag = float(row["det_quality_flag"])
                spurious_probability = float(row["spurious_prob"])
                point_like_probability = float(row["point_like_prob"])
            except (KeyError, TypeError, ValueError):
                continue
            flag_vis = _optional_float(row, "flag_vis")
            spurious_flag = _optional_float(row, "spurious_flag")
            point_like_flag = _optional_float(row, "point_like_flag")
            if not (
                np.isfinite(flux_vis)
                and flux_vis > 0.0
                and np.isfinite(fluxerr_vis)
                and fluxerr_vis > 0.0
                and flux_vis / fluxerr_vis >= COLOR_SFR_VIS_SNR_FLOOR
                and vis_det == 1.0
                and np.isfinite(det_quality_flag)
                and det_quality_flag < 4.0
                and np.isfinite(spurious_probability)
                and spurious_probability <= 0.5
                and np.isfinite(point_like_probability)
                and 0.0 <= point_like_probability < 0.5
                and (flag_vis is None or flag_vis == 0.0)
                and (spurious_flag is None or spurious_flag == 0.0)
                and (point_like_flag is None or point_like_flag != 1.0)
            ):
                continue
            try:
                nisp_flux = tuple(
                    float(row[f"flux_{band}_2fwhm_aper_uJy"])
                    for band in _RATIO_COLUMNS
                )
                nisp_error = tuple(
                    float(row[f"fluxerr_{band}_2fwhm_aper_uJy"])
                    for band in _RATIO_COLUMNS
                )
            except (KeyError, TypeError, ValueError):
                continue
            if not all(
                np.isfinite(flux) and np.isfinite(error) and error > 0.0
                for flux, error in zip(nisp_flux, nisp_error, strict=True)
            ):
                continue
            galaxy_weight = 1.0 - point_like_probability
            row_ratio = tuple(flux / flux_vis for flux in nisp_flux)
            row_variance = tuple(
                (error**2 + value**2 * fluxerr_vis**2) / flux_vis**2
                for value, error in zip(row_ratio, nisp_error, strict=True)
            )
            row_magnitude = float(uJy_to_ab_mag(flux_vis))

            morph_radius = _optional_float(
                row, "morph_sersic_vis_radius_arcsec",
            )
            morph_axis_ratio = _optional_float(
                row, "morph_sersic_vis_axis_ratio",
            )
            morph_flags = _optional_float(row, "morph_sersic_visnir_flags")
            row_resolved = bool(
                morph_radius is not None
                and morph_axis_ratio is not None
                and np.isfinite(morph_radius)
                and morph_radius > 0.0
                and np.isfinite(morph_axis_ratio)
                and 0.0 < morph_axis_ratio <= 1.0
                and (morph_flags is None or morph_flags == 0.0)
            )
            if row_resolved:
                circularized = morph_radius * math.sqrt(morph_axis_ratio)
                row_resolved = 0.03 <= circularized < 10.0
            if row_resolved:
                row_log_radius = math.log10(circularized)
            else:
                row_log_radius = float(radius_law.mean(row_magnitude))

            sfr = _optional_float(row, "phz_pp_median_sfr")
            mass = _optional_float(row, "phz_pp_median_stellarmass")
            phys_flags = _optional_float(row, "phz_phys_flags")
            phys_quality = _optional_float(row, "phz_phys_quality_flag")
            row_sfr_valid = bool(
                sfr is not None and mass is not None
                and np.isfinite(sfr) and np.isfinite(mass)
                and (phys_flags is None or phys_flags == 0.0)
                and (phys_quality is None or phys_quality == 0.0)
                and (sfr - mass) < COLOR_SFR_PATHOLOGICAL_LOGSSFR
            )
            if row_sfr_valid:
                row_class = (
                    SFR_CLASS_QUENCHED
                    if (sfr - mass) < COLOR_SFR_ACTIVITY_THRESHOLD_LOGSSFR
                    else SFR_CLASS_STAR_FORMING
                )
                row_log_sfr = float(sfr)
            else:
                row_class = SFR_CLASS_UNKNOWN
                row_log_sfr = float("nan")

            magnitude.append(row_magnitude)
            log_radius.append(row_log_radius)
            resolved.append(row_resolved)
            ratio.append(row_ratio)
            ratio_var.append(row_variance)
            weight.append(galaxy_weight)
            log_sfr.append(row_log_sfr)
            sfr_valid.append(row_sfr_valid)
            sfr_class.append(row_class)

    return ColorSFRRows(
        magnitude=np.asarray(magnitude, dtype=np.float64),
        log_radius=np.asarray(log_radius, dtype=np.float64),
        resolved=np.asarray(resolved, dtype=bool),
        ratio=np.asarray(ratio, dtype=np.float64).reshape(-1, 3),
        ratio_var=np.asarray(ratio_var, dtype=np.float64).reshape(-1, 3),
        weight=np.asarray(weight, dtype=np.float64),
        log_sfr=np.asarray(log_sfr, dtype=np.float64),
        sfr_valid=np.asarray(sfr_valid, dtype=bool),
        sfr_class=np.asarray(sfr_class, dtype=np.uint8),
        catalog_rows=catalog_rows,
    )


def fit_conditional_color_sfr_payload(
    catalog_path: str | Path,
    meta_path: str | Path,
    *,
    radius_law: ConditionalRadiusLaw,
    expected_catalog_version: int = 7,
    require_catalog_version: bool = True,
    tree_count: int = COLOR_SFR_FOREST_TREES,
    min_leaf_weight: float = COLOR_SFR_MIN_LEAF_WEIGHT,
    fit_seed: int = COLOR_SFR_FIT_SEED,
    minimum_rows: int = 200,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the ``color_sfr_model`` artifact section plus fit diagnostics.

    ``require_catalog_version`` exists only for local end-to-end smokes on a
    stale cache (every row lands in the unresolved-radius regime there);
    production fits must keep it enabled.
    """
    catalog_path = Path(catalog_path)
    meta_path = Path(meta_path)
    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Euclid population metadata is unavailable") from exc
    catalog_version = int(meta.get("catalog_version") or 0)
    if require_catalog_version and catalog_version != expected_catalog_version:
        raise ValueError(
            f"colour+SFR fit requires catalog_version "
            f"{expected_catalog_version}, got {catalog_version}; re-run the "
            "population query to refresh the cache"
        )
    selected = read_color_sfr_rows(catalog_path, radius_law=radius_law)
    catalog_rows = selected.catalog_rows
    row_count = int(selected.weight.size)
    if row_count < int(minimum_rows):
        raise ValueError(
            f"colour+SFR fit selected only {row_count} rows; at least "
            f"{int(minimum_rows)} are required"
        )
    expected_catalog_rows = int(meta.get("rows") or 0)
    if expected_catalog_rows and expected_catalog_rows != catalog_rows:
        raise ValueError("The cached Euclid catalogue row count is stale")

    magnitude_array = selected.magnitude
    log_radius_array = selected.log_radius
    resolved_array = selected.resolved.astype(np.float64)
    ratio_array = selected.ratio
    ratio_var_array = selected.ratio_var
    weight_array = selected.weight
    log_sfr_array = selected.log_sfr
    sfr_valid_array = selected.sfr_valid
    sfr_class_array = selected.sfr_class
    if not np.any(sfr_valid_array):
        raise ValueError("colour+SFR fit found no usable PHZ SFR rows")

    sfr_rank_array = np.full(row_count, np.nan, dtype=np.float64)
    sfr_rank_array[sfr_valid_array] = weighted_mid_quantiles(
        log_sfr_array[sfr_valid_array], weight_array[sfr_valid_array],
    )

    # Split-criterion targets: asinh-softened ratios plus log SFR, with
    # missing SFRs imputed FIT-TIME-ONLY by k-NN over (magnitude, colours) so
    # SFR has a voice in where the regime boundaries fall. Sampled SFR values
    # always come from real SFR-valid rows (the sampler's borrow policy).
    softened = np.arcsinh(ratio_array / COLOR_SFR_ASINH_SOFTENING)
    knn_features = np.column_stack((magnitude_array, softened))
    feature_scale = np.maximum(np.std(knn_features, axis=0), 1e-6)
    knn_features = knn_features / feature_scale[None, :]
    filled_log_sfr = log_sfr_array.copy()
    missing = ~sfr_valid_array
    if np.any(missing):
        neighbors = KNeighborsRegressor(
            n_neighbors=int(min(32, int(np.sum(sfr_valid_array)))),
            weights="distance",
        )
        neighbors.fit(
            knn_features[sfr_valid_array], log_sfr_array[sfr_valid_array],
        )
        filled_log_sfr[missing] = neighbors.predict(knn_features[missing])

    targets = np.column_stack((softened, filled_log_sfr))
    target_scale = np.maximum(np.std(targets, axis=0), 1e-9)
    targets = (targets - np.mean(targets, axis=0)) / target_scale[None, :]
    features = np.column_stack(
        (magnitude_array, log_radius_array, resolved_array),
    )
    total_weight = float(np.sum(weight_array))
    leaf_fraction = float(
        min(0.5, max(1e-9, float(min_leaf_weight) / total_weight))
    )
    forest = RandomForestRegressor(
        n_estimators=int(tree_count),
        bootstrap=True,
        min_weight_fraction_leaf=leaf_fraction,
        random_state=int(fit_seed),
        n_jobs=-1,
    )
    forest.fit(features, targets, sample_weight=weight_array)
    leaf_ids = forest.apply(features)

    trees: list[dict[str, Any]] = []
    tree_digest = hashlib.sha256()
    leaf_counts: list[int] = []
    leaf_row_sizes: list[float] = []
    for tree_index, estimator in enumerate(forest.estimators_):
        structure = estimator.tree_
        node_count = int(structure.node_count)
        assignments = leaf_ids[:, tree_index].astype(np.int64)
        row_order = np.argsort(assignments, kind="mergesort").astype(np.int32)
        counts = np.bincount(assignments, minlength=node_count)
        node_row_start = np.zeros(node_count + 1, dtype=np.int32)
        node_row_start[1:] = np.cumsum(counts, dtype=np.int64)
        arrays = {
            "feature": np.asarray(structure.feature, dtype=np.int32),
            "threshold": np.asarray(structure.threshold, dtype=np.float64),
            "children_left": np.asarray(
                structure.children_left, dtype=np.int32,
            ),
            "children_right": np.asarray(
                structure.children_right, dtype=np.int32,
            ),
            "row_order": row_order,
            "node_row_start": node_row_start,
        }
        encoded = {"node_count": node_count}
        for name, values in arrays.items():
            dtype = "<f8" if name == "threshold" else "<i4"
            encoded[f"{name}_zlib_base64"] = encode_array(values, dtype)
            tree_digest.update(
                np.asarray(values, dtype=dtype, order="C").tobytes(order="C")
            )
        trees.append(encoded)
        leaf_counts.append(int(np.sum(structure.children_left == -1)))
        leaf_row_sizes.append(float(np.median(counts[counts > 0])))

    rows_section: dict[str, Any] = {}
    row_arrays = {
        "ratio": (ratio_array, "<f4"),
        "ratio_var": (ratio_var_array, "<f4"),
        "log_sfr": (log_sfr_array, "<f4"),
        "weight": (weight_array, "<f4"),
        "sfr_rank": (sfr_rank_array, "<f4"),
        "sfr_valid": (sfr_valid_array.astype(np.uint8), "|u1"),
        "sfr_class": (sfr_class_array, "|u1"),
    }
    for name, (values, dtype) in row_arrays.items():
        rows_section[f"{name}_zlib_base64"] = encode_array(values, dtype)
        rows_section[f"{name}_sha256"] = array_sha256(values, dtype)

    catalog_sha256 = _sha256_file(catalog_path)
    identity = {
        "version": COLOR_SFR_MODEL_VERSION,
        "kind": COLOR_SFR_MODEL_KIND,
        "catalog_version": catalog_version,
        "catalog_sha256": catalog_sha256,
        "row_count": row_count,
        "tree_count": int(tree_count),
        "min_leaf_weight": float(min_leaf_weight),
        "fit_seed": int(fit_seed),
        "vis_snr_floor": COLOR_SFR_VIS_SNR_FLOOR,
        "selection": _SELECTION,
        "weight_policy": _WEIGHT_POLICY,
        "row_sha256": {
            name: rows_section[f"{name}_sha256"] for name in row_arrays
        },
        "tree_sha256": tree_digest.hexdigest(),
        "re_imputation": {
            "pivot_mag": float(radius_law.pivot_mag),
            "intercept_log10_arcsec": float(
                radius_law.intercept_log10_arcsec
            ),
            "slope_log10_arcsec_per_mag": float(
                radius_law.slope_log10_arcsec_per_mag
            ),
        },
    }
    calibration_fingerprint = hashlib.sha256(json.dumps(
        identity, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()

    payload = {
        "version": COLOR_SFR_MODEL_VERSION,
        "kind": COLOR_SFR_MODEL_KIND,
        "row_count": row_count,
        "tree_count": int(tree_count),
        "feature_names": list(COLOR_SFR_FEATURE_NAMES),
        "color_space": "nisp_over_vis_2fwhm_flux_ratio",
        "ratio_floor": COLOR_SFR_RATIO_FLOOR,
        "asinh_softening": COLOR_SFR_ASINH_SOFTENING,
        "sfr_units": "log10 Msun/yr (PHZ_PP_MEDIAN_SFR)",
        "activity_threshold_logssfr": COLOR_SFR_ACTIVITY_THRESHOLD_LOGSSFR,
        "pathological_logssfr": COLOR_SFR_PATHOLOGICAL_LOGSSFR,
        "deconvolution": (
            "analytic one-component extreme deconvolution within the forest "
            "neighbourhood, on ROBUST statistics (weighted-median location, "
            "16-84 half-width observed spread, weighted-median reported "
            "noise variance): intrinsic variance = observed minus noise, "
            "floored at zero; the drawn row's true ratio is sampled from "
            "its Gaussian posterior"
        ),
        "vis_snr_floor": COLOR_SFR_VIS_SNR_FLOOR,
        "selection": _SELECTION,
        "weight_policy": _WEIGHT_POLICY,
        "min_leaf_weight": float(min_leaf_weight),
        "min_weight_fraction_leaf": leaf_fraction,
        "fit_seed": int(fit_seed),
        "sklearn_version": str(sklearn.__version__),
        "catalog_version": catalog_version,
        "catalog_sha256": catalog_sha256,
        "tree_sha256": tree_digest.hexdigest(),
        "calibration_fingerprint": calibration_fingerprint,
        "re_imputation": identity["re_imputation"],
        "rows": rows_section,
        "trees": trees,
    }
    sampler = ConditionalColorSFRSampler(payload)

    magnitude_edges = np.arange(
        math.floor(float(np.min(magnitude_array))),
        math.ceil(float(np.max(magnitude_array))) + 0.5,
        0.5,
    )
    bin_index = np.clip(
        np.searchsorted(magnitude_edges, magnitude_array, side="right") - 1,
        0, magnitude_edges.size - 2,
    )
    observed_variance: list[list[float | None]] = []
    noise_variance: list[list[float | None]] = []
    sfr_valid_fraction: list[float | None] = []
    for index in range(magnitude_edges.size - 1):
        inside = bin_index == index
        bin_weight = weight_array[inside]
        if not np.any(inside) or float(np.sum(bin_weight)) <= 0.0:
            observed_variance.append([None, None, None])
            noise_variance.append([None, None, None])
            sfr_valid_fraction.append(None)
            continue
        # Same robust estimators the sampler's deconvolution uses.
        bin_observed: list[float | None] = []
        bin_noise: list[float | None] = []
        for band in range(3):
            low, high = weighted_quantile(
                ratio_array[inside, band], bin_weight, (0.16, 0.84),
            )
            bin_observed.append(float((0.5 * (high - low)) ** 2))
            variance_column = ratio_var_array[inside, band]
            median_variance = float(weighted_quantile(
                variance_column, bin_weight, 0.5,
            ))
            sane = variance_column <= 25.0 * max(median_variance, 1e-300)
            bin_noise.append(float(np.average(
                variance_column[sane], weights=bin_weight[sane],
            )))
        observed_variance.append(bin_observed)
        noise_variance.append(bin_noise)
        sfr_valid_fraction.append(float(
            np.sum(bin_weight[sfr_valid_array[inside]])
            / np.sum(bin_weight)
        ))
    diagnostic_magnitude = 0.5 * (magnitude_edges[:-1] + magnitude_edges[1:])
    mean_colors = sampler.mean_colors(diagnostic_magnitude)

    diagnostics = {
        "catalog_rows": catalog_rows,
        "selected_rows": row_count,
        "selected_galaxy_weight": total_weight,
        "sfr_valid_rows": int(np.sum(sfr_valid_array)),
        "sfr_valid_weight_fraction": float(
            np.sum(weight_array[sfr_valid_array]) / total_weight
        ),
        "resolved_radius_weight_fraction": float(
            np.sum(weight_array[resolved_array > 0.0]) / total_weight
        ),
        "negative_ratio_row_fraction": [
            float(np.mean(ratio_array[:, band] <= 0.0)) for band in range(3)
        ],
        "quenched_weight_fraction": float(
            np.sum(weight_array[sfr_class_array == SFR_CLASS_QUENCHED])
            / total_weight
        ),
        "leaf_count_by_tree": leaf_counts,
        "median_leaf_rows_by_tree": leaf_row_sizes,
        "magnitude_edges": magnitude_edges.tolist(),
        "observed_ratio_variance_by_magnitude": observed_variance,
        "noise_ratio_variance_by_magnitude": noise_variance,
        "sfr_valid_weight_fraction_by_magnitude": sfr_valid_fraction,
        "mean_colors_by_magnitude": {
            "magnitude": diagnostic_magnitude.tolist(),
            "vis_minus_y": mean_colors[:, 0].tolist(),
            "y_minus_j": mean_colors[:, 1].tolist(),
            "j_minus_h": mean_colors[:, 2].tolist(),
        },
    }
    return payload, diagnostics
