#!/usr/bin/env python3
"""Fit a fast, catalog-only TNG VIS number-count prior.

This deliberately does not load or render SKIRT FITS stamps.  It reproduces
the field proposal from:

* ``tng_properties.csv`` for the atlas mass distribution;
* the configured redshift and target-mass priors;
* the generator's coarse VIS magnitude predictor and m=28 pre-render cut;
* the current test+validate source sidecars for a small empirical correction
  between the coarse predictor and rendered truth flux.

The fitted curve is a single smooth (piecewise-linear in log weight) proposal
weight versus predicted VIS magnitude.  It keeps a requested raw TNG draw
budget fixed while fitting the rising 20--28 AB branch measured from the deep
COSMOS2025 extraction.  Euclid MER remains a detection-level comparison, not a
latent-population target.  Outputs are diagnostics only: common-detector
validation is still required before enabling a fitted curve in field generation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "euclid_mpl_cache")
)

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

from euclid_polish.config import Config
from euclid_polish.photometry import electrons_to_ab_mag
from euclid_polish.sky.generation.redshift_model import (
    _z_inverse_cdf_grid,
    physical_pc_to_arcsec,
    predicted_vis_mag,
)

plt.switch_backend("Agg")


DEFAULT_REMOTE_CACHE = (
    "data/_fasrc_cache/n/netscratch/lconnor_lab/Lab/abelotserkovtsev/"
    "EuclidPolish/data/_tng_infographics/tng_properties.csv"
)
DEFAULT_LOCAL_PROPERTIES = "data/_tng_infographics/tng_properties.csv"
DEFAULT_RECORDS = (
    "data/_fasrc_cache/n/netscratch/lconnor_lab/Lab/abelotserkovtsev/"
    "EuclidPolish/data/images/records_v2"
)
DEFAULT_EUCLID = "data/population_comparison/euclid_population.csv"
DEFAULT_EUCLID_META = "data/population_comparison/euclid_population_meta.json"
DEFAULT_OUTPUT_DIR = "data/population_comparison"
DEFAULT_COSMOS_COUNTS = (
    "data/population_comparison/cosmos2025/cosmos2025_number_counts.csv"
)

MAG_BINS = np.arange(20.0, 28.0001, 0.5)
COSMOS_FIT_MIN = 20.0
COSMOS_FIT_MAX = 28.0
WEIGHT_KNOTS = np.asarray(
    (18.0, 21.0, 22.5, 23.5, 24.0, 24.5, 25.0,
     25.5, 26.0, 27.0, 28.0, 29.5, 31.5),
    dtype=np.float64,
)


@dataclass(frozen=True)
class Atlas:
    ids: np.ndarray
    logmass: np.ndarray
    reff_kpc: np.ndarray
    checksum: str


@dataclass(frozen=True)
class TruthCalibration:
    magnitudes: np.ndarray
    residual_noise: np.ndarray
    atlas_correction: dict[str, float]
    global_correction: float
    regression_beta: np.ndarray
    field_count: int
    atlas_ids_seen: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_properties_path(explicit: str | None) -> Path:
    candidates = [
        Path(explicit) if explicit else None,
        Path(DEFAULT_REMOTE_CACHE),
        Path(DEFAULT_LOCAL_PROPERTIES),
    ]
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "No tng_properties.csv found. Pull the FASRC copy or pass --properties."
    )


def read_atlas(path: Path) -> Atlas:
    rows: list[tuple[str, float, float]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                gid = str(row["id"]).strip()
                logmass = math.log10(float(row["mass_stars"]))
                reff = float(row["reff"])
            except (KeyError, TypeError, ValueError):
                continue
            if gid and np.isfinite(logmass) and np.isfinite(reff) and reff > 0:
                rows.append((gid, logmass, reff))
    if not rows:
        raise ValueError(f"No usable atlas rows in {path}")
    rows.sort(key=lambda item: item[1])
    return Atlas(
        ids=np.asarray([row[0] for row in rows], dtype=str),
        logmass=np.asarray([row[1] for row in rows], dtype=np.float64),
        reff_kpc=np.asarray([row[2] for row in rows], dtype=np.float64),
        checksum=_sha256(path),
    )


def _source_paths(records_dir: Path) -> list[Path]:
    paths = [
        records_dir / "sources_test.csv",
        records_dir / "sources_validate.csv",
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing current source sidecar(s): {missing}")
    return paths


def read_truth_calibration(records_dir: Path) -> TruthCalibration:
    observations: list[tuple[str, float, float, float]] = []
    magnitudes: list[float] = []
    residual_by_atlas: dict[str, list[float]] = defaultdict(list)
    field_count = 0

    for path in _source_paths(records_dir):
        fields: set[int] = set()
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                fields.add(int(row["field_index"]))
                if (
                    row.get("render") != "tng"
                    or not row.get("flux_vis_e")
                    or not row.get("logmass")
                    or not row.get("z")
                ):
                    continue
                try:
                    gid = str(row.get("subhalo_id", "")).strip()
                    flux_vis = float(row["flux_vis_e"])
                    z = float(row["z"])
                    logmass = float(row["logmass"])
                except (TypeError, ValueError):
                    continue
                if (
                    not np.isfinite(flux_vis)
                    or flux_vis <= 0.0
                    or not np.isfinite(z)
                    or not np.isfinite(logmass)
                ):
                    continue
                actual = float(electrons_to_ab_mag(
                    flux_vis, Config.BAND_VIS
                ))
                predicted = float(predicted_vis_mag(logmass, z))
                if not np.isfinite(actual) or not np.isfinite(predicted):
                    continue
                residual = actual - predicted
                magnitudes.append(actual)
                residual_by_atlas[gid].append(residual)
                observations.append((gid, residual, z, logmass))
        field_count += len(fields)

    if not observations or field_count <= 0:
        raise ValueError(f"No usable TNG truth rows under {records_dir}")

    all_residuals = np.asarray([row[1] for row in observations])
    global_correction = float(np.median(all_residuals))
    shrinkage = 3.0
    atlas_correction = {
        gid: float(
            (len(values) * np.median(values)
             + shrinkage * global_correction)
            / (len(values) + shrinkage)
        )
        for gid, values in residual_by_atlas.items()
    }

    design = np.asarray([
        (1.0, z - 0.9, (z - 0.9) ** 2, logmass - 9.1)
        for _gid, _residual, z, logmass in observations
    ])
    response = np.asarray([
        residual - atlas_correction.get(gid, global_correction)
        for gid, residual, _z, _logmass in observations
    ])
    beta = np.linalg.lstsq(design, response, rcond=None)[0]
    residual_noise = response - design @ beta

    return TruthCalibration(
        magnitudes=np.asarray(magnitudes),
        residual_noise=np.asarray(residual_noise),
        atlas_correction=atlas_correction,
        global_correction=global_correction,
        regression_beta=np.asarray(beta),
        field_count=field_count,
        atlas_ids_seen=len(residual_by_atlas),
    )


def read_euclid_magnitudes(path: Path) -> np.ndarray:
    values: list[float] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("type", "")).lower() == "star":
                continue
            try:
                value = float(row["mag_vis"])
            except (KeyError, TypeError, ValueError):
                continue
            if np.isfinite(value):
                values.append(value)
    if not values:
        raise ValueError(f"No usable nonstellar Euclid magnitudes in {path}")
    return np.asarray(values)


def read_binned_cosmos_counts(
    path: Path,
    *,
    selection: str = "clean",
    bins: np.ndarray = MAG_BINS,
) -> np.ndarray:
    """Integrate the extractor's differential counts into ``bins``.

    The COSMOS extraction uses 0.25-mag bins and records densities per unit
    magnitude.  This routine integrates their overlap with the TNG fitter's
    0.5-mag bins, so no assumption about matching bin edges is hidden.
    """
    density_key = f"{selection}_density_per_mag_arcmin2"
    result = np.zeros(len(bins) - 1, dtype=np.float64)
    covered = np.zeros(len(bins) - 1, dtype=np.float64)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or density_key not in reader.fieldnames:
            raise ValueError(
                f"{path} does not contain COSMOS selection {selection!r}"
            )
        for row in reader:
            try:
                source_lo = float(row["mag_lo"])
                source_hi = float(row["mag_hi"])
                density_per_mag = float(row[density_key])
            except (KeyError, TypeError, ValueError):
                continue
            if (
                not np.isfinite(density_per_mag)
                or source_hi <= source_lo
            ):
                continue
            for index in range(len(result)):
                overlap = max(
                    0.0,
                    min(source_hi, bins[index + 1])
                    - max(source_lo, bins[index]),
                )
                result[index] += density_per_mag * overlap
                covered[index] += overlap
    required = (bins[:-1] >= COSMOS_FIT_MIN) & (bins[1:] <= COSMOS_FIT_MAX)
    if np.any(covered[required] < np.diff(bins)[required] - 1.0e-9):
        raise ValueError(
            f"{path} does not fully cover {COSMOS_FIT_MIN:g}–"
            f"{COSMOS_FIT_MAX:g} mag"
        )
    return result


def _mass_inverse_cdf() -> tuple[np.ndarray, np.ndarray]:
    logmass = np.linspace(
        Config.TNG_MF_LOGM_MIN, Config.TNG_MF_LOGM_MAX, 2048
    )
    ratio = 10.0 ** (logmass - Config.TNG_MF_LOGM_STAR)
    pdf = ratio ** (Config.TNG_MF_ALPHA + 1.0) * np.exp(-ratio)
    cdf = np.cumsum(pdf)
    cdf -= cdf[0]
    cdf /= cdf[-1]
    return cdf, logmass


def draw_catalog_proposals(
    atlas: Atlas,
    calibration: TruthCalibration,
    *,
    sample_count: int,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)

    z_cdf, z_grid = _z_inverse_cdf_grid(
        Config.TNG_Z_FORM,
        Config.TNG_Z0,
        Config.TNG_Z_PHI_SCALE,
        Config.TNG_Z_MIN,
        Config.TNG_Z_MAX,
    )
    z = np.interp(rng.random(sample_count), z_cdf, z_grid)

    mass_cdf, mass_grid = _mass_inverse_cdf()
    target_logmass = np.interp(
        rng.random(sample_count), mass_cdf, mass_grid
    )

    lower = np.searchsorted(atlas.logmass, target_logmass, side="left")
    upper = np.searchsorted(
        atlas.logmass,
        target_logmass + np.log10(Config.TNG_MASS_WINDOW),
        side="right",
    )
    lower = np.minimum(lower, len(atlas.logmass) - 1)
    upper = np.where(upper > lower, upper, len(atlas.logmass))
    span = np.maximum(upper - lower, 1)
    atlas_index = lower + (rng.random(sample_count) * span).astype(np.int64)
    atlas_logmass = atlas.logmass[atlas_index]
    mass_scale = np.minimum(
        1.0, 10.0 ** (target_logmass - atlas_logmass)
    )

    z_lookup = np.linspace(
        Config.TNG_Z_MIN, Config.TNG_Z_MAX, 4096
    )
    faint_mass = float(Config.TNG_MF_LOGM_MIN)
    mag_lookup = np.asarray([
        predicted_vis_mag(faint_mass, value) for value in z_lookup
    ])
    coarse_mag = (
        np.interp(z, z_lookup, mag_lookup)
        - 2.5 * (target_logmass - faint_mass)
    )

    correction_lookup = np.asarray([
        calibration.atlas_correction.get(
            gid, calibration.global_correction
        )
        for gid in atlas.ids
    ])
    beta = calibration.regression_beta
    predicted_mean_mag = (
        coarse_mag
        + correction_lookup[atlas_index]
        + beta[0]
        + beta[1] * (z - 0.9)
        + beta[2] * (z - 0.9) ** 2
        + beta[3] * (target_logmass - 9.1)
    )
    draft_mag = (
        predicted_mean_mag
        + rng.choice(
            calibration.residual_noise,
            size=sample_count,
            replace=True,
        )
    )
    keep = (
        coarse_mag <= Config.TNG_FAINT_SKIP_MAG_VIS
        if Config.TNG_FAINT_SKIP_MAG_VIS > 0
        else np.ones(sample_count, dtype=bool)
    )

    # The CSV half-mass radius is not identical to the SKIRT light radius, but
    # this gives a fast property-level size draft with the same mass/redshift
    # scaling as field generation.
    kpc_to_arcsec_lookup = np.asarray([
        physical_pc_to_arcsec(1000.0, value) for value in z_lookup
    ])
    angular_scale_arcsec = (
        atlas.reff_kpc[atlas_index]
        * np.interp(z, z_lookup, kpc_to_arcsec_lookup)
    )
    compactness = Config.TNG_COMPACT_C0 * (1.0 + z) ** Config.TNG_COMPACT_BETA
    draft_re_arcsec = (
        angular_scale_arcsec
        * mass_scale ** Config.TNG_MASS_SIZE_ALPHA
        / compactness
    )

    return {
        "z": z,
        "target_logmass": target_logmass,
        "atlas_index": atlas_index,
        "mass_scale": mass_scale,
        "coarse_mag": coarse_mag,
        "predicted_mean_mag": predicted_mean_mag,
        "draft_mag": draft_mag,
        "draft_re_arcsec": draft_re_arcsec,
        "keep": keep,
    }

def density_histogram(
    values: np.ndarray,
    *,
    bins: np.ndarray,
    area_arcmin2: float,
) -> np.ndarray:
    return np.histogram(values, bins=bins)[0].astype(np.float64) / area_arcmin2


def weighted_draft_density(
    draft_mag: np.ndarray,
    keep: np.ndarray,
    predicted_mean_mag: np.ndarray,
    log_weights: np.ndarray,
    *,
    knots: np.ndarray = WEIGHT_KNOTS,
    bins: np.ndarray = MAG_BINS,
    raw_density: float = Config.TNG_GAL_DENSITY_ARCMIN2,
) -> tuple[np.ndarray, np.ndarray]:
    log_weight = np.interp(
        predicted_mean_mag,
        knots,
        log_weights,
        left=log_weights[0],
        right=log_weights[-1],
    )
    weights = np.exp(np.clip(log_weight, -5.0, 5.0))
    indices = np.digitize(draft_mag, bins) - 1
    valid = (indices >= 0) & (indices < len(bins) - 1)
    numerator = np.bincount(
        indices[valid],
        weights=(weights * keep)[valid],
        minlength=len(bins) - 1,
    )
    density = raw_density * numerator / weights.sum()
    return density, weights


def fit_log_weights(
    draft_mag: np.ndarray,
    keep: np.ndarray,
    predicted_mean_mag: np.ndarray,
    target_density: np.ndarray,
    *,
    knots: np.ndarray = WEIGHT_KNOTS,
    bins: np.ndarray = MAG_BINS,
    fit_min: float = COSMOS_FIT_MIN,
    fit_max: float = COSMOS_FIT_MAX,
    raw_density: float = Config.TNG_GAL_DENSITY_ARCMIN2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centers = 0.5 * (bins[:-1] + bins[1:])
    free = np.arange(3, len(knots))

    def expand(parameters: np.ndarray) -> np.ndarray:
        full = np.zeros(len(knots), dtype=np.float64)
        full[free] = parameters
        return full

    def residuals(parameters: np.ndarray) -> np.ndarray:
        full = expand(parameters)
        density, weights = weighted_draft_density(
            draft_mag,
            keep,
            predicted_mean_mag,
            full,
            knots=knots,
            bins=bins,
            raw_density=raw_density,
        )
        # COSMOS is a deep latent-population target, so use its full rising
        # branch rather than importing the Euclid detection turnover.
        importance = np.where(
            (centers >= fit_min) & (centers < fit_max),
            4.0,
            0.5,
        )
        data = np.log((density + 0.1) / (target_density + 0.1))
        smoothness = 0.12 * np.diff(full, n=2)
        normalization = np.asarray([15.0 * np.log(weights.mean())])
        return np.concatenate((importance * data, smoothness, normalization))

    result = least_squares(
        residuals,
        np.zeros(len(free), dtype=np.float64),
        bounds=(-2.5, 2.5),
        max_nfev=200,
    )
    full = expand(result.x)
    density, weights = weighted_draft_density(
        draft_mag,
        keep,
        predicted_mean_mag,
        full,
        knots=knots,
        bins=bins,
        raw_density=raw_density,
    )
    return full, density, weights


def _field_area_from_comparison(default_fields: int) -> tuple[float, int]:
    path = Path(DEFAULT_OUTPUT_DIR) / "comparison.json"
    try:
        payload = json.loads(path.read_text())
        per_field = float(payload["geometry"]["field_area_arcmin2"])
        fields = int(payload["population"]["synthetic_field_count"])
        if per_field > 0 and fields > 0:
            return per_field * fields, fields
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        pass
    per_field = (256.0 * Config.VIS_PIXEL_SCALE_ARCSEC / 60.0) ** 2
    return per_field * default_fields, default_fields


def _euclid_area(path: Path) -> float:
    try:
        payload = json.loads(path.read_text())
        area = float(payload["area_arcmin2"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        area = 0.0
    if area <= 0:
        raise ValueError(f"No positive area_arcmin2 in {path}")
    return area


def _plot(
    path: Path,
    centers: np.ndarray,
    synthetic_density: np.ndarray,
    euclid_density: np.ndarray,
    cosmos_density: np.ndarray,
    baseline_density: np.ndarray,
    fitted_density: np.ndarray,
    target_density: np.ndarray,
) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    ax.step(
        centers, euclid_density, where="mid", linewidth=2.4,
        color="#1479ff", label="Euclid MER",
    )
    ax.step(
        centers, synthetic_density, where="mid", linewidth=2.2,
        color="#ef6c00", linestyle="--", label="current rendered truth",
    )
    ax.step(
        centers, cosmos_density, where="mid", linewidth=2.8,
        color="#111111", linestyle="-.", label="COSMOS2025 raw population",
    )
    ax.plot(
        centers, baseline_density, color="#6f42c1", linewidth=1.8,
        marker="o", markersize=4, label="catalog-only baseline draft",
    )
    ax.plot(
        centers, fitted_density, color="#008f5d", linewidth=2.6,
        marker="s", markersize=4.5, label="catalog-only fitted draft",
    )
    ax.plot(
        centers, target_density, color="#222222", linewidth=1.4,
        linestyle=":", alpha=0.65, label="fit target",
    )
    ax.axvspan(
        COSMOS_FIT_MIN, COSMOS_FIT_MAX, color="#008f5d", alpha=0.08
    )
    ax.set(
        xlabel="VIS magnitude (AB)",
        ylabel="objects / arcmin² / 0.5 mag",
        title="TNG catalog-only VIS number-count fit",
    )
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run(args: argparse.Namespace) -> dict[str, Any]:
    properties_path = resolve_properties_path(args.properties)
    records_dir = Path(args.records_dir)
    euclid_path = Path(args.euclid)
    euclid_meta_path = Path(args.euclid_meta)
    cosmos_counts_path = Path(args.cosmos_counts)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    atlas = read_atlas(properties_path)
    calibration = read_truth_calibration(records_dir)
    euclid_magnitudes = read_euclid_magnitudes(euclid_path)
    cosmos_density = read_binned_cosmos_counts(
        cosmos_counts_path, selection=args.cosmos_selection
    )
    proposals = draw_catalog_proposals(
        atlas,
        calibration,
        sample_count=args.samples,
        seed=args.seed,
    )

    synthetic_area, synthetic_fields = _field_area_from_comparison(
        calibration.field_count
    )
    euclid_area = _euclid_area(euclid_meta_path)
    synthetic_density = density_histogram(
        calibration.magnitudes,
        bins=MAG_BINS,
        area_arcmin2=synthetic_area,
    )
    euclid_density = density_histogram(
        euclid_magnitudes,
        bins=MAG_BINS,
        area_arcmin2=euclid_area,
    )
    raw_density = float(args.raw_density)
    baseline_density = (
        np.histogram(
            proposals["draft_mag"][proposals["keep"]],
            bins=MAG_BINS,
        )[0].astype(np.float64)
        / args.samples
        * raw_density
    )

    centers = 0.5 * (MAG_BINS[:-1] + MAG_BINS[1:])
    target_density = synthetic_density.copy()
    fit_region = (
        (centers >= COSMOS_FIT_MIN) & (centers < COSMOS_FIT_MAX)
    )
    # The deep COSMOS population—not the selected Euclid MER detections—is the
    # latent truth target throughout its empirically rising branch.
    target_density[fit_region] = cosmos_density[fit_region]

    log_weights, fitted_density, proposal_weights = fit_log_weights(
        proposals["draft_mag"],
        proposals["keep"],
        proposals["predicted_mean_mag"],
        target_density,
        raw_density=raw_density,
    )
    fitted_retained_density = (
        raw_density
        * np.sum(proposal_weights * proposals["keep"])
        / np.sum(proposal_weights)
    )
    baseline_retained_density = raw_density * float(
        np.mean(proposals["keep"])
    )

    rows = []
    for index, center in enumerate(centers):
        rows.append({
            "mag_lo": float(MAG_BINS[index]),
            "mag_hi": float(MAG_BINS[index + 1]),
            "mag_center": float(center),
            "synthetic_truth_density": float(synthetic_density[index]),
            "euclid_mer_density": float(euclid_density[index]),
            "cosmos2025_population_density": float(cosmos_density[index]),
            "catalog_baseline_density": float(baseline_density[index]),
            "fit_target_density": float(target_density[index]),
            "catalog_fitted_density": float(fitted_density[index]),
        })

    csv_path = output_dir / "tng_catalog_vis_fit.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    plot_path = output_dir / "tng_catalog_vis_fit.png"
    _plot(
        plot_path,
        centers,
        synthetic_density,
        euclid_density,
        cosmos_density,
        baseline_density,
        fitted_density,
        target_density,
    )

    payload: dict[str, Any] = {
        "version": 1,
        "method": "catalog-only importance fit; no SKIRT or field rendering",
        "selection_caveat": (
            "COSMOS is a deep NIRCam-selected raw population using HST F814W "
            "as the VIS proxy and has no completeness correction. Euclid remains "
            "a detection-level comparison only. Validate the fitted generator "
            "through common detector photometry before production generation."
        ),
        "inputs": {
            "properties_csv": str(properties_path),
            "properties_sha256": atlas.checksum,
            "atlas_rows": int(len(atlas.ids)),
            "truth_rows": int(len(calibration.magnitudes)),
            "truth_fields": int(synthetic_fields),
            "euclid_rows": int(len(euclid_magnitudes)),
            "euclid_area_arcmin2": float(euclid_area),
            "cosmos_counts_csv": str(cosmos_counts_path),
            "cosmos_counts_sha256": _sha256(cosmos_counts_path),
            "cosmos_selection": str(args.cosmos_selection),
        },
        "proposal": {
            "samples": int(args.samples),
            "seed": int(args.seed),
            "raw_density_arcmin2": raw_density,
            "mf_alpha": float(Config.TNG_MF_ALPHA),
            "mf_logm_star": float(Config.TNG_MF_LOGM_STAR),
            "z_form": str(Config.TNG_Z_FORM),
            "faint_skip_mag_vis": float(Config.TNG_FAINT_SKIP_MAG_VIS),
            "baseline_retained_density_arcmin2": baseline_retained_density,
            "fitted_retained_density_arcmin2": float(fitted_retained_density),
        },
        "photometric_calibration": {
            "atlas_ids_seen": int(calibration.atlas_ids_seen),
            "global_mag_correction": calibration.global_correction,
            "residual_scatter_mag": float(
                np.std(calibration.residual_noise)
            ),
            "regression_beta": calibration.regression_beta.tolist(),
        },
        "fit": {
            "region_mag": [COSMOS_FIT_MIN, COSMOS_FIT_MAX],
            "cosmos_target_density_arcmin2": float(
                np.sum(cosmos_density[fit_region])
            ),
            "raw_budget_slack_after_cosmos_fit_arcmin2": float(
                raw_density - np.sum(cosmos_density[fit_region])
            ),
            "knots_predicted_vis_mag": WEIGHT_KNOTS.tolist(),
            "proposal_weights": np.exp(log_weights).tolist(),
            "normalization_mean_weight": float(np.mean(proposal_weights)),
            "bins": rows,
        },
        "outputs": {
            "csv": str(csv_path),
            "plot": str(plot_path),
        },
    }
    json_path = output_dir / "tng_catalog_vis_fit.json"
    payload["outputs"]["json"] = str(json_path)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--properties", default=None)
    parser.add_argument("--records-dir", default=DEFAULT_RECORDS)
    parser.add_argument("--euclid", default=DEFAULT_EUCLID)
    parser.add_argument("--euclid-meta", default=DEFAULT_EUCLID_META)
    parser.add_argument("--cosmos-counts", default=DEFAULT_COSMOS_COUNTS)
    parser.add_argument(
        "--cosmos-selection",
        choices=("population", "clean", "isolated", "generator_ready"),
        default="clean",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--samples", type=int, default=1_500_000)
    parser.add_argument("--seed", type=int, default=73032)
    parser.add_argument(
        "--raw-density",
        type=float,
        default=Config.TNG_GAL_DENSITY_ARCMIN2,
        help="Raw TNG proposals per arcmin2 for the catalog-only fit.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.samples < 10_000:
        raise SystemExit("--samples must be at least 10000")
    if args.raw_density <= 0:
        raise SystemExit("--raw-density must be positive")
    payload = run(args)
    print(json.dumps({
        "properties": payload["inputs"]["properties_csv"],
        "atlas_rows": payload["inputs"]["atlas_rows"],
        "samples": payload["proposal"]["samples"],
        "baseline_retained_density_arcmin2": (
            payload["proposal"]["baseline_retained_density_arcmin2"]
        ),
        "fitted_retained_density_arcmin2": (
            payload["proposal"]["fitted_retained_density_arcmin2"]
        ),
        "weights": payload["fit"]["proposal_weights"],
        "outputs": payload["outputs"],
    }, indent=2))


if __name__ == "__main__":
    main()
