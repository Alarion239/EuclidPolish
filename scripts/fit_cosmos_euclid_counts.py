#!/usr/bin/env python3
"""Fit Euclid VIS detections as an observation of the COSMOS population.

COSMOS2025 provides the latent galaxy count shape.  The fit is intentionally
restricted to an observation layer:

* a global population normalization;
* a COSMOS F814W -> Euclid VIS magnitude offset and Gaussian scatter;
* a logistic Euclid detection completeness curve.

This prevents the Euclid catalog turnover from being interpreted as a physical
decline in the faint galaxy population.  The output plot keeps the latent and
detected curves visually distinct and includes the current rendered synthetic
truth when its catalog-only fit CSV is available.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import tempfile
from dataclasses import asdict, dataclass
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
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares

plt.switch_backend("Agg")


DEFAULT_COSMOS_COUNTS = (
    "data/population_comparison/cosmos2025/cosmos2025_number_counts.csv"
)
DEFAULT_EUCLID = "data/population_comparison/euclid_population.csv"
DEFAULT_EUCLID_META = "data/population_comparison/euclid_population_meta.json"
DEFAULT_SYNTHETIC = "data/population_comparison/tng_catalog_vis_fit.csv"
DEFAULT_OUTPUT_DIR = "data/population_comparison/cosmos2025"

DISPLAY_BINS = np.arange(20.0, 28.0001, 0.5)
MODEL_STEP = 0.01
MODEL_GRID = np.arange(17.0 + MODEL_STEP / 2, 32.0, MODEL_STEP)


@dataclass(frozen=True)
class ObservationFit:
    population_scale: float
    vis_minus_f814w_mag: float
    magnitude_slope: float
    scatter_mag: float
    completeness_m50: float
    completeness_width_mag: float
    poisson_deviance: float
    dof: int


def read_cosmos_density(
    path: Path,
    *,
    selection: str = "clean",
) -> tuple[np.ndarray, np.ndarray]:
    """Read the extractor's differential counts in objects/arcmin²/mag."""
    key = f"{selection}_density_per_mag_arcmin2"
    centers: list[float] = []
    density: list[float] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or key not in reader.fieldnames:
            raise ValueError(f"{path} has no {key!r} column")
        for row in reader:
            try:
                center = float(row["mag_center"])
                value = float(row[key])
            except (KeyError, TypeError, ValueError):
                continue
            if np.isfinite(center) and np.isfinite(value) and value >= 0:
                centers.append(center)
                density.append(value)
    if len(centers) < 3:
        raise ValueError(f"No usable COSMOS number counts in {path}")
    order = np.argsort(centers)
    return (
        np.asarray(centers, dtype=np.float64)[order],
        np.asarray(density, dtype=np.float64)[order],
    )


def read_euclid_magnitudes(
    path: Path,
    *,
    maximum_spurious_probability: float = 0.5,
) -> np.ndarray:
    """Read clean non-star Euclid detections without calling them galaxies."""
    values: list[float] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("type", "")).strip().lower() == "star":
                continue
            raw_spurious = str(row.get("spurious_prob", "")).strip()
            try:
                spurious = float(raw_spurious) if raw_spurious else 0.0
                magnitude = float(row["mag_vis"])
            except (KeyError, TypeError, ValueError):
                continue
            if (
                np.isfinite(magnitude)
                and np.isfinite(spurious)
                and spurious <= maximum_spurious_probability
            ):
                values.append(magnitude)
    if not values:
        raise ValueError(f"No usable non-star Euclid magnitudes in {path}")
    return np.asarray(values, dtype=np.float64)


def _integrate_grid(
    density_per_mag: np.ndarray,
    *,
    bins: np.ndarray = DISPLAY_BINS,
    grid: np.ndarray = MODEL_GRID,
) -> np.ndarray:
    indices = np.digitize(grid, bins) - 1
    valid = (indices >= 0) & (indices < len(bins) - 1)
    return np.bincount(
        indices[valid],
        weights=density_per_mag[valid] * MODEL_STEP,
        minlength=len(bins) - 1,
    ).astype(np.float64)


def completeness_curve(
    magnitudes: np.ndarray,
    *,
    m50: float,
    width: float,
) -> np.ndarray:
    argument = np.clip((magnitudes - m50) / width, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(argument))


def observation_model(
    intrinsic_density_per_mag: np.ndarray,
    *,
    population_scale: float,
    magnitude_offset: float,
    magnitude_slope: float,
    scatter_mag: float,
    completeness_m50: float,
    completeness_width_mag: float,
    bins: np.ndarray = DISPLAY_BINS,
    grid: np.ndarray = MODEL_GRID,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project latent COSMOS counts into the Euclid detected magnitude bins."""
    pivot = 24.0
    source_magnitude = pivot + (
        grid - pivot - magnitude_offset
    ) / magnitude_slope
    shifted = np.interp(
        source_magnitude,
        grid,
        intrinsic_density_per_mag,
        left=0.0,
        right=0.0,
    ) / magnitude_slope
    blurred = gaussian_filter1d(
        shifted,
        sigma=scatter_mag / MODEL_STEP,
        mode="constant",
        cval=0.0,
    )
    transferred = population_scale * blurred
    completeness = completeness_curve(
        grid,
        m50=completeness_m50,
        width=completeness_width_mag,
    )
    latent_binned = _integrate_grid(transferred, bins=bins, grid=grid)
    detected_binned = _integrate_grid(
        transferred * completeness,
        bins=bins,
        grid=grid,
    )
    return latent_binned, detected_binned, completeness


def _signed_poisson_residual(observed: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    mu = np.clip(predicted, 1e-12, None)
    term = mu - observed
    positive = observed > 0
    term[positive] += observed[positive] * np.log(observed[positive] / mu[positive])
    deviance = np.maximum(2.0 * term, 0.0)
    return np.sign(mu - observed) * np.sqrt(deviance)


def fit_observation_layer(
    cosmos_density_per_mag: np.ndarray,
    euclid_counts: np.ndarray,
    euclid_area_arcmin2: float,
    *,
    fit_population_scale: bool = False,
    bins: np.ndarray = DISPLAY_BINS,
    grid: np.ndarray = MODEL_GRID,
) -> tuple[ObservationFit, np.ndarray, np.ndarray, np.ndarray]:
    """Fit an affine photometric transfer and logistic selection curve.

    The COSMOS normalization is fixed by default. One 36 arcmin² Euclid cone
    cannot separate a global density change from cosmic variance and the
    F814W-to-VIS transfer. Freeing it is therefore only a local-field
    sensitivity calculation.
    """
    if euclid_area_arcmin2 <= 0:
        raise ValueError("Euclid area must be positive")

    def unpack(
        parameters: np.ndarray,
    ) -> tuple[float, float, float, float, float, float]:
        index = 0
        if fit_population_scale:
            scale = float(np.exp(parameters[index]))
            index += 1
        else:
            scale = 1.0
        return (
            scale,
            float(parameters[index]),
            float(np.exp(parameters[index + 1])),
            float(np.exp(parameters[index + 2])),
            float(parameters[index + 3]),
            float(np.exp(parameters[index + 4])),
        )

    def residuals(parameters: np.ndarray) -> np.ndarray:
        scale, offset, slope, scatter, m50, width = unpack(parameters)
        _latent, predicted_density, _completeness = observation_model(
            cosmos_density_per_mag,
            population_scale=scale,
            magnitude_offset=offset,
            magnitude_slope=slope,
            scatter_mag=scatter,
            completeness_m50=m50,
            completeness_width_mag=width,
            bins=bins,
            grid=grid,
        )
        predicted_counts = predicted_density * euclid_area_arcmin2
        data = _signed_poisson_residual(euclid_counts, predicted_counts)
        index = 1 if fit_population_scale else 0
        priors = [
            offset / 0.60,
            parameters[index + 1] / 0.20,
            (parameters[index + 2] - math.log(0.15)) / 1.00,
            (m50 - 25.2) / 1.50,
            (parameters[index + 4] - math.log(0.35)) / 1.00,
        ]
        if fit_population_scale:
            priors.insert(0, parameters[0] / 0.30)
        return np.concatenate((data, np.asarray(priors)))

    core_initial = [0.0, 0.0, math.log(0.15), 25.2, math.log(0.35)]
    core_lower = [-1.0, math.log(0.60), math.log(0.02), 23.5, math.log(0.04)]
    core_upper = [1.0, math.log(1.40), math.log(1.00), 28.0, math.log(2.00)]
    if fit_population_scale:
        initial = np.asarray([0.0, *core_initial])
        lower = np.asarray([math.log(0.30), *core_lower])
        upper = np.asarray([math.log(3.00), *core_upper])
    else:
        initial = np.asarray(core_initial)
        lower = np.asarray(core_lower)
        upper = np.asarray(core_upper)
    result = least_squares(
        residuals,
        initial,
        bounds=(lower, upper),
        max_nfev=1000,
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
    )
    scale, offset, slope, scatter, m50, width = unpack(result.x)
    latent, detected, completeness = observation_model(
        cosmos_density_per_mag,
        population_scale=scale,
        magnitude_offset=offset,
        magnitude_slope=slope,
        scatter_mag=scatter,
        completeness_m50=m50,
        completeness_width_mag=width,
        bins=bins,
        grid=grid,
    )
    predicted_counts = detected * euclid_area_arcmin2
    poisson_residual = _signed_poisson_residual(euclid_counts, predicted_counts)
    fit = ObservationFit(
        population_scale=scale,
        vis_minus_f814w_mag=offset,
        magnitude_slope=slope,
        scatter_mag=scatter,
        completeness_m50=m50,
        completeness_width_mag=width,
        poisson_deviance=float(np.sum(poisson_residual**2)),
        dof=max(1, len(euclid_counts) - (6 if fit_population_scale else 5)),
    )
    return fit, latent, detected, completeness


def _read_area(path: Path) -> float:
    payload = json.loads(path.read_text())
    area = float(payload["area_arcmin2"])
    if area <= 0:
        raise ValueError(f"No positive area_arcmin2 in {path}")
    return area


def _read_synthetic_density(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    values: list[float] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                values.append(float(row["synthetic_truth_density"]))
            except (KeyError, TypeError, ValueError):
                return None
    result = np.asarray(values, dtype=np.float64)
    return result if len(result) == len(DISPLAY_BINS) - 1 else None


def _plot(
    path: Path,
    *,
    centers: np.ndarray,
    cosmos_density: np.ndarray,
    fitted_latent_density: np.ndarray,
    euclid_density: np.ndarray,
    euclid_error: np.ndarray,
    predicted_detected_density: np.ndarray,
    local_latent_density: np.ndarray,
    local_predicted_detected_density: np.ndarray,
    synthetic_density: np.ndarray | None,
    completeness: np.ndarray,
    fit: ObservationFit,
) -> None:
    fig, (ax, lower) = plt.subplots(
        2,
        1,
        figsize=(9.6, 7.4),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": (3.2, 1.25), "hspace": 0.06},
    )
    ax.step(
        centers,
        cosmos_density,
        where="mid",
        color="#242424",
        linewidth=2.5,
        linestyle=(0, (7, 3)),
        label="COSMOS latent population",
    )
    ax.step(
        centers,
        fitted_latent_density,
        where="mid",
        color="#008c68",
        linewidth=2.8,
        label="fitted latent population",
    )
    ax.step(
        centers,
        local_latent_density,
        where="mid",
        color="#008c68",
        linewidth=1.8,
        linestyle=(0, (2, 2)),
        label="latent + local normalization sensitivity",
    )
    ax.errorbar(
        centers,
        euclid_density,
        yerr=euclid_error,
        color="#1267d6",
        marker="o",
        markersize=5.5,
        markerfacecolor="white",
        markeredgewidth=1.8,
        linestyle="none",
        capsize=2.5,
        label="Euclid non-star detections",
        zorder=5,
    )
    ax.plot(
        centers,
        predicted_detected_density,
        color="#cf3d2e",
        linewidth=2.6,
        marker="D",
        markersize=4.5,
        label="COSMOS through fitted Euclid selection",
    )
    ax.plot(
        centers,
        local_predicted_detected_density,
        color="#e68a00",
        linewidth=1.9,
        linestyle="--",
        marker="^",
        markersize=4,
        label="local-normalization detection fit",
    )
    if synthetic_density is not None:
        ax.step(
            centers,
            synthetic_density,
            where="mid",
            color="#8d48b5",
            linewidth=2.0,
            linestyle=":",
            label="current synthetic truth",
        )
    ax.set_ylabel("objects / arcmin² / 0.5 mag")
    ax.set_title("COSMOS latent counts fitted to Euclid VIS detections")
    ax.grid(alpha=0.20)
    ax.legend(frameon=False, ncol=2, fontsize=9)

    display_completeness = np.interp(centers, MODEL_GRID, completeness)
    lower.plot(
        centers,
        display_completeness,
        color="#cf3d2e",
        linewidth=2.5,
        marker="D",
        markersize=4,
        label="fitted completeness",
    )
    lower.axhline(0.5, color="#6b7280", linewidth=1.0, linestyle="--")
    lower.axvline(
        fit.completeness_m50,
        color="#cf3d2e",
        linewidth=1.1,
        linestyle=":",
    )
    lower.set(
        xlabel="catalog VIS magnitude (AB)",
        ylabel="detection\nprobability",
        ylim=(-0.04, 1.04),
        xlim=(DISPLAY_BINS[0], DISPLAY_BINS[-1]),
    )
    lower.grid(alpha=0.20)
    reliable_vis_max = (
        24.0
        + fit.magnitude_slope * (27.5 - 24.0)
        + fit.vis_minus_f814w_mag
    )
    for axis in (ax, lower):
        axis.axvspan(
            reliable_vis_max,
            DISPLAY_BINS[-1],
            color="#8b93a1",
            alpha=0.10,
            linewidth=0,
        )
    ax.text(
        reliable_vis_max + 0.04,
        0.97,
        "COSMOS turnover-sensitive",
        transform=ax.get_xaxis_transform(),
        color="#687080",
        fontsize=8.5,
        ha="left",
        va="top",
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run(args: argparse.Namespace) -> dict[str, Any]:
    cosmos_path = Path(args.cosmos_counts)
    euclid_path = Path(args.euclid)
    euclid_meta_path = Path(args.euclid_meta)
    synthetic_path = Path(args.synthetic)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cosmos_centers, cosmos_native_density = read_cosmos_density(
        cosmos_path,
        selection=args.cosmos_selection,
    )
    cosmos_model_density = np.interp(
        MODEL_GRID,
        cosmos_centers,
        cosmos_native_density,
        left=0.0,
        right=0.0,
    )
    euclid_magnitudes = read_euclid_magnitudes(
        euclid_path,
        maximum_spurious_probability=args.maximum_spurious_probability,
    )
    euclid_area = _read_area(euclid_meta_path)
    euclid_meta = json.loads(euclid_meta_path.read_text())
    euclid_counts = np.histogram(euclid_magnitudes, bins=DISPLAY_BINS)[0]
    euclid_density = euclid_counts / euclid_area
    euclid_error = np.sqrt(euclid_counts) / euclid_area

    fit, fitted_latent, predicted_detected, completeness = fit_observation_layer(
        cosmos_model_density,
        euclid_counts,
        euclid_area,
    )
    local_fit, local_latent, local_predicted_detected, _ = fit_observation_layer(
        cosmos_model_density,
        euclid_counts,
        euclid_area,
        fit_population_scale=True,
    )
    cosmos_binned = _integrate_grid(cosmos_model_density)
    synthetic_density = _read_synthetic_density(synthetic_path)
    centers = 0.5 * (DISPLAY_BINS[:-1] + DISPLAY_BINS[1:])

    rows: list[dict[str, float]] = []
    display_completeness = np.interp(centers, MODEL_GRID, completeness)
    for index, center in enumerate(centers):
        rows.append({
            "mag_lo": float(DISPLAY_BINS[index]),
            "mag_hi": float(DISPLAY_BINS[index + 1]),
            "mag_center": float(center),
            "cosmos_latent_density": float(cosmos_binned[index]),
            "fitted_latent_density": float(fitted_latent[index]),
            "euclid_detected_density": float(euclid_density[index]),
            "euclid_poisson_error": float(euclid_error[index]),
            "predicted_detected_density": float(predicted_detected[index]),
            "local_fitted_latent_density": float(local_latent[index]),
            "local_predicted_detected_density": float(
                local_predicted_detected[index]
            ),
            "fitted_completeness": float(display_completeness[index]),
            "synthetic_truth_density": (
                float(synthetic_density[index])
                if synthetic_density is not None
                else float("nan")
            ),
        })

    csv_path = output_dir / "cosmos_euclid_density_fit.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    plot_path = output_dir / "cosmos_euclid_density_fit.png"
    if not args.no_plot:
        _plot(
            plot_path,
            centers=centers,
            cosmos_density=cosmos_binned,
            fitted_latent_density=fitted_latent,
            euclid_density=euclid_density,
            euclid_error=euclid_error,
            predicted_detected_density=predicted_detected,
            local_latent_density=local_latent,
            local_predicted_detected_density=local_predicted_detected,
            synthetic_density=synthetic_density,
            completeness=completeness,
            fit=fit,
        )

    m28 = MODEL_GRID < 28.0
    raw_cosmos_m28 = float(
        np.sum(cosmos_model_density[m28]) * MODEL_STEP
    )
    payload: dict[str, Any] = {
        "version": 1,
        "method": (
            "COSMOS latent number counts passed through a fitted affine "
            "F814W-to-VIS transfer with scatter and logistic Euclid completeness"
        ),
        "interpretation": (
            "Euclid non-star rows are detections, not confirmed galaxies. "
            "COSMOS sets the latent shape; Euclid calibrates the observation "
            "layer and, with at least three separated cones, a model-dependent "
            "latent normalization. This is not the generator's raw draw budget; "
            "calibrate that with the common detector applied to real and "
            "synthetic fields."
        ),
        "inputs": {
            "cosmos_counts_csv": str(cosmos_path),
            "cosmos_selection": args.cosmos_selection,
            "euclid_catalog_csv": str(euclid_path),
            "euclid_area_arcmin2": euclid_area,
            "euclid_cone_count": int(euclid_meta.get("cone_count", 1)),
            "euclid_cones": euclid_meta.get("cones"),
            "euclid_nonstar_rows_used": int(len(euclid_magnitudes)),
            "maximum_spurious_probability": float(
                args.maximum_spurious_probability
            ),
            "synthetic_truth_csv": (
                str(synthetic_path) if synthetic_density is not None else None
            ),
        },
        "fit": asdict(fit),
        "local_normalization_sensitivity_fit": asdict(local_fit),
        "euclid_latent_density_estimate": {
            "density_arcmin2": float(
                local_fit.population_scale * raw_cosmos_m28
            ),
            "cone_count": int(euclid_meta.get("cone_count", 1)),
            "use_local_normalization": (
                int(euclid_meta.get("cone_count", 1)) >= 3
            ),
            "method": (
                "free population normalization in the F814W-to-VIS "
                "observation fit over spatially separated Euclid cones"
            ),
            "caveat": (
                "Completeness-model extrapolation to the unseen population; "
                "not directly comparable to raw TNG draws. Requires at least "
                "three separated cones; one cone is retained only as a local "
                "cosmic-variance sensitivity estimate."
            ),
        },
        "reliability": {
            "cosmos_f814w_turnover_sensitive_above_mag": 27.5,
            "fixed_fit_vis_turnover_sensitive_above_mag": (
                24.0
                + fit.magnitude_slope * (27.5 - 24.0)
                + fit.vis_minus_f814w_mag
            ),
        },
        "latent_density": {
            "cosmos_m20_to_m28_arcmin2": float(
                np.sum(cosmos_binned)
            ),
            "fitted_m20_to_m28_arcmin2": float(
                np.sum(fitted_latent)
            ),
            "cosmos_m_lt_28_arcmin2": raw_cosmos_m28,
            "fitted_m_lt_28_arcmin2": (
                fit.population_scale * raw_cosmos_m28
            ),
            "locally_renormalized_m_lt_28_arcmin2": (
                local_fit.population_scale * raw_cosmos_m28
            ),
        },
        "detected_density": {
            "euclid_m20_to_m28_arcmin2": float(np.sum(euclid_density)),
            "predicted_m20_to_m28_arcmin2": float(
                np.sum(predicted_detected)
            ),
            "local_prediction_m20_to_m28_arcmin2": float(
                np.sum(local_predicted_detected)
            ),
        },
        "bins": rows,
        "outputs": {
            "csv": str(csv_path),
            "plot": None if args.no_plot else str(plot_path),
        },
    }
    json_path = output_dir / "cosmos_euclid_density_fit.json"
    payload["outputs"]["json"] = str(json_path)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    from euclid_polish.sky.generation.cosmos_tng_prior import (
        brightness_transfer_payload,
    )
    transfer = brightness_transfer_payload(json_path)
    if transfer is not None:
        transfer_path = output_dir / "photometric_transfer_candidate.json"
        transfer_path.write_text(json.dumps(transfer, indent=2, sort_keys=True))
        payload["outputs"]["photometric_transfer_candidate"] = str(
            transfer_path
        )
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cosmos-counts", default=DEFAULT_COSMOS_COUNTS)
    parser.add_argument(
        "--cosmos-selection",
        choices=("population", "clean", "isolated", "generator_ready"),
        default="clean",
    )
    parser.add_argument("--euclid", default=DEFAULT_EUCLID)
    parser.add_argument("--euclid-meta", default=DEFAULT_EUCLID_META)
    parser.add_argument("--synthetic", default=DEFAULT_SYNTHETIC)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--maximum-spurious-probability",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="fit and write numeric artifacts without rendering a figure",
    )
    return parser


def main() -> None:
    payload = run(build_parser().parse_args())
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
