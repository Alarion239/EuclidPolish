#!/usr/bin/env python3
"""Compare two treatments of sub-resolution Euclid MER galaxy radii.

This is a local catalogue experiment.  It shares the current COSMOS latent
population fit and refits only the Euclid response under three policies:
keep exact MER radii, drop unresolved rows entirely, or retain their magnitude
and fractional galaxy membership while left-censoring the radius.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "euclid_mpl_cache")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from euclid_polish.config import Config
from euclid_polish.population.joint_galaxy import (
    EUCLID_LOG_RE_EDGES,
    EUCLID_MAG_EDGES,
    fit_euclid_response,
    fit_payload,
    fit_schechter_evolution,
    fit_size_evolution,
    latent_population_cube,
    read_cosmos_population,
    read_euclid_population,
)

DEFAULT_COSMOS = Config.COSMOS_POPULATION_PRIOR_PATH
DEFAULT_EUCLID = "data/population_comparison/euclid_population.csv"
DEFAULT_EUCLID_META = "data/population_comparison/euclid_population_meta.json"
DEFAULT_OUTPUT_DIR = "data/population_comparison/cosmos2025"
OUTPUT_JSON = "euclid_unresolved_policy_comparison.json"
OUTPUT_PNG = "euclid_unresolved_policy_comparison.png"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(temporary, path)


def _plot(
    path: Path, results: dict[str, dict[str, Any]], threshold: float,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5), constrained_layout=True)
    titles = {
        "keep": "Current: use exact proxy radii",
        "drop": "Drop unresolved objects entirely",
        "censor": "Keep counts; censor unresolved radii",
    }
    colors = {"observed": "#1267d6", "model": "#cf3d2e"}
    for column, policy in enumerate(("keep", "drop", "censor")):
        item = results[policy]
        magnitude = item["magnitude"]
        radius = item["radius"]
        top = axes[0, column]
        bottom = axes[1, column]
        top.scatter(
            magnitude["x"], magnitude["observed"], s=22,
            facecolors="none", edgecolors=colors["observed"],
            linewidths=1.4, label="Euclid observed",
        )
        top.plot(
            magnitude["x"], magnitude["model"],
            color=colors["model"], linewidth=2.2, label="fitted response",
        )
        top.set_yscale("log")
        top.set(
            title=titles[policy], xlabel="VIS AB magnitude",
            ylabel="objects / arcmin² / mag",
        )

        resolved = np.asarray(radius["x"]) >= threshold
        bottom.scatter(
            np.asarray(radius["x"])[resolved],
            np.asarray(radius["observed"])[resolved],
            s=22, facecolors="none", edgecolors=colors["observed"],
            linewidths=1.4, label="resolved-radius bins",
        )
        bottom.plot(
            np.asarray(radius["x"])[resolved],
            np.asarray(radius["model"])[resolved],
            color=colors["model"], linewidth=2.2, label="fitted response",
        )
        if policy == "keep":
            bottom.scatter(
                np.asarray(radius["x"])[~resolved],
                np.asarray(radius["observed"])[~resolved],
                s=22, facecolors="none", edgecolors="#7a3db8",
                linewidths=1.4, label="exact sub-resolution proxy",
            )
            bottom.plot(
                np.asarray(radius["x"])[~resolved],
                np.asarray(radius["model"])[~resolved],
                color="#7a3db8", linewidth=2.0,
            )
        elif policy == "censor":
            bottom.scatter(
                [threshold], [radius["unresolved_observed_density"]],
                marker="<", s=70, color="#7a3db8",
                label=f"censored mass below {threshold:.2f}″",
            )
            bottom.scatter(
                [threshold], [radius["unresolved_model_density"]],
                marker="x", s=55, color=colors["model"],
                label="model censored mass",
            )
        bottom.axvline(threshold, color="0.45", linestyle="--", linewidth=1.2)
        bottom.set_xscale("log")
        bottom.set_yscale("log")
        bottom.set(
            xlabel="MER angular-radius proxy / arcsec",
            ylabel="objects / arcmin² / dex",
        )
        for axis in (top, bottom):
            axis.grid(alpha=0.2)
            axis.legend(frameon=False, fontsize=8)
        top.text(
            0.04, 0.05,
            f"deviance/dof = {item['reduced_deviance']:.2f}",
            transform=top.transAxes, fontsize=9,
        )
    fig.suptitle(
        f"Euclid unresolved-galaxy policy comparison (rMER < {threshold:.2f}″)",
        fontsize=16,
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run(args: argparse.Namespace) -> dict[str, Any]:
    threshold = float(args.unresolved_radius_arcsec)
    if not 0.075 < threshold < 0.30:
        raise ValueError("unresolved threshold must be between 0.075 and 0.30 arcsec")
    cosmos = read_cosmos_population(args.cosmos)
    euclid = read_euclid_population(args.euclid)
    euclid_meta = json.loads(Path(args.euclid_meta).read_text())
    area = float(euclid_meta["area_arcmin2"])
    lf_fit, _lf_observed, _lf_predicted = fit_schechter_evolution(
        np.asarray(cosmos["magnitude"]), np.asarray(cosmos["redshift"]),
    )
    size_fit = fit_size_evolution(
        np.asarray(cosmos["magnitude"]), np.asarray(cosmos["redshift"]),
        np.asarray(cosmos["radius_arcsec"]),
    )
    cube = latent_population_cube(lf_fit, size_fit)
    threshold_log = math.log10(threshold)
    radius_edges = np.unique(np.append(EUCLID_LOG_RE_EDGES, threshold_log))
    radius_centers = np.power(
        10.0, 0.5 * (radius_edges[:-1] + radius_edges[1:])
    )
    unresolved_bins = radius_edges[1:] <= threshold_log + 1e-10
    resolved_rows = np.asarray(euclid["radius_arcsec"]) >= threshold
    total_weight = float(np.sum(euclid["weight"]))
    unresolved_weight = float(np.sum(np.asarray(euclid["weight"])[~resolved_rows]))
    results: dict[str, dict[str, Any]] = {}
    for policy in ("keep", "drop", "censor"):
        fit, observed, predicted_density = fit_euclid_response(
            cube, euclid, area_arcmin2=area,
            unresolved_policy=policy,
            unresolved_radius_arcsec=threshold,
            log_radius_edges=radius_edges,
        )
        row_selection = resolved_rows if policy == "drop" else np.ones(
            len(resolved_rows), dtype=bool,
        )
        magnitude_observed, _ = np.histogram(
            np.asarray(euclid["magnitude"])[row_selection],
            bins=EUCLID_MAG_EDGES,
            weights=np.asarray(euclid["weight"])[row_selection],
        )
        model_bins = ~unresolved_bins if policy == "drop" else np.ones(
            len(unresolved_bins), dtype=bool,
        )
        magnitude_width = np.diff(EUCLID_MAG_EDGES)
        radius_width = np.diff(radius_edges)
        unresolved_width = threshold_log - radius_edges[0]
        results[policy] = {
            "fit": fit_payload(fit),
            "reduced_deviance": fit.poisson_deviance / fit.dof,
            "retained_weighted_galaxies": float(np.sum(
                np.asarray(euclid["weight"])[row_selection]
            )),
            "magnitude": {
                "x": (0.5 * (EUCLID_MAG_EDGES[:-1] + EUCLID_MAG_EDGES[1:])).tolist(),
                "observed": (magnitude_observed / area / magnitude_width).tolist(),
                "model": (
                    np.sum(predicted_density[:, model_bins], axis=1)
                    / magnitude_width
                ).tolist(),
            },
            "radius": {
                "x": radius_centers.tolist(),
                "observed": (
                    np.sum(observed, axis=0) / area / radius_width
                ).tolist(),
                "model": (
                    np.sum(predicted_density, axis=0) / radius_width
                ).tolist(),
                "unresolved_observed_density": (
                    float(np.sum(observed[:, unresolved_bins]))
                    / area / unresolved_width
                ),
                "unresolved_model_density": (
                    float(np.sum(predicted_density[:, unresolved_bins]))
                    / unresolved_width
                ),
            },
        }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / OUTPUT_PNG
    json_path = output_dir / OUTPUT_JSON
    payload = {
        "version": 1,
        "kind": "euclid_unresolved_radius_policy_comparison",
        "unresolved_radius_arcsec": threshold,
        "total_weighted_galaxies": total_weight,
        "unresolved_weighted_galaxies": unresolved_weight,
        "unresolved_fraction": unresolved_weight / total_weight,
        "policies": results,
        "plot": str(plot_path),
        "interpretation": (
            "drop removes unresolved objects from both counts and size; censor "
            "retains their magnitude and galaxy weight but uses only the event "
            "r_MER below the threshold"
        ),
        "tng_used": False,
    }
    _plot(plot_path, results, threshold)
    _atomic_json(json_path, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cosmos", default=DEFAULT_COSMOS)
    parser.add_argument("--euclid", default=DEFAULT_EUCLID)
    parser.add_argument("--euclid-meta", default=DEFAULT_EUCLID_META)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--unresolved-radius-arcsec", type=float, default=0.10)
    return parser


def main() -> None:
    payload = run(build_parser().parse_args())
    print(f"Wrote {payload['plot']}")
    print(
        "unresolved weighted fraction "
        f"{100.0 * payload['unresolved_fraction']:.3f}%"
    )
    for policy, result in payload["policies"].items():
        print(
            f"{policy}: reduced deviance {result['reduced_deviance']:.3f}; "
            f"weighted galaxies {result['retained_weighted_galaxies']:.1f}"
        )
    print("No TNG catalogue or FASRC resource was used.")


if __name__ == "__main__":
    main()
