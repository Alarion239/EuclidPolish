#!/usr/bin/env python3
"""Rerun the synthetic-sky population calibration on Euclid DR1 Wide.

The same chain the console runs on Q1 (Galaxy distributions › Query + fit,
Star distribution › Query + fit), with ``EUCLID_POLISH_CALIBRATION_RELEASE=dr1``
so every archive query hits the DR1 Wide tables over the frozen 300-tile sample
and every artifact lands in ``<DATA_DIR>/population_comparison/dr1/``. The
active Q1 calibration is never read as an input or overwritten, and nothing is
activated: the job leaves *candidates* to compare against Q1.

Steps (each is resumable; the aperture counts checkpoint per bin):

  galaxy_counts   MER + PHZ VIS-aperture counts      -> q1_galaxy_aperture_counts.json
  galaxy_radii    Sersic R_e + MER FWHM aggregates   -> q1_galaxy_sersic_radius_statistics.json
  catalogue       colour+SFR catalogue, 24 x 5' cones -> euclid_population.csv (+ meta, PHZ PDFs)
  galaxy_fit      count law + joint galaxy candidate  -> calibrations/euclid_galaxy_population_candidate.json
  star_counts     PHZ star counts                     -> q1_phz_star_counts.json
  star_colors     stellar colour sample + Gaia cones  -> q1_stellar_color_sample.csv, gaia_population.csv
  star_fit        stellar prior candidate             -> calibrations/star_population_candidate.json
  noise           MER RMS-map levels (S3 windows)     -> sky/observation/mer_noise_levels_dr1.json

(The ``q1_`` file names are kept so the console readers find them; the folder
says which release they hold.)

    python scripts/run_dr1_population_calibration.py [--steps galaxy_counts,...]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from euclid_polish.sky.observation.calibration_release import (  # noqa: E402
    DR1_POPULATION_CONE_COUNT,
    RELEASE_ENV,
    active_release,
    dr1_population_cone_centers,
    population_dir,
)
from euclid_polish.web.helpers import (  # noqa: E402
    population_calibration,
    population_comparison,
    q1_galaxy_counts,
    q1_galaxy_radius_statistics,
    q1_star_counts,
    q1_stellar_colors,
    star_population,
)
from euclid_polish.web.helpers.galaxy_distributions import (  # noqa: E402
    build_galaxy_distributions,
)

STEPS = (
    "galaxy_counts", "galaxy_radii", "catalogue", "galaxy_fit",
    "star_counts", "star_colors", "star_fit", "noise",
)
CONE_RADIUS_ARCMIN = 5.0


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def progress(prefix: str):
    last = {"t": 0.0}

    def report(done: int, total: int, label: str) -> None:
        now = time.monotonic()
        if now - last["t"] >= 30.0 or done >= total:
            last["t"] = now
            log(f"{prefix} {done}/{total} {label}")
    return report


def galaxy_counts() -> dict:
    result = q1_galaxy_counts.query_q1_galaxy_aperture_counts(
        progress=progress("galaxy counts"))
    return {k: result[k] for k in ("completed_queries", "total_queries", "footprint_area_deg2")}


def galaxy_radii() -> dict:
    result = q1_galaxy_radius_statistics.query_q1_galaxy_radius_statistics(
        progress=progress("galaxy radii"))
    return {k: result[k] for k in ("completed_queries", "total_queries", "complete")}


def catalogue() -> dict:
    meta = population_comparison.query_euclid_population_multi(
        count=DR1_POPULATION_CONE_COUNT,
        radius_arcmin=CONE_RADIUS_ARCMIN,
        centers=dr1_population_cone_centers(),
        progress=progress("catalogue"),
    )
    return {k: meta.get(k) for k in ("rows", "cone_count", "area_arcmin2")}


def galaxy_fit() -> dict:
    law = q1_galaxy_counts.fit_q1_galaxy_aperture_counts()
    candidate = population_calibration.fit_euclid_joint_galaxy_candidate()
    try:
        build_galaxy_distributions()
    except Exception as exc:  # noqa: BLE001 - plots only; the candidate is written
        log(f"galaxy plots not cached: {type(exc).__name__}: {exc}")
    f2 = (law.get("apertures") or {}).get("f2") or {}
    return {"law": f2.get("law") or {}, "fingerprint": candidate.get("fingerprint")}


def star_counts() -> dict:
    result = q1_star_counts.query_q1_phz_star_counts(progress=progress("star counts"))
    return {k: result[k] for k in ("expected_stars", "footprint_area_deg2")}


def star_colors() -> dict:
    result = q1_stellar_colors.query_q1_stellar_color_sample(progress=progress("star colours"))
    return {"euclid_rows": result["euclid"]["rows"], "gaia_rows": result["gaia"]["rows"],
            "archive": result["euclid"]["archive"]}


def star_fit() -> dict:
    fit = star_population.fit_star_population()
    for include_training in (False, True):
        try:
            star_population.star_distribution_payload(
                include_training=include_training, persist=True)
        except Exception as exc:  # noqa: BLE001 - plots only; the candidate is written
            log(f"star plot cache not written ({type(exc).__name__}: {exc})")
    return {"fingerprint": fit.get("fingerprint"), "version": fit.get("version")}


def noise() -> dict:
    script = REPO / "scripts" / "download_mer_noise_levels_dr1.py"
    plan = population_dir() / "mer_noise_levels" / "plan.json"
    if not plan.exists():
        subprocess.run([sys.executable, str(script), "plan"], check=True)
    subprocess.run([sys.executable, str(script), "acquire", "--max-minutes", "180"], check=True)
    subprocess.run([sys.executable, str(script), "finalize"], check=True)
    return {"table": "euclid_polish/sky/observation/mer_noise_levels_dr1.json"}


RUNNERS = {
    "galaxy_counts": galaxy_counts, "galaxy_radii": galaxy_radii, "catalogue": catalogue,
    "galaxy_fit": galaxy_fit, "star_counts": star_counts, "star_colors": star_colors,
    "star_fit": star_fit, "noise": noise,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", default=",".join(STEPS),
                        help=f"comma-separated subset of {', '.join(STEPS)}")
    parser.add_argument("--keep-going", action="store_true",
                        help="run later steps even if one fails")
    args = parser.parse_args()
    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    unknown = sorted(set(steps) - set(STEPS))
    if unknown:
        parser.error(f"unknown steps: {unknown}")

    os.environ[RELEASE_ENV] = "dr1"
    release = active_release()
    out_dir = population_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"release {release.name}: {release.survey}, {release.area_deg2:.1f} deg², "
        f"tables {dict(release.tables)}; outputs in {out_dir}")
    summary_path = out_dir / "calibration_run.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    failed = []
    for step in steps:
        log(f"=== {step}")
        t0 = time.monotonic()
        try:
            result = RUNNERS[step]()
            summary[step] = {"ok": True, "seconds": round(time.monotonic() - t0, 1),
                             "finished": time.strftime("%Y-%m-%dT%H:%M:%S"), **result}
            log(f"{step} done in {time.monotonic() - t0:.0f} s: {result}")
        except Exception as exc:  # noqa: BLE001 - record and decide below
            traceback.print_exc()
            summary[step] = {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                             "seconds": round(time.monotonic() - t0, 1)}
            failed.append(step)
        summary_path.write_text(json.dumps(summary, indent=2, default=str) + "\n")
        if failed and not args.keep_going:
            break
    log(f"finished; failed steps: {failed or 'none'}; summary {summary_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
