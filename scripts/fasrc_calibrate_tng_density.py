#!/usr/bin/env python
"""Generate an isolated matched-seed TNG density sweep and fit its response."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import tensorflow as tf

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from euclid_polish.config import Config
from euclid_polish.image.core import Image
from euclid_polish.sky.generation.source_catalog import read_sources
from euclid_polish.web.helpers.population_calibration import (
    fit_density_response,
)
from euclid_polish.web.helpers.tng_prior import DetectionAccumulator


def _densities(value: str) -> list[float]:
    result = sorted({float(part) for part in value.split(",") if part.strip()})
    if len(result) < 3 or any(number <= 0 for number in result):
        raise argparse.ArgumentTypeError("densities needs at least three positives")
    return result


def _detect(records_dir: Path) -> list[float]:
    dirty = records_dir / "dirty_validate.tfrecord"
    sources = read_sources(str(records_dir / "sources_validate.csv"))
    accumulator = DetectionAccumulator()
    for index, raw in enumerate(tf.data.TFRecordDataset([str(dirty)])):
        image = Image.from_tfrecord(raw)
        accumulator.add(image.data[..., 0], sources.get(index, []))
    payload = accumulator.payload()
    return [
        max(float(pos) - float(neg) - float(star), 0.0)
        for pos, neg, star in zip(
            payload["positive"], payload["negative"],
            payload["matched_stars"], strict=True,
        )
    ]


def _run_density(args: argparse.Namespace, density: float) -> list[float]:
    label = f"{density:g}".replace(".", "p")
    records = args.output_dir / f"records_{label}"
    cmd = [
        sys.executable, "scripts/run_pipeline.py",
        "--records-dir", str(records),
        "--ntrain", "0", "--nvalid", str(args.fields), "--ntest", "0",
        "--image-size", str(args.image_size),
        "--seed", str(args.seed),
        "--gen-workers", str(args.workers),
        "--galaxy-density-arcmin2", f"{density:g}",
        "--galaxy-thinning-max-density-arcmin2", f"{max(args.densities):g}",
        "--star-density-arcmin2", f"{args.star_density:g}",
        "--star-mag-slope", f"{args.star_mag_slope:g}",
        "--star-mag-bright", f"{args.star_mag_bright:g}",
        "--star-mag-faint", f"{args.star_mag_faint:g}",
        "--cosmos-vis-offset-mag", f"{args.transfer_offset:.12g}",
        "--cosmos-vis-magnitude-slope", f"{args.transfer_slope:.12g}",
        "--cosmos-vis-scatter-mag", f"{args.transfer_scatter:.12g}",
        "--cosmos-vis-transfer-source",
        f"fixed_normalization_fit:{args.transfer_fingerprint}:density_sweep",
        "--cosmos-vis-transfer-artifact-json", args.transfer_artifact_json,
        "--regenerate-splits", "validate", "--skip-train",
    ]
    subprocess.run(cmd, cwd=_ROOT, check=True)
    return _detect(records)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--densities", type=_densities,
                        default=_densities("240,280,320,360,400"))
    parser.add_argument("--fields", type=int, default=100)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=510)
    parser.add_argument("--seed", type=int, default=71032)
    parser.add_argument("--field-area-arcmin2", type=float, required=True)
    parser.add_argument("--euclid-field-detections", type=json.loads, required=True)
    parser.add_argument("--euclid-cone-densities", type=json.loads, default=[])
    parser.add_argument("--transfer-offset", type=float, required=True)
    parser.add_argument("--transfer-slope", type=float, required=True)
    parser.add_argument("--transfer-scatter", type=float, required=True)
    parser.add_argument("--transfer-fingerprint", required=True)
    parser.add_argument("--transfer-artifact-json", required=True)
    parser.add_argument("--star-density", type=float,
                        default=Config.DEFAULT_STAR_DENSITY_ARCMIN2)
    parser.add_argument("--star-mag-slope", type=float,
                        default=Config.STAR_MAG_SLOPE)
    parser.add_argument("--star-mag-bright", type=float,
                        default=Config.STAR_MAG_BRIGHT)
    parser.add_argument("--star-mag-faint", type=float,
                        default=Config.STAR_MAG_FAINT)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(Config.DATA_DIR) / "population_comparison" /
        "calibrations" / "tng_density_sweep",
    )
    args = parser.parse_args(argv)
    if args.fields < 2:
        parser.error("--fields must be at least 2")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    detections = [_run_density(args, density) for density in args.densities]
    result = fit_density_response(
        args.densities,
        detections,
        [float(value) for value in args.euclid_field_detections],
        transfer_fingerprint=args.transfer_fingerprint,
        active_transfer_fingerprint=args.transfer_fingerprint,
        field_area_arcmin2=args.field_area_arcmin2,
        euclid_cone_detection_densities=[
            float(value) for value in args.euclid_cone_densities
        ],
    )
    identity = {
        "transfer_fingerprint": args.transfer_fingerprint,
        "densities": args.densities,
        "fields": args.fields,
        "seed": args.seed,
    }
    result["calibration_fingerprint"] = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode("utf-8")
    ).hexdigest()
    result["seed"] = args.seed
    result["nested_thinning_max_density_arcmin2"] = max(args.densities)
    result["records_root"] = str(args.output_dir)
    output = args.output_dir.parent / "tng_density_calibration.json"
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True))
    os.replace(temporary, output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
