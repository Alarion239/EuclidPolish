#!/usr/bin/env python3
"""Build the validated remote TNG VIS half-light-radius manifest."""

from __future__ import annotations

import argparse
import json
import os
import sys

from euclid_polish.config import Config
from euclid_polish.tng.radius_manifest import (
    build_manifest,
    write_parameter_summary,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tng-dir", default=Config.TNG_SKIRT_DIR)
    parser.add_argument("--properties", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--summary", default="")
    parser.add_argument(
        "--workers", type=int,
        default=max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))),
    )
    args = parser.parse_args(argv)
    properties = args.properties or os.path.join(
        Config.DATA_DIR, "_tng_infographics", "tng_properties.csv"
    )
    output = args.output or os.path.join(
        Config.DATA_DIR, "_tng_infographics", "tng_radius_manifest.json"
    )
    summary = args.summary or os.path.join(
        Config.DATA_DIR, "_tng_infographics", "tng_atlas_parameters.csv"
    )
    report = build_manifest(
        args.tng_dir, properties_path=properties, output_path=output,
        workers=args.workers,
    )
    summary_meta = None
    if report.get("valid"):
        summary_meta = write_parameter_summary(
            summary, report, properties_path=properties,
        )
    print(json.dumps({
        key: report.get(key) for key in (
            "valid", "expected_count", "valid_count", "failed_count",
            "atlas_inventory_fingerprint", "manifest_fingerprint",
        )
    } | {
        "parameter_summary": summary if summary_meta else None,
        "parameter_summary_fingerprint": (
            summary_meta.get("summary_fingerprint") if summary_meta else None
        ),
    }, sort_keys=True))
    if report.get("failures"):
        for failure in report["failures"]:
            print(
                f"{failure['subhalo_id']} O{failure['orientation']}: "
                f"{failure['error']}",
                file=sys.stderr,
            )
    return 0 if report.get("valid") else 2


if __name__ == "__main__":
    raise SystemExit(main())
