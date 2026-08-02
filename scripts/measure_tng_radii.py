#!/usr/bin/env python3
"""Build the validated remote TNG VIS half-light-radius manifest."""

from __future__ import annotations

import argparse
import json
import os
import sys

from euclid_polish.config import Config
from euclid_polish.sky.generation.tng_radius_manifest import (
    build_manifest,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tng-dir", default=Config.TNG_SKIRT_DIR)
    parser.add_argument("--properties", default="")
    parser.add_argument("--output", default="")
    args = parser.parse_args(argv)
    properties = args.properties or os.path.join(
        Config.DATA_DIR, "_tng_infographics", "tng_properties.csv"
    )
    output = args.output or os.path.join(
        Config.DATA_DIR, "_tng_infographics", "tng_radius_manifest.json"
    )
    report = build_manifest(
        args.tng_dir, properties_path=properties, output_path=output,
    )
    print(json.dumps({
        key: report.get(key) for key in (
            "valid", "expected_count", "valid_count", "failed_count",
            "atlas_inventory_fingerprint", "manifest_fingerprint",
        )
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
