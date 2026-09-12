#!/usr/bin/env python3
"""Independent four-band noise acquisition, measurement, and standalone report."""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
# Matplotlib reads its config directory on import; set a writable default first.
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/euclid_noise_mpl")

from euclid_polish.noise_assessment.archive import acquire_mer, initialize, read_json  # noqa: E402
from euclid_polish.noise_assessment.exposures import (  # noqa: E402
    acquire_exposures,
    measure_exposures,
)
from euclid_polish.noise_assessment.measurement import (  # noqa: E402
    measure_mer,
    select_validation,
)
from euclid_polish.noise_assessment.report import build_report  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPO / "data/population_comparison/noise_assessment")
    sub = parser.add_subparsers(dest="stage", required=True)
    init = sub.add_parser("init", help="Freeze the 44 saved positions and four exact example patches")
    init.add_argument(
        "--source-manifest",
        type=Path,
        default=REPO / "data/euclid_sky/vis_noise_samples/vis_noise_sampling_manifest.json",
    )
    init.add_argument(
        "--parents-manifest",
        type=Path,
        default=REPO / "data/euclid_sky/archive_fields/archive_fields_manifest.json",
    )
    init.add_argument(
        "--examples", type=Path, default=REPO / "data/population_comparison/mask_examples/provenance.json"
    )
    init.add_argument("--size", type=int, default=256)
    for name in ("acquire-mer", "acquire-exposures"):
        p = sub.add_parser(name, help="Retrieve public products sequentially; resume from checksummed cache")
        p.add_argument(
            "--pilot", action="store_true", help="Verify one pointing before expanding acquisition"
        )
    sub.add_parser("measure-mer", help="Compare exact legacy MAD with official total RMS")
    sub.add_parser("select", help="Freeze nine RMS-stratified pointings plus the bright-star stress case")
    p = sub.add_parser(
        "measure-exposures", help="Measure disjoint exposure differences and conditional median predictions"
    )
    p.add_argument("--draws", type=int, default=512)
    p.add_argument(
        "--size", type=int, default=80, help="Difference side in native-scale output pixels; bounds memory"
    )
    sub.add_parser(
        "report", help="Render PNG figures and a self-contained HTML report from saved measurements"
    )
    sub.add_parser("status", help="Show acquisition and measurement coverage without downloading")
    args = parser.parse_args()

    if args.stage == "init":
        initialize(args.output, args.source_manifest, args.parents_manifest, args.examples, args.size)
    elif args.stage == "acquire-mer":
        acquire_mer(args.output, pilot=args.pilot)
    elif args.stage == "acquire-exposures":
        acquire_exposures(args.output, pilot=args.pilot)
    elif args.stage == "measure-mer":
        measure_mer(args.output)
    elif args.stage == "select":
        print(select_validation(args.output))
    elif args.stage == "measure-exposures":
        measure_exposures(args.output, draws=args.draws, size=args.size)
    elif args.stage == "report":
        print(build_report(args.output))
    elif args.stage == "status":
        manifest = read_json(args.output / "manifest.json")
        print(
            "MER:",
            dict(
                Counter(
                    r.get("status", "pending") for p in manifest["patches"] for r in p.get("mer", {}).values()
                )
            ),
        )
        for filename, key in (
            ("summary.json", "mer_measurements"),
            ("summary.json", "exposure_measurements"),
        ):
            path = args.output / filename
            if path.exists():
                print(key + ":", dict(Counter(r["status"] for r in read_json(path).get(key, []))))


if __name__ == "__main__":
    main()
