#!/usr/bin/env python
"""Download and align one cached JWST/Euclid overlap row.

Run the overlap discovery first, then pass the tile and JWST observation id
shown in its CSV.  The same cached workflow is exposed by the WebUI page.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from euclid_polish.web.helpers.jwst_euclid import download_and_align_pair, find_overlap_row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tile", required=True, help="Euclid VIS tile_index from the overlap CSV")
    parser.add_argument("--jwst", required=True, help="JWST observation id from the overlap CSV")
    parser.add_argument("--archive", choices=("esa", "mast"), default="esa")
    parser.add_argument("--size-arcsec", type=float, default=30.0)
    args = parser.parse_args(argv)

    row = find_overlap_row(args.archive, args.tile, args.jwst)
    if row is None:
        parser.error("no matching row in data/jwst_euclid_overlap/*.csv")
    manifest = download_and_align_pair(
        row,
        size_arcsec=args.size_arcsec,
        progress=lambda done, total, label: print(f"[{done}/{total}] {label}", flush=True),
    )
    print(json.dumps({
        "field_id": manifest["field_id"],
        "directory": str(Path("data/jwst_euclid_overlap/paired_fields") / manifest["field_id"]),
        "jwst_product": manifest["jwst_product"],
        "shape": manifest["shape"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
