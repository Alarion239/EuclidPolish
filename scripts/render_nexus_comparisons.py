#!/usr/bin/env python3
"""Render Euclid LR / SR / NEXUS comparison plates for cached NEXUS tiles.

A thin CLI over :mod:`euclid_polish.web.helpers.nexus_plates` — the same
renderer the Figures › Plates job runs (``POST /api/figures/nexus-plates``).
Each tile is read through the C9 ``real`` viewer collection: the registered
four-band Euclid LR, the SR of one model spec (``--model``: production, mean,
rbf, member:member_<N>, gate:<variant>; the RBF-era cached NEXUS SRs are
``rbf``) and native NEXUS, so a plate shows exactly the arrays the viewer
shows. All panels cover the same 25.5″ tile. ``--band VIS`` (or another
band) stretches every panel on its own; ``--band temp`` renders LR and SR in
the viewer's "Temp" colour while NEXUS stays native grey. NEXUS is an
external morphological reference in a different band and unit, not a
photometric truth for the SR.

    python scripts/render_nexus_comparisons.py --tiles 40,42,70,178 --model rbf --band VIS,temp --tag m169-188
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from euclid_polish.web.helpers import nexus_plates


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tiles", default=",".join(str(t) for t in nexus_plates.DEFAULT_TILES),
                        help="comma-separated NEXUS tile numbers or real-tile ids (f200w-NNNN)")
    parser.add_argument("--band", default=nexus_plates.DEFAULT_BAND,
                        help="Euclid band(s) for LR/SR (VIS, Y_E, J_E, H_E) or 'temp'; comma list")
    parser.add_argument("--model", default="production",
                        help="model spec of the SR panel (production, mean, rbf, member:…, gate:…)")
    parser.add_argument("--tag", default="",
                        help="output sub-directory, e.g. the member range (default <model>-<date>)")
    parser.add_argument("--out-dir", default="",
                        help="plates root (default output/nexus_comparisons)")
    args = parser.parse_args()

    out_root = Path(args.out_dir) if args.out_dir else None
    try:
        for band in [item.strip() for item in args.band.split(",") if item.strip()]:
            record = nexus_plates.render_plates(
                args.tiles, band=band, model=args.model, tag=args.tag or None,
                out_root=out_root,
                progress=lambda step, total, label: print(f"[{step}/{total}] {label}"),
            )
            directory = (out_root or nexus_plates.plates_root()) / record["tag"]
            for tile in record["tiles"]:
                print(f"wrote {directory / tile['file']}")
            print(f"wrote {directory / record['sheet']}")
    except nexus_plates.PlateError as exc:
        raise SystemExit(f"error: {exc}") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
