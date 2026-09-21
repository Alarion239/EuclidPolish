#!/usr/bin/env python3
"""Render Euclid LR / SR / NEXUS comparison plates for cached NEXUS tiles.

Each tile is read through the ``nexus-field`` viewer collection (registered
four-band Euclid LR, the STARFULL combiner SR, native NEXUS), so a plate
shows exactly the arrays the viewer shows. All panels cover the same
25.5″ north-up tile. Every panel gets its own asinh display stretch: NEXUS
is an external morphological reference in a different band and unit, not a
photometric truth for the SR.

    python scripts/render_nexus_comparisons.py --tiles 16,18,31,88 --tag m169-178
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from euclid_polish.web.helpers.jwst_euclid import (
    _read_nexus_field_manifest,
    nexus_field_id,
)
from euclid_polish.web.helpers.viewer_data import get_cube

PANELS = (
    ("lr", "Euclid {band} (LR)", "0.10″/pix", "#1F6FB2"),
    ("sr", "Super-resolved {band}", "0.05″/pix", "#2E8B57"),
    ("jwst", "NEXUS {filter}", "0.03″/pix", "#D9760A"),
)


def _asinh_display(data: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(np.asarray(data, dtype=np.float32), nan=0.0)
    values = np.clip(values, 0.0, None)
    positive = values[values > 0]
    scale = float(np.percentile(positive, 90.0)) if positive.size else 1.0
    stretched = np.arcsinh(values / max(scale, 1e-12))
    lo, hi = np.percentile(stretched, [0.5, 99.5])
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((stretched - lo) / (hi - lo), 0.0, 1.0)


def _plane(cube: np.ndarray, info: dict, band: str) -> np.ndarray:
    bands = [str(name) for name in info.get("bands", [])]
    index = bands.index(band) if band in bands else 0
    return cube[..., index]


def _render(axes_row, index: int, params: dict, band: str, filter_name: str,
            *, titles: bool) -> None:
    for ax, (tier, title, scale, color) in zip(axes_row, PANELS, strict=True):
        cube, info = get_cube("nexus-field", index, tier, params)
        ax.imshow(_asinh_display(_plane(cube, info, band)), origin="lower",
                  cmap="gray", interpolation="nearest", vmin=0.0, vmax=1.0)
        if titles:
            ax.set_title(f"{title.format(band=band, filter=filter_name)}\n{scale}",
                         color="white", fontsize=12, fontweight="bold", pad=8)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(color)
            spine.set_linewidth(2.5)
    axes_row[0].set_ylabel(f"tile {index}", color="white", fontsize=12,
                           fontweight="bold")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field", default=nexus_field_id("F200W"))
    parser.add_argument("--tiles", default="16,18,31,88",
                        help="comma-separated viewer tile indices")
    parser.add_argument("--band", default="VIS", help="Euclid band for LR/SR")
    parser.add_argument("--tag", default="current",
                        help="output sub-directory, e.g. the member range")
    parser.add_argument("--out-dir", default="output/nexus_comparisons")
    args = parser.parse_args()

    manifest = _read_nexus_field_manifest(args.field)
    if manifest is None:
        raise SystemExit(f"no cached NEXUS field {args.field!r}")
    params = {"field": args.field}
    filter_name = str(manifest.get("filter") or "JWST")
    tiles = [int(item) for item in args.tiles.split(",") if item.strip()]
    out_dir = os.path.join(args.out_dir, args.tag)
    os.makedirs(out_dir, exist_ok=True)

    for index in tiles:
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.9), dpi=200,
                                 facecolor="black")
        _render(axes, index, params, args.band, filter_name, titles=True)
        fig.subplots_adjust(left=0.03, right=0.99, bottom=0.02, top=0.88,
                            wspace=0.03)
        path = os.path.join(out_dir, f"nexus_tile{index:03d}_{args.band}.png")
        fig.savefig(path, facecolor="black")
        plt.close(fig)
        print(f"wrote {path}")

    fig, axes = plt.subplots(len(tiles), 3, figsize=(10.5, 3.6 * len(tiles)),
                             dpi=200, facecolor="black", squeeze=False)
    for row, index in enumerate(tiles):
        _render(axes[row], index, params, args.band, filter_name, titles=row == 0)
    fig.subplots_adjust(left=0.05, right=0.99, bottom=0.01, top=0.95,
                        wspace=0.03, hspace=0.04)
    sheet = os.path.join(out_dir, f"nexus_tiles_{args.band}.png")
    fig.savefig(sheet, facecolor="black")
    plt.close(fig)
    print(f"wrote {sheet}")

    records = manifest.get("tiles", [])
    with open(os.path.join(out_dir, "provenance.json"), "w", encoding="utf-8") as handle:
        json.dump({
            "field_id": args.field,
            "band": args.band,
            "tiles": [{
                "index": index,
                "ra_deg": records[index].get("ra_deg"),
                "dec_deg": records[index].get("dec_deg"),
                "inference": records[index].get("inference"),
            } for index in tiles],
        }, handle, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
