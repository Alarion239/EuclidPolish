"""Kron VIS flux kept against VIS magnitude on the isolated NEXUS galaxies,
drawn from the per-galaxy fluxes :mod:`scripts.flux_conservation` saves
(``<report>_fluxes.npz``).

    python scripts/plot_flux_conservation.py \\
        data/jwst_euclid_overlap/flux_conservation/report_fluxes.npz \\
        data/jwst_euclid_overlap/flux_conservation/report_258_fluxes.npz \\
        --label "universal:member_258=258 pre-blur (375k)"

Several reports are merged on ``object_id``; each re-measures the input, so
the first one sets the isolated sample. Left: the median SR/input ratio per
half-magnitude bin. Right: the pull against the MER catalogue, with the
input's own pull for reference.
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table

from euclid_polish.config import Config
from euclid_polish.photometry import uJy_to_electrons
from scripts.flux_conservation import ISOLATED_PULL

KEY = "kron_vis"
MAG_EDGES = np.arange(21.0, 25.01, 0.5)
PULL_EDGES = np.linspace(-8.0, 8.0, 65)


def load(paths: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """``(object_id, vis_mag, input, {spec: SR})`` Kron VIS fluxes, every
    report's rows put in the first report's order."""
    first = np.load(paths[0])
    ids = np.asarray(first["object_id"])
    models: dict[str, np.ndarray] = {}
    for path in paths:
        z = np.load(path)
        index = {oid: i for i, oid in enumerate(np.asarray(z["object_id"]))}
        order = np.array([index.get(oid, -1) for oid in ids])
        for name in z.files:
            spec, _, key = name.rpartition(":")
            if key != KEY or spec == "lr" or spec in models:
                continue
            values = np.asarray(z[name], np.float64)[order]
            values[order < 0] = np.nan
            models[spec] = values
    return ids, np.asarray(first["vis_mag"], np.float64), np.asarray(first[f"lr:{KEY}"], np.float64), models


def catalogue_kron(ids: np.ndarray, path: str) -> tuple[np.ndarray, np.ndarray]:
    """The MER Kron VIS flux and error of each galaxy, in electrons."""
    table = Table.read(path)
    row = {oid: i for i, oid in enumerate(np.asarray(table["object_id"]))}
    take = np.array([row[oid] for oid in ids])
    conv = uJy_to_electrons(1.0, Config.get_band(Config.LR_INPUT_BAND_NAMES[0]))
    def col(name):
        return np.asarray(table[name].filled(np.nan) if hasattr(table[name], "filled") else table[name],
                          np.float64)[take] * conv
    return col("flux_detection_total"), col("fluxerr_detection_total")


def isolated(lr: np.ndarray, catalogue: np.ndarray, error: np.ndarray) -> np.ndarray:
    """The input gives back the catalogue's Kron flux: MER masked no neighbour."""
    return (np.abs(lr - catalogue) <= ISOLATED_PULL * error) & (error > 0) & (catalogue > 0)


def binned_median(values: np.ndarray, mag: np.ndarray, edges: np.ndarray = MAG_EDGES):
    """Bin centres and the median of the finite values in each (NaN if empty)."""
    centres = 0.5 * (edges[:-1] + edges[1:])
    medians = np.full(centres.size, np.nan)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        sel = (mag >= lo) & (mag < hi) & np.isfinite(values)
        if sel.any():
            medians[i] = np.median(values[sel])
    return centres, medians


def default_label(spec: str) -> str:
    kind, _, name = spec.partition(":")
    if kind == "milestone":
        member, _, step = name.partition("@")
        return f"{member.removeprefix('member_')} @ {int(step) / 1e6:g}M"
    return name.removeprefix("member_") if name else spec


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("reports", nargs="+", help="<report>_fluxes.npz files, merged on object_id")
    p.add_argument("--catalogue", default="data/jwst_euclid_overlap/flux_conservation/mer_nexus.fits")
    p.add_argument("--label", action="append", default=[], metavar="SPEC=LABEL",
                   help="legend label of one model (repeatable)")
    p.add_argument("--out", default="data/jwst_euclid_overlap/flux_conservation/kron_vis.png")
    args = p.parse_args()
    labels = dict(item.split("=", 1) for item in args.label)

    ids, mag, lr, models = load(args.reports)
    catalogue, error = catalogue_kron(ids, args.catalogue)
    keep = isolated(lr, catalogue, error)

    fig, (left, right) = plt.subplots(1, 2, figsize=(13.2, 4.6))
    for spec, sr in models.items():
        label = labels.get(spec, default_label(spec))
        ok = keep & np.isfinite(sr)
        centres, medians = binned_median(sr[ok] / lr[ok], mag[ok])
        left.plot(centres, medians, "o-", label=label)
        right.hist((sr[ok] - catalogue[ok]) / error[ok], PULL_EDGES, density=True, histtype="step",
                   lw=1.3, label=label)
    right.hist((lr[keep] - catalogue[keep]) / error[keep], PULL_EDGES, density=True, color="0.8",
               label="Euclid input")
    left.axhline(1.0, color="k", lw=1)
    left.set(title=f"Isolated NEXUS galaxies (n={int(keep.sum())})", xlabel="VIS Kron magnitude (AB)",
             ylabel="SR / input, Kron VIS flux (median)")
    right.set(title="Kron VIS pull in catalogue σ", xlabel="(SR − MER catalogue) / MER error")
    left.legend(fontsize=8, loc="lower left")
    right.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out, dpi=100)
    print(f"wrote {args.out}: {int(keep.sum())} isolated galaxies, {len(models)} models")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
