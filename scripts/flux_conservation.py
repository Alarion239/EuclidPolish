"""Flux conservation on the NEXUS tiles: the Euclid MER catalogue's own
photometry redone on each model's SR and compared with the reported flux and
error (:mod:`euclid_polish.eval.flux_conservation`).

    python scripts/flux_conservation.py --models universal:member_249,production \\
        --out data/jwst_euclid_overlap/flux_conservation/report.json

Galaxies: MER sources detected in VIS, not spurious, ``POINT_LIKE_PROB`` <
0.1, Kron S/N ≥ 10, isolated (the input gives back the catalogue's Kron flux
within 0.5σ: MER masked no neighbour), with the Kron ellipse and the 4-FWHM matched aperture
inside the tile. ``lr`` (the input, each pixel split in four — exactly the
input's photometry) is always measured: it is the baseline every pull is
read against and sets the aperture corrections. The catalogue is fetched
once from the Euclid archive (public Q1) into ``--catalogue``.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
from astropy.table import Table
from astropy.wcs import WCS
from astroquery.esa.euclid import Euclid

from euclid_polish.config import Config
from euclid_polish.eval import flux_conservation as fc
from euclid_polish.photometry import uJy_to_ab_mag, uJy_to_electrons
from euclid_polish.training.inference import scaled_wcs_header
from euclid_polish.web.helpers import model_catalog
from euclid_polish.web.helpers import real_tiles as rt

#: The NEXUS tiles' footprint (deg) with a margin.
FOOTPRINT = {"ra": (268.10, 268.83), "dec": (65.06, 65.34)}
MARGIN_LR_PX = 4
#: Largest |input − catalogue| Kron pull (σ) of a galaxy kept as isolated.
ISOLATED_PULL = 0.5


def fetch_catalogue(path: str) -> Table:
    if os.path.isfile(path):
        return Table.read(path)
    cols = ["object_id", "right_ascension", "declination", "vis_det", "point_like_prob",
            "spurious_flag", "det_quality_flag", "semimajor_axis", "ellipticity", "position_angle",
            "kron_radius", "fwhm", "segmentation_area", "flux_detection_total",
            "fluxerr_detection_total", "flux_segmentation", "fluxerr_segmentation"]
    for band in fc.BANDS:
        cols.append(f"flag_{band}")
        for n in fc.APERTURE_MULTIPLES:
            cols += [f"flux_{band}_{n}fwhm_aper", f"fluxerr_{band}_{n}fwhm_aper"]
    query = (f"SELECT {', '.join(cols)} FROM catalogue.mer_catalogue "
             f"WHERE right_ascension BETWEEN {FOOTPRINT['ra'][0]} AND {FOOTPRINT['ra'][1]} "
             f"AND declination BETWEEN {FOOTPRINT['dec'][0]} AND {FOOTPRINT['dec'][1]} "
             "AND spurious_flag = 0")
    table = Euclid.launch_job_async(query).get_results()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    table.write(path, overwrite=True)
    return table


def galaxies(table: Table) -> Table:
    def col(name):
        return np.asarray(table[name].filled(np.nan) if hasattr(table[name], "filled")
                          else table[name], np.float64)

    keep = ((col("vis_det") == 1) & (col("point_like_prob") < 0.1)
            & (col("flux_detection_total") >= 10 * col("fluxerr_detection_total"))
            & (col("kron_radius") > 0) & np.isfinite(col("ellipticity")) & (col("fwhm") > 0))
    return table[keep]


class _Members:
    def __init__(self, runner, lr):
        self.runner, self.lr, self.memory = runner, lr, {}

    def get(self, label):
        if label not in self.memory:
            self.memory[label] = self.runner.predict(self.lr, label)
        return self.memory[label]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", required=True, help="comma list of model specs")
    p.add_argument("--catalogue", default="data/jwst_euclid_overlap/flux_conservation/mer_nexus.fits")
    p.add_argument("--out", default="data/jwst_euclid_overlap/flux_conservation/report.json")
    p.add_argument("--tiles", type=int, default=None, help="at most this many tiles")
    args = p.parse_args()
    specs = [s.strip() for s in args.models.split(",") if s.strip()]
    items = {s: model_catalog.resolve_spec(s) for s in specs}
    for spec, item in items.items():
        if not item.available:
            raise SystemExit(f"{spec}: {item.reason}")
    runner = model_catalog.EnsembleMemberRunner(labels=model_catalog.needed_members(items.values()))
    cat = galaxies(fetch_catalogue(args.catalogue))
    ra, dec = np.asarray(cat["right_ascension"], float), np.asarray(cat["declination"], float)
    entries = [e for e in rt.list_entries("nexus") if e.extras.get("jwst_file")][:args.tiles]
    measured = {name: [] for name in ("lr", *specs)}
    rows_out, started = [], time.monotonic()
    for t, entry in enumerate(entries):
        lr = np.asarray(rt.get_tile("nexus", entry.id, entry=entry).lr_e, np.float32)
        header = rt.lr_header(entry)
        x, y = WCS(header).celestial.world_to_pixel_values(ra, dec)
        h, w = lr.shape[:2]
        reach = np.maximum(np.asarray(cat["kron_radius"], float),
                           2.0 * np.asarray(cat["fwhm"], float) / fc.LR_PIXEL_ARCSEC) + MARGIN_LR_PX
        inside = np.flatnonzero((x - reach > 0) & (x + reach < w - 1) & (y - reach > 0) & (y + reach < h - 1))
        if not inside.size:
            continue
        rows = [{k: float(cat[k][i]) for k in ("kron_radius", "ellipticity", "position_angle", "fwhm")}
                for i in inside]
        target = float(np.median([r["fwhm"] for r in rows]))

        sx, sy = WCS(scaled_wcs_header(header, 2)).celestial.world_to_pixel_values(ra[inside], dec[inside])
        xy_sr = np.stack([sx, sy], 1)
        lr_sr = np.repeat(np.repeat(lr, 2, 0), 2, 1) / 4.0      # the input on the SR grid, flux kept
        measured["lr"] += fc.measure(lr_sr, xy_sr, rows, target, scale=2)
        members = _Members(runner, lr)
        for spec in specs:
            sr = model_catalog.predict(items[spec], lr, members)
            measured[spec] += fc.measure(sr, xy_sr, rows, target, scale=2)
        rows_out += [int(i) for i in inside]
        print(f"tile {t + 1}/{len(entries)} {entry.id}: {inside.size} galaxies "
              f"({len(rows_out)} total, {time.monotonic() - started:.0f} s)", flush=True)
    sub = cat[rows_out]
    vis_mag = uJy_to_ab_mag(np.asarray(sub["flux_detection_total"], float))
    report = {"meta": {"field": "NEXUS deep tiles (Euclid Q1 EDF-N)", "n_tiles": len(entries),
                       "n_galaxies": len(rows_out), "catalogue": os.path.basename(args.catalogue),
                       "native_fwhm_arcsec": fc.NATIVE_FWHM_ARCSEC,
                       "matched_fwhm_arcsec_median": round(float(np.median(sub["fwhm"])), 4)},
              "corrections": {}, "models": {}}
    conv = {band: uJy_to_electrons(1.0, Config.get_band(name))
            for band, name in zip(fc.BANDS, Config.LR_INPUT_BAND_NAMES, strict=True)}
    def column(name, key):
        return np.array([m[key] for m in measured[name]], np.float64)
    tests = [("kron_vis", "flux_detection_total", "fluxerr_detection_total", "vis")]
    tests += [(f"ap{n}_{b}", f"flux_{b}_{n}fwhm_aper", f"fluxerr_{b}_{n}fwhm_aper", b)
              for b in fc.BANDS for n in fc.APERTURE_MULTIPLES]
    lr_cols = {key: column("lr", key) for key, *_ in tests}
    def values(name, band):
        return np.asarray(sub[name].filled(np.nan) if hasattr(sub[name], "filled") else sub[name],
                          np.float64) * conv[band]
    # Isolated: the input gives back the catalogue's Kron flux, so MER masked
    # no neighbour (deblended sources lose their neighbours' pixels there).
    kron_c, kron_e = values("flux_detection_total", "vis"), values("fluxerr_detection_total", "vis")
    isolated = np.abs(lr_cols["kron_vis"] - kron_c) <= ISOLATED_PULL * kron_e
    report["meta"]["n_isolated"] = int(isolated.sum())
    for key, flux, err, band in tests:
        c, e = values(flux, band), values(err, band)
        c[~isolated] = np.nan
        bright = c >= 20 * e
        corr = 1.0 if key == "kron_vis" else fc.aperture_correction(lr_cols[key][bright], c[bright])
        report["corrections"][key] = round(corr, 4)
        for name in ("lr", *specs):
            report["models"].setdefault(name, {})[key] = fc.summarize(
                column(name, key), lr_cols[key], c, e, vis_mag, corr)
    with open(args.out + ".tmp", "w") as handle:
        json.dump(report, handle, indent=1)
    os.replace(args.out + ".tmp", args.out)
    np.savez_compressed(os.path.splitext(args.out)[0] + "_fluxes.npz",
                        object_id=np.asarray(sub["object_id"]), vis_mag=vis_mag,
                        **{f"{name}:{key}": column(name, key) for name in measured for key, *_ in tests})
    for name in ("lr", *specs):
        k = report["models"][name]["kron_vis"]
        print(f"{name}: Kron VIS pull {k['pull_median']:+.2f} ± {k['pull_mad']:.2f}σ, "
              f"SR/input {k['ratio_input_median']:.4f}, within 1σ {k['within_1sigma']:.0%}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
