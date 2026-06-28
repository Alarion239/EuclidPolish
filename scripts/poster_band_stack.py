#!/usr/bin/env python3
"""Transpose per-band TNG stacks into one band-stacked FITS for a single viewpoint.

The ``tng_stack`` step card on http://127.0.0.1:8765/tng ("download latest
stacked FITS") gives you, per Euclid band, a multi-extension FITS:

    PrimaryHDU (TNGID/BAND/NORIENT) + ImageHDU O1..O5   (one per SKIRT viewpoint)

Download one such file for each band and you have ``Gal_VIS.fits``,
``Gal_Y.fits``, ``Gal_J.fits``, ``Gal_H.fits`` — each holding all 5 viewpoints
of ONE band. This script does the transpose: pick a viewpoint number 1..5
(O1..O5) and pull that frame out of each of the 4 files, bundling the bands into
one multi-extension FITS:

    PrimaryHDU (TNGID/ORIENT/NBAND) + ImageHDU VIS/Y/J/H   (one per band)

Each band's pixel data and per-frame header are preserved verbatim (the same
header that ``tng_stack`` copied from the original SKIRT frame).

Usage
-----
    python3 scripts/poster_band_stack.py \\
        Gal_VIS.fits Gal_Y.fits Gal_J.fits Gal_H.fits 3 \\
        -o Gal_O3_bands.fits

The 4 files may be given in any order — each band is identified by the ``BAND``
keyword in its PrimaryHDU (falling back to the filename). The viewpoint number
(1..5) may appear anywhere among the positional arguments. If ``-o`` is omitted
the output is written next to the inputs as ``<TNGID>_O<N>_bands.fits``.
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np
from astropy.io import fits

# Canonical Euclid band order for the output extensions (mirrors
# fasrc_tng_infographic.SINGLE_BANDS = TNG_FITS_BANDS).
BANDS = ("VIS", "Y", "J", "H")
ORIENTATIONS = (1, 2, 3, 4, 5)


def _band_of(path: str) -> str:
    """Identify a per-band stack's band from its PrimaryHDU BAND keyword,
    falling back to the filename (e.g. ``Gal_VIS.fits`` -> ``VIS``)."""
    try:
        with fits.open(path, memmap=False) as hdul:
            band = str(hdul[0].header.get("BAND", "")).strip().upper()
        if band in BANDS:
            return band
    except Exception:
        pass
    # Filename fallback: match a band token bounded by non-alphanumerics.
    stem = os.path.basename(path).upper()
    for b in BANDS:
        if re.search(rf"(?<![A-Z0-9]){b}(?![A-Z0-9])", stem):
            return b
    raise ValueError(
        f"cannot determine Euclid band for {path!r} (no BAND header, no "
        f"recognizable band token in the filename)")


def _frame_hdu(path: str, orient: int) -> fits.ImageHDU:
    """Return the ``O{orient}`` ImageHDU from a per-band stack, data+header
    preserved. Matches by EXTNAME, falling back to the ORIENT keyword."""
    with fits.open(path, memmap=False) as hdul:
        name = f"O{orient}"
        for hdu in hdul[1:]:
            if str(hdu.header.get("EXTNAME", "")).upper() == name:
                return fits.ImageHDU(data=np.asarray(hdu.data),
                                     header=hdu.header.copy())
        for hdu in hdul[1:]:
            if int(hdu.header.get("ORIENT", -1)) == orient:
                return fits.ImageHDU(data=np.asarray(hdu.data),
                                     header=hdu.header.copy())
    raise FileNotFoundError(
        f"no viewpoint O{orient} extension in {path!r}")


def build_band_stack(paths: list[str], orient: int) -> fits.HDUList:
    """Bundle viewpoint ``orient`` from each per-band file into one FITS with
    one ImageHDU per band, ordered VIS/Y/J/H."""
    by_band = {}
    for p in paths:
        b = _band_of(p)
        if b in by_band:
            raise ValueError(
                f"two files map to band {b}: {by_band[b]!r} and {p!r}")
        by_band[b] = p
    missing = [b for b in BANDS if b not in by_band]
    if missing:
        raise ValueError(f"missing band file(s) for: {', '.join(missing)}")

    primary = fits.PrimaryHDU()
    primary.header["ORIENT"] = (orient, "SKIRT viewpoint index (O1..O5)")
    primary.header["NBAND"] = (len(BANDS), "number of stacked Euclid bands")
    hdul = fits.HDUList([primary])
    tngid = None
    for band in BANDS:
        hdu = _frame_hdu(by_band[band], orient)
        hdu.header["EXTNAME"] = band
        hdu.header["BAND"] = (band, "Euclid band of this frame")
        hdu.header["ORIENT"] = (orient, "SKIRT viewpoint index")
        hdu.name = band
        if tngid is None:
            tngid = hdu.header.get("TNGID")
        hdul.append(hdu)
    if tngid is not None:
        primary.header["TNGID"] = (str(tngid),
                                   "IllustrisTNG TNG50-1 subhalo id")
    return hdul


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("args", nargs="+",
                   help="The 4 per-band FITS files (any order) plus the "
                        "viewpoint number 1..5 (anywhere among them).")
    p.add_argument("-o", "--out", default=None,
                   help="Output FITS path (default: <TNGID>_O<N>_bands.fits "
                        "next to the inputs).")
    return p.parse_args(argv)


def _split_positionals(items: list[str]) -> tuple[list[str], int]:
    """Separate the bare 1..5 viewpoint number from the FITS paths."""
    orients = [a for a in items if a.strip() in {str(o) for o in ORIENTATIONS}]
    paths = [a for a in items if a not in orients]
    if len(orients) != 1:
        raise SystemExit(
            "expected exactly one viewpoint number (1..5) among the "
            f"arguments, found {len(orients)}: {orients}")
    if len(paths) != len(BANDS):
        raise SystemExit(
            f"expected {len(BANDS)} per-band FITS files, found {len(paths)}: "
            f"{paths}")
    return paths, int(orients[0])


def main(argv: list[str] | None = None) -> int:
    ns = parse_args(argv)
    paths, orient = _split_positionals(ns.args)
    for p in paths:
        if not os.path.isfile(p):
            raise SystemExit(f"no such file: {p}")

    hdul = build_band_stack(paths, orient)

    out = ns.out
    if out is None:
        tngid = hdul[0].header.get("TNGID", "TNG")
        out = os.path.join(os.path.dirname(os.path.abspath(paths[0])),
                           f"{tngid}_O{orient}_bands.fits")
    hdul.writeto(out, overwrite=True)
    bands = ", ".join(h.name for h in hdul[1:])
    print(f"wrote {out}  (O{orient}: {bands})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
