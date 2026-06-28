#!/usr/bin/env python
"""Dump everything the downloaded Euclid cutouts carry about orientation /
position angle, to settle whether the telescope ROLL is recoverable from the
FITS headers (vs. only from the diffraction-spike orientation in the pixels).

For ``--n`` cutouts of a band it:
  1. prints the FULL header of the first cutout (every keyword once),
  2. for each cutout prints the WCS-derived pixel scale + the position angle of
     the +y axis (East of North) + the cutout-centre sky position, and lists any
     orientation keywords present (CD/PC/CDELT/CROTA, ORIENTAT, PA_*, ROLL*, …),
  3. summarises whether the orientation VARIES across field position (if it does,
     the header encodes the roll and we can read it; if it's ~constant North-up,
     the mosaic is reprojected and the roll lives only in the spikes), plus the
     union of every header keyword seen.

Read-only. Run where the cutouts live (FASRC):
    python scripts/inspect_cutout_orientation.py --band VIS --n 30 --size 256
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from euclid_polish.config import Config

# Keywords that (in various missions' conventions) carry orientation / roll.
ORIENT_KEYS = [
    "CD1_1", "CD1_2", "CD2_1", "CD2_2",
    "PC1_1", "PC1_2", "PC2_1", "PC2_2",
    "CDELT1", "CDELT2", "CROTA1", "CROTA2",
    "ORIENTAT", "PA_V3", "PA_APER", "PA_OBS", "POSANG", "POS_ANGL",
    "ROLL", "ROLL_REF", "ROLL_NOM", "ROLLANGL", "SUNANGLE", "PA",
]


def _orientation(header) -> dict | None:
    """WCS pixel scale + position angle of the +y axis (deg E of N) + the
    cutout-centre sky position. ``None`` if the header has no usable WCS."""
    try:
        w = WCS(header)
        m = np.asarray(w.pixel_scale_matrix, dtype=float)   # deg/pix, incl. rotation
        ny, nx = (header.get("NAXIS2", 0), header.get("NAXIS1", 0))
        lon, lat = w.all_pix2world(nx / 2.0, ny / 2.0, 0)
    except Exception:
        return None
    sx = float(np.hypot(m[0, 0], m[1, 0]) * 3600.0)
    sy = float(np.hypot(m[0, 1], m[1, 1]) * 3600.0)
    pa = float(np.degrees(np.arctan2(m[0, 1], m[1, 1])))
    det = float(m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0])
    return {"pixscale_x": sx, "pixscale_y": sy, "pa_y_deg": pa,
            "parity": ("flipped" if det > 0 else "normal"),
            "ra": float(lon), "dec": float(lat)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--band", default="VIS", help="Band to inspect (default VIS).")
    ap.add_argument("--size", type=int, default=Config.DEFAULT_CUTOUT_SIZE,
                    help="Cutout side (px), to match the filename suffix.")
    ap.add_argument("--n", type=int, default=20, help="Cutouts to inspect.")
    ap.add_argument("--cutout-dir", default=None,
                    help="Override the band cutout directory.")
    ap.add_argument("--full-header", action="store_true",
                    help="Dump the FULL header of every cutout (not just the first).")
    args = ap.parse_args()

    cutout_dir = args.cutout_dir or Config.cutout_dir_for_band(args.band)
    pattern = os.path.join(cutout_dir, f"star_*_{args.size}.fits")
    files = sorted(glob.glob(pattern))[: args.n]
    print(f"band={args.band}  size={args.size}  dir={cutout_dir}")
    if not files:
        print(f"  no cutouts matching {pattern}")
        return 1
    print(f"  inspecting {len(files)} cutout(s)\n")

    all_keys: set = set()
    pas = []
    print(f"{'file':28s} {'pa_y(deg)':>10s} {'pixscale\"':>10s} {'parity':>8s} "
          f"{'ra':>10s} {'dec':>9s}   orient-keys")
    for i, path in enumerate(files):
        try:
            with fits.open(path) as hdul:
                header = hdul[0].header.copy()
        except Exception as e:
            print(f"  {os.path.basename(path)}: unreadable ({type(e).__name__})")
            continue
        all_keys.update(header.keys())
        present = [k for k in ORIENT_KEYS if k in header]

        if i == 0 or args.full_header:
            print("=" * 72)
            print(f"FULL HEADER — {os.path.basename(path)}")
            print("=" * 72)
            print(header.tostring(sep="\n", padding=False, endcard=False))
            print("=" * 72 + "\n")

        o = _orientation(header)
        if o is None:
            print(f"{os.path.basename(path):28s} {'(no WCS)':>10s}")
            continue
        pas.append(o["pa_y_deg"])
        print(f"{os.path.basename(path):28s} {o['pa_y_deg']:10.4f} "
              f"{o['pixscale_y']:10.4f} {o['parity']:>8s} "
              f"{o['ra']:10.5f} {o['dec']:9.5f}   "
              f"{', '.join(f'{k}={header[k]}' for k in present) or '(none)'}")

    print("\n" + "-" * 72)
    if pas:
        arr = np.asarray(pas)
        spread = float(arr.max() - arr.min())
        print(f"position angle of +y axis: mean={arr.mean():.4f}°  "
              f"std={arr.std():.4f}°  spread={spread:.4f}°  "
              f"(min={arr.min():.4f}, max={arr.max():.4f})")
        if spread > 0.5:
            print("  → orientation VARIES across cutouts: the WCS encodes the "
                  "local roll — readable directly (no spike-angle needed).")
        else:
            print("  → orientation is ~constant (reprojected/North-up mosaic): "
                  "the header does NOT give the roll; it lives in the spikes.")
    orient_present = sorted(k for k in ORIENT_KEYS if k in all_keys)
    print(f"\norientation keywords present in any cutout: "
          f"{orient_present or '(none of the usual ones)'}")
    print(f"\nALL header keywords seen ({len(all_keys)}):")
    print("  " + ", ".join(sorted(all_keys)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
