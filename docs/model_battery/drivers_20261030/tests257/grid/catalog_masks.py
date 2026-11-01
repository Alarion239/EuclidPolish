"""Per-window catalogue flags on the real tiles (MER, NEXUS footprint):
star_near (a point-like source close enough to light the window) and
gal_in (the window centre inside a galaxy's MER Kron aperture). No TF.

    python catalog_masks.py   -> prep_cat.npz
"""
import json
import os

import sys
sys.path.insert(0, os.getcwd())
import numpy as np
from astropy.table import Table
from astropy.wcs import WCS

from euclid_polish.web.helpers import real_tiles as rt

HERE = os.path.dirname(os.path.abspath(__file__))
P = np.load(os.path.join(HERE, "prep.npz"))
META = json.load(open(os.path.join(HERE, "prep.json")))
ORIGINS = P["origins"]
HALF_DIAG = 32 / np.sqrt(2.0) * np.sqrt(2.0)        # window half-diagonal in LR px (32-px footprint) ~ 22.6
KRON_ANGLE_OFFSET_DEG = 0.0


def col(t, n):
    x = t[n]
    return np.asarray(x.filled(np.nan) if hasattr(x, "filled") else x, np.float64)


def main():
    from euclid_polish.eval import flux_conservation as fc
    off = float(getattr(fc, "KRON_ANGLE_OFFSET_DEG", 0.0))
    cat = Table.read("data/jwst_euclid_overlap/flux_conservation/mer_nexus.fits")
    ra, dec = col(cat, "right_ascension"), col(cat, "declination")
    plp, mag = col(cat, "point_like_prob"), 23.9 - 2.5 * np.log10(np.clip(col(cat, "flux_detection_total"), 1e-6, None))
    star = np.isfinite(plp) & (plp > 0.5) & np.isfinite(mag)
    gal = (col(cat, "vis_det") == 1) & (plp < 0.1) & (col(cat, "kron_radius") > 0) & np.isfinite(col(cat, "ellipticity"))
    # light reach of a star (LR px): 10 px at VIS 21 mag, x10**(0.25 dm): 32 px at 19, 100 px at 17, 316 at 15
    r_star = 10.0 * 10 ** (0.25 * (21.0 - mag))
    kr, ell, pa = col(cat, "kron_radius"), col(cat, "ellipticity"), col(cat, "position_angle")
    entries = {e.id: e for e in rt.list_entries("nexus")}
    cy = (ORIGINS[:, 0] + 32) / 2.0 - 0.5                 # window centres, LR px
    cx = (ORIGINS[:, 1] + 32) / 2.0 - 0.5
    star_near, gal_in, gal_big = [], [], []
    for tid in META["tiles"]:
        w = WCS(rt.lr_header(entries[tid])).celestial
        x, y = w.world_to_pixel_values(ra, dec)
        near = (x > -400) & (x < 655) & (y > -400) & (y < 655)
        s = np.flatnonzero(near & star)
        d = np.hypot(cx[:, None] - x[s][None], cy[:, None] - y[s][None])          # (N, S)
        star_near.append((d < (r_star[s][None] + HALF_DIAG)).any(1))
        g = np.flatnonzero(near & gal)
        dx, dy = cx[:, None] - x[g][None], cy[:, None] - y[g][None]
        th = np.deg2rad(pa[g] + off)[None]
        u = dx * np.cos(th) + dy * np.sin(th)
        v = -dx * np.sin(th) + dy * np.cos(th)
        a = kr[g][None]
        b = a * (1.0 - ell[g][None])
        r_ell = np.hypot(u / a, v / np.maximum(b, 1e-3))
        inside = r_ell <= 1.0
        gal_in.append(inside.any(1))
        gal_big.append((inside & (a >= 20.0)).any(1))
    np.savez(os.path.join(HERE, "prep_cat.npz"), star_near=np.array(star_near), gal_in=np.array(gal_in),
             gal_big=np.array(gal_big))
    out = P["out_real"]
    sn, gi, gb = np.array(star_near), np.array(gal_in), np.array(gal_big)
    print("outskirt windows", int(out.sum()), "near a star", int((out & sn).sum()), "in a galaxy Kron aperture",
          int((out & gi).sum()), "clean (no star, in galaxy)", int((out & ~sn & gi).sum()),
          "clean + big galaxy (Kron >= 2\")", int((out & ~sn & gb).sum()),
          "tiles", int(((out & ~sn & gi).sum(1) > 0).sum()))
    print("all inner windows near a star:", float(sn[:, P["inner"]].mean()))


if __name__ == "__main__":
    main()
