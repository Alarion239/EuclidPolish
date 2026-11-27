"""Real-sky bright-star check: how much of a real DR1 bright star's spikes and
halo a model keeps in its SR, and how much of a spike line crossing a field
from a star outside it.

The target is the deconvolved sky, which has neither spikes nor a halo at
these radii, so a model should keep as little of them as possible. Inputs
(``--inputs``, built by output/bright_star_artifacts/scripts/bs_inputs.py
from DR1 Wide cutouts): ``centred`` (N, 384, 384, 4) electrons on 0.1″
pixels with the star at the centre, and ``offfield`` (M, 256, 256, 4) whose
star lies 15-41″ away along one of its spikes, with JSON metadata (star
position in crop pixels, spike angle).

Per model and band (VIS, H), the SR is sum-binned back to 0.1″ and compared
with the input:

* centred: azimuthal-median halo and spike ridge light (summed ±2.5° across
  each of the six highest peaks of the halo-subtracted, radius-averaged polar
  profile) at 8-12″ and 12-18″; ``keep = SR / input``;
* off-field: light across the known spike line (mean of the 8-px band on the
  ridge above the median 6-12 px off it); ``keep = SR / input``.

Light, not peak height: a model that sharpens a spike keeps the same light
in a narrower, taller ridge.

Blank input pixels are left out of every statistic. SR outputs are written
to ``<out>/sr_<slug>.npz`` and the numbers to ``<out>/bright_star_check.json``.

    python scripts/bright_star_check.py milestone:member_280@200000 \\
        milestone:member_281@200000 --inputs inputs.npz --out output/bright_star_check
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.signal import find_peaks

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from euclid_polish.eval.battery.common import ModelRunner, resolve  # noqa: E402

BANDS = {"VIS": 0, "H_E": 3}
RADII_ARCSEC = ((8.0, 12.0), (12.0, 18.0))
PIX = 0.1
N_THETA = 1440
RIDGE_HALF_WIDTH = 10            # polar samples (2.5°) summed either side of a ridge


def binned(sr: np.ndarray) -> np.ndarray:
    """SR (2h, 2w, c) sum-binned 2x2 onto the input's 0.1″ grid."""
    h, w = sr.shape[0] // 2, sr.shape[1] // 2
    return sr[:2 * h, :2 * w].reshape(h, 2, w, 2, -1).sum(axis=(1, 3))


def halo_and_ridge(img: np.ndarray, blank: np.ndarray, sx: float, sy: float,
                   lo: float, hi: float) -> tuple[float, float]:
    """Azimuthal-median halo and six-peak spike ridge (per pixel) at lo-hi arcsec."""
    r = np.arange(lo / PIX, hi / PIX, 1.0)
    th = np.linspace(0.0, 2 * np.pi, N_THETA, endpoint=False)
    yy, xx = sy + r[:, None] * np.sin(th), sx + r[:, None] * np.cos(th)
    P = map_coordinates(np.where(blank, 0.0, img), [yy, xx], order=1, cval=np.nan)
    M = map_coordinates(blank.astype(np.float32), [yy, xx], order=0, cval=1.0) > 0.5
    P[M] = np.nan
    halo_r = np.nanmedian(P, axis=1)
    prof = np.nanmean(P - halo_r[:, None], axis=0)
    prof = np.nan_to_num(prof - np.nanmedian(prof))
    pk, _ = find_peaks(np.concatenate([prof, prof[:N_THETA // 24]]), distance=N_THETA // 24)
    pk = np.unique(pk % N_THETA)
    if not pk.size:
        return float(np.nanmedian(halo_r)), 0.0
    top = pk[np.argsort(prof[pk])[::-1][:6]]
    # Light across each ridge (±2.5°), so a sharpened ridge of the same flux
    # counts the same as the input's.
    win = np.arange(-RIDGE_HALF_WIDTH, RIDGE_HALF_WIDTH + 1)
    ridge = float(np.mean([prof[(t + win) % N_THETA].sum() for t in top]))
    return float(np.nanmedian(halo_r)), ridge


def line_amplitude(img: np.ndarray, blank: np.ndarray, sx: float, sy: float, angle: float) -> float:
    """Spike-line amplitude across the ridge from (sx, sy) at ``angle``."""
    yy, xx = np.indices(img.shape[:2], dtype=np.float64)
    dx, dy = xx - sx, yy - sy
    along = dx * np.cos(angle) + dy * np.sin(angle)
    perp = np.abs(dy * np.cos(angle) - dx * np.sin(angle))
    ok = (along > 0) & ~blank
    on, off = img[ok & (perp < 4.0)], img[ok & (perp > 6.0) & (perp < 12.0)]
    if on.size < 20 or off.size < 20:
        return float("nan")
    # Light across the line (8 px wide band) above the local background: a
    # sharpened line of the same flux counts the same.
    return float((np.mean(on) - np.median(off)) * 8.0)


def measure(lr: np.ndarray, sr: np.ndarray, meta: dict, kind: str) -> dict:
    out = {}
    srb = binned(sr)
    for band, k in BANDS.items():
        blank = lr[..., k] == 0.0
        if kind == "centred":
            for lo, hi in RADII_ARCSEC:
                h_in, r_in = halo_and_ridge(lr[..., k], blank, meta["sx"], meta["sy"], lo, hi)
                h_sr, r_sr = halo_and_ridge(srb[..., k], blank, meta["sx"], meta["sy"], lo, hi)
                tag = f"{band}_{lo:g}-{hi:g}"
                out[f"halo_keep_{tag}"] = h_sr / h_in if h_in > 0 else float("nan")
                out[f"ridge_keep_{tag}"] = r_sr / r_in if r_in > 0 else float("nan")
        else:
            a_in = line_amplitude(lr[..., k], blank, meta["sx"], meta["sy"], meta["angle"])
            a_sr = line_amplitude(srb[..., k], blank, meta["sx"], meta["sy"], meta["angle"])
            out[f"line_keep_{band}"] = a_sr / a_in if a_in > 0 else float("nan")
            out[f"line_in_{band}"] = a_in
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("specs", nargs="+", help="model specs")
    p.add_argument("--inputs", required=True, help="bright_star_check_inputs.npz")
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=None, help="first N fields per set (smoke)")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    z = np.load(args.inputs)
    n = args.limit
    sets = {"centred": (z["centred"][:n], json.loads(str(z["centred_meta"]))[:n]),
            "offfield": (z["offfield"][:n], json.loads(str(z["offfield_meta"]))[:n])}
    path = os.path.join(args.out, "bright_star_check.json")
    results = json.load(open(path)) if os.path.exists(path) else {}
    runner = ModelRunner()
    for spec in args.specs:
        info = resolve(spec)
        rows, srs = {}, {}
        for kind, (lrs, metas) in sets.items():
            outs = []
            for i, (lr, meta) in enumerate(zip(lrs, metas, strict=True)):
                sr = runner.sr(info, lr)
                srs[f"{kind}_{i}"] = sr.astype(np.float16)
                outs.append(measure(lr, sr, meta, kind))
            rows[kind] = outs
            keys = sorted(outs[0])
            summary = {k: float(np.nanmedian([o[k] for o in outs])) for k in keys}
            rows[f"{kind}_median"] = summary
            shown = " ".join(f"{k}={v:.3f}" for k, v in summary.items() if "keep" in k)
            print(info.name, kind, shown, flush=True)
        np.savez_compressed(os.path.join(args.out, f"sr_{info.slug}.npz"), **srs)
        results[info.spec] = rows
        with open(path, "w") as fh:
            json.dump(results, fh, indent=1)
    runner.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
