#!/usr/bin/env python
"""Render TNG50 SKIRT-atlas image infographics on FASRC for the WebUI.

Runs on a FASRC node (as the ``tng_grid`` / ``tng_stack`` SLURM jobs, or by
hand), where the downloaded galaxy FITS live. Like
``scripts/fasrc_inspect_tile.py`` it writes bytes to stdout, or — with
``--save`` — to the standard artifact path the WebUI then fetches.

  ``--mode grid --band {VIS|Y|J|H|RGB} --downsample {1|2|4} --seed S``
      A 5×5 PNG grid: 5 seeded-random galaxies (rows) × their 5 viewpoints
      (cols). Single Euclid band (asinh-grayscale) or an ``make_lupton_rgb``
      RGB from VIS+NISP. ``block_mean`` downsamples ×1/×2/×4.

  ``--mode stack --band B [--id N] [--seed S]``
      Bundle the 5 viewpoint frames of one band for a galaxy (random if no id)
      into one multi-extension FITS (PrimaryHDU + ImageHDU O1..O5) → stdout.

  ``--mode histograms``
      2×2 property panel (SFR / stellar mass / halo mass / effective radius).
      The WebUI renders this **locally** (no FITS needed — just the galaxy id
      list + the TNG API); this CLI mode is kept for debugging on the node.
      The property + plotting logic lives in ``euclid_polish.tng.properties``.

Galaxies are self-enumerated from the download tree — every
``<tng_dir>/<subhalo_id>/`` that holds a ``.done`` marker.
"""

from __future__ import annotations

import argparse
import io
import os
import random
import sys
import warnings

import matplotlib

matplotlib.use("Agg")                       # headless, before pyplot
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from euclid_polish.config import Config
from euclid_polish.observability.reporter import Reporter
from euclid_polish.skirt.image import block_mean, load_skirt_frame
from euclid_polish.sky.generation.tng_galaxy import (
    TNG_FITS_BANDS,
    tng_fits_path,
)
from euclid_polish.tng.properties import (
    _fig_to_png,
    load_api_key,
    placeholder_png,
    render_histograms_for_ids,
)

ORIENTATIONS: tuple[int, ...] = (1, 2, 3, 4, 5)
GRID_GALAXIES = 5
# RGB channels as FITS band tokens (R, G, B) = (H, J, VIS), matching
# Config.Color.RGB_SCHEMES["vis_nisp"] = (H_E, J_E, VIS).
RGB_FITS_BANDS: tuple[str, str, str] = ("H", "J", "VIS")
SINGLE_BANDS = tuple(TNG_FITS_BANDS)         # ("VIS", "Y", "J", "H")

# Where a SLURM job writes its rendered artifact (under the download root),
# keyed by mode. Fixed names → each job overwrites the previous result, which
# the WebUI then fetches via the matching /tng/result/<…> route.
INFOGRAPHIC_SUBDIR = "_infographics"
OUTPUT_NAMES = {
    "histograms": "histograms.png",
    "grid":       "grid.png",
    "stack":      "stack.fits",
}


def default_output_path(tng_dir: str, mode: str) -> str:
    """Standard artifact path for a job-rendered infographic of ``mode``."""
    return os.path.join(tng_dir, INFOGRAPHIC_SUBDIR, OUTPUT_NAMES[mode])


# ---------------------------------------------------------------------------
# Galaxy enumeration + selection
# ---------------------------------------------------------------------------

def list_downloaded_ids(tng_dir: str) -> list[str]:
    """Subhalo ids that finished downloading (their folder holds a .done)."""
    if not os.path.isdir(tng_dir):
        return []
    ids = [
        name for name in os.listdir(tng_dir)
        if os.path.isfile(os.path.join(tng_dir, name, Config.Tng.DONE_MARKER))
    ]
    try:
        return sorted(ids, key=int)
    except ValueError:
        return sorted(ids)


def pick_ids(ids: list[str], k: int, seed: int) -> list[str]:
    """Pick up to ``k`` ids. ``seed >= 0`` is reproducible; ``seed < 0`` draws a
    fresh random subset each call (so a re-submitted grid/stack job re-rolls)."""
    rng = random.Random(seed if seed >= 0 else None)
    pool = list(ids)
    if not pool:
        return []
    if len(pool) <= k:
        rng.shuffle(pool)
        return pool
    return rng.sample(pool, k)


def render_histograms(tng_dir: str, *, api_key: str = "",
                      max_workers: int = 16, reporter=None) -> bytes:
    """Enumerate the locally-downloaded galaxies, then plot (CLI/debug path —
    the WebUI renders histograms locally via euclid_polish.tng.properties)."""
    return render_histograms_for_ids(
        tng_dir, list_downloaded_ids(tng_dir), api_key,
        max_workers=max_workers, reporter=reporter)


# ---------------------------------------------------------------------------
# Image grid + stacked FITS (need the FITS pixels → these stay FASRC jobs)
# ---------------------------------------------------------------------------

def _grayscale_norm(arr: np.ndarray) -> np.ndarray:
    """Asinh stretch (scale = 90th pct of positive flux) + [0.5, 99.5] clip."""
    d = np.clip(np.asarray(arr, dtype=np.float32), 0.0, None)
    pos = d[d > 0]
    scale = float(np.percentile(pos, 90)) if pos.size else 1.0
    s = np.arcsinh(d / max(scale, 1e-12))
    lo, hi = np.percentile(s, [0.5, 99.5])
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((s - lo) / (hi - lo), 0.0, 1.0)


def _load_cell_band(gdir: str, gid: str, orient: int, fits_band: str,
                    downsample: int) -> np.ndarray | None:
    path = tng_fits_path(gdir, gid, orient, fits_band)
    if not os.path.isfile(path):
        return None
    arr = load_skirt_frame(path)
    if downsample > 1:
        arr = block_mean(arr, downsample)
    return arr


def render_cell(gdir: str, gid: str, orient: int, band: str,
                downsample: int) -> np.ndarray | None:
    """One panel: 2-D [0,1] grayscale, or (H,W,3) uint8 RGB, or None if missing."""
    if band == "RGB":
        chans = []
        for tok in RGB_FITS_BANDS:                 # H, J, VIS
            a = _load_cell_band(gdir, gid, orient, tok, downsample)
            if a is None:
                return None
            chans.append(np.clip(a, 0.0, None))
        r, g, b = chans
        from astropy.visualization import make_lupton_rgb
        ref = float(np.percentile(
            np.concatenate([r.ravel(), g.ravel(), b.ravel()]), 99.5))
        return make_lupton_rgb(r, g, b, Q=8, stretch=max(ref, 1e-12))
    a = _load_cell_band(gdir, gid, orient, band, downsample)
    return None if a is None else _grayscale_norm(a)


def render_grid(tng_dir: str, band: str, downsample: int, seed: int, *,
                ids: list[str] | None = None, note: str = "") -> bytes:
    # Explicit ids (chosen locally by the selection mode) win; otherwise pick a
    # seeded-random set on the node.
    chosen = (list(ids)[:GRID_GALAXIES] if ids
              else pick_ids(list_downloaded_ids(tng_dir), GRID_GALAXIES, seed))
    if not chosen:
        return placeholder_png("No galaxies downloaded yet.")
    nrows = len(chosen)
    fig, axes = plt.subplots(nrows, len(ORIENTATIONS),
                             figsize=(len(ORIENTATIONS) * 2.1, nrows * 2.1),
                             squeeze=False)
    for r, gid in enumerate(chosen):
        gdir = os.path.join(tng_dir, gid)
        for c, orient in enumerate(ORIENTATIONS):
            ax = axes[r][c]
            ax.set_xticks([]); ax.set_yticks([])
            img = render_cell(gdir, gid, orient, band, downsample)
            if img is None:
                ax.text(0.5, 0.5, "—", ha="center", va="center",
                        transform=ax.transAxes, color="#bbb")
            elif img.ndim == 2:
                ax.imshow(img, cmap="gray", origin="lower",
                          interpolation="nearest")
            else:
                ax.imshow(img, origin="lower", interpolation="nearest")
            if r == 0:
                ax.set_title(f"O{orient}", fontsize=10)
            if c == 0:
                ax.set_ylabel(f"TNG{gid}", fontsize=9)
    title = f"TNG50-1 — {band} — downsample ×{downsample}"
    if note:
        title += f"  ·  {note}"
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _fig_to_png(fig)


def build_stack_hdul(tng_dir: str, gid: str, band: str) -> fits.HDUList:
    """Multi-extension FITS: PrimaryHDU + one ImageHDU (O1..O5) per viewpoint.

    Original pixel data and per-frame headers are preserved verbatim.
    """
    gdir = os.path.join(tng_dir, gid)
    primary = fits.PrimaryHDU()
    primary.header["TNGID"] = (str(gid), "IllustrisTNG TNG50-1 subhalo id")
    primary.header["BAND"] = (band, "Euclid band of the stacked frames")
    primary.header["NORIENT"] = (len(ORIENTATIONS), "number of viewpoints")
    hdul = fits.HDUList([primary])
    for orient in ORIENTATIONS:
        path = tng_fits_path(gdir, gid, orient, band)
        if not os.path.isfile(path):
            continue
        with fits.open(path, memmap=False) as src:
            data = np.asarray(src[0].data)
            hdr = src[0].header.copy(strip=True)
        hdr["EXTNAME"] = f"O{orient}"
        hdr["TNGID"] = str(gid)
        hdr["ORIENT"] = (orient, "SKIRT viewpoint index")
        hdul.append(fits.ImageHDU(data=data, header=hdr, name=f"O{orient}"))
    if len(hdul) == 1:
        raise FileNotFoundError(
            f"no {band} frames found for galaxy {gid} in {gdir}")
    return hdul


def build_stack_bytes(tng_dir: str, gid: str, band: str) -> bytes:
    buf = io.BytesIO()
    build_stack_hdul(tng_dir, gid, band).writeto(buf, overwrite=True)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", required=True,
                   choices=("histograms", "grid", "stack"))
    p.add_argument("--tng-dir", default=Config.TNG_SKIRT_DIR,
                   help=f"Download root. Default: {Config.TNG_SKIRT_DIR}")
    p.add_argument("--band", default="VIS",
                   help="VIS|Y|J|H|RGB (grid) or VIS|Y|J|H (stack).")
    p.add_argument("--downsample", type=int, default=1, choices=(1, 2, 4))
    p.add_argument("--seed", type=int, default=0,
                   help="Seed for the random galaxy pick (grid / random stack).")
    p.add_argument("--ids", default="",
                   help="Comma-separated subhalo ids to render in the grid "
                        "(chosen locally by selection mode); overrides --seed.")
    p.add_argument("--note", default="",
                   help="Free-text note appended to the grid title (e.g. the "
                        "selection mode).")
    p.add_argument("--id", default="",
                   help="Subhalo id for stack mode (blank → seeded random).")
    p.add_argument("--workers", type=int, default=16,
                   help="Concurrent TNG-API requests for the histogram fetch.")
    p.add_argument("--api-key-file", default=Config.Tng.API_KEY_FILE)
    # Output target. Default (neither given) → bytes to stdout (interactive).
    # SLURM jobs pass --save to write the standard artifact path that the
    # WebUI's /tng/result/<…> route then fetches; --out overrides the path.
    p.add_argument("--out", default="",
                   help="Write the rendered bytes to this file instead of stdout.")
    p.add_argument("--save", action="store_true",
                   help="Write to the standard artifact path for this mode "
                        "(<tng-dir>/_infographics/<name>) — used by the jobs.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    # Binary stdout — mute every chatty warning so the PNG/FITS stays clean.
    warnings.filterwarnings("ignore")
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    args = parse_args(argv)
    reporter = Reporter.from_env()

    if args.mode == "histograms":
        key = load_api_key(args.api_key_file)
        out = render_histograms(args.tng_dir, api_key=key,
                                max_workers=args.workers, reporter=reporter)
    elif args.mode == "grid":
        band = args.band.upper()
        if band not in (*SINGLE_BANDS, "RGB"):
            sys.stderr.write(f"bad band {band!r}\n")
            return 2
        ids = [s.strip() for s in args.ids.split(",") if s.strip()] or None
        out = render_grid(args.tng_dir, band, args.downsample, args.seed,
                          ids=ids, note=args.note)
    elif args.mode == "stack":
        band = args.band.upper()
        if band not in SINGLE_BANDS:
            sys.stderr.write(f"bad stack band {band!r}\n")
            return 2
        gid = args.id.strip()
        if not gid:
            picked = pick_ids(list_downloaded_ids(args.tng_dir), 1, args.seed)
            if not picked:
                sys.stderr.write("no downloaded galaxies\n")
                return 3
            gid = picked[0]
        try:
            out = build_stack_bytes(args.tng_dir, gid, band)
        except FileNotFoundError as e:
            sys.stderr.write(f"{e}\n")
            return 4
    else:
        sys.stderr.write(f"unknown mode {args.mode!r}\n")
        return 1

    out_path = args.out or (default_output_path(args.tng_dir, args.mode)
                            if args.save else "")
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "wb") as f:
            f.write(out)
        sys.stderr.write(f"wrote {len(out)} bytes → {out_path}\n")
    else:
        sys.stdout.buffer.write(out)
        sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
