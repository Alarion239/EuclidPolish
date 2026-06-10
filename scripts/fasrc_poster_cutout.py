#!/usr/bin/env python
"""Generate ONE random object as a clean 4-band Euclid cutout — for the poster.

Runs on a FASRC node (as the ``poster_cutout`` SLURM step card on the
Visualization tab) where the real COSMOS2025 master catalog lives. Picks a
single object of the requested kind, centred in the field, and renders the
*clean* HR sky (0.05″/pix, no PSF, no noise) in all four Euclid bands
(VIS, Y_E, J_E, H_E). No forward model is applied — this is the idealised
ground-truth object, the same clean scene the training generator produces
before convolution + noise.

Four modes (one object per mode, chosen at random):

  --mode sersic   a single analytic Sérsic bulge+disk galaxy (COSMOS row)
  --mode star     a single point source (PSF-free delta; fixed G-type colour)
  --mode lens     a gravitational lens system — SIE + shear deflection with a
                  real TNG50 deflector and lensed source, the same pure-TNG
                  lens model the main training pipeline uses (needs the TNG
                  atlas downloaded)
  --mode tng      a single real TNG50 SKIRT galaxy stamp

Outputs (under ``$EUCLID_POLISH_DATA_DIR/_poster/``, fixed names so each run
overwrites the previous result the WebUI then fetches):

  poster_cutout.fits   PrimaryHDU (OBJTYPE/SEED/… header) + one ImageHDU per
                       band (EXTNAME = VIS / Y_E / J_E / H_E), clean HR e⁻.
  poster_cutout.png    1×4 asinh-grayscale band montage (preview).

Usage
-----
    python scripts/fasrc_poster_cutout.py --mode lens --save
    python scripts/fasrc_poster_cutout.py --mode sersic --seed 7 --image-size 256
"""
from __future__ import annotations

import argparse
import io
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
from astropy.io import fits

# All imports at module scope (never function-scoped) — see project convention.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Make ``euclid_polish`` importable when run as a bare script (``python
# scripts/fasrc_poster_cutout.py``) and not just via ``python -m`` — the same
# bootstrap every other script under scripts/ uses.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from euclid_polish.config import Config
from euclid_polish.sky.cosmos2025 import ensure_prefiltered_catalog, open_cosmos2025
from euclid_polish.sky.multiband_generator import (
    MultiBandGeneratorConfig,
    MultiBandSimulator,
)
from euclid_polish.tng.properties import _fig_to_png

MODES = ("sersic", "star", "lens", "tng")
# Output band order matches the generator's channel order.
BAND_NAMES: Tuple[str, ...] = Config.LR_INPUT_BAND_NAMES  # ("VIS","Y_E","J_E","H_E")

OUTPUT_SUBDIR = "_poster"
FITS_NAME = "poster_cutout.fits"
PNG_NAME = "poster_cutout.png"

# ASCII label per mode (safe for FITS headers, which are ASCII-only).
MODE_LABEL = {
    "sersic": "Sersic galaxy",
    "star":   "Star (point source)",
    "lens":   "Gravitational lens",
    "tng":    "TNG50 galaxy",
}
# Pretty title per mode for the PNG montage (matplotlib renders unicode fine).
MODE_TITLE = {
    "sersic": "Sérsic galaxy",
    "star":   "Star (point source)",
    "lens":   "Gravitational lens",
    "tng":    "TNG50 galaxy",
}


# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------

def output_dir() -> str:
    return os.path.join(Config.DATA_DIR, OUTPUT_SUBDIR)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _counts_for_mode(mode: str) -> Dict[str, int]:
    """Explicit per-type source counts: exactly one object, nothing else."""
    base = dict(n_galaxies=0, n_stars=0, n_lenses=0, n_big=0)
    if mode in ("sersic", "tng"):
        base["n_galaxies"] = 1
    elif mode == "star":
        base["n_stars"] = 1
    elif mode == "lens":
        base["n_lenses"] = 1
    return base


# A poster lens must be *eye-visible*, unlike the honest training population
# (typical lenses hide their arcs deep inside the deflector light): require
# θ_E to clear this fraction of the deflector's visible (μ-truncated) radius,
# and the lensed source to have kept this much VIS flux after cosmological
# dimming — otherwise the cutout is just a bright galaxy with wings.
LENS_MIN_THETA_E_VISIBLE_FRAC = 0.5
LENS_MIN_SOURCE_VIS_E = 1000.0


def _lens_is_showable(rec: dict) -> bool:
    r_vis = rec.get("lens_visible_r_arcsec")
    src_e = rec.get("source_flux_vis_e")
    if r_vis is None or src_e is None:      # legacy/Sersic record → no check
        return True
    return (rec["theta_E_arcsec"] >= LENS_MIN_THETA_E_VISIBLE_FRAC * r_vis
            and src_e >= LENS_MIN_SOURCE_VIS_E)


def _record_ok(mode: str, meta: dict) -> bool:
    """Did the scene actually contain the requested object?

    A lens sample can fail (``_add_lens`` returns None on a RuntimeError) and a
    TNG slot silently falls back to Sérsic if its stamp can't load — so we don't
    just trust the requested count, we check the rendered records."""
    if mode == "star":
        return meta["n_stars"] == 1
    if mode == "lens":
        return (meta["n_lenses"] == 1
                and _lens_is_showable(meta["lenses"][0]))
    if mode == "sersic":
        gals = meta["galaxies"]
        return len(gals) == 1 and gals[0].get("render") == "sersic"
    if mode == "tng":
        gals = meta["galaxies"]
        return len(gals) == 1 and gals[0].get("render") == "tng"
    return False


def generate_cutout(
    mode: str, *, seed: int, image_size: int, max_tries: int = 16,
) -> Tuple[np.ndarray, dict, int]:
    """Render one centred random object's clean 4-band HR field.

    Returns ``(data, source_meta, used_seed)`` where ``data`` has shape
    ``(image_size, image_size, 4)`` in electrons and ``source_meta`` is the
    single object's parameter record from the generator metadata.
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; choose from {MODES}")

    # Pure-TNG mode (tng_fraction=1) makes every galaxy / lens-light / lensed
    # source a real TNG50 stamp — exactly how the main training pipeline runs
    # (fasrc_pipeline defaults tng_fraction=1.0). Both `tng` and `lens` modes
    # use it so the poster object matches the training scenes: in particular,
    # `lens` then goes through MultiBandSimulator._add_lens_pure (TNG deflector
    # + TNG lensed source, SIE+shear geometry), the same lens model the
    # pipeline produces — not the legacy analytic-Sérsic catalog path. The
    # `sersic`/`star` modes stay analytic (tng_fraction=0).
    pure_tng_mode = mode in ("tng", "lens")
    cfg = MultiBandGeneratorConfig(
        image_size=image_size,
        pixel_scale=Config.DEFAULT_PIXEL_SCALE,
        tng_fraction=(1.0 if pure_tng_mode else 0.0),
    )
    cat = open_cosmos2025(path=ensure_prefiltered_catalog(Config.COSMOS2025_CATALOG_PATH))
    sim = MultiBandSimulator(cat, cfg)
    if pure_tng_mode and not sim.tng_galaxies:
        raise RuntimeError(
            f"no downloaded TNG galaxies under {cfg.tng_galaxy_dir} — run the "
            f"TNG atlas download first ('{mode}' uses pure-TNG mode to match "
            "the training pipeline), or pick another mode.")

    # Centre the single object: _random_pix is the sole source of source
    # positions, so overriding it places every object at the field centre.
    centre = ((image_size - 1) / 2.0, (image_size - 1) / 2.0)
    sim._random_pix = lambda rng: centre  # type: ignore[method-assign]

    counts = _counts_for_mode(mode)
    seq = seed if seed >= 0 else int(np.random.SeedSequence().entropy % (2**32))
    if mode == "lens":
        max_tries *= 6      # the showability cut rejects most honest draws
    for attempt in range(max_tries):
        used = (seq + attempt) % (2**32)
        rng = np.random.default_rng(used)
        img, meta = sim.simulate_field(rng, **counts)
        if _record_ok(mode, meta):
            recs = {"sersic": meta["galaxies"], "tng": meta["galaxies"],
                    "star": meta["stars"], "lens": meta["lenses"]}[mode]
            return np.asarray(img.data, dtype=np.float32), recs[0], used
    raise RuntimeError(
        f"could not generate a '{mode}' object in {max_tries} tries "
        f"(seed base {seq}). Lens/TNG draws can fail; try a different seed.")


# ---------------------------------------------------------------------------
# FITS + PNG writers
# ---------------------------------------------------------------------------

def _header_value(v):
    """Coerce a metadata value into something FITS headers accept."""
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float, str)):
        return v
    if isinstance(v, (list, tuple)):
        return ",".join(f"{float(x):.6g}" for x in v)
    return str(v)


def build_cutout_hdul(
    data: np.ndarray, *, mode: str, seed: int, image_size: int,
    source_meta: dict,
) -> fits.HDUList:
    """PrimaryHDU (provenance header) + one ImageHDU per band (clean HR e⁻)."""
    primary = fits.PrimaryHDU()
    h = primary.header
    h["OBJTYPE"] = (mode, "poster cutout object kind")
    h["OBJLABEL"] = (MODE_LABEL[mode], "human-readable object kind")
    h["SEED"] = (int(seed), "RNG seed used for this scene")
    h["IMGSIZE"] = (int(image_size), "HR field side (pixels)")
    h["PIXSCALE"] = (float(Config.DEFAULT_PIXEL_SCALE), "arcsec/pixel (HR)")
    h["CLEAN"] = (True, "clean sky: no PSF, no noise")
    h["NBAND"] = (len(BAND_NAMES), "number of bands")
    # Stamp the object's own parameters (skip the per-band flux vectors that
    # already live on the per-band HDUs; keep scalar params).
    for k, v in source_meta.items():
        if k == "flux_e_per_band":
            continue
        key = str(k).upper().replace("_", "")[:8]
        try:
            h[key] = (_header_value(v), str(k))
        except Exception:
            pass

    hdul = fits.HDUList([primary])
    flux = source_meta.get("flux_e_per_band")
    for k, name in enumerate(BAND_NAMES):
        hdu = fits.ImageHDU(data=np.asarray(data[..., k], dtype=np.float32),
                            name=name)
        hdu.header["EXTNAME"] = name
        hdu.header["BAND"] = (name, "Euclid band")
        hdu.header["OBJTYPE"] = mode
        hdu.header["BUNIT"] = ("electron", "clean sky flux over the exposure stack")
        if flux is not None and k < len(flux):
            hdu.header["FLUXE"] = (float(flux[k]), "total source flux (electrons)")
        hdul.append(hdu)
    return hdul


def _grayscale_norm(arr: np.ndarray) -> np.ndarray:
    """Asinh stretch (scale = 90th pct of positive flux) + [0.5, 99.5] clip.

    Same house stretch as scripts/fasrc_tng_infographic.py so the poster
    cutouts match the rest of the TNG/infographic imagery."""
    d = np.clip(np.asarray(arr, dtype=np.float32), 0.0, None)
    pos = d[d > 0]
    scale = float(np.percentile(pos, 90)) if pos.size else 1.0
    s = np.arcsinh(d / max(scale, 1e-12))
    lo, hi = np.percentile(s, [0.5, 99.5])
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((s - lo) / (hi - lo), 0.0, 1.0)


def render_preview_png(
    data: np.ndarray, *, mode: str, seed: int, source_meta: dict,
) -> bytes:
    """1×4 band montage (asinh-grayscale), titled with the object kind."""
    n = len(BAND_NAMES)
    fig, axes = plt.subplots(1, n, figsize=(n * 2.4, 2.7))
    for k, (ax, name) in enumerate(zip(axes, BAND_NAMES)):
        ax.imshow(_grayscale_norm(data[..., k]), origin="lower", cmap="gray")
        ax.set_title(name, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
    # A short caption with the most telling parameter per mode.
    extra = ""
    if mode == "lens" and "theta_E_arcsec" in source_meta:
        extra = f" — θ_E = {source_meta['theta_E_arcsec']:.2f}″"
    elif mode == "star" and "mag_vis" in source_meta:
        extra = f" — VIS mag {source_meta['mag_vis']:.1f}"
    elif mode == "tng" and "subhalo_id" in source_meta:
        extra = f" — subhalo {source_meta['subhalo_id']}"
    fig.suptitle(f"{MODE_TITLE[mode]} (clean, seed {seed}){extra}",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _fig_to_png(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", required=True, choices=MODES,
                   help="Object kind to generate (one random object).")
    p.add_argument("--seed", type=int, default=-1,
                   help="RNG seed; -1 (default) → fresh random object each run.")
    p.add_argument("--image-size", type=int, default=Config.DEFAULT_IMAGE_SIZE,
                   help="HR field side in pixels (0.05\"/pix).")
    p.add_argument("--save", action="store_true",
                   help=f"Write {FITS_NAME} + {PNG_NAME} under "
                        f"$EUCLID_POLISH_DATA_DIR/{OUTPUT_SUBDIR}/. Without it, "
                        "the FITS bytes go to stdout.")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    size = max(16, int(args.image_size))
    print(f"[poster] mode={args.mode} seed={args.seed} image_size={size}")

    data, source_meta, used_seed = generate_cutout(
        args.mode, seed=args.seed, image_size=size)
    print(f"[poster] generated '{args.mode}' (used seed {used_seed}); "
          f"total flux = {float(data.sum()):.3e} e⁻")

    hdul = build_cutout_hdul(
        data, mode=args.mode, seed=used_seed, image_size=size,
        source_meta=source_meta)

    if not args.save:
        buf = io.BytesIO()
        hdul.writeto(buf, overwrite=True)
        sys.stdout.buffer.write(buf.getvalue())
        return 0

    out_dir = output_dir()
    os.makedirs(out_dir, exist_ok=True)
    fits_path = os.path.join(out_dir, FITS_NAME)
    png_path = os.path.join(out_dir, PNG_NAME)
    hdul.writeto(fits_path, overwrite=True)
    with open(png_path, "wb") as fh:
        fh.write(render_preview_png(
            data, mode=args.mode, seed=used_seed, source_meta=source_meta))
    print(f"[poster] wrote {fits_path}")
    print(f"[poster] wrote {png_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
