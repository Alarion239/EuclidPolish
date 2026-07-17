#!/usr/bin/env python
"""Generate ONE random object as a clean 4-band Euclid cutout — for the poster.

Runs on a FASRC node (as the ``poster_cutout`` SLURM step card on the
Visualization tab) where the real COSMOS2025 master catalog lives. Picks a
single object of the requested kind, centred in the field, and renders the
*clean* HR sky (0.05″/pix, no PSF, no noise) in all four Euclid bands
(VIS, Y_E, J_E, H_E). No forward model is applied — this is the idealised
ground-truth object, the same clean scene the training generator produces
before convolution + noise.

Five modes (one object per mode, chosen at random — except ``field``):

  --mode sersic   a single analytic Sérsic bulge+disk galaxy (COSMOS row)
  --mode star     a single point source (PSF-free delta; median stellar colour)
  --mode lens     a gravitational lens system — SIE + shear deflection with a
                  real TNG50 deflector and lensed source, the same pure-TNG
                  lens model the main training pipeline uses (needs the TNG
                  atlas downloaded)
  --mode tng      a single real TNG50 SKIRT galaxy stamp
  --mode field    a full random field — Poisson source counts, sources at
                  random positions: exactly what the main training pipeline
                  generates (pure-TNG mode). Additionally forward-models the
                  clean scene to a noisy mock-Euclid stack and reconstructs
                  it with the trained model from ``--ckpt-dir``.

Outputs (under ``$EUCLID_POLISH_DATA_DIR/_poster/``, fixed names so each run
overwrites the previous result the WebUI then fetches):

  poster_cutout.fits   PrimaryHDU (OBJTYPE/SEED/… header) + one ImageHDU per
                       band (EXTNAME = VIS/Y_E/J_E/H_E), clean HR e⁻.
                       Field mode stacks the whole triplet into this ONE
                       file: CLEAN_VIS…CLEAN_H_E (0.05″), DIRTY_VIS…DIRTY_H_E
                       (0.10″, noisy), and SR (0.05″, checkpoint in header) —
                       so a single download carries everything.
  poster_cutout.png    preview montage (field: clean | dirty | SR VIS).

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

# All imports at module scope (never function-scoped) — see project convention.
import matplotlib
import numpy as np
from astropy.io import fits

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Make ``euclid_polish`` importable when run as a bare script (``python
# scripts/fasrc_poster_cutout.py``) and not just via ``python -m`` — the same
# bootstrap every other script under scripts/ uses.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import contextlib

from euclid_polish.config import Config
from euclid_polish.image import Image
from euclid_polish.psf.psf_library import load_all_band_psf_sets
from euclid_polish.sky.generation.cosmos2025 import ensure_prefiltered_catalog, open_cosmos2025
from euclid_polish.sky.generation.sky_simulator import (
    SkySimulator,
    SkySimulatorConfig,
)
from euclid_polish.sky.observation.observation_simulator import (
    ObservationSimulator,
    ObservationSimulatorConfig,
)
from euclid_polish.tng.properties import _fig_to_png
from euclid_polish.training.inference import (
    load_model_from_checkpoint,
    reconstruct,
)
from euclid_polish.visualization.color import eye_rgb

MODES = ("sersic", "star", "lens", "tng", "field")
# Output band order matches the generator's channel order.
BAND_NAMES: tuple[str, ...] = Config.LR_INPUT_BAND_NAMES  # ("VIS","Y_E","J_E","H_E")

OUTPUT_SUBDIR = "_poster"
FITS_NAME = "poster_cutout.fits"
PNG_NAME = "poster_cutout.png"
# Field-mode star density (per arcmin²): a showcase boost so a poster field
# carries a few point sources (~2 per 510 px field). The TRAINING default
# stays at the real-sky 1.389/arcmin² — that value is pinned for a
# controlled comparison (see Config.DEFAULT_STAR_DENSITY_ARCMIN2's note);
# changing it there is an experiment decision, not a poster one.
FIELD_STAR_DENSITY_ARCMIN2 = 12.0

# ASCII label per mode (safe for FITS headers, which are ASCII-only).
MODE_LABEL = {
    "sersic": "Sersic galaxy",
    "star":   "Star (point source)",
    "lens":   "Gravitational lens",
    "tng":    "TNG50 galaxy",
    "field":  "Random field (clean+dirty+SR)",
}
# Pretty title per mode for the PNG montage (matplotlib renders unicode fine).
MODE_TITLE = {
    "sersic": "Sérsic galaxy",
    "star":   "Star (point source)",
    "lens":   "Gravitational lens",
    "tng":    "TNG50 galaxy",
    "field":  "Random field",
}


# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------

def output_dir() -> str:
    return os.path.join(Config.DATA_DIR, OUTPUT_SUBDIR)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _counts_for_mode(mode: str) -> dict[str, int]:
    """Explicit per-type source counts: exactly one object, nothing else.

    ``field`` overrides nothing — the simulator draws its Poisson counts
    exactly as the main training pipeline does."""
    if mode == "field":
        return {}
    base = {"n_sersic": 0, "n_tng": 0, "n_stars": 0, "n_lenses": 0}
    if mode == "sersic":
        base["n_sersic"] = 1
    elif mode == "tng":
        base["n_tng"] = 1
    elif mode == "star":
        base["n_stars"] = 1
    elif mode == "lens":
        base["n_lenses"] = 1
    return base


# A poster lens must be *eye-visible*, unlike the honest training population
# (typical lenses hide their arcs deep inside the deflector light). The
# generator rejects unshowable systems ANALYTICALLY before rendering
# (cfg.lens_require_showable → cached native-photometry predictors); the
# check below re-verifies the RENDERED record as a backstop, since the
# predictors approximate (mean VIS profile, deterministic drift).
LENS_MIN_THETA_E_VISIBLE_FRAC = Config.LENS_SHOWABLE_THETA_E_FRAC
LENS_MIN_SOURCE_VIS_E = Config.LENS_SHOWABLE_MIN_SRC_VIS_E


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
    if mode == "field":
        # Poisson counts can come up all-zero on a tiny field; ask for at
        # least one rendered source so the preview isn't a blank frame.
        return (meta["n_galaxies"] + meta["n_stars"] + meta["n_lenses"]) > 0
    if mode == "sersic":
        gals = meta["galaxies"]
        return len(gals) == 1 and gals[0].get("render") == "sersic"
    if mode == "tng":
        gals = meta["galaxies"]
        return len(gals) == 1 and gals[0].get("render") == "tng"
    return False


def generate_cutout(
    mode: str, *, seed: int, image_size: int, max_tries: int = 16,
) -> tuple[np.ndarray, dict, int]:
    """Render one centred random object's clean 4-band HR field.

    Returns ``(data, source_meta, used_seed)`` where ``data`` has shape
    ``(image_size, image_size, 4)`` in electrons and ``source_meta`` is the
    single object's parameter record from the generator metadata.
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; choose from {MODES}")

    # TNG-stamp modes (tng, lens, field) use real TNG50 stamps matching the
    # training pipeline: no Sersic catalog needed for lens/tng/field.
    # `sersic`/`star` modes stay all-analytic.
    tng_mode = mode in ("tng", "lens", "field")
    cfg = SkySimulatorConfig(
        image_size=image_size,
        pixel_scale=Config.DEFAULT_PIXEL_SCALE,
        sersic_density_arcmin2=(0.0 if tng_mode
                                else Config.DEFAULT_GAL_DENSITY_ARCMIN2),
        tng_density_arcmin2=(Config.DEFAULT_GAL_DENSITY_ARCMIN2 if tng_mode
                             else 0.0),
        # Poster lenses must be eye-visible: rejected analytically inside
        # _add_lens_pure before any stamp is rendered.
        lens_require_showable=(mode == "lens"),
        # Field showcase: a few stars per field (training keeps its own
        # pinned density).
        star_density_arcmin2=(FIELD_STAR_DENSITY_ARCMIN2 if mode == "field"
                              else Config.DEFAULT_STAR_DENSITY_ARCMIN2),
    )
    # TNG modes render nothing Sersic, so COSMOS is not needed.
    cat = None if tng_mode else open_cosmos2025(
        path=ensure_prefiltered_catalog(Config.COSMOS2025_CATALOG_PATH))
    sim = SkySimulator(cat, cfg)
    if tng_mode and not sim.tng_galaxies:
        raise RuntimeError(
            f"no downloaded TNG galaxies under {cfg.tng_galaxy_dir} — run the "
            f"TNG atlas download first ('{mode}' uses TNG stamps to match "
            "the training pipeline), or pick another mode.")

    # Centre the single object: _random_pix is the sole source of source
    # positions, so overriding it places every object at the field centre.
    # A full field keeps the simulator's own random placement.
    if mode != "field":
        centre = ((image_size - 1) / 2.0, (image_size - 1) / 2.0)
        sim._random_pix = lambda rng: centre  # type: ignore[method-assign]

    counts = _counts_for_mode(mode)
    seq = seed if seed >= 0 else int(np.random.SeedSequence().entropy % (2**32))
    if mode == "lens":
        # The analytic pre-rejection handles most of the selection inside
        # _add_lens_pure; whole-field retries remain only for the rare
        # post-render backstop failures.
        max_tries *= 3
    for attempt in range(max_tries):
        used = (seq + attempt) % (2**32)
        rng = np.random.default_rng(used)
        img, meta = sim.simulate_field(rng, **counts)
        if _record_ok(mode, meta):
            if mode == "field":
                # Whole-field summary instead of a single object's record.
                rec = {k: meta[k] for k in
                       ("field_area_arcmin2", "n_galaxies",
                        "n_stars", "n_lenses")}
            else:
                rec = {"sersic": meta["galaxies"], "tng": meta["galaxies"],
                       "star": meta["stars"], "lens": meta["lenses"]}[mode][0]
            return np.asarray(img.data, dtype=np.float32), rec, used
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
        with contextlib.suppress(Exception):
            h[key] = (_header_value(v), str(k))

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


def forward_and_reconstruct(
    clean_4ch: np.ndarray, seed: int, *, psf_dir: str, ckpt_dir: str,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Field mode's second half: clean HR → noisy mock-Euclid LR → SR.

    The same forward model and inference path the training pipeline uses
    (per-band ePSF convolution, sum-rebin to 0.10″, Poisson/read noise and
    artifacts; then the latest checkpoint in ``ckpt_dir``). Returns
    ``(dirty_lr_4ch, sr_hr, checkpoint_name)`` — ``sr_hr`` is the 4-band
    SR cube for a VIS+NISP-output checkpoint, or the 2-D VIS plane for a
    legacy single-output one.
    """
    psf_sets = load_all_band_psf_sets(
        psf_dir=psf_dir, require_empirical=False,
        target_pixel_scale=Config.DEFAULT_PIXEL_SCALE)
    fwd = ObservationSimulator(psf_sets_by_band=psf_sets,
                           config=ObservationSimulatorConfig(add_noise=True))
    hr = Image(
        data=np.asarray(clean_4ch, dtype=np.float32),
        pixel_scale_arcsec=Config.DEFAULT_PIXEL_SCALE,
        band_names=BAND_NAMES, is_clean=True)
    # Noise RNG decoupled from the scene seed (+1) so re-rolling the scene
    # never reuses a noise stream.
    lr, _hr_vis = fwd.process(hr, np.random.default_rng(seed + 1))

    model = load_model_from_checkpoint(
        ckpt_dir, Config.DEFAULT_REBIN_FACTOR, Config.DEFAULT_NUM_RES_BLOCKS,
        nchan_out=Config.NUM_HR_CHANNELS)
    _lr_vis, sr = reconstruct(model, np.asarray(lr.data, dtype=np.float32))
    # Checkpoint name for provenance, from the TF checkpoint state file
    # (avoids importing tensorflow at this level just for the name).
    ckpt_name = os.path.basename(ckpt_dir)
    state = os.path.join(ckpt_dir, "checkpoint")
    if os.path.isfile(state):
        with open(state, encoding="utf-8") as fh:
            first = fh.readline()
        if '"' in first:
            ckpt_name = first.split('"')[1]
    return (np.asarray(lr.data, dtype=np.float32),
            np.asarray(sr, dtype=np.float32), ckpt_name)


def append_field_companions(
    hdul: fits.HDUList, dirty: np.ndarray, sr: np.ndarray,
    *, ckpt_name: str,
) -> None:
    """Stack the dirty LR bands and the SR image into the field FITS.

    One file holds the whole triplet (CLEAN_* / DIRTY_* / SR extensions),
    so the WebUI's single "pull latest FITS" fetches everything. The clean
    band HDUs are renamed with a CLEAN_ prefix for symmetry.
    """
    for name in BAND_NAMES:                       # VIS → CLEAN_VIS, …
        hdul[name].name = f"CLEAN_{name}"
    for k, name in enumerate(BAND_NAMES):
        hdu = fits.ImageHDU(data=np.asarray(dirty[..., k], dtype=np.float32),
                            name=f"DIRTY_{name}")
        hdu.header["BAND"] = (name, "Euclid band")
        hdu.header["PIXSCALE"] = (0.10, "arcsec/pixel (LR archive grid)")
        hdu.header["BUNIT"] = ("electron", "noisy LR flux over the stack")
        hdul.append(hdu)
    sr = np.asarray(sr, dtype=np.float32)
    if sr.ndim == 3:
        # 4-band SR cube → one SR_<band> extension per band, mirroring
        # the DIRTY_<band> convention.
        for k, name in enumerate(BAND_NAMES):
            sp = fits.ImageHDU(data=np.ascontiguousarray(sr[..., k]),
                               name=f"SR_{name}")
            sp.header["BAND"] = (name, "Euclid band")
            sp.header["PIXSCALE"] = (float(Config.DEFAULT_PIXEL_SCALE),
                                     "arcsec/pixel (HR grid)")
            sp.header["CKPT"] = (ckpt_name, "checkpoint used for the reconstruction")
            sp.header["BUNIT"] = ("electron", "super-resolved sky")
            hdul.append(sp)
        return
    sp = fits.ImageHDU(data=sr, name="SR")
    sp.header["PIXSCALE"] = (float(Config.DEFAULT_PIXEL_SCALE),
                             "arcsec/pixel (HR grid)")
    sp.header["CKPT"] = (ckpt_name, "checkpoint used for the reconstruction")
    sp.header["BUNIT"] = ("electron", "super-resolved VIS sky")
    hdul.append(sp)


def render_field_preview_png(
    clean: np.ndarray, dirty: np.ndarray, sr: np.ndarray,
    *, seed: int, source_meta: dict,
) -> bytes:
    """Field mode preview: clean VIS | mock-Euclid VIS | super-resolved.

    A 4-band SR cube gets two panels — the VIS plane plus the physical
    "eye" color composite (per-pixel blackbody T → Planckian-locus hue;
    the model emits all bands now, so SR has color).
    """
    sr = np.asarray(sr, dtype=np.float32)
    sr_cube = sr if sr.ndim == 3 else None
    sr_vis  = sr[..., 0] if sr_cube is not None else sr
    panels = [("clean VIS (0.05″/pix)", clean[..., 0]),
              ("mock Euclid VIS (0.10″/pix)", dirty[..., 0]),
              ("super-resolved VIS (0.05″/pix)", sr_vis)]
    if sr_cube is not None:
        panels.append(("super-resolved eye color (4-band)", None))
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(n * 3.0, 3.3))
    for ax, (title, im) in zip(np.atleast_1d(axes), panels, strict=False):
        if im is None:
            rgb = eye_rgb(sr_cube, band_names=BAND_NAMES, stretch="asinh",
                          asinh_scale_e=float(Config.STRETCH_SCALE_E))
            ax.imshow(np.clip(rgb, 0.0, 1.0), origin="lower")
        else:
            ax.imshow(_grayscale_norm(im), origin="lower", cmap="gray")
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(
        f"Random field (seed {seed}) — {source_meta.get('n_galaxies', 0)} gal, "
        f"{source_meta.get('n_stars', 0)} stars, "
        f"{source_meta.get('n_lenses', 0)} lenses", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _fig_to_png(fig)


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
    for k, (ax, name) in enumerate(zip(axes, BAND_NAMES, strict=False)):
        ax.imshow(_grayscale_norm(data[..., k]), origin="lower", cmap="gray")
        ax.set_title(name, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
    # A short caption with the most telling parameter per mode.
    extra = ""
    if mode == "lens" and "theta_E_arcsec" in source_meta:
        extra = f" — θ_E = {source_meta['theta_E_arcsec']:.2f}″"
    elif mode == "field":
        extra = (f" — {source_meta.get('n_galaxies', 0)} gal, "
                 f"{source_meta.get('n_stars', 0)} stars, "
                 f"{source_meta.get('n_lenses', 0)} lenses")
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

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
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
    p.add_argument("--ckpt-dir", default=Config.DEFAULT_CHECKPOINT_DIR,
                   help="Checkpoint dir for the field-mode reconstruction "
                        "(default: $EUCLID_POLISH_CKPT_DIR / ./ckpt/wdsr).")
    p.add_argument("--psf-dir", default=Config.EUCLID_PSF_DIR,
                   help="Per-band ePSF dir for the field-mode forward model "
                        "(Gaussian fallback where a band's file is missing).")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
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

    # Field mode: forward-model + reconstruct, so the result is the full
    # clean / dirty / SR triplet the pipeline trains on.
    dirty = sr = None
    if args.mode == "field":
        print(f"[poster] forward-modelling + reconstructing "
              f"(ckpt dir: {args.ckpt_dir}) …")
        dirty, sr, ckpt_name = forward_and_reconstruct(
            data, used_seed, psf_dir=args.psf_dir, ckpt_dir=args.ckpt_dir)
        print(f"[poster] reconstruction used checkpoint {ckpt_name}")
        append_field_companions(hdul, dirty, sr, ckpt_name=ckpt_name)

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
    print(f"[poster] wrote {fits_path}"
          + (" (CLEAN_*/DIRTY_*/SR extensions)" if args.mode == "field" else ""))
    if args.mode == "field":
        png = render_field_preview_png(
            data, dirty, sr, seed=used_seed, source_meta=source_meta)
    else:
        png = render_preview_png(
            data, mode=args.mode, seed=used_seed, source_meta=source_meta)
    with open(png_path, "wb") as fh:
        fh.write(png)
    print(f"[poster] wrote {png_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
