#!/usr/bin/env python
"""Generate ONE random object as a clean 4-band Euclid cutout — for the poster.

Runs on a FASRC node (as the ``poster_cutout`` SLURM step — Runs › Steps, also
embedded in the Figures poster plate) where the downloaded TNG50 SKIRT atlas
lives. Picks a single object of the requested kind, centred in the field, and
renders the *clean* HR sky (0.05″/pix, no PSF, no noise) in all four Euclid
bands (VIS, Y_E, J_E, H_E). No forward model is applied — this is the
idealised ground-truth object: the starfull HR target the training generator
builds (its starless scene plus the scene's point sources).

Galaxies and lenses draw from the activated Euclid joint galaxy-population
artifact and stars from the activated Gaia+Euclid stellar prior — the
populations synthetic generation uses. The console freezes both at submit and
stages them as JSON files beside the job script
(``--joint-galaxy-population-file`` / ``--star-prior-file``); a direct run
without them reads the artifacts activated under ``$EUCLID_POLISH_DATA_DIR``.

Four modes (one object per mode, chosen at random — except ``field``):

  --mode star     a single point source (PSF-free delta; magnitude and colour
                  drawn from the stellar prior)
  --mode lens     a gravitational lens system — SIE + shear deflection with a
                  real TNG50 deflector and lensed source, the same pure-TNG
                  lens model the main training pipeline uses (needs the TNG
                  atlas downloaded)
  --mode tng      a single real TNG50 SKIRT galaxy stamp
  --mode field    a full random field — Poisson source counts at the galaxy
                  and star densities of the two artifacts (the default lens
                  density), sources at random positions, as synthetic
                  generation renders a validate/test field. Additionally
                  forward-models the scene to a noisy mock-Euclid stack (stars
                  carried as the separate star plane, as for validate/test)
                  and reconstructs it with the production model: the spatial
                  gate over the STARFULL ensemble members it reads, or their
                  plain mean (logged) when no current gate loads.

Outputs (under ``$EUCLID_POLISH_DATA_DIR/_poster/``, fixed names so each run
overwrites the previous result the WebUI then fetches):

  poster_cutout.fits   PrimaryHDU (OBJTYPE/SEED/… header) + one ImageHDU per
                       band (EXTNAME = VIS/Y_E/J_E/H_E), clean HR e⁻.
                       Field mode stacks the whole triplet into this ONE
                       file: CLEAN_VIS…CLEAN_H_E (0.05″), DIRTY_VIS…DIRTY_H_E
                       (0.10″, noisy), and SR_VIS…SR_H_E (0.05″; each header
                       names the model — combiner and members) — so a single
                       download carries everything.
  poster_cutout.png    preview montage (field: clean | dirty | SR VIS | SR
                       eye colour).

Usage
-----
    python scripts/fasrc_poster_cutout.py --mode lens --save
    python scripts/fasrc_poster_cutout.py --mode tng --seed 7 --image-size 256
"""
from __future__ import annotations

import argparse
import io
import json
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
from euclid_polish.eval.catalog_runner import eval_model_identity
from euclid_polish.eval.ensemble_infer import load_eval_ensemble, sr_from_model
from euclid_polish.image import Image
from euclid_polish.psf.psf_library import load_all_band_psf_sets
from euclid_polish.sky.generation.phz_galaxy_prior import (
    population_prior_from_payload,
)
from euclid_polish.sky.generation.sky_simulator import (
    SkySimulator,
    SkySimulatorConfig,
    _deposit_star,
    star_band_magnitudes_from_record,
)
from euclid_polish.sky.observation.observation_simulator import (
    ObservationSimulator,
    ObservationSimulatorConfig,
)
from euclid_polish.tng.properties import _fig_to_png
from euclid_polish.visualization.color import eye_rgb
from euclid_polish.web.helpers.population_calibration import (
    active_star,
    joint_galaxy_state,
)

MODES = ("star", "lens", "tng", "field")
# Modes that draw from the stellar prior / the galaxy-population artifact
# (the TNG-backed ones: galaxies, lenses).
STAR_MODES = ("star", "field")
GALAXY_MODES = ("lens", "tng", "field")
# Output band order matches the generator's channel order.
BAND_NAMES: tuple[str, ...] = Config.LR_INPUT_BAND_NAMES  # ("VIS","Y_E","J_E","H_E")

OUTPUT_SUBDIR = "_poster"
FITS_NAME = "poster_cutout.fits"
PNG_NAME = "poster_cutout.png"

# ASCII label per mode (safe for FITS headers, which are ASCII-only).
MODE_LABEL = {
    "star":   "Star (point source)",
    "lens":   "Gravitational lens",
    "tng":    "TNG50 galaxy",
    "field":  "Random field (clean+dirty+SR)",
}
# Pretty title per mode for the PNG montage (matplotlib renders unicode fine).
MODE_TITLE = {
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

def galaxy_density_arcmin2(payload: dict) -> float:
    """The galaxy-population artifact's generation surface density — the
    density synthetic generation renders (``SyntheticGenerateStep`` and
    ``run_pipeline._freeze_generation_population`` read the same key)."""
    try:
        density = float(payload["generation"]["surface_density_arcmin2"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "galaxy population artifact has no finite surface density"
        ) from exc
    if not np.isfinite(density) or density <= 0.0:
        raise ValueError(
            "galaxy population artifact has no finite surface density")
    return density


def star_density_arcmin2(payload: dict) -> float:
    """The stellar prior's surface density — what synthetic generation uses
    (the generator default when the artifact carries none)."""
    density = (payload.get("population") or {}).get("density_arcmin2")
    return (float(density) if density is not None
            else Config.DEFAULT_STAR_DENSITY_ARCMIN2)


def star_plane(shape: tuple[int, ...], stars: list[dict]) -> np.ndarray | None:
    """The scene's drawn stars as a sparse HR plane, or ``None`` without stars.

    The generator renders a STARLESS scene and only records its stars; this
    deposits them the way ``run_pipeline._forward_with_stars`` does for the
    validate/test fields (nearest HR pixel, four-band magnitudes)."""
    if not stars:
        return None
    plane = np.zeros(shape, dtype=np.float32)
    for s in stars:
        _deposit_star(plane, float(s["x_pix"]), float(s["y_pix"]),
                      float(s["mag_vis"]),
                      band_magnitudes=star_band_magnitudes_from_record(s))
    return plane


def _counts_for_mode(mode: str) -> dict[str, int]:
    """Explicit per-type source counts: exactly one object, nothing else.

    ``field`` overrides nothing — the simulator draws its Poisson counts at
    the configured densities, as synthetic generation does."""
    if mode == "field":
        return {}
    base = {"n_galaxies": 0, "n_stars": 0, "n_lenses": 0}
    if mode == "tng":
        base["n_galaxies"] = 1
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
    if r_vis is None or src_e is None:
        return True
    return (rec["theta_E_arcsec"] >= LENS_MIN_THETA_E_VISIBLE_FRAC * r_vis
            and src_e >= LENS_MIN_SOURCE_VIS_E)


def _record_ok(mode: str, meta: dict) -> bool:
    """Did the scene actually contain the requested object?

    A lens sample can fail (``_add_lens`` returns ``None`` on a runtime error),
    so do not trust the requested count alone."""
    if mode == "star":
        return meta["n_stars"] == 1
    if mode == "lens":
        return (meta["n_lenses"] == 1
                and _lens_is_showable(meta["lenses"][0]))
    if mode == "field":
        # Poisson counts can come up all-zero on a tiny field; ask for at
        # least one rendered source so the preview isn't a blank frame.
        return (meta["n_galaxies"] + meta["n_stars"] + meta["n_lenses"]) > 0
    if mode == "tng":
        gals = meta["galaxies"]
        return len(gals) == 1 and gals[0].get("render") == "tng"
    return False


def generate_cutout(
    mode: str,
    *,
    seed: int,
    image_size: int,
    max_tries: int = 16,
    star_prior_payload: dict | None = None,
    galaxy_population_payload: dict | None = None,
) -> tuple[np.ndarray, np.ndarray | None, dict, int]:
    """Render one centred random object's clean 4-band HR field.

    Returns ``(scene, stars, source_meta, used_seed)``: ``scene`` is the
    STARLESS ``(image_size, image_size, 4)`` field in electrons (galaxies +
    lenses), ``stars`` the scene's point sources as a separate HR plane of the
    same shape (``None`` without stars) — the clean sky is ``scene + stars``
    — and ``source_meta`` the single object's parameter record from the
    generator metadata (a whole-field summary in field mode).
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; choose from {MODES}")
    if mode in STAR_MODES and star_prior_payload is None:
        raise ValueError(
            f"'{mode}' mode requires the activated empirical stellar prior"
        )
    if mode in GALAXY_MODES and galaxy_population_payload is None:
        raise ValueError(
            f"'{mode}' mode requires the activated galaxy-population artifact"
        )

    field = mode == "field"
    cfg = SkySimulatorConfig(
        image_size=image_size,
        pixel_scale=Config.DEFAULT_PIXEL_SCALE,
        # A field draws Poisson counts at the artifacts' densities, as
        # synthetic generation does; the single-object modes pass explicit
        # counts instead, so their densities stay zero (and the star mode
        # never needs the TNG atlas).
        galaxy_density_arcmin2=(
            galaxy_density_arcmin2(galaxy_population_payload) if field else 0.0
        ),
        star_density_arcmin2=(
            star_density_arcmin2(star_prior_payload) if field else 0.0
        ),
        lens_density_arcmin2=Config.LENS_DENSITY_ARCMIN2 if field else 0.0,
        # Poster lenses must be eye-visible: rejected analytically inside
        # _add_lens_pure before any stamp is rendered.
        lens_require_showable=(mode == "lens"),
        star_prior_payload=star_prior_payload,
    )
    # TNG-stamp modes (tng, lens, field) use real TNG50 stamps with the
    # artifact's photometry and colours, matching the training pipeline.
    prior = (
        population_prior_from_payload(galaxy_population_payload)
        if mode in GALAXY_MODES else None
    )
    sim = SkySimulator(prior, cfg)
    if mode in GALAXY_MODES and not sim.tng_atlas:
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
                       ("field_area_arcmin2", "galaxy_density_arcmin2",
                        "star_density_arcmin2", "n_galaxies",
                        "n_stars", "n_lenses")}
            else:
                rec = {
                    "tng": meta["galaxies"],
                    "star": meta["stars"],
                    "lens": meta["lenses"],
                }[mode][0]
            scene = np.asarray(img.data, dtype=np.float32)
            return scene, star_plane(scene.shape, meta["stars"]), rec, used
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
    scene: np.ndarray, stars: np.ndarray | None, seed: int, *, psf_dir: str,
    ensemble_dir: str | None = None, num_res_blocks: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict, str]:
    """Field mode's second half: clean HR → noisy mock-Euclid LR → SR.

    The forward synthetic generation runs for a validate/test field: per-band
    ePSF convolution (training PSF-warp and saturation-mask defaults),
    sum-rebin to 0.10″, Poisson/read noise and artifacts, with the stars
    passed as the separate ``star_hr_4ch`` plane so saturation tells star
    cores from galaxy light. The SR is the production model
    (:func:`load_eval_ensemble` + :func:`sr_from_model`): the spatial gate
    over the STARFULL members it reads, or their plain mean — logged — when
    no current gate loads. Returns ``(dirty_lr_4ch, sr_hr_4ch, identity,
    model_label)``; ``identity`` is :func:`eval_model_identity`'s
    ``{member_labels, combiner_kind, combiner_fingerprint, run_labels}``.
    """
    psf_sets = load_all_band_psf_sets(
        psf_dir=psf_dir, require_empirical=False,
        target_pixel_scale=Config.DEFAULT_PIXEL_SCALE)
    fwd = ObservationSimulator(
        psf_sets_by_band=psf_sets,
        config=ObservationSimulatorConfig(
            add_noise=True,
            psf_warp_prob=Config.TRAIN_PSF_WARP_PROB,
            psf_warp_alpha_max=Config.TRAIN_PSF_WARP_ALPHA_MAX,
            psf_warp_sigma=Config.TRAIN_PSF_WARP_SIGMA,
            saturation_mask_prob=Config.TRAIN_SATURATION_MASK_PROB,
        ))
    hr = Image(
        data=np.asarray(scene, dtype=np.float32),
        pixel_scale_arcsec=Config.DEFAULT_PIXEL_SCALE,
        band_names=BAND_NAMES, is_clean=True)
    # Noise RNG decoupled from the scene seed (+1) so re-rolling the scene
    # never reuses a noise stream.
    lr, _hr = fwd.process(hr, np.random.default_rng(seed + 1),
                          star_hr_4ch=stars)
    dirty = np.asarray(lr.data, dtype=np.float32)

    model = load_eval_ensemble(ensemble_dir, num_res_blocks,
                               log=lambda msg: print(f"[poster] {msg}"))
    _lr_vis, sr, _members = sr_from_model(model, dirty)
    return (dirty, np.asarray(sr, dtype=np.float32),
            eval_model_identity(model), str(model.label))


def _ascii(value) -> str:
    """FITS header text must be ASCII (member labels carry a '·')."""
    return (str(value).replace("·", ".")
            .encode("ascii", "replace").decode("ascii"))


def model_cards(identity: dict, model_label: str) -> list[tuple[str, object, str]]:
    """Header cards naming the production model behind an SR: its combiner
    (``member_mean`` = the plain mean) and members — the full fitted list
    and the ones that ran (the combiner's reads)."""
    members = [_ascii(v) for v in identity.get("member_labels") or []]
    run = [_ascii(v) for v in identity.get("run_labels") or members]
    return [
        ("MODEL", _ascii(model_label), ""),   # no comment: labels run long
        ("COMBINER", identity.get("combiner_kind") or "member_mean",
         "production combiner"),
        # The combiner artifact's sha256: 64 hex characters fill the card,
        # so no comment.
        ("COMBFP", identity.get("combiner_fingerprint") or "", ""),
        ("NMEMBERS", len(members), "members the model was fitted on"),
        ("NRUN", len(run), "members that ran"),
        ("MEMBERS", ",".join(members), "fitted member labels"),
        ("RUNMEMB", ",".join(run), "member labels that ran"),
    ]


def append_field_companions(
    hdul: fits.HDUList, dirty: np.ndarray, sr: np.ndarray,
    *, identity: dict, model_label: str,
) -> None:
    """Stack the dirty LR bands and the 4-band SR cube into the field FITS.

    One file holds the whole triplet (CLEAN_* / DIRTY_* / SR_* extensions),
    so the WebUI's single "pull latest FITS" fetches everything. The clean
    band HDUs are renamed with a CLEAN_ prefix for symmetry; every SR_<band>
    header names the model (:func:`model_cards`).
    """
    sr = np.asarray(sr, dtype=np.float32)
    if sr.ndim != 3 or sr.shape[-1] != len(BAND_NAMES):
        raise ValueError(f"SR must be a {len(BAND_NAMES)}-band cube, got {sr.shape}")
    for name in BAND_NAMES:                       # VIS → CLEAN_VIS, …
        hdul[name].name = f"CLEAN_{name}"
    for k, name in enumerate(BAND_NAMES):
        hdu = fits.ImageHDU(data=np.asarray(dirty[..., k], dtype=np.float32),
                            name=f"DIRTY_{name}")
        hdu.header["BAND"] = (name, "Euclid band")
        hdu.header["PIXSCALE"] = (0.10, "arcsec/pixel (LR archive grid)")
        hdu.header["BUNIT"] = ("electron", "noisy LR flux over the stack")
        hdul.append(hdu)
    # One SR_<band> extension per band, mirroring the DIRTY_<band> convention.
    cards = model_cards(identity, model_label)
    for k, name in enumerate(BAND_NAMES):
        sp = fits.ImageHDU(data=np.ascontiguousarray(sr[..., k]),
                           name=f"SR_{name}")
        sp.header["BAND"] = (name, "Euclid band")
        sp.header["PIXSCALE"] = (float(Config.DEFAULT_PIXEL_SCALE),
                                 "arcsec/pixel (HR grid)")
        sp.header["BUNIT"] = ("electron", "super-resolved sky")
        for key, value, comment in cards:
            sp.header[key] = (value, comment)
        hdul.append(sp)


def render_field_preview_png(
    clean: np.ndarray, dirty: np.ndarray, sr: np.ndarray,
    *, seed: int, source_meta: dict,
) -> bytes:
    """Field mode preview: clean VIS | mock-Euclid VIS | super-resolved VIS
    | the SR's physical "eye" color composite (per-pixel blackbody T →
    Planckian-locus hue over the 4-band SR cube).
    """
    sr = np.asarray(sr, dtype=np.float32)
    panels = [("clean VIS (0.05″/pix)", clean[..., 0]),
              ("mock Euclid VIS (0.10″/pix)", dirty[..., 0]),
              ("super-resolved VIS (0.05″/pix)", sr[..., 0]),
              ("super-resolved eye color (4-band)", None)]
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(n * 3.0, 3.3))
    for ax, (title, im) in zip(np.atleast_1d(axes), panels, strict=False):
        if im is None:
            rgb = eye_rgb(sr, band_names=BAND_NAMES, stretch="asinh",
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
    p.add_argument("--ensemble-dir", default=None,
                   help="Ensemble base dir for the field-mode reconstruction "
                        "(default: <ckpt parent>/ensemble, i.e. next to "
                        "$EUCLID_POLISH_CKPT_DIR). The production gate under "
                        "<vis>/ensemble/starfull picks the members it reads.")
    p.add_argument("--num-res-blocks", type=int,
                   default=Config.DEFAULT_NUM_RES_BLOCKS)
    p.add_argument("--psf-dir", default=Config.EUCLID_PSF_DIR,
                   help="Per-band ePSF dir for the field-mode forward model "
                        "(Gaussian fallback where a band's file is missing).")
    p.add_argument("--joint-galaxy-population-json", default="",
                   help="Activated galaxy-population artifact as JSON "
                        "(lens/tng/field).")
    p.add_argument("--joint-galaxy-population-file", default="",
                   help="Path to the activated galaxy-population JSON "
                        "artifact (lens/tng/field).")
    p.add_argument("--star-prior-json", default="",
                   help="Activated empirical stellar-population artifact as "
                        "JSON (star/field).")
    p.add_argument("--star-prior-file", default="",
                   help="Path to the activated stellar-population JSON "
                        "artifact (star/field).")
    return p.parse_args(argv)


def _read_payload(inline: str, path: str, *, label: str) -> dict | None:
    """One JSON-object artifact given inline or as a staged file; ``None``
    when neither is given."""
    inline, path = str(inline or "").strip(), str(path or "").strip()
    if inline and path:
        raise ValueError(f"{label}: pass the JSON or the file, not both")
    if path:
        with open(path, encoding="utf-8") as fh:
            inline = fh.read().strip()
    if not inline:
        return None
    payload = json.loads(inline)
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _active_galaxy_population() -> dict | None:
    """The locally activated galaxy-population artifact, if it is current."""
    state = joint_galaxy_state()
    return state.get("active") if state.get("is_active") else None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    size = max(16, int(args.image_size))
    print(f"[poster] mode={args.mode} seed={args.seed} image_size={size}")

    # The submitted job carries frozen artifacts; a direct run falls back to
    # the ones activated under $EUCLID_POLISH_DATA_DIR.
    star_prior_payload = galaxy_population_payload = None
    if args.mode in STAR_MODES:
        star_prior_payload = _read_payload(
            args.star_prior_json, args.star_prior_file,
            label="stellar prior") or active_star()
    if args.mode in GALAXY_MODES:
        galaxy_population_payload = _read_payload(
            args.joint_galaxy_population_json,
            args.joint_galaxy_population_file,
            label="galaxy population") or _active_galaxy_population()

    scene, stars, source_meta, used_seed = generate_cutout(
        args.mode,
        seed=args.seed,
        image_size=size,
        star_prior_payload=star_prior_payload,
        galaxy_population_payload=galaxy_population_payload,
    )
    data = scene if stars is None else scene + stars
    print(f"[poster] generated '{args.mode}' (used seed {used_seed}); "
          f"total flux = {float(data.sum()):.3e} e⁻")

    hdul = build_cutout_hdul(
        data, mode=args.mode, seed=used_seed, image_size=size,
        source_meta=source_meta)

    # Field mode: forward-model + reconstruct, so the result is the full
    # clean / dirty / SR triplet the pipeline trains on.
    dirty = sr = None
    if args.mode == "field":
        print("[poster] forward-modelling + reconstructing with the "
              "production model …")
        dirty, sr, identity, model_label = forward_and_reconstruct(
            scene, stars, used_seed, psf_dir=args.psf_dir,
            ensemble_dir=args.ensemble_dir,
            num_res_blocks=args.num_res_blocks)
        print(f"[poster] reconstruction: {model_label}")
        append_field_companions(hdul, dirty, sr, identity=identity,
                                model_label=model_label)

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
          + (" (CLEAN_*/DIRTY_*/SR_* extensions)" if args.mode == "field" else ""))
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
