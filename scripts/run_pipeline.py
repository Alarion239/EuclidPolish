#!/usr/bin/env python
"""Non-interactive pipeline driver for the multi-band EuclidPolish workflow.

Mirrors the three CLI menu steps but drives them sequentially without prompts:

    1. Generate clean HR fields with COSMOS2025 galaxies + stars + lenses
       (saved as ``clean_{train,validate,test}.tfrecord`` in v2 schema,
       4 channels).
    2. Run the per-band forward model HR → LR (PSF convolution + detector/MER
       noise), including a seeded elastic PSF deformation. NISP Y/J/H noise is
       drawn per native 0.30" dither and resampled to the 0.10" archive grid.
       Saved as ``dirty_{train,validate,test}.tfrecord``, 4-channel LR at
       0.10″/pix.
    3. Train WDSR (4-channel input, 4-channel VIS+NISP HR target).

Any step can be skipped via ``--skip-{generate,convolve,train}``.

All file paths and constants come from :mod:`euclid_polish.config`.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

# Cap per-process BLAS threads BEFORE numpy import so the parallel
# generate+forward workers map 1:1 onto their CPUs instead of each
# spawning a thread per core. Affects only numpy/scipy (generation +
# forward model); TF training uses its own thread pools, so this does NOT
# slow the train step. ``setdefault`` honours an explicit override.
os.environ.setdefault("OMP_NUM_THREADS",      "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS",      "1")

# Make ``euclid_polish`` importable when running this file directly.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


import contextlib

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from euclid_polish.config import Config
from euclid_polish.image import Image
from euclid_polish.image.tfio import (
    open_writer,
    tfrecord_path,
)
from euclid_polish.model import Model
from euclid_polish.observability.reporter import Reporter
from euclid_polish.observability.resource_sampler import ResourceSampler
from euclid_polish.psf.psf_library import load_all_band_psf_sets
from euclid_polish.sky.generation.cosmos_tng_prior import (
    CosmosTngPrior,
    F814WToVisTransfer,
)
from euclid_polish.sky.generation.gen_provenance import (
    ShardStampPlan,
    make_generation_context,
)
from euclid_polish.sky.generation.sky_simulator import (
    SkySimulator,
    SkySimulatorConfig,
    _deposit_star,
    star_band_magnitudes_from_record,
)
from euclid_polish.sky.generation.source_catalog import (
    SOURCE_COLS,
    SourceCatalogWriter,
    concat_source_csvs,
    read_sources,
)
from euclid_polish.sky.observation.observation_simulator import (
    ObservationSimulator,
    ObservationSimulatorConfig,
)
from euclid_polish.training.stage_timer import StageTimer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log(msg: str) -> None:
    print(f"[{_ts()}] {msg}", flush=True)


def _banner(msg: str) -> None:
    bar = "=" * 70
    print(f"\n{bar}\n[{_ts()}] {msg}\n{bar}", flush=True)


_SPLITS = ("train", "validate", "test")


def _parse_regenerate_splits(value: str) -> tuple[str, ...]:
    """Parse a comma-separated, de-duplicated list of dataset splits."""
    splits = tuple(dict.fromkeys(
        part.strip().lower() for part in value.split(",") if part.strip()
    ))
    invalid = [split for split in splits if split not in _SPLITS]
    if not splits:
        raise argparse.ArgumentTypeError(
            "--regenerate-splits needs train, validate, and/or test")
    if invalid:
        raise argparse.ArgumentTypeError(
            "invalid regeneration split(s): " + ", ".join(invalid))
    return splits


def _regenerate_splits(args: argparse.Namespace) -> tuple[str, ...]:
    """Targeted destructive splits, with a legacy-Namespace fallback."""
    value = getattr(args, "regenerate_splits", ())
    if isinstance(value, str):
        return _parse_regenerate_splits(value) if value.strip() else ()
    return tuple(value or ())


def _split_selected(args: argparse.Namespace, subset: str) -> bool:
    """No target means normal all-split resume; a target means only those."""
    targets = _regenerate_splits(args)
    return not targets or subset in targets


def _split_forced(args: argparse.Namespace, subset: str) -> bool:
    return bool(getattr(args, "force", False) or
                subset in _regenerate_splits(args))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--catalog",
        default="",
        help=argparse.SUPPRESS,  # accepted only for old submitted-job replay
    )
    ap.add_argument("--psf-dir",        default=Config.EUCLID_PSF_DIR,
                    help="Directory containing per-band ePSF FITS files; "
                         "missing bands fall back to Gaussian.")
    ap.add_argument("--psf-warp-prob", type=float,
                    default=Config.TRAIN_PSF_WARP_PROB,
                    help="Probability that each generated dirty exposure "
                         "receives an elastic deformation of its shared "
                         "empirical PSF. Default 1; 0 disables it.")
    ap.add_argument("--psf-warp-alpha-max", type=float,
                    default=Config.TRAIN_PSF_WARP_ALPHA_MAX,
                    help="Maximum observation-PSF elastic displacement in HR "
                         "pixels; alpha is drawn uniformly from [0, max] "
                         "for each exposure (default 20).")
    ap.add_argument("--psf-warp-sigma", type=float,
                    default=Config.TRAIN_PSF_WARP_SIGMA,
                    help="Gaussian smoothing scale of the elastic "
                         "displacement field in HR pixels (default 3).")
    ap.add_argument("--saturation-mask-prob", type=float,
                    default=Config.TRAIN_SATURATION_MASK_PROB,
                    help="Probability that an above-well source receives a "
                         "dark rectangular core in generated train, "
                         "validate, and test data. Default 0.2; capped at "
                         "0.5 so intact bright cores remain common.")
    ap.add_argument("--records-dir",    default=Config.RECORDS_DIR_V2)
    ap.add_argument("--checkpoint-dir", default=Config.DEFAULT_CHECKPOINT_DIR)
    ap.add_argument("--ntrain",         type=int, default=6400)
    ap.add_argument("--nvalid",         type=int, default=100)
    ap.add_argument("--ntest",          type=int, default=100,
                    help="Held-out test images (default 100). Evals run on this "
                         "set, not validate (which the trainer keeps for "
                         "save-best). 0 disables the test split.")
    ap.add_argument("--image-size",     type=int, default=252,
                    help="HR field side (HR pixels). Must be divisible by 6 "
                         "(LCM of VIS rebin=2 and NISP rebin=6).")
    ap.add_argument("--steps",          type=int, default=Config.DEFAULT_TRAIN_STEPS)
    ap.add_argument("--batch-size",     type=int, default=Config.DEFAULT_BATCH_SIZE)
    ap.add_argument("--evaluate-every", type=int, default=Config.DEFAULT_EVALUATE_EVERY)
    ap.add_argument("--num-res-blocks", type=int, default=Config.DEFAULT_NUM_RES_BLOCKS)
    ap.add_argument("--require-empirical-psf", action="store_true",
                    help="Fail if any band lacks a real ePSF (no Gaussian fallback).")
    ap.add_argument("--gen-workers", type=int, default=1,
                    help="Parallelise synthetic generation across this many "
                         "processes. >1 runs a COMBINED generate+forward pass "
                         "(each worker renders clean → hr+dirty for a "
                         "contiguous index range into its own TFRecord "
                         "shards, then the shards are concatenated in order). "
                         "Requires both generate and convolve (i.e. neither "
                         "--skip-generate nor --skip-convolve); falls back to "
                         "the serial two-step path otherwise.")
    ap.add_argument("--galaxy-density-arcmin2", type=float,
                    default=Config.GALAXY_DENSITY_ARCMIN2,
                    help="COSMOS-conditioned TNG galaxies per arcmin².")
    ap.add_argument("--galaxy-thinning-max-density-arcmin2", type=float,
                    default=None,
                    help="Calibration-only master density. Shared seeds make "
                         "lower densities exact nested source thinnings.")
    ap.add_argument("--cosmos-prior", default=Config.COSMOS_TNG_PRIOR_PATH,
                    help="Joint COSMOS2025 population-prior NPZ.")
    ap.add_argument("--tng-dir", default=Config.TNG_SKIRT_DIR,
                    help="Downloaded TNG SKIRT atlas directory.")
    ap.add_argument("--tng-properties", default="",
                    help="TNG stellar-properties CSV (required for strict generation).")
    ap.add_argument("--tng-radius-manifest", default="",
                    help="Versioned measured TNG VIS-radius manifest.")
    ap.add_argument("--cosmos-vis-offset-mag", type=float, default=None,
                    help="Embedded fitted VIS-F814W magnitude offset.")
    ap.add_argument("--cosmos-vis-magnitude-slope", type=float, default=None,
                    help="Embedded fitted F814W-to-VIS magnitude slope.")
    ap.add_argument("--cosmos-vis-scatter-mag", type=float, default=None,
                    help="Embedded fitted F814W-to-VIS scatter.")
    ap.add_argument("--cosmos-vis-transfer-source", default="",
                    help="Provenance label for the embedded brightness fit.")
    ap.add_argument("--cosmos-vis-transfer-artifact-json", default="",
                    help="Exact activated transfer artifact embedded by the web job.")
    ap.add_argument("--star-density-arcmin2", type=float,
                    default=Config.DEFAULT_STAR_DENSITY_ARCMIN2,
                    help="Stellar surface density (stars/arcmin²).")
    ap.add_argument("--star-mag-slope", type=float, default=Config.STAR_MAG_SLOPE,
                    help="Star-count slope α in dN/dm ∝ 10^(α·m) "
                         "(high-Galactic-latitude value ~0.14–0.35).")
    ap.add_argument("--star-mag-bright", type=float, default=Config.STAR_MAG_BRIGHT,
                    help="Brightest synthetic star (VIS mag).")
    ap.add_argument("--star-mag-faint", type=float, default=Config.STAR_MAG_FAINT,
                    help="Faintest synthetic star (VIS mag).")
    ap.add_argument("--star-prior-json", default="",
                    help="Activated versioned Gaia/Euclid stellar-prior JSON.")
    ap.add_argument("--lens-density-arcmin2", type=float,
                    default=Config.LENS_DENSITY_ARCMIN2,
                    help="Strong-lens surface density (lenses/arcmin²).")
    ap.add_argument("--lens-sigma-v-min-kms", type=float,
                    default=Config.LENS_SIGMA_V_MIN_KMS,
                    help="Min lens velocity dispersion (km/s); σ_v² sets θ_E.")
    ap.add_argument("--lens-sigma-v-max-kms", type=float,
                    default=Config.LENS_SIGMA_V_MAX_KMS,
                    help="Max lens velocity dispersion (km/s); σ_v² sets θ_E.")
    ap.add_argument("--seed", type=int, default=-1,
                    help="Master RNG seed for generation/forward-model. "
                         "-1 (default) draws a fresh entropy seed each run. The "
                         "seed actually used is recorded on the run's "
                         "Process.generation provenance record, so passing the "
                         "stored value here replays a run deterministically.")
    ap.add_argument("--skip-generate",  action="store_true")
    ap.add_argument("--skip-convolve",  action="store_true")
    ap.add_argument("--skip-train",     action="store_true")
    ap.add_argument("--onthefly-train", action="store_true",
                    help="On-the-fly training mode: generate the TRAIN split "
                         "as clean_train ONLY — no hr_train, no dirty_train. "
                         "On-the-fly training reads clean_train and builds the "
                         "LR + target live (fresh PSF/noise/stars per visit), "
                         "so both would be dead weight (~13 GB + a forward per "
                         "field). validate/test still get the full clean + hr "
                         "+ dirty triple (training validation + evaluation read "
                         "them). Any stale hr_train/dirty_train (+ provenance "
                         "sidecars) left by an earlier record-mode run is "
                         "DELETED — they must not linger.")
    regeneration = ap.add_mutually_exclusive_group()
    regeneration.add_argument(
        "--force", action="store_true",
        help="Regenerate every subset from scratch, ignoring already-complete "
             "data on disk (default: resume — skip complete subsets).")
    regeneration.add_argument(
        "--regenerate-splits", type=_parse_regenerate_splits, default=(),
        metavar="SPLITS",
        help="Delete and rebuild only these comma-separated splits (train, "
             "validate, test). Unselected splits are left untouched even if "
             "incomplete.")
    ap.add_argument("--stages-csv", default="",
                    help="Path to per-stage timings CSV. "
                         "Default: <records-dir>/stages_${SLURM_JOB_ID}.csv "
                         "(or stages_local.csv outside SLURM).")
    return ap.parse_args(argv)


# ---------------------------------------------------------------------------
# Reproducible seeding
#
# One master ``run_seed`` per invocation (``--seed`` when >= 0, else entropy)
# is recorded on the run's Process.generation provenance and used to derive
# every per-(subset, stream) / per-shard RNG. Storing the one int is enough to
# replay the whole run. Stream tags are kept large so they never collide with
# the small shard ids the parallel path threads through the 3rd seed slot.
# ---------------------------------------------------------------------------

_STREAM_GEN = 10_000   # clean-scene generation draws
_STREAM_FWD = 10_001   # forward-model (noise / artifact) draws

#: Distinct per-subset tags so train / validate / test draw independent RNG
#: streams from the one run_seed (else test would alias validate's noise).
_SUBSET_TAGS = {"train": 1, "validate": 2, "test": 3}


def _subset_tag(subset: str) -> int:
    return _SUBSET_TAGS.get(subset, 0)


def _ntest(args: argparse.Namespace) -> int:
    """Held-out test-set size; 0 when ``--ntest`` is absent (e.g. a partial
    args built programmatically), so the test split is simply skipped."""
    return int(getattr(args, "ntest", 0))


def _resolve_run_seed(args: argparse.Namespace) -> int:
    """The run's master seed: ``--seed`` when >= 0, else a fresh entropy draw."""
    s = getattr(args, "seed", -1)
    if s is not None and int(s) >= 0:
        return int(s)
    return int.from_bytes(os.urandom(8), "little")


def _subset_rng(run_seed: int, subset: str,
                stream: int) -> np.random.Generator:
    """Deterministic RNG for one (subset, stream), derived from ``run_seed``."""
    return np.random.default_rng([run_seed, _subset_tag(subset), stream])


# ---------------------------------------------------------------------------
# Step 1: clean multi-band scene generation
# ---------------------------------------------------------------------------

def _generator_config_from_args(args: argparse.Namespace) -> SkySimulatorConfig:
    """Build the generator config from CLI args (shared by serial + parallel)."""
    return SkySimulatorConfig(
        image_size=args.image_size,
        pixel_scale=Config.DEFAULT_PIXEL_SCALE,
        galaxy_density_arcmin2=args.galaxy_density_arcmin2,
        galaxy_thinning_max_density_arcmin2=getattr(
            args, "galaxy_thinning_max_density_arcmin2", None,
        ),
        cosmos_prior_path=args.cosmos_prior,
        tng_galaxy_dir=getattr(args, "tng_dir", Config.TNG_SKIRT_DIR),
        tng_properties_csv=getattr(args, "tng_properties", ""),
        tng_radius_manifest_path=(
            getattr(args, "tng_radius_manifest", "")
            or os.path.join(Config.DATA_DIR, "_tng_infographics",
                             "tng_radius_manifest.json")
        ),
        strict_population_artifacts=True,
        star_density_arcmin2=args.star_density_arcmin2,
        star_mag_slope=args.star_mag_slope,
        star_mag_bright=args.star_mag_bright,
        star_mag_faint=args.star_mag_faint,
        star_prior_payload=(
            json.loads(args.star_prior_json) if args.star_prior_json else None
        ),
        lens_density_arcmin2=args.lens_density_arcmin2,
        lens_sigma_v_min_kms=args.lens_sigma_v_min_kms,
        lens_sigma_v_max_kms=args.lens_sigma_v_max_kms,
    )


def _photometric_transfer_from_args(
    args: argparse.Namespace,
) -> F814WToVisTransfer | None:
    """Return an explicitly embedded transfer, or defer to the local fit.

    Web-submitted FASRC jobs embed all three fitted coefficients because the
    analysis artifact lives on the web host and may not exist on the cluster.
    Older CLI invocations without these flags retain the file-based behavior.
    """
    values = (
        getattr(args, "cosmos_vis_offset_mag", None),
        getattr(args, "cosmos_vis_magnitude_slope", None),
        getattr(args, "cosmos_vis_scatter_mag", None),
    )
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError(
            "COSMOS brightness transfer needs offset, slope, and scatter"
        )
    offset, slope, scatter = (float(value) for value in values)
    if not all(np.isfinite(value) for value in (offset, slope, scatter)):
        raise ValueError("COSMOS brightness transfer must be finite")
    if slope <= 0.0 or scatter < 0.0:
        raise ValueError(
            "COSMOS brightness slope must be positive and scatter non-negative"
        )
    return F814WToVisTransfer(
        offset_mag=offset,
        magnitude_slope=slope,
        scatter_mag=scatter,
        source=(
            str(getattr(args, "cosmos_vis_transfer_source", "")).strip()
            or "embedded_cli"
        ),
    )


def _observation_config_from_args(
    args: argparse.Namespace,
) -> ObservationSimulatorConfig:
    """Build the forward config shared by train, validate, and test.

    ``getattr`` fallbacks preserve programmatic callers that construct a
    minimal legacy Namespace while CLI/WebUI runs get the configured warp.
    """
    saturation_mask_prob = float(getattr(
        args, "saturation_mask_prob", Config.TRAIN_SATURATION_MASK_PROB,
    ))
    if not 0.0 <= saturation_mask_prob <= Config.TRAIN_SATURATION_MASK_PROB_MAX:
        raise ValueError(
            "saturation_mask_prob must be in [0, "
            f"{Config.TRAIN_SATURATION_MASK_PROB_MAX:g}] for training data"
        )
    return ObservationSimulatorConfig(
        add_noise=True,
        psf_warp_prob=float(getattr(
            args, "psf_warp_prob", Config.TRAIN_PSF_WARP_PROB)),
        psf_warp_alpha_max=float(getattr(
            args, "psf_warp_alpha_max", Config.TRAIN_PSF_WARP_ALPHA_MAX)),
        psf_warp_sigma=float(getattr(
            args, "psf_warp_sigma", Config.TRAIN_PSF_WARP_SIGMA)),
        saturation_mask_prob=saturation_mask_prob,
    )


def step_generate(args: argparse.Namespace) -> None:
    _banner(f"STEP 1: Generate clean 4-band HR fields  "
            f"({args.ntrain} train + {args.nvalid} valid, "
            f"{args.image_size}² @ {Config.DEFAULT_PIXEL_SCALE}\"/pix)")

    cat = (CosmosTngPrior(
               args.cosmos_prior,
               photometric_transfer=_photometric_transfer_from_args(args),
           )
           if args.galaxy_density_arcmin2 > 0.0 else None)
    if cat is not None:
        _log(f"COSMOS joint prior: {len(cat)} latent galaxies")

    cfg = _generator_config_from_args(args)
    sim = SkySimulator(cat, cfg)
    os.makedirs(args.records_dir, exist_ok=True)

    # One master seed for the whole step, recorded on the generation run so it
    # can be replayed via --seed; every per-subset RNG is derived from it.
    run_seed = _resolve_run_seed(args)
    gen_ctx = make_generation_context(cfg, seed=run_seed)
    _log(f"  run_seed={run_seed}  (replay with --seed {run_seed})")

    subsets = (("train", args.ntrain), ("validate", args.nvalid),
               ("test", _ntest(args)))
    targets = _regenerate_splits(args)
    if targets:
        _log("  targeted regeneration: " + ", ".join(targets) +
             " (all other splits remain untouched)")

    # A destructive run removes every selected split before writing any of
    # them. This prevents a wall-time kill from leaving matching-count stale
    # finals that a later resume could mistake for the new calibration.
    if getattr(args, "force", False) or targets:
        for subset, n in subsets:
            if n > 0 and _split_selected(args, subset):
                _cleanup_parts(args.records_dir, subset)
                _remove_subset_finals(args.records_dir, subset)

    # Structured progress for the WebUI (no terminal for tqdm under SLURM).
    # In targeted mode, the total includes only the selected splits.
    reporter = Reporter.from_env()
    # Sample CPU (and GPU, if the train step runs on one) through the whole
    # pipeline so the WebUI shows live utilisation — confirms the parallel
    # generate workers are actually busy. Daemon thread; dies at exit.
    ResourceSampler(reporter).start()
    reporter.set_stage("generating clean HR fields")
    grand_total = sum(int(n) for subset, n in subsets
                      if _split_selected(args, subset))
    done = 0

    for subset, n in subsets:
        if n <= 0:
            continue
        if not _split_selected(args, subset):
            _log(f"  {subset}: not selected — leaving existing data untouched")
            continue
        if not _split_forced(args, subset) and _subset_complete(
                args.records_dir, subset, ("clean", "sources"), n):
            done += n
            _log(f"  {subset}: clean already complete ({n} records) — skipping")
            reporter.set_step(done, grand_total, f"{subset} already complete")
            continue
        # Per-subset RNG derived from the run's master seed → the whole run
        # replays from the single recorded run_seed.
        rng = _subset_rng(run_seed, subset, _STREAM_GEN)
        _log(f"  {subset}: generating {n} images  (run_seed={run_seed})")
        t0 = time.perf_counter()
        # Stream each image to disk as it's generated — accumulating
        # 6400 510² × 4-channel float32 fields would cost ~26 GB of RSS
        # and OOM-kill on the FASRC default --mem=32G.
        with open_writer(f"clean_{subset}",
                                   records_dir=args.records_dir) as w, \
             SourceCatalogWriter(
                 tfrecord_path(args.records_dir, f"sources_{subset}")
                 .replace(".tfrecord", ".csv")) as sources:
            # Train draws no fixed stars (on-the-fly injects them per visit);
            # validate/test draw + record fixed stars for a reproducible LR.
            n_stars = 0 if subset == "train" else None
            for i in tqdm(range(n), desc=f"  {subset}", unit="img"):
                sky, meta = sim.simulate_field(rng, n_stars=n_stars)
                sky.index = i
                sky.subset = subset
                if gen_ctx is not None:
                    with contextlib.suppress(Exception):
                        sky.stamp = gen_ctx.stamp("clean", subset)
                w.write(sky, index=i)
                sources.add_field(i, meta)
                done += 1
                reporter.set_step(done, grand_total, f"generate {subset} {i + 1}/{n}")
            path, count = w.path, w.count
        if gen_ctx is not None:
            with contextlib.suppress(Exception):
                gen_ctx.finalize("clean", subset, path)
        _log(f"  {subset}: done — {count} → {path}  "
             f"({time.perf_counter() - t0:.1f} s)")


# ---------------------------------------------------------------------------
# Step 2: per-band PSF convolution + detector/MER noise
# ---------------------------------------------------------------------------

def step_convolve(args: argparse.Namespace) -> None:
    _banner("STEP 2: HR → LR  (per-band PSF + native detector/MER noise)")

    targets = _regenerate_splits(args)
    if targets and getattr(args, "skip_generate", False):
        _log("  targeted regeneration: " + ", ".join(targets) +
             " (all other splits remain untouched)")

    # In a convolve-only targeted run, preserve clean inputs while removing
    # just the stale forward products. In a normal generate+convolve run these
    # files were already removed by step_generate, so this is harmless.
    if getattr(args, "force", False) or targets:
        for subset in _SPLITS:
            if _split_selected(args, subset):
                _remove_subset_finals(
                    args.records_dir, subset, kinds=("hr", "dirty"))

    psf_sets = load_all_band_psf_sets(
        psf_dir=args.psf_dir,
        require_empirical=args.require_empirical_psf,
        target_pixel_scale=Config.DEFAULT_PIXEL_SCALE,
    )
    for name, pset in psf_sets.items():
        _log(f"  PSF[{name}]: {pset.n} kernel(s), shape={pset.shape}, "
             f"{pset.pixel_scale}\"/pix")

    fwd = ObservationSimulator(
        psf_sets_by_band=psf_sets,
        config=_observation_config_from_args(args),
    )
    _log(
        "  PSF warp: "
        f"p={fwd.config.psf_warp_prob:g}, "
        f"alpha_max={fwd.config.psf_warp_alpha_max:g} HR px, "
        f"sigma={fwd.config.psf_warp_sigma:g} HR px"
    )
    _log(
        "  saturation dark-core probability: "
        f"{fwd.config.saturation_mask_prob:g}"
    )
    _log(
        "  NISP noise: 4 native 0.30\" dithers → "
        f"{fwd.config.nisp_resample_kernel} MER resample → 0.10\" (/9 flux)"
    )

    # One master seed for the forward step, recorded on its run so the noise /
    # artifact realisations can be replayed via --seed.
    run_seed = _resolve_run_seed(args)
    conv_ctx = make_generation_context(fwd.config, seed=run_seed)
    _log(f"  run_seed={run_seed}  (replay with --seed {run_seed})")

    # Structured progress for the WebUI — one cumulative bar across both
    # subsets present. Pre-count the clean records (re-iterating is ~ms).
    reporter = Reporter.from_env()
    reporter.set_stage("forward-modelling HR → LR")
    counts = {}
    for subset in ("train", "validate", "test"):
        p = tfrecord_path(args.records_dir, f"clean_{subset}")
        counts[subset] = (sum(1 for _ in tf.data.TFRecordDataset(p))
                          if os.path.exists(p) else 0)
    grand_total = sum(count for subset, count in counts.items()
                      if _split_selected(args, subset))
    done = 0

    onthefly_train = bool(getattr(args, "onthefly_train", False))
    n_expected_by_subset = {"train": args.ntrain, "validate": args.nvalid,
                            "test": _ntest(args)}
    for subset in ("train", "validate", "test"):
        if not _split_selected(args, subset):
            _log(f"  {subset}: not selected — leaving existing data untouched")
            continue
        clean_path = tfrecord_path(args.records_dir, f"clean_{subset}")
        if not os.path.exists(clean_path):
            _log(f"⚠️  {clean_path} not found, skipping {subset}")
            continue

        # --onthefly-train: the TRAIN split is clean-only. On-the-fly training
        # reads clean_train and builds the LR + target live, so hr/dirty are
        # dead weight — don't forward-model them at all. Drop any stale ones
        # left by an earlier record-mode run (they must not linger).
        if onthefly_train and subset == "train":
            _remove_subset_finals(args.records_dir, "train", kinds=("hr", "dirty"))
            done += counts[subset]
            _log("  train: on-the-fly mode — clean only, skipping hr + dirty")
            reporter.set_step(done, grand_total, "train clean-only (on-the-fly)")
            continue

        n_expected = n_expected_by_subset[subset]
        if not _split_forced(args, subset) and _subset_complete(
                args.records_dir, subset, ("hr", "dirty"), n_expected):
            done += counts[subset]
            _log(f"  {subset}: already complete — skipping")
            reporter.set_step(done, grand_total, f"{subset} already complete")
            continue

        # Stream records from the clean TFRecord (do NOT materialise the
        # whole list — same OOM hazard as step_generate at 6400 images).
        clean_ds = tf.data.TFRecordDataset(clean_path)
        # The scene records are starless; re-inject each field's FIXED stars
        # (recorded in the source CSV) before the forward so the LR carries
        # star contamination while the HR target stays starfull.
        stars_by_field = {}
        sources_csv = tfrecord_path(
            args.records_dir, f"sources_{subset}").replace(".tfrecord", ".csv")
        for fidx, rows in read_sources(sources_csv).items():
            fld = [r for r in rows if r.get("type") == "star"]
            if fld:
                stars_by_field[fidx] = fld

        # The clean record file id (if stamped) is the lineage parent of the
        # hr+dirty files produced from it. Peek the first record once.
        clean_parent = None
        if conv_ctx is not None:
            try:
                first = Image.from_tfrecord(next(iter(clean_ds)))
                cs = first.prov_stamp()
                clean_parent = cs.id if cs is not None else None
            except Exception:
                clean_parent = None
        n_total = counts[subset]
        # Per-subset forward-model RNG derived from the run's master seed.
        rng = _subset_rng(run_seed, subset, _STREAM_FWD)

        _log(f"  {subset}: forward-modelling {n_total} fields  "
             f"(run_seed={run_seed})")
        t0 = time.perf_counter()
        # Two streaming writers (one for hr_, one for dirty_); clean_ is
        # NOT rewritten — the 4-band record is kept for inspection.
        with open_writer(f"hr_{subset}",
                                   records_dir=args.records_dir) as hr_w, \
             open_writer(f"dirty_{subset}", records_dir=args.records_dir) as lr_w:
            for i, raw in enumerate(tqdm(clean_ds, desc=f"  {subset}",
                                         unit="img", total=n_total)):
                hr_4ch = Image.from_tfrecord(raw)
                lr, hr = _forward_with_stars(
                    fwd, hr_4ch, stars_by_field.get(i), rng)
                lr.index = hr.index = i
                lr.subset = hr.subset = subset
                if conv_ctx is not None:
                    try:
                        parents = (clean_parent,) if clean_parent is not None else ()
                        hr.stamp = conv_ctx.stamp("hr", subset, parents=parents)
                        lr.stamp = conv_ctx.stamp("dirty", subset,
                                                  parents=parents)
                    except Exception:
                        pass
                hr_w.write(hr, index=i)
                lr_w.write(lr, index=i)
                done += 1
                reporter.set_step(done, grand_total, f"forward {subset} {i + 1}/{n_total}")
        if conv_ctx is not None:
            try:
                parents = (clean_parent,) if clean_parent is not None else ()
                conv_ctx.finalize("hr", subset,
                                  tfrecord_path(args.records_dir, f"hr_{subset}"),
                                  parents=parents)
                conv_ctx.finalize(
                    "dirty", subset,
                    tfrecord_path(args.records_dir, f"dirty_{subset}"),
                    parents=parents)
            except Exception:
                pass
        _log(f"  {subset}: done in {time.perf_counter() - t0:.1f} s → kept "
             f"clean + wrote hr + dirty {subset}")


# ---------------------------------------------------------------------------
# Parallel combined generate + forward (one process per index range)
# ---------------------------------------------------------------------------

def _concat_tfrecords(part_paths: list[str], out_path: str) -> None:
    """Byte-concatenate TFRecord shard files into one TFRecord.

    A TFRecord file is a bare back-to-back sequence of self-framed records
    (no header/footer), so concatenating shards in order yields a valid
    TFRecord whose records appear in shard order. Missing/empty parts are
    skipped."""
    with open(out_path, "wb") as out:
        for p in part_paths:
            if os.path.exists(p) and os.path.getsize(p) > 0:
                with open(p, "rb") as f:
                    shutil.copyfileobj(f, out, length=4 * 1024 * 1024)


def _shard_bounds(n: int, n_shards: int) -> list[tuple[int, int]]:
    """Contiguous ``[start, end)`` ranges partitioning ``[0, n)``."""
    return [(round(k * n / n_shards), round((k + 1) * n / n_shards))
            for k in range(n_shards)]


# ---------------------------------------------------------------------------
# Resume support: detect already-complete subsets so a resubmitted job only
# generates what's left (e.g. a SLURM job killed after ``train`` finishes).
# ---------------------------------------------------------------------------

def _count_tfrecords(path: str) -> int | None:
    """Number of examples in a TFRecord file, or None if it can't be read in
    full. A missing file or a record truncated by a job killed mid-merge
    (``tf.errors.DataLossError``) both return None — i.e. 'not complete'."""
    if not os.path.exists(path):
        return None
    try:
        return sum(1 for _ in tf.data.TFRecordDataset(path))
    except tf.errors.DataLossError:
        return None


def _sources_complete(csv_path: str, expected_n: int) -> bool:
    """True iff the source sidecar exists (expected_n <= 0 is trivially OK).

    Sidecar rows are sparse — a field that renders no galaxies/lenses writes no
    row — so a field_index-coverage check would false-flag a complete run whose
    last field is empty. ``concat_source_csvs`` is atomic, so the final CSV only
    ever exists in complete form; existence is therefore a sound signal. The
    per-subset TFRecord count check is the authoritative guard."""
    if expected_n <= 0:
        return True
    return os.path.exists(csv_path)


def _forward_with_stars(fwd, sky_starless: Image, stars, rng) -> tuple:
    """Re-deposit the field's fixed stars, then forward → ``(lr, hr)`` — BOTH
    with stars (the STARFULL pair generated together, second).

    The scene is stored starless (the ``clean`` record, generated first =
    starless target). Here we add back the recorded ``stars`` (HR deltas,
    pre-PSF) so ``lr`` (dirty) carries realistic contamination and ``hr`` is
    the starfull HR target — exactly the pair the old pipeline produced, only
    reconstructed from starless + stars instead of stored with stars.
    """
    if stars:
        star_scene = np.zeros_like(sky_starless.data)
        for s in stars:
            _deposit_star(
                star_scene,
                float(s["x_pix"]),
                float(s["y_pix"]),
                float(s["mag_vis"]),
                band_magnitudes=star_band_magnitudes_from_record(s),
            )
    else:
        star_scene = None
    return fwd.process(
        sky_starless, rng=rng, star_hr_4ch=star_scene,
    )                                               # (lr, hr) both starfull


def _remove_subset_finals(records_dir: str, subset: str,
                          kinds: tuple[str, ...] = ("clean", "hr", "dirty",
                                                    "sources")) -> None:
    """Delete ``subset``'s FINAL record files (+ provenance sidecars).

    Destructive regeneration deletes up-front (rather than lazily overwriting
    at merge time). Otherwise a wall-time kill can leave OLD finals on disk,
    and a follow-up resume can count them as complete and keep stale data."""
    removed = []
    for kind in kinds:
        path = tfrecord_path(records_dir, f"{kind}_{subset}")
        if kind == "sources":
            path = path.replace(".tfrecord", ".csv")
        if os.path.isfile(path):
            os.remove(path)
            removed.append(os.path.basename(path))
    if removed:
        _log(f"  {subset}: deleted stale final(s) {', '.join(removed)} "
             "(destructive regeneration starts clean)")
    for sc in glob.glob(os.path.join(records_dir, "*.skytfrecordartifact.json")):
        try:
            with open(sc) as f:
                meta = json.load(f)
            d = meta.get("descriptors", {})
            if d.get("kind") in kinds and d.get("subset") == subset:
                os.remove(sc)
        except (OSError, ValueError):
            pass


def _subset_complete(records_dir: str, subset: str,
                     kinds, expected_n: int) -> bool:
    """True iff every TFRecord ``kind`` for ``subset`` has exactly ``expected_n``
    records and, when 'sources' is requested, the sidecar exists. A count that
    differs from ``expected_n`` (e.g. a resubmit with a different n) is treated
    as incomplete, so the subset is regenerated to match the request."""
    # Leftover shard parts = an UNFINISHED generation or merge (a successful
    # merge deletes them last). Whatever the final files count, the subset is
    # not done — without this, a run killed mid-merge leaves a mix of newly-
    # merged and stale finals whose counts all "match" and the resume skips.
    if glob.glob(os.path.join(records_dir, f"*_{subset}.part*")):
        return False
    for kind in kinds:
        if kind == "sources":
            csv_path = tfrecord_path(records_dir, f"sources_{subset}").replace(
                ".tfrecord", ".csv")
            if not _sources_complete(csv_path, expected_n):
                return False
        else:
            if _count_tfrecords(tfrecord_path(records_dir, f"{kind}_{subset}")) \
                    != expected_n:
                return False
    return True


def _cleanup_parts(records_dir: str, subset: str) -> None:
    """Remove leftover per-shard part files for ``subset`` from a prior run.

    Orphan parts survive when a resumed run uses a different shard count (the
    merge only reads the freshly-computed parts list), wasting disk; deleting
    them before regenerating keeps the records dir clean."""
    for kind in ("clean", "hr", "dirty", "sources"):
        for p in glob.glob(os.path.join(records_dir, f"{kind}_{subset}.part*")):
            with contextlib.suppress(OSError):
                os.remove(p)


# ---------------------------------------------------------------------------
# Shard-level resume: salvage the intact records a killed run left on disk so a
# resubmit only regenerates the shortfall, instead of redoing the whole subset.
# A SLURM SIGKILL can leave a part file with a half-written final record; we
# drop that bad tail, align the clean/hr/dirty views to a common length, and
# filter the source sidecar to the surviving fields.
# ---------------------------------------------------------------------------

def _salvage_tfrecord(path: str) -> int:
    """Rewrite ``path`` keeping only its leading run of intact records.

    Streams the good records into a sibling temp and atomically replaces the
    original, stopping at the first truncated/corrupt record (which a killed
    writer leaves as a ``tf.errors.DataLossError`` at the tail). Returns the
    surviving record count (0 for a missing/empty file)."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return 0
    tmp = path + ".salvage"
    n = 0
    writer = tf.io.TFRecordWriter(tmp)
    try:
        for raw in tf.data.TFRecordDataset(path):
            writer.write(raw.numpy())          # raw serialized record bytes
            n += 1
    except tf.errors.DataLossError:
        pass                                   # truncated/corrupt tail — stop
    finally:
        writer.close()
    os.replace(tmp, path)
    return n


def _truncate_tfrecord(path: str, keep: int) -> None:
    """Keep only the first ``keep`` records of ``path`` (rewrite in place).

    Aligns the clean/hr/dirty views of a shard when a kill landed between
    writing the three views of one field, so they stay position-paired."""
    tmp = path + ".trunc"
    n = 0
    with tf.io.TFRecordWriter(tmp) as w:
        for raw in tf.data.TFRecordDataset(path):
            if n >= keep:
                break
            w.write(raw.numpy())
            n += 1
    os.replace(tmp, path)


def _part_indices(path: str) -> list[int]:
    """Stored ``index`` of every record in ``path`` (decodes only that
    feature, never the pixel array)."""
    feats = {"index": tf.io.FixedLenFeature([], tf.int64)}
    return [int(tf.io.parse_single_example(raw, feats)["index"].numpy())
            for raw in tf.data.TFRecordDataset(path)]


def _filter_sources_part(csv_path: str, keep: set[int]) -> None:
    """Keep the header plus only complete rows whose ``field_index`` is in
    ``keep``. Drops a half-written final line from a killed run (wrong column
    count) and orphan rows for fields whose image was not saved."""
    if not os.path.isfile(csv_path):
        return
    ncommas = len(SOURCE_COLS) - 1
    tmp = csv_path + ".tmp"
    with open(csv_path, newline="") as fin, open(tmp, "w", newline="") as fout:
        first = True
        for line in fin:
            if first:                          # header
                fout.write(line)
                first = False
                continue
            head = line.split(",", 1)[0].strip()
            if line.count(",") == ncommas and head.isdigit() and int(head) in keep:
                fout.write(line)
    os.replace(tmp, csv_path)


def _existing_part_sids(records_dir: str, subset: str) -> list[int]:
    """Ascending shard ids that have a ``clean_<subset>.partNNNN`` on disk."""
    prefix = os.path.join(records_dir, f"clean_{subset}.part")
    sids = []
    for p in glob.glob(prefix + "*"):
        tail = p[len(prefix):].split(".", 1)[0]
        if tail.isdigit():
            sids.append(int(tail))
    return sorted(sids)


def _salvage_shard(records_dir: str, subset: str, sid: int,
                   cap: int,
                   kinds: tuple[str, ...] = ("clean", "hr", "dirty"),
                   ) -> tuple[int, list[int]]:
    """Salvage one shard's clean/hr/dirty/sources parts to a common, intact
    prefix of at most ``cap`` records. Returns ``(kept, index_list)``.

    The views are truncated to the shortest that survived (a kill between
    writing the views of one field), and the sources sidecar is filtered to
    the surviving field indices. ``kinds`` is just ``("clean",)`` for an
    ``--onthefly-train`` run (its train shards write only the clean part). A
    shard with nothing salvageable (or ``cap <= 0``) is deleted and returns
    ``(0, [])``."""
    paths = {k: tfrecord_path(records_dir, f"{k}_{subset}.part{sid:04d}")
             for k in kinds}
    src = tfrecord_path(
        records_dir, f"sources_{subset}.part{sid:04d}").replace(
            ".tfrecord", ".csv")
    counts = {k: _salvage_tfrecord(p) for k, p in paths.items()}
    keep = min(min(counts.values()), max(0, cap))
    if keep <= 0:
        for p in (*paths.values(), src):
            with contextlib.suppress(OSError):
                os.remove(p)
        return 0, []
    for k, p in paths.items():
        if counts[k] > keep:
            _truncate_tfrecord(p, keep)
    idx = _part_indices(paths["clean"])
    _filter_sources_part(src, set(idx))
    return keep, idx


def _salvage_subset(records_dir: str, subset: str,
                    n: int,
                    kinds: tuple[str, ...] = ("clean", "hr", "dirty"),
                    ) -> tuple[int, list[int], int]:
    """Salvage every prior-run shard for ``subset`` to a clean prefix, capped
    so the kept total never exceeds ``n`` (a resubmit with a smaller ``n``
    discards the surplus). Returns ``(total_kept, used_indices,
    next_free_shard_id)`` — new shards take ids at/above ``next_free_shard_id``
    so they never clobber a salvaged part."""
    sids = _existing_part_sids(records_dir, subset)
    done = 0
    used: list[int] = []
    for sid in sids:
        kept, idx = _salvage_shard(records_dir, subset, sid, cap=n - done,
                                   kinds=kinds)
        done += kept
        used.extend(idx)
    next_sid = (max(sids) + 1) if sids else 0
    return done, used, next_sid


def _merge_subset(records_dir: str, subset: str,
                  kinds: tuple[str, ...] = ("clean", "hr", "dirty")) -> None:
    """Concatenate every on-disk shard for ``subset`` into the final clean/hr/
    dirty TFRecords + sources CSV, then delete the parts.

    Shards merge in ascending id order, shared across the kinds so records
    stay position-aligned (the dataset pairs by position). ``kinds`` is just
    ``("clean",)`` on an ``--onthefly-train`` run — merging an absent hr/dirty
    kind would silently produce a 0-record file. Each output is built in a temp then
    atomically renamed, and parts are deleted only after every kind is
    merged — so a kill mid-merge leaves the parts intact for the next resume
    rather than a half-merged final file."""
    sids = _existing_part_sids(records_dir, subset)
    for kind in kinds:
        parts = [tfrecord_path(records_dir, f"{kind}_{subset}.part{sid:04d}")
                 for sid in sids]
        out = tfrecord_path(records_dir, f"{kind}_{subset}")
        _concat_tfrecords(parts, out + ".tmp")
        os.replace(out + ".tmp", out)
    src_parts = [tfrecord_path(records_dir, f"sources_{subset}.part{sid:04d}")
                 .replace(".tfrecord", ".csv") for sid in sids]
    concat_source_csvs(src_parts, tfrecord_path(
        records_dir, f"sources_{subset}").replace(".tfrecord", ".csv"))
    _cleanup_parts(records_dir, subset)


def _generate_convolve_range(sim, fwd, records_dir: str, subset: str,
                             start: int, count: int, shard_id: int,
                             seed, plan=None,
                             write_forward: bool = True,
                             ) -> tuple[str, int, int]:
    """Generate clean → forward to hr+dirty for ``[start, start+count)`` and
    write the triple to per-shard TFRecords.

    Pure (no globals / pool) so it's unit-testable with an injected
    sim/fwd. Records are written ``clean[i] → hr[i] → dirty[i]`` in the same
    order, so concatenating shards in id order keeps clean/hr/dirty
    position-aligned (the dataset pairs by position, not by stored index).

    ``write_forward=False`` (``--onthefly-train``): the forward model never
    runs and ONLY the clean part is written — no hr, no dirty. On-the-fly
    training reads ``clean_train`` and builds the LR + target live (injecting a
    fresh star realization per visit), so both would be dead weight.
    """
    rng = np.random.default_rng(seed)
    tag = f"{subset}.part{shard_id:04d}"
    # Per-worker progress → the parent's events file (shared, append-atomic).
    # Keyed by shard_id so the consumer can sum a cumulative count and tell
    # how many processes are busy. Rate-limited so 16 workers don't flood
    # the file; register at 0 up front so a just-started shard shows.
    reporter = Reporter.from_env()
    reporter.set_worker_step(shard_id, 0, count, subset)
    last_emit = time.perf_counter()
    sources_part = tfrecord_path(records_dir,
                                 f"sources_{tag}").replace(".tfrecord", ".csv")
    with open_writer(f"clean_{tag}", records_dir=records_dir) as cw, \
         (open_writer(f"hr_{tag}", records_dir=records_dir)
          if write_forward else contextlib.nullcontext()) as hw, \
         (open_writer(f"dirty_{tag}", records_dir=records_dir)
          if write_forward else contextlib.nullcontext()) as dw, \
         SourceCatalogWriter(sources_part) as sources:
        for local, i in enumerate(range(start, start + count), start=1):
            # Scene is starless. The training split (clean-only) draws no fixed
            # stars — on-the-fly training injects a fresh realization per
            # visit; validate/test draw + record fixed stars for a
            # reproducible LR.
            sky, meta = sim.simulate_field(rng, n_stars=(None if write_forward
                                                         else 0))
            sky.index = i
            sky.subset = subset
            if plan is not None:
                with contextlib.suppress(Exception):
                    sky.stamp = plan.clean_stamp(subset)
            cw.write(sky, index=i)
            if write_forward:
                lr, hr = _forward_with_stars(fwd, sky, meta.get("stars"), rng)
                lr.index = hr.index = i
                lr.subset = hr.subset = subset
                if plan is not None:
                    try:
                        hr.stamp = plan.hr_stamp(subset)
                        lr.stamp = plan.dirty_stamp(subset)
                    except Exception:
                        pass
                hw.write(hr, index=i)
                dw.write(lr, index=i)
            sources.add_field(i, meta)
            now = time.perf_counter()
            if local == count or (now - last_emit) >= 2.0:
                reporter.set_worker_step(shard_id, local, count, subset)
                last_emit = now
    return subset, shard_id, count


# Worker-process globals, built once per worker by ``_gen_init_worker``.
_W_SIM = None
_W_FWD = None
_W_RECORDS_DIR = ""


def _gen_init_worker(prior_path, image_size, psf_dir,
                     require_empirical_psf, records_dir,
                     galaxy_density_arcmin2=Config.GALAXY_DENSITY_ARCMIN2,
                     galaxy_thinning_max_density_arcmin2=None,
                     star_density_arcmin2=Config.DEFAULT_STAR_DENSITY_ARCMIN2,
                     star_mag_slope=Config.STAR_MAG_SLOPE,
                     star_mag_bright=Config.STAR_MAG_BRIGHT,
                     star_mag_faint=Config.STAR_MAG_FAINT,
                     star_prior_json="",
                     lens_density_arcmin2=Config.LENS_DENSITY_ARCMIN2,
                     lens_sigma_v_min_kms=Config.LENS_SIGMA_V_MIN_KMS,
                     lens_sigma_v_max_kms=Config.LENS_SIGMA_V_MAX_KMS,
                     psf_warp_prob=Config.TRAIN_PSF_WARP_PROB,
                     psf_warp_alpha_max=Config.TRAIN_PSF_WARP_ALPHA_MAX,
                     psf_warp_sigma=Config.TRAIN_PSF_WARP_SIGMA,
                     saturation_mask_prob=Config.TRAIN_SATURATION_MASK_PROB,
                     cosmos_vis_offset_mag=None,
                     cosmos_vis_magnitude_slope=None,
                     cosmos_vis_scatter_mag=None,
                     cosmos_vis_transfer_source="",
                     tng_galaxy_dir=Config.TNG_SKIRT_DIR,
                     tng_properties_csv="",
                     tng_radius_manifest_path="",
                     ) -> None:
    """ProcessPool initializer: build the (small, filtered) catalog +
    simulator + forward model once per worker. The COSMOS2025 FITS is
    memmapped and only the filtered columns are held, so each worker's copy
    is a few MB — no 10 GB-per-worker blow-up."""
    global _W_SIM, _W_FWD, _W_RECORDS_DIR
    transfer = _photometric_transfer_from_args(argparse.Namespace(
        cosmos_vis_offset_mag=cosmos_vis_offset_mag,
        cosmos_vis_magnitude_slope=cosmos_vis_magnitude_slope,
        cosmos_vis_scatter_mag=cosmos_vis_scatter_mag,
        cosmos_vis_transfer_source=cosmos_vis_transfer_source,
    ))
    cat = (CosmosTngPrior(prior_path, photometric_transfer=transfer)
           if prior_path and galaxy_density_arcmin2 > 0.0 else None)
    _W_SIM = SkySimulator(
        cat, SkySimulatorConfig(image_size=image_size,
                                      pixel_scale=Config.DEFAULT_PIXEL_SCALE,
                                      galaxy_density_arcmin2=galaxy_density_arcmin2,
                                      galaxy_thinning_max_density_arcmin2=(
                                          galaxy_thinning_max_density_arcmin2
                                      ),
                                      cosmos_prior_path=prior_path or "",
                                      tng_galaxy_dir=tng_galaxy_dir,
                                      tng_properties_csv=tng_properties_csv,
                                      tng_radius_manifest_path=(
                                          tng_radius_manifest_path
                                          or os.path.join(
                                              Config.DATA_DIR,
                                              "_tng_infographics",
                                              "tng_radius_manifest.json",
                                          )
                                      ),
                                      strict_population_artifacts=True,
                                      star_density_arcmin2=star_density_arcmin2,
                                      star_mag_slope=star_mag_slope,
                                      star_mag_bright=star_mag_bright,
                                      star_mag_faint=star_mag_faint,
                                      star_prior_payload=(
                                          json.loads(star_prior_json)
                                          if star_prior_json else None
                                      ),
                                      lens_density_arcmin2=lens_density_arcmin2,
                                      lens_sigma_v_min_kms=lens_sigma_v_min_kms,
                                      lens_sigma_v_max_kms=lens_sigma_v_max_kms),
    )
    psf_sets = load_all_band_psf_sets(
        psf_dir=psf_dir, require_empirical=require_empirical_psf,
        target_pixel_scale=Config.DEFAULT_PIXEL_SCALE,
    )
    _W_FWD = ObservationSimulator(
        psf_sets_by_band=psf_sets,
        config=ObservationSimulatorConfig(
            add_noise=True,
            psf_warp_prob=float(psf_warp_prob),
            psf_warp_alpha_max=float(psf_warp_alpha_max),
            psf_warp_sigma=float(psf_warp_sigma),
            saturation_mask_prob=float(saturation_mask_prob),
        ),
    )
    _W_RECORDS_DIR = records_dir


def _gen_convolve_shard(task) -> tuple[str, int, int]:
    """Top-level pool entry point → ``_generate_convolve_range`` with the
    worker-global sim/fwd."""
    subset, start, count, shard_id, seed, plan, write_forward = task
    return _generate_convolve_range(
        _W_SIM, _W_FWD, _W_RECORDS_DIR, subset, start, count, shard_id, seed,
        plan=plan, write_forward=write_forward,
    )


def step_generate_and_convolve_parallel(args: argparse.Namespace) -> None:
    _banner(f"STEP 1+2 (parallel ×{args.gen_workers}): generate clean HR + "
            f"forward-model to dirty Euclid LR")
    reporter = Reporter.from_env()
    os.makedirs(args.records_dir, exist_ok=True)
    workers = max(1, int(args.gen_workers))

    prior_path = (None if args.galaxy_density_arcmin2 <= 0.0
                  else args.cosmos_prior)

    # One master seed for the whole parallel step, recorded on the generation
    # run; every shard's RNG is derived from it, so the run replays via --seed.
    run_seed = _resolve_run_seed(args)
    # Provenance (best-effort): one generation run for the whole parallel step;
    # per-subset ids are pre-minted in this parent and shipped to the workers.
    observation_cfg = _observation_config_from_args(args)
    gen_ctx = make_generation_context(
        _generator_config_from_args(args), seed=run_seed)
    _log(f"  run_seed={run_seed}  (replay with --seed {run_seed})")
    _log(
        "  PSF warp: "
        f"p={observation_cfg.psf_warp_prob:g}, "
        f"alpha_max={observation_cfg.psf_warp_alpha_max:g} HR px, "
        f"sigma={observation_cfg.psf_warp_sigma:g} HR px"
    )
    _log(
        "  saturation dark-core probability: "
        f"{observation_cfg.saturation_mask_prob:g}"
    )
    _log(
        "  NISP noise: 4 native 0.30\" dithers → "
        f"{observation_cfg.nisp_resample_kernel} MER resample → 0.10\" (/9 flux)"
    )

    onthefly_train = bool(getattr(args, "onthefly_train", False))
    subsets = (("train", args.ntrain), ("validate", args.nvalid),
               ("test", _ntest(args)))
    targets = _regenerate_splits(args)
    if targets:
        _log("  targeted regeneration: " + ", ".join(targets) +
             " (all other splits remain untouched)")

    # Destructive modes discard every selected subset UP-FRONT (parts + final
    # files + sidecars), before any generation starts. Deleting lazily leaves
    # stale matching-count finals behind when a job hits its wall clock.
    if getattr(args, "force", False) or targets:
        for subset, n in subsets:
            if n > 0 and _split_selected(args, subset):
                _cleanup_parts(args.records_dir, subset)
                _remove_subset_finals(args.records_dir, subset)
    for subset, n in subsets:
        if n <= 0:
            continue
        if not _split_selected(args, subset):
            _log(f"  {subset}: not selected — leaving existing data untouched")
            continue
        # --onthefly-train: the TRAIN split gets clean records ONLY (on-the-fly
        # training reads clean_train and builds LR+target live); validate/test
        # keep the full triple (training validation + evaluation read them).
        write_forward = not (onthefly_train and subset == "train")
        rec_kinds = (("clean", "hr", "dirty") if write_forward
                     else ("clean",))
        # Clean-only train split: drop any stale hr/dirty finals a prior
        # record-mode run left behind so nothing downstream reads them.
        if not write_forward:
            _remove_subset_finals(args.records_dir, subset, kinds=("hr", "dirty"))
        if not _split_forced(args, subset) and _subset_complete(
                args.records_dir, subset, (*rec_kinds, "sources"), n):
            _log(f"  {subset}: already complete ({n} records) — skipping")
            continue

        # Resume: salvage the intact records a killed run left on disk so we
        # only regenerate the shortfall. --force discarded them up-front.
        if _split_forced(args, subset):
            done, used_idx, base_sid = 0, [], 0
        else:
            done, used_idx, base_sid = _salvage_subset(
                args.records_dir, subset, n, kinds=rec_kinds)
            if done:
                _log(f"  {subset}: resuming — salvaged {done}/{n} pairs from a "
                     f"prior run; generating {n - done} more")
        remaining = n - done

        # Pre-mint this subset's three file ids in the parent; every shard
        # stamps with the same plan (records share one id per file).
        plan = None
        if gen_ctx is not None:
            try:
                plan = ShardStampPlan(
                    run_id=gen_ctx.run_id,
                    clean_id=gen_ctx.file_id("clean", subset),
                    hr_id=gen_ctx.file_id("hr", subset),
                    dirty_id=gen_ctx.file_id("dirty", subset),
                )
            except Exception:
                plan = None

        # New fields take indices above every salvaged one and shard ids above
        # every salvaged shard, so the index↔sources map stays unique and new
        # parts never clobber a kept one.
        tasks = []
        if remaining > 0:
            base_idx = (max(used_idx) + 1) if used_idx else 0
            # More shards than workers → finer progress + load balancing AND a
            # finer resume granularity (completed shards survive a later kill).
            # ~256 images/shard, but at least one per worker, never more than
            # there are images.
            n_shards = min(remaining, max(workers, math.ceil(remaining / 256)))
            for k, (start, end) in enumerate(_shard_bounds(remaining, n_shards)):
                if end > start:
                    sid = base_sid + k
                    # SeedSequence material → reproducible, independent per
                    # shard; all derived from the one recorded run_seed.
                    seed = [run_seed, _subset_tag(subset), sid]
                    tasks.append((subset, base_idx + start, end - start,
                                  sid, seed, plan, write_forward))

        t0 = time.perf_counter()
        if tasks:
            reporter.set_stage(f"generate+forward {subset} (×{workers})")
            # Announce the parallel phase: total items + worker count. The
            # workers report their own progress; the consumer sums a cumulative
            # bar and counts active processes.
            reporter.set_parallel(remaining, workers, label=subset)
            _log(f"  {subset}: {remaining} pairs across {len(tasks)} shards, "
                 f"{workers} workers (run_seed={run_seed})")
            with ProcessPoolExecutor(
                max_workers=workers, initializer=_gen_init_worker,
                initargs=(prior_path, args.image_size, args.psf_dir,
                          args.require_empirical_psf, args.records_dir,
                          args.galaxy_density_arcmin2,
                          getattr(
                              args,
                              "galaxy_thinning_max_density_arcmin2",
                              None,
                          ),
                          args.star_density_arcmin2,
                          args.star_mag_slope, args.star_mag_bright,
                          args.star_mag_faint, args.star_prior_json,
                          args.lens_density_arcmin2,
                          args.lens_sigma_v_min_kms, args.lens_sigma_v_max_kms,
                          getattr(args, "psf_warp_prob",
                                  Config.TRAIN_PSF_WARP_PROB),
                          getattr(args, "psf_warp_alpha_max",
                                  Config.TRAIN_PSF_WARP_ALPHA_MAX),
                          getattr(args, "psf_warp_sigma",
                                  Config.TRAIN_PSF_WARP_SIGMA),
                          getattr(args, "saturation_mask_prob",
                                  Config.TRAIN_SATURATION_MASK_PROB),
                          getattr(args, "cosmos_vis_offset_mag", None),
                          getattr(args, "cosmos_vis_magnitude_slope", None),
                          getattr(args, "cosmos_vis_scatter_mag", None),
                          getattr(args, "cosmos_vis_transfer_source", ""),
                          getattr(args, "tng_dir", Config.TNG_SKIRT_DIR),
                          getattr(args, "tng_properties", ""),
                          getattr(args, "tng_radius_manifest", "")),
            ) as pool:
                futs = [pool.submit(_gen_convolve_shard, t) for t in tasks]
                for fut in as_completed(futs):
                    fut.result()   # surface worker exceptions; progress is
                                   # driven by the workers' set_worker_step.

        # Merge salvaged + new shards IN ID ORDER (atomic; parts kept until the
        # whole merge succeeds, so a mid-merge kill resumes cleanly).
        reporter.set_stage(f"merging {subset} shards")
        _merge_subset(args.records_dir, subset, kinds=rec_kinds)

        # Persist the merged file-level artifacts (hr+dirty parent on clean).
        if gen_ctx is not None:
            try:
                clean_id = gen_ctx.file_id("clean", subset)
                gen_ctx.finalize("clean", subset,
                                 tfrecord_path(args.records_dir, f"clean_{subset}"))
                if write_forward:
                    gen_ctx.finalize("hr", subset,
                                     tfrecord_path(args.records_dir, f"hr_{subset}"),
                                     parents=(clean_id,))
                    gen_ctx.finalize(
                        "dirty", subset,
                        tfrecord_path(args.records_dir, f"dirty_{subset}"),
                        parents=(clean_id,))
            except Exception:
                pass
        _log(f"  {subset}: done in {time.perf_counter() - t0:.1f} s → "
             + ("clean only" if not write_forward else "clean + hr + dirty")
             + f" {subset}")


# ---------------------------------------------------------------------------
# Step 3: train WDSR (4-channel in, 4-channel out)
# ---------------------------------------------------------------------------

def step_train(args: argparse.Namespace) -> None:
    _banner(f"STEP 3: Train WDSR  ({args.steps} steps, batch {args.batch_size}, "
            f"eval every {args.evaluate_every})")

    scale = Config.DEFAULT_REBIN_FACTOR

    print(f"  TFRecords:   {args.records_dir}")
    print(f"  Checkpoints: {args.checkpoint_dir}")

    # Same --seed knob as generation: a fixed value makes the run reproducible
    # (recorded on a Process.training); -1 keeps fresh entropy.
    run_seed = _resolve_run_seed(args)
    print(f"  run_seed={run_seed}  (replay with --seed {run_seed})")
    m = Model(args.checkpoint_dir, scale=scale,
              num_res_blocks=args.num_res_blocks, seed=run_seed)
    m.train(
        tfrecord_path(args.records_dir, "dirty_train"),
        tfrecord_path(args.records_dir, "clean_train"),
        steps=args.steps,
        evaluate_every=args.evaluate_every,
    )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> int:
    # ``t_script_start`` brackets the "init" stage — everything that
    # happened between Python booting up and the first real stage
    # starting (module imports, dataset loaders ready, etc).
    t_script_start = time.time()
    t0_perf = time.perf_counter()
    args = parse_args()
    print(f"Pipeline started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  args = {vars(args)}")

    # Per-stage timings CSV — default sits next to the SLURM .out file
    # via ``$EUCLID_POLISH_DATA_DIR/images/records_v2/stages_<jobid>.csv``
    # so the FASRC dashboard can fetch it back without knowing the
    # exact records-dir layout.
    slurm_jobid = os.environ.get("SLURM_JOB_ID", "local")
    stages_path = args.stages_csv or os.path.join(
        args.records_dir, f"stages_{slurm_jobid}.csv",
    )
    timer = StageTimer(
        csv_path=stages_path,
        jobid=slurm_jobid,
        params={
            "n_train": args.ntrain, "n_valid": args.nvalid,
            "n_test": _ntest(args),
            "regenerate_splits": ",".join(_regenerate_splits(args)),
            "image_size": args.image_size, "batch_size": args.batch_size,
            "steps": args.steps,
        },
    )
    print(f"  stage timings → {stages_path}")

    # ``init`` is everything from ``t_script_start`` up to the first
    # stage. Mark it now so it's persisted even if a later stage fails.
    timer.mark("init", params_dependent=False,
               started_at=t_script_start, ended_at=time.time())

    # Parallel combined path: one process per index range does generate +
    # forward in a single pass, then shards are concatenated. Only when both
    # stages are wanted (the standalone re-convolve case stays serial).
    parallel_gen = (int(getattr(args, "gen_workers", 1) or 1) > 1
                    and not args.skip_generate and not args.skip_convolve)
    if parallel_gen:
        with timer.stage("generate", params_dependent=True):
            step_generate_and_convolve_parallel(args)
    else:
        if not args.skip_generate:
            with timer.stage("generate", params_dependent=True):
                step_generate(args)
        else:
            print("STEP 1 skipped (--skip-generate)")

        if not args.skip_convolve:
            with timer.stage("convolve", params_dependent=True):
                step_convolve(args)
        else:
            print("STEP 2 skipped (--skip-convolve)")

    if not args.skip_train:
        with timer.stage("train", params_dependent=True):
            step_train(args)
    else:
        print("STEP 3 skipped (--skip-train)")

    dt = time.perf_counter() - t0_perf
    print(f"\nDone in {dt/60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
