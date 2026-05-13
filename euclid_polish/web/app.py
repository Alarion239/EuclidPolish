"""
Flask app factory + routes for the EuclidPolish web UI.
"""

from __future__ import annotations

# Force the non-interactive matplotlib backend BEFORE anything that
# imports matplotlib gets a chance to lock in macOS's GUI backend. The
# job registry spawns worker threads for inference / plotting; macOS's
# default backend only works from the main thread and would otherwise
# crash with "Cannot create a GUI FigureManager outside the main thread"
# whenever a job called ``plt.figure``.
import matplotlib  # noqa: E402  (must precede any other matplotlib user)
matplotlib.use("Agg")

import glob
import io
import json
import os
import re
import shlex
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from flask import (
    Flask, Response, abort, jsonify, render_template, request, send_file,
    stream_with_context, url_for,
)

from euclid_polish.config import BandConfig, Config
from euclid_polish.euclid.catalog import StarCatalog
from euclid_polish.euclid.psf_library import (
    load_all_band_psfs, psf_inventory, psf_path_for_band,
)
from euclid_polish.euclid.types import PSF
from euclid_polish.web import (
    fasrc_config, fasrc_jobs, fasrc_log_parser, git_ops,
)
from euclid_polish.web.fasrc_mirror import MIRROR
from euclid_polish.web.jobs import REGISTRY
from euclid_polish.web.remote import (
    BitwardenError, RemoteState, SSHConfig, SSHError, SSHSession, STATE,
)


# Regex guard for cutout filenames: alphanumerics, underscores, dashes, dots.
# Rejects any path-traversal sequences ("..", "/").
_CUTOUT_FNAME_RE = re.compile(r"^[A-Za-z0-9_.\-]+\.fits$")


# ---------------------------------------------------------------------------
# Read-only status helpers
# ---------------------------------------------------------------------------

def _catalog_status() -> Dict[str, Any]:
    cat = StarCatalog()
    if not cat.exists():
        return {"present": False}
    summary = cat.get_summary()
    return {"present": True, "summary": summary, "path": cat.catalog_path}


def _psf_status() -> Dict[str, Any]:
    inv = psf_inventory()
    bands = []
    for b in Config.BANDS:
        path = inv.get(b.name)
        item = {
            "name":           b.name,
            "fwhm":           b.psf_fwhm_arcsec,
            "oversampling":   b.epsf_oversampling,
            "epsf_pixel_scale": b.epsf_pixel_scale_arcsec,
            "empirical":      path is not None,
            "path":           path,
        }
        if path:
            try:
                psf = PSF.from_fits(path)
                item["shape"]      = list(psf.data.shape)
                item["pixel_scale"]= psf.pixel_scale
            except Exception as e:
                item["error"] = str(e)
        bands.append(item)
    return {"bands": bands}


def _tfrecords_status() -> Dict[str, Any]:
    out = {"dir": Config.RECORDS_DIR_V2, "files": []}
    if os.path.isdir(Config.RECORDS_DIR_V2):
        for fname in sorted(os.listdir(Config.RECORDS_DIR_V2)):
            full = os.path.join(Config.RECORDS_DIR_V2, fname)
            if not os.path.isfile(full):
                continue
            try:
                size_mb = os.path.getsize(full) / 1e6
            except OSError:
                size_mb = 0
            out["files"].append({"name": fname, "size_mb": round(size_mb, 1)})
    return out


def _checkpoints_status() -> Dict[str, Any]:
    out = {"dir": Config.DEFAULT_CHECKPOINT_DIR, "files": []}
    if os.path.isdir(Config.DEFAULT_CHECKPOINT_DIR):
        for fname in sorted(os.listdir(Config.DEFAULT_CHECKPOINT_DIR)):
            full = os.path.join(Config.DEFAULT_CHECKPOINT_DIR, fname)
            if os.path.isfile(full):
                size_mb = os.path.getsize(full) / 1e6
                out["files"].append({"name": fname, "size_mb": round(size_mb, 1)})
    return out


def _cutout_layout_status(output_dir: str = Config.DEFAULT_OUTPUT_DIR,
                          preview_n: int = 8) -> Dict[str, Any]:
    """Count cutout FITS files per band under ``output_dir/cutouts/<band>/``.

    Also returns up to ``preview_n`` filenames per band for inline thumbnails.
    """
    cutout_root = os.path.join(output_dir, "cutouts")
    bands_info = []
    total = 0
    for band in Config.BANDS:
        band_dir = Config.cutout_dir_for_band(band.name, root=cutout_root)
        files: list[str] = []
        if os.path.isdir(band_dir):
            files = sorted(
                f for f in os.listdir(band_dir)
                if f.lower().endswith(".fits") and _CUTOUT_FNAME_RE.match(f)
            )
        n = len(files)
        total += n
        bands_info.append({
            "name": band.name, "dir": band_dir, "count": n,
            "native_scale": band.pixel_scale_lr_arcsec,
            "preview": files[:preview_n],
        })
    return {"root": cutout_root, "bands": bands_info, "total": total}


def _list_vis_pngs() -> list[Dict[str, Any]]:
    """Recent PNGs under data/vis/, newest first."""
    pngs: list[Dict[str, Any]] = []
    if not os.path.isdir(Config.VIS_DIR):
        return pngs
    for dirpath, _, files in os.walk(Config.VIS_DIR):
        for fname in files:
            if not fname.lower().endswith(".png"):
                continue
            full = os.path.join(dirpath, fname)
            try:
                mtime = os.path.getmtime(full)
                size_kb = os.path.getsize(full) / 1024
            except OSError:
                continue
            rel = os.path.relpath(full, Config.VIS_DIR)
            pngs.append({
                "rel":     rel,
                "mtime":   mtime,
                "size_kb": round(size_kb, 1),
            })
    pngs.sort(key=lambda d: d["mtime"], reverse=True)
    return pngs


# ---------------------------------------------------------------------------
# Background-job target functions
# ---------------------------------------------------------------------------

def _job_generate(cap, image_size: int, n_train: int, n_valid: int,
                  lens_density: float) -> Dict[str, Any]:
    """Run the multi-band clean-field generator with live progress."""
    from euclid_polish.sky.cosmos2025 import open_cosmos2025
    from euclid_polish.sky.multiband_generator import (
        MultiBandGeneratorConfig, MultiBandSimulator,
    )
    from euclid_polish.sky.tfrecord import open_multiband_writer

    print(f"Generating clean fields: image_size={image_size}, n_train={n_train}, "
          f"n_valid={n_valid}, lens_density={lens_density}")
    catalog = open_cosmos2025()
    print(f"Catalog: {type(catalog).__name__} ({len(catalog)} galaxies)")
    cfg = MultiBandGeneratorConfig(
        image_size=image_size,
        pixel_scale=Config.DEFAULT_PIXEL_SCALE,
        lens_density_arcmin2=lens_density,
    )
    sim = MultiBandSimulator(catalog, cfg)
    os.makedirs(Config.RECORDS_DIR_V2, exist_ok=True)
    result: Dict[str, Any] = {}
    total_n = n_train + n_valid
    done = 0
    for subset, n in (("train", n_train), ("validate", n_valid)):
        # Entropy-seeded master RNG so each click of the web button
        # produces fresh fields. Seed logged for after-the-fact replay.
        master_seed = int.from_bytes(os.urandom(8), "little")
        rng = np.random.default_rng(master_seed)
        print(f"  {subset}: master_seed={master_seed}")
        # Stream each image to disk to bound RSS — accumulating a list
        # of 6400 510² × 4-channel fields would cost ~26 GB.
        with open_multiband_writer(f"clean_{subset}",
                                   records_dir=Config.RECORDS_DIR_V2) as w:
            for i in range(n):
                sky, _ = sim.simulate_field(rng)
                sky.index = i
                sky.subset = subset
                w.write(sky, index=i)
                done += 1
                cap.tick(done, total_n, f"generating {subset} {i+1}/{n}")
            path, count = w.path, w.count
        print(f"  ✓ {path}")
        result[subset] = {"path": path, "count": count}
    return result


def _job_forward(cap) -> Dict[str, Any]:
    """Apply the multi-band forward model with progress tracking."""
    import tensorflow as tf
    from euclid_polish.euclid.psf_library import load_all_band_psfs
    from euclid_polish.sky.multiband_forward import (
        MultiBandForward, MultiBandForwardConfig,
    )
    from euclid_polish.sky.tfrecord import (
        open_multiband_writer, tfrecord_path,
    )
    from euclid_polish.sky.types import MultiBandSkyImage

    psfs = load_all_band_psfs()
    print("Loaded PSFs:")
    for name, psf in psfs.items():
        print(f"  {name:5s} shape={psf.data.shape} scale={psf.pixel_scale:.3f}\"")
    fwd = MultiBandForward(psfs_by_band=psfs,
                           config=MultiBandForwardConfig(add_noise=True))

    # Pre-count total for the progress bar
    totals = {}
    for subset in ("train", "validate"):
        path = tfrecord_path(Config.RECORDS_DIR_V2, f"clean_{subset}")
        if os.path.exists(path):
            totals[subset] = sum(1 for _ in tf.data.TFRecordDataset(path))
    grand_total = sum(totals.values())
    done = 0
    result: Dict[str, Any] = {}

    for subset in ("train", "validate"):
        clean = tfrecord_path(Config.RECORDS_DIR_V2, f"clean_{subset}")
        if not os.path.exists(clean):
            print(f"  ⚠️  no clean_{subset} on disk, skipping")
            continue
        # Entropy-seeded forward-model RNG — fresh noise/CR/streak draw
        # every run, with the master seed logged for replay.
        master_seed = int.from_bytes(os.urandom(8), "little")
        rng = np.random.default_rng(master_seed)
        print(f"  forward {subset}: master_seed={master_seed}")
        # Stream LR + HR pairs to disk — at 6400 fields the in-memory
        # list approach was costing ~13 GB and risking another OOM after
        # the generator's fix.
        n = totals.get(subset, 0)
        with open_multiband_writer(f"hr_{subset}",
                                   records_dir=Config.RECORDS_DIR_V2) as hr_w, \
             open_multiband_writer(f"dirty_{subset}",
                                   records_dir=Config.RECORDS_DIR_V2) as lr_w:
            for i, raw in enumerate(tf.data.TFRecordDataset(clean)):
                hr_4ch = MultiBandSkyImage.from_tfrecord(raw)
                lr, hr = fwd.process(hr_4ch, rng=rng)
                lr.index = i; hr.index = i
                lr.subset = subset; hr.subset = subset
                hr_w.write(hr, index=i)
                lr_w.write(lr, index=i)
                done += 1
                cap.tick(done, grand_total, f"forward-model {subset}")
        result[subset] = {"n": n}
        print(f"  ✓ {subset}: kept clean ({len(hr_imgs)} 4-band) + "
              f"wrote hr ({len(hr_imgs)} 1-band VIS target) + "
              f"dirty ({len(lr_imgs)} LR-4ch)")
    return result


def _job_query_brightest(cap, num_stars: int, output_dir: str,
                         magnitude_limit: Optional[float] = None,
                         magnitude_min: Optional[float] = None) -> Dict[str, Any]:
    """Query Euclid archive for the N brightest stars in the given mag window."""
    window = []
    if magnitude_min is not None:   window.append(f"mag>{magnitude_min}")
    if magnitude_limit is not None: window.append(f"mag<{magnitude_limit}")
    win_str = (" [" + ", ".join(window) + "]") if window else ""
    print(f"querying {num_stars} brightest stars{win_str} into {output_dir}")
    cat = StarCatalog(output_dir)
    result = cat.query_brightest_stars(
        num_stars=num_stars,
        magnitude_limit=magnitude_limit,
        magnitude_min=magnitude_min,
    )
    print(result["message"])
    return result


def _job_query_region(cap, ra: float, dec: float, radius: float,
                      mag_limit: float, output_dir: str,
                      magnitude_min: Optional[float] = None) -> Dict[str, Any]:
    """Query Euclid archive for stars in a cone, optionally excluding bright ones."""
    extra = f" mag>{magnitude_min}" if magnitude_min is not None else ""
    print(f"querying ra={ra}, dec={dec}, radius={radius}°, mag<{mag_limit}"
          f"{extra} → {output_dir}")
    cat = StarCatalog(output_dir)
    result = cat.query_euclid_catalog(
        ra=ra, dec=dec, radius=radius, magnitude_limit=mag_limit,
        magnitude_min=magnitude_min,
    )
    print(result["message"])
    return result


def _job_download_cutouts(cap, bands: list[str], cutout_size_vis_pixels: int,
                          max_workers: int, output_dir: str) -> Dict[str, Any]:
    """Download cutouts for each selected band; tqdm-driven progress bar."""
    from euclid_polish.euclid.downloader import (
        DownloadConfig, EuclidCutoutDownloader,
    )

    cat = StarCatalog(output_dir)
    arcsec = cutout_size_vis_pixels * Config.BAND_VIS.pixel_scale_lr_arcsec
    print(f"selected bands: {bands}")
    print(f"angular field: {arcsec:.2f}\"  (= {cutout_size_vis_pixels} VIS px)")
    per_band: dict[str, dict] = {}
    for band_name in bands:
        native = Config.get_band(band_name).cutout_size_for_arcsec(arcsec)
        print(f"\n=== {band_name}  native_size = {native} ===")
        cfg = DownloadConfig.for_band(
            band_name,
            cutout_size_vis_pixels=cutout_size_vis_pixels,
            max_workers=max_workers,
        )
        dl = EuclidCutoutDownloader(cat, cfg)
        # The downloader uses tqdm internally; hook drives the UI bar.
        with cap.tqdm_hook(label=f"download {band_name}"):
            r = dl.download(show_progress=True)
        per_band[band_name] = r
        print(f"  → downloaded {r['downloaded']}, valid={r['valid']}, "
              f"corrupted={r['corrupted']}, failed={r.get('failed', 0)}")
    return per_band


def _job_extract_psf(cap, band_name: str, num_stars: int,
                     cutout_size: int, output_size: int | None,
                     output_dir: str, psf_dir: str) -> Dict[str, Any]:
    """Extract a per-band empirical ePSF from local cutouts."""
    from euclid_polish.euclid.psf_extractor import (
        PSFExtractionConfig, PSFExtractor,
    )

    band = Config.get_band(band_name)
    cutout_dir = Config.cutout_dir_for_band(
        band.name, root=os.path.join(output_dir, "cutouts"),
    )
    if not os.path.isdir(cutout_dir):
        raise FileNotFoundError(f"no cutout dir for {band.name}: {cutout_dir}")
    cfg = PSFExtractionConfig(
        psf_size=cutout_size - 1 if cutout_size % 2 == 0 else cutout_size - 2,
        output_size=output_size,
        oversampling=band.epsf_oversampling,
        progress_bar=False,
    )
    extractor = PSFExtractor(cfg)
    files = extractor.get_cutout_files(cutout_dir, cutout_size=cutout_size)
    if not files:
        raise FileNotFoundError(
            f"no cutouts of size {cutout_size} in {cutout_dir}"
        )
    selected = extractor.select_files(files, num_stars=num_stars)
    print(f"using {len(selected)} of {len(files)} stars for {band.name}")
    with cap.tqdm_hook(label=f"EPSFBuilder {band_name}"):
        extractor.build_epsf(selected)
    psf = extractor.to_psf(band.epsf_pixel_scale_arcsec)
    os.makedirs(psf_dir, exist_ok=True)
    saved = psf.save(psf_dir, filename=band.psf_fits_filename)
    print(f"saved {saved}: shape={psf.data.shape}, "
          f"pixel_scale={psf.pixel_scale:.4f}\"/pix")
    return {"path": saved, "shape": list(psf.data.shape),
            "pixel_scale": psf.pixel_scale}


def _job_check_integrity(cap, output_dir: str) -> Dict[str, Any]:
    """Scan every cutout under output_dir/cutouts/<band>/ for corruption."""
    import glob
    from euclid_polish.euclid.validator import FitsValidator

    validator = FitsValidator()
    cutout_root = os.path.join(output_dir, "cutouts")
    bands = Config.LR_INPUT_BAND_NAMES
    summary: dict[str, dict] = {}
    # Pre-count for the unified progress bar
    all_files = []
    for name in bands:
        band_dir = Config.cutout_dir_for_band(name, root=cutout_root)
        if os.path.isdir(band_dir):
            all_files.extend((name, f) for f in glob.glob(os.path.join(band_dir, "*.fits")))
    total = len(all_files)
    done = 0
    for name in bands:
        band_dir = Config.cutout_dir_for_band(name, root=cutout_root)
        if not os.path.isdir(band_dir):
            summary[name] = {"valid": 0, "corrupted": 0, "absent": True}
            continue
        files = glob.glob(os.path.join(band_dir, "*.fits"))
        ok = bad = 0
        for f in files:
            is_valid, _ = validator.validate_basic_integrity(f)
            if is_valid:
                ok += 1
            else:
                bad += 1
            done += 1
            cap.tick(done, total, f"checking {name}")
        summary[name] = {"valid": ok, "corrupted": bad, "absent": False}
        print(f"{name}: valid={ok}, corrupted={bad}, total={len(files)}")
    return summary


def _job_train(cap, steps: int, batch_size: int, num_res_blocks: int,
               evaluate_every: int, checkpoint_dir: str) -> Dict[str, Any]:
    """Train the WDSR model on the v2 multi-band TFRecords."""
    from tf_keras.optimizers.schedules import PiecewiseConstantDecay
    from euclid_polish.training import Trainer
    from euclid_polish.training.data_multiband import MultiBandEuclidDataset
    from euclid_polish.training.models.wdsr import wdsr

    scale = Config.DEFAULT_REBIN_FACTOR
    print(f"training: steps={steps}, batch={batch_size}, ckpt={checkpoint_dir}")
    train = MultiBandEuclidDataset(scale=scale, subset="train").dataset(
        batch_size=batch_size, random_transform=True,
    )
    valid = MultiBandEuclidDataset(scale=scale, subset="validate").dataset(
        batch_size=1, random_transform=False, repeat_count=1,
    )
    model = wdsr(scale=scale, num_res_blocks=num_res_blocks,
                 nchan_in=Config.NUM_LR_CHANNELS, nchan_out=Config.NUM_HR_CHANNELS)
    schedule = PiecewiseConstantDecay(boundaries=[200_000], values=[1e-3, 5e-4])
    trainer = Trainer(model=model, learning_rate=schedule,
                      checkpoint_dir=checkpoint_dir)
    # Trainer.train uses tqdm internally — the hook drives our progress bar.
    with cap.tqdm_hook(label="training"):
        trainer.train(train, valid, steps=steps,
                      evaluate_every=evaluate_every)
    return {"checkpoint_dir": checkpoint_dir, "steps": steps}


def _job_evaluate(cap, checkpoint_dir: str, num_res_blocks: int) -> Dict[str, Any]:
    """Run validation PSNRs against the latest checkpoint."""
    import tensorflow as tf
    from euclid_polish.training import Trainer
    from euclid_polish.training.data_multiband import MultiBandEuclidDataset
    from euclid_polish.training.inference import load_model_from_checkpoint

    scale = Config.DEFAULT_REBIN_FACTOR
    print(f"evaluating checkpoints under {checkpoint_dir}")
    if not tf.train.latest_checkpoint(checkpoint_dir):
        raise FileNotFoundError(f"no checkpoint in {checkpoint_dir}")
    model = load_model_from_checkpoint(
        checkpoint_dir, scale, num_res_blocks,
        nchan_in=Config.NUM_LR_CHANNELS, nchan_out=Config.NUM_HR_CHANNELS,
    )
    valid_ds = MultiBandEuclidDataset(scale=scale, subset="validate").dataset(
        batch_size=1, random_transform=False, repeat_count=1,
    )
    metrics = Trainer(model=model, checkpoint_dir=checkpoint_dir).evaluate(valid_ds)
    out = {k: float(v.numpy()) for k, v in metrics.items()}
    print(f"  psnr_stretched = {out['psnr_stretched']:.3f} dB")
    print(f"  psnr_raw       = {out['psnr_raw']:.3f} dB")
    return out


def _resolve_training_log(checkpoint_dir: str) -> Optional[str]:
    """Pick the current ``training_log.csv`` or fall back to legacy
    ``training_log.jsonl`` so logs from runs before the CSV switch still
    plot. Returns the path that exists, or None if neither does."""
    for name in ("training_log.csv", "training_log.jsonl"):
        p = os.path.join(checkpoint_dir, name)
        if os.path.exists(p):
            return p
    return None


def _job_plot_training_log(cap, checkpoint_dir: str) -> Dict[str, Any]:
    """Render the training-log PNG (loss + PSNR vs step)."""
    from euclid_polish.training.log_plot import plot_training_log
    log_path = _resolve_training_log(checkpoint_dir)
    if log_path is None:
        raise FileNotFoundError(
            f"no training_log.csv or .jsonl in {checkpoint_dir}"
        )
    out_path = os.path.join(Config.VIS_DIR, "training_log.png")
    os.makedirs(Config.VIS_DIR, exist_ok=True)
    plot_training_log(log_path, out_path)
    print(f"wrote {out_path}")
    return {"path": out_path}


def _job_reconstruct(cap, checkpoint_dir: str, num_res_blocks: int,
                     subset: str, n_images: int) -> Dict[str, Any]:
    """Run inference on N random LR records; render side-by-side PNGs."""
    import tensorflow as tf
    from euclid_polish.sky.tfrecord import (
        read_multiband_skyimages, tfrecord_path,
    )
    from euclid_polish.training.inference import (
        load_model_from_checkpoint, reconstruct, plot_reconstruction,
    )

    scale = Config.DEFAULT_REBIN_FACTOR
    if not tf.train.latest_checkpoint(checkpoint_dir):
        raise FileNotFoundError(f"no checkpoint in {checkpoint_dir}")
    model = load_model_from_checkpoint(
        checkpoint_dir, scale, num_res_blocks,
        nchan_in=Config.NUM_LR_CHANNELS, nchan_out=Config.NUM_HR_CHANNELS,
    )

    lr_path = tfrecord_path(Config.RECORDS_DIR_V2, f"dirty_{subset}")
    # ``hr_<subset>`` is the 1-channel VIS target the trainer fit
    # against; ``clean_<subset>`` is the 4-channel HR cube (kept for
    # inspection only). Prefer the 1-channel record so the residual
    # plot's ``hr_data - sr_data`` shapes match; fall back to slicing
    # channel 0 out of the clean record on older datasets that don't
    # carry a separate ``hr_<subset>`` file.
    hr_path     = tfrecord_path(Config.RECORDS_DIR_V2, f"hr_{subset}")
    clean_path  = tfrecord_path(Config.RECORDS_DIR_V2, f"clean_{subset}")
    if not os.path.exists(lr_path):
        raise FileNotFoundError(f"no records in {Config.RECORDS_DIR_V2}")
    lr_records = read_multiband_skyimages(lr_path, num_images=10_000)
    if os.path.exists(hr_path):
        hr_records = read_multiband_skyimages(hr_path, num_images=10_000)
    elif os.path.exists(clean_path):
        hr_records = read_multiband_skyimages(clean_path, num_images=10_000)
    else:
        hr_records = []
    hr_by_idx = {h.index: h for h in hr_records}
    n = min(n_images, len(lr_records))
    rng = np.random.default_rng()
    chosen = rng.choice(len(lr_records), size=n, replace=False)
    out_dir = Config.VIS_RECONSTRUCTION_DIR
    os.makedirs(out_dir, exist_ok=True)
    out_paths = []
    for k, i in enumerate(chosen):
        lr_img = lr_records[i]
        lr_data, sr_data = reconstruct(model, lr_img.data)
        hr_data = None
        if lr_img.index in hr_by_idx:
            raw = hr_by_idx[lr_img.index].data
            # plot_reconstruction expects a 2-D VIS HR. Slice channel 0
            # when the record carries the legacy 4-channel cube.
            if raw.ndim == 3 and raw.shape[-1] >= 1:
                hr_data = raw[..., 0]
            elif raw.ndim == 2:
                hr_data = raw
        out = os.path.join(out_dir, f"reconstruct_idx{lr_img.index:04d}.png")
        plot_reconstruction(lr_data, sr_data, hr_data=hr_data, output_path=out)
        out_paths.append(out)
        cap.tick(k + 1, n, f"reconstructing idx {lr_img.index}")
        print(f"  ✓ {out}")
    return {"output_dir": out_dir, "n": len(out_paths), "paths": out_paths}


def _job_reconstruct_euclid_cutout(
    cap,
    ra: float,
    dec: float,
    checkpoint_dir: str,
    num_res_blocks: int,
    cutout_size_vis_pixels: int,
) -> Dict[str, Any]:
    """Download a 4-band Euclid cutout at one sky position, run SR, save PNG.

    Unlike ``_job_reconstruct`` (which iterates over synthetic TFRecords),
    this fetches a real cutout at ``(ra, dec)`` for every band, converts
    each from the archive's ADU s⁻¹ units to electrons-over-the-stack (so
    the model sees the same scale it was trained on), stacks the four
    bands into ``(H, W, 4)``, and runs the model. The HR target is
    unknown for real data, so the output plot is LR/SR-only.
    """
    import tempfile
    import tensorflow as tf
    from astropy.io import fits
    from euclid_polish.euclid.downloader import fetch_cutout_at
    from euclid_polish.training.inference import (
        load_model_from_checkpoint, reconstruct, plot_reconstruction,
    )

    if not tf.train.latest_checkpoint(checkpoint_dir):
        raise FileNotFoundError(f"no checkpoint in {checkpoint_dir}")
    scale = Config.DEFAULT_REBIN_FACTOR
    model = load_model_from_checkpoint(
        checkpoint_dir, scale, num_res_blocks,
        nchan_in=Config.NUM_LR_CHANNELS, nchan_out=Config.NUM_HR_CHANNELS,
    )

    # Fetch each band into a temp dir; per-band MAGZERO from each header
    # drives the per-band ADU/s → electrons conversion so the model sees
    # the same calibration scale the simulator uses.
    bands_data: Dict[str, np.ndarray] = {}
    bands_info: Dict[str, Dict[str, Any]] = {}
    with tempfile.TemporaryDirectory(prefix="euclid_infer_") as tmpdir:
        for k, band_name in enumerate(Config.LR_INPUT_BAND_NAMES):
            cap.tick(k, len(Config.LR_INPUT_BAND_NAMES) + 1,
                     f"downloading {band_name} cutout")
            band = Config.get_band(band_name)
            outf = os.path.join(tmpdir, f"{band_name}.fits")
            ok, err = fetch_cutout_at(
                ra=ra, dec=dec, band_name=band_name, output_file=outf,
                cutout_size_vis_pixels=cutout_size_vis_pixels,
            )
            if not ok:
                raise RuntimeError(f"{band_name}: {err}")
            with fits.open(outf) as hdul:
                arr = hdul[0].data.astype(np.float32)
                header = hdul[0].header
            magzero = float(header.get("MAGZERO",
                                       band.sim_zeropoint_e))
            # m_AB = MAGZERO - 2.5·log10(F_archive)  (archive units = ADU/s)
            # m_AB = ZP_stack_e - 2.5·log10(F_e_over_stack)
            #  ⇒ F_e = F_archive · 10^((ZP_stack_e − MAGZERO)/2.5)
            adu_to_e = 10 ** ((band.sim_zeropoint_e - magzero) / 2.5)
            data_e = (arr * adu_to_e).astype(np.float32)
            bands_data[band_name] = data_e
            bands_info[band_name] = {
                "shape":      data_e.shape,
                "magzero":    magzero,
                "adu_to_e":   adu_to_e,
                "pix_mean":   float(np.mean(data_e)),
                "pix_std":    float(np.std(data_e)),
            }
            print(f"  {band_name}: shape={data_e.shape}, MAGZERO={magzero:.3f}, "
                  f"ADU/s→e⁻ factor={adu_to_e:.1f}")

    # All four cutouts must land on the same VIS-LR grid (the MER mosaic
    # pipeline delivers every band at 0.10″/pix). Anything else is a bug
    # in the archive query we should not silently paper over.
    shapes = {n: bands_data[n].shape for n in Config.LR_INPUT_BAND_NAMES}
    base_shape = shapes["VIS"]
    if any(s != base_shape for s in shapes.values()):
        raise RuntimeError(
            f"per-band shapes disagree: {shapes}; expected all bands at "
            "the same VIS LR grid (0.10″/pix)."
        )

    lr_cube = np.stack(
        [bands_data[n] for n in Config.LR_INPUT_BAND_NAMES], axis=-1,
    )   # (H, W, 4)
    cap.tick(len(Config.LR_INPUT_BAND_NAMES),
             len(Config.LR_INPUT_BAND_NAMES) + 1, "running model")
    _, sr_data = reconstruct(model, lr_cube)
    lr_vis = lr_cube[..., 0]

    out_dir = Config.VIS_RECONSTRUCTION_DIR
    os.makedirs(out_dir, exist_ok=True)
    tag = f"ra{ra:.4f}_dec{dec:+.4f}".replace("+", "p").replace("-", "m")
    out_path = os.path.join(out_dir, f"euclid_{tag}.png")
    plot_reconstruction(lr_vis, sr_data, hr_data=None, output_path=out_path)
    cap.tick(len(Config.LR_INPUT_BAND_NAMES) + 1,
             len(Config.LR_INPUT_BAND_NAMES) + 1, "saved PNG")
    print(f"  ✓ {out_path}")
    return {
        "output_path":  out_path,
        "ra":           ra,
        "dec":          dec,
        "cutout_size":  cutout_size_vis_pixels,
        "bands":        bands_info,
    }


def _job_viz_star_positions(cap, output_dir: str) -> Dict[str, Any]:
    from euclid_polish.visualization.methods import draw_star_positions
    cat = StarCatalog(output_dir)
    data = cat.load()
    out = Config.VIS_STAR_POSITIONS
    os.makedirs(os.path.dirname(out), exist_ok=True)
    draw_star_positions(data["stars"], out)
    print(f"wrote {out}")
    return {"path": out}


def _job_viz_psf(cap, band_name: str | None, psf_dir: str) -> Dict[str, Any]:
    """Render the four-band PSF inspection panel (or one band)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from euclid_polish.euclid.psf_library import load_all_band_psfs

    psfs = load_all_band_psfs(target_pixel_scale=Config.DEFAULT_PIXEL_SCALE,
                              psf_dir=psf_dir)
    bands = [band_name] if band_name else list(psfs.keys())
    n = len(bands)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, name in zip(axes, bands):
        psf = psfs[name]
        d = np.clip(psf.data, 1e-8, None)
        ax.imshow(np.log10(d), cmap="viridis", origin="lower")
        ax.set_title(f"{name}  shape={psf.data.shape}  "
                     f"@ {psf.pixel_scale:.3f}\"/pix")
        ax.set_xticks([]); ax.set_yticks([])
    out = os.path.join(Config.VIS_PSF_DIR,
                       f"psf_{band_name or 'all'}.png")
    os.makedirs(Config.VIS_PSF_DIR, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return {"path": out}


def _job_demo_lens(cap, n_lenses: int) -> Dict[str, Any]:
    """Quick end-to-end demo: generate one field, run forward, save PNGs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    from euclid_polish.euclid.psf_library import load_all_band_psfs
    from euclid_polish.sky.cosmos2025 import open_cosmos2025
    from euclid_polish.sky.multiband_forward import (
        MultiBandForward, MultiBandForwardConfig,
    )
    from euclid_polish.sky.multiband_generator import (
        MultiBandGeneratorConfig, MultiBandSimulator,
    )

    out_dir = os.path.join(Config.VIS_DIR, "web_demo")
    os.makedirs(out_dir, exist_ok=True)
    cat = open_cosmos2025()
    sim = MultiBandSimulator(
        cat, MultiBandGeneratorConfig(image_size=510),
    )

    rng = np.random.default_rng()
    sky, meta = sim.simulate_field(rng, n_lenses=n_lenses)
    print(f"generated 510² field: {meta['n_galaxies']} gal, "
          f"{meta['n_stars']} stars, {meta['n_lenses']} lenses")
    psfs = load_all_band_psfs()
    fwd = MultiBandForward(psfs_by_band=psfs,
                           config=MultiBandForwardConfig(add_noise=True))
    lr, hr = fwd.process(sky, rng=np.random.default_rng(42))

    bands = Config.LR_INPUT_BAND_NAMES
    lens_positions = [(L["x_pix"], L["y_pix"]) for L in meta["lenses"]]
    lens_theta_E   = [L["theta_E_arcsec"] for L in meta["lenses"]]

    def _save_panel(data_4ch, title, fname, *, scale_per_band=True, lens_px_scale=0.05):
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        for ax, k in zip(axes.flat, range(4)):
            scale = Config.get_band(bands[k]).asinh_stretch_scale_e
            stretched = np.arcsinh(data_4ch[..., k] / scale)
            lo, hi = np.percentile(stretched, [1.0, 99.7])
            ax.imshow(stretched, cmap="gray_r", origin="lower",
                      vmin=lo, vmax=hi)
            ax.set_title(f"{bands[k]}", fontsize=11)
            ax.set_xticks([]); ax.set_yticks([])
            for (xp, yp), te in zip(lens_positions, lens_theta_E):
                ax.add_patch(mpatches.Circle(
                    (xp / (lens_px_scale / 0.05), yp / (lens_px_scale / 0.05)),
                    te / lens_px_scale,
                    fill=False, ec="red", lw=1.0, ls="--",
                ))
        fig.suptitle(title, fontsize=12)
        fig.tight_layout()
        out = os.path.join(out_dir, fname)
        fig.savefig(out, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ wrote {out}")
        return out

    hr_path = _save_panel(sky.data, "HR clean 510² (0.05\"/pix)",
                          "demo_hr.png", lens_px_scale=0.05)
    lr_path = _save_panel(lr.data,  "LR dirty 255² (0.10\"/pix, +Poisson+read)",
                          "demo_lr.png", lens_px_scale=0.10)
    return {"hr": hr_path, "lr": lr_path,
            "n_lenses": meta["n_lenses"], "n_galaxies": meta["n_galaxies"]}


# ---------------------------------------------------------------------------
# Cutout visualization helpers
# ---------------------------------------------------------------------------

def _resolve_cutout_path(band_name: str, filename: str,
                         output_dir: str) -> str:
    """Map ``(band, filename)`` → safe absolute FITS path.

    Refuses anything that doesn't pass the strict filename regex or that
    resolves outside the per-band cutout directory after symlink expansion.
    """
    if not _CUTOUT_FNAME_RE.match(filename):
        abort(400)
    try:
        Config.get_band(band_name)
    except Exception:
        abort(404)
    band_dir = Config.cutout_dir_for_band(
        band_name, root=os.path.join(output_dir, "cutouts"),
    )
    full = os.path.realpath(os.path.join(band_dir, filename))
    if not full.startswith(os.path.realpath(band_dir) + os.sep):
        abort(403)
    if not os.path.isfile(full):
        abort(404)
    return full


def _render_fits_to_png(fits_path: str, band: BandConfig,
                        size: Optional[int] = None) -> bytes:
    """Load a cutout FITS, apply per-band asinh stretch, return PNG bytes.

    - Asinh scale comes from ``band.asinh_stretch_scale_e``.
    - vmin/vmax clip at the 1.0 / 99.7 percentiles of the stretched image.
    - Optional ``size`` resamples (nearest-neighbour) to a square thumbnail.
    """
    from astropy.io import fits
    from PIL import Image

    with fits.open(fits_path, memmap=False) as hdul:
        data = None
        for hdu in hdul:
            if hdu.data is not None and getattr(hdu.data, "ndim", 0) == 2:
                data = np.asarray(hdu.data, dtype=np.float32)
                break
    if data is None:
        abort(415)

    finite = np.isfinite(data)
    if not finite.any():
        data = np.zeros_like(data)
    else:
        # Replace NaN/inf with the median of finite pixels so the stretch is well-defined.
        data = np.where(finite, data, np.nanmedian(data[finite]))

    stretched = np.arcsinh(data / float(band.asinh_stretch_scale_e))
    lo, hi = np.percentile(stretched, [1.0, 99.7])
    if hi <= lo:
        hi = lo + 1.0
    norm = np.clip((stretched - lo) / (hi - lo), 0.0, 1.0)
    # gray_r style: bright pixels → dark ink. Easier to read against white UI.
    img8 = (255 * (1.0 - norm)).astype(np.uint8)
    # FITS orientation: origin lower-left. PIL is origin upper-left → flip.
    img8 = np.flipud(img8)
    pil = Image.fromarray(img8, mode="L")
    if size is not None and size > 0 and size != pil.size[0]:
        pil = pil.resize((int(size), int(size)), Image.NEAREST)
    buf = io.BytesIO()
    pil.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _list_band_cutouts(band_name: str, output_dir: str) -> List[str]:
    """Sorted list of cutout filenames present for ``band_name``."""
    band_dir = Config.cutout_dir_for_band(
        band_name, root=os.path.join(output_dir, "cutouts"),
    )
    if not os.path.isdir(band_dir):
        return []
    return sorted(
        f for f in os.listdir(band_dir)
        if f.lower().endswith(".fits") and _CUTOUT_FNAME_RE.match(f)
    )


# ---------------------------------------------------------------------------
# View renderers: serve PNGs of pipeline artefacts for the center pane.
# These all return bytes; the route layer wraps in send_file.
# ---------------------------------------------------------------------------

def _render_psf_panel_png(band: Optional[str]) -> bytes:
    """Render one band (or all four) on a log-stretch panel as PNG bytes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from euclid_polish.euclid.psf_library import load_all_band_psfs

    psfs = load_all_band_psfs()
    if band and band != "all":
        if band not in psfs:
            abort(404)
        names = [band]
    else:
        names = [b.name for b in Config.BANDS if b.name in psfs]
    if not names:
        abort(404)
    n = len(names)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4.2), squeeze=False)
    for ax, name in zip(axes[0], names):
        p = psfs[name]
        d = np.clip(p.data, 1e-8, None)
        ax.imshow(np.log10(d), cmap="viridis", origin="lower")
        ax.set_title(f"{name}  {p.data.shape[0]}×{p.data.shape[1]}  "
                     f"@ {p.pixel_scale:.3f}\"/pix", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, dpi=110, bbox_inches="tight", format="png")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def _render_sky_record_png(subset: str, kind: str, band: str,
                           index: int) -> bytes:
    """Render one image from the multi-band TFRecords with asinh stretch.

    ``subset`` ∈ {"train", "validate"},
    ``kind`` ∈ {"clean", "dirty", "hr"}
      • ``clean`` → 4-band HR clean record
      • ``dirty`` → 4-band LR dirty record (PSF + noise + artifacts)
      • ``hr``    → 1-band VIS HR target (the network's training output)
    ``band`` ∈ one of ``Config.LR_INPUT_BAND_NAMES``; ignored for ``hr``
      (always VIS) but accepted so the toolbar JS can pass any value.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import tensorflow as tf
    from euclid_polish.sky.tfrecord import (
        read_multiband_skyimages, tfrecord_path,
    )

    if subset not in ("train", "validate"):
        abort(400)
    if kind not in ("clean", "dirty", "hr"):
        abort(400)
    if band not in Config.LR_INPUT_BAND_NAMES:
        abort(400)
    name = f"{kind}_{subset}"
    path = tfrecord_path(Config.RECORDS_DIR_V2, name)
    if not os.path.exists(path):
        abort(404)
    # Stream just enough records to reach ``index``.
    max_to_read = max(index + 1, 1)
    records = read_multiband_skyimages(path, num_images=max_to_read)
    if not records or index >= len(records):
        abort(404)
    img = records[min(index, len(records) - 1)]
    data = img.data
    # HR records have shape (H, W, 1) (VIS only). LR + clean (4-ch) have 4 channels.
    if data.shape[-1] == 1:
        plane = data[..., 0]
        band_name = "VIS"
    else:
        k = list(Config.LR_INPUT_BAND_NAMES).index(band) if band != "HR" else 0
        plane = data[..., k]
        band_name = band
    bcfg = Config.get_band(band_name)
    stretched = np.arcsinh(plane / float(bcfg.asinh_stretch_scale_e))
    lo, hi = np.percentile(stretched, [1.0, 99.7])
    if hi <= lo: hi = lo + 1.0
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.imshow(stretched, cmap="gray_r", origin="lower", vmin=lo, vmax=hi)
    ax.set_title(f"{kind} {subset} · {band_name} · idx {img.index}  "
                 f"({data.shape[0]}×{data.shape[1]} @ "
                 f"{img.pixel_scale_arcsec:.3f}\"/pix)", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, dpi=110, bbox_inches="tight", format="png")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def _render_catalog_view_png(view: str, output_dir: str) -> bytes:
    """Render a catalog visualization: positions or magnitude histogram."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cat = StarCatalog(output_dir)
    if not cat.exists():
        abort(404)
    data = cat.load()
    stars = data.get("stars", [])
    if not stars:
        abort(404)
    if view == "positions":
        from euclid_polish.visualization.methods import plot_star_positions
        fig = plot_star_positions(stars)
    elif view == "magnitudes":
        mags = [s.get("magnitude") for s in stars
                if s.get("magnitude") is not None]
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.hist(mags, bins=40, color="#2a5db0", edgecolor="white")
        ax.set_xlabel("VIS magnitude (AB)"); ax.set_ylabel("count")
        ax.set_title(f"Catalog mag distribution  "
                     f"(median = {float(np.median(mags)):.2f})")
        fig.tight_layout()
    else:
        abort(400)
    buf = io.BytesIO()
    fig.savefig(buf, dpi=110, bbox_inches="tight", format="png")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def _record_count(name: str) -> int:
    """Total records in a multi-band tfrecord (0 if absent)."""
    import tensorflow as tf
    from euclid_polish.sky.tfrecord import tfrecord_path
    p = tfrecord_path(Config.RECORDS_DIR_V2, name)
    if not os.path.exists(p):
        return 0
    return sum(1 for _ in tf.data.TFRecordDataset(p))


# ---------------------------------------------------------------------------
# App factory + routes
# ---------------------------------------------------------------------------

def create_app() -> Flask:
    here = os.path.dirname(os.path.abspath(__file__))
    app = Flask(
        __name__,
        template_folder=os.path.join(here, "templates"),
        static_folder=os.path.join(here, "static"),
    )

    # ---------------- Dashboard ----------------
    @app.route("/")
    def index():
        return render_template(
            "index.html",
            catalog=_catalog_status(),
            psfs=_psf_status(),
            tfrecords=_tfrecords_status(),
            checkpoints=_checkpoints_status(),
        )

    # ---------------- Catalog page ----------------
    @app.route("/catalog")
    def catalog_page():
        from euclid_polish.euclid import auth
        return render_template(
            "catalog.html",
            status=_catalog_status(),
            bands=Config.BANDS,
            authenticated=auth.is_authenticated(),
            current_user=auth.current_user(),
            cutout_layout=_cutout_layout_status(),
        )

    @app.route("/catalog/query-brightest", methods=["POST"])
    def catalog_query_brightest():
        n   = int(request.form.get("num_stars", 200))
        out = request.form.get("output_dir", Config.DEFAULT_OUTPUT_DIR)
        mag_lim_raw = request.form.get("magnitude_limit", "").strip()
        mag_min_raw = request.form.get("magnitude_min", "").strip()
        mag_lim = float(mag_lim_raw) if mag_lim_raw else None
        mag_min = float(mag_min_raw) if mag_min_raw else None
        win = ""
        if mag_min is not None: win += f" mag>{mag_min}"
        if mag_lim is not None: win += f" mag<{mag_lim}"
        job_id = REGISTRY.spawn(
            label=f"query {n} brightest stars{win}",
            target=lambda cap: _job_query_brightest(
                cap, n, out, magnitude_limit=mag_lim, magnitude_min=mag_min),
        )
        return jsonify({"job_id": job_id})

    @app.route("/catalog/query-region", methods=["POST"])
    def catalog_query_region():
        ra  = float(request.form.get("ra"))
        dec = float(request.form.get("dec"))
        rad = float(request.form.get("radius"))
        mag = float(request.form.get("magnitude_limit"))
        mag_min_raw = request.form.get("magnitude_min", "").strip()
        mag_min = float(mag_min_raw) if mag_min_raw else None
        out = request.form.get("output_dir", Config.DEFAULT_OUTPUT_DIR)
        extra = f" mag>{mag_min}" if mag_min is not None else ""
        job_id = REGISTRY.spawn(
            label=f"query region ra={ra:.2f} dec={dec:.2f} r={rad}° mag<{mag}{extra}",
            target=lambda cap: _job_query_region(
                cap, ra, dec, rad, mag, out, magnitude_min=mag_min),
        )
        return jsonify({"job_id": job_id})

    @app.route("/catalog/integrity", methods=["POST"])
    def catalog_integrity():
        out = request.form.get("output_dir", Config.DEFAULT_OUTPUT_DIR)
        job_id = REGISTRY.spawn(
            label="check cutouts integrity",
            target=lambda cap: _job_check_integrity(cap, out),
        )
        return jsonify({"job_id": job_id})

    # ---------------- Authentication ----------------
    @app.route("/auth/login", methods=["POST"])
    def auth_login():
        from euclid_polish.euclid import auth
        user = request.form.get("username", "").strip()
        pwd  = request.form.get("password", "").strip()
        if not user or not pwd:
            return jsonify({"ok": False, "error": "Missing username or password"}), 400
        try:
            auth.login(user, pwd)
            return jsonify({"ok": True, "user": auth.current_user()})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/auth/logout", methods=["POST"])
    def auth_logout():
        from euclid_polish.euclid import auth
        try:
            auth.logout()
        except Exception:
            pass
        return jsonify({"ok": True})

    # ---------------- Cutouts page ----------------
    @app.route("/cutouts")
    def cutouts_page():
        return render_template(
            "cutouts.html",
            catalog=_catalog_status(),
            bands=Config.BANDS,
            cutout_layout=_cutout_layout_status(),
            default_vis_pixels=Config.DEFAULT_CUTOUT_SIZE,
        )

    @app.route("/cutouts/download", methods=["POST"])
    def cutouts_download():
        bands  = request.form.getlist("bands")
        if not bands:
            return jsonify({"ok": False, "error": "no bands selected"}), 400
        vis_px = int(request.form.get("cutout_size_vis_pixels",
                                       Config.DEFAULT_CUTOUT_SIZE))
        workers = int(request.form.get("max_workers", 8))
        out = request.form.get("output_dir", Config.DEFAULT_OUTPUT_DIR)
        job_id = REGISTRY.spawn(
            label=f"download cutouts ({'+'.join(bands)}, {vis_px} VIS px)",
            target=lambda cap: _job_download_cutouts(
                cap, bands, vis_px, workers, out,
            ),
        )
        return jsonify({"job_id": job_id})

    # ---------------- Cutout gallery + live FITS→PNG ----------------
    @app.route("/cutouts/<band_name>")
    def cutouts_gallery(band_name: str):
        """Per-band paginated thumbnail gallery."""
        try:
            band = Config.get_band(band_name)
        except Exception:
            abort(404)
        out_dir = request.args.get("output_dir", Config.DEFAULT_OUTPUT_DIR)
        try:
            page = max(1, int(request.args.get("page", 1)))
        except ValueError:
            page = 1
        per_page = 60
        files = _list_band_cutouts(band_name, out_dir)
        total = len(files)
        n_pages = max(1, (total + per_page - 1) // per_page)
        page = min(page, n_pages)
        start = (page - 1) * per_page
        end   = start + per_page
        return render_template(
            "cutouts_gallery.html",
            band=band, files=files[start:end],
            total=total, page=page, n_pages=n_pages,
            per_page=per_page, output_dir=out_dir,
        )

    @app.route("/cutout-image/<band_name>/<path:filename>")
    def cutout_image(band_name: str, filename: str):
        out_dir = request.args.get("output_dir", Config.DEFAULT_OUTPUT_DIR)
        try:
            size = int(request.args.get("size", 0)) or None
        except ValueError:
            size = None
        if size is not None and (size < 16 or size > 2048):
            abort(400)
        try:
            band = Config.get_band(band_name)
        except ValueError:
            abort(404)
        fits_path = _resolve_cutout_path(band_name, filename, out_dir)
        png = _render_fits_to_png(fits_path, band, size=size)
        return send_file(io.BytesIO(png), mimetype="image/png",
                         max_age=3600)

    # ---------------- PSFs page ----------------
    @app.route("/psfs")
    def psfs_page():
        return render_template(
            "psfs.html",
            status=_psf_status(),
            bands=Config.BANDS,
            default_num_stars=200,
            default_output_size=1024,
        )

    @app.route("/psfs/extract", methods=["POST"])
    def psfs_extract():
        band_name   = request.form.get("band")
        num_stars   = int(request.form.get("num_stars", 200))
        cutout_size = int(request.form.get("cutout_size", Config.DEFAULT_CUTOUT_SIZE))
        output_raw  = request.form.get("output_size", "").strip()
        output_size = int(output_raw) if output_raw else None
        out_dir     = request.form.get("output_dir", Config.DEFAULT_OUTPUT_DIR)
        psf_dir     = request.form.get("psf_dir",    Config.EUCLID_PSF_DIR)
        job_id = REGISTRY.spawn(
            label=f"extract {band_name} ePSF ({num_stars} stars)",
            target=lambda cap: _job_extract_psf(
                cap, band_name, num_stars, cutout_size, output_size,
                out_dir, psf_dir,
            ),
        )
        return jsonify({"job_id": job_id})

    @app.route("/psfs/visualize", methods=["POST"])
    def psfs_visualize():
        band_name = request.form.get("band") or None
        psf_dir   = request.form.get("psf_dir", Config.EUCLID_PSF_DIR)
        job_id = REGISTRY.spawn(
            label=f"render PSF panel ({band_name or 'all bands'})",
            target=lambda cap: _job_viz_psf(cap, band_name, psf_dir),
        )
        return jsonify({"job_id": job_id})

    # ---------------- Sky generation + forward ----------------
    @app.route("/sky")
    def sky_page():
        return render_template("sky.html",
                               tfrecords=_tfrecords_status(),
                               default_image_size=510,
                               default_n_train=20,
                               default_n_valid=4,
                               default_lens_density=Config.LENS_DENSITY_ARCMIN2)

    @app.route("/sky/generate", methods=["POST"])
    def sky_generate():
        n_train = int(request.form.get("n_train", 20))
        n_valid = int(request.form.get("n_valid", 4))
        image_size = int(request.form.get("image_size", 510))
        lens_density = float(request.form.get("lens_density",
                                              Config.LENS_DENSITY_ARCMIN2))
        job_id = REGISTRY.spawn(
            label=f"generate {n_train}+{n_valid} @ {image_size}²",
            target=lambda cap: _job_generate(
                cap, image_size, n_train, n_valid, lens_density,
            ),
        )
        return jsonify({"job_id": job_id})

    @app.route("/sky/forward", methods=["POST"])
    def sky_forward():
        job_id = REGISTRY.spawn(
            label="forward model (PSF + noise)",
            target=lambda cap: _job_forward(cap),
        )
        return jsonify({"job_id": job_id})

    # ---------------- Training page ----------------
    @app.route("/training")
    def training_page():
        return render_template(
            "training.html",
            tfrecords=_tfrecords_status(),
            checkpoints=_checkpoints_status(),
            default_steps=Config.DEFAULT_TRAIN_STEPS,
            default_batch=Config.DEFAULT_BATCH_SIZE,
            default_num_res_blocks=Config.DEFAULT_NUM_RES_BLOCKS,
            default_eval_every=Config.DEFAULT_EVALUATE_EVERY,
        )

    @app.route("/training/train", methods=["POST"])
    def training_train():
        steps = int(request.form.get("steps", Config.DEFAULT_TRAIN_STEPS))
        batch = int(request.form.get("batch_size", Config.DEFAULT_BATCH_SIZE))
        nrb   = int(request.form.get("num_res_blocks", Config.DEFAULT_NUM_RES_BLOCKS))
        eval_every = int(request.form.get("evaluate_every",
                                           Config.DEFAULT_EVALUATE_EVERY))
        ckpt_dir = request.form.get("checkpoint_dir", Config.DEFAULT_CHECKPOINT_DIR)
        job_id = REGISTRY.spawn(
            label=f"train WDSR ({steps} steps, batch {batch})",
            target=lambda cap: _job_train(cap, steps, batch, nrb,
                                          eval_every, ckpt_dir),
        )
        return jsonify({"job_id": job_id})

    @app.route("/training/evaluate", methods=["POST"])
    def training_evaluate():
        ckpt_dir = request.form.get("checkpoint_dir", Config.DEFAULT_CHECKPOINT_DIR)
        nrb = int(request.form.get("num_res_blocks", Config.DEFAULT_NUM_RES_BLOCKS))
        job_id = REGISTRY.spawn(
            label="evaluate latest checkpoint",
            target=lambda cap: _job_evaluate(cap, ckpt_dir, nrb),
        )
        return jsonify({"job_id": job_id})

    @app.route("/training/plot-log", methods=["POST"])
    def training_plot_log():
        ckpt_dir = request.form.get("checkpoint_dir", Config.DEFAULT_CHECKPOINT_DIR)
        job_id = REGISTRY.spawn(
            label="plot training log",
            target=lambda cap: _job_plot_training_log(cap, ckpt_dir),
        )
        return jsonify({"job_id": job_id})

    # ---------------- Inference page ----------------
    @app.route("/inference")
    def inference_page():
        # Most-recent reconstruction PNGs (newest first).
        recon_pngs: list[Dict[str, Any]] = []
        rdir = Config.VIS_RECONSTRUCTION_DIR
        if os.path.isdir(rdir):
            for fname in os.listdir(rdir):
                if not fname.lower().endswith(".png"):
                    continue
                full = os.path.join(rdir, fname)
                try:
                    mtime = os.path.getmtime(full)
                except OSError:
                    continue
                rel = os.path.relpath(full, Config.VIS_DIR)
                recon_pngs.append({"rel": rel, "name": fname, "mtime": mtime})
            recon_pngs.sort(key=lambda d: d["mtime"], reverse=True)
        return render_template(
            "inference.html",
            checkpoints=_checkpoints_status(),
            tfrecords=_tfrecords_status(),
            recon_pngs=recon_pngs,
            default_num_res_blocks=Config.DEFAULT_NUM_RES_BLOCKS,
            default_n_images=4,
        )

    @app.route("/inference/reconstruct", methods=["POST"])
    def inference_reconstruct():
        ckpt_dir = request.form.get("checkpoint_dir", Config.DEFAULT_CHECKPOINT_DIR)
        nrb = int(request.form.get("num_res_blocks", Config.DEFAULT_NUM_RES_BLOCKS))
        subset = request.form.get("subset", "validate")
        n = int(request.form.get("n_images", 4))
        job_id = REGISTRY.spawn(
            label=f"reconstruct {n} {subset} images",
            target=lambda cap: _job_reconstruct(cap, ckpt_dir, nrb, subset, n),
        )
        return jsonify({"job_id": job_id})

    @app.route("/inference/reconstruct-euclid", methods=["POST"])
    def inference_reconstruct_euclid():
        try:
            ra  = float(request.form["ra"])
            dec = float(request.form["dec"])
        except (KeyError, ValueError):
            return jsonify({"error": "ra and dec must be valid floats (degrees)"}), 400
        if not (0.0 <= ra < 360.0):
            return jsonify({"error": f"ra={ra} out of range [0, 360)"}), 400
        if not (-90.0 <= dec <= 90.0):
            return jsonify({"error": f"dec={dec} out of range [-90, 90]"}), 400
        ckpt_dir = request.form.get("checkpoint_dir", Config.DEFAULT_CHECKPOINT_DIR)
        nrb = int(request.form.get("num_res_blocks", Config.DEFAULT_NUM_RES_BLOCKS))
        size = int(request.form.get("cutout_size", 512))
        if not (32 <= size <= 4096):
            return jsonify({"error": f"cutout_size={size} out of range [32, 4096]"}), 400
        job_id = REGISTRY.spawn(
            label=f"infer Euclid cutout @ ({ra:.4f}, {dec:+.4f})",
            target=lambda cap: _job_reconstruct_euclid_cutout(
                cap, ra, dec, ckpt_dir, nrb, size,
            ),
        )
        return jsonify({"job_id": job_id})

    # ---------------- Visualization page ----------------
    @app.route("/visualization")
    def visualization_page():
        return render_template("visualization.html",
                               pngs=_list_vis_pngs())

    @app.route("/visualization/demo", methods=["POST"])
    def visualization_demo():
        n_lenses = int(request.form.get("n_lenses", 3))
        job_id = REGISTRY.spawn(
            label=f"demo: 510² field with {n_lenses} lenses",
            target=lambda cap: _job_demo_lens(cap, n_lenses),
        )
        return jsonify({"job_id": job_id})

    @app.route("/visualization/star-positions", methods=["POST"])
    def visualization_star_positions():
        out = request.form.get("output_dir", Config.DEFAULT_OUTPUT_DIR)
        job_id = REGISTRY.spawn(
            label="plot star positions",
            target=lambda cap: _job_viz_star_positions(cap, out),
        )
        return jsonify({"job_id": job_id})

    # ---------------- Live view renderers (PNG) ----------------
    @app.route("/view/psfs")
    def view_psfs():
        band = request.args.get("band", "all")
        png = _render_psf_panel_png(None if band == "all" else band)
        return send_file(io.BytesIO(png), mimetype="image/png", max_age=0)

    @app.route("/view/sky")
    def view_sky():
        subset = request.args.get("subset", "train")
        kind   = request.args.get("kind",   "clean")
        band   = request.args.get("band",   "VIS")
        try:
            idx = int(request.args.get("i", 0))
        except ValueError:
            idx = 0
        png = _render_sky_record_png(subset, kind, band, idx)
        return send_file(io.BytesIO(png), mimetype="image/png", max_age=0)

    @app.route("/view/catalog")
    def view_catalog():
        view = request.args.get("view", "positions")
        out  = request.args.get("output_dir", Config.DEFAULT_OUTPUT_DIR)
        png = _render_catalog_view_png(view, out)
        return send_file(io.BytesIO(png), mimetype="image/png", max_age=0)

    @app.route("/view/training-log")
    def view_training_log():
        ckpt = request.args.get("checkpoint_dir", Config.DEFAULT_CHECKPOINT_DIR)
        log_path = _resolve_training_log(ckpt)
        out_png  = os.path.join(Config.VIS_DIR, "training_log.png")
        if log_path is None:
            abort(404)
        # Render if missing or stale.
        if (not os.path.exists(out_png)
                or os.path.getmtime(log_path) > os.path.getmtime(out_png)):
            from euclid_polish.training.log_plot import plot_training_log
            os.makedirs(Config.VIS_DIR, exist_ok=True)
            plot_training_log(log_path, out_png)
        return send_file(out_png, mimetype="image/png", max_age=0)

    @app.route("/api/sky/totals")
    def api_sky_totals():
        return jsonify({
            "clean_train":    _record_count("clean_train"),
            "clean_validate": _record_count("clean_validate"),
            "dirty_train":    _record_count("dirty_train"),
            "dirty_validate": _record_count("dirty_validate"),
            "hr_train":       _record_count("hr_train"),
            "hr_validate":    _record_count("hr_validate"),
        })

    # ---------------- Static PNG server (data/vis/) ----------------
    @app.route("/vis/<path:relpath>")
    def serve_vis(relpath: str):
        full = os.path.realpath(os.path.join(Config.VIS_DIR, relpath))
        vis_root = os.path.realpath(Config.VIS_DIR)
        # Refuse anything that resolves outside data/vis (path traversal).
        if not full.startswith(vis_root + os.sep):
            abort(403)
        if not os.path.isfile(full):
            abort(404)
        return send_file(full, mimetype="image/png")

    # ---------------- Job tracker API ----------------
    @app.route("/api/jobs")
    def api_jobs():
        return jsonify(REGISTRY.list())

    @app.route("/api/jobs/<job_id>")
    def api_job(job_id: str):
        job = REGISTRY.get(job_id)
        if not job:
            abort(404)
        return jsonify(job.to_dict())

    @app.route("/api/status")
    def api_status():
        return jsonify({
            "catalog":     _catalog_status(),
            "psfs":        _psf_status(),
            "tfrecords":   _tfrecords_status(),
            "checkpoints": _checkpoints_status(),
        })

    # =========================================================================
    # Git tab — local commit / push / pull, no remote auth needed.
    # =========================================================================

    @app.route("/git")
    def git_page():
        return render_template(
            "git.html",
            status=git_ops.status(),
            log_entries=git_ops.log(15),
        )

    @app.route("/api/git/status")
    def api_git_status():
        return jsonify({"status": git_ops.status(),
                        "log": git_ops.log(15)})

    @app.route("/api/git/diff")
    def api_git_diff():
        staged = request.args.get("staged", "0") in ("1", "true", "yes")
        return jsonify({"diff": git_ops.diff(staged=staged)})

    @app.route("/git/commit", methods=["POST"])
    def git_commit():
        msg = request.form.get("message", "").strip()
        out = git_ops.commit(msg)
        code = 200 if out.get("ok") else 400
        return jsonify(out), code

    @app.route("/git/push", methods=["POST"])
    def git_push():
        out = git_ops.push()
        code = 200 if out.get("ok") else 400
        return jsonify(out), code

    @app.route("/git/pull", methods=["POST"])
    def git_pull():
        out = git_ops.pull()
        code = 200 if out.get("ok") else 400
        return jsonify(out), code

    @app.route("/git/fetch", methods=["POST"])
    def git_fetch():
        out = git_ops.fetch()
        code = 200 if out.get("ok") else 400
        return jsonify(out), code

    # =========================================================================
    # FASRC tab — Bitwarden-driven SSH ControlMaster, SLURM submission,
    # live log streaming, checkpoint auto-mirror.
    # =========================================================================

    @app.route("/fasrc")
    def fasrc_page():
        cfg = fasrc_config.load()
        return render_template(
            "fasrc.html",
            cfg=cfg,
            state=STATE.public_status(),
            recent=fasrc_jobs.DB.list_recent(20),
        )

    # ---- config -----------------------------------------------------------

    @app.route("/api/fasrc/config", methods=["GET", "POST"])
    def api_fasrc_config():
        if request.method == "POST":
            patch = {k: v for k, v in request.form.items()}
            cfg = fasrc_config.update(patch)
        else:
            cfg = fasrc_config.load()
        return jsonify(cfg.to_dict())

    # ---- auth -------------------------------------------------------------

    @app.route("/api/fasrc/status")
    def api_fasrc_status():
        return jsonify(STATE.public_status())

    @app.route("/api/fasrc/unlock", methods=["POST"])
    def api_fasrc_unlock():
        master = request.form.get("master_password", "")
        try:
            STATE.bw.unlock(master)
        except BitwardenError as e:
            return jsonify({"ok": False, "error": str(e)}), 400
        return jsonify({"ok": True, "status": STATE.public_status()})

    @app.route("/api/fasrc/lock", methods=["POST"])
    def api_fasrc_lock():
        STATE.bw.lock()
        return jsonify({"ok": True, "status": STATE.public_status()})

    @app.route("/api/fasrc/connect", methods=["POST"])
    def api_fasrc_connect():
        cfg = fasrc_config.load()
        if not cfg.ssh_user:
            return jsonify({"ok": False,
                            "error": "set ssh_user in Settings first"}), 400
        if not STATE.bw.unlocked:
            return jsonify({"ok": False,
                            "error": "unlock Bitwarden first"}), 400
        try:
            pwd  = STATE.bw.get_password(cfg.bw_item)
            totp = STATE.bw.get_totp(cfg.bw_item)
        except BitwardenError as e:
            return jsonify({"ok": False, "error": str(e)}), 400
        STATE.ssh = SSHSession(SSHConfig(
            user=cfg.ssh_user, host=cfg.ssh_host,
            socket=cfg.control_socket,
            control_persist=cfg.control_persist,
        ))
        try:
            STATE.ssh.connect(pwd, totp)
        except SSHError as e:
            STATE.ssh = None
            return jsonify({"ok": False, "error": str(e)}), 400
        STATE.connected_at = time.time()
        return jsonify({"ok": True, "status": STATE.public_status()})

    @app.route("/api/fasrc/disconnect", methods=["POST"])
    def api_fasrc_disconnect():
        if STATE.ssh:
            STATE.ssh.disconnect()
        STATE.ssh = None
        STATE.connected_at = None
        MIRROR.stop()
        return jsonify({"ok": True, "status": STATE.public_status()})

    # ---- remote info ------------------------------------------------------

    @app.route("/api/fasrc/git-status")
    def api_fasrc_git_status():
        if not STATE.ssh or not STATE.ssh.is_connected():
            return jsonify({"ok": False, "error": "not connected"}), 400
        cfg = fasrc_config.load()
        repo = cfg.repo_path
        cmds = (
            f"cd {repo} && "
            f"git rev-parse --abbrev-ref HEAD && "
            f"git fetch --quiet && "
            f"git rev-list --left-right --count HEAD...@{{u}} 2>/dev/null && "
            f"git log -1 --pretty=format:'%h%x09%s%x09%cr'"
        )
        rc, out, err = STATE.ssh.run(cmds, timeout=30)
        if rc != 0:
            return jsonify({"ok": False, "error": err.strip() or out.strip()}), 500
        lines = out.strip().splitlines()
        branch  = lines[0] if len(lines) > 0 else ""
        counts  = (lines[1].split() if len(lines) > 1 else ["0", "0"])
        ahead   = int(counts[0]) if counts and counts[0].isdigit() else 0
        behind  = int(counts[1]) if len(counts) > 1 and counts[1].isdigit() else 0
        last    = lines[2].split("\t", 2) if len(lines) > 2 else []
        last_commit = ({"hash": last[0], "subject": last[1], "relative": last[2]}
                       if len(last) == 3 else {})
        return jsonify({"ok": True, "repo": repo, "branch": branch,
                        "ahead": ahead, "behind": behind,
                        "last": last_commit})

    @app.route("/api/fasrc/git-pull", methods=["POST"])
    def api_fasrc_git_pull():
        """``git pull`` + auto-update conda env when ``environment.yml`` moved.

        Returns ``env_update_needed: True`` whenever the pull's diff
        touches ``environment.yml``; the UI then kicks off the
        ``/api/fasrc/env-update`` SSE stream automatically so the user
        doesn't have to remember.
        """
        if not STATE.ssh or not STATE.ssh.is_connected():
            return jsonify({"ok": False, "error": "not connected"}), 400
        cfg = fasrc_config.load()
        # Run ``git pull`` and ask in the same shell what just moved.
        # ``ORIG_HEAD..HEAD`` is everything fetched; if the pull was a
        # no-op the second command emits an empty list.
        rc, out, err = STATE.ssh.run(
            f"cd {shlex.quote(cfg.repo_path)} && "
            f"git pull --ff-only && "
            f"echo '__CHANGED__' && "
            f"git diff --name-only ORIG_HEAD..HEAD 2>/dev/null || true",
            timeout=60,
        )
        out_text = (out + err).strip()
        changed_files: list[str] = []
        if "__CHANGED__" in out:
            head, _, tail = out.partition("__CHANGED__")
            out_text = head.strip()
            changed_files = [line for line in tail.splitlines() if line.strip()]
        env_update_needed = any(
            f.endswith("environment.yml") for f in changed_files
        )
        return jsonify({
            "ok":                rc == 0,
            "stdout":            out_text,
            "changed_files":     changed_files,
            "env_update_needed": env_update_needed,
            "error":             "" if rc == 0 else
                                  (err.strip() or out.strip()),
        })

    @app.route("/api/fasrc/data-listing")
    def api_fasrc_data_listing():
        if not STATE.ssh or not STATE.ssh.is_connected():
            return jsonify({"ok": False, "error": "not connected"}), 400
        cfg = fasrc_config.load()
        # Each section is guarded with `[ -d path ] &&` so a missing
        # directory (common on a fresh netscratch dir) doesn't sink the
        # whole listing. The trailing ``exit 0`` keeps the SSH call
        # green even if every section is empty.
        # ``du -shL`` dereferences symlinks: COSMOS2025 / euclid_psf are
        # typically symlinks into <repo>/data on holylabs, and the naked
        # ``du`` reports the link itself (~60 B, rounds to 0). ``-L``
        # follows the link and reports the contents. ``find -L`` mirrors
        # that semantic for the tfrecord / checkpoint sweeps below.
        cmd = (
            f"{{ "
            f"  [ -d {shlex.quote(cfg.data_dir)} ] && "
            f"    du -shL {shlex.quote(cfg.data_dir)}/* 2>/dev/null | sort -k2 ; "
            f"  echo '---' ; "
            f"  [ -d {shlex.quote(cfg.data_dir)} ] && "
            f"    find -L {shlex.quote(cfg.data_dir)} -maxdepth 3 -type f "
            f"      -name '*.tfrecord' -printf '%p\\t%s\\n' 2>/dev/null ; "
            f"  echo '---' ; "
            f"  [ -d {shlex.quote(cfg.ckpt_dir)} ] && "
            f"    find -L {shlex.quote(cfg.ckpt_dir)} -maxdepth 2 -type f "
            f"      -printf '%p\\t%s\\t%TY-%Tm-%Td %TH:%TM\\n' 2>/dev/null ; "
            f"}}; exit 0"
        )
        rc, out, err = STATE.ssh.run(cmd, timeout=30)
        if rc != 0:
            return jsonify({"ok": False,
                            "error": f"remote du/find failed: {err.strip()}"}), 500
        sections = out.split("---")
        du_lines     = (sections[0].splitlines() if len(sections) > 0 else [])
        tfr_lines    = (sections[1].splitlines() if len(sections) > 1 else [])
        ckpt_lines   = (sections[2].splitlines() if len(sections) > 2 else [])

        def _split(line: str, n: int) -> list[str]:
            parts = line.split("\t" if "\t" in line else None, n - 1)
            return parts + [""] * (n - len(parts))

        return jsonify({
            "ok": True,
            "data_dir": cfg.data_dir,
            "ckpt_dir": cfg.ckpt_dir,
            "du": [line.split(None, 1) for line in du_lines if line.strip()],
            "tfrecords": [
                {"path": p, "size": int(s) if s.isdigit() else 0}
                for line in tfr_lines if line.strip()
                for p, s in [_split(line, 2)[:2]]
            ],
            "checkpoints": [
                {"path": p, "size": int(s) if s.isdigit() else 0, "mtime": m}
                for line in ckpt_lines if line.strip()
                for p, s, m in [_split(line, 3)[:3]]
            ],
        })

    @app.route("/api/fasrc/bootstrap-data", methods=["POST"])
    def api_fasrc_bootstrap_data():
        """Re-create the symlinks that point ``data_dir`` at the durable
        copy of the same data under ``{repo_path}/data/`` on holylabs.
        Idempotent: re-runnable after a netscratch purge, after committing
        new PSFs, or after Globus uploads a fresh COSMOS catalog —
        without manual cleanup.

        Targets (source → link name under ``data_dir``):
          - ``{repo_path}/data/euclid_psf``  → ``euclid_psf``   (ships via git)
          - ``{repo_path}/data/COSMOS2025``  → ``COSMOS2025``   (Globus-uploaded)
        """
        if not STATE.ssh or not STATE.ssh.is_connected():
            return jsonify({"ok": False, "error": "not connected"}), 400
        cfg = fasrc_config.load()
        repo_data = f"{cfg.repo_path}/data"
        targets = [
            ("euclid_psf", f"{repo_data}/euclid_psf"),
            ("COSMOS2025", f"{repo_data}/COSMOS2025"),
        ]
        link_cmds = []
        for name, src in targets:
            link_cmds.append(
                f"if [ -e {shlex.quote(src)} ]; then "
                f"  ln -sfn {shlex.quote(src)} {shlex.quote(name)} "
                f"    && echo 'linked: {name} -> {src}' "
                f"    || echo 'FAILED: ln -sfn {src} {name}'; "
                f"else "
                f"  echo 'MISSING source: {src} — upload via Globus first'; "
                f"fi"
            )
        cmd = (
            f"mkdir -p {shlex.quote(cfg.data_dir)} && "
            f"cd {shlex.quote(cfg.data_dir)} && {{ "
            + " ; ".join(link_cmds)
            + "; echo '---'; ls -l . | head -40; "
            + "}"
        )
        rc, out, err = STATE.ssh.run(cmd, timeout=20)
        return jsonify({
            "ok":     rc == 0,
            "output": out.strip(),
            "error":  err.strip() if rc != 0 else "",
        })

    @app.route("/api/fasrc/queue")
    def api_fasrc_queue():
        if not STATE.ssh or not STATE.ssh.is_connected():
            return jsonify({"ok": False, "error": "not connected"}), 400
        rc, out, err = STATE.ssh.run(
            f"squeue -h -u $USER --format='{fasrc_jobs.SQUEUE_FMT}'",
            timeout=15,
        )
        if rc != 0:
            return jsonify({"ok": False, "error": err.strip()}), 500
        rows = fasrc_jobs.parse_squeue(out)

        # Reconcile sqlite state with the live queue: any tracked job that
        # has fallen off squeue is finished (sacct would be more precise
        # but is slow on the login node — last_seen + ended_at is enough).
        live_ids = {r["jobid"] for r in rows}
        for stored in fasrc_jobs.DB.list_recent(50):
            if stored["state"] in ("COMPLETED", "FAILED", "CANCELLED",
                                   "TIMEOUT", "DONE"):
                continue
            if stored["jobid"] not in live_ids and stored["started_at"]:
                fasrc_jobs.DB.update_state(
                    stored["jobid"], state="DONE",
                    ended_at=time.time(),
                )
        # Push live state for jobs that ARE running.
        for r in rows:
            if r.get("state") == "RUNNING":
                fasrc_jobs.DB.update_state(
                    r["jobid"], state="RUNNING",
                    started_at=time.time() - _parse_slurm_time(r.get("time")),
                )
            else:
                fasrc_jobs.DB.update_state(r["jobid"], state=r.get("state", "?"))

        return jsonify({"ok": True, "rows": rows})

    _parse_slurm_time = fasrc_jobs.parse_slurm_time

    # ---- submission -------------------------------------------------------

    @app.route("/api/fasrc/presets")
    def api_fasrc_presets():
        """Return the submission preset table so the JS form can render
        the dropdown and auto-fill resource fields when a preset changes."""
        # Stripping the python-only ``skip_flags`` / ``needs_train_knobs``
        # would lose data the UI uses — leave them in.
        return jsonify({"presets": fasrc_jobs.PRESETS})

    @app.route("/api/fasrc/eta")
    def api_fasrc_eta():
        try:
            steps = int(request.args.get("steps", 0))
        except ValueError:
            steps = 0
        spt = fasrc_jobs.secs_per_step_history()
        return jsonify({
            "secs_per_step": spt,
            "history_n":     len(fasrc_jobs.DB.list_completed(8)),
            "eta_seconds":   fasrc_jobs.eta_for_submission(steps),
        })

    @app.route("/api/fasrc/submit", methods=["POST"])
    def api_fasrc_submit():
        if not STATE.ssh or not STATE.ssh.is_connected():
            return jsonify({"ok": False, "error": "not connected"}), 400
        cfg = fasrc_config.load()
        f = request.form
        preset_name = f.get("preset", "custom")
        preset = fasrc_jobs.resolve_preset(preset_name)
        try:
            params = {
                "partition":  f.get("partition",  cfg.partition),
                "n_gpus":     int(f.get("n_gpus",     cfg.n_gpus)),
                "n_cpus":     int(f.get("n_cpus",     cfg.n_cpus)),
                "memory":          f.get("memory",     cfg.memory),
                "time_limit":      f.get("time_limit", cfg.time_limit),
                "n_train":    int(f.get("n_train",    cfg.n_train)),
                "n_valid":    int(f.get("n_valid",    cfg.n_valid)),
                "image_size": int(f.get("image_size", cfg.image_size)),
                "batch_size": int(f.get("batch_size", cfg.batch_size)),
                "steps":      int(f.get("steps",      cfg.steps)),
                "extra_flags":     f.get("extra_flags", "").strip(),
            }
        except (TypeError, ValueError) as e:
            return jsonify({"ok": False, "error": f"bad form field: {e}"}), 400

        # Preset → append the right --skip-* flags so the user can't
        # accidentally request a CPU-only "convolve" job that then tries
        # to train (or vice-versa). The free-form ``extra_flags`` field
        # is preserved so users can still pass one-off args.
        skip = (preset.get("skip_flags") or "").strip()
        if skip:
            params["extra_flags"] = (params["extra_flags"] + " " + skip).strip()
        params["preset"] = preset_name

        label = f.get("label", "").strip() or (
            f"{preset.get('label', preset_name)}: "
            f"{params['steps']} steps on {params['n_train']}+"
            f"{params['n_valid']} fields"
        )
        built = fasrc_jobs.build_sbatch_script(
            label=label, params=params, cfg=cfg,
        )
        # Drop the script into the repo's logs/jobs dir on FASRC.
        remote_script = f"{cfg.repo_path}/{built['script']}"
        write_cmd = (
            f"mkdir -p {cfg.repo_path}/{os.path.dirname(built['script'])} && "
            f"cat > {remote_script} <<'__EUCLID_POLISH_EOF__'\n"
            f"{built['body']}"
            f"__EUCLID_POLISH_EOF__\n"
            f"chmod +x {remote_script}"
        )
        rc, _out, err = STATE.ssh.run(write_cmd, timeout=20)
        if rc != 0:
            return jsonify({"ok": False,
                            "error": f"failed to write script: {err.strip()}"}), 500

        rc, out, err = STATE.ssh.run(
            f"cd {cfg.repo_path} && sbatch {built['script']}", timeout=20,
        )
        if rc != 0:
            return jsonify({"ok": False,
                            "error": f"sbatch failed: {err.strip()}"}), 500
        # sbatch output: "Submitted batch job 12345"
        m = re.search(r"Submitted batch job (\d+)", out)
        if not m:
            return jsonify({"ok": False,
                            "error": f"unparseable sbatch output: {out}"}), 500
        slurm_id = m.group(1)
        fasrc_jobs.DB.insert(
            slurm_id,
            label=label,
            params=params,
            script_path=remote_script,
            log_path=f"{cfg.repo_path}/{built['out']}",
            err_path=f"{cfg.repo_path}/{built['err']}",
        )
        return jsonify({"ok": True, "jobid": slurm_id,
                        "label": label, "params": params,
                        "log_path":   f"{cfg.repo_path}/{built['out']}"})

    @app.route("/api/fasrc/extend-time", methods=["POST"])
    def api_fasrc_extend_time():
        """Add wall time to a running job via ``scontrol update job=…
        TimeLimit=+HH:MM:SS``.

        SLURM lets users extend their own running jobs up to the
        partition's ``MaxTime``. We don't try to read the partition cap
        client-side — if SLURM refuses, its error message comes back in
        the response. Safety cap of 168 h (one week) per single call.
        """
        if not STATE.ssh or not STATE.ssh.is_connected():
            return jsonify({"ok": False, "error": "not connected"}), 400
        jid   = request.form.get("jobid", "").strip()
        hours = request.form.get("hours", "1").strip()
        if not jid.isdigit():
            return jsonify({"ok": False, "error": "bad jobid"}), 400
        try:
            h = float(hours)
        except (TypeError, ValueError):
            return jsonify({"ok": False, "error": "hours must be numeric"}), 400
        if h <= 0 or h > 168:
            return jsonify({"ok": False,
                            "error": "hours must be 0 < h ≤ 168"}), 400

        secs = int(round(h * 3600))
        hh, rem = divmod(secs, 3600)
        mm, ss  = divmod(rem, 60)
        delta = f"+{hh:02d}:{mm:02d}:{ss:02d}"

        # ``scontrol update`` is silent on success — chase it with
        # ``scontrol show`` so we can echo the new effective TimeLimit
        # back to the UI.
        cmd = (f"scontrol update job={jid} TimeLimit={delta} && "
               f"scontrol show job={jid} | tr ' ' '\\n' | grep TimeLimit=")
        rc, out, err = STATE.ssh.run(cmd, timeout=15)
        if rc != 0:
            return jsonify({"ok": False,
                            "error": err.strip() or out.strip()
                                     or "scontrol failed"}), 500
        return jsonify({
            "ok":         True,
            "delta":      delta,
            "scontrol":   out.strip(),
        })

    @app.route("/api/fasrc/cancel", methods=["POST"])
    def api_fasrc_cancel():
        if not STATE.ssh or not STATE.ssh.is_connected():
            return jsonify({"ok": False, "error": "not connected"}), 400
        jid = request.form.get("jobid", "").strip()
        if not jid.isdigit():
            return jsonify({"ok": False, "error": "bad job id"}), 400
        rc, _, err = STATE.ssh.run(f"scancel {jid}", timeout=10)
        if rc != 0:
            return jsonify({"ok": False, "error": err.strip()}), 500
        fasrc_jobs.DB.update_state(jid, state="CANCELLED",
                                   ended_at=time.time())
        return jsonify({"ok": True})

    @app.route("/api/fasrc/jobs")
    def api_fasrc_jobs_list():
        rows = fasrc_jobs.DB.list_recent(30)
        for r in rows:
            r["eta_seconds"] = (
                fasrc_jobs.eta_for_running(r)
                if r["state"] in ("RUNNING", "PENDING") else None
            )
        return jsonify({"jobs": rows})

    # ---- past-runs browser (Logs tab) ---------------------------------------
    #
    # Combines two sources so the user sees every run that left a log on
    # FASRC, regardless of how it was submitted:
    #   (1) ``JobDB`` — every job submitted from this UI, with its SLURM
    #       jobid, label, state, and timestamps.
    #   (2) Remote ``find <repo>/logs/jobs -name '*.out' -o -name '*.err'``
    #       — picks up jobs submitted directly via sbatch from the CLI,
    #       which the DB has no record of.
    # Rows are de-duplicated by base name (the ``euclid-YYYYMMDD-HHMMSS``
    # prefix that pairs an ``.out`` with its ``.err``); UI-submitted jobs
    # therefore get their full DB metadata, CLI-submitted jobs just get
    # the file timestamps + sizes.

    @app.route("/api/fasrc/runs")
    def api_fasrc_runs():
        if not STATE.ssh or not STATE.ssh.is_connected():
            return jsonify({"ok": False, "error": "not connected"}), 400
        cfg = fasrc_config.load()
        log_dir = f"{cfg.repo_path}/{cfg.logs_subdir}/jobs"

        # 1. Scan remote for every .out / .err — one cheap SSH call.
        # ``stat -c '%Y\t%s\t%n'`` works on GNU coreutils (FASRC); falls
        # through with empty output if the dir doesn't exist yet.
        cmd = (
            f"{{ [ -d {shlex.quote(log_dir)} ] && "
            f"find {shlex.quote(log_dir)} -maxdepth 2 -type f "
            f"\\( -name '*.out' -o -name '*.err' \\) "
            f"-printf '%T@\\t%s\\t%p\\n' 2>/dev/null "
            f"| sort -rn -k1,1 | head -240 ; }}; exit 0"
        )
        rc, out, _err = STATE.ssh.run(cmd, timeout=15)
        files: Dict[str, Dict[str, Any]] = {}     # keyed by base name
        if rc == 0:
            for line in out.splitlines():
                parts = line.split("\t")
                if len(parts) != 3:
                    continue
                try:
                    mtime = float(parts[0])
                    size  = int(parts[1])
                except ValueError:
                    continue
                full = parts[2]
                base = os.path.basename(full)
                if base.endswith(".out"):
                    stem, kind = base[:-4], "out"
                elif base.endswith(".err"):
                    stem, kind = base[:-4], "err"
                else:
                    continue
                rec = files.setdefault(stem, {"name": stem, "mtime": 0.0})
                rec[f"{kind}_path"] = full
                rec[f"{kind}_size"] = size
                rec["mtime"] = max(rec["mtime"], mtime)

        # 2. Overlay JobDB rows (gives us jobid + state + label + params).
        db_by_name: Dict[str, Dict[str, Any]] = {}
        for row in fasrc_jobs.DB.list_recent(120):
            lp = row.get("log_path") or ""
            base = os.path.basename(lp)
            stem = base[:-4] if base.endswith(".out") else base
            if stem:
                db_by_name[stem] = row

        runs: List[Dict[str, Any]] = []
        for stem, rec in files.items():
            db_row = db_by_name.get(stem) or {}
            try:
                params = json.loads(db_row.get("params_json") or "{}")
            except (TypeError, ValueError):
                params = {}
            runs.append({
                "name":         stem,
                "jobid":        db_row.get("jobid"),
                "label":        db_row.get("label"),
                "state":        db_row.get("state"),
                "submitted_at": db_row.get("submitted_at") or rec["mtime"],
                "started_at":   db_row.get("started_at"),
                "ended_at":     db_row.get("ended_at"),
                "out_path":     rec.get("out_path"),
                "err_path":     rec.get("err_path"),
                "out_size":     rec.get("out_size", 0),
                "err_size":     rec.get("err_size", 0),
                "mtime":        rec["mtime"],
                "params":       params,
            })

        # Also surface DB-known jobs whose log files have been deleted
        # (so the user can still see the row even if the .out is gone).
        for stem, row in db_by_name.items():
            if stem in files:
                continue
            try:
                params = json.loads(row.get("params_json") or "{}")
            except (TypeError, ValueError):
                params = {}
            runs.append({
                "name":         stem,
                "jobid":        row.get("jobid"),
                "label":        row.get("label"),
                "state":        row.get("state"),
                "submitted_at": row.get("submitted_at") or 0.0,
                "started_at":   row.get("started_at"),
                "ended_at":     row.get("ended_at"),
                "out_path":     row.get("log_path"),
                "err_path":     row.get("err_path"),
                "out_size":     0,
                "err_size":     0,
                "mtime":        row.get("submitted_at") or 0.0,
                "missing":      True,
                "params":       params,
            })
        runs.sort(key=lambda r: r["mtime"], reverse=True)
        return jsonify({"ok": True, "log_dir": log_dir, "runs": runs[:100]})

    @app.route("/api/fasrc/runs/log")
    def api_fasrc_runs_log():
        """Tail of one log file on FASRC.

        Path is supplied by the client (echoed back from ``/api/fasrc/runs``).
        We verify it falls under the configured logs dir and ends in
        ``.out`` / ``.err`` before reading — a stronger guarantee than
        relying on the URL not containing ``..``.
        """
        if not STATE.ssh or not STATE.ssh.is_connected():
            return jsonify({"ok": False, "error": "not connected"}), 400
        path = (request.args.get("path") or "").strip()
        try:
            lines = int(request.args.get("lines", 1000))
        except ValueError:
            lines = 1000
        lines = max(50, min(lines, 10_000))
        if not path:
            return jsonify({"ok": False, "error": "missing path"}), 400
        if not (path.endswith(".out") or path.endswith(".err")):
            return jsonify({"ok": False, "error": "path must end in .out or .err"}), 400
        cfg = fasrc_config.load()
        log_root = f"{cfg.repo_path}/{cfg.logs_subdir}/"
        if not path.startswith(log_root):
            return jsonify({"ok": False, "error": f"path must live under {log_root}"}), 400
        # Defensive: reject any sneaky path components.
        if ".." in path.split("/"):
            return jsonify({"ok": False, "error": "bad path"}), 400

        cmd = (
            f"{{ [ -f {shlex.quote(path)} ] && "
            f"  tail -n {lines} {shlex.quote(path)} 2>/dev/null || true ; "
            f"}}; exit 0"
        )
        rc, out, _err = STATE.ssh.run(cmd, timeout=20)
        if rc != 0:
            return jsonify({"ok": False, "error": "ssh tail failed"}), 500
        return jsonify({"ok": True, "path": path, "lines": lines,
                        "content": out})

    # ---- parsed live status (.out + .err + training_log.jsonl) -----------

    @app.route("/api/fasrc/training-status")
    def api_fasrc_training_status():
        """Single JSON dict the UI polls every few seconds.

        Identifies the currently running job (RUNNING state in squeue,
        cross-referenced against local sqlite), reads the tail of its
        ``.out`` / ``.err`` / ``training_log.jsonl`` over SSH, and
        returns a parsed summary. Errors are reported in-band as
        ``{"ok": False, "error": ...}`` rather than raising — a 5xx
        here would just look like the dashboard "disconnecting" to
        the user, when really the SSH is fine and only one log read
        misbehaved.
        """
        if not STATE.ssh or not STATE.ssh.is_connected():
            return jsonify({"ok": False, "error": "not connected"}), 400

        try:
            return _build_training_status()
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({
                "ok":    False,
                "error": f"{type(e).__name__}: {e}",
            }), 200

    def _build_training_status():
        cfg = fasrc_config.load()

        # 1. Identify the running job. Trust live squeue > sqlite.
        rc, sq_out, _err = STATE.ssh.run(
            f"squeue -h -u $USER --format='{fasrc_jobs.SQUEUE_FMT}'",
            timeout=10,
        )
        running_rows = []
        if rc == 0:
            for row in fasrc_jobs.parse_squeue(sq_out):
                if row.get("state") == "RUNNING":
                    running_rows.append(row)
        if not running_rows:
            return jsonify({"ok": True, "running": False,
                            "queue_rows": fasrc_jobs.parse_squeue(sq_out)
                                          if rc == 0 else []})

        # Prefer a row whose jobid matches one we submitted from this UI.
        known_ids = {r["jobid"] for r in fasrc_jobs.DB.list_recent(20)}
        live = next((r for r in running_rows if r["jobid"] in known_ids),
                    running_rows[0])
        jobid  = live["jobid"]
        stored = fasrc_jobs.DB.get(jobid)
        log_path = (stored or {}).get("log_path") \
                   or f"{cfg.repo_path}/logs/jobs/{live['name']}.out"
        err_path = (stored or {}).get("err_path") \
                   or log_path.replace(".out", ".err")
        # Trainer writes ``training_log.csv``; pre-CSV runs left
        # ``training_log.jsonl`` — try the new name first and fall back
        # so partially-mirrored ckpt dirs still light up the dashboard.
        train_log_csv   = f"{cfg.ckpt_dir}/training_log.csv"
        train_log_jsonl = f"{cfg.ckpt_dir}/training_log.jsonl"

        # 2. Fetch the relevant log tails in one SSH call. Wrapped in
        # ``{ ...; exit 0; }`` so a missing optional file (e.g. the
        # training log before the first eval) doesn't fail the whole
        # route. Every section is independently guarded with
        # ``2>/dev/null`` + ``|| true``. The training-log block first
        # tries the CSV, then the legacy JSONL — the parser auto-detects
        # the format from the leading character.
        cmd = (
            f"{{ "
            f"  echo __OUT__ ; "
            f"  tail -n 500 {shlex.quote(log_path)} 2>/dev/null || true ; "
            f"  echo __ERR__ ; "
            f"  tail -n 200 {shlex.quote(err_path)} 2>/dev/null || true ; "
            f"  echo __JSONL__ ; "
            # For the CSV path, the header is line 1 — we MUST keep it for
            # csv.DictReader. ``head -n 1 + tail -n 200`` covers files of
            # any size and the awk dedupes the duplicate when the file is
            # shorter than 200 rows (header would otherwise appear twice).
            f"  if [ -f {shlex.quote(train_log_csv)} ]; then "
            f"    {{ head -n 1 {shlex.quote(train_log_csv)} ; "
            f"       tail -n 200 {shlex.quote(train_log_csv)} ; }} 2>/dev/null "
            f"    | awk '!seen[$0]++' || true ; "
            f"  elif [ -f {shlex.quote(train_log_jsonl)} ]; then "
            f"    tail -n 200 {shlex.quote(train_log_jsonl)} 2>/dev/null || true ; "
            f"  fi ; "
            f"}}; exit 0"
        )
        rc, out, _err = STATE.ssh.run(cmd, timeout=15)
        # rc should always be 0 thanks to ``exit 0``; if it isn't, ssh
        # itself failed (e.g. socket died). Surface a clear message
        # instead of a generic 500.
        if rc != 0:
            return jsonify({
                "ok":    False,
                "error": "ssh failed to read remote logs — "
                         "try refreshing or reconnecting",
            })

        sections = {"__OUT__": "", "__ERR__": "", "__JSONL__": ""}
        current  = None
        for line in out.splitlines():
            if line in sections:
                current = line
                continue
            if current:
                sections[current] += line + "\n"

        elapsed_s = fasrc_jobs.parse_slurm_time(live.get("time"))

        history_spt = fasrc_jobs.secs_per_step_history()
        summary = fasrc_log_parser.summarise(
            out_text=sections["__OUT__"],
            err_text=sections["__ERR__"],
            jsonl_text=sections["__JSONL__"],
            elapsed_seconds=elapsed_s,
            history_secs_per_step=history_spt,
        )

        # 3. Side-effect: keep sqlite up to date so /api/fasrc/jobs is
        # accurate for the recent-submissions panel.
        if summary["progress"]:
            fasrc_jobs.DB.update_progress(
                jobid,
                summary["progress"]["current"],
                summary["progress"]["total"],
            )
        if stored and stored["started_at"] is None:
            fasrc_jobs.DB.update_state(
                jobid, state="RUNNING",
                started_at=time.time() - elapsed_s,
            )

        # 4. Activate the auto-mirror during the train stage and trigger
        # an immediate sync whenever a fresh "Checkpoint saved" line
        # shows up. ``MIRROR.trigger()`` calls rsync synchronously and
        # can block for minutes on large checkpoint dirs — dispatch on
        # a daemon thread so the status poll stays snappy.
        if summary["stage"] == "train":
            if not MIRROR.status.enabled:
                MIRROR.start()
            if (summary["last_checkpoint"]
                    and MIRROR.status.last_checkpoint_line
                        != summary["last_checkpoint"]):
                MIRROR.status.last_checkpoint_line = summary["last_checkpoint"]
                import threading as _t
                _t.Thread(target=MIRROR.trigger, daemon=True,
                          name="mirror-trigger").start()

        return jsonify({
            "ok":       True,
            "running":  True,
            "job": {
                "jobid":           jobid,
                "name":            live.get("name", ""),
                "state":           live.get("state", ""),
                "elapsed_seconds": elapsed_s,
                "elapsed":         live.get("time", ""),
                "time_limit":      live.get("time_limit", ""),
                "node":            live.get("reason", ""),
                "start_time":      live.get("start_time", ""),
                "log_path":        log_path,
                "err_path":        err_path,
                "label":           (stored or {}).get("label", ""),
                "params":          json.loads((stored or {}).get("params_json") or "null"),
            },
            "stage":             summary["stage"],
            "stage_index":       summary["stage_index"],
            "pipeline_done":     summary["pipeline_done"],
            "progress":          summary["progress"],
            "latest_metrics":    summary["latest_metrics"],
            "last_checkpoint":   summary["last_checkpoint"],
            "validations":       summary["validations"],
            "latest_validation": summary["latest_validation"],
            "eta_seconds":       summary["eta_seconds"],
            "queue_rows":        running_rows,
        })

    # ---- live log stream (SSE) -------------------------------------------

    @app.route("/api/fasrc/log/<jobid>")
    def api_fasrc_log_stream(jobid: str):
        if not jobid.isdigit():
            abort(400)
        row = fasrc_jobs.DB.get(jobid)
        if not row:
            abort(404)
        log_path = row["log_path"]
        # Stream both files in case the user wants stderr (`?which=err`).
        which = request.args.get("which", "out")
        if which == "err":
            log_path = row["err_path"]

        def _gen():
            if not STATE.ssh or not STATE.ssh.is_connected():
                yield "event: error\ndata: not connected\n\n"
                return
            # tail with retry — file may not exist until SLURM starts the job.
            cmd = (f"tail -F -n 200 {log_path} 2>/dev/null || "
                   f"(while [ ! -f {log_path} ]; do sleep 2; done && "
                   f" tail -F -n 200 {log_path})")
            try:
                for line in STATE.ssh.stream(cmd):
                    prog = fasrc_jobs.parse_progress(line)
                    if prog:
                        fasrc_jobs.DB.update_progress(jobid, prog[0], prog[1])
                    # SSE framing: one event per line, multiline data uses
                    # repeated ``data:`` lines.
                    safe = line.replace("\r", "")
                    yield f"data: {safe}\n\n"
            except SSHError as e:
                yield f"event: error\ndata: {e}\n\n"
        return Response(stream_with_context(_gen()),
                        mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache",
                                 "X-Accel-Buffering": "no"})

    # ---- checkpoint auto-mirror -------------------------------------------

    @app.route("/api/fasrc/mirror/status")
    def api_fasrc_mirror_status():
        s = MIRROR.status
        return jsonify({
            "enabled":     s.enabled,
            "last_run_at": s.last_run_at,
            "last_rc":     s.last_rc,
            "last_error":  s.last_error,
            "last_stdout": s.last_stdout,
            "remote_dir":  s.remote_dir,
            "local_dir":   s.local_dir,
            "period_seconds": MIRROR.period,
        })

    @app.route("/api/fasrc/mirror/start", methods=["POST"])
    def api_fasrc_mirror_start():
        if not STATE.ssh or not STATE.ssh.is_connected():
            return jsonify({"ok": False, "error": "not connected"}), 400
        try:
            MIRROR.period = max(15, int(request.form.get("period", 60)))
        except ValueError:
            pass
        MIRROR.start()
        return jsonify({"ok": True, "status": api_fasrc_mirror_status().json})

    @app.route("/api/fasrc/mirror/stop", methods=["POST"])
    def api_fasrc_mirror_stop():
        MIRROR.stop()
        return jsonify({"ok": True})

    @app.route("/api/fasrc/mirror/trigger", methods=["POST"])
    def api_fasrc_mirror_trigger():
        if not STATE.ssh or not STATE.ssh.is_connected():
            return jsonify({"ok": False, "error": "not connected"}), 400
        MIRROR.trigger()
        return jsonify({"ok": True})

    # ---- per-stage timings (CSV from run_pipeline.py's StageTimer) -------

    @app.route("/api/fasrc/stages/<jobid>")
    def api_fasrc_stages(jobid: str):
        """Parse the remote ``stages_<jobid>.csv`` into JSON rows so the
        UI can render the per-stage breakdown. The CSV lives next to the
        TFRecords on netscratch (see ``run_pipeline.py``'s ``--stages-csv``
        default)."""
        if not jobid.isdigit() and jobid != "local":
            abort(400)
        if not STATE.ssh or not STATE.ssh.is_connected():
            return jsonify({"ok": False, "error": "not connected"}), 400
        cfg = fasrc_config.load()
        path = f"{cfg.data_dir}/images/records_v2/stages_{jobid}.csv"
        rc, out, err = STATE.ssh.run(
            f"if [ -f {shlex.quote(path)} ]; then "
            f"  cat {shlex.quote(path)}; "
            f"else "
            f"  echo MISSING; "
            f"fi", timeout=10,
        )
        if rc != 0:
            return jsonify({"ok": False, "error": err.strip()}), 500
        if out.strip() == "MISSING":
            return jsonify({"ok": True, "path": path, "rows": []})

        import csv
        import io as _io
        reader = csv.DictReader(_io.StringIO(out))
        rows = []
        for r in reader:
            try:
                rows.append({
                    "stage":             r.get("stage", ""),
                    "started_at":        float(r.get("started_at",  "0") or 0),
                    "ended_at":          float(r.get("ended_at",    "0") or 0),
                    "duration_seconds":  float(r.get("duration_seconds", "0") or 0),
                    "params_dependent":  bool(int(r.get("params_dependent", "0") or 0)),
                    "n_train":           r.get("n_train", ""),
                    "n_valid":           r.get("n_valid", ""),
                    "image_size":        r.get("image_size", ""),
                    "batch_size":        r.get("batch_size", ""),
                    "steps":             r.get("steps", ""),
                })
            except (ValueError, TypeError):
                # Skip malformed rows (e.g. a partial write captured mid-flight).
                continue
        return jsonify({"ok": True, "path": path, "rows": rows})

    # ---- conda env update -------------------------------------------------

    def _build_env_update_cmd(cfg) -> str:
        """`module load python` + `yes | mamba env update -p … -f environment.yml`.

        FASRC uses lmod, which is exposed as the ``module`` shell function
        once ``/etc/profile.d/lmod.sh`` is sourced — non-interactive SSH
        bash doesn't pull that automatically, so we do it ourselves.
        The ``yes |`` keeps mamba 2.x's "Proceed ([y]/n)?" prompt from
        stalling the stream.
        """
        return (
            "set -o pipefail; "
            "[ -f /etc/profile.d/lmod.sh ] && source /etc/profile.d/lmod.sh; "
            f"cd {shlex.quote(cfg.repo_path)} && "
            "module purge 2>/dev/null || true; "
            "module load python && "
            "echo '--- mamba: '$(which mamba) && "
            f"yes | mamba env update -p {shlex.quote(cfg.conda_env_path)} "
            "-f environment.yml 2>&1"
        )

    @app.route("/api/fasrc/env-update")
    def api_fasrc_env_update():
        if not STATE.ssh or not STATE.ssh.is_connected():
            return Response(
                "event: error\ndata: not connected\n\n",
                mimetype="text/event-stream", status=400,
            )
        cfg = fasrc_config.load()
        cmd = _build_env_update_cmd(cfg)

        def _gen():
            yield f"data: $ remote: cd {cfg.repo_path}\n\n"
            yield f"data: $ module load python\n\n"
            yield (f"data: $ yes | mamba env update -p "
                   f"{cfg.conda_env_path} -f environment.yml\n\n")
            try:
                for line in STATE.ssh.stream(cmd):
                    yield f"data: {line.replace(chr(13), '')}\n\n"
                yield "event: done\ndata: complete\n\n"
            except SSHError as e:
                yield f"event: error\ndata: {e}\n\n"
        return Response(stream_with_context(_gen()),
                        mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache",
                                 "X-Accel-Buffering": "no"})

    return app


def main() -> None:
    """Run the Flask app on 127.0.0.1:8765."""
    import argparse
    ap = argparse.ArgumentParser(description="EuclidPolish localhost web UI")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    app = create_app()
    print(f"\nEuclidPolish web UI on http://{args.host}:{args.port}\n")
    app.run(host=args.host, port=args.port, debug=args.debug,
            use_reloader=False)


if __name__ == "__main__":
    main()
