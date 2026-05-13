"""
Flask app factory + routes for the EuclidPolish web UI.
"""

from __future__ import annotations

import glob
import io
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from flask import (
    Flask, abort, jsonify, render_template, request, send_file, url_for,
)

from euclid_polish.config import BandConfig, Config
from euclid_polish.euclid.catalog import StarCatalog
from euclid_polish.euclid.psf_library import (
    load_all_band_psfs, psf_inventory, psf_path_for_band,
)
from euclid_polish.euclid.types import PSF
from euclid_polish.web.jobs import REGISTRY


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
    from euclid_polish.sky.tfrecord import write_multiband_skyimages

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
    for subset, n, seed in (("train", n_train, 0), ("validate", n_valid, n_train)):
        imgs = []
        for i in range(n):
            sky, _ = sim.simulate_field(np.random.default_rng(seed + i))
            sky.index = i
            sky.subset = subset
            imgs.append(sky)
            done += 1
            cap.tick(done, total_n, f"generating {subset} {i+1}/{n}")
        path = write_multiband_skyimages(imgs, f"clean_{subset}",
                                         records_dir=Config.RECORDS_DIR_V2)
        print(f"  ✓ {path}")
        result[subset] = {"path": path, "count": len(imgs)}
    return result


def _job_forward(cap) -> Dict[str, Any]:
    """Apply the multi-band forward model with progress tracking."""
    import tensorflow as tf
    from euclid_polish.euclid.psf_library import load_all_band_psfs
    from euclid_polish.sky.multiband_forward import (
        MultiBandForward, MultiBandForwardConfig,
    )
    from euclid_polish.sky.tfrecord import (
        tfrecord_path, write_multiband_skyimages,
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
        records = list(tf.data.TFRecordDataset(clean))
        rng = np.random.default_rng(0xEC11D + (1 if subset == "validate" else 0))
        lr_imgs, hr_imgs = [], []
        for raw in records:
            hr_4ch = MultiBandSkyImage.from_tfrecord(raw)
            lr, hr = fwd.process(hr_4ch, rng=rng)
            lr_imgs.append(lr); hr_imgs.append(hr)
            done += 1
            cap.tick(done, grand_total, f"forward-model {subset}")
        # Do NOT overwrite clean_{subset}: that 4-band file is the
        # inspection-friendly clean record. The 1-channel VIS HR target
        # used by training is written to a separate ``hr_{subset}`` file.
        write_multiband_skyimages(hr_imgs, f"hr_{subset}",
                                  records_dir=Config.RECORDS_DIR_V2)
        write_multiband_skyimages(lr_imgs, f"dirty_{subset}",
                                  records_dir=Config.RECORDS_DIR_V2)
        result[subset] = {"n": len(records)}
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


def _job_plot_training_log(cap, checkpoint_dir: str) -> Dict[str, Any]:
    """Render the training-log PNG (loss + PSNR vs step)."""
    from euclid_polish.training.log_plot import plot_training_log
    log_path = os.path.join(checkpoint_dir, "training_log.jsonl")
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"no training_log.jsonl in {checkpoint_dir}")
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
    hr_path = tfrecord_path(Config.RECORDS_DIR_V2, f"clean_{subset}")
    if not os.path.exists(lr_path):
        raise FileNotFoundError(f"no records in {Config.RECORDS_DIR_V2}")
    lr_records = read_multiband_skyimages(lr_path, num_images=10_000)
    hr_records = (read_multiband_skyimages(hr_path, num_images=10_000)
                  if os.path.exists(hr_path) else [])
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
        hr_data = hr_by_idx[lr_img.index].data if lr_img.index in hr_by_idx else None
        out = os.path.join(out_dir, f"reconstruct_idx{lr_img.index:04d}.png")
        plot_reconstruction(lr_data, sr_data, hr_data=hr_data, output_path=out)
        out_paths.append(out)
        cap.tick(k + 1, n, f"reconstructing idx {lr_img.index}")
        print(f"  ✓ {out}")
    return {"output_dir": out_dir, "n": len(out_paths), "paths": out_paths}


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
        log_path = os.path.join(ckpt, "training_log.jsonl")
        out_png  = os.path.join(Config.VIS_DIR, "training_log.png")
        if not os.path.exists(log_path):
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
