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

from euclid_polish.sky.cosmos2025 import open_cosmos2025
from euclid_polish.sky.multiband_generator import (
MultiBandGeneratorConfig, MultiBandSimulator,
)
from euclid_polish.sky.tfrecord import open_multiband_writer
import tensorflow as tf
from euclid_polish.euclid.psf_library import load_all_band_psfs
from euclid_polish.sky.multiband_forward import (
MultiBandForward, MultiBandForwardConfig,
)
from euclid_polish.sky.tfrecord import (
open_multiband_writer, tfrecord_path,
)
from euclid_polish.sky.types import MultiBandSkyImage
from tf_keras.optimizers.schedules import PiecewiseConstantDecay
from euclid_polish.training import Trainer
from euclid_polish.training.data_multiband import MultiBandEuclidDataset
from euclid_polish.training.models.wdsr import wdsr
from euclid_polish.training.inference import load_model_from_checkpoint
from euclid_polish.training.log_plot import plot_training_log
from euclid_polish.sky.tfrecord import (
read_multiband_skyimages, tfrecord_path,
)
from euclid_polish.training.inference import (
load_model_from_checkpoint, reconstruct, plot_reconstruction,
scaled_wcs_header,
)
from astropy.io import fits as _fits
from astropy.io import fits
from euclid_polish.euclid.downloader import fetch_cutout_at
from scipy import signal as scipy_signal
from euclid_polish.sky.multiband_forward import MultiBandForward
from euclid_polish.visualization.methods import draw_star_positions
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from astropy.visualization import (
AsinhStretch, ImageNormalize, MinMaxInterval,
)
from PIL import Image
from euclid_polish.web.fasrc_fetcher import _local_path_for
from euclid_polish.visualization.color import calibrated_rgb_panel
from euclid_polish.visualization.methods import plot_star_positions
from euclid_polish.sky.tfrecord import tfrecord_path
import os as _os
from euclid_polish.euclid import auth
from euclid_polish.web.fasrc_fetcher import list_remote_dir
from euclid_polish.web import fasrc_fetcher as _fasrc_fetcher
import importlib.util
from scipy.signal import fftconvolve
from matplotlib.colors import AsinhNorm
import importlib.util as _ilu
from euclid_polish.web.fasrc_fetcher import (
is_allowed_remote_path, run_remote_python,
)
from euclid_polish.web.fasrc_pipeline import (
    REGISTRY as STEP_REGISTRY, StepResources, render_sbatch_body,
)
from euclid_polish.web.job_status import JobStatusFetcher
import io as _io
from euclid_polish.training.log_plot import plot_training_records
import traceback
import threading as _t
import csv
import argparse
matplotlib.use("Agg")

import glob
import io
import json
import os
import re
import shlex
import shutil
import textwrap
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from flask import (
    Flask, Response, abort, jsonify, redirect, render_template, request,
    send_file, stream_with_context, url_for,
)

from euclid_polish.config import BandConfig, Config
from euclid_polish.euclid.catalog import StarCatalog
from euclid_polish.euclid.psf_library import (
    load_all_band_psfs, load_all_band_psf_sets, psf_inventory, psf_path_for_band,
)
from euclid_polish.euclid.types import PSF
from euclid_polish.web import (
    fasrc_config, fasrc_jobs, fasrc_log_parser, git_ops,
)
from euclid_polish.web.fasrc_mirror import MIRROR
from euclid_polish.web.jobs import REGISTRY
from euclid_polish.web.remote import (
    RemoteState, SSHConfig, SSHError, SSHSession, STATE,
)


# Regex guard for cutout filenames: alphanumerics, underscores, dashes, dots.
# Rejects any path-traversal sequences ("..", "/").
_CUTOUT_FNAME_RE = re.compile(r"^[A-Za-z0-9_.\-]+\.fits$")


# ---------------------------------------------------------------------------
# Read-only status helpers
# ---------------------------------------------------------------------------

def _fasrc_catalog_remote_path() -> str:
    """Remote path of the catalog the FASRC-side query writes."""
    cfg = fasrc_config.load()
    return f"{cfg.data_dir}/euclid_stars/{Config.CATALOG_FILE}"


def _fasrc_catalog_dir(force: bool = True) -> Optional[str]:
    """Pull the FASRC ``stars.csv`` into the local cache; return the dir
    holding it (for ``StarCatalog(<dir>)``), or None if the remote catalog
    isn't there / can't be fetched.

    The brightest-N query runs on the FASRC login node and writes the
    catalog to netscratch ``$DATA_DIR/euclid_stars``. The laptop UI must
    read *that* copy — never a stale local one — for the summary + plots,
    so we rsync it down on demand."""
    res = _fasrc_fetcher.fetch_one_file(_fasrc_catalog_remote_path(), force=force)
    if not res.ok or not res.local_path or not os.path.isfile(res.local_path):
        return None
    return os.path.dirname(res.local_path)


def _catalog_status() -> Dict[str, Any]:
    cat_dir = _fasrc_catalog_dir()
    if cat_dir is None:
        return {"present": False}
    cat = StarCatalog(cat_dir)
    if not cat.exists():
        return {"present": False}
    summary = cat.get_summary()
    # Show the *remote* path so the page makes clear this is the FASRC
    # (netscratch) catalog, not the local cache copy we render from.
    return {"present": True, "summary": summary,
            "path": _fasrc_catalog_remote_path()}


def _fasrc_psf_dir(force: bool = True) -> Optional[str]:
    """Pull the FASRC Euclid ePSFs into the local cache; return the dir
    holding them, or None if none are on FASRC yet.

    The all-band extraction job writes ``euclid_psf_<band>.fits`` to
    netscratch ``$DATA_DIR/euclid_psf``, so the laptop UI must rsync those
    down and read them — not a stale local copy. The four band files share
    one remote dir, so they land in one local cache dir."""
    cfg = fasrc_config.load()
    local_dir: Optional[str] = None
    for band in Config.BANDS:
        remote = f"{cfg.data_dir}/euclid_psf/{band.psf_fits_filename}"
        res = _fasrc_fetcher.fetch_one_file(remote, force=force)
        if res.ok and res.local_path and os.path.isfile(res.local_path):
            local_dir = os.path.dirname(res.local_path)
    return local_dir


def _psf_status() -> Dict[str, Any]:
    psf_dir = _fasrc_psf_dir()
    # No FASRC PSFs yet → every band shows the Gaussian fallback (rather
    # than reading whatever stale ePSF might sit in the local data dir).
    inv = (psf_inventory(psf_dir=psf_dir) if psf_dir
           else {b.name: None for b in Config.BANDS})
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
                # Number of position-dependent cluster PSFs in the file
                # (NPSF header on the multi-extension format; 1 for a legacy
                # single-PSF FITS).
                with fits.open(path) as _hdul:
                    item["n_psf"] = int(_hdul[0].header.get("NPSF", 1))
            except Exception as e:
                item["error"] = str(e)
        bands.append(item)
    return {"bands": bands}


def _submit_quick_fasrc_job(
    ssh, *,
    argv: List[str],
    label: str,
    step_id: str,
    resources: StepResources,
    params: Dict[str, Any],
):
    """Render + submit a short CPU SLURM job for a quick FASRC task
    (catalog query / photometry verify) and register it in the JobDB.

    The login-node path (a blocking SSH command inside a daemon thread)
    cannot be cancelled — there is no stop hook. Routing the same work
    through SLURM instead means the job lands in the JobDB, shows up in the
    FASRC job panel, and is cancellable via the standard ``scancel`` button
    on ``/fasrc``. Returns ``(slurm_id, payload)`` from
    :func:`fasrc_jobs.submit_sbatch_script` (``slurm_id`` is None on failure,
    ``payload`` the jsonify-able dict to return either way)."""
    cfg = fasrc_config.load()
    ts = time.strftime("%Y%m%d-%H%M%S")
    job_name = f"euclid-{step_id}-{ts}"
    built = render_sbatch_body(
        job_name=job_name,
        relative_log_dir="logs/euclid_pipeline",
        resources=resources,
        cfg=cfg,
        label=label,
        cmd_argv=argv,
        banner_line=f"Web-submitted job: {step_id} - {label}",
        step_id=step_id,
    )
    params_for_db = dict(params)
    params_for_db.update(resources.to_dict())
    params_for_db["step_id"] = step_id
    return fasrc_jobs.submit_sbatch_script(
        ssh, cfg=cfg, built=built, label=label,
        params=params_for_db, step_id=step_id,
    )


def _tfrecords_status(records_dir: Optional[str] = None) -> Dict[str, Any]:
    """List on-disk TFRecord files under ``records_dir``.

    Defaults to ``Config.RECORDS_DIR_V2`` (the locally-generated sky
    records) so existing callers stay unchanged. The HST Catalog
    passes its own dir to reuse this for the FASRC-cached records.
    """
    d = records_dir or Config.RECORDS_DIR_V2
    out = {"dir": d, "files": []}
    if os.path.isdir(d):
        for fname in sorted(os.listdir(d)):
            full = os.path.join(d, fname)
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
    """Recent PNGs under data/vis/, newest first.

    Each entry includes ``inspect_fits`` — a project-relative path to a
    same-stem ``.fits`` sibling if one exists. The visualization gallery
    uses this to route the thumbnail click to ``/inspect`` instead of
    just popping the raw PNG.
    """
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
            inspect_fits = None
            stem = os.path.splitext(full)[0]
            for ext in (".fits", ".fit"):
                cand = stem + ext
                if os.path.isfile(cand):
                    inspect_fits = _safe_relpath(os.path.realpath(cand))
                    break
            pngs.append({
                "rel":          rel,
                "mtime":        mtime,
                "size_kb":      round(size_kb, 1),
                "inspect_fits": inspect_fits,
            })
    pngs.sort(key=lambda d: d["mtime"], reverse=True)
    return pngs


# ---------------------------------------------------------------------------
# Background-job target functions
# ---------------------------------------------------------------------------

def _resolve_training_log(checkpoint_dir: str) -> Optional[str]:
    """Pick the current ``training_log.csv`` or fall back to legacy
    ``training_log.jsonl`` so logs from runs before the CSV switch still
    plot. Returns the path that exists, or None if neither does."""
    for name in ("training_log.csv", "training_log.jsonl"):
        p = os.path.join(checkpoint_dir, name)
        if os.path.exists(p):
            return p
    return None


def _login_node_generate_cmd(cfg, remote_tmp: str, hr_image_size: int,
                             n_pairs: int) -> str:
    """Shell command that generates ``n_pairs`` synthetic *validate* pairs
    at ``hr_image_size`` on the FASRC **login node** — a plain command, not
    an sbatch job.

    Reuses ``scripts/run_pipeline.py`` (the same generator ``/sky`` uses);
    with ``EUCLID_POLISH_DATA_DIR`` pointed at the netscratch data dir, the
    script's defaults resolve the 10 GB COSMOS catalog and the FASRC ePSFs.
    Output TFRecords go to the throwaway ``remote_tmp`` so the training
    records are never touched. Mirrors the conda activation prologue the
    sbatch wrapper uses, minus the GPU module (generation is CPU-only).
    """
    q = shlex.quote
    return textwrap.dedent(f"""
        set -e
        export EUCLID_POLISH_DATA_DIR={q(cfg.data_dir)}
        mkdir -p {q(remote_tmp)}
        module purge 2>/dev/null || true
        module load python 2>/dev/null || true
        if [ -z "${{CONDA_SHLVL:-}}" ]; then
          CONDA_BASE="$(conda info --base 2>/dev/null || true)"
          if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
            source "$CONDA_BASE/etc/profile.d/conda.sh"
          fi
          if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/mamba.sh" ]; then
            source "$CONDA_BASE/etc/profile.d/mamba.sh"
          fi
        fi
        mamba activate {q(cfg.conda_env_path)}
        cd {q(cfg.repo_path)}
        python -u scripts/run_pipeline.py \
          --ntrain 0 --nvalid {int(n_pairs)} --image-size {int(hr_image_size)} \
          --records-dir {q(remote_tmp)} --skip-train --gen-workers 1
    """).strip()


def _job_generate_reconstruct(
    cap, checkpoint_dir: str, num_res_blocks: int,
    hr_image_size: int, n_pairs: int,
    asinh_scale: Optional[float] = None,
) -> Dict[str, Any]:
    """Generate fresh synthetic pair(s) on the FASRC login node, pull them
    down, run the model locally, and render LR | SR | HR | forward(SR) |
    residual with the FASRC PSF the checkpoint trained against.

    Flow: (1) pull the Euclid ePSFs from FASRC; (2) run ``run_pipeline.py``
    on the **login node** (not sbatch) writing one-or-more validate pairs to
    a throwaway remote dir; (3) rsync them down; (4) WDSR inference locally;
    (5) ``forward(SR)`` with the *same* FASRC PSF; (6) write FITS + PNG into
    ``VIS_RECONSTRUCTION_DIR``. The remote/local temp dirs are cleaned up.
    """
    if STATE.ssh is None or not STATE.ssh.is_connected():
        raise RuntimeError(
            "not connected to FASRC — connect on the FASRC tab first; "
            "login-node generation needs the SSH session"
        )
    cfg = fasrc_config.load()
    total = n_pairs + 2  # PSF pull + login-node gen + one tick per scene

    # 1. Pull the FASRC ePSFs so generation AND the local forward(SR) use
    #    the same PSF the checkpoint trained against. Writes to the cache,
    #    never data/euclid_psf.
    cap.tick(0, total, "pulling FASRC ePSFs")
    psf_dir = _fasrc_psf_dir(force=True)
    if not psf_dir:
        raise FileNotFoundError("no Euclid ePSFs on FASRC to pull")
    print(f"  ✓ FASRC ePSFs → {psf_dir}")

    # 2. Generate on the login node into a throwaway dir.
    remote_tmp = f"{cfg.data_dir}/_inference_gen/{uuid.uuid4().hex}"
    local_tmp = os.path.join(Config.DATA_DIR, "_inference_gen", uuid.uuid4().hex)
    cap.tick(1, total,
             f"generating {n_pairs} pair(s) @ {hr_image_size}px on FASRC login node")
    print(f"  login-node generate → {remote_tmp}")
    try:
        rc, out, err = STATE.ssh.run(
            _login_node_generate_cmd(cfg, remote_tmp, hr_image_size, n_pairs),
            timeout=900,
        )
        if rc != 0:
            tail = (err.strip() or out.strip())[-2000:]
            raise RuntimeError(f"login-node generation failed (rc={rc}):\n{tail}")

        # 3. Pull the pair(s) down. ``rsync -a`` tries to preserve perms, so
        #    a Linux→macOS pull can exit rc=23 ("unable to escalate mode")
        #    while still copying every file — don't fail on rc alone; the
        #    file-existence check below is the real gate.
        rc, _o, perr = STATE.ssh.rsync_pull(remote_tmp + "/", local_tmp, timeout=600)
        if rc != 0:
            print(f"  ⚠ rsync exited rc={rc} (continuing): {perr.strip()[:300]}")

        lr_path    = tfrecord_path(local_tmp, "dirty_validate")
        hr_path    = tfrecord_path(local_tmp, "hr_validate")
        clean_path = tfrecord_path(local_tmp, "clean_validate")
        if not os.path.exists(lr_path):
            raise FileNotFoundError(
                f"login-node generation produced no dirty records in {local_tmp}")
        lr_records    = read_multiband_skyimages(lr_path, num_images=10_000)
        hr_records    = (read_multiband_skyimages(hr_path, num_images=10_000)
                         if os.path.exists(hr_path) else [])
        clean_records = (read_multiband_skyimages(clean_path, num_images=10_000)
                         if os.path.exists(clean_path) else [])
        hr_by_idx    = {h.index: h for h in hr_records}
        clean_by_idx = {c.index: c for c in clean_records}

        # 4. Model — load once.
        scale = Config.DEFAULT_REBIN_FACTOR
        if not tf.train.latest_checkpoint(checkpoint_dir):
            raise FileNotFoundError(f"no checkpoint in {checkpoint_dir}")
        model = load_model_from_checkpoint(
            checkpoint_dir, scale, num_res_blocks,
            nchan_in=Config.NUM_LR_CHANNELS, nchan_out=Config.NUM_HR_CHANNELS,
        )

        out_dir = Config.VIS_RECONSTRUCTION_DIR
        os.makedirs(out_dir, exist_ok=True)
        out_paths = []
        for k, lr_img in enumerate(lr_records):
            # Keep the full 4-band LR cube for the color composite — the
            # 2-D ``lr_data`` returned by reconstruct() is VIS-only.
            lr_cube_for_color = (lr_img.data
                                 if lr_img.data.ndim == 3
                                    and lr_img.data.shape[-1] == Config.NUM_LR_CHANNELS
                                 else None)
            lr_data, sr_data = reconstruct(model, lr_img.data)

            # HR color from the CLEAN (noise-free) record; residual/PSNR from
            # the 1-channel hr_<subset>, falling back to clean channel 0.
            hr_cube_for_color = None
            if lr_img.index in clean_by_idx:
                raw = clean_by_idx[lr_img.index].data
                if raw.ndim == 3 and raw.shape[-1] == Config.NUM_LR_CHANNELS:
                    hr_cube_for_color = raw
            hr_data = None
            if lr_img.index in hr_by_idx:
                raw = hr_by_idx[lr_img.index].data
                hr_data = raw[..., 0] if raw.ndim == 3 else raw
            elif hr_cube_for_color is not None:
                hr_data = hr_cube_for_color[..., 0]

            # forward(SR) with the FASRC PSF (matches gen + training).
            predicted = residual = None
            try:
                predicted, residual = _forward_model_sr_residual(
                    sr_data, lr_data, psf_dir=psf_dir)
            except Exception as e:  # noqa: BLE001 — panel is best-effort
                print(f"  ⚠ forward(SR) failed: {e}")

            stem = f"gensynth_{hr_image_size}px_idx{lr_img.index:04d}"
            out = os.path.join(out_dir, stem + ".png")
            plot_reconstruction(lr_data, sr_data, hr_data=hr_data,
                                output_path=out,
                                lr_cube=lr_cube_for_color,
                                hr_cube=hr_cube_for_color,
                                asinh_scale=asinh_scale,
                                predicted_dirty=predicted,
                                residual=residual)

            def _write_fits(name: str, data2d, label: str) -> None:
                if data2d is None:
                    return
                arr = np.ascontiguousarray(np.asarray(data2d, dtype=np.float32))
                hdu = _fits.PrimaryHDU(arr)
                hdu.header["OBJECT"] = (f"EuclidPolish {label} (VIS)", "panel label")
                hdu.header["IDX"]    = (int(lr_img.index), "scene index")
                hdu.header["HRSIZE"] = (int(hr_image_size), "HR side px (0.05in/px)")
                hdu.header["CKPT"]   = (str(checkpoint_dir)[:60], "checkpoint dir")
                hdu.header["PSFSRC"] = ("FASRC", "ePSF pulled from FASRC (training PSF)")
                hdu.header["ASINH"]  = (float(asinh_scale or Config.STRETCH_SCALE_E),
                                        "asinh stretch knee used for the plot")
                hdu.header["BUNIT"]  = ("e-", "electrons (raw, sign preserved)")
                hdu.writeto(os.path.join(out_dir, name), overwrite=True)

            _write_fits(stem + ".fits",           sr_data,   "SR")
            _write_fits(stem + "_lr.fits",        lr_data,   "LR dirty")
            _write_fits(stem + "_hr.fits",        hr_data,   "HR clean")
            _write_fits(stem + "_srforward.fits", predicted, "VIS PSF (x) SR + 2x rebin")
            _write_fits(stem + "_residual.fits",  residual,  "LR - forward(SR)")
            out_paths.append(out)
            cap.tick(k + 2, total, f"reconstructed scene {lr_img.index}")
            print(f"  ✓ {out}")
        return {"output_dir": out_dir, "n": len(out_paths), "paths": out_paths}
    finally:
        # Best-effort cleanup: remote throwaway dir + local temp pull.
        try:
            STATE.ssh.run(f"rm -rf {shlex.quote(remote_tmp)}", timeout=30)
        except Exception:  # noqa: BLE001
            pass
        shutil.rmtree(local_tmp, ignore_errors=True)


def _forward_model_sr_residual(
    sr_data: np.ndarray, lr_vis: np.ndarray,
    psf_dir: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Push SR back through the VIS forward chain and diff against the LR.

    Convolves SR with the empirical VIS ePSF, sum-rebins ×2 to the
    0.10″/pix LR grid (the deterministic ``EuclidVISForwardOp`` chain),
    crops to the common shape, and returns ``(predicted_dirty, residual)``
    with ``residual = lr_vis − predicted_dirty``. This is the round-trip
    self-consistency check — a well-behaved model reproduces the observed
    Euclid LR. May raise on PSF-load / shape errors; callers handle it.

    ``psf_dir`` overrides which Euclid ePSFs to convolve with. The
    generate+reconstruct path passes the FASRC-pulled PSF dir so the
    forward op uses the *same* PSF the checkpoint trained against (and
    that the login-node generation used), not the local committed copy.
    """
    psfs = load_all_band_psfs(
        target_pixel_scale=Config.DEFAULT_PIXEL_SCALE,
        **({"psf_dir": psf_dir} if psf_dir else {}),
    )
    vis_psf = psfs[Config.BAND_VIS.name]
    sr_hr = scipy_signal.fftconvolve(
        sr_data, vis_psf.data, mode="same",
    ).astype(np.float32)
    rebin_factor = int(round(
        Config.BAND_VIS.pixel_scale_lr_arcsec / Config.DEFAULT_PIXEL_SCALE
    ))
    predicted = MultiBandForward.sum_rebin(sr_hr, rebin_factor)
    # ``sum_rebin`` may trim a row/col if HR isn't divisible by the rebin
    # factor — match the LR shape by cropping to the smaller.
    h = min(predicted.shape[0], lr_vis.shape[0])
    w = min(predicted.shape[1], lr_vis.shape[1])
    predicted = predicted[:h, :w].astype(np.float32)
    residual = (lr_vis[:h, :w].astype(np.float32) - predicted).astype(np.float32)
    return predicted, residual


def _job_reconstruct_euclid_cutout(
    cap,
    ra: float,
    dec: float,
    checkpoint_dir: str,
    num_res_blocks: int,
    cutout_size_vis_pixels: int,
    asinh_scale: Optional[float] = None,
    show_all_bands: bool = False,
) -> Dict[str, Any]:
    """Download a 4-band Euclid cutout at one sky position, run SR, save PNG.

    Unlike ``_job_generate_reconstruct`` (which renders a synthetic scene),
    this fetches a real cutout at ``(ra, dec)`` for every band, converts
    each from the archive's ADU s⁻¹ units to electrons-over-the-stack (so
    the model sees the same scale it was trained on), stacks the four
    bands into ``(H, W, 4)``, and runs the model.

    The downloaded FITS files (one per band) plus the super-resolved
    VIS output are written to a single ``Config.EUCLID_INFERENCE_DIR/
    cutouts/latest/`` slot that is wiped at the start of every call, so
    each run *replaces* the previous records rather than accumulating one
    directory per position. The input RA/Dec/size are preserved in the
    SR FITS header for provenance.
    """

    if not tf.train.latest_checkpoint(checkpoint_dir):
        raise FileNotFoundError(f"no checkpoint in {checkpoint_dir}")
    scale = Config.DEFAULT_REBIN_FACTOR
    model = load_model_from_checkpoint(
        checkpoint_dir, scale, num_res_blocks,
        nchan_in=Config.NUM_LR_CHANNELS, nchan_out=Config.NUM_HR_CHANNELS,
    )

    # Single overwrite slot: wipe every previous cutout record (the old
    # per-position directories included) so each call replaces the prior
    # run rather than accumulating one directory per position. The input
    # RA/Dec/size live in the SR FITS header for provenance.
    cutouts_root = os.path.join(Config.EUCLID_INFERENCE_DIR, "cutouts")
    if os.path.isdir(cutouts_root):
        shutil.rmtree(cutouts_root)
    cache_dir = os.path.join(cutouts_root, "latest")
    os.makedirs(cache_dir, exist_ok=True)

    # Fetch each band; per-band MAGZERO from each header drives the
    # per-band ADU/s → electrons conversion so the model sees the same
    # calibration scale the simulator uses.
    bands_data: Dict[str, np.ndarray] = {}
    bands_info: Dict[str, Dict[str, Any]] = {}
    vis_header = None
    for k, band_name in enumerate(Config.LR_INPUT_BAND_NAMES):
        cap.tick(k, len(Config.LR_INPUT_BAND_NAMES) + 2,
                 f"downloading {band_name} cutout")
        band = Config.get_band(band_name)
        outf = os.path.join(cache_dir, f"{band_name}.fits")
        ok, err = fetch_cutout_at(
            ra=ra, dec=dec, band_name=band_name, output_file=outf,
            cutout_size_vis_pixels=cutout_size_vis_pixels,
        )
        if not ok:
            raise RuntimeError(f"{band_name}: {err}")
        with fits.open(outf) as hdul:
            arr = hdul[0].data.astype(np.float32)
            header = hdul[0].header
        if band_name == "VIS":
            vis_header = header.copy()
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
            "fits_path":  outf,
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
             len(Config.LR_INPUT_BAND_NAMES) + 3, "running model")
    _, sr_data = reconstruct(model, lr_cube)
    lr_vis = lr_cube[..., 0]

    # ESA cutout headers carry an EXTNAME that's invalid on a PrimaryHDU;
    # strip it (and let silentfix handle the rest) so the writes don't
    # trip FITS verification.
    def _clean_hdr(hdr):
        if hdr is None:
            return fits.Header()
        h = hdr.copy()
        for kbad in ("EXTNAME", "XTENSION"):
            if kbad in h:
                del h[kbad]
        return h

    # Stacked 4-band original (electrons): one image plane per band in
    # LR_INPUT_BAND_NAMES order (band 0 = VIS), carrying the VIS WCS so it
    # overlays the SR on-sky — the same downloadable output the standalone
    # infer_euclid_cutout.py script produces.
    stack = np.moveaxis(lr_cube, -1, 0).astype(np.float32)   # (4, H, W)
    stack_hdr = _clean_hdr(vis_header)
    stack_hdr["OBJECT"] = "Euclid LR stack (electrons)"
    stack_hdr["BUNIT"]  = "electron"
    stack_hdr["BANDS"]  = (",".join(Config.LR_INPUT_BAND_NAMES),
                           "NAXIS3 plane order")
    stack_path = os.path.join(cache_dir, "original_stack.fits")
    fits.PrimaryHDU(stack, header=stack_hdr).writeto(
        stack_path, overwrite=True, output_verify="silentfix")
    print(f"  ✓ saved stacked original → {stack_path}")

    # ────────────────────────────────────────────────────────────────
    # Self-consistency check: forward-model the SR and compare to the
    # observed dirty LR. If the network is doing its job, applying the
    # VIS forward chain (fftconvolve with empirical ePSF + 2× sum-rebin)
    # to SR should produce something close to the actual dirty LR; the
    # residual flags pixel-level over- or under-reconstruction.
    # ────────────────────────────────────────────────────────────────

    cap.tick(len(Config.LR_INPUT_BAND_NAMES) + 1,
             len(Config.LR_INPUT_BAND_NAMES) + 3,
             "forward-modelling SR for residual")
    try:
        predicted_dirty, residual = _forward_model_sr_residual(sr_data, lr_vis)
        print(f"  ✓ residual stats: mean={residual.mean():.3g}, "
              f"std={residual.std():.3g}, "
              f"|max|={np.abs(residual).max():.3g}")
    except Exception as e:
        print(f"  residual computation skipped: {type(e).__name__}: {e}")
        predicted_dirty = None
        residual = None

    # Save SR with the 2× magnified VIS WCS so it overlays the stacked
    # original on-sky (0.05″/pix vs 0.10″/pix).
    sr_fits_path = os.path.join(cache_dir, "SR.fits")
    sr_hdr = (_clean_hdr(scaled_wcs_header(vis_header, scale))
              if vis_header is not None else fits.Header())
    sr_hdu = fits.PrimaryHDU(sr_data.astype(np.float32), header=sr_hdr)
    sr_hdu.header["OBJECT"]   = "Euclid SR (WDSR VIS)"
    sr_hdu.header["BUNIT"]    = "electron"
    sr_hdu.header["RA"]       = (float(ra),  "Input RA (deg)")
    sr_hdu.header["DEC"]      = (float(dec), "Input Dec (deg)")
    sr_hdu.header["CSIZE"]    = (int(cutout_size_vis_pixels),
                                 "Input VIS cutout size (px)")
    sr_hdu.header["CKPT"]     = (str(checkpoint_dir)[:60], "Checkpoint dir")
    sr_hdu.header["ASINH"]    = (float(asinh_scale or Config.STRETCH_SCALE_E),
                                 "asinh stretch knee used for plot")
    sr_hdu.writeto(sr_fits_path, overwrite=True, output_verify="silentfix")
    print(f"  ✓ saved SR  → {sr_fits_path}")

    if predicted_dirty is not None and residual is not None:
        pd_path = os.path.join(cache_dir, "SR_forward.fits")
        ph = fits.PrimaryHDU(predicted_dirty)
        # FITS header values must be plain ASCII — no unicode math symbols.
        ph.header["OBJECT"] = "VIS-PSF conv SR + 2x rebin (predicted LR)"
        ph.writeto(pd_path, overwrite=True)
        res_path = os.path.join(cache_dir, "residual.fits")
        rh = fits.PrimaryHDU(residual)
        rh.header["OBJECT"] = "Dirty(VIS) - VIS-PSF conv SR + 2x rebin"
        rh.writeto(res_path, overwrite=True)
        print(f"  ✓ saved SR_forward + residual → {cache_dir}")

    out_dir = Config.VIS_RECONSTRUCTION_DIR
    os.makedirs(out_dir, exist_ok=True)
    # Drop stale Euclid-cutout renders (one used to be written per
    # position); leave the synthetic reconstruction PNGs alone.
    for stale in glob.glob(os.path.join(out_dir, "euclid_*.png")):
        try:
            os.remove(stale)
        except OSError:
            pass
    out_path = os.path.join(out_dir, "euclid_latest.png")
    plot_reconstruction(lr_vis, sr_data, hr_data=None,
                        output_path=out_path, lr_cube=lr_cube,
                        asinh_scale=asinh_scale,
                        show_all_bands=show_all_bands,
                        predicted_dirty=predicted_dirty,
                        residual=residual)
    cap.tick(len(Config.LR_INPUT_BAND_NAMES) + 3,
             len(Config.LR_INPUT_BAND_NAMES) + 3, "saved PNG")
    print(f"  ✓ {out_path}")
    return {
        "output_path":  out_path,
        "cache_dir":    cache_dir,
        "sr_fits_path": sr_fits_path,
        "ra":           ra,
        "dec":          dec,
        "cutout_size":  cutout_size_vis_pixels,
        "bands":        bands_info,
        "residual_std": float(residual.std()) if residual is not None else None,
    }


def _plot_lr_input(lr_cube, output_path, asinh_scale, *, label=""):
    """Render the 4-band LR input as an asinh montage (no model needed).

    Used by the round-trip inspector before any checkpoint exists, so the
    user can still eyeball the real-Euclid stamps the network will be fed.
    """
    scale = float(asinh_scale) if asinh_scale and asinh_scale > 0 \
            else float(Config.STRETCH_SCALE_E)
    names = Config.LR_INPUT_BAND_NAMES
    fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 4.2))
    if len(names) == 1:
        axes = [axes]
    for ax, name in zip(axes, names):
        plane = np.arcsinh(lr_cube[..., names.index(name)] / scale)
        ax.imshow(plane, origin="lower", cmap="gray", interpolation="nearest")
        ax.set_title(f"{name} (asinh)", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"Round-trip LR input · {label}  (0.10\"/pix, electrons)",
                 fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _job_roundtrip_inspect(
    cap,
    pos_id: int,
    checkpoint_dir: str,
    num_res_blocks: int,
    asinh_scale: Optional[float] = None,
    show_all_bands: bool = False,
) -> Dict[str, Any]:
    """Inspect one real-Euclid round-trip cutout — and its round-trip
    reconstruction once a checkpoint exists.

    Pulls the bundled ``sky_<pos_id>.fits`` (the 4-band cutouts step 1 of
    the round-trip pipeline writes on FASRC) into the local cache, scales
    each band by ``t_total_s`` to match the round-trip TFRecords the
    trainer saw, and renders the LR input. When the local checkpoint
    mirror has a checkpoint, it also runs ``M → SR``, forward-models SR
    back to the Euclid LR grid, and shows the round-trip residual — the
    exact self-consistency the round-trip loss optimises.
    """
    cfg = fasrc_config.load()
    remote = f"{cfg.data_dir}/euclid_sky/cutouts/sky_{int(pos_id):04d}.fits"
    cap.tick(0, 4, f"fetching sky_{int(pos_id):04d}.fits from FASRC")
    res = _fasrc_fetcher.fetch_one_file(remote)
    if not res.ok:
        raise RuntimeError(f"could not fetch {remote}: {res.error}")
    local = res.local_path

    cap.tick(1, 4, "reading 4-band bundle")
    bands_data: Dict[str, np.ndarray] = {}
    bands_info: Dict[str, Dict[str, Any]] = {}
    with fits.open(local) as hdul:
        names_present = {h.name for h in hdul if getattr(h, "name", "")}
        primary_hdr = hdul[0].header
        for band_name in Config.LR_INPUT_BAND_NAMES:
            if band_name not in names_present:
                raise RuntimeError(
                    f"bundle {os.path.basename(local)} is missing band "
                    f"{band_name} (HDUs: {sorted(names_present)})"
                )
            band = Config.get_band(band_name)
            # Archive e⁻/s → total electrons over the stack (× t_total_s),
            # matching fasrc_generate_euclid_roundtrip_tfrecords.py.
            data_e = (np.asarray(hdul[band_name].data, dtype=np.float32)
                      * float(band.t_total_s))
            bands_data[band_name] = data_e
            bands_info[band_name] = {
                "shape":     list(data_e.shape),
                "t_total_s": float(band.t_total_s),
                "pix_mean":  float(np.mean(data_e)),
                "pix_std":   float(np.std(data_e)),
            }
    ra  = float(primary_hdr.get("RA",  float("nan")))
    dec = float(primary_hdr.get("DEC", float("nan")))

    shapes = {n: bands_data[n].shape for n in Config.LR_INPUT_BAND_NAMES}
    if len(set(shapes.values())) != 1:
        raise RuntimeError(f"per-band shapes disagree: {shapes}")
    lr_cube = np.stack(
        [bands_data[n] for n in Config.LR_INPUT_BAND_NAMES], axis=-1,
    )
    lr_vis = lr_cube[..., 0]

    out_dir = Config.VIS_RECONSTRUCTION_DIR
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "roundtrip_latest.png")

    has_ckpt = bool(tf.train.latest_checkpoint(checkpoint_dir))
    residual_std = None
    if has_ckpt:
        cap.tick(2, 4, "running model + round-trip forward")
        model = load_model_from_checkpoint(
            checkpoint_dir, Config.DEFAULT_REBIN_FACTOR, num_res_blocks,
            nchan_in=Config.NUM_LR_CHANNELS, nchan_out=Config.NUM_HR_CHANNELS,
        )
        _, sr_data = reconstruct(model, lr_cube)
        try:
            predicted_dirty, residual = _forward_model_sr_residual(sr_data, lr_vis)
            residual_std = float(residual.std())
        except Exception as e:  # noqa: BLE001 — residual is a bonus panel
            print(f"  residual skipped: {type(e).__name__}: {e}")
            predicted_dirty, residual = None, None
        plot_reconstruction(
            lr_vis, sr_data, hr_data=None, output_path=out_path,
            lr_cube=lr_cube, asinh_scale=asinh_scale,
            show_all_bands=show_all_bands,
            predicted_dirty=predicted_dirty, residual=residual,
        )
    else:
        cap.tick(2, 4, "no checkpoint yet — rendering LR input only")
        _plot_lr_input(lr_cube, out_path, asinh_scale,
                       label=f"sky_{int(pos_id):04d}")
    cap.tick(4, 4, "done")
    print(f"  ✓ {out_path}")
    return {
        "output_path":    out_path,
        "pos_id":         int(pos_id),
        "ra":             ra,
        "dec":            dec,
        "has_checkpoint": has_ckpt,
        "shape":          list(lr_vis.shape),
        "bands":          bands_info,
        "residual_std":   residual_std,
        "local_bundle":   local,
    }


def _job_viz_star_positions(cap, output_dir: str) -> Dict[str, Any]:
    cat = StarCatalog(output_dir)
    data = cat.load()
    out = Config.VIS_STAR_POSITIONS
    os.makedirs(os.path.dirname(out), exist_ok=True)
    draw_star_positions(data["stars"], out)
    print(f"wrote {out}")
    return {"path": out}


def _job_viz_psf(cap, band_name: str | None, psf_dir: str) -> Dict[str, Any]:
    """Render the four-band PSF inspection panel (or one band)."""
    matplotlib.use("Agg")

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
        ax.imshow(np.log10(d), cmap="viridis", origin="lower",
                  interpolation="nearest")
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
    matplotlib.use("Agg")

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
    psf_sets = load_all_band_psf_sets()
    fwd = MultiBandForward(psf_sets_by_band=psf_sets,
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
                      vmin=lo, vmax=hi, interpolation="nearest")
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

    # Also persist the underlying 4-band cubes as FITS so each panel's
    # raw pixels are inspectable via the universal /inspect view. One
    # FITS per (kind, band) — matches the per-band slicing the rest of
    # the UI uses for sky records.
    def _save_cube_per_band(data_4ch, kind: str, px_scale_arcsec: float) -> None:
        for k, band_name in enumerate(bands):
            plane = np.ascontiguousarray(
                np.asarray(data_4ch[..., k], dtype=np.float32),
            )
            hdu = _fits.PrimaryHDU(plane)
            hdu.header["OBJECT"] = (f"EuclidPolish demo {kind} {band_name}",
                                    "panel label")
            hdu.header["KIND"]   = (kind, "hr | lr")
            hdu.header["BAND"]   = (band_name, "band name")
            hdu.header["NLENS"]  = (int(meta["n_lenses"]),
                                     "number of injected lenses")
            hdu.header["NGAL"]   = (int(meta["n_galaxies"]),
                                     "number of galaxies in the field")
            hdu.header["BUNIT"]  = ("e-", "electrons (raw)")
            hdu.header["CDELT1"] = (-px_scale_arcsec / 3600.0,
                                     "pixel scale (degrees)")
            hdu.header["CDELT2"] = ( px_scale_arcsec / 3600.0,
                                     "pixel scale (degrees)")
            out = os.path.join(out_dir, f"demo_{kind}_{band_name}.fits")
            hdu.writeto(out, overwrite=True)
            print(f"  ✓ wrote {out}")
    _save_cube_per_band(sky.data, "hr", 0.05)
    _save_cube_per_band(lr.data,  "lr", 0.10)

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


def _render_fits_to_png_adaptive(fits_path: str, size: int) -> bytes:
    """Render any 2-D FITS image to PNG with a data-adaptive stretch.

    The band-aware :func:`_render_fits_to_png` hardcodes an asinh knee
    (``band.asinh_stretch_scale_e``, ~1000 e⁻ by default) tuned for
    Euclid sky cutouts. That stretch is meaningless for files outside
    that domain — a unit-flux PSF sums to 1 over 511² pixels (values
    ~10⁻⁶–10⁻²), a differential kernel can have signed wings, dark
    frames hover near zero. Applying the cutout knee to those leaves
    `arcsinh(x / 1000) ≈ x / 1000` with a near-empty histogram, then
    the 1.0/99.7-percentile clip stretches numerical noise instead of
    real structure — what looked "weird" on /inspect for the diff
    kernel was exactly this mismatch.

    The /inspect route can't know what kind of file it's showing, so
    use ``MinMaxInterval + AsinhStretch(a=0.01)`` from
    astropy.visualization. MinMax keeps the full data range in view
    (a kernel's central peak sits 5 decades above the typical pixel —
    anything that trims outliers, like ZScale or a percentile clip,
    pushes the peak into saturation and the centre renders as a
    checkerboard instead of a smooth blob). The aggressive asinh knee
    (``a=0.01`` = transition at 1 % of the normalised range) compresses
    that dynamic range so the faint wings and the bright core are
    visible in the same frame.
    """

    with fits.open(fits_path, memmap=False) as hdul:
        data = None
        for hdu in hdul:
            if hdu.data is not None and getattr(hdu.data, "ndim", 0) == 2:
                data = np.asarray(hdu.data, dtype=np.float64)
                break
    if data is None:
        abort(415)

    finite = np.isfinite(data)
    if not finite.any():
        data = np.zeros_like(data)
    else:
        data = np.where(finite, data, np.nanmedian(data[finite]))

    # ``clip=True`` guarantees the output stays in [0, 1] even when the
    # underlying stretch would push values outside (signed residuals
    # near the edges of the data range).
    norm = ImageNormalize(
        data, interval=MinMaxInterval(), stretch=AsinhStretch(a=0.01),
        clip=True,
    )
    normed = np.asarray(norm(data), dtype=np.float64)
    img8 = (255 * (1.0 - normed)).astype(np.uint8)   # gray_r: bright → dark
    img8 = np.flipud(img8)                            # FITS lower-origin → PIL
    pil = Image.fromarray(img8, mode="L")
    if size and size != pil.size[0]:
        pil = pil.resize((int(size), int(size)), Image.NEAREST)
    buf = io.BytesIO()
    pil.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _render_fits_to_png(fits_path: str, band: BandConfig,
                        size: Optional[int] = None) -> bytes:
    """Load a cutout FITS, apply per-band asinh stretch, return PNG bytes.

    - Asinh stretch knee comes from ``band.asinh_stretch_scale_e``.
    - vmin/vmax clip at the 1.0 / 99.7 percentiles of the stretched image.
    - Optional ``size`` resamples (nearest-neighbour) to a square thumbnail.

    Use this only for files whose pixel scale you know (Euclid sky
    cutouts, per-band thumbnails). For the universal /inspect view —
    where the file could be anything — use
    :func:`_render_fits_to_png_adaptive` instead, which derives the
    contrast window from the data and doesn't assume cutout units.
    """

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
# Sky-record FITS export. The sky generator writes TFRecords (not FITS),
# so the inspector route needs an on-demand exporter that slices out one
# (subset, kind, band, index) and writes a real FITS file.
# ---------------------------------------------------------------------------

def _export_sky_record_fits(
    subset: str, kind: str, band: str, index: int,
    records_dir: Optional[str] = None,
) -> str:
    """Materialise one sky record as a single-band FITS file.

    Caches the result under ``data/vis/sky_fits/`` so repeat clicks on
    the same record don't re-read the TFRecord. Returns the absolute
    path to the saved file. ``records_dir`` defaults to the local
    ``RECORDS_DIR_V2``; the /sky viewer passes the FASRC cache dir.
    """

    if subset not in ("train", "validate"):
        abort(400)
    if kind not in ("clean", "dirty", "hr"):
        abort(400)
    band_names = list(Config.LR_INPUT_BAND_NAMES)
    if kind == "hr":
        if band != "VIS":
            abort(400)
        band_idx = 0
    else:
        if band not in band_names:
            abort(400)
        band_idx = band_names.index(band)
    try:
        idx = int(index)
    except (TypeError, ValueError):
        abort(400)
    if idx < 0:
        abort(400)

    name = f"{kind}_{subset}"
    src_path = tfrecord_path(records_dir or Config.RECORDS_DIR_V2, name)
    if not os.path.exists(src_path):
        abort(404)

    out_dir = os.path.realpath(os.path.join(Config.VIS_DIR, "sky_fits"))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{kind}_{subset}_{band}_{idx:04d}.fits")
    if os.path.isfile(out_path) and os.path.getsize(out_path) > 0:
        return out_path

    # Stream just enough records to reach ``idx`` — TFRecords don't have
    # random access so we read sequentially. Cheap for typical idx ≤ ~50.
    records = read_multiband_skyimages(src_path, num_images=idx + 1)
    if not records or idx >= len(records):
        abort(404)
    record = records[idx]
    data = record.data
    if data.ndim == 2:
        plane = data
    elif data.ndim == 3:
        if band_idx >= data.shape[-1]:
            abort(404)
        plane = data[..., band_idx]
    else:
        abort(415)

    hdu = fits.PrimaryHDU(np.ascontiguousarray(plane, dtype=np.float32))
    hdu.header["OBJECT"] = (f"EuclidPolish {kind} {band}", "kind + band")
    hdu.header["SUBSET"] = (subset, "TFRecord subset")
    hdu.header["IDX"]    = (idx, "record index within subset")
    hdu.header["BAND"]   = (band, "band name (VIS, Y_E, J_E, H_E)")
    hdu.header["KIND"]   = (kind, "clean | dirty | hr")
    hdu.header["BUNIT"]  = ("e-", "electrons (raw, sign preserved)")
    if kind == "clean":
        hdu.header["CDELT1"] = (-Config.DEFAULT_PIXEL_SCALE / 3600.0,
                                 "HR pixel scale (degrees)")
        hdu.header["CDELT2"] = ( Config.DEFAULT_PIXEL_SCALE / 3600.0,
                                 "HR pixel scale (degrees)")
    elif kind == "dirty":
        hdu.header["CDELT1"] = (-Config.VIS_PIXEL_SCALE_ARCSEC / 3600.0,
                                 "LR pixel scale (degrees)")
        hdu.header["CDELT2"] = ( Config.VIS_PIXEL_SCALE_ARCSEC / 3600.0,
                                 "LR pixel scale (degrees)")
    hdu.writeto(out_path, overwrite=True)
    return out_path


# ---------------------------------------------------------------------------
# Universal FITS inspector — header view + download for any FITS in our
# data tree. Every page that shows an image card links its thumbnails
# through ``/inspect?fits=<path>``; this is the only place that exposes
# raw FITS files to the browser, so the safety check lives here.
# ---------------------------------------------------------------------------

def _inspectable_roots() -> List[str]:
    """Real paths under which a user may request any FITS file via /inspect.

    Anything outside this set is rejected with HTTP 403 — this prevents a
    crafted ``?fits=../../../etc/passwd`` from escaping the data tree.
    All roots are normalised via :func:`os.path.realpath` so symlinks
    can't bypass the check either.
    """
    roots = [
        Config.DEFAULT_OUTPUT_DIR,        # Euclid star cutouts
        Config.EUCLID_PSF_DIR,             # band PSFs
        Config.EUCLID_INFERENCE_DIR,       # real-Euclid inference outputs
        Config.VIS_DIR,                     # reconstruction PNG/FITS, demos, sky_fits
        Config.RECORDS_DIR_V2,              # TFRecords (not FITS) — kept for completeness
        Config.FASRC_CACHE_DIR,            # rsync'd-from-FASRC FITS files
    ]
    out = []
    for p in roots:
        if not p:
            continue
        try:
            out.append(os.path.realpath(p))
        except OSError:
            pass
    return out


def _resolve_inspectable_fits(raw_path: str) -> str:
    """Validate ``raw_path`` and return its real absolute path.

    Aborts the request with the appropriate HTTP error:

    * 400 — empty / non-FITS extension
    * 403 — resolves outside the allowed roots
    * 404 — does not exist on disk
    """
    if not raw_path:
        abort(400)
    p = raw_path if os.path.isabs(raw_path) else os.path.normpath(
        os.path.join(os.getcwd(), raw_path)
    )
    real = os.path.realpath(p)
    if not real.lower().endswith((".fits", ".fit", ".fits.gz")):
        abort(400)
    if not os.path.isfile(real):
        abort(404)
    for root in _inspectable_roots():
        if real == root or real.startswith(root + os.sep):
            return real
    abort(403)


def _read_fits_header_rows(path: str) -> List[Dict[str, Any]]:
    """Return one row per HDU with its header laid out for the table view.

    Each row: ``{hdu_index, name, kind, shape, dtype, cards: [(key, value, comment), ...]}``.
    """
    rows: List[Dict[str, Any]] = []
    with fits.open(path, memmap=False) as hdul:
        for i, hdu in enumerate(hdul):
            cards: List[Tuple[str, str, str]] = []
            for card in hdu.header.cards:
                key  = str(card.keyword)
                val  = str(card.value)
                # Trim long string values — full text is visible in the
                # downloaded FITS, the UI just needs an at-a-glance view.
                if len(val) > 80:
                    val = val[:77] + "…"
                cmt  = str(card.comment)
                cards.append((key, val, cmt))
            shape = getattr(hdu.data, "shape", None) if hdu.is_image else None
            dtype = getattr(hdu.data, "dtype", None) if hdu.is_image else None
            rows.append({
                "hdu_index": i,
                "name":      hdu.name or f"HDU{i}",
                "kind":      type(hdu).__name__,
                "shape":     list(shape) if shape is not None else None,
                "dtype":     str(dtype) if dtype is not None else None,
                "cards":     cards,
            })
    return rows


def _fits_file_info(path: str) -> Dict[str, Any]:
    """Lightweight ``ls``-style metadata for the inspector header card."""
    st = os.stat(path)
    return {
        "abspath":  path,
        "basename": os.path.basename(path),
        "size_kb":  round(st.st_size / 1024, 1),
        "mtime":    st.st_mtime,
    }


def _safe_relpath(real_abs: str) -> str:
    """Return the project-rooted relative form of an absolute path.

    Used to build ``?fits=`` query strings that work regardless of the
    server's CWD. Falls back to the absolute path on failure.
    """
    try:
        rel = os.path.relpath(real_abs, os.getcwd())
        return rel if not rel.startswith("..") else real_abs
    except ValueError:
        return real_abs


# ---------------------------------------------------------------------------
# View renderers: serve PNGs of pipeline artefacts for the center pane.
# These all return bytes; the route layer wraps in send_file.
# ---------------------------------------------------------------------------

def _render_psf_panel_png(band: Optional[str]) -> bytes:
    """Render one band (or all four) on a log-stretch panel as PNG bytes."""
    matplotlib.use("Agg")

    # Render the FASRC-extracted PSFs (pulled to the local cache), not a
    # stale local copy. None → nothing on FASRC yet.
    psf_dir = _fasrc_psf_dir()
    if not psf_dir:
        abort(404)
    psfs = load_all_band_psfs(psf_dir=psf_dir)
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
        ax.imshow(np.log10(d), cmap="viridis", origin="lower",
                  interpolation="nearest")
        ax.set_title(f"{name}  {p.data.shape[0]}×{p.data.shape[1]}  "
                     f"@ {p.pixel_scale:.3f}\"/pix", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, dpi=110, bbox_inches="tight", format="png")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def _arrays_to_fits_bytes(
    arrays,
    header_meta=None,
    primary_name: Optional[str] = None,
) -> bytes:
    """Pack a dict of ``{name: array}`` into a multi-HDU FITS file.

    The first array goes into the PrimaryHDU; the rest become
    ``ImageHDU`` extensions with their dict key as ``EXTNAME``.
    ``header_meta`` keys are copied into every HDU's header
    (FITS keywords are uppercased and truncated to 8 chars).

    Used by the "Download FITS" sister endpoints of every renderer
    so the user can pull the raw linear array even when the view
    they're looking at is asinh-stretched, percentile-clipped, and
    colormapped for display.
    """
    header_meta = dict(header_meta or {})
    hdus = []
    items = list(arrays.items())
    if not items:
        raise ValueError("arrays dict is empty")
    for i, (name, arr) in enumerate(items):
        arr = np.asarray(arr, dtype=np.float32)
        if i == 0:
            hdu = _fits.PrimaryHDU(data=arr)
        else:
            hdu = _fits.ImageHDU(data=arr)
        # EXTNAME on the primary HDU isn't strictly required but lots
        # of viewers (DS9, ginga) treat it as the human-readable label
        # so set it consistently.
        hdu.header["EXTNAME"] = name[:60]
        for k, v in header_meta.items():
            key = str(k).upper()[:8]
            if isinstance(v, (int, float, bool, str)):
                hdu.header[key] = v
        if primary_name and i == 0:
            hdu.header["EXTNAME"] = primary_name[:60]
        hdus.append(hdu)
    buf = io.BytesIO()
    _fits.HDUList(hdus).writeto(buf, overwrite=True)
    buf.seek(0)
    return buf.getvalue()


def _render_sky_record_png(subset: str, kind: str, band: str,
                           index: int,
                           records_dir: Optional[str] = None) -> bytes:
    """Render one image from the multi-band TFRecords.

    ``records_dir`` defaults to ``Config.RECORDS_DIR_V2`` (the locally-
    generated sky records). The HST Catalog passes its own dir so it
    can reuse the exact same renderer against the FASRC-cached HST
    TFRecords — they use the same MultiBandSkyImage schema, so the
    only thing that changes is where the file lives.

    ``subset`` ∈ {"train", "validate"},
    ``kind`` ∈ {"clean", "dirty", "hr"}
      • ``clean`` → 4-band HR clean record
      • ``dirty`` → 4-band LR dirty record (PSF + noise + artifacts)
      • ``hr``    → 1-band VIS HR target (the network's training output)
    ``band``:
      * one of ``Config.LR_INPUT_BAND_NAMES`` → grayscale asinh of that band
      * ``"color"`` → 4-band Lupton RGB (solar-balanced). Requires the
        record to carry all four bands — clean/dirty only. For the
        VIS-only ``hr`` record, ``color`` falls back to VIS grayscale.
    """
    matplotlib.use("Agg")

    if subset not in ("train", "validate"):
        abort(400)
    if kind not in ("clean", "dirty", "hr"):
        abort(400)
    if band not in Config.LR_INPUT_BAND_NAMES and band != "color":
        abort(400)
    name = f"{kind}_{subset}"
    path = tfrecord_path(records_dir or Config.RECORDS_DIR_V2, name)
    if not os.path.exists(path):
        abort(404)
    # Stream just enough records to reach ``index``.
    max_to_read = max(index + 1, 1)
    records = read_multiband_skyimages(path, num_images=max_to_read)
    if not records or index >= len(records):
        abort(404)
    img = records[min(index, len(records) - 1)]
    data = img.data

    if band == "color" and data.shape[-1] >= len(Config.LR_INPUT_BAND_NAMES):
        # 4-band solar-balanced RGB with per-channel [p1, p99.5]
        # normalisation — same dynamic-range convention as the
        # grayscale single-band panels in the rest of the sky tab.
        # ``clean`` records live on the HR grid (band-independent
        # asinh-stretch knee unspecified) — use the VIS knee as the
        # shared reference, matching how the rest of the UI treats HR.
        rgb = calibrated_rgb_panel(
            data, band_names=Config.LR_INPUT_BAND_NAMES,
            scheme="vis_nisp", reference="solar",
            stretch="asinh",
            asinh_scale_e=float(Config.BAND_VIS.asinh_stretch_scale_e),
        )
        fig, ax = plt.subplots(figsize=(6.5, 6.5))
        ax.imshow(np.clip(rgb, 0.0, 1.0), origin="lower",
                  interpolation="nearest")
        ax.set_title(
            f"{kind} {subset} · color (4-band, solar) · idx {img.index}  "
            f"({data.shape[0]}×{data.shape[1]} @ "
            f"{img.pixel_scale_arcsec:.3f}\"/pix)", fontsize=10,
        )
        ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, dpi=110, bbox_inches="tight", format="png")
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()

    # Grayscale single-band rendering. HR records carry shape (H, W, 1)
    # (VIS only), so any band selection falls back to channel 0.
    if data.shape[-1] == 1 or band == "color":
        plane = data[..., 0]
        band_name = "VIS"
    else:
        k = list(Config.LR_INPUT_BAND_NAMES).index(band)
        plane = data[..., k]
        band_name = band
    bcfg = Config.get_band(band_name)
    stretched = np.arcsinh(plane / float(bcfg.asinh_stretch_scale_e))
    lo, hi = np.percentile(stretched, [1.0, 99.7])
    if hi <= lo: hi = lo + 1.0
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.imshow(stretched, cmap="gray_r", origin="lower", vmin=lo, vmax=hi,
              interpolation="nearest")
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


def _render_sky_record_pair_png(subset: str, band: str, index: int,
                                records_dir: Optional[str] = None) -> bytes:
    """Side-by-side triptych for one (clean, dirty, HR) pair.

    Pulls the three TFRecord files at the same ``index`` and renders
    them in a single figure: clean HR (4-band) ┃ dirty LR (4-band) ┃
    HR-VIS target (1-band).

    Display range = SHARED 99.7-percentile clip across **all three
    panels combined**. Sharing the clip across panels means "mid-grey
    here" corresponds to the same electron count as "mid-grey there",
    which is the invariant a side-by-side comparison needs. The
    single-panel kinds (clean / dirty / hr alone) keep the standard
    per-image clip from :func:`_render_sky_record_png`.

    ``band`` ∈ ``Config.LR_INPUT_BAND_NAMES`` ∪ {"color"}. With
    ``band="color"`` the clean + dirty panels use the 4-band Lupton
    RGB; the HR panel always uses VIS grayscale (it's a single-band
    record).
    """
    matplotlib.use("Agg")

    if subset not in ("train", "validate"):
        abort(400)
    if band not in Config.LR_INPUT_BAND_NAMES and band != "color":
        abort(400)

    base_dir = records_dir or Config.RECORDS_DIR_V2
    panels: Dict[str, Any] = {}
    for kind in ("clean", "dirty", "hr"):
        path = tfrecord_path(base_dir, f"{kind}_{subset}")
        if not os.path.exists(path):
            abort(404)
        records = read_multiband_skyimages(path, num_images=max(index + 1, 1))
        if not records or index >= len(records):
            abort(404)
        panels[kind] = records[min(index, len(records) - 1)]

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.4))

    # Pre-compute the SHARED 99.7-percentile clip across all three
    # grayscale panels. Skipped when band="color" because color panels
    # use the Lupton RGB renderer instead of asinh.
    def _grayscale_plane(img, band_name: str):
        data = img.data
        if data.shape[-1] == 1:
            plane = data[..., 0]
            bn = "VIS"
        else:
            k = list(Config.LR_INPUT_BAND_NAMES).index(band_name)
            plane = data[..., k]
            bn = band_name
        knee = float(Config.get_band(bn).asinh_stretch_scale_e)
        return np.arcsinh(plane / knee)

    shared_lo, shared_hi = 0.0, 1.0
    if band != "color":
        all_stretched = [_grayscale_plane(panels[k], band)
                         for k in ("clean", "dirty", "hr")]
        union = np.concatenate([s.ravel() for s in all_stretched])
        shared_lo, shared_hi = np.percentile(union, [1.0, 99.7])
        if shared_hi <= shared_lo:
            shared_hi = shared_lo + 1.0

    def _show_grayscale(ax, img, band_name: str, title: str) -> None:
        stretched = _grayscale_plane(img, band_name)
        ax.imshow(stretched, cmap="gray_r", origin="lower",
                  vmin=shared_lo, vmax=shared_hi,
                  interpolation="nearest")
        ax.set_title(
            f"{title}  ({img.data.shape[0]}×{img.data.shape[1]} @ "
            f"{img.pixel_scale_arcsec:.3f}\"/pix)", fontsize=10,
        )
        ax.set_xticks([]); ax.set_yticks([])

    def _show_color(ax, img, title: str) -> None:
        rgb = calibrated_rgb_panel(
            img.data, band_names=Config.LR_INPUT_BAND_NAMES,
            scheme="vis_nisp", reference="solar", stretch="asinh",
            asinh_scale_e=float(Config.BAND_VIS.asinh_stretch_scale_e),
        )
        ax.imshow(np.clip(rgb, 0.0, 1.0), origin="lower",
                  interpolation="nearest")
        ax.set_title(
            f"{title}  ({img.data.shape[0]}×{img.data.shape[1]} @ "
            f"{img.pixel_scale_arcsec:.3f}\"/pix)", fontsize=10,
        )
        ax.set_xticks([]); ax.set_yticks([])

    for ax, kind, title in [
        (axes[0], "clean", f"clean HR · idx {panels['clean'].index}"),
        (axes[1], "dirty", f"dirty LR · idx {panels['dirty'].index}"),
        (axes[2], "hr",    f"HR target · idx {panels['hr'].index}"),
    ]:
        img = panels[kind]
        # HR target is VIS-only — never has 4 bands, so always grayscale.
        if band == "color" and kind != "hr":
            _show_color(ax, img, title)
        else:
            _show_grayscale(ax, img, band, title)

    fig.suptitle(
        f"HST → Euclid pair · {subset} · idx {index} · band={band}",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, dpi=100, bbox_inches="tight", format="png")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def _render_catalog_view_png(view: str, output_dir: str) -> bytes:
    """Render a catalog visualization: positions or magnitude histogram."""
    matplotlib.use("Agg")

    cat = StarCatalog(output_dir)
    if not cat.exists():
        abort(404)
    data = cat.load()
    stars = data.get("stars", [])
    if not stars:
        abort(404)
    if view == "positions":
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


def _record_count(name: str, records_dir: Optional[str] = None) -> Optional[int]:
    """Records in a multi-band tfrecord.

    Returns:
      * ``0``    — file does not exist
      * ``int``  — full record count
      * ``None`` — file present but partially corrupt (e.g. ``DataLossError``
        from a truncated rsync). Returning ``None`` instead of raising
        keeps callers like ``/api/hst-pairs/totals`` from 500-ing the
        whole response when one shard is bad.
    """
    p = tfrecord_path(records_dir or Config.RECORDS_DIR_V2, name)
    if not os.path.exists(p):
        return 0
    try:
        return sum(1 for _ in tf.data.TFRecordDataset(p))
    except tf.errors.DataLossError:
        # Truncated / partially-synced shard. Caller renders as "—".
        return None
    except Exception:
        # Any other read failure (permissions, corrupt header, etc.)
        # — treat the same way so the page stays usable.
        return None


# ---------------------------------------------------------------------------
# App factory + routes
# ---------------------------------------------------------------------------

def _try_startup_ssh_connect() -> Optional[str]:
    """Attempt one SSH connect during app startup. Return None on success.

    Stores the active session on :data:`STATE`. Failures are returned as
    a short error string for the connection-error page to display.

    Honours the ``EUCLID_POLISH_DISABLE_AUTO_SSH=1`` env var as a hard
    kill-switch — when set, this function is a no-op (returns a
    diagnostic string but never opens a socket or touches ``STATE.ssh``).
    This is **load-bearing for tests**: pytest imports ``create_app``
    which used to silently dial out to the user's real FASRC via their
    ControlMaster socket, blowing past any ``STATE.ssh = stub``
    monkeypatch the test had installed and submitting real SLURM jobs
    through every test that posted to a submit endpoint. The env var
    lets the test harness disable the auto-connect entirely so the
    stub stays in effect.
    """
    if _os.environ.get("EUCLID_POLISH_DISABLE_AUTO_SSH", "").strip() in (
        "1", "true", "yes", "on",
    ):
        return ("auto-connect disabled by "
                "EUCLID_POLISH_DISABLE_AUTO_SSH env var (test mode)")
    cfg = fasrc_config.load()
    if not cfg.ssh_user:
        return "ssh_user is unset — open Settings and configure it"
    try:
        STATE.ssh = SSHSession(SSHConfig(
            user=cfg.ssh_user, host=cfg.ssh_host,
            socket=cfg.control_socket,
            control_persist=cfg.control_persist,
        ))
        STATE.ssh.connect()
    except (SSHError, Exception) as e:
        STATE.ssh = None
        return f"{type(e).__name__}: {e}"
    STATE.connected_at = time.time()
    # Same catch-up as in /api/fasrc/connect — jobs that finished
    # while the server was offline get their state + sacct post-mortem
    # recorded now. Best-effort; failures here are swallowed so the
    # auto-connect message stays clean.
    try:
        fasrc_jobs.sync_pending_on_connect(STATE.ssh)
    except Exception:
        pass
    return None


def create_app() -> Flask:
    here = os.path.dirname(os.path.abspath(__file__))
    app = Flask(
        __name__,
        template_folder=os.path.join(here, "templates"),
        static_folder=os.path.join(here, "static"),
    )

    # ---------------------------------------------------------------- #
    # Auto-connect to FASRC on launch. If we can't connect, every
    # non-connection-related page redirects to /connection-error where
    # the user can edit ssh_user / ssh_host and retry. We deliberately
    # store the last error message so the error page can show *why*.
    # ---------------------------------------------------------------- #
    app.config["FASRC_STARTUP_ERROR"] = _try_startup_ssh_connect()

    # Paths that remain reachable even when SSH is down — the settings
    # form, the retry endpoint, static assets, and the connection-error
    # page itself. Everything else gets gated below.
    _ALWAYS_REACHABLE_PREFIXES = (
        "/connection-error",
        "/api/fasrc/",            # connect/settings/login/etc.
        "/static/",
        "/api/status",
    )

    @app.before_request
    def _enforce_ssh_gate():
        # Quickest possible check first.
        if STATE.ssh is not None and STATE.ssh.is_connected():
            return None
        if request.path == "/connection-error":
            return None
        if any(request.path.startswith(p) for p in _ALWAYS_REACHABLE_PREFIXES):
            return None
        # All other paths: redirect to the error page so the user can
        # fix settings + retry without poking through the UI for it.
        return redirect(url_for("connection_error_page"))

    @app.route("/connection-error", methods=["GET", "POST"])
    def connection_error_page():
        """Settings-edit + retry-connect page shown when SSH is down."""
        if request.method == "POST":
            # User edited settings; persist + retry.
            patch = {
                "ssh_user":     request.form.get("ssh_user", "").strip(),
                "ssh_host":     request.form.get("ssh_host", "").strip(),
                "control_socket": request.form.get("control_socket", "").strip(),
            }
            patch = {k: v for k, v in patch.items() if v}
            if patch:
                fasrc_config.update(patch)
            err = _try_startup_ssh_connect()
            app.config["FASRC_STARTUP_ERROR"] = err
            if err is None:
                # Connected — bounce to the dashboard.
                return redirect(url_for("index"))
            # Fall through to re-render the page with the new error.

        cfg_loaded = fasrc_config.load()
        return render_template(
            "connection_error.html",
            error=app.config.get("FASRC_STARTUP_ERROR") or "not connected",
            cfg=cfg_loaded,
        )

    @app.route("/api/connection/retry", methods=["POST"])
    def api_connection_retry():
        """POST-only retry hook so the existing /fasrc tab can also trigger reconnect."""
        err = _try_startup_ssh_connect()
        app.config["FASRC_STARTUP_ERROR"] = err
        if err is None:
            return jsonify({"ok": True})
        return jsonify({"ok": False, "error": err}), 502

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
        """Submit the brightest-N archive query as a short SLURM job.

        The catalog must land on the shared netscratch ``$DATA_DIR`` where
        the cutout-download / PSF jobs read it, and the query hits the ESA
        TAP service — the ``shared`` partition's compute nodes have outbound
        internet (the cutout-download job runs there too). Running it through
        SLURM rather than the login node means the standard Cancel-job button
        on ``/fasrc`` can stop it (login-node jobs can't be cancelled). The
        output lands in the sbatch log, tracked by the FASRC job panel."""
        if not STATE.ssh or not STATE.ssh.is_connected():
            return jsonify({
                "ok": False,
                "error": "Connect to FASRC first (the job lands on "
                         "netscratch where the download/PSF jobs read it).",
            }), 400
        n   = int(request.form.get("num_stars", 200))
        mag_lim_raw = request.form.get("magnitude_limit", "").strip()
        mag_min_raw = request.form.get("magnitude_min", "").strip()
        mag_lim = float(mag_lim_raw) if mag_lim_raw else None
        mag_min = float(mag_min_raw) if mag_min_raw else None
        snr_min_raw = request.form.get("snr_min", "").strip()
        snr_min = float(snr_min_raw) if snr_min_raw else None
        win = ""
        if mag_min is not None: win += f" mag>{mag_min}"
        if mag_lim is not None: win += f" mag<{mag_lim}"
        if snr_min is not None: win += f" snr≥{snr_min:g}"

        # No --output-dir: the remote script defaults to its own
        # $DATA_DIR/euclid_stars, which the sbatch template points at
        # netscratch via EUCLID_POLISH_DATA_DIR.
        argv = ["scripts/query_brightest_stars.py", "--num-stars", str(n)]
        if mag_min is not None: argv += ["--magnitude-min", f"{mag_min:g}"]
        if mag_lim is not None: argv += ["--magnitude-limit", f"{mag_lim:g}"]
        if snr_min is not None: argv += ["--snr-min", f"{snr_min:g}"]

        slurm_id, payload = _submit_quick_fasrc_job(
            STATE.ssh,
            argv=argv,
            label=f"query {n} brightest stars{win}",
            step_id="query_brightest_stars",
            resources=StepResources(partition="shared", n_cpus=1, n_gpus=0,
                                    memory="4G", time_limit="30:00"),
            params={"num_stars": n,
                    "magnitude_min": mag_min_raw, "magnitude_limit": mag_lim_raw,
                    "snr_min": snr_min_raw},
        )
        if slurm_id is None:
            return jsonify(payload), 500
        return jsonify(payload)

    @app.route("/cutouts/verify-photometry", methods=["POST"])
    def cutouts_verify_photometry():
        """Submit the read-only photometry-scale check as a short SLURM job.

        ``verify_star_photometry.py`` aperture-measures N downloaded VIS
        cutouts (electrons) and compares to each star's catalog PSF flux —
        a median ratio ≈ 1 confirms the absolute electron scale the
        star-anchor delta-targets are built on. It only reads the cutouts on
        netscratch + prints a report. Routed through SLURM (not the login
        node) so the standard Cancel-job button on ``/fasrc`` can stop it;
        the ratio table lands in the sbatch log."""
        if not STATE.ssh or not STATE.ssh.is_connected():
            return jsonify({
                "ok": False,
                "error": "Connect to FASRC first (the check reads the "
                         "cutouts on netscratch).",
            }), 400
        n    = int(request.form.get("n", 40))
        size = int(request.form.get("size", 256))
        argv = ["scripts/verify_star_photometry.py",
                "--n", str(n), "--size", str(size)]

        slurm_id, payload = _submit_quick_fasrc_job(
            STATE.ssh,
            argv=argv,
            label=f"verify photometry scale (n={n}, size={size}px)",
            step_id="verify_photometry",
            resources=StepResources(partition="shared", n_cpus=1, n_gpus=0,
                                    memory="8G", time_limit="30:00"),
            params={"n": n, "size": size},
        )
        if slurm_id is None:
            return jsonify(payload), 500
        return jsonify(payload)

    # ---------------- Authentication ----------------
    @app.route("/auth/login", methods=["POST"])
    def auth_login():
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
        try:
            auth.logout()
        except Exception:
            pass
        return jsonify({"ok": True})

    # ---------------- Euclid archive credentials (for FASRC download) -----
    # The cutout-download job runs on FASRC and logs into the Euclid
    # archive there. We write the credentials to the remote
    # ``~/.euclid_credentials`` (the file ``auth.login`` falls back to) via
    # the SSH channel — the password is sent as heredoc stdin (never in a
    # process argv), is stored mode-600 on the remote, and never touches
    # the laptop disk or the job DB.
    _EUCLID_CREDS_REMOTE = '"$HOME/.euclid_credentials"'

    @app.route("/euclid-auth/save", methods=["POST"])
    def euclid_auth_save():
        if not STATE.ssh or not STATE.ssh.is_connected():
            return jsonify({"ok": False, "error": "not connected to FASRC"}), 400
        user = request.form.get("euclid_user", "").strip()
        pw   = request.form.get("euclid_password", "")
        if not user or not pw:
            return jsonify({"ok": False,
                            "error": "username and password are required"}), 400
        if "\n" in user or "\n" in pw:
            return jsonify({"ok": False, "error": "invalid characters"}), 400
        # Quoted heredoc → body is literal (no shell expansion of the
        # password); umask 077 + chmod 600 keep it private on the remote.
        write_cmd = (
            f"umask 077; cat > {_EUCLID_CREDS_REMOTE} <<'__EUCLID_CREDS_EOF__'\n"
            f"{user}\n{pw}\n"
            "__EUCLID_CREDS_EOF__\n"
            f"chmod 600 {_EUCLID_CREDS_REMOTE}"
        )
        rc, _out, err = STATE.ssh.run(write_cmd, timeout=15)
        if rc != 0:
            return jsonify({"ok": False,
                            "error": f"failed to write credentials: {err.strip()}"}), 500
        return jsonify({"ok": True, "user": user})

    @app.route("/euclid-auth/status")
    def euclid_auth_status():
        """Is a credentials file present on FASRC? Returns the username
        (line 1) for display — never the password."""
        if not STATE.ssh or not STATE.ssh.is_connected():
            return jsonify({"present": False, "connected": False})
        rc, out, _err = STATE.ssh.run(
            f"test -e {_EUCLID_CREDS_REMOTE} && head -1 {_EUCLID_CREDS_REMOTE} || true",
            timeout=10,
        )
        lines = [ln for ln in out.splitlines() if ln.strip()]
        user = lines[0].strip() if lines else ""
        return jsonify({"present": bool(user), "connected": True,
                        "user": user or None})

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

    # ---------------- HST PSF page (mirrors /psfs but reads from FASRC) ----
    @app.route("/hst-psf")
    def hst_psf_page():
        cfg_loaded = fasrc_config.load()
        psf_remote_path = f"{cfg_loaded.data_dir}/hst_psf/F814W.fits"
        kernel_remote_path = f"{cfg_loaded.data_dir}/hst_psf/diff_kernel_VIS.fits"
        # Cheap directory listing — one find round-trip.
        ok, entries, list_err = list_remote_dir(
            f"{cfg_loaded.data_dir}/hst_psf",
            glob_pattern="*.fits",
        )
        files: List[Dict[str, Any]] = []
        for e in (entries or []):
            files.append({
                "name":         e["name"],
                "size_mb":      round(e["size"] / (1024 * 1024), 2),
                "mtime":        e["mtime"],
                "remote_path":  f"{cfg_loaded.data_dir}/hst_psf/{e['name']}",
            })
        files.sort(key=lambda d: d["name"])
        return render_template(
            "hst_psf.html",
            files=files,
            list_ok=ok,
            list_err=list_err,
            psf_remote_path=psf_remote_path,
            kernel_remote_path=kernel_remote_path,
            remote_dir=f"{cfg_loaded.data_dir}/hst_psf",
        )

    # ---------------- HST cutouts (per-star gallery on FASRC) -------------
    @app.route("/hst-cutouts")
    def hst_cutouts_page():
        cfg_loaded = fasrc_config.load()
        # Use the existing pagination knob from cutouts.html for symmetry.
        try:
            page = max(1, int(request.args.get("page", 1)))
        except ValueError:
            page = 1
        per_page = 60
        ok, entries, list_err = list_remote_dir(
            f"{cfg_loaded.data_dir}/hst_stars",
            glob_pattern="star_*.fits",
            max_entries=2000,
        )
        # Newest tile run is the most interesting; sort by mtime desc.
        entries = sorted(entries or [], key=lambda e: -float(e["mtime"]))
        total = len(entries)
        n_pages = max(1, (total + per_page - 1) // per_page)
        page = min(page, n_pages)
        start = (page - 1) * per_page
        end   = start + per_page
        page_items = []
        for e in entries[start:end]:
            page_items.append({
                "name":         e["name"],
                "size_kb":      round(e["size"] / 1024, 1),
                "remote_path":  f"{cfg_loaded.data_dir}/hst_stars/{e['name']}",
            })
        return render_template(
            "hst_cutouts.html",
            files=page_items,
            total=total, page=page, n_pages=n_pages,
            list_ok=ok, list_err=list_err,
            remote_dir=f"{cfg_loaded.data_dir}/hst_stars",
        )

    @app.route("/hst-psf/preview.png")
    def hst_psf_preview_png():
        """Pull the F814W PSF + render an asinh PNG for the page header.

        The ``?force=1`` query arg bypasses the rsync cache — used by
        the Sync button after the user has rebuilt the PSF on FASRC.
        """
        cfg_loaded = fasrc_config.load()
        remote = f"{cfg_loaded.data_dir}/hst_psf/F814W.fits"
        force = request.args.get("force") in ("1", "true", "True")
        result = _fasrc_fetcher.fetch_one_file(remote, force=force)
        if not result.ok:
            abort(404)
        png = _render_fits_to_png(result.local_path, Config.BAND_VIS, size=320)
        # When forced, also disable HTTP-level caching so the browser
        # actually paints the freshly-rendered PNG instead of the one
        # the proxy/UA stored 5 min ago.
        max_age = 0 if force else 600
        return send_file(io.BytesIO(png), mimetype="image/png", max_age=max_age)

    def _compute_validate_arrays():
        """Shared preprocessing for ``/hst-psf/validate.png`` and the
        ``A⊛H``-save endpoint.

        Returns ``(e, h, krn, a_conv_h, paths)`` where ``paths`` is a
        dict of the source FITS files used (for provenance headers).

        Aborts the request with 404 if any of the three on-disk files
        is missing — callers don't need to re-check.
        """

        # 1. Resolve local file paths.
        cfg = fasrc_config.load()
        hst_path = _local_path_for(f"{cfg.data_dir}/hst_psf/F814W.fits")
        krn_path = _local_path_for(f"{cfg.data_dir}/hst_psf/diff_kernel_VIS.fits")
        euc_path = os.path.join(Config.EUCLID_PSF_DIR, "euclid_psf_VIS.fits")
        for p in (hst_path, krn_path, euc_path):
            if not os.path.isfile(p):
                abort(404, description=f"missing FITS: {p}")

        # 2. Load script helpers via importlib so we apply the SAME
        # preprocessing the kernel solver did. (The script isn't on
        # the import path; cheap to load once per request.)
        repo_root = os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))
        repo_root = os.path.dirname(repo_root)        # …/EuclidPolish
        script = os.path.join(repo_root, "scripts",
                              "fasrc_compute_differential_kernel.py")
        spec = importlib.util.spec_from_file_location("_fdk", script)
        fdk  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fdk)

        # 3. Load FITS arrays + pixel scales.
        def _read(path):
            with fits.open(path, memmap=False) as hdul:
                return (np.asarray(hdul[0].data, dtype=np.float64),
                        float(hdul[0].header.get("PIXSCALE", 0.05)))
        hst_raw, hst_scale = _read(hst_path)
        euc_raw, euc_scale = _read(euc_path)
        krn,     _         = _read(krn_path)

        # 4. Mirror the script's pipeline: resample → common-side crop
        # → renormalise → bg-subtract → border-zero. We **do not**
        # recenter here: ``compute_differential_kernel`` does its own
        # internal recentering AND undoes the net shift on the kernel,
        # so the saved A satisfies ``A ⊛ H_unrecentered = E_unrecentered``
        # — the contract every downstream caller relies on
        # (fasrc_generate_hst_tfrecords applies A to raw HST cutouts).
        # Validate against the same un-recentered arrays the kernel was
        # designed to work with; if rel.RMS comes out high, the kernel
        # IS bad (not just being tested against the wrong thing).
        common_side  = 511
        border_pixels = 10
        e = fdk._resample_to_hr_grid(euc_raw, euc_scale)
        h = fdk._resample_to_hr_grid(hst_raw, hst_scale)
        e = fdk._centre_crop_to(e, common_side)
        h = fdk._centre_crop_to(h, common_side)
        e = e / e.sum(); h = h / h.sum()
        e = fdk._bg_subtract_and_clip(e)
        h = fdk._bg_subtract_and_clip(h)
        e = fdk._zero_borders(e, border_pixels=border_pixels)
        h = fdk._zero_borders(h, border_pixels=border_pixels)

        # 5. Apply A to H (mode='same' gives back the input shape).
        a_conv_h = fftconvolve(h, krn, mode="same")

        return e, h, krn, a_conv_h, {
            "hst": hst_path, "kernel": krn_path, "euclid": euc_path,
        }

    @app.route("/hst-psf/validate.png")
    def hst_psf_validate_png():
        """Sanity-check the differential kernel: render ``E`` vs ``A⊛H``
        vs residual side-by-side.

        Loads three FITS files entirely from the *local* state — no
        FASRC round-trip:

          * ``A``: rsync'd diff kernel in the fetcher's cache.
          * ``H``: rsync'd HST F814W ePSF, same cache.
          * ``E``: Euclid VIS empirical PSF in ``data/euclid_psf/``.

        Runs the exact same preprocessing the kernel solver did
        (resample, common-side crop, bg-subtract, border-zero,
        renormalise, sub-pixel recenter) on H and E so the comparison
        is honest. Then ``A ⊛ H_cleaned`` is computed via fftconvolve
        and rendered alongside ``E_cleaned`` and their residual.
        """
        e, h, krn, a_conv_h, _ = _compute_validate_arrays()

        # 6. Scalar diagnostics so the user can read off the numbers
        # instead of squinting at the pictures.
        peak_e        = float(e.max())
        peak_a_h      = float(a_conv_h.max())
        flux_e        = float(e.sum())
        flux_a_h      = float(a_conv_h.sum())
        residual      = a_conv_h - e
        rms_residual  = float(np.sqrt((residual ** 2).mean()))
        rms_e         = float(np.sqrt((e ** 2).mean()))
        rel_rms       = rms_residual / rms_e if rms_e > 0 else float("nan")

        # 7. Render the 3-panel figure.

        # Shared asinh stretch for the two PSF panels — the linear scale
        # value is what controls how aggressively faint structure is
        # boosted. Pick something that brings out the spikes without
        # saturating the core; 1 % of the peak is a sensible default.
        asinh_scale = max(peak_e, peak_a_h) * 0.01
        psf_norm = AsinhNorm(
            linear_width=asinh_scale, vmin=0.0,
            vmax=max(peak_e, peak_a_h),
        )
        # Residual stretch: symmetric asinh around 0 with a divergent
        # colormap. AsinhNorm is sign-preserving, so the same stretch
        # behaviour applies in both halves. Use 1 % of max|res| as the
        # linear-width so faint structure shows up but the deepest
        # excursions don't saturate the colormap.
        r_lim = float(np.max(np.abs(residual)))
        if r_lim <= 0:
            r_lim = 1.0     # avoid /0 in AsinhNorm
        res_norm = AsinhNorm(
            linear_width=max(r_lim * 0.01, 1e-12),
            vmin=-r_lim, vmax=+r_lim,
        )

        fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), dpi=110)
        axes[0].imshow(e,        cmap="gray_r", norm=psf_norm, origin="lower",
                       interpolation="nearest")
        axes[0].set_title(f"E (target Euclid)\npeak={peak_e:.3e}, "
                          f"flux={flux_e:.4f}", fontsize=10)
        axes[1].imshow(a_conv_h, cmap="gray_r", norm=psf_norm, origin="lower",
                       interpolation="nearest")
        axes[1].set_title(f"A ⊛ H (kernel result)\npeak={peak_a_h:.3e}, "
                          f"flux={flux_a_h:.4f}", fontsize=10)
        axes[2].imshow(residual, cmap="RdBu_r", norm=res_norm, origin="lower",
                       interpolation="nearest")
        axes[2].set_title(f"A⊛H − E (asinh)\nmax|res|={r_lim:.3e}, "
                          f"RMS/RMS(E)={rel_rms:.3f}", fontsize=10)
        for ax in axes:
            ax.set_xticks([]); ax.set_yticks([])

        peak_ratio = peak_a_h / peak_e if peak_e > 0 else float("nan")
        fig.suptitle(
            f"Kernel sanity check — peak(A⊛H)/peak(E)={peak_ratio:.3f}  "
            f"flux(A⊛H)={flux_a_h:.4f}  rel.RMS residual={rel_rms:.3f}",
            fontsize=10, y=1.02,
        )
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.2)
        plt.close(fig)
        return send_file(io.BytesIO(buf.getvalue()),
                         mimetype="image/png", max_age=0)

    @app.route("/api/hst-psf/save-a-conv-h", methods=["POST"])
    def api_hst_psf_save_a_conv_h():
        """Compute A⊛H via the same chain validate.png uses and save it
        to a sibling FITS in the HST-PSF cache directory.

        Returns JSON with the inspect-relative path so the UI can build
        ``/inspect?fits=…`` and ``/inspect/download?fits=…`` links —
        bytewise inspection without re-uploading or saving by hand.

        Side-effect (intentional): the saved file lives under the FASRC
        cache, which is in ``_inspectable_roots`` already, so the
        inspector accepts it without further whitelist tweaks.
        """

        _, _, _, a_conv_h, paths = _compute_validate_arrays()

        cfg = fasrc_config.load()
        out_path = _local_path_for(
            f"{cfg.data_dir}/hst_psf/a_conv_h_validate.fits"
        )
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        hdu = _fits.PrimaryHDU(
            np.ascontiguousarray(a_conv_h, dtype=np.float32),
        )
        h = hdu.header
        h["OBJECT"]   = ("A conv H (validation)",
                         "kernel applied to HST F814W ePSF")
        h["EUCLBAND"] = ("VIS", "Euclid band the kernel maps INTO")
        h["HSTFILT"]  = ("F814W", "HST filter the kernel maps FROM")
        h["PIXSCALE"] = (Config.DEFAULT_PIXEL_SCALE, "arcsec / pix")
        h["BUNIT"]    = ("", "dimensionless (sums to ~1)")
        h["COMMENT"]  = "Generated by /api/hst-psf/save-a-conv-h"
        h["SRC_HST"]  = (os.path.basename(paths["hst"]),
                         "source HST ePSF FITS")
        h["SRC_KRN"]  = (os.path.basename(paths["kernel"]),
                         "source differential kernel FITS")
        h["SRC_EUC"]  = (os.path.basename(paths["euclid"]),
                         "source Euclid VIS PSF FITS")
        hdu.writeto(out_path, overwrite=True)

        rel = _safe_relpath(os.path.realpath(out_path))
        return jsonify({
            "ok":          True,
            "rel_path":    rel,
            "size_bytes":  int(os.path.getsize(out_path)),
            "shape":       list(a_conv_h.shape),
            "peak":        float(a_conv_h.max()),
            "sum":         float(a_conv_h.sum()),
        })

    @app.route("/api/hst-psf/sync", methods=["POST"])
    def api_hst_psf_sync():
        """Re-rsync the HST PSF + differential kernel from FASRC.

        Bypasses the fetcher's 5-minute cache (``force=True``) so a
        kernel you just rebuilt on FASRC shows up locally without
        waiting. Reports per-file status. Files that don't exist on
        FASRC are reported as failed individually but don't block the
        others.
        """
        cfg_loaded = fasrc_config.load()
        targets = {
            "psf":    f"{cfg_loaded.data_dir}/hst_psf/F814W.fits",
            "kernel": f"{cfg_loaded.data_dir}/hst_psf/diff_kernel_VIS.fits",
        }
        results: Dict[str, Dict[str, Any]] = {}
        any_ok = False
        for key, remote in targets.items():
            r = _fasrc_fetcher.fetch_one_file(remote, force=True)
            entry: Dict[str, Any] = {
                "remote_path": remote,
                "ok":          r.ok,
                "size_bytes":  r.size_bytes,
            }
            if r.ok and r.local_path:
                try:
                    entry["local_mtime"] = os.path.getmtime(r.local_path)
                except OSError:
                    entry["local_mtime"] = None
                any_ok = True
            else:
                entry["error"] = r.error
            results[key] = entry
        return jsonify({"ok": any_ok, "files": results})

    @app.route("/hst-cutouts/preview.png")
    def hst_cutout_preview_png():
        """Pull one HST star cutout + render a thumbnail PNG.

        Same pattern as /cutout-image but the source FITS lives on FASRC,
        not locally. Respects the fetcher's 50 MB cap (star stamps are
        ~260 KB so this is never a concern for this endpoint).
        """
        remote = request.args.get("remote_path", "").strip()
        if not remote:
            abort(400)
        try:
            size = int(request.args.get("size", 140))
        except ValueError:
            size = 140
        if size < 16 or size > 1024:
            abort(400)
        result = _fasrc_fetcher.fetch_one_file(remote)
        if not result.ok:
            abort(404)
        png = _render_fits_to_png(result.local_path, Config.BAND_VIS, size=size)
        return send_file(io.BytesIO(png), mimetype="image/png", max_age=3600)

    # ---------------- HST tiles inspector (header + random cutout) -------
    @app.route("/hst-tiles")
    def hst_tiles_page():
        cfg_loaded = fasrc_config.load()
        remote_dir = f"{cfg_loaded.data_dir}/hst_hlsp"
        # Depth 3 catches files at both the canonical flat layout AND
        # the in-progress scratch layout astroquery uses while a job
        # is still running: <hst_hlsp>/mastDownload/HLSP/<obs_id>/<file>.
        # Without this, the page would show 0 new tiles for the entire
        # duration of a multi-hour download, then jump to the full count
        # only when the post-download flatten runs.
        ok, entries, list_err = list_remote_dir(
            remote_dir, glob_pattern="hlsp_cosmos_*.fits",
            max_entries=200, max_depth=3,
        )
        # De-duplicate by basename, preferring the larger size (catches
        # the brief window where a file exists both flat and nested
        # mid-flatten — flat is the canonical copy when it's complete).
        best: Dict[str, Dict[str, Any]] = {}
        for e in (entries or []):
            name = e["name"]
            prev = best.get(name)
            if prev is None or e["size"] > prev["size"]:
                best[name] = e
        tiles = [
            {
                "name":        e["name"],
                "size_gb":     round(e["size"] / 1e9, 2),
                "mtime":       e["mtime"],
                "remote_path": f"{remote_dir}/{e['name']}",
            }
            for e in best.values()
        ]
        tiles.sort(key=lambda t: t["name"])
        return render_template(
            "hst_tiles.html",
            tiles=tiles, list_ok=ok, list_err=list_err,
            remote_dir=remote_dir,
        )

    @app.route("/fasrc/tile/header")
    def fasrc_tile_header():
        """JSON: full FITS header of one tile (no big transfer)."""
        path = request.args.get("path", "").strip()
        if not path or not is_allowed_remote_path(path):
            abort(400)
        rc, out, err = run_remote_python(
            "scripts/fasrc_inspect_tile.py",
            ["--path", path, "--mode", "header"],
            binary=False, timeout=20,
        )
        if rc != 0:
            return jsonify({"ok": False,
                            "error": err.strip() or out[:500]}), 502
        try:
            payload = json.loads(out)
        except json.JSONDecodeError as e:
            return jsonify({"ok": False,
                            "error": f"bad header JSON: {e}"}), 502
        payload["ok"] = True
        return jsonify(payload)

    @app.route("/fasrc/tile/cutout.png")
    def fasrc_tile_cutout_png():
        """Stream a single random PNG cutout from a remote tile.

        The remote script emits the PNG to stdout and a JSON sidecar
        (centre, sigma-clipped stats, blank flag) to stderr. We surface
        the stats via an ``X-Cutout-Stats`` response header so the page
        JS can display "median, σ, is-blank" next to the image — handy
        for diagnosing "why does this cutout look like a gradient?"
        cases (usually a bright star or empty tile-edge region).
        """
        path = request.args.get("path", "").strip()
        if not path or not is_allowed_remote_path(path):
            abort(400)
        try:
            size = int(request.args.get("size", 256))
            seed = int(request.args.get("seed", -1))
        except ValueError:
            abort(400)
        if size < 32 or size > 1024:
            abort(400)
        rc, out_bytes, err = run_remote_python(
            "scripts/fasrc_inspect_tile.py",
            ["--path", path, "--mode", "cutout",
             "--size", str(size), "--seed", str(seed)],
            binary=True, timeout=30,
        )
        if rc != 0 or not out_bytes:
            return jsonify({"ok": False,
                            "error": (err if isinstance(err, str) else err.decode(errors="replace")).strip() or "empty cutout"}), 502
        resp = send_file(
            io.BytesIO(out_bytes), mimetype="image/png", max_age=0,
        )
        # err contains the JSON sidecar; pass it through as a header so
        # the page JS can render it without a second SSH round-trip.
        if isinstance(err, str) and err.strip():
            resp.headers["X-Cutout-Stats"] = err.strip().splitlines()[-1]
        return resp

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
        # Synthetic records are now generated on FASRC, so list the remote
        # records_v2 dir (one shallow ls) rather than the local disk. SSH is
        # up on this page (the connection gate is upstream); on failure we
        # just show "no records yet".
        cfg_loaded = fasrc_config.load()
        records_dir = f"{cfg_loaded.data_dir}/images/records_v2"
        tfrecords: List[Dict[str, Any]] = []
        ok, entries, _ = list_remote_dir(
            records_dir, glob_pattern="*.tfrecord", max_entries=50,
        )
        if ok:
            for e in sorted(entries, key=lambda r: str(r.get("name", ""))):
                tfrecords.append({
                    "name":    e["name"],
                    "size_mb": f"{int(e.get('size', 0)) / 1e6:.1f}",
                })
        return render_template("sky.html",
                               records_dir=records_dir,
                               tfrecords=tfrecords)

    @app.route("/roundtrip")
    def roundtrip_page():
        """Dedicated page for the round-trip self-supervised pipeline.

        Mounts the three FASRC steps that build and use the real-Euclid
        round-trip records (download cutouts, chop into TFRecords, train
        the WDSR with the round-trip mix). The viz column shows a live
        disk-state summary so the user can see at a glance whether each
        step has produced its output yet.
        """
        cfg_loaded = fasrc_config.load()
        cutouts_dir = f"{cfg_loaded.data_dir}/euclid_sky/cutouts"
        records_dir = f"{cfg_loaded.data_dir}/images/records_v2_euclid_roundtrip"

        # Bundle summary — each ``sky_NNNN.fits`` carries all four bands,
        # so one ``find`` over the cutouts dir gives us the per-position
        # count and aggregate disk size in a single SSH round-trip. SSH
        # may be down on this page render (the connection gate is
        # downstream); the helper returns ok=False quietly and we just
        # show "no cutouts yet" in that case.
        cutouts_summary: List[Dict[str, Any]] = []
        ok, entries, _ = list_remote_dir(
            cutouts_dir, glob_pattern="sky_*.fits", max_entries=100_000,
        )
        if ok and entries:
            n = len(entries)
            size_gb = sum(int(e.get("size", 0)) for e in entries) / 1e9
            cutouts_summary.append({
                "band":    "all 4 bands (bundled)",
                "n":       n,
                "size_gb": f"{size_gb:.2f}",
            })

        # Round-trip TFRecord listing — one shallow ls.
        tfrecords: List[Dict[str, Any]] = []
        ok, entries, _ = list_remote_dir(
            records_dir, glob_pattern="*.tfrecord", max_entries=20,
        )
        if ok:
            for e in sorted(entries, key=lambda r: str(r.get("name", ""))):
                tfrecords.append({
                    "name":    e["name"],
                    "size_gb": f"{int(e.get('size', 0)) / 1e9:.2f}",
                })

        return render_template(
            "roundtrip.html",
            cutouts_dir=cutouts_dir,
            records_dir=records_dir,
            cutouts_summary=cutouts_summary,
            tfrecords=tfrecords,
            default_num_res_blocks=Config.DEFAULT_NUM_RES_BLOCKS,
        )

    @app.route("/roundtrip/inspect", methods=["POST"])
    def roundtrip_inspect():
        """Fetch one round-trip sky cutout from FASRC and render it — plus
        its round-trip reconstruction when a local checkpoint exists."""
        if not STATE.ssh or not STATE.ssh.is_connected():
            return jsonify({"error": "not connected to FASRC — connect "
                            "first to fetch the cutout bundle"}), 400
        try:
            pos_id = int(request.form["pos_id"])
        except (KeyError, ValueError):
            return jsonify({"error": "pos_id must be an integer"}), 400
        if pos_id < 0:
            return jsonify({"error": f"pos_id={pos_id} must be >= 0"}), 400
        ckpt_dir = request.form.get("checkpoint_dir",
                                    Config.DEFAULT_CHECKPOINT_DIR)
        nrb = int(request.form.get("num_res_blocks",
                                   Config.DEFAULT_NUM_RES_BLOCKS))
        asinh = _parse_asinh_scale(request.form.get("asinh_scale", ""))
        show_all = request.form.get("show_all_bands", "").lower() in (
            "1", "on", "true", "yes",
        )
        job_id = REGISTRY.spawn(
            label=f"round-trip inspect sky_{pos_id:04d}",
            target=lambda cap: _job_roundtrip_inspect(
                cap, pos_id, ckpt_dir, nrb,
                asinh_scale=asinh, show_all_bands=show_all,
            ),
        )
        return jsonify({"job_id": job_id})

    # ---------------- Training page ----------------
    @app.route("/training")
    def training_page():
        return render_template(
            "training.html",
            tfrecords=_tfrecords_status(),
            checkpoints=_checkpoints_status(),
        )

    # ---------------- Inference page ----------------
    @app.route("/inference")
    def inference_page():
        # Most-recent reconstruction PNGs (newest first). Each thumbnail
        # links to its sidecar SR.fits when one exists on disk.
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
                # Synthetic path saves PNG + same-stem FITS in this dir.
                fits_rel = None
                stem = os.path.splitext(fname)[0]
                fits_local = os.path.join(rdir, f"{stem}.fits")
                if os.path.isfile(fits_local):
                    fits_rel = os.path.relpath(fits_local, Config.VIS_DIR)
                recon_pngs.append({
                    "rel": rel, "name": fname, "mtime": mtime,
                    "fits_rel": fits_rel,
                })
            recon_pngs.sort(key=lambda d: d["mtime"], reverse=True)

        # Persistent Euclid inference cutouts: one entry per cache dir.
        # Each entry exposes the four LR FITS + the SR FITS as download
        # links so the user can re-load them in their own tools.
        euclid_runs: list[Dict[str, Any]] = []
        eroot = os.path.join(Config.EUCLID_INFERENCE_DIR, "cutouts")
        if os.path.isdir(eroot):
            for tag in os.listdir(eroot):
                d = os.path.join(eroot, tag)
                if not os.path.isdir(d):
                    continue
                files = []
                for name in ("original_stack.fits",
                             "VIS.fits", "Y_E.fits", "J_E.fits", "H_E.fits",
                             "SR.fits", "SR_forward.fits", "residual.fits"):
                    f = os.path.join(d, name)
                    if os.path.isfile(f):
                        rel = os.path.relpath(f, Config.EUCLID_INFERENCE_DIR)
                        files.append({
                            "name":     name,
                            "rel":      rel,
                            "size_kb":  int(os.path.getsize(f) / 1024),
                        })
                if files:
                    # The dir name is now a fixed "latest" slot, so read the
                    # real position out of the SR header for a useful label.
                    label = tag
                    sr_local = os.path.join(d, "SR.fits")
                    if os.path.isfile(sr_local):
                        try:
                            hdr = fits.getheader(sr_local)
                            if hdr.get("RA") is not None and hdr.get("DEC") is not None:
                                label = (f"RA {float(hdr['RA']):.4f}, "
                                         f"Dec {float(hdr['DEC']):+.4f}")
                                if hdr.get("CSIZE") is not None:
                                    label += f"  ({int(hdr['CSIZE'])} px)"
                        except Exception:  # noqa: BLE001 — label is cosmetic
                            pass
                    euclid_runs.append({
                        "tag":   tag,
                        "label": label,
                        "files": files,
                        "mtime": max(os.path.getmtime(os.path.join(d, f["name"]))
                                     for f in files),
                    })
            euclid_runs.sort(key=lambda d: d["mtime"], reverse=True)

        return render_template(
            "inference.html",
            checkpoints=_checkpoints_status(),
            tfrecords=_tfrecords_status(),
            recon_pngs=recon_pngs,
            euclid_runs=euclid_runs,
            default_num_res_blocks=Config.DEFAULT_NUM_RES_BLOCKS,
        )

    def _parse_asinh_scale(raw: str) -> Optional[float]:
        """Form parser for the asinh-scale knob. Empty string / 0 / bad
        input → None, which makes plot_reconstruction fall back to
        Config.STRETCH_SCALE_E (1000 e⁻, the training default)."""
        raw = (raw or "").strip()
        if not raw:
            return None
        try:
            val = float(raw)
        except ValueError:
            return None
        return val if val > 0 else None

    @app.route("/inference/generate-reconstruct", methods=["POST"])
    def inference_generate_reconstruct():
        ckpt_dir = request.form.get("checkpoint_dir", Config.DEFAULT_CHECKPOINT_DIR)
        nrb = int(request.form.get("num_res_blocks", Config.DEFAULT_NUM_RES_BLOCKS))
        asinh = _parse_asinh_scale(request.form.get("asinh_scale", ""))
        try:
            hr_size = int(request.form.get("hr_image_size", 510))
        except (TypeError, ValueError):
            return jsonify({"error": "hr_image_size must be an integer"}), 400
        # The generator rebins NISP ×6, so the HR side must be a multiple of
        # 6. Clamp to a sane range and round to the nearest multiple.
        hr_size = max(60, min(2048, hr_size))
        if hr_size % 6:
            hr_size = int(round(hr_size / 6)) * 6
        try:
            n_pairs = int(request.form.get("n_pairs", 1))
        except (TypeError, ValueError):
            n_pairs = 1
        n_pairs = max(1, min(8, n_pairs))
        job_id = REGISTRY.spawn(
            label=f"gen+reconstruct {n_pairs}×{hr_size}px (FASRC login-node gen)",
            target=lambda cap: _job_generate_reconstruct(
                cap, ckpt_dir, nrb, hr_size, n_pairs, asinh_scale=asinh,
            ),
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
        asinh = _parse_asinh_scale(request.form.get("asinh_scale", ""))
        # HTML checkbox: present in form data → ``"on"`` (or whatever
        # ``value=`` was set to); absent if unchecked. Parse truthy.
        show_all = request.form.get("show_all_bands", "").lower() in (
            "1", "on", "true", "yes",
        )
        job_id = REGISTRY.spawn(
            label=f"infer Euclid cutout @ ({ra:.4f}, {dec:+.4f})",
            target=lambda cap: _job_reconstruct_euclid_cutout(
                cap, ra, dec, ckpt_dir, nrb, size,
                asinh_scale=asinh, show_all_bands=show_all,
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

    def _sky_records_remote_dir() -> str:
        cfg = fasrc_config.load()
        return f"{cfg.data_dir}/images/records_v2"

    def _sky_records_local_dir() -> str:
        """Local cache dir mirroring the remote synthetic records dir.

        Same convention as :func:`fasrc_fetcher._local_path_for`, so the
        viewer reads exactly what ``/api/sky/sync`` (fetch_one_file)
        writes. The synthetic generator runs on FASRC, so the preview
        renders the synced shards — not a stale local copy."""
        any_path = f"{_sky_records_remote_dir()}/clean_validate.tfrecord"
        return os.path.dirname(_local_path_for(any_path))

    @app.route("/view/sky")
    def view_sky():
        subset = request.args.get("subset", "validate")
        kind   = request.args.get("kind",   "clean")
        band   = request.args.get("band",   "VIS")
        try:
            idx = int(request.args.get("i", 0))
        except ValueError:
            idx = 0
        # Render from the FASRC-synced records cache (populated by
        # /api/sky/sync), not the local data dir.
        png = _render_sky_record_png(
            subset, kind, band, idx, records_dir=_sky_records_local_dir(),
        )
        return send_file(io.BytesIO(png), mimetype="image/png", max_age=0)

    @app.route("/view/catalog")
    def view_catalog():
        view = request.args.get("view", "positions")
        # Render from the FASRC catalog (pulled to the local cache), not a
        # stale local stars.csv — the query writes it on netscratch.
        out = _fasrc_catalog_dir(force=True)
        if out is None:
            abort(404)
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
            os.makedirs(Config.VIS_DIR, exist_ok=True)
            plot_training_log(log_path, out_png)
        return send_file(out_png, mimetype="image/png", max_age=0)

    @app.route("/api/sky/totals")
    def api_sky_totals():
        local = _sky_records_local_dir()
        return jsonify({
            name: _record_count(name, records_dir=local)
            for name in ("clean_train", "clean_validate",
                         "dirty_train", "dirty_validate",
                         "hr_train",    "hr_validate")
        })

    @app.route("/api/sky/sync", methods=["POST"])
    def api_sky_sync():
        """Rsync the synthetic TFRecord shards from FASRC into the local
        cache so the preview can render them.

        ``include_train`` (default false) controls whether the large
        train-split files are pulled; validate shards are always included.
        Lifts the fetcher's 50 MB cap to 5 GB since TFRecords are large and
        this is an explicit user-requested transfer."""
        remote_dir = _sky_records_remote_dir()
        include_train = (request.values.get("include_train", "false")
                         .lower() in ("1", "true", "yes", "on"))
        targets: Dict[str, str] = {}
        for kind in ("clean", "dirty", "hr"):
            targets[f"{kind}_validate"] = f"{remote_dir}/{kind}_validate.tfrecord"
            if include_train:
                targets[f"{kind}_train"] = f"{remote_dir}/{kind}_train.tfrecord"
        max_bytes = 5 * 1024 * 1024 * 1024
        results: Dict[str, Dict[str, Any]] = {}
        any_ok = False
        for key, remote in targets.items():
            r = _fasrc_fetcher.fetch_one_file(remote, force=True, max_bytes=max_bytes)
            entry: Dict[str, Any] = {"ok": r.ok, "size_bytes": r.size_bytes}
            if r.ok:
                any_ok = True
            else:
                entry["error"] = r.error
            results[key] = entry
        return jsonify({"ok": any_ok, "files": results,
                        "include_train": include_train})

    # =========================================================================
    # HST Catalog — same viewer as /sky but pointed at the FASRC-cached HST
    # TFRecord pairs that scripts/fasrc_generate_hst_tfrecords.py wrote.
    # =========================================================================

    def _hst_pairs_remote_dir() -> str:
        cfg = fasrc_config.load()
        return f"{cfg.data_dir}/images/records_v2_hst"

    def _hst_pairs_local_dir() -> str:
        """Local cache dir mirroring the remote HST records dir.

        Uses the same convention as :func:`fasrc_fetcher._local_path_for`
        so the page reads exactly what ``fetch_one_file`` writes — no
        chance of pointing at the wrong directory."""
        any_path = f"{_hst_pairs_remote_dir()}/clean_validate.tfrecord"
        return os.path.dirname(_local_path_for(any_path))

    @app.route("/hst-pairs")
    def hst_pairs_page():
        local_dir = _hst_pairs_local_dir()
        return render_template(
            "hst_pairs.html",
            tfrecords=_tfrecords_status(local_dir),
            local_dir=local_dir,
            remote_dir=_hst_pairs_remote_dir(),
            # Validation set is small (~80 MB per file × 3 files); train
            # set is multi-GB. Default the index nav to validate so the
            # page works the moment the user clicks Sync.
            default_subset="validate",
        )

    def _load_diff_kernel_local() -> Optional[np.ndarray]:
        """Return the analytic differential kernel from the local cache.

        Returns ``None`` (without raising) when the kernel FITS isn't
        synced yet, so callers can abort with a useful 404 message.
        """
        cfg = fasrc_config.load()
        krn_path = _local_path_for(
            f"{cfg.data_dir}/hst_psf/diff_kernel_VIS.fits"
        )
        if not os.path.isfile(krn_path):
            return None
        with fits.open(krn_path) as hdul:
            return np.asarray(hdul[0].data, dtype=np.float32)

    def _hst_analytic_lr_cube(
        clean_hr_data: np.ndarray, kernel: np.ndarray, *, rebin: int = 2,
    ) -> np.ndarray:
        """Per-band ``fftconvolve(clean_HR, A)`` then sum-rebin × ``rebin``.

        Returns the noiseless deterministic forward at the LR grid —
        what a perfect forward model with no trained CNN and no noise
        injection would produce. Lets the user isolate ringing
        artifacts the CNN was meant to suppress.
        """
        out_hr = np.empty_like(clean_hr_data, dtype=np.float32)
        for c in range(clean_hr_data.shape[-1]):
            out_hr[..., c] = fftconvolve(
                clean_hr_data[..., c].astype(np.float32), kernel,
                mode="same",
            )
        return MultiBandSkyImage.rebin_array(out_hr, rebin)

    def _compute_hst_pair_arrays(subset, kind, index, records_dir):
        """Raw-array companion to ``_render_sky_record_png`` /
        ``_render_sky_record_pair_png``. Returns
        ``({name: array}, meta_dict)`` for the kind requested.

        For triptych ``kind="pair"`` the 4-band records keep their
        ``(H, W, C)`` shape so DS9 sees a proper data cube; FITS
        stores axes in NAXIS3-then-2 ordering so it surveys as
        "bands × H × W" via image-cube viewers.

        ``kind="dirty_analytic"`` is computed live from the matching
        ``clean`` record by applying the cached differential kernel
        (no CNN, no noise) — used by the live debug view to surface
        ringing artifacts directly attributable to the bare kernel.
        """
        if subset not in ("train", "validate"):
            abort(400)
        if kind not in ("clean", "dirty", "hr", "pair", "dirty_analytic"):
            abort(400)

        def _load(name):
            path = tfrecord_path(records_dir, name)
            if not os.path.exists(path):
                abort(404)
            recs = read_multiband_skyimages(path, num_images=max(index + 1, 1))
            if not recs or index >= len(recs):
                abort(404)
            return recs[min(index, len(recs) - 1)]

        meta_base = {
            "KIND":   kind,
            "SUBSET": subset,
            "INDEX":  int(index),
        }

        if kind == "pair":
            rs = {k: _load(f"{k}_{subset}") for k in ("clean", "dirty", "hr")}
            return (
                {k.upper(): np.asarray(rs[k].data, dtype=np.float32)
                 for k in ("clean", "dirty", "hr")},
                {**meta_base,
                 "PXSCALEC": float(rs["clean"].pixel_scale_arcsec),
                 "PXSCALED": float(rs["dirty"].pixel_scale_arcsec),
                 "PXSCALEH": float(rs["hr"].pixel_scale_arcsec)},
            )

        if kind == "dirty_analytic":
            clean = _load(f"clean_{subset}")
            kernel = _load_diff_kernel_local()
            if kernel is None:
                abort(404, description=(
                    "diff_kernel_VIS.fits not in local cache — run the "
                    "kernel step on FASRC and pull it down first."
                ))
            lr = _hst_analytic_lr_cube(
                np.asarray(clean.data, dtype=np.float32), kernel,
            )
            # HR pixel scale (0.05") halves to 0.10" after sum-rebin ×2.
            lr_pxscale = float(clean.pixel_scale_arcsec) * 2.0
            return (
                {"DIRTY_ANALYTIC": lr},
                {**meta_base, "PXSCALE": lr_pxscale,
                 "PXSCALEC": float(clean.pixel_scale_arcsec)},
            )

        rec = _load(f"{kind}_{subset}")
        return (
            {kind.upper(): np.asarray(rec.data, dtype=np.float32)},
            {**meta_base, "PXSCALE": float(rec.pixel_scale_arcsec)},
        )

    def _render_hst_dirty_analytic_png(
        subset: str, band: str, index: int, records_dir: str,
    ) -> bytes:
        """Render ``sum_rebin(A ⊛ clean_HR)`` for one index — live, no CNN.

        Same asinh / Lupton conventions as :func:`_render_sky_record_png`
        so the output is directly comparable to the cached ``dirty``
        record; the only difference is which forward operator produced
        the LR side.
        """
        matplotlib.use("Agg")
        if subset not in ("train", "validate"):
            abort(400)
        if band not in Config.LR_INPUT_BAND_NAMES and band != "color":
            abort(400)
        arrays, meta = _compute_hst_pair_arrays(
            subset, "dirty_analytic", index, records_dir,
        )
        data = arrays["DIRTY_ANALYTIC"]
        pxscale = float(meta["PXSCALE"])

        if band == "color" and data.shape[-1] >= len(Config.LR_INPUT_BAND_NAMES):
            rgb = calibrated_rgb_panel(
                data, band_names=Config.LR_INPUT_BAND_NAMES,
                scheme="vis_nisp", reference="solar", stretch="asinh",
                asinh_scale_e=float(Config.BAND_VIS.asinh_stretch_scale_e),
            )
            fig, ax = plt.subplots(figsize=(6.5, 6.5))
            ax.imshow(np.clip(rgb, 0.0, 1.0), origin="lower",
                      interpolation="nearest")
            ax.set_title(
                f"dirty (analytic A, no CNN, no noise) {subset} · color "
                f"· idx {index}  ({data.shape[0]}×{data.shape[1]} @ "
                f"{pxscale:.3f}\"/pix)", fontsize=10,
            )
        else:
            if data.shape[-1] == 1:
                plane, band_name = data[..., 0], "VIS"
            else:
                k = list(Config.LR_INPUT_BAND_NAMES).index(band)
                plane, band_name = data[..., k], band
            bcfg = Config.get_band(band_name)
            stretched = np.arcsinh(plane / float(bcfg.asinh_stretch_scale_e))
            lo, hi = np.percentile(stretched, [1.0, 99.7])
            if hi <= lo:
                hi = lo + 1.0
            fig, ax = plt.subplots(figsize=(6.5, 6.5))
            ax.imshow(stretched, cmap="gray_r", origin="lower",
                      vmin=lo, vmax=hi, interpolation="nearest")
            ax.set_title(
                f"dirty (analytic A, no CNN, no noise) {subset} · "
                f"{band_name} · idx {index}  "
                f"({data.shape[0]}×{data.shape[1]} @ "
                f"{pxscale:.3f}\"/pix)", fontsize=10,
            )
        ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, dpi=110, bbox_inches="tight", format="png")
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()

    @app.route("/view/hst-pair")
    def view_hst_pair():
        subset = request.args.get("subset", "validate")
        kind   = request.args.get("kind",   "clean")
        band   = request.args.get("band",   "VIS")
        try:
            idx = int(request.args.get("i", 0))
        except ValueError:
            idx = 0
        # kind="pair" routes to the side-by-side triptych so the user
        # can compare HR clean / LR dirty / HR target at one index
        # without chip-toggling. ``dirty_analytic`` runs the bare
        # differential kernel live (no CNN, no noise) on the matching
        # ``clean`` HR record; everything else uses the single-image
        # renderer reading directly from the cached TFRecords.
        if kind == "pair":
            png = _render_sky_record_pair_png(
                subset, band, idx,
                records_dir=_hst_pairs_local_dir(),
            )
        elif kind == "dirty_analytic":
            png = _render_hst_dirty_analytic_png(
                subset, band, idx,
                records_dir=_hst_pairs_local_dir(),
            )
        else:
            png = _render_sky_record_png(
                subset, kind, band, idx,
                records_dir=_hst_pairs_local_dir(),
            )
        return send_file(io.BytesIO(png), mimetype="image/png", max_age=0)

    @app.route("/view/hst-pair.fits")
    def view_hst_pair_fits():
        """Raw-array FITS companion of ``/view/hst-pair``.

        Returns the linear electron pixel values for the requested
        kind. The ``band`` query arg controls slicing:

        * ``band`` in ``Config.LR_INPUT_BAND_NAMES`` (e.g. ``VIS``)
          → return only that band as a 2-D ``(H, W)`` image. Lets the
          user A/B one band against another viewer (e.g. compare
          ``dirty`` VIS to ``dirty_analytic`` VIS in DS9 without
          slicing).
        * ``band=color`` or absent → full ``(H, W, C)`` cube, same
          as before.
        * Single-band records (e.g. ``hr``) are returned as-is in
          either case.
        """
        subset = request.args.get("subset", "validate")
        kind   = request.args.get("kind",   "clean")
        band   = request.args.get("band",   "color")
        try:
            idx = int(request.args.get("i", 0))
        except ValueError:
            idx = 0
        arrays, meta = _compute_hst_pair_arrays(
            subset, kind, idx,
            records_dir=_hst_pairs_local_dir(),
        )
        # Slice down to one band when the chip selected a real band.
        # Triptych ``kind="pair"`` keeps all panels cubed because there
        # the user wants to inspect three records together; slicing
        # one out doesn't compose with the others.
        if (band in Config.LR_INPUT_BAND_NAMES
                and kind != "pair"):
            k_idx = list(Config.LR_INPUT_BAND_NAMES).index(band)
            sliced: Dict[str, np.ndarray] = {}
            for ext_name, arr in arrays.items():
                if arr.ndim == 3 and arr.shape[-1] > k_idx:
                    sliced[ext_name] = arr[..., k_idx]
                else:
                    # Single-band records (e.g. HR) pass through.
                    sliced[ext_name] = arr
            arrays = sliced
            meta = {**meta, "BAND": band}
        data = _arrays_to_fits_bytes(arrays, header_meta=meta)
        band_tag = f"_{band}" if band in Config.LR_INPUT_BAND_NAMES else ""
        fname = f"hst_{kind}_{subset}_{idx}{band_tag}.fits"
        return send_file(
            io.BytesIO(data), mimetype="application/fits",
            as_attachment=True, download_name=fname, max_age=0,
        )

    @app.route("/api/hst-pairs/totals")
    def api_hst_pairs_totals():
        local = _hst_pairs_local_dir()
        return jsonify({
            name: _record_count(name, records_dir=local)
            for name in ("clean_train", "clean_validate",
                         "dirty_train", "dirty_validate",
                         "hr_train",    "hr_validate")
        })

    @app.route("/api/hst-pairs/status")
    def api_hst_pairs_status():
        return jsonify(_tfrecords_status(_hst_pairs_local_dir()))

    @app.route("/api/hst-pairs/sync", methods=["POST"])
    def api_hst_pairs_sync():
        """Rsync HST TFRecord files from FASRC into the local cache.

        Form arg ``include_train`` (default false) controls whether the
        large train-split files are pulled. Validation files (~80 MB
        each, 3 files) are always included since they're small enough
        that "sync to view" makes sense; train can be many GB.

        Lifts the fetcher's default 50 MB cap to 5 GB per file because
        TFRecord files are intentionally large and the user explicitly
        asked for this transfer.
        """
        remote_dir = _hst_pairs_remote_dir()
        include_train = (request.values.get("include_train", "false")
                         .lower() in ("1", "true", "yes", "on"))
        targets: Dict[str, str] = {}
        for kind in ("clean", "dirty", "hr"):
            targets[f"{kind}_validate"] = (
                f"{remote_dir}/{kind}_validate.tfrecord"
            )
            if include_train:
                targets[f"{kind}_train"] = (
                    f"{remote_dir}/{kind}_train.tfrecord"
                )

        max_bytes = 5 * 1024 * 1024 * 1024
        results: Dict[str, Dict[str, Any]] = {}
        any_ok = False
        for key, remote in targets.items():
            r = _fasrc_fetcher.fetch_one_file(remote, force=True, max_bytes=max_bytes)
            entry: Dict[str, Any] = {
                "remote_path": remote,
                "ok":          r.ok,
                "size_bytes":  r.size_bytes,
            }
            if r.ok and r.local_path:
                try:
                    entry["local_mtime"] = os.path.getmtime(r.local_path)
                except OSError:
                    entry["local_mtime"] = None
                any_ok = True
            else:
                entry["error"] = r.error
            results[key] = entry
        return jsonify({"ok": any_ok, "files": results,
                        "include_train": include_train})

    @app.route("/sky/fits")
    def sky_fits():
        """Export one sky-record band+index as FITS and return the file."""
        path = _export_sky_record_fits(
            subset=request.args.get("subset", ""),
            kind=request.args.get("kind", ""),
            band=request.args.get("band", ""),
            index=request.args.get("i", "0"),
            records_dir=_sky_records_local_dir(),
        )
        return send_file(path, as_attachment=True,
                         download_name=os.path.basename(path),
                         mimetype="application/fits")

    @app.route("/sky/inspect")
    def sky_inspect():
        """Export the requested record then redirect into the inspector."""
        path = _export_sky_record_fits(
            subset=request.args.get("subset", ""),
            kind=request.args.get("kind", ""),
            band=request.args.get("band", ""),
            index=request.args.get("i", "0"),
            records_dir=_sky_records_local_dir(),
        )
        return redirect(url_for("inspect_fits_page",
                                fits=_safe_relpath(path)))

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

    @app.route("/inference-files/<path:relpath>")
    def serve_inference_files(relpath: str):
        """Serve FITS / PNG files from data/euclid_inference/.

        Used by the inference UI to download the persisted cutouts and
        SR result. Path is jailed to ``Config.EUCLID_INFERENCE_DIR`` to
        prevent traversal — anything resolving outside that tree 403s.
        """
        root = os.path.realpath(Config.EUCLID_INFERENCE_DIR)
        full = os.path.realpath(os.path.join(root, relpath))
        if not full.startswith(root + os.sep):
            abort(403)
        if not os.path.isfile(full):
            abort(404)
        # FITS gets the application/fits MIME so browsers prompt to
        # save instead of trying to render it as text.
        mt = ("application/fits" if full.lower().endswith(".fits")
              else "application/octet-stream")
        return send_file(
            full, mimetype=mt, as_attachment=True,
            download_name=os.path.basename(full),
        )

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
    # Universal FITS inspector — every image card across the UI links here.
    # =========================================================================

    @app.route("/inspect")
    def inspect_fits_page():
        path = _resolve_inspectable_fits(request.args.get("fits", ""))
        info = _fits_file_info(path)
        rows = _read_fits_header_rows(path)
        # Project-relative path is what shows in the UI + what the
        # download/preview routes echo back (so refresh from a bookmark
        # keeps working as long as the file is still at that location).
        rel = _safe_relpath(path)
        return render_template(
            "inspect_fits.html",
            file=info,
            hdus=rows,
            rel=rel,
            # Roots are displayed so the user can confirm which data
            # subtree the file came from (useful when triaging mismatched
            # outputs from multiple runs).
            allowed_roots=_inspectable_roots(),
        )

    @app.route("/inspect/download")
    def inspect_fits_download():
        path = _resolve_inspectable_fits(request.args.get("fits", ""))
        return send_file(
            path, as_attachment=True,
            download_name=os.path.basename(path),
            mimetype="application/fits",
        )

    @app.route("/fasrc/file/inspect")
    def fasrc_file_inspect():
        """Fetch one file from FASRC (cached) then redirect to /inspect.

        Query param: ``remote_path=<absolute path on FASRC>``. Subject
        to all the safeguards in :mod:`euclid_polish.web.fasrc_fetcher`
        (size cap, allowed roots, cache TTL).
        """
        remote = request.args.get("remote_path", "").strip()
        if not remote:
            abort(400)
        result = _fasrc_fetcher.fetch_one_file(remote)
        if not result.ok:
            return render_template(
                "fasrc_fetch_error.html", remote=remote, error=result.error,
            ), 502
        # Hand off to the existing inspector with the local cache path.
        return redirect(url_for(
            "inspect_fits_page",
            fits=_safe_relpath(result.local_path),
        ))

    @app.route("/fasrc/file/download")
    def fasrc_file_download():
        """Fetch one file from FASRC (cached) and send it back directly."""
        remote = request.args.get("remote_path", "").strip()
        if not remote:
            abort(400)
        result = _fasrc_fetcher.fetch_one_file(remote)
        if not result.ok:
            return jsonify({"ok": False, "error": result.error}), 502
        return send_file(
            result.local_path, as_attachment=True,
            download_name=os.path.basename(remote),
        )

    @app.route("/inspect/preview.png")
    def inspect_fits_preview():
        path = _resolve_inspectable_fits(request.args.get("fits", ""))
        try:
            size = int(request.args.get("size", 512))
        except ValueError:
            size = 512
        if size < 16 or size > 2048:
            abort(400)
        # /inspect is universal — could be a sky cutout, a PSF, a diff
        # kernel, a residual map, anything in the allowed roots. The
        # band-aware renderer assumes Euclid cutout units (~1000 e⁻ asinh
        # knee) and silently misrenders everything else; use the
        # data-adaptive ZScale + Asinh renderer instead.
        png = _render_fits_to_png_adaptive(path, size=size)
        # /inspect is interactive debugging — when the underlying FITS
        # gets regenerated (e.g. you re-run the differential-kernel
        # script with different params), the preview must reflect the
        # new file on the next page load. A long ``max_age`` here means
        # the browser shows the stale render for an hour, which is the
        # opposite of what an inspector is for.
        resp = send_file(io.BytesIO(png), mimetype="image/png", max_age=0)
        resp.headers["Cache-Control"] = "no-store, must-revalidate"
        return resp

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
        # Don't call STATE.public_status() here — it runs ``bw --version``
        # and ``ssh -O check`` subprocesses (up to ~600 ms combined on a
        # warm cache, several seconds on a cold one) and the template
        # doesn't even use the result. The page's own JS fetches connection
        # state via /api/fasrc/status after the DOM loads, which is async
        # and doesn't block render.
        cfg = fasrc_config.load()
        return render_template(
            "fasrc.html",
            cfg=cfg,
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

    @app.route("/api/fasrc/connect", methods=["POST"])
    def api_fasrc_connect():
        cfg = fasrc_config.load()
        if not cfg.ssh_user:
            return jsonify({"ok": False,
                            "error": "set ssh_user in Settings first"}), 400
        STATE.ssh = SSHSession(SSHConfig(
            user=cfg.ssh_user, host=cfg.ssh_host,
            socket=cfg.control_socket,
            control_persist=cfg.control_persist,
        ))
        try:
            STATE.ssh.connect()
        except SSHError as e:
            STATE.ssh = None
            return jsonify({"ok": False, "error": str(e)}), 400
        STATE.connected_at = time.time()
        # Catch up on any jobs that finished while the server was offline:
        # squeue no longer lists them, so reconcile marks them DONE and
        # the ssh-passing path fetches their sacct accounting into the
        # CSV log. Best-effort — failures here must not block connect.
        try:
            fasrc_jobs.sync_pending_on_connect(STATE.ssh)
        except Exception:
            pass
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
        # Single source of truth for "is this job still alive?": rows
        # not present in squeue get marked DONE (if we'd seen them run)
        # or UNKNOWN (if we never did — the user can then look at .err
        # to figure out what happened, instead of seeing RUNNING forever).
        fasrc_jobs.reconcile_with_squeue(rows, ssh=STATE.ssh)
        return jsonify({"ok": True, "rows": rows})

    _parse_slurm_time = fasrc_jobs.parse_slurm_time

    # ---- submission -------------------------------------------------------

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

    def _require_confirm(form):
        """Shared confirm-token guard for the two FASRC submit endpoints.

        Returns ``None`` when the form carries the explicit-confirm
        token; otherwise returns a ``(flask response, 400)`` tuple the
        caller can return directly.
        """
        if str(form.get("confirm", "")).lower() in ("yes", "true", "1"):
            return None
        return jsonify({
            "ok": False,
            "error": (
                "missing explicit confirmation token. Refresh the page "
                "and click Submit again — the flow shows a dialog with "
                "the full payload before any FASRC submit."
            ),
        }), 400

    @app.route("/api/fasrc/submit", methods=["POST"])
    def api_fasrc_submit():
        """``run_pipeline.py`` submission (API path).

        Submits a ``scripts/run_pipeline.py`` job through the shared
        sbatch helper. The only registered run_pipeline step is
        ``synthetic_generate``; an optional ``step`` form field can name
        another registered step, otherwise it defaults to that.
        """
        if not STATE.ssh or not STATE.ssh.is_connected():
            return jsonify({"ok": False, "error": "not connected"}), 400
        confirm_err = _require_confirm(request.form)
        if confirm_err is not None:
            return confirm_err

        cfg = fasrc_config.load()
        f = request.form
        step_name = f.get("step") or f.get("preset") or "synthetic_generate"
        try:
            step = STEP_REGISTRY.get(step_name)
        except KeyError:
            # Unknown names fall back to the synthetic generator so a stale
            # frontend can't 404 the submit.
            step = STEP_REGISTRY.get("synthetic_generate")
            step_name = "synthetic_generate"

        # Resources from the form, falling back to the step's defaults
        # (so the legacy form fields keep working even when fields are
        # left blank).
        try:
            resources = StepResources.from_form(f, step.defaults)
        except ValueError as e:
            return jsonify({"ok": False, "error": str(e)}), 400

        # Training params — passed through to ``build_command``. We
        # validate the numerics here so a bad form field 400s instead of
        # blowing up inside the renderer.
        try:
            params = {
                "n_train":     int(f.get("n_train",    cfg.n_train)),
                "n_valid":     int(f.get("n_valid",    cfg.n_valid)),
                "image_size":  int(f.get("image_size", cfg.image_size)),
                "batch_size":  int(f.get("batch_size", cfg.batch_size)),
                "steps":       int(f.get("steps",      cfg.steps)),
                "extra_flags": f.get("extra_flags", "").strip(),
            }
        except (TypeError, ValueError) as e:
            return jsonify({"ok": False, "error": f"bad form field: {e}"}), 400
        params.update(resources.to_dict())

        label = f.get("label", "").strip() or (
            f"{step.label}: {params['steps']} steps on "
            f"{params['n_train']}+{params['n_valid']} fields"
        )
        built = step.build_sbatch_body(
            params=params, resources=resources, cfg=cfg, label=label,
        )
        slurm_id, payload = fasrc_jobs.submit_sbatch_script(
            STATE.ssh, cfg=cfg, built=built, label=label,
            params=params, step_id=step.step_id,
        )
        if slurm_id is None:
            return jsonify(payload), 500
        return jsonify(payload)

    # =========================================================================
    # HST pipeline (FASRC submissions for the 5-step HST → Euclid workflow)
    # =========================================================================

    @app.route("/api/fasrc/hst/status")
    def api_fasrc_hst_status():
        """Per-step: name, defaults, last-runtime median, on-disk status.

        Uses ``STATE.ssh`` to ``test`` for each artifact's existence; if
        SSH isn't connected we return only the static defaults so the UI
        can still render its forms.
        """
        cfg_loaded = fasrc_config.load()
        ssh_ok = bool(STATE.ssh and STATE.ssh.is_connected())

        steps_payload = []
        for step in STEP_REGISTRY.all():
            median = fasrc_jobs.median_runtime_for_step(step.step_id)
            history = fasrc_jobs.runtime_history_for_step(
                step.step_id, limit=5,
            )
            steps_payload.append({
                "step_id":     step.step_id,
                "label":       step.label,
                "description": step.description,
                "needs_gpu":   step.needs_gpu,
                "fixed_cpus":  step.fixed_cpus,
                "defaults":    step.defaults.to_dict(),
                "median_runtime_s":   median,
                "runtime_history_s":  history,
            })

        # Cheap probes for "does this artifact exist on FASRC?" — single
        # ``test -e`` per check, batched in one SSH round-trip. Keep
        # this list in sync with the ``produces`` map in fasrc.html
        # (the JS side maps each step_id to one of these keys).
        artifacts = {
            "tiles": None, "psf": None, "kernel": None,
            "records": None, "ckpt": None,
            # Round-trip pipeline artifacts (Chunk C3 + web wiring):
            #   euclid_sky      — sky-position catalog written by the
            #                     sky-download step (cutouts arrive in
            #                     subdirs whose names depend on the
            #                     requested size; gate on the catalog
            #                     instead, which exists as soon as the
            #                     position generation has run).
            #   roundtrip_records — LR-only TFRecord produced by the
            #                     stack/chop step. Single-file probe so
            #                     the UI can flip ✓ as soon as the
            #                     first shard lands.
            "euclid_sky": None, "roundtrip_records": None,
            # Per-page Euclid star-cutout pipeline:
            #   euclid_cutouts — VIS cutout subdir, written by the
            #                    download_euclid_cutouts step.
            #   euclid_psf     — VIS empirical ePSF, written by the
            #                    extract_euclid_psf (all-band) step.
            "euclid_cutouts": None, "euclid_psf": None,
            # Synthetic generation (/sky page).
            "synthetic_records": None,
        }
        if ssh_ok:
            paths = {
                "tiles":   f"{cfg_loaded.data_dir}/hst_hlsp/download_summary.json",
                "psf":     f"{cfg_loaded.data_dir}/hst_psf/F814W.fits",
                "kernel":  f"{cfg_loaded.data_dir}/hst_psf/diff_kernel_VIS.fits",
                "records": f"{cfg_loaded.data_dir}/images/records_v2_hst/clean_train.tfrecord",
                "ckpt":    f"{cfg_loaded.ckpt_dir}/checkpoint",
                "euclid_sky":
                    f"{cfg_loaded.data_dir}/euclid_sky/sky_positions.csv",
                "roundtrip_records":
                    f"{cfg_loaded.data_dir}/images/records_v2_euclid_roundtrip/dirty_train.tfrecord",
                "euclid_cutouts":
                    f"{cfg_loaded.data_dir}/euclid_stars/cutouts/VIS",
                "euclid_psf":
                    f"{cfg_loaded.data_dir}/euclid_psf/euclid_psf_VIS.fits",
                "synthetic_records":
                    f"{cfg_loaded.data_dir}/images/records_v2/clean_train.tfrecord",
            }
            probe = " && ".join(
                f"(test -e {shlex.quote(p)} && echo {k}=1 || echo {k}=0)"
                for k, p in paths.items()
            )
            try:
                rc, out, _err = STATE.ssh.run(probe, timeout=10)
                if rc == 0:
                    for line in out.splitlines():
                        line = line.strip()
                        if "=" in line:
                            k, v = line.split("=", 1)
                            if k in artifacts:
                                artifacts[k] = (v == "1")
            except Exception:
                pass

        return jsonify({
            "ssh_connected": ssh_ok,
            "steps":         steps_payload,
            "artifacts":     artifacts,
            "remote_paths": {
                "data_dir":    cfg_loaded.data_dir,
                "ckpt_dir":    cfg_loaded.ckpt_dir,
                "logs_dir":    f"logs/hst_pipeline",
            },
        })

    @app.route("/api/fasrc/hst/<step_id>/submit", methods=["POST"])
    def api_fasrc_hst_submit(step_id: str):
        """Generic submission for any HST-pipeline step.

        Two defences, in order of evaluation:

        1. ``confirm=yes`` token — proves the frontend dialog was
           shown (catches stale-JS tabs and most accidental submits).
        2. SSH connected (sanity check before the work starts).

        Any failure returns 400 and DOES NOT touch the SSH session,
        so no sbatch call, no script write, nothing reaches FASRC."""

        if not STATE.ssh or not STATE.ssh.is_connected():
            return jsonify({"ok": False, "error": "not connected"}), 400
        try:
            step = STEP_REGISTRY.get(step_id)
        except KeyError:
            return jsonify({"ok": False, "error": f"unknown step: {step_id}"}), 404

        cfg_loaded = fasrc_config.load()
        form = request.form.to_dict()

        confirm_err = _require_confirm(form)
        if confirm_err is not None:
            return confirm_err

        # A step with a locked CPU count renders the field as read-only
        # text, so the form may not carry ``n_cpus`` at all. Inject the
        # locked value before the strict parse (which requires it) — the
        # value is forced again below regardless of what the form sent.
        if step.fixed_cpus is not None:
            form["n_cpus"] = str(step.fixed_cpus)

        try:
            resources = StepResources.from_form_strict(form)
        except ValueError as e:
            return jsonify({"ok": False, "error": str(e)}), 400

        # If the step locks the CPU count, force it regardless of what
        # the form sent. Prevents the user from over-allocating cores
        # for a single-threaded job.
        if step.fixed_cpus is not None:
            resources.n_cpus = int(step.fixed_cpus)

        # All form values are passed as ``params``; the step picks out
        # what it needs (e.g. ``n_tiles``, ``hst_fraction``).
        label = (form.get("label", "").strip()
                 or f"HST pipeline · {step.label}")
        built = step.build_sbatch_body(
            params=form,
            resources=resources,
            cfg=cfg_loaded,
            label=label,
        )

        params_for_db = dict(form)
        params_for_db.update(resources.to_dict())
        params_for_db["step_id"] = step_id
        slurm_id, payload = fasrc_jobs.submit_sbatch_script(
            STATE.ssh, cfg=cfg_loaded, built=built, label=label,
            params=params_for_db, step_id=step_id,
        )
        if slurm_id is None:
            return jsonify(payload), 500
        return jsonify(payload)

    @app.route("/api/fasrc/refresh-accounting", methods=["POST"])
    def api_fasrc_refresh_accounting():
        """One-shot: re-pull sacct for every finalised job and re-record.

        Use after a change to how a post-mortem stat is computed (e.g. the
        CPU-utilisation fix) to backfill existing history rows."""
        if not STATE.ssh or not STATE.ssh.is_connected():
            return jsonify({"ok": False, "error": "not connected"}), 400
        return jsonify(fasrc_jobs.refresh_all_post_mortems(STATE.ssh))

    @app.route("/api/fasrc/hst/runtime-history/<step_id>")
    def api_fasrc_hst_runtime_history(step_id: str):
        return jsonify({
            "step_id":  step_id,
            "history":  fasrc_jobs.runtime_history_for_step(step_id, limit=10),
            "median":   fasrc_jobs.median_runtime_for_step(step_id),
        })

    @app.route("/api/fasrc/hst/<step_id>/history", methods=["GET", "POST"])
    def api_fasrc_hst_step_history(step_id: str):
        """Per-step run history + best-match prefill suggestion.

        Powers the "Previous runs" panel and the resources prefill under
        each step's form. The client posts the current task-params
        dict (everything in the form except resource fields and meta
        keys); we serialise it the same way the submitter does so the
        latest-match lookup uses string equality on ``params_json``.

        Response shape:

        ```json
        {
          "ok":       true,
          "step_id":  "extract_psf",
          "history":  [<row>, ...],          # newest first, all states
          "match":    <row> | null,          # latest exact-match row
          "task_params_json": "..."          # canonical serialisation
        }
        ```
        """
        try:
            STEP_REGISTRY.get(step_id)
        except KeyError:
            return jsonify({"ok": False, "error": f"unknown step: {step_id}"}), 404

        # Accept task params via form POST (UI flow) OR query string GET
        # (curl-friendly debugging). Strip the same meta keys the
        # submitter strips so the match lookup is symmetric.
        raw = request.form.to_dict() if request.method == "POST" else request.args.to_dict()
        task_params = {
            k: v for k, v in raw.items()
            if k not in ("partition", "n_cpus", "n_gpus", "memory",
                         "time_limit", "confirm", "label", "preset")
        }
        params_json = json.dumps(
            task_params, ensure_ascii=False,
            separators=(",", ":"), sort_keys=True,
        ) if task_params else ""

        history = fasrc_jobs.JOBLOG.history_for_step(step_id)
        match   = fasrc_jobs.JOBLOG.latest_match(step_id, params_json)
        return jsonify({
            "ok":               True,
            "step_id":          step_id,
            "history":          history,
            "match":            match,
            "task_params_json": params_json,
        })

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

    @app.route("/api/fasrc/current-submission")
    def api_fasrc_current_submission():
        """Return the user's most-recent live submission + its event-stream status.

        Looks at the JobDB for the latest job in ``PENDING`` or
        ``RUNNING`` state, reconciles its row against ``squeue`` so the
        elapsed/limit/node columns are fresh, and folds its ``.events``
        JSONL into a :class:`JobStatus` (stage, full stage history,
        step progress, warnings, errors).

        Response::

            { "ok": true, "current": null }   # no active job
            { "ok": true,
              "current": { "job": { ... DB row + squeue overrides ... },
                           "status": { stage, stages, step, warnings,
                                       errors, has_events, ... } } }

        Used by the FASRC page's "Current Submission" tab. Replaces the
        WDSR-specific ``/api/fasrc/training-status`` for the general job
        case; the training-status endpoint stays for the trainer's own
        live-metrics view.
        """
        if not STATE.ssh or not STATE.ssh.is_connected():
            return jsonify({"ok": False, "error": "not connected"}), 400

        # Reconcile DB rows against squeue first — without this, a job
        # that already finished still shows up as RUNNING in the DB and
        # the tab would lie. One squeue call per refresh; the same one
        # the Logs tab already makes.
        rc_q, out_q, _err_q = STATE.ssh.run(
            f"squeue -h -u $USER --format='{fasrc_jobs.SQUEUE_FMT}'",
            timeout=15,
        )
        squeue_rows: List[Dict[str, Any]] = []
        if rc_q == 0:
            squeue_rows = fasrc_jobs.parse_squeue(out_q)
            fasrc_jobs.reconcile_with_squeue(squeue_rows, ssh=STATE.ssh)

        # Pick the newest still-live row. ``list_recent`` orders by
        # submitted_at DESC, so the first matching row IS the newest.
        recent = fasrc_jobs.DB.list_recent(limit=10)
        current_row = next(
            (r for r in recent if r.get("state") in ("PENDING", "RUNNING")),
            None,
        )
        if current_row is None:
            return jsonify({"ok": True, "current": None})

        # Merge live squeue fields into the row so the UI sees the
        # current ``start_time`` (PENDING jobs only), ``reason`` (why
        # SLURM hasn't started it: Priority / Resources / …), updated
        # elapsed ``time``, and assigned ``nodes``. reconcile_with_squeue
        # only persists state + started_at, so these have to be merged
        # in at the response layer.
        jid = str(current_row.get("jobid", "")).strip()
        live = next((r for r in squeue_rows if r.get("jobid") == jid), None)
        if live is not None:
            for k in ("start_time", "reason", "nodes", "time", "time_limit"):
                v = live.get(k)
                if v is not None and v != "":
                    current_row[k] = v

        # Fold the live event stream into a JobStatus.
        fetcher = JobStatusFetcher(ssh=STATE.ssh)
        status  = fetcher.fetch(events_path=current_row.get("events_path"))
        return jsonify({
            "ok":      True,
            "current": {
                "job":    current_row,
                "status": status.to_dict(),
            },
        })

    @app.route("/api/fasrc/jobs/<jobid>/status")
    def api_fasrc_job_status(jobid: str):
        """Return the structured status for one job.

        Reads the job's ``.events`` JSONL stream (written on FASRC by
        :class:`euclid_polish.observability.Reporter`) and folds it into
        the :class:`JobStatus` shape the ``JobStatusCard`` widget polls
        for. Returns an empty status (``has_events=False``) when the
        job is queued / pre-Reporter / disconnected — never 500s on a
        missing file."""
        row = fasrc_jobs.DB.get(jobid)
        if not row:
            return jsonify({"ok": False, "error": "unknown jobid"}), 404
        fetcher = JobStatusFetcher(ssh=STATE.ssh)
        status  = fetcher.fetch(events_path=row.get("events_path"))
        return jsonify({
            "ok":          True,
            "jobid":       jobid,
            "state":       row.get("state"),
            "events_path": row.get("events_path"),
            "status":      status.to_dict(),
        })

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

        # 0. Reconcile DB state against the live queue *before* we read
        # rows out of sqlite. Without this, any job that finished
        # while the Logs tab wasn't open shows up as RUNNING forever,
        # and jobs that disappeared from squeue before ever starting
        # (sbatch rejected, queue purged, etc.) stay PENDING. One
        # extra cheap squeue call per Logs-tab load is well worth it.
        rc_q, out_q, _err_q = STATE.ssh.run(
            f"squeue -h -u $USER --format='{fasrc_jobs.SQUEUE_FMT}'",
            timeout=15,
        )
        if rc_q == 0:
            fasrc_jobs.reconcile_with_squeue(
                fasrc_jobs.parse_squeue(out_q), ssh=STATE.ssh,
            )

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

    @app.route("/api/fasrc/runs/ckpt-bundle.tar")
    def api_fasrc_runs_ckpt_bundle():
        """Stream a tar of the FASRC ckpt dir so the user can download
        the trained model after a job completes.

        We tar on the remote (one ssh + tar pipeline) and pipe bytes
        back through the ControlMaster — no temp file involved. Bundle
        size is bounded by ``max_to_keep=3`` in the trainer, so usually
        a few × 7 MB plus the training_log.
        """
        if not STATE.ssh or not STATE.ssh.is_connected():
            return jsonify({"ok": False, "error": "not connected"}), 400
        cfg = fasrc_config.load()
        ckpt_dir = cfg.ckpt_dir
        # Defensive: refuse if cfg.ckpt_dir points anywhere weird.
        if not ckpt_dir or ".." in ckpt_dir.split("/"):
            return jsonify({"ok": False, "error": "invalid ckpt_dir"}), 400
        parent = os.path.dirname(ckpt_dir.rstrip("/")) or "/"
        leaf   = os.path.basename(ckpt_dir.rstrip("/"))
        # ``tar -C parent leaf`` makes the tarball self-contained: it
        # unpacks into ``leaf/`` regardless of where the user extracts
        # it. ``--ignore-failed-read`` keeps going if a single ckpt file
        # is being rewritten while we tar.
        cmd = (
            f"tar -C {shlex.quote(parent)} -cf - "
            f"  --ignore-failed-read {shlex.quote(leaf)} 2>/dev/null"
        )
        rc, out, err = STATE.ssh.run(cmd, timeout=300, binary=True)
        if rc != 0 or not out:
            return jsonify({
                "ok": False,
                "error": f"tar failed: {err[:200] if err else 'no output'}",
            }), 500
        filename = f"{leaf}.tar"
        return Response(
            out, mimetype="application/x-tar",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Content-Length":      str(len(out)),
            },
        )

    @app.route("/api/fasrc/runs/training-plot.png")
    def api_fasrc_runs_training_plot():
        """PNG of the validation log restricted to one run's wall-time window.

        The trainer writes ``training_log.csv`` (append-only across all
        sessions sharing the same ckpt dir). We fetch the file over SSH,
        filter rows by ``[started_at, ended_at]``, render with
        :func:`plot_training_records`, and return the bytes.
        """
        if not STATE.ssh or not STATE.ssh.is_connected():
            return jsonify({"ok": False, "error": "not connected"}), 400
        try:
            started_at = float(request.args.get("started_at", "0"))
            ended_at   = float(request.args.get("ended_at",   "0"))
        except ValueError:
            return jsonify({"ok": False, "error": "bad started_at/ended_at"}), 400
        if ended_at <= 0:
            ended_at = time.time() + 1.0   # ongoing run — include up to now
        if started_at <= 0:
            return jsonify({"ok": False, "error": "missing started_at"}), 400

        cfg = fasrc_config.load()
        csv_path   = f"{cfg.ckpt_dir}/training_log.csv"
        jsonl_path = f"{cfg.ckpt_dir}/training_log.jsonl"

        # Fetch whichever log file exists. Cap to 50k lines (~20 MB) so
        # the SSH payload stays bounded for very long training histories.
        cmd = (
            f"{{ "
            f"  if [ -f {shlex.quote(csv_path)} ]; then "
            f"    head -n 50000 {shlex.quote(csv_path)} ; "
            f"  elif [ -f {shlex.quote(jsonl_path)} ]; then "
            f"    head -n 50000 {shlex.quote(jsonl_path)} ; "
            f"  fi ; "
            f"}}; exit 0"
        )
        rc, text, _err = STATE.ssh.run(cmd, timeout=30)
        if rc != 0 or not text.strip():
            return jsonify({"ok": False,
                            "error": "training log not found on remote"}), 404

        # Parse + filter by wall-time window.
        all_rows = fasrc_log_parser.parse_training_log(
            text, max_records=10_000_000,
        )
        rows = [r for r in all_rows
                if started_at <= r.get("wall_time", 0.0) <= ended_at]
        if not rows:
            return jsonify({"ok": False,
                            "error": f"no training-log rows in window "
                                     f"[{started_at:.0f}, {ended_at:.0f}]"}), 404

        # Render in-memory.
        tmp_png = os.path.join(Config.VIS_DIR, "tmp_training_plot.png")
        os.makedirs(Config.VIS_DIR, exist_ok=True)
        plot_training_records(
            rows, tmp_png,
            title_suffix=f"\n(this run only: {len(rows)} evals)",
        )
        with open(tmp_png, "rb") as fh:
            data = fh.read()
        return Response(data, mimetype="image/png",
                        headers={"Cache-Control": "no-cache"})

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

    # ---- parsed live status (.out + .err + training_log) -----------------
    #
    # The sidebar polls this every couple of seconds AND every page in the
    # app pulls it on initial render — without a cache that's a fresh
    # squeue + multi-file tail per poll, which is what makes the UI feel
    # sluggish. A 2-second TTL coalesces bursts and keeps live progress
    # visibly fresh (next poll arrives just after expiry).

    _TRAINING_STATUS_CACHE: Dict[str, Any] = {"at": 0.0, "resp": None}
    _TRAINING_STATUS_TTL_S: float = 2.0

    @app.route("/api/fasrc/training-status")
    def api_fasrc_training_status():
        """Single JSON dict the UI polls every few seconds.

        Identifies the currently running job (RUNNING state in squeue,
        cross-referenced against local sqlite), reads the tail of its
        ``.out`` / ``.err`` / ``training_log`` over SSH, and returns a
        parsed summary. Errors are reported in-band as
        ``{"ok": False, "error": ...}`` rather than raising — a 5xx
        here would just look like the dashboard "disconnecting" to
        the user, when really the SSH is fine and only one log read
        misbehaved.
        """
        if not STATE.ssh or not STATE.ssh.is_connected():
            return jsonify({"ok": False, "error": "not connected"}), 400

        # Serve from the cache if a recent (<2 s) response is available.
        now = time.monotonic()
        if (_TRAINING_STATUS_CACHE["resp"] is not None
                and now - _TRAINING_STATUS_CACHE["at"] < _TRAINING_STATUS_TTL_S):
            return _TRAINING_STATUS_CACHE["resp"]

        try:
            resp = _build_training_status()
        except Exception as e:
            traceback.print_exc()
            resp = jsonify({
                "ok":    False,
                "error": f"{type(e).__name__}: {e}",
            }), 200
        _TRAINING_STATUS_CACHE["at"]   = now
        _TRAINING_STATUS_CACHE["resp"] = resp
        return resp

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
        """One-shot rsync from remote ckpt dir → local mirror.

        Used by the Training tab's "Sync now" button AND the Logs
        tab's "Pull checkpoints" button. ``MIRROR.trigger`` runs
        synchronously, so the caller learns the final ``last_rc`` /
        ``last_error`` straight from the response without polling.
        """
        if not STATE.ssh or not STATE.ssh.is_connected():
            return jsonify({"ok": False, "error": "not connected"}), 400
        MIRROR.trigger()
        s = MIRROR.status
        return jsonify({
            "ok":          (s.last_rc == 0),
            "last_rc":     s.last_rc,
            "last_error":  s.last_error,
            "last_stdout": s.last_stdout,
            "remote_dir":  s.remote_dir,
            "local_dir":   s.local_dir,
            "last_run_at": s.last_run_at,
        })

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
