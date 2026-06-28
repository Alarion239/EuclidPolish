"""jobs_impl helpers for the EuclidPolish web UI (extracted from app.py)."""
from __future__ import annotations

import glob
import os
import shlex
import shutil
import textwrap
import uuid
from collections.abc import Callable
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from astropy.io import fits
from astropy.io import fits as _fits
from scipy import signal as scipy_signal

from euclid_polish.catalog.downloader import fetch_cutout_at
from euclid_polish.catalog.photometry import adu_per_s_to_electrons_factor
from euclid_polish.config import Config
from euclid_polish.eval.sr_provenance import stamp_sr_fits
from euclid_polish.image.tfio import read_images, tfrecord_path
from euclid_polish.psf.psf_library import load_all_band_psfs
from euclid_polish.sky.observation.observation_simulator import ObservationSimulator
from euclid_polish.training.inference import (
    load_model_from_checkpoint,
    plot_reconstruction,
    reconstruct,
    scaled_wcs_header,
)
from euclid_polish.web import fasrc_config
from euclid_polish.web import fasrc_fetcher as _fasrc_fetcher
from euclid_polish.web.fasrc_jobs import _conda_activate_snippet
from euclid_polish.web.helpers.status import _fasrc_psf_dir
from euclid_polish.web.remote import STATE


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
    # All-TNG mode (redshift realism, COSMOS catalog skipped).
    tng_flag = f" --sersic-density-arcmin2 0 --tng-density-arcmin2 {Config.DEFAULT_GAL_DENSITY_ARCMIN2}"
    _conda_block = _conda_activate_snippet(cfg.conda_env_path)
    return textwrap.dedent(f"""
        set -e
        export EUCLID_POLISH_DATA_DIR={q(cfg.data_dir)}
        mkdir -p {q(remote_tmp)}
        module purge 2>/dev/null || true
        __CONDA_BLOCK__
        cd {q(cfg.repo_path)}
        python -u scripts/run_pipeline.py \
          --ntrain 0 --nvalid {int(n_pairs)} --image-size {int(hr_image_size)} \
          --records-dir {q(remote_tmp)} --skip-train --gen-workers 1{tng_flag}
    """).replace("__CONDA_BLOCK__", _conda_block).strip()


def _job_generate_reconstruct(
    cap, checkpoint_dir: str, num_res_blocks: int,
    hr_image_size: int, n_pairs: int,
    asinh_scale: float | None = None,
) -> dict[str, Any]:
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
             f"generating {n_pairs} pair(s) @ {hr_image_size}px, TNG "
             "on FASRC login node")
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
        lr_records    = read_images(lr_path, num_images=10_000)
        hr_records    = (read_images(hr_path, num_images=10_000)
                         if os.path.exists(hr_path) else [])
        clean_records = (read_images(clean_path, num_images=10_000)
                         if os.path.exists(clean_path) else [])
        hr_by_idx    = {h.index: h for h in hr_records}
        clean_by_idx = {c.index: c for c in clean_records}

        # 4. Model — load once.
        scale = Config.DEFAULT_REBIN_FACTOR
        if not tf.train.latest_checkpoint(checkpoint_dir):
            raise FileNotFoundError(f"no checkpoint in {checkpoint_dir}")
        model = load_model_from_checkpoint(
            checkpoint_dir, scale, num_res_blocks,
            nchan_out=Config.NUM_HR_CHANNELS,   # nchan_in inferred from ckpt
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
                # The hr record is 4-band since the VIS+NISP-output
                # change — it can back the color panel when the clean
                # record is absent.
                if (hr_cube_for_color is None and raw.ndim == 3
                        and raw.shape[-1] == Config.NUM_LR_CHANNELS):
                    hr_cube_for_color = raw
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
            # TWO colored reconstructions per scene — the same LR | SR | HR
            # figure rendered once per color regime: "eye" (physical
            # blackbody-T colors, absolute) and "solar" (solar-balanced
            # adaptive windows). Both land in the gallery side by side.
            scene_pngs = []
            for regime, mode in (("eye", "eye"), ("solar", "calibrated")):
                out = os.path.join(out_dir, f"{stem}_{regime}.png")
                plot_reconstruction(lr_data, sr_data, hr_data=hr_data,
                                    output_path=out,
                                    lr_cube=lr_cube_for_color,
                                    hr_cube=hr_cube_for_color,
                                    asinh_scale=asinh_scale,
                                    predicted_dirty=predicted,
                                    residual=residual,
                                    rgb_mode=mode)
                scene_pngs.append(out)

            def _write_fits(path: str, data2d, label: str, *, lr_img=lr_img) -> None:
                if data2d is None:
                    return
                arr = np.ascontiguousarray(np.asarray(data2d, dtype=np.float32))
                band_note = "VIS"
                if arr.ndim == 3:
                    # 4-band cube (the VIS+NISP model) → NAXIS3 plane per
                    # band, same convention as original_stack.fits.
                    arr = np.ascontiguousarray(np.moveaxis(arr, -1, 0))
                    band_note = "4-band"
                hdu = _fits.PrimaryHDU(arr)
                if band_note == "4-band":
                    hdu.header["BANDS"] = (
                        ",".join(Config.LR_INPUT_BAND_NAMES),
                        "NAXIS3 plane order (band 0 = VIS)")
                hdu.header["OBJECT"] = (f"EuclidPolish {label} ({band_note})",
                                        "panel label")
                hdu.header["IDX"]    = (int(lr_img.index), "scene index")
                hdu.header["HRSIZE"] = (int(hr_image_size), "HR side px (0.05in/px)")
                hdu.header["CKPT"]   = (str(checkpoint_dir)[:60], "checkpoint dir")
                hdu.header["PSFSRC"] = ("FASRC", "ePSF pulled from FASRC (training PSF)")
                hdu.header["ASINH"]  = (float(asinh_scale or Config.STRETCH_SCALE_E),
                                        "asinh stretch knee used for the plot")
                hdu.header["BUNIT"]  = ("e-", "electrons (raw, sign preserved)")
                os.makedirs(os.path.dirname(path), exist_ok=True)
                hdu.writeto(path, overwrite=True)

            # SR FITS next to the PNG — the /inference gallery links to this.
            _write_fits(os.path.join(out_dir, stem + ".fits"), sr_data, "SR")

            # Inspectable per-scene FITS set, mirroring the real-Euclid cutout
            # outputs so each synthetic scene can be inspected/downloaded as
            # FITS on /inference: the 4-band LR cube + SR (+ HR clean, the
            # ground truth that only the synthetic path has).
            syn_dir = os.path.join(Config.EUCLID_INFERENCE_DIR, "synthetic", stem)
            os.makedirs(syn_dir, exist_ok=True)
            if lr_cube_for_color is not None:
                stack = np.moveaxis(np.ascontiguousarray(
                    np.asarray(lr_cube_for_color, dtype=np.float32)), -1, 0)
                sh = _fits.Header()
                sh["OBJECT"] = "EuclidPolish synthetic LR stack (electrons)"
                sh["BUNIT"]  = "electron"
                sh["BANDS"]  = (",".join(Config.LR_INPUT_BAND_NAMES),
                                "NAXIS3 plane order (band 0 = VIS)")
                sh["IDX"]    = (int(lr_img.index), "scene index")
                _fits.PrimaryHDU(stack, header=sh).writeto(
                    os.path.join(syn_dir, "original_stack.fits"),
                    overwrite=True, output_verify="silentfix")
            _write_fits(os.path.join(syn_dir, "SR.fits"), sr_data, "SR")
            _write_fits(os.path.join(syn_dir, "HR.fits"), hr_data, "HR clean")
            # Purge superseded / deprecated per-scene flat FITS + the old
            # single-regime PNG naming from previous runs.
            for _stale in (stem + "_lr.fits", stem + "_hr.fits",
                           stem + "_srforward.fits", stem + "_residual.fits",
                           stem + ".png"):
                try:
                    os.remove(os.path.join(out_dir, _stale))
                except OSError:
                    pass
            out_paths.extend(scene_pngs)
            cap.tick(k + 2, total, f"reconstructed scene {lr_img.index}")
            for out in scene_pngs:
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
    psf_dir: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
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

    ``sr_data`` may be the 4-band SR cube (the VIS+NISP model) — only
    its VIS plane (channel 0) is round-tripped here.
    """
    sr_data = np.asarray(sr_data)
    if sr_data.ndim == 3:
        sr_data = sr_data[..., 0]
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
    predicted = ObservationSimulator.sum_rebin(sr_hr, rebin_factor)
    # ``sum_rebin`` may trim a row/col if HR isn't divisible by the rebin
    # factor — match the LR shape by cropping to the smaller.
    h = min(predicted.shape[0], lr_vis.shape[0])
    w = min(predicted.shape[1], lr_vis.shape[1])
    predicted = predicted[:h, :w].astype(np.float32)
    residual = (lr_vis[:h, :w].astype(np.float32) - predicted).astype(np.float32)
    return predicted, residual


def reconstruct_cutout_at(
    model,
    ra: float,
    dec: float,
    cutout_size_vis_pixels: int,
    out_dir: str,
    *,
    asinh_scale: float | None = None,
    show_all_bands: bool = False,
    checkpoint_dir: str = "",
    render: bool = True,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    """Fetch a 4-band real Euclid cutout at ``(ra, dec)``, run SR, write outputs.

    This is the per-object body shared by the single-position WebUI job
    (:func:`_job_reconstruct_euclid_cutout`) and the batch catalog evaluator
    (``scripts/fasrc_eval_catalog.py``). It fetches each band, converts the
    archive's ADU s⁻¹ to electrons-over-the-stack via the per-band ``MAGZERO``
    (so the model sees the same scale it trained on), stacks to ``(H, W, 4)``,
    runs ``reconstruct``, forward-models the SR for a self-consistency
    residual, and writes ``original_stack.fits`` + ``SR.fits`` (and, when
    ``render``, ``eye.png`` + ``solar.png``) into ``out_dir``.

    ``out_dir`` is created if absent and used as-is — callers that want a
    single overwrite slot must wipe it themselves. ``progress`` is an optional
    ``(done, total, label)`` callback (e.g. wrapping a job's ``cap.tick``).

    Returns a dict with the output paths, per-band info, and the
    forward-model residual metrics.
    """
    scale = Config.DEFAULT_REBIN_FACTOR
    band_names = Config.LR_INPUT_BAND_NAMES
    total = len(band_names) + 3
    os.makedirs(out_dir, exist_ok=True)

    def _tick(done: int, label: str) -> None:
        if progress is not None:
            progress(done, total, label)

    # Fetch each band; per-band MAGZERO from each header drives the
    # per-band ADU/s → electrons conversion so the model sees the same
    # calibration scale the simulator uses.
    bands_data: dict[str, np.ndarray] = {}
    bands_info: dict[str, dict[str, Any]] = {}
    vis_header = None
    for k, band_name in enumerate(band_names):
        _tick(k, f"loading {band_name} cutout")
        band = Config.get_band(band_name)
        outf = os.path.join(out_dir, f"{band_name}.fits")
        if os.path.isfile(outf) and os.path.getsize(outf) > 0:
            print(f"  {band_name}: reusing cached cutout → {outf}")
        else:
            _tick(k, f"downloading {band_name} cutout")
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
        magzero = float(header.get("MAGZERO", band.sim_zeropoint_e))
        # Single source of truth for archive ADU/s → electrons-over-stack.
        adu_to_e = adu_per_s_to_electrons_factor(magzero, band)
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
    shapes = {n: bands_data[n].shape for n in band_names}
    base_shape = shapes["VIS"]
    if any(s != base_shape for s in shapes.values()):
        raise RuntimeError(
            f"per-band shapes disagree: {shapes}; expected all bands at "
            "the same VIS LR grid (0.10″/pix)."
        )

    lr_cube = np.stack([bands_data[n] for n in band_names], axis=-1)  # (H,W,4)
    _tick(len(band_names), "running model")
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
    # overlays the SR on-sky.
    stack = np.moveaxis(lr_cube, -1, 0).astype(np.float32)   # (4, H, W)
    stack_hdr = _clean_hdr(vis_header)
    stack_hdr["OBJECT"] = "Euclid LR stack (electrons)"
    stack_hdr["BUNIT"]  = "electron"
    stack_hdr["BANDS"]  = (",".join(band_names), "NAXIS3 plane order")
    stack_path = os.path.join(out_dir, "original_stack.fits")
    fits.PrimaryHDU(stack, header=stack_hdr).writeto(
        stack_path, overwrite=True, output_verify="silentfix")
    print(f"  ✓ saved stacked original → {stack_path}")

    # NOTE: there is deliberately NO forward-model self-consistency residual
    # for real cutouts. That comparison would push SR back through *our*
    # committed VIS PSF, but the true Euclid PSF is position-dependent and
    # unknown at an arbitrary (RA, Dec) — so a "predicted LR" and its residual
    # measure the PSF mismatch, not the reconstruction, and are misleading.
    # The quantitative signal for real targets is the Zoobot morphology
    # comparison; the only pixel-level check that survives an unknown PSF is
    # flux conservation (total counts are invariant under a *normalised* PSF),
    # computed directly below.
    _tick(len(band_names) + 1, "rendering")

    # Save SR with the 2× magnified VIS WCS so it overlays the stacked
    # original on-sky (0.05″/pix vs 0.10″/pix). A 4-band SR cube is
    # written one plane per band (NAXIS3), same convention as the
    # original_stack file.
    sr_fits_path = os.path.join(out_dir, "SR.fits")
    sr_hdr = (_clean_hdr(scaled_wcs_header(vis_header, scale))
              if vis_header is not None else fits.Header())
    sr_arr = np.asarray(sr_data, dtype=np.float32)
    sr_is_cube = sr_arr.ndim == 3
    if sr_is_cube:
        sr_arr = np.ascontiguousarray(np.moveaxis(sr_arr, -1, 0))
    sr_hdu = fits.PrimaryHDU(sr_arr, header=sr_hdr)
    sr_hdu.header["OBJECT"]   = ("Euclid SR (WDSR, 4-band)" if sr_is_cube
                                 else "Euclid SR (WDSR VIS)")
    if sr_is_cube:
        sr_hdu.header["BANDS"] = (",".join(band_names),
                                  "NAXIS3 plane order (band 0 = VIS)")
    sr_hdu.header["BUNIT"]    = "electron"
    sr_hdu.header["RA"]       = (float(ra),  "Input RA (deg)")
    sr_hdu.header["DEC"]      = (float(dec), "Input Dec (deg)")
    sr_hdu.header["CSIZE"]    = (int(cutout_size_vis_pixels),
                                 "Input VIS cutout size (px)")
    sr_hdu.header["CKPT"]     = (str(checkpoint_dir)[:60], "Checkpoint dir")
    sr_hdu.header["ASINH"]    = (float(asinh_scale or Config.STRETCH_SCALE_E),
                                 "asinh stretch knee used for plot")
    # Provenance: stamp the SR with its model lineage (best-effort, before write).
    try:
        stamp_sr_fits(
            sr_hdu.header, checkpoint_dir=str(checkpoint_dir),
            sr_fits_path=sr_fits_path,
            descriptors={"ra": float(ra), "dec": float(dec),
                         "cutout_size": int(cutout_size_vis_pixels),
                         "bands": ",".join(band_names)},
        )
    except Exception as exc:
        print(f"  [provenance] SR.fits not stamped: {exc}")
    sr_hdu.writeto(sr_fits_path, overwrite=True, output_verify="silentfix")
    print(f"  ✓ saved SR  → {sr_fits_path}")

    # TWO colored LR → SR renders — the same figure once per color regime:
    # "eye" (physical blackbody-T colors, absolute) and "solar"
    # (solar-balanced adaptive windows).
    png_paths: list[str] = []
    if render:
        for regime, mode in (("eye", "eye"), ("solar", "calibrated")):
            out_path = os.path.join(out_dir, f"{regime}.png")
            plot_reconstruction(lr_vis, sr_data, hr_data=None,
                                output_path=out_path, lr_cube=lr_cube,
                                asinh_scale=asinh_scale,
                                show_all_bands=show_all_bands,
                                rgb_mode=mode)
            png_paths.append(out_path)
            print(f"  ✓ {out_path}")
    _tick(total, "saved outputs")

    # Flux conservation — the one pixel-level sanity check that doesn't depend
    # on the (unknown, position-dependent) true PSF: a normalised PSF + sum-
    # rebin conserves total counts, so Σ(forward(SR)) ≡ Σ(SR). We therefore
    # compare Σ(SR_VIS) to Σ(LR_VIS) directly — ratio ≈ 1 means the
    # deconvolution neither invented nor destroyed flux.
    _sr = np.asarray(sr_data)
    sr_vis = _sr[..., 0] if _sr.ndim == 3 else _sr
    lr_sum = float(np.sum(lr_vis))
    sr_sum = float(np.sum(sr_vis))
    metrics = {
        "lr_total_e":            lr_sum,
        "sr_total_e":            sr_sum,
        "flux_ratio_sr_over_lr": (sr_sum / lr_sum) if lr_sum != 0 else None,
    }

    return {
        "out_dir":      out_dir,
        "png_paths":    png_paths,
        "sr_fits_path": sr_fits_path,
        "stack_fits_path": stack_path,
        "ra":           float(ra),
        "dec":          float(dec),
        "cutout_size":  int(cutout_size_vis_pixels),
        "bands":        bands_info,
        "metrics":      metrics,
    }


def _job_reconstruct_euclid_cutout(
    cap,
    ra: float,
    dec: float,
    checkpoint_dir: str,
    num_res_blocks: int,
    cutout_size_vis_pixels: int,
    asinh_scale: float | None = None,
    show_all_bands: bool = False,
) -> dict[str, Any]:
    """Download a 4-band Euclid cutout at one sky position, run SR, save PNG.

    Thin wrapper over :func:`reconstruct_cutout_at`: it loads the model, wipes
    the single ``Config.EUCLID_INFERENCE_DIR/cutouts/latest/`` overwrite slot
    (so each run *replaces* the previous record), runs the shared per-object
    body into it, then copies the two color renders to the gallery's fixed
    ``euclid_latest_{eye,solar}.png`` names. The input RA/Dec/size are
    preserved in the SR FITS header for provenance.
    """
    if not tf.train.latest_checkpoint(checkpoint_dir):
        raise FileNotFoundError(f"no checkpoint in {checkpoint_dir}")
    scale = Config.DEFAULT_REBIN_FACTOR
    model = load_model_from_checkpoint(
        checkpoint_dir, scale, num_res_blocks,
        nchan_out=Config.NUM_HR_CHANNELS,   # nchan_in inferred from ckpt
    )

    # Single overwrite slot: wipe every previous cutout record so each call
    # replaces the prior run rather than accumulating one directory per
    # position.
    cutouts_root = os.path.join(Config.EUCLID_INFERENCE_DIR, "cutouts")
    if os.path.isdir(cutouts_root):
        shutil.rmtree(cutouts_root)
    cache_dir = os.path.join(cutouts_root, "latest")

    res = reconstruct_cutout_at(
        model, ra, dec, cutout_size_vis_pixels, cache_dir,
        asinh_scale=asinh_scale, show_all_bands=show_all_bands,
        checkpoint_dir=checkpoint_dir,
        progress=lambda done, total, label: cap.tick(done, total, label),
    )

    out_dir = Config.VIS_RECONSTRUCTION_DIR
    os.makedirs(out_dir, exist_ok=True)
    # Drop stale Euclid-cutout renders (one used to be written per
    # position); leave the synthetic reconstruction PNGs alone.
    for stale in glob.glob(os.path.join(out_dir, "euclid_*.png")):
        try:
            os.remove(stale)
        except OSError:
            pass
    out_pngs = []
    for regime, src in zip(("eye", "solar"), res["png_paths"]):
        dst = os.path.join(out_dir, f"euclid_latest_{regime}.png")
        shutil.copyfile(src, dst)
        out_pngs.append(dst)

    return {
        "output_path":  out_pngs[0] if out_pngs else None,
        "output_paths": out_pngs,
        "cache_dir":    cache_dir,
        "sr_fits_path": res["sr_fits_path"],
        "ra":           ra,
        "dec":          dec,
        "cutout_size":  cutout_size_vis_pixels,
        "bands":        res["bands"],
        "flux_ratio":   res["metrics"]["flux_ratio_sr_over_lr"],
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
    asinh_scale: float | None = None,
    show_all_bands: bool = False,
) -> dict[str, Any]:
    """Inspect one real-Euclid round-trip cutout — and its round-trip
    reconstruction once a checkpoint exists.

    Pulls the bundled ``sky_<pos_id>.fits`` (the 4-band cutouts step 1 of
    the round-trip pipeline writes on FASRC) into the local cache,
    converts each band from archive e⁻/s to total electrons via its
    MAGZERO zeropoint factor (the same conversion the round-trip
    TFRecords the trainer saw now use), and renders the LR input. When
    the local checkpoint
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
    bands_data: dict[str, np.ndarray] = {}
    bands_info: dict[str, dict[str, Any]] = {}
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
            # Archive e⁻/s → total electrons over the stack via the band's
            # MAGZERO zeropoint factor — the SAME conversion as the
            # direct-cutout reconstruct and the round-trip TFRecord
            # generator (verify_star_photometry-validated). MAGZERO is
            # preserved per band in the sky bundle.
            band_hdr = hdul[band_name].header
            magzero = float(band_hdr.get("MAGZERO", band.sim_zeropoint_e))
            adu_to_e = adu_per_s_to_electrons_factor(magzero, band)
            data_e = (np.asarray(hdul[band_name].data, dtype=np.float32)
                      * adu_to_e)
            bands_data[band_name] = data_e
            bands_info[band_name] = {
                "shape":    list(data_e.shape),
                "magzero":  magzero,
                "adu_to_e": adu_to_e,
                "pix_mean": float(np.mean(data_e)),
                "pix_std":  float(np.std(data_e)),
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
            nchan_out=Config.NUM_HR_CHANNELS,   # nchan_in inferred from ckpt
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
