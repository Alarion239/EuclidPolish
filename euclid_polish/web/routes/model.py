"""model routes for the EuclidPolish web UI (extracted from app.py)."""
from __future__ import annotations

from astropy.io import fits
from euclid_polish.config import Config
from euclid_polish.web.jobs import REGISTRY
from flask import jsonify
from flask import render_template
from flask import request
from typing import Any
from typing import Dict
from typing import Optional
import os
from euclid_polish.web.helpers.forms import _parse_asinh_scale
from euclid_polish.web.helpers.jobs_impl import _job_generate_reconstruct, _job_reconstruct_euclid_cutout
from euclid_polish.web.helpers.status import _checkpoints_status, _ckpt_dir_for_kind, _tfrecords_status


def register(app):

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

    @app.route("/inference/generate-reconstruct", methods=["POST"])
    def inference_generate_reconstruct():
        ckpt_dir = _ckpt_dir_for_kind(
            request.form.get("checkpoint_dir", Config.DEFAULT_CHECKPOINT_DIR),
            request.form.get("ckpt_kind"))
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
        ckpt_dir = _ckpt_dir_for_kind(
            request.form.get("checkpoint_dir", Config.DEFAULT_CHECKPOINT_DIR),
            request.form.get("ckpt_kind"))
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
