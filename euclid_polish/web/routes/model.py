"""model routes for the EuclidPolish web UI (extracted from app.py)."""
from __future__ import annotations

import os
from typing import Any

from astropy.io import fits
from flask import jsonify, redirect, render_template, request

from euclid_polish.config import Config
from euclid_polish.web import job_config
from euclid_polish.web.helpers.forms import _parse_asinh_scale
from euclid_polish.web.helpers.jobs_impl import (
    _job_generate_reconstruct,
    _job_reconstruct_euclid_cutout,
)
from euclid_polish.web.helpers.status import (
    _checkpoints_status,
    _tfrecords_status,
)
from euclid_polish.web.jobs import REGISTRY


def _inference_gallery() -> dict[str, Any]:
    """Recent reconstruction PNGs + the persistent Euclid/synthetic inference
    runs (newest first). Shared by the /inference page render and the JSON
    endpoint the React console reads."""
    recon_pngs: list[dict[str, Any]] = []
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
            fits_rel = None
            stem = os.path.splitext(fname)[0]
            for suffix in ("_eye", "_solar"):
                if stem.endswith(suffix):
                    stem = stem[: -len(suffix)]
                    break
            fits_local = os.path.join(rdir, f"{stem}.fits")
            if os.path.isfile(fits_local):
                fits_rel = os.path.relpath(fits_local, Config.VIS_DIR)
            recon_pngs.append({"rel": rel, "name": fname, "mtime": mtime,
                               "fits_rel": fits_rel})
        recon_pngs.sort(key=lambda d: d["mtime"], reverse=True)

    def _runs(root: str, names: tuple[str, ...], label_from_sr: bool) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        if not os.path.isdir(root):
            return out
        for tag in os.listdir(root):
            d = os.path.join(root, tag)
            if not os.path.isdir(d):
                continue
            files = []
            for name in names:
                f = os.path.join(d, name)
                if os.path.isfile(f):
                    files.append({
                        "name": name,
                        "rel": os.path.relpath(f, Config.EUCLID_INFERENCE_DIR),
                        "size_kb": int(os.path.getsize(f) / 1024),
                    })
            if not files:
                continue
            label = tag
            sr_local = os.path.join(d, "SR.fits")
            if label_from_sr and os.path.isfile(sr_local):
                try:
                    hdr = fits.getheader(sr_local)
                    if hdr.get("RA") is not None and hdr.get("DEC") is not None:
                        label = (f"RA {float(hdr['RA']):.4f}, "
                                 f"Dec {float(hdr['DEC']):+.4f}")
                        if hdr.get("CSIZE") is not None:
                            label += f"  ({int(hdr['CSIZE'])} px)"
                except Exception:  # noqa: BLE001 — label is cosmetic
                    pass
            out.append({"tag": tag, "label": label, "files": files,
                        "mtime": max(os.path.getmtime(os.path.join(d, x["name"]))
                                     for x in files)})
        out.sort(key=lambda d: d["mtime"], reverse=True)
        return out

    euclid_runs = _runs(
        os.path.join(Config.EUCLID_INFERENCE_DIR, "cutouts"),
        ("original_stack.fits", "VIS.fits", "Y_E.fits", "J_E.fits", "H_E.fits", "SR.fits"),
        label_from_sr=True)
    synthetic_runs = _runs(
        os.path.join(Config.EUCLID_INFERENCE_DIR, "synthetic"),
        ("original_stack.fits", "SR.fits", "HR.fits"), label_from_sr=False)
    return {"recon_pngs": recon_pngs, "euclid_runs": euclid_runs,
            "synthetic_runs": synthetic_runs}


def register(app):

    # ---------------- Training (folded into /ensemble) ----------------
    @app.route("/training")
    def training_page():
        """Training is ensemble-only now — the /ensemble page owns TFRecord
        status, the ensemble_train step card, curves and member management."""
        return redirect("/ensemble", code=302)

    # ---------------- Inference page ----------------
    @app.route("/inference")
    def inference_page():
        gallery = _inference_gallery()
        return render_template(
            "inference.html",
            checkpoints=_checkpoints_status(),
            tfrecords=_tfrecords_status(),
            recon_pngs=gallery["recon_pngs"],
            euclid_runs=gallery["euclid_runs"],
            synthetic_runs=gallery["synthetic_runs"],
            default_num_res_blocks=Config.DEFAULT_NUM_RES_BLOCKS,
        )

    @app.route("/api/inference/recent.json")
    def api_inference_recent():
        """The reconstruction gallery + persistent inference runs as JSON for
        the React console (PNGs at /vis/<rel>, run FITS at /inference-files/<rel>)."""
        return jsonify(_inference_gallery())

    @app.route("/inference/generate-reconstruct", methods=["POST"])
    def inference_generate_reconstruct():
        # The model is THE ensemble (registry-active members, mean
        # prediction); HR size + asinh come from the universal /config tab.
        # Only n_pairs remains page-level.
        jc = job_config.load()
        asinh = _parse_asinh_scale(str(jc.asinh_scale))
        hr_size = jc.hr_image_size
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
        # Always pure-TNG generation (redshift realism, COSMOS skipped).
        job_id = REGISTRY.spawn(
            label=f"gen+reconstruct {n_pairs}×{hr_size}px (FASRC login-node gen)",
            target=lambda cap: _job_generate_reconstruct(
                cap, hr_size, n_pairs, asinh_scale=asinh,
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
        # The model is THE ensemble; asinh comes from the universal /config
        # tab. cutout_size stays page-level (a real-cutout knob).
        size = int(request.form.get("cutout_size", 512))
        if not (32 <= size <= 4096):
            return jsonify({"error": f"cutout_size={size} out of range [32, 4096]"}), 400
        asinh = _parse_asinh_scale(str(job_config.load().asinh_scale))
        # HTML checkbox: present in form data → ``"on"`` (or whatever
        # ``value=`` was set to); absent if unchecked. Parse truthy.
        show_all = request.form.get("show_all_bands", "").lower() in (
            "1", "on", "true", "yes",
        )
        job_id = REGISTRY.spawn(
            label=f"infer Euclid cutout @ ({ra:.4f}, {dec:+.4f})",
            target=lambda cap: _job_reconstruct_euclid_cutout(
                cap, ra, dec, size,
                asinh_scale=asinh, show_all_bands=show_all,
            ),
        )
        return jsonify({"job_id": job_id})
