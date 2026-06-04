"""sky routes for the EuclidPolish web UI (extracted from app.py)."""
from __future__ import annotations

from euclid_polish.config import Config
from euclid_polish.web import fasrc_config
from euclid_polish.web.fasrc_fetcher import list_remote_dir
from euclid_polish.web.jobs import REGISTRY
from euclid_polish.web.remote import STATE
from flask import jsonify
from flask import render_template
from flask import request
from typing import Any
from typing import Dict
from typing import List
from euclid_polish.web.helpers.forms import _parse_asinh_scale
from euclid_polish.web.helpers.jobs_impl import _job_roundtrip_inspect


def register(app):

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
