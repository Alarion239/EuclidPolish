"""psfs routes for the EuclidPolish web UI (extracted from app.py)."""
from __future__ import annotations

from typing import Any

from flask import jsonify, render_template

from euclid_polish.config import Config
from euclid_polish.web import fasrc_config
from euclid_polish.web import fasrc_fetcher as _fasrc_fetcher
from euclid_polish.web.helpers.status import _psf_status


def register(app):

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

    @app.route("/api/euclid-psf/sync", methods=["POST"])
    def api_euclid_psf_sync():
        """Force a re-rsync of the four Euclid band ePSFs from FASRC.

        The /psfs page reads the local cache only (no rsync on load), so this
        is how you pull a freshly-extracted PSF down — ``force=True`` bypasses
        the fetcher's TTL cache. Per-band status; bands not yet on FASRC are
        reported failed individually without blocking the others."""
        cfg_loaded = fasrc_config.load()
        results: dict[str, dict[str, Any]] = {}
        any_ok = False
        for band in Config.BANDS:
            remote = f"{cfg_loaded.data_dir}/euclid_psf/{band.psf_fits_filename}"
            r = _fasrc_fetcher.fetch_one_file(
                remote, force=True,
                max_bytes=Config.WebFetch.MAX_PSF_PULL_BYTES)
            entry: dict[str, Any] = {
                "remote_path": remote, "ok": r.ok, "size_bytes": r.size_bytes,
            }
            if r.ok and r.local_path:
                any_ok = True
            else:
                entry["error"] = r.error
            results[band.name] = entry
        return jsonify({"ok": any_ok, "files": results})
