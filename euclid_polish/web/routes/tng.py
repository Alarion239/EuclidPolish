"""TNG50 SKIRT-atlas download page for the EuclidPolish web UI.

Hosts the ``download_tng_skirt`` FASRC step card (bulk-fetch the whole
IllustrisTNG TNG50-1 SKIRT atlas as dusty Euclid VIS+NISP FITS) and a small
"what's on FASRC" summary derived from the per-galaxy ``.done`` markers.
"""
from __future__ import annotations

from flask import render_template

from euclid_polish.config import Config
from euclid_polish.web import fasrc_config
from euclid_polish.web.fasrc_fetcher import list_remote_dir


def register(app):

    @app.route("/tng")
    def tng_page():
        cfg = fasrc_config.load()
        tng_dir = f"{cfg.data_dir}/{Config.Tng.SKIRT_SUBDIR}"
        # Completed galaxies = ``.done`` sentinels one level under tng_skirt.
        # Depth 2 so ``tng_skirt/<subhalo_id>/.done`` is reached; missing dir
        # degrades to an empty list (ok=True) rather than an error.
        ok, entries, err = list_remote_dir(
            tng_dir,
            glob_pattern=Config.Tng.DONE_MARKER,
            max_entries=2000,
            max_depth=2,
        )
        return render_template(
            "tng.html",
            tng_dir=tng_dir,
            n_done=(len(entries) if ok else None),
            list_err=(None if ok else err),
        )
