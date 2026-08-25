"""psfs routes for the EuclidPolish web UI (extracted from app.py)."""
from __future__ import annotations

import shlex
import textwrap
from typing import Any

from flask import jsonify, render_template, request

from euclid_polish.config import Config
from euclid_polish.web import fasrc_config
from euclid_polish.web import fasrc_fetcher as _fasrc_fetcher
from euclid_polish.web.fasrc_jobs import _conda_activate_snippet
from euclid_polish.web.helpers.fits_render import _psf_preview_payload
from euclid_polish.web.helpers.status import PSF_CLUSTERS_META, _psf_status
from euclid_polish.web.remote import STATE


def _clusters_meta_remote_cmd(cfg) -> str:
    """Login-node command that (re)dumps the VIS ePSF cluster metadata to
    ``<data_dir>/euclid_psf/euclid_psf_clusters.json`` on FASRC.

    astropy opens the multi-hundred-MB kernel stack LAZILY — only the HDU
    headers (RA/Dec centroid + NSTARS per cluster) are read, so this runs in
    seconds and the subsequent rsync moves kilobytes, not the FITS.
    """
    fits_remote = f"{cfg.data_dir}/euclid_psf/{Config.BAND_VIS.psf_fits_filename}"
    out_remote = f"{cfg.data_dir}/euclid_psf/{PSF_CLUSTERS_META}"
    py = textwrap.dedent("""\
        import json, os
        from astropy.io import fits
        src, out = os.environ["EP_FITS"], os.environ["EP_OUT"]
        clusters = []
        with fits.open(src) as hdul:
            for hdu in hdul[1:]:
                h = hdu.header
                if "RA" in h and "DEC" in h:
                    clusters.append({"ra": float(h["RA"]),
                                     "dec": float(h["DEC"]),
                                     "n_stars": int(h.get("NSTARS", 0))})
        tmp = out + ".part"
        with open(tmp, "w") as f:
            json.dump({"source": os.path.basename(src),
                       "n_clusters": len(clusters),
                       "clusters": clusters}, f)
        os.replace(tmp, out)
        print(f"{len(clusters)} clusters -> {out}")
        """)
    return (
        _conda_activate_snippet(cfg.conda_env_path)
        + f"\nEP_FITS={shlex.quote(fits_remote)} EP_OUT={shlex.quote(out_remote)}"
        + f" python - <<'PYEOF'\n{py}PYEOF\n"
    )


def _sync_clusters_meta(cfg) -> dict[str, Any]:
    """Regenerate the cluster-metadata JSON on FASRC and rsync it down.

    Returns a status dict (never raises): ``{ok, n_clusters?, error?}``.
    """
    if STATE.ssh is None or not STATE.ssh.is_connected():
        return {"ok": False, "error": "not connected to FASRC"}
    try:
        rc, out, err = STATE.ssh.run(_clusters_meta_remote_cmd(cfg), timeout=180)
    except Exception as e:  # noqa: BLE001 — surfaced to the UI, not fatal
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    if rc != 0:
        tail = (err.strip() or out.strip())[-500:]
        return {"ok": False, "error": f"remote header dump failed (rc={rc}): {tail}"}
    r = _fasrc_fetcher.fetch_one_file(
        f"{cfg.data_dir}/euclid_psf/{PSF_CLUSTERS_META}",
        force=True, max_bytes=16 * 1024 * 1024)
    if not r.ok or r.local_path is None:
        return {"ok": False, "error": r.error or "rsync of the metadata failed"}
    import json

    try:
        with open(r.local_path) as f:
            n = int(json.load(f).get("n_clusters", 0))
    except (OSError, ValueError):
        n = 0
    return {"ok": True, "n_clusters": n, "local_path": r.local_path}


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
        reported failed individually without blocking the others. Also
        refreshes the cluster-metadata JSON (best-effort) so the cluster map
        tracks the full FASRC extraction."""
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
        meta = _sync_clusters_meta(cfg_loaded)
        return jsonify({"ok": any_ok, "files": results, "clusters_meta": meta})

    @app.route("/api/euclid-psf/preview")
    def api_euclid_psf_preview():
        """Serve bounded PSF arrays for the React canvas renderer.

        Synchronisation remains an explicit user action. A page render cannot
        silently trigger SSH/rsync or confuse a FASRC outage with rendering.
        """
        return jsonify(_psf_preview_payload(request.args.get("band", "all")))

    @app.route("/api/euclid-psf/sync-meta", methods=["POST"])
    def api_euclid_psf_sync_meta():
        """Metadata-ONLY sync for the cluster map: dump the per-cluster
        centroids + star counts from the VIS ePSF headers on the FASRC login
        node (lazy astropy open — seconds), then rsync the kilobyte JSON down.
        The cluster plot renders from this JSON, so it shows EVERY cluster of
        the FASRC extraction without pulling the kernel stack."""
        return jsonify(_sync_clusters_meta(fasrc_config.load()))
