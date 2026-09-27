"""ePSF routes of Data › PSFs.

``GET /api/euclid-psf/inventory`` is local (the synchronised cache only);
the two syncs are background jobs (``kind="psf-sync"``, one at a time) that
record each band's outcome, so the inventory can tell "not synchronised yet"
from "FASRC has no empirical PSF for this band" (Gaussian fallback).
"""
from __future__ import annotations

import json
import shlex
import textwrap
import threading
from typing import Any

from flask import jsonify

from euclid_polish.config import Config
from euclid_polish.web import fasrc_config
from euclid_polish.web import fasrc_fetcher as _fasrc_fetcher
from euclid_polish.web.fasrc_gate import requires_fasrc
from euclid_polish.web.fasrc_jobs import _conda_activate_snippet
from euclid_polish.web.helpers.status import (
    PSF_CLUSTERS_META,
    psf_inventory_payload,
    remote_missing,
    write_psf_sync_status,
)
from euclid_polish.web.jobs import REGISTRY as JOB_REGISTRY
from euclid_polish.web.remote import STATE

SYNC_JOB_KIND = "psf-sync"
_SPAWN_LOCK = threading.Lock()


def _clusters_meta_remote_cmd(cfg) -> str:
    """Login-node command that (re)dumps the ePSF cluster metadata to
    ``<data_dir>/euclid_psf/euclid_psf_clusters.json`` on FASRC.

    astropy opens the multi-hundred-MB kernel stacks LAZILY — only the HDU
    headers are read (RA/Dec centroid + NSTARS from VIS, FWHM of every band
    file present), so this runs in seconds and the subsequent rsync moves
    kilobytes, not the FITS. ``fwhm_arcsec`` stays the VIS FWHM (the atlas
    layer reads it); ``fwhm_by_band`` has every band.
    """
    base = f"{cfg.data_dir}/euclid_psf"
    band_files = {band.name: f"{base}/{band.psf_fits_filename}" for band in Config.BANDS}
    out_remote = f"{base}/{PSF_CLUSTERS_META}"
    py = textwrap.dedent("""\
        import json, os
        from astropy.io import fits
        bands, out = json.loads(os.environ["EP_BANDS"]), os.environ["EP_OUT"]
        vis = os.environ["EP_VIS"]

        def fwhms(path):
            if not os.path.exists(path):
                return {}
            got = {}
            with fits.open(path, lazy_load_hdus=True) as hdul:
                for i, hdu in enumerate(hdul[1:]):
                    h = hdu.header
                    if "FWHM" in h:
                        got[i] = float(h["FWHM"])
            return got

        per_band = {name: fwhms(path) for name, path in bands.items()}
        clusters = []
        with fits.open(bands[vis], lazy_load_hdus=True) as hdul:
            for i, hdu in enumerate(hdul[1:]):
                h = hdu.header
                if "RA" in h and "DEC" in h:
                    by_band = {name: got.get(i) for name, got in per_band.items()}
                    clusters.append({"index": i + 1,
                                     "ra": float(h["RA"]),
                                     "dec": float(h["DEC"]),
                                     "n_stars": int(h.get("NSTARS", 0)),
                                     "fwhm_arcsec": by_band.get(vis),
                                     "fwhm_by_band": by_band})
        tmp = out + ".part"
        with open(tmp, "w") as f:
            json.dump({"source": os.path.basename(bands[vis]),
                       "n_clusters": len(clusters),
                       "clusters": clusters}, f)
        os.replace(tmp, out)
        print(f"{len(clusters)} clusters -> {out}")
        """)
    return (
        _conda_activate_snippet(cfg.conda_env_path)
        + f"\nEP_BANDS={shlex.quote(json.dumps(band_files))}"
        + f" EP_VIS={shlex.quote(Config.BAND_VIS.name)} EP_OUT={shlex.quote(out_remote)}"
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
    try:
        with open(r.local_path) as f:
            n = int(json.load(f).get("n_clusters", 0))
    except (OSError, ValueError):
        n = 0
    return {"ok": True, "n_clusters": n, "local_path": r.local_path}


def _job_sync_epsfs(cap) -> dict[str, Any]:
    """Force a re-rsync of the four band ePSFs + the cluster metadata."""
    cfg = fasrc_config.load()
    results: dict[str, dict[str, Any]] = {}
    total = len(Config.BANDS) + 1
    for position, band in enumerate(Config.BANDS):
        cap.tick(position, total, f"pulling the {band.name} ePSF")
        remote = f"{cfg.data_dir}/euclid_psf/{band.psf_fits_filename}"
        r = _fasrc_fetcher.fetch_one_file(
            remote, force=True, max_bytes=Config.WebFetch.MAX_PSF_PULL_BYTES)
        entry: dict[str, Any] = {"remote_path": remote, "ok": bool(r.ok and r.local_path),
                                 "size_bytes": r.size_bytes}
        if not entry["ok"]:
            entry["error"] = r.error
            entry["missing_remote"] = remote_missing(r.error)
        cap.write(f"{band.name}: " + (f"{(r.size_bytes or 0) / 1e6:.1f} MB\n" if entry["ok"]
                                      else f"not pulled — {r.error}\n"))
        results[band.name] = entry
    cap.tick(total - 1, total, "cluster metadata")
    meta = _sync_clusters_meta(cfg)
    cap.write(f"cluster metadata: {meta.get('n_clusters', 0)} clusters\n" if meta.get("ok")
              else f"cluster metadata: {meta.get('error')}\n")
    write_psf_sync_status(results, meta)
    cap.tick(total, total, "done")
    any_ok = any(entry["ok"] for entry in results.values())
    if not any_ok and not any(entry.get("missing_remote") for entry in results.values()):
        raise RuntimeError("no ePSF was pulled: " + "; ".join(
            f"{band}: {entry.get('error')}" for band, entry in results.items())[:600])
    return {"ok": any_ok, "files": results, "clusters_meta": meta}


def _job_sync_meta(cap) -> dict[str, Any]:
    cap.tick(0, 1, "dumping the cluster headers on FASRC")
    meta = _sync_clusters_meta(fasrc_config.load())
    write_psf_sync_status({}, meta)
    cap.tick(1, 1, "done")
    if not meta.get("ok"):
        raise RuntimeError(meta.get("error") or "metadata sync failed")
    return meta


def _running_sync() -> str | None:
    for job in JOB_REGISTRY.list(summary=True):
        if job.get("kind") == SYNC_JOB_KIND and job.get("status") == "running":
            return str(job["job_id"])
    return None


def _spawn(label: str, target) -> dict[str, Any]:
    with _SPAWN_LOCK:
        running = _running_sync()
        if running:
            return {"ok": True, "job_id": running, "already_running": True}
        return {"ok": True, "job_id": JOB_REGISTRY.spawn(label, target, kind=SYNC_JOB_KIND)}


def register(app):

    @app.route("/api/euclid-psf/inventory")
    def api_euclid_psf_inventory():
        """Per-band ePSF state (``empirical`` / ``no_empirical`` /
        ``not_cached``, see ``status.psf_inventory_payload``) and the cluster
        table (RA/Dec, star count, per-band FWHM). Local cache only."""
        return jsonify(psf_inventory_payload())

    @app.route("/api/euclid-psf/sync", methods=["POST"])
    @requires_fasrc
    def api_euclid_psf_sync():
        """Force a re-rsync of the four Euclid band ePSFs from FASRC (a job).

        Data › PSFs reads the local cache only (no rsync on load), so this is
        how a freshly-extracted PSF comes down — ``force=True`` bypasses the
        fetcher's TTL cache, with the larger ePSF pull cap. Bands not on
        FASRC are reported individually (``missing_remote``) without blocking
        the others; the cluster metadata JSON is refreshed too. Job result:
        ``{ok, files: {band: {ok, remote_path, size_bytes, error?,
        missing_remote?}}, clusters_meta}``."""
        return jsonify(_spawn("PSFs: sync ePSFs from FASRC", _job_sync_epsfs))

    @app.route("/api/euclid-psf/sync-meta", methods=["POST"])
    @requires_fasrc
    def api_euclid_psf_sync_meta():
        """Metadata-ONLY sync (a job): dump the per-cluster centroids, star
        counts and per-band FWHM from the ePSF headers on the FASRC login
        node (lazy astropy open — seconds), then rsync the kilobyte JSON
        down. Job result ``{ok, n_clusters, local_path}``."""
        return jsonify(_spawn("PSFs: sync cluster metadata", _job_sync_meta))
