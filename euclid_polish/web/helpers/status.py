"""status helpers for the EuclidPolish web UI (extracted from app.py)."""
from __future__ import annotations

import os
from typing import Any

import tensorflow as tf
from astropy.io import fits

from euclid_polish.catalog.catalog_object import CatalogObject, summarize
from euclid_polish.config import Config
from euclid_polish.image.tfio import tfrecord_path
from euclid_polish.observability.training_log import TrainingLog
from euclid_polish.psf import PSF
from euclid_polish.psf.psf_library import psf_inventory
from euclid_polish.web import fasrc_config
from euclid_polish.web import fasrc_fetcher as _fasrc_fetcher
from euclid_polish.web.fasrc_fetcher import _local_path_for
from euclid_polish.web.helpers._const import _CUTOUT_FNAME_RE
from euclid_polish.web.helpers.paths import _safe_relpath
from euclid_polish.web.remote import STATE


def _fasrc_catalog_remote_path() -> str:
    """Remote path of the catalog the FASRC-side query writes."""
    cfg = fasrc_config.load()
    return f"{cfg.data_dir}/euclid_stars/{Config.CATALOG_FILE}"


def _fasrc_catalog_dir(force: bool = True) -> str | None:
    """Pull the FASRC ``stars.csv`` into the local cache; return the dir
    holding it (the ``<dir>/stars.csv`` the catalog reads), or None if the
    remote catalog isn't there / can't be fetched.

    The brightest-N query runs on the FASRC login node and writes the
    catalog to netscratch ``$DATA_DIR/euclid_stars``. The laptop UI must
    read *that* copy — never a stale local one — for the summary + plots,
    so we rsync it down on demand."""
    res = _fasrc_fetcher.fetch_one_file(_fasrc_catalog_remote_path(), force=force)
    if not res.ok or not res.local_path or not os.path.isfile(res.local_path):
        return None
    return os.path.dirname(res.local_path)


def _valid_4band_stars(force: bool = False):
    """Stars valid in ALL 4 bands at ONE common cutout size, from the FASRC
    ``stars.csv``. Returns ``(size, [star_id, ...])`` (ids sorted) for the
    size with the most such stars, or ``(None, [])`` when there's no catalog
    or none qualify. ``force`` re-pulls the catalog; navigation passes False
    so it reads the already-cached copy instead of rsync-ing per image."""
    cat_dir = _fasrc_catalog_dir(force=force)
    if cat_dir is None:
        return None, []
    objects = CatalogObject.read(os.path.join(cat_dir, Config.CATALOG_FILE))
    if not objects:
        return None, []
    band_names = [b.name for b in Config.BANDS]
    by_size: dict[int, list[int]] = {}
    for o in objects:
        # Sizes valid in EVERY band = intersection of each band's valid sizes;
        # any band with no valid size disqualifies the star.
        per_band: list[set] = []
        for bn in band_names:
            sizes = set(o.valid_sizes(bn))
            if not sizes:
                per_band = None
                break
            per_band.append(sizes)
        if per_band is None or o.id is None:
            continue
        for sz in set.intersection(*per_band):
            by_size.setdefault(sz, []).append(int(o.id))
    if not by_size:
        return None, []
    best = max(by_size, key=lambda sz: len(by_size[sz]))
    return best, sorted(by_size[best])


def _ensure_local_star_cutout(band: str, sid: int, size: int) -> str | None:
    """Local canonical path of star ``sid``'s ``band`` cutout, PERSISTENTLY
    saved under ``data/euclid_stars/cutouts/<band>/`` (the same layout the
    gallery + /inspect read). Pulled from FASRC once on first request;
    later views read the saved copy — no re-pull, no LRU eviction. Returns
    None when not cached and FASRC isn't reachable."""
    local_dir = Config.cutout_dir_for_band(
        band, root=os.path.join(Config.DEFAULT_OUTPUT_DIR, Config.CUTOUTS_SUBDIR))
    fname = f"star_{int(sid):04d}_{int(size)}.fits"
    local_path = os.path.join(local_dir, fname)
    if os.path.isfile(local_path):
        return local_path                       # already saved — no pull
    if not STATE.ssh or not STATE.ssh.is_connected():
        return None
    cfg = fasrc_config.load()
    remote = f"{cfg.data_dir}/euclid_stars/cutouts/{band}/{fname}"
    os.makedirs(local_dir, exist_ok=True)
    try:
        STATE.ssh.rsync_pull(remote, local_dir, timeout=120)
    except Exception:
        return None
    return local_path if os.path.isfile(local_path) else None


def _catalog_status() -> dict[str, Any]:
    cat_dir = _fasrc_catalog_dir()
    if cat_dir is None:
        return {"present": False}
    path = os.path.join(cat_dir, Config.CATALOG_FILE)
    if not os.path.exists(path):
        return {"present": False}
    summary = summarize(CatalogObject.read(path))
    # Show the *remote* path so the page makes clear this is the FASRC
    # (netscratch) catalog, not the local cache copy we render from.
    return {"present": True, "summary": summary,
            "path": _fasrc_catalog_remote_path()}


def _fasrc_psf_dir(force: bool = True) -> str | None:
    """Pull the FASRC Euclid ePSFs into the local cache; return the dir
    holding them, or None if none are on FASRC yet.

    The all-band extraction job writes ``euclid_psf_<band>.fits`` to
    netscratch ``$DATA_DIR/euclid_psf``, so the laptop UI must rsync those
    down and read them — not a stale local copy. The four band files share
    one remote dir, so they land in one local cache dir."""
    cfg = fasrc_config.load()
    local_dir: str | None = None
    for band in Config.BANDS:
        remote = f"{cfg.data_dir}/euclid_psf/{band.psf_fits_filename}"
        res = _fasrc_fetcher.fetch_one_file(
            remote, force=force,
            max_bytes=Config.WebFetch.MAX_PSF_PULL_BYTES)
        if res.ok and res.local_path and os.path.isfile(res.local_path):
            local_dir = os.path.dirname(res.local_path)
    return local_dir


def _cached_fasrc_psf_dir() -> str | None:
    """Local cache dir holding the FASRC ePSFs **without** triggering a fetch.

    Page renders read whatever was last synced (fast, no SSH round-trip); the
    explicit "Synchronise" button (``/api/euclid-psf/sync``) is the only thing
    that re-rsyncs. Returns None if nothing has been synced down yet."""
    cfg = fasrc_config.load()
    local_dir: str | None = None
    for band in Config.BANDS:
        local = _local_path_for(
            f"{cfg.data_dir}/euclid_psf/{band.psf_fits_filename}")
        if os.path.isfile(local):
            local_dir = os.path.dirname(local)
    return local_dir


def _psf_status() -> dict[str, Any]:
    # Read the local cache only — no rsync on page load. The Synchronise
    # button pulls fresh ePSFs from FASRC on demand.
    psf_dir = _cached_fasrc_psf_dir()
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


def _tfrecords_status(records_dir: str | None = None) -> dict[str, Any]:
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


def _checkpoints_status() -> dict[str, Any]:
    """Every checkpoint file on disk, recursing into sub-tracks.

    Walks the whole checkpoint dir (not just the top level) so the second
    save-best track under ``loss_best/`` is listed alongside the root
    (save-best-score / PSNR) track. Each entry carries the ``folder`` it
    lives in so the UI can group + show locations.
    """
    ckpt_dir = Config.DEFAULT_CHECKPOINT_DIR
    out: dict[str, Any] = {"dir": ckpt_dir, "files": []}
    # The standard 4-channel model lives in ``ckpt_dir``; the VIS-only
    # (1-channel) model in the sibling ``ckpt_dir-vis``. List both so the UI
    # shows whichever exist. Each entry's ``variant`` ("" / "vis") +
    # ``subdir`` (""/"loss_best") locate it unambiguously.
    roots = [("", ckpt_dir), ("vis", ckpt_dir.rstrip("/") + "-vis")]
    for variant, root in roots:
        if not os.path.isdir(root):
            continue
        for dirpath, _dirs, files in os.walk(root):
            subdir = os.path.relpath(dirpath, root)
            subdir = "" if subdir == "." else subdir
            for fname in sorted(files):
                full = os.path.join(dirpath, fname)
                if os.path.isfile(full):
                    rel = os.path.join(subdir, fname) if subdir else fname
                    out["files"].append({
                        "name":    fname,
                        "rel":     rel,                       # loss_best/ckpt-11.index
                        "variant": variant,                   # "" (4ch) or "vis"
                        "subdir":  subdir,                    # "" (root) or "loss_best"
                        "folder":  os.path.normpath(dirpath),  # ckpt/wdsr/loss_best
                        "size_mb": round(os.path.getsize(full) / 1e6, 1),
                    })
    # 4-channel first, then VIS-only; root track before sub-tracks; sorted.
    out["files"].sort(key=lambda f: (f["variant"], f["subdir"], f["name"]))
    return out


def _cutout_layout_status(output_dir: str = Config.DEFAULT_OUTPUT_DIR,
                          preview_n: int = 8) -> dict[str, Any]:
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


def _list_vis_pngs() -> list[dict[str, Any]]:
    """Recent PNGs under data/vis/, newest first.

    Each entry includes ``inspect_fits`` — a project-relative path to a
    same-stem ``.fits`` sibling if one exists. The visualization gallery
    uses this to route the thumbnail click to ``/inspect`` instead of
    just popping the raw PNG.
    """
    pngs: list[dict[str, Any]] = []
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


def _resolve_training_log(checkpoint_dir: str) -> str | None:
    """Pick the current ``training_log.csv`` or fall back to legacy
    ``training_log.jsonl`` so logs from runs before the CSV switch still
    plot. Returns the path that exists, or None if neither does."""
    for name in (TrainingLog.FILENAME, "training_log.jsonl"):
        p = os.path.join(checkpoint_dir, name)
        if os.path.exists(p):
            return p
    return None


def _ckpt_dir_for_kind(checkpoint_dir: str, kind: str | None,
                       vis_only: Any = False) -> str:
    """Resolve the inference checkpoint dir for the chosen save-best track.

    The trainer keeps two checkpoint sets: PSNR-best at ``checkpoint_dir``
    and combined-loss-best in the ``loss_best/`` subfolder. ``kind="loss"``
    selects the latter; anything else (incl. None) → the PSNR-best root.

    ``vis_only`` (truthy, incl. the form strings "1"/"on"/"true"/"yes")
    selects the VIS-only (1-channel) model, whose checkpoints live in the
    sibling ``<checkpoint_dir>-vis`` directory. The ``-vis`` suffix is applied
    to the *base* before the ``loss_best`` subfolder, so both tracks of the
    VIS-only model are reachable. The model itself self-describes its channel
    count on load, so only the path needs steering here.
    """
    truthy = str(vis_only).strip().lower() in ("1", "on", "true", "yes") \
        if not isinstance(vis_only, bool) else vis_only
    base = checkpoint_dir.rstrip("/") + "-vis" if truthy else checkpoint_dir
    if (kind or "").strip().lower() == "loss":
        return os.path.join(base, "loss_best")
    return base


def _record_count(name: str, records_dir: str | None = None) -> int | None:
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
