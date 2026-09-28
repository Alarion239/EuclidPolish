"""status helpers for the EuclidPolish web UI (extracted from app.py)."""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, cast

from astropy.io import fits

from euclid_polish.catalog.catalog_object import CatalogObject, summarize
from euclid_polish.config import Config
from euclid_polish.ensemble import default_ensemble_dir
from euclid_polish.ensemble_registry import active_member_dirs
from euclid_polish.image.tfio import tfrecord_path
from euclid_polish.psf import PSF
from euclid_polish.psf.psf_library import psf_inventory
from euclid_polish.web import fasrc_config
from euclid_polish.web import fasrc_fetcher as _fasrc_fetcher
from euclid_polish.web.fasrc_fetcher import _local_path_for
from euclid_polish.web.helpers import sky_records
from euclid_polish.web.helpers.paths import _safe_relpath, _sky_records_local_dir
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
    so we rsync it down on demand.

    A non-forced read (navigation) falls back to the last synchronised copy
    when the pull fails — FASRC offline, or the cache older than the fetch
    TTL with the pull refused: a stale mirror beats "no catalogue". A forced
    read (an explicit refresh) reports the failure instead."""
    res = _fasrc_fetcher.fetch_one_file(_fasrc_catalog_remote_path(), force=force)
    if not res.ok or not res.local_path or not os.path.isfile(res.local_path):
        return None if force else _cached_fasrc_catalog_dir()
    return os.path.dirname(res.local_path)


def _cached_fasrc_catalog_dir() -> str | None:
    """Directory of the already-synchronised stellar catalog, if present.

    Presentation renders are cache-only: opening a figure page must not turn
    into an implicit rsync or require a live FASRC connection.
    """
    local = _local_path_for(_fasrc_catalog_remote_path())
    return os.path.dirname(local) if os.path.isfile(local) else None


def catalog_cache_info() -> dict[str, Any]:
    """Freshness of the synchronised FASRC ``stars.csv`` mirror (no SSH):
    ``{present, path (remote), local_path, size_bytes, mtime, age_s}`` — the
    mtime is the time of the last pull (the fetcher stamps it)."""
    remote = _fasrc_catalog_remote_path()
    local = _local_path_for(remote)
    try:
        st = os.stat(local)
    except OSError:
        return {"present": False, "path": remote, "local_path": None, "size_bytes": None,
                "mtime": None, "age_s": None}
    return {"present": True, "path": remote, "local_path": local, "size_bytes": int(st.st_size),
            "mtime": float(st.st_mtime), "age_s": max(0.0, time.time() - float(st.st_mtime))}


_CATALOG_LOCK = threading.Lock()
_CATALOG_MEMO: dict[str, tuple[tuple[int, int], list[CatalogObject]]] = {}
_VALID4_MEMO: dict[str, tuple[tuple[int, int], tuple[int | None, list[CatalogObject]]]] = {}


def _catalog_key(path: str) -> tuple[int, int] | None:
    try:
        st = os.stat(path)
    except OSError:
        return None
    return int(st.st_size), int(st.st_mtime_ns)


def read_catalog_objects(path: str) -> list[CatalogObject]:
    """``CatalogObject.read(path)`` memoised per file state (the 43k-row
    mirror takes ~1.3 s to parse; the cutouts viewer asks per cube)."""
    key = _catalog_key(path)
    if key is None:
        return []
    real = os.path.realpath(path)
    with _CATALOG_LOCK:
        hit = _CATALOG_MEMO.get(real)
        if hit is not None and hit[0] == key:
            return hit[1]
    objects = CatalogObject.read(real)
    with _CATALOG_LOCK:
        _CATALOG_MEMO.clear()                   # one catalogue at a time
        _CATALOG_MEMO[real] = (key, objects)
    return objects


def _valid_4band_stars(force: bool = False):
    """Stars valid in ALL 4 bands at ONE common cutout size, from the FASRC
    ``stars.csv``. Returns ``(size, [star_id, ...])`` (ids sorted) for the
    size with the most such stars, or ``(None, [])`` when there's no catalog
    or none qualify. ``force`` re-pulls the catalog; navigation passes False
    so it reads the already-cached copy instead of rsync-ing per image."""
    size, objects = _valid_4band_star_objects(force=force)
    return size, [int(o.id) for o in objects if o.id is not None]


def _valid_4band_star_objects(force: bool = False):
    """:func:`_valid_4band_stars` with the catalogue rows: ``(size,
    [CatalogObject, ...])`` sorted by id — so callers also get each star's
    ``ra``/``dec`` without re-reading ``stars.csv``."""
    return _valid_4band_star_objects_at(_fasrc_catalog_dir(force=force))


def _cached_valid_4band_star_objects():
    """:func:`_valid_4band_star_objects` over the synchronised mirror only —
    no SSH, no rsync (the cutouts navigator and its totals; works offline)."""
    return _valid_4band_star_objects_at(_cached_fasrc_catalog_dir())


def _cached_valid_4band_stars():
    """:func:`_valid_4band_stars` over the synchronised mirror only."""
    size, objects = _cached_valid_4band_star_objects()
    return size, [int(o.id) for o in objects if o.id is not None]


def _valid_4band_star_objects_at(cat_dir: str | None):
    if cat_dir is None:
        return None, []
    path = os.path.join(cat_dir, Config.CATALOG_FILE)
    key = _catalog_key(path)
    if key is None:
        return None, []
    real = os.path.realpath(path)
    with _CATALOG_LOCK:
        hit = _VALID4_MEMO.get(real)
        if hit is not None and hit[0] == key:
            return hit[1]
    result = _valid_4band_select(read_catalog_objects(path))
    with _CATALOG_LOCK:
        _VALID4_MEMO.clear()
        _VALID4_MEMO[real] = (key, result)
    return result


def _valid_4band_select(objects: list[CatalogObject]):
    if not objects:
        return None, []
    band_names = [b.name for b in Config.BANDS]
    by_size: dict[int, list[CatalogObject]] = {}
    for o in objects:
        # Sizes valid in EVERY band = intersection of each band's valid sizes;
        # any band with no valid size disqualifies the star.
        per_band: list[set[int]] | None = []
        for bn in band_names:
            sizes = set(o.valid_sizes(bn))
            if not sizes:
                per_band = None
                break
            per_band.append(sizes)
        if per_band is None or o.id is None:
            continue
        for sz in set.intersection(*per_band):
            by_size.setdefault(sz, []).append(o)
    if not by_size:
        return None, []
    best = max(by_size, key=lambda sz: len(by_size[sz]))
    return best, sorted(by_size[best], key=lambda o: int(o.id))


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


def _catalog_status_at(cat_dir: str | None) -> dict[str, Any]:
    """Summary of the ``stars.csv`` in ``cat_dir`` (``present: False`` if none)."""
    if cat_dir is None:
        return {"present": False}
    path = os.path.join(cat_dir, Config.CATALOG_FILE)
    if not os.path.exists(path):
        return {"present": False}
    summary = summarize(read_catalog_objects(path))
    # Show the *remote* path so the page makes clear this is the FASRC
    # (netscratch) catalog, not the local cache copy we render from.
    return {"present": True, "summary": summary,
            "path": _fasrc_catalog_remote_path()}


def _catalog_status() -> dict[str, Any]:
    """Re-pull the FASRC catalogue (forced rsync), then summarise it."""
    return _catalog_status_at(_fasrc_catalog_dir())


def _cached_catalog_status() -> dict[str, Any]:
    """Summary of the already-synchronised catalogue — no SSH, no rsync
    (``GET /api/status``); ``POST /api/status/refresh-catalog`` re-pulls."""
    return {**_catalog_status_at(_cached_fasrc_catalog_dir()), "cached": True}


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


#: Cluster-metadata sidecar living next to the ePSF FITS on FASRC — per
#: cluster centroid RA/Dec + star count, dumped from the VIS ePSF headers.
#: Kilobytes, so the /psfs cluster map can show EVERY FASRC cluster without
#: pulling the multi-hundred-MB kernel stack down.
PSF_CLUSTERS_META = "euclid_psf_clusters.json"


def _cached_psf_clusters_json() -> str | None:
    """Local cached copy of the FASRC cluster-metadata JSON, or ``None`` when
    it has not been synced yet. Read-only — no rsync on page load."""
    cfg = fasrc_config.load()
    local = _local_path_for(f"{cfg.data_dir}/euclid_psf/{PSF_CLUSTERS_META}")
    return local if os.path.isfile(local) else None


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


def psf_sync_status_path() -> str:
    """Where the last ePSF sync records each band's outcome (in the cache
    root: evicting it only loses the "no empirical PSF" distinction)."""
    return os.path.join(Config.FASRC_CACHE_DIR, "euclid_psf_sync.json")


def read_psf_sync_status() -> dict[str, Any] | None:
    try:
        with open(psf_sync_status_path()) as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def write_psf_sync_status(bands: dict[str, dict[str, Any]],
                          clusters_meta: dict[str, Any] | None = None) -> dict[str, Any]:
    """Merge one sync's per-band outcomes into the status file."""
    previous = read_psf_sync_status() or {}
    merged = dict(previous.get("bands") or {})
    now = time.time()
    for band, outcome in bands.items():
        merged[band] = {**outcome, "checked_at": now}
    payload = {"checked_at": now, "bands": merged,
               "clusters_meta": clusters_meta if clusters_meta is not None
               else previous.get("clusters_meta")}
    path = psf_sync_status_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w") as handle:
        json.dump(payload, handle)
    os.replace(tmp, path)
    return payload


def remote_missing(error: str | None) -> bool:
    """A fetch error that means "the file does not exist on FASRC"."""
    text = (error or "").lower()
    return "no such file" in text or "not found on remote" in text


def _psf_band_path(band) -> str:
    cfg = fasrc_config.load()
    return _local_path_for(f"{cfg.data_dir}/euclid_psf/{band.psf_fits_filename}")


_PSF_HEADER_MEMO: dict[str, tuple[tuple[int, int], dict[str, Any]]] = {}


def _header_float(header, *keys: str) -> float | None:
    for key in keys:
        if key in header:
            try:
                value = float(cast(Any, header[key]))
            except (TypeError, ValueError):
                continue
            if value == value:                  # not NaN
                return value
    return None


def psf_file_summary(path: str) -> dict[str, Any]:
    """Headers-only summary of one cached ePSF file (memoised per file
    state): ``{n_psf, shape, pixel_scale, fwhm_arcsec, clusters: [{index,
    ra, dec, n_stars, fwhm_arcsec}]}`` — HDU0 is the mean kernel, HDU1…K the
    spatial clusters. Pixels are never read."""
    key = _catalog_key(path)
    real = os.path.realpath(path)
    if key is not None:
        hit = _PSF_HEADER_MEMO.get(real)
        if hit is not None and hit[0] == key:
            return hit[1]
    with fits.open(real, memmap=True, lazy_load_hdus=True) as hdul:
        primary = hdul[0].header
        shape = [int(primary.get("NAXIS2", 0) or 0), int(primary.get("NAXIS1", 0) or 0)]
        clusters = []
        for index, hdu in enumerate(hdul[1:]):
            header = hdu.header
            if int(header.get("NAXIS", 0) or 0) < 2:
                continue
            n_stars = header.get("NSTARS")
            clusters.append({
                "index": index + 1,
                "ra": _header_float(header, "RA"),
                "dec": _header_float(header, "DEC"),
                "n_stars": int(cast(Any, n_stars)) if n_stars is not None else None,
                "fwhm_arcsec": _header_float(header, "FWHM"),
            })
        n_psf = primary.get("NPSF")
        summary = {
            "n_psf": int(cast(Any, n_psf)) if n_psf is not None else max(1, len(clusters)),
            "shape": shape,
            "pixel_scale": _header_float(primary, "PXSCALE", "PIXSCALE"),
            "fwhm_arcsec": _header_float(primary, "FWHM"),
            "clusters": clusters,
        }
    if key is not None:
        _PSF_HEADER_MEMO[real] = (key, summary)
    return summary


def psf_inventory_payload() -> dict[str, Any]:
    """The ePSF inventory of Synthetic › PSF (local cache only — no SSH).

    Per band ``state``:

    * ``empirical`` — the FASRC ePSF is synchronised locally (``path``,
      ``synced_at``, header summary);
    * ``no_empirical`` — the last sync found no ePSF on FASRC for this band:
      generation uses the Gaussian fallback (``fwhm`` from the config);
    * ``not_cached`` — not synchronised yet (or the last sync failed for
      another reason, ``error``): FASRC may well have one.

    ``clusters``: one row per spatial cluster (from the synced metadata JSON,
    else the cached VIS headers) with RA/Dec, star count and the per-band
    FWHM wherever a cached band file carries it."""
    status = read_psf_sync_status() or {}
    sync_bands = status.get("bands") or {}
    bands = []
    summaries: dict[str, dict[str, Any]] = {}
    for band in Config.BANDS:
        path = _psf_band_path(band)
        last = sync_bands.get(band.name) or None
        item: dict[str, Any] = {
            "name": band.name,
            "fwhm": band.psf_fwhm_arcsec,
            "oversampling": band.epsf_oversampling,
            "epsf_pixel_scale": band.epsf_pixel_scale_arcsec,
            "last_sync": last,
        }
        try:
            st = os.stat(path)
        except OSError:
            st = None
        if st is not None:
            item.update(state="empirical", empirical=True, path=_safe_relpath(path) or path,
                        size_bytes=int(st.st_size), synced_at=float(st.st_mtime))
            try:
                summary = psf_file_summary(path)
                summaries[band.name] = summary
                item.update(n_psf=summary["n_psf"], shape=summary["shape"],
                            pixel_scale=summary["pixel_scale"],
                            measured_fwhm=summary["fwhm_arcsec"])
            except Exception as exc:  # noqa: BLE001 - a broken file is reported, not raised
                item["error"] = f"{type(exc).__name__}: {exc}"
        elif last and not last.get("ok") and remote_missing(last.get("error")):
            item.update(state="no_empirical", empirical=False)
        else:
            item.update(state="not_cached", empirical=False,
                        error=(last or {}).get("error") if last and not last.get("ok") else None)
        bands.append(item)

    clusters: list[dict[str, Any]] = []
    meta_path = _cached_psf_clusters_json()
    meta_rows: list[dict[str, Any]] = []
    if meta_path:
        try:
            with open(meta_path) as handle:
                raw = json.load(handle).get("clusters", [])
            meta_rows = [row for row in raw if isinstance(row, dict)]
        except (OSError, ValueError, AttributeError):
            meta_rows = []
    vis = summaries.get(Config.BAND_VIS.name, {}).get("clusters", [])
    base = meta_rows or vis
    for position, row in enumerate(base):
        index = int(row.get("index") or position + 1)
        fwhm_by_band: dict[str, float | None] = {}
        for name, summary in summaries.items():
            match = next((c for c in summary["clusters"] if c["index"] == index), None)
            fwhm_by_band[name] = match.get("fwhm_arcsec") if match else None
        recorded = row.get("fwhm_by_band")
        if isinstance(recorded, dict):
            for name, value in recorded.items():
                if fwhm_by_band.get(name) is None and isinstance(value, int | float):
                    fwhm_by_band[name] = float(value)
        if fwhm_by_band.get(Config.BAND_VIS.name) is None:
            vis_fwhm = row.get("fwhm_arcsec")
            if isinstance(vis_fwhm, int | float):
                fwhm_by_band[Config.BAND_VIS.name] = float(vis_fwhm)
        n_stars = row.get("n_stars")
        clusters.append({
            "index": index, "id": f"cluster-{index:03d}",
            "ra": row.get("ra") if isinstance(row.get("ra"), int | float) else None,
            "dec": row.get("dec") if isinstance(row.get("dec"), int | float) else None,
            "n_stars": int(n_stars) if isinstance(n_stars, int | float) else None,
            "fwhm_by_band": fwhm_by_band,
        })
    meta_stat = _catalog_key(meta_path) if meta_path else None
    return {
        "bands": bands,
        "generation": _records_psf_generation(),
        "clusters": clusters,
        "clusters_source": "metadata" if meta_rows else ("vis_headers" if vis else None),
        "clusters_meta": {"present": bool(meta_path),
                          "synced_at": os.path.getmtime(meta_path) if meta_stat else None},
        "last_sync": status.get("checked_at"),
    }


def _records_psf_generation() -> dict[str, Any] | None:
    """The PSFs the last generation run used, from the local records'
    provenance (the newest split that recorded them): ``{subset, psf_kinds:
    {band: empirical | gaussian}, run, created}``, or ``None`` when no local
    split recorded them (records generated before the stamp, or not synced)."""
    records_dir = _sky_records_local_dir()
    if not records_dir or not os.path.isdir(records_dir):
        return None
    found = [(subset, info) for subset in sky_records.SUBSETS
             if (info := sky_records.records_generation(records_dir, subset)) is not None]
    if not found:
        return None
    subset, info = max(found, key=lambda item: str(item[1].get("created") or ""))
    return {"subset": subset, **info}


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
                    primary = cast(fits.PrimaryHDU, _hdul[0])
                    item["n_psf"] = int(cast(
                        str | int, primary.header.get("NPSF", 1),
                    ))
            except Exception as e:
                item["error"] = str(e)
        bands.append(item)
    return {"bands": bands}


def _tfrecords_status(records_dir: str | None = None) -> dict[str, Any]:
    """List on-disk TFRecord files under ``records_dir``.

    Defaults to ``Config.RECORDS_DIR_V2`` (the locally-generated sky
    records) so existing callers stay unchanged; callers may pass their
    own dir to reuse this for FASRC-cached records.
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
    """Every checkpoint file of every ACTIVE ensemble member, recursing into
    sub-tracks.

    THE model is the ensemble: files live in ``<ensemble>/member_NN/`` (plus
    the ``loss_best/`` second save-best track inside each member). Each entry
    carries its ``member`` and ``folder`` so the UI can group + show
    locations. Archived members (registry tombstones) are not listed.
    """
    base = default_ensemble_dir()
    out: dict[str, Any] = {"dir": base, "files": []}
    for root in active_member_dirs(base):
        member = os.path.basename(root)
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
                        "member":  member,                    # member_00
                        "subdir":  subdir,                    # "" (root) or "loss_best"
                        "folder":  os.path.normpath(dirpath),  # …/member_00/loss_best
                        "size_mb": round(os.path.getsize(full) / 1e6, 1),
                    })
    # Members in order; root track before sub-tracks; sorted.
    out["files"].sort(key=lambda f: (f["member"], f["subdir"], f["name"]))
    return out


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


def _record_count(name: str, records_dir: str | None = None) -> int | None:
    """Records in a multi-band tfrecord (headers only, see
    :func:`sky_records.record_count`).

    Returns:
      * ``0``    — file does not exist
      * ``int``  — full record count
      * ``None`` — file present but partially corrupt (a truncated rsync, a
        bad header). Returning ``None`` instead of raising keeps callers from
        500-ing the whole response when one shard is bad.
    """
    return sky_records.record_count(tfrecord_path(records_dir or Config.RECORDS_DIR_V2, name))
