"""Open-everything integrity pass over downloaded star cutouts.

The downloader flags corruption as it goes, but a dedicated pass that *opens
every file on disk* is the durable guarantee: it catches cutouts that were
truncated/partially written and re-derives each ``(band, size)`` validity
flag from whether the FITS actually opens and carries finite data. Run it
right after a download so downstream consumers (PSF extraction, star-anchor
generation, the ``/star-cutouts`` gallery) can trust the catalog's
"valid in all 4 bands" without re-opening every file themselves.

Operates on a catalog directory: cutouts live under
``<output_dir>/cutouts/<band>/`` and the catalog CSV at
``<output_dir>/<CATALOG_FILE>``, read/written as a ``list[CatalogObject]``.
"""

from __future__ import annotations

import glob
import os
from typing import Any, cast

import numpy as np
from astropy.io import fits

from euclid_polish.catalog.catalog_object import CatalogObject
from euclid_polish.catalog.downloader import _FNAME_RE
from euclid_polish.config import Config


def _catalog_path(output_dir: str) -> str:
    return os.path.join(output_dir, Config.CATALOG_FILE)


def cutout_openable(path: str) -> bool:
    """True iff ``path`` opens as a FITS with non-empty, partly-finite data.

    Swallows every error (missing file, truncated download, unreadable
    header, all-NaN plane) → ``False``, so a bad cutout is simply marked
    corrupted rather than crashing the pass."""
    try:
        with fits.open(path, memmap=False) as hdul:
            primary = cast(fits.PrimaryHDU, hdul[0])
            data = primary.data
        arr = np.asarray(data, dtype=np.float32)
        return arr.size > 0 and bool(np.isfinite(arr).any())
    except Exception:
        return False


def _radec_from_header(path: str):
    """Recover ``(ra, dec)`` from a cutout's WCS reference (CRVAL1/2), the star
    position the downloader centred on. ``(None, None)`` if unreadable."""
    try:
        with fits.open(path, memmap=False) as hdul:
            h = cast(fits.PrimaryHDU, hdul[0]).header
        ra = float(cast(Any, h["CRVAL1"]))
        dec = float(cast(Any, h["CRVAL2"]))
        if np.isfinite(ra) and np.isfinite(dec):
            return ra, dec
    except Exception:
        pass
    return None, None


def rebuild_catalog_from_cutouts(
    output_dir: str,
    band_names: list[str] | None = None,
    *,
    reporter: Any = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Recover a corrupted/incomplete catalog from the cutouts already on disk.

    Scans ``<output_dir>/cutouts/<band>/star_<id>_<size>.fits`` and **adds any
    star id present on disk but missing from the catalog** — recovering its sky
    position from the FITS WCS (``CRVAL1/2``); magnitude/PSF flux are not stored
    in the cutouts, so they come back NaN (re-queryable later). Existing rows
    keep their metadata. It then re-derives every per-``(band,size)`` validity
    flag via :func:`validate_all_cutouts` and persists (atomically).

    Use after an OOM-killed download truncated the catalog: the cutout FITS are
    the durable record, this rebuilds the index over them. Returns a summary."""
    if band_names is None:
        band_names = [b.name for b in Config.BANDS]
    objects = CatalogObject.read(_catalog_path(output_dir))
    by_id = {int(o.id): o for o in objects if o.id is not None}
    cutouts_root = os.path.join(output_dir, Config.CUTOUTS_SUBDIR)

    id_to_path: dict[int, str] = {}
    for bn in band_names:
        band_dir = Config.cutout_dir_for_band(bn, root=cutouts_root)
        for path in glob.glob(os.path.join(band_dir, "star_[0-9]*_*.fits")):
            m = _FNAME_RE.search(os.path.basename(path))
            if m:
                id_to_path.setdefault(int(m.group(1)), path)

    on_disk = sorted(id_to_path)
    missing = [sid for sid in on_disk if sid not in by_id]
    no_radec = 0
    for sid in missing:
        ra, dec = _radec_from_header(id_to_path[sid])
        if ra is None:
            no_radec += 1
        objects.append(CatalogObject(
            ra=ra if ra is not None else float("nan"),
            dec=dec if dec is not None else float("nan"),
            id=sid, magnitude=float("nan")))

    summary: dict[str, Any] = {
        "ids_on_disk":        len(on_disk),
        "already_in_catalog": len(on_disk) - len(missing),
        "recovered":          len(missing),
        "missing_radec":      no_radec,
        "catalog_before":     len(by_id),
        "catalog_after":      len(objects),
        "dry_run":            bool(dry_run),
    }
    if dry_run:
        return summary
    res = validate_all_cutouts(output_dir, band_names, objects=objects,
                               reporter=reporter)
    summary["validated_cutouts"] = res["checked"]
    summary["valid_all_bands"] = res["valid_all_bands"]
    return summary


def validate_all_cutouts(
    output_dir: str,
    band_names: list[str] | None = None,
    *,
    objects: list[CatalogObject] | None = None,
    reporter: Any = None,
) -> dict[str, Any]:
    """Re-derive per-``(band, size)`` validity by opening every cutout on disk.

    For each ``star_<id>_<size>.fits`` under each band's cutout dir, set the
    object's ``valid``/``corrupted`` flag from :func:`cutout_openable`, then
    persist. ``objects`` is read from the catalog when not supplied; it is
    mutated and written back. ``reporter`` (optional) drives a progress bar.

    Returns ``{"checked", "unopenable", "valid_all_bands", "n_bands"}``.
    """
    if band_names is None:
        band_names = [b.name for b in Config.BANDS]
    if objects is None:
        objects = CatalogObject.read(_catalog_path(output_dir))
    by_id = {int(o.id): o for o in objects if o.id is not None}
    cutouts_root = os.path.join(output_dir, Config.CUTOUTS_SUBDIR)

    tasks = []
    for bn in band_names:
        band_dir = Config.cutout_dir_for_band(bn, root=cutouts_root)
        for path in sorted(glob.glob(os.path.join(band_dir, "star_[0-9]*_*.fits"))):
            m = _FNAME_RE.search(os.path.basename(path))
            if m:
                tasks.append((int(m.group(1)), bn, int(m.group(2)), path))

    n = len(tasks)
    unopenable = 0
    for i, (sid, bn, size, path) in enumerate(tasks):
        o = by_id.get(sid)
        if o is None:
            continue
        if cutout_openable(path):
            o.set_valid(size, band=bn)
        else:
            o.set_corrupted(size, band=bn)
            unopenable += 1
        if reporter is not None and (i % 200 == 0 or i == n - 1):
            reporter.set_step(i + 1, n, f"{bn} star {sid:04d}")

    CatalogObject.write(objects, _catalog_path(output_dir))

    want = set(band_names)
    valid_all = sum(1 for o in objects if want <= set(o.valid_bands()))
    return {
        "checked":          n,
        "unopenable":       unopenable,
        "valid_all_bands":  valid_all,
        "n_bands":          len(band_names),
    }


def purge_incomplete_cutouts(
    output_dir: str,
    band_names: list[str] | None = None,
    *,
    objects: list[CatalogObject] | None = None,
    dry_run: bool = False,
    drop_catalog_rows: bool = True,
    reporter: Any = None,
) -> dict[str, Any]:
    """Delete every cutout of any star that is **not valid in all bands**.

    A star is *complete* iff it has a valid cutout in **every** band of
    ``band_names`` (any size — the same ``want <= valid_bands`` test
    :func:`validate_all_cutouts` reports as ``valid_all_bands``). A star that
    is valid in only some bands — or missing a band entirely — is useless to
    the 4-channel pipeline / ePSF, so **all** of its cutout FITS are removed
    from disk (every band, every size). Complete stars are left untouched.

    Validity is read from the catalog flags, so run :func:`validate_all_cutouts`
    first if the on-disk files may have changed since the flags were written.

    With ``drop_catalog_rows`` (default), purged stars are also removed from
    the catalog so it stays consistent with disk; otherwise the rows are kept.
    ``dry_run`` reports what *would* be deleted, touching neither disk nor
    catalog.

    Returns ``{"n_bands", "complete_stars", "incomplete_stars",
    "deleted_files", "failed_deletes", "dropped_rows", "dry_run"}``.
    """
    if band_names is None:
        band_names = [b.name for b in Config.BANDS]
    if objects is None:
        objects = CatalogObject.read(_catalog_path(output_dir))
    want = set(band_names)
    cutouts_root = os.path.join(output_dir, Config.CUTOUTS_SUBDIR)

    purge_ids = {int(o.id) for o in objects
                 if o.id is not None and not (want <= set(o.valid_bands()))}
    complete_count = sum(1 for o in objects
                         if o.id is not None and want <= set(o.valid_bands()))

    # Map every on-disk cutout to its star id, robust to id zero-padding
    # (filenames are ``star_<id>_<size>.fits`` with ``:04d`` *minimum* width).
    id_to_paths: dict[int, list[str]] = {}
    for bn in band_names:
        band_dir = Config.cutout_dir_for_band(bn, root=cutouts_root)
        for path in glob.glob(os.path.join(band_dir, "star_[0-9]*_*.fits")):
            m = _FNAME_RE.search(os.path.basename(path))
            if m:
                id_to_paths.setdefault(int(m.group(1)), []).append(path)

    deleted = failed = 0
    targets = sorted(purge_ids)
    for i, sid in enumerate(targets):
        for path in id_to_paths.get(sid, []):
            if dry_run:
                deleted += 1
                continue
            try:
                os.remove(path)
                deleted += 1
            except OSError:
                failed += 1
        if reporter is not None and targets and (i % 200 == 0 or i == len(targets) - 1):
            reporter.set_step(i + 1, len(targets), f"purge star {sid:04d}")

    dropped = 0
    if not dry_run and drop_catalog_rows and purge_ids:
        kept = [o for o in objects
                if not (o.id is not None and int(o.id) in purge_ids)]
        CatalogObject.write(kept, _catalog_path(output_dir))
        dropped = len(purge_ids)

    return {
        "n_bands":          len(band_names),
        "complete_stars":   complete_count,
        "incomplete_stars": len(purge_ids),
        "deleted_files":    deleted,
        "failed_deletes":   failed,
        "dropped_rows":     dropped,
        "dry_run":          bool(dry_run),
    }
