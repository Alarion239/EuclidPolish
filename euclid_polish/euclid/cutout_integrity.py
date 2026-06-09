"""Open-everything integrity pass over downloaded star cutouts.

The downloader flags corruption as it goes, but a dedicated pass that *opens
every file on disk* is the durable guarantee: it catches cutouts that were
truncated/partially written and re-derives each ``(band, size)`` validity
flag from whether the FITS actually opens and carries finite data. Run it
right after a download so downstream consumers (PSF extraction, star-anchor
generation, the ``/star-cutouts`` gallery) can trust the catalog's
"valid in all 4 bands" without re-opening every file themselves.
"""

from __future__ import annotations

import glob
import os
import re
from typing import Any, Dict, List, Optional

import numpy as np
from astropy.io import fits

from euclid_polish.config import Config
from euclid_polish.euclid.catalog import StarCatalog

#: Cutout filenames are ``star_<id>_<size>.fits`` (id zero-padded to ≥4).
_FNAME_RE = re.compile(r"star_(\d+)_(\d+)\.fits$")


def cutout_openable(path: str) -> bool:
    """True iff ``path`` opens as a FITS with non-empty, partly-finite data.

    Swallows every error (missing file, truncated download, unreadable
    header, all-NaN plane) → ``False``, so a bad cutout is simply marked
    corrupted rather than crashing the pass."""
    try:
        with fits.open(path, memmap=False) as hdul:
            data = hdul[0].data
        arr = np.asarray(data, dtype=np.float32)
        return arr.size > 0 and bool(np.isfinite(arr).any())
    except Exception:
        return False


def _radec_from_header(path: str):
    """Recover ``(ra, dec)`` from a cutout's WCS reference (CRVAL1/2), the star
    position the downloader centred on. ``(None, None)`` if unreadable."""
    try:
        with fits.open(path, memmap=False) as hdul:
            h = hdul[0].header
        ra, dec = float(h["CRVAL1"]), float(h["CRVAL2"])
        if np.isfinite(ra) and np.isfinite(dec):
            return ra, dec
    except Exception:
        pass
    return None, None


def rebuild_catalog_from_cutouts(
    cat: StarCatalog,
    band_names: Optional[List[str]] = None,
    *,
    reporter: Any = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Recover a corrupted/incomplete catalog from the cutouts already on disk.

    Scans ``<output_dir>/cutouts/<band>/star_<id>_<size>.fits`` and **adds any
    star id present on disk but missing from the catalog** — recovering its sky
    position from the FITS WCS (``CRVAL1/2``); magnitude/PSF flux are not stored
    in the cutouts, so they come back NaN (re-queryable later). Existing rows
    keep their metadata. It then re-derives every per-``(band,size)`` validity
    flag via :func:`validate_all_cutouts` and persists (atomically).

    Use after an OOM-killed download truncated ``stars.csv``: the cutout FITS are
    the durable record, this rebuilds the index over them. Returns a summary."""
    if band_names is None:
        band_names = [b.name for b in Config.BANDS]
    catalog = cat.load()
    by_id = {int(s["id"]): s for s in catalog.get("stars", [])
             if s.get("id") is not None}
    cutouts_root = os.path.join(cat.output_dir, Config.CUTOUTS_SUBDIR)

    id_to_path: Dict[int, str] = {}
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
        catalog["stars"].append({
            "id":        sid,
            "ra":        ra if ra is not None else float("nan"),
            "dec":       dec if dec is not None else float("nan"),
            "magnitude": float("nan"),
        })
    ids = [int(s["id"]) for s in catalog["stars"] if s.get("id") is not None]
    catalog["next_id"] = (max(ids) + 1) if ids else 0

    summary: Dict[str, Any] = {
        "ids_on_disk":        len(on_disk),
        "already_in_catalog": len(on_disk) - len(missing),
        "recovered":          len(missing),
        "missing_radec":      no_radec,
        "catalog_before":     len(by_id),
        "catalog_after":      len(catalog["stars"]),
        "dry_run":            bool(dry_run),
    }
    if dry_run:
        return summary
    res = validate_all_cutouts(cat, catalog, band_names, reporter=reporter)
    summary["validated_cutouts"] = res["checked"]
    summary["valid_all_bands"] = res["valid_all_bands"]
    return summary


def validate_all_cutouts(
    cat: StarCatalog,
    catalog: Dict[str, Any],
    band_names: Optional[List[str]] = None,
    *,
    reporter: Any = None,
) -> Dict[str, Any]:
    """Re-derive per-``(band, size)`` validity by opening every cutout on disk.

    For each ``star_<id>_<size>.fits`` under each band's cutout dir, set the
    star's ``valid``/``corrupted`` flag from :func:`cutout_openable`, then
    persist via ``cat.save``. ``reporter`` (optional) drives a progress bar.

    Returns ``{"checked", "unopenable", "valid_all_bands", "n_bands"}``.
    """
    if band_names is None:
        band_names = [b.name for b in Config.BANDS]
    stars = catalog.get("stars", [])
    by_id = {int(s["id"]): s for s in stars if "id" in s}

    # Cutouts live under ``<output_dir>/cutouts/<band>/`` — the SAME root the
    # downloader writes to (``catalog.output_dir`` + CUTOUTS_SUBDIR), not the
    # fixed STAR_CUTOUTS_ROOT, so a custom output dir still resolves.
    cutouts_root = os.path.join(cat.output_dir, Config.CUTOUTS_SUBDIR)

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
        star = by_id.get(sid)
        if star is None:
            continue
        if cutout_openable(path):
            StarCatalog.set_valid(star, size, band=bn)
        else:
            StarCatalog.set_corrupted(star, size, band=bn)
            unopenable += 1
        if reporter is not None and (i % 200 == 0 or i == n - 1):
            reporter.set_step(i + 1, n, f"{bn} star {sid:04d}")

    cat.save(catalog)

    want = set(band_names)
    valid_all = sum(1 for s in stars
                    if want <= set(StarCatalog.valid_bands(s)))
    return {
        "checked":          n,
        "unopenable":       unopenable,
        "valid_all_bands":  valid_all,
        "n_bands":          len(band_names),
    }
