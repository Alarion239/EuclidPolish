"""``CatalogObject`` — one Euclid catalogue object (a queried row).

Holds the object's sky position + PSF photometry and its per-``(band, size)``
cutout-download status, and (de)serialises a list of objects to the ``stars.csv``
catalog file with a stable provenance sidecar.

This is local-only: no network, no credentials. :class:`EuclidCatalog` produces
``CatalogObject`` instances from a live archive query; everything in this module
operates on objects already in hand.

The CSV schema is one row per object::

    id,ra,dec,magnitude,flux_psf_uJy,fluxerr_psf_uJy,valid:<band>:<size>,...

Flag columns ``{kind}:{band}:{size}`` (``kind ∈ {valid, corrupted,
download_failed}``) grow lazily as ``(band, size)`` pairs are touched; a ``True``
cell sets that flag, an empty cell means "no info".
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from euclid_polish.config import Config
from euclid_polish.catalog.validator import angular_separation_arcsec
from euclid_polish.provenance.ids import ProvId
from euclid_polish.provenance.records import Stamp

_FLAG_KINDS = ("valid", "corrupted", "download_failed")
_FLAG_RE = re.compile(r"^(valid|corrupted|download_failed):([^:]+):(\d+)$")
_BASE_COLS = ("id", "ra", "dec", "magnitude", "flux_psf_uJy", "fluxerr_psf_uJy")
_OPT_FLOAT_COLS = ("flux_psf_uJy", "fluxerr_psf_uJy")

#: Serialises catalog writes across threads — parallel band downloads share one
#: ``stars.csv`` and would otherwise race on the temp file.
_write_lock = threading.Lock()


def _finite_float(value) -> Optional[float]:
    """``float(value)`` if finite, else ``None`` (handles masked / NaN / missing)."""
    if value is None or (hasattr(value, "mask") and bool(value.mask)):
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


@dataclass
class CatalogObject:
    """A Euclid catalogue object: position, PSF photometry, download status."""

    ra: float
    dec: float
    id: Optional[int] = None
    magnitude: Optional[float] = None
    flux_psf_uJy: Optional[float] = None
    fluxerr_psf_uJy: Optional[float] = None
    kind: str = "star"
    flags: Dict[str, Dict[str, Dict[str, bool]]] = field(
        default_factory=lambda: {k: {} for k in _FLAG_KINDS})

    def __post_init__(self) -> None:
        for k in _FLAG_KINDS:
            self.flags.setdefault(k, {})

    # ------------------------------------------------------------------
    # Per-(band, size) cutout-download status
    # ------------------------------------------------------------------

    def _read_flag(self, kind: str, band: str, size: Optional[int]) -> bool:
        slot = self.flags.get(kind, {}).get(band, {})
        if not isinstance(slot, dict):
            return False
        if size is None:
            return any(slot.values())
        return bool(slot.get(str(size), False))

    def _write_flag(self, kind: str, band: str, size: int, value: bool) -> None:
        if value:
            self.flags.setdefault(kind, {}).setdefault(band, {})[str(size)] = True
        else:
            slot = self.flags.get(kind, {}).get(band, {})
            slot.pop(str(size), None)
            if not slot:
                self.flags.get(kind, {}).pop(band, None)

    def has_any(self, kind: str) -> bool:
        """True if *any* ``(band, size)`` flag of ``kind`` is set."""
        return any(any(sizes.values()) for sizes in self.flags.get(kind, {}).values()
                   if isinstance(sizes, dict))

    def is_valid(self, size: Optional[int] = None, band: str = "VIS") -> bool:
        return self._read_flag("valid", band, size)

    def set_valid(self, size: int, band: str = "VIS") -> None:
        """Mark valid and clear any matching corrupted flag (a successful
        re-download supersedes a prior failure)."""
        self._write_flag("valid", band, size, True)
        self._write_flag("corrupted", band, size, False)

    def is_corrupted(self, size: Optional[int] = None, band: str = "VIS") -> bool:
        return self._read_flag("corrupted", band, size)

    def set_corrupted(self, size: int, band: str = "VIS") -> None:
        self._write_flag("corrupted", band, size, True)
        self._write_flag("valid", band, size, False)

    def is_download_failed(self, size: Optional[int] = None, band: str = "VIS") -> bool:
        return self._read_flag("download_failed", band, size)

    def set_download_failed(self, size: int, band: str = "VIS") -> None:
        self._write_flag("download_failed", band, size, True)

    def clear_download_failed(self, size: int, band: str = "VIS") -> None:
        """Drop a ``download_failed`` flag so the object is retried."""
        self._write_flag("download_failed", band, size, False)

    def valid_sizes(self, band: str = "VIS") -> List[int]:
        slot = self.flags.get("valid", {}).get(band, {})
        return [int(k) for k, ok in slot.items() if ok] if isinstance(slot, dict) else []

    def valid_bands(self, size: Optional[int] = None) -> List[str]:
        """Bands with at least one valid cutout (``size=None`` accepts any)."""
        out: List[str] = []
        for band, sizes in self.flags.get("valid", {}).items():
            if not isinstance(sizes, dict):
                continue
            if size is None and any(sizes.values()):
                out.append(band)
            elif size is not None and sizes.get(str(size), False):
                out.append(band)
        return out

    # ------------------------------------------------------------------
    # Row serialization
    # ------------------------------------------------------------------

    def to_row(self) -> Dict[str, Any]:
        """This object as a flat CSV row dict (base scalars + flag columns)."""
        row: Dict[str, Any] = {
            "id":        self.id,
            "ra":        _finite_float(self.ra),
            "dec":       _finite_float(self.dec),
            "magnitude": _finite_float(self.magnitude),
        }
        for k in _OPT_FLOAT_COLS:
            v = _finite_float(getattr(self, k))
            if v is not None:
                row[k] = v
        for kind in _FLAG_KINDS:
            for band, sizes in self.flags.get(kind, {}).items():
                for size, ok in sizes.items():
                    if ok:
                        row[f"{kind}:{band}:{size}"] = True
        return row

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> "CatalogObject":
        """Build a :class:`CatalogObject` from a CSV row dict."""
        v = _finite_float(row.get("id"))
        obj_id = int(v) if v is not None else None
        flags: Dict[str, Dict[str, Dict[str, bool]]] = {k: {} for k in _FLAG_KINDS}
        for col, val in row.items():
            m = _FLAG_RE.match(str(col))
            if not m:
                continue
            truthy = val is True or (isinstance(val, str)
                                     and val.lower() in ("true", "1", "t"))
            if not truthy:
                continue
            kind, band, size = m.group(1), m.group(2), m.group(3)
            flags[kind].setdefault(band, {})[size] = True
        return cls(
            ra=_finite_float(row.get("ra")) or float("nan"),
            dec=_finite_float(row.get("dec")) or float("nan"),
            id=obj_id,
            magnitude=_finite_float(row.get("magnitude")),
            flux_psf_uJy=_finite_float(row.get("flux_psf_uJy")),
            fluxerr_psf_uJy=_finite_float(row.get("fluxerr_psf_uJy")),
            flags=flags,
        )

    # ------------------------------------------------------------------
    # Catalog file I/O (the whole stars.csv)
    # ------------------------------------------------------------------

    @classmethod
    def read(cls, path: str) -> "List[CatalogObject]":
        """Read every object from a ``stars.csv`` catalog (empty list if absent).

        A truncated / half-written file is copied to ``.corrupt`` and treated as
        empty rather than raising.
        """
        if not os.path.exists(path):
            return []
        try:
            df = pd.read_csv(path)
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            try:
                if os.path.getsize(path) > 0:
                    shutil.copy2(path, path + ".corrupt")
            except OSError:
                pass
            return []
        return [cls.from_row(row.to_dict()) for _, row in df.iterrows()]

    @classmethod
    def write(cls, objects: "List[CatalogObject]", path: str) -> None:
        """Atomically write ``objects`` to ``path`` and stamp a stable prov id.

        The write renders to a unique temp file then renames, keeps a one-deep
        ``.bak`` of the prior catalog, and is serialised across threads. A stable
        :class:`ProvId` is minted once into ``<path>.prov.json`` and reused.
        """
        out_dir = os.path.dirname(path) or "."
        os.makedirs(out_dir, exist_ok=True)
        rows = [o.to_row() for o in objects]
        df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=list(_BASE_COLS))
        flag_cols = sorted(c for c in df.columns if c not in _BASE_COLS)
        base = [c for c in _BASE_COLS if c in df.columns or not rows]
        df = df.reindex(columns=list(base) + flag_cols)
        with _write_lock:
            fd, tmp = tempfile.mkstemp(dir=out_dir,
                                       prefix=os.path.basename(path) + ".",
                                       suffix=".tmp")
            os.close(fd)
            try:
                df.to_csv(tmp, index=False)
                if os.path.exists(path):
                    try:
                        shutil.copy2(path, path + ".bak")
                    except OSError:
                        pass
                os.replace(tmp, path)
            finally:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
        cls._ensure_prov(path)

    # -- provenance: a stable id per catalog file (minted once, reused) -- #

    @staticmethod
    def prov_id(path: str) -> Optional[ProvId]:
        """The catalog's stable :class:`ProvId`, or ``None`` if never written."""
        p = path + ".prov.json"
        if not os.path.exists(p):
            return None
        try:
            with open(p) as f:
                return Stamp.from_json(f.read()).id
        except Exception:
            return None

    @staticmethod
    def _ensure_prov(path: str) -> None:
        try:
            if CatalogObject.prov_id(path) is not None:
                return
            stamp = Stamp(id=ProvId.mint(lambda _id: False), schema_version=3)
            tmp = path + ".prov.json.tmp"
            with open(tmp, "w") as f:
                f.write(stamp.to_json())
            os.replace(tmp, path + ".prov.json")
        except Exception:
            pass


def next_id(objects: "List[CatalogObject]") -> int:
    """One past the largest assigned id (0 for an empty / id-less list)."""
    ids = [o.id for o in objects if isinstance(o.id, int)]
    return max(ids) + 1 if ids else 0


# ---------------------------------------------------------------------------
# List-level operations (dedup / ingest / aggregation)
# ---------------------------------------------------------------------------

def _is_duplicate(ra: float, dec: float, objects: "List[CatalogObject]",
                  tol_arcsec: float) -> bool:
    """True if ``(ra, dec)`` lies within ``tol_arcsec`` of any object's position."""
    for o in objects:
        if angular_separation_arcsec(ra, dec, o.ra, o.dec) < tol_arcsec:
            return True
    return False


def merge_new(existing: "List[CatalogObject]",
              candidates: "List[CatalogObject]", *,
              tol_arcsec: float = Config.Matching.CATALOG_POSITION_TOL_ARCSEC,
              limit: Optional[int] = None) -> Dict[str, Any]:
    """Append non-duplicate ``candidates`` to ``existing`` with fresh ids.

    Candidates within ``tol_arcsec`` of an already-present object (existing or a
    just-added candidate) are skipped. ``limit`` caps how many are added. Mutates
    ``existing`` in place; returns ``{"added", "skipped", "objects"}``.
    """
    nid = next_id(existing)
    added = skipped = 0
    for cand in candidates:
        if limit is not None and added >= limit:
            break
        if _is_duplicate(cand.ra, cand.dec, existing, tol_arcsec):
            skipped += 1
            continue
        cand.id = nid
        existing.append(cand)
        nid += 1
        added += 1
    return {"added": added, "skipped": skipped, "objects": existing}


def by_status(objects: "List[CatalogObject]") -> Dict[str, "List[CatalogObject]"]:
    """Bucket objects by any-``(band, size)`` download status."""
    return {
        "valid":     [o for o in objects if o.has_any("valid")],
        "corrupted": [o for o in objects if o.has_any("corrupted")],
        "failed":    [o for o in objects if o.has_any("download_failed")],
        "pending":   [o for o in objects
                      if not o.has_any("valid")
                      and not o.has_any("corrupted")
                      and not o.has_any("download_failed")],
        "all":       objects,
    }


def summarize(objects: "List[CatalogObject]") -> Dict[str, Any]:
    """Aggregate counts over a catalog: totals, per-status, per-band, mag range."""
    buckets = by_status(objects)
    summary: Dict[str, Any] = {
        "total":     len(objects),
        "valid":     len(buckets["valid"]),
        "corrupted": len(buckets["corrupted"]),
        "failed":    len(buckets["failed"]),
        "pending":   len(buckets["pending"]),
        "next_id":   next_id(objects),
    }
    summary["valid_by_band"] = {
        b.name: sum(1 for o in objects if o.is_valid(band=b.name))
        for b in Config.BANDS
    }
    mags = [o.magnitude for o in objects
            if o.magnitude is not None and np.isfinite(o.magnitude)]
    if mags:
        summary["mag_min"] = min(mags)
        summary["mag_max"] = max(mags)
    return summary
