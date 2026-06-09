"""Star catalog (CSV) management.

The catalog is one CSV file per output directory with a single row
per star and the schema:

    id,ra,dec,magnitude,valid:<band>:<size>,corrupted:<band>:<size>,...

Flag columns are named ``{kind}:{band}:{size}`` for
``kind ∈ {valid, corrupted, download_failed}``. A ``True`` cell sets
the flag for that ``(band, size)``; an empty cell means "no info".
The columns grow lazily as new ``(band, size)`` pairs get touched.

In memory, each star is a plain dict with the fixed scalar keys plus
three nested dicts:

    {
      "id": int, "ra": float, "dec": float, "magnitude": float,
      "valid":           {band: {str(size): True}},
      "corrupted":       {band: {str(size): True}},
      "download_failed": {band: {str(size): True}},
    }

This is the *only* on-disk and in-memory format — no JSON fallback,
no band-less bool fallback. Existing JSON catalogs can be converted
with ``scripts/convert_stars_json_to_csv.py``.
"""

import os
import re
import shutil
import tempfile
import threading
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
from astroquery.esa.euclid import Euclid

from euclid_polish.config import Config
from euclid_polish.euclid.photometry import uJy_to_ab_mag
from euclid_polish.euclid.validator import angular_separation_arcsec

from euclid_polish.config import Config as _Cfg

_FLAG_KINDS = ("valid", "corrupted", "download_failed")
_FLAG_RE    = re.compile(r"^(valid|corrupted|download_failed):([^:]+):(\d+)$")
_BASE_COLS  = ("id", "ra", "dec", "magnitude")


# ---------------------------------------------------------------------------
# Row ⇄ star-dict conversion
# ---------------------------------------------------------------------------

def _unmask_float(value) -> Optional[float]:
    """Astropy masked-table cell → ``float``, or ``None`` when masked/missing."""
    if value is None or (hasattr(value, "mask") and bool(value.mask)):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _nan_or(value, cast):
    """Cast ``value`` or return NaN/None when missing.

    Real catalog rows always carry all four scalar fields, but tests and
    in-memory star dicts under construction may omit them — round-trip
    them as NaN rather than raising.
    """
    if value is None:
        return float("nan")
    if isinstance(value, float) and np.isnan(value):
        return value
    return cast(value)


def _row_to_star(row: Dict[str, Any]) -> Dict[str, Any]:
    """Pandas row → canonical star dict (scalars + nested flag dicts)."""
    raw_id = row.get("id")
    star: Dict[str, Any] = {
        "id":        int(raw_id) if raw_id is not None and not (isinstance(raw_id, float) and np.isnan(raw_id)) else None,
        "ra":        _nan_or(row.get("ra"),        float),
        "dec":       _nan_or(row.get("dec"),       float),
        "magnitude": _nan_or(row.get("magnitude"), float),
    }
    # Optional raw PSF photometry (µJy) — present for star-anchor stars,
    # absent for older catalogs / test rows (kept out of the star dict then).
    for k in ("flux_psf_uJy", "fluxerr_psf_uJy"):
        v = row.get(k)
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            star[k] = float(v)
    flags: Dict[str, Dict[str, Dict[str, bool]]] = {k: {} for k in _FLAG_KINDS}
    for col, val in row.items():
        m = _FLAG_RE.match(str(col))
        if not m:
            continue
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue
        # CSV reads boolean-like strings ("True"/"False") as strings; coerce.
        truthy = val is True or (isinstance(val, str)
                                 and val.lower() in ("true", "1", "t"))
        if not truthy and val is not True:
            continue
        kind, band, size = m.group(1), m.group(2), m.group(3)
        flags[kind].setdefault(band, {})[size] = True
    for kind in _FLAG_KINDS:
        if flags[kind]:
            star[kind] = flags[kind]
    return star


def _star_to_row(star: Dict[str, Any]) -> Dict[str, Any]:
    """Canonical star dict → flat row dict for pandas. Missing scalars
    are written as NaN so partially-populated stars round-trip cleanly."""
    row: Dict[str, Any] = {
        "id":        star.get("id"),
        "ra":        _nan_or(star.get("ra"),        float),
        "dec":       _nan_or(star.get("dec"),       float),
        "magnitude": _nan_or(star.get("magnitude"), float),
    }
    for k in ("flux_psf_uJy", "fluxerr_psf_uJy"):
        v = star.get(k)
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            row[k] = float(v)
    for kind in _FLAG_KINDS:
        nested = star.get(kind)
        if not isinstance(nested, dict):
            continue
        for band, sizes in nested.items():
            if not isinstance(sizes, dict):
                continue
            for size, ok in sizes.items():
                if not ok:
                    continue
                row[f"{kind}:{band}:{size}"] = True
    return row


# ---------------------------------------------------------------------------
# StarCatalog
# ---------------------------------------------------------------------------

class StarCatalog:
    """Read/write helper for the per-output-dir ``stars.csv`` catalog."""

    DEFAULT_BAND = "VIS"

    #: Serialises ``save`` across threads. Parallel band downloads share one
    #: ``stars.csv``; without this, two ``save`` calls racing on the same temp
    #: file truncated the catalog, and a later band then read a short catalog
    #: and decided it had nothing left to download (silent ``downloaded=0``).
    #: Process-wide (class attribute) because separate ``StarCatalog`` instances
    #: for different bands point at the same file.
    _save_lock = threading.Lock()

    def __init__(self, output_dir: str = Config.DEFAULT_OUTPUT_DIR):
        self.output_dir = output_dir
        self.catalog_path = os.path.join(output_dir, Config.CATALOG_FILE)

    # ── file I/O ──────────────────────────────────────────────────────────

    def exists(self) -> bool:
        return os.path.exists(self.catalog_path)

    def load(self) -> Dict[str, Any]:
        """Return ``{"stars": [...], "next_id": int}``.

        ``next_id`` is derived from the data — one past the largest ID
        present, or 0 for an empty catalog.
        """
        if not self.exists():
            return {"stars": [], "next_id": 0}
        try:
            df = pd.read_csv(self.catalog_path)
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            # File exists but is empty / truncated / half-written (e.g. an
            # OOM-killed download interrupted a non-atomic write). Preserve the
            # bytes to ``.corrupt`` for forensics/recovery before treating it as
            # empty — otherwise the next save would silently overwrite a corrupt
            # catalog with a fresh small batch and orphan every on-disk cutout.
            try:
                if os.path.getsize(self.catalog_path) > 0:
                    shutil.copy2(self.catalog_path, self.catalog_path + ".corrupt")
            except OSError:
                pass
            return {"stars": [], "next_id": 0}
        # ``df.iterrows`` yields Series objects with NaN for missing flag
        # columns; ``to_dict`` keeps those as floats and ``_row_to_star``
        # drops them.
        stars = [_row_to_star(row.to_dict()) for _, row in df.iterrows()]
        ids = [s["id"] for s in stars if isinstance(s.get("id"), int)]
        next_id = (max(ids) + 1) if ids else 0
        return {"stars": stars, "next_id": next_id}

    def save(self, catalog: Dict[str, Any]) -> None:
        """Write the catalog. ``next_id`` is recomputed on load, so it's
        OK if the caller mutates it — we ignore that field here."""
        os.makedirs(self.output_dir, exist_ok=True)
        stars = catalog.get("stars", [])
        rows = [_star_to_row(s) for s in stars]
        if rows:
            df = pd.DataFrame(rows)
        else:
            df = pd.DataFrame(columns=list(_BASE_COLS))
        flag_cols = sorted(c for c in df.columns if c not in _BASE_COLS)
        df = df.reindex(columns=list(_BASE_COLS) + flag_cols)
        # Atomic write: render to a *uniquely named* temp file then rename.
        # ``to_csv`` is NOT atomic, so a crash/OOM mid-write would otherwise
        # truncate the live catalog — the index of which on-disk cutouts exist.
        # A one-deep ``.bak`` of the prior good catalog gives an immediate
        # recovery copy. The unique temp name + ``_save_lock`` make concurrent
        # saves (parallel band downloads) safe: a shared ``stars.csv.tmp`` used
        # to be written by two bands at once, producing a truncated catalog.
        with self._save_lock:
            fd, tmp = tempfile.mkstemp(
                dir=self.output_dir, prefix=Config.CATALOG_FILE + ".",
                suffix=".tmp")
            os.close(fd)
            try:
                df.to_csv(tmp, index=False)
                if os.path.exists(self.catalog_path):
                    try:
                        shutil.copy2(self.catalog_path, self.catalog_path + ".bak")
                    except OSError:
                        pass
                os.replace(tmp, self.catalog_path)
            finally:
                if os.path.exists(tmp):
                    try:
                        os.remove(tmp)
                    except OSError:
                        pass

    # ── per-(band, size) flag primitives ──────────────────────────────────
    #
    # All flag dicts are guaranteed nested ``{band: {str(size): True}}``
    # — no bool / band-less variants ever reach this code.

    @staticmethod
    def _read_flag(star: Dict[str, Any], kind: str,
                   band: str, size: Optional[int]) -> bool:
        band_slot = star.get(kind, {}).get(band, {})
        if not isinstance(band_slot, dict):
            return False
        if size is None:
            return any(band_slot.values())
        return bool(band_slot.get(str(size), False))

    @staticmethod
    def _write_flag(star: Dict[str, Any], kind: str,
                    band: str, size: int, value: bool) -> None:
        nested = star.get(kind)
        if not isinstance(nested, dict):
            nested = {} if value else None
        if value:
            band_slot = nested.setdefault(band, {})
            band_slot[str(size)] = True
            star[kind] = nested
        else:
            if not isinstance(nested, dict):
                return
            band_slot = nested.get(band, {})
            band_slot.pop(str(size), None)
            if not band_slot:
                nested.pop(band, None)
            if not nested:
                star.pop(kind, None)

    # ── public accessors (call sites in the downloader / extractor) ───────

    @staticmethod
    def is_valid(star: Dict[str, Any], size: Optional[int] = None,
                 band: str = "VIS") -> bool:
        """``size=None`` → True if *any* size in ``band`` is valid."""
        return StarCatalog._read_flag(star, "valid", band, size)

    @staticmethod
    def set_valid(star: Dict[str, Any], size: int, band: str = "VIS") -> None:
        """Mark valid and clear any matching corrupted flag (a successful
        re-download supersedes a prior failure)."""
        StarCatalog._write_flag(star, "valid", band, size, True)
        StarCatalog._write_flag(star, "corrupted", band, size, False)

    @staticmethod
    def is_corrupted(star: Dict[str, Any], size: Optional[int] = None,
                     band: str = "VIS") -> bool:
        return StarCatalog._read_flag(star, "corrupted", band, size)

    @staticmethod
    def set_corrupted(star: Dict[str, Any], size: int, band: str = "VIS") -> None:
        StarCatalog._write_flag(star, "corrupted", band, size, True)
        StarCatalog._write_flag(star, "valid", band, size, False)

    @staticmethod
    def is_download_failed(star: Dict[str, Any], size: Optional[int] = None,
                           band: str = "VIS") -> bool:
        return StarCatalog._read_flag(star, "download_failed", band, size)

    @staticmethod
    def set_download_failed(star: Dict[str, Any], size: int,
                            band: str = "VIS") -> None:
        StarCatalog._write_flag(star, "download_failed", band, size, True)

    @staticmethod
    def clear_download_failed(star: Dict[str, Any], size: int,
                              band: str = "VIS") -> None:
        """Drop a ``download_failed`` flag so the star is retried.

        ``download_failed`` is meant for "no VIS tile covers this star", but a
        transient TAP/session error during mosaic resolution marks *every*
        pending star failed too. ``--retry-failed`` clears these so genuinely
        coverable stars get another attempt (truly uncovered ones re-flag)."""
        StarCatalog._write_flag(star, "download_failed", band, size, False)

    @staticmethod
    def valid_sizes(star: Dict[str, Any], band: str = "VIS") -> List[int]:
        band_slot = star.get("valid", {}).get(band, {})
        if not isinstance(band_slot, dict):
            return []
        return [int(k) for k, ok in band_slot.items() if ok]

    @staticmethod
    def valid_bands(star: Dict[str, Any],
                    size: Optional[int] = None) -> List[str]:
        """Bands with at least one valid cutout. ``size=None`` accepts
        any; pass a specific size to require it."""
        nested = star.get("valid", {})
        if not isinstance(nested, dict):
            return []
        out: List[str] = []
        for band, sizes in nested.items():
            if not isinstance(sizes, dict):
                continue
            if size is None:
                if any(sizes.values()):
                    out.append(band)
            elif sizes.get(str(size), False):
                out.append(band)
        return out

    def _has_any_flag(self, star: Dict[str, Any], kind: str) -> bool:
        nested = star.get(kind)
        if not isinstance(nested, dict):
            return False
        return any(any(sizes.values()) for sizes in nested.values()
                   if isinstance(sizes, dict))

    # ── catalog queries ───────────────────────────────────────────────────

    def get_stars_by_status(self) -> Dict[str, List[Dict[str, Any]]]:
        """Categorise stars by ``any-(band, size)`` status."""
        stars = self.load().get("stars", [])
        return {
            "valid":     [s for s in stars if self._has_any_flag(s, "valid")],
            "corrupted": [s for s in stars if self._has_any_flag(s, "corrupted")],
            "failed":    [s for s in stars if self._has_any_flag(s, "download_failed")],
            "pending":   [s for s in stars
                          if not self._has_any_flag(s, "valid")
                          and not self._has_any_flag(s, "corrupted")
                          and not self._has_any_flag(s, "download_failed")],
            "all":       stars,
        }

    def get_star_by_id(self, star_id: int) -> Optional[Dict[str, Any]]:
        for star in self.load().get("stars", []):
            if star.get("id") == star_id:
                return star
        return None

    def get_summary(self) -> Dict[str, Any]:
        catalog = self.load()
        stars = catalog.get("stars", [])

        valid     = [s for s in stars if self._has_any_flag(s, "valid")]
        corrupted = [s for s in stars if self._has_any_flag(s, "corrupted")]
        failed    = [s for s in stars if self._has_any_flag(s, "download_failed")]
        pending   = [s for s in stars
                     if not self._has_any_flag(s, "valid")
                     and not self._has_any_flag(s, "corrupted")
                     and not self._has_any_flag(s, "download_failed")]

        summary = {
            "total":     len(stars),
            "valid":     len(valid),
            "corrupted": len(corrupted),
            "failed":    len(failed),
            "pending":   len(pending),
            "next_id":   catalog.get("next_id", 0),
        }
        summary["valid_by_band"] = {
            b.name: sum(1 for s in stars if self.is_valid(s, band=b.name))
            for b in _Cfg.BANDS
        }
        mags = [s["magnitude"] for s in stars if s.get("magnitude") is not None]
        if mags:
            summary["mag_min"] = min(mags)
            summary["mag_max"] = max(mags)
        return summary

    # ── ingest / query helpers ────────────────────────────────────────────

    def _is_duplicate_star(self, ra: float, dec: float,
                           existing_stars: List[Dict],
                           tolerance_arcsec: float = Config.Matching.CATALOG_POSITION_TOL_ARCSEC
                           ) -> bool:
        for star in existing_stars:
            if angular_separation_arcsec(ra, dec,
                                         star["ra"], star["dec"]
                                         ) < tolerance_arcsec:
                return True
        return False

    def _query_bright_stars(self, ra: float, dec: float, radius: float,
                            magnitude_limit: float,
                            num_stars: Optional[int] = None,
                            magnitude_min: Optional[float] = None
                            ) -> List[Dict[str, Any]]:
        """ADQL cone query on ``mer_catalogue`` (legacy synchronous path).

        Uses ``flux_vis_psf`` (TPHOT PSF flux, µJy) + ``fluxerr_vis_psf`` like
        the async path; magnitudes are proper AB (``Config.AB_ZP_UJY``)."""
        query = f"""
        SELECT TOP 100000
            right_ascension, declination, flux_vis_psf, fluxerr_vis_psf
        FROM catalogue.mer_catalogue
        WHERE CONTAINS(
            POINT('ICRS', right_ascension, declination),
            CIRCLE('ICRS', {ra}, {dec}, {radius})
        ) = 1
          AND flux_vis_psf IS NOT NULL
          AND flux_vis_psf > 0
        """
        try:
            job = Euclid.launch_job(query)
            results = job.get_results()
            if results is None or len(results) == 0:
                return []

            stars: List[Dict[str, Any]] = []
            for row in results:
                flux = _unmask_float(row["flux_vis_psf"])
                if flux is None or not np.isfinite(flux) or flux <= 0:
                    continue
                ferr = _unmask_float(row["fluxerr_vis_psf"])
                mag = uJy_to_ab_mag(flux)
                if mag >= magnitude_limit:
                    continue
                if magnitude_min is not None and mag <= magnitude_min:
                    continue
                stars.append({
                    "ra":              float(row["right_ascension"]),
                    "dec":             float(row["declination"]),
                    "magnitude":       mag,
                    "flux_psf_uJy":    flux,
                    "fluxerr_psf_uJy": ferr if (ferr is not None and np.isfinite(ferr)) else float("nan"),
                })
            stars.sort(key=lambda s: s["magnitude"])
            if num_stars is not None:
                stars = stars[:num_stars]
            return stars
        except Exception as e:
            print(f"    Query failed: {e}")
            return []

    def query_euclid_catalog(self, ra: float, dec: float, radius: float,
                             magnitude_limit: float,
                             num_stars: Optional[int] = None,
                             magnitude_min: Optional[float] = None
                             ) -> Dict[str, Any]:
        """Cone query + dedupe + persist. Faint-end cap is mandatory;
        ``magnitude_min`` adds a bright-end cutoff to skip saturating stars."""
        if (magnitude_min is not None
                and magnitude_min >= magnitude_limit):
            raise ValueError(
                f"magnitude_min ({magnitude_min}) must be < "
                f"magnitude_limit ({magnitude_limit}) — the window would be empty."
            )
        catalog = self.load()
        existing_stars = catalog["stars"]
        next_id = catalog["next_id"]

        available = [s for s in existing_stars
                     if not self._has_any_flag(s, "corrupted")
                     and not self._has_any_flag(s, "download_failed")]
        existing_available_count = len(available)

        if num_stars is not None:
            num_to_add = max(0, num_stars - existing_available_count)
            if num_to_add == 0:
                return {
                    "added":    0,
                    "skipped":  0,
                    "total":    len(existing_stars),
                    "next_id":  next_id,
                    "message":  f"Already have {existing_available_count}/{num_stars} available stars",
                }
        else:
            num_to_add = None

        new_stars = self._query_bright_stars(
            ra=ra, dec=dec, radius=radius,
            magnitude_limit=magnitude_limit,
            num_stars=num_to_add,
            magnitude_min=magnitude_min,
        )
        if not new_stars:
            return {
                "added":   0, "skipped":  0,
                "total":   len(existing_stars),
                "next_id": next_id,
                "message": "No new stars found in this region",
            }
        return self._ingest_stars(new_stars, catalog, num_to_add=num_to_add)

    def _ingest_stars(self, new_stars: List[Dict[str, Any]],
                      catalog: Dict[str, Any],
                      num_to_add: Optional[int] = None) -> Dict[str, Any]:
        """Dedupe + assign IDs + persist. Shared by all query methods."""
        existing_stars = catalog["stars"]
        next_id = catalog["next_id"]

        added_count = 0
        skipped_count = 0
        for star in new_stars:
            if num_to_add is not None and added_count >= num_to_add:
                break
            if self._is_duplicate_star(star["ra"], star["dec"], existing_stars):
                skipped_count += 1
            else:
                star["id"] = next_id
                existing_stars.append(star)
                next_id += 1
                added_count += 1

        catalog["stars"]   = existing_stars
        catalog["next_id"] = next_id
        self.save(catalog)

        return {
            "added":   added_count,
            "skipped": skipped_count,
            "total":   len(existing_stars),
            "next_id": next_id,
            "message": f"Added {added_count} stars → {len(existing_stars)} total in catalog",
        }

    def query_brightest_stars(self, num_stars: int,
                              ra: Optional[float] = None,
                              dec: Optional[float] = None,
                              radius: Optional[float] = None,
                              magnitude_limit: Optional[float] = None,
                              magnitude_min: Optional[float] = None,
                              snr_min: Optional[float] = None,
                              ) -> Dict[str, Any]:
        """Server-side TOP-N query sorted by VIS **PSF** flux (descending).

        Uses ``flux_vis_psf`` (TPHOT PSF-fitting photometry, µJy) — the
        point-source-optimal total flux — and its error ``fluxerr_vis_psf``.
        Magnitudes are proper AB (µJy zeropoint ``Config.AB_ZP_UJY``); the raw
        flux + error are stored so the star-anchor delta uses the physical
        flux directly (and a zeropoint tweak never needs a re-query).
        ``snr_min`` keeps only well-measured stars (``flux/fluxerr ≥ snr_min``).
        """
        if num_stars <= 0:
            raise ValueError("num_stars must be positive")
        if (magnitude_min is not None and magnitude_limit is not None
                and magnitude_min >= magnitude_limit):
            raise ValueError(
                f"magnitude_min ({magnitude_min}) must be < "
                f"magnitude_limit ({magnitude_limit}) — the window would be empty."
            )

        cone_given = [v is not None for v in (ra, dec, radius)]
        if any(cone_given) and not all(cone_given):
            raise ValueError("ra, dec, and radius must be provided together")

        where_clauses = [
            "flux_vis_psf IS NOT NULL",
            "flux_vis_psf > 0",
            "fluxerr_vis_psf IS NOT NULL",
            "fluxerr_vis_psf > 0",
        ]
        # Magnitude window → µJy PSF-flux bounds via the AB µJy zeropoint.
        if magnitude_limit is not None:
            flux_min = 10 ** ((Config.AB_ZP_UJY - magnitude_limit) / 2.5)
            where_clauses.append(f"flux_vis_psf > {flux_min}")
        if magnitude_min is not None:
            flux_max = 10 ** ((Config.AB_ZP_UJY - magnitude_min) / 2.5)
            where_clauses.append(f"flux_vis_psf < {flux_max}")
        if snr_min is not None and snr_min > 0:
            where_clauses.append(f"flux_vis_psf > {float(snr_min)} * fluxerr_vis_psf")
        if all(cone_given):
            where_clauses.append(
                f"CONTAINS(POINT('ICRS', right_ascension, declination), "
                f"CIRCLE('ICRS', {ra}, {dec}, {radius})) = 1"
            )

        query = (
            f"SELECT TOP {num_stars} right_ascension, declination, "
            f"flux_vis_psf, fluxerr_vis_psf "
            f"FROM catalogue.mer_catalogue "
            f"WHERE {' AND '.join(where_clauses)} "
            f"ORDER BY flux_vis_psf DESC"
        )

        try:
            job = Euclid.launch_job_async(query)
            results = job.get_results()
        except Exception as e:
            print(f"    Async query failed: {e}")
            return {"added": 0, "skipped": 0,
                    "total": 0, "next_id": 0,
                    "message": f"Query failed: {e}"}

        if results is None or len(results) == 0:
            catalog = self.load()
            return {"added": 0, "skipped": 0,
                    "total":   len(catalog.get("stars", [])),
                    "next_id": catalog.get("next_id", 0),
                    "message": "No stars returned"}

        new_stars: List[Dict[str, Any]] = []
        for row in results:
            flux = _unmask_float(row["flux_vis_psf"])
            ferr = _unmask_float(row["fluxerr_vis_psf"])
            # Even with the WHERE filter, astropy returns masked tables — guard.
            if flux is None or not np.isfinite(flux) or flux <= 0:
                continue
            new_stars.append({
                "ra":             float(row["right_ascension"]),
                "dec":            float(row["declination"]),
                "magnitude":      uJy_to_ab_mag(flux),
                "flux_psf_uJy":   flux,
                "fluxerr_psf_uJy": ferr if (ferr is not None and np.isfinite(ferr)) else float("nan"),
            })

        catalog = self.load()
        return self._ingest_stars(new_stars, catalog, num_to_add=num_stars)
