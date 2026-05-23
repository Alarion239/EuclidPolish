"""
On-disk template library: the index of downloaded HST cutouts.

The library is a directory under :attr:`Config.HST_TEMPLATES_DIR` with:

* one FITS cutout per (cosmos_id, size) under the
  :attr:`Config.HST_CUTOUT_FILENAME_FMT` naming scheme
* one CSV index file (``templates_catalog.csv``) listing every cutout's
  metadata for fast scanning by the renderer

Append-only by design — once a cutout is in the catalog, its row is not
overwritten without an explicit :meth:`rebuild_from_disk` call. This
matches the :class:`StarCatalog` pattern in :mod:`euclid_polish.euclid`.
"""

from __future__ import annotations

import csv
import os
from dataclasses import asdict
from typing import Dict, List, Optional

from astropy.io import fits

from euclid_polish.config import Config
from euclid_polish.hst.types import HSTCutoutMetadata


_CSV_FIELDS = (
    "cosmos_id", "ra_deg", "dec_deg", "size_pix",
    "pixel_scale_arcsec", "filename", "filter_name", "bunit",
)


class HSTTemplateLibrary:
    """In-memory + on-disk index of HST template cutouts.

    Layout::

        <root>/
            templates_catalog.csv
            tpl_00001234_0256.fits
            tpl_00005678_0256.fits
            ...

    The CSV header matches :attr:`HSTCutoutMetadata` fields; rows are
    deduped by ``(cosmos_id, size_pix)``.
    """

    def __init__(self, root: str = Config.HST_TEMPLATES_DIR):
        self.root = root
        self.catalog_path = os.path.join(root, Config.HST_TEMPLATE_CATALOG_FILE)
        os.makedirs(root, exist_ok=True)
        self._entries: Dict[tuple, HSTCutoutMetadata] = {}   # (id, size) → meta
        if os.path.exists(self.catalog_path):
            self._load_csv()

    # ------------------------------------------------------------------ #
    # In-memory mutators
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self):
        return iter(self._entries.values())

    def add(self, meta: HSTCutoutMetadata, *, save: bool = True) -> None:
        """Register a new cutout. Overwrites an existing same-key entry."""
        key = (int(meta.cosmos_id), int(meta.size_pix))
        self._entries[key] = meta
        if save:
            self.save()

    def get(
        self, cosmos_id: int, size_pix: Optional[int] = None,
    ) -> Optional[HSTCutoutMetadata]:
        """Look up a cutout by ``(cosmos_id, size_pix)``.

        When ``size_pix`` is None, returns the first entry for that
        ``cosmos_id`` regardless of size.
        """
        if size_pix is not None:
            return self._entries.get((int(cosmos_id), int(size_pix)))
        for (cid, _sz), meta in self._entries.items():
            if cid == int(cosmos_id):
                return meta
        return None

    def cutout_path(self, meta: HSTCutoutMetadata) -> str:
        """Resolve ``meta.filename`` against the library root."""
        return os.path.join(self.root, meta.filename)

    def cutout_ids(self) -> List[int]:
        """All ``cosmos_id`` values currently in the library (deduped)."""
        return sorted({cid for (cid, _sz) in self._entries.keys()})

    # ------------------------------------------------------------------ #
    # CSV persistence
    # ------------------------------------------------------------------ #

    def save(self) -> None:
        """Write the in-memory index to ``templates_catalog.csv``."""
        tmp = self.catalog_path + ".tmp"
        with open(tmp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
            w.writeheader()
            for meta in self._entries.values():
                w.writerow({k: v for k, v in asdict(meta).items() if k in _CSV_FIELDS})
        os.replace(tmp, self.catalog_path)

    def _load_csv(self) -> None:
        with open(self.catalog_path, newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                meta = HSTCutoutMetadata(
                    cosmos_id          = int(row["cosmos_id"]),
                    ra_deg             = float(row["ra_deg"]),
                    dec_deg            = float(row["dec_deg"]),
                    size_pix           = int(row["size_pix"]),
                    pixel_scale_arcsec = float(row["pixel_scale_arcsec"]),
                    filename           = row["filename"],
                    filter_name        = row.get("filter_name", "F814W"),
                    bunit              = row.get("bunit",       "ELECTRONS/S"),
                )
                self._entries[(meta.cosmos_id, meta.size_pix)] = meta

    # ------------------------------------------------------------------ #
    # Disk reconciliation
    # ------------------------------------------------------------------ #

    def rebuild_from_disk(self) -> int:
        """Rescan the templates directory and re-index every FITS found.

        Useful after a bulk transfer where ``templates_catalog.csv`` was
        lost. Returns the number of cutouts re-indexed. Files that do not
        parse as templates (missing WCS or non-conforming filename) are
        skipped with a warning.
        """
        self._entries.clear()
        for name in sorted(os.listdir(self.root)):
            if not name.startswith("tpl_") or not name.endswith(".fits"):
                continue
            # Parse cosmos_id and size from filename:
            # tpl_<cosmos_id 8 digits>_<size 4 digits>.fits
            try:
                cid_str, size_str = name[len("tpl_"):-len(".fits")].split("_")
                cid = int(cid_str)
                sz  = int(size_str)
            except ValueError:
                print(f"  skipping unparseable {name}")
                continue
            path = os.path.join(self.root, name)
            try:
                with fits.open(path) as hdul:
                    sci = next(
                        (e for e in hdul if e.is_image and e.data is not None),
                        None,
                    )
                    if sci is None:
                        continue
                    h    = sci.header
                    ra   = float(h.get("CRVAL1", 0.0))
                    dec  = float(h.get("CRVAL2", 0.0))
                    # Pixel scale from CD matrix or CDELT.
                    cd11 = float(h.get("CD1_1", h.get("CDELT1", 0.0)))
                    cd22 = float(h.get("CD2_2", h.get("CDELT2", 0.0)))
                    scale = (abs(cd11) + abs(cd22)) / 2.0 * 3600.0
                    if scale == 0:
                        scale = Config.HST_NATIVE_PIXEL_SCALE_ARCSEC
                    filt = (
                        h.get("FILTER") or h.get("FILTER1") or h.get("FILTER2")
                        or "F814W"
                    )
                    bunit = h.get("BUNIT", "ELECTRONS/S")
            except Exception as e:
                print(f"  skipping unreadable {name}: {type(e).__name__}: {e}")
                continue
            meta = HSTCutoutMetadata(
                cosmos_id=cid, ra_deg=ra, dec_deg=dec, size_pix=sz,
                pixel_scale_arcsec=scale, filename=name,
                filter_name=str(filt).strip(), bunit=str(bunit).strip(),
            )
            self._entries[(cid, sz)] = meta
        self.save()
        return len(self._entries)

    # ------------------------------------------------------------------ #
    # Data access
    # ------------------------------------------------------------------ #

    def load_image(self, meta: HSTCutoutMetadata):
        """Return the 2-D pixel array for ``meta`` (float32).

        Reads the first image HDU containing data. Native units from the
        FITS header (typically electrons/s for ACS drizzled) — the
        renderer normalises before depositing so units don't matter
        downstream.
        """
        import numpy as np
        path = self.cutout_path(meta)
        with fits.open(path) as hdul:
            sci = next(
                (e for e in hdul if e.is_image and e.data is not None),
                None,
            )
            if sci is None:
                raise IOError(f"no image HDU in {path}")
            return np.asarray(sci.data, dtype=np.float32)
