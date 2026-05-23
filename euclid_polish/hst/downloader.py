"""
HST cutout downloader (MAST HAPcut backend).

Mirrors :mod:`euclid_polish.euclid.downloader`: a single class wrapping
the underlying archive call so the rest of the codebase deals with
``(ra, dec, size) → on-disk FITS`` rather than astroquery quirks. We use
``astroquery.mast.Hapcut`` — HAP (Hubble Advanced Products) is the
canonical drizzled-science archive that serves *server-side* cutouts;
the alternative (downloading multi-GB mosaic tiles and cutting locally)
is much heavier for the same result.

Failure mode to know about: HAPcut returns ``0 cutouts`` if no HAP
coverage exists at the requested position. The COSMOS field has dense
ACS/F814W HAP coverage but not 100% — expect ~10-20% miss rate at
arbitrary positions. The downloader records failures so callers can
skip them on retry.
"""

from __future__ import annotations

import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
import astropy.units as u
from tqdm import tqdm

from euclid_polish.config import Config


@dataclass
class HSTDownloadConfig:
    """Configuration for one HST cutout download job.

    Attributes
    ----------
    output_dir
        Where cutouts are written. Created on first use.
    size_pix
        Native HST pixel side of the requested cutout (square).
    filter_name
        HST filter to request. HAPcut may return additional filters at
        the position; we keep only the matching one (or any if missing).
    max_workers
        Parallel HAPcut requests. MAST tolerates moderate concurrency;
        lower this if you see throttling.
    """

    output_dir:   str = Config.HST_TEMPLATES_DIR
    size_pix:     int = Config.HST_DEFAULT_CUTOUT_PIX
    filter_name:  str = "F814W"
    max_workers:  int = 4

    def validate(self) -> Tuple[bool, Optional[str]]:
        if self.size_pix <= 0:
            return False, "size_pix must be positive"
        if self.max_workers <= 0:
            return False, "max_workers must be positive"
        if not self.filter_name:
            return False, "filter_name must be non-empty"
        return True, None


class HSTCutoutDownloader:
    """Download HST drizzled-science cutouts and persist them as FITS.

    Usage
    -----
    >>> dl = HSTCutoutDownloader(HSTDownloadConfig(size_pix=256))
    >>> dl.download_one(cosmos_id=42, ra=150.1, dec=2.2)
    True

    The cutout filename follows
    :attr:`Config.HST_CUTOUT_FILENAME_FMT` so :class:`HSTTemplateLibrary`
    can locate it by ``cosmos_id`` alone.
    """

    def __init__(self, config: Optional[HSTDownloadConfig] = None):
        self.config = config or HSTDownloadConfig()
        ok, why = self.config.validate()
        if not ok:
            raise ValueError(f"Invalid HSTDownloadConfig: {why}")
        os.makedirs(self.config.output_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Single-position download
    # ------------------------------------------------------------------ #

    def cutout_path(self, cosmos_id: int) -> str:
        """Return the on-disk path for ``cosmos_id``'s cached cutout."""
        fname = Config.HST_CUTOUT_FILENAME_FMT.format(
            cosmos_id=int(cosmos_id), size=self.config.size_pix,
        )
        return os.path.join(self.config.output_dir, fname)

    def already_have(self, cosmos_id: int) -> bool:
        """True if a non-empty cutout already exists on disk for ``cosmos_id``."""
        p = self.cutout_path(cosmos_id)
        return os.path.exists(p) and os.path.getsize(p) > 0

    def download_one(
        self, cosmos_id: int, ra: float, dec: float, overwrite: bool = False,
    ) -> bool:
        """Download a single cutout. Return True on success.

        Re-uses an existing on-disk file unless ``overwrite=True``. Writes
        the cached file atomically (``.tmp`` → rename) so a partial
        download cannot leave a corrupted FITS in the library.
        """
        # Lazy import: astroquery.mast.Hapcut is slow to import, and this
        # downloader is constructed even in tests that never call it.
        from astroquery.mast import Hapcut

        out = self.cutout_path(cosmos_id)
        if not overwrite and self.already_have(cosmos_id):
            return True

        coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
        try:
            hduls = Hapcut.get_cutouts(coordinates=coord, size=self.config.size_pix)
        except Exception as e:
            tqdm.write(f"  HST cutout failed @ ({ra:.5f}, {dec:.5f}): "
                       f"{type(e).__name__}: {e}")
            return False

        if not hduls:
            return False

        chosen = self._pick_filter_match(hduls, self.config.filter_name)
        if chosen is None:
            return False

        # Atomic write — never leave a partial FITS in the library on
        # interrupt or crash mid-flush.
        tmp = out + ".tmp"
        try:
            chosen.writeto(tmp, overwrite=True)
            shutil.move(tmp, out)
        except Exception as e:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
            tqdm.write(f"  HST write failed @ {out}: {type(e).__name__}: {e}")
            return False
        finally:
            for h in hduls:
                try:
                    h.close()
                except Exception:
                    pass
        return True

    @staticmethod
    def _pick_filter_match(
        hduls: List[fits.HDUList], wanted_filter: str,
    ) -> Optional[fits.HDUList]:
        """Pick the HDUL whose SCI HDU matches ``wanted_filter``.

        HAPcut returns one HDUL per matching HAP product at the position.
        Headers may store the filter under any of FILTER / FILTER1 /
        FILTER2 / PHOTPLAM; we accept the first explicit match and fall
        back to the first valid SCI HDU if no filter keyword is set.
        """
        wf = wanted_filter.upper()
        # Helper: pull a filter token from an HDU's header (whichever key
        # holds it).
        def _filter_token(hdu: fits.PrimaryHDU | fits.ImageHDU) -> Optional[str]:
            h = hdu.header
            for k in ("FILTER", "FILTER1", "FILTER2"):
                v = h.get(k)
                if v and isinstance(v, str) and v.strip():
                    return v.strip().upper()
            return None

        fallback: Optional[fits.HDUList] = None
        for hdul in hduls:
            sci = next(
                (e for e in hdul if e.is_image and e.data is not None
                 and getattr(e, "name", "").upper() in ("SCI", "PRIMARY")),
                None,
            )
            if sci is None:
                continue
            tok = _filter_token(sci) or _filter_token(hdul[0])
            if tok and wf in tok:
                return hdul
            if fallback is None:
                fallback = hdul
        return fallback

    # ------------------------------------------------------------------ #
    # Parallel batch download
    # ------------------------------------------------------------------ #

    def download_many(
        self,
        positions: List[Tuple[int, float, float]],
        *,
        show_progress: bool = True,
        overwrite: bool = False,
    ) -> dict:
        """Download cutouts for every ``(cosmos_id, ra, dec)`` in parallel.

        Returns a summary dict with ``downloaded``, ``skipped``,
        ``failed_ids``. Files already on disk are skipped unless
        ``overwrite=True``.
        """
        downloaded = 0
        skipped    = 0
        failed_ids: List[int] = []

        def _one(item):
            cosmos_id, ra, dec = item
            if not overwrite and self.already_have(cosmos_id):
                return cosmos_id, "skip"
            ok = self.download_one(cosmos_id, ra, dec, overwrite=overwrite)
            return cosmos_id, ("ok" if ok else "fail")

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as pool:
            futures = [pool.submit(_one, p) for p in positions]
            iterator = as_completed(futures)
            if show_progress:
                iterator = tqdm(
                    iterator,
                    total=len(futures),
                    desc=f"HST cutouts ({self.config.max_workers}x)",
                )
            for fut in iterator:
                cid, status = fut.result()
                if status == "ok":
                    downloaded += 1
                elif status == "skip":
                    skipped += 1
                else:
                    failed_ids.append(cid)

        return {
            "downloaded": downloaded,
            "skipped":    skipped,
            "failed":     len(failed_ids),
            "failed_ids": failed_ids,
            "size_pix":   self.config.size_pix,
            "filter":     self.config.filter_name,
        }
