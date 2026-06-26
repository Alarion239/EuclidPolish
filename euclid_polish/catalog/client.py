"""``EuclidCatalog`` — the authenticated client for all Euclid archive operations.

Construction authenticates eagerly::

    cat = EuclidCatalog(login="user", password="pw")   # explicit
    cat = EuclidCatalog()                               # EUCLID_USER / EUCLID_PASSWORD

and raises :class:`EuclidAuthError` if no credentials are available or the login
fails. The instance owns the session (astroquery's ``Euclid`` is a process-global
singleton; a class-level lock serialises (re)login). Queries return
:class:`~euclid_polish.catalog.catalog_object.CatalogObject` lists; downloads
return cutout FITS / a multi-band :class:`~euclid_polish.image.Image`.
"""

from __future__ import annotations

import math
import os
import threading
from typing import List, Optional

import numpy as np
from astroquery.esa.euclid import Euclid

from euclid_polish.catalog.archive import EuclidArchive
from euclid_polish.catalog.catalog_object import CatalogObject
from euclid_polish.catalog.downloader import fetch_cutout_at
from euclid_polish.catalog.photometry import uJy_to_ab_mag
from euclid_polish.config import Config


class EuclidAuthError(RuntimeError):
    """Raised when Euclid authentication cannot be established."""


def _finite(value) -> Optional[float]:
    if value is None or (hasattr(value, "mask") and bool(value.mask)):
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


# Galaxy selection window (clean, resolved, bigger-end, not point-like).
_GAL_DIAM_LO_ARCSEC = 2.0
_GAL_DIAM_HI_ARCSEC = 5.0
_GAL_MAG_FLOOR = 23.0


class EuclidCatalog:
    """Authenticated access to the Euclid science archive."""

    #: astroquery's ``Euclid`` is a process-global singleton — one login lock.
    _login_lock = threading.Lock()

    def __init__(self, login: Optional[str] = None, password: Optional[str] = None,
                 *, _skip_login: bool = False) -> None:
        self._user: Optional[str] = None
        self._password: Optional[str] = None
        if _skip_login:
            return
        user, pw = self._resolve_credentials(login, password)
        self._do_login(user, pw)

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_credentials(login: Optional[str],
                             password: Optional[str]) -> "tuple[str, str]":
        if login and password:
            return login, password
        env_user = os.environ.get("EUCLID_USER")
        env_pw = os.environ.get("EUCLID_PASSWORD")
        if env_user and env_pw:
            return env_user, env_pw
        raise EuclidAuthError(
            "No Euclid credentials: set EUCLID_USER and EUCLID_PASSWORD, "
            "or pass login=/password=.")

    def _do_login(self, user: str, password: str) -> None:
        with self._login_lock:
            try:
                Euclid.login(user=user, password=password)
            except Exception as e:
                raise EuclidAuthError(f"Euclid login failed: {e}") from e
        self._user, self._password = user, password

    def relogin(self) -> bool:
        """Refresh the session after a mid-batch TAP expiry. ``False`` on failure."""
        if not self._user or not self._password:
            return False
        try:
            with self._login_lock:
                Euclid.login(user=self._user, password=self._password)
            return True
        except Exception:
            return False

    @classmethod
    def _unauthenticated(cls) -> "EuclidCatalog":
        """A client that skipped the network login — tests / offline only."""
        return cls(_skip_login=True)

    # ------------------------------------------------------------------
    # Queries → CatalogObject lists
    # ------------------------------------------------------------------

    def query_bright_stars(
        self, num_stars: int, *,
        ra: Optional[float] = None, dec: Optional[float] = None,
        radius: Optional[float] = None,
        magnitude_limit: Optional[float] = None,
        magnitude_min: Optional[float] = None,
        snr_min: Optional[float] = None,
        require_unmasked: bool = True,
    ) -> List[CatalogObject]:
        """Top-``num_stars`` mask-free point sources by VIS PSF flux (descending).

        Magnitudes are AB (``Config.AB_ZP_UJY``); the raw PSF flux + error are
        kept on each object. ``require_unmasked`` adds ``det_quality_flag = 0``
        (the MER clean-point-source cut). ``snr_min`` keeps ``flux ≥ snr·err``.
        A cone (``ra``/``dec``/``radius``) is optional but must be given together.
        Returns objects in flux order; persistence/dedup is the caller's job.
        """
        if num_stars <= 0:
            raise ValueError("num_stars must be positive")
        if (magnitude_min is not None and magnitude_limit is not None
                and magnitude_min >= magnitude_limit):
            raise ValueError("magnitude_min must be < magnitude_limit")
        cone = [v is not None for v in (ra, dec, radius)]
        if any(cone) and not all(cone):
            raise ValueError("ra, dec, and radius must be provided together")

        where = ["flux_vis_psf IS NOT NULL", "flux_vis_psf > 0",
                 "fluxerr_vis_psf IS NOT NULL", "fluxerr_vis_psf > 0"]
        if magnitude_limit is not None:
            where.append(f"flux_vis_psf > {10 ** ((Config.AB_ZP_UJY - magnitude_limit) / 2.5)}")
        if magnitude_min is not None:
            where.append(f"flux_vis_psf < {10 ** ((Config.AB_ZP_UJY - magnitude_min) / 2.5)}")
        if snr_min is not None and snr_min > 0:
            where.append(f"flux_vis_psf > {float(snr_min)} * fluxerr_vis_psf")
        if require_unmasked:
            where.append("det_quality_flag = 0")
        if all(cone):
            where.append(
                f"CONTAINS(POINT('ICRS', right_ascension, declination), "
                f"CIRCLE('ICRS', {ra}, {dec}, {radius})) = 1")

        query = (f"SELECT TOP {num_stars} right_ascension, declination, "
                 f"flux_vis_psf, fluxerr_vis_psf FROM catalogue.mer_catalogue "
                 f"WHERE {' AND '.join(where)} ORDER BY flux_vis_psf DESC")
        results = self._launch(query, async_=True)

        objs: List[CatalogObject] = []
        for row in results or []:
            flux = _finite(row["flux_vis_psf"])
            if flux is None or flux <= 0:
                continue
            ferr = _finite(row["fluxerr_vis_psf"])
            objs.append(CatalogObject(
                ra=float(row["right_ascension"]), dec=float(row["declination"]),
                magnitude=uJy_to_ab_mag(flux), flux_psf_uJy=flux,
                fluxerr_psf_uJy=ferr if ferr is not None else float("nan"),
                kind="star"))
        return objs

    def query_galaxies(self, ra: float, dec: float, radius_deg: float, *,
                       mag_floor: float = _GAL_MAG_FLOOR) -> List[CatalogObject]:
        """Clean, resolved, bigger-end galaxies in a cone (``kind='galaxy'``).

        Server-side cuts: extended (``point_like_flag = 0``), not spurious, clean
        (``det_quality_flag = 0``), segmentation area within the
        2–5″ diameter window. Here we drop sources fainter than ``mag_floor``.
        """
        area_lo = math.pi * ((_GAL_DIAM_LO_ARCSEC / 2.0) / Config.VIS_PIXEL_SCALE_ARCSEC) ** 2
        area_hi = math.pi * ((_GAL_DIAM_HI_ARCSEC / 2.0) / Config.VIS_PIXEL_SCALE_ARCSEC) ** 2
        query = (
            "SELECT TOP 100000 object_id, right_ascension, declination, "
            "segmentation_area, flux_vis_psf FROM catalogue.mer_catalogue "
            f"WHERE CONTAINS(POINT('ICRS', right_ascension, declination), "
            f"CIRCLE('ICRS', {ra}, {dec}, {radius_deg})) = 1 "
            "AND point_like_flag = 0 AND spurious_flag = 0 AND det_quality_flag = 0 "
            f"AND segmentation_area BETWEEN {area_lo:.1f} AND {area_hi:.1f} "
            "AND flux_vis_psf IS NOT NULL AND flux_vis_psf > 0")
        results = self._launch(query, async_=False)

        objs: List[CatalogObject] = []
        for row in results or []:
            r = _finite(row["right_ascension"])
            d = _finite(row["declination"])
            flux = _finite(row["flux_vis_psf"])
            if r is None or d is None or flux is None or flux <= 0:
                continue
            mag = uJy_to_ab_mag(flux)
            if mag >= mag_floor:
                continue
            objs.append(CatalogObject(ra=r, dec=d, magnitude=mag,
                                      flux_psf_uJy=flux, kind="galaxy"))
        return objs

    def _launch(self, query: str, *, async_: bool):
        """Run an ADQL query, returning its result table (or ``None``)."""
        job = (Euclid.launch_job_async(query) if async_ else Euclid.launch_job(query))
        return job.get_results() if job is not None else None

    # ------------------------------------------------------------------
    # Downloads
    # ------------------------------------------------------------------

    def fetch_cutout(self, ra: float, dec: float, band: str, output_file: str,
                     size: int = 512) -> "tuple[bool, Optional[str]]":
        """Download one band's FITS cutout at ``(ra, dec)``. Returns ``(ok, err)``.

        ``size`` is the reference cutout side in VIS pixels (0.10″/pix grid).
        """
        return fetch_cutout_at(ra=ra, dec=dec, band_name=band,
                               output_file=output_file, cutout_size_vis_pixels=size)

    def fetch_image(self, ra: float, dec: float, size: int):
        """Download a 4-band electron-unit :class:`~euclid_polish.image.Image` at ``(ra, dec)``."""
        return EuclidArchive.fetch(ra=ra, dec=dec, size=size)
