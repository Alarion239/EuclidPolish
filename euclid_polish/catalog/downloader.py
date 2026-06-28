"""Cutout-download primitives for the Euclid archive.

Stateless helpers the :class:`~euclid_polish.catalog.client.EuclidCatalog`
batch downloader composes: mosaic-tile resolution, the on-disk cutout scan,
position matching, a single ``get_cutout`` call, and the saturation check.
``fetch_cutout_at`` is the standalone single-position fetch used by ad-hoc
inference (it bypasses any catalog).
"""

from __future__ import annotations

import glob
import os
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astroquery.esa.euclid import Euclid
from tqdm import tqdm

from euclid_polish.catalog.validator import FitsValidator, angular_separation_arcsec
from euclid_polish.config import Config

#: Matches ``star_<id>_<size>.fits``; ``:04d`` is a *minimum* width so IDs ≥ 10000
#: render with 5+ digits — ``\d+`` matches any digit count.
_FNAME_RE = re.compile(r"star_(\d+)_(\d+)\.fits$")

#: Serialises the TAP mosaic-tile query across threads. ``astroquery``'s
#: ``Euclid`` is a process-wide singleton wrapping one TAP session; firing
#: several ``launch_job_async`` jobs at once (parallel band downloads each
#: resolving their tiles) shares that session's connection/job state and is not
#: thread-safe. The query is ~50 s vs a ~20 min download phase, so serialising
#: it costs nothing while removing the race.
_TAP_QUERY_LOCK = threading.Lock()


def query_mosaic_tiles(query: str, *, retries: int = 2,
                       relogin: Callable[[], bool] | None = None):
    """Run a ``sedm.mosaic_product`` ADQL query, tolerant of a dropped session.

    ``Euclid.launch_job_async`` can return ``None`` (e.g. the TAP session expired
    after a long download) — calling ``.get_results()`` on it then raised a
    confusing ``AttributeError: 'NoneType'``. This guards that, and **retries**
    so a session that lapsed between bands self-heals. When ``relogin`` is given,
    it is invoked between attempts to refresh the session.

    The TAP call is serialised by ``_TAP_QUERY_LOCK`` so concurrent band
    downloads don't run several async jobs on the shared session at once.

    Returns ``(results_or_None, error_message)``.
    """
    last = ""
    for attempt in range(max(1, retries)):
        try:
            with _TAP_QUERY_LOCK:
                job = Euclid.launch_job_async(query)
                results = job.get_results() if job is not None else None
            if results is not None:
                return results, ""
            last = "launch_job_async returned None (TAP session expired?)"
        except Exception as e:                       # noqa: BLE001 — surfaced below
            last = f"{type(e).__name__}: {e}"
        if attempt + 1 < retries:
            if relogin is not None:
                try:
                    relogin()
                except Exception:                    # noqa: BLE001
                    pass
            time.sleep(3.0)
    return None, last


def core_is_saturated(data: np.ndarray, core_size: int) -> bool:
    """True if the central ``core_size × core_size`` region is clipped.

    The Euclid MER pipeline zeros pixels above well capacity, so an
    oversaturated / clipped star has a hole (non-finite or ``<= 0`` values) at
    its centre. Cutouts are centred on the star, so we inspect the central core.
    ``core_size <= 0`` disables the check (returns False).
    """
    if core_size is None or core_size < 1 or data is None:
        return False
    ny, nx = data.shape[:2]
    cy, cx = ny // 2, nx // 2
    sh = int(core_size) // 2
    core = data[max(0, cy - sh): cy + sh + 1, max(0, cx - sh): cx + sh + 1]
    if core.size == 0:
        return True
    return bool((~np.isfinite(core)).any() or (core <= 0).any())


@dataclass
class DownloadConfig:
    """Configuration for cutout downloading (one band at a time).

    Driven by a :class:`euclid_polish.config.BandConfig`. The ``band``
    attribute selects:

      * which catalog flag slot to write (per-band valid/corrupted/failed)
      * which on-disk subdirectory to use (``cutouts/<band_name>/``)
      * which Euclid archive instrument + filter to query
      * which pixel scale to use for the cutout-radius calculation.

    Pass a :class:`BandConfig` directly via :meth:`for_band` for a fully
    populated config; the bare init exists for back-compat tests.
    """
    cutout_size: int = Config.DEFAULT_CUTOUT_SIZE
    position_tolerance: float = Config.Matching.DOWNLOAD_POSITION_TOL_ARCSEC
    size_tolerance: int = Config.Matching.DOWNLOAD_SIZE_TOL_PIXELS
    max_workers: int = 8  # parallel cutout HTTPS fetches
    band: str = "VIS"
    instrument: str = "VIS"
    filter_name: str | None = None
    pixel_scale_arcsec: float = Config.VIS_PIXEL_SCALE_ARCSEC
    # Saturation/clipping rejection: a downloaded cutout is rejected when
    # its central ``saturation_core_size × saturation_core_size`` region
    # (centred on the star) contains any non-finite or ``<= 0`` pixel —
    # the Euclid MER pipeline zeros out detector pixels above well
    # capacity, so a clipped/saturated core shows up as that hole. Same
    # check the PSF extractor applies. 0 disables.
    saturation_core_size: int = 7

    @classmethod
    def for_band(
        cls,
        band_name: str,
        *,
        cutout_size_vis_pixels: int | None = None,
        cutout_size: int | None = None,
        **overrides,
    ) -> DownloadConfig:
        """Build a download config from a band name.

        Cutout size can be specified two ways:

        * ``cutout_size_vis_pixels`` (preferred): the *reference* size in
          VIS pixels at 0.10″/pix. Each band's native cutout size is
          derived so every band covers the same angular field. Example:
          ``cutout_size_vis_pixels=512`` → 51.2″ on a side → VIS
          fetches 512 px, NISP fetches ~171 px.
        * ``cutout_size``: explicit native pixel count for this band
          (bypasses the conversion; useful for band-by-band control).

        Pulls instrument / filter / native pixel scale from the band's
        :class:`BandConfig`; remaining knobs default unless overridden.
        """
        b = Config.get_band(band_name)

        if cutout_size is not None and cutout_size_vis_pixels is not None:
            raise ValueError(
                "Pass either cutout_size or cutout_size_vis_pixels, not both."
            )
        if cutout_size_vis_pixels is not None:
            arcsec_side = (
                int(cutout_size_vis_pixels)
                * Config.BAND_VIS.pixel_scale_lr_arcsec
            )
            cutout_size = b.cutout_size_for_arcsec(arcsec_side)

        defaults = {
            "band": b.name,
            "instrument": b.archive_instrument,
            "filter_name": b.archive_filter or None,
            "pixel_scale_arcsec": b.pixel_scale_lr_arcsec,
        }
        if cutout_size is not None:
            defaults["cutout_size"] = int(cutout_size)
        defaults.update(overrides)
        return cls(**defaults)

    def validate(self) -> tuple[bool, str | None]:
        """Validate configuration."""
        if self.cutout_size <= 0:
            return False, "Cutout size must be positive"
        if self.position_tolerance <= 0:
            return False, "Position tolerance must be positive"
        if self.max_workers <= 0:
            return False, "max_workers must be positive"
        if self.instrument not in ("VIS", "NISP"):
            return False, f"instrument must be VIS or NISP, got {self.instrument!r}"
        if self.instrument == "NISP" and not self.filter_name:
            return False, "NISP downloads require a filter_name (e.g. 'NIR_Y')"
        if self.pixel_scale_arcsec <= 0:
            return False, "pixel_scale_arcsec must be positive"
        return True, None


def positions_match(ra1: float, dec1: float, ra2: float, dec2: float,
                    tol_arcsec: float) -> bool:
    """True if two sky positions agree within ``tol_arcsec``."""
    return angular_separation_arcsec(ra1, dec1, ra2, dec2) < tol_arcsec


def scan_cutouts(
    cutout_dir: str, validator: FitsValidator | None = None,
) -> tuple[
    dict[int, list[tuple[float, float, int, str]]],
    list[tuple[int, int | None, str]],
]:
    """Scan ``cutout_dir`` for ``star_<id>_<size>.fits`` and read each WCS centre.

    A single star can have multiple cutouts at different sizes
    (e.g. ``star_0042_256.fits`` and ``star_0042_512.fits``); all are returned.

    Returns ``(positions, corrupted)`` where
    ``positions = {id: [(ra, dec, size, path), ...]}`` and
    ``corrupted = [(id, size_or_None, path), ...]``.
    """
    validator = validator or FitsValidator()
    positions: dict[int, list[tuple[float, float, int, str]]] = {}
    corrupted: list[tuple[int, int | None, str]] = []

    fits_files = glob.glob(os.path.join(cutout_dir, "star_[0-9]*_*.fits"))

    for filepath in fits_files:
        filename = os.path.basename(filepath)
        m = _FNAME_RE.search(filename)
        if m is None:
            print(f"Warning: unexpected cutout filename {filename!r}")
            continue
        star_id = int(m.group(1))
        size = int(m.group(2))
        try:
            header = validator.get_header(filepath)
            if header and 'CRVAL1' in header and 'CRVAL2' in header:
                positions.setdefault(star_id, []).append(
                    (float(header['CRVAL1']), float(header['CRVAL2']), size, filepath))
            else:
                corrupted.append((star_id, size, filepath))
        except (KeyError, OSError) as e:
            corrupted.append((star_id, size, filepath))
            print(f"Warning: Could not read {filename}: {e}")

    return positions, corrupted


def resolve_mosaics(
    objects: list[Any], config: DownloadConfig, *,
    relogin: Callable[[], bool] | None = None,
) -> dict[int, dict[str, Any]]:
    """Map every object to its mosaic tile in a single ADQL query.

    Pulls the full mosaic catalogue from ``sedm.mosaic_product`` once
    (filtered by instrument + optional filter), then matches each object to its
    nearest tile centre (tiles are ~0.5° wide, so the nearest centre is the
    containing tile within tile-overlap noise).

    Returns ``{id: {'file_path': '<path>/<name>', 'tile_index': int}}``; objects
    not covered by any tile are absent.
    """
    if not objects:
        return {}

    where_parts = [f"instrument_name = '{config.instrument}'"]
    if config.filter_name:
        where_parts.append(f"filter_name = '{config.filter_name}'")
    query = (
        "SELECT file_path, file_name, tile_index, ra, dec "
        "FROM sedm.mosaic_product "
        f"WHERE {' AND '.join(where_parts)}"
    )
    mosaics, err = query_mosaic_tiles(query, relogin=relogin)
    if mosaics is None:
        print(f"  Mosaic lookup failed: {err}")
        return {}
    if len(mosaics) == 0:
        print(f"  Mosaic lookup returned 0 tiles for "
              f"instrument={config.instrument} "
              f"filter={config.filter_name or '-'}")
        return {}

    mosaic_coords = SkyCoord(
        ra=np.asarray(mosaics['ra']) * u.degree,
        dec=np.asarray(mosaics['dec']) * u.degree,
        frame='icrs',
    )
    obj_coords = SkyCoord(
        ra=np.asarray([o.ra for o in objects]) * u.degree,
        dec=np.asarray([o.dec for o in objects]) * u.degree,
        frame='icrs',
    )

    idx, sep, _ = obj_coords.match_to_catalog_sky(mosaic_coords)

    max_sep = 0.5 * u.degree  # tile half-diagonal-ish; loose upper bound
    out: dict[int, dict[str, Any]] = {}
    for i, obj in enumerate(objects):
        if sep[i] > max_sep:
            continue
        m = mosaics[int(idx[i])]
        out[obj.id] = {
            'file_path': f"{m['file_path']}/{m['file_name']}",
            'tile_index': int(m['tile_index']),
        }
    return out


def download_one_cutout(
    ra: float, dec: float, config: DownloadConfig,
    cutout_radius_arcmin: float, output_file: str, mosaic: dict[str, Any],
) -> bool:
    """Download a single cutout at ``(ra, dec)`` using a pre-resolved tile.

    Returns True iff ``get_cutout`` produced a non-empty file.
    """
    coord = SkyCoord(ra=ra, dec=dec, unit=(u.degree, u.degree), frame='icrs')
    try:
        Euclid.get_cutout(
            file_path=mosaic['file_path'],
            instrument=config.instrument,
            id=mosaic['tile_index'],
            coordinate=coord,
            radius=cutout_radius_arcmin * u.arcmin,
            output_file=output_file,
        )
    except Exception as e:
        tqdm.write(f"  get_cutout failed for ({ra:.4f}, {dec:.4f}): "
                   f"{type(e).__name__}: {e}")
        return False
    return os.path.exists(output_file) and os.path.getsize(output_file) > 0


# ---------------------------------------------------------------------------
# Ad-hoc single-position cutout helper (used by the inference UI)
# ---------------------------------------------------------------------------

def fetch_cutout_at(
    ra: float,
    dec: float,
    band_name: str,
    output_file: str,
    cutout_size_vis_pixels: int = 512,
) -> tuple[bool, str | None]:
    """Fetch a single Euclid cutout at one (ra, dec) for one band.

    Bypasses the on-disk star catalog — meant for ad-hoc inference where the
    caller just wants the four-band cube at a given sky position.

    Parameters
    ----------
    ra, dec : float
        ICRS coordinates in degrees.
    band_name : str
        One of ``"VIS" / "Y_E" / "J_E" / "H_E"``.
    output_file : str
        Where to write the FITS. Parent dir must exist.
    cutout_size_vis_pixels : int
        Reference cutout size in VIS pixels at 0.10″/pix. Each band's native
        cutout count is derived so every band covers the same angular field
        (~51.2″ for the default 512 px).

    Returns
    -------
    (ok, error_message) :
        ``ok=True`` on success (file exists and is non-empty); ``ok=False`` with
        an explanation otherwise. Failures are reported without raising.
    """
    cfg = DownloadConfig.for_band(
        band_name, cutout_size_vis_pixels=cutout_size_vis_pixels,
    )
    ok, err = cfg.validate()
    if not ok:
        return False, err

    arcsec_side = cfg.cutout_size * cfg.pixel_scale_arcsec
    cutout_radius_arcmin = (arcsec_side / 2.0) / 60.0

    where = [f"instrument_name = '{cfg.instrument}'"]
    if cfg.filter_name:
        where.append(f"filter_name = '{cfg.filter_name}'")
    query = (
        "SELECT file_path, file_name, tile_index, ra, dec "
        "FROM sedm.mosaic_product "
        f"WHERE {' AND '.join(where)}"
    )
    mosaics, err = query_mosaic_tiles(query)
    if mosaics is None:
        return False, f"mosaic lookup failed: {err}"
    if len(mosaics) == 0:
        return False, "no mosaic tiles matched the band query"

    star_coord = SkyCoord(ra=ra * u.degree, dec=dec * u.degree, frame="icrs")
    tile_coords = SkyCoord(
        ra=np.asarray(mosaics["ra"]) * u.degree,
        dec=np.asarray(mosaics["dec"]) * u.degree,
        frame="icrs",
    )
    idx, sep, _ = star_coord.match_to_catalog_sky(tile_coords)
    if sep > 0.5 * u.degree:
        return False, (
            f"closest {band_name} tile is {sep[0].to_value(u.degree):.2f}° away "
            f"— position likely outside Euclid Q1 coverage"
        )

    m = mosaics[int(idx)]
    file_path = f"{m['file_path']}/{m['file_name']}"
    try:
        Euclid.get_cutout(
            file_path=file_path,
            instrument=cfg.instrument,
            id=int(m["tile_index"]),
            coordinate=star_coord,
            radius=cutout_radius_arcmin * u.arcmin,
            output_file=output_file,
        )
    except Exception as e:
        return False, f"get_cutout failed: {type(e).__name__}: {e}"

    if not (os.path.exists(output_file) and os.path.getsize(output_file) > 0):
        return False, "downloaded file is empty"
    return True, None
