"""Collection registry feeding the unified client-side cutout viewer.

The viewer renders raw N-band float cubes in the browser. This module is the
server side of that contract (C6 in the web-console plan): it abstracts every
heterogeneous data source behind one tiny interface so the ``/viewer`` routes
don't care where pixels come from.

A *collection* is a named source of indexable cutouts. Each registered
collection provides:

* ``meta(params) -> dict`` — ``count``, ``tiers`` (``[{key, label, unit?,
  hidden?, disabled?}]``), ``default_tier``, ``band_names`` and ``objects``
  (per index: a stable ``id``, ``label``, ``ra``/``dec`` in degrees when the
  object is on the sky, and the tiers available for it when they differ).
* ``cube(index, tier, params) -> (ndarray (H, W, C) float32, info)`` where
  ``info`` carries ``label``, ``asinh``, ``pixscale`` and, when known,
  ``unit`` (``"e-"``, ``"MJy/sr"``, ``"ADU/s"``, ``"arb"``) and ``wcs`` (the
  compact celestial WCS keywords of *that tier's* pixel grid, FITS 1-based,
  axis 1 = column; see :func:`celestial_wcs_keywords`).

==================  ================================  ============================
collection          tiers                             source
==================  ================================  ============================
``sky``             dirty (LR), hr, bhr, sr           synthetic TFRecords
``cutouts``         real                              real star cutouts (FITS)
``evaluation``      LR / SR / HR / BHR / std / pcaN    eval-store object FITS
``ensemble``        lr, sr (production gate), mean,   evaluation cube cache +
                    std, hr, bhr, combiners, members  records
``archive-fields``  lr                                multipoint archive FITS
``real-field``      lr, sr (mean), combiners, …       cached 100-tile real field
``jwst-euclid``     lr, sr, jwst, jwst_blur           saved JWST × Euclid pairs
``nexus-field``     lr, sr, jwst, jwst_blur           NEXUS tiled field
``psfs``            VIS / Y_E / J_E / H_E             cached FASRC ePSF clusters
``real``            lr, jwst, m:<spec>                any real tile (``source``;
                                                      ``models`` = spec list)
==================  ================================  ============================

Band order is always ``Config.LR_INPUT_BAND_NAMES = (VIS, Y_E, J_E, H_E)``.
SR grids are 2× the LR grid: their WCS is the LR WCS magnified ×2
(``CD/2``, ``CRPIX → 2·CRPIX − 0.5``), see :func:`scaled_wcs_keywords`.
"""
from __future__ import annotations

import contextlib
import csv
import json
import math
import os
import re
import warnings
from collections import OrderedDict
from collections.abc import Callable, Mapping
from typing import Any, cast

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import gaussian_filter

from euclid_polish.config import Config
from euclid_polish.ensemble import pca_field
from euclid_polish.eval.combiner import (
    ACTIVE_COMBINER_KINDS,
    COMBINER_MODELS,
    RAW_INCREMENTAL_MINMEANMAX_RBF_KIND,
    load_combiner,
)
from euclid_polish.eval.ensemble_cube_cache import load_cached_field_lr
from euclid_polish.eval.spatial_gate import SPATIAL_GATE_KIND
from euclid_polish.image.tfio import read_images, tfrecord_path
from euclid_polish.psf.core import PSF
from euclid_polish.training.target_blur import (
    blur_target_array,
    validate_target_fwhm_arcsec,
)
from euclid_polish.web import job_config
from euclid_polish.web.helpers import (
    archive_fields,
    jwst_euclid,
    model_catalog,
    real_field,
    real_tiles,
    sky_records,
)
from euclid_polish.web.helpers.paths import _sky_records_local_dir
from euclid_polish.web.helpers.status import (
    _cached_fasrc_psf_dir,
    _cached_psf_clusters_json,
    _ensure_local_star_cutout,
    _record_count,
    _valid_4band_star_objects,
    _valid_4band_stars,
)

#: Band names + channel order shared by every collection.
BAND_NAMES: tuple[str, ...] = tuple(Config.LR_INPUT_BAND_NAMES)
BHR_FWHM_PARAM = "bhr_fwhm_arcsec"
BHR_FWHM_MAX_ARCSEC = float(Config.TARGET_PSF_FWHM_MAX_ARCSEC)


def _image_hdu(
    hdu: object,
) -> fits.PrimaryHDU | fits.ImageHDU | fits.CompImageHDU | None:
    """Narrow one heterogeneous HDUList member to an image HDU."""
    if isinstance(hdu, (fits.PrimaryHDU, fits.ImageHDU, fits.CompImageHDU)):
        return hdu
    return None


def _bhr_fwhm_arcsec(
    params: Mapping[str, str],
    default: float = Config.TARGET_PSF_FWHM_ARCSEC,
) -> float:
    """Validated viewer-controlled target-preview FWHM."""
    raw = params.get(BHR_FWHM_PARAM)
    try:
        fwhm = validate_target_fwhm_arcsec(
            default if raw is None else float(raw),
        )
    except (TypeError, ValueError) as exc:
        raise ViewerError(400, f"invalid {BHR_FWHM_PARAM}") from exc
    if fwhm > BHR_FWHM_MAX_ARCSEC:
        raise ViewerError(
            400,
            f"{BHR_FWHM_PARAM} must be <= {BHR_FWHM_MAX_ARCSEC:g}",
        )
    return fwhm


def receptive_field_constants() -> list[dict[str, Any]]:
    """Return the WDSR model receptive fields used by the shared viewer.

    The main WDSR path has a 3-pixel entry convolution, one 3-pixel
    convolution per residual block, and a 3-pixel reconstruction convolution.
    The two same-padded sides of each convolution add two input pixels to the
    receptive-field side, hence ``2 * blocks + 5``.  Store the angular side as
    well as the LR-pixel side so the client can keep the annotation stable
    while changing between LR/SR/HR tiles.
    """
    fields = []
    for blocks in (8, 16, 32):
        pixels = 2 * blocks + 5
        fields.append({
            "label": f"{blocks}b",
            "blocks": blocks,
            "pixels": pixels,
            "angular_side_arcsec": pixels * float(Config.VIS_PIXEL_SCALE_ARCSEC),
        })
    return fields


class ViewerError(Exception):
    """Raised by a collection loader; ``code`` maps to an HTTP status."""

    def __init__(self, code: int, message: str = ""):
        super().__init__(message or f"viewer error {code}")
        self.code = code


def color_constants() -> dict[str, Any]:
    """The per-band calibration constants the JS renderer needs.

    Sent once with every ``meta`` response so the browser-side colour math
    (AB-flux normalisation, solar balance, Planckian-locus temperature fit)
    is computed from the exact same numbers as ``visualization/color.py``.
    """
    bands = {}
    for name in BAND_NAMES:
        b = Config.get_band(name)
        bands[name] = {
            "t_total_s": float(b.t_total_s),
            "zeropoint_ab": float(b.zeropoint_ab_e_per_s),
            # Stack zeropoint (AB mag of 1 e⁻ over the full integration) —
            # precomputed server-side so the JS magnitude readout consumes the
            # SAME BandConfig.sim_zeropoint_e anchor as all Python photometry
            # instead of re-deriving it from zeropoint_ab + t_total.
            "zeropoint_ab_e_total": float(b.sim_zeropoint_e),
            "solar_ab_mag": float(Config.Color.SOLAR_AB_MAG[name]),
            "pivot_um": float(Config.Color.PIVOT_WAVELENGTH_UM[name]),
            "asinh_scale_e": float(b.asinh_stretch_scale_e),
        }
    return {
        "band_names": list(BAND_NAMES),
        "bands": bands,
        "rgb_scheme": list(Config.Color.RGB_SCHEMES["vis_nisp"]),  # [H_E,J_E,VIS]
        "default_asinh": float(Config.STRETCH_SCALE_E),
    }


def _as_hwc(arr: np.ndarray, *, layout: str = "auto") -> np.ndarray:
    """Normalise a FITS/record array to ``(H, W, C)`` float32.

    ``layout="chw"`` is the FITS cube convention (the band axis is NAXIS3,
    i.e. numpy axis 0) and ``"hwc"`` the record / ``.npy`` one; ``"auto"``
    treats a leading axis strictly shorter than both others as channel-first.
    Channel counts above four (multi-knee heads: ``(24, H, W)``) are kept,
    never misread as an image row. ``(H, W)`` becomes one channel.
    """
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 2:
        return a[..., None]
    if a.ndim != 3:
        raise ViewerError(415, f"expected 2-D/3-D array, got {a.shape}")
    if layout == "chw" or (
            layout == "auto" and a.shape[0] < a.shape[1] and a.shape[0] < a.shape[2]):
        return np.moveaxis(a, 0, -1)
    return a


# ---------------------------------------------------------------------------
# Celestial WCS + units (contract C6)
# ---------------------------------------------------------------------------

_WCS_KEYS = ("CTYPE1", "CTYPE2", "CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2",
             "CD1_1", "CD1_2", "CD2_1", "CD2_2")


def celestial_wcs_keywords(source: Any) -> dict[str, Any] | None:
    """Compact celestial WCS of a FITS header / astropy ``WCS``, or ``None``.

    Always the CD-matrix form (``CD = PC · CDELT`` when the source uses
    PC/CDELT), FITS 1-based pixel convention, axis 1 = column (x) — what
    ``X-Cube-WCS`` carries. ``None`` when the source has no celestial WCS.
    """
    if source is None:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wcs = source if isinstance(source, WCS) else WCS(source)
            wcs = wcs.celestial
        if not wcs.has_celestial or wcs.naxis != 2:
            return None
        matrix = np.asarray(wcs.pixel_scale_matrix, dtype=np.float64)
        ctype = [str(value) for value in wcs.wcs.ctype]
        if not all(ctype) or not np.all(np.isfinite(matrix)):
            return None
        return {
            "CTYPE1": ctype[0], "CTYPE2": ctype[1],
            "CRVAL1": float(wcs.wcs.crval[0]), "CRVAL2": float(wcs.wcs.crval[1]),
            "CRPIX1": float(wcs.wcs.crpix[0]), "CRPIX2": float(wcs.wcs.crpix[1]),
            "CD1_1": float(matrix[0, 0]), "CD1_2": float(matrix[0, 1]),
            "CD2_1": float(matrix[1, 0]), "CD2_2": float(matrix[1, 1]),
        }
    except Exception:  # noqa: BLE001 - heterogeneous archive headers
        return None


def shifted_wcs_keywords(keywords: Mapping[str, Any] | None, *,
                         dx: float, dy: float) -> dict[str, Any] | None:
    """WCS of a crop whose pixel (0, 0) is source pixel ``(dx, dy)`` (0-based)."""
    if keywords is None:
        return None
    out = dict(keywords)
    out["CRPIX1"] = float(out["CRPIX1"]) - float(dx)
    out["CRPIX2"] = float(out["CRPIX2"]) - float(dy)
    return out


def scaled_wcs_keywords(keywords: Mapping[str, Any] | None,
                        factor: int) -> dict[str, Any] | None:
    """WCS of a grid ``factor``× finer on the same footprint (SR = 2).

    ``CRPIX → factor·CRPIX − (factor − 1)/2`` and ``CD /= factor``: the
    ``factor²`` fine pixels exactly subdivide each coarse pixel (the
    pixel-shuffle geometry), matching :func:`scaled_wcs_header`.
    """
    if keywords is None:
        return None
    out = dict(keywords)
    offset = (factor - 1) / 2.0
    for key in ("CRPIX1", "CRPIX2"):
        out[key] = float(out[key]) * factor - offset
    for key in ("CD1_1", "CD1_2", "CD2_1", "CD2_2"):
        out[key] = float(out[key]) / factor
    return out


def _world_centre(keywords: Mapping[str, Any] | None, height: int,
                  width: int) -> tuple[float, float] | None:
    """``(ra, dec)`` of an image's centre under ``keywords``."""
    if keywords is None:
        return None
    try:
        wcs = WCS(fits.Header(dict(keywords)))
        ra, dec = wcs.pixel_to_world_values((width - 1) / 2.0, (height - 1) / 2.0)
    except Exception:  # noqa: BLE001
        return None
    ra, dec = float(ra) % 360.0, float(dec)
    return (ra, dec) if math.isfinite(ra) and math.isfinite(dec) else None


_UNIT_ALIASES = {
    "electron": "e-", "electrons": "e-", "e-": "e-", "e": "e-",
    "electron/pixel": "e-", "electrons/pixel": "e-", "e-/pixel": "e-",
    "mjy/sr": "MJy/sr", "mjysr-1": "MJy/sr", "mjy.sr-1": "MJy/sr",
    "adu/s": "ADU/s", "adu/sec": "ADU/s", "count/s": "ADU/s", "counts/s": "ADU/s",
}


def unit_from_header(header: Mapping[str, Any] | None, default: str = "arb") -> str:
    """The viewer unit of a FITS ``BUNIT`` (``e-``, ``MJy/sr``, ``ADU/s``)."""
    raw = "".join(str((header or {}).get("BUNIT") or "").strip().lower().split())
    return _UNIT_ALIASES.get(raw, default)


_HEADER_CACHE: OrderedDict[tuple[str, int, int], fits.Header | None] = OrderedDict()
_HEADER_CACHE_MAX = 256


def _fits_header(path: str | os.PathLike[str], hdu: int = 0) -> fits.Header | None:
    """One HDU's header without reading pixels (cached by path + mtime)."""
    try:
        stat = os.stat(path)
    except OSError:
        return None
    key = (os.fspath(path), int(stat.st_mtime_ns), int(hdu))
    if key in _HEADER_CACHE:
        _HEADER_CACHE.move_to_end(key)
        return _HEADER_CACHE[key]
    try:
        header: fits.Header | None = fits.getheader(path, hdu)
    except (OSError, IndexError, KeyError):
        header = None
    _HEADER_CACHE[key] = header
    if len(_HEADER_CACHE) > _HEADER_CACHE_MAX:
        _HEADER_CACHE.popitem(last=False)
    return header


_FILE_WCS_CACHE: OrderedDict[tuple[str, int, int], dict[str, Any] | None] = OrderedDict()


def _file_wcs(path: str | os.PathLike[str], hdu: int = 0) -> dict[str, Any] | None:
    """Compact WCS of one FITS HDU (cached by path + mtime; ``None`` if none)."""
    try:
        stat = os.stat(path)
    except OSError:
        return None
    key = (os.fspath(path), int(stat.st_mtime_ns), int(hdu))
    if key not in _FILE_WCS_CACHE:
        _FILE_WCS_CACHE[key] = celestial_wcs_keywords(_fits_header(path, hdu))
        if len(_FILE_WCS_CACHE) > _HEADER_CACHE_MAX:
            _FILE_WCS_CACHE.popitem(last=False)
    else:
        _FILE_WCS_CACHE.move_to_end(key)
    keywords = _FILE_WCS_CACHE[key]
    return dict(keywords) if keywords is not None else None


# ---------------------------------------------------------------------------
# sky — multi-band TFRecords (FASRC-synced cache)
# ---------------------------------------------------------------------------

# Tiers offered for sky records: LR (the dirty record), raw HR (the starfull
# scene), BHR (that scene with the target PSF), and SR (model output, generated
# on demand by the /sky button — disabled until at least one SR cube exists).
# The clean record is the deliberately starless target and must not be
# substituted for HR.
_SKY_RECORD_TIERS = [
    {"key": "dirty", "label": "LR", "unit": "e-"},
    {"key": "hr", "label": "HR", "unit": "e-"},
]


def _sky_subset(params: dict[str, str]) -> str:
    # Default to the held-out test split — the eval set the /sky sync pulls.
    subset = (params.get("subset") or "test").strip()
    if subset not in sky_records.SUBSETS:
        raise ViewerError(400, f"subset must be {'|'.join(sky_records.SUBSETS)}")
    return subset


def _sky_meta(params: dict[str, str]) -> dict[str, Any]:
    subset = _sky_subset(params)
    records_dir = _sky_records_local_dir()
    tiers: list[dict[str, Any]] = [
        dict(t) for t in _SKY_RECORD_TIERS
        if os.path.exists(tfrecord_path(records_dir, f"{t['key']}_{subset}"))
    ]
    counts = {t["key"]: (_record_count(f"{t['key']}_{subset}", records_dir) or 0)
              for t in tiers}
    if "hr" in counts:
        hr_position = next(i for i, tier in enumerate(tiers)
                           if tier["key"] == "hr")
        tiers.insert(hr_position + 1, {
            "key": "bhr", "label": "BHR (blurred HR)", "unit": "e-",
        })
        counts["bhr"] = counts["hr"]
    count = max(counts.values()) if counts else 0
    # SR is always offered so the user can see it exists; it's disabled until
    # the model has been run over the records (the "Generate SR" button).
    n_sr = sky_records.sr_count(subset)
    tiers.append({"key": "sr", "label": "SR", "disabled": n_sr == 0, "unit": "e-"})
    counts["sr"] = n_sr
    default = "dirty" if any(t["key"] == "dirty" for t in tiers) else (
        tiers[0]["key"] if tiers else "dirty")
    return {
        "count": count,
        "tiers": tiers,
        "default_tier": default,
        "band_names": list(BAND_NAMES),
        "tier_counts": counts,
        # Synthetic scenes: positional ids within the split, no sky position.
        "objects": [{"id": f"{subset}:{index}", "label": f"{subset} · idx {index}"}
                    for index in range(count)],
    }


def _sky_cube(index: int, tier: str, params: dict[str, str]):
    subset = _sky_subset(params)
    if tier == "sr":
        path = sky_records.sr_path(subset, index)
        if not os.path.isfile(path):
            raise ViewerError(404, "SR not generated for this record")
        cube = _as_hwc(np.load(path), layout="hwc")
        return cube, {
            "label": f"sr · {subset} · idx {index}",
            "asinh": float(Config.STRETCH_SCALE_E),
            "pixscale": float(Config.DEFAULT_PIXEL_SCALE),
            "unit": "e-",
        }
    if tier not in ("dirty", "hr", "bhr"):
        raise ViewerError(400, "bad tier")
    record_kind = "hr" if tier == "bhr" else tier
    path = tfrecord_path(_sky_records_local_dir(), f"{record_kind}_{subset}")
    if not os.path.exists(path):
        raise ViewerError(404, "records not synced")
    records = read_images(path, num_images=max(index + 1, 1))
    if not records or index >= len(records):
        raise ViewerError(404, "index out of range")
    rec = records[index]
    cube = _as_hwc(rec.data, layout="hwc")
    if tier == "bhr":
        cube = blur_target_array(
            cube, _bhr_fwhm_arcsec(params),
            pixel_scale_arcsec=rec.pixel_scale_arcsec,
        )
    label = "BHR (blurred HR)" if tier == "bhr" else tier
    info = {
        "label": f"{label} · {subset} · idx {rec.index}",
        "asinh": float(Config.STRETCH_SCALE_E),
        "pixscale": float(getattr(rec, "pixel_scale_arcsec", 0.0) or 0.0),
        "unit": "e-",
    }
    return cube, info


# ---------------------------------------------------------------------------
# cutouts — real Euclid stars valid in all 4 bands (per-band FITS, stacked)
# ---------------------------------------------------------------------------

def _cutouts_meta(params: dict[str, str]) -> dict[str, Any]:
    _size, stars = _valid_4band_star_objects(force=False)
    objects = []
    for star in stars:
        ra, dec = _finite_float(star.ra), _finite_float(star.dec)
        objects.append({
            "id": str(int(star.id)), "label": f"star {int(star.id)}",
            **({"ra": ra, "dec": dec} if ra is not None and dec is not None else {}),
        })
    return {
        "count": len(objects),
        # Raw archive cutouts (rate units, MAGZERO in the header).
        "tiers": [{"key": "real", "label": "Euclid", "unit": "ADU/s"}],
        "default_tier": "real",
        "band_names": list(BAND_NAMES),
        "objects": objects,
    }


def _read_fits_plane(path: str) -> np.ndarray:
    with fits.open(path, memmap=False) as hdul:
        for raw_hdu in hdul:
            hdu = _image_hdu(raw_hdu)
            if hdu is not None and hdu.data is not None and hdu.data.ndim == 2:
                return np.asarray(hdu.data, dtype=np.float32)
    raise ViewerError(415, "no 2-D plane in FITS")


def _cutouts_cube(index: int, tier: str, params: dict[str, str]):
    size, ids = _valid_4band_stars(force=False)
    if not ids or size is None:
        raise ViewerError(404, "no valid-in-4-bands stars")
    if index < 0 or index >= len(ids):
        raise ViewerError(404, "index out of range")
    sid = ids[index]
    planes = []
    vis_path = None
    for band in BAND_NAMES:
        path = _ensure_local_star_cutout(band, sid, size)
        if not path:
            raise ViewerError(404, f"{band} cutout unavailable")
        vis_path = vis_path or path
        planes.append(_read_fits_plane(path))
    shapes = {p.shape for p in planes}
    if len(shapes) != 1:
        raise ViewerError(415, f"band cutouts disagree in shape: {shapes}")
    cube = np.stack(planes, axis=-1)
    header = _fits_header(vis_path) if vis_path else None
    info = {
        "label": f"star {sid} · {size}px",
        "asinh": float(Config.STRETCH_SCALE_E),
        "pixscale": float(Config.VIS_PIXEL_SCALE_ARCSEC),
        # Every band is cut on the VIS grid, so the VIS WCS holds for all.
        "wcs": celestial_wcs_keywords(header),
        "unit": unit_from_header(header, default="ADU/s"),
    }
    return cube, info


# ---------------------------------------------------------------------------
# evaluation — per-object LR/SR/HR FITS in the shared eval store
# ---------------------------------------------------------------------------

_EVAL_TIER_FILES = {
    "LR": "original_stack.fits",
    "SR": "SR.fits",
    "HR": "HR.fits",
    "std": "std.fits",
}
def _read_eval_manifest(root: str) -> list[dict[str, str]]:
    """Rows of the shared eval store's ``manifest.csv`` (empty when absent)."""
    path = os.path.join(root, "manifest.csv")
    if not os.path.isfile(path):
        return []
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _eval_objects() -> list[dict[str, Any]]:
    """Return manifest objects with their available on-disk tiers and grade."""
    root = os.path.abspath(Config.EVAL_RESULTS_DIR)
    rows = _read_eval_manifest(root)
    objs: list[dict[str, Any]] = []
    for r in rows:
        if str(r.get("ok", "")).lower() != "true":
            continue
        sub = r.get("out_subdir") or r.get("id")
        if not sub:
            continue
        obj_dir = os.path.join(root, sub)
        tiers = [k for k, fn in _EVAL_TIER_FILES.items()
                 if os.path.isfile(os.path.join(obj_dir, fn))]
        # BHR is persisted by new synthetic evaluations and can be derived from
        # HR for older raw-target results, so it follows every available HR.
        if "HR" in tiers:
            tiers.insert(tiers.index("HR") + 1, "BHR")
        if not tiers:
            continue
        grade = (r.get("grade") or "").strip()
        pca_n, pca_amps, pca_var = 0, [], []
        dj = os.path.join(obj_dir, "disagreement.json")
        if os.path.isfile(dj):
            with contextlib.suppress(OSError, ValueError):
                with open(dj) as f:
                    _dmeta = json.load(f)
                pca_n = int(_dmeta.get("pca_n", 0) or 0)
                pca_amps = list(_dmeta.get("pca_amps", []) or [])
                pca_var = list(_dmeta.get("pca_var", []) or [])
        # The morph tier is client-animated (SR mean + pca cubes), so it has no
        # file of its own — advertise it per object or the viewer's per-object
        # availability gate keeps the movie chip disabled for every object.
        if pca_n > 0 and "SR" in tiers:
            tiers.append("morph")
        position = {}
        ra, dec = _finite_float(r.get("ra")), _finite_float(r.get("dec"))
        if ra is not None and dec is not None:
            position = {"ra": ra, "dec": dec}
        objs.append({
            "subdir": sub,
            "label": (f"{r.get('id', sub)}" + (f" · {grade}" if grade else "")),
            "grade": grade,
            "tiers": tiers,
            "pca_n": pca_n,
            "pca_amps": pca_amps,
            "pca_var": pca_var,
            **position,
        })
    return objs


def _eval_meta(params: dict[str, str]) -> dict[str, Any]:
    objs = _eval_objects()
    # All tiers seen across the run, ordered LR→SR→HR→BHR→std, for the chip strip.
    order = ["LR", "SR", "HR", "BHR", "std"]
    seen = {t for o in objs for t in o["tiers"]}
    tiers = [{"key": k, "label": ("stdSR" if k == "std" else k), "unit": "e-"}
             for k in order if k in seen]
    pca_n = max((int(o.get("pca_n", 0) or 0) for o in objs), default=0)
    pca_amps = [list(o.get("pca_amps", []) or []) for o in objs]
    pca_var = [list(o.get("pca_var", []) or []) for o in objs]
    if pca_n > 0:
        tiers.append({"key": "morph", "label": "disagreement movie"})
    default = "SR" if any(t["key"] == "SR" for t in tiers) else (
        tiers[0]["key"] if tiers else "SR")
    return {
        "count": len(objs),
        "tiers": tiers,
        "default_tier": default,
        "band_names": list(BAND_NAMES),
        "pca_n": pca_n,
        "pca_amps": pca_amps,
        "pca_var": pca_var,
        "objects": [{"id": str(o.get("id") or o["subdir"]), "label": o["label"],
                     "grade": o["grade"], "tiers": o["tiers"], "subdir": o["subdir"],
                     **{key: o[key] for key in ("ra", "dec") if key in o}}
                    for o in objs],
    }


def _eval_cube(index: int, tier: str, params: dict[str, str]):
    objs = _eval_objects()
    if index < 0 or index >= len(objs):
        raise ViewerError(404, "index out of range")
    obj = objs[index]
    root = os.path.abspath(Config.EVAL_RESULTS_DIR)
    asinh = float(Config.STRETCH_SCALE_E)
    key = None
    if tier.startswith("pca") and tier[3:].isdigit():
        path = os.path.join(root, obj["subdir"], f"{tier}.fits")
        if not os.path.isfile(path):
            raise ViewerError(404, f"{tier} not available for this object")
    else:
        # The shared morph animation fetches the ensemble mean as lower-case
        # "sr", but eval tier keys are upper-case (LR/SR/HR); resolve case-
        # insensitively so the disagreement movie works on the eval page too.
        key = next((k for k in (*_EVAL_TIER_FILES, "BHR")
                    if k.lower() == tier.lower()), None)
        if key is None or key not in obj["tiers"]:
            raise ViewerError(404, f"{tier} not available for this object")
        if key == "BHR":
            # Always derive the interactive BHR from raw HR. A persisted
            # BHR.fits records the evaluation target, but cannot respond to the
            # viewer slider.
            path = os.path.join(root, obj["subdir"], _EVAL_TIER_FILES["HR"])
        else:
            path = os.path.join(root, obj["subdir"], _EVAL_TIER_FILES[key])
    with fits.open(path, memmap=False) as hdul:
        primary = cast(fits.PrimaryHDU, hdul[0])
        data = primary.data
        header = primary.header
        with contextlib.suppress(TypeError, ValueError):
            asinh = float(cast(
                str | float, header.get("ASINH", asinh),
            ))
    if data is None:
        raise ViewerError(415, "primary FITS HDU contains no image")
    cube = _as_hwc(data, layout="chw")
    if key == "BHR":
        cube = blur_target_array(
            cube, _bhr_fwhm_arcsec(params),
            pixel_scale_arcsec=Config.DEFAULT_PIXEL_SCALE,
        )
    is_lr = tier.lower() in {"lr", "original", "original_stack"}
    tier_scale = (Config.VIS_PIXEL_SCALE_ARCSEC if is_lr
                  else Config.DEFAULT_PIXEL_SCALE)
    display_tier = "BHR (blurred HR)" if key == "BHR" else tier
    info = {"label": f"{obj['label']} · {display_tier}", "asinh": asinh,
            "pixscale": float(tier_scale),
            "unit": unit_from_header(header, default="e-"),
            **_eval_tier_wcs(os.path.join(root, obj["subdir"]), is_lr=is_lr,
                             synthetic="HR" in obj["tiers"])}
    if tier.startswith("pca"):
        info["unit"] = "arb"            # unit-norm eigen-images
    return cube, info


def _eval_tier_wcs(obj_dir: str, *, is_lr: bool, synthetic: bool) -> dict[str, Any]:
    """``{"wcs": …}`` for an eval tier: the LR cutout's WCS, or that WCS ×2
    for every SR-grid tier (SR / std / pcaN share the SR grid). Synthetic
    objects (they carry an HR truth) have no sky position."""
    if synthetic:
        return {}
    lr = _file_wcs(os.path.join(obj_dir, _EVAL_TIER_FILES["LR"]))
    wcs = lr if is_lr else scaled_wcs_keywords(lr, 2)
    return {"wcs": wcs} if wcs else {}


# ---------------------------------------------------------------------------
# ensemble — LR / SR (production gate) / mean / stdSR / HR disagreement viewer
# ---------------------------------------------------------------------------
#
# The /ensemble "Evaluate" job caches the ensemble-mean and per-pixel std
# (stdSR) cubes under <vis>/ensemble/<regime>/cubes/{sr,std}_<recidx>.npy plus a
# viz_index.json {subset, indices, member_labels}. LR/HR are read back from the
# sky records by record index. Contract C6: the public ``sr`` tier is the
# PRODUCTION combiner (``ACTIVE_COMBINER_KINDS[0]``, the spatial gate); the
# cached mean is its own ``mean`` tier (also the centre of the disagreement
# movie, ``morph_base_tier``); other loadable combiners (RBF kinds) stay as
# extra tiers. The regime defaults to STARFULL.

#: The combiner behind the ``sr`` tier.
PRODUCTION_COMBINER_KIND = ACTIVE_COMBINER_KINDS[0]
PRODUCTION_SR_LABEL = "SR · production gate"
MEAN_LABEL = "Mean of members"

_ENSEMBLE_TIERS = [
    {"key": "lr", "label": "LR", "unit": "e-"},
    {"key": "sr", "label": PRODUCTION_SR_LABEL, "unit": "e-"},
    {"key": "mean", "label": MEAN_LABEL, "unit": "e-"},
    # stdSR stays available (it powers the ±σ magnitude on the mean frame) but
    # is hidden from the chip row per the trimmed tier set.
    {"key": "std", "label": "stdSR", "hidden": True, "unit": "e-"},
    {"key": "hr", "label": "HR", "unit": "e-"},
    {"key": "bhr", "label": "BHR (blurred HR)", "unit": "e-"},
]

#: How many PCA components the on-the-fly (member-subset) disagreement movie
#: keeps — matches the baked ``ENSEMBLE_PCA_COMPONENTS``.
_MORPH_PCA_COMPONENTS = 3


def _ensemble_starless(params: dict[str, str]) -> bool:
    """The star regime the viewer is showing (``?mode=starfull|starless``).
    The two regimes' cubes are fully detached; STARFULL is the default (the
    production regime since 3aa5c86)."""
    return (params.get("mode", "starfull") or "starfull").lower() == "starless"


def _ensemble_target(starless: bool) -> tuple[str, str]:
    """Return the record kind and viewer label for one ensemble regime."""
    return (("clean", "Clean (starless goal)")
            if starless else ("hr", "HR"))


def _blurred_target_label(target_label: str) -> str:
    """Viewer label for the PSF-stabilised form of a raw target."""
    source = "Clean" if target_label.startswith("Clean") else "HR"
    return f"BHR (blurred {source})"


def _ensemble_cubes_dir(starless: bool) -> str:
    regime = "starless" if starless else "starfull"
    return os.path.join(Config.VIS_DIR, "ensemble", regime, "cubes")


def _ensemble_manifest(starless: bool) -> dict[str, Any]:
    p = os.path.join(_ensemble_cubes_dir(starless), "viz_index.json")
    if os.path.isfile(p):
        with contextlib.suppress(OSError, ValueError), open(p) as f:
            return json.load(f)
    return {"subset": "", "indices": []}


def _combiner_available(starless: bool, man: Mapping[str, Any], kind: str) -> bool:
    """A combiner tier is offered when its cube is baked or it loads for the
    cached membership (computed on the fly then)."""
    labels = man.get("member_labels", []) or []
    return bool(man.get(f"has_combiner_{kind}")
                or (labels and _load_field_combiner(starless, labels, kind) is not None))


def _ensemble_meta(params: dict[str, str]) -> dict[str, Any]:
    starless = _ensemble_starless(params)
    target_kind, target_label = _ensemble_target(starless)
    man = _ensemble_manifest(starless)
    idxs = man.get("indices", [])
    sub = man.get("subset", "")
    rdir = _sky_records_local_dir()
    has_target = bool(sub) and bool(rdir) and os.path.exists(
        tfrecord_path(rdir, f"{target_kind}_{sub}"))
    target_labels = {
        "hr": target_label,
        "bhr": _blurred_target_label(target_label),
    }
    tiers = [({**t, "label": target_labels[t["key"]]}
              if t["key"] in target_labels else dict(t))
             for t in _ENSEMBLE_TIERS
             if t["key"] not in target_labels or has_target]
    production = _combiner_available(starless, man, PRODUCTION_COMBINER_KIND)
    has_mean = bool(idxs) and os.path.isfile(os.path.join(
        _ensemble_cubes_dir(starless), f"sr_{int(idxs[0]):05d}.npy"))
    if not production:
        tiers = [tier for tier in tiers if tier["key"] != "sr"]
    if not has_mean:
        tiers = [tier for tier in tiers if tier["key"] != "mean"]
    # Every other active combiner (the RBF kinds) gets its own selectable
    # tier after the mean, computed on demand when not baked by the eval.
    position = 1 + max((i for i, tier in enumerate(tiers)
                        if tier["key"] in {"mean", "sr"}), default=0)
    for kind in reversed(ACTIVE_COMBINER_KINDS[1:]):
        if _combiner_available(starless, man, kind):
            spec = COMBINER_MODELS[kind]
            tiers.insert(position, {"key": spec.cube_prefix,
                                    "label": f"SR · {spec.label}", "unit": "e-"})
    # Individual member SR tiers, labelled from the eval. HIDDEN from the tier
    # chip row (they'd swamp it at 22 members) but still loadable on demand:
    # the React member panel searches/sorts them and toggles one in via the
    # engine's setTiers, and their cubes feed the member-subset movie.
    member_labels = man.get("member_labels", []) or []
    tiers += [{"key": f"member{i}", "label": f"SR {lab}", "hidden": True,
               "unit": "e-"}
              for i, lab in enumerate(member_labels)]
    # PCA disagreement basis for the morphing animation: per-field amplitudes
    # (population std the members span along each component), aligned to the
    # viewer index order. The pcaN cubes are fetched on demand (not listed as
    # static tiers). JSON keys are strings.
    pca_n = int(man.get("pca_n", 0) or 0)
    amps_by = man.get("pca_amps", {}) or {}
    pca_amps = [list(amps_by.get(str(int(i)), [])) for i in idxs]
    # The morphing "disagreement movie" is a client-animated tier (mean + PCA
    # components); the viewer special-cases it (no fetchable cube).
    if pca_n > 0:
        tiers.append({"key": "morph", "label": "disagreement movie"})
    default = ("sr" if production else "mean" if has_mean else "lr")
    return {
        "count": len(idxs),
        "tiers": tiers,
        "default_tier": default,
        "band_names": list(BAND_NAMES),
        "subset": sub,
        "regime": "starless" if starless else "starfull",
        "pca_n": pca_n,
        "pca_amps": pca_amps,
        # The movie animates mean + Σ amp·PC: its centre is the mean tier.
        "morph_base_tier": "mean",
        "production_combiner": PRODUCTION_COMBINER_KIND,
        # Member index → label, for the React panel to join psnr/loss/depth
        # (from status.json) and drive the member-subset disagreement movie.
        "member_labels": list(member_labels),
        "pca_max": _MORPH_PCA_COMPONENTS,
        # Synthetic test fields: ids are split:record-index, no sky position.
        "objects": [{"id": f"{sub}:{int(i)}", "label": f"{sub} · idx {int(i)}"}
                    for i in idxs],
    }


def _ensemble_record_cube(sub: str, n_read: int, kind: str, rec_index: int,
                          *, blurred_fwhm_arcsec: float | None = None):
    """LR/goal record matched by ``.index`` (dirty, clean, or hr)."""
    rdir = _sky_records_local_dir()
    path = tfrecord_path(rdir, f"{kind}_{sub}") if rdir else ""
    if not rdir or not os.path.exists(path):
        raise ViewerError(404, f"{kind} records not available")
    recs = read_images(path, num_images=max(n_read, 1))
    rec = {r.index: r for r in recs}.get(rec_index)
    if rec is None:
        raise ViewerError(404, f"record {rec_index} not found")
    data = _as_hwc(rec.data, layout="hwc")
    if blurred_fwhm_arcsec is not None:
        data = blur_target_array(
            data, blurred_fwhm_arcsec,
            pixel_scale_arcsec=rec.pixel_scale_arcsec,
        )
    return (data,
            float(getattr(rec, "pixel_scale_arcsec", 0.0) or 0.0))


# On-the-fly member-subset PCA. The disagreement movie normally decomposes ALL
# members' variation about the mean (baked pca0…N cubes). When the viewer asks
# for a SUBSET (``?members=0,3,7``) we recompute PCA over just those members
# from their cached ``member{i}`` cubes — the SVD of a k-row residual matrix is
# tens of ms, so this is fully interactive. A tiny LRU lets the sr + pca0…N
# fetches for one frame share a single SVD.
_SUBSET_PCA_CACHE: OrderedDict[tuple, tuple] = OrderedDict()
_SUBSET_PCA_MAX = 8


def _parse_member_subset(raw: str | None, n_members: int) -> list[int] | None:
    """``"0,3,7"`` → sorted unique valid member indices, or ``None`` (all)."""
    if not raw or not str(raw).strip():
        return None
    out: list[int] = []
    for tok in str(raw).split(","):
        tok = tok.strip()
        if tok.isdigit():
            i = int(tok)
            if 0 <= i < n_members and i not in out:
                out.append(i)
    # A single member has no residual subspace — treat as "no subset".
    return sorted(out) if len(out) >= 2 else None


def _subset_pca(starless: bool, rec_index: int, subset: list[int]):
    """``(mean, components, amplitudes, var_explained)`` of the member-subset
    residuals for one field, from the cached ``member{i}`` cubes. LRU-cached."""
    key = ("starless" if starless else "starfull", int(rec_index), tuple(subset))
    hit = _SUBSET_PCA_CACHE.get(key)
    if hit is not None:
        _SUBSET_PCA_CACHE.move_to_end(key)
        return hit
    cdir = _ensemble_cubes_dir(starless)
    stack = []
    for i in subset:
        p = os.path.join(cdir, f"member{i}_{int(rec_index):05d}.npy")
        if not os.path.isfile(p):
            raise ViewerError(404, f"member{i} cube missing")
        stack.append(np.load(p).astype(np.float32))
    res = pca_field(np.stack(stack, axis=0), n_components=_MORPH_PCA_COMPONENTS)
    _SUBSET_PCA_CACHE[key] = res
    if len(_SUBSET_PCA_CACHE) > _SUBSET_PCA_MAX:
        _SUBSET_PCA_CACHE.popitem(last=False)
    return res


def _ensemble_regime_dir(starless: bool) -> str:
    """Regime root (parent of ``cubes/``) — where ``combiner/`` lives."""
    return os.path.dirname(_ensemble_cubes_dir(starless))


def _load_field_combiner(starless: bool, member_labels: list[str],
                         model_kind: str = PRODUCTION_COMBINER_KIND):
    """The regime's fitted combiner if it exists AND its membership matches the
    cube stack (``member_labels``), else ``None``. Cheap (a small artifact)."""
    if not member_labels:
        return None
    try:
        return load_combiner(_ensemble_regime_dir(starless),
                             member_labels=list(member_labels),
                             artifact_dir=COMBINER_MODELS[model_kind].artifact_dir)
    except Exception:
        return None


# On-the-fly combiner reconstruction (per field), so the "combiner" tier shows
# whenever a combiner is fitted — not only when the last eval baked comb_ cubes.
_COMB_CUBE_CACHE: OrderedDict[tuple, np.ndarray] = OrderedDict()
_COMB_CUBE_MAX = 8


def _combiner_field_cube(starless: bool, rec_index: int,
                         member_labels: list[str],
                         model_kind: str = PRODUCTION_COMBINER_KIND) -> np.ndarray:
    """The combiner reconstruction ``(H,W,C)`` for one field, applied to the
    cached full member stack. LRU-cached; raises 404 if no combiner / cubes."""
    key = ("starless" if starless else "starfull", int(rec_index),
           tuple(member_labels), model_kind)
    hit = _COMB_CUBE_CACHE.get(key)
    if hit is not None:
        _COMB_CUBE_CACHE.move_to_end(key)
        return hit
    comb = _load_field_combiner(starless, member_labels, model_kind)
    if comb is None:
        raise ViewerError(404, "no combiner for this regime")
    cdir = _ensemble_cubes_dir(starless)
    stack = []
    for i in range(len(member_labels)):
        p = os.path.join(cdir, f"member{i}_{int(rec_index):05d}.npy")
        if not os.path.isfile(p):
            raise ViewerError(404, f"member{i} cube missing")
        stack.append(np.load(p).astype(np.float32))
    lr = None
    if getattr(comb, "use_lr", False):
        try:
            with open(os.path.join(cdir, "viz_index.json")) as handle:
                subset = str(json.load(handle).get("subset", "test"))
        except (OSError, ValueError):
            subset = "test"
        lr = load_cached_field_lr(cdir, int(rec_index),
                                  records_dir=_sky_records_local_dir(), subset=subset)
        if lr is None:
            raise ViewerError(404, "LR input for the spatial gate is missing")
    out = np.asarray(comb.apply_field(np.stack(stack, axis=0), lr=lr), np.float32)
    _COMB_CUBE_CACHE[key] = out
    if len(_COMB_CUBE_CACHE) > _COMB_CUBE_MAX:
        _COMB_CUBE_CACHE.popitem(last=False)
    return out


def _ensemble_cube(index: int, tier: str, params: dict[str, str]):
    starless = _ensemble_starless(params)
    target_kind, target_label = _ensemble_target(starless)
    man = _ensemble_manifest(starless)
    idxs = man.get("indices", [])
    sub = man.get("subset", "")
    if index < 0 or index >= len(idxs):
        raise ViewerError(404, "index out of range")
    rec_index = int(idxs[index])

    # Member-subset disagreement movie: recompute the mean (``mean``, and
    # ``sr`` for the movie engine that fetches ``sr`` as its centre) and the
    # PCA eigen-images on the fly for the requested members. A combiner fitted
    # for the full ordered membership cannot take fewer member channels, so
    # a subset always means the subset mean. amp/var are subset-dependent →
    # returned as headers so the animation reads the right spread.
    is_pca = tier.startswith("pca") and tier[3:].isdigit()
    subset = _parse_member_subset(
        params.get("members"), len(man.get("member_labels", []) or []))
    if subset is not None and (tier in ("sr", "mean") or is_pca):
        mean, comps, amps, var = _subset_pca(starless, rec_index, subset)
        tag = f"{len(subset)} of {len(man.get('member_labels', []) or [])} members"
        if not is_pca:
            return _as_hwc(mean, layout="hwc"), {
                "label": f"SR (subset mean · {tag}) · {sub} · idx {rec_index}",
                "asinh": float(Config.STRETCH_SCALE_E),
                "pixscale": float(Config.DEFAULT_PIXEL_SCALE), "unit": "e-"}
        k = int(tier[3:])
        if k >= len(comps):
            raise ViewerError(404, "pca component out of range")
        return _as_hwc(comps[k], layout="hwc"), {
            "label": f"PC{k} · {tag}", "asinh": float(Config.STRETCH_SCALE_E),
            "pixscale": float(Config.DEFAULT_PIXEL_SCALE), "amp": float(amps[k]),
            "var": float(var[k]) if k < len(var) else 0.0, "unit": "arb"}
    # ``sr`` is the production combiner (C6); every combiner tier prefers a
    # baked model-specific cube, else applies the fitted model to the cached
    # member stack on the fly.
    tier_kinds = {COMBINER_MODELS[kind].cube_prefix: kind
                  for kind in ACTIVE_COMBINER_KINDS}
    tier_kinds["sr"] = PRODUCTION_COMBINER_KIND
    if tier in tier_kinds:
        model_kind = tier_kinds[tier]
        prefix = COMBINER_MODELS[model_kind].cube_prefix
        baked = os.path.join(_ensemble_cubes_dir(starless),
                             f"{prefix}_{rec_index:05d}.npy")
        cube = (_as_hwc(np.load(baked), layout="hwc") if os.path.isfile(baked)
                else _as_hwc(_combiner_field_cube(
                    starless, rec_index, man.get("member_labels", []) or [], model_kind),
                    layout="hwc"))
        label = (PRODUCTION_SR_LABEL if tier == "sr"
                 else f"SR · {COMBINER_MODELS[model_kind].label}")
        return cube, {"label": f"{label} · {sub} · idx {rec_index}",
                      "asinh": float(Config.STRETCH_SCALE_E),
                      "pixscale": float(Config.DEFAULT_PIXEL_SCALE), "unit": "e-"}
    # The cached ensemble mean is stored as ``sr_<rec>.npy`` by the evaluation.
    if tier == "mean":
        path = os.path.join(_ensemble_cubes_dir(starless), f"sr_{rec_index:05d}.npy")
        if not os.path.isfile(path):
            raise ViewerError(404, "mean cube missing")
        return _as_hwc(np.load(path), layout="hwc"), {
            "label": f"{MEAN_LABEL} · {sub} · idx {rec_index}",
            "asinh": float(Config.STRETCH_SCALE_E),
            "pixscale": float(Config.DEFAULT_PIXEL_SCALE), "unit": "e-"}
    # Records are written index==position from 0, so reading up to the largest
    # cached index covers every LR/goal field we need.
    n_read = (max(int(i) for i in idxs) + 1) if idxs else 1
    # sr / std, the PCA eigen-images (pca0…) and individual member SRs
    # (member0…) are cached .npy cubes; LR and the regime goal come from the
    # records. pcaN are served on demand for the animation (not advertised as
    # static tiers). The stable ``hr`` tier key means "goal" here: clean for
    # starless and hr for starfull.
    is_npy = (tier == "std"
              or (tier.startswith("pca") and tier[3:].isdigit())
              or (tier.startswith("member") and tier[6:].isdigit()))
    if is_npy:
        path = os.path.join(_ensemble_cubes_dir(starless),
                            f"{tier}_{rec_index:05d}.npy")
        if not os.path.isfile(path):
            raise ViewerError(404, f"{tier} cube missing")
        cube, pix = _as_hwc(np.load(path), layout="hwc"), float(Config.DEFAULT_PIXEL_SCALE)
    elif tier == "lr":
        cube, pix = _ensemble_record_cube(sub, n_read, "dirty", rec_index)
    elif tier in {"hr", "bhr"}:
        cube, pix = _ensemble_record_cube(
            sub, n_read, target_kind, rec_index,
            blurred_fwhm_arcsec=(
                _bhr_fwhm_arcsec(params) if tier == "bhr" else None
            ),
        )
    else:
        raise ViewerError(400, "bad tier")
    labels = {"lr": "LR", "std": "stdSR (member std)", "hr": target_label,
              "bhr": _blurred_target_label(target_label)}
    if tier.startswith("member") and tier[6:].isdigit():
        mlabels = man.get("member_labels", []) or []
        mi = int(tier[6:])
        label = f"SR {mlabels[mi]}" if mi < len(mlabels) else tier
    else:
        label = labels.get(tier, tier)
    info = {"label": f"{label} · {sub} · idx {rec_index}",
            "asinh": float(Config.STRETCH_SCALE_E), "pixscale": pix,
            "unit": "arb" if is_pca else "e-"}
    # Baked full-ensemble PCA: surface the per-field amplitude/variance from the
    # manifest so the client reads amps from the cube header uniformly (subset
    # and full paths alike), not a separate meta lookup.
    if is_pca:
        k = int(tier[3:])
        amp = (man.get("pca_amps", {}) or {}).get(str(rec_index), [])
        var = (man.get("pca_var", {}) or {}).get(str(rec_index), [])
        if k < len(amp):
            info["amp"] = float(amp[k])
        if k < len(var):
            info["var"] = float(var[k])
    return cube, info


# ---------------------------------------------------------------------------
# archive-fields — shared independent-pointing four-band archive samples
# ---------------------------------------------------------------------------

def _archive_fields_meta(_params: dict[str, str]) -> dict[str, Any]:
    status = archive_fields.availability()
    # This collection is the real side of the Synthetic–Real comparison, so it
    # offers only tiles that were not steered away from bright stars.
    fields = (
        list(archive_fields.iter_comparison_fields()) if status["ready"] else []
    )
    tier = {"key": "lr", "label": "Archive LR", "unit": "e-"}
    objects = []
    for field in fields:
        label_field = archive_fields.position_field(field)
        objects.append({
            "id": str(field.sample_id),
            "label": (
                f"{label_field} · pointing {field.source_sample_id + 1} · "
                f"{field.position_name} · sample {field.sample_id + 1}"
            ),
            "tiers": ["lr"],
            "sample_id": field.sample_id,
            "source_sample_id": field.source_sample_id,
            "parent_id": field.parent_id,
            # Position-derived label; the manifest's own string is kept.
            "field": label_field,
            "stored_field": field.field,
            "ra": field.ra,
            "dec": field.dec,
            "position_name": field.position_name,
        })
    return {
        "count": len(fields),
        "tiers": [tier],
        "default_tier": "lr",
        "band_names": list(BAND_NAMES),
        "archive": status,
        "objects": objects,
    }


def _archive_fields_cube(index: int, tier: str, _params: dict[str, str]):
    if tier.lower() != "lr":
        raise ViewerError(400, "archive fields provide only the LR tier")
    status = archive_fields.availability()
    if not status["ready"]:
        raise ViewerError(404, "multipoint archive collection is unavailable")
    fields = list(archive_fields.iter_comparison_fields())
    if index < 0 or index >= len(fields):
        raise ViewerError(404, "archive sample index out of range")
    field = fields[index]
    try:
        cube = archive_fields.load_field(field)
    except archive_fields.ArchiveFieldError as exc:
        raise ViewerError(415, str(exc)) from exc
    return cube, {
        "label": (
            f"Archive LR · {archive_fields.position_field(field)} · pointing "
            f"{field.source_sample_id + 1} · {field.position_name} · "
            f"sample {field.sample_id + 1}"
        ),
        "asinh": float(Config.STRETCH_SCALE_E),
        "pixscale": float(Config.VIS_PIXEL_SCALE_ARCSEC),
        "bands": list(BAND_NAMES),
        "transfer_group": "euclid",
        # load_field converts every band to electrons on the VIS grid.
        "unit": "e-",
        "wcs": _file_wcs(field.path, 1),
    }


# ---------------------------------------------------------------------------
# real-field — legacy ad-hoc single-pointing inference workspace
# ---------------------------------------------------------------------------

def _real_field_manifest(params: dict[str, str]) -> dict[str, Any]:
    identifier = (params.get("field") or "").strip()
    if identifier:
        try:
            with real_field.manifest_path(identifier).open() as f:
                return json.load(f)
        except (OSError, ValueError):
            raise ViewerError(404, "real field not cached") from None
    manifest = real_field.latest_field()
    if manifest is None:
        raise ViewerError(404, "no real Euclid field cached")
    return manifest


def _real_field_geometry(manifest: Mapping[str, Any]) -> tuple[int, int]:
    """``(tile_size, grid_side)`` of a cached field (LR pixels, tiles/side)."""
    tile = int(manifest.get("tile_size", real_field.TILE_SIZE) or real_field.TILE_SIZE)
    side = int(manifest.get("grid_side", real_field.GRID_SIDE) or real_field.GRID_SIDE)
    return tile, side


def _real_field_tile_wcs(manifest: Mapping[str, Any], index: int) -> dict[str, Any] | None:
    """LR WCS of one tile: the field's ``original_stack.fits`` WCS shifted by
    the tile's pixel offset (tiles are ``tile_size`` squares, row-major)."""
    field = _file_wcs(real_field.field_dir(str(manifest["field_id"]))
                      / "original_stack.fits")
    tile, side = _real_field_geometry(manifest)
    row, col = divmod(int(index), side)
    return shifted_wcs_keywords(field, dx=col * tile, dy=row * tile)


def _real_field_meta(params: dict[str, str]) -> dict[str, Any]:
    manifest = _real_field_manifest(params)
    labels = list(manifest.get("member_labels", []) or [])
    tiers = [
        {"key": "lr", "label": "LR", "unit": "e-"},
        {"key": "sr", "label": "SR (mean)", "unit": "e-"},
        {"key": "std", "label": "stdSR", "hidden": True, "unit": "e-"},
    ]
    for kind, spec in COMBINER_MODELS.items():
        if kind not in ACTIVE_COMBINER_KINDS:
            continue
        if kind in set(manifest.get("combiner_kinds", []) or []):
            tiers.append({"key": spec.cube_prefix, "label": spec.label, "unit": "e-"})
    tiers += [{"key": f"member{i}", "label": f"SR {label}", "hidden": True,
               "unit": "e-"}
              for i, label in enumerate(labels)]
    if int(manifest.get("pca_n", 0) or 0) > 0:
        tiers.append({"key": "morph", "label": "disagreement movie"})
    count = int(manifest.get("count", 0) or 0)
    tile, side = _real_field_geometry(manifest)
    identifier = str(manifest.get("field_id", ""))
    objects = []
    for i in range(count):
        centre = _world_centre(_real_field_tile_wcs(manifest, i), tile, tile)
        objects.append({
            "id": f"{identifier}/{i:03d}",
            "label": f"tile {i + 1:03d} · row {i // side + 1}, col {i % side + 1}",
            "tiers": [t["key"] for t in tiers],
            **({"ra": centre[0], "dec": centre[1]} if centre else {}),
        })
    return {
        "count": count, "tiers": tiers, "default_tier": "sr",
        "band_names": list(BAND_NAMES), "member_labels": labels,
        "pca_n": int(manifest.get("pca_n", 0) or 0),
        "pca_amps": [list((manifest.get("pca_amps", {}) or {}).get(str(i), []))
                     for i in range(count)],
        "pca_var": [list((manifest.get("pca_var", {}) or {}).get(str(i), []))
                    for i in range(count)],
        "objects": objects,
    }


def _real_field_cube(index: int, tier: str, params: dict[str, str]):
    manifest = _real_field_manifest(params)
    count = int(manifest.get("count", 0) or 0)
    if index < 0 or index >= count:
        raise ViewerError(404, "tile index out of range")
    path = (real_field.field_dir(str(manifest["field_id"])) / "cubes"
            / f"{tier}_{index:03d}.npy")
    if not path.is_file():
        raise ViewerError(404, f"{tier} cube is not cached")
    cube = _as_hwc(np.load(path), layout="hwc")
    labels = list(manifest.get("member_labels", []) or [])
    if tier.startswith("member") and tier[6:].isdigit():
        mi = int(tier[6:])
        label = f"SR {labels[mi]}" if mi < len(labels) else tier
    elif tier == "sr":
        label = "SR (STARFULL mean)"
    elif tier == "std":
        label = "stdSR (STARFULL members)"
    elif tier == "lr":
        label = "LR"
    else:
        label = next((COMBINER_MODELS[kind].label for kind in ACTIVE_COMBINER_KINDS
                      if COMBINER_MODELS[kind].cube_prefix == tier), tier)
    is_lr = tier.lower() == "lr"
    tier_scale = (Config.VIS_PIXEL_SCALE_ARCSEC if is_lr
                  else Config.DEFAULT_PIXEL_SCALE)
    lr_wcs = _real_field_tile_wcs(manifest, index)
    return cube, {"label": f"{label} · tile {index + 1:03d}",
                  "asinh": float(Config.STRETCH_SCALE_E),
                  "pixscale": float(tier_scale),
                  "unit": "arb" if tier.startswith("pca") else "e-",
                  "wcs": lr_wcs if is_lr else scaled_wcs_keywords(lr_wcs, 2)}


# ---------------------------------------------------------------------------
# psfs — one navigable object per spatial ePSF cluster, one tier per band
# ---------------------------------------------------------------------------

def _psf_paths() -> dict[str, str]:
    """Return the already-synchronised FASRC ePSF paths by band."""
    psf_dir = _cached_fasrc_psf_dir()
    if not psf_dir:
        return {}
    return {
        band.name: os.path.join(psf_dir, band.psf_fits_filename)
        for band in Config.BANDS
        if os.path.isfile(os.path.join(psf_dir, band.psf_fits_filename))
    }


def _psf_count(path: str) -> int:
    """Read the cluster count from FITS headers without materialising pixels."""
    with fits.open(path, memmap=True) as hdul:
        primary = cast(fits.PrimaryHDU, hdul[0])
        header_count = primary.header.get("NPSF")
        if header_count is not None:
            return max(1, int(cast(str | int, header_count)))
        image_hdus = [h for h in hdul if getattr(h, "data", None) is not None]
        return max(1, len(image_hdus) - 1) if len(image_hdus) > 1 else 1


def _psf_cluster_positions() -> list[dict[str, Any]]:
    """Cluster centroids from the synced metadata sidecar (``[]`` if absent)."""
    path = _cached_psf_clusters_json()
    if not path:
        return []
    try:
        with open(path) as handle:
            clusters = json.load(handle).get("clusters", [])
    except (OSError, ValueError, AttributeError):
        return []
    return [cluster if isinstance(cluster, dict) else {} for cluster in clusters]


def _psf_meta(_params: dict[str, str]) -> dict[str, Any]:
    paths = _psf_paths()
    counts = {name: _psf_count(path) for name, path in paths.items()}
    count = max(counts.values(), default=0)
    tiers = [
        {"key": name, "label": name, "disabled": name not in counts, "unit": "arb"}
        for name in BAND_NAMES
    ]
    positions = _psf_cluster_positions()
    objects = []
    for index in range(count):
        position = positions[index] if index < len(positions) else {}
        ra, dec = _finite_float(position.get("ra")), _finite_float(position.get("dec"))
        objects.append({
            "id": f"cluster-{index + 1:03d}",
            "label": f"PSF cluster {index + 1:03d}",
            "tiers": [name for name, n in counts.items() if index < n],
            **({"ra": ra, "dec": dec} if ra is not None and dec is not None else {}),
        })
    return {
        "count": count,
        "tiers": tiers,
        "default_tier": next(iter(counts), BAND_NAMES[0]),
        "band_names": list(BAND_NAMES),
        "objects": objects,
        "source": "FASRC cache",
        "render_mode": "log",
        "empty_label": "No synchronised FASRC PSFs are available.",
    }


def _psf_preview_warp_settings() -> tuple[float, float]:
    """Current persisted training warp ``(alpha_max, sigma)`` for the demo."""
    cfg = job_config.load()
    return float(cfg.psf_warp_alpha_max), float(cfg.psf_warp_sigma)


def _psf_cube(index: int, tier: str, params: dict[str, str]):
    if tier not in BAND_NAMES:
        raise ViewerError(400, f"bad PSF band: {tier}")
    path = _psf_paths().get(tier)
    if path is None:
        raise ViewerError(404, f"{tier} PSF not synchronised")

    with fits.open(path, memmap=True) as hdul:
        image_hdus = [
            image_hdu for raw_hdu in hdul
            if (image_hdu := _image_hdu(raw_hdu)) is not None
            and image_hdu.data is not None
        ]
        cluster_hdus = image_hdus[1:] if len(image_hdus) > 1 else image_hdus
        if index < 0 or index >= len(cluster_hdus):
            raise ViewerError(404, "PSF cluster out of range")
        hdu = cluster_hdus[index]
        data = np.asarray(hdu.data, dtype=np.float32).copy()
        header = hdu.header
        pixel_scale = float(cast(
            str | float,
            header.get("PXSCALE", header.get("PIXSCALE", 0.0)),
        ))
        n_stars = header.get("NSTARS")
        label = f"{tier} PSF · cluster {index + 1:03d}"
        if n_stars is not None:
            label += f" · {int(cast(str | int, n_stars)):,} stars"

    if params.get("psf_warp") == "1":
        try:
            preview_seed = int(params.get("psf_warp_seed", "0"))
        except ValueError as exc:
            raise ViewerError(400, "psf_warp_seed must be an integer") from exc
        if preview_seed < 0 or preview_seed > np.iinfo(np.uint32).max:
            raise ViewerError(400, "psf_warp_seed must be a uint32")

        # Derive both draws from the visible sample seed.  The same request
        # parameters therefore reproduce the same alpha + displacement field
        # in every band, exactly like one shared training PSFSample.
        alpha_max, sigma = _psf_preview_warp_settings()
        if alpha_max < 0.0 or sigma <= 0.0:
            raise ViewerError(400, "invalid persisted PSF warp settings")
        rng = np.random.default_rng(preview_seed)
        alpha = float(rng.uniform(0.0, alpha_max))
        warp_seed = int(rng.integers(
            0, np.iinfo(np.uint32).max, dtype=np.uint32,
        ))
        data = PSF(
            data=data,
            pixel_scale=pixel_scale,
        ).elastic_warp(
            alpha,
            sigma,
            seed=warp_seed,
        ).data
        # This label travels in ``X-Cube-Label``.  Werkzeug's development
        # server serialises HTTP headers as Latin-1, so keep parameter names
        # ASCII even though the React control can safely render Greek symbols.
        label += f" · warped alpha={alpha:.1f}, sigma={sigma:g} px"
    return _as_hwc(data), {
        "label": label,
        "asinh": float(Config.STRETCH_SCALE_E),
        "pixscale": pixel_scale,
        "unit": "arb",                    # normalised kernel, not on the sky
    }


# ---------------------------------------------------------------------------
# jwst-euclid — one saved paired field, preserving each native image grid
# ---------------------------------------------------------------------------

_PAIR_ID = re.compile(r"^[A-Za-z0-9._-]{1,220}$")


def _jwst_euclid_pair(params: dict[str, str]) -> tuple[dict[str, Any], str]:
    """Load a verified paired-field manifest without exposing its cache path."""
    identifier = (params.get("field") or "").strip()
    if not _PAIR_ID.fullmatch(identifier):
        raise ViewerError(404, "paired field not found")
    directory = jwst_euclid.pair_root() / identifier
    try:
        with (directory / "manifest.json").open(encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, ValueError):
        raise ViewerError(404, "paired field not found") from None
    if (not isinstance(manifest, dict)
            or not jwst_euclid._cached_pair_is_usable(directory, manifest)):
        raise ViewerError(404, "paired field is incomplete")
    return jwst_euclid.enrich_manifest_metadata(directory, manifest), str(directory)


def _pair_file(directory: str, relative: object) -> str:
    """Resolve a manifest FITS path below one saved-pair directory."""
    if not isinstance(relative, str) or not relative:
        raise ViewerError(404, "paired field product is missing")
    root = os.path.realpath(directory)
    path = os.path.realpath(os.path.join(root, relative))
    if os.path.commonpath((root, path)) != root or not os.path.isfile(path):
        raise ViewerError(404, "paired field product is missing")
    return path


def _pair_image(path: str) -> tuple[np.ndarray, Any, Any]:
    """Read the first usable celestial image from a cached pair FITS file."""
    try:
        with fits.open(path, memmap=False) as hdul:
            primary = cast(fits.PrimaryHDU, hdul[0]).header
            for raw_hdu in hdul:
                hdu = _image_hdu(raw_hdu)
                if hdu is None:
                    continue
                data = hdu.data
                if data is None or np.ndim(data) != 2:
                    continue
                header = hdu.header.copy()
                try:
                    wcs = WCS(header).celestial
                    if not wcs.has_celestial:
                        wcs = WCS(primary).celestial
                    if wcs.has_celestial:
                        return np.asarray(data, np.float32), header, wcs
                except Exception:  # noqa: BLE001 - heterogeneous archive headers
                    continue
    except OSError as exc:
        raise ViewerError(404, "paired field FITS is unreadable") from exc
    raise ViewerError(404, "paired field FITS has no celestial image")


def _pair_cube_and_header(path: str) -> tuple[np.ndarray, fits.Header]:
    """A 2-D image or a small channel-first cube from a pair product, plus
    the header of the HDU it came from (its WCS and ``BUNIT``)."""
    try:
        with fits.open(path, memmap=False) as hdul:
            for hdu in hdul:
                data = getattr(hdu, "data", None)
                if data is None or np.ndim(data) not in (2, 3):
                    continue
                return (_as_hwc(np.asarray(data, np.float32), layout="chw"),
                        hdu.header.copy())
    except OSError as exc:
        raise ViewerError(404, "paired field FITS is unreadable") from exc
    raise ViewerError(404, "paired field FITS has no image cube")


def _pair_cube(path: str) -> np.ndarray:
    """Read a 2-D image or a small channel-first cube from a pair product."""
    return _pair_cube_and_header(path)[0]


def _jwst_band_entries(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalise old one-product manifests and current multi-filter locations."""
    entries = manifest.get("jwst_bands", [])
    if isinstance(entries, list) and entries:
        return [dict(entry) for entry in entries if isinstance(entry, dict)]
    files = manifest.get("files", {}) or {}
    return [{
        "key": "jwst0",
        "filter": _jwst_band_name(manifest),
        "file": files.get("jwst_native"),
        "metadata": manifest.get("jwst_metadata", {}),
        "native_is_field_cutout": bool(manifest.get("jwst_native_is_field_cutout")),
    }]


def _pair_native_jwst(
    manifest: dict[str, Any], directory: str, entry: dict[str, Any],
) -> tuple[np.ndarray, Any]:
    """Return JWST at its source pixel scale, cropping legacy full products only."""
    data, _header, wcs = _pair_image(_pair_file(directory, entry.get("file")))
    if entry.get("native_is_field_cutout"):
        return data, wcs
    try:
        coordinate = SkyCoord(
            ra=float(manifest["ra_deg"]), dec=float(manifest["dec_deg"]), unit="deg", frame="icrs",
        )
        return jwst_euclid._native_sky_cutout(
            data, wcs, coordinate, float(manifest["size_arcsec"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ViewerError(404, "legacy JWST product has no usable field geometry") from exc


def _robust_display_scale(data: np.ndarray) -> float:
    """Return a display-only factor that puts the robust bright end at white.

    The canvas viewer uses ``30 * 100`` as its default white reference.  The
    archive products have unrelated units and calibrations, so this is not a
    flux conversion or a resampling operation; native FITS values stay intact.
    """
    finite = np.asarray(data, np.float32)
    finite = finite[np.isfinite(finite)]
    positive = finite[finite > 0]
    if positive.size:
        bright = float(np.nanpercentile(positive, 99.5))
    elif finite.size:
        bright = float(np.nanpercentile(np.abs(finite), 99.5))
    else:
        bright = 1.0
    return 3000.0 / max(bright, 1e-12)


def _jwst_sr_pixel_blur(
    cube: np.ndarray,
    info: Mapping[str, Any],
    *,
    sr_pixel_scale_arcsec: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply a Gaussian whose FWHM is one SR pixel to native-grid JWST.

    The comparison tier deliberately stays on the JWST grid.  Converting the
    angular SR-pixel width to a native-JWST sigma avoids silently resampling
    either image, while the copied display scale ensures the native and
    blurred JWST tiers differ only by this convolution.
    """
    jwst_pixel_scale = float(info.get("pixscale", 0.0))
    if not math.isfinite(jwst_pixel_scale) or jwst_pixel_scale <= 0:
        raise ViewerError(404, "JWST pixel scale is unavailable for SR-matched blur")
    if not math.isfinite(sr_pixel_scale_arcsec) or sr_pixel_scale_arcsec <= 0:
        raise ViewerError(500, "SR pixel scale is invalid")

    sigma_pixels = (
        sr_pixel_scale_arcsec
        / jwst_pixel_scale
        / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    )
    source = np.asarray(cube, dtype=np.float32)
    sigma = (sigma_pixels, sigma_pixels, 0.0)
    finite = np.isfinite(source)
    if np.all(finite):
        blurred = gaussian_filter(source, sigma=sigma, mode="reflect")
    else:
        # FITS/WCS cutouts can carry NaN rims.  Normalised convolution keeps
        # them from poisoning neighbouring valid pixels, then restores the
        # original invalid footprint rather than inventing sky coverage.
        values = gaussian_filter(np.where(finite, source, 0.0), sigma=sigma, mode="reflect")
        weights = gaussian_filter(finite.astype(np.float32), sigma=sigma, mode="reflect")
        blurred = np.full_like(source, np.nan)
        np.divide(values, weights, out=blurred, where=weights > 1e-6)
        blurred[~finite] = np.nan

    blurred_info = dict(info)
    blurred_info["label"] = (
        f"{info.get('label') or 'JWST'} · Gaussian blur · "
        f"FWHM {sr_pixel_scale_arcsec:.3f}\" (1 SR px)"
    )
    return np.asarray(blurred, dtype=np.float32), blurred_info


def _jwst_band_name(manifest: dict[str, Any]) -> str:
    """Return the real JWST filter/pupil name, excluding non-band CLEAR."""
    metadata = manifest.get("jwst_metadata", {}) or {}
    candidates = (
        metadata.get("filter"),
        metadata.get("pupil"),
        manifest.get("jwst_filters"),
    )
    for candidate in candidates:
        text = str(candidate or "").strip().upper()
        if text and text not in {"CLEAR", "N/A", "NONE", "UNKNOWN"}:
            return text
    return "JWST"


def _saved_jwst_euclid_pairs() -> list[tuple[dict[str, Any], str]]:
    """Return saved paired fields in the stable location-carousel order."""
    pairs: list[tuple[dict[str, Any], str]] = []
    for manifest in jwst_euclid.saved_pairs():
        identifier = str(manifest.get("field_id") or "")
        if not _PAIR_ID.fullmatch(identifier):
            continue
        directory = jwst_euclid.pair_root() / identifier
        pairs.append((manifest, str(directory)))
    if not pairs:
        raise ViewerError(404, "no saved JWST × Euclid fields")
    return pairs


def _jwst_filter_wavelength_um(entry: Mapping[str, Any]) -> float:
    """Approximate a JWST filter pivot from its standard ``F###`` name."""
    text = str(entry.get("filter") or "").upper()
    match = re.search(r"F(\d{3,4})", text)
    return float(match.group(1)) / 100.0 if match else math.inf


def _jwst_approx_color_band(entry: Mapping[str, Any]) -> tuple[str, dict[str, Any]] | None:
    """Describe a JWST filter for display-only temperature colouring.

    Archive products can be expressed in different calibrated surface-brightness
    units and need their actual throughput curves for photometric colour work.
    The viewer therefore uses only the wavelength encoded in ``F###`` as an
    effective pivot and marks this path display-only.
    """
    name = str(entry.get("filter") or "").strip().upper()
    pivot_um = _jwst_filter_wavelength_um(entry)
    if not name or not math.isfinite(pivot_um):
        return None
    return name, {
        "pivot_um": pivot_um,
        # The JS temperature renderer only needs a relative normalisation.
        # Avoid exposing a fictitious JWST AB zero point or magnitude readout.
        "zeropoint_ab_e_total": 1.0,
        "display_only": True,
    }


def _jwst_colour_channel_groups(entries: list[dict[str, Any]]) -> tuple[list[int], list[int], list[int]]:
    """Assign every available filter to blue, green, or red display light."""
    ordered = sorted(range(len(entries)), key=lambda index: _jwst_filter_wavelength_um(entries[index]))
    if len(ordered) == 1:
        return ordered, ordered, ordered
    if len(ordered) == 2:
        return [ordered[0]], [ordered[0]], [ordered[1]]
    blue, green, red = (list(chunk) for chunk in np.array_split(np.asarray(ordered), 3))
    return blue, green, red


def _jwst_aligned_planes(
    manifest: dict[str, Any], directory: str,
) -> tuple[list[dict[str, Any]], list[np.ndarray], str, float, Any]:
    """Return the usable JWST planes on the finest native JWST display WCS
    (entries, planes, reference filter, reference pixel scale, reference WCS)."""
    loaded: list[tuple[dict[str, Any], np.ndarray, Any, float]] = []
    for entry in _jwst_band_entries(manifest):
        try:
            data, wcs = _pair_native_jwst(manifest, directory, entry)
        except ViewerError:
            continue
        metadata = entry.get("metadata", {}) or {}
        scales = metadata.get("pixel_scale_arcsec", [])
        scale = float(scales[0]) if isinstance(scales, list) and scales else math.inf
        if not math.isfinite(scale) or scale <= 0:
            scale = math.inf
        loaded.append((entry, np.asarray(data, np.float32), wcs, scale))
    if not loaded:
        raise ViewerError(404, "saved field has no usable JWST images")

    reference_index = min(range(len(loaded)), key=lambda index: loaded[index][3])
    reference_entry, reference_data, reference_wcs, reference_scale = loaded[reference_index]
    aligned_entries: list[dict[str, Any]] = []
    aligned_planes: list[np.ndarray] = []
    for entry, data, wcs, _scale in loaded:
        try:
            plane = data if wcs is reference_wcs else jwst_euclid.align_to_target(
                data, wcs, reference_wcs, reference_data.shape,
            )
        except Exception:  # noqa: BLE001 - a non-overlapping camera need not break colour
            continue
        if np.any(np.isfinite(plane)):
            aligned_entries.append(entry)
            aligned_planes.append(np.asarray(plane, np.float32))
    if not aligned_planes:
        raise ViewerError(404, "JWST cameras do not overlap on this field")
    reference_filter = str(reference_entry.get("filter") or "JWST")
    return (aligned_entries, aligned_planes, reference_filter, reference_scale,
            reference_wcs)


def _jwst_colour_cube(manifest: dict[str, Any], directory: str) -> tuple[np.ndarray, dict[str, Any]]:
    """Build a display-only RGB composite while retaining the source FITS grids.

    Native JWST files remain untouched in the cache.  For the viewer only,
    every usable filter is sampled onto the finest saved JWST WCS, then split
    by wavelength into blue/green/red groups.  The aligned native brightnesses
    are combined directly: one shared display stretch is applied to the final
    cube, so inter-filter brightness ratios remain intact.
    """
    (aligned_entries, aligned_planes, reference_filter, reference_scale,
     reference_wcs) = _jwst_aligned_planes(manifest, directory)

    blue_indices, green_indices, red_indices = _jwst_colour_channel_groups(aligned_entries)

    def channel(indices: list[int]) -> np.ndarray:
        stack = np.stack([aligned_planes[index] for index in indices], axis=0)
        # WCS resampling leaves a NaN rim where a coarser camera has no source
        # pixels.  A fully empty edge is expected and transparent in display.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            plane = np.nanmedian(stack, axis=0)
        return np.nan_to_num(plane, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    blue, green, red = channel(blue_indices), channel(green_indices), channel(red_indices)
    cube = np.stack([red, green, blue], axis=-1)

    def names(indices: list[int]) -> str:
        return "+".join(str(aligned_entries[index].get("filter") or "JWST") for index in indices)

    return cube, {
        "label": (
            f"JWST colour | R {names(red_indices)} | G {names(green_indices)} | "
            f"B {names(blue_indices)} | display WCS {reference_filter}"
        ),
        "asinh": 0.05,
        "pixscale": reference_scale if math.isfinite(reference_scale) else 0.0,
        "bands": ["JWST-R", "JWST-G", "JWST-B"],
        "direct_rgb": True,
        "display_scale": _robust_display_scale(cube),
        "transfer_group": "jwst",
        "unit": "arb",                     # display-only colour composite
        "wcs": celestial_wcs_keywords(reference_wcs),
    }


def _jwst_temperature_cube(manifest: dict[str, Any], directory: str) -> tuple[np.ndarray, dict[str, Any]]:
    """Build an approximate JWST temperature cube from native brightnesses."""
    (entries, planes, reference_filter, reference_scale,
     reference_wcs) = _jwst_aligned_planes(manifest, directory)
    bands_and_meta = [_jwst_approx_color_band(entry) for entry in entries]
    usable = [item for item in bands_and_meta if item is not None]
    if len(usable) < 2:
        raise ViewerError(404, "JWST temperature colour needs at least two named filters")
    names = [item[0] for item in usable]
    selected_planes = [
        np.nan_to_num(planes[index], nan=0.0, posinf=0.0, neginf=0.0)
        for index, item in enumerate(bands_and_meta) if item is not None
    ]
    cube = np.stack(selected_planes, axis=-1).astype(np.float32)
    return cube, {
        "label": (
            f"JWST temperature · approximate F### pivots ({'+'.join(names)}) | "
            f"display WCS {reference_filter}"
        ),
        "asinh": 100.0,
        "pixscale": reference_scale if math.isfinite(reference_scale) else 0.0,
        "bands": names,
        "display_scale": _robust_display_scale(cube),
        "transfer_group": "jwst",
        "unit": "arb",
        "wcs": celestial_wcs_keywords(reference_wcs),
    }


def _manifest_position(manifest: Mapping[str, Any]) -> dict[str, float]:
    """``{"ra", "dec"}`` of a pair/tile manifest (``ra_deg``/``dec_deg``)."""
    ra, dec = _finite_float(manifest.get("ra_deg")), _finite_float(manifest.get("dec_deg"))
    return {"ra": ra, "dec": dec} if ra is not None and dec is not None else {}


def _image_unit(path: str, default: str = "arb") -> str:
    """Viewer unit of the first HDU carrying a ``BUNIT`` (headers only);
    ``default`` (itself normalised, e.g. a manifest's ``"MJy/sr"``) else."""
    fallback = _UNIT_ALIASES.get("".join(default.lower().split()), default)
    try:
        with fits.open(path, memmap=True, lazy_load_hdus=True) as hdul:
            for hdu in hdul:
                if hdu.header.get("BUNIT"):
                    return unit_from_header(hdu.header, default=fallback)
    except OSError:
        pass
    return fallback


def _pair_lr_wcs(directory: str, relative: object) -> dict[str, Any] | None:
    """WCS of a pair/tile LR product (the HDU :func:`_pair_cube` reads)."""
    try:
        return celestial_wcs_keywords(_pair_cube_and_header(_pair_file(directory, relative))[1])
    except ViewerError:
        return None


def _jwst_euclid_meta(params: dict[str, str]) -> dict[str, Any]:
    pairs = _saved_jwst_euclid_pairs()
    seen_filters: set[str] = set()
    jwst_band_options = [{"value": "colour", "label": "JWST colour"}]
    filter_entries = sorted(
        (entry for manifest, _directory in pairs for entry in _jwst_band_entries(manifest)),
        key=_jwst_filter_wavelength_um,
    )
    for entry in filter_entries:
        band = str(entry.get("filter") or "").strip().upper()
        if not band or band in seen_filters:
            continue
        seen_filters.add(band)
        jwst_band_options.append({"value": band, "label": band})
    if len(seen_filters) >= 2:
        jwst_band_options.insert(1, {
            "value": "temperature",
            "label": "JWST temperature · approximate",
        })
    jwst_color_bands = dict(
        item for entry in filter_entries if (item := _jwst_approx_color_band(entry)) is not None
    )
    tiers = [
        {"key": "lr", "label": "LR · Euclid VIS"},
        {"key": "sr", "label": "SR · STARFULL combiner", "unit": "e-"},
        {"key": "jwst", "label": "JWST"},
        {"key": "jwst_blur", "label": "JWST · Gaussian blur · FWHM 1 SR px"},
    ]
    return {
        "count": len(pairs),
        "tiers": tiers,
        "default_tier": "lr",
        "band_names": list(BAND_NAMES),
        "color_label": "Euclid colour",
        "jwst_band_options": jwst_band_options,
        "extra_color_bands": jwst_color_bands,
        "missing_tier_labels": {"sr": "Generate SR"},
        "transfer_groups": ["euclid", "jwst"],
        "objects": [
            {
                "id": str(manifest.get("field_id") or index),
                **_manifest_position(manifest),
                "label": f"{index} · {manifest.get('target_name') or 'paired field'}",
                # Keep SR in every comparison row.  Before inference its tile
                # is an explicit "Generate SR" affordance rather than a hidden
                # tier, while after inference the same tile receives the result.
                "tiers": [tier["key"] for tier in tiers],
                "jwst_bands": ["colour"] + (
                    ["temperature"] if len(_jwst_band_entries(manifest)) >= 2 else []
                ) + [
                    str(entry.get("filter") or "").strip().upper()
                    for entry in _jwst_band_entries(manifest)
                    if str(entry.get("filter") or "").strip()
                ],
            }
            for index, (manifest, _directory) in enumerate(pairs)
        ],
    }


def _jwst_euclid_cube(index: int, tier: str, params: dict[str, str]):
    pairs = _saved_jwst_euclid_pairs()
    if not 0 <= index < len(pairs):
        raise ViewerError(404, "paired field index out of range")
    manifest, directory = pairs[index]
    files = manifest.get("files", {}) or {}
    inference = manifest.get("inference", {}) or {}
    inference_files = inference.get("files", {}) if isinstance(inference, dict) else {}
    lr_source = inference_files.get("lr") or files.get("euclid")
    if tier == "lr":
        cube, header = _pair_cube_and_header(_pair_file(directory, lr_source))
        bands = list(BAND_NAMES[:cube.shape[-1]])
        return cube, {
            "label": "LR · Euclid VIS",
            "asinh": float(Config.STRETCH_SCALE_E),
            "pixscale": float(Config.VIS_PIXEL_SCALE_ARCSEC),
            "bands": bands,
            "display_scale": _robust_display_scale(cube),
            "transfer_group": "euclid",
            "unit": unit_from_header(
                header, default="e-" if inference_files.get("lr") else "arb"),
            "wcs": celestial_wcs_keywords(header),
        }
    if tier == "sr":
        source = inference_files.get("starfull")
        if not source:
            raise ViewerError(404, "STARFULL inference is not available for this field")
        cube = _pair_cube(_pair_file(directory, source))
        bands = list(BAND_NAMES[:cube.shape[-1]])
        return cube, {
            "label": str(inference.get("combiner_label") or "SR · STARFULL combiner"),
            "asinh": float(Config.STRETCH_SCALE_E),
            "pixscale": float(inference.get("pixel_scale_arcsec") or Config.DEFAULT_PIXEL_SCALE),
            "bands": bands,
            "display_scale": _robust_display_scale(cube),
            "transfer_group": "euclid",
            "unit": "e-",
            # SR grid = the LR grid magnified ×2 (one consistent rule).
            "wcs": scaled_wcs_keywords(
                _pair_lr_wcs(directory, lr_source), 2),
        }
    if tier in {"jwst", "jwst_blur"}:
        choice = str(params.get("jwst_band") or "colour").strip().upper()
        if choice in {"", "COLOUR"}:
            result = _jwst_colour_cube(manifest, directory)
        elif choice == "TEMPERATURE":
            result = _jwst_temperature_cube(manifest, directory)
        else:
            entry = next(
                (candidate for candidate in _jwst_band_entries(manifest)
                 if str(candidate.get("filter") or "").strip().upper() == choice),
                None,
            )
            if entry is None:
                raise ViewerError(404, f"JWST band {choice} is unavailable for this field")
            data, native_wcs = _pair_native_jwst(manifest, directory, entry)
            metadata = entry.get("metadata", {}) or {}
            scales = metadata.get("pixel_scale_arcsec", [])
            scale = float(scales[0]) if isinstance(scales, list) and scales else 0.0
            result = _as_hwc(data), {
                "label": f"JWST native · {choice} · display-normalised only",
                "asinh": 100.0,
                "pixscale": scale if math.isfinite(scale) else 0.0,
                "bands": [choice],
                "display_scale": _robust_display_scale(data),
                "transfer_group": "jwst",
                "unit": _image_unit(_pair_file(directory, entry.get("file")),
                                    default=str(metadata.get("units") or "arb")),
                "wcs": celestial_wcs_keywords(native_wcs),
            }
        if tier == "jwst_blur":
            sr_scale = float(inference.get("pixel_scale_arcsec") or Config.DEFAULT_PIXEL_SCALE)
            return _jwst_sr_pixel_blur(*result, sr_pixel_scale_arcsec=sr_scale)
        return result
    raise ViewerError(400, "bad paired-field tier")


# ---------------------------------------------------------------------------
# nexus-field — one full NEXUS mosaic covered by 255×255 Euclid tiles
# ---------------------------------------------------------------------------

def _nexus_field(params: dict[str, str]) -> tuple[dict[str, Any], str]:
    identifier = (params.get("field") or "").strip()
    if not _PAIR_ID.fullmatch(identifier):
        raise ViewerError(404, "NEXUS tiled field not found")
    manifest = jwst_euclid._read_nexus_field_manifest(identifier)
    if manifest is None:
        raise ViewerError(404, "NEXUS tiled field not found")
    return manifest, str(jwst_euclid.nexus_field_root() / identifier)


def _nexus_jwst_unit(tiles: list[Any]) -> str:
    """NEXUS NIRCam mosaics are MJy/sr; trust a tile's recorded unit first."""
    for tile in tiles:
        metadata = tile.get("jwst_metadata", {}) if isinstance(tile, Mapping) else {}
        recorded = str((metadata or {}).get("units") or "").strip()
        if recorded:
            return _UNIT_ALIASES.get("".join(recorded.lower().split()), recorded)
    return "MJy/sr"


def _nexus_field_meta(params: dict[str, str]) -> dict[str, Any]:
    manifest, directory = _nexus_field(params)
    tiles = manifest.get("tiles", [])
    if not isinstance(tiles, list) or not tiles:
        raise ViewerError(404, "NEXUS tiled field has no covered Euclid tiles")

    def _available_tiers(tile: Mapping[str, Any]) -> list[str]:
        lr_file = tile.get("lr_file")
        inference = tile.get("inference", {})
        files = inference.get("files", {}) if isinstance(inference, Mapping) else {}
        sr_file = files.get("starfull") if isinstance(files, Mapping) else None
        has_registered_lr = (
            isinstance(lr_file, str) and os.path.isfile(os.path.join(directory, lr_file))
        )
        has_sr = (
            isinstance(sr_file, str) and os.path.isfile(os.path.join(directory, sr_file))
        )
        tiers = ["lr"]
        if has_registered_lr or has_sr:
            tiers.append("sr")
        tiers.extend(("jwst", "jwst_blur"))
        return tiers

    jwst_unit = _nexus_jwst_unit(tiles)
    identifier = str(manifest.get("field_id") or "nexus")
    return {
        "count": len(tiles),
        "tiers": [
            {"key": "lr", "label": "LR · Euclid · 255 px", "unit": "e-"},
            {"key": "sr", "label": "SR · STARFULL combiner", "unit": "e-"},
            {"key": "jwst", "label": f"NEXUS {manifest.get('filter') or 'JWST'} · native",
             "unit": jwst_unit},
            {"key": "jwst_blur", "label": "JWST · Gaussian blur · FWHM 1 SR px",
             "unit": jwst_unit},
        ],
        "default_tier": "lr",
        "band_names": list(BAND_NAMES),
        "color_label": "Euclid colour",
        "missing_tier_labels": {"sr": "Generate SR"},
        "transfer_groups": ["euclid", "jwst"],
        "objects": [{
            "id": f"{identifier}/{int(tile.get('index', index)):04d}",
            **_manifest_position(tile),
            "label": (
                f"{index} · RA {float(tile.get('ra_deg', 0.0)):.5f}, "
                f"Dec {float(tile.get('dec_deg', 0.0)):.5f}"
            ),
            "tiers": _available_tiers(tile),
            "jwst_bands": [str(manifest.get("filter") or "JWST")],
        } for index, tile in enumerate(tiles) if isinstance(tile, Mapping)],
    }


def _nexus_field_cube(index: int, tier: str, params: dict[str, str]):
    manifest, directory = _nexus_field(params)
    tiles = manifest.get("tiles", [])
    if not isinstance(tiles, list) or not 0 <= index < len(tiles):
        raise ViewerError(404, "NEXUS tiled field index out of range")
    tile = tiles[index]
    if not isinstance(tile, Mapping):
        raise ViewerError(404, "NEXUS tile is invalid")
    lr_source = tile.get("lr_file") or tile.get("euclid_file")
    if tier == "lr":
        cube, header = _pair_cube_and_header(_pair_file(directory, lr_source))
        bands = list(BAND_NAMES[:cube.shape[-1]])
        return cube, {
            "label": "LR · Euclid VIS+Y+J+H · matched 255 × 255 tile" if len(bands) == 4
            else "Euclid VIS · matched 255 × 255 tile",
            "asinh": float(Config.STRETCH_SCALE_E),
            "pixscale": float(Config.VIS_PIXEL_SCALE_ARCSEC),
            # The registered NEXUS Euclid cube is already in the same raw
            # electron units as the normal Inference Tile viewer.  Do not
            # apply the archive-only robust scale here: it made faint
            # background/noise much brighter before the shared asinh clip.
            "bands": bands,
            "transfer_group": "euclid",
            "unit": "e-",
            "wcs": celestial_wcs_keywords(header),
        }
    if tier == "sr":
        inference = tile.get("inference", {}) if isinstance(tile, Mapping) else {}
        files = inference.get("files", {}) if isinstance(inference, Mapping) else {}
        source = files.get("starfull") if isinstance(files, Mapping) else None
        if not isinstance(source, str) or not source:
            raise ViewerError(404, "STARFULL inference is not available for this NEXUS tile")
        cube = _pair_cube(_pair_file(directory, source))
        return cube, {
            "label": str(inference.get("combiner_label") or "SR · STARFULL combiner"),
            "asinh": float(Config.STRETCH_SCALE_E),
            "pixscale": float(inference.get("pixel_scale_arcsec") or Config.DEFAULT_PIXEL_SCALE),
            "bands": list(BAND_NAMES[:cube.shape[-1]]),
            "transfer_group": "euclid",
            "unit": "e-",
            # SR grid = the tile's LR grid magnified ×2 (one consistent rule).
            "wcs": scaled_wcs_keywords(_pair_lr_wcs(directory, lr_source), 2),
        }
    if tier in {"jwst", "jwst_blur"}:
        cube, header = _pair_cube_and_header(_pair_file(directory, tile.get("jwst_file")))
        metadata = tile.get("jwst_metadata", {}) if isinstance(tile, Mapping) else {}
        scales = metadata.get("pixel_scale_arcsec", []) if isinstance(metadata, Mapping) else []
        scale = float(scales[0]) if isinstance(scales, list) and scales else 0.0
        result = cube, {
            "label": f"NEXUS native · {manifest.get('filter') or 'JWST'}",
            "asinh": 100.0, "pixscale": scale if math.isfinite(scale) else 0.0,
            "bands": [str(manifest.get("filter") or "JWST")],
            "display_scale": _robust_display_scale(cube),
            "transfer_group": "jwst",
            "unit": unit_from_header(header, default=_nexus_jwst_unit([tile])),
            "wcs": celestial_wcs_keywords(header),
        }
        if tier == "jwst_blur":
            inference = tile.get("inference", {}) if isinstance(tile, Mapping) else {}
            sr_scale = float(
                inference.get("pixel_scale_arcsec") or Config.DEFAULT_PIXEL_SCALE
            ) if isinstance(inference, Mapping) else float(Config.DEFAULT_PIXEL_SCALE)
            return _jwst_sr_pixel_blur(*result, sr_pixel_scale_arcsec=sr_scale)
        return result
    raise ViewerError(400, "bad NEXUS tiled-field tier")


# ---------------------------------------------------------------------------
# real — any real tile of the real-tile store (contract C9)
# ---------------------------------------------------------------------------
#
# ``?source=`` picks the real-tile source (nexus, tile, field, archive, eval,
# poster, pair); ``?models=`` (comma list of model specs) the ``m:<spec>``
# tiers — default: every spec with at least one output in the source. A tile's
# outputs are the C9 store merged with its legacy SRs (NEXUS whole-field /
# pair inference, poster), see ``real_tiles.tile_outputs``. Object ids are the
# real-tile ids, so ``?id=`` works as for every collection.

_SPEC_ORDER = ("production", "mean", "rbf")


def _real_source(params: dict[str, str]) -> str:
    source = (params.get("source") or "").strip()
    try:
        return real_tiles.check_source(source)
    except real_tiles.RealTileError as exc:
        raise ViewerError(404, str(exc)) from exc


def _real_default_specs(outputs: list[dict[str, Any]]) -> list[str]:
    seen = {spec for tile in outputs for spec in tile}

    def order(spec: str) -> tuple[int, str]:
        return (_SPEC_ORDER.index(spec) if spec in _SPEC_ORDER else
                len(_SPEC_ORDER) + (0 if spec.startswith("member:") else 1), spec)

    return sorted(seen, key=order)


def _real_specs(params: dict[str, str], outputs: list[dict[str, Any]]) -> list[str]:
    raw = params.get("models")
    if not raw:
        return _real_default_specs(outputs)
    try:
        return model_catalog.parse_specs(raw)
    except ValueError as exc:
        raise ViewerError(400, str(exc)) from exc


def _real_meta(params: dict[str, str]) -> dict[str, Any]:
    source = _real_source(params)
    entries = real_tiles.list_entries(source)
    catalog = {item.spec: item for item in model_catalog.list_specs()}
    current = {spec: item.fingerprint for spec, item in catalog.items()}
    tile_outputs = [real_tiles.tile_outputs(entry, current) for entry in entries]
    specs = _real_specs(params, tile_outputs)
    has_jwst = any(entry.has_jwst for entry in entries)
    tiers: list[dict[str, Any]] = [{"key": "lr", "label": "LR · Euclid", "unit": "e-"}]
    if has_jwst:
        tiers.append({"key": "jwst", "label": "JWST · native", "unit": "MJy/sr"})
    for spec in specs:
        item = catalog.get(spec)
        tiers.append({"key": f"m:{spec}", "label": item.label if item else spec,
                      "unit": "e-", "spec": spec,
                      "available": bool(item and item.available)})
    objects = []
    for entry, outputs in zip(entries, tile_outputs, strict=True):
        states = {spec: model_catalog.output_state(outputs[spec], current)
                  for spec in specs if spec in outputs}
        objects.append({
            "id": entry.id, "label": entry.label,
            **({"ra": entry.ra, "dec": entry.dec}
               if entry.ra is not None and entry.dec is not None else {}),
            "field": entry.field, "ref": entry.ref,
            "tiers": (["lr"] + (["jwst"] if entry.has_jwst else [])
                      + [f"m:{spec}" for spec in states]),
            "model_states": states,
            "legacy_models": [spec for spec in states if outputs[spec].get("legacy")],
            "model_ready": entry.model_ready,
        })
    return {
        "count": len(objects), "tiers": tiers, "default_tier": "lr",
        "band_names": list(BAND_NAMES), "source": source, "models": specs,
        "transfer_groups": ["euclid", "jwst"] if has_jwst else ["euclid"],
        "missing_tier_labels": {f"m:{spec}": "Run this model (Sky → Experiments)"
                                for spec in specs},
        "objects": objects,
    }


def _real_entry(index: int, params: dict[str, str]) -> real_tiles.TileEntry:
    entries = real_tiles.list_entries(_real_source(params))
    if not 0 <= index < len(entries):
        raise ViewerError(404, "real tile index out of range")
    return entries[index]


def _pixscale_of(header: Any, default: float) -> float:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            matrix = WCS(header).celestial.pixel_scale_matrix
        return float(math.sqrt(abs(np.linalg.det(matrix))) * 3600.0)
    except Exception:  # noqa: BLE001 - heterogeneous headers
        return default


def _real_cube(index: int, tier: str, params: dict[str, str]):
    entry = _real_entry(index, params)
    if tier == "lr":
        try:
            tile = real_tiles.get_tile(entry.source, entry.id, entry=entry)
        except real_tiles.RealTileError as exc:
            raise ViewerError(exc.code, str(exc)) from exc
        return tile.lr_e, {
            "label": f"LR · {entry.label}", "asinh": float(Config.STRETCH_SCALE_E),
            "pixscale": float(entry.pixscale), "bands": list(entry.bands),
            "transfer_group": "euclid", "unit": "e-",
            "wcs": celestial_wcs_keywords(tile.wcs_header),
        }
    if tier == "jwst":
        planes = real_tiles.jwst_planes(entry)
        if not planes:
            raise ViewerError(404, f"{entry.ref} has no JWST image")
        wanted = (params.get("jwst_band") or "").strip().upper()
        plane = next((p for p in planes if p["band"].upper() == wanted), planes[0])
        cube = np.asarray(plane["data"], np.float32)[..., None]
        return cube, {
            "label": f"JWST {plane['band']} · native", "asinh": 100.0,
            "pixscale": _pixscale_of(plane["header"], 0.0), "bands": [plane["band"]],
            "display_scale": _robust_display_scale(cube), "transfer_group": "jwst",
            "unit": unit_from_header(plane["header"], default=plane["unit"]),
            "wcs": celestial_wcs_keywords(plane["header"]),
            "jwst_bands": [p["band"] for p in planes],
        }
    if tier.startswith("m:"):
        current = model_catalog.current_fingerprints()
        try:
            spec = model_catalog.canonical_spec(tier[2:])
            cube, header, meta = real_tiles.load_output(entry, spec, current=current.get(spec))
        except ValueError as exc:
            raise ViewerError(400, str(exc)) from exc
        except FileNotFoundError as exc:
            raise ViewerError(404, f"{spec} has not been run on {entry.ref} yet") from exc
        state = model_catalog.output_state(meta, current)
        return cube, {
            "label": (f"{meta.get('label') or spec}"
                      + (" · legacy" if meta.get("legacy") else "")
                      + ("" if state == "current" else f" · {state}")),
            "asinh": float(Config.STRETCH_SCALE_E),
            "pixscale": float(entry.pixscale) / model_catalog.SR_FACTOR,
            "bands": list(BAND_NAMES[:cube.shape[-1]]), "transfer_group": "euclid",
            "unit": "e-", "wcs": celestial_wcs_keywords(header),
            "model_state": state, "legacy": bool(meta.get("legacy")),
        }
    raise ViewerError(400, f"bad real-tile tier {tier!r} (lr, jwst, m:<spec>)")


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

_Meta = Callable[[dict[str, str]], dict[str, Any]]
_Cube = Callable[[int, str, dict[str, str]], tuple[np.ndarray, dict[str, Any]]]

_REGISTRY: dict[str, tuple[_Meta, _Cube]] = {
    "sky": (_sky_meta, _sky_cube),
    "cutouts": (_cutouts_meta, _cutouts_cube),
    "evaluation": (_eval_meta, _eval_cube),
    "ensemble": (_ensemble_meta, _ensemble_cube),
    "archive-fields": (_archive_fields_meta, _archive_fields_cube),
    "real-field": (_real_field_meta, _real_field_cube),
    "jwst-euclid": (_jwst_euclid_meta, _jwst_euclid_cube),
    "nexus-field": (_nexus_field_meta, _nexus_field_cube),
    "psfs": (_psf_meta, _psf_cube),
    "real": (_real_meta, _real_cube),
}


def get_meta(collection: str, params: dict[str, str]) -> dict[str, Any]:
    if collection not in _REGISTRY:
        raise ViewerError(404, "unknown collection")
    meta = _REGISTRY[collection][0](params)
    meta["collection"] = collection
    if any(str(tier.get("key", "")).lower() == "bhr"
           for tier in meta.get("tiers", [])):
        _bhr_fwhm_arcsec(params)
        meta["bhr_fwhm_control"] = {
            "param": BHR_FWHM_PARAM,
            "default_arcsec": float(Config.TARGET_PSF_FWHM_ARCSEC),
            "min_arcsec": 0.0,
            "max_arcsec": BHR_FWHM_MAX_ARCSEC,
            "step_arcsec": 0.001,
        }
    # This is shared metadata rather than collection-specific UI state: the
    # generic Tile viewer is mounted by all routes and must annotate every
    # tile at the same physical/angular receptive-field sizes.
    meta["receptive_fields"] = receptive_field_constants()
    color = color_constants()
    extra_color_bands = meta.pop("extra_color_bands", {})
    if isinstance(extra_color_bands, dict):
        color["bands"].update(extra_color_bands)
    meta["color"] = color
    if meta.get("render_mode"):
        meta["color"]["render_mode"] = meta["render_mode"]
    return meta


def index_of(objects: list[Mapping[str, Any]], object_id: str) -> int:
    """Position of the object whose meta ``id`` is ``object_id`` in a meta's
    ``objects``; :class:`ViewerError` 404 when it is not there."""
    for index, obj in enumerate(objects):
        if str(obj.get("id")) == object_id:
            return index
    raise ViewerError(404, f"unknown object id: {object_id}")


def resolve_index(collection: str, object_id: str,
                  params: dict[str, str]) -> int:
    """Position of the object whose stable meta ``id`` is ``object_id``,
    looked up in the collection's meta built with the same ``params``."""
    if collection not in _REGISTRY:
        raise ViewerError(404, "unknown collection")
    return index_of(_REGISTRY[collection][0](params).get("objects") or [], object_id)


def get_cube(collection: str, index: int, tier: str,
             params: dict[str, str]) -> tuple[np.ndarray, dict[str, Any]]:
    if collection not in _REGISTRY:
        raise ViewerError(404, "unknown collection")
    cube, info = _REGISTRY[collection][1](index, tier, params)
    return np.ascontiguousarray(cube, dtype=np.float32), info


__all__ = [
    "BAND_NAMES",
    "BHR_FWHM_PARAM",
    "COMBINER_MODELS",
    "MEAN_LABEL",
    "PRODUCTION_COMBINER_KIND",
    "PRODUCTION_SR_LABEL",
    "RAW_INCREMENTAL_MINMEANMAX_RBF_KIND",
    "SPATIAL_GATE_KIND",
    "ViewerError",
    "celestial_wcs_keywords",
    "color_constants",
    "get_cube",
    "get_meta",
    "index_of",
    "receptive_field_constants",
    "resolve_index",
    "scaled_wcs_keywords",
    "shifted_wcs_keywords",
    "unit_from_header",
]
