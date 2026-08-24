"""Collection registry feeding the unified client-side cutout viewer.

The viewer (``static/cutout_viewer.js``) renders raw N-band float cubes in
the browser. This module is the server side of that contract: it abstracts
the three heterogeneous data sources behind one tiny interface so the
``/viewer`` routes don't care where pixels come from.

A *collection* is a named source of indexable cutouts. Each registered
collection provides:

* ``meta(params) -> dict`` — ``count``, ``tiers`` (``[{key,label}]``),
  ``default_tier``, ``band_names``, and optionally an ``objects`` list
  (per-index label/grade/available-tiers, used by ``evaluation``).
* ``cube(index, tier, params) -> (ndarray (H, W, C) float32, info)`` where
  ``info`` is ``{label, asinh, pixscale}``.

Three collections are registered:

==============  =========  ======================================  ==========
collection      params     tiers                                    source
==============  =========  ======================================  ==========
``sky``         subset     dirty→LR, hr→HR, bhr→BHR                  TFRecords
``cutouts``     —          real→Euclid                               per-band FITS
``evaluation``  —          LR / SR / HR (per object)                 object FITS
``psfs``        —          VIS / Y_E / J_E / H_E cluster kernels     FASRC ePSF FITS
==============  =========  ======================================  ==========

Band order is always ``Config.LR_INPUT_BAND_NAMES = (VIS, Y_E, J_E, H_E)``.
"""
from __future__ import annotations

import contextlib
import json
import math
import os
import re
import warnings
from collections import OrderedDict
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
from astropy.io import fits

from euclid_polish.config import Config
from euclid_polish.eval.combiner import (
    ACTIVE_COMBINER_KINDS,
    COMBINER_MODELS,
    RAW_INCREMENTAL_MINMEANMAX_RBF_KIND,
)
from euclid_polish.image.tfio import read_images, tfrecord_path
from euclid_polish.psf.core import PSF
from euclid_polish.training.target_blur import (
    blur_target_array,
    validate_target_fwhm_arcsec,
)
from euclid_polish.web.helpers import sky_records
from euclid_polish.web.helpers.paths import _sky_records_local_dir
from euclid_polish.web.helpers.status import (
    _ensure_local_star_cutout,
    _record_count,
    _valid_4band_stars,
)

#: Band names + channel order shared by every collection.
BAND_NAMES: tuple[str, ...] = tuple(Config.LR_INPUT_BAND_NAMES)
BHR_FWHM_PARAM = "bhr_fwhm_arcsec"
BHR_FWHM_MAX_ARCSEC = float(Config.TARGET_PSF_FWHM_MAX_ARCSEC)


def _bhr_fwhm_arcsec(
    params: Mapping[str, str],
    default: float = Config.TARGET_PSF_FWHM_ARCSEC,
) -> float:
    """Validated viewer-controlled target-preview FWHM."""
    raw = params.get(BHR_FWHM_PARAM)
    try:
        fwhm = validate_target_fwhm_arcsec(default if raw is None else raw)
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


def _as_hwc(arr: np.ndarray) -> np.ndarray:
    """Normalise a FITS/record array to ``(H, W, C)`` float32.

    Accepts ``(C, H, W)`` (FITS cube convention), ``(H, W, C)`` (records),
    or ``(H, W)`` (single band → 1 channel).
    """
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 2:
        return a[..., None]
    if a.ndim != 3:
        raise ViewerError(415, f"expected 2-D/3-D array, got {a.shape}")
    # FITS cubes are (C, H, W) with a small leading axis; records are
    # (H, W, C). Disambiguate by which axis is the short (band) one.
    if a.shape[0] <= 4 and a.shape[0] < a.shape[-1]:
        return np.moveaxis(a, 0, -1)
    return a


# ---------------------------------------------------------------------------
# sky — multi-band TFRecords (FASRC-synced cache)
# ---------------------------------------------------------------------------

# Tiers offered for sky records: LR (the dirty record), raw HR (the starfull
# scene), BHR (that scene with the target PSF), and SR (model output, generated
# on demand by the /sky button — disabled until at least one SR cube exists).
# The clean record is the deliberately starless target and must not be
# substituted for HR.
_SKY_RECORD_TIERS = [
    {"key": "dirty", "label": "LR"},
    {"key": "hr", "label": "HR"},
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
    tiers = [dict(t) for t in _SKY_RECORD_TIERS
             if os.path.exists(tfrecord_path(records_dir, f"{t['key']}_{subset}"))]
    counts = {t["key"]: (_record_count(f"{t['key']}_{subset}", records_dir) or 0)
              for t in tiers}
    if "hr" in counts:
        hr_position = next(i for i, tier in enumerate(tiers)
                           if tier["key"] == "hr")
        tiers.insert(hr_position + 1, {
            "key": "bhr", "label": "BHR (blurred HR)",
        })
        counts["bhr"] = counts["hr"]
    count = max(counts.values()) if counts else 0
    # SR is always offered so the user can see it exists; it's disabled until
    # the model has been run over the records (the "Generate SR" button).
    n_sr = sky_records.sr_count(subset)
    tiers.append({"key": "sr", "label": "SR", "disabled": n_sr == 0})
    counts["sr"] = n_sr
    default = "dirty" if any(t["key"] == "dirty" for t in tiers) else (
        tiers[0]["key"] if tiers else "dirty")
    return {
        "count": count,
        "tiers": tiers,
        "default_tier": default,
        "band_names": list(BAND_NAMES),
        "tier_counts": counts,
    }


def _sky_cube(index: int, tier: str, params: dict[str, str]):
    subset = _sky_subset(params)
    if tier == "sr":
        path = sky_records.sr_path(subset, index)
        if not os.path.isfile(path):
            raise ViewerError(404, "SR not generated for this record")
        cube = _as_hwc(np.load(path))
        return cube, {
            "label": f"sr · {subset} · idx {index}",
            "asinh": float(Config.STRETCH_SCALE_E),
            "pixscale": float(Config.DEFAULT_PIXEL_SCALE),
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
    cube = _as_hwc(rec.data)
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
    }
    return cube, info


# ---------------------------------------------------------------------------
# cutouts — real Euclid stars valid in all 4 bands (per-band FITS, stacked)
# ---------------------------------------------------------------------------

def _cutouts_meta(params: dict[str, str]) -> dict[str, Any]:
    _size, ids = _valid_4band_stars(force=False)
    return {
        "count": len(ids),
        "tiers": [{"key": "real", "label": "Euclid"}],
        "default_tier": "real",
        "band_names": list(BAND_NAMES),
    }


def _read_fits_plane(path: str) -> np.ndarray:
    with fits.open(path, memmap=False) as hdul:
        for hdu in hdul:
            if hdu.data is not None and getattr(hdu.data, "ndim", 0) == 2:
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
    for band in BAND_NAMES:
        path = _ensure_local_star_cutout(band, sid, size)
        if not path:
            raise ViewerError(404, f"{band} cutout unavailable")
        planes.append(_read_fits_plane(path))
    shapes = {p.shape for p in planes}
    if len(shapes) != 1:
        raise ViewerError(415, f"band cutouts disagree in shape: {shapes}")
    cube = np.stack(planes, axis=-1)
    info = {
        "label": f"star {sid} · {size}px",
        "asinh": float(Config.STRETCH_SCALE_E),
        "pixscale": float(Config.VIS_PIXEL_SCALE_ARCSEC),
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
def _eval_objects() -> list[dict[str, Any]]:
    """Return manifest objects with their available on-disk tiers and grade."""
    from euclid_polish.web.routes.evaluation import _read_manifest  # local: avoid cycle

    root = os.path.abspath(Config.EVAL_RESULTS_DIR)
    rows = _read_manifest(root)
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
        objs.append({
            "subdir": sub,
            "label": (f"{r.get('id', sub)}" + (f" · {grade}" if grade else "")),
            "grade": grade,
            "tiers": tiers,
            "pca_n": pca_n,
            "pca_amps": pca_amps,
            "pca_var": pca_var,
        })
    return objs


def _eval_meta(params: dict[str, str]) -> dict[str, Any]:
    objs = _eval_objects()
    # All tiers seen across the run, ordered LR→SR→HR→BHR→std, for the chip strip.
    order = ["LR", "SR", "HR", "BHR", "std"]
    seen = {t for o in objs for t in o["tiers"]}
    tiers = [{"key": k, "label": ("stdSR" if k == "std" else k)}
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
        "objects": [{"label": o["label"], "grade": o["grade"],
                     "tiers": o["tiers"], "subdir": o["subdir"]}
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
        data = hdul[0].data
        with contextlib.suppress(TypeError, ValueError):
            asinh = float(hdul[0].header.get("ASINH", asinh))
    cube = _as_hwc(data)
    if key == "BHR":
        cube = blur_target_array(
            cube, _bhr_fwhm_arcsec(params),
            pixel_scale_arcsec=Config.DEFAULT_PIXEL_SCALE,
        )
    tier_scale = (Config.VIS_PIXEL_SCALE_ARCSEC
                  if tier.lower() in {"lr", "original", "original_stack"}
                  else Config.DEFAULT_PIXEL_SCALE)
    display_tier = "BHR (blurred HR)" if key == "BHR" else tier
    info = {"label": f"{obj['label']} · {display_tier}", "asinh": asinh,
            "pixscale": float(tier_scale)}
    return cube, info


# ---------------------------------------------------------------------------
# ensemble — LR / SR(combiner) / stdSR(std) / HR disagreement viewer
# ---------------------------------------------------------------------------
#
# The /ensemble "Evaluate" job caches the ensemble-mean and per-pixel std
# (stdSR) cubes under <vis>/ensemble/cubes/{sr,std}_<recidx>.npy plus a
# viz_index.json {subset, indices}. LR/HR are read back from the sky records by
# record index.  The public ``sr`` viewer tier is replaced by the primary fitted
# combiner below; the cached mean remains the centre of the disagreement movie.

_ENSEMBLE_TIERS = [
    {"key": "lr", "label": "LR"},
    {"key": "sr", "label": "SR (minibatched convex all-asinh RBF)"},
    # stdSR stays available (it powers the ±σ magnitude on the mean frame) but
    # is hidden from the chip row per the trimmed tier set.
    {"key": "std", "label": "stdSR", "hidden": True},
    {"key": "hr", "label": "HR"},
    {"key": "bhr", "label": "BHR (blurred HR)"},
]

#: How many PCA components the on-the-fly (member-subset) disagreement movie
#: keeps — matches the baked ``ENSEMBLE_PCA_COMPONENTS``.
_MORPH_PCA_COMPONENTS = 3


def _ensemble_starless(params: dict[str, str]) -> bool:
    """The star regime the viewer is showing (``?mode=starfull|starless``). The
    two regimes' cubes are fully detached; default starless (the production
    reconstruction)."""
    return (params.get("mode", "starless") or "starless").lower() != "starfull"


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
    member_labels0 = man.get("member_labels", []) or []
    primary_available = bool(
        man.get(f"has_combiner_{RAW_INCREMENTAL_MINMEANMAX_RBF_KIND}")
        or man.get("has_combiner")
        or (member_labels0 and _load_field_combiner(
            starless, member_labels0,
            RAW_INCREMENTAL_MINMEANMAX_RBF_KIND) is not None)
    )
    if not primary_available:
        tiers = [tier for tier in tiers if tier["key"] != "sr"]
    # Each ordinary combiner model gets its own selectable tier. Cubes are
    # computed on demand from the shared member cache when not baked by eval.
    for kind, key, label in reversed(tuple(
            (kind, spec.cube_prefix, spec.label)
            for kind, spec in COMBINER_MODELS.items()
            if kind in ACTIVE_COMBINER_KINDS
            and kind != RAW_INCREMENTAL_MINMEANMAX_RBF_KIND)):
        if man.get(f"has_combiner_{kind}") or (
                member_labels0 and _load_field_combiner(
                    _ensemble_starless(params), member_labels0, kind) is not None):
            tiers.insert(2, {"key": key, "label": label})
    # Individual member SR tiers, labelled from the eval. HIDDEN from the tier
    # chip row (they'd swamp it at 22 members) but still loadable on demand:
    # the React member panel searches/sorts them and toggles one in via the
    # engine's setTiers, and their cubes feed the member-subset movie.
    member_labels = man.get("member_labels", []) or []
    tiers += [{"key": f"member{i}", "label": f"SR {lab}", "hidden": True}
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
    return {
        "count": len(idxs),
        "tiers": tiers,
        "default_tier": "sr" if primary_available else "lr",
        "band_names": list(BAND_NAMES),
        "subset": sub,
        "pca_n": pca_n,
        "pca_amps": pca_amps,
        # Member index → label, for the React panel to join psnr/loss/depth
        # (from status.json) and drive the member-subset disagreement movie.
        "member_labels": list(member_labels),
        "pca_max": _MORPH_PCA_COMPONENTS,
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
    data = _as_hwc(rec.data)
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
    from euclid_polish.ensemble import pca_field
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
                         model_kind: str = "raw_incremental_minmeanmax_rbf"):
    """The regime's fitted combiner if it exists AND its membership matches the
    cube stack (``member_labels``), else ``None``. Cheap (an ~8 KB npz)."""
    if not member_labels:
        return None
    from euclid_polish.eval.combiner import load_combiner
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
                         model_kind: str = "raw_incremental_minmeanmax_rbf") -> np.ndarray:
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
    out = np.asarray(comb.apply_field(np.stack(stack, axis=0)), np.float32)
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

    # Member-subset disagreement movie: recompute sr (=subset mean) and the PCA
    # eigen-images on the fly for the requested members. amp/var are subset-
    # dependent → returned as headers so the animation reads the right spread.
    is_pca = tier.startswith("pca") and tier[3:].isdigit()
    subset = _parse_member_subset(
        params.get("members"), len(man.get("member_labels", []) or []))
    if subset is not None and (tier == "sr" or is_pca):
        mean, comps, amps, var = _subset_pca(starless, rec_index, subset)
        tag = f"{len(subset)} of {len(man.get('member_labels', []) or [])} members"
        if tier == "sr":
            return _as_hwc(mean), {
                "label": f"SR (subset mean · {tag}) · {sub} · idx {rec_index}",
                "asinh": float(Config.STRETCH_SCALE_E),
                "pixscale": float(Config.DEFAULT_PIXEL_SCALE)}
        k = int(tier[3:])
        if k >= len(comps):
            raise ViewerError(404, "pca component out of range")
        return _as_hwc(comps[k]), {
            "label": f"PC{k} · {tag}", "asinh": float(Config.STRETCH_SCALE_E),
            "pixscale": float(Config.DEFAULT_PIXEL_SCALE), "amp": float(amps[k]),
            "var": float(var[k]) if k < len(var) else 0.0}
    # The main SR tier is the primary fitted combiner.  Keep ``sr`` as the
    # stable viewer/API key while avoiding a duplicate primary-combiner chip.
    # A requested member subset is handled above as a subset mean because a
    # combiner fitted for the full ordered membership cannot accept fewer
    # member channels.
    if tier == "sr":
        tier = COMBINER_MODELS[
            RAW_INCREMENTAL_MINMEANMAX_RBF_KIND].cube_prefix

    # Combiner reconstruction: prefer a baked model-specific cube; otherwise
    # apply that fitted model to the cached member stack on the fly.
    tier_kinds = {COMBINER_MODELS[kind].cube_prefix: kind
                  for kind in ACTIVE_COMBINER_KINDS}
    if tier in tier_kinds:
        model_kind = tier_kinds[tier]
        prefix = COMBINER_MODELS[model_kind].cube_prefix
        baked = os.path.join(_ensemble_cubes_dir(starless),
                             f"{prefix}_{rec_index:05d}.npy")
        cube = (_as_hwc(np.load(baked)) if os.path.isfile(baked)
                else _as_hwc(_combiner_field_cube(
                    starless, rec_index, man.get("member_labels", []) or [], model_kind)))
        label = COMBINER_MODELS[model_kind].label
        return cube, {"label": f"SR ({label}) · {sub} · idx {rec_index}",
                      "asinh": float(Config.STRETCH_SCALE_E),
                      "pixscale": float(Config.DEFAULT_PIXEL_SCALE)}
    # Records are written index==position from 0, so reading up to the largest
    # cached index covers every LR/goal field we need.
    n_read = (max(int(i) for i in idxs) + 1) if idxs else 1
    # sr / std, the PCA eigen-images (pca0…) and individual member SRs
    # (member0…) are cached .npy cubes; LR and the regime goal come from the
    # records. pcaN are served on demand for the animation (not advertised as
    # static tiers). The stable ``hr`` tier key means "goal" here: clean for
    # starless and hr for starfull.
    is_npy = (tier in ("sr", "std")
              or (tier.startswith("pca") and tier[3:].isdigit())
              or (tier.startswith("member") and tier[6:].isdigit()))
    if is_npy:
        path = os.path.join(_ensemble_cubes_dir(starless),
                            f"{tier}_{rec_index:05d}.npy")
        if not os.path.isfile(path):
            raise ViewerError(404, f"{tier} cube missing")
        cube, pix = _as_hwc(np.load(path)), float(Config.DEFAULT_PIXEL_SCALE)
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
    labels = {"lr": "LR", "sr": "SR (ensemble mean)",
              "std": "stdSR (member std)", "hr": target_label,
              "bhr": _blurred_target_label(target_label),
              "comb": "SR (combiner)"}
    if tier.startswith("member") and tier[6:].isdigit():
        mlabels = man.get("member_labels", []) or []
        mi = int(tier[6:])
        label = f"SR {mlabels[mi]}" if mi < len(mlabels) else tier
    else:
        label = labels.get(tier, tier)
    info = {"label": f"{label} · {sub} · idx {rec_index}",
            "asinh": float(Config.STRETCH_SCALE_E), "pixscale": pix}
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
# real-field — cached 10x10 tiles from one real Euclid archive field
# ---------------------------------------------------------------------------

def _real_field_manifest(params: dict[str, str]) -> dict[str, Any]:
    from euclid_polish.web.helpers.real_field import latest_field, manifest_path

    identifier = (params.get("field") or "").strip()
    if identifier:
        try:
            with manifest_path(identifier).open() as f:
                return json.load(f)
        except (OSError, ValueError):
            raise ViewerError(404, "real field not cached") from None
    manifest = latest_field()
    if manifest is None:
        raise ViewerError(404, "no real Euclid field cached")
    return manifest


def _real_field_meta(params: dict[str, str]) -> dict[str, Any]:
    manifest = _real_field_manifest(params)
    labels = list(manifest.get("member_labels", []) or [])
    tiers = [
        {"key": "lr", "label": "LR"},
        {"key": "sr", "label": "SR (mean)"},
        {"key": "std", "label": "stdSR", "hidden": True},
    ]
    for kind, spec in COMBINER_MODELS.items():
        if kind not in ACTIVE_COMBINER_KINDS:
            continue
        if kind in set(manifest.get("combiner_kinds", []) or []):
            tiers.append({"key": spec.cube_prefix, "label": spec.label})
    tiers += [{"key": f"member{i}", "label": f"SR {label}", "hidden": True}
              for i, label in enumerate(labels)]
    if int(manifest.get("pca_n", 0) or 0) > 0:
        tiers.append({"key": "morph", "label": "disagreement movie"})
    count = int(manifest.get("count", 0) or 0)
    side = int(manifest.get("grid_side", 10) or 10)
    return {
        "count": count, "tiers": tiers, "default_tier": "sr",
        "band_names": list(BAND_NAMES), "member_labels": labels,
        "pca_n": int(manifest.get("pca_n", 0) or 0),
        "pca_amps": [list((manifest.get("pca_amps", {}) or {}).get(str(i), []))
                     for i in range(count)],
        "pca_var": [list((manifest.get("pca_var", {}) or {}).get(str(i), []))
                    for i in range(count)],
        "objects": [
            {"label": f"tile {i + 1:03d} · row {i // side + 1}, col {i % side + 1}",
             "tiers": [t["key"] for t in tiers]}
            for i in range(count)
        ],
    }


def _real_field_cube(index: int, tier: str, params: dict[str, str]):
    manifest = _real_field_manifest(params)
    count = int(manifest.get("count", 0) or 0)
    if index < 0 or index >= count:
        raise ViewerError(404, "tile index out of range")
    from euclid_polish.web.helpers.real_field import field_dir
    path = field_dir(str(manifest["field_id"])) / "cubes" / f"{tier}_{index:03d}.npy"
    if not path.is_file():
        raise ViewerError(404, f"{tier} cube is not cached")
    cube = _as_hwc(np.load(path))
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
    tier_scale = (Config.VIS_PIXEL_SCALE_ARCSEC
                  if tier.lower() == "lr" else Config.DEFAULT_PIXEL_SCALE)
    return cube, {"label": f"{label} · tile {index + 1:03d}",
                  "asinh": float(Config.STRETCH_SCALE_E),
                  "pixscale": float(tier_scale)}


# ---------------------------------------------------------------------------
# psfs — one navigable object per spatial ePSF cluster, one tier per band
# ---------------------------------------------------------------------------

def _psf_paths() -> dict[str, str]:
    """Return the already-synchronised FASRC ePSF paths by band."""
    from euclid_polish.web.helpers.status import _cached_fasrc_psf_dir

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
        header_count = hdul[0].header.get("NPSF")
        if header_count is not None:
            return max(1, int(header_count))
        image_hdus = [h for h in hdul if getattr(h, "data", None) is not None]
        return max(1, len(image_hdus) - 1) if len(image_hdus) > 1 else 1


def _psf_meta(_params: dict[str, str]) -> dict[str, Any]:
    paths = _psf_paths()
    counts = {name: _psf_count(path) for name, path in paths.items()}
    count = max(counts.values(), default=0)
    tiers = [
        {"key": name, "label": name, "disabled": name not in counts}
        for name in BAND_NAMES
    ]
    objects = [
        {
            "label": f"PSF cluster {index + 1:03d}",
            "tiers": [name for name, n in counts.items() if index < n],
        }
        for index in range(count)
    ]
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
    from euclid_polish.web import job_config

    cfg = job_config.load()
    return float(cfg.psf_warp_alpha_max), float(cfg.psf_warp_sigma)


def _psf_cube(index: int, tier: str, params: dict[str, str]):
    if tier not in BAND_NAMES:
        raise ViewerError(400, f"bad PSF band: {tier}")
    path = _psf_paths().get(tier)
    if path is None:
        raise ViewerError(404, f"{tier} PSF not synchronised")

    with fits.open(path, memmap=True) as hdul:
        image_hdus = [h for h in hdul if getattr(h, "data", None) is not None]
        cluster_hdus = image_hdus[1:] if len(image_hdus) > 1 else image_hdus
        if index < 0 or index >= len(cluster_hdus):
            raise ViewerError(404, "PSF cluster out of range")
        hdu = cluster_hdus[index]
        data = np.asarray(hdu.data, dtype=np.float32).copy()
        header = hdu.header
        pixel_scale = float(header.get(
            "PXSCALE", header.get("PIXSCALE", 0.0)))
        n_stars = header.get("NSTARS")
        label = f"{tier} PSF · cluster {index + 1:03d}"
        if n_stars is not None:
            label += f" · {int(n_stars):,} stars"

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
    }


# ---------------------------------------------------------------------------
# jwst-euclid — one saved paired field, preserving each native image grid
# ---------------------------------------------------------------------------

_PAIR_ID = re.compile(r"^[A-Za-z0-9._-]{1,220}$")


def _jwst_euclid_pair(params: dict[str, str]) -> tuple[dict[str, Any], str]:
    """Load a verified paired-field manifest without exposing its cache path."""
    from euclid_polish.web.helpers.jwst_euclid import (
        _cached_pair_is_usable,
        enrich_manifest_metadata,
        pair_root,
    )

    identifier = (params.get("field") or "").strip()
    if not _PAIR_ID.fullmatch(identifier):
        raise ViewerError(404, "paired field not found")
    directory = pair_root() / identifier
    try:
        with (directory / "manifest.json").open(encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, ValueError):
        raise ViewerError(404, "paired field not found") from None
    if not isinstance(manifest, dict) or not _cached_pair_is_usable(directory, manifest):
        raise ViewerError(404, "paired field is incomplete")
    return enrich_manifest_metadata(directory, manifest), str(directory)


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
    from astropy.wcs import WCS

    try:
        with fits.open(path, memmap=False) as hdul:
            primary = hdul[0].header
            for hdu in hdul:
                data = getattr(hdu, "data", None)
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


def _pair_cube(path: str) -> np.ndarray:
    """Read a 2-D image or a small channel-first cube from a pair product."""
    try:
        with fits.open(path, memmap=False) as hdul:
            for hdu in hdul:
                data = getattr(hdu, "data", None)
                if data is None or np.ndim(data) not in (2, 3):
                    continue
                return _as_hwc(np.asarray(data, np.float32))
    except OSError as exc:
        raise ViewerError(404, "paired field FITS is unreadable") from exc
    raise ViewerError(404, "paired field FITS has no image cube")


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
    from astropy.coordinates import SkyCoord

    from euclid_polish.web.helpers.jwst_euclid import _native_sky_cutout

    data, _, wcs = _pair_image(_pair_file(directory, entry.get("file")))
    if entry.get("native_is_field_cutout"):
        return data, wcs
    try:
        coordinate = SkyCoord(
            ra=float(manifest["ra_deg"]), dec=float(manifest["dec_deg"]), unit="deg", frame="icrs",
        )
        return _native_sky_cutout(data, wcs, coordinate, float(manifest["size_arcsec"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ViewerError(404, "legacy JWST product has no usable field geometry") from exc


def _pair_asinh(cube: np.ndarray) -> float:
    finite = np.abs(cube[np.isfinite(cube)])
    if finite.size == 0:
        return 1.0
    return max(float(np.nanpercentile(finite, 95.0)), 1e-8)


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
    from scipy.ndimage import gaussian_filter

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


def _jwst_filter_tint(band: str) -> list[float]:
    """Give an uncalibrated single JWST filter a clearly labelled display tint."""
    match = re.search(r"F(\d{3,4})", band.upper())
    wavelength_um = float(match.group(1)) / 100.0 if match else None
    if wavelength_um is None:
        return [0.92, 0.92, 0.96]
    if wavelength_um <= 1.2:
        return [0.38, 0.62, 1.0]
    if wavelength_um <= 1.8:
        return [0.32, 0.92, 0.66]
    if wavelength_um <= 2.6:
        return [1.0, 0.72, 0.26]
    return [1.0, 0.34, 0.30]


def _saved_jwst_euclid_pairs() -> list[tuple[dict[str, Any], str]]:
    """Return saved paired fields in the stable location-carousel order."""
    from euclid_polish.web.helpers.jwst_euclid import (
        pair_root,
        saved_pairs,
    )

    pairs: list[tuple[dict[str, Any], str]] = []
    for manifest in saved_pairs():
        identifier = str(manifest.get("field_id") or "")
        if not _PAIR_ID.fullmatch(identifier):
            continue
        directory = pair_root() / identifier
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
) -> tuple[list[dict[str, Any]], list[np.ndarray], str, float]:
    """Return the usable JWST planes on the finest native JWST display WCS."""
    from euclid_polish.web.helpers.jwst_euclid import align_to_target

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
            plane = data if wcs is reference_wcs else align_to_target(
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
    return aligned_entries, aligned_planes, reference_filter, reference_scale


def _jwst_colour_cube(manifest: dict[str, Any], directory: str) -> tuple[np.ndarray, dict[str, Any]]:
    """Build a display-only RGB composite while retaining the source FITS grids.

    Native JWST files remain untouched in the cache.  For the viewer only,
    every usable filter is sampled onto the finest saved JWST WCS, then split
    by wavelength into blue/green/red groups.  The aligned native brightnesses
    are combined directly: one shared display stretch is applied to the final
    cube, so inter-filter brightness ratios remain intact.
    """
    aligned_entries, aligned_planes, reference_filter, reference_scale = _jwst_aligned_planes(
        manifest, directory,
    )

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
    }


def _jwst_temperature_cube(manifest: dict[str, Any], directory: str) -> tuple[np.ndarray, dict[str, Any]]:
    """Build an approximate JWST temperature cube from native brightnesses."""
    entries, planes, reference_filter, reference_scale = _jwst_aligned_planes(manifest, directory)
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
    }


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
        {"key": "sr", "label": "SR · STARFULL combiner"},
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
    if tier == "lr":
        source = inference_files.get("lr") or files.get("euclid")
        cube = _pair_cube(_pair_file(directory, source))
        bands = list(BAND_NAMES[:cube.shape[-1]])
        return cube, {
            "label": "LR · Euclid VIS",
            "asinh": float(Config.STRETCH_SCALE_E),
            "pixscale": float(Config.VIS_PIXEL_SCALE_ARCSEC),
            "bands": bands,
            "display_scale": _robust_display_scale(cube),
            "transfer_group": "euclid",
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
            data, _wcs = _pair_native_jwst(manifest, directory, entry)
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
    from euclid_polish.web.helpers.jwst_euclid import (
        _read_nexus_field_manifest,
        nexus_field_root,
    )

    identifier = (params.get("field") or "").strip()
    if not _PAIR_ID.fullmatch(identifier):
        raise ViewerError(404, "NEXUS tiled field not found")
    manifest = _read_nexus_field_manifest(identifier)
    if manifest is None:
        raise ViewerError(404, "NEXUS tiled field not found")
    return manifest, str(nexus_field_root() / identifier)


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

    return {
        "count": len(tiles),
        "tiers": [
            {"key": "lr", "label": "LR · Euclid · 255 px"},
            {"key": "sr", "label": "SR · STARFULL combiner"},
            {"key": "jwst", "label": f"NEXUS {manifest.get('filter') or 'JWST'} · native"},
            {"key": "jwst_blur", "label": "JWST · Gaussian blur · FWHM 1 SR px"},
        ],
        "default_tier": "lr",
        "band_names": list(BAND_NAMES),
        "color_label": "Euclid colour",
        "missing_tier_labels": {"sr": "Generate SR"},
        "transfer_groups": ["euclid", "jwst"],
        "objects": [{
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
    if tier == "lr":
        lr_file = tile.get("lr_file")
        cube = _pair_cube(_pair_file(directory, lr_file or tile.get("euclid_file")))
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
        }
    if tier in {"jwst", "jwst_blur"}:
        cube = _pair_cube(_pair_file(directory, tile.get("jwst_file")))
        metadata = tile.get("jwst_metadata", {}) if isinstance(tile, Mapping) else {}
        scales = metadata.get("pixel_scale_arcsec", []) if isinstance(metadata, Mapping) else []
        scale = float(scales[0]) if isinstance(scales, list) and scales else 0.0
        result = cube, {
            "label": f"NEXUS native · {manifest.get('filter') or 'JWST'}",
            "asinh": 100.0, "pixscale": scale if math.isfinite(scale) else 0.0,
            "bands": [str(manifest.get("filter") or "JWST")],
            "display_scale": _robust_display_scale(cube),
            "transfer_group": "jwst",
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
# registry
# ---------------------------------------------------------------------------

_Meta = Callable[[dict[str, str]], dict[str, Any]]
_Cube = Callable[[int, str, dict[str, str]], tuple[np.ndarray, dict[str, Any]]]

_REGISTRY: dict[str, tuple[_Meta, _Cube]] = {
    "sky": (_sky_meta, _sky_cube),
    "cutouts": (_cutouts_meta, _cutouts_cube),
    "evaluation": (_eval_meta, _eval_cube),
    "ensemble": (_ensemble_meta, _ensemble_cube),
    "real-field": (_real_field_meta, _real_field_cube),
    "jwst-euclid": (_jwst_euclid_meta, _jwst_euclid_cube),
    "nexus-field": (_nexus_field_meta, _nexus_field_cube),
    "psfs": (_psf_meta, _psf_cube),
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


def get_cube(collection: str, index: int, tier: str,
             params: dict[str, str]) -> tuple[np.ndarray, dict[str, Any]]:
    if collection not in _REGISTRY:
        raise ViewerError(404, "unknown collection")
    cube, info = _REGISTRY[collection][1](index, tier, params)
    return np.ascontiguousarray(cube, dtype=np.float32), info
