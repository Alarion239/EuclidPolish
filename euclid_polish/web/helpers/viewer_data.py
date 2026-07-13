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
``sky``         subset     dirty→LR, clean→HR, hr→HR-target          TFRecords
``cutouts``     —          real→Euclid                               per-band FITS
``evaluation``  —          LR / SR / HR (per object)                 object FITS
==============  =========  ======================================  ==========

Band order is always ``Config.LR_INPUT_BAND_NAMES = (VIS, Y_E, J_E, H_E)``.
"""
from __future__ import annotations

import contextlib
import json
import math
import os
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

import numpy as np
from astropy.io import fits

from euclid_polish.config import Config
from euclid_polish.eval.lensfinder_eval import per_object_plens
from euclid_polish.image.tfio import read_images, tfrecord_path
from euclid_polish.web.helpers import sky_records
from euclid_polish.web.helpers.paths import _sky_records_local_dir
from euclid_polish.web.helpers.status import (
    _ensure_local_star_cutout,
    _record_count,
    _valid_4band_stars,
)

#: Band names + channel order shared by every collection.
BAND_NAMES: tuple[str, ...] = tuple(Config.LR_INPUT_BAND_NAMES)


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

# Tiers offered for sky records: LR (the dirty record), HR (the clean
# record), and SR (model output, generated on demand by the /sky button —
# disabled until at least one SR cube exists). The "hr target" record is the
# same clean 4-band sky since the VIS+NISP-output change, so it isn't a
# separate tier.
_SKY_RECORD_TIERS = [
    {"key": "dirty", "label": "LR"},
    {"key": "clean", "label": "HR"},
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
            "pixscale": 0.0,
        }
    if tier not in ("dirty", "clean"):
        raise ViewerError(400, "bad tier")
    path = tfrecord_path(_sky_records_local_dir(), f"{tier}_{subset}")
    if not os.path.exists(path):
        raise ViewerError(404, "records not synced")
    records = read_images(path, num_images=max(index + 1, 1))
    if not records or index >= len(records):
        raise ViewerError(404, "index out of range")
    rec = records[index]
    cube = _as_hwc(rec.data)
    info = {
        "label": f"{tier} · {subset} · idx {rec.index}",
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
#: Viewer tier key → ``lens_scores.csv`` recon column for the P(lens) lookup.
_EVAL_TIER_PLENS = {"LR": "lr", "SR": "sr", "HR": "hr"}


def _eval_objects() -> list[dict[str, Any]]:
    """Manifest objects (ok rows) with on-disk tiers, label/grade, and P(lens).

    ``plens`` carries the headed lens-finder's prediction per tier (``{"LR":
    0.12, "SR": 0.87, ...}``) so the viewer can show what the model thinks for
    each render. Only finite scores present in ``lens_scores.csv`` are included;
    a tier with no score (e.g. an unscored run, or HR for a real lens) is simply
    absent and the viewer shows no badge for it.
    """
    from euclid_polish.web.routes.evaluation import _read_manifest  # local: avoid cycle

    root = os.path.abspath(Config.EVAL_RESULTS_DIR)
    rows = _read_manifest(root)
    plens_by_id = per_object_plens(root)        # {id: {"lr": P, "sr": P, "hr": P}}
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
        if not tiers:
            continue
        grade = (r.get("grade") or "").strip()
        scores = plens_by_id.get(r.get("id") or sub, {})
        plens = {tier: float(scores[col])
                 for tier, col in _EVAL_TIER_PLENS.items()
                 if math.isfinite(scores.get(col, float("nan")))}
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
            "plens": plens,
            "pca_n": pca_n,
            "pca_amps": pca_amps,
            "pca_var": pca_var,
        })
    return objs


def _eval_meta(params: dict[str, str]) -> dict[str, Any]:
    objs = _eval_objects()
    # All tiers seen across the run, ordered LR→SR→HR→std, for the chip strip.
    order = ["LR", "SR", "HR", "std"]
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
                     "tiers": o["tiers"], "subdir": o["subdir"],
                     "plens": o["plens"]}
                    for o in objs],
    }


def _eval_cube(index: int, tier: str, params: dict[str, str]):
    objs = _eval_objects()
    if index < 0 or index >= len(objs):
        raise ViewerError(404, "index out of range")
    obj = objs[index]
    root = os.path.abspath(Config.EVAL_RESULTS_DIR)
    asinh = float(Config.STRETCH_SCALE_E)
    if tier.startswith("pca") and tier[3:].isdigit():
        path = os.path.join(root, obj["subdir"], f"{tier}.fits")
        if not os.path.isfile(path):
            raise ViewerError(404, f"{tier} not available for this object")
    else:
        # The shared morph animation fetches the ensemble mean as lower-case
        # "sr", but eval tier keys are upper-case (LR/SR/HR); resolve case-
        # insensitively so the disagreement movie works on the eval page too.
        key = next((k for k in _EVAL_TIER_FILES if k.lower() == tier.lower()), None)
        if key is None or key not in obj["tiers"]:
            raise ViewerError(404, f"{tier} not available for this object")
        path = os.path.join(root, obj["subdir"], _EVAL_TIER_FILES[key])
    with fits.open(path, memmap=False) as hdul:
        data = hdul[0].data
        with contextlib.suppress(TypeError, ValueError):
            asinh = float(hdul[0].header.get("ASINH", asinh))
    cube = _as_hwc(data)
    info = {"label": f"{obj['label']} · {tier}", "asinh": asinh, "pixscale": 0.0}
    return cube, info


# ---------------------------------------------------------------------------
# ensemble — LR / SR(mean) / stdSR(std) / HR for the disagreement viewer
# ---------------------------------------------------------------------------
#
# The /ensemble "Evaluate" job caches the ensemble-mean (SR) and per-pixel std
# (stdSR) cubes under <vis>/ensemble/cubes/{sr,std}_<recidx>.npy plus a
# viz_index.json {subset, indices}. LR/HR are read back from the sky records by
# record index, so only the computed cubes are duplicated.

_ENSEMBLE_TIERS = [
    {"key": "lr", "label": "LR"},
    {"key": "sr", "label": "SR (mean)"},
    # stdSR stays available (it powers the ±σ magnitude on the mean frame) but
    # is hidden from the chip row per the trimmed tier set.
    {"key": "std", "label": "stdSR", "hidden": True},
    {"key": "hr", "label": "HR"},
]

#: How many PCA components the on-the-fly (member-subset) disagreement movie
#: keeps — matches the baked ``ENSEMBLE_PCA_COMPONENTS``.
_MORPH_PCA_COMPONENTS = 3


def _ensemble_starless(params: dict[str, str]) -> bool:
    """The star regime the viewer is showing (``?mode=starfull|starless``). The
    two regimes' cubes are fully detached; default starless (the production
    reconstruction)."""
    return (params.get("mode", "starless") or "starless").lower() != "starfull"


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
    man = _ensemble_manifest(_ensemble_starless(params))
    idxs = man.get("indices", [])
    sub = man.get("subset", "")
    rdir = _sky_records_local_dir()
    has_hr = bool(sub) and bool(rdir) and os.path.exists(
        tfrecord_path(rdir, f"hr_{sub}"))
    tiers = [dict(t) for t in _ENSEMBLE_TIERS if t["key"] != "hr" or has_hr]
    member_labels0 = man.get("member_labels", []) or []
    # Combiner reconstruction tier: show it whenever a fitted combiner EXISTS for
    # this regime + membership — its cube is computed on the fly from the cached
    # member stack, so it no longer depends on the last eval having baked comb_
    # cubes. Placed right after the ensemble mean.
    if man.get("has_combiner") or (
            member_labels0 and _load_field_combiner(
                _ensemble_starless(params), member_labels0) is not None):
        tiers.insert(2, {"key": "comb", "label": "combiner"})
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
        "default_tier": "sr",
        "band_names": list(BAND_NAMES),
        "subset": sub,
        "pca_n": pca_n,
        "pca_amps": pca_amps,
        # Member index → label, for the React panel to join psnr/loss/depth
        # (from status.json) and drive the member-subset disagreement movie.
        "member_labels": list(member_labels),
        "pca_max": _MORPH_PCA_COMPONENTS,
    }


def _ensemble_record_cube(sub: str, n_read: int, kind: str, rec_index: int):
    """LR (``kind='dirty'``) or HR (``kind='hr'``) record matched by ``.index``."""
    rdir = _sky_records_local_dir()
    path = tfrecord_path(rdir, f"{kind}_{sub}") if rdir else ""
    if not rdir or not os.path.exists(path):
        raise ViewerError(404, f"{kind} records not available")
    recs = read_images(path, num_images=max(n_read, 1))
    rec = {r.index: r for r in recs}.get(rec_index)
    if rec is None:
        raise ViewerError(404, f"record {rec_index} not found")
    return (_as_hwc(rec.data),
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


def _load_field_combiner(starless: bool, member_labels: list[str]):
    """The regime's fitted combiner if it exists AND its membership matches the
    cube stack (``member_labels``), else ``None``. Cheap (an ~8 KB npz)."""
    if not member_labels:
        return None
    from euclid_polish.eval.combiner import load_combiner
    try:
        return load_combiner(_ensemble_regime_dir(starless),
                             member_labels=list(member_labels))
    except Exception:
        return None


# On-the-fly combiner reconstruction (per field), so the "combiner" tier shows
# whenever a combiner is fitted — not only when the last eval baked comb_ cubes.
_COMB_CUBE_CACHE: OrderedDict[tuple, np.ndarray] = OrderedDict()
_COMB_CUBE_MAX = 8


def _combiner_field_cube(starless: bool, rec_index: int,
                         member_labels: list[str]) -> np.ndarray:
    """The combiner reconstruction ``(H,W,C)`` for one field, applied to the
    cached full member stack. LRU-cached; raises 404 if no combiner / cubes."""
    key = ("starless" if starless else "starfull", int(rec_index),
           tuple(member_labels))
    hit = _COMB_CUBE_CACHE.get(key)
    if hit is not None:
        _COMB_CUBE_CACHE.move_to_end(key)
        return hit
    comb = _load_field_combiner(starless, member_labels)
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
                "asinh": float(Config.STRETCH_SCALE_E), "pixscale": 0.0}
        k = int(tier[3:])
        if k >= len(comps):
            raise ViewerError(404, "pca component out of range")
        return _as_hwc(comps[k]), {
            "label": f"PC{k} · {tag}", "asinh": float(Config.STRETCH_SCALE_E),
            "pixscale": 0.0, "amp": float(amps[k]),
            "var": float(var[k]) if k < len(var) else 0.0}
    # Combiner reconstruction: prefer a baked comb_ cube; otherwise apply the
    # fitted combiner to the cached member stack on the fly (so the tier works
    # even when the last eval predates the combiner fit).
    if tier == "comb":
        baked = os.path.join(_ensemble_cubes_dir(starless),
                             f"comb_{rec_index:05d}.npy")
        cube = (_as_hwc(np.load(baked)) if os.path.isfile(baked)
                else _as_hwc(_combiner_field_cube(
                    starless, rec_index, man.get("member_labels", []) or [])))
        return cube, {"label": f"SR (combiner) · {sub} · idx {rec_index}",
                      "asinh": float(Config.STRETCH_SCALE_E), "pixscale": 0.0}
    # Records are written index==position from 0, so reading up to the largest
    # cached index covers every LR/HR field we need.
    n_read = (max(int(i) for i in idxs) + 1) if idxs else 1
    # sr / std, the PCA eigen-images (pca0…) and individual member SRs
    # (member0…) are cached .npy cubes; lr / hr come from the records. pcaN are
    # served on demand for the animation (not advertised as static tiers).
    is_npy = (tier in ("sr", "std")
              or (tier.startswith("pca") and tier[3:].isdigit())
              or (tier.startswith("member") and tier[6:].isdigit()))
    if is_npy:
        path = os.path.join(_ensemble_cubes_dir(starless),
                            f"{tier}_{rec_index:05d}.npy")
        if not os.path.isfile(path):
            raise ViewerError(404, f"{tier} cube missing")
        cube, pix = _as_hwc(np.load(path)), 0.0
    elif tier == "lr":
        cube, pix = _ensemble_record_cube(sub, n_read, "dirty", rec_index)
    elif tier == "hr":
        cube, pix = _ensemble_record_cube(sub, n_read, "hr", rec_index)
    else:
        raise ViewerError(400, "bad tier")
    labels = {"lr": "LR", "sr": "SR (ensemble mean)",
              "std": "stdSR (member std)", "hr": "HR",
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


# Lens isolation deliberately uses a separate collection instead of pretending
# its ``lens_*`` targets are production ``hr_*`` records.  The viewer contract
# is otherwise identical to the ensemble collection, so the React viewer gets
# the same tiers, member-subset PCA movie, and combiner behavior.
def _lens_isolation_manifest() -> dict[str, Any]:
    from euclid_polish.web.helpers.lens_isolation_viz import cubes_dir

    path = os.path.join(cubes_dir(), "viz_index.json")
    if os.path.isfile(path):
        with contextlib.suppress(OSError, ValueError), open(path) as handle:
            return json.load(handle)
    return {"subset": "", "indices": []}


def _load_lens_isolation_combiner(member_labels: list[str]):
    if not member_labels:
        return None
    from euclid_polish.eval.combiner import load_combiner
    from euclid_polish.web.helpers.lens_isolation_viz import output_dir

    with contextlib.suppress(Exception):
        return load_combiner(output_dir(), member_labels=list(member_labels))
    return None


def _lens_isolation_meta(_params: dict[str, str]) -> dict[str, Any]:
    from euclid_polish.web.helpers.lens_isolation_viz import records_dir

    manifest = _lens_isolation_manifest()
    indices = manifest.get("indices", []) or []
    subset = str(manifest.get("subset", "") or "")
    has_target = bool(subset) and os.path.isfile(
        tfrecord_path(records_dir(), f"lens_{subset}")
    )
    tiers = [dict(t) for t in _ENSEMBLE_TIERS if t["key"] != "hr" or has_target]
    for tier in tiers:
        if tier["key"] == "hr":
            tier["label"] = "lens target"
    member_labels = list(manifest.get("member_labels", []) or [])
    if manifest.get("has_combiner") or _load_lens_isolation_combiner(member_labels) is not None:
        tiers.insert(2, {"key": "comb", "label": "combiner"})
    tiers += [
        {"key": f"member{i}", "label": f"SR {label}", "hidden": True}
        for i, label in enumerate(member_labels)
    ]
    pca_n = int(manifest.get("pca_n", 0) or 0)
    amplitudes = manifest.get("pca_amps", {}) or {}
    if pca_n > 0:
        tiers.append({"key": "morph", "label": "disagreement movie"})
    return {
        "count": len(indices),
        "tiers": tiers,
        "default_tier": "sr",
        "band_names": list(BAND_NAMES),
        "subset": subset,
        "pca_n": pca_n,
        "pca_amps": [list(amplitudes.get(str(int(index)), [])) for index in indices],
        "member_labels": member_labels,
        "pca_max": _MORPH_PCA_COMPONENTS,
        "target_label": "lens target",
    }


def _lens_isolation_record_cube(subset: str, n_read: int, kind: str, rec_index: int):
    from euclid_polish.web.helpers.lens_isolation_viz import records_dir

    path = tfrecord_path(records_dir(), f"{kind}_{subset}")
    if not os.path.isfile(path):
        raise ViewerError(404, f"{kind} records not available")
    record = {item.index: item for item in read_images(path, num_images=max(n_read, 1))}.get(
        rec_index
    )
    if record is None:
        raise ViewerError(404, f"record {rec_index} not found")
    return _as_hwc(record.data), float(getattr(record, "pixel_scale_arcsec", 0.0) or 0.0)


def _lens_isolation_subset_pca(rec_index: int, subset: list[int]):
    from euclid_polish.ensemble import pca_field
    from euclid_polish.web.helpers.lens_isolation_viz import cubes_dir

    key = ("lens-isolation", int(rec_index), tuple(subset))
    hit = _SUBSET_PCA_CACHE.get(key)
    if hit is not None:
        _SUBSET_PCA_CACHE.move_to_end(key)
        return hit
    stack = []
    for member_index in subset:
        path = os.path.join(cubes_dir(), f"member{member_index}_{int(rec_index):05d}.npy")
        if not os.path.isfile(path):
            raise ViewerError(404, f"member{member_index} cube missing")
        stack.append(np.load(path).astype(np.float32))
    result = pca_field(np.stack(stack), n_components=_MORPH_PCA_COMPONENTS)
    _SUBSET_PCA_CACHE[key] = result
    if len(_SUBSET_PCA_CACHE) > _SUBSET_PCA_MAX:
        _SUBSET_PCA_CACHE.popitem(last=False)
    return result


def _lens_isolation_combiner_cube(rec_index: int, member_labels: list[str]) -> np.ndarray:
    from euclid_polish.web.helpers.lens_isolation_viz import cubes_dir

    key = ("lens-isolation", int(rec_index), tuple(member_labels))
    hit = _COMB_CUBE_CACHE.get(key)
    if hit is not None:
        _COMB_CUBE_CACHE.move_to_end(key)
        return hit
    combiner = _load_lens_isolation_combiner(member_labels)
    if combiner is None:
        raise ViewerError(404, "no lens-isolation combiner")
    stack = []
    for member_index in range(len(member_labels)):
        path = os.path.join(cubes_dir(), f"member{member_index}_{rec_index:05d}.npy")
        if not os.path.isfile(path):
            raise ViewerError(404, f"member{member_index} cube missing")
        stack.append(np.load(path).astype(np.float32))
    result = np.asarray(combiner.apply_field(np.stack(stack)), np.float32)
    _COMB_CUBE_CACHE[key] = result
    if len(_COMB_CUBE_CACHE) > _COMB_CUBE_MAX:
        _COMB_CUBE_CACHE.popitem(last=False)
    return result


def _lens_isolation_cube(index: int, tier: str, params: dict[str, str]):
    from euclid_polish.web.helpers.lens_isolation_viz import cubes_dir

    manifest = _lens_isolation_manifest()
    indices = manifest.get("indices", []) or []
    subset_name = str(manifest.get("subset", "") or "")
    if index < 0 or index >= len(indices):
        raise ViewerError(404, "index out of range")
    rec_index = int(indices[index])
    member_labels = list(manifest.get("member_labels", []) or [])
    is_pca = tier.startswith("pca") and tier[3:].isdigit()
    member_subset = _parse_member_subset(params.get("members"), len(member_labels))
    if member_subset is not None and (tier == "sr" or is_pca):
        mean, components, amplitudes, variance = _lens_isolation_subset_pca(
            rec_index, member_subset
        )
        tag = f"{len(member_subset)} of {len(member_labels)} members"
        if tier == "sr":
            return _as_hwc(mean), {
                "label": f"SR (subset mean · {tag}) · {subset_name} · idx {rec_index}",
                "asinh": float(Config.STRETCH_SCALE_E),
                "pixscale": 0.0,
            }
        component = int(tier[3:])
        if component >= len(components):
            raise ViewerError(404, "pca component out of range")
        return _as_hwc(components[component]), {
            "label": f"PC{component} · {tag}",
            "asinh": float(Config.STRETCH_SCALE_E),
            "pixscale": 0.0,
            "amp": float(amplitudes[component]),
            "var": float(variance[component]) if component < len(variance) else 0.0,
        }
    if tier == "comb":
        baked = os.path.join(cubes_dir(), f"comb_{rec_index:05d}.npy")
        cube = np.load(baked) if os.path.isfile(baked) else _lens_isolation_combiner_cube(
            rec_index, member_labels
        )
        return _as_hwc(cube), {
            "label": f"SR (combiner) · {subset_name} · idx {rec_index}",
            "asinh": float(Config.STRETCH_SCALE_E),
            "pixscale": 0.0,
        }
    n_read = max((int(value) for value in indices), default=0) + 1
    is_npy = (
        tier in {"sr", "std"}
        or is_pca
        or (tier.startswith("member") and tier[6:].isdigit())
    )
    if is_npy:
        path = os.path.join(cubes_dir(), f"{tier}_{rec_index:05d}.npy")
        if not os.path.isfile(path):
            raise ViewerError(404, f"{tier} cube missing")
        cube, pixel_scale = _as_hwc(np.load(path)), 0.0
    elif tier == "lr":
        cube, pixel_scale = _lens_isolation_record_cube(
            subset_name, n_read, "dirty", rec_index
        )
    elif tier == "hr":
        cube, pixel_scale = _lens_isolation_record_cube(
            subset_name, n_read, "lens", rec_index
        )
    else:
        raise ViewerError(400, "bad tier")
    labels = {
        "lr": "LR",
        "sr": "SR (ensemble mean)",
        "std": "stdSR (member std)",
        "hr": "lens target",
    }
    if tier.startswith("member") and tier[6:].isdigit():
        member_index = int(tier[6:])
        label = f"SR {member_labels[member_index]}" if member_index < len(member_labels) else tier
    else:
        label = labels.get(tier, tier)
    info = {
        "label": f"{label} · {subset_name} · idx {rec_index}",
        "asinh": float(Config.STRETCH_SCALE_E),
        "pixscale": pixel_scale,
    }
    if is_pca:
        component = int(tier[3:])
        amplitudes = (manifest.get("pca_amps", {}) or {}).get(str(rec_index), [])
        variance = (manifest.get("pca_var", {}) or {}).get(str(rec_index), [])
        if component < len(amplitudes):
            info["amp"] = float(amplitudes[component])
        if component < len(variance):
            info["var"] = float(variance[component])
    return cube, info


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
    "lens-isolation": (_lens_isolation_meta, _lens_isolation_cube),
}


def get_meta(collection: str, params: dict[str, str]) -> dict[str, Any]:
    if collection not in _REGISTRY:
        raise ViewerError(404, "unknown collection")
    meta = _REGISTRY[collection][0](params)
    meta["collection"] = collection
    meta["color"] = color_constants()
    return meta


def get_cube(collection: str, index: int, tier: str,
             params: dict[str, str]) -> tuple[np.ndarray, dict[str, Any]]:
    if collection not in _REGISTRY:
        raise ViewerError(404, "unknown collection")
    cube, info = _REGISTRY[collection][1](index, tier, params)
    return np.ascontiguousarray(cube, dtype=np.float32), info
