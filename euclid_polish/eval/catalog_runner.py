"""Run the SR model over a catalog of real targets — locally, in-process.

This is the catalog-evaluation loop shared by the WebUI's local background job
(``/api/evaluation/run-eval``) and the ``scripts/eval_catalog.py`` CLI. It
fetches a 4-band Euclid cutout at every catalog (RA, Dec), runs the model,
writes per-object FITS (``SR.fits`` + ``original_stack.fits``) and a
``manifest.csv``. PNG rendering is left to the gallery (local, on demand).

Progress and logs are reported through plain callbacks so the same loop drives
the WebUI job's progress bar / log panel and the CLI's stdout:

    run_catalog_eval(..., on_progress=cap.tick, log=cap.write)
"""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import traceback
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

import numpy as np
from astropy.io import fits

from euclid_polish.config import Config
from euclid_polish.ensemble import default_ensemble_dir
from euclid_polish.ensemble_registry import regime_labels
from euclid_polish.eval import lens_catalog
from euclid_polish.eval.combiner import COMBINER_MODELS, combiner_artifact_fingerprint
from euclid_polish.eval.ensemble_infer import (
    PRODUCTION_COMBINER_KIND,
    load_eval_ensemble,
    load_production_combiner,
    starfull_regime_dir,
)
from euclid_polish.eval.eval_catalog import read_eval_catalog
from euclid_polish.eval.progress import tqdm_progress
from euclid_polish.web.helpers.jobs_impl import reconstruct_cutout_at

# Per-object metric keys (from reconstruct_cutout_at) + the manifest columns.
_METRIC_KEYS = ("lr_total_e", "sr_total_e", "flux_ratio_sr_over_lr")
MANIFEST_COLS = (
    ["id", "ra", "dec", "grade", "ok", "error", "out_subdir"]
    + list(_METRIC_KEYS)
    + ["psnr_lr_hr", "psnr_sr_hr"]
)
_MANIFEST_COLS = MANIFEST_COLS
_SAFE_ID = re.compile(r"[^A-Za-z0-9._-]+")

#: Canonical evaluation stamp sizes (px). The LR cutout lives on the VIS grid;
#: SR and HR live on the 2× grid. Every eval object is center-cropped to these
#: (or dropped when smaller — see :func:`enforce_object_sizes`) so the gallery
#: and quantitative summaries see one coherent 53/106 geometry.
EVAL_LR_SIZE = 53
EVAL_HR_SIZE = 2 * EVAL_LR_SIZE   # 106


def _safe_id(obj_id: str) -> str:
    s = _SAFE_ID.sub("_", obj_id).strip("_")
    return s or "obj"


def object_output_dir(out_dir: str, obj_id: str) -> str:
    return os.path.join(out_dir, _safe_id(obj_id))


def _base_manifest_row(obj, grade: str | None = None) -> dict[str, Any]:
    return {
        "id": obj["id"], "ra": obj["ra"], "dec": obj["dec"],
        "grade": grade if grade is not None else (obj.get("grade") or ""),
        "ok": False, "error": "", "out_subdir": _safe_id(obj["id"]),
        **dict.fromkeys(_METRIC_KEYS, ""),
    }


# ---------------------------------------------------------------------------
# model identity — which STARFULL members + production combiner made an SR
# ---------------------------------------------------------------------------
#
# Every object records the model that produced its SR in ``members.json``:
# ``{member_labels, combiner_kind, combiner_fingerprint}`` (``combiner_kind``
# is ``None`` for the plain member mean). The reuse key and the staleness the
# Sky › Catalog-eval tab shows both compare it with the model an evaluation
# would load NOW (:func:`current_eval_identity`).

MEMBERS_FILE = "members.json"
_IDENTITY_KEYS = ("member_labels", "combiner_kind", "combiner_fingerprint")


def _production_fingerprint(kind: str | None) -> str | None:
    if not kind:
        return None
    return combiner_artifact_fingerprint(starfull_regime_dir(),
                                         COMBINER_MODELS[str(kind)].artifact_dir)


def identity_for(member_labels: Sequence[str], combiner_kind: str | None) -> dict[str, Any]:
    """The identity of an SR made by ``member_labels`` through ``combiner_kind``
    (``None`` = the plain member mean), fingerprinting the fitted artifact."""
    return {"member_labels": [str(x) for x in member_labels],
            "combiner_kind": combiner_kind or None,
            "combiner_fingerprint": _production_fingerprint(combiner_kind)}


def eval_model_identity(model: Any) -> dict[str, Any]:
    """The identity of a loaded eval model (``load_eval_ensemble``)."""
    return identity_for(getattr(model, "member_labels", []) or [],
                        getattr(model, "combiner_kind", None))


def current_eval_identity(ensemble_dir: str | None = None, *,
                          labels: Sequence[str] | None = None) -> dict[str, Any]:
    """The identity :func:`load_eval_ensemble` would load now — the ACTIVE
    STARFULL members and the production combiner when one is fitted for
    exactly them (else the member mean) — without loading any network."""
    members = (list(labels) if labels is not None
               else list(regime_labels(ensemble_dir or default_ensemble_dir(), False)))
    current = (load_production_combiner(members) is not None) if members else False
    return identity_for(members, PRODUCTION_COMBINER_KIND if current else None)


def read_model_identity(obj_dir: str) -> dict[str, Any] | None:
    """The object's recorded ``members.json`` (``None`` when absent/unreadable)."""
    try:
        with open(os.path.join(obj_dir, MEMBERS_FILE)) as f:
            recorded = json.load(f)
    except (OSError, ValueError):
        return None
    return recorded if isinstance(recorded, dict) else None


def record_model_identity(obj_dir: str, identity: Mapping[str, Any]) -> None:
    """Merge the model identity into the object's ``members.json``."""
    recorded = read_model_identity(obj_dir) or {}
    recorded.update({key: identity.get(key) for key in _IDENTITY_KEYS})
    recorded["member_labels"] = list(recorded.get("member_labels") or [])
    os.makedirs(obj_dir, exist_ok=True)
    with open(os.path.join(obj_dir, MEMBERS_FILE), "w") as f:
        json.dump(recorded, f)


def _combiner_matches(recorded: Mapping[str, Any], identity: Mapping[str, Any]) -> bool:
    return ("combiner_kind" in recorded
            and recorded.get("combiner_kind") == identity.get("combiner_kind")
            and recorded.get("combiner_fingerprint") == identity.get("combiner_fingerprint"))


def object_model_state(obj_dir: str, identity: Mapping[str, Any]) -> dict[str, Any]:
    """``{state: current|stale|unknown, reason, n_members, combiner_kind,
    combiner_fingerprint}`` of one object's SR against ``identity``."""
    recorded = read_model_identity(obj_dir)
    if recorded is None:
        return {"state": "unknown", "reason": "no model recorded (members.json missing)",
                "n_members": None, "combiner_kind": None, "combiner_fingerprint": None}
    labels = [str(x) for x in recorded.get("member_labels") or []]
    out = {"n_members": len(labels), "combiner_kind": recorded.get("combiner_kind"),
           "combiner_fingerprint": recorded.get("combiner_fingerprint")}
    want = [str(x) for x in identity.get("member_labels") or []]
    if labels != want:
        return {**out, "state": "stale", "reason": (
            f"membership changed: made by {len(labels)} member(s), "
            f"{len(want)} active STARFULL now")}
    if "combiner_kind" not in recorded:
        return {**out, "state": "stale",
                "reason": "made before the combiner was recorded (plain mean or older combiner)"}
    if not _combiner_matches(recorded, identity):
        made = recorded.get("combiner_kind") or "member mean"
        now = identity.get("combiner_kind") or "member mean"
        return {**out, "state": "stale", "reason": (
            f"combiner changed: made by {made}, now {now}" if made != now
            else f"combiner changed: {made} was refitted")}
    return {**out, "state": "current", "reason": None}


def can_reuse_eval_object(obj_dir: str, *,
                          require_disagreement: bool = False,
                          member_labels: list[str] | None = None,
                          identity: Mapping[str, Any] | None = None) -> bool:
    """True when an object already has the real-lens evaluation FITS outputs.

    With ``require_disagreement`` (set when the ensemble has >1 models), also
    require the disagreement cubes (``std.fits`` + ``pca0.fits``) so an object
    that only carries a plain ``SR.fits`` is re-run — letting the ensemble add
    the stdSR + disagreement-movie cubes — instead of being skipped as done.

    ``member_labels`` is the membership fingerprint: when given (alongside
    ``require_disagreement``), the object's ``members.json`` must exist and
    record the SAME labels — outputs produced by a different membership (e.g.
    before a member was archived) are stale and must be regenerated.
    ``identity`` (:func:`current_eval_identity`) adds the production combiner:
    the recorded ``combiner_kind`` + ``combiner_fingerprint`` must match too,
    so a refitted (or newly fitted) gate regenerates the SRs it would change.
    """
    needed = ["original_stack.fits", "SR.fits"]
    if require_disagreement:
        needed += ["std.fits", "pca0.fits"]
    if not all(
        os.path.isfile(os.path.join(obj_dir, name))
        and os.path.getsize(os.path.join(obj_dir, name)) > 0
        for name in needed
    ):
        return False
    if require_disagreement and member_labels is not None:
        recorded = read_model_identity(obj_dir)
        if recorded is None:
            return False
        if list(recorded.get("member_labels") or []) != list(member_labels):
            return False
        if identity is not None and not _combiner_matches(recorded, identity):
            return False
    return True


def _vis_plane(arr):
    data = np.asarray(arr)
    if data.ndim == 3:
        if data.shape[0] == Config.NUM_LR_CHANNELS:
            return data[0]
        if data.shape[-1] == Config.NUM_LR_CHANNELS:
            return data[..., 0]
        return data[0]
    return data


def reuse_catalog_object(obj, out_dir: str, *, grade: str | None = None,
                         from_cache: bool = True,
                         log: Callable[[str], None] | None = None
                         ) -> dict[str, Any]:
    """Build a manifest row from on-disk LR/SR FITS (the flux metrics).

    ``from_cache`` only affects the log wording: ``True`` (the default) means the
    FITS were already present from a prior run (a genuine cache hit — no download
    happened), ``False`` means they were just downloaded this run and we are only
    reading them back to compute the post-crop metrics. The wording matters
    because this is called in both branches and a "reusing" line after a fresh
    download otherwise reads as wasteful re-downloading.
    """
    emit = log or (lambda m: None)
    rec = _base_manifest_row(obj, grade=grade)
    obj_dir = object_output_dir(out_dir, obj["id"])
    try:
        with fits.open(os.path.join(obj_dir, "original_stack.fits")) as hdul:
            primary = cast(fits.PrimaryHDU, hdul[0])
            lr_vis = _vis_plane(primary.data)
        with fits.open(os.path.join(obj_dir, "SR.fits")) as hdul:
            primary = cast(fits.PrimaryHDU, hdul[0])
            sr_vis = _vis_plane(primary.data)
        lr_sum = float(np.sum(lr_vis))
        sr_sum = float(np.sum(sr_vis))
        rec.update({
            "ok": True,
            "lr_total_e": lr_sum,
            "sr_total_e": sr_sum,
            "flux_ratio_sr_over_lr": (sr_sum / lr_sum) if lr_sum else "",
        })
        emit(f"  ↻ {obj['id']}: reusing existing LR/SR FITS (no download)"
             if from_cache else
             f"  ✓ {obj['id']}: metrics from freshly-downloaded FITS")
    except Exception as e:  # noqa: BLE001 — keep batch semantics
        rec["error"] = f"{type(e).__name__}: {e}"
        emit(f"  ! {obj['id']} cache unusable: {rec['error']}")
    return rec


def default_catalog_path() -> str:
    return os.path.join(Config.EVAL_CATALOG_DIR, "lens_catalog", "lenses.csv")


def crop_offsets(shape: Sequence[int], size: int) -> tuple[int, int]:
    """``(sy, sx)``: the first kept row / column of :func:`center_crop`."""
    h, w = int(shape[-2]), int(shape[-1])
    if h <= size and w <= size:
        return 0, 0
    return max(0, (h - size) // 2), max(0, (w - size) // 2)


def center_crop(arr, size: int, offsets: tuple[int, int] | None = None):
    """Center-crop the trailing two (spatial) axes of ``arr`` to ``size``×``size``.

    Works for 2-D planes and channel-first cubes ``(C, H, W)`` alike. Centering
    on the array's central pixel means an even source cropped to an even ``size``
    stays even, while ``size`` is otherwise honored exactly; the crop is a no-op
    when both spatial dims are already ≤ ``size``. ``offsets`` overrides the
    centre (``(sy, sx)`` of the first kept row / column).
    """
    a = np.asarray(arr)
    h, w = a.shape[-2], a.shape[-1]
    if h <= size and w <= size:
        return a
    sy, sx = offsets if offsets is not None else crop_offsets(a.shape, size)
    return a[..., sy:sy + size, sx:sx + size]


def _shift_crpix(header: fits.Header, sy: int, sx: int) -> fits.Header:
    """The WCS of a crop starting at row ``sy`` / column ``sx`` (FITS axis 1 =
    column): the reference pixel moves by the offset, the sky stays put."""
    out = header.copy()
    if sx and "CRPIX1" in out:
        out["CRPIX1"] = float(out["CRPIX1"]) - sx
    if sy and "CRPIX2" in out:
        out["CRPIX2"] = float(out["CRPIX2"]) - sy
    return out


#: The object FITS held at the canonical geometry: (file, side, required).
#: Everything but the LR stack lives on the 2× (SR) grid.
OBJECT_PLANES = (("original_stack.fits", EVAL_LR_SIZE, True),
                 ("SR.fits", EVAL_HR_SIZE, True),
                 ("HR.fits", EVAL_HR_SIZE, False),
                 ("BHR.fits", EVAL_HR_SIZE, False),
                 ("mean.fits", EVAL_HR_SIZE, False),
                 ("std.fits", EVAL_HR_SIZE, False),
                 ("pca0.fits", EVAL_HR_SIZE, False),
                 ("pca1.fits", EVAL_HR_SIZE, False),
                 ("pca2.fits", EVAL_HR_SIZE, False))


def enforce_object_sizes(obj_dir: str, *,
                         log: Callable[[str], None] | None = None) -> bool:
    """Crop an object's FITS to the canonical eval sizes; signal drop if smaller.

    ``original_stack.fits`` (LR / VIS grid) is held at ``EVAL_LR_SIZE``² and
    ``SR.fits`` / ``HR.fits`` / ``mean.fits`` / ``std.fits`` / ``pcaN.fits``
    (2× grid) at ``EVAL_HR_SIZE``². Larger stamps are center-cropped down; a
    stamp smaller than its target in either spatial axis means the object can't
    be represented at the canonical geometry, so this returns ``False`` (the
    caller drops it) **without modifying any file**. ``HR.fits`` is optional
    (real A/B/C cutouts have no HR); the LR and SR planes are required. Returns
    ``True`` once every present plane met (or exceeded, and was cropped to) its
    target.

    The crops keep the sky under every pixel: each file's ``CRPIX1/2`` moves by
    its crop offset, and a 2×-grid plane that is exactly twice the LR stack is
    cut at twice the LR offset (not its own centre), so the SR stamp stays the
    LR stamp magnified ×2 — the LR WCS ×2 the viewer derives for it holds.
    """
    emit = log or (lambda m: None)
    tag = os.path.basename(obj_dir.rstrip(os.sep))

    # 1) Validate every plane's size before touching anything on disk.
    loaded = []
    for name, size, required in OBJECT_PLANES:
        path = os.path.join(obj_dir, name)
        if not os.path.isfile(path):
            if required:
                emit(f"  ✗ {tag}: missing {name} — dropping")
                return False
            continue
        with fits.open(path) as hdul:
            primary = cast(fits.PrimaryHDU, hdul[0])
            data = np.asarray(primary.data)
            header = primary.header.copy()
        h, w = data.shape[-2], data.shape[-1]
        if h < size or w < size:
            emit(f"  ✗ {tag}: {name} {w}×{h} < {size}×{size} — dropping")
            return False
        loaded.append((path, name, size, data, header))

    # 2) Crop to the exact target (rewrite only when the shape actually changes).
    lr_shape = loaded[0][3].shape[-2:]
    lr_off = crop_offsets(lr_shape, EVAL_LR_SIZE)
    for path, name, size, data, header in loaded:
        if name == "original_stack.fits":
            offsets = lr_off
        elif (tuple(data.shape[-2:]) == (2 * lr_shape[0], 2 * lr_shape[1])
              and size == 2 * EVAL_LR_SIZE):
            offsets = (2 * lr_off[0], 2 * lr_off[1])      # nested in the LR crop
        else:
            offsets = crop_offsets(data.shape, size)
        cropped = center_crop(data, size, offsets)
        if cropped.shape != data.shape:
            fits.PrimaryHDU(np.ascontiguousarray(cropped),
                            header=_shift_crpix(header, *offsets)).writeto(
                path, overwrite=True, output_verify="silentfix")
            emit(f"  ✂ {tag}: {name} → {size}×{size}")
    return True


def seed_object_from_cache(source_dir: str, out_dir: str, obj_id: str) -> bool:
    """Copy an object's cached LR/SR FITS from ``source_dir`` into ``out_dir``.

    Lets a fresh run reuse already-downloaded cutouts (e.g. crop them to a new
    size) without re-fetching from the archive. No-op when ``source_dir`` lacks
    the object or the destination already has it; returns ``True`` if it copied.
    """
    src_dir = object_output_dir(source_dir, obj_id)
    dst_dir = object_output_dir(out_dir, obj_id)
    if os.path.abspath(src_dir) == os.path.abspath(dst_dir):
        return False
    if can_reuse_eval_object(dst_dir) or not can_reuse_eval_object(src_dir):
        return False
    os.makedirs(dst_dir, exist_ok=True)
    for name in ("original_stack.fits", "SR.fits"):
        src = os.path.join(src_dir, name)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(dst_dir, name))
    return True


def write_manifest_upsert(
    manifest_path: str,
    rows: list[dict[str, Any]],
    fieldnames=MANIFEST_COLS,
    drop_ids: set[str] | None = None,
) -> None:
    """Write ``rows`` into a shared manifest, preserving unrelated objects.

    ``drop_ids`` removes those existing rows instead of preserving them —
    used to retire objects superseded by a regeneration (a plain upsert keys
    by id, so rows whose ids never recur would otherwise live forever).
    """
    existing: list[dict[str, Any]] = []
    if os.path.isfile(manifest_path):
        with open(manifest_path, newline="") as f:
            existing = list(csv.DictReader(f))
    if drop_ids:
        existing = [r for r in existing if str(r.get("id", "")) not in drop_ids]

    order: list[str] = []
    by_id: dict[str, dict[str, Any]] = {}
    for row in existing + rows:
        obj_id = str(row.get("id", ""))
        if not obj_id:
            continue
        if obj_id not in by_id:
            order.append(obj_id)
        merged = dict(by_id.get(obj_id, {}))
        merged.update(row)
        by_id[obj_id] = merged

    cols = list(fieldnames)
    for row in by_id.values():
        for key in row:
            if key not in cols:
                cols.append(key)

    os.makedirs(os.path.dirname(manifest_path) or ".", exist_ok=True)
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for obj_id in order:
            row = by_id[obj_id]
            writer.writerow({key: row.get(key, "") for key in cols})


def eval_catalog_object(model, obj, out_dir: str, *, cutout_size: int,
                        asinh_scale: float | None, checkpoint: str,
                        grade: str | None = None, render: bool = False,
                        log: Callable[[str], None] | None = None
                        ) -> dict[str, Any]:
    """Reconstruct one catalog object → write its FITS, return a manifest row.

    ``obj`` is an ``{id, ra, dec, grade}`` dict (from read_eval_catalog).
    ``grade`` overrides ``obj['grade']`` when given (used to tag the group).
    A failure is captured in the row's ``error`` (never raised) so one bad
    object can't kill a run.
    """
    emit = log or (lambda m: None)
    obj_id = obj["id"]
    obj_dir = object_output_dir(out_dir, obj_id)
    rec = _base_manifest_row(obj, grade=grade)
    identity = eval_model_identity(model)
    ensemble = model.n_members > 1
    if can_reuse_eval_object(obj_dir, require_disagreement=ensemble,
                             member_labels=(list(model.member_labels) if ensemble else None),
                             identity=identity if ensemble else None):
        enforce_object_sizes(obj_dir, log=emit)
        return reuse_catalog_object(obj, out_dir, grade=grade, log=emit)
    try:
        res = reconstruct_cutout_at(
            model, obj["ra"], obj["dec"], cutout_size, obj_dir,
            asinh_scale=asinh_scale, checkpoint_dir=checkpoint, render=render)
        # The SR now comes from THIS model: record it (members + combiner)
        # as the object's reuse key and staleness reference.
        record_model_identity(obj_dir, identity)
        for k in _METRIC_KEYS:
            rec[k] = res["metrics"].get(k)
        rec["ok"] = True
    except Exception as e:  # noqa: BLE001 — one bad object must not kill the run
        rec["error"] = f"{type(e).__name__}: {e}"
        emit(f"  ! {obj_id} skipped: {rec['error']}")
        traceback.print_exc()
    return rec


def run_catalog_eval(
    *,
    out_dir: str,
    catalog_path: str | None = None,
    ensemble_dir: str | None = None,
    num_res_blocks: int | None = None,
    cutout_size: int = 256,
    grade: str | None = None,
    max_n: int | None = None,
    asinh_scale: float | None = None,
    render: bool = False,
    on_progress: Callable[[int, int, str], None] | None = None,
    log: Callable[[str], None] | None = None,
    model: Any = None,
) -> dict[str, Any]:
    """Evaluate the model over a catalog into ``out_dir``; return a summary.

    ``catalog_path=None`` uses the default lens catalog and auto-fetches it from
    Zenodo if it's missing; an explicit path that's missing raises. ``on_progress``
    is called ``(done, total, label)`` per object, ``log`` with human lines.
    """
    def _emit(msg: str) -> None:
        (log or print)(msg)

    if on_progress is None:                     # local/CLI run → visible bar
        on_progress = tqdm_progress("catalog")

    def _tick(done: int, total: int, label: str = "") -> None:
        if on_progress is not None:
            on_progress(done, total, label)

    ensemble_dir = ensemble_dir or default_ensemble_dir()
    catalog = catalog_path or default_catalog_path()

    if not os.path.isfile(catalog):
        if catalog_path:
            raise FileNotFoundError(f"catalog not found: {catalog}")
        _emit(f"catalog {catalog} not found — fetching from Zenodo…")
        lens_catalog.fetch(catalog)

    rows = read_eval_catalog(catalog, grade=grade, max_n=(max_n or None))
    n = len(rows)
    _emit(f"catalog {catalog}: {n} object(s)"
          + (f" (grade {grade})" if grade else ""))
    if n == 0:
        _emit("nothing to evaluate")
        return {"out_dir": out_dir, "n": 0, "n_ok": 0, "n_skip": 0,
                "manifest": None}

    os.makedirs(out_dir, exist_ok=True)
    # Reuse key: the STARFULL members + production combiner the model has (or
    # would have — cheap registry probe, no network loaded).
    identity = (eval_model_identity(model) if model is not None
                else current_eval_identity(ensemble_dir))
    ensemble = len(identity["member_labels"]) > 1

    def _reusable(obj_id: str) -> bool:
        return can_reuse_eval_object(
            object_output_dir(out_dir, obj_id), require_disagreement=ensemble,
            member_labels=identity["member_labels"] if ensemble else None,
            identity=identity if ensemble else None)

    needs_model = any(not _reusable(row["id"]) for row in rows)
    if needs_model and model is None:
        model = load_eval_ensemble(ensemble_dir, num_res_blocks, log=_emit)
    elif not needs_model:
        _emit("all catalog outputs already present — reusing cached FITS")

    manifest_path = os.path.join(out_dir, "manifest.csv")
    n_ok = n_skip = 0
    out_rows: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        _tick(i, n, f"{row['id']} ({i + 1}/{n})")
        _emit(f"[{i + 1}/{n}] {row['id']}  ra={row['ra']:.5f} "
              f"dec={row['dec']:.5f}")
        if model is None:                       # every object is current
            enforce_object_sizes(object_output_dir(out_dir, row["id"]), log=_emit)
            rec = reuse_catalog_object(row, out_dir, log=_emit)
        else:
            rec = eval_catalog_object(
                model, row, out_dir, cutout_size=cutout_size,
                asinh_scale=asinh_scale, checkpoint=ensemble_dir,
                render=render, log=_emit)
        n_ok, n_skip = (n_ok + 1, n_skip) if rec["ok"] else (n_ok, n_skip + 1)
        out_rows.append(rec)
        write_manifest_upsert(manifest_path, out_rows, _MANIFEST_COLS)

    _tick(n, n, "done")
    _emit(f"\n✓ done: {n_ok} ok, {n_skip} skipped → {manifest_path}")
    return {"out_dir": out_dir, "n": n, "n_ok": n_ok, "n_skip": n_skip,
            "manifest": manifest_path}
