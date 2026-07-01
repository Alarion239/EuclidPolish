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
import os
import re
import traceback
from collections.abc import Callable
from typing import Any

from euclid_polish.config import Config
from euclid_polish.eval.eval_catalog import read_eval_catalog

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
#: (or dropped when smaller — see :func:`enforce_object_sizes`) so the gallery,
#: the Zoobot representations and the lens-finder all see one coherent geometry —
#: the same 53/106 the lens-finder training stamps use.
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


def can_reuse_eval_object(obj_dir: str, *,
                          require_disagreement: bool = False) -> bool:
    """True when an object already has the real-lens evaluation FITS outputs.

    With ``require_disagreement`` (set when the eval model is an ensemble), also
    require the disagreement cubes (``std.fits`` + ``pca0.fits``) so an object
    that only carries a single-model ``SR.fits`` is re-run — letting the ensemble
    add the stdSR + disagreement-movie cubes — instead of being skipped as done.
    """
    needed = ["original_stack.fits", "SR.fits"]
    if require_disagreement:
        needed += ["std.fits", "pca0.fits"]
    return all(
        os.path.isfile(os.path.join(obj_dir, name))
        and os.path.getsize(os.path.join(obj_dir, name)) > 0
        for name in needed
    )


def _vis_plane(arr):
    import numpy as np

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
    import numpy as np
    from astropy.io import fits

    emit = log or (lambda m: None)
    rec = _base_manifest_row(obj, grade=grade)
    obj_dir = object_output_dir(out_dir, obj["id"])
    try:
        with fits.open(os.path.join(obj_dir, "original_stack.fits")) as hdul:
            lr_vis = _vis_plane(hdul[0].data)
        with fits.open(os.path.join(obj_dir, "SR.fits")) as hdul:
            sr_vis = _vis_plane(hdul[0].data)
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


def center_crop(arr, size: int):
    """Center-crop the trailing two (spatial) axes of ``arr`` to ``size``×``size``.

    Works for 2-D planes and channel-first cubes ``(C, H, W)`` alike. Centering
    on the array's central pixel means an even source cropped to an even ``size``
    stays even, while ``size`` is otherwise honored exactly; the crop is a no-op
    when both spatial dims are already ≤ ``size``.
    """
    import numpy as np

    a = np.asarray(arr)
    h, w = a.shape[-2], a.shape[-1]
    if h <= size and w <= size:
        return a
    sy = max(0, (h - size) // 2)
    sx = max(0, (w - size) // 2)
    return a[..., sy:sy + size, sx:sx + size]


def enforce_object_sizes(obj_dir: str, *,
                         log: Callable[[str], None] | None = None) -> bool:
    """Crop an object's FITS to the canonical eval sizes; signal drop if smaller.

    ``original_stack.fits`` (LR / VIS grid) is held at ``EVAL_LR_SIZE``² and
    ``SR.fits`` / ``HR.fits`` (2× grid) at ``EVAL_HR_SIZE``². Larger stamps are
    center-cropped down; a stamp smaller than its target in either spatial axis
    means the object can't be represented at the canonical geometry, so this
    returns ``False`` (the caller drops it) **without modifying any file**.
    ``HR.fits`` is optional (real A/B/C cutouts have no HR); the LR and SR planes
    are required. Returns ``True`` once every present plane met (or exceeded, and
    was cropped to) its target.
    """
    import numpy as np
    from astropy.io import fits

    emit = log or (lambda m: None)
    tag = os.path.basename(obj_dir.rstrip(os.sep))
    plan = (("original_stack.fits", EVAL_LR_SIZE, True),
            ("SR.fits", EVAL_HR_SIZE, True),
            ("HR.fits", EVAL_HR_SIZE, False),
            ("std.fits", EVAL_HR_SIZE, False),
            ("pca0.fits", EVAL_HR_SIZE, False),
            ("pca1.fits", EVAL_HR_SIZE, False),
            ("pca2.fits", EVAL_HR_SIZE, False))

    # 1) Validate every plane's size before touching anything on disk.
    loaded = []
    for name, size, required in plan:
        path = os.path.join(obj_dir, name)
        if not os.path.isfile(path):
            if required:
                emit(f"  ✗ {tag}: missing {name} — dropping")
                return False
            continue
        with fits.open(path) as hdul:
            data = np.asarray(hdul[0].data)
            header = hdul[0].header.copy()
        h, w = data.shape[-2], data.shape[-1]
        if h < size or w < size:
            emit(f"  ✗ {tag}: {name} {w}×{h} < {size}×{size} — dropping")
            return False
        loaded.append((path, name, size, data, header))

    # 2) Crop to the exact target (rewrite only when the shape actually changes).
    for path, name, size, data, header in loaded:
        cropped = center_crop(data, size)
        if cropped.shape != data.shape:
            fits.PrimaryHDU(np.ascontiguousarray(cropped), header=header).writeto(
                path, overwrite=True, output_verify="silentfix")
            emit(f"  ✂ {tag}: {name} → {size}×{size}")
    return True


def seed_object_from_cache(source_dir: str, out_dir: str, obj_id: str) -> bool:
    """Copy an object's cached LR/SR FITS from ``source_dir`` into ``out_dir``.

    Lets a fresh run reuse already-downloaded cutouts (e.g. crop them to a new
    size) without re-fetching from the archive. No-op when ``source_dir`` lacks
    the object or the destination already has it; returns ``True`` if it copied.
    """
    import shutil

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
) -> None:
    """Write ``rows`` into a shared manifest, preserving unrelated objects."""
    existing: list[dict[str, Any]] = []
    if os.path.isfile(manifest_path):
        with open(manifest_path, newline="") as f:
            existing = list(csv.DictReader(f))

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


def load_eval_model(checkpoint: str | None = None,
                    num_res_blocks: int | None = None):
    """Load the local SR model once (raises if no checkpoint). Shared by the
    single-catalog and grouped runners."""
    import tensorflow as tf

    from euclid_polish.training.inference import load_model_from_checkpoint

    checkpoint = checkpoint or Config.DEFAULT_CHECKPOINT_DIR
    num_res_blocks = num_res_blocks or Config.DEFAULT_NUM_RES_BLOCKS
    if not tf.train.latest_checkpoint(checkpoint):
        raise FileNotFoundError(f"no checkpoint in {checkpoint}")
    return load_model_from_checkpoint(
        checkpoint, Config.DEFAULT_REBIN_FACTOR, num_res_blocks,
        nchan_out=Config.NUM_HR_CHANNELS,   # nchan_in inferred from ckpt
    )


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
    rec = _base_manifest_row(obj, grade=grade)
    if can_reuse_eval_object(object_output_dir(out_dir, obj_id),
                             require_disagreement=hasattr(model, "member_arrays")):
        enforce_object_sizes(object_output_dir(out_dir, obj_id), log=emit)
        return reuse_catalog_object(obj, out_dir, grade=grade, log=emit)
    try:
        from euclid_polish.web.helpers.jobs_impl import reconstruct_cutout_at

        res = reconstruct_cutout_at(
            model, obj["ra"], obj["dec"], cutout_size,
            object_output_dir(out_dir, obj_id),
            asinh_scale=asinh_scale, checkpoint_dir=checkpoint, render=render)
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
    checkpoint: str | None = None,
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
        from euclid_polish.eval.progress import tqdm_progress
        on_progress = tqdm_progress("catalog")

    def _tick(done: int, total: int, label: str = "") -> None:
        if on_progress is not None:
            on_progress(done, total, label)

    checkpoint = checkpoint or Config.DEFAULT_CHECKPOINT_DIR
    catalog = catalog_path or default_catalog_path()

    if not os.path.isfile(catalog):
        if catalog_path:
            raise FileNotFoundError(f"catalog not found: {catalog}")
        from euclid_polish.eval import lens_catalog
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
    needs_model = any(
        not can_reuse_eval_object(object_output_dir(out_dir, row["id"]))
        for row in rows
    )
    if needs_model and model is None:
        _emit(f"loading model from {checkpoint}")
        model = load_eval_model(checkpoint, num_res_blocks)
    elif not needs_model:
        _emit("all catalog outputs already present — reusing cached FITS")

    manifest_path = os.path.join(out_dir, "manifest.csv")
    n_ok = n_skip = 0
    out_rows: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        _tick(i, n, f"{row['id']} ({i + 1}/{n})")
        _emit(f"[{i + 1}/{n}] {row['id']}  ra={row['ra']:.5f} "
              f"dec={row['dec']:.5f}")
        rec = eval_catalog_object(
            model, row, out_dir, cutout_size=cutout_size,
            asinh_scale=asinh_scale, checkpoint=checkpoint,
            render=render, log=_emit)
        n_ok, n_skip = (n_ok + 1, n_skip) if rec["ok"] else (n_ok, n_skip + 1)
        out_rows.append(rec)
        write_manifest_upsert(manifest_path, out_rows, _MANIFEST_COLS)

    _tick(n, n, "done")
    _emit(f"\n✓ done: {n_ok} ok, {n_skip} skipped → {manifest_path}")
    return {"out_dir": out_dir, "n": n, "n_ok": n_ok, "n_skip": n_skip,
            "manifest": manifest_path}
