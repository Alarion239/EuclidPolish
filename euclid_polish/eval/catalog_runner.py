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
from typing import Any, Callable, Dict, List, Optional

from euclid_polish.config import Config
from euclid_polish.euclid.eval_catalog import read_eval_catalog

# Per-object metric keys (from reconstruct_cutout_at) + the manifest columns.
_METRIC_KEYS = ("lr_total_e", "sr_total_e", "flux_ratio_sr_over_lr")
_MANIFEST_COLS = (
    ["id", "ra", "dec", "grade", "ok", "error", "out_subdir"]
    + list(_METRIC_KEYS)
)
_SAFE_ID = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_id(obj_id: str) -> str:
    s = _SAFE_ID.sub("_", obj_id).strip("_")
    return s or "obj"


def default_catalog_path() -> str:
    return os.path.join(Config.EVAL_CATALOG_DIR, "lens_catalog", "lenses.csv")


def run_catalog_eval(
    *,
    out_dir: str,
    catalog_path: Optional[str] = None,
    checkpoint: Optional[str] = None,
    num_res_blocks: Optional[int] = None,
    cutout_size: int = 256,
    grade: Optional[str] = None,
    max_n: Optional[int] = None,
    asinh_scale: Optional[float] = None,
    render: bool = False,
    on_progress: Optional[Callable[[int, int, str], None]] = None,
    log: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Evaluate the model over a catalog into ``out_dir``; return a summary.

    ``catalog_path=None`` uses the default lens catalog and auto-fetches it from
    Zenodo if it's missing; an explicit path that's missing raises. ``on_progress``
    is called ``(done, total, label)`` per object, ``log`` with human lines.
    """
    def _emit(msg: str) -> None:
        (log or print)(msg)

    def _tick(done: int, total: int, label: str = "") -> None:
        if on_progress is not None:
            on_progress(done, total, label)

    # Heavy imports (TF) deferred so importing this module stays cheap.
    import tensorflow as tf
    from euclid_polish.training.inference import load_model_from_checkpoint
    from euclid_polish.web.helpers.jobs_impl import reconstruct_cutout_at

    checkpoint = checkpoint or Config.DEFAULT_CHECKPOINT_DIR
    num_res_blocks = num_res_blocks or Config.DEFAULT_NUM_RES_BLOCKS
    catalog = catalog_path or default_catalog_path()

    if not os.path.isfile(catalog):
        if catalog_path:
            raise FileNotFoundError(f"catalog not found: {catalog}")
        from euclid_polish.euclid import lens_catalog
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

    if not tf.train.latest_checkpoint(checkpoint):
        raise FileNotFoundError(f"no checkpoint in {checkpoint}")
    _emit(f"loading model from {checkpoint}")
    model = load_model_from_checkpoint(
        checkpoint, Config.DEFAULT_REBIN_FACTOR, num_res_blocks,
        nchan_out=Config.NUM_HR_CHANNELS,   # nchan_in inferred from ckpt
    )

    os.makedirs(out_dir, exist_ok=True)
    manifest_path = os.path.join(out_dir, "manifest.csv")
    n_ok = n_skip = 0
    with open(manifest_path, "w", newline="") as fmani:
        writer = csv.DictWriter(fmani, fieldnames=_MANIFEST_COLS)
        writer.writeheader()
        for i, row in enumerate(rows):
            obj_id = row["id"]
            sub = _safe_id(obj_id)
            _tick(i, n, f"{obj_id} ({i + 1}/{n})")
            _emit(f"[{i + 1}/{n}] {obj_id}  ra={row['ra']:.5f} "
                  f"dec={row['dec']:.5f}")
            rec: Dict[str, Any] = {
                "id": obj_id, "ra": row["ra"], "dec": row["dec"],
                "grade": row["grade"] or "", "ok": False, "error": "",
                "out_subdir": sub,
            }
            rec.update({k: "" for k in _METRIC_KEYS})
            try:
                res = reconstruct_cutout_at(
                    model, row["ra"], row["dec"], cutout_size,
                    os.path.join(out_dir, sub),
                    asinh_scale=asinh_scale, checkpoint_dir=checkpoint,
                    render=render,
                )
                for k in _METRIC_KEYS:
                    rec[k] = res["metrics"].get(k)
                rec["ok"] = True
                n_ok += 1
            except Exception as e:  # noqa: BLE001 — one bad object must not kill the run
                msg = f"{type(e).__name__}: {e}"
                rec["error"] = msg
                n_skip += 1
                _emit(f"  ! skipped: {msg}")
                traceback.print_exc()
            writer.writerow(rec)
            fmani.flush()

    _tick(n, n, "done")
    _emit(f"\n✓ done: {n_ok} ok, {n_skip} skipped → {manifest_path}")
    return {"out_dir": out_dir, "n": n, "n_ok": n_ok, "n_skip": n_skip,
            "manifest": manifest_path}
