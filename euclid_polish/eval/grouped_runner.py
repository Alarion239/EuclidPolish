"""Unified grouped evaluation: A / B / C lens grades + synthetic, one run.

Runs the SR model over N real lens cutouts per grade (A/B/C — LR+SR only) plus
N synthetic validation triptychs (LR+SR+HR), into a single run dir with one
``manifest.csv`` whose ``grade`` column is the group ∈ {A, B, C, synthetic}.
Shares the SR model load across all groups and runs locally, in-process.
"""

from __future__ import annotations

import csv
import os
from typing import Any, Callable, Dict, List, Optional

from euclid_polish.config import Config
from euclid_polish.euclid.eval_catalog import read_eval_catalog
from euclid_polish.eval import catalog_runner, synthetic_runner

#: Manifest columns for a grouped run (superset: PSNR is synthetic-only).
GROUPED_COLS = [
    "id", "ra", "dec", "grade", "ok", "error", "out_subdir",
    "lr_total_e", "sr_total_e", "flux_ratio_sr_over_lr",
    "psnr_lr_hr", "psnr_sr_hr",
]
LENS_GRADES = ("A", "B", "C")


def run_grouped_analysis(
    out_dir: str, n: int, *,
    cutout_size: int = 256,
    catalog_path: Optional[str] = None,
    checkpoint: Optional[str] = None,
    num_res_blocks: Optional[int] = None,
    asinh_scale: Optional[float] = None,
    seed: int = 0,
    grades=LENS_GRADES,
    include_synthetic: bool = True,
    on_progress: Optional[Callable[[int, int, str], None]] = None,
    log: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Prepare the four-group dataset into ``out_dir``; write one manifest."""
    def _emit(m): (log or print)(m)

    checkpoint = checkpoint or Config.DEFAULT_CHECKPOINT_DIR
    catalog = catalog_path or catalog_runner.default_catalog_path()
    if not os.path.isfile(catalog):
        if catalog_path:
            raise FileNotFoundError(f"catalog not found: {catalog}")
        from euclid_polish.euclid import lens_catalog
        _emit(f"catalog {catalog} not found — fetching from Zenodo…")
        lens_catalog.fetch(catalog)

    # Plan the work so the single progress bar spans every group.
    lens_plan = []
    for g in grades:
        rows = read_eval_catalog(catalog, grade=g, max_n=n)
        lens_plan.append((g, rows))
        _emit(f"grade {g}: {len(rows)} lens(es)")
    total = sum(len(r) for _, r in lens_plan) + (n if include_synthetic else 0)
    if total == 0:
        _emit("nothing to evaluate")
        return {"out_dir": out_dir, "n": 0, "manifest": None}

    _emit(f"loading model from {checkpoint}")
    model = catalog_runner.load_eval_model(checkpoint, num_res_blocks)
    os.makedirs(out_dir, exist_ok=True)

    done = [0]
    def _tick(label):
        if on_progress:
            on_progress(done[0], total, label)

    all_rows: List[Dict[str, Any]] = []
    for g, rows in lens_plan:
        for obj in rows:
            _tick(f"{g}: {obj['id']}")
            rec = catalog_runner.eval_catalog_object(
                model, obj, out_dir, cutout_size=cutout_size,
                asinh_scale=asinh_scale, checkpoint=checkpoint, grade=g,
                log=_emit)
            rec.setdefault("psnr_lr_hr", "")
            rec.setdefault("psnr_sr_hr", "")
            all_rows.append(rec)
            done[0] += 1
            _tick(f"{g}: {obj['id']}")

    if include_synthetic:
        _emit("synthetic group…")
        base = total - n
        syn = synthetic_runner.run_synthetic_eval(
            out_dir, n, model=model, asinh_scale=asinh_scale, seed=seed,
            on_progress=(lambda i, t, lbl: on_progress(base + i, total, lbl))
            if on_progress else None,
            log=_emit)
        all_rows.extend(syn["rows"])
        done[0] = total

    manifest_path = os.path.join(out_dir, "manifest.csv")
    with open(manifest_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=GROUPED_COLS, extrasaction="ignore")
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r.get(k, "") for k in GROUPED_COLS})

    if on_progress:
        on_progress(total, total, "done")
    n_ok = sum(1 for r in all_rows if r.get("ok"))
    _emit(f"\n✓ grouped analysis: {n_ok}/{len(all_rows)} ok → {manifest_path}")
    return {"out_dir": out_dir, "n": len(all_rows), "n_ok": n_ok,
            "manifest": manifest_path,
            "groups": {g: sum(1 for r in all_rows if r["grade"] == g and r.get("ok"))
                       for g in list(grades) + (["synthetic"] if include_synthetic else [])}}
