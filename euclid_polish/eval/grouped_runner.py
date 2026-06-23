"""Unified grouped evaluation: A / B / C lens grades + synthetic, one run.

Runs the SR model over N real lens cutouts per grade (A/B/C — LR+SR only) plus
N synthetic validation triptychs (LR+SR+HR), into a single run dir with one
``manifest.csv`` whose ``grade`` column is the group ∈ {A, B, C, synthetic}.
Shares the SR model load across all groups and runs locally, in-process.
"""

from __future__ import annotations

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
    stamp_m: int = 64,
    seed: int = 0,
    grades=LENS_GRADES,
    include_synthetic: bool = True,
    lens_crop_vis: Optional[int] = None,
    lens_source_dir: Optional[str] = None,
    unique_fields: bool = True,
    on_progress: Optional[Callable[[int, int, str], None]] = None,
    log: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Prepare the four-group dataset into ``out_dir``; write one manifest.

    ``lens_crop_vis`` center-crops each real A/B/C cutout to that many VIS pixels
    (and its SR to ``2×``) after it is downloaded or reused — handy for matching
    the synthetic stamp field of view. ``lens_source_dir`` seeds real-lens FITS
    from an existing run dir so cached cutouts are reused (and merely re-cropped)
    instead of re-downloaded. ``unique_fields=False`` lets a synthetic field host
    both a syn-lens and a syn-gal stamp, maximizing yield from a small validation
    set.
    """
    def _emit(m): (log or print)(m)

    if on_progress is None:                     # local/CLI run → visible bar
        from euclid_polish.eval.progress import tqdm_progress
        on_progress = tqdm_progress("grouped")

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
    n_lens = sum(len(r) for _, r in lens_plan)
    total = n_lens + (2 * n if include_synthetic else 0)
    if total == 0:
        _emit("nothing to evaluate")
        return {"out_dir": out_dir, "n": 0, "manifest": None}

    # Load the SR model only for real lenses that do not already have cached
    # LR/SR FITS. Synthetic can be regenerated cheaply; A/B/C cutouts should
    # not be redownloaded when their existing outputs are present.
    def _reusable(obj_id: str) -> bool:
        if catalog_runner.can_reuse_eval_object(
                catalog_runner.object_output_dir(out_dir, obj_id)):
            return True
        return bool(lens_source_dir) and catalog_runner.can_reuse_eval_object(
            catalog_runner.object_output_dir(lens_source_dir, obj_id))

    model = None
    needs_lens_model = any(
        not _reusable(obj["id"])
        for _, rows in lens_plan
        for obj in rows
    )
    if needs_lens_model:
        _emit(f"loading model from {checkpoint}")
        model = catalog_runner.load_eval_model(checkpoint, num_res_blocks)
    elif n_lens:
        _emit("A/B/C lens outputs already present — reusing cached FITS")
    os.makedirs(out_dir, exist_ok=True)

    done = [0]
    def _tick(label):
        if on_progress:
            on_progress(done[0], total, label)

    all_rows: List[Dict[str, Any]] = []
    for g, rows in lens_plan:
        for obj in rows:
            _tick(f"{g}: {obj['id']}")
            obj_dir = catalog_runner.object_output_dir(out_dir, obj["id"])
            # Seed cached cutouts from a prior run so they are re-cropped, not
            # re-downloaded.
            if lens_source_dir and not catalog_runner.can_reuse_eval_object(obj_dir):
                catalog_runner.seed_object_from_cache(
                    lens_source_dir, out_dir, obj["id"])
            if catalog_runner.can_reuse_eval_object(obj_dir):
                if lens_crop_vis:               # crop first → metrics match stamp
                    catalog_runner.crop_object_fits(obj_dir, lens_crop_vis, log=_emit)
                rec = catalog_runner.reuse_catalog_object(
                    obj, out_dir, grade=g, log=_emit)
            else:
                rec = catalog_runner.eval_catalog_object(
                    model, obj, out_dir, cutout_size=cutout_size,
                    asinh_scale=asinh_scale, checkpoint=checkpoint, grade=g,
                    log=_emit)
                if rec.get("ok") and lens_crop_vis:
                    catalog_runner.crop_object_fits(obj_dir, lens_crop_vis, log=_emit)
            rec.setdefault("psnr_lr_hr", "")
            rec.setdefault("psnr_sr_hr", "")
            all_rows.append(rec)
            done[0] += 1
            _tick(f"{g}: {obj['id']}")

    if include_synthetic:
        _emit("synthetic subgroups (syn-lens / syn-gal)…")
        base = n_lens
        try:
            syn = synthetic_runner.run_synthetic_eval(
                out_dir, n, model=model, asinh_scale=asinh_scale, stamp_m=stamp_m,
                seed=seed, unique_fields=unique_fields,
                on_progress=(lambda i, t, lbl: on_progress(base + i, base + t, lbl))
                if on_progress else None,
                log=_emit)
            all_rows.extend(syn["rows"])
        except Exception as e:  # noqa: BLE001 — synthetic must not kill A/B/C
            _emit(f"synthetic subgroups skipped: {type(e).__name__}: {e}")
        done[0] = total

    manifest_path = os.path.join(out_dir, "manifest.csv")
    catalog_runner.write_manifest_upsert(manifest_path, all_rows, GROUPED_COLS)

    if on_progress:
        on_progress(total, total, "done")
    n_ok = sum(1 for r in all_rows if r.get("ok"))
    _emit(f"\n✓ grouped analysis: {n_ok}/{len(all_rows)} ok → {manifest_path}")
    groups: Dict[str, int] = {}
    for r in all_rows:
        if r.get("ok"):
            groups[r["grade"]] = groups.get(r["grade"], 0) + 1
    return {"out_dir": out_dir, "n": len(all_rows), "n_ok": n_ok,
            "manifest": manifest_path, "groups": groups}
