"""Unified grouped evaluation: A / B / C lens grades + synthetic, one run.

Runs the SR model over N real lens cutouts per grade (A/B/C — LR+SR only) plus
N synthetic validation triptychs (LR+SR+HR), into a single run dir with one
``manifest.csv`` whose ``grade`` column is the group ∈ {A, B, C, synthetic}.
Shares the SR model load across all groups and runs locally, in-process.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable
from typing import Any

from euclid_polish.config import Config
from euclid_polish.eval import catalog_runner, galaxy_catalog, synthetic_runner
from euclid_polish.eval.catalog_runner import EVAL_HR_SIZE, EVAL_LR_SIZE
from euclid_polish.eval.eval_catalog import read_eval_catalog

#: Manifest columns for a grouped run (superset: PSNR is synthetic-only).
GROUPED_COLS = [
    "id", "ra", "dec", "grade", "ok", "error", "out_subdir",
    "lr_total_e", "sr_total_e", "flux_ratio_sr_over_lr",
    "psnr_lr_hr", "psnr_sr_hr",
]
LENS_GRADES = ("A", "B", "C")

#: Each non-lens class (real-gal, syn-lens, syn-gal) is sized to match the real
#: lens total: N lenses per grade across the 3 A/B/C grades = 3N. So all four
#: groups are balanced at 3N — 3N real lenses, 3N real galaxies, 3N syn-lens, 3N
#: syn-gal — by drawing ``CLASS_MULT × n`` of each non-lens class.
CLASS_MULT = len(LENS_GRADES)

#: The Euclid cutout service can round a pixel-size request DOWN (a 53-px request
#: comes back 52²), which enforce_object_sizes would then drop. Request a couple
#: of extra VIS px so the stamp always lands ≥ EVAL_LR_SIZE; it is center-cropped
#: back to the canonical geometry afterwards, so the pad never reaches the output.
DOWNLOAD_PAD = 2


def _galaxy_plan(log: Callable[[str], None],
                 max_n: int | None = None) -> list[dict[str, Any]]:
    """Rows for the real-galaxy group, read from the cached galaxy catalog.

    Cache-only: the Euclid archive is queried solely by the standalone
    Query-galaxies step (``/api/evaluation/query-galaxies`` →
    :func:`galaxy_catalog.build`), which writes ``galaxies.csv``. The grouped run
    never logs in or queries — it just consumes that cache, capped at ``max_n``
    (the balanced 3N target; ``None`` = all). A missing or empty cache yields an
    empty plan with a hint, so galaxies never kill the A/B/C run.
    """
    gal_csv = galaxy_catalog.default_out_csv()
    if not os.path.isfile(gal_csv):
        log(f"real galaxies: no cached catalog at {gal_csv} — run the "
            "Query-galaxies step (Euclid archive login) first")
        return []
    rows = read_eval_catalog(gal_csv, max_n=max_n)
    if rows:
        log(f"real galaxies: {len(rows)} cached from {gal_csv}"
            + (f" (capped at {max_n})" if max_n is not None else ""))
    else:
        log(f"real galaxies: cached catalog empty ({gal_csv})")
    return rows


def run_grouped_analysis(
    out_dir: str, n: int, *,
    cutout_size: int = EVAL_LR_SIZE,
    catalog_path: str | None = None,
    checkpoint: str | None = None,
    num_res_blocks: int | None = None,
    asinh_scale: float | None = None,
    stamp_m: int = EVAL_HR_SIZE,
    seed: int = 0,
    grades=LENS_GRADES,
    include_synthetic: bool = True,
    include_galaxies: bool = True,
    lens_source_dir: str | None = None,
    unique_fields: bool = True,
    on_progress: Callable[[int, int, str], None] | None = None,
    log: Callable[[str], None] | None = None,
    model: Any = None,
) -> dict[str, Any]:
    """Prepare the four-group dataset into ``out_dir``; write one manifest.

    Every object is held at the canonical eval geometry: a ``EVAL_LR_SIZE``² LR
    stamp beside a ``EVAL_HR_SIZE``² SR/HR stamp. Real A/B/C cutouts are downloaded
    directly at the LR side (``cutout_size`` defaults to ``EVAL_LR_SIZE`` VIS px →
    the model yields ``EVAL_HR_SIZE``² SR), so there's nothing to throw away;
    :func:`~euclid_polish.eval.catalog_runner.enforce_object_sizes` then only trims
    a rare archive-rounding overhang or drops a stamp truncated at a survey edge. A
    larger ``cutout_size`` still works (it is center-cropped down). ``lens_source_dir``
    seeds real-lens FITS from an existing run dir so cached cutouts are reused (and
    merely re-cropped) instead of re-downloaded. ``unique_fields=False`` lets a
    synthetic field host both a syn-lens and a syn-gal stamp, maximizing yield from
    a small validation set.
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
        from euclid_polish.eval import lens_catalog
        _emit(f"catalog {catalog} not found — fetching from Zenodo…")
        lens_catalog.fetch(catalog)

    # Plan the work so the single progress bar spans every group.
    lens_plan = []
    for g in grades:
        rows = read_eval_catalog(catalog, grade=g, max_n=n)
        lens_plan.append((g, rows))
        _emit(f"grade {g}: {len(rows)} lens(es)")
    if include_galaxies:
        # Balance to 3N real galaxies (matches the 3N real lenses).
        gal_rows = _galaxy_plan(_emit, max_n=CLASS_MULT * n)
        if gal_rows:
            lens_plan.append(("gal", gal_rows))    # processed by the same A/B/C loop
    n_lens = sum(len(r) for _, r in lens_plan)     # now includes galaxies
    # Synthetic contributes 3N syn-lens + 3N syn-gal = 2·3N stamps.
    total = n_lens + (2 * CLASS_MULT * n if include_synthetic else 0)
    if total == 0:
        _emit("nothing to evaluate")
        return {"out_dir": out_dir, "n": 0, "manifest": None}

    # Load the SR model only for real lenses that do not already have cached
    # LR/SR FITS. Synthetic can be regenerated cheaply; A/B/C cutouts should
    # not be redownloaded when their existing outputs are present.
    # When the eval model is an ensemble we (re)generate the disagreement cubes
    # (stdSR + movie), so an object that only has a single-model SR.fits must be
    # re-run to add them — not treated as done. Cheap probe, no model load.
    from euclid_polish.ensemble import ensemble_available
    want_disagreement = (hasattr(model, "member_arrays") if model is not None
                         else ensemble_available())

    def _reusable(obj_id: str) -> bool:
        if catalog_runner.can_reuse_eval_object(
                catalog_runner.object_output_dir(out_dir, obj_id),
                require_disagreement=want_disagreement):
            return True
        return bool(lens_source_dir) and catalog_runner.can_reuse_eval_object(
            catalog_runner.object_output_dir(lens_source_dir, obj_id),
            require_disagreement=want_disagreement)

    needs_lens_model = any(
        not _reusable(obj["id"])
        for _, rows in lens_plan
        for obj in rows
    )
    if needs_lens_model and model is None:
        from euclid_polish.eval.ensemble_infer import load_eval_ensemble_or_single
        model = load_eval_ensemble_or_single(checkpoint, num_res_blocks, log=_emit)
    elif n_lens and not needs_lens_model:
        _emit("A/B/C lens outputs already present — reusing cached FITS")
    os.makedirs(out_dir, exist_ok=True)

    done = [0]
    def _tick(label):
        if on_progress:
            on_progress(done[0], total, label)

    # Download a hair larger than the canonical LR side so the archive's
    # round-down never lands below it (enforce_object_sizes crops back to target).
    download_size = max(cutout_size, EVAL_LR_SIZE + DOWNLOAD_PAD)

    all_rows: list[dict[str, Any]] = []
    for g, rows in lens_plan:
        for obj in rows:
            _tick(f"{g}: {obj['id']}")
            obj_dir = catalog_runner.object_output_dir(out_dir, obj["id"])
            # Seed cached cutouts from a prior run so they are re-cropped, not
            # re-downloaded.
            if lens_source_dir and not catalog_runner.can_reuse_eval_object(obj_dir):
                catalog_runner.seed_object_from_cache(
                    lens_source_dir, out_dir, obj["id"])
            # Existence is checked BEFORE any archive download: a cutout already
            # on disk is reused as-is and never re-fetched.
            from_cache = catalog_runner.can_reuse_eval_object(
                obj_dir, require_disagreement=want_disagreement)
            if from_cache:
                _emit(f"  • {obj['id']}: already present locally — skipping download")
                produced, err = True, ""
            else:
                r0 = catalog_runner.eval_catalog_object(
                    model, obj, out_dir, cutout_size=download_size,
                    asinh_scale=asinh_scale, checkpoint=checkpoint, grade=g,
                    log=_emit)
                produced, err = bool(r0.get("ok")), r0.get("error", "")
            # Hold the canonical geometry (LR EVAL_LR_SIZE², SR EVAL_HR_SIZE²),
            # then read the flux metrics off the cropped FITS so they describe
            # the stamp the gallery and Zoobot actually see. A stamp that came
            # out smaller than the target is dropped.
            if produced and catalog_runner.enforce_object_sizes(obj_dir, log=_emit):
                rec = catalog_runner.reuse_catalog_object(
                    obj, out_dir, grade=g, from_cache=from_cache, log=_emit)
            else:
                if produced:                    # produced but below target → drop
                    shutil.rmtree(obj_dir, ignore_errors=True)
                    err = (f"stamp below {EVAL_LR_SIZE}×{EVAL_LR_SIZE} LR / "
                           f"{EVAL_HR_SIZE}×{EVAL_HR_SIZE} SR")
                rec = catalog_runner._base_manifest_row(obj, grade=g)
                rec["error"] = err
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
                out_dir, CLASS_MULT * n,            # 3N syn-lens + 3N syn-gal
                model=model, asinh_scale=asinh_scale, stamp_m=stamp_m,
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
    groups: dict[str, int] = {}
    for r in all_rows:
        if r.get("ok"):
            groups[r["grade"]] = groups.get(r["grade"], 0) + 1
    return {"out_dir": out_dir, "n": len(all_rows), "n_ok": n_ok,
            "manifest": manifest_path, "groups": groups}
