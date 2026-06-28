"""Web helpers for the ensemble page: status, the disagreement render, and the
test-set evaluation job.

The disagreement render is the hallucination cross-check made visual: LR, the
ensemble-mean SR, the per-pixel spread across members (where members disagree =
where the SR is invented), and the HR truth when available.
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np

from euclid_polish.config import Config
from euclid_polish.ensemble import EnsembleModel, evaluate_on_records
from euclid_polish.eval.subsets import eval_subset
from euclid_polish.image.tfio import read_images, tfrecord_path
from euclid_polish.model import _checkpoint_exists
from euclid_polish.provenance.checkpoint import read_checkpoint_provenance
from euclid_polish.provenance.defaults import default_store
from euclid_polish.visualization.base import BaseVisualizer
from euclid_polish.web.helpers.paths import _sky_records_local_dir

_MEMBER_GLOB = "member_*"


def ensemble_dir() -> str:
    """Base directory of the ensemble: ``<ckpt parent>/ensemble`` (members in
    ``member_NN/`` beside the single-model ``wdsr`` checkpoint)."""
    parent = os.path.dirname(Config.DEFAULT_CHECKPOINT_DIR.rstrip("/")) or "."
    return os.path.join(parent, "ensemble")


def _ensemble_out_dir() -> str:
    d = os.path.join(Config.VIS_DIR, "ensemble")
    os.makedirs(d, exist_ok=True)
    return d


def _member_seed(member_dir: str) -> int | None:
    """The seed a member was trained with (from its checkpoint → training-run
    provenance), or ``None`` if unavailable."""
    try:
        stamp = read_checkpoint_provenance(member_dir)
        if stamp is None or stamp.produced_by is None:
            return None
        run = default_store().get_or_none(stamp.produced_by)
        return getattr(run, "seed", None) if run is not None else None
    except Exception:                                   # noqa: BLE001 — best-effort
        return None


def ensemble_status() -> dict:
    """Everything the ensemble page renders: members (+ seeds), test-data
    presence, recent disagreement PNGs, and the latest eval summary."""
    base = ensemble_dir()
    members = []
    for d in sorted(glob.glob(os.path.join(base, _MEMBER_GLOB))):
        if os.path.isdir(d) and _checkpoint_exists(d):
            members.append({"name": os.path.basename(d), "seed": _member_seed(d)})

    rdir = _sky_records_local_dir()
    sub = eval_subset(rdir) if rdir else "test"
    test_present = bool(rdir) and os.path.exists(
        tfrecord_path(rdir, f"dirty_{sub}"))

    out_dir = os.path.join(Config.VIS_DIR, "ensemble")   # read-only here
    pngs = []
    for p in sorted(glob.glob(os.path.join(out_dir, "*.png")),
                    key=os.path.getmtime, reverse=True)[:24]:
        pngs.append({"name": os.path.basename(p),
                     "rel": os.path.relpath(p, Config.VIS_DIR)})

    summary = None
    summary_path = os.path.join(out_dir, "eval_summary.json")
    if os.path.isfile(summary_path):
        try:
            with open(summary_path) as f:
                summary = json.load(f)
        except (OSError, json.JSONDecodeError):
            summary = None

    return {
        "base_dir": base,
        "members": members,
        "n_members": len(members),
        "records_dir": rdir,
        "eval_subset": sub,
        "test_present": test_present,
        "result_pngs": pngs,
        "eval_summary": summary,
    }


def _vis(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, np.float32)
    return a[..., 0] if a.ndim == 3 else a


def _render_disagreement_png(lr_vis, sr_vis, std_vis, hr_vis, out_png) -> str:
    """LR | ensemble-mean SR | disagreement (std) | HR — VIS planes, asinh."""
    cols = 4 if hr_vis is not None else 3
    viz = BaseVisualizer(rows=1, cols=cols, figsize=(5.2 * cols, 5.0))
    viz.add_scale_panel(lr_vis, stretch="asinh", title_suffix=" — LR (VIS)")
    viz.add_scale_panel(sr_vis, stretch="asinh",
                        title_suffix=" — ensemble mean SR (VIS)")
    viz.add_scale_panel(std_vis, stretch="asinh", cmap="magma",
                        colorbar_label="member std (e⁻)",
                        title_suffix=" — disagreement ≈ hallucination")
    if hr_vis is not None:
        viz.add_scale_panel(hr_vis, stretch="asinh", title_suffix=" — HR (VIS)")
    viz.save_figure(out_png)
    return out_png


def job_ensemble_render(cap, *, index: int) -> dict:
    """Run the ensemble on one held-out test field and render its disagreement."""
    base = ensemble_dir()
    ens = EnsembleModel(base, scale=Config.DEFAULT_REBIN_FACTOR,
                        num_res_blocks=Config.DEFAULT_NUM_RES_BLOCKS)
    if ens.n_members == 0:
        raise RuntimeError(
            f"no ensemble members under {base}/{_MEMBER_GLOB}; train an ensemble "
            "first (EnsembleModel.train / scripts/train_ensemble.py).")
    rdir = _sky_records_local_dir()
    if not rdir:
        raise RuntimeError("no local sky records — sync them on the /sky page.")
    sub = eval_subset(rdir)

    lr_recs = read_images(tfrecord_path(rdir, f"dirty_{sub}"), num_images=index + 1)
    if index >= len(lr_recs):
        raise RuntimeError(
            f"only {len(lr_recs)} {sub} fields available; index {index} out of range.")
    lr = lr_recs[index]
    hr_path = tfrecord_path(rdir, f"hr_{sub}")
    hr = None
    if os.path.exists(hr_path):
        hr_by = {h.index: h for h in read_images(hr_path, num_images=index + 1)}
        hr = hr_by.get(lr.index)

    cap.tick(0, ens.n_members, f"running {ens.n_members} members")
    mean, std = ens.predict(lr.data)
    cap.tick(ens.n_members, ens.n_members, "rendering")

    out_png = os.path.join(_ensemble_out_dir(),
                           f"ensemble_{sub}_idx{lr.index:04d}.png")
    _render_disagreement_png(_vis(lr.data), _vis(mean), _vis(std),
                             _vis(hr.data) if hr is not None else None, out_png)
    print(f"  ✓ {ens.n_members}-member disagreement → {out_png}")
    return {"png": out_png, "n_members": ens.n_members, "index": lr.index}


def job_ensemble_evaluate(cap, *, num_images: int) -> dict:
    """Evaluate the ensemble on the held-out test set; persist + return the summary."""
    base = ensemble_dir()
    rdir = _sky_records_local_dir()
    if not rdir:
        raise RuntimeError("no local sky records — sync them on the /sky page.")

    def _prog(i, n, lbl):
        cap.tick(i, n, lbl)

    out = evaluate_on_records(base, rdir, num_images=int(num_images),
                              on_progress=_prog)
    with open(os.path.join(_ensemble_out_dir(), "eval_summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    return out
