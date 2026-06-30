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
import shutil

import numpy as np

from euclid_polish.config import Config
from euclid_polish.ensemble import EnsembleModel, evaluate_on_records, pca_field
from euclid_polish.eval.power_spectrum import (
    EnsembleSpectrumAccumulator,
    render_ensemble_power_spectrum,
)
from euclid_polish.eval.subsets import eval_subset
from euclid_polish.image.tfio import read_images, tfrecord_path
from euclid_polish.model import _checkpoint_exists
from euclid_polish.provenance.checkpoint import read_checkpoint_provenance
from euclid_polish.provenance.defaults import default_store
from euclid_polish.visualization.base import BaseVisualizer
from euclid_polish.web import fasrc_config
from euclid_polish.web.helpers.paths import _sky_records_local_dir
from euclid_polish.web.remote import STATE

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
            lb = os.path.join(d, "loss_best")
            has_lb = os.path.isdir(lb) and _checkpoint_exists(lb)
            members.append({"name": os.path.basename(d), "seed": _member_seed(d),
                            "has_loss_best": has_lb})
    # Each seed contributes its PSNR-best checkpoint and (when present) its
    # loss-best one — the ensemble loads/uses BOTH (include_loss_best=True).
    n_models = sum(1 + (1 if m["has_loss_best"] else 0) for m in members)

    rdir = _sky_records_local_dir()
    sub = eval_subset(rdir) if rdir else "test"
    test_present = bool(rdir) and os.path.exists(
        tfrecord_path(rdir, f"dirty_{sub}"))

    out_dir = os.path.join(Config.VIS_DIR, "ensemble")   # read-only here
    # The power-spectrum summary gets its own card; keep it out of the gallery.
    ps_path = os.path.join(out_dir, "ensemble_power_spectrum.png")
    power_spectrum_png = (os.path.relpath(ps_path, Config.VIS_DIR)
                          if os.path.isfile(ps_path) else None)
    pngs = []
    for p in sorted(glob.glob(os.path.join(out_dir, "*.png")),
                    key=os.path.getmtime, reverse=True)[:24]:
        if os.path.basename(p) == "ensemble_power_spectrum.png":
            continue
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
        "n_models": n_models,
        "records_dir": rdir,
        "eval_subset": sub,
        "test_present": test_present,
        "result_pngs": pngs,
        "power_spectrum_png": power_spectrum_png,
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


#: How many of the scored fields to persist as float cubes for the client-side
#: viewer (LR/SR/stdSR/HR). The metrics still use every field; only the viewer
#: cache is capped so data/vis stays bounded.
ENSEMBLE_VIZ_FIELDS = 24

#: How many PCA components of the member-residual subspace to cache per field
#: for the morphing animation (M=5 members → residuals span M-1=4 dims; the top
#: 3 carry the disagreement).
ENSEMBLE_PCA_COMPONENTS = 3


def _ensemble_cubes_dir() -> str:
    return os.path.join(_ensemble_out_dir(), "cubes")


def job_ensemble_evaluate(cap, *, num_images: int) -> dict:
    """Evaluate the ensemble on the held-out test set; persist + return the summary.

    Also caches the first :data:`ENSEMBLE_VIZ_FIELDS` fields' ensemble-mean (SR)
    and per-pixel std (stdSR) cubes under ``<vis>/ensemble/cubes/`` so the
    ``ensemble`` viewer collection can show LR · SR · stdSR · HR client-side.
    """
    base = ensemble_dir()
    rdir = _sky_records_local_dir()
    if not rdir:
        raise RuntimeError("no local sky records — sync them on the /sky page.")
    sub = eval_subset(rdir)

    cubes_dir = _ensemble_cubes_dir()
    shutil.rmtree(cubes_dir, ignore_errors=True)      # fresh viz set per eval
    os.makedirs(cubes_dir, exist_ok=True)
    saved: list[int] = []
    pca_amps: dict[int, list[float]] = {}        # rec_index → [a0, a1, a2]
    ps_acc: list = [None]                        # lazy EnsembleSpectrumAccumulator

    def _on_field(rec_index, _lr_cube, preds, mean, std, hr_cube):
        # Power spectrum over ALL fields that have HR (VIS band): HR vs
        # ensemble-mean (+ coherence r(k)) and the member-disagreement spectrum.
        if hr_cube is not None:
            hr_v, mean_v = _vis(hr_cube), _vis(mean)
            mem = np.asarray(preds, np.float32)
            mem_v = mem[..., 0] if mem.ndim == 4 else mem      # (M, H, W)
            if ps_acc[0] is None:
                ps_acc[0] = EnsembleSpectrumAccumulator(
                    int(hr_v.shape[0]), float(Config.DEFAULT_PIXEL_SCALE))
            ps_acc[0].add(hr_v, mean_v, mem_v)

        # LR/HR are read back from the records by the viewer; persist the
        # computed mean (SR) + std (stdSR) and the PCA disagreement basis
        # (mean + Σ aᵢ·sin·compᵢ powers the morphing animation). Cap the set.
        if len(saved) >= ENSEMBLE_VIZ_FIELDS:
            return
        rec = int(rec_index)
        np.save(os.path.join(cubes_dir, f"sr_{rec:05d}.npy"),
                np.asarray(mean, dtype=np.float32))
        np.save(os.path.join(cubes_dir, f"std_{rec:05d}.npy"),
                np.asarray(std, dtype=np.float32))
        _m, comps, amps = pca_field(preds, n_components=ENSEMBLE_PCA_COMPONENTS)
        for i, comp in enumerate(comps):
            np.save(os.path.join(cubes_dir, f"pca{i}_{rec:05d}.npy"),
                    np.asarray(comp, dtype=np.float32))
        pca_amps[rec] = [float(a) for a in amps]
        saved.append(rec)

    def _prog(i, n, lbl):
        cap.tick(i, n, lbl)

    out = evaluate_on_records(base, rdir, num_images=int(num_images),
                              on_field=_on_field, on_progress=_prog)
    with open(os.path.join(cubes_dir, "viz_index.json"), "w") as f:
        json.dump({"subset": sub, "indices": saved,
                   "pca_n": ENSEMBLE_PCA_COMPONENTS, "pca_amps": pca_amps}, f)

    # Power-spectrum summary (HR vs ensemble-mean coherence + disagreement).
    if ps_acc[0] is not None and float(ps_acc[0].bc.sum()) > 0:
        curves = ps_acc[0].curves()
        ps_png = os.path.join(_ensemble_out_dir(), "ensemble_power_spectrum.png")
        render_ensemble_power_spectrum(ps_png, curves, n_fields=ps_acc[0].n_fields)
        with open(os.path.join(_ensemble_out_dir(),
                               "ensemble_power_spectrum.json"), "w") as f:
            json.dump({k: [None if not np.isfinite(x) else round(float(x), 6)
                           for x in np.asarray(v, float)]
                       for k, v in curves.items()}, f)
        out["power_spectrum_fields"] = int(ps_acc[0].n_fields)

    with open(os.path.join(_ensemble_out_dir(), "eval_summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    out["viz_fields"] = len(saved)
    return out


def remote_ensemble_dir() -> str:
    """The ensemble dir on FASRC: sibling of the remote checkpoint dir."""
    cfg = fasrc_config.load()
    parent = os.path.dirname(cfg.ckpt_dir.rstrip("/")) or "."
    return os.path.join(parent, "ensemble")


def job_ensemble_pull(cap) -> dict:
    """Download the trained ensemble (``member_NN/``) from FASRC to the local
    checkpoint tree, so the render / evaluate actions can run it locally."""
    if STATE.ssh is None or not STATE.ssh.is_connected():
        raise RuntimeError("not connected to FASRC — connect on the FASRC tab first.")
    remote = remote_ensemble_dir()
    local = ensemble_dir()
    os.makedirs(local, exist_ok=True)
    cap.tick(0, 0, f"rsync {remote} → {local}")
    # rsync -a can exit non-zero on perm-preserve (Linux→macOS) while still
    # copying every file; the member count below is the real success gate.
    rc, _out, err = STATE.ssh.rsync_pull(remote.rstrip("/") + "/", local,
                                         timeout=3600)
    n = len([d for d in glob.glob(os.path.join(local, _MEMBER_GLOB))
             if _checkpoint_exists(d)])
    if n == 0:
        raise RuntimeError(
            f"pulled 0 members (rsync rc={rc}: {err.strip()[:300]}). Has the "
            "ensemble_train job finished and written members at "
            f"{remote} on FASRC?")
    print(f"  ✓ pulled {n} ensemble member(s) → {local}")
    return {"local": local, "n_members": n}
