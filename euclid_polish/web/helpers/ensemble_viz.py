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
import re
import shlex
import shutil

import numpy as np

from euclid_polish import ensemble_registry
from euclid_polish.config import Config
from euclid_polish.ensemble import (
    EnsembleModel,
    default_ensemble_dir,
    evaluate_member_on_records,
    evaluate_on_records,
    member_fingerprint,
    pca_field,
)
from euclid_polish.eval.ensemble_diagnostics import (
    EnsembleDiagnosticsAccumulator,
    render_calibration,
    render_std_vs_brightness,
    render_std_vs_error,
)
from euclid_polish.eval.power_spectrum import (
    EnsembleSpectrumAccumulator,
    render_ensemble_power_spectrum,
)
from euclid_polish.eval.subsets import eval_subset
from euclid_polish.image.tfio import read_images, tfrecord_path
from euclid_polish.model import _checkpoint_exists
from euclid_polish.provenance.checkpoint import read_checkpoint_provenance
from euclid_polish.provenance.defaults import default_store
from euclid_polish.provenance.gitinfo import capture_git
from euclid_polish.training.inference import infer_checkpoint_num_res_blocks
from euclid_polish.training.trainer import prune_orphaned_checkpoints
from euclid_polish.tracking import TrackingError
from euclid_polish.tracking import default_store as tracking_default_store
from euclid_polish.visualization.base import BaseVisualizer
from euclid_polish.web import fasrc_config
from euclid_polish.web.helpers.paths import _sky_records_local_dir
from euclid_polish.web.remote import STATE

_MEMBER_GLOB = "member_*"


def ensemble_dir() -> str:
    """Base directory of the ensemble — the canonical
    :func:`euclid_polish.ensemble.default_ensemble_dir`."""
    return default_ensemble_dir()


def _ensemble_out_dir() -> str:
    # Config.VIS_DIR may be relative (the default is "./data/vis"). Flask's
    # send_file resolves relative paths against app.root_path, so keep all
    # ensemble artifacts pinned to the process cwd instead.
    d = os.path.abspath(os.path.join(Config.VIS_DIR, "ensemble"))
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


def _member_last_step(member_dir: str) -> int | None:
    """Last logged step from the tail of training_log.csv (cheap; None when
    unreadable). Good enough for display — the trainer reads the
    authoritative step from the checkpoint itself."""
    p = os.path.join(member_dir, "training_log.csv")
    try:
        with open(p, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 4096))
            lines = f.read().decode(errors="replace").strip().splitlines()
        for line in reversed(lines):
            head = line.split(",", 1)[0]
            if head.isdigit():
                return int(head)
        return None
    except OSError:
        return None


def _member_origin(member_dir: str) -> dict | None:
    """The ``origin.json`` a training run wrote when it CREATED the member
    (op add/fork, fork source, seed, commit) — synced down with the member."""
    try:
        with open(os.path.join(member_dir, "origin.json")) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _dir_size_mb(d: str) -> float:
    total = 0
    for dp, _dirs, fns in os.walk(d):
        for fn in fns:
            try:
                total += os.path.getsize(os.path.join(dp, fn))
            except OSError:
                pass
    return total / 1e6


# --------------------------------------------------------------------------- #
# Per-member test PSNR (asinh space) — fingerprint-cached, so an unchanged
# member is never re-scored. The members table shows these with a rank.
# --------------------------------------------------------------------------- #

#: Held-out fields per member score. Fixed so cached values stay comparable
#: across refreshes (a different count would be a different metric).
MEMBER_PSNR_FIELDS = 100


def _member_psnr_cache_path() -> str:
    # abspath WITHOUT makedirs — read on every page render.
    return os.path.abspath(os.path.join(Config.VIS_DIR, "ensemble",
                                        "member_psnr.json"))


def _eval_records_fingerprint(records_dir: str | None,
                              subset: str) -> str | None:
    """Identity of the eval dataset itself: size+mtime of the ``dirty_`` and
    ``hr_`` records the PSNR scores against. A regenerated test set keeps its
    subset name, field count AND the member checkpoints unchanged — without
    this in the cache key, every PSNR shown after a dataset regen silently
    referred to the OLD records. rsync preserves mtimes, so a no-op sync
    keeps the fingerprint stable while a real change bumps it."""
    if not records_dir:
        return None
    parts = []
    for kind in ("dirty", "hr"):
        p = tfrecord_path(records_dir, f"{kind}_{subset}")
        try:
            st = os.stat(p)
        except OSError:
            return None
        parts.append(f"{kind}:{st.st_size}:{st.st_mtime_ns}")
    return "|".join(parts)


def _load_member_psnr_cache() -> dict:
    try:
        with open(_member_psnr_cache_path()) as f:
            cache = json.load(f)
        return cache if isinstance(cache, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _member_psnr_entry(cache: dict, name: str, mdir: str, subset: str,
                       records_fp: str | None = None) -> dict | None:
    """The member's cached score, or ``None`` when it must be (re)computed —
    missing, different subset/field count, the checkpoint changed, or the
    EVAL RECORDS themselves changed (a regenerated test set is a different
    metric even though subset/count/checkpoint all stay the same)."""
    if (cache.get("subset") != subset
            or int(cache.get("num_images", 0) or 0) != MEMBER_PSNR_FIELDS):
        return None
    if records_fp is not None and cache.get("records_fp") != records_fp:
        return None
    e = (cache.get("members") or {}).get(name)
    if not e:
        return None
    fp = member_fingerprint(mdir)
    if fp is None or e.get("fingerprint") != fp:
        return None
    return e


def update_member_psnr_cache(scores: dict[str, dict], subset: str,
                             records_fp: str | None = None) -> None:
    """Merge ``{member_name: {fingerprint, psnr, n_scored}}`` into the cache.

    A subset/field-count/eval-records change invalidates wholesale (different
    metric); entries for members no longer on disk are left alone — they are
    ignored on read and rewritten on the next refresh.
    """
    cache = _load_member_psnr_cache()
    if (cache.get("subset") != subset
            or int(cache.get("num_images", 0) or 0) != MEMBER_PSNR_FIELDS
            or cache.get("records_fp") != records_fp):
        cache = {"subset": subset, "num_images": MEMBER_PSNR_FIELDS,
                 "records_fp": records_fp, "members": {}}
    cache.setdefault("members", {}).update(scores)
    path = os.path.join(_ensemble_out_dir(), "member_psnr.json")
    with open(path, "w") as f:
        json.dump(cache, f, indent=2)


def training_curves_payload() -> list[dict]:
    """Training series for the in-browser curves — registry-ACTIVE members
    only. An archived member's directory can linger on disk (or come back from
    a FASRC leftover), and the series reader globs ``member_*`` — without this
    filter a tombstoned member kept showing in the PSNR curves. Each entry is
    enriched with trunk depth and the cached test PSNR so the chart can color
    lines by depth or by a test-PSNR gradient."""
    from euclid_polish.training.log_plot import ensemble_training_series

    base = ensemble_dir()
    active = {os.path.basename(d)
              for d in ensemble_registry.active_member_dirs(base)}
    rdir = _sky_records_local_dir()
    sub = eval_subset(rdir) if rdir else "test"
    rec_fp = _eval_records_fingerprint(rdir, sub)
    cache = _load_member_psnr_cache()
    out = []
    for s in ensemble_training_series(base):
        if s["name"] not in active:
            continue
        d = os.path.join(base, s["name"])
        entry = _member_psnr_entry(cache, s["name"], d, sub,
                                   records_fp=rec_fp)
        s["blocks"] = infer_checkpoint_num_res_blocks(d)
        s["test_psnr"] = (entry or {}).get("psnr")
        # Reconstruction norm from origin.json; members created before the
        # loss knob existed all trained with the then-hardcoded L1.
        origin = _member_origin(d)
        s["loss"] = ((origin or {}).get("loss_norm") or "l1")
        out.append(s)
    return out


def job_member_psnr(cap) -> dict:
    """Score each active member's test-set PSNR (asinh space), skipping members
    whose checkpoint fingerprint already has a cached score — so re-running
    after nothing changed costs nothing, and after a pull only the members that
    actually changed are re-evaluated."""
    base = ensemble_dir()
    rdir = _sky_records_local_dir()
    if not rdir:
        raise RuntimeError("no local sky records — sync them on the /sky page.")
    sub = eval_subset(rdir)
    rec_fp = _eval_records_fingerprint(rdir, sub)
    cache = _load_member_psnr_cache()
    dirs = [d for d in ensemble_registry.active_member_dirs(base)
            if os.path.isdir(d) and _checkpoint_exists(d)]
    todo = [d for d in dirs
            if _member_psnr_entry(cache, os.path.basename(d), d, sub,
                                  records_fp=rec_fp) is None]
    reused = [os.path.basename(d) for d in dirs if d not in todo]
    if reused:
        print(f"  • cached (checkpoint unchanged): {', '.join(reused)}")
    scores: dict[str, dict] = {}
    for i, d in enumerate(todo):
        name = os.path.basename(d)
        fp = member_fingerprint(d)
        cap.tick(i, len(todo), name)
        out = evaluate_member_on_records(
            d, rdir, subset=sub, num_images=MEMBER_PSNR_FIELDS,
            on_progress=lambda j, n, lbl, _i=i: cap.tick(
                _i, len(todo), f"{name}: {lbl}"))
        scores[name] = {"fingerprint": fp,
                        "psnr": out["psnr_stretched"],
                        "n_scored": out["n_scored"]}
        print(f"  ✓ {name}: {out['psnr_stretched']:.3f} dB "
              f"(asinh, {out['n_scored']} {sub} fields)")
    if scores:
        update_member_psnr_cache(scores, sub, records_fp=rec_fp)
    cap.tick(len(todo), len(todo), "done")
    return {"evaluated": sorted(scores), "reused": reused, "subset": sub}


def ensemble_status() -> dict:
    """Everything the ensemble page renders: registry-active members
    (+ seeds, sizes, cached test PSNR + rank), archived tombstones,
    test-data presence, and the latest eval summary (+ staleness)."""
    base = ensemble_dir()
    reg = ensemble_registry.load_registry(base)

    rdir = _sky_records_local_dir()
    sub = eval_subset(rdir) if rdir else "test"
    test_present = bool(rdir) and os.path.exists(
        tfrecord_path(rdir, f"dirty_{sub}"))
    status_rec_fp = _eval_records_fingerprint(rdir, sub)

    psnr_cache = _load_member_psnr_cache()
    members = []
    for d in [os.path.join(base, n) for n in reg["active"]]:
        if os.path.isdir(d) and _checkpoint_exists(d):
            lb = os.path.join(d, "loss_best")
            has_lb = os.path.isdir(lb) and _checkpoint_exists(lb)
            name = os.path.basename(d)
            # Cached test PSNR (asinh space): only shown while the checkpoint
            # it was scored on is the one on disk — a changed member reads "—"
            # until the next refresh re-scores it (and only it).
            entry = _member_psnr_entry(psnr_cache, name, d, sub,
                                       records_fp=status_rec_fp)
            origin = _member_origin(d)
            members.append({"name": name, "seed": _member_seed(d),
                            "has_loss_best": has_lb,
                            "size_mb": round(_dir_size_mb(d), 1),
                            "step": _member_last_step(d),
                            "blocks": infer_checkpoint_num_res_blocks(d),
                            "origin": origin,
                            "loss": ((origin or {}).get("loss_norm") or "l1"),
                            "psnr": (entry or {}).get("psnr")})
    # Rank by cached PSNR (1 = best); unscored members rank last, unranked.
    by_psnr = sorted((m for m in members if m["psnr"] is not None),
                     key=lambda m: -m["psnr"])
    ranks = {m["name"]: i + 1 for i, m in enumerate(by_psnr)}
    for m in members:
        m["psnr_rank"] = ranks.get(m["name"])
    # The ensemble uses each member's PSNR-best checkpoint only; loss_best/
    # stays on disk as a fork source but is not an ensemble model.
    n_models = len(members)
    active_labels = ensemble_registry.active_labels(base)

    # Same abspath as _ensemble_out_dir() but WITHOUT the makedirs — a page
    # render is read-only and must not create directories.
    out_dir = os.path.abspath(os.path.join(Config.VIS_DIR, "ensemble"))
    ps_path = os.path.join(out_dir, "ensemble_power_spectrum.png")
    power_spectrum_png = (os.path.relpath(ps_path, Config.VIS_DIR)
                          if os.path.isfile(ps_path) else None)
    # The Evaluations card can render as long as EITHER a figure already
    # exists or the per-field cube cache does (figures then render lazily).
    evaluations_available = bool(power_spectrum_png) or os.path.isfile(
        os.path.join(out_dir, "cubes", "viz_index.json"))

    summary = None
    summary_stale = False
    summary_path = os.path.join(out_dir, "eval_summary.json")
    if os.path.isfile(summary_path):
        try:
            with open(summary_path) as f:
                summary = json.load(f)
        except (OSError, json.JSONDecodeError):
            summary = None
    if summary is not None:
        # Membership changed since this eval ran → numbers describe a
        # different ensemble. Shown as a badge, not silently deleted.
        recorded = [str(x) for x in (summary.get("member_labels")
                    or summary.get("per_member_labels") or [])]
        summary_stale = recorded != active_labels

    return {
        "base_dir": base,
        "members": members,
        "archived": list(reg["archived"]),
        "n_members": len(members),
        "n_models": n_models,
        "records_dir": rdir,
        "eval_subset": sub,
        "test_present": test_present,
        "psnr_fields": MEMBER_PSNR_FIELDS,
        "power_spectrum_png": power_spectrum_png,
        "evaluations_available": evaluations_available,
        "eval_summary": summary,
        "eval_summary_stale": summary_stale,
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
#: Safety ceiling on how many evaluated fields to cache as viewer/animation
#: cubes (5 npy per field). ALL evaluated fields up to this are cached — raised
#: from a flat 24 so the browser + morph aren't limited to a slice of the test
#: set. ``data/vis`` is transient (cleared each eval).
ENSEMBLE_VIZ_FIELDS_MAX = 200

#: How many PCA components of the member-residual subspace to cache per field
#: for the morphing animation (M=5 members → residuals span M-1=4 dims; the top
#: 3 carry the disagreement).
ENSEMBLE_PCA_COMPONENTS = 3


def _ensemble_cubes_dir() -> str:
    return os.path.join(_ensemble_out_dir(), "cubes")


def job_ensemble_evaluate(cap, *, num_images: int) -> dict:
    """Evaluate the ensemble on the held-out test set; persist + return the summary.

    Also caches every evaluated field's ensemble-mean (SR), per-pixel std
    (stdSR) and PCA cubes under ``<vis>/ensemble/cubes/`` (up to
    :data:`ENSEMBLE_VIZ_FIELDS_MAX`) so the ``ensemble`` viewer + morph can show
    the whole test set client-side.
    """
    base = ensemble_dir()
    rdir = _sky_records_local_dir()
    if not rdir:
        raise RuntimeError("no local sky records — sync them on the /sky page.")
    sub = eval_subset(rdir)
    viz_cap = min(int(num_images), ENSEMBLE_VIZ_FIELDS_MAX)

    cubes_dir = _ensemble_cubes_dir()
    shutil.rmtree(cubes_dir, ignore_errors=True)      # fresh viz set per eval
    os.makedirs(cubes_dir, exist_ok=True)
    saved: list[int] = []
    pca_amps: dict[int, list[float]] = {}        # rec_index → [a0, a1, a2]
    pca_var: dict[int, list[float]] = {}         # rec_index → variance explained
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
        if len(saved) >= viz_cap:
            return
        rec = int(rec_index)
        np.save(os.path.join(cubes_dir, f"sr_{rec:05d}.npy"),
                np.asarray(mean, dtype=np.float32))
        np.save(os.path.join(cubes_dir, f"std_{rec:05d}.npy"),
                np.asarray(std, dtype=np.float32))
        _m, comps, amps, var_exp = pca_field(
            preds, n_components=ENSEMBLE_PCA_COMPONENTS)
        for i, comp in enumerate(comps):
            np.save(os.path.join(cubes_dir, f"pca{i}_{rec:05d}.npy"),
                    np.asarray(comp, dtype=np.float32))
        pca_amps[rec] = [float(a) for a in amps]
        pca_var[rec] = [float(v) for v in var_exp]
        # Individual member SR cubes → per-member viewer tiers.
        for i, mem in enumerate(np.asarray(preds, dtype=np.float32)):
            np.save(os.path.join(cubes_dir, f"member{i}_{rec:05d}.npy"), mem)
        saved.append(rec)

    def _prog(i, n, lbl):
        cap.tick(i, n, lbl)

    out = evaluate_on_records(base, rdir, num_images=int(num_images),
                              on_field=_on_field, on_progress=_prog)
    member_labels = list(out.get("member_labels", []))

    # The full eval already scored every member — bank the stretched PSNRs in
    # the per-member cache (free ride: no extra inference), but only when this
    # eval used the cache's canonical field count, so the numbers stay one
    # metric. Labels are "NN·psnr" → dir "member_NN".
    if int(num_images) == MEMBER_PSNR_FIELDS and out.get("n_scored"):
        scores = {}
        for lbl, p in zip(member_labels,
                          out.get("per_member_psnr_stretched", [])):
            name = f"member_{lbl.split('·')[0]}"
            fp = member_fingerprint(os.path.join(base, name))
            if fp is not None:
                scores[name] = {"fingerprint": fp, "psnr": float(p),
                                "n_scored": int(out["n_scored"])}
        if scores:
            update_member_psnr_cache(
                scores, sub, records_fp=_eval_records_fingerprint(rdir, sub))
    with open(os.path.join(cubes_dir, "viz_index.json"), "w") as f:
        json.dump({"subset": sub, "indices": saved,
                   "pca_n": ENSEMBLE_PCA_COMPONENTS, "pca_amps": pca_amps,
                   "pca_var": pca_var,
                   "member_labels": member_labels,
                   # Eval-dataset identity: the cubes are position-keyed into
                   # THESE records — regenerated records make them garbage.
                   "records_fp": _eval_records_fingerprint(rdir, sub)}, f)

    # Power-spectrum summary (HR vs ensemble-mean coherence + disagreement).
    if ps_acc[0] is not None and float(ps_acc[0].bc.sum()) > 0:
        curves = ps_acc[0].curves()
        ps_png = os.path.join(_ensemble_out_dir(), "ensemble_power_spectrum.png")
        render_ensemble_power_spectrum(ps_png, curves, n_fields=ps_acc[0].n_fields)

        def _jsonable(v):                       # curves are 1-D or (M, nbins) 2-D
            a = np.asarray(v, float)
            fmt = lambda x: None if not np.isfinite(x) else round(float(x), 6)  # noqa: E731
            return ([fmt(x) for x in a] if a.ndim <= 1
                    else [[fmt(x) for x in row] for row in a])

        with open(os.path.join(_ensemble_out_dir(),
                               "ensemble_power_spectrum.json"), "w") as f:
            json.dump({k: _jsonable(v) for k, v in curves.items()}, f)
        out["power_spectrum_fields"] = int(ps_acc[0].n_fields)

    # The pixel-level diagnostic figures render lazily from the fresh cubes —
    # drop the ones from the previous eval so the page never serves stale plots.
    for png in EVAL_DIAGNOSTIC_PNGS.values():
        try:
            os.remove(os.path.join(_ensemble_out_dir(), png))
        except FileNotFoundError:
            pass

    with open(os.path.join(_ensemble_out_dir(), "eval_summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    out["viz_fields"] = len(saved)
    return out


def _iter_cached_fields():
    """Yield ``(hr_vis, mean_vis, members_vis)`` per cached evaluated field.

    Streams the mean-SR (``sr_*.npy``) + individual member (``member*_*.npy``)
    cubes the last Evaluate wrote, paired with HR from the records — so any
    evaluation figure can be recomputed (e.g. after a code fix) in seconds with
    NO model inference and no full re-run. Yields nothing when the cache is
    missing, or when the membership changed since the cubes were written (a
    member archived/added → the position-keyed cubes are invalid).
    """
    cubes_dir = _ensemble_cubes_dir()
    man_path = os.path.join(cubes_dir, "viz_index.json")
    if not os.path.isfile(man_path):
        return
    with open(man_path) as f:
        man = json.load(f)
    if ([str(x) for x in man.get("member_labels", []) or []]
            != ensemble_registry.active_labels(ensemble_dir())):
        return
    idxs = [int(i) for i in man.get("indices", [])]
    sub = man.get("subset", "")
    n_members = len(man.get("member_labels", []))
    rdir = _sky_records_local_dir()
    hr_path = tfrecord_path(rdir, f"hr_{sub}") if rdir else ""
    if not idxs or n_members == 0 or not rdir or not os.path.exists(hr_path):
        return
    # Eval-dataset identity: the SR cubes were computed against the records
    # named in the manifest — pairing them with REGENERATED records would
    # silently mix two datasets (old SR vs new HR). Legacy manifests without
    # the fingerprint are treated as stale for the same reason.
    if man.get("records_fp") != _eval_records_fingerprint(rdir, sub):
        return

    hr_by = {r.index: r for r in read_images(hr_path, num_images=max(idxs) + 1)}
    for rec in idxs:
        sr_f = os.path.join(cubes_dir, f"sr_{rec:05d}.npy")
        hr = hr_by.get(rec)
        if not os.path.isfile(sr_f) or hr is None:
            continue
        members = [np.load(mf) for i in range(n_members)
                   if os.path.isfile(mf := os.path.join(cubes_dir, f"member{i}_{rec:05d}.npy"))]
        if not members:
            continue
        yield (_vis(np.asarray(hr.data, np.float32)), _vis(np.load(sr_f)),
               np.stack([_vis(m) for m in members], 0))


def _member_meta_from_labels(labels) -> list[dict]:
    """Per-member ``{"loss", "blocks"}`` for the power-spectrum line
    grouping, positional with ``labels`` ("NN·psnr" → member_NN)."""
    base = ensemble_dir()
    meta = []
    for lbl in labels:
        d = os.path.join(base, f"member_{str(lbl).split('·')[0]}")
        origin = _member_origin(d)
        meta.append({"loss": ((origin or {}).get("loss_norm") or "l1"),
                     "blocks": infer_checkpoint_num_res_blocks(d)})
    return meta


def regenerate_power_spectrum(color_by: str | None = None) -> str | None:
    """Re-render the ensemble power spectrum from the CACHED per-field cubes
    (see :func:`_iter_cached_fields`). ``color_by`` ∈ {"loss", "depth"}
    colors the per-member lines by that grouping. Returns the PNG path, or
    ``None`` if nothing is cached."""
    acc = None
    for hr_v, mean_v, mem_v in _iter_cached_fields():
        if acc is None:
            acc = EnsembleSpectrumAccumulator(
                int(hr_v.shape[0]), float(Config.DEFAULT_PIXEL_SCALE))
        acc.add(hr_v, mean_v, mem_v)
    if acc is None or float(acc.bc.sum()) <= 0:
        return None
    member_meta = None
    if color_by in ("loss", "depth"):
        man_path = os.path.join(_ensemble_cubes_dir(), "viz_index.json")
        try:
            with open(man_path) as f:
                member_meta = _member_meta_from_labels(
                    json.load(f).get("member_labels", []))
        except (OSError, json.JSONDecodeError):
            member_meta = None
    ps_png = os.path.join(_ensemble_out_dir(), "ensemble_power_spectrum.png")
    render_ensemble_power_spectrum(ps_png, acc.curves(), n_fields=acc.n_fields,
                                   member_meta=member_meta, color_by=color_by)
    return ps_png


#: Diagnostic figures rendered from the cached cubes: URL slug → PNG basename.
#: One sweep renders all three (they share the same pixel statistics pass).
EVAL_DIAGNOSTIC_PNGS = {
    "std-error": "ensemble_std_vs_error.png",
    "std-brightness": "ensemble_std_vs_brightness.png",
    "calibration": "ensemble_calibration.png",
}


def regenerate_eval_diagnostics() -> dict[str, str] | None:
    """Render the pixel-level diagnostic figures from the CACHED cubes.

    One pass over :func:`_iter_cached_fields` feeds a single
    :class:`EnsembleDiagnosticsAccumulator`; all figures in
    :data:`EVAL_DIAGNOSTIC_PNGS` are (re)rendered together. Returns
    ``{slug: png_path}`` or ``None`` when nothing is cached.
    """
    acc = EnsembleDiagnosticsAccumulator()
    for hr_v, mean_v, mem_v in _iter_cached_fields():
        acc.add(hr_v, mean_v, mem_v)
    if acc.n_fields == 0:
        return None
    out_dir = _ensemble_out_dir()
    renderers = {"std-error": render_std_vs_error,
                 "std-brightness": render_std_vs_brightness,
                 "calibration": render_calibration}
    out = {}
    for slug, render in renderers.items():
        png = os.path.join(out_dir, EVAL_DIAGNOSTIC_PNGS[slug])
        if render(png, acc):
            out[slug] = png
    return out or None


def _delete_remote_member(name: str) -> str:
    """Best-effort ``rm -rf`` of the member's dir on FASRC; returns a status
    line for the job output + campaign log. Never raises — the local archive
    already succeeded, so a remote hiccup only downgrades to a reminder."""
    remote = f"{remote_ensemble_dir().rstrip('/')}/{name}"
    if STATE.ssh is None or not STATE.ssh.is_connected():
        return (f"NOT deleted on FASRC (not connected) — remove {remote} "
                "there manually.")
    # rm -rf guard: absolute, reasonably deep, and unmistakably a member dir.
    if not (remote.startswith("/") and remote.count("/") >= 4
            and "/ensemble/member_" in remote):
        return f"NOT deleted on FASRC (refused unsafe path {remote!r})."
    try:
        rc, _out, err = STATE.ssh.run(f"rm -rf {shlex.quote(remote)}",
                                      timeout=120)
        if rc == 0:
            return f"deleted on FASRC ({remote})."
        return (f"FASRC delete failed (rc={rc}: {err.strip()[:200]}) — "
                f"remove {remote} manually.")
    except Exception as e:  # noqa: BLE001 — remote cleanup is best-effort
        return (f"FASRC delete failed ({type(e).__name__}: {e}) — "
                f"remove {remote} manually.")


def job_archive_member(cap, *, name: str) -> dict:
    """Retire one ensemble member: zip → tracking, tombstone, delete, purge.

    The zip lands in the active tracking campaign's ``models/``; the registry
    gets a permanent tombstone (so a FASRC mirror pulling the dir back never
    re-activates it); the local member dir is deleted; the FASRC-side copy is
    deleted too (best-effort, needs the SSH session); and the position-keyed
    ensemble cube cache is purged eagerly (eval summaries/plots invalidate
    lazily via the membership fingerprint).
    """
    if not re.fullmatch(r"member_\d{2,}", name or ""):
        raise RuntimeError(f"invalid member name {name!r}")
    base = ensemble_dir()
    reg = ensemble_registry.load_registry(base)
    if name not in reg["active"]:
        raise RuntimeError(f"{name} is not an active ensemble member")
    src = os.path.join(base, name)
    store = tracking_default_store()
    if not store.has_current():
        raise RuntimeError(
            "no active tracking campaign — start one on the /tracking page "
            "so the archived member has somewhere to go.")
    cap.tick(0, 3, f"zipping {name}")
    try:
        meta = store.archive_model_zip(
            src, f"ensemble-{name}",
            comment=f"archived from ensemble ({base})")
    except TrackingError as e:
        raise RuntimeError(f"archive failed: {e}") from e
    commit = (capture_git() or {}).get("short")
    cap.tick(1, 3, "updating registry")
    ensemble_registry.archive_member_entry(
        base, name, zip_path=os.path.join("models", meta["name"]),
        commit=commit)
    cap.tick(2, 4, "deleting member dir + caches")
    shutil.rmtree(src, ignore_errors=True)
    shutil.rmtree(_ensemble_cubes_dir(), ignore_errors=True)  # position-keyed
    cap.tick(3, 4, "deleting FASRC copy")
    remote_status = _delete_remote_member(name)
    store.append_log(
        f"Archived ensemble member `{name}` → `models/{meta['name']}` "
        f"({meta['size_bytes'] / 1e6:.1f} MB). Local member dir deleted; "
        f"cube cache purged. FASRC copy: {remote_status}")
    print(f"  ✓ {name} → tracking {meta['name']}; caches purged; "
          f"FASRC copy: {remote_status}")
    return {"zip": meta["name"], "member": name,
            "remote": remote_status}


def remote_ensemble_dir() -> str:
    """The ensemble dir on FASRC: sibling of the remote checkpoint dir."""
    cfg = fasrc_config.load()
    parent = os.path.dirname(cfg.ckpt_dir.rstrip("/")) or "."
    return os.path.join(parent, "ensemble")


def changed_members_from_itemize(out: str) -> set[str]:
    """Member names with CONTENT changes in ``rsync --itemize-changes`` output.

    A member counts as changed when a file under it would be created (``+``)
    or transferred for a checksum/size/time difference (``c``/``s``/``t``).
    Attribute-only lines (perms/owner — chronic on Linux→macOS pulls, where
    ``-a``'s perm-preserve half-fails) are ignored, else every member would
    read "changed" on every pull and the skip would never fire.

    Flag strings are 11 chars on rsync 3.x but 9 on macOS openrsync
    (protocol 29, no ACL/xattr columns) — accept both.
    """
    changed: set[str] = set()
    for line in out.splitlines():
        parts = line.rstrip().split(" ", 1)
        if len(parts) != 2:
            continue
        flags, path = parts
        if not path.startswith("member_") or len(flags) < 9:
            continue
        if flags[0] in ("<", ">", "c") and any(
                ch in flags[2:] for ch in ("+", "c", "s", "t")):
            changed.add(path.split("/", 1)[0])
    return changed


def job_ensemble_pull(cap) -> dict:
    """Download the trained ensemble (``member_NN/``) from FASRC to the local
    checkpoint tree, so the render / evaluate actions can run it locally.

    Member-aware: one ``--dry-run --itemize-changes`` probe decides which
    members actually changed on FASRC; only those are downloaded (and orphan-
    pruned). An unchanged ensemble downloads nothing — and the PSNR refresh
    afterwards is fingerprint-cached, so it re-scores only what was pulled.
    """
    if STATE.ssh is None or not STATE.ssh.is_connected():
        raise RuntimeError("not connected to FASRC — connect on the FASRC tab first.")
    remote = remote_ensemble_dir()
    local = ensemble_dir()
    os.makedirs(local, exist_ok=True)

    # Tombstones win: an archived member may still have a leftover dir on
    # FASRC (archives predating the remote-delete feature) — excluding it from
    # the rsync keeps it from resurrecting locally (which is exactly how a
    # tombstoned member reappeared in the training curves).
    tombstoned = {t["name"]
                  for t in ensemble_registry.load_registry(local)["archived"]}
    excludes = [f"--exclude=/{n}/" for n in sorted(tombstoned)]

    cap.tick(0, 0, "probing FASRC for changed members (rsync dry-run)")
    probe_rc, probe_out, _perr = STATE.ssh.rsync_pull(
        remote.rstrip("/") + "/", local,
        extra_args=["--dry-run", "--itemize-changes", *excludes], timeout=600)
    changed = changed_members_from_itemize(probe_out) - tombstoned

    rc, err = 0, ""
    if probe_rc != 0 and not probe_out.strip():
        # Probe itself failed (transport error, not perm noise) — fall back to
        # the old full-tree pull rather than wrongly concluding "no changes".
        cap.tick(0, 0, f"rsync {remote} → {local}")
        # rsync -a can exit non-zero on perm-preserve (Linux→macOS) while
        # still copying every file; the member count below is the success gate.
        rc, _out, err = STATE.ssh.rsync_pull(remote.rstrip("/") + "/", local,
                                             extra_args=excludes, timeout=3600)
        changed = {os.path.basename(d)
                   for d in glob.glob(os.path.join(local, _MEMBER_GLOB))
                   } - tombstoned
        print("  • change probe failed — pulled the full tree")
    elif not changed:
        print("  ✓ all members up to date on FASRC — nothing to download")
    else:
        for i, name in enumerate(sorted(changed)):
            cap.tick(i, len(changed), f"pull {name}")
            rc, _out, err = STATE.ssh.rsync_pull(
                f"{remote.rstrip('/')}/{name}/", os.path.join(local, name),
                timeout=3600)
        print(f"  ✓ downloaded {len(changed)} changed member(s): "
              + ", ".join(sorted(changed)))

    n = len([d for d in glob.glob(os.path.join(local, _MEMBER_GLOB))
             if _checkpoint_exists(d)])
    if n == 0:
        raise RuntimeError(
            f"pulled 0 members (rsync rc={rc}: {err.strip()[:300]}). Has the "
            "ensemble_train job finished and written members at "
            f"{remote} on FASRC?")
    # The pull rsyncs WITHOUT --delete, so checkpoint generations from
    # earlier pulls accumulate locally (member dirs doubled: 44.6 → 89.2 MB).
    # Sweep files no manifest references — only where something was pulled.
    pruned = 0
    for name in sorted(changed):
        d = os.path.join(local, name)
        for track in (d, os.path.join(d, "loss_best")):
            if os.path.isdir(track):
                pruned += prune_orphaned_checkpoints(track)
    if pruned:
        print(f"  • pruned {pruned} stale checkpoint file(s)")
    # Refresh the per-member test PSNRs — fingerprint-cached, so only members
    # the pull actually changed get re-scored (nothing new → costs nothing).
    psnr: dict = {}
    try:
        psnr = job_member_psnr(cap)
    except Exception as e:  # noqa: BLE001 — the pull itself succeeded
        print(f"  ! member PSNR refresh skipped: {type(e).__name__}: {e}")
    return {"local": local, "n_members": n,
            "changed": sorted(changed), "psnr": psnr}
