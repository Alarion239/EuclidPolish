"""Web helpers for the ensemble page: status, the disagreement render, and the
test-set evaluation job.

The disagreement render is the hallucination cross-check made visual: LR, the
ensemble-mean SR, the per-pixel spread across members (where members disagree =
where the SR is invented), and the HR truth when available.
"""

from __future__ import annotations

import base64
import contextlib
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
    render_std_vs_brightness,
    render_std_vs_error,
)
from euclid_polish.eval.power_spectrum import (
    LR_NYQUIST_CYC_ARCSEC,
    EnsembleSpectrumAccumulator,
    ensemble_ps_plot_curves,
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


_LEGACY_LAYOUT_MIGRATED = False

#: Artifacts that lived flat under ``<ensemble>/`` before the starfull/starless
#: regime split — moved once into their regime dir so an upgrade doesn't orphan a
#: fitted combiner, its eval payloads or the cached cubes (which is why the
#: combiner card + viewer would go blank after relaunching on the new code).
_LEGACY_FLAT_NAMES = (
    "combiner", "combiner_evals.json", "cubes", "cubes_validate",
    "ensemble_evals.json", "ensemble_power_spectrum.json",
    "ensemble_power_spectrum.png", "eval_summary.json",
)


def _migrate_legacy_flat_layout(out_dir: str) -> None:
    """One-time: relocate pre-regime-split flat artifacts into their regime dir.

    The pre-split layout was single-regime; its regime is read from the flat
    combiner (``starfull`` flag), defaulting to starfull (the historical
    combiner regime, and the pre-knob member default). Idempotent + best-effort;
    a move only happens when the source exists and the destination doesn't."""
    global _LEGACY_LAYOUT_MIGRATED
    if _LEGACY_LAYOUT_MIGRATED:
        return
    _LEGACY_LAYOUT_MIGRATED = True
    try:
        if not any(os.path.exists(os.path.join(out_dir, n))
                   for n in _LEGACY_FLAT_NAMES):
            return
        starless = False
        cj = os.path.join(out_dir, "combiner", "combiner.json")
        if os.path.isfile(cj):
            with contextlib.suppress(OSError, ValueError):
                with open(cj) as f:
                    starless = not bool(json.load(f).get("starfull", True))
        regime_dir = os.path.join(out_dir, "starless" if starless else "starfull")
        for name in _LEGACY_FLAT_NAMES:
            src = os.path.join(out_dir, name)
            dst = os.path.join(regime_dir, name)
            if os.path.exists(src) and not os.path.exists(dst):
                os.makedirs(regime_dir, exist_ok=True)
                shutil.move(src, dst)
    except Exception:                                   # noqa: BLE001 — best-effort
        pass


def _ensemble_out_dir() -> str:
    # Config.VIS_DIR may be relative (the default is "./data/vis"). Flask's
    # send_file resolves relative paths against app.root_path, so keep all
    # ensemble artifacts pinned to the process cwd instead.
    d = os.path.abspath(os.path.join(Config.VIS_DIR, "ensemble"))
    os.makedirs(d, exist_ok=True)
    _migrate_legacy_flat_layout(d)
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


def _eval_records_fingerprint(records_dir: str | None, subset: str, *,
                              starless: bool = False) -> str | None:
    """Identity of the eval dataset itself: size+mtime of the ``dirty_`` and the
    regime's TARGET records the reconstruction is scored against — ``clean_`` for
    starless (star-erased), ``hr_`` for starfull. A regenerated test set keeps
    its subset name, field count AND the member checkpoints unchanged — without
    this in the cache key, cached figures shown after a dataset regen silently
    referred to the OLD records. rsync preserves mtimes, so a no-op sync keeps
    the fingerprint stable while a real change bumps it. (Default starfull/``hr``
    so the member-PSNR cache and existing starfull cubes are unaffected.)"""
    if not records_dir:
        return None
    parts = []
    for kind in ("dirty", "clean" if starless else "hr"):
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
        # Per-member asinh knee (electrons) for the "by knee" coloring; None →
        # the per-band default (the client renders it as 100).
        s["asinh_knee"] = (origin or {}).get("asinh_knee")
        # Star regime (origin.json): starless members erase stars (clean
        # target), starfull reconstruct them (hr target). Pre-knob members
        # have no field → starfull. Drives the /ensemble mode toggle + filter.
        s["starless"] = bool((origin or {}).get("starless", False))
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
                            # Per-member asinh knee (electrons); None → default
                            # 100. Shown in the members table + "by knee" color.
                            "asinh_knee": (origin or {}).get("asinh_knee"),
                            # Star regime for the table's Regime column + the
                            # mode-toggle filter (origin.json; pre-knob members
                            # → starfull). Must match member_is_starless().
                            "starless": bool((origin or {}).get("starless", False)),
                            "psnr": (entry or {}).get("psnr")})
    # Rank by cached PSNR (1 = best) WITHIN each star regime. starfull and
    # starless are scored against different targets (hr vs clean) and render in
    # separate tables, so a shared 1..N enumeration across both would be
    # meaningless — each regime restarts at 1. Unscored members rank last,
    # unranked.
    ranks: dict[str, int] = {}
    for regime in (False, True):
        group = sorted((m for m in members
                        if m["starless"] is regime and m["psnr"] is not None),
                       key=lambda m: -m["psnr"])
        for i, m in enumerate(group):
            ranks[m["name"]] = i + 1
    for m in members:
        m["psnr_rank"] = ranks.get(m["name"])
    # The ensemble uses each member's PSNR-best checkpoint only; loss_best/
    # stays on disk as a fork source but is not an ensemble model.
    n_models = len(members)

    # Same abspath as _ensemble_out_dir() but WITHOUT the makedirs — a page
    # render is read-only and must not create directories. Artifacts are keyed
    # by star regime (starfull/starless are fully detached), so check both.
    out_dir = os.path.abspath(os.path.join(Config.VIS_DIR, "ensemble"))
    regime_dirs = [os.path.join(out_dir, r) for r in ("starless", "starfull")]
    ps_path = next((p for d in regime_dirs
                    if os.path.isfile(p := os.path.join(
                        d, "ensemble_power_spectrum.png"))), None)
    power_spectrum_png = (os.path.relpath(ps_path, Config.VIS_DIR)
                          if ps_path else None)
    # The Evaluations card can render as long as EITHER a figure already
    # exists or a per-field cube cache does (figures then render lazily) — in
    # either regime.
    evaluations_available = bool(power_spectrum_png) or any(
        os.path.isfile(os.path.join(d, "cubes", "viz_index.json"))
        for d in regime_dirs)

    summary = None
    summary_stale = False
    summary_starless = False
    summary_path = None
    for r in ("starless", "starfull"):
        p = os.path.join(out_dir, r, "eval_summary.json")
        if os.path.isfile(p):
            summary_path, summary_starless = p, (r == "starless")
            break
    if summary_path:
        try:
            with open(summary_path) as f:
                summary = json.load(f)
        except (OSError, json.JSONDecodeError):
            summary = None
    if summary is not None:
        # Membership changed since this eval ran → the numbers describe a
        # different ensemble. Compare against THIS regime's active members only:
        # starfull and starless are detached, so adding starless members must NOT
        # mark the starfull summary stale. Shown as a badge, not silently deleted.
        recorded = [str(x) for x in (summary.get("member_labels")
                    or summary.get("per_member_labels") or [])]
        summary_stale = recorded != _regime_labels(base, summary_starless)

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


def _lr_on_hr_grid(lr_cube, n: int) -> np.ndarray | None:
    """The LR VIS plane bicubic-resampled onto the ``(n, n)`` HR grid — the
    no-super-resolution baseline for the power-spectrum r(k) reference. Returns
    ``None`` if the LR cube is missing/degenerate."""
    if lr_cube is None:
        return None
    a = np.asarray(_vis(lr_cube), np.float64)
    if a.ndim != 2 or a.size == 0:
        return None
    if a.shape == (n, n):
        return a
    from scipy.ndimage import zoom
    up = zoom(a, (n / a.shape[0], n / a.shape[1]), order=3)  # bicubic baseline
    return up[:n, :n]


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


def _regime_slug(starless: bool) -> str:
    return "starless" if starless else "starfull"


def _ensemble_regime_dir(starless: bool) -> str:
    """Per-regime artifact root — ``<ensemble>/starless/`` or
    ``<ensemble>/starfull/``. The starfull and starless reconstructions are
    FULLY DETACHED: cubes, eval payloads, power spectrum, diagnostics and the
    fitted combiner each live under their regime's dir and never clobber."""
    d = os.path.join(_ensemble_out_dir(), _regime_slug(starless))
    os.makedirs(d, exist_ok=True)
    return d


def _ensemble_cubes_dir(subset: str | None = None, *, starless: bool) -> str:
    """The per-field cube bucket for one regime. No ``subset`` → ``cubes/`` (the
    TEST-eval bucket). A named ``subset`` (e.g. ``"validate"``, used by the
    combiner fit) → a sibling ``cubes_<subset>/`` so the buckets never clobber.
    Both sit under the regime dir, so starfull/starless cubes stay separate."""
    name = "cubes" if not subset else f"cubes_{subset}"
    return os.path.join(_ensemble_regime_dir(starless), name)


def _cache_field_cubes(cubes_dir: str, rec: int, preds: np.ndarray,
                       mean: np.ndarray, std: np.ndarray, *,
                       pca_components: int = ENSEMBLE_PCA_COMPONENTS
                       ) -> tuple[list[float], list[float]]:
    """Write one field's cubes (``sr_``, ``std_``, ``pcaN_``, ``memberi_``) into
    ``cubes_dir`` and return ``(pca_amps, pca_var)``. Shared by the test-eval and
    the validate combiner-fit caching so both lay out identical buckets."""
    rec = int(rec)
    np.save(os.path.join(cubes_dir, f"sr_{rec:05d}.npy"),
            np.asarray(mean, dtype=np.float32))
    np.save(os.path.join(cubes_dir, f"std_{rec:05d}.npy"),
            np.asarray(std, dtype=np.float32))
    _m, comps, amps, var_exp = pca_field(preds, n_components=pca_components)
    for i, comp in enumerate(comps):
        np.save(os.path.join(cubes_dir, f"pca{i}_{rec:05d}.npy"),
                np.asarray(comp, dtype=np.float32))
    for i, mem in enumerate(np.asarray(preds, dtype=np.float32)):
        np.save(os.path.join(cubes_dir, f"member{i}_{rec:05d}.npy"), mem)
    return [float(a) for a in amps], [float(v) for v in var_exp]


def _jsonable(v):
    """NaN-safe JSON conversion for 1-D or 2-D float arrays (NaN → None)."""
    a = np.asarray(v, float)
    fmt = lambda x: None if not np.isfinite(x) else round(float(x), 6)  # noqa: E731
    return ([fmt(x) for x in a] if a.ndim <= 1
            else [[fmt(x) for x in row] for row in a])


def _vis_stretched_psnr(a_vis, hr_vis) -> float:
    """Stretched-space PSNR (dB) of a VIS plane vs HR — the same asinh metric
    the ensemble/member curves use, so combiner vs mean vs member is
    apples-to-apples."""
    knee = float(Config.STRETCH_SCALE_E)
    peak = float(Config.PSNR_PEAK_STRETCHED)
    aa = np.arcsinh(np.asarray(a_vis, np.float64) / knee)
    hh = np.arcsinh(np.asarray(hr_vis, np.float64) / knee)
    mse = float(np.mean((aa - hh) ** 2))
    if mse <= 0.0:
        return float("inf")
    return float(10.0 * np.log10(peak * peak / mse))


class _CombinerMetricAcc:
    """Running VIS stretched-PSNR of the ensemble mean, the combiner, and each
    member vs HR — for the combiner comparison block (combiner vs mean vs best
    single member). Fed the same VIS planes as the spectrum accumulator, live
    and on lazy re-render, so both paths agree."""

    def __init__(self) -> None:
        self.mean = 0.0
        self.comb = 0.0
        self.mem: np.ndarray | None = None
        self.n = 0
        self.n_comb = 0

    def add(self, hr_v, mean_v, mem_v, comb_v) -> None:
        self.mean += _vis_stretched_psnr(mean_v, hr_v)
        mem_v = np.asarray(mem_v)
        if self.mem is None:
            self.mem = np.zeros(len(mem_v))
        for i, m in enumerate(mem_v):
            self.mem[i] += _vis_stretched_psnr(m, hr_v)
        if comb_v is not None:
            self.comb += _vis_stretched_psnr(comb_v, hr_v)
            self.n_comb += 1
        self.n += 1

    def block(self, member_labels) -> dict | None:
        if not self.n:
            return None
        mem = (self.mem / self.n) if self.mem is not None else np.array([])
        best_i = int(np.argmax(mem)) if mem.size else -1
        has_comb = self.n_comb > 0
        return {
            "available": bool(has_comb),
            "psnr": (self.comb / self.n_comb) if has_comb else None,
            "ensemble_mean_psnr": self.mean / self.n,
            "best_member_psnr": float(mem[best_i]) if mem.size else None,
            "best_member_label": (member_labels[best_i]
                                  if 0 <= best_i < len(member_labels) else None),
        }


def _evals_payload(ps_curves: dict | None, diag: EnsembleDiagnosticsAccumulator,
                   member_labels: list, subset: str,
                   combiner: dict | None = None) -> dict:
    """The complete Evaluations-card dataset, JSON-ready.

    Everything the FRONTEND renderers draw — power-spectrum curves,
    diagnostic histograms, calibration stats, per-member loss/depth meta and
    the guide constants — so styling choices (member-line coloring, tab
    switches) are instant client-side redraws; the cubes are only touched to
    (re)compute this payload."""
    payload: dict = {
        "subset": subset,
        "n_fields": int(diag.n_fields),
        "n_members": int(diag.n_members),
        "members": [{"label": str(lbl), **meta}
                    for lbl, meta in zip(
                        member_labels,
                        _member_meta_from_labels(member_labels))],
        "guides": {
            "lr_scale": 0.5 / LR_NYQUIST_CYC_ARCSEC,
            "vis_fwhm": float(Config.get_band("VIS").psf_fwhm_arcsec),
            "theta_min": float(Config.DEFAULT_PIXEL_SCALE),
            "rn_vis": float(Config.get_band("VIS").read_noise_e),
        },
        **diag.to_payload(),
    }
    payload["ps"] = None
    if ps_curves is not None:
        cv = ensemble_ps_plot_curves(ps_curves)
        payload["ps"] = {k: _jsonable(v) for k, v in cv.items()}
    payload["combiner"] = combiner       # test-time combiner metrics (or None)
    return payload


# ---------------------------------------------------------------------------
# Combiner (starfull): local fit on validate + payload
# ---------------------------------------------------------------------------

def _combiner_payload_path(starless: bool) -> str:
    return os.path.join(_ensemble_regime_dir(starless), "combiner_evals.json")


def _validate_records_present(rdir: str | None, *, starless: bool) -> bool:
    target = "clean" if starless else "hr"
    return bool(rdir) and all(
        os.path.exists(tfrecord_path(rdir, f"{k}_validate"))
        for k in ("dirty", target))


def _regime_labels(base: str, starless: bool) -> list[str]:
    """The active members' labels (``NN·psnr``) of one star regime, computed
    without loading any model — the per-regime membership fingerprint for cheap
    combiner/cube staleness checks. Delegates to the canonical registry helper
    so it matches exactly what :class:`EnsembleModel` loads for that regime."""
    try:
        return ensemble_registry.regime_labels(base, starless)
    except Exception:
        return []


def _reuse_validate_cubes(val_dir: str, records_fp) -> tuple[list[int], list[str]] | None:
    """If ``cubes_validate/`` holds a manifest matching the current validate
    records fingerprint, return ``(indices, member_labels)`` so the fit can reuse
    the cached member inference. Otherwise ``None`` (must re-infer)."""
    try:
        with open(os.path.join(val_dir, "viz_index.json")) as f:
            man = json.load(f)
    except (OSError, ValueError):
        return None
    if str(man.get("subset")) != "validate":
        return None
    if records_fp is not None and man.get("records_fp") != records_fp:
        return None
    labels = [str(x) for x in man.get("member_labels", []) or []]
    indices = [int(i) for i in man.get("indices", []) or []]
    if not labels or not indices:
        return None
    return indices, labels


def job_combiner_fit(cap, *, num_images: int, n_kernels: int = 12,
                     min_usage: float = 0.0, starless: bool = False) -> dict:
    """Fit the combiner (per-band RBF brightness gate) for one star regime
    LOCALLY on the validate split. Reuses cached validate member cubes when
    their fingerprint matches; else runs member inference once and caches them
    (``<regime>/cubes_validate/``). Persists to ``<regime>/combiner/`` + writes
    the combiner payload. Starfull fuses star reconstructions; starless fuses
    the star-erasing members (still useful — different members denoise/erase
    faint structure differently)."""
    from euclid_polish.eval.combiner import (
        BAND_NAMES, FitBufferAccumulator, fit_combiner, save_combiner)
    from euclid_polish.eval.ensemble_cube_cache import load_cached_member_stack

    base = ensemble_dir()
    rdir = _sky_records_local_dir()
    target = "clean" if starless else "hr"
    if not rdir:
        raise RuntimeError("no local sky records — sync them on the /sky page.")
    if not _validate_records_present(rdir, starless=starless):
        raise RuntimeError(
            "validate records not synced — enable 'Include validate' on the "
            f"/sky page so dirty_validate + {target}_validate are local.")

    fp = _eval_records_fingerprint(rdir, "validate", starless=starless)
    val_dir = _ensemble_cubes_dir("validate", starless=starless)
    acc = FitBufferAccumulator(BAND_NAMES)

    reuse = _reuse_validate_cubes(val_dir, fp)
    if reuse is not None:
        indices, labels = reuse
        hr_by = {h.index: h for h in read_images(
            tfrecord_path(rdir, f"{target}_validate"), num_images=int(num_images))}
        n = len(indices)
        for k, idx in enumerate(indices):
            stack = load_cached_member_stack(idx, subset="validate",
                                             cubes_dir=val_dir, active=labels)
            hr = hr_by.get(idx)
            if stack is not None and hr is not None:
                acc.add(stack, np.asarray(hr.data, np.float32))
            cap.tick(k + 1, n, f"reuse field {idx}")
    else:
        shutil.rmtree(val_dir, ignore_errors=True)
        os.makedirs(val_dir, exist_ok=True)
        saved: list[int] = []

        def _on_field(rec_index, _lr, preds, mean, std, hr):
            _cache_field_cubes(val_dir, rec_index, preds, mean, std)
            saved.append(int(rec_index))
            if hr is not None:
                acc.add(np.asarray(preds, np.float32), np.asarray(hr, np.float32))

        out = evaluate_on_records(
            base, rdir, subset="validate", num_images=int(num_images),
            starless=bool(starless), on_field=_on_field,
            on_progress=lambda i, n, l: cap.tick(i, n, l))
        labels = list(out.get("member_labels", []))
        with open(os.path.join(val_dir, "viz_index.json"), "w") as f:
            json.dump({"subset": "validate", "indices": saved,
                       "member_labels": labels, "records_fp": fp,
                       "pca_n": ENSEMBLE_PCA_COMPONENTS}, f)

    buffers = acc.buffers()
    if not any(np.asarray(X).size for X, _ in buffers.values()):
        raise RuntimeError("no validate pixels collected — check the records.")
    comb = fit_combiner(buffers, labels, n_kernels=int(n_kernels),
                        min_usage=float(min_usage))
    comb.records_fp = fp
    comb.starfull = not bool(starless)
    comb.fit_meta = {"subset": "validate", "num_images": int(num_images)}
    save_combiner(comb, _ensemble_regime_dir(starless))
    compute_combiner_payload(starless)
    return {"n_members": len(labels), "n_kernels": int(n_kernels),
            "min_usage": float(min_usage), "val_l1": comb.val_l1,
            "surviving": comb.surviving_members(), "subset": "validate",
            "regime": _regime_slug(starless)}


def compute_combiner_payload(starless: bool) -> dict | None:
    """(Re)compute the combiner payload from the persisted combiner: the
    per-band effective-weight (Jacobian) curves, survivors, and val loss.
    Returns the payload, or ``None`` when no combiner is saved."""
    from euclid_polish.eval.combiner import load_combiner
    comb = load_combiner(_ensemble_regime_dir(starless))
    if comb is None:
        return None
    stale = list(comb.member_labels) != _regime_labels(ensemble_dir(), starless)
    eff = {}
    for b in comb.bands:
        ew = comb.effective_weights(b)
        eff[b] = {"brightness_asinh": _jsonable(ew["brightness_asinh"]),
                  "brightness_e": _jsonable(ew["brightness_e"]),
                  "jacobian": _jsonable(ew["jacobian"])}
    payload = {
        "available": True, "stale": bool(stale), "kind": comb.kind,
        "regime": _regime_slug(starless),
        "member_labels": list(comb.member_labels),
        "members": [{"label": str(lbl), **meta} for lbl, meta in
                    zip(comb.member_labels,
                        _member_meta_from_labels(comb.member_labels))],
        "n_kernels": int(comb.n_kernels), "min_usage": float(comb.min_usage),
        "val_l1": comb.val_l1, "band_names": list(comb.band_names),
        "surviving": comb.surviving_members(), "eff_weights": eff,
    }
    with open(_combiner_payload_path(starless), "w") as f:
        json.dump(payload, f)
    return payload


def _evals_payload_path(starless: bool) -> str:
    return os.path.join(_ensemble_regime_dir(starless), "ensemble_evals.json")


def _diag_samples_path(starless: bool) -> str:
    """Sidecar for the pixel back-tracing samples (per histogram cell → example
    ``(field, y, x)`` locations). Loaded only on a heatmap-cell click, so it
    lives apart from ``ensemble_evals.json`` (which drives every redraw)."""
    return os.path.join(_ensemble_regime_dir(starless),
                        "ensemble_diag_samples.json")


def _write_diag_samples(starless: bool, diag) -> None:
    """Persist the accumulator's back-tracing reservoirs. Best-effort — a write
    failure never fails the evaluation (the plots still render, just without the
    click-to-inspect examples)."""
    try:
        with open(_diag_samples_path(starless), "w") as f:
            json.dump(diag.samples_payload(), f)
    except OSError:
        pass


#: Zoom stamps returned per back-traced pixel (half-window in HR pixels). Bigger
#: than the histogram cell needs → gives the eye some context around the pixel.
PIXEL_TRACE_HALF = 20
#: Example pixels returned per clicked cell (one per field, reservoir-sampled).
PIXEL_TRACE_STAMPS = 8


def _b64_f32(arr: np.ndarray) -> str:
    """Little-endian float32 C-order → base64. The stamps travel as compact
    typed blobs (not JSON number arrays) so a full-colour multi-band window is
    ~half the size and decodes to a Float32Array in one step."""
    return base64.b64encode(
        np.ascontiguousarray(arr, dtype="<f4").tobytes()).decode("ascii")


def _lr_cube_on_hr_grid(lr_cube, n: int):
    """All LR bands bicubic-resampled onto the ``(n, n)`` HR grid — the same
    no-super-resolution baseline the field viewer's LR tier shows, but keeping
    every band so the stamp can be coloured. ``None`` if the LR cube is bad."""
    if lr_cube is None:
        return None
    a = np.asarray(lr_cube, np.float64)
    if a.ndim == 2:
        a = a[..., None]
    if a.ndim != 3 or a.size == 0:
        return None
    if a.shape[0] == n and a.shape[1] == n:
        return a
    from scipy.ndimage import zoom
    up = zoom(a, (n / a.shape[0], n / a.shape[1], 1), order=3)  # bicubic
    return up[:n, :n]


def pixel_trace(starless: bool, diag: str, i: int, j: int,
                *, half: int = PIXEL_TRACE_HALF,
                max_stamps: int = PIXEL_TRACE_STAMPS) -> dict:
    """Back-trace one heatmap cell to real image stamps.

    Given a diagnostic (``"std_err"`` | ``"bright_std"``) and its histogram cell
    ``(i, j)``, read the sidecar's example pixel locations for that cell and cut
    a ``(2·half+1)²`` window around each — the real pixels that landed in the
    clicked cell. Each stamp carries the **full N-band** LR, HR and SR cubes
    (SR = the regime's combiner where available, else the ensemble mean) as
    base64 float32 so the frontend can render them with the field viewer's exact
    colour / knee / brightness, plus the single-band cross-member σ, and the
    per-pixel VIS numbers that place the pixel in the plot. Windows are zero-
    padded to a fixed ``(2·half+1)²`` with the sampled pixel at the centre.
    Returns ``{stamps: [...], ...}`` (``stamps`` empty when nothing sampled)."""
    S = 2 * int(half) + 1
    out = {"diag": diag, "i": int(i), "j": int(j), "half": int(half),
           "size": S, "bands": list(Config.LR_INPUT_BAND_NAMES),
           "stretch": float(Config.STRETCH_SCALE_E), "stamps": []}
    try:
        with open(_diag_samples_path(starless)) as f:
            side = json.load(f)
    except (OSError, json.JSONDecodeError):
        return out
    cells = side.get(diag) or {}
    picks = cells.get(f"{int(i)},{int(j)}") or []
    if not picks:
        return out

    cubes_dir = _ensemble_cubes_dir(starless=starless)
    rdir = _sky_records_local_dir()
    if not rdir:
        return out
    sub = eval_subset(rdir)
    target = "clean" if starless else "hr"
    hr_path = tfrecord_path(rdir, f"{target}_{sub}")
    if not os.path.exists(hr_path):
        return out

    # Group the (field, y, x) picks by field so each record + cube reads once.
    picks = [tuple(int(v) for v in p) for p in picks][:max_stamps]
    by_rec: dict[int, list[tuple[int, int]]] = {}
    for rec, y, x in picks:
        by_rec.setdefault(rec, []).append((y, x))
    max_idx = max(by_rec) + 1
    hr_by = {r.index: r for r in read_images(hr_path, num_images=max_idx)}
    # LR baseline (dirty records), matched by index — optional (skip the LR tier
    # if the dirty records aren't synced).
    lr_path = tfrecord_path(rdir, f"dirty_{sub}")
    lr_by = ({r.index: r for r in read_images(lr_path, num_images=max_idx)}
             if os.path.exists(lr_path) else {})

    def _crop(cube, y, x):
        """Zero-padded (S, S, C) window centred on (y, x) at (half, half)."""
        if cube.ndim == 2:
            cube = cube[..., None]
        H, W, C = cube.shape
        win = np.zeros((S, S, C), np.float32)
        y0, y1 = max(0, y - half), min(H, y + half + 1)
        x0, x1 = max(0, x - half), min(W, x + half + 1)
        win[y0 - (y - half):y1 - (y - half),
            x0 - (x - half):x1 - (x - half)] = cube[y0:y1, x0:x1]
        return win

    for rec, coords in by_rec.items():
        sr_f = os.path.join(cubes_dir, f"sr_{rec:05d}.npy")
        std_f = os.path.join(cubes_dir, f"std_{rec:05d}.npy")
        comb_f = os.path.join(cubes_dir, f"comb_{rec:05d}.npy")
        hr_rec = hr_by.get(rec)
        if not (os.path.isfile(sr_f) and os.path.isfile(std_f)
                and hr_rec is not None):
            continue
        hr_cube = np.asarray(hr_rec.data, np.float32)           # (H, W, C)
        n = int(hr_cube.shape[0])
        # SR = combiner where this field has it, else the ensemble mean.
        use_comb = os.path.isfile(comb_f)
        sr_cube = np.load(comb_f if use_comb else sr_f).astype(np.float32)
        std_v = _vis(np.load(std_f)).astype(np.float32)          # scalar σ (VIS)
        lr_rec = lr_by.get(rec)
        lr_cube = _lr_cube_on_hr_grid(
            np.asarray(lr_rec.data, np.float32), n) if lr_rec is not None else None
        hr_v, sr_v = _vis(hr_cube), _vis(sr_cube)
        for (y, x) in coords:
            if not (0 <= y < n and 0 <= x < n):
                continue
            hv, sv, dv = float(hr_v[y, x]), float(sr_v[y, x]), float(std_v[y, x])
            stamp = {
                "field": int(rec), "y": int(y), "x": int(x), "center": int(half),
                "sr_is_combiner": bool(use_comb),
                "hr": _b64_f32(_crop(hr_cube, y, x)),
                "sr": _b64_f32(_crop(sr_cube, y, x)),
                "std": _b64_f32(_crop(std_v, y, x)[..., 0]),
                "hr_val": hv, "sr_val": sv, "std_val": dv,
                "err_val": abs(sv - hv),
                "bright_asinh": float(np.arcsinh(hv / Config.STRETCH_SCALE_E)),
            }
            if lr_cube is not None:
                stamp["lr"] = _b64_f32(_crop(lr_cube, y, x))
            out["stamps"].append(stamp)
    return out


def compute_evaluation_payload(starless: bool) -> dict | None:
    """(Re)compute the Evaluations payload from the CACHED cubes — ONE sweep
    fills both the spectrum and the pixel-diagnostics accumulators — and
    persist it to the regime's ``ensemble_evals.json``. Returns the payload, or
    ``None`` when nothing (valid) is cached."""
    ps_acc = None
    diag = EnsembleDiagnosticsAccumulator()
    cmet = _CombinerMetricAcc()
    for hr_v, mean_v, mem_v, comb_v, lr_v, rec in _iter_cached_fields(starless):
        if ps_acc is None:
            ps_acc = EnsembleSpectrumAccumulator(
                int(hr_v.shape[0]), float(Config.DEFAULT_PIXEL_SCALE))
        ps_acc.add(hr_v, mean_v, mem_v, combiner=comb_v, lr=lr_v)
        diag.add(hr_v, mean_v, mem_v, combiner=comb_v, field_index=rec)
        cmet.add(hr_v, mean_v, mem_v, comb_v)
    if diag.n_fields == 0:
        return None
    _write_diag_samples(starless, diag)
    man_path = os.path.join(_ensemble_cubes_dir(starless=starless),
                            "viz_index.json")
    try:
        with open(man_path) as f:
            man = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    curves = (ps_acc.curves() if ps_acc is not None
              and float(ps_acc.bc.sum()) > 0 else None)
    payload = _evals_payload(curves, diag, man.get("member_labels", []),
                             man.get("subset", ""),
                             combiner=cmet.block(man.get("member_labels", [])))
    payload["regime"] = _regime_slug(starless)
    with open(_evals_payload_path(starless), "w") as f:
        json.dump(payload, f)
    return payload


def job_ensemble_evaluate(cap, *, num_images: int,
                          starless: bool = True) -> dict:
    """Evaluate the ensemble on the held-out test set; persist + return the summary.

    ``starless`` selects the regime: STARLESS members scored against the
    starless ``clean`` target (erase stars), STARFULL against the starfull
    ``hr`` target (reconstruct them). Only members of the matching regime are
    evaluated together (their targets differ, so a mixed mean is meaningless).

    Also caches every evaluated field's ensemble-mean (SR), per-pixel std
    (stdSR) and PCA cubes under ``<vis>/ensemble/<regime>/cubes/`` (up to
    :data:`ENSEMBLE_VIZ_FIELDS_MAX`) so the ``ensemble`` viewer + morph can show
    the whole test set client-side. Every artifact this writes (cubes, evals
    payload, power spectrum, diagnostics, summary) lives under the regime dir,
    so starfull and starless never clobber each other.
    """
    base = ensemble_dir()
    rdir = _sky_records_local_dir()
    if not rdir:
        raise RuntimeError("no local sky records — sync them on the /sky page.")
    sub = eval_subset(rdir)
    viz_cap = min(int(num_images), ENSEMBLE_VIZ_FIELDS_MAX)
    out_dir = _ensemble_regime_dir(starless)

    cubes_dir = _ensemble_cubes_dir(starless=starless)
    shutil.rmtree(cubes_dir, ignore_errors=True)      # fresh viz set per eval
    os.makedirs(cubes_dir, exist_ok=True)
    saved: list[int] = []
    pca_amps: dict[int, list[float]] = {}        # rec_index → [a0, a1, a2]
    pca_var: dict[int, list[float]] = {}         # rec_index → variance explained
    ps_acc: list = [None]                        # lazy EnsembleSpectrumAccumulator
    diag_acc = EnsembleDiagnosticsAccumulator()  # pixel-level diagnostics
    cmet = _CombinerMetricAcc()                  # combiner vs mean vs member PSNR

    # Apply this regime's fitted combiner (if any) as an extra reconstruction
    # series. Starfull fuses star reconstructions; starless fuses the
    # star-erasing members. Skipped if no combiner is saved for this regime or
    # it is stale for the current membership.
    from euclid_polish.eval.combiner import load_combiner
    combiner_model = load_combiner(
        out_dir, member_labels=_regime_labels(base, starless))

    def _on_field(rec_index, lr_cube, preds, mean, std, hr_cube):
        comb_full = None
        # Power spectrum over ALL fields that have HR (VIS band): HR vs
        # ensemble-mean (+ coherence r(k)) and the member-disagreement spectrum.
        if hr_cube is not None:
            hr_v, mean_v = _vis(hr_cube), _vis(mean)
            mem = np.asarray(preds, np.float32)
            mem_v = mem[..., 0] if mem.ndim == 4 else mem      # (M, H, W)
            if combiner_model is not None and mem.ndim == 4 and \
                    mem.shape[0] == len(combiner_model.member_labels):
                comb_full = combiner_model.apply_field(mem)    # (H, W, C) electrons
            comb_v = _vis(comb_full) if comb_full is not None else None
            lr_v = _lr_on_hr_grid(lr_cube, int(hr_v.shape[0]))  # baseline r(k)
            if ps_acc[0] is None:
                ps_acc[0] = EnsembleSpectrumAccumulator(
                    int(hr_v.shape[0]), float(Config.DEFAULT_PIXEL_SCALE))
            ps_acc[0].add(hr_v, mean_v, mem_v, combiner=comb_v, lr=lr_v)
            # Back-tracing samples only for fields whose cubes are cached (within
            # viz_cap) — beyond that the sr_/std_ stamps don't exist to show.
            # Error is scored against the combiner when present (the shipped
            # point estimate), else the ensemble mean.
            diag_acc.add(hr_v, mean_v, mem_v, combiner=comb_v,
                         field_index=(int(rec_index) if len(saved) < viz_cap
                                      else None))
            cmet.add(hr_v, mean_v, mem_v, comb_v)

        # LR/HR are read back from the records by the viewer; persist the
        # computed mean (SR) + std (stdSR) and the PCA disagreement basis
        # (mean + Σ aᵢ·sin·compᵢ powers the morphing animation). Cap the set.
        if len(saved) >= viz_cap:
            return
        rec = int(rec_index)
        amps, var_exp = _cache_field_cubes(cubes_dir, rec, preds, mean, std)
        if comb_full is not None:
            np.save(os.path.join(cubes_dir, f"comb_{rec:05d}.npy"),
                    np.asarray(comb_full, dtype=np.float32))
        pca_amps[rec] = amps
        pca_var[rec] = var_exp
        saved.append(rec)

    def _prog(i, n, lbl):
        cap.tick(i, n, lbl)

    out = evaluate_on_records(base, rdir, num_images=int(num_images),
                              starless=bool(starless),
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
    combiner_block = cmet.block(member_labels)
    has_combiner = bool(combiner_model is not None and cmet.n_comb > 0)
    with open(os.path.join(cubes_dir, "viz_index.json"), "w") as f:
        json.dump({"subset": sub, "indices": saved,
                   "pca_n": ENSEMBLE_PCA_COMPONENTS, "pca_amps": pca_amps,
                   "pca_var": pca_var,
                   "member_labels": member_labels,
                   "has_combiner": has_combiner,
                   # Eval-dataset identity: the cubes are position-keyed into
                   # THESE records — regenerated records make them garbage.
                   "records_fp": _eval_records_fingerprint(rdir, sub, starless=starless)}, f)

    # Power-spectrum summary (HR vs ensemble-mean coherence + disagreement).
    curves = None
    if ps_acc[0] is not None and float(ps_acc[0].bc.sum()) > 0:
        curves = ps_acc[0].curves()
        ps_png = os.path.join(out_dir, "ensemble_power_spectrum.png")
        render_ensemble_power_spectrum(ps_png, curves, n_fields=ps_acc[0].n_fields)
        with open(os.path.join(out_dir,
                               "ensemble_power_spectrum.json"), "w") as f:
            json.dump({k: _jsonable(v) for k, v in curves.items()}, f)
        out["power_spectrum_fields"] = int(ps_acc[0].n_fields)

    # Frontend Evaluations payload — the SAME pass already filled both
    # accumulators, so this is a free serialization (no cube re-read).
    if diag_acc.n_fields:
        payload = _evals_payload(curves, diag_acc, member_labels, sub,
                                 combiner=combiner_block)
        payload["regime"] = _regime_slug(starless)
        with open(_evals_payload_path(starless), "w") as f:
            json.dump(payload, f)
        _write_diag_samples(starless, diag_acc)

    # Combiner comparison numbers into the run summary.
    if combiner_block is not None and combiner_block.get("available"):
        out["combiner_psnr"] = combiner_block["psnr"]
        out["combiner_vs_mean_db"] = (
            combiner_block["psnr"] - combiner_block["ensemble_mean_psnr"])
        out["combiner_vs_best_member_db"] = (
            combiner_block["psnr"] - (combiner_block["best_member_psnr"] or 0.0))

    # The pixel-level diagnostic figures render lazily from the fresh cubes —
    # drop the ones from the previous eval so the page never serves stale plots.
    for png in EVAL_DIAGNOSTIC_PNGS.values():
        try:
            os.remove(os.path.join(out_dir, png))
        except FileNotFoundError:
            pass

    out["regime"] = _regime_slug(starless)
    with open(os.path.join(out_dir, "eval_summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    out["viz_fields"] = len(saved)
    return out


def _iter_cached_fields(starless: bool):
    """Yield ``(target_vis, mean_vis, members_vis, combiner_vis, lr_vis, rec)``
    per cached field of one regime (``rec`` = the field's record index — the key
    the ``sr_``/``std_`` cubes and the back-tracing sidecar are stored under).

    Streams the mean-SR (``sr_*.npy``) + individual member (``member*_*.npy``)
    cubes the last Evaluate wrote for this regime, paired with the regime's
    TARGET from the records (``clean`` for starless, ``hr`` for starfull) — so
    any evaluation figure can be recomputed (e.g. after a code fix) in seconds
    with NO model inference and no full re-run. ``combiner_vis`` is the VIS
    plane of the ``comb_*.npy`` cube when present, else ``None``; ``lr_vis`` is
    the LR plane bicubic-resampled onto the HR grid (the no-SR baseline for
    r(k)), or ``None`` when the dirty records are absent. Yields nothing when
    the cache is missing, or when the regime's membership changed since the
    cubes were written (a member archived/added → position-keyed cubes invalid).
    """
    cubes_dir = _ensemble_cubes_dir(starless=starless)
    man_path = os.path.join(cubes_dir, "viz_index.json")
    if not os.path.isfile(man_path):
        return
    with open(man_path) as f:
        man = json.load(f)
    if ([str(x) for x in man.get("member_labels", []) or []]
            != _regime_labels(ensemble_dir(), starless)):
        return
    idxs = [int(i) for i in man.get("indices", [])]
    sub = man.get("subset", "")
    n_members = len(man.get("member_labels", []))
    rdir = _sky_records_local_dir()
    target = "clean" if starless else "hr"
    hr_path = tfrecord_path(rdir, f"{target}_{sub}") if rdir else ""
    if not idxs or n_members == 0 or not rdir or not os.path.exists(hr_path):
        return
    # Eval-dataset identity: the SR cubes were computed against the records
    # named in the manifest — pairing them with REGENERATED records would
    # silently mix two datasets (old SR vs new HR). Legacy manifests without
    # the fingerprint are treated as stale for the same reason.
    if man.get("records_fp") != _eval_records_fingerprint(rdir, sub, starless=starless):
        return

    hr_by = {r.index: r for r in read_images(hr_path, num_images=max(idxs) + 1)}
    # LR baseline (optional): the dirty records, matched by index. Absent → the
    # r_lr curve is simply skipped, no error.
    lr_path = tfrecord_path(rdir, f"dirty_{sub}") if rdir else ""
    lr_by = ({r.index: r for r in read_images(lr_path, num_images=max(idxs) + 1)}
             if lr_path and os.path.exists(lr_path) else {})
    for rec in idxs:
        sr_f = os.path.join(cubes_dir, f"sr_{rec:05d}.npy")
        hr = hr_by.get(rec)
        if not os.path.isfile(sr_f) or hr is None:
            continue
        members = [np.load(mf) for i in range(n_members)
                   if os.path.isfile(mf := os.path.join(cubes_dir, f"member{i}_{rec:05d}.npy"))]
        if not members:
            continue
        comb_f = os.path.join(cubes_dir, f"comb_{rec:05d}.npy")
        comb_v = _vis(np.load(comb_f)) if os.path.isfile(comb_f) else None
        hr_v = _vis(np.asarray(hr.data, np.float32))
        lr_rec = lr_by.get(rec)
        lr_v = (_lr_on_hr_grid(np.asarray(lr_rec.data, np.float32),
                               int(hr_v.shape[0])) if lr_rec is not None else None)
        yield (hr_v, _vis(np.load(sr_f)),
               np.stack([_vis(m) for m in members], 0), comb_v, lr_v, rec)


def _member_meta_from_labels(labels) -> list[dict]:
    """Per-member ``{"loss", "blocks", "asinh_knee", "step", "psnr"}`` for line
    coloring (loss / depth / knee / test-PSNR gradient), positional with
    ``labels`` ("NN·psnr" → member_NN)."""
    base = ensemble_dir()
    rdir = _sky_records_local_dir()
    sub = eval_subset(rdir) if rdir else "test"
    rec_fp = _eval_records_fingerprint(rdir, sub)
    cache = _load_member_psnr_cache()
    meta = []
    for lbl in labels:
        name = f"member_{str(lbl).split('·')[0]}"
        d = os.path.join(base, name)
        origin = _member_origin(d)
        entry = _member_psnr_entry(cache, name, d, sub, records_fp=rec_fp)
        meta.append({"loss": ((origin or {}).get("loss_norm") or "l1"),
                     "blocks": infer_checkpoint_num_res_blocks(d),
                     "asinh_knee": (origin or {}).get("asinh_knee"),
                     "step": _member_last_step(d),
                     "psnr": (entry or {}).get("psnr")})
    return meta


def regenerate_power_spectrum(starless: bool,
                              color_by: str | None = None) -> str | None:
    """Re-render one regime's ensemble power spectrum from the CACHED per-field
    cubes (see :func:`_iter_cached_fields`). ``color_by`` ∈ {"loss", "depth",
    "knee"} colors the per-member lines by that grouping. Returns the PNG path,
    or ``None`` if nothing is cached."""
    acc = None
    for hr_v, mean_v, mem_v, comb_v, lr_v, _rec in _iter_cached_fields(starless):
        if acc is None:
            acc = EnsembleSpectrumAccumulator(
                int(hr_v.shape[0]), float(Config.DEFAULT_PIXEL_SCALE))
        acc.add(hr_v, mean_v, mem_v, combiner=comb_v, lr=lr_v)
    if acc is None or float(acc.bc.sum()) <= 0:
        return None
    member_meta = None
    if color_by in ("loss", "depth", "knee"):
        man_path = os.path.join(_ensemble_cubes_dir(starless=starless),
                                "viz_index.json")
        try:
            with open(man_path) as f:
                member_meta = _member_meta_from_labels(
                    json.load(f).get("member_labels", []))
        except (OSError, json.JSONDecodeError):
            member_meta = None
    ps_png = os.path.join(_ensemble_regime_dir(starless),
                          "ensemble_power_spectrum.png")
    render_ensemble_power_spectrum(ps_png, acc.curves(), n_fields=acc.n_fields,
                                   member_meta=member_meta, color_by=color_by)
    return ps_png


#: Diagnostic figures rendered from the cached cubes: URL slug → PNG basename.
#: One sweep renders both (they share the same pixel statistics pass).
EVAL_DIAGNOSTIC_PNGS = {
    "std-error": "ensemble_std_vs_error.png",
    "std-brightness": "ensemble_std_vs_brightness.png",
}


def regenerate_eval_diagnostics(starless: bool) -> dict[str, str] | None:
    """Render one regime's pixel-level diagnostic figures from the CACHED cubes.

    One pass over :func:`_iter_cached_fields` feeds a single
    :class:`EnsembleDiagnosticsAccumulator`; all figures in
    :data:`EVAL_DIAGNOSTIC_PNGS` are (re)rendered together. Returns
    ``{slug: png_path}`` or ``None`` when nothing is cached.
    """
    acc = EnsembleDiagnosticsAccumulator()
    for hr_v, mean_v, mem_v, comb_v, _lr_v, rec in _iter_cached_fields(starless):
        acc.add(hr_v, mean_v, mem_v, combiner=comb_v, field_index=rec)
    if acc.n_fields == 0:
        return None
    _write_diag_samples(starless, acc)
    out_dir = _ensemble_regime_dir(starless)
    renderers = {"std-error": render_std_vs_error,
                 "std-brightness": render_std_vs_brightness}
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
    # Every position-keyed cube bucket: test eval + validate combiner fit, in
    # BOTH star regimes (the archived member belongs to one, but purging all is
    # cheap and keeps no stale position-keyed cubes behind).
    for sl in (True, False):
        shutil.rmtree(_ensemble_cubes_dir(starless=sl), ignore_errors=True)
        shutil.rmtree(_ensemble_cubes_dir("validate", starless=sl),
                      ignore_errors=True)
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
