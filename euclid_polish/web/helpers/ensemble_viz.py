"""Web helpers for the Models workspace (the ensemble): member status, the
members table, member inspector and training curves, cached per-member
PSNR, the test-set evaluation job and the diagnostics
rebuilt from its cached cubes (power spectrum, pixel diagnostics with
back-tracing, PSNR vs knee, the Y/J/H payloads), the combiner fit / variant /
compare / promote jobs, member archive / restore / pull, and the
train-command preview.

The evaluation caches, per field, the ensemble-mean SR and the per-pixel
spread across members (where members disagree = where the SR is invented) —
the hallucination cross-check the ``ensemble`` viewer shows. (The standalone
disagreement render was removed with the legacy pages, 0ad56d9.)
"""

from __future__ import annotations

import base64
import contextlib
import dataclasses
import glob
import hashlib
import json
import os
import re
import shlex
import shutil
import zipfile
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import UTC, datetime

import numpy as np
from scipy.ndimage import zoom

from euclid_polish import ensemble_registry
from euclid_polish.config import Config
from euclid_polish.ensemble import (
    default_ensemble_dir,
    evaluate_member_on_records,
    evaluate_on_records,
    member_fingerprint,
    member_fingerprints,
    member_is_starless,
    pca_field,
)
from euclid_polish.eval import spatial_gate_compare as sgc
from euclid_polish.eval.combiner import (
    BAND_NAMES,
    COMBINER_MODELS,
    RAW_INCREMENTAL_MINMEANMAX_RBF_KIND,
    combiner_artifact_fingerprint,
    combiner_model_spec,
    combiner_region_ids,
    fit_combiner_minibatched,
    load_combiner,
    normalize_model_kind,
    save_combiner,
)
from euclid_polish.eval.ensemble_cube_cache import (
    BucketSync,
    bucket_member_path,
    bucket_member_paths,
    is_label_keyed,
    load_cached_field_lr,
    load_cached_member_stack,
    manifest_member_labels,
    member_cube_path,
    migrate_positional_bucket,
    missing_member_cubes,
    prune_bucket_fields,
    read_bucket_manifest,
    recorded_fingerprints,
    save_cached_field_lr,
    sync_bucket_members,
    write_bucket_manifest,
)
from euclid_polish.eval.ensemble_diagnostics import EnsembleDiagnosticsAccumulator
from euclid_polish.eval.gate_members import (
    DEFAULT_USED_THRESHOLD,
    member_peak_weights,
)
from euclid_polish.eval.knee_psnr import (
    KNEE_GRID_E,
    integrated_psnr,
    knee_psnr,
    stretched_truth,
)
from euclid_polish.eval.power_spectrum import (
    LR_NYQUIST_CYC_ARCSEC,
    EnsembleSpectrumAccumulator,
    EnsembleSpectrumCurves,
    ensemble_ps_plot_curves,
    render_ensemble_power_spectrum,
)
from euclid_polish.eval.spatial_gate import (
    SPATIAL_GATE_KIND,
    SpatialGateCombiner,
    band_scales,
    joined_after_fit,
    reads_available,
    restrict_to_available,
)
from euclid_polish.eval.spatial_gate_fit import (
    LazyMemberRunner,
    build_blackout_fields,
    fit_spatial_gate,
    load_cube_fields,
    split_holdout,
)
from euclid_polish.eval.subsets import eval_subset
from euclid_polish.image import Image
from euclid_polish.image.collection import ImageSet
from euclid_polish.image.tfio import read_images, tfrecord_path
from euclid_polish.model import _checkpoint_exists
from euclid_polish.provenance.checkpoint import read_checkpoint_provenance
from euclid_polish.provenance.defaults import default_store
from euclid_polish.provenance.gitinfo import capture_git
from euclid_polish.tracking import TrackingError
from euclid_polish.tracking import default_store as tracking_default_store
from euclid_polish.training import log_plot
from euclid_polish.training.inference import infer_checkpoint_num_res_blocks
from euclid_polish.training.target_blur import (
    blur_target_array,
    validate_target_fwhm_arcsec,
)
from euclid_polish.training.trainer import TRAINING_LOG_FILENAME, prune_orphaned_checkpoints
from euclid_polish.web import fasrc_config, fasrc_jobs, job_config
from euclid_polish.web.fasrc_pipeline import REGISTRY as STEP_REGISTRY
from euclid_polish.web.helpers.paths import _sky_records_local_dir
from euclid_polish.web.helpers.purge_requests import request_stale_purge
from euclid_polish.web.remote import STATE


def _record_index(record: Image) -> int:
    """Return the required persisted index of one TFRecord image."""
    index = record.index
    if index is None:
        raise RuntimeError("cached evaluation records must carry an index")
    return index


_MEMBER_GLOB = "member_*"

#: Raised by every job that needs the locally synced test/validate records.
_NO_SKY_RECORDS = "no local sky records — sync them on Synthetic › Records."


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
            with contextlib.suppress(OSError, ValueError), open(cj) as f:
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
            with contextlib.suppress(OSError):
                total += os.path.getsize(os.path.join(dp, fn))
    return total / 1e6


# --------------------------------------------------------------------------- #
# Per-member test PSNR (asinh space) — fingerprint-cached, so an unchanged
# member is never re-scored. The members table shows these with a rank.
# --------------------------------------------------------------------------- #

#: Held-out fields per member score. Fixed so cached values stay comparable
#: across refreshes (a different count would be a different metric).
MEMBER_PSNR_FIELDS = 100
#: Identity of the member-scoring rule. Bump it whenever the score of an
#: unchanged checkpoint would change, so cached scores are recomputed.
#: "regime-target": starless members are scored against ``clean_``, starfull
#: members against ``hr_`` (earlier caches scored everyone against ``hr_``).
MEMBER_PSNR_SCORING = "regime-target"


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


def _member_scoring_records_fingerprint(records_dir: str | None,
                                        subset: str) -> str | None:
    """Records identity for the per-member score cache.

    Starless members are scored against ``clean_`` and starfull members against
    ``hr_`` (see :func:`euclid_polish.ensemble.evaluate_member_on_records`), so a
    regenerated file of either kind must invalidate the cached scores.
    """
    starfull = _eval_records_fingerprint(records_dir, subset)
    starless = _eval_records_fingerprint(records_dir, subset, starless=True)
    if starfull is None and starless is None:
        return None
    return f"{starfull}||{starless}"


_RAW_INCREMENTAL_MINMEANMAX_RBF_KIND = RAW_INCREMENTAL_MINMEANMAX_RBF_KIND
_RBF_KIND = _RAW_INCREMENTAL_MINMEANMAX_RBF_KIND
_PCA_GATE_KINDS = {_RAW_INCREMENTAL_MINMEANMAX_RBF_KIND}
#: Spatial gate fit: share of validate fields held out for checkpoint
#: selection, and how many training fields get a blackout-augmented copy.
SPATIAL_GATE_HOLDOUT_FRACTION = 0.15
SPATIAL_GATE_BLACKOUT_FIELDS = 40
_ORDINARY_COMBINER_KINDS = tuple(COMBINER_MODELS)
_PCA_WEIGHT_SURFACE_SCHEMA = 3


def _normalize_combiner_kind(kind: str | None) -> str:
    return normalize_model_kind(kind)


def _combiner_artifact_dir(kind: str | None) -> str:
    return COMBINER_MODELS[_normalize_combiner_kind(kind)].artifact_dir


def _combiner_payload_name(kind: str | None) -> str:
    return COMBINER_MODELS[_normalize_combiner_kind(kind)].payload_name


def _combiner_cube_prefix(kind: str | None) -> str:
    return COMBINER_MODELS[_normalize_combiner_kind(kind)].cube_prefix


def _combiner_fingerprint(regime_dir: str, kind: str | None = None) -> str | None:
    """Identity of one fitted combiner artifact (size + mtime)."""
    npz = os.path.join(regime_dir, _combiner_artifact_dir(kind), "combiner.npz")
    try:
        st = os.stat(npz)
    except OSError:
        return None
    return f"{st.st_size}:{st.st_mtime_ns}"


def _eval_identity(base: str, rdir: str | None, sub: str, regime_dir: str,
                   *, starless: bool, num_images: int,
                   target_fwhm_arcsec: float = Config.TARGET_PSF_FWHM_ARCSEC) -> dict:
    """Everything that determines an evaluation's numbers: the eval dataset
    (records fingerprint), the exact member weights (per-member checkpoint
    fingerprints), the fitted combiner and the requested field count. Two evals
    with the same identity produce the same results — so a cached one with a
    matching identity is reused instead of re-running model inference."""
    target_fwhm = validate_target_fwhm_arcsec(target_fwhm_arcsec)
    labels = _regime_labels(base, starless)
    fingerprints = member_fingerprints(base, labels)
    member_fps = [fingerprints[str(lbl)] for lbl in labels]
    return {
        "records_fp": _eval_records_fingerprint(rdir, sub, starless=starless),
        "subset": sub,
        "num_images": int(num_images),
        "regime": _regime_slug(starless),
        "target_psf_fwhm_arcsec": target_fwhm,
        "member_fps": member_fps,
        # Kept for older summary readers; ``combiner_fps`` is the complete
        # identity now that the ordinary combiners are independent.
        "combiner_fp": _combiner_fingerprint(regime_dir, _RBF_KIND),
        "combiner_fps": {kind: _combiner_fingerprint(regime_dir, kind)
                         for kind in _ORDINARY_COMBINER_KINDS},
    }


def _read_eval_summary(starless: bool) -> dict | None:
    try:
        with open(os.path.join(_ensemble_regime_dir(starless),
                               "eval_summary.json")) as f:
            d = json.load(f)
        return d if isinstance(d, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _reusable_eval(starless: bool, identity: dict) -> dict | None:
    """The cached summary of a completed evaluation whose identity matches
    ``identity`` AND whose cubes are still on disk pointing at the same dataset
    and made by the same member checkpoints — else ``None`` (a real
    re-evaluation is needed). ``eval_summary.json`` is written last, so its
    presence with a matching identity means that run finished; the
    cube-manifest ``records_fp`` and ``member_fps`` guard against a cube
    wipe/regen or a refill since then, and every field the manifest lists must
    still hold every member's cube."""
    summary = _read_eval_summary(starless)
    if not summary or summary.get("eval_identity") != identity:
        return None
    cubes_dir = _ensemble_cubes_dir(starless=starless)
    man = read_bucket_manifest(cubes_dir)
    if man is None or not is_label_keyed(man):
        return None
    if man.get("records_fp") != identity["records_fp"]:
        return None
    if float(man.get("target_psf_fwhm_arcsec", -1.0)) != float(
            identity["target_psf_fwhm_arcsec"]):
        return None
    labels = manifest_member_labels(man)
    recorded = recorded_fingerprints(man)
    if [recorded[label] for label in labels] != list(identity.get("member_fps") or []):
        return None
    # A run stopped half-way (a forced one empties the bucket first) leaves
    # the previous run's summary behind: it stands only for a complete bucket.
    if not _bucket_current(cubes_dir, labels, recorded):
        return None
    return summary


def _archive_stale_path(starless: bool) -> str:
    """Durable queue of archived members whose cached cubes need rebuilding."""
    return os.path.join(_ensemble_regime_dir(starless), "archive_stale.json")


def _pending_archived_members(starless: bool) -> list[str]:
    """Archived member names waiting for this regime's next evaluation."""
    try:
        with open(_archive_stale_path(starless)) as f:
            names = json.load(f).get("members", [])
    except (OSError, ValueError, AttributeError):
        return []
    return [str(name) for name in names
            if re.fullmatch(r"member_\d{2,}", str(name))]


def _mark_archive_stale(starless: bool, name: str) -> None:
    """Remember an archive without touching the expensive cube cache yet."""
    names = _pending_archived_members(starless)
    if name not in names:
        names.append(name)
    _atomic_json(_archive_stale_path(starless), {"members": names})


def _unmark_archive_stale(starless: bool, name: str) -> None:
    """Forget a queued archive: the member was restored before the next
    evaluation, so its cached cubes and the combiners reading it stay."""
    names = _pending_archived_members(starless)
    if name not in names:
        return
    names.remove(name)
    if names:
        _atomic_json(_archive_stale_path(starless), {"members": names})
    else:
        _clear_archive_stale(starless)


def _clear_archive_stale(starless: bool) -> None:
    with contextlib.suppress(FileNotFoundError):
        os.remove(_archive_stale_path(starless))


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
    missing, different subset/field count or scoring rule, the checkpoint changed, or the
    EVAL RECORDS themselves changed (a regenerated test set is a different
    metric even though subset/count/checkpoint all stay the same)."""
    if (cache.get("subset") != subset
            or int(cache.get("num_images", 0) or 0) != MEMBER_PSNR_FIELDS
            or cache.get("scoring") != MEMBER_PSNR_SCORING):
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
            or cache.get("records_fp") != records_fp
            or cache.get("scoring") != MEMBER_PSNR_SCORING):
        cache = {"subset": subset, "num_images": MEMBER_PSNR_FIELDS,
                 "records_fp": records_fp, "scoring": MEMBER_PSNR_SCORING,
                 "members": {}}
    cache.setdefault("members", {}).update(scores)
    path = os.path.join(_ensemble_out_dir(), "member_psnr.json")
    with open(path, "w") as f:
        json.dump(cache, f, indent=2)


def job_member_psnr(cap) -> dict:
    """Score each active member's test-set PSNR (asinh space), skipping members
    whose checkpoint fingerprint already has a cached score — so re-running
    after nothing changed costs nothing, and after a pull only the members that
    actually changed are re-evaluated."""
    base = ensemble_dir()
    rdir = _sky_records_local_dir()
    if not rdir:
        raise RuntimeError(_NO_SKY_RECORDS)
    sub = eval_subset(rdir)
    rec_fp = _member_scoring_records_fingerprint(rdir, sub)
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
            on_progress=lambda j, n, lbl, _i=i, _name=name: cap.tick(
                _i, len(todo), f"{_name}: {lbl}"))
        scores[name] = {"fingerprint": fp,
                        "psnr": out["psnr_stretched"],
                        "n_scored": out["n_scored"]}
        print(f"  ✓ {name}: {out['psnr_stretched']:.3f} dB "
              f"(asinh, {out['n_scored']} {sub} fields)")
    if scores:
        update_member_psnr_cache(scores, sub, records_fp=rec_fp)
    cap.tick(len(todo), len(todo), "done")
    return {"evaluated": sorted(scores), "reused": reused, "subset": sub}


def ensemble_status(starless: bool | None = None) -> dict:
    """The ``/ensemble/status.json`` payload (Home reads it as a fallback):
    registry-active members (+ seeds, sizes, cached test PSNR + rank),
    archived tombstones, test-data presence, and the latest eval summary
    (+ staleness).

    ``starless`` picks WHICH regime's eval summary + staleness to report — the
    badge/stats must reflect the regime the user is viewing (``?mode=``), not
    just the first one present. ``None`` (the removed classic page; now only
    the tests omit it) keeps the old first-present behaviour (starless
    priority)."""
    base = ensemble_dir()
    reg = ensemble_registry.load_registry(base)

    rdir = _sky_records_local_dir()
    sub = eval_subset(rdir) if rdir else "test"
    test_present = bool(rdir) and os.path.exists(
        tfrecord_path(rdir, f"dirty_{sub}"))
    status_rec_fp = _member_scoring_records_fingerprint(rdir, sub)

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
            seed = (origin or {}).get("seed")
            members.append({"name": name,
                            # origin.json records the seed; the provenance
                            # store lookup (~0.4 s per member) is the fallback.
                            "seed": seed if seed is not None else _member_seed(d),
                            "has_loss_best": has_lb,
                            "size_mb": round(_dir_size_mb(d), 1),
                            "step": _member_last_step(d),
                            "blocks": infer_checkpoint_num_res_blocks(d),
                            "origin": origin,
                            "loss": ((origin or {}).get("loss_norm") or "l1"),
                            # Per-member asinh knee (electrons); None → default
                            # 100. Shown in the members table + "by knee" color.
                            "asinh_knee": (origin or {}).get("asinh_knee"),
                            # Star regime (origin.json; pre-knob members →
                            # starfull); Home counts members per regime by it.
                            # Must match member_is_starless().
                            "starless": bool((origin or {}).get("starless", False)),
                            "psnr": (entry or {}).get("psnr")})
    # Rank by cached PSNR (1 = best) WITHIN each star regime. starfull and
    # starless are scored against different targets (hr vs clean), so a
    # shared 1..N enumeration across both would be meaningless — each regime
    # restarts at 1. Unscored members rank last, unranked.
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
    # Evaluations are available (for the removed classic Evaluations card) as
    # long as EITHER a figure already exists or a per-field cube cache does
    # (figures then render lazily) — in either regime.
    evaluations_available = bool(power_spectrum_png) or any(
        os.path.isfile(os.path.join(d, "cubes", "viz_index.json"))
        for d in regime_dirs)

    summary = None
    summary_stale = False
    summary_starless = False
    summary_path = None
    # Report the REQUESTED regime's summary (mode-specific badge); fall back to
    # the first present when no regime is requested (``starless=None``).
    order = (("starless", "starfull") if starless is None
             else ("starless",) if starless else ("starfull",))
    for r in order:
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
        # The viewed regime's evaluation exists AND matches the current members
        # — the signal the removed SPA Ensemble page's Fit-combiner button
        # gated on (fit against a known baseline, not a stale/absent one); no
        # current reader.
        "evaluations_ready": bool(summary is not None and not summary_stale),
    }


def _vis(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, np.float32)
    return a[..., 0] if a.ndim == 3 else a


def _plane(arr: np.ndarray, band: int) -> np.ndarray:
    """Channel ``band`` of an ``(H, W, C)`` cube (a 2-D array is VIS only, so
    it answers band 0 and raises for any other band)."""
    a = np.asarray(arr, np.float32)
    if a.ndim == 3:
        return a[..., int(band)]
    if int(band) != 0:
        raise IndexError(f"a single-band array has no band {band}")
    return a


def _lr_on_hr_grid(lr_cube, n: int, band: int = 0) -> np.ndarray | None:
    """The LR plane of ``band`` (VIS by default) bicubic-resampled onto the
    ``(n, n)`` HR grid — the no-super-resolution baseline for the power-spectrum
    r(k) reference. Returns ``None`` if the LR cube is missing/degenerate."""
    if lr_cube is None:
        return None
    try:
        a = np.asarray(_plane(lr_cube, band), np.float64)
    except IndexError:
        return None
    if a.ndim != 2 or a.size == 0:
        return None
    if a.shape == (n, n):
        return a
    up = zoom(a, (n / a.shape[0], n / a.shape[1]), order=3)  # bicubic baseline
    return up[:n, :n]


#: How many of the scored fields to persist as float cubes for the client-side
#: viewer (LR/SR/stdSR/HR). The metrics still use every field; only the viewer
#: cache is capped so data/vis stays bounded.
#: Safety ceiling on how many evaluated fields to cache as viewer/animation
#: cubes (``sr_``, ``std_``, ``pcaN_``, one ``member_<key>_`` per member,
#: ``lr_`` and each baked combiner's npy per field). ALL evaluated fields up to
#: this are cached — raised from a flat 24 so the browser + morph aren't
#: limited to a slice of the test set. Member cubes persist across evaluations
#: (keyed by member label + checkpoint fingerprint); fields outside the
#: evaluated set are pruned.
ENSEMBLE_VIZ_FIELDS_MAX = 200

#: How many PCA components of the member-residual subspace to cache per field
#: for the morphing animation (M members → residuals span at most M-1 dims;
#: only the top 3 are cached).
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
                       lr: np.ndarray | None = None,
                       pca_components: int = ENSEMBLE_PCA_COMPONENTS
                       ) -> tuple[list[float], list[float]]:
    """Write one field's aggregate cubes (``sr_``, ``std_``, ``pcaN_`` of the
    member stack ``preds`` and, when given, the LR input ``lr_``) into
    ``cubes_dir`` and return ``(pca_amps, pca_var)``. Shared by the test-eval,
    the validate combiner-fit caching and the archive rebuild so all lay out
    identical buckets; the member cubes themselves are written per member
    (:func:`_run_missing_members`). ``sr_`` is written last, so its presence
    means the field's aggregates are complete (:func:`_drop_field_means`)."""
    rec = int(rec)
    if lr is not None:
        save_cached_field_lr(cubes_dir, rec, lr)
    np.save(os.path.join(cubes_dir, f"std_{rec:05d}.npy"),
            np.asarray(std, dtype=np.float32))
    for stale in glob.glob(os.path.join(cubes_dir, f"pca*_{rec:05d}.npy")):
        os.remove(stale)
    _m, comps, amps, var_exp = pca_field(preds, n_components=pca_components)
    for i, comp in enumerate(comps):
        np.save(os.path.join(cubes_dir, f"pca{i}_{rec:05d}.npy"),
                np.asarray(comp, dtype=np.float32))
    np.save(_field_mean_path(cubes_dir, rec), np.asarray(mean, dtype=np.float32))
    return [float(a) for a in amps], [float(v) for v in var_exp]


def _field_mean_path(cubes_dir: str, rec: int) -> str:
    """A field's cached ensemble mean (``sr_<rec>.npy``)."""
    return os.path.join(cubes_dir, f"sr_{int(rec):05d}.npy")


def _drop_field_means(cubes_dir: str, rec: int | None = None) -> None:
    """Delete one field's (``rec``) or every field's ``sr_`` before the member
    stack it averages changes: a fill or rebuild interrupted half-way then
    leaves the fields it did not reach without one, so the next fill
    recomputes their aggregates instead of trusting stale ones."""
    paths = ([_field_mean_path(cubes_dir, rec)] if rec is not None
             else glob.glob(os.path.join(cubes_dir, "sr_*.npy")))
    for path in paths:
        with contextlib.suppress(FileNotFoundError):
            os.remove(path)


def _drop_combiner_cubes(cubes_dir: str) -> None:
    """Delete every combiner output cube of a bucket: an evaluation bakes
    afresh those of the combiners that apply to its member stack."""
    for kind in COMBINER_MODELS:
        for path in glob.glob(os.path.join(cubes_dir, f"{_combiner_cube_prefix(kind)}_*.npy")):
            os.remove(path)


def _jsonable(v):
    """NaN-safe JSON conversion for 1-D or 2-D float arrays (NaN → None)."""
    if isinstance(v, dict):
        return {str(k): _jsonable(value) for k, value in v.items()}
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


def _vis_stretched_l1(a_vis, hr_vis) -> float:
    """Mean absolute VIS error in the same asinh space used by PSNR."""
    knee = float(Config.STRETCH_SCALE_E)
    aa = np.arcsinh(np.asarray(a_vis, np.float64) / knee)
    hh = np.arcsinh(np.asarray(hr_vis, np.float64) / knee)
    return float(np.mean(np.abs(aa - hh)))


class _CombinerMetricAcc:
    """Running VIS asinh PSNR and L1 for mean, combiner, and each member."""

    def __init__(self) -> None:
        self.mean = 0.0
        self.comb = 0.0
        self.mem: np.ndarray | None = None
        self.mean_l1 = 0.0
        self.comb_l1 = 0.0
        self.mem_l1: np.ndarray | None = None
        self.n = 0
        self.n_comb = 0

    def add(self, hr_v, mean_v, mem_v, comb_v) -> None:
        self.mean += _vis_stretched_psnr(mean_v, hr_v)
        self.mean_l1 += _vis_stretched_l1(mean_v, hr_v)
        mem_v = np.asarray(mem_v)
        if self.mem is None:
            self.mem = np.zeros(len(mem_v))
            self.mem_l1 = np.zeros(len(mem_v))
        assert self.mem_l1 is not None
        for i, m in enumerate(mem_v):
            self.mem[i] += _vis_stretched_psnr(m, hr_v)
            self.mem_l1[i] += _vis_stretched_l1(m, hr_v)
        if comb_v is not None:
            self.comb += _vis_stretched_psnr(comb_v, hr_v)
            self.comb_l1 += _vis_stretched_l1(comb_v, hr_v)
            self.n_comb += 1
        self.n += 1

    def block(self, member_labels) -> dict | None:
        if not self.n:
            return None
        mem = (self.mem / self.n) if self.mem is not None else np.array([])
        mem_l1 = (self.mem_l1 / self.n
                  if self.mem_l1 is not None else np.array([]))
        best_i = int(np.argmax(mem)) if mem.size else -1
        best_l1_i = int(np.argmin(mem_l1)) if mem_l1.size else -1
        has_comb = self.n_comb > 0
        return {
            "available": bool(has_comb),
            "psnr": (self.comb / self.n_comb) if has_comb else None,
            "asinh_l1": (self.comb_l1 / self.n_comb) if has_comb else None,
            "ensemble_mean_psnr": self.mean / self.n,
            "ensemble_mean_asinh_l1": self.mean_l1 / self.n,
            "best_member_psnr": float(mem[best_i]) if mem.size else None,
            "best_member_label": (member_labels[best_i]
                                  if 0 <= best_i < len(member_labels) else None),
            "best_member_asinh_l1": (float(mem_l1[best_l1_i])
                                      if mem_l1.size else None),
            "best_member_l1_label": (
                member_labels[best_l1_i]
                if 0 <= best_l1_i < len(member_labels) else None),
        }


def _summary_headline(model_cmet: dict[str, _CombinerMetricAcc],
                      labels: list) -> dict:
    """The eval summary's headline numbers, every one the VIS asinh PSNR
    (knee ``Config.STRETCH_SCALE_E``) over the scored test fields:

    * ``ensemble_psnr`` — the plain mean of the members;
    * ``mean_member_psnr`` / ``best_member_psnr`` (+ ``best_member_label``) —
      the average and the best single member;
    * ``ensemble_vs_mean_member_db`` / ``ensemble_vs_best_member_db`` — the
      mean's gain over them; ``ensemble_gain_db`` is the former (one meaning
      everywhere, as :meth:`EnsembleModel.evaluate` defines it);
    * per combiner kind ``<kind>_combiner_psnr``, ``…_vs_mean_db`` (over the
      ensemble MEAN), ``…_vs_best_member_db`` and ``…_vs_mean_member_db``.
      The production combiner is the spatial gate (``spatial_gate_*`` keys);
      the bare ``combiner_psnr`` / ``combiner_vs_mean_db`` keys are the RBF's
      (kept for older readers)."""
    base = next((c for c in model_cmet.values() if c.n), None)
    if base is None:
        return {}
    per_member = (base.mem / base.n) if base.mem is not None else np.array([])
    ensemble = float(base.mean / base.n)
    mean_member = float(np.mean(per_member)) if per_member.size else None
    best_i = int(np.argmax(per_member)) if per_member.size else -1
    best = float(per_member[best_i]) if per_member.size else None
    out: dict = {
        "psnr_metric": "vis_asinh",
        "psnr_knee_e": float(Config.STRETCH_SCALE_E),
        "n_scored": int(base.n),
        "ensemble_psnr": ensemble,
        "mean_member_psnr": mean_member,
        "best_member_psnr": best,
        "best_member_label": (str(labels[best_i])
                              if 0 <= best_i < len(labels) else None),
        "per_member_vis_psnr": [float(x) for x in per_member],
        "ensemble_vs_mean_member_db": (ensemble - mean_member
                                       if mean_member is not None else None),
        "ensemble_vs_best_member_db": (ensemble - best
                                       if best is not None else None),
    }
    out["ensemble_gain_db"] = out["ensemble_vs_mean_member_db"]
    for kind, cmet in model_cmet.items():
        block = cmet.block(labels)
        if not (block and block.get("available")):
            continue
        out[f"{kind}_combiner_psnr"] = block["psnr"]
        out[f"{kind}_combiner_vs_mean_db"] = block["psnr"] - block["ensemble_mean_psnr"]
        out[f"{kind}_combiner_vs_best_member_db"] = (
            block["psnr"] - (block["best_member_psnr"] or 0.0))
        if mean_member is not None:
            out[f"{kind}_combiner_vs_mean_member_db"] = block["psnr"] - mean_member
        if kind == _RBF_KIND:
            out["combiner_psnr"] = block["psnr"]
            out["combiner_vs_mean_db"] = block["psnr"] - block["ensemble_mean_psnr"]
            out["combiner_vs_best_member_db"] = (
                block["psnr"] - (block["best_member_psnr"] or 0.0))
    return out


def _evals_payload(ps_curves: EnsembleSpectrumCurves | None,
                   diag: EnsembleDiagnosticsAccumulator,
                   member_labels: list, subset: str,
                   combiner: dict | None = None,
                   model_combiners: dict[str, dict | None] | None = None,
                   coherence: dict | None = None,
                   band: str = "VIS") -> dict:
    """The complete Evaluations-card dataset, JSON-ready, for one ``band``
    (VIS is ``ensemble_evals.json``; the other bands are
    :func:`compute_band_evaluation_payloads`).

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
                        _member_meta_from_labels(member_labels),
                        strict=True)],
        "band": band,
        "guides": {
            "lr_scale": 0.5 / LR_NYQUIST_CYC_ARCSEC,
            "theta_min": float(Config.DEFAULT_PIXEL_SCALE),
            "band": band,
            "psf_fwhm": float(Config.get_band(band).psf_fwhm_arcsec),
            "read_noise": float(Config.get_band(band).read_noise_e),
            # the VIS names older readers use
            "vis_fwhm": float(Config.get_band("VIS").psf_fwhm_arcsec),
            "rn_vis": float(Config.get_band("VIS").read_noise_e),
        },
        **diag.to_payload(),
    }
    payload["ps"] = None
    if ps_curves is not None:
        cv = ensemble_ps_plot_curves(ps_curves)
        payload["ps"] = {k: _jsonable(v) for k, v in cv.items()}
    payload["coherence"] = None
    if coherence is not None:
        # Replace positional member IDs with the current human-readable labels
        # from viz_index.json. The score computation itself remains independent
        # of labels and therefore stays reusable for cached-cube rebuilds.
        c = dict(coherence)
        score_rows = []
        for row in coherence.get("scores", []):
            item = dict(row)
            sid = str(item.get("id", ""))
            if sid.startswith("member_"):
                try:
                    pos = int(sid.split("_", 1)[1])
                except ValueError:
                    pos = -1
                item["label"] = (str(member_labels[pos])
                                  if 0 <= pos < len(member_labels)
                                  else sid.replace("_", " "))
            else:
                fallback_label = {
                    "ensemble_mean": "ensemble mean",
                    "lr_baseline": "LR baseline",
                    "combiner": "combiner",
                    "model_agreement": "model agreement",
                }.get(sid)
                combiner_spec = COMBINER_MODELS.get(sid.removesuffix("_combiner"))
                item["label"] = (
                    fallback_label
                    if fallback_label is not None
                    else combiner_spec.label
                    if sid.endswith("_combiner") and combiner_spec is not None
                    else sid.replace("_", " ")
                )
            score_rows.append(item)
        c["scores"] = score_rows
        payload["coherence"] = c
    model_blocks = {
        str(kind): (dict(block) if block is not None else None)
        for kind, block in (model_combiners or {}).items()
    }
    if payload["coherence"] is not None:
        coherence_by_id = {
            str(row.get("id", "")): row
            for row in payload["coherence"].get("scores", [])
        }
        for kind, block in model_blocks.items():
            row = coherence_by_id.get(f"{kind}_combiner")
            if block is not None and row is not None:
                block["coherence_overall"] = row.get("overall")
                block["coherence_sr"] = row.get("sr")
    payload["combiner"] = combiner       # test-time combiner metrics (or None)
    payload["model_combiners"] = model_blocks
    return payload


# ---------------------------------------------------------------------------
# Combiner (starfull): local fit on validate + payload
# ---------------------------------------------------------------------------

def _combiner_payload_path(starless: bool, model_kind: str | None = None) -> str:
    """Per-model inspection payload. The RBF name stays legacy-compatible."""
    return os.path.join(_ensemble_regime_dir(starless),
                        _combiner_payload_name(model_kind))


def _combiner_signature(comb) -> str:
    """Stable signature for the fitted combiner parameters and member order."""
    h = hashlib.sha256()
    h.update(str(comb.kind).encode())
    h.update(json.dumps(list(comb.member_labels), separators=(",", ":")).encode())
    for value in (comb.coefficients, comb.centers, comb.scales,
                  comb.sigmas, comb.increment_ids,
                  comb.reference_features, comb.output_floors):
        a = np.ascontiguousarray(value)
        h.update(str(a.dtype).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def _unavailable_hr_weight_diagnostic(comb, *, target: str, reason: str) -> dict:
    return {
        "available": False,
        "reason": reason,
        "subset": "validate",
        "target": target,
        "member_labels": list(comb.weight_labels),
        "source_member_labels": list(comb.member_labels),
        "bands": {},
        "n_fields": 0,
        "n_pixels": 0,
        "combiner_signature": _combiner_signature(comb),
        "records_fp": comb.records_fp,
    }


def _hr_weight_diagnostic_from_bucket(comb, *, starless: bool,
                                      cubes_dir: str, target: str,
                                      max_pixels_per_bin: int = 10_000) -> dict:
    """Aggregate fitted weights by the corresponding validation target brightness.

    The target is used only to label/bin already-computed weights. It is never
    passed to ``comb``. The member stack and target are read from the same
    validation cache, and the manifest/member/fingerprint checks prevent
    silently pairing weights with a different dataset.
    """
    manifest = read_bucket_manifest(cubes_dir)
    if manifest is None:
        return _unavailable_hr_weight_diagnostic(
            comb, target=target, reason="validation cube manifest is missing")

    labels = manifest_member_labels(manifest)
    if labels != list(comb.member_labels):
        return _unavailable_hr_weight_diagnostic(
            comb, target=target, reason="validation cubes do not match combiner members")
    if str(manifest.get("subset", "")) != "validate":
        return _unavailable_hr_weight_diagnostic(
            comb, target=target, reason="validation cube manifest has the wrong subset")
    cache_fp = manifest.get("records_fp") or manifest.get("dirty_records_fp")
    if comb.records_fp is not None and cache_fp != comb.records_fp:
        return _unavailable_hr_weight_diagnostic(
            comb, target=target, reason="validation cubes do not match combiner records")

    indices = [int(i) for i in manifest.get("indices", []) or []]
    rdir = _sky_records_local_dir()
    target_path = tfrecord_path(rdir, f"{target}_validate") if rdir else ""
    if not indices or not rdir or not os.path.exists(target_path):
        return _unavailable_hr_weight_diagnostic(
            comb, target=target, reason="validation target records are missing")

    targets = {r.index: r for r in read_images(
        target_path, num_images=max(indices) + 1)}
    n_bins = 25
    lo, hi = map(float, comb.level_range)
    edges = np.linspace(lo, hi, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    source_m = len(comb.member_labels)
    weight_labels = list(comb.weight_labels)
    weight_m = len(weight_labels)
    sums = {name: np.zeros((n_bins, weight_m), np.float64) for name in comb.band_names}
    counts = {name: np.zeros(n_bins, np.int64) for name in comb.band_names}
    samples = {name: [[] for _ in range(n_bins)] for name in comb.band_names}
    per_field = max(1, int(max_pixels_per_bin) // max(len(indices), 1))
    rng = np.random.default_rng(0)
    used_fields = set()

    for rec in indices:
        target_image = targets.get(rec)
        if target_image is None:
            continue
        paths = bucket_member_paths(manifest, cubes_dir, labels, rec)
        if not all(path is not None and os.path.isfile(path) for path in paths):
            continue
        stack = np.stack([np.load(path).astype(np.float32) for path in paths], 0)
        truth = np.asarray(target_image.data, np.float32)
        if truth.ndim != 3 or stack.ndim != 4 or stack.shape[1:] != truth.shape:
            continue
        used_fields.add(int(rec))
        for ci, name in enumerate(comb.band_names):
            scale = float(Config.get_band(name).asinh_stretch_scale_e)
            x = np.arcsinh(stack[..., ci].reshape(source_m, -1).T / scale)
            y = np.arcsinh(truth[..., ci].reshape(-1) / scale)
            bin_idx = np.clip(np.digitize(y, edges) - 1, 0, n_bins - 1)
            for bi in np.unique(bin_idx):
                sel = np.flatnonzero(bin_idx == bi)
                if not len(sel):
                    continue
                pick = sel if len(sel) <= per_field else rng.choice(
                    sel, size=per_field, replace=False)
                # The diagnostic must stay bounded at K=64: evaluating every
                # pixel of 100 510² fields costs billions of RBF operations.
                # A deterministic, per-field brightness-stratified reservoir
                # gives each occupied brightness bin equal representation and
                # is exactly the sample used for its percentile ribbons.
                picked_weights = np.asarray(
                    comb.band_weights(name, x[pick]), np.float64)
                sums[name][bi] += picked_weights.sum(axis=0)
                counts[name][bi] += len(pick)
                samples[name][bi].append(picked_weights.astype(np.float32))

    if not used_fields:
        return _unavailable_hr_weight_diagnostic(
            comb, target=target, reason="no matching validation fields were found")

    bands = {}
    for name in comb.band_names:
        mean = np.full((n_bins, weight_m), np.nan, np.float64)
        valid = counts[name] > 0
        mean[valid] = sums[name][valid] / counts[name][valid, None]
        p16 = np.full_like(mean, np.nan)
        p84 = np.full_like(mean, np.nan)
        for bi, rows in enumerate(samples[name]):
            if not rows:
                continue
            sample = np.concatenate(rows, axis=0)
            p16[bi] = np.percentile(sample, 16, axis=0)
            p84[bi] = np.percentile(sample, 84, axis=0)
        scale = float(Config.get_band(name).asinh_stretch_scale_e)
        bands[name] = {
            "brightness_asinh": _jsonable(centers),
            "brightness_e": _jsonable(np.sinh(centers) * scale),
            "mean": _jsonable(mean),
            "p16": _jsonable(p16),
            "p84": _jsonable(p84),
            "counts": [int(x) for x in counts[name]],
        }

    return {
        "available": True,
        "subset": "validate",
        "target": target,
        "member_labels": weight_labels,
        "source_member_labels": list(comb.member_labels),
        "bands": bands,
        "n_fields": len(used_fields),
        "n_pixels": int(max((int(x.sum()) for x in counts.values()), default=0)),
        "combiner_signature": _combiner_signature(comb),
        "records_fp": comb.records_fp,
    }


def _hr_weight_diagnostic(comb, *, starless: bool,
                          ) -> dict:
    target = "clean" if starless else "hr"
    if comb.kind == _RAW_INCREMENTAL_MINMEANMAX_RBF_KIND:
        return _unavailable_hr_weight_diagnostic(
            comb, target=target,
            reason="this residual combiner has no member-routing weights")
    cubes_dir = _ensemble_cubes_dir("validate", starless=starless)
    return _hr_weight_diagnostic_from_bucket(
        comb, starless=starless, cubes_dir=cubes_dir, target=target)


def _cached_hr_weight_diagnostic(existing: dict | None, comb) -> dict | None:
    if not isinstance(existing, dict):
        return None
    if (existing.get("combiner_signature") == _combiner_signature(comb)
            and existing.get("member_labels") == list(comb.weight_labels)
            and existing.get("source_member_labels", list(comb.member_labels))
                == list(comb.member_labels)
            and existing.get("records_fp") == comb.records_fp):
        return existing
    return None


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


def _open_member_bucket(cubes_dir: str, *, records: dict, labels: list[str],
                        fingerprints: dict[str, str | None],
                        adopt: dict[str, str | None] | None = None,
                        wipe: bool = False) -> BucketSync:
    """Ready a member-cube bucket for an incremental fill: a positional bucket
    is renamed to the label keying (adopting ``adopt``); a bucket made from
    other records — any of the ``records`` manifest keys (subset, records
    fingerprint, target PSF) differs — or ``wipe`` is emptied, the only case
    in which every member is re-inferred; then its membership is synced to
    ``labels`` at ``fingerprints`` (departed and changed members' cubes are
    deleted, :func:`~euclid_polish.eval.ensemble_cube_cache.sync_bucket_members`;
    a departure also drops every field's mean, so the fill recomputes the
    aggregates)."""
    os.makedirs(cubes_dir, exist_ok=True)
    manifest = migrate_positional_bucket(cubes_dir, adopt=adopt) or {}
    if wipe or any(manifest.get(key) != value for key, value in records.items()):
        shutil.rmtree(cubes_dir, ignore_errors=True)
        os.makedirs(cubes_dir, exist_ok=True)
        manifest = {}
    if any(label not in labels for label in manifest_member_labels(manifest)):
        _drop_field_means(cubes_dir)
    return sync_bucket_members(cubes_dir, {**manifest, **records}, labels, fingerprints)


def _bucket_current(cubes_dir: str, labels: list[str],
                    fingerprints: dict[str, str | None]) -> bool:
    """Whether a label-keyed bucket holds, for exactly ``labels``, a cube of
    every listed field made by the checkpoints ``fingerprints`` name."""
    manifest = read_bucket_manifest(cubes_dir)
    if not is_label_keyed(manifest) or manifest_member_labels(manifest) != list(labels):
        return False
    recorded = recorded_fingerprints(manifest)
    if any(recorded[label] != fingerprints.get(label) for label in labels):
        return False
    indices = [int(i) for i in (manifest or {}).get("indices", []) or []]
    return bool(indices) and not any(missing_member_cubes(cubes_dir, labels, rec)
                                     for rec in indices)


def _proven_test_fingerprints(starless: bool, manifest: dict) -> dict[str, str | None] | None:
    """The member fingerprints a positional TEST bucket provably holds: those
    its evaluation recorded when it RAN the members (``eval_summary.json``
    identity, same members and records). A summary rebuilt from cubes stamped
    the checkpoints current at that time, not the ones that made the cubes, so
    it proves nothing (``None``)."""
    summary = _read_eval_summary(starless) or {}
    identity = summary.get("eval_identity") or {}
    labels = manifest_member_labels(manifest)
    fps = identity.get("member_fps")
    if (summary.get("recomputed_from_cubes") or not labels
            or [str(v) for v in summary.get("member_labels") or []] != labels
            or not isinstance(fps, list) or len(fps) != len(labels)
            or identity.get("records_fp") != manifest.get("records_fp")):
        return None
    return dict(zip(labels, fps, strict=True))


def _migrate_test_bucket(starless: bool) -> None:
    """Rename the regime's positional test bucket to the label keying,
    adopting the fingerprints its evaluation proves (else its members are
    re-inferred by the next fill)."""
    cubes_dir = _ensemble_cubes_dir(starless=starless)
    manifest = read_bucket_manifest(cubes_dir)
    if manifest is not None and not is_label_keyed(manifest):
        migrate_positional_bucket(cubes_dir,
                                  adopt=_proven_test_fingerprints(starless, manifest))


def _run_missing_members(cubes_dir: str, runner, lr: np.ndarray, rec: int,
                         missing: list[str]) -> dict[str, np.ndarray]:
    """Run ``missing`` members on one field's LR and store their cubes."""
    if not missing:
        return {}
    ran = dict(zip(missing, np.asarray(runner(lr, missing), np.float32), strict=True))
    for label, cube in ran.items():
        np.save(member_cube_path(cubes_dir, label, rec), cube)
    return ran


def _member_stack_of(cubes_dir: str, labels: list[str], rec: int,
                     ran: dict[str, np.ndarray]) -> np.ndarray:
    """The ``(M, H, W, C)`` stack of ``labels`` for one field: the members
    just run, the rest from their cached cubes."""
    return np.stack([ran[label] if label in ran
                     else np.load(member_cube_path(cubes_dir, label, rec))
                     for label in labels]).astype(np.float32)


def _fill_validate_cubes(cap, cubes_dir: str, *, base: str, starless: bool,
                         records_dir: str, records: dict, labels: list[str],
                         num_images: int) -> list[int]:
    """Bring the validate bucket to ``labels`` at their current checkpoints,
    inferring only what it lacks, and return its fields: those it already
    holds plus the first ``num_images`` records. A field whose member stack
    changed (a member inferred or dropped — its ``sr_`` is gone) gets its
    aggregates recomputed from the stack; nothing is read or run when the
    bucket is already current."""
    fingerprints = member_fingerprints(base, labels)
    sync = _open_member_bucket(cubes_dir, records=records, labels=labels,
                               fingerprints=fingerprints)
    held = {int(i) for i in sync.manifest.get("indices", []) or []}
    if len(held) >= int(num_images) and not any(
            missing_member_cubes(cubes_dir, labels, rec)
            or not os.path.isfile(_field_mean_path(cubes_dir, rec)) for rec in held):
        return sorted(held)
    runner = LazyMemberRunner(base, starless=starless, labels=labels)
    limit = max(int(num_images), (max(held) + 1) if held else 0)
    kept: list[int] = []
    subset = str(records["subset"])
    for position, image in enumerate(ImageSet.read(
            tfrecord_path(records_dir, f"dirty_{subset}"), num_images=limit)):
        rec = _record_index(image)
        if rec not in held and position >= int(num_images):
            continue
        lr = np.asarray(image.data, np.float32)
        missing = missing_member_cubes(cubes_dir, labels, rec)
        if missing:
            _drop_field_means(cubes_dir, rec)
        ran = _run_missing_members(cubes_dir, runner, lr, rec, missing)
        if not os.path.isfile(_field_mean_path(cubes_dir, rec)):
            preds = _member_stack_of(cubes_dir, labels, rec, ran)
            _cache_field_cubes(cubes_dir, rec, preds, preds.mean(0), preds.std(0), lr=lr)
        kept.append(rec)
        cap.tick(position + 1, limit, f"{subset} field {rec}: "
                 + (f"{len(missing)} member(s) inferred" if missing else "cached"))
    write_bucket_manifest(cubes_dir, {**sync.manifest, "indices": sorted(kept),
                                      "pca_n": ENSEMBLE_PCA_COMPONENTS})
    prune_bucket_fields(cubes_dir, kept)
    return sorted(kept)


def _collect_bounded_ablation_patches(
    comb, *, indices: list[int], labels: list[str], val_dir: str,
    records_dir: str, target: str, max_fields: int = 4,
    max_patches: int = 64, patch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect one compact validation patch per occupied VIS RBF region.

    Candidate centres combine bright truth pixels, high member-disagreement
    pixels, and a coarse spatial grid. Only one full cached member field is
    resident at a time; returned storage is hard-capped at roughly 24 MB for a
    20-member ensemble.
    """

    wanted = sorted(int(i) for i in indices)[:max(1, int(max_fields))]
    if not wanted:
        return (np.empty((0, len(labels), patch_size, patch_size, 4), np.float32),
                np.empty((0, patch_size, patch_size, 4), np.float32),
                np.empty((0, 4), np.int32))
    target_iter = iter(ImageSet.read(
        tfrecord_path(records_dir, f"{target}_validate"),
        num_images=max(wanted) + 1))
    current = next(target_iter, None)
    selected: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    half = int(patch_size) // 2
    vis_scale = float(Config.get_band("VIS").asinh_stretch_scale_e)

    for idx in wanted:
        while current is not None and _record_index(current) < idx:
            current = next(target_iter, None)
        record = current if current is not None and _record_index(current) == idx else None
        stack = load_cached_member_stack(
            idx, subset="validate", cubes_dir=val_dir, active=labels)
        if record is None or stack is None:
            continue
        stack = np.asarray(stack, np.float32)
        truth = np.asarray(record.data, np.float32)
        _m, height, width, _c = stack.shape
        if height < patch_size or width < patch_size:
            continue
        vis = np.arcsinh(stack[..., 0] / vis_scale)
        truth_vis = np.abs(np.arcsinh(truth[..., 0] / vis_scale)).reshape(-1)
        disagreement = (np.max(vis, axis=0) - np.min(vis, axis=0)).reshape(-1)
        budget = min(1024, height * width)
        bright = np.argpartition(truth_vis, -budget)[-budget:]
        disagree = np.argpartition(disagreement, -budget)[-budget:]
        gy, gx = np.mgrid[half:height - half:8, half:width - half:8]
        grid = (gy.reshape(-1) * width + gx.reshape(-1)).astype(np.int64)
        ordered = np.concatenate((bright[::-1], disagree[::-1], grid))
        seen_positions: set[int] = set()
        valid: list[int] = []
        for flat in ordered:
            flat = int(flat)
            if flat in seen_positions:
                continue
            seen_positions.add(flat)
            y, x = divmod(flat, width)
            if half <= y < height - half and half <= x < width - half:
                valid.append(flat)
        if not valid:
            continue
        ys = np.asarray(valid, np.int64) // width
        xs = np.asarray(valid, np.int64) % width
        X = vis[:, ys, xs].T
        regions = combiner_region_ids(comb, X, band="VIS")
        for pos, region in zip(valid, regions, strict=True):
            region = int(region)
            if region in selected:
                continue
            y, x = divmod(int(pos), width)
            selected[region] = (
                stack[:, y - half:y + half, x - half:x + half, :].copy(),
                truth[y - half:y + half, x - half:x + half, :].copy(),
            )
            if len(selected) >= int(max_patches):
                break
        del vis, disagreement, truth_vis, stack, truth
        if len(selected) >= int(max_patches):
            break

    vis_regions = np.asarray(sorted(selected), np.int32)
    if not len(vis_regions):
        return (np.empty((0, len(labels), patch_size, patch_size, 4), np.float32),
                np.empty((0, patch_size, patch_size, 4), np.float32),
                np.empty((0, 4), np.int32))
    patch_stacks = np.stack([selected[int(r)][0] for r in vis_regions]).astype(np.float32)
    patch_truth = np.stack([selected[int(r)][1] for r in vis_regions]).astype(np.float32)
    centres = patch_stacks[:, :, half, half, :]
    region_columns = []
    for ci, name in enumerate(comb.band_names):
        scale = float(Config.get_band(name).asinh_stretch_scale_e)
        X = np.arcsinh(centres[..., ci] / scale)
        region_columns.append(combiner_region_ids(comb, X, band=name))
    regions = np.stack(region_columns, axis=1).astype(np.int32)
    return patch_stacks, patch_truth, regions


def _prepare_validate_cubes(cap, *, starless: bool, num_images: int,
                            target_fwhm: float):
    """The regime's cached validate member cubes, brought up to the active
    members: only members whose cubes are missing or were made by another
    checkpoint are inferred, departed members are dropped, and a change of the
    records or the target PSF re-infers everyone (:func:`_fill_validate_cubes`).
    → ``(base, records_dir, records_fp, validate_dir, indices, labels,
    target)`` — shared by the production combiner fit and the named
    spatial-gate variant fit."""
    base = ensemble_dir()
    records_dir = _sky_records_local_dir()
    target = "clean" if starless else "hr"
    if not records_dir:
        raise RuntimeError(_NO_SKY_RECORDS)
    if not _validate_records_present(records_dir, starless=starless):
        raise RuntimeError(
            "validate records not synced — sync the validate split on Synthetic › Records "
            f"(Sync from FASRC) so dirty_validate + {target}_validate are local.")

    records_fp = _eval_records_fingerprint(
        records_dir, "validate", starless=starless)
    validate_dir = _ensemble_cubes_dir("validate", starless=starless)
    labels = _regime_labels(base, starless)
    if not labels:
        raise RuntimeError(f"no active {_regime_slug(starless)} members with a checkpoint "
                           f"under {base}")
    indices = _fill_validate_cubes(
        cap, validate_dir, base=base, starless=starless, records_dir=records_dir,
        records={"subset": "validate", "records_fp": records_fp,
                 "target_psf_fwhm_arcsec": float(target_fwhm)},
        labels=labels, num_images=int(num_images))
    if not indices:
        raise RuntimeError("no validate fields collected — check the records.")
    return base, records_dir, records_fp, validate_dir, indices, labels, target


def job_combiner_fit(cap, *, num_images: int, n_kernels: int = 128,
                     min_usage: float | None = None,
                     starless: bool = False,
                     model_kind: str = _RAW_INCREMENTAL_MINMEANMAX_RBF_KIND,
                     score_test: bool = True,
                     gate_members: list[str] | None = None,
                     target_fwhm_arcsec: float = Config.TARGET_PSF_FWHM_ARCSEC) -> dict:
    """Fit a combiner on the validate cubes, then optionally test-score.

    ``gate_members`` (member numbers, spatial gate only) fits a pruned gate
    that reads just those members."""
    del min_usage

    model_kind = _normalize_combiner_kind(model_kind)
    target_fwhm = validate_target_fwhm_arcsec(target_fwhm_arcsec)
    (base, records_dir, records_fp, validate_dir, indices, labels,
     target) = _prepare_validate_cubes(cap, starless=starless,
                                       num_images=num_images,
                                       target_fwhm=target_fwhm)

    if model_kind == SPATIAL_GATE_KIND:
        combiner = _fit_spatial_gate_on_validate(
            cap, base=base, labels=labels, indices=indices,
            records_dir=records_dir, records_fp=records_fp,
            validate_dir=validate_dir, starless=starless, target=target,
            target_fwhm=target_fwhm, gate_members=gate_members)
        combiner.fit_meta.update({"subset": "validate",
                                  "num_images": int(num_images)})
        return _save_and_score_combiner(
            cap, combiner, starless=starless, model_kind=model_kind,
            score_test=score_test, n_members=len(labels))

    fingerprints = member_fingerprints(base, labels)

    def validation_fields(requested_indices):
        wanted = sorted({int(value) for value in requested_indices})
        if not wanted:
            return
        target_iter = iter(ImageSet.read(
            tfrecord_path(records_dir, f"{target}_validate"),
            num_images=max(wanted) + 1))
        current_target = next(target_iter, None)
        for index in wanted:
            while (current_target is not None
                   and _record_index(current_target) < int(index)):
                current_target = next(target_iter, None)
            target_record = (
                current_target
                if current_target is not None
                and _record_index(current_target) == int(index)
                else None)
            stack = load_cached_member_stack(
                index, subset="validate", cubes_dir=validate_dir, active=labels,
                fingerprints=fingerprints)
            if stack is not None and target_record is not None:
                target_record = dataclasses.replace(
                    target_record,
                    data=blur_target_array(
                        target_record.data, target_fwhm,
                        pixel_scale_arcsec=target_record.pixel_scale_arcsec,
                    ),
                )
                yield (
                    int(index),
                    np.asarray(stack, np.float32),
                    np.asarray(target_record.data, np.float32),
                )

    if int(n_kernels) <= 0:
        n_kernels = combiner_model_spec().default_kernels
    member_meta = _member_meta_from_labels(labels)
    member_psnr = np.asarray(
        [row.get("psnr", np.nan) for row in member_meta], np.float64)
    if member_psnr.shape != (len(labels),) or not np.any(np.isfinite(member_psnr)):
        member_psnr = None
    combiner = fit_combiner_minibatched(
        validation_fields, indices, labels, band_names=BAND_NAMES,
        n_kernels=int(n_kernels),
        model_kind=model_kind,
        member_validation_psnr=member_psnr,
        progress=lambda current, total, label: cap.tick(current, total, label))
    combiner.records_fp = records_fp
    combiner.starfull = not bool(starless)
    combiner.fit_meta.update({
        "subset": "validate",
        "num_images": int(num_images),
        "model_kind": combiner.kind,
        "pruning": "not_applicable",
        "features": "all member asinh inferences",
        "pixel_source": "all pixels from disjoint validation fields",
    })
    return _save_and_score_combiner(
        cap, combiner, starless=starless, model_kind=model_kind,
        score_test=score_test, n_members=len(labels))


def _fit_spatial_gate_on_validate(cap, *, base: str, labels: list[str],
                                  indices: list[int], records_dir: str,
                                  records_fp, validate_dir: str, starless: bool,
                                  target: str, target_fwhm: float,
                                  gate_members: list[str] | None = None):
    """Fit the spatial gate on the cached validate member cubes, plus
    blackout-augmented copies of the training fields (one cached extra
    member-inference pass, reused on later fits)."""
    fields, _labels = load_cube_fields(
        validate_dir, records_dir, "validate", target_name=target,
        target_fwhm_arcsec=target_fwhm, indices=indices,
        progress=lambda i, n, label: cap.tick(i, n, label))
    if len(fields) < 2:
        raise RuntimeError("the spatial gate needs at least two validate fields")
    active = None
    if gate_members:
        wanted = {str(v) for v in gate_members}
        active = [i for i, label in enumerate(labels)
                  if str(label).split("·")[0] in wanted]
        unknown = wanted - {str(labels[i]).split("·")[0] for i in active}
        if unknown:
            raise RuntimeError(f"not active {_regime_slug(starless)} members: "
                               f"{', '.join(sorted(unknown))}")
    train, holdout = split_holdout(
        fields, max(1, round(SPATIAL_GATE_HOLDOUT_FRACTION * len(fields))), seed=0)
    blackout = build_blackout_fields(
        train, labels, LazyMemberRunner(base, starless=starless, labels=labels),
        _ensemble_cubes_dir("validate_blackout", starless=starless),
        max_fields=SPATIAL_GATE_BLACKOUT_FIELDS, seed=0,
        source_fingerprint=str(records_fp),
        progress=lambda i, n, label: cap.tick(i, n, label))
    combiner = fit_spatial_gate(
        train + blackout, holdout, labels, active_members=active,
        progress=lambda i, n, label: cap.tick(i, n, label),
        log=lambda message: print(f"[spatial gate] {message}", flush=True))
    combiner.records_fp = records_fp
    combiner.starfull = not bool(starless)
    combiner.fit_meta["blackout_fields"] = len(blackout)
    return combiner


def _save_and_score_combiner(cap, combiner, *, starless: bool, model_kind: str,
                             score_test: bool, n_members: int) -> dict:
    """Persist a freshly fitted combiner, refresh its payload and (optionally)
    score it on the cached test cubes without re-running the members."""
    regime_dir = _ensemble_regime_dir(starless)
    save_combiner(
        combiner, regime_dir, artifact_dir=_combiner_artifact_dir(model_kind))
    compute_combiner_payload(starless, model_kind=model_kind)

    test_summary = None
    if score_test:
        cap.tick(0, 1, "scoring combiner on the test set")
        if _apply_combiner_to_test_cubes(
                starless, model_kind, progress=cap.tick):
            previous = (_read_eval_summary(starless) or {}).get("eval_identity") or {}
            test_summary = _reevaluate_from_cached_cubes(
                starless, num_images=previous.get("num_images"),
                progress=cap.tick)
    cap.tick(1, 1, "done")
    result = {
        "n_members": int(n_members),
        "n_kernels": int(combiner.n_kernels),
        "model_kind": combiner.kind,
        "fitted_models": [combiner.kind],
        "val_l1": combiner.val_l1,
        "subset": "validate",
        "regime": _regime_slug(starless),
        "test_scored": test_summary is not None,
        "training_mode": "all_validation_pixels_minibatch",
    }
    if test_summary is not None:
        prefix = f"{model_kind}_combiner"
        result["combiner_psnr"] = test_summary.get(f"{prefix}_psnr")
        result["combiner_vs_mean_db"] = test_summary.get(
            f"{prefix}_vs_mean_db")
    return result


def _shared_pca_weight_diagnostic(comb, *, starless: bool,
                                  max_rows: int = 100_000,
                                  per_field: int = 4096) -> dict:
    """PC1 x PC2 gate surfaces from cached validation member pixels."""

    val_dir = _ensemble_cubes_dir("validate", starless=starless)
    try:
        with open(os.path.join(val_dir, "viz_index.json")) as handle:
            manifest = json.load(handle)
    except (OSError, ValueError):
        return {"available": False,
                "reason": "no matching validation cube cache for PCA"}
    if (list(manifest.get("member_labels") or []) != list(comb.member_labels)
            or (comb.records_fp is not None
                and manifest.get("records_fp") != comb.records_fp)):
        return {"available": False,
                "reason": "validation cube cache does not match this fit"}

    rng = np.random.default_rng(271828)
    rows: list[np.ndarray] = []
    used = 0
    field_indices = sorted(
        int(i) for i in manifest.get("indices", []) or [])
    field_budget = min(
        int(per_field),
        max(1, int(np.ceil(max_rows / max(1, len(field_indices))))),
    )
    for rec in field_indices:
        if used >= int(max_rows):
            break
        stack = load_cached_member_stack(
            rec, subset="validate", cubes_dir=val_dir,
            active=list(comb.member_labels), require_current=False)
        if stack is None:
            continue
        members, height, width, channels = stack.shape
        n_pixels = height * width
        take = min(field_budget, n_pixels, int(max_rows) - used)
        pick = (np.arange(n_pixels) if take == n_pixels
                else rng.choice(n_pixels, size=take, replace=False))
        rows.append(stack.reshape(members, n_pixels, channels)[:, pick, :]
                    .transpose(1, 0, 2).astype(np.float32, copy=False))
        used += take
    if not rows:
        return {"available": False,
                "reason": "validation member cubes contain no PCA pixels"}
    pixels = np.concatenate(rows, axis=0)
    surface = comb.pca_weight_surface(pixels)
    if not surface.get("available"):
        return {"available": False,
                "reason": "too few validation pixels for a two-axis PCA"}
    common = {
        "available": True,
        "schema": _PCA_WEIGHT_SURFACE_SCHEMA,
        "n_pixels": int(surface.get("n_pixels", 0)),
        "n_fields": len(manifest.get("indices", []) or []),
        "feature_space": surface.get("feature_space"),
        "conditioning_note": surface.get("conditioning_note"),
        "projection_method": surface.get("projection_method"),
        "integration_neighbors": surface.get("integration_neighbors"),
        "records_fp": manifest.get("records_fp"),
        # FeatureGrid-compatible names let the existing interactive 3-D renderer
        # draw this diagnostic with exactly the same camera and weight scale.
        "mean_asinh": _jsonable(surface["pc1"]),
        "std_asinh": _jsonable(surface["pc2"]),
        "std_log": _jsonable(surface["pc2"]),
        "center_mean_asinh": _jsonable(surface.get("center_pc1", [])),
        "center_std_asinh": _jsonable(surface.get("center_pc2", [])),
        "center_std_log": _jsonable(surface.get("center_pc2", [])),
        "x_label": "PC1",
        "y_label": "PC2",
        "y_is_log": False,
        "z_label": surface.get("z_label", "relative weight [0-1]"),
        "surface_labels": [str(label) for label in
                           surface.get("surface_labels", [])],
        "explained_variance_ratio": _jsonable(
            surface["explained_variance_ratio"]),
        "feature_names": [str(name) for name in surface["feature_names"]],
        "loadings": np.round(np.asarray(surface["loadings"], float), 6).tolist(),
        "projected_density": _jsonable(surface.get("projected_density", [])),
        "integrated_weights": _jsonable(surface.get("integrated_weights", [])),
        "peak_weights": _jsonable(surface.get("peak_weights", [])),
    }
    weights = np.asarray(surface["weights"], float)
    common["weights"] = np.round(weights, 6).tolist()
    return common


def compute_combiner_payload(starless: bool,
                             model_kind: str | None = None) -> dict | None:
    """Serialize the incremental combiner's fit and PCA diagnostics."""
    model_kind = _normalize_combiner_kind(model_kind)
    combiner = load_combiner(
        _ensemble_regime_dir(starless),
        artifact_dir=_combiner_artifact_dir(model_kind))
    if combiner is None:
        return None
    if isinstance(combiner, SpatialGateCombiner):
        return _spatial_gate_payload(combiner, starless=starless)
    payload_path = _combiner_payload_path(starless, model_kind)
    previous = None
    with contextlib.suppress(OSError, ValueError), open(payload_path) as handle:
        previous = json.load(handle)
    artifact_fp = combiner_artifact_fingerprint(
        _ensemble_regime_dir(starless), _combiner_artifact_dir(model_kind))
    cached = (previous or {}).get("pca_weight_surface") or {}
    if (cached.get("schema") == _PCA_WEIGHT_SURFACE_SCHEMA
            and cached.get("artifact_fp") == artifact_fp
            and cached.get("records_fp") == combiner.records_fp):
        surface = cached
    else:
        surface = _shared_pca_weight_diagnostic(
            combiner, starless=starless)
        surface["artifact_fp"] = artifact_fp
    weight_labels = list(combiner.weight_labels)
    integrated_weights = np.asarray(
        surface.get("integrated_weights", []), np.float64)
    peak_weights = np.asarray(surface.get("peak_weights", []), np.float64)
    if (integrated_weights.shape == (len(weight_labels),)
            and peak_weights.shape == (len(weight_labels),)):
        shared_peaks = peak_weights.tolist()
        shared_integrals = integrated_weights.tolist()
    else:
        shared_peaks = []
        shared_integrals = []
    member_weight_peaks = dict.fromkeys(combiner.band_names, shared_peaks)
    member_weight_integrals = dict.fromkeys(
        combiner.band_names, shared_integrals)
    member_meta = _member_meta_from_labels(weight_labels)
    payload = {
        "available": True,
        "stale": list(combiner.member_labels) != _regime_labels(
            ensemble_dir(), starless),
        "kind": combiner.kind,
        "regime": _regime_slug(starless),
        "member_labels": weight_labels,
        "source_member_labels": list(combiner.member_labels),
        "members": [
            {"label": str(label), "role": "source_member", **member_meta[index]}
            for index, label in enumerate(weight_labels)
        ],
        "n_kernels": int(combiner.n_kernels),
        "min_usage": 0.0,
        "val_l1": combiner.val_l1,
        "band_names": list(combiner.band_names),
        "eff_weights": {},
        "member_weight_peaks": member_weight_peaks,
        "member_weight_integrals": member_weight_integrals,
        "surviving": combiner.surviving_members(),
        "feature_grid": {},
        "pca_weight_surface": surface,
        "hr_weights": {"available": False, "bands": {},
                       "member_labels": [], "n_fields": 0, "n_pixels": 0},
        "fit_meta": combiner.fit_meta,
    }
    _atomic_json(payload_path, payload)
    return payload


_GATE_DIAGNOSTIC_SCHEMA = 1
_GATE_DIAGNOSTIC_FIELDS = 8
_GATE_BRIGHTNESS_EDGES = (0.02, 0.1, 0.5, 2.0)
_GATE_BRIGHTNESS_NAMES = ("sky", "faint", "mid", "bright", "core")


def _spatial_gate_weight_diagnostic(comb: SpatialGateCombiner, *, starless: bool,
                                    max_fields: int = _GATE_DIAGNOSTIC_FIELDS) -> dict:
    """How much weight the gate gives each member, per band: over all pixels,
    over source pixels, and by brightness (the member-mean asinh level), from
    the gate's held-out validate fields."""
    val_dir = _ensemble_cubes_dir("validate", starless=starless)
    try:
        with open(os.path.join(val_dir, "viz_index.json")) as handle:
            manifest = json.load(handle)
    except (OSError, ValueError):
        return {"available": False, "reason": "no validation cube cache"}
    cube_labels = {str(v) for v in manifest.get("member_labels") or []}
    if not reads_available(comb.read_labels, cube_labels):
        return {"available": False,
                "reason": "validation cube cache lacks members this gate reads"}
    # The brightness level is the mean of every fitted member with a cube
    # (all of them unless unread members were archived since the fit); the
    # weights read only the gate's members and come back full width.
    level_labels = [str(v) for v in comb.member_labels if str(v) in cube_labels]
    read_rows = [level_labels.index(str(v)) for v in comb.read_labels]
    cached = {int(i) for i in manifest.get("indices", []) or []}
    preferred = [int(i) for i in comb.fit_meta.get("holdout_fields", []) or []]
    fields = [i for i in preferred if i in cached] or sorted(cached)
    records_dir = _sky_records_local_dir()
    scales = band_scales(comb.band_names).astype(np.float32)
    n_members, n_bands = len(comb.member_labels), len(comb.band_names)
    n_bins = len(_GATE_BRIGHTNESS_NAMES)
    usage = np.zeros((n_members, n_bands))
    usage_source = np.zeros((n_members, n_bands))
    by_bin = np.zeros((n_bands, n_bins, n_members))
    bin_pixels = np.zeros((n_bands, n_bins))
    n_pixels = n_source = n_fields = 0
    for rec in fields[:int(max_fields)]:
        stack = load_cached_member_stack(rec, subset="validate", cubes_dir=val_dir,
                                         active=level_labels, require_current=False)
        lr = (load_cached_field_lr(val_dir, rec, records_dir=records_dir,
                                   subset="validate") if comb.use_lr else None)
        if stack is None or (comb.use_lr and lr is None):
            continue
        weights = comb.weights_field(stack[read_rows], lr=lr)  # (H, W, M, C)
        level = np.arcsinh(stack.mean(axis=0) / scales)       # (H, W, C)
        usage += weights.sum(axis=(0, 1))
        n_pixels += level.shape[0] * level.shape[1]
        source = level[..., 0] > _GATE_BRIGHTNESS_EDGES[1]
        usage_source += weights[source].sum(axis=0)
        n_source += int(source.sum())
        for c in range(n_bands):
            bins = np.digitize(level[..., c], _GATE_BRIGHTNESS_EDGES)
            for b in range(n_bins):
                hit = bins == b
                by_bin[c, b] += weights[..., c][hit].sum(axis=0)
                bin_pixels[c, b] += hit.sum()
        n_fields += 1
    if not n_fields:
        return {"available": False,
                "reason": "no validate member cubes available for the gate"}
    names = list(comb.band_names)
    mean_by_bin = by_bin / np.maximum(bin_pixels[..., None], 1.0)
    return {
        "available": True,
        "schema": _GATE_DIAGNOSTIC_SCHEMA,
        "n_fields": int(n_fields),
        "n_pixels": int(n_pixels),
        "brightness_edges_asinh": list(_GATE_BRIGHTNESS_EDGES),
        "brightness_names": list(_GATE_BRIGHTNESS_NAMES),
        "usage": {band: (usage[:, c] / max(n_pixels, 1)).tolist()
                  for c, band in enumerate(names)},
        "usage_source": {band: (usage_source[:, c] / max(n_source, 1)).tolist()
                         for c, band in enumerate(names)},
        "usage_by_brightness": {band: mean_by_bin[c].tolist()
                                for c, band in enumerate(names)},
        "brightness_pixels": {band: bin_pixels[c].astype(int).tolist()
                              for c, band in enumerate(names)},
    }


def _spatial_gate_payload(comb: SpatialGateCombiner, *, starless: bool) -> dict:
    """The combiner card's dataset for the spatial gate (weight diagnostics
    are cached per fitted artifact, so page loads stay cheap)."""
    regime_dir = _ensemble_regime_dir(starless)
    payload_path = _combiner_payload_path(starless, SPATIAL_GATE_KIND)
    artifact_fp = combiner_artifact_fingerprint(
        regime_dir, _combiner_artifact_dir(SPATIAL_GATE_KIND))
    previous = None
    with contextlib.suppress(OSError, ValueError), open(payload_path) as handle:
        previous = json.load(handle)
    diagnostic = (previous or {}).get("gate_diagnostics") or {}
    if (diagnostic.get("schema") != _GATE_DIAGNOSTIC_SCHEMA
            or diagnostic.get("artifact_fp") != artifact_fp):
        diagnostic = _spatial_gate_weight_diagnostic(comb, starless=starless)
        diagnostic["artifact_fp"] = artifact_fp
    labels = list(comb.member_labels)
    member_meta = _member_meta_from_labels(labels)
    usage = diagnostic.get("usage") or {}
    active = _regime_labels(ensemble_dir(), starless)
    try:
        peaks: list[float] | None = member_peak_weights(diagnostic, len(labels))
    except ValueError:
        peaks = None
    payload = {
        "available": True,
        # Valid while every member the gate READS is active (joined members
        # are a note, unread members may leave or be retrained).
        "stale": not reads_available(comb.read_labels, active),
        "joined_after_fit": joined_after_fit(labels, active),
        "read_labels": comb.read_labels,
        # The "used by the gate" rule (eval/gate_members.py): peak weight =
        # max over bands of the all-pixel, source and brightness-bin means.
        "member_peak_weights": peaks,
        "used_threshold": DEFAULT_USED_THRESHOLD,
        "used_by_gate": (None if peaks is None
                         else [p >= DEFAULT_USED_THRESHOLD for p in peaks]),
        "kind": comb.kind,
        "regime": _regime_slug(starless),
        "member_labels": labels,
        "source_member_labels": labels,
        "members": [{"label": str(label), "role": "source_member", **member_meta[i]}
                    for i, label in enumerate(labels)],
        "n_kernels": int(comb.n_kernels),
        "n_parameters": int(comb.n_kernels),
        "use_lr": bool(comb.use_lr),
        "min_usage": 0.0,
        "val_l1": comb.val_l1,
        "band_names": list(comb.band_names),
        "eff_weights": {},
        "member_weight_peaks": {},
        "member_weight_integrals": usage,
        "surviving": comb.surviving_members(),
        "feature_grid": {},
        "pca_weight_surface": {
            "available": False,
            "reason": ("the spatial gate weighs members from each pixel's "
                       "neighbourhood, not from per-pixel member values alone")},
        "hr_weights": {"available": False, "bands": {},
                       "member_labels": [], "n_fields": 0, "n_pixels": 0},
        "gate_diagnostics": diagnostic,
        "fit_meta": comb.fit_meta,
    }
    _atomic_json(payload_path, payload)
    return payload


# ---------------------------------------------------------------------------
# Evaluation: payload files, pixel back-tracing, the per-band and PSNR-vs-knee
# diagnostics, archive reconciliation and the test-set evaluation job; then
# member archive and the FASRC pull.
# ---------------------------------------------------------------------------



def _atomic_json(path: str, value: dict) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(value, f, indent=2)
    os.replace(tmp, path)


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
    up = zoom(a, (n / a.shape[0], n / a.shape[1], 1), order=3)  # bicubic
    return up[:n, :n]


def pixel_trace(starless: bool, diag: str, i: int, j: int,
                model_kind: str | None = None,
                axis_mode: str | None = None,
                *, band: str = "VIS",
                half: int = PIXEL_TRACE_HALF,
                max_stamps: int = PIXEL_TRACE_STAMPS) -> dict:
    """Back-trace one heatmap cell to real image stamps.

    Given a diagnostic (``"std_err"`` | ``"bright_std"`` |
    ``"combiner_feature_error"``, the last only in an older sidecar) and its
    histogram cell ``(i, j)``, read the sidecar's example pixel locations for
    that cell and cut a ``(2·half+1)²`` window around each — the real pixels
    that landed in the clicked cell. Each stamp carries the **full N-band**
    LR, HR and SR cubes (SR = the baked cube of the combiner ``model_kind``
    names, else the ensemble mean; a field without that combiner's cube is
    skipped, never substituted) as
    base64 float32 so the frontend can render them with the field viewer's exact
    colour / knee / brightness, plus the single-band cross-member σ, and the
    per-pixel numbers of ``band`` (the diagnostics' band, VIS by default)
    that place the pixel in the plot. Windows are zero-padded to a fixed
    ``(2·half+1)²`` with the sampled pixel at the centre.
    Returns ``{stamps: [...], ...}`` (``stamps`` empty when nothing sampled)."""
    band_names = list(Config.LR_INPUT_BAND_NAMES)
    if band not in band_names:
        raise ValueError(f"unknown band {band!r}")
    band_index = band_names.index(band)
    S = 2 * int(half) + 1
    selected_kind = str(model_kind or "")
    selected_axis = str(axis_mode or "")
    out = {"diag": diag, "model_kind": selected_kind or None,
           "axis_mode": selected_axis or None,
           "i": int(i), "j": int(j), "half": int(half),
           "size": S, "bands": list(Config.LR_INPUT_BAND_NAMES), "band": band,
           "stretch": float(Config.STRETCH_SCALE_E), "stamps": []}
    try:
        with open(diag_samples_path(starless, band)) as f:
            side = json.load(f)
    except (OSError, json.JSONDecodeError):
        return out
    if diag == "std_err" and selected_kind:
        cells = (side.get("std_err_models") or {}).get(selected_kind) or {}
    elif diag == "combiner_feature_error":
        cells = ((side.get(diag) or {}).get(selected_axis) or {}).get(
            selected_kind) or {}
    else:
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
    manifest = {}
    with contextlib.suppress(OSError, json.JSONDecodeError), open(
        os.path.join(cubes_dir, "viz_index.json")) as handle:
        manifest = json.load(handle)
    target_fwhm = validate_target_fwhm_arcsec(
        manifest.get("target_psf_fwhm_arcsec", Config.TARGET_PSF_FWHM_ARCSEC))
    hr_by = {
        r.index: dataclasses.replace(
            r,
            data=blur_target_array(
                r.data, target_fwhm,
                pixel_scale_arcsec=r.pixel_scale_arcsec,
            ),
        )
        for r in read_images(hr_path, num_images=max_idx)
    }
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
        if selected_kind == "ensemble_mean" or not selected_kind:
            model_f = sr_f
        elif selected_kind in COMBINER_MODELS:
            model_f = os.path.join(
                cubes_dir,
                f"{COMBINER_MODELS[selected_kind].cube_prefix}_{rec:05d}.npy")
        else:
            model_f = sr_f
        hr_rec = hr_by.get(rec)
        if not (os.path.isfile(sr_f) and os.path.isfile(std_f)
                and hr_rec is not None):
            continue
        hr_cube = np.asarray(hr_rec.data, np.float32)           # (H, W, C)
        n = int(hr_cube.shape[0])
        # The trace must show the reconstruction whose error made the selected
        # plot cell.  Falling back to another combiner would make the provenance
        # look plausible while being scientifically wrong.
        use_comb = selected_kind in COMBINER_MODELS
        if not os.path.isfile(model_f):
            continue
        sr_cube = np.load(model_f).astype(np.float32)
        try:
            std_v = _plane(np.load(std_f), band_index)           # scalar σ (band)
            hr_v, sr_v = _plane(hr_cube, band_index), _plane(sr_cube, band_index)
        except IndexError:
            continue
        lr_rec = lr_by.get(rec)
        lr_cube = _lr_cube_on_hr_grid(
            np.asarray(lr_rec.data, np.float32), n) if lr_rec is not None else None
        for (y, x) in coords:
            if not (0 <= y < n and 0 <= x < n):
                continue
            hv, sv, dv = float(hr_v[y, x]), float(sr_v[y, x]), float(std_v[y, x])
            stamp = {
                "field": int(rec), "y": int(y), "x": int(x), "center": int(half),
                "sr_is_combiner": bool(use_comb),
                "model_kind": selected_kind or "ensemble_mean",
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
    model_cmet = {kind: _CombinerMetricAcc() for kind in _ORDINARY_COMBINER_KINDS}
    for hr_v, mean_v, mem_v, model_v, lr_v, rec in _iter_cached_fields(starless):
        if ps_acc is None:
            ps_acc = EnsembleSpectrumAccumulator(
                int(hr_v.shape[0]), float(Config.DEFAULT_PIXEL_SCALE))
        ps_acc.add(hr_v, mean_v, mem_v, model_combiners=model_v, lr=lr_v)
        diag.add(hr_v, mean_v, mem_v, combiners=model_v, field_index=rec)
        for kind, cmet in model_cmet.items():
            cmet.add(hr_v, mean_v, mem_v, model_v.get(kind))
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
    coherence = (ps_acc.coherence_scores() if ps_acc is not None
                 and float(ps_acc.bc.sum()) > 0 else None)
    payload = _evals_payload(curves, diag, man.get("member_labels", []),
                             man.get("subset", ""),
                             combiner=model_cmet[_RBF_KIND].block(man.get("member_labels", [])),
                             model_combiners={kind: cmet.block(man.get("member_labels", []))
                                              for kind, cmet in model_cmet.items()},
                             coherence=coherence)
    payload["regime"] = _regime_slug(starless)
    with open(_evals_payload_path(starless), "w") as f:
        json.dump(payload, f)
    return payload


def refresh_evaluation_diagnostics(starless: bool) -> dict | None:
    """Refresh only pixel diagnostics + trace samples from cached cubes.

    This is the schema-migration path for an otherwise current
    ``ensemble_evals.json``.  It deliberately preserves the cached spectrum,
    coherence and metric blocks, avoiding their substantially more expensive
    FFT pass when a new pixel-level plot is introduced.
    """
    path = _evals_payload_path(starless)
    try:
        with open(path) as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return compute_evaluation_payload(starless)

    diag = EnsembleDiagnosticsAccumulator()
    for hr_v, mean_v, mem_v, model_v, _lr_v, rec in _iter_cached_fields(starless):
        diag.add(hr_v, mean_v, mem_v, combiners=model_v, field_index=rec)
    if diag.n_fields == 0:
        return None
    _write_diag_samples(starless, diag)
    diagnostics = diag.to_payload()
    payload.update(diagnostics)
    payload["n_fields"] = int(diag.n_fields)
    payload["n_members"] = int(diag.n_members)
    payload["regime"] = _regime_slug(starless)
    with open(path, "w") as f:
        json.dump(payload, f)
    return payload


#: The bands measured beyond VIS; each has its own evaluation payload next to
#: ``ensemble_evals.json`` (same schema, one band).
EVAL_EXTRA_BANDS: tuple[str, ...] = tuple(Config.LR_INPUT_BAND_NAMES[1:])
#: Bumped when the per-band payload changes shape → the next refresh rebuilds.
_BAND_EVALS_SCHEMA = 1


def _band_evals_path(starless: bool, band: str) -> str:
    return os.path.join(_ensemble_regime_dir(starless), f"ensemble_evals_{band}.json")


def _band_diag_samples_path(starless: bool, band: str) -> str:
    return os.path.join(_ensemble_regime_dir(starless),
                        f"ensemble_diag_samples_{band}.json")


def diag_samples_path(starless: bool, band: str = "VIS") -> str:
    """The back-tracing sidecar of one band's diagnostics."""
    return (_diag_samples_path(starless) if band == "VIS"
            else _band_diag_samples_path(starless, band))


def _band_evals_identity(starless: bool) -> dict | None:
    """What a per-band payload depends on: the scored fields, the members and
    every baked combiner (as the PSNR-vs-knee curves). ``None`` without a
    cached evaluation."""
    manifest = _read_test_manifest(starless)
    if manifest is None:
        return None
    identity = {key: value for key, value in _knee_psnr_identity(starless, manifest).items()
                if key not in ("schema", "knees")}
    return {"schema": _BAND_EVALS_SCHEMA, **identity}


def band_evals_state(starless: bool) -> dict[str, str]:
    """``{band: "current" | "stale" | "missing"}`` for the bands beyond VIS."""
    identity = _band_evals_identity(starless)
    out: dict[str, str] = {}
    for band in EVAL_EXTRA_BANDS:
        try:
            with open(_band_evals_path(starless, band)) as handle:
                stored = json.load(handle).get("identity")
        except (OSError, ValueError, AttributeError):
            out[band] = "missing"
            continue
        out[band] = "current" if identity is not None and stored == identity else "stale"
    return out


def read_band_evals(starless: bool, band: str) -> dict | None:
    """One band's cached evaluation payload with ``stale`` set (its identity
    differs from the current evaluation), or ``None`` when it was never
    computed. Cache only: nothing is recomputed."""
    if band not in EVAL_EXTRA_BANDS:
        raise ValueError(f"unknown band {band!r}; the extra bands are {', '.join(EVAL_EXTRA_BANDS)}")
    try:
        with open(_band_evals_path(starless, band)) as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return None
    payload["stale"] = payload.get("identity") != _band_evals_identity(starless)
    return payload


def compute_band_evaluation_payloads(
        starless: bool, *, bands: tuple[str, ...] | None = None,
        progress: Callable[[int, int, str], None] | None = None) -> dict[str, dict] | None:
    """The evaluation diagnostics of the bands beyond VIS, from the CACHED
    cubes (no model inference): one sweep reads each field once and feeds a
    spectrum and a pixel-diagnostics accumulator per band. Writes
    ``ensemble_evals_<band>.json`` and its back-tracing sidecar per band and
    returns ``{band: payload}``, or ``None`` when nothing valid is cached.
    The member-pair spectra are left out (the VIS payload keeps them)."""
    names = list(Config.LR_INPUT_BAND_NAMES)
    wanted = tuple(names.index(b) for b in (bands or EVAL_EXTRA_BANDS) if b in names and b != "VIS")
    manifest = _read_test_manifest(starless)
    identity = _band_evals_identity(starless)
    if not wanted or manifest is None or identity is None:
        return None
    labels = [str(x) for x in manifest.get("member_labels", []) or []]
    total = len(manifest.get("indices", []) or [])
    spectra: dict[int, EnsembleSpectrumAccumulator] = {}
    diags = {band: EnsembleDiagnosticsAccumulator() for band in wanted}
    for position, (rec, planes) in enumerate(_iter_cached_field_bands(starless, wanted), 1):
        for band, (hr_v, mean_v, mem_v, model_v, lr_v) in planes.items():
            spectrum = spectra.get(band)
            if spectrum is None:
                spectrum = spectra[band] = EnsembleSpectrumAccumulator(
                    int(hr_v.shape[0]), float(Config.DEFAULT_PIXEL_SCALE),
                    collect_pairwise=False)
            spectrum.add(hr_v, mean_v, mem_v, model_combiners=model_v, lr=lr_v)
            diags[band].add(hr_v, mean_v, mem_v, combiners=model_v, field_index=rec)
        if progress is not None:
            progress(position, total, f"Y, J, H diagnostics: test field {rec}")
    out: dict[str, dict] = {}
    for band in wanted:
        diag = diags[band]
        if diag.n_fields == 0:
            continue
        spectrum = spectra.get(band)
        measured = spectrum is not None and float(spectrum.bc.sum()) > 0
        payload = _evals_payload(
            spectrum.curves() if measured else None, diag, labels,
            manifest.get("subset", ""),
            coherence=spectrum.coherence_scores() if measured else None,
            band=names[band])
        payload["regime"] = _regime_slug(starless)
        payload["identity"] = identity
        with open(_band_evals_path(starless, names[band]), "w") as handle:
            json.dump(payload, handle)
        try:
            with open(_band_diag_samples_path(starless, names[band]), "w") as handle:
                json.dump(diag.samples_payload(), handle)
        except OSError:
            pass
        out[names[band]] = payload
    return out or None


def _refresh_band_evals(starless: bool, progress) -> None:
    """Keep the Y/J/H diagnostics in step with the cubes; skipped while every
    band is current (best-effort: never fails the evaluation)."""
    try:
        state = band_evals_state(starless)
        if state and all(value == "current" for value in state.values()):
            return
        compute_band_evaluation_payloads(starless, progress=progress)
    except Exception as exc:  # noqa: BLE001 — diagnostic only
        print(f"[ensemble] Y/J/H diagnostics not refreshed: {exc}")


def _rebuild_bucket_dropping_member(cubes_dir: str, member_nn: str,
                                    *, keep_combiner: bool = False,
                                    keep_combiners: dict[str, bool] | None = None
                                    ) -> bool:
    """Drop one member from a cube bucket, REUSING the cached per-member
    inference (no model re-run).

    The archived member's cubes are deleted and the aggregate ``sr_``/``std_``/
    ``pcaN_`` cubes are recomputed from the remaining stack (the ensemble mean IS
    the plain member mean, so this is exact). ``keep_combiner`` preserves the
    ``comb_`` cubes + the manifest flag — set only when the archived member was
    PRUNED by the combiner (weight 0 everywhere), so its output is unchanged;
    otherwise the combiner is stale and its cubes are dropped. A positional
    bucket is first renamed to the label keying (fingerprints unproven).
    Returns ``True`` iff this bucket contained the member (and was rebuilt)."""
    old_labels = manifest_member_labels(read_bucket_manifest(cubes_dir))
    drop = next((lbl for lbl in old_labels if lbl.split("·")[0] == member_nn), None)
    if drop is None:
        return False
    keep_by_kind = {
        kind: bool((keep_combiners or {}).get(
            kind, keep_combiner if kind == _RBF_KIND else False))
        for kind in _ORDINARY_COMBINER_KINDS
    }
    # An RBF can stay only when the archived member was globally pruned. Drop
    # stale baked outputs independently.
    for kind, keep in keep_by_kind.items():
        if not keep:
            for path in glob.glob(os.path.join(cubes_dir,
                                               f"{_combiner_cube_prefix(kind)}_*.npy")):
                os.remove(path)
    _drop_field_means(cubes_dir)
    man = migrate_positional_bucket(cubes_dir) or {}
    new_labels = [lbl for lbl in old_labels if lbl != drop]
    recorded = recorded_fingerprints(man)
    man = sync_bucket_members(cubes_dir, man, new_labels,
                              {label: recorded[label] for label in new_labels}).manifest
    pca_amps: dict[str, list[float]] = {}
    pca_var: dict[str, list[float]] = {}
    for rec in (int(i) for i in man.get("indices", []) or []):
        if not new_labels or missing_member_cubes(cubes_dir, new_labels, rec):
            continue
        preds = _member_stack_of(cubes_dir, new_labels, rec, {})
        amps, var = _cache_field_cubes(cubes_dir, rec, preds, preds.mean(0), preds.std(0))
        pca_amps[str(rec)] = amps
        pca_var[str(rec)] = var
    man["has_combiner"] = bool(keep_by_kind[_RBF_KIND])
    for kind, keep in keep_by_kind.items():
        man[f"has_combiner_{kind}"] = bool(keep)
    man["pca_amps"] = pca_amps
    man["pca_var"] = pca_var
    write_bucket_manifest(cubes_dir, man)
    return True


_KNEE_PSNR_SCHEMA = 2
_KNEE_PSNR_WORKERS = 6


def _knee_psnr_path(starless: bool) -> str:
    return os.path.join(_ensemble_regime_dir(starless), "ensemble_knee_psnr.json")


def _read_test_manifest(starless: bool) -> dict | None:
    try:
        with open(os.path.join(_ensemble_cubes_dir(starless=starless),
                               "viz_index.json")) as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def _knee_psnr_identity(starless: bool, manifest: dict) -> dict:
    """What the curves depend on: the scored fields, the members and the
    checkpoints that made their cubes (a continued member's cubes are
    re-inferred in place) and every baked combiner (a refit changes its
    artifact fingerprint)."""
    regime_dir = _ensemble_regime_dir(starless)
    return {
        "schema": _KNEE_PSNR_SCHEMA,
        "knees": list(KNEE_GRID_E),
        "records_fp": manifest.get("records_fp"),
        "subset": manifest.get("subset"),
        "indices": sorted(int(i) for i in manifest.get("indices", []) or []),
        "member_labels": [str(x) for x in manifest.get("member_labels", []) or []],
        "member_fps": recorded_fingerprints(manifest),
        "target_psf_fwhm_arcsec": manifest.get("target_psf_fwhm_arcsec"),
        "combiner_fps": {kind: _combiner_fingerprint(regime_dir, kind)
                         for kind in COMBINER_MODELS
                         if manifest.get(f"has_combiner_{kind}")},
    }


def compute_knee_psnr_payload(starless: bool, *, force: bool = False,
                              progress: Callable[[int, int, str], None]
                              | None = None) -> dict | None:
    """PSNR-vs-knee curves (every band) for every member, the ensemble mean
    and each baked combiner over the regime's cached test cubes, plus each
    model's integrated PSNR (mean over log knee). Reuses the saved payload
    when nothing it depends on changed; ``None`` when there are no current
    cubes or records to score."""
    manifest = _read_test_manifest(starless)
    if manifest is None:
        return None
    labels = [str(x) for x in manifest.get("member_labels", []) or []]
    if not labels or labels != _regime_labels(ensemble_dir(), starless):
        return None
    identity = _knee_psnr_identity(starless, manifest)
    path = _knee_psnr_path(starless)
    if not force:
        with contextlib.suppress(OSError, ValueError), open(path) as handle:
            cached = json.load(handle)
            if cached.get("identity") == identity:
                return cached
    fields = _knee_psnr_field_curves(starless, manifest, identity, progress)
    if fields is None:
        return None
    sums: np.ndarray | None = None
    for curve in fields["curves"]:                     # field order, as before
        sums = curve if sums is None else sums + curve
    assert sums is not None
    n_fields = len(fields["fields"])
    curves = sums / n_fields
    model_ids = fields["model_ids"]
    member_meta = _member_meta_from_labels(labels)
    models = []
    for m, model_id in enumerate(model_ids):
        entry: dict = {"id": model_id,
                       "psnr": np.round(curves[m], 4).tolist(),
                       "integrated": np.round(integrated_psnr(curves[m]), 4).tolist()}
        if m < len(labels):
            meta = member_meta[m]
            entry.update(kind="member", label=labels[m], loss=meta.get("loss"),
                         asinh_knee=meta.get("asinh_knee"), blocks=meta.get("blocks"),
                         asinh_knees=meta.get("asinh_knees"),
                         output_knee=meta.get("output_knee"))
        elif model_id == "ensemble_mean":
            entry.update(kind="mean", label="ensemble mean")
        else:
            entry.update(kind="combiner", label=COMBINER_MODELS[model_id].label)
        models.append(entry)
    payload = {
        "available": True,
        "identity": identity,
        "regime": _regime_slug(starless),
        "knees": list(KNEE_GRID_E),
        "bands": list(BAND_NAMES),
        "n_fields": int(n_fields),
        "integration": {"from_e": KNEE_GRID_E[0], "to_e": KNEE_GRID_E[-1],
                        "weighting": "uniform in log10(knee), trapezoid rule"},
        "models": models,
    }
    _atomic_json(path, payload)
    return payload


def knee_psnr_fields(starless: bool, *,
                     progress: Callable[[int, int, str], None] | None = None
                     ) -> dict | None:
    """The per-field PSNR-vs-knee curves behind :func:`compute_knee_psnr_payload`
    (same loop, same models, nothing averaged): ``{identity, labels, model_ids,
    fields: [record index], curves: (fields, models, knees, bands)}`` (a field
    whose cubes or target record are missing is left out of ``fields``). Model
    ids are ``member_<i>`` (positional with ``labels``), ``ensemble_mean`` and
    each baked combiner kind. ``None`` when the cubes are missing, belong to
    another membership or their records changed. Writes nothing."""
    manifest = _read_test_manifest(starless)
    if manifest is None:
        return None
    labels = [str(x) for x in manifest.get("member_labels", []) or []]
    if not labels or labels != _regime_labels(ensemble_dir(), starless):
        return None
    return _knee_psnr_field_curves(
        starless, manifest, _knee_psnr_identity(starless, manifest), progress)


def _knee_psnr_field_curves(starless: bool, manifest: dict, identity: dict,
                            progress: Callable[[int, int, str], None] | None
                            ) -> dict | None:
    """Score every cached test field at every knee (see :func:`knee_psnr_fields`)."""
    labels = [str(x) for x in manifest.get("member_labels", []) or []]
    rdir = _sky_records_local_dir()
    subset = str(manifest.get("subset", ""))
    target_name = "clean" if starless else "hr"
    target_path = tfrecord_path(rdir, f"{target_name}_{subset}") if rdir else ""
    if (not identity["indices"] or not target_path or not os.path.isfile(target_path)
            or manifest.get("records_fp")
            != _eval_records_fingerprint(rdir, subset, starless=starless)):
        return None
    fwhm = validate_target_fwhm_arcsec(
        manifest.get("target_psf_fwhm_arcsec", Config.TARGET_PSF_FWHM_ARCSEC))
    cubes_dir = _ensemble_cubes_dir(starless=starless)
    combiner_kinds = [kind for kind in COMBINER_MODELS
                      if manifest.get(f"has_combiner_{kind}")]
    model_ids = ([f"member_{i}" for i in range(len(labels))]
                 + ["ensemble_mean"] + combiner_kinds)

    def field_curves(rec: int, target: np.ndarray) -> np.ndarray | None:
        tag = f"{rec:05d}"
        paths = (bucket_member_paths(manifest, cubes_dir, labels, rec)
                 + [os.path.join(cubes_dir, f"sr_{tag}.npy")]
                 + [os.path.join(cubes_dir, f"{COMBINER_MODELS[k].cube_prefix}_{tag}.npy")
                    for k in combiner_kinds])
        if not all(p is not None and os.path.isfile(p) for p in paths):
            return None
        truth = stretched_truth(target)
        return np.stack([knee_psnr(np.load(p), target, truth_asinh=truth)
                         for p in paths])                     # (models, K, C)

    wanted = set(identity["indices"])
    total, done = len(wanted), 0
    recs: list[int] = []
    curves: list[np.ndarray] = []
    pending: list[tuple[int, Future]] = []

    def drain(limit: int) -> None:
        nonlocal done
        while len(pending) > limit:
            rec, future = pending.pop(0)
            result = future.result()
            done += 1
            if result is not None:
                recs.append(rec)
                curves.append(result)
            if progress is not None:
                progress(done, total, "PSNR vs knee")

    with ThreadPoolExecutor(max_workers=_KNEE_PSNR_WORKERS) as pool:
        for image in ImageSet.read(target_path, num_images=max(wanted) + 1):
            rec = _record_index(image)
            if rec not in wanted:
                continue
            target = blur_target_array(np.asarray(image.data, np.float32), fwhm,
                                       pixel_scale_arcsec=image.pixel_scale_arcsec)
            pending.append((rec, pool.submit(field_curves, rec, target)))
            drain(2 * _KNEE_PSNR_WORKERS)
        drain(0)
    if not curves:
        return None
    return {"identity": identity, "labels": labels, "model_ids": model_ids,
            "fields": recs,
            "curves": np.stack(curves)}


def knee_psnr_status(starless: bool) -> dict:
    """The saved curves for the regime, flagged ``stale`` when the cubes,
    members or combiners moved on since they were computed."""
    try:
        with open(_knee_psnr_path(starless)) as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return {"available": False, "stale": False,
                "reason": "PSNR-vs-knee curves not computed yet"}
    manifest = _read_test_manifest(starless)
    payload["stale"] = (manifest is None
                        or payload.get("identity") != _knee_psnr_identity(starless, manifest))
    return payload


def job_knee_psnr(cap, *, starless: bool) -> dict:
    payload = compute_knee_psnr_payload(
        starless, force=True, progress=lambda i, n, label: cap.tick(i, n, label))
    if payload is None:
        raise RuntimeError("no current cached test cubes to score — run "
                           "“Evaluate on test set” first.")
    return {"regime": payload["regime"], "n_fields": payload["n_fields"],
            "models": len(payload["models"])}


def _refresh_knee_psnr(starless: bool, progress) -> None:
    """Keep the PSNR-vs-knee curves in step with the cubes (best-effort: a
    diagnostic never fails the evaluation that triggered it)."""
    try:
        compute_knee_psnr_payload(starless, progress=progress)
    except Exception as exc:  # noqa: BLE001 — diagnostic only
        print(f"[ensemble] PSNR-vs-knee curves not refreshed: {exc}")


def _reevaluate_from_cached_cubes(starless: bool,
                                  *, num_images: int | None = None,
                                  progress: Callable[[int, int, str], None]
                                  | None = None) -> dict | None:
    """Re-derive a full evaluation ENTIRELY from the cached per-member cubes —
    no model inference. Recomputes the ensemble/member PSNR summary, the pixel
    diagnostics + back-trace samples, the power spectrum and the combiner block,
    then rewrites ``eval_summary.json`` (with a fresh identity so a later
    ``Evaluate`` reuses it) and the evals payload. Returns the summary, or
    ``None`` when no valid cubes are cached. Used after archiving a member so the
    ensemble updates cheaply instead of forcing a full re-inference."""
    base = ensemble_dir()
    rdir = _sky_records_local_dir()
    if not rdir:
        return None
    sub = eval_subset(rdir)
    out_dir = _ensemble_regime_dir(starless)
    try:
        with open(os.path.join(_ensemble_cubes_dir(starless=starless),
                               "viz_index.json")) as f:
            cached_manifest = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    target_fwhm = validate_target_fwhm_arcsec(
        cached_manifest.get("target_psf_fwhm_arcsec",
                            Config.TARGET_PSF_FWHM_ARCSEC))

    ps_acc = None
    diag = EnsembleDiagnosticsAccumulator()
    model_cmet = {kind: _CombinerMetricAcc() for kind in _ORDINARY_COMBINER_KINDS}
    progress_total = max(0, int(num_images or 0))
    for position, (hr_v, mean_v, mem_v, model_v, lr_v, rec) in enumerate(
            _iter_cached_fields(starless), 1):
        if ps_acc is None:
            ps_acc = EnsembleSpectrumAccumulator(
                int(hr_v.shape[0]), float(Config.DEFAULT_PIXEL_SCALE))
        ps_acc.add(hr_v, mean_v, mem_v, model_combiners=model_v, lr=lr_v)
        diag.add(hr_v, mean_v, mem_v, combiners=model_v, field_index=rec)
        for kind, cmet in model_cmet.items():
            cmet.add(hr_v, mean_v, mem_v, model_v.get(kind))
        if progress is not None:
            progress(position, progress_total,
                     f"reevaluating cached test field {rec}")
    if not model_cmet[_RBF_KIND].n:
        return None
    _write_diag_samples(starless, diag)

    try:
        with open(os.path.join(_ensemble_cubes_dir(starless=starless),
                               "viz_index.json")) as f:
            man = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    labels = [str(x) for x in man.get("member_labels", []) or []]
    curves = (ps_acc.curves() if ps_acc is not None
              and float(ps_acc.bc.sum()) > 0 else None)
    coherence = (ps_acc.coherence_scores() if ps_acc is not None
                 and float(ps_acc.bc.sum()) > 0 else None)
    comb_block = model_cmet[_RBF_KIND].block(labels)
    payload = _evals_payload(curves, diag, labels, man.get("subset", ""),
                             combiner=comb_block,
                             model_combiners={kind: cmet.block(labels)
                                              for kind, cmet in model_cmet.items()},
                             coherence=coherence)
    payload["regime"] = _regime_slug(starless)
    with open(_evals_payload_path(starless), "w") as f:
        json.dump(payload, f)
    if curves is not None:
        with open(os.path.join(out_dir, "ensemble_power_spectrum.json"), "w") as f:
            json.dump({k: _jsonable(v) for k, v in curves.items()}, f)

    base_cmet = model_cmet[_RBF_KIND]
    per_member = ((base_cmet.mem / base_cmet.n).tolist()
                  if base_cmet.mem is not None else [])
    if num_images is None:
        num_images = base_cmet.n
    summary = {
        "regime": _regime_slug(starless),
        "member_labels": list(labels),
        "per_member_psnr_stretched": [float(x) for x in per_member],
        "recomputed_from_cubes": True,
        "reused": False,
        **_summary_headline(model_cmet, labels),
    }
    identity = _eval_identity(
        base, rdir, sub, out_dir, starless=starless, num_images=int(num_images),
        target_fwhm_arcsec=target_fwhm)
    # The checkpoints that MADE the cubes (none recorded for a positional
    # bucket), not those current now: a member continued since keeps the next
    # Evaluate from reusing this summary, so it re-infers that member.
    recorded = recorded_fingerprints(man)
    identity["member_fps"] = [recorded.get(label) for label in labels]
    summary["eval_identity"] = identity
    with open(os.path.join(out_dir, "eval_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    _refresh_knee_psnr(starless, progress)
    _refresh_band_evals(starless, progress)
    return summary


def _combiner_for_stack(regime_dir: str, labels: list[str], model_kind: str
                        ) -> tuple[object | None, list[int]]:
    """``(combiner, positions)``: the fitted ``model_kind`` combiner applied
    to a label-keyed member stack (``labels``, e.g. a cube bucket) and the
    positions in it of the members it takes, in its order.

    A spatial gate applies while every member it READS is in the stack:
    members that joined after its fit are skipped, and unread members that
    left since (archived) are dropped from an in-memory copy — the artifact
    is untouched and the math identical. The RBF needs exactly its fitted
    members. ``(None, [])`` when the combiner is absent or cannot apply."""
    artifact_dir = _combiner_artifact_dir(model_kind)
    if model_kind != SPATIAL_GATE_KIND:
        comb = load_combiner(regime_dir, member_labels=labels, artifact_dir=artifact_dir)
        return (comb, list(range(len(labels)))) if comb is not None else (None, [])
    gate = load_combiner(regime_dir, available_labels=labels, artifact_dir=artifact_dir)
    gate = restrict_to_available(gate, labels) if gate is not None else None
    positions = (sgc.member_positions(gate.member_labels, labels)
                 if gate is not None else None)
    return (gate, positions) if positions is not None else (None, [])


def _apply_combiner_to_test_cubes(starless: bool,
                                  model_kind: str | None = None, *,
                                  progress: Callable[[int, int, str], None]
                                  | None = None) -> bool:
    """Apply one fitted model to cached TEST member cubes without re-inference
    (a spatial gate reads only its members' cubes, by label)."""
    model_kind = _normalize_combiner_kind(model_kind)
    prefix = _combiner_cube_prefix(model_kind)
    cubes_dir = _ensemble_cubes_dir(starless=starless)
    man_path = os.path.join(cubes_dir, "viz_index.json")
    try:
        with open(man_path) as f:
            man = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    labels = [str(x) for x in man.get("member_labels", []) or []]
    comb, positions = _combiner_for_stack(_ensemble_regime_dir(starless), labels,
                                          model_kind)
    if comb is None:                     # no combiner, or stale for these cubes
        return False
    applied = 0
    indices = [int(i) for i in man.get("indices", []) or []]
    needs_lr = bool(getattr(comb, "use_lr", False))
    records_dir = _sky_records_local_dir() if needs_lr else None
    for position, rec in enumerate(indices, 1):
        tag = f"{rec:05d}"
        stack = []
        for p in positions:
            mf = bucket_member_path(man, cubes_dir, labels[p], rec)
            if mf is None or not os.path.isfile(mf):
                break
            stack.append(np.load(mf))
        lr = (load_cached_field_lr(cubes_dir, rec, records_dir=records_dir,
                                   subset=str(man.get("subset", "test")))
              if needs_lr else None)
        if len(stack) == len(positions) and (lr is not None or not needs_lr):
            comb_full = comb.apply_field(
                np.stack(stack, 0), lr=lr)     # (H, W, C) electrons
            np.save(os.path.join(cubes_dir, f"{prefix}_{tag}.npy"),
                    np.asarray(comb_full, np.float32))
            applied += 1
        if progress is not None:
            progress(position, len(indices),
                     f"applying combiner to test field {rec}")
    if applied == 0:
        return False
    man[f"has_combiner_{model_kind}"] = True
    if model_kind == _RBF_KIND:  # legacy viewer/cache contract
        man["has_combiner"] = True
    write_bucket_manifest(cubes_dir, man)
    return True


def _apply_all_combiners_to_test_cubes(starless: bool) -> bool:
    """Refresh all fitted ordinary combiners from one shared member cache."""
    # Do not pass a generator to ``any``: it would stop after a fitted RBF and
    # silently skip refreshing a model-specific cube cache.
    applied = [_apply_combiner_to_test_cubes(starless, kind)
               for kind in _ORDINARY_COMBINER_KINDS]
    return any(applied)


def _reconcile_combiner_on_archives(regime_dir: str, starless: bool,
                                   member_nns: list[str],
                                   model_kind: str = _RBF_KIND) -> bool:
    """Keep the regime's combiner when none of the departed members were used.

    Every queued archive is handled as one batch.  This matters when several
    pruned members were archived before the next evaluation: an intermediate
    combiner label set would not match the final active ensemble.

    A spatial gate is left byte-identical while every member it reads stays
    active: its validity is keyed on its reads, so rewriting it without the
    unread members would only change its fingerprint and make every output
    it produced stale for nothing (its consumers drop unread members that
    left in memory, :func:`restrict_to_available`). A gate that lost a member
    it reads is moved to a ``spatial_gate_backup_*`` directory, not deleted:
    it can be promoted back once that member is restored.
    """
    model_kind = _normalize_combiner_kind(model_kind)
    artifact_dir = _combiner_artifact_dir(model_kind)
    comb = load_combiner(regime_dir, artifact_dir=artifact_dir)
    if comb is None:
        return False

    def _drop() -> bool:
        path = os.path.join(regime_dir, artifact_dir)
        if isinstance(comb, SpatialGateCombiner):
            stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
            backup, n = f"{GATE_BACKUP_PREFIX}{stamp}", 1
            while os.path.exists(os.path.join(regime_dir, backup)):
                backup, n = f"{GATE_BACKUP_PREFIX}{stamp}-{n}", n + 1
            os.replace(path, os.path.join(regime_dir, backup))
            print(f"[ensemble] {artifact_dir} reads an archived member — moved to "
                  f"{backup} (promote it again once the member is restored)")
        else:
            shutil.rmtree(path, ignore_errors=True)
        with contextlib.suppress(FileNotFoundError):
            os.remove(_combiner_payload_path(starless, model_kind))
        return False

    positions = [i for i, lbl in enumerate(comb.member_labels)
                 if str(lbl).split("·")[0] in set(member_nns)]
    if any(not comb.member_pruned(pos) for pos in positions):
        return _drop()                            # a departed member was used
    active = _regime_labels(ensemble_dir(), starless)
    if isinstance(comb, SpatialGateCombiner):
        # Valid while every member it reads is active (members that joined
        # after its fit do not matter either); never rewritten.
        return True if reads_available(comb.read_labels, active) else _drop()
    for pos in reversed(positions):
        comb = comb.without_member(pos)
    # Keep any other combiner only for exactly the active ensemble.
    if list(comb.member_labels) != active:
        return _drop()
    save_combiner(comb, regime_dir, artifact_dir=artifact_dir)
    return True


def _rebuild_pending_archive_caches(starless: bool) -> bool:
    """Apply queued archives to cached cubes just before the next evaluation.

    Archiving is intentionally cheap: it only records the stale member.  The
    next evaluation performs this cache-only reconciliation once, batching all
    archives made since the prior evaluation.  Returns whether the test bucket
    was rebuilt and can therefore provide a fresh evaluation without inference.
    """
    names = _pending_archived_members(starless)
    member_nns = [name.split("_", 1)[1] for name in names]
    keep_comb = {
        kind: _reconcile_combiner_on_archives(
            _ensemble_regime_dir(starless), starless, member_nns, kind)
        for kind in _ORDINARY_COMBINER_KINDS
    }
    rebuilt_test = False
    for member_nn in member_nns:
        rebuilt_test = (_rebuild_bucket_dropping_member(
            _ensemble_cubes_dir(starless=starless), member_nn,
            keep_combiners=keep_comb) or rebuilt_test)
        _rebuild_bucket_dropping_member(
            _ensemble_cubes_dir("validate", starless=starless), member_nn,
            keep_combiners=keep_comb)
    return rebuilt_test


def job_ensemble_evaluate(cap, *, num_images: int,
                          starless: bool, force: bool = False,
                          target_fwhm_arcsec: float = Config.TARGET_PSF_FWHM_ARCSEC) -> dict:
    """Evaluate the ensemble on the held-out test set; persist + return the summary.

    ``starless`` (required — no regime default here; the routes default to
    STARFULL) selects the regime: STARLESS members scored against the
    starless ``clean`` target (erase stars), STARFULL against the starfull
    ``hr`` target (reconstruct them). Only members of the matching regime are
    evaluated together (their targets differ, so a mixed mean is meaningless).

    Idempotent: a completed evaluation with the SAME identity — the dataset, the
    member weights, the fitted combiner and the field count all unchanged —
    is never re-run. Its cheap browser figures (payload + back-trace samples) are
    rebuilt from the cached cubes (no model inference) and the cached metrics are
    returned. Pass ``force=True`` to bypass and re-infer every member.

    Otherwise the evaluation reads the cached member cubes (keyed by member
    label + checkpoint fingerprint) and runs ONLY the members whose cube of a
    field is missing or was made by another checkpoint (a new or continued
    member); members that left are dropped from the cache. Only regenerated
    records or another target PSF re-infer everyone.

    Also caches every evaluated field's member cubes, ensemble-mean (SR),
    per-pixel std (stdSR) and PCA cubes under ``<vis>/ensemble/<regime>/cubes/``
    (up to :data:`ENSEMBLE_VIZ_FIELDS_MAX`) so the ``ensemble`` viewer + morph
    can show the whole test set client-side. Every artifact this writes (cubes,
    evals payload, power spectrum, diagnostics, summary) lives under the regime
    dir, so starfull and starless never clobber each other.
    """
    target_fwhm = validate_target_fwhm_arcsec(target_fwhm_arcsec)
    base = ensemble_dir()
    rdir = _sky_records_local_dir()
    if not rdir:
        raise RuntimeError(_NO_SKY_RECORDS)
    sub = eval_subset(rdir)
    target = "clean" if starless else "hr"
    required_records = [tfrecord_path(rdir, f"dirty_{sub}"),
                        tfrecord_path(rdir, f"{target}_{sub}")]
    missing_records = [path for path in required_records if not os.path.isfile(path)]
    if missing_records:
        names = ", ".join(os.path.basename(path) for path in missing_records)
        raise RuntimeError(
            f"missing local {sub} record shard(s): {names}. "
            "Use Synthetic › Records › Sync from FASRC; it pulls test + validate together.")

    _migrate_test_bucket(starless)
    cubes_dir = _ensemble_cubes_dir(starless=starless)
    labels_now = _regime_labels(base, starless)
    fingerprints = member_fingerprints(base, labels_now)

    pending_archives = _pending_archived_members(starless)
    if pending_archives:
        cap.tick(0, 1, "rebuilding stale ensemble from cached cubes (no re-inference)")
        rebuilt_test = _rebuild_pending_archive_caches(starless)
        # Only a cache that is complete and current for the active members can
        # stand in for an evaluation; otherwise the evaluation below fills it.
        if rebuilt_test and _bucket_current(cubes_dir, labels_now, fingerprints):
            cached_summary = _reevaluate_from_cached_cubes(
                starless, num_images=int(num_images))
            if cached_summary is not None:
                _clear_archive_stale(starless)
                cap.tick(1, 1, "rebuilt stale ensemble from cached cubes")
                cached_summary["recomputed_from_archives"] = list(pending_archives)
                print(f"[ensemble evaluate] rebuilt {_regime_slug(starless)} "
                      f"after archiving {', '.join(pending_archives)} from cached cubes")
                return cached_summary
        # The cache might predate the archived member or be incomplete. Keep
        # the marker through the normal evaluation, then clear it only after a
        # new complete summary is written below.

    viz_cap = min(int(num_images), ENSEMBLE_VIZ_FIELDS_MAX)
    out_dir = _ensemble_regime_dir(starless)

    # Never re-infer on an unchanged dataset+model. If a completed evaluation
    # with this exact identity is already cached, rebuild only the cheap figures
    # from the stored cubes (seconds, no GPU) and return the cached metrics.
    identity = _eval_identity(base, rdir, sub, out_dir,
                              starless=starless, num_images=int(num_images),
                              target_fwhm_arcsec=target_fwhm)
    if not force:
        cached = _reusable_eval(starless, identity)
        if cached is not None:
            cap.tick(0, 1, "cached evaluation found — rebuilding figures (no inference)")
            compute_evaluation_payload(starless)   # payload + back-trace samples
            _refresh_knee_psnr(starless, lambda i, n, label: cap.tick(i, n, label))
            _refresh_band_evals(starless, lambda i, n, label: cap.tick(i, n, label))
            cap.tick(1, 1, "reused cached evaluation (dataset + model unchanged)")
            summary = dict(cached)
            summary["reused"] = True
            summary["regime"] = _regime_slug(starless)
            summary.setdefault("viz_fields", 0)
            print(f"[ensemble evaluate] reused cached result for "
                  f"{_regime_slug(starless)} (identity unchanged)")
            return summary

    if not labels_now:
        raise RuntimeError(f"no active {_regime_slug(starless)} members with a checkpoint "
                           f"under {base}")
    records_fp = _eval_records_fingerprint(rdir, sub, starless=starless)
    bucket = _open_member_bucket(
        cubes_dir, records={"subset": sub, "records_fp": records_fp,
                            "target_psf_fwhm_arcsec": target_fwhm},
        labels=labels_now, fingerprints=fingerprints, wipe=bool(force))
    _drop_combiner_cubes(cubes_dir)
    # The bucket lists no field until this run writes its manifest: a run
    # stopped half-way leaves member cubes to reuse, never a partial stack
    # that a cache re-score, the knee curves or a comparison reads as an
    # evaluation.
    write_bucket_manifest(cubes_dir, {
        **bucket.manifest, "indices": [], "pca_amps": {}, "pca_var": {},
        **{key: False for key in bucket.manifest if key.startswith("has_combiner")}})
    runner = LazyMemberRunner(base, starless=starless, labels=labels_now)
    saved: list[int] = []
    pca_amps: dict[int, list[float]] = {}        # rec_index → [a0, a1, a2]
    pca_var: dict[int, list[float]] = {}         # rec_index → variance explained
    ps_acc: list = [None]                        # lazy EnsembleSpectrumAccumulator
    diag_acc = EnsembleDiagnosticsAccumulator()  # pixel-level diagnostics
    model_cmet = {kind: _CombinerMetricAcc() for kind in _ORDINARY_COMBINER_KINDS}

    # Load every independently persisted ordinary model. They share member
    # predictions but retain distinct output cubes and diagnostics.
    # A spatial gate takes the members it reads by label (members that joined
    # after its fit are skipped); the RBF needs exactly the active members.
    models = {kind: _combiner_for_stack(out_dir, labels_now, kind)
              for kind in _ORDINARY_COMBINER_KINDS}

    def _member_stack(lr_image) -> np.ndarray:
        """The field's member stack: cached cubes plus the members that lack a
        current one (run and stored). Fields past the viewer cap are not
        cached, so every member runs on them."""
        lr = np.asarray(lr_image.data, np.float32)
        if len(saved) >= viz_cap:
            return np.asarray(runner(lr, labels_now), np.float32)
        rec = _record_index(lr_image)
        ran = _run_missing_members(cubes_dir, runner, lr, rec,
                                   missing_member_cubes(cubes_dir, labels_now, rec))
        return _member_stack_of(cubes_dir, labels_now, rec, ran)

    def _on_field(rec_index, lr_cube, preds, mean, std, hr_cube):
        model_full: dict[str, np.ndarray] = {}
        # Power spectrum over ALL fields that have HR (VIS band): HR vs
        # ensemble-mean (+ coherence r(k)) and the member-disagreement spectrum.
        if hr_cube is not None:
            hr_v, mean_v = _vis(hr_cube), _vis(mean)
            mem = np.asarray(preds, np.float32)
            mem_v = mem[..., 0] if mem.ndim == 4 else mem      # (M, H, W)
            if mem.ndim == 4 and mem.shape[0] == len(labels_now):
                everyone = list(range(mem.shape[0]))
                for kind, (model, positions) in models.items():
                    if model is not None:          # no copy of the whole stack
                        stack = mem if positions == everyone else mem[positions]
                        model_full[kind] = model.apply_field(stack, lr=lr_cube)
            model_v = {kind: (_vis(image) if image is not None else None)
                       for kind, image in model_full.items()}
            lr_v = _lr_on_hr_grid(lr_cube, int(hr_v.shape[0]))  # baseline r(k)
            if ps_acc[0] is None:
                ps_acc[0] = EnsembleSpectrumAccumulator(
                    int(hr_v.shape[0]), float(Config.DEFAULT_PIXEL_SCALE))
            ps_acc[0].add(hr_v, mean_v, mem_v, model_combiners=model_v,
                          lr=lr_v)
            # Back-tracing samples only for fields whose cubes are cached (within
            # viz_cap) — beyond that the sr_/std_ stamps don't exist to show.
            # The headline error is the ensemble mean's; every loaded combiner
            # is also scored in its own per-model histogram.
            diag_acc.add(hr_v, mean_v, mem_v, combiners=model_v,
                         field_index=(int(rec_index) if len(saved) < viz_cap
                                      else None))
            for kind, cmet in model_cmet.items():
                cmet.add(hr_v, mean_v, mem_v, model_v.get(kind))

        # LR/HR are read back from the records by the viewer; persist the
        # computed mean (SR) + std (stdSR) and the PCA disagreement basis
        # (mean + Σ aᵢ·sin·compᵢ powers the morphing animation). Cap the set.
        if len(saved) >= viz_cap:
            return
        rec = int(rec_index)
        amps, var_exp = _cache_field_cubes(cubes_dir, rec, preds, mean, std,
                                           lr=lr_cube)
        for kind, image in model_full.items():
            np.save(os.path.join(cubes_dir,
                                 f"{_combiner_cube_prefix(kind)}_{rec:05d}.npy"),
                    np.asarray(image, dtype=np.float32))
        pca_amps[rec] = amps
        pca_var[rec] = var_exp
        saved.append(rec)

    def _prog(i, n, lbl):
        cap.tick(i, n, lbl)

    out = evaluate_on_records(base, rdir, num_images=int(num_images),
                              starless=bool(starless),
                              target_fwhm_arcsec=target_fwhm,
                              member_stack=_member_stack, member_labels=labels_now,
                              on_field=_on_field, on_progress=_prog)
    member_labels = list(out.get("member_labels", []))
    if runner.seconds:
        print(f"[ensemble evaluate] member inference on {len(runner.seconds)} field(s) "
              f"({len(bucket.added)} new, {len(bucket.refreshed)} changed, "
              f"{len(bucket.dropped)} dropped member(s)); the rest from cached cubes")

    # The full eval already scored every member — bank the stretched PSNRs in
    # the per-member cache (free ride: no extra inference), but only when this
    # eval used the cache's canonical field count, so the numbers stay one
    # metric. Labels are "NN·psnr" → dir "member_NN".
    if int(num_images) == MEMBER_PSNR_FIELDS and out.get("n_scored"):
        scores = {}
        for lbl, p in zip(
            member_labels,
            out.get("per_member_psnr_stretched", []),
            strict=False,
        ):
            name = f"member_{lbl.split('·')[0]}"
            fp = member_fingerprint(os.path.join(base, name))
            if fp is not None:
                scores[name] = {"fingerprint": fp, "psnr": float(p),
                                "n_scored": int(out["n_scored"])}
        if scores:
            update_member_psnr_cache(
                scores, sub,
                records_fp=_member_scoring_records_fingerprint(rdir, sub))
    combiner_block = model_cmet[_RBF_KIND].block(member_labels)
    has_by_kind = {
        kind: bool(models[kind][0] is not None and cmet.n_comb > 0)
        for kind, cmet in model_cmet.items()
    }
    write_bucket_manifest(cubes_dir, {
        **bucket.manifest,
        "subset": sub, "indices": saved,
        "pca_n": ENSEMBLE_PCA_COMPONENTS, "pca_amps": pca_amps,
        "pca_var": pca_var,
        "member_labels": member_labels,
        "target_psf_fwhm_arcsec": target_fwhm,
        "has_combiner": has_by_kind[_RBF_KIND],
        **{f"has_combiner_{kind}": has for kind, has in has_by_kind.items()},
        # Eval-dataset identity: the cubes are keyed into THESE records —
        # regenerated records make them garbage.
        "records_fp": records_fp})
    prune_bucket_fields(cubes_dir, saved)

    # Power-spectrum summary (HR vs ensemble-mean coherence + disagreement).
    curves = None
    coherence = None
    if ps_acc[0] is not None and float(ps_acc[0].bc.sum()) > 0:
        curves = ps_acc[0].curves()
        coherence = ps_acc[0].coherence_scores()
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
                                 combiner=combiner_block,
                                 model_combiners={kind: cmet.block(member_labels)
                                                  for kind, cmet in model_cmet.items()},
                                 coherence=coherence)
        payload["regime"] = _regime_slug(starless)
        with open(_evals_payload_path(starless), "w") as f:
            json.dump(payload, f)
        _write_diag_samples(starless, diag_acc)

    # The headline numbers (ensemble mean, members, every combiner) in ONE
    # metric — the VIS asinh PSNR of the metric accumulator — exactly as a
    # rebuild from the cached cubes writes them. EnsembleModel.evaluate's
    # raw-electron PSNRs stay available under *_raw_e.
    for key in ("ensemble_psnr", "mean_member_psnr", "best_member_psnr",
                "ensemble_gain_db", "ensemble_vs_mean_member_db",
                "ensemble_vs_best_member_db"):
        if key in out:
            out[f"{key}_raw_e"] = out.pop(key)
    out.pop("best_member_label", None)
    if model_cmet[_RBF_KIND].n:
        out.update(_summary_headline(model_cmet, member_labels))

    out["regime"] = _regime_slug(starless)
    # Stamp the identity LAST (with the summary) so its presence means this run
    # finished — the reuse guard keys off it to skip a redundant re-evaluation.
    out["eval_identity"] = identity
    out["reused"] = False
    with open(os.path.join(out_dir, "eval_summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    if pending_archives:
        _clear_archive_stale(starless)
    print(json.dumps(out, indent=2))
    _refresh_knee_psnr(starless, lambda i, n, label: cap.tick(i, n, label))
    _refresh_band_evals(starless, lambda i, n, label: cap.tick(i, n, label))
    out["viz_fields"] = len(saved)
    return out


def _iter_cached_fields(starless: bool):
    """The VIS planes of :func:`_iter_cached_field_bands`: yield
    ``(target_vis, mean_vis, members_vis, model_vis, lr_vis, rec)``."""
    for rec, planes in _iter_cached_field_bands(starless, (0,)):
        hr_v, mean_v, mem_v, model_v, lr_v = planes[0]
        yield hr_v, mean_v, mem_v, model_v, lr_v, rec


def _iter_cached_field_bands(starless: bool, bands: tuple[int, ...]):
    """Yield ``(rec, {band: (target, mean, members, model, lr)})`` per cached
    field — one read of each cube serves every requested band index.

    ``rec`` is the field's record index — the key the ``sr_``/``std_`` cubes
    and the back-tracing sidecar are stored under. A band a cube lacks is left
    out of that field's dict.

    Streams the mean-SR (``sr_*.npy``) + individual member (``member_*_*.npy``)
    cubes the last Evaluate wrote for this regime, paired with the regime's
    TARGET from the records (``clean`` for starless, ``hr`` for starfull) — so
    any evaluation figure can be recomputed (e.g. after a code fix) in seconds
    with NO model inference and no full re-run. ``model`` maps every
    registered ordinary combiner kind to that band of its baked cube (or
    ``None``). ``lr`` is the LR plane bicubic-resampled onto the HR grid (the no-SR
    baseline for r(k)), or ``None`` when the dirty records are absent. Yields
    nothing when the cache is missing, or when the regime's membership changed
    since the cubes were written (a member archived/added: the figures would
    describe another ensemble).
    """
    cubes_dir = _ensemble_cubes_dir(starless=starless)
    man_path = os.path.join(cubes_dir, "viz_index.json")
    if not os.path.isfile(man_path):
        return
    with open(man_path) as f:
        man = json.load(f)
    labels = manifest_member_labels(man)
    if labels != _regime_labels(ensemble_dir(), starless):
        return
    idxs = [int(i) for i in man.get("indices", [])]
    sub = man.get("subset", "")
    n_members = len(labels)
    target_fwhm = validate_target_fwhm_arcsec(
        man.get("target_psf_fwhm_arcsec", Config.TARGET_PSF_FWHM_ARCSEC))
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

    # Stream targets and optional LR records in index order. Materialising every
    # 510²×4 target field costs hundreds of MB and defeats cached-cube refreshes.
    target_iter = iter(ImageSet.read(hr_path, num_images=max(idxs) + 1))
    current_target = next(target_iter, None)
    # LR baseline (optional): absent → the r_lr curve is simply skipped, no
    # error. It follows the same lazy alignment as the target stream.
    lr_path = tfrecord_path(rdir, f"dirty_{sub}") if rdir else ""
    lr_iter = (iter(ImageSet.read(lr_path, num_images=max(idxs) + 1))
               if lr_path and os.path.exists(lr_path) else None)
    current_lr = next(lr_iter, None) if lr_iter is not None else None
    for rec in sorted(idxs):
        while (current_target is not None
               and _record_index(current_target) < int(rec)):
            current_target = next(target_iter, None)
        hr = (current_target if current_target is not None
              and _record_index(current_target) == int(rec) else None)
        while (lr_iter is not None and current_lr is not None
               and _record_index(current_lr) < int(rec)):
            current_lr = next(lr_iter, None)
        lr_rec = (current_lr if current_lr is not None
                  and _record_index(current_lr) == int(rec) else None)
        sr_f = os.path.join(cubes_dir, f"sr_{rec:05d}.npy")
        if not os.path.isfile(sr_f) or hr is None:
            continue
        member_paths = bucket_member_paths(man, cubes_dir, labels, rec)
        if not all(mf is not None and os.path.isfile(mf) for mf in member_paths):
            continue                  # a fill stopped half-way: no partial stacks
        members = [np.load(mf) for mf in member_paths]
        model_cubes = {}
        for kind, spec in COMBINER_MODELS.items():
            model_f = os.path.join(cubes_dir, f"{spec.cube_prefix}_{rec:05d}.npy")
            model_cubes[kind] = (np.load(model_f)
                                 if (man.get(f"has_combiner_{kind}")
                                     and os.path.isfile(model_f)) else None)
        hr_cube = blur_target_array(
            np.asarray(hr.data, np.float32), target_fwhm,
            pixel_scale_arcsec=hr.pixel_scale_arcsec)
        sr_cube = np.load(sr_f)
        lr_cube = (np.asarray(lr_rec.data, np.float32)
                   if lr_rec is not None else None)
        planes = {}
        for band in bands:
            try:
                hr_v = _plane(hr_cube, band)
                model_v = {kind: (_plane(cube, band) if cube is not None else None)
                           for kind, cube in model_cubes.items()}
                planes[band] = (
                    hr_v, _plane(sr_cube, band),
                    np.stack([_plane(m, band) for m in members], 0), model_v,
                    _lr_on_hr_grid(lr_cube, int(hr_v.shape[0]), band)
                    if lr_cube is not None else None)
            except IndexError:
                continue          # a cube without this band: skip the band here
        if planes:
            yield rec, planes


def _member_meta_from_labels(labels) -> list[dict]:
    """Per-member ``{"loss", "blocks", "asinh_knee", "asinh_knees",
    "output_knee", "step", "psnr"}`` for line coloring (loss / depth / knee /
    test-PSNR gradient), positional with ``labels`` ("NN·psnr" → member_NN).
    ``asinh_knees`` marks a multi-knee member (``output_knee`` set: one image
    out; unset: one image per knee)."""
    base = ensemble_dir()
    rdir = _sky_records_local_dir()
    sub = eval_subset(rdir) if rdir else "test"
    rec_fp = _member_scoring_records_fingerprint(rdir, sub)
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
                     "asinh_knees": (origin or {}).get("asinh_knees"),
                     "output_knee": (origin or {}).get("output_knee"),
                     "step": _member_last_step(d),
                     "psnr": (entry or {}).get("psnr")})
    return meta


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
    """Retire one ensemble member: zip → tracking, tombstone, delete, mark stale.

    The zip lands in the active tracking campaign's ``models/``; the registry
    gets a permanent tombstone (so a FASRC mirror pulling the dir back never
    re-activates it); the local member dir is deleted; the FASRC-side copy is
    deleted too (best-effort, needs the SSH session).

    The archive path itself does not touch cached cubes. It records the
    affected regime as stale (its next evaluation batches pending archives and
    re-derives the summary from the remaining members' cached cubes, without
    model inference) and requests a stale-cube purge, which deletes the
    archived member's cubes once the console is idle
    (:mod:`euclid_polish.web.helpers.stale_purge`).
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
            "no active tracking campaign — start one in Notebook › Log (New campaign…) "
            "so the archived member has somewhere to go.")
    archived_starless = member_is_starless(src)
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
    _mark_archive_stale(archived_starless, name)
    cap.tick(2, 3, "marked ensemble stale — rebuild queued for next evaluation")
    shutil.rmtree(src, ignore_errors=True)
    cap.tick(3, 3, "deleting FASRC copy")
    remote_status = _delete_remote_member(name)
    regime = _regime_slug(archived_starless)
    store.append_log(
        f"Archived ensemble member `{name}` → `models/{meta['name']}` "
        f"({meta['size_bytes'] / 1e6:.1f} MB). Local member dir deleted; "
        f"{regime} evaluation marked stale; cached cubes will rebuild on the next "
        f"evaluation (no re-inference). FASRC copy: {remote_status}")
    print(f"  ✓ {name} → tracking {meta['name']}; {regime} evaluation marked "
          f"stale (cached-cube rebuild queued for next evaluation); "
          f"FASRC copy: {remote_status}")
    request_stale_purge(f"archived {name}")
    return {"zip": meta["name"], "member": name,
            "remote": remote_status, "stale_regime": regime}


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


def job_ensemble_pull(cap, *, members: list[str] | None = None,
                      dry_run: bool = False) -> dict:
    """Download the trained ensemble (``member_NN/``) from FASRC to the local
    checkpoint tree, so the evaluate / combiner / SR actions can run it locally.

    Member-aware: one ``--dry-run --itemize-changes`` probe decides which
    members actually changed on FASRC; only those are downloaded (and orphan-
    pruned). An unchanged ensemble downloads nothing — and the PSNR refresh
    afterwards is fingerprint-cached, so it re-scores only what was pulled.

    ``members`` (any member spelling) restricts the download to those members
    (Models › Members, Pull from FASRC); a requested member the probe finds
    unchanged is reported in ``up_to_date``. ``dry_run`` stops after the
    probe and returns what WOULD be pulled (``changed``) — nothing is
    downloaded or re-scored.
    """
    wanted = ({ensemble_registry.member_name(m) for m in members}
              if members else None)
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
    if dry_run:
        if probe_rc != 0 and not probe_out.strip():
            raise RuntimeError("the FASRC change probe failed — try again, or pull "
                               "without the dry run")
        print(f"  • {len(changed)} member(s) changed on FASRC: "
              + (", ".join(sorted(changed)) or "none"))
        return {"dry_run": True, "changed": sorted(changed),
                "tombstoned_skipped": sorted(tombstoned)}
    up_to_date: list[str] = []
    if wanted is not None:
        up_to_date = sorted(wanted - changed - tombstoned)
        changed &= wanted

    rc, err = 0, ""
    if probe_rc != 0 and not probe_out.strip() and wanted is not None:
        # Probe failed but the user named members: pull exactly those.
        changed = set(wanted) - tombstoned
        for i, name in enumerate(sorted(changed)):
            cap.tick(i, len(changed), f"pull {name}")
            rc, _out, err = STATE.ssh.rsync_pull(
                f"{remote.rstrip('/')}/{name}/", os.path.join(local, name),
                timeout=3600)
        print("  • change probe failed — pulled the requested members")
    elif probe_rc != 0 and not probe_out.strip():
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
    if changed:
        # Their cached cubes were made by the old checkpoints.
        request_stale_purge(f"pulled {', '.join(sorted(changed))}")
    # Refresh the per-member test PSNRs — fingerprint-cached, so only members
    # the pull actually changed get re-scored (nothing new → costs nothing).
    psnr: dict = {}
    try:
        psnr = job_member_psnr(cap)
    except Exception as e:  # noqa: BLE001 — the pull itself succeeded
        print(f"  ! member PSNR refresh skipped: {type(e).__name__}: {e}")
    return {"local": local, "n_members": n,
            "changed": sorted(changed), "up_to_date": up_to_date,
            "requested": sorted(wanted) if wanted is not None else None,
            "psnr": psnr}


# ===========================================================================
# Models workspace (spec §8.2, named Ensemble there): the joined members
# table, one member's inspector payload, training curves, the combiner variant
# registry and its compare / fit / promote jobs, the overview's headline +
# staleness, archived members (restore from zip) and the train-command preview.
# ===========================================================================

#: Per-band validation PSNR columns of ``training_log.csv``.
_BAND_LOG_COLUMNS = {"VIS": "psnr_vis", "Y_E": "psnr_y_e",
                     "J_E": "psnr_j_e", "H_E": "psnr_h_e"}
#: SLURM states of a job that is still going (the member may still grow).
_LIVE_SLURM_STATES = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING",
                      "REQUEUED", "RESIZING", "SUSPENDED"}
#: A promotion's automatic backup of the previous production gate.
GATE_BACKUP_PREFIX = "spatial_gate_backup_"
_VARIANT_NAME = re.compile(r"^spatial_gate_[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
#: Named compare reports (newest wins as ``spatial_gate_comparison.json``).
_COMPARE_DIR = "spatial_gate_comparisons"
_COMPARE_LATEST = "spatial_gate_comparison.json"
_COMPARE_ID = re.compile(r"^[0-9]{8}-[0-9]{6}(?:-[0-9]+)?$")


def _regime_dir_ro(starless: bool) -> str:
    """The regime artifact dir WITHOUT creating it (read-only GET paths)."""
    return os.path.abspath(os.path.join(Config.VIS_DIR, "ensemble",
                                        _regime_slug(starless)))


def _read_json_file(path: str) -> dict | None:
    try:
        with open(path) as handle:
            value = json.load(handle)
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _iso_mtime(path: str) -> str | None:
    try:
        stamp = os.path.getmtime(path)
    except OSError:
        return None
    return datetime.fromtimestamp(stamp, UTC).isoformat(timespec="seconds")


def _finite(value) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _as_int(value) -> int | None:
    v = _finite(value)
    return int(v) if v is not None else None


# ---- training curves ------------------------------------------------------ #

def _member_training_series(member_dir: str) -> dict | None:
    """One member's rollback-deduped validation history: joint + per-band
    PSNR (asinh), the combined training loss, the raw loss, the gradient norm
    (mean / max) and the wall time per 1000 steps. ``None`` without a log."""
    path = os.path.join(member_dir, TRAINING_LOG_FILENAME)
    if not os.path.isfile(path):
        return None
    try:
        recs = log_plot.read_training_log(path)
    except (FileNotFoundError, ValueError):
        return None
    recs = [r for r in recs if str(r.get("is_baseline", "")).strip()
            not in ("1", "1.0", "true", "True")]
    recs = log_plot.dedupe_latest_per_step(recs)

    def col(key: str) -> list[list[float]]:
        # 5 significant digits: plenty for a chart, ~40 % less JSON.
        out = []
        for r in recs:
            v = _finite(r.get(key))
            if v is not None and r.get("step") is not None:
                out.append([int(r["step"]), float(f"{v:.5g}")])
        return out

    step_time = []
    previous = None
    for r in recs:
        step, dur = r.get("step"), _finite(r.get("duration_s"))
        if step is not None and dur is not None and dur > 0:
            span = int(step) - (previous if previous is not None else 0)
            if span > 0:
                step_time.append([int(step), float(f"{dur * 1000.0 / span:.4g}")])
        if step is not None:
            previous = int(step)
    series = {
        "psnr": col("psnr_stretched"),
        "band_psnr": {band: col(key) for band, key in _BAND_LOG_COLUMNS.items()},
        "loss_series": col("combined_loss"),
        "train_loss": col("loss"),
        "gnorm": col("gnorm_avg"),
        "gnorm_max": col("gnorm_max"),
        "step_time": step_time,
    }
    if not (series["psnr"] or series["loss_series"]):
        return None
    return series


def training_curves_payload() -> list[dict]:
    """Training series for the in-browser curves — registry-ACTIVE members
    only (an archived member's directory can linger on disk or come back from
    a FASRC leftover; it never shows).

    Each entry: ``{name, label, starless, psnr, band_psnr{VIS,Y_E,J_E,H_E},
    loss_series, loss (= loss_series, deprecated alias), train_loss, gnorm,
    gnorm_max, step_time}`` — every series ``[[step, value], …]``; ``step_time``
    is seconds per 1000 steps — plus the facets the chart colours by:
    ``loss_norm`` (the member's reconstruction loss ``l1``/``l2``/…),
    ``blocks``, ``asinh_knee``, ``asinh_knees``, ``output_knee``,
    ``knee_loss``, ``test_psnr`` and ``target_steps``. The series is never
    overwritten by the norm (the old payload did that)."""
    base = ensemble_dir()
    rdir = _sky_records_local_dir()
    sub = eval_subset(rdir) if rdir else "test"
    rec_fp = _member_scoring_records_fingerprint(rdir, sub)
    cache = _load_member_psnr_cache()
    out = []
    for d in ensemble_registry.active_member_dirs(base):
        if not os.path.isdir(d):
            continue
        name = os.path.basename(d)
        series = _member_training_series(d)
        if series is None:
            continue
        origin = _member_origin(d) or {}
        entry = _member_psnr_entry(cache, name, d, sub, records_fp=rec_fp)
        out.append({
            "name": name,
            "label": ensemble_registry.member_label(name),
            **series,
            "loss": series["loss_series"],
            "loss_norm": origin.get("loss_norm") or "l1",
            "blocks": infer_checkpoint_num_res_blocks(d),
            "asinh_knee": origin.get("asinh_knee"),
            "asinh_knees": origin.get("asinh_knees"),
            "output_knee": origin.get("output_knee"),
            "knee_loss": origin.get("knee_loss"),
            "target_steps": _as_int(origin.get("target_steps")),
            "test_psnr": (entry or {}).get("psnr"),
            "starless": bool(origin.get("starless", False)),
        })
    return out


# ---- training jobs (the local FASRC job log) -------------------------------- #

def _split_names(raw) -> list[str]:
    return [t.strip() for t in str(raw or "").split(",") if t.strip()]


def training_jobs() -> list[dict]:
    """Every ``ensemble_train`` submission in the local job log (newest
    first), with the members it created or continued and its recipe."""
    try:
        rows = fasrc_jobs.JOBLOG.history_for_step("ensemble_train")
    except Exception:  # noqa: BLE001 — a missing/corrupt log is "no jobs"
        return []
    out = []
    for r in rows:
        try:
            params = json.loads(r.get("params_json") or "{}")
        except (TypeError, ValueError):
            params = {}
        if not isinstance(params, dict):
            params = {}
        mode = str(params.get("mode") or "add")
        names = _split_names(params.get("member_names")) or _split_names(params.get("members"))
        out.append({
            "jobid": str(r.get("jobid") or ""),
            "state": str(r.get("state") or "").upper() or None,
            "submitted_at": r.get("submitted_at") or None,
            "started_at": r.get("started_at") or None,
            "ended_at": r.get("ended_at") or None,
            "elapsed_seconds": _finite(r.get("elapsed_seconds")),
            "req_time_limit": r.get("req_time_limit") or None,
            "req_memory": r.get("req_memory") or None,
            "req_cpus": _as_int(r.get("req_cpus")),
            "req_gpus": _as_int(r.get("req_gpus")),
            "partition": r.get("partition") or None,
            "gpu_util_mean": _finite(r.get("gpu_util_mean")),
            "mode": mode,
            "member_names": names,
            "steps": _as_int(params.get("steps")),
            "continue_basis": params.get("continue_basis"),
            "target_steps": _as_int(params.get("target_steps")),
            "extra_steps": _as_int(params.get("extra_steps")),
            "params": {k: v for k, v in params.items() if not str(k).startswith("_")},
        })
    return out


def _jobs_by_member(jobs: list[dict]) -> dict[str, dict]:
    """member name → the NEWEST training job that created or continued it."""
    out: dict[str, dict] = {}
    for job in jobs:                               # newest first
        for name in job["member_names"]:
            out.setdefault(name, job)
    return out


def member_progress(step: int | None, origin: dict | None, job: dict | None) -> dict:
    """Steps reached vs the target, and whether the member stopped short.

    The target is the continue job's ``target_steps`` when the newest job
    continued the member up to N steps, else ``origin.json``'s. A member whose
    job is still live is ``running``; one below its target once the job ended
    is ``timeout`` (the SLURM time limit, or a crash, stopped it)."""
    origin = origin or {}
    target = _as_int(origin.get("target_steps"))
    if job and job.get("mode") == "continue" and job.get("continue_basis") == "target" \
            and job.get("target_steps"):
        target = int(job["target_steps"])
    live = bool(job and (job.get("state") or "") in _LIVE_SLURM_STATES)
    if live:
        status = "running"
    elif step is None or not target:
        status = "unknown"
    elif step >= target:
        status = "complete"
    else:
        status = "timeout"
    return {"target_steps": target,
            "fraction": (min(1.0, step / target) if step is not None and target else None),
            "status": status, "timeout": status == "timeout"}


# ---- one member row -------------------------------------------------------- #

def _knee_rows_by_label(knee: dict) -> dict[str, dict]:
    if not knee.get("available"):
        return {}
    return {str(m.get("label")): m for m in knee.get("models", []) or []
            if m.get("kind") == "member"}


def _gate_usage(starless: bool) -> dict:
    """The production gate's cached per-member usage (Combiners payload) —
    read from the saved payload only, never recomputed on a page load."""
    payload = _read_json_file(os.path.join(
        _regime_dir_ro(starless), COMBINER_MODELS[SPATIAL_GATE_KIND].payload_name))
    if not payload or not payload.get("available"):
        return {"available": False}
    diag = payload.get("gate_diagnostics") or {}
    reads = payload.get("read_labels")
    return {"available": True, "stale": bool(payload.get("stale")),
            "labels": [str(v) for v in payload.get("member_labels") or []],
            "read_labels": None if reads is None else [str(v) for v in reads],
            "bands": [str(v) for v in payload.get("band_names") or []],
            "usage": diag.get("usage") or payload.get("member_weight_integrals") or {},
            "usage_source": diag.get("usage_source") or {},
            "usage_by_brightness": diag.get("usage_by_brightness") or {},
            "brightness_names": diag.get("brightness_names") or []}


def _gate_peak(gate: dict, i: int) -> dict | None:
    """Member ``i``'s peak share of the gate's weight — the largest, over the
    bands, of its all-pixel, source-pixel and every brightness-bin mean (the
    ``used by the gate`` rule, eval/gate_members.py) — and where it is:
    ``{value, band, bin}`` (``bin`` = a brightness-bin name, ``"sources"`` or
    ``None`` for all pixels). ``None`` without a diagnostic."""
    best: dict | None = None

    def consider(value, band: str, where: str | None) -> None:
        nonlocal best
        v = _finite(value)
        if v is not None and (best is None or v > best["value"]):
            best = {"value": v, "band": band, "bin": where}

    for band, values in (gate.get("usage") or {}).items():
        if i < len(values or []):
            consider(values[i], str(band), None)
    for band, values in (gate.get("usage_source") or {}).items():
        if i < len(values or []):
            consider(values[i], str(band), "sources")
    names = list(gate.get("brightness_names") or [])
    for band, rows in (gate.get("usage_by_brightness") or {}).items():
        for b, row in enumerate(rows or []):
            if i < len(row or []):
                consider(row[i], str(band), names[b] if b < len(names) else None)
    return best


def _coherence_by_label(starless: bool) -> dict[str, dict]:
    evals = _read_json_file(os.path.join(_regime_dir_ro(starless), "ensemble_evals.json"))
    out: dict[str, dict] = {}
    for row in ((evals or {}).get("coherence") or {}).get("scores", []) or []:
        if str(row.get("id", "")).startswith("member_"):
            out[str(row.get("label"))] = {"overall": row.get("overall"), "sr": row.get("sr")}
    return out


@dataclasses.dataclass
class _MemberContext:
    """Everything a member row joins, read once per request."""

    base: str
    starless: bool
    sub: str
    rec_fp: str | None
    psnr_cache: dict
    jobs: dict[str, dict]
    knee: dict
    knee_rows: dict[str, dict]
    gate: dict
    coherence: dict[str, dict]
    vis_psnr: dict[str, float | None]
    vis_psnr_meta: dict | None


def _summary_vis_psnr(summary: dict | None) -> tuple[dict[str, float | None], dict | None]:
    """label → the headline (VIS asinh) test PSNR of each member from
    eval_summary.json: the metric of the Leaderboard's best-member test PSNR, NOT
    the member-PSNR cache (joint 4-band). ``per_member_vis_psnr`` is the
    explicit key; summaries recomputed from cubes before it existed carry the
    same VIS numbers as ``per_member_psnr_stretched`` (the TF-evaluate path's
    key of that name is joint 4-band, so it is only trusted from cubes)."""
    s = summary or {}
    labels = [str(x) for x in (s.get("member_labels") or s.get("per_member_labels") or [])]
    vals = s.get("per_member_vis_psnr")
    if vals is None and s.get("recomputed_from_cubes"):
        vals = s.get("per_member_psnr_stretched")
    if not labels or not isinstance(vals, list) or len(vals) != len(labels):
        return {}, None
    meta = {"metric": s.get("psnr_metric") or "vis_asinh",
            "knee_e": s.get("psnr_knee_e"), "n_scored": s.get("n_scored")}
    return {lbl: _finite(v) for lbl, v in zip(labels, vals, strict=True)}, meta


def _member_context(starless: bool) -> _MemberContext:
    base = ensemble_dir()
    rdir = _sky_records_local_dir()
    sub = eval_subset(rdir) if rdir else "test"
    knee = knee_psnr_status(starless)
    vis_psnr, vis_psnr_meta = _summary_vis_psnr(_read_eval_summary(starless))
    return _MemberContext(
        base=base, starless=starless, sub=sub,
        rec_fp=_member_scoring_records_fingerprint(rdir, sub),
        psnr_cache=_load_member_psnr_cache(), jobs=_jobs_by_member(training_jobs()),
        knee=knee, knee_rows=_knee_rows_by_label(knee), gate=_gate_usage(starless),
        coherence=_coherence_by_label(starless),
        vis_psnr=vis_psnr, vis_psnr_meta=vis_psnr_meta)


def _member_row(name: str, ctx: _MemberContext) -> dict:
    d = os.path.join(ctx.base, name)
    label = ensemble_registry.member_label(name)
    origin = _member_origin(d)
    o = origin or {}
    step = _member_last_step(d)
    job = ctx.jobs.get(name)
    entry = _member_psnr_entry(ctx.psnr_cache, name, d, ctx.sub, records_fp=ctx.rec_fp)
    knee_row = ctx.knee_rows.get(label)
    knee_integrated = None
    if knee_row is not None:
        vals = [_finite(v) for v in knee_row.get("integrated") or []]
        bands = list(ctx.knee.get("bands") or BAND_NAMES)
        knee_integrated = dict(zip(bands, vals, strict=False))
        finite_vals = [v for v in vals if v is not None]
        knee_integrated["mean"] = (float(np.mean(finite_vals)) if finite_vals else None)
    gate_usage = gate_usage_source = gate_usage_peak = used_by_gate = None
    if ctx.gate.get("available") and ctx.gate.get("read_labels") is not None:
        # Production SR runs this member iff the production gate reads it.
        used_by_gate = label in ctx.gate["read_labels"]
    if ctx.gate.get("available") and label in ctx.gate["labels"]:
        i = ctx.gate["labels"].index(label)
        gate_usage_peak = _gate_peak(ctx.gate, i)
        gate_usage = {b: _finite((ctx.gate["usage"].get(b) or [None] * (i + 1))[i])
                      for b in ctx.gate["bands"]}
        if ctx.gate["usage_source"]:
            gate_usage_source = {b: _finite((ctx.gate["usage_source"].get(b) or [None] * (i + 1))[i])
                                 for b in ctx.gate["bands"]}
    lb = os.path.join(d, "loss_best")
    seed = o.get("seed")
    return {
        "name": name, "label": label,
        "starless": bool(o.get("starless", False)),
        "regime": "starless" if o.get("starless") else "starfull",
        "origin": origin,
        "op": o.get("op"), "forked_from": o.get("forked_from"),
        "loss": o.get("loss_norm") or "l1",
        "blocks": infer_checkpoint_num_res_blocks(d),
        "asinh_knee": o.get("asinh_knee"),
        "asinh_knees": o.get("asinh_knees"),
        "output_knee": o.get("output_knee"),
        "knee_loss": o.get("knee_loss"),
        "noise_aug": o.get("noise_aug"), "bootstrap": o.get("bootstrap"),
        "icnr": o.get("icnr"), "seed": seed if seed is not None else _member_seed(d),
        "commit": o.get("commit"), "created_at": o.get("created_at"),
        "noise_model": o.get("noise_model"),
        "step": step, **member_progress(step, origin, job),
        "job": ({k: job[k] for k in ("jobid", "state", "submitted_at", "ended_at",
                                     "elapsed_seconds", "req_time_limit", "gpu_util_mean",
                                     "mode")} if job else None),
        "psnr": (entry or {}).get("psnr"),
        "vis_psnr": ctx.vis_psnr.get(label),
        "knee_integrated": knee_integrated,
        "gate_usage": gate_usage, "gate_usage_source": gate_usage_source,
        "gate_usage_peak": gate_usage_peak, "used_by_gate": used_by_gate,
        "coherence": ctx.coherence.get(label),
        "has_loss_best": os.path.isdir(lb) and _checkpoint_exists(lb),
        "size_mb": round(_dir_size_mb(d), 1),
    }


def members_payload(starless: bool) -> dict:
    """The Models › Members roster: one joined row per ACTIVE member of the
    regime (status + origin.json + training job + knee-integrated PSNR per
    band + production-gate usage + spectral coherence), the archived
    tombstones (with their zip location, for restore) and the join's own
    freshness flags."""
    ctx = _member_context(starless)
    reg = ensemble_registry.load_registry(ctx.base)
    rows = []
    for name in reg["active"]:
        d = os.path.join(ctx.base, name)
        if not (os.path.isdir(d) and _checkpoint_exists(d)):
            continue
        if ensemble_registry.member_is_starless(d) != bool(starless):
            continue
        rows.append(_member_row(name, ctx))
    ranked = sorted((r for r in rows if r["psnr"] is not None), key=lambda r: -r["psnr"])
    rank = {r["name"]: i + 1 for i, r in enumerate(ranked)}
    ranked_knee = sorted((r for r in rows if (r["knee_integrated"] or {}).get("mean") is not None),
                         key=lambda r: -r["knee_integrated"]["mean"])
    knee_rank = {r["name"]: i + 1 for i, r in enumerate(ranked_knee)}
    for r in rows:
        r["psnr_rank"] = rank.get(r["name"])
        r["knee_rank"] = knee_rank.get(r["name"])
    other = sum(1 for n in reg["active"]
                if ensemble_registry.member_is_starless(os.path.join(ctx.base, n)) != bool(starless))
    return {
        "regime": _regime_slug(starless),
        "members": rows,
        "other_regime_members": other,
        "archived": ensemble_registry.archived_members(ctx.base, Config.TRACKING_DIR),
        "knee": {"available": bool(ctx.knee.get("available")),
                 "stale": bool(ctx.knee.get("stale")),
                 "n_fields": ctx.knee.get("n_fields")},
        "gate": {"available": bool(ctx.gate.get("available")),
                 "stale": bool(ctx.gate.get("stale")),
                 "n_members": len(ctx.gate.get("labels") or [])},
        "psnr_fields": MEMBER_PSNR_FIELDS,
        "vis_psnr": ctx.vis_psnr_meta,
        "eval_subset": ctx.sub,
    }


def member_detail(name: str) -> dict | None:
    """One member's inspector payload (active or archived): the joined row,
    its origin, training series, PSNR-vs-knee curve (with the mean and the
    production gate for reference) and production-gate usage by brightness.
    ``None`` for a name the registry has never seen."""
    name = ensemble_registry.member_name(name)
    base = ensemble_dir()
    reg = ensemble_registry.load_registry(base)
    tombstone = next((t for t in ensemble_registry.archived_members(base, Config.TRACKING_DIR)
                      if t.get("name") == name), None)
    d = os.path.join(base, name)
    active = name in reg["active"]
    if not active and tombstone is None:
        return None
    starless = ensemble_registry.member_is_starless(d)
    out: dict = {"name": name, "label": ensemble_registry.member_label(name),
                 "active": active, "archived": tombstone,
                 "regime": _regime_slug(starless), "row": None,
                 "curves": None, "knee": None, "gate": None}
    if not active:
        return out
    ctx = _member_context(starless)
    # The table's row (with its ranks among the regime's members).
    rows = {r["name"]: r for r in members_payload(starless)["members"]}
    out["row"] = rows.get(name) or _member_row(name, ctx)
    out["curves"] = _member_training_series(d)
    label = out["label"]
    if ctx.knee.get("available"):
        models = ctx.knee.get("models", []) or []
        pick = [m for m in models if m.get("label") == label or m.get("kind") in ("mean", "combiner")]
        out["knee"] = {"knees": ctx.knee.get("knees"), "bands": ctx.knee.get("bands"),
                       "stale": bool(ctx.knee.get("stale")),
                       "models": [{k: m.get(k) for k in ("id", "kind", "label", "psnr", "integrated")}
                                  for m in pick]}
    if ctx.gate.get("available") and label in ctx.gate["labels"]:
        i = ctx.gate["labels"].index(label)
        out["gate"] = {
            "stale": ctx.gate["stale"], "bands": ctx.gate["bands"],
            "brightness_names": ctx.gate["brightness_names"],
            "usage": {b: _finite((v or [None] * (i + 1))[i]) for b, v in ctx.gate["usage"].items()},
            "usage_source": {b: _finite((v or [None] * (i + 1))[i])
                             for b, v in ctx.gate["usage_source"].items()},
            "by_brightness": {b: [_finite(row[i]) if i < len(row) else None for row in rows]
                              for b, rows in ctx.gate["usage_by_brightness"].items()},
            "uniform": 1.0 / max(1, len(ctx.gate["labels"])),
        }
    return out


# ---- archive / restore ----------------------------------------------------- #

def _safe_extract(zf: zipfile.ZipFile, dest: str) -> None:
    root = os.path.realpath(dest)
    for info in zf.infolist():
        target = os.path.realpath(os.path.join(dest, info.filename))
        if target != root and not target.startswith(root + os.sep):
            raise RuntimeError(f"refusing unsafe zip entry {info.filename!r}")
    zf.extractall(dest)


def job_restore_member(cap, *, name: str) -> dict:
    """Bring an archived member back: unzip its tracking archive into the
    ensemble dir and move its tombstone back to active. The regime's
    evaluation, knee curves and combiner then read stale (membership changed)
    until re-evaluated / refitted — shown by the Models › Leaderboard checks.
    An archive the next evaluation has not applied yet is dequeued, so that
    evaluation keeps the member's cached cubes and the combiners reading it."""
    name = ensemble_registry.member_name(name)
    base = ensemble_dir()
    tomb = next((t for t in ensemble_registry.archived_members(base, Config.TRACKING_DIR)
                 if t.get("name") == name), None)
    if tomb is None:
        raise RuntimeError(f"{name} is not archived")
    if not tomb.get("zip_found"):
        raise RuntimeError(f"the archive zip of {name} ({tomb.get('zip')}) is not in any "
                           "tracking campaign — it cannot be restored")
    dest = os.path.join(base, name)
    if os.path.exists(dest):
        raise RuntimeError(f"{dest} already exists — move it away before restoring")
    tmp = os.path.join(base, f".restore-{name}")
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    cap.tick(0, 3, f"unzipping {os.path.basename(tomb['zip_path'])}")
    try:
        with zipfile.ZipFile(tomb["zip_path"]) as zf:
            _safe_extract(zf, tmp)
        if not _checkpoint_exists(tmp):
            raise RuntimeError(f"the archive of {name} holds no checkpoint")
        cap.tick(1, 3, "installing the member directory")
        os.rename(tmp, dest)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    cap.tick(2, 3, "updating the registry")
    ensemble_registry.restore_member_entry(base, name)
    starless = ensemble_registry.member_is_starless(dest)
    _unmark_archive_stale(starless, name)
    regime = _regime_slug(starless)
    with contextlib.suppress(Exception):
        tracking_default_store().append_log(
            f"Restored ensemble member `{name}` from `{tomb.get('zip')}` "
            f"({tomb.get('campaign')}); {regime} evaluation is stale until re-run.")
    cap.tick(3, 3, "done")
    print(f"  ✓ {name} restored from {tomb['zip_path']} ({regime})")
    return {"member": name, "zip": tomb["zip_path"], "regime": regime}


# ---- combiner variants ----------------------------------------------------- #

def variant_dir(starless: bool, name: str, *, must_exist: bool = True) -> str:
    """The directory of a gate variant: ``name`` is its directory
    (``spatial_gate_<x>``) or its model spec (``gate:<x>``, ``production``).
    Raises :class:`ValueError` on a malformed name (or a missing one when
    ``must_exist``)."""
    raw = str(name or "").strip()
    if raw == "production":
        raw = COMBINER_MODELS[SPATIAL_GATE_KIND].artifact_dir
    elif raw.startswith("gate:"):
        raw = "spatial_gate_" + raw.removeprefix("gate:")
    if not _VARIANT_NAME.fullmatch(raw):
        raise ValueError(f"not a spatial gate variant name: {name!r} "
                         "(spatial_gate_<letters, digits, . _ ->)")
    d = os.path.join(_regime_dir_ro(starless), raw)
    if must_exist and not os.path.isfile(os.path.join(d, "combiner.json")):
        raise ValueError(f"no gate variant {raw}")
    return d


def _fit_summary(fit_meta: dict) -> dict:
    keep = ("model", "mix_space", "width", "use_lr", "loss", "loss_knees_e", "steps",
            "steps_run", "complete", "batch_size", "crop", "learning_rate",
            "uniform_crop_fraction", "blackout_fields", "fit_seconds", "subset",
            "num_images", "seed", "eval_every", "holdout_count", "variant", "fitted_via",
            "members_requested", "used_threshold", "promoted_from", "promoted_at",
            "best_member_per_band")
    out = {k: fit_meta[k] for k in keep if k in fit_meta}
    out["train_field_count"] = len(fit_meta.get("train_fields") or [])
    out["holdout_field_count"] = len(fit_meta.get("holdout_fields") or [])
    return out


def _history_rows(fit_meta: dict) -> list[dict]:
    rows = []
    for h in fit_meta.get("history") or []:
        if not isinstance(h, dict):
            continue
        rows.append({k: h.get(k) for k in ("step", "loss", "train_loss", "vis_psnr",
                                           "band_psnr", "integrated_psnr",
                                           "vis_integrated_psnr") if k in h})
    return rows


def _latest_compare(starless: bool) -> dict | None:
    return _read_json_file(os.path.join(_regime_dir_ro(starless), _COMPARE_LATEST))


def _promotion_state(directory: str, manifest: dict, reads: list[str],
                     active: list[str], *, production: bool = False) -> dict:
    """``{ok, reason}``: whether :func:`job_combiner_promote` would install
    this variant now (``force`` aside), and why not."""
    if production:
        return {"ok": False, "reason": "already the production gate"}
    reason = sgc.promotion_refusal(directory, manifest)
    if reason is None and not reads_available(reads, active):
        missing = [lb for lb in reads if lb not in active]
        reason = ("it reads members that are not active: "
                  + ", ".join(missing[:6]) + ("…" if len(missing) > 6 else ""))
    return {"ok": reason is None, "reason": reason}


def combiner_variants(starless: bool) -> dict:
    """The combiner variant registry of a regime: every ``spatial_gate_*``
    directory (the production gate, named variants, promotion backups; the
    legacy RBF is never listed), each with its fit summary, held-out loss history, membership
    against the active members, test PSNR (production: the eval summary;
    variants: the latest compare report) and knee-integrated PSNR (production:
    the knee payload; variants: the latest compare report)."""
    regime_dir = _regime_dir_ro(starless)
    base = ensemble_dir()
    active = _regime_labels(base, starless)
    manifest = _read_test_manifest(starless) or {}
    cube_labels = [str(v) for v in manifest.get("member_labels", []) or []]
    summary = _read_eval_summary(starless) or {}
    knee = knee_psnr_status(starless)
    knee_gate = next((m for m in knee.get("models", []) or []
                      if m.get("id") == SPATIAL_GATE_KIND), None) if knee.get("available") else None
    report = _latest_compare(starless) or {}
    report_knee = (report.get("knee") or {}).get("methods") or {}
    natural = (report.get("groups") or {}).get("natural") or {}
    blackout = (report.get("groups") or {}).get("blackout") or {}
    production_dir = COMBINER_MODELS[SPATIAL_GATE_KIND].artifact_dir
    rows = []
    candidates = sorted(glob.glob(os.path.join(regime_dir, "spatial_gate_*")))
    for d in candidates:
        name = os.path.basename(d)
        m = _read_json_file(os.path.join(d, "combiner.json"))
        if m is None or m.get("kind") != SPATIAL_GATE_KIND or not _VARIANT_NAME.fullmatch(name):
            continue
        labels = [str(v) for v in m.get("member_labels") or []]
        fit_meta = m.get("fit_meta") or {}
        active_members = m.get("active_members")
        reads = ([labels[int(i)] for i in active_members if int(i) < len(labels)]
                 if isinstance(active_members, list) else labels)
        production = name == production_dir
        method = f"gate:{name}"
        row = {
            "name": name, "kind": "gate",
            "spec": "production" if production else f"gate:{name.removeprefix('spatial_gate_')}",
            "production": production, "backup": name.startswith(GATE_BACKUP_PREFIX),
            "member_labels": labels, "reads": reads, "n_members": len(labels),
            "n_reads": len(reads), "pruned": isinstance(active_members, list),
            "mix_space": m.get("mix_space", "asinh"), "use_lr": bool(m.get("use_lr")),
            "width": m.get("width"), "fitted_at": _iso_mtime(os.path.join(d, "combiner.npz")),
            "fingerprint": combiner_artifact_fingerprint(regime_dir, name),
            # Current while every member the gate READS is active; "extra"
            # members joined after the fit (refit to consider them).
            "membership": {"current": reads_available(reads, active),
                           "missing": [lb for lb in labels if lb not in active],
                           "missing_reads": [lb for lb in reads if lb not in active],
                           "extra": joined_after_fit(labels, active)},
            "promotion": _promotion_state(d, m, reads, active, production=production),
            # A gate applies by label while the cubes hold every member it
            # reads (load_gate_methods drops unread members that left).
            "applies_to_test_cubes": sgc.member_positions(reads, cube_labels) is not None,
            "fit": _fit_summary(fit_meta),
            "selected": fit_meta.get("selected"), "baseline": fit_meta.get("baseline_holdout"),
            "history": _history_rows(fit_meta),
            "test": ({"source": "compare", "report": report.get("id"),
                      "band_psnr": (natural.get(method) or {}).get("band_psnr"),
                      "blackout_band_psnr": (blackout.get(method) or {}).get("band_psnr")}
                     if method in natural else None),
            "knee": ({"source": "compare", "report": report.get("id"),
                      "integrated": report_knee[method]["integrated"],
                      "psnr": report_knee[method]["psnr"]} if method in report_knee else None),
        }
        if production:
            row["eval"] = {"psnr": summary.get(f"{SPATIAL_GATE_KIND}_combiner_psnr"),
                           "vs_mean_db": summary.get(f"{SPATIAL_GATE_KIND}_combiner_vs_mean_db"),
                           "vs_best_member_db": summary.get(
                               f"{SPATIAL_GATE_KIND}_combiner_vs_best_member_db")}
            if knee_gate is not None:
                row["knee"] = {"source": "knee", "stale": bool(knee.get("stale")),
                               "integrated": knee_gate.get("integrated"),
                               "psnr": knee_gate.get("psnr")}
        rows.append(row)
    return {"regime": _regime_slug(starless), "production": production_dir,
            "active_members": active, "cube_members": cube_labels,
            "variants": rows,
            "compare": ({"id": report.get("id"), "created": report.get("created"),
                         "methods": report.get("methods"), "n_fields": report.get("n_fields")}
                        if report else None)}


# ---- compare --------------------------------------------------------------- #

def compare_reports(starless: bool) -> list[dict]:
    """Saved compare reports, newest first (id, created, methods, fields)."""
    folder = os.path.join(_regime_dir_ro(starless), _COMPARE_DIR)
    out = []
    for path in glob.glob(os.path.join(folder, "*.json")):
        rid = os.path.basename(path)[:-5]
        rep = _read_json_file(path) if _COMPARE_ID.fullmatch(rid) else None
        if rep is None:
            continue
        out.append({"id": rid, "created": rep.get("created"), "methods": rep.get("methods"),
                    "n_fields": rep.get("n_fields"), "gates_requested": rep.get("gates_requested")})
    out.sort(key=lambda r: r["id"], reverse=True)
    return out


def read_compare_report(starless: bool, report_id: str | None = None) -> dict | None:
    """One saved compare report (``None`` → the latest)."""
    if not report_id:
        return _latest_compare(starless)
    if not _COMPARE_ID.fullmatch(str(report_id)):
        raise ValueError(f"bad report id {report_id!r}")
    return _read_json_file(os.path.join(_regime_dir_ro(starless), _COMPARE_DIR,
                                        f"{report_id}.json"))


def _report_id(folder: str) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    rid, n = stamp, 1
    while os.path.exists(os.path.join(folder, f"{rid}.json")):
        n += 1
        rid = f"{stamp}-{n}"
    return rid


def default_compare_gates(starless: bool) -> list[str]:
    """Every gate directory that applies to the regime's test cubes."""
    data = combiner_variants(starless)
    return [v["name"] for v in data["variants"]
            if v["kind"] == "gate" and v["applies_to_test_cubes"] and not v["backup"]]


def job_combiner_compare(cap, *, starless: bool, gates: list[str] | None = None,
                         blackout_fields: int = 40, seed: int = 0,
                         include_rbf: bool = True, knee: bool = True) -> dict:
    """Score gate variants (default: every applicable one) against the mean,
    the RBF and every member on the cached test cubes + blackout copies —
    ``scripts/fit_spatial_gate.py compare`` as a local job. Saves the report
    under ``<regime>/spatial_gate_comparisons/<id>.json`` and as the latest
    ``spatial_gate_comparison.json``."""
    regime_dir = _ensemble_regime_dir(starless)
    records_dir = _sky_records_local_dir()
    if not records_dir:
        raise RuntimeError(_NO_SKY_RECORDS)
    names = [variant_dir(starless, g) for g in (gates or default_compare_gates(starless))]
    names = [os.path.basename(p) for p in names]
    if not names:
        raise RuntimeError("no gate variant applies to the current test cubes")
    manifest = _read_test_manifest(starless) or {}
    labels = [str(v) for v in manifest.get("member_labels", []) or []]
    active = _regime_labels(ensemble_dir(), starless)
    runner = (LazyMemberRunner(ensemble_dir(), starless=starless, labels=labels)
              if labels and labels == active else None)
    if runner is None and blackout_fields > 0:
        print("  • the test cubes' members differ from the active members — "
              "only already-cached blackout fields can be scored")
    result = sgc.run_compare(
        regime_dir=regime_dir, records_dir=records_dir, gates=names, runner=runner,
        blackout_fields=int(blackout_fields), seed=int(seed),
        target_name="clean" if starless else "hr", include_rbf=include_rbf, knee=knee,
        progress=lambda i, n, label: cap.tick(i, n, label),
        log=lambda message: print(message, flush=True))
    folder = os.path.join(regime_dir, _COMPARE_DIR)
    os.makedirs(folder, exist_ok=True)
    rid = _report_id(folder)
    report = {**result.report, "id": rid, "gates_requested": names,
              "regime": _regime_slug(starless)}
    _atomic_json(os.path.join(folder, f"{rid}.json"), report)
    _atomic_json(os.path.join(regime_dir, _COMPARE_LATEST), report)
    print(sgc.format_report(report))
    return {"report_id": rid, "methods": report["methods"], "n_fields": report["n_fields"]}


# ---- fit a named variant --------------------------------------------------- #

def check_new_variant_name(starless: bool, out_name: str, *, overwrite: bool = False) -> str:
    """Validate a fit destination: a ``spatial_gate_<x>`` name that is neither
    the production gate nor a promotion backup, and (unless ``overwrite``)
    not an existing variant. Returns the directory."""
    d = variant_dir(starless, out_name, must_exist=False)
    name = os.path.basename(d)
    if name == COMBINER_MODELS[SPATIAL_GATE_KIND].artifact_dir:
        raise ValueError("a fit never writes the production gate — name a variant, "
                         "then promote it")
    if name.startswith(GATE_BACKUP_PREFIX):
        raise ValueError(f"{GATE_BACKUP_PREFIX}* names are reserved for promotion backups")
    if (name.startswith(_COMPARE_LATEST.removesuffix(".json"))
            or name.endswith((".json", "_evals"))):
        # spatial_gate_comparisons/ (compare reports), spatial_gate_comparison.json
        # and the <combiner>_evals.json sidecars share the variant namespace
        raise ValueError(f"{name} is reserved for compare reports / eval sidecars")
    if os.path.exists(d):
        manifest = _read_json_file(os.path.join(d, "combiner.json")) if os.path.isdir(d) else None
        if not (manifest or {}).get("kind") == SPATIAL_GATE_KIND:
            raise ValueError(f"{name} exists and is not a gate variant — pick another name")
        if not overwrite:
            raise ValueError(f"{name} already exists — pick another name or allow overwrite")
    return d


def job_gate_variant_fit(cap, *, starless: bool, out_name: str, width: int = 32,
                         use_lr: bool = False, steps: int = 2000, batch_size: int = 8,
                         crop: int = 192, learning_rate: float = 2e-3,
                         eval_every: int = 250, holdout: int = 15,
                         blackout_fields: int = SPATIAL_GATE_BLACKOUT_FIELDS, seed: int = 0,
                         members: list[str] | None = None, loss_knees: str = "all",
                         mix_space: str = "linear", num_images: int = 100,
                         target_fwhm_arcsec: float = Config.TARGET_PSF_FWHM_ARCSEC,
                         overwrite: bool = False, compare_after: bool = True) -> dict:
    """Fit a spatial gate into a NAMED variant directory (never production)
    on the regime's validate member cubes (re-inferred when stale), then —
    by default — compare it with the production gate on the test cubes."""
    out_dir = check_new_variant_name(starless, out_name, overwrite=overwrite)
    name = os.path.basename(out_dir)
    knees = sgc.parse_loss_knees(loss_knees)
    target_fwhm = validate_target_fwhm_arcsec(target_fwhm_arcsec)
    (base, records_dir, records_fp, validate_dir, indices, labels,
     target) = _prepare_validate_cubes(cap, starless=starless, num_images=num_images,
                                       target_fwhm=target_fwhm)
    fields, _labels = load_cube_fields(
        validate_dir, records_dir, "validate", target_name=target,
        target_fwhm_arcsec=target_fwhm, indices=indices,
        progress=lambda i, n, label: cap.tick(i, n, label))
    comb = sgc.fit_gate_variant(
        fields, labels, out_dir=out_dir, holdout=int(holdout), seed=int(seed),
        blackout_fields=int(blackout_fields),
        runner=LazyMemberRunner(base, starless=starless, labels=labels),
        blackout_dir=_ensemble_cubes_dir("validate_blackout", starless=starless),
        source_fingerprint=str(records_fp), width=int(width), use_lr=bool(use_lr),
        steps=int(steps), batch_size=int(batch_size), crop=int(crop),
        learning_rate=float(learning_rate), eval_every=int(eval_every),
        members=members, loss_knees=knees, mix_space=mix_space,
        starfull=not starless, records_fp=records_fp,
        extra_meta={"variant": name, "fitted_via": "web", "subset": "validate",
                    "num_images": int(num_images), "members_requested": members or None},
        progress=lambda i, n, label: cap.tick(i, n, label),
        log=lambda message: print(f"[spatial gate {name}] {message}", flush=True))
    result = {"variant": name, "n_members": len(labels),
              "selected": comb.fit_meta.get("selected"), "report_id": None}
    if compare_after:
        gates = [g for g in (COMBINER_MODELS[SPATIAL_GATE_KIND].artifact_dir, name)
                 if os.path.isfile(os.path.join(_regime_dir_ro(starless), g, "combiner.json"))]
        try:
            result["report_id"] = job_combiner_compare(
                cap, starless=starless, gates=gates)["report_id"]
        except (RuntimeError, ValueError) as exc:
            print(f"  ! compare after fit skipped: {exc}")
    return result


# ---- promote --------------------------------------------------------------- #

def job_combiner_promote(cap, *, starless: bool, variant: str, force: bool = False) -> dict:
    """Make a gate variant the production gate. The current production
    artifact is first copied to ``spatial_gate_backup_<UTC stamp>`` (promote
    that backup to roll back). Then the production payload, the test cubes'
    gate outputs, the eval summary and the knee curves are refreshed from the
    cached cubes (no member inference) when the variant fits the cubes.

    A variant that a fit is still writing, or whose fit did not complete
    (``fit_meta.complete`` not true), is always refused
    (:func:`~euclid_polish.eval.spatial_gate_compare.promotion_refusal`). A
    variant that reads a member which is not active is refused unless
    ``force`` (it would make the production model unavailable); members that
    joined after its fit do not matter (a note: refit to consider them)."""
    src = variant_dir(starless, variant)
    name = os.path.basename(src)
    prod_name = COMBINER_MODELS[SPATIAL_GATE_KIND].artifact_dir
    if name == prod_name:
        raise RuntimeError(f"{name} is already the production gate")
    manifest = _read_json_file(os.path.join(src, "combiner.json")) or {}
    refusal = sgc.promotion_refusal(src, manifest)
    if refusal is not None:
        raise RuntimeError(f"not promoting: {refusal}")
    labels = [str(v) for v in manifest.get("member_labels") or []]
    active_members = manifest.get("active_members")
    reads = ([labels[int(i)] for i in active_members if int(i) < len(labels)]
             if isinstance(active_members, list) else labels)
    active = _regime_labels(ensemble_dir(), starless)
    if not reads_available(reads, active) and not force:
        missing = [lb for lb in reads if lb not in active]
        raise RuntimeError(
            f"{name} reads {len(missing)} member(s) that are not active in the "
            f"{_regime_slug(starless)} ensemble ({', '.join(missing[:6])}) — promoting "
            "it would leave no current production model (pass force to promote anyway)")
    joined = joined_after_fit(labels, active)
    if joined:
        print(f"  • note: {len(joined)} member(s) joined after {name}'s fit; "
              "refit to consider them")
    regime_dir = _ensemble_regime_dir(starless)
    prod = os.path.join(regime_dir, prod_name)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    backup = None
    cap.tick(0, 5, "backing up the production gate")
    if os.path.isdir(prod):
        backup = f"{GATE_BACKUP_PREFIX}{stamp}"
        shutil.copytree(prod, os.path.join(regime_dir, backup))
    tmp = os.path.join(regime_dir, f".promote-{stamp}")
    old = os.path.join(regime_dir, f".replaced-{stamp}")
    shutil.copytree(src, tmp)
    meta_path = os.path.join(tmp, "combiner.json")
    tmp_manifest = _read_json_file(meta_path) or {}
    tmp_manifest.setdefault("fit_meta", {}).update({
        "promoted_from": name, "promoted_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "replaced_backup": backup})
    _atomic_json(meta_path, tmp_manifest)
    cap.tick(1, 5, f"installing {name} as production")
    if os.path.isdir(prod):
        os.rename(prod, old)
    try:
        os.rename(tmp, prod)
    except OSError:
        if os.path.isdir(old) and not os.path.exists(prod):
            os.rename(old, prod)            # put the previous production back
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    shutil.rmtree(old, ignore_errors=True)
    result = {"promoted": name, "backup": backup, "test_rescored": False, "summary": None}
    cap.tick(2, 5, "refreshing the production gate payload")
    try:
        compute_combiner_payload(starless, model_kind=SPATIAL_GATE_KIND)
    except Exception as exc:  # noqa: BLE001 — the promotion itself succeeded
        print(f"  ! gate payload not refreshed: {type(exc).__name__}: {exc}")
    cap.tick(3, 5, "re-applying the gate to the cached test cubes")
    try:
        if _apply_combiner_to_test_cubes(starless, SPATIAL_GATE_KIND, progress=cap.tick):
            previous = (_read_eval_summary(starless) or {}).get("eval_identity") or {}
            summary = _reevaluate_from_cached_cubes(
                starless, num_images=previous.get("num_images"), progress=cap.tick)
            result["test_rescored"] = summary is not None
            if summary:
                result["summary"] = {k: summary.get(k) for k in (
                    "spatial_gate_combiner_psnr", "spatial_gate_combiner_vs_mean_db",
                    "spatial_gate_combiner_vs_best_member_db")}
        else:
            print("  • the promoted gate does not fit the cached test cubes — "
                  "re-evaluate to score it")
    except Exception as exc:  # noqa: BLE001
        print(f"  ! test cubes not re-scored: {type(exc).__name__}: {exc}")
    with contextlib.suppress(Exception):
        tracking_default_store().append_log(
            f"Promoted spatial gate `{name}` to production ({_regime_slug(starless)}); "
            f"previous production backed up as `{backup}`.")
    cap.tick(5, 5, "done")
    print(f"  ✓ {name} → {prod_name} (backup {backup})")
    return result


# ---- overview -------------------------------------------------------------- #

def ensemble_overview(starless: bool) -> dict:
    """Models › Leaderboard: headline numbers (each with its definition) and the
    staleness checks, all from local files (fast, offline)."""
    base = ensemble_dir()
    regime_dir = _regime_dir_ro(starless)
    active = _regime_labels(base, starless)
    summary = _read_eval_summary(starless)
    identity = (summary or {}).get("eval_identity") or {}
    rdir = _sky_records_local_dir()
    sub = eval_subset(rdir) if rdir else "test"
    records_fp = _eval_records_fingerprint(rdir, sub, starless=starless)
    knee = knee_psnr_status(starless)
    prod_manifest = _read_json_file(os.path.join(
        regime_dir, COMBINER_MODELS[SPATIAL_GATE_KIND].artifact_dir, "combiner.json"))
    gate_labels = [str(v) for v in (prod_manifest or {}).get("member_labels") or []]
    checks = []

    def check(cid, ok, tone, title, detail, action=None):
        checks.append({"id": cid, "ok": bool(ok), "tone": "good" if ok else tone,
                       "title": title, "detail": detail, "action": action})

    recorded = [str(x) for x in ((summary or {}).get("member_labels")
                                 or (summary or {}).get("per_member_labels") or [])]
    if summary is None:
        check("evaluation", False, "warn", "No evaluation yet",
              "Evaluate the ensemble on the test records.", "evaluate")
    else:
        check("eval-members", recorded == active, "warn", "Evaluation vs members",
              "The evaluation matches the active members." if recorded == active else
              f"Evaluated {len(recorded)} members; {len(active)} are active now.", "evaluate")
        rec_ok = identity.get("records_fp") in (None, records_fp)
        check("eval-records", rec_ok, "warn", "Evaluation vs records",
              "The test records are the ones evaluated." if rec_ok else
              "The test records changed since the evaluation (regenerated or re-synced).",
              "evaluate")
        fps = identity.get("combiner_fps") or {}
        gate_fp = _combiner_fingerprint(regime_dir, SPATIAL_GATE_KIND)
        gate_ok = fps.get(SPATIAL_GATE_KIND) in (None, gate_fp) or not gate_fp
        check("eval-gate", gate_ok, "warn", "Evaluation vs production gate",
              "The production gate is the one evaluated." if gate_ok else
              "The production gate changed (refit or promotion) since the evaluation.",
              "evaluate")
    if prod_manifest is None:
        check("gate", False, "bad", "No production gate",
              "Fit a gate variant and promote it.", "combiners")
    else:
        prod_active = (prod_manifest or {}).get("active_members")
        gate_reads = ([gate_labels[int(i)] for i in prod_active if int(i) < len(gate_labels)]
                      if isinstance(prod_active, list) else gate_labels)
        valid = reads_available(gate_reads, active)
        joined = joined_after_fit(gate_labels, active)
        missing = [lb for lb in gate_reads if lb not in active]
        if not valid:
            detail = (f"It reads {len(missing)} member(s) that are not active "
                      f"({', '.join(missing[:6])}): production falls back to the mean.")
        elif joined:
            detail = (f"Reads {len(gate_reads)} of the {len(active)} active members; "
                      f"{len(joined)} joined after this fit — refit to consider them.")
        else:
            detail = (f"Reads {len(gate_reads)} of the {len(active)} active members."
                      if len(gate_reads) != len(active) else
                      f"Fitted for the {len(active)} active members.")
        check("gate-members", valid and not joined, "info" if valid else "warn",
              "Production gate vs members", detail, "combiners")
    if not knee.get("available"):
        check("knee", False, "warn", "No PSNR-vs-knee curves",
              "Compute them from the cached test cubes.", "knee")
    else:
        check("knee", not knee.get("stale"), "warn", "Knee curves",
              "Current for the cubes and combiners." if not knee.get("stale") else
              "The cubes or combiners changed since the curves were computed.", "knee")
    pending = _pending_archived_members(starless)
    if pending:
        check("archive-pending", False, "info", "Archived members pending",
              f"{', '.join(pending)} left; the next evaluation rebuilds from cached cubes.",
              "evaluate")

    knee_models = {m.get("id"): m for m in knee.get("models", []) or []} if knee.get("available") else {}
    members_knee = [m for m in knee_models.values() if m.get("kind") == "member"]

    def knee_mean(m):
        vals = [v for v in (m or {}).get("integrated") or [] if v is not None]
        return float(np.mean(vals)) if vals else None

    best_knee = max(members_knee, key=lambda m: knee_mean(m) or -1e9, default=None)
    s = dict(summary or {})
    if s and s.get("best_member_psnr") is None:
        # Summaries written before the headline keys: the evals payload's
        # metric block carries the same VIS asinh best member.
        block = ((_read_json_file(os.path.join(regime_dir, "ensemble_evals.json")) or {})
                 .get("combiner") or {})
        s["best_member_psnr"] = block.get("best_member_psnr")
        s["best_member_label"] = s.get("best_member_label") or block.get("best_member_label")
    headline = {
        "metric": s.get("psnr_metric") or ("vis_asinh" if s.get("recomputed_from_cubes") else None),
        "knee_e": s.get("psnr_knee_e", float(Config.STRETCH_SCALE_E)),
        "n_scored": s.get("n_scored"),
        "production": {"psnr": s.get(f"{SPATIAL_GATE_KIND}_combiner_psnr"),
                       "vs_mean_db": s.get(f"{SPATIAL_GATE_KIND}_combiner_vs_mean_db"),
                       "vs_best_member_db": s.get(f"{SPATIAL_GATE_KIND}_combiner_vs_best_member_db")},
        "mean": {"psnr": s.get("ensemble_psnr"),
                 "vs_mean_member_db": s.get("ensemble_vs_mean_member_db", s.get("ensemble_gain_db"))},
        "best_member": {"psnr": s.get("best_member_psnr"), "label": s.get("best_member_label"),
                        "mean_member_psnr": s.get("mean_member_psnr")},
        "knee": {"available": bool(knee.get("available")), "stale": bool(knee.get("stale")),
                 "n_fields": knee.get("n_fields"), "integration": knee.get("integration"),
                 "production": knee_mean(knee_models.get(SPATIAL_GATE_KIND)),
                 "production_bands": (knee_models.get(SPATIAL_GATE_KIND) or {}).get("integrated"),
                 "mean": knee_mean(knee_models.get("ensemble_mean")),
                 "best_member": knee_mean(best_knee),
                 "best_member_label": (best_knee or {}).get("label")},
    }
    return {"regime": _regime_slug(starless), "active_members": active,
            "n_members": len(active), "records_dir": rdir, "eval_subset": sub,
            "test_present": bool(rdir) and os.path.exists(tfrecord_path(rdir, f"dirty_{sub}")),
            "evaluated_at": _iso_mtime(os.path.join(regime_dir, "eval_summary.json")),
            "summary": summary, "headline": headline, "checks": checks,
            "production_gate": {"available": prod_manifest is not None,
                                "n_members": len(gate_labels),
                                "mix_space": (prod_manifest or {}).get("mix_space"),
                                "fitted_at": _iso_mtime(os.path.join(
                                    regime_dir, COMBINER_MODELS[SPATIAL_GATE_KIND].artifact_dir,
                                    "combiner.npz")),
                                "promoted_from": ((prod_manifest or {}).get("fit_meta") or {}).get(
                                    "promoted_from")}}


# ---- train preview --------------------------------------------------------- #

def train_command_preview(form: dict) -> dict:
    """What an ``ensemble_train`` submit with this form would run, without
    touching FASRC: the member names it would allocate (from the local
    registry, tombstones never reused), the array shape and the
    ``train_ensemble.py`` argv. Raises :class:`ValueError` (a
    ``TaskParamError`` included) for a form the submit would refuse."""
    step = STEP_REGISTRY.get("ensemble_train")
    params = step.fill_task_params(dict(form))
    for key, value in job_config.fasrc_params_for("ensemble_train").items():
        params.setdefault(key, value)
    blank_seed = str(params.get("base_seed", "")).strip() in ("", "-1")
    prepared = step.prepare_params(params)
    star = bool(prepared.pop("_star_prior_json", None))
    if star:
        prepared["_star_prior_file"] = "logs/pipeline/<job>.star-population.<sha>.json"
    if blank_seed:
        prepared["base_seed"] = ""
    command = step.build_command(prepared)
    names = (_split_names(prepared.get("member_names"))
             or _split_names(prepared.get("members")))
    array = step.array_shape(prepared)
    return {"ok": True, "mode": str(prepared.get("mode") or "add"),
            "member_names": names, "count": len(names),
            "array": ({"tasks": array[0], "max_parallel": array[1]} if array else None),
            "command": ["python", *command],
            "command_text": shlex.join(["python", *command]),
            "base_seed": None if blank_seed else prepared.get("base_seed"),
            "star_prior": star,
            "params": {k: v for k, v in prepared.items() if not str(k).startswith("_")}}
