# Ensemble-Only Model Architecture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `EnsembleModel` the only public model struct (single model = ensemble of 1), archive the legacy single-model checkpoints into tracking as zips, and add registry-backed per-member archiving with lazy cache invalidation.

**Architecture:** A `registry.json`-style file at `ckpt/ensemble_registry.json` (outside the ensemble dir so the FASRC `--delete-after` mirror can't wipe it) is the source of truth for active members, with tombstones for archived ones. All eval/inference loads go through `load_eval_ensemble()`; disagreement artifacts are gated on `n_members > 1`. Archiving zips a member into the current tracking campaign and tombstones it; caches record membership labels and self-delete on mismatch.

**Tech Stack:** Python 3.12, Flask WebUI, TensorFlow checkpoints, pytest.

**Spec:** `docs/superpowers/specs/2026-07-01-ensemble-only-model-design.md`

Key discovered facts the plan relies on:
- `HSTTrainStep` (step_id `train`, `fasrc_pipeline.py:955`) shells to `scripts/fasrc_train_with_hst.py`, deleted in f1fc0a1 → the step is dead; remove it. `EnsembleTrainStep` is the only real training path.
- `fasrc_mirror.py` rsyncs remote `cfg.ckpt_dir` → local `DEFAULT_CHECKPOINT_DIR` with `--delete-after`. Repoint to ensemble dirs; registry tombstones stop mirrored-back members from re-activating.
- Cube cache files are keyed by *stack position* (`member{i}_*.npy`), so any membership change invalidates the whole cubes dir.
- `ensemble_viz.ensemble_dir()` and `ensemble._default_ensemble_dir()` duplicate the same path logic — unify in `ensemble.py`.

---

### Task 1: Registry module

**Files:**
- Create: `euclid_polish/ensemble_registry.py`
- Test: `tests/test_ensemble_registry.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_ensemble_registry.py
import json
import os

from euclid_polish import ensemble_registry as er


def _mk_member(base, i, *, loss_best=False):
    d = os.path.join(base, f"member_{i:02d}")
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, "checkpoint"), "w").write("x")
    if loss_best:
        lb = os.path.join(d, "loss_best")
        os.makedirs(lb, exist_ok=True)
        open(os.path.join(lb, "checkpoint"), "w").write("x")
    return d


def test_bootstrap_discovers_members_and_persists(tmp_path):
    base = str(tmp_path / "ensemble")
    _mk_member(base, 0)
    _mk_member(base, 1, loss_best=True)
    reg = er.load_registry(base)
    assert reg["active"] == ["member_00", "member_01"]
    assert reg["archived"] == []
    # persisted OUTSIDE the ensemble dir (mirror --delete-after safety)
    assert os.path.isfile(er.registry_path(base))
    assert not er.registry_path(base).startswith(base + os.sep)


def test_missing_dir_dropped_archived_never_reactivated(tmp_path):
    base = str(tmp_path / "ensemble")
    _mk_member(base, 0)
    _mk_member(base, 1)
    er.load_registry(base)
    er.archive_member_entry(base, "member_01", zip_path="z.zip", commit="abc")
    reg = er.load_registry(base)          # dir still on disk (mirror pulled it back)
    assert reg["active"] == ["member_00"]
    assert reg["archived"][0]["name"] == "member_01"
    # a vanished active member is dropped from active on load
    import shutil
    shutil.rmtree(os.path.join(base, "member_00"))
    assert er.load_registry(base)["active"] == []


def test_active_member_dirs_and_labels(tmp_path):
    base = str(tmp_path / "ensemble")
    _mk_member(base, 0, loss_best=True)
    _mk_member(base, 2)
    dirs = er.active_member_dirs(base)
    assert [os.path.basename(d) for d in dirs] == ["member_00", "member_02"]
    assert er.active_labels(base) == ["00·psnr", "00·loss", "02·psnr"]


def test_archive_unknown_member_raises(tmp_path):
    base = str(tmp_path / "ensemble")
    _mk_member(base, 0)
    er.load_registry(base)
    import pytest
    with pytest.raises(ValueError):
        er.archive_member_entry(base, "member_09", zip_path="z", commit=None)
```

- [ ] **Step 2: Run tests, verify they fail** — `pytest tests/test_ensemble_registry.py -x -q` → ImportError.

- [ ] **Step 3: Implement**

```python
# euclid_polish/ensemble_registry.py
"""Source of truth for which ensemble members are active.

The registry file lives at ``<ensemble parent>/ensemble_registry.json`` — one
level ABOVE the ensemble dir on purpose: the FASRC checkpoint auto-mirror
rsyncs the ensemble dir with ``--delete-after``, which would delete any
local-only file inside it.

Bootstrap rule: any ``member_*`` directory with a checkpoint that the registry
has never seen (neither active nor archived) is auto-added to ``active``.
Archived tombstones are permanent — a member dir reappearing on disk (e.g.
mirrored back from FASRC) is NOT re-activated.
"""

from __future__ import annotations

import glob
import os
from datetime import datetime, timezone
from typing import Any

from euclid_polish.tracking._utils import _read_json, _write_json

_MEMBER_GLOB = "member_*"
REGISTRY_FILENAME = "ensemble_registry.json"


def _checkpoint_exists(d: str) -> bool:
    # Mirrors sky_records.checkpoint_present — cheap, no TF import.
    return (os.path.isfile(os.path.join(d, "checkpoint"))
            or bool(glob.glob(os.path.join(d, "*.index"))))


def registry_path(base_dir: str) -> str:
    parent = os.path.dirname(os.path.abspath(base_dir).rstrip("/")) or "."
    return os.path.join(parent, REGISTRY_FILENAME)


def _member_dirs_on_disk(base_dir: str) -> list[str]:
    return sorted(
        d for d in glob.glob(os.path.join(base_dir, _MEMBER_GLOB))
        if os.path.isdir(d) and _checkpoint_exists(d))


def load_registry(base_dir: str) -> dict[str, Any]:
    """Load + bootstrap the registry; persists any change it makes."""
    reg = _read_json(registry_path(base_dir)) or {}
    active = [str(n) for n in reg.get("active", [])]
    archived = list(reg.get("archived", []))
    seen = set(active) | {str(t.get("name")) for t in archived}
    on_disk = {os.path.basename(d) for d in _member_dirs_on_disk(base_dir)}
    changed = False
    for name in sorted(on_disk - seen):          # bootstrap new members
        active.append(name)
        changed = True
    kept = [n for n in active if n in on_disk]   # drop vanished actives
    if kept != active:
        active, changed = kept, True
    out = {"active": sorted(active), "archived": archived}
    if changed or not os.path.isfile(registry_path(base_dir)):
        if on_disk or archived or os.path.isdir(base_dir):
            _write_json(registry_path(base_dir), out)
    return out


def active_member_dirs(base_dir: str) -> list[str]:
    return [os.path.join(base_dir, n) for n in load_registry(base_dir)["active"]]


def active_labels(base_dir: str) -> list[str]:
    """Model labels the ensemble will load, aligned with member order:
    ``NN·psnr`` always, plus ``NN·loss`` when ``loss_best/`` has a checkpoint."""
    labels: list[str] = []
    for d in active_member_dirs(base_dir):
        idx = os.path.basename(d).removeprefix("member_")
        labels.append(f"{idx}·psnr")
        lb = os.path.join(d, "loss_best")
        if os.path.isdir(lb) and _checkpoint_exists(lb):
            labels.append(f"{idx}·loss")
    return labels


def archive_member_entry(base_dir: str, name: str, *, zip_path: str,
                         commit: str | None) -> dict[str, Any]:
    """Move ``name`` from active → archived tombstone. Returns the registry."""
    reg = load_registry(base_dir)
    if name not in reg["active"]:
        raise ValueError(f"{name!r} is not an active ensemble member")
    reg["active"] = [n for n in reg["active"] if n != name]
    reg["archived"].append({
        "name": name,
        "archived_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "zip": zip_path,
        "commit": commit,
    })
    _write_json(registry_path(base_dir), reg)
    return reg
```

(Check `euclid_polish/tracking/_utils.py` exports `_read_json`/`_write_json`; they are imported by `store.py` from there.)

- [ ] **Step 4: Run tests** — `pytest tests/test_ensemble_registry.py -q` → all pass.
- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat(ensemble): member registry with archive tombstones"`

---

### Task 2: Ensemble core reads the registry; `default_ensemble_dir()` unified

**Files:**
- Modify: `euclid_polish/ensemble.py` (discovery in `__init__`, `ensemble_available`, `_default_ensemble_dir` → public, new `upsample_batch`)
- Modify: `euclid_polish/web/helpers/ensemble_viz.py:37-41` (`ensemble_dir()` delegates)
- Test: `tests/test_ensemble_registry.py` (extend), existing ensemble tests must stay green

- [ ] **Step 1: Write failing test** (in `tests/test_ensemble_registry.py`)

```python
def test_ensemble_available_respects_registry(tmp_path, monkeypatch):
    from euclid_polish import ensemble as ens
    base = str(tmp_path / "ensemble")
    _mk_member(base, 0)
    assert ens.ensemble_available(base) is True
    er.load_registry(base)
    er.archive_member_entry(base, "member_00", zip_path="z", commit=None)
    assert ens.ensemble_available(base) is False   # dir on disk, but archived


def test_default_ensemble_dir_is_ckpt_sibling(monkeypatch, tmp_path):
    from euclid_polish import ensemble as ens
    from euclid_polish.config import Config
    monkeypatch.setattr(Config, "DEFAULT_CHECKPOINT_DIR", str(tmp_path / "ckpt/wdsr"))
    assert ens.default_ensemble_dir() == str(tmp_path / "ckpt/ensemble")
```

- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement.** In `ensemble.py`:
  - Rename `_default_ensemble_dir` → `default_ensemble_dir` (public), keep body.
  - `__init__` discovery block: replace the glob two-liner with `dirs = ensemble_registry.active_member_dirs(base_dir)`; keep the `n_members` cap and the rest (loss_best loading unchanged). Filter still guards `_checkpoint_exists`.
  - `ensemble_available()`: `return bool(ensemble_registry.load_registry(base_dir or default_ensemble_dir())["active"])` — but keep the checkpoint check via `active_member_dirs` (registry already verifies on-disk presence).
  - `load_ensemble()`: use `default_ensemble_dir()` instead of the inline duplicate.
  - Add batch upsampling for the sky route (mirrors `Model.upsample_batch` semantics):

```python
    def upsample_batch(self, lr_images, *, on_progress=None, log=None):
        """Ensemble-mean SR for every image (list of :class:`Image`)."""
        self._require_members()
        lr_list = list(lr_images)
        out = []
        for i, lr in enumerate(lr_list):
            out.append(self.upsample(lr))
            if on_progress is not None:
                on_progress(i + 1, len(lr_list), f"field {lr.index}")
        return out
```

  - In `ensemble_viz.py`: `def ensemble_dir(): return default_ensemble_dir()` (import from `euclid_polish.ensemble`).
- [ ] **Step 4: Run** — `pytest tests/test_ensemble_registry.py tests/test_ensemble*.py -q` (plus any existing ensemble tests found via `pytest -q -k ensemble`).
- [ ] **Step 5: Commit** — `git commit -m "refactor(ensemble): registry-driven member discovery, public default_ensemble_dir"`

---

### Task 3: Tracking zip archive

**Files:**
- Modify: `euclid_polish/tracking/store.py` (new method after `backup_model`, ~line 336)
- Test: `tests/test_tracking_store_zip.py`

- [ ] **Step 1: Failing test**

```python
# tests/test_tracking_store_zip.py
import os
import zipfile

from euclid_polish.tracking.store import TrackingStore


def _store(tmp_path):
    s = TrackingStore(str(tmp_path / "tracking"))
    s.start_campaign("zip test")          # confirm actual API name in store.py
    return s


def test_archive_model_zip_roundtrip(tmp_path):
    src = tmp_path / "member_00"
    (src / "loss_best").mkdir(parents=True)
    (src / "checkpoint").write_text("root")
    (src / "ckpt-5.index").write_text("idx")
    (src / "loss_best" / "checkpoint").write_text("lb")
    s = _store(tmp_path)
    meta = s.archive_model_zip(str(src), "ensemble-member_00", comment="bye")
    zpath = os.path.join(s.current_dir, "models", meta["name"])
    assert meta["name"].endswith(".zip") and os.path.isfile(zpath)
    with zipfile.ZipFile(zpath) as z:
        names = set(z.namelist())
    assert {"checkpoint", "ckpt-5.index", "loss_best/checkpoint"} <= names
    assert meta["kind"] == "model-zip" and meta["size_bytes"] > 0


def test_archive_model_zip_missing_src(tmp_path):
    import pytest
    from euclid_polish.tracking.store import TrackingError
    s = _store(tmp_path)
    with pytest.raises(TrackingError):
        s.archive_model_zip(str(tmp_path / "nope"), "x")
```

(Adjust the campaign-creation call to the real method name — check `store.py` for it, e.g. `new_campaign`/`start_campaign`, and whether `TrackingStore(...)` takes a root path; mimic `tests/test_tracking_routes.py` fixtures.)

- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement** in `store.py`:

```python
    def archive_model_zip(self, src_dir: str, name: str,
                          comment: str = "") -> dict[str, Any]:
        """Zip a checkpoint directory tree into ``current/models/<name>.zip``.

        Unlike :meth:`backup_model` (restorable loose copy of the primary
        tracks), this captures the FULL tree verbatim — used when a model is
        retired (ensemble-member archive / single-model migration), so the
        source dir can be deleted afterwards.
        """
        import zipfile
        with _LOCK:
            self._require_current()
            if not os.path.isdir(src_dir):
                raise TrackingError(f"source dir not found: {src_dir}")
            stem = _slugify(name, default="model")
            dest_dir = os.path.join(self.current_dir, "models")
            os.makedirs(dest_dir, exist_ok=True)
            dest = _unique_path(dest_dir, stem + ".zip")
            files: list[str] = []
            with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as zf:
                for dp, _dirs, fns in os.walk(src_dir):
                    for fn in sorted(fns):
                        full = os.path.join(dp, fn)
                        rel = os.path.relpath(full, src_dir)
                        zf.write(full, rel)
                        files.append(rel)
            if not files:
                os.remove(dest)
                raise TrackingError(f"nothing to archive under {src_dir}")
            meta = self._record_meta(
                kind="model-zip", name=os.path.basename(dest), comment=comment,
                source_path=src_dir, files=files,
                size_bytes=os.path.getsize(dest))
            _write_json(dest + ".meta.json", meta)
            return meta
```

Also update `_backups_in` (store.py:344-367) so `models/` picks up zip sidecars: in the models loop, additionally scan for `*.zip.meta.json` files in `models_dir` (like the fits/images loop) and append them. The tracking page then lists archived zips alongside dir backups. Guard `model_backup_dir`'s time-travel path: zips have no dir — time-travel button should not render for `kind == "model-zip"` (template check in `tracking.html` where backups render, line ~88: wrap the ⏱ button in `{% if m.kind != 'model-zip' %}`).

- [ ] **Step 4: Run** — `pytest tests/test_tracking_store_zip.py tests/test_tracking_routes.py -q`.
- [ ] **Step 5: Commit** — `git commit -m "feat(tracking): archive_model_zip — zip a checkpoint tree into the campaign"`

---

### Task 4: Ensemble-only eval loader + `sr_from_model` gating

**Files:**
- Modify: `euclid_polish/eval/ensemble_infer.py` (replace `load_eval_ensemble_or_single`)
- Modify: `euclid_polish/eval/grouped_runner.py:105,138-158`
- Modify: `euclid_polish/eval/catalog_runner.py:64-80,283,308-309,350-370` (delete `load_eval_model`)
- Modify: `euclid_polish/web/routes/evaluation.py:283` (log line)
- Modify: `scripts/eval_catalog.py`, `scripts/eval_grouped.py` (`--checkpoint` → `--ensemble-dir`)
- Test: rewrite `tests/test_grouped_runner_model_default.py`

- [ ] **Step 1: Rewrite the loader test** (failing):

```python
# tests/test_grouped_runner_model_default.py  (replace file)
"""load_eval_ensemble(): always an EnsembleModel; clear error when empty."""
import pytest

from euclid_polish.eval import ensemble_infer as ei


class _FakeEns:
    def __init__(self, n): self.n_members = n


def test_returns_ensemble(monkeypatch):
    monkeypatch.setattr("euclid_polish.ensemble.load_ensemble",
                        lambda **kw: _FakeEns(3))
    out = ei.load_eval_ensemble(log=lambda m: None)
    assert out.n_members == 3


def test_zero_members_raises(monkeypatch):
    monkeypatch.setattr("euclid_polish.ensemble.load_ensemble",
                        lambda **kw: _FakeEns(0))
    with pytest.raises(RuntimeError, match="no active ensemble members"):
        ei.load_eval_ensemble(log=lambda m: None)


def test_sr_from_model_hides_members_for_singleton():
    import numpy as np

    class _One:
        n_members = 1
        def member_arrays(self, lr):
            return np.zeros((1, 4, 4, 1), np.float32)
    _lr_vis, _sr, members = ei.sr_from_model(_One(), np.zeros((2, 2, 1)))
    assert members is None                      # single member → no std cubes
```

- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement.** `ensemble_infer.py` becomes:

```python
"""Ensemble inference for the evaluators: mean SR + (multi-member) stack."""

from __future__ import annotations

from typing import Any

import numpy as np

from euclid_polish.config import Config


def sr_from_model(model: Any, lr_cube: np.ndarray
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """``(lr_vis, sr, members)`` for one LR cube.

    ``sr`` is the ensemble mean. ``members`` is the ``(M, H, W, C)`` stack when
    the ensemble has >1 model (disagreement is meaningful), else ``None`` so a
    1-member ensemble writes no all-zero std/PCA cubes.
    """
    lr = np.asarray(lr_cube, dtype=np.float32)
    members = np.asarray(model.member_arrays(lr), dtype=np.float32)
    sr = members.mean(axis=0)
    lr_vis = lr[..., 0] if lr.ndim == 3 else lr
    return lr_vis, sr, (members if getattr(model, "n_members", 1) > 1 else None)


def load_eval_ensemble(base_dir: str | None = None,
                       num_res_blocks: int | None = None, *, log=None) -> Any:
    """THE eval model: the ensemble under ``base_dir`` (default location).

    Raises RuntimeError when the registry has no active members — there is no
    single-model fallback anymore (a lone model is an ensemble of 1).
    """
    emit = log or (lambda m: None)
    from euclid_polish.ensemble import load_ensemble
    ens = load_ensemble(base_dir, num_res_blocks=num_res_blocks
                        or Config.DEFAULT_NUM_RES_BLOCKS)
    if ens.n_members < 1:
        raise RuntimeError(
            "no active ensemble members — train one (scripts/train_ensemble.py"
            " --n-members 1 works) or pull members on the /ensemble page.")
    emit(f"using ensemble mean ({ens.n_members} models)")
    return ens
```

  - `grouped_runner.py`: param `checkpoint` → `ensemble_dir` (keep position/keyword compat by renaming at the signature and the one internal use); drop `checkpoint = checkpoint or Config.DEFAULT_CHECKPOINT_DIR`; `want_disagreement = (model.n_members > 1 if model is not None else len(active_labels(default_ensemble_dir())) > 1)`; the load becomes `model = load_eval_ensemble(ensemble_dir, num_res_blocks, log=_emit)`. Grep for `run_grouped_analysis(` callers (`web/routes/evaluation.py`, `scripts/eval_grouped.py`) and update the kwarg.
  - `catalog_runner.py`: same rename in `run_catalog_eval`; `eval_catalog_object` reuse check → `require_disagreement=model.n_members > 1`; delete `load_eval_model` (only remaining caller was the old fallback); the `checkpoint` arg it forwards to `reconstruct_cutout_at(..., checkpoint_dir=...)` becomes the ensemble base dir (only used for provenance labeling — verify with grep inside `reconstruct_cutout_at`).
  - `evaluation.py:283` log line → `cap.write("model: ensemble mean (registry-active members)\n")`.
  - Scripts: `--checkpoint` → `--ensemble-dir`, default `None` (loader resolves).
- [ ] **Step 4: Run** — `pytest tests/test_grouped_runner_model_default.py -q` + `pytest -q -k "grouped or catalog_runner or evaluation" `.
- [ ] **Step 5: Commit** — `git commit -m "refactor(eval): ensemble-only loader; disagreement gated on n_members>1"`

---

### Task 5: Membership fingerprint in eval-object reuse

**Files:**
- Modify: `euclid_polish/eval/disagreement.py` (write `members.json` beside the cubes)
- Modify: `euclid_polish/eval/catalog_runner.py` `can_reuse_eval_object`
- Modify: `euclid_polish/eval/grouped_runner.py` / `synthetic_runner.py` (pass labels)
- Test: `tests/test_eval_reuse_fingerprint.py`

- [ ] **Step 1: Failing test**

```python
# tests/test_eval_reuse_fingerprint.py
import json
import os

from euclid_polish.eval.catalog_runner import can_reuse_eval_object


def _touch(d, *names):
    os.makedirs(d, exist_ok=True)
    for n in names:
        with open(os.path.join(d, n), "wb") as f:
            f.write(b"x")


def test_reuse_requires_matching_member_labels(tmp_path):
    d = str(tmp_path / "obj")
    _touch(d, "original_stack.fits", "SR.fits", "std.fits", "pca0.fits")
    labels = ["00·psnr", "01·psnr"]
    # no members.json yet → stale (pre-fingerprint outputs) when labels demanded
    assert not can_reuse_eval_object(d, require_disagreement=True,
                                     member_labels=labels)
    with open(os.path.join(d, "members.json"), "w") as f:
        json.dump({"member_labels": labels}, f)
    assert can_reuse_eval_object(d, require_disagreement=True,
                                 member_labels=labels)
    assert not can_reuse_eval_object(d, require_disagreement=True,
                                     member_labels=["00·psnr"])


def test_single_member_reuse_unchanged(tmp_path):
    d = str(tmp_path / "obj")
    _touch(d, "original_stack.fits", "SR.fits")
    assert can_reuse_eval_object(d, require_disagreement=False)
```

- [ ] **Step 2: Run, verify fail** (unexpected kwarg).
- [ ] **Step 3: Implement.**
  - `can_reuse_eval_object(obj_dir, *, require_disagreement=False, member_labels=None)`: after the existing file checks, when `require_disagreement and member_labels is not None`, also require `members.json` present with `member_labels == list(member_labels)` (json read wrapped, any error → False).
  - `write_disagreement_cubes(out_dir, members, *, member_labels=None)` in `disagreement.py`: after writing cubes, when labels given dump `{"member_labels": member_labels}` to `members.json`.
  - Thread labels: `grouped_runner` computes `labels = model.member_labels if model is not None else active_labels(default_ensemble_dir())` once and passes `member_labels=labels if want_disagreement else None` into its `_reusable`; `catalog_runner.eval_catalog_object` passes `model.member_labels` when `model.n_members > 1`; `jobs_impl.reconstruct_cutout_at` (line ~502) and `synthetic_runner.py` (line ~316) pass `member_labels=model.member_labels` / the loaded stack's labels into `write_disagreement_cubes`. For the synthetic cache-reuse path the labels come from the cube-cache manifest (Task 6 returns them).
- [ ] **Step 4: Run** — `pytest tests/test_eval_reuse_fingerprint.py -q` and touched suites.
- [ ] **Step 5: Commit** — `git commit -m "feat(eval): membership fingerprint gates eval-object reuse"`

---

### Task 6: Lazy cube-cache invalidation

**Files:**
- Modify: `euclid_polish/eval/ensemble_cube_cache.py`
- Modify: `euclid_polish/eval/synthetic_runner.py:252-257` (use new return)
- Modify: `euclid_polish/web/helpers/ensemble_viz.py` (`ensemble_status` marks stale summary)
- Test: `tests/test_ensemble_cube_cache.py` (extend or create)

- [ ] **Step 1: Failing test**

```python
# tests/test_ensemble_cube_cache.py (add)
import json
import os

import numpy as np

from euclid_polish.eval import ensemble_cube_cache as ecc


def _write_cache(d, labels, idx=3):
    os.makedirs(d, exist_ok=True)
    for i in range(len(labels)):
        np.save(os.path.join(d, f"member{i}_{idx:05d}.npy"),
                np.zeros((4, 4, 1), np.float32))
    with open(os.path.join(d, "viz_index.json"), "w") as f:
        json.dump({"subset": "test", "indices": [idx],
                   "member_labels": labels}, f)


def test_stale_membership_deletes_cache(tmp_path):
    d = str(tmp_path / "cubes")
    _write_cache(d, ["00·psnr", "01·psnr"])
    out = ecc.load_cached_member_stack(
        3, subset="test", cubes_dir=d, active_labels=["00·psnr"])
    assert out is None
    assert not os.path.isfile(os.path.join(d, "viz_index.json"))  # purged
    assert not any(f.startswith("member") for f in os.listdir(d) if os.path.isdir(d)) \
        if os.path.isdir(d) else True


def test_matching_membership_hits(tmp_path):
    d = str(tmp_path / "cubes")
    _write_cache(d, ["00·psnr"])
    out = ecc.load_cached_member_stack(
        3, subset="test", cubes_dir=d, active_labels=["00·psnr"])
    assert out is not None and out.shape[0] == 1
```

- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement.** In `ensemble_cube_cache.py`:
  - Signature: `load_cached_member_stack(field_index, *, subset, cubes_dir=None, active_labels=None) -> np.ndarray | None`; add `cached_member_labels(cubes_dir=None) -> list[str] | None` (reads manifest labels; used by synthetic runner for fingerprints).
  - When `active_labels is None`, compute from the registry: `active_labels = ensemble_registry.active_labels(default_ensemble_dir())`.
  - After loading the manifest: `if list(man.get("member_labels", [])) != list(active_labels): shutil.rmtree(d, ignore_errors=True); return None` — the lazy delete the user asked for (whole dir: files are position-keyed).
  - `synthetic_runner.py`: pass nothing extra (default registry path is right); where it reuses `cached`, fetch labels via `cached_member_labels()` for the `write_disagreement_cubes(member_labels=...)` call from Task 5.
  - `ensemble_viz.ensemble_status()`: after loading `eval_summary.json`, compare `summary.get("member_labels") or summary.get("per_member_labels")` with `active_labels(base)`; add `"eval_summary_stale": bool(mismatch)` to the returned dict (UI in Task 8 renders a badge). Same check for `regenerate_power_spectrum` is unnecessary — it reads the cubes dir, which self-purges via the loader path; but guard it: if manifest labels mismatch registry, return None.
- [ ] **Step 4: Run** — `pytest tests/test_ensemble_cube_cache.py -q` + synthetic runner tests.
- [ ] **Step 5: Commit** — `git commit -m "feat(eval): cube cache self-purges when ensemble membership changes"`

---

### Task 7: Archive-member job + route

**Files:**
- Modify: `euclid_polish/web/helpers/ensemble_viz.py` (new `job_archive_member`)
- Modify: `euclid_polish/web/routes/ensemble.py` (new POST route)
- Test: `tests/test_ensemble_archive_route.py`

- [ ] **Step 1: Failing test** (pattern-match the Flask client fixtures in `tests/test_web.py`):

```python
# tests/test_ensemble_archive_route.py
import os

import numpy as np


def test_archive_member_zips_tombstones_and_purges(tmp_path, monkeypatch):
    from euclid_polish import ensemble_registry as er
    from euclid_polish.config import Config
    from euclid_polish.web.helpers import ensemble_viz as ev

    monkeypatch.setattr(Config, "DEFAULT_CHECKPOINT_DIR",
                        str(tmp_path / "ckpt/wdsr"))
    monkeypatch.setattr(Config, "VIS_DIR", str(tmp_path / "vis"))
    monkeypatch.setattr(Config, "TRACKING_DIR", str(tmp_path / "tracking"))
    base = ev.ensemble_dir()
    for i in (0, 1):
        d = os.path.join(base, f"member_{i:02d}")
        os.makedirs(d)
        open(os.path.join(d, "checkpoint"), "w").write("x")
    # a cube cache exists → must be purged (position-keyed)
    cubes = os.path.join(Config.VIS_DIR, "ensemble", "cubes")
    os.makedirs(cubes)
    open(os.path.join(cubes, "viz_index.json"), "w").write("{}")

    # tracking store needs an active campaign (create via default store API)
    from euclid_polish.tracking.store import TrackingStore
    # ... instantiate/start per the real store API and monkeypatch
    #     ensemble_viz's store accessor to it

    class _Cap:  # job capture stub
        def tick(self, *a): pass
        def write(self, *a): pass

    out = ev.job_archive_member(_Cap(), name="member_01")
    assert not os.path.isdir(os.path.join(base, "member_01"))
    assert out["zip"].endswith(".zip")
    reg = er.load_registry(base)
    assert reg["active"] == ["member_00"]
    assert reg["archived"][0]["name"] == "member_01"
    assert not os.path.isdir(cubes)             # eager purge
```

- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement.** In `ensemble_viz.py`:

```python
def job_archive_member(cap, *, name: str) -> dict:
    """Retire one ensemble member: zip → tracking, tombstone, delete, purge."""
    import re
    from euclid_polish import ensemble_registry
    from euclid_polish.provenance.gitinfo import capture_git
    from euclid_polish.tracking.defaults import default_store  # confirm accessor

    if not re.fullmatch(r"member_\d{2,}", name or ""):
        raise RuntimeError(f"invalid member name {name!r}")
    base = ensemble_dir()
    reg = ensemble_registry.load_registry(base)
    if name not in reg["active"]:
        raise RuntimeError(f"{name} is not an active member")
    src = os.path.join(base, name)
    store = default_store()
    cap.tick(0, 3, f"zipping {name}")
    meta = store.archive_model_zip(src, f"ensemble-{name}",
                                   comment=f"archived from ensemble ({base})")
    commit = (capture_git(cwd=os.getcwd()) or {}).get("short")
    cap.tick(1, 3, "updating registry")
    ensemble_registry.archive_member_entry(
        base, name, zip_path=os.path.join("models", meta["name"]),
        commit=commit)
    cap.tick(2, 3, "deleting member dir + caches")
    shutil.rmtree(src, ignore_errors=True)
    shutil.rmtree(_ensemble_cubes_dir(), ignore_errors=True)   # position-keyed
    print(f"  ✓ {name} → tracking {meta['name']}; caches purged")
    return {"zip": meta["name"], "member": name}
```

(Resolve the real tracking-store accessor — grep `tracking_default_store` in `web/routes/tracking.py` and import the same one. Add a friendly error when no campaign is active: catch `TrackingError` and re-raise with "start a campaign on /tracking first".) Add a log-line into the campaign log via `store.append_log(...)` noting the archive + FASRC-side reminder.

Route in `routes/ensemble.py`:

```python
    @app.route("/ensemble/archive-member", methods=["POST"])
    def ensemble_archive_member():
        name = (request.form.get("member") or "").strip()
        job_id = REGISTRY.spawn(
            f"ensemble: archive {name} → tracking",
            target=lambda cap: job_archive_member(cap, name=name),
        )
        return jsonify({"job_id": job_id})
```

- [ ] **Step 4: Run** — `pytest tests/test_ensemble_archive_route.py -q`.
- [ ] **Step 5: Commit** — `git commit -m "feat(ensemble): archive-member job — zip to tracking, tombstone, purge caches"`

---

### Task 8: Ensemble page = model manager (+ fold /training in)

**Files:**
- Modify: `euclid_polish/web/templates/ensemble.html` (member rows: archive button; archived history; TFRecords card; train-step hint)
- Modify: `euclid_polish/web/helpers/ensemble_viz.py` (`ensemble_status`: per-member `size_mb`, `labels`, plus `archived` tombstones + `tfrecords`)
- Modify: `euclid_polish/web/routes/model.py` (`/training` → redirect; delete training page handler)
- Delete: `euclid_polish/web/templates/training.html`
- Modify: `euclid_polish/web/templates/base.html` (drop the Training nav link if present; check `grep -rn training base.html`)
- Test: `tests/test_web.py` (update `/training` expectations), quick route smoke test

- [ ] **Step 1: Failing test** — add to an appropriate web test module:

```python
def test_training_redirects_to_ensemble(client):
    r = client.get("/training")
    assert r.status_code in (301, 302) and "/ensemble" in r.headers["Location"]
```

- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement.**
  - `ensemble_status()` additions: per member `size_mb` (walk dir), and top-level `archived: reg["archived"]` + `tfrecords: _tfrecords_status()`.
  - `ensemble.html`: members list becomes a table — name · seed · tracks (psnr/+loss) · size · **📦 archive** button posting `member=<name>` to `/ensemble/archive-member` with a `confirm()` naming the zip destination; collapsed `<details>` "Archived members" section from tombstones (name, date, zip, commit); a TFRecords card (copy the file-list markup pattern from `training.html` before deleting it); keep the existing `ensemble_train` step card (it is already the only working train path). Render the `eval_summary_stale` badge next to the eval table when set ("membership changed since this eval — re-run Evaluate").
  - `routes/model.py`: replace the training page handler with `return redirect("/ensemble", code=302)` (`from flask import redirect`); remove the now-unused `_tfrecords_status`/`_checkpoints_status` imports *if* unused after Task 9.
  - Delete `training.html`; grep for `url_for('training_page')`/`"/training"` references (`base.html`, docs) and update.
- [ ] **Step 4: Run** — `pytest tests/test_web.py -q`; then `python scripts/serve.py` smoke + `curl -sI localhost:PORT/training | head -3` if a quick manual check is cheap.
- [ ] **Step 5: Commit** — `git commit -m "feat(ensemble page): member manager with archive; fold /training in"`

---

### Task 9: Inference + sky + status go ensemble-only

**Files:**
- Modify: `euclid_polish/web/routes/model.py:157-222` (drop `ckpt_kind`/`vis_only`; jobs take no ckpt dir)
- Modify: `euclid_polish/web/helpers/jobs_impl.py` (`_job_generate_reconstruct`, `_job_reconstruct_euclid_cutout` load the ensemble; `reconstruct(model, ...)` call → `sr_from_model`)
- Modify: `euclid_polish/web/templates/inference.html:160-235` (remove ckpt-kind/vis-only selectors; show member count)
- Modify: `euclid_polish/web/helpers/status.py` (`_checkpoints_status` walks ensemble members; delete `_ckpt_dir_for_kind`)
- Modify: `euclid_polish/web/routes/views.py:195-235` (`api_sky_generate_sr` uses the ensemble; `view_training_log` drops the `-vis` special case)
- Modify: `euclid_polish/web/helpers/sky_records.py` (`checkpoint_present` → ensemble-aware)
- Modify: `euclid_polish/web/routes/files.py` (status uses new `_checkpoints_status`)
- Test: `tests/test_web.py` (vis-only tests removed/replaced)

- [ ] **Step 1: Adjust tests first.** In `tests/test_web.py`: delete `test_view_training_log_reads_vis_only_dir` and `test_delete_model_vis_only_targets_vis_dir` (delete-model route goes away in Task 10); update any `_ckpt_dir_for_kind` tests to the new discovery. Add:

```python
def test_checkpoints_status_lists_ensemble_members(tmp_path, monkeypatch):
    from euclid_polish.config import Config
    from euclid_polish.web.helpers.status import _checkpoints_status
    monkeypatch.setattr(Config, "DEFAULT_CHECKPOINT_DIR",
                        str(tmp_path / "ckpt/wdsr"))
    m = tmp_path / "ckpt/ensemble/member_00"
    (m / "loss_best").mkdir(parents=True)
    (m / "checkpoint").write_text("x")
    (m / "ckpt-3.index").write_text("x")
    (m / "loss_best" / "checkpoint").write_text("x")
    out = _checkpoints_status()
    members = {f["member"] for f in out["files"]}
    assert members == {"member_00"}
    assert any(f["subdir"] == "loss_best" for f in out["files"])
```

- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement.**
  - `_checkpoints_status()`: root = `default_ensemble_dir()`; walk each `active_member_dirs()`; entry keys: `name`, `rel`, `member` (replaces `variant`), `subdir`, `folder`, `size_mb`.
  - Delete `_ckpt_dir_for_kind` (grep confirms only `routes/model.py` imports it).
  - `jobs_impl._job_generate_reconstruct`: drop `checkpoint_dir`/`num_res_blocks` params; replace the `load_model_from_checkpoint` block with `model = load_eval_ensemble(log=...)`; replace `lr_data, sr_data = reconstruct(model, lr_img.data)` with `lr_data, sr_data, _members = sr_from_model(model, lr_img.data)` (VIS plane semantics match — verify `reconstruct` returns the VIS-only lr; `sr_from_model` does the same).
  - `_job_reconstruct_euclid_cutout`: same — drop the `tf.train.latest_checkpoint` gate + keras load; `model = load_eval_ensemble()`; `checkpoint_dir=` arg fed to `reconstruct_cutout_at` becomes `default_ensemble_dir()`.
  - `routes/model.py`: both POST handlers stop reading `ckpt_kind`/`vis_only` and stop passing ckpt args into the jobs.
  - `views.py api_sky_generate_sr`: gate on `ensemble_available()`; `model = load_eval_ensemble()`; `model.upsample_batch(lr, on_progress=..., log=...)` — EnsembleModel's version accepts these kwargs (Task 2); error string → "no active ensemble members".
  - `sky_records.checkpoint_present()` → keep name, body: `from euclid_polish.ensemble import ensemble_available; return ensemble_available()` when no explicit `checkpoint` given (explicit arg keeps old dir check for callers that pass one — grep callers).
  - `views.py view_training_log`: remove the `-vis` branch (png name always `training_log.png`).
- [ ] **Step 4: Run** — `pytest tests/test_web.py tests/test_inference_shapes.py -q`.
- [ ] **Step 5: Commit** — `git commit -m "refactor(web): inference + sky + status are ensemble-only"`

---

### Task 10: FASRC cleanup — dead train step, mirror repoint, delete-model removal

**Files:**
- Modify: `euclid_polish/web/fasrc_pipeline.py` (delete `HSTTrainStep`, its registration; check the registry list near `EXPERIMENTAL_LANES_ENABLED` at line ~188)
- Modify: `euclid_polish/web/fasrc_mirror.py:50-95` (mirror remote/local = ensemble dirs)
- Modify: `euclid_polish/web/routes/fasrc.py` (delete `/api/fasrc/delete-model` route; drop `vis_only` in the training-log fetch at ~1224)
- Modify: `euclid_polish/web/templates/fasrc.html` (drop vis toggles at lines ~190, ~578; relabel mirror copy)
- Test: `tests/test_web.py` (delete-model tests removed in Task 9 step 1); pipeline-step tests if any (`grep -rn "HSTTrainStep\|step_id=\"train\"" tests/`)

- [ ] **Step 1: Failing check** — `grep -rn "fasrc_train_with_hst" euclid_polish/` should end empty after the change; add/adjust a step-registry test if one exists (`pytest -q -k fasrc_pipeline`).
- [ ] **Step 2: Implement.**
  - Delete the `HSTTrainStep` class (fasrc_pipeline.py:955-1042) and remove it from the step registry (find where instances are listed, near the bottom of the module or in `fasrc.py`); the comment block at line ~1105 ("training now goes exclusively through HSTTrainStep") → "…through EnsembleTrainStep".
  - `fasrc_mirror._sync_once` + `start`: `remote_base = remote_ensemble_dir()` (import from `ensemble_viz` — or inline the same sibling-of-`cfg.ckpt_dir` logic to avoid a web-helper import cycle; check imports) and `local_base = default_ensemble_dir()`; `cfg.local_ckpt_mirror` override keeps working when set. Keep `--delete-after` (registry lives outside the dir — Task 1).
  - Remove `api_fasrc_delete_model` wholesale (superseded by per-member archive; single-ckpt dirs no longer exist locally). Grep `delete-model` in templates/JS (`training.html` was the only consumer; it is deleted).
  - `fasrc.py` training-log fetch: remove `vis_only` query handling; log path = remote ensemble member? Simplest: keep it reading `cfg.ckpt_dir`'s log for legacy runs but drop only the `-vis` branch.
- [ ] **Step 3: Run** — `pytest -q -k "fasrc or web"`.
- [ ] **Step 4: Commit** — `git commit -m "cleanup(fasrc): drop dead single-model train step + delete-model; mirror ensemble dir"`

---

### Task 11: Migration script + config comment

**Files:**
- Create: `scripts/migrate_single_model.py`
- Modify: `euclid_polish/config.py:589-591` (comment: the dir is now only the path anchor for `ensemble/`)
- Test: `tests/test_migrate_single_model.py`

- [ ] **Step 1: Failing test**

```python
# tests/test_migrate_single_model.py
import os


def test_migrate_zips_and_deletes(tmp_path, monkeypatch):
    from euclid_polish.config import Config
    monkeypatch.setattr(Config, "DEFAULT_CHECKPOINT_DIR",
                        str(tmp_path / "ckpt/wdsr"))
    monkeypatch.setattr(Config, "TRACKING_DIR", str(tmp_path / "tracking"))
    for d in ("ckpt/wdsr", "ckpt/wdsr-vis"):
        p = tmp_path / d
        p.mkdir(parents=True)
        (p / "checkpoint").write_text("x")
    from scripts.migrate_single_model import migrate
    out = migrate()                       # creates campaign if none active
    assert not (tmp_path / "ckpt/wdsr").exists()
    assert not (tmp_path / "ckpt/wdsr-vis").exists()
    assert len(out["archived"]) == 2
    out2 = migrate()                      # idempotent
    assert out2["archived"] == []
```

- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement** `scripts/migrate_single_model.py`:

```python
#!/usr/bin/env python
"""One-shot: archive the legacy single-model checkpoints into tracking.

Zips ``Config.DEFAULT_CHECKPOINT_DIR`` (and its ``-vis`` sibling) into the
current tracking campaign via ``TrackingStore.archive_model_zip``, logs a
campaign note (incl. the FASRC-side cleanup reminder), then deletes the local
dirs. Idempotent: already-missing dirs are skipped. Run once after the
ensemble-only refactor lands.
"""
from __future__ import annotations

import os
import shutil

from euclid_polish.config import Config


def migrate() -> dict:
    # import here so the test's Config monkeypatching applies
    from euclid_polish.web.routes.tracking import ...  # use the same
    # default-store accessor as job_archive_member (Task 7); if no campaign
    # is active, start one named "single-model retirement".
    store = ...
    archived = []
    for d, tag in ((Config.DEFAULT_CHECKPOINT_DIR, "wdsr-single-model"),
                   (Config.DEFAULT_CHECKPOINT_DIR.rstrip("/") + "-vis",
                    "wdsr-vis-single-model")):
        if not os.path.isdir(d):
            continue
        meta = store.archive_model_zip(
            d, tag, comment="ensemble-only migration: single model retired")
        shutil.rmtree(d)
        archived.append(meta["name"])
    if archived:
        store.append_log(
            "Ensemble-only migration: archived " + ", ".join(archived)
            + ". REMINDER: the FASRC-side single-model ckpt dir "
              "(cfg.ckpt_dir) still exists remotely — remove it manually "
              "when convenient.")
    return {"archived": archived}


if __name__ == "__main__":
    out = migrate()
    print(out or "nothing to migrate")
```

(Fill the store accessor from the real code; auto-start a campaign when none is active — check `TrackingStore` for the create API and `has_current()`.)

- [ ] **Step 4: Run tests.**
- [ ] **Step 5: Commit** — `git commit -m "feat: single-model → tracking migration script"`

---

### Task 12: Full test sweep, run migration, docs, push

- [ ] **Step 1:** `pytest -q` full suite; fix fallout (expect: tests importing `load_eval_ensemble_or_single`, `_ckpt_dir_for_kind`, training-page templates).
- [ ] **Step 2:** Run the migration locally: `python scripts/migrate_single_model.py` (verify `ckpt/wdsr*` gone, zips in `tracking/current/models/`, campaign log updated). `ls ckpt/` should show only `ensemble/` (+ registry after first load).
- [ ] **Step 3:** Update `README.md` / docs mentions of `/training` and single-model checkpoints (`grep -rn "ckpt/wdsr\|/training" README.md docs/ --include=*.md | grep -v superpowers`).
- [ ] **Step 4:** Commit + push; suggest `scripts/track.py sync` to mirror the new zips to holylabs.
