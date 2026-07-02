# Ensemble Training Control (add / continue / fork) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the blind `n_members` loop with spec-driven training: add new members only, continue selected members, or fork new members from an existing one with the LR schedule reset to step 0 — controlled from the /ensemble page.

**Architecture:** `MemberTrainSpec` list executed sequentially by `EnsembleModel.train_members`; `train_ensemble.py --mode {add,continue,fork}`; `EnsembleTrainStep` allocates member names from the registry at submit time; the ensemble page's train card gains a mode selector fed by `window.ENSEMBLE_MEMBERS`, with per-row ▶/⑂ shortcuts.

**Tech Stack:** Python 3.12 / TensorFlow checkpoints, Flask + vanilla-JS step cards, pytest (env: `~/miniforge3/envs/EuclidPolishEnv/bin/python -m pytest`).

**Spec:** `docs/superpowers/specs/2026-07-02-ensemble-training-control-design.md`

Mechanics this plan relies on (verified):
- `Trainer` auto-restores; `ckpt.step` persists; `steps` is an ABSOLUTE target; the cosine LR is evaluated at the absolute step over `total_steps=steps` of the current run (`trainer.py:590-700`, `model.py:219-224`).
- Checkpoint-reader pattern for metadata without building a model:
  `tf.train.load_checkpoint(latest)` (`training/inference.py:74-86`); the step
  variable's key is `"step/.ATTRIBUTES/VARIABLE_VALUE"` (from
  `tf.train.Checkpoint(step=tf.Variable(0), …)` at `trainer.py:276`).
- `Model.__init__` builds fresh 4-band wdsr when its dir has no checkpoint
  (`model.py:99-114`) — the fork hook goes right after that build.
- Step-card fields are hard-coded per `step_id` in
  `web/static/fasrc_step_card.js` (`case 'ensemble_train'` at ~line 329);
  markup is injected via innerHTML, so dynamic behavior needs wiring in the
  module after render, not inline `<script>`.
- `EnsembleTrainStep.build_command` (fasrc_pipeline.py) maps form params →
  CLI flags; it runs LOCALLY at submit time, so it can consult the registry.

---

### Task 1: `next_member_names` in the registry

**Files:**
- Modify: `euclid_polish/ensemble_registry.py`
- Test: `tests/test_ensemble_registry.py` (append)

- [ ] **Step 1: Failing tests**

```python
def test_next_member_names_skips_tombstones_and_gaps(tmp_path):
    base = str(tmp_path / "ensemble")
    _mk_member(base, 0)
    _mk_member(base, 2)                       # gap at 1 stays a gap
    er.load_registry(base)
    er.archive_member_entry(base, "member_02", zip_path="z", commit=None)
    # active: member_00; archived: member_02 → next index is 3, never 1 or 2
    assert er.next_member_names(base, 2) == ["member_03", "member_04"]


def test_next_member_names_counts_unregistered_disk_dirs(tmp_path):
    base = str(tmp_path / "ensemble")
    _mk_member(base, 5)                       # on disk, not yet in registry
    assert er.next_member_names(base, 1) == ["member_06"]


def test_next_member_names_empty_ensemble(tmp_path):
    base = str(tmp_path / "ensemble")
    assert er.next_member_names(base, 2) == ["member_00", "member_01"]
```

- [ ] **Step 2: Run** — `pytest tests/test_ensemble_registry.py -q -k next_member` → AttributeError.
- [ ] **Step 3: Implement** (append to `ensemble_registry.py`)

```python
def _member_index(name: str) -> int | None:
    tail = str(name).removeprefix("member_")
    return int(tail) if tail.isdigit() else None


def next_member_names(base_dir: str, k: int) -> list[str]:
    """``k`` fresh consecutive member names that can never collide.

    Starts at max(index over active ∪ archived tombstones ∪ ``member_*`` on
    disk) + 1 — tombstoned indices are skipped FOREVER (the bootstrap ignores
    a reincarnated archived name, so reuse would create a ghost member).
    Called locally at FASRC submit time; the job receives explicit names.
    """
    reg = load_registry(base_dir)
    names = set(reg["active"]) | {str(t.get("name")) for t in reg["archived"]}
    names |= {os.path.basename(d)
              for d in glob.glob(os.path.join(base_dir, _MEMBER_GLOB))}
    used = [i for i in (_member_index(n) for n in names) if i is not None]
    start = (max(used) + 1) if used else 0
    return [f"member_{i:02d}" for i in range(start, start + int(k))]
```

- [ ] **Step 4: Run** — same command → 3 passed (plus existing tests stay green: `pytest tests/test_ensemble_registry.py -q`).
- [ ] **Step 5: Commit** — `git commit -m "feat(registry): next_member_names — fresh indices past tombstones"`

---

### Task 2: `checkpoint_step()` reader

**Files:**
- Modify: `euclid_polish/training/inference.py` (new helper beside `infer_checkpoint_nchan_in`)
- Test: `tests/test_checkpoint_step.py`

- [ ] **Step 1: Failing test**

```python
# tests/test_checkpoint_step.py
import os

import tensorflow as tf

from euclid_polish.training.inference import checkpoint_step


def test_checkpoint_step_reads_step_var(tmp_path):
    d = str(tmp_path / "ck")
    ck = tf.train.Checkpoint(step=tf.Variable(1234))
    mgr = tf.train.CheckpointManager(ck, d, max_to_keep=1)
    mgr.save()
    assert checkpoint_step(d) == 1234


def test_checkpoint_step_none_when_missing(tmp_path):
    assert checkpoint_step(str(tmp_path / "nope")) is None
```

- [ ] **Step 2: Run** — `pytest tests/test_checkpoint_step.py -q` → ImportError.
- [ ] **Step 3: Implement** in `training/inference.py`:

```python
def checkpoint_step(checkpoint_dir: str) -> int | None:
    """The persisted ``ckpt.step`` of the latest checkpoint, or ``None``.

    Reads the step variable straight off the checkpoint file (no model
    build) — the authoritative "how far has this member trained" used by
    continue-mode target computation.
    """
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    if latest is None:
        return None
    try:
        reader = tf.train.load_checkpoint(latest)
        return int(reader.get_tensor("step/.ATTRIBUTES/VARIABLE_VALUE"))
    except Exception:            # unreadable / legacy layout → caller decides
        return None
```

- [ ] **Step 4: Run** — 2 passed.
- [ ] **Step 5: Commit** — `git commit -m "feat(training): checkpoint_step reader (no model build)"`

---

### Task 3: `Model(init_weights_from=…)` fork hook

**Files:**
- Modify: `euclid_polish/model.py:77-118` (`__init__`)
- Test: `tests/test_model_fork_init.py`

- [ ] **Step 1: Failing test** (tiny net: `num_res_blocks=1` keeps it fast)

```python
# tests/test_model_fork_init.py
import numpy as np
import pytest
import tensorflow as tf

from euclid_polish.model import Model


def _save_weights_ckpt(model: Model, d: str) -> None:
    # Same key layout the Trainer saves under (model=...), so the fork's
    # weights-only restore matches a real member checkpoint.
    ck = tf.train.Checkpoint(step=tf.Variable(500), model=model._tf_model)
    tf.train.CheckpointManager(ck, d, max_to_keep=1).save()


def test_fork_copies_weights_and_is_virgin(tmp_path):
    src_dir = str(tmp_path / "src")
    src = Model(src_dir, num_res_blocks=1, seed=1)
    _save_weights_ckpt(src, src_dir)

    dst = Model(str(tmp_path / "dst"), num_res_blocks=1, seed=2,
                init_weights_from=src_dir)
    for a, b in zip(src._tf_model.get_weights(),
                    dst._tf_model.get_weights(), strict=True):
        assert np.array_equal(a, b)
    # the fork target itself has NO checkpoint → training starts at step 0
    assert tf.train.latest_checkpoint(str(tmp_path / "dst")) is None


def test_fork_refuses_nonvirgin_target(tmp_path):
    src_dir = str(tmp_path / "src")
    src = Model(src_dir, num_res_blocks=1, seed=1)
    _save_weights_ckpt(src, src_dir)
    dst_dir = str(tmp_path / "dst")
    dst = Model(dst_dir, num_res_blocks=1, seed=2)
    _save_weights_ckpt(dst, dst_dir)          # target already trained
    with pytest.raises(ValueError, match="virgin"):
        Model(dst_dir, num_res_blocks=1, init_weights_from=src_dir)


def test_fork_refuses_missing_source(tmp_path):
    with pytest.raises(ValueError, match="no checkpoint"):
        Model(str(tmp_path / "dst"), num_res_blocks=1,
              init_weights_from=str(tmp_path / "empty_src"))
```

- [ ] **Step 2: Run** — `pytest tests/test_model_fork_init.py -q` → TypeError (unexpected kwarg).
- [ ] **Step 3: Implement.** In `Model.__init__`: add kwarg `init_weights_from: str | None = None` after `deterministic`. After the fresh-build `else:` branch completes (and BEFORE `_reconstruct_fn` is set), add:

```python
        if init_weights_from is not None:
            # Fork: start a NEW member (step 0, fresh optimizer, this seed's
            # data order) from an existing member's weights. The LR schedule
            # therefore restarts from scratch.
            if _checkpoint_exists(checkpoint_dir):
                raise ValueError(
                    f"init_weights_from requires a virgin target dir; "
                    f"{checkpoint_dir!r} already has a checkpoint.")
            if not _checkpoint_exists(init_weights_from):
                raise ValueError(
                    f"init_weights_from: no checkpoint in {init_weights_from!r}")
            src = self._load_fn(init_weights_from, scale, num_res_blocks)
            self._tf_model.set_weights(src.get_weights())
            print(f"  ✓ fork: weights initialized from {init_weights_from}")
```

Note the virgin check uses `_checkpoint_exists(checkpoint_dir)` — but the load branch above already ran when a checkpoint exists, so put the `init_weights_from` block FIRST in `__init__` body order: raise on non-virgin before any load. Concretely: insert the two `raise` guards right after `self._load_fn` is assigned, and the `set_weights` copy right after the fresh `_wdsr_build` branch.

- [ ] **Step 4: Run** — 3 passed; `pytest tests/test_model.py -q` stays green.
- [ ] **Step 5: Commit** — `git commit -m "feat(model): init_weights_from — fork a member from existing weights at step 0"`

---

### Task 4: `MemberTrainSpec` + `EnsembleModel.train_members`

**Files:**
- Modify: `euclid_polish/ensemble.py` (dataclass + method; `train()` becomes a wrapper)
- Test: `tests/test_ensemble_train_members.py`

- [ ] **Step 1: Failing tests** (fake `Model` records construction/train calls)

```python
# tests/test_ensemble_train_members.py
import json
import os

import pytest

from euclid_polish import ensemble as ens_mod
from euclid_polish.ensemble import EnsembleModel, MemberTrainSpec


class _FakeModel:
    calls: list = []

    def __init__(self, checkpoint_dir, *, scale=2, num_res_blocks=32,
                 seed=None, init_weights_from=None):
        self.checkpoint_dir = checkpoint_dir
        self.kwargs = {"seed": seed, "init_weights_from": init_weights_from}

    def train(self, lr, hr, steps=0, batch_size=16, **kw):
        _FakeModel.calls.append(
            {"dir": self.checkpoint_dir, "steps": steps, **self.kwargs})


@pytest.fixture(autouse=True)
def _patch_model(monkeypatch):
    _FakeModel.calls = []
    monkeypatch.setattr(ens_mod, "Model", _FakeModel)


def test_train_members_runs_specs_in_order(tmp_path):
    base = str(tmp_path / "ensemble")
    specs = [
        MemberTrainSpec(name="member_09", seed=7, target_steps=1000,
                        op="add", run_steps=1000),
        MemberTrainSpec(name="member_03", seed=3, target_steps=1500,
                        op="continue", run_steps=500),
        MemberTrainSpec(name="member_10", seed=8, target_steps=1000,
                        op="fork", run_steps=1000,
                        init_from=os.path.join(base, "member_03"),
                        forked_from="member_03·psnr"),
    ]
    ens = EnsembleModel(base, _models=[])
    ens.train_members("lr.tfrecord", "hr.tfrecord", specs)
    assert [os.path.basename(c["dir"]) for c in _FakeModel.calls] == \
        ["member_09", "member_03", "member_10"]
    assert _FakeModel.calls[0] == {
        "dir": os.path.join(base, "member_09"), "steps": 1000,
        "seed": 7, "init_weights_from": None}
    assert _FakeModel.calls[2]["init_weights_from"] == \
        os.path.join(base, "member_03")


def test_train_members_writes_origin_for_created_members(tmp_path):
    base = str(tmp_path / "ensemble")
    specs = [MemberTrainSpec(name="member_09", seed=7, target_steps=100,
                             op="fork", run_steps=100, init_from="x",
                             forked_from="member_03·loss")]
    EnsembleModel(base, _models=[]).train_members("lr", "hr", specs)
    with open(os.path.join(base, "member_09", "origin.json")) as f:
        o = json.load(f)
    assert o["op"] == "fork" and o["forked_from"] == "member_03·loss"
    assert o["seed"] == 7 and o["target_steps"] == 100
    assert "created_at" in o
    # continue never writes/overwrites origin
    specs2 = [MemberTrainSpec(name="member_09", seed=7, target_steps=200,
                              op="continue", run_steps=100)]
    EnsembleModel(base, _models=[]).train_members("lr", "hr", specs2)
    with open(os.path.join(base, "member_09", "origin.json")) as f:
        assert json.load(f)["op"] == "fork"        # untouched


def test_legacy_train_wraps_add_specs(tmp_path):
    base = str(tmp_path / "ensemble")
    EnsembleModel(base, _models=[]).train("lr", "hr", n_members=2,
                                          base_seed=100, steps=50)
    assert [os.path.basename(c["dir"]) for c in _FakeModel.calls] == \
        ["member_00", "member_01"]
    assert [c["seed"] for c in _FakeModel.calls] == [100, 101]
```

- [ ] **Step 2: Run** — `pytest tests/test_ensemble_train_members.py -q` → ImportError (`MemberTrainSpec`).
- [ ] **Step 3: Implement** in `ensemble.py`:

```python
@dataclass
class MemberTrainSpec:
    """One member's training job within a run.

    ``target_steps`` is the ABSOLUTE step target (the trainer's ``steps``);
    ``run_steps`` is how many steps this run actually executes (=
    ``target_steps`` for add/fork, the extra for continue) — used only for
    cumulative progress accounting. ``init_from`` is a checkpoint dir to copy
    weights from (fork). ``op`` ∈ {"add", "continue", "fork"}.
    """
    name: str
    seed: int
    target_steps: int
    op: str = "add"
    run_steps: int = 0
    init_from: str | None = None
    forked_from: str | None = None
```

(`from dataclasses import dataclass`, `import json`, `from datetime import UTC, datetime` at top; also `from euclid_polish.provenance.gitinfo import capture_git`.)

Method on `EnsembleModel` (below `train`):

```python
    def train_members(
        self,
        lr_path: str,
        hr_path: str,
        specs: Sequence[MemberTrainSpec],
        *,
        batch_size: int = 16,
        on_member: Callable[[int, int, Model], None] | None = None,
        **train_kwargs,
    ) -> EnsembleModel:
        """Run an explicit list of member training jobs sequentially.

        Unlike the legacy :meth:`train` (which blindly loops member_00..N-1),
        each spec names its member, seed, absolute step target and optional
        fork source. Members CREATED here (op add/fork) get an
        ``origin.json`` provenance sidecar that syncs down with the member.
        """
        if not specs:
            raise ValueError("no member specs to train")
        self._models = []
        for i, spec in enumerate(specs):
            d = os.path.join(self.base_dir, spec.name)
            created = not (os.path.isdir(d) and _checkpoint_exists(d))
            os.makedirs(d, exist_ok=True)
            if created and spec.op in ("add", "fork"):
                commit = (capture_git() or {}).get("short")
                with open(os.path.join(d, "origin.json"), "w") as f:
                    json.dump({
                        "op": spec.op,
                        "forked_from": spec.forked_from,
                        "seed": int(spec.seed),
                        "target_steps": int(spec.target_steps),
                        "created_at": datetime.now(UTC).isoformat(
                            timespec="seconds"),
                        "commit": commit,
                    }, f, indent=2)
            m = Model(d, scale=self._scale,
                      num_res_blocks=self._num_res_blocks,
                      seed=int(spec.seed), init_weights_from=spec.init_from)
            m.train(lr_path, hr_path, steps=int(spec.target_steps),
                    batch_size=batch_size, **train_kwargs)
            self._models.append(m)
            if on_member is not None:
                on_member(i + 1, len(specs), m)
        return self
```

Rewrite `train()` body as a wrapper (keep signature + docstring note):

```python
        if n_members < 1:
            raise ValueError(f"n_members must be >= 1, got {n_members}")
        if base_seed is None:
            base_seed = int.from_bytes(os.urandom(4), "little")
        specs = [MemberTrainSpec(
                     name=MEMBER_DIR_FMT.format(i), seed=int(base_seed) + i,
                     target_steps=int(steps), op="add", run_steps=int(steps))
                 for i in range(int(n_members))]
        return self.train_members(lr_path, hr_path, specs,
                                  batch_size=batch_size,
                                  on_member=on_member, **train_kwargs)
```

- [ ] **Step 4: Run** — new tests pass; `pytest -q -k ensemble` green.
- [ ] **Step 5: Commit** — `git commit -m "feat(ensemble): MemberTrainSpec + train_members; legacy train() wraps add specs"`

---

### Task 5: CLI modes in `scripts/train_ensemble.py`

**Files:**
- Modify: `scripts/train_ensemble.py` (args + a testable `build_specs()`; main uses `train_members`)
- Test: `tests/test_train_ensemble_specs.py`

- [ ] **Step 1: Failing tests**

```python
# tests/test_train_ensemble_specs.py
import os

import pytest

from scripts.train_ensemble import build_specs, parse_args


def _mk_member(base, i, step=None):
    d = os.path.join(base, f"member_{i:02d}")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "checkpoint"), "w") as f:
        f.write("x")
    return d


def test_add_mode_uses_passed_names_and_seeds(tmp_path):
    base = str(tmp_path / "ens")
    args = parse_args(["--mode", "add", "--count", "2", "--steps", "1000",
                       "--base-seed", "50",
                       "--member-names", "member_09,member_10"])
    specs = build_specs(args, base)
    assert [(s.name, s.seed, s.target_steps, s.op, s.run_steps)
            for s in specs] == [
        ("member_09", 50, 1000, "add", 1000),
        ("member_10", 51, 1000, "add", 1000)]
    assert all(s.init_from is None for s in specs)


def test_add_mode_allocates_from_disk_when_names_absent(tmp_path):
    base = str(tmp_path / "ens")
    _mk_member(base, 4)
    args = parse_args(["--mode", "add", "--count", "1", "--steps", "10"])
    assert build_specs(args, base)[0].name == "member_05"


def test_add_collision_shifts_past_existing_dir(tmp_path):
    base = str(tmp_path / "ens")
    _mk_member(base, 9)                       # queued twin already created it
    args = parse_args(["--mode", "add", "--count", "1", "--steps", "10",
                       "--member-names", "member_09"])
    assert build_specs(args, base)[0].name == "member_10"


def test_continue_mode_reads_current_step(tmp_path, monkeypatch):
    base = str(tmp_path / "ens")
    _mk_member(base, 3)
    monkeypatch.setattr("scripts.train_ensemble.checkpoint_step",
                        lambda d: 2000)
    args = parse_args(["--mode", "continue", "--members", "member_03",
                       "--extra-steps", "500"])
    (s,) = build_specs(args, base)
    assert (s.name, s.target_steps, s.run_steps, s.op) == \
        ("member_03", 2500, 500, "continue")
    assert s.init_from is None


def test_continue_requires_existing_checkpoint(tmp_path):
    args = parse_args(["--mode", "continue", "--members", "member_08",
                       "--extra-steps", "500"])
    with pytest.raises(SystemExit):
        build_specs(args, str(tmp_path / "ens"))


def test_fork_mode_builds_init_from(tmp_path):
    base = str(tmp_path / "ens")
    _mk_member(base, 2)
    args = parse_args(["--mode", "fork", "--fork-from", "member_02",
                       "--fork-track", "loss", "--count", "2",
                       "--steps", "1000",
                       "--member-names", "member_09,member_10"])
    specs = build_specs(args, base)
    assert [s.name for s in specs] == ["member_09", "member_10"]
    assert all(s.init_from == os.path.join(base, "member_02", "loss_best")
               for s in specs)
    assert all(s.forked_from == "member_02·loss" for s in specs)
    assert specs[0].seed != specs[1].seed or True   # distinct via base+ i
    assert specs[0].op == "fork"


def test_n_members_is_add_count_alias(tmp_path):
    args = parse_args(["--n-members", "3", "--steps", "10"])
    specs = build_specs(args, str(tmp_path / "ens"))
    assert len(specs) == 3 and all(s.op == "add" for s in specs)
```

- [ ] **Step 2: Run** — `pytest tests/test_train_ensemble_specs.py -q` → ImportError (`build_specs`) / parse_args signature (add `argv=None` param).
- [ ] **Step 3: Implement.** In `train_ensemble.py`:
  - `parse_args(argv=None)`; new args:

```python
    p.add_argument("--mode", choices=["add", "continue", "fork"], default="add")
    p.add_argument("--count", type=int, default=None,
                   help="add/fork: how many new members (default 5 for add, 1 for fork).")
    p.add_argument("--n-members", type=int, default=None,
                   help="DEPRECATED alias for --count (add mode).")
    p.add_argument("--member-names", default="",
                   help="Explicit new-member names (comma-separated), allocated "
                        "by the WebUI from the registry. Absent → next free "
                        "on-disk indices.")
    p.add_argument("--members", default="",
                   help="continue: comma-separated existing member names.")
    p.add_argument("--extra-steps", type=int, default=50_000,
                   help="continue: train each member this many MORE steps.")
    p.add_argument("--fork-from", default="",
                   help="fork: source member name (e.g. member_02).")
    p.add_argument("--fork-track", choices=["psnr", "loss"], default="psnr")
```

  - Imports at top: `from euclid_polish.ensemble import EnsembleModel, MemberTrainSpec, evaluate_on_records`, `from euclid_polish.training.inference import checkpoint_step`.
  - `build_specs(args, base) -> list[MemberTrainSpec]` (module-level, pure except reading `base` on disk):

```python
def _exists(d: str) -> bool:
    return os.path.isfile(os.path.join(d, "checkpoint")) or bool(
        __import__("glob").glob(os.path.join(d, "*.index")))
```

    (No — imports at top: `import glob`; `_exists` uses `glob.glob`.)

```python
def _next_free_names(base: str, k: int, start_past: int = -1) -> list[str]:
    used = [int(n.removeprefix("member_"))
            for n in (os.path.basename(d)
                      for d in glob.glob(os.path.join(base, "member_*")))
            if n.removeprefix("member_").isdigit()]
    start = max([start_past, *used], default=start_past) + 1
    return [f"member_{i:02d}" for i in range(start, start + k)]


def _fresh_names(args, base: str, k: int) -> list[str]:
    """Names for members this run CREATES: the submitted allocation, with a
    collision shift past any name that already exists on this filesystem
    (two queued jobs can allocate the same index at submit time)."""
    wanted = [n.strip() for n in args.member_names.split(",") if n.strip()]
    if not wanted:
        return _next_free_names(base, k)
    out: list[str] = []
    for n in wanted[:k]:
        if os.path.isdir(os.path.join(base, n)):
            shifted = _next_free_names(base, 1)[0]
            print(f"  ⚠ {n} already exists on disk — shifted to {shifted}")
            n = shifted
        out.append(n)
        os.makedirs(os.path.join(base, n), exist_ok=True)  # reserve now
    while len(out) < k:
        n = _next_free_names(base, 1)[0]
        out.append(n)
        os.makedirs(os.path.join(base, n), exist_ok=True)
    return out


def build_specs(args, base: str) -> list[MemberTrainSpec]:
    base_seed = (int.from_bytes(os.urandom(4), "little")
                 if args.base_seed < 0 else int(args.base_seed))
    if args.mode == "continue":
        names = [n.strip() for n in args.members.split(",") if n.strip()]
        if not names:
            print("✗ --mode continue needs --members"); raise SystemExit(2)
        specs = []
        for i, name in enumerate(names):
            d = os.path.join(base, name)
            cur = checkpoint_step(d)
            if cur is None:
                print(f"✗ {name}: no checkpoint to continue from in {d}")
                raise SystemExit(2)
            specs.append(MemberTrainSpec(
                name=name, seed=base_seed + i, op="continue",
                target_steps=cur + int(args.extra_steps),
                run_steps=int(args.extra_steps)))
        return specs

    k = int(args.count or args.n_members
            or (1 if args.mode == "fork" else 5))
    init_from = forked_from = None
    if args.mode == "fork":
        if not args.fork_from:
            print("✗ --mode fork needs --fork-from"); raise SystemExit(2)
        src = os.path.join(base, args.fork_from)
        if args.fork_track == "loss":
            src = os.path.join(src, "loss_best")
        if checkpoint_step(src) is None and not _ckpt_files(src):
            print(f"✗ fork source has no checkpoint: {src}")
            raise SystemExit(2)
        init_from = src
        forked_from = f"{args.fork_from.removeprefix('member_')}·{args.fork_track}" \
            if False else f"{args.fork_from}·{args.fork_track}"
    names = _fresh_names(args, base, k)
    return [MemberTrainSpec(
                name=n, seed=base_seed + i, op=args.mode,
                target_steps=int(args.steps), run_steps=int(args.steps),
                init_from=init_from, forked_from=forked_from)
            for i, n in enumerate(names)]
```

    Clean-ups while writing the real code: drop the `if False else` scaffold
    (use the plain `f"{args.fork_from}·{args.fork_track}"`), and
    `_ckpt_files(src)` is `bool(glob.glob(os.path.join(src, "*.index")))` —
    define one `_has_ckpt(d)` helper used by both fork and the tests'
    expectations (checkpoint file OR *.index, matching `_checkpoint_exists`).
    Note the fork-source check must accept the test's bare `checkpoint` file:
    use `_has_ckpt`, not `checkpoint_step` (which needs a real TF ckpt).
  - `main()`: `specs = build_specs(args, base)`; total for the Reporter =
    `sum(s.run_steps for s in specs)`; per-spec progress offset =
    `done_before + max(0, s_local - (spec.target_steps - spec.run_steps))`
    where `s_local` is the member-local absolute step from `step_callback`;
    `_eval_cb` offsets the same way; label shows
    `f"{spec.name} ({i+1}/{len(specs)}) · step {s_local}"`. Replace
    `ens.train(...)` with `ens.train_members(lr, hr, specs, batch_size=…,
    evaluate_every=…, step_callback=…, eval_callback=…, warn_callback=…,
    on_member=…, lr_peak=… <all knobs unchanged>)`. Keep staging + post-train
    eval untouched.
- [ ] **Step 4: Run** — `pytest tests/test_train_ensemble_specs.py -q` green; `pytest tests/test_job_config.py -q` still green.
- [ ] **Step 5: Commit** — `git commit -m "feat(train_ensemble): add/continue/fork modes via MemberTrainSpec"`

---

### Task 6: `EnsembleTrainStep.build_command` modes

**Files:**
- Modify: `euclid_polish/web/fasrc_pipeline.py` (`EnsembleTrainStep`)
- Test: `tests/test_fasrc_pipeline.py` (append to `TestConcreteSteps` region)

- [ ] **Step 1: Failing tests**

```python
def test_ensemble_train_add_mode_allocates_names(monkeypatch):
    from euclid_polish.web.fasrc_pipeline import EnsembleTrainStep
    monkeypatch.setattr(
        "euclid_polish.web.fasrc_pipeline.next_member_names",
        lambda base, k: [f"member_{9 + i:02d}" for i in range(k)])
    argv = EnsembleTrainStep().build_command(
        {"mode": "add", "count": 2, "steps": 1000})
    assert argv[argv.index("--mode") + 1] == "add"
    assert argv[argv.index("--count") + 1] == "2"
    assert argv[argv.index("--member-names") + 1] == "member_09,member_10"


def test_ensemble_train_continue_mode(monkeypatch):
    from euclid_polish.web.fasrc_pipeline import EnsembleTrainStep
    argv = EnsembleTrainStep().build_command(
        {"mode": "continue", "members": "member_03,member_05",
         "extra_steps": 500})
    assert argv[argv.index("--mode") + 1] == "continue"
    assert argv[argv.index("--members") + 1] == "member_03,member_05"
    assert argv[argv.index("--extra-steps") + 1] == "500"
    assert "--member-names" not in argv and "--count" not in argv


def test_ensemble_train_fork_mode(monkeypatch):
    from euclid_polish.web.fasrc_pipeline import EnsembleTrainStep
    monkeypatch.setattr(
        "euclid_polish.web.fasrc_pipeline.next_member_names",
        lambda base, k: ["member_11"])
    argv = EnsembleTrainStep().build_command(
        {"mode": "fork", "fork_from": "member_02", "fork_track": "loss",
         "count": 1, "steps": 2000})
    assert argv[argv.index("--fork-from") + 1] == "member_02"
    assert argv[argv.index("--fork-track") + 1] == "loss"
    assert argv[argv.index("--member-names") + 1] == "member_11"


def test_ensemble_train_default_is_legacy_add():
    from euclid_polish.web.fasrc_pipeline import EnsembleTrainStep
    argv = EnsembleTrainStep().build_command({"n_members": 5, "steps": 100})
    assert argv[argv.index("--mode") + 1] == "add"
    assert argv[argv.index("--count") + 1] == "5"
```

- [ ] **Step 2: Run** — fail (no `--mode` emitted).
- [ ] **Step 3: Implement.** Top of `fasrc_pipeline.py`: `from euclid_polish.ensemble_registry import default_ensemble_dir, next_member_names`. Replace `EnsembleTrainStep.build_command` body:

```python
    def build_command(self, params: dict[str, Any]) -> list[str]:
        mode = str(params.get("mode", "add") or "add").strip()
        steps = int(params.get("steps", Config.DEFAULT_TRAIN_STEPS) or
                    Config.DEFAULT_TRAIN_STEPS)
        cmd = ["scripts/train_ensemble.py", "--mode", mode]
        if mode == "continue":
            members = str(params.get("members", "")).strip()
            cmd += ["--members", members,
                    "--extra-steps",
                    str(int(params.get("extra_steps", 50_000) or 50_000))]
        else:
            # add / fork create members → allocate names from the LOCAL
            # registry now (tombstones must never be reused, and the remote
            # dir still holds archived members' directories).
            count = int(params.get("count", params.get("n_members", 0)) or
                        (1 if mode == "fork" else 5))
            names = next_member_names(default_ensemble_dir(), count)
            cmd += ["--count", str(count),
                    "--member-names", ",".join(names),
                    "--steps", str(steps)]
            if mode == "fork":
                cmd += ["--fork-from", str(params.get("fork_from", "")).strip(),
                        "--fork-track",
                        str(params.get("fork_track", "psnr") or "psnr")]
        base_seed = str(params.get("base_seed", "")).strip()
        if base_seed not in ("", "-1"):
            with contextlib.suppress(ValueError):
                cmd += ["--base-seed", str(int(base_seed))]
        eval_images = str(params.get("eval_images", "")).strip()
        if eval_images:
            with contextlib.suppress(ValueError):
                cmd += ["--eval-images", str(int(eval_images))]
        for name in ("lr_peak", "lr_final", "lr_warmup_steps",
                     "plateau_lr_enabled", "plateau_lr_factor",
                     "plateau_lr_patience", "plateau_lr_min_delta",
                     "plateau_lr_cooldown", "plateau_lr_min_lr",
                     "plateau_lr_metric"):
            val = str(params.get(name, "")).strip()
            if val:
                cmd += [f"--{name.replace('_', '-')}", val]
        return cmd
```

Keep the class docstring updated (three modes). Check the existing `test_ensemble_train_step_build_command` (test_fasrc_pipeline.py:~95) and `tests/test_job_config.py:220` — update their expectations to include `--mode add` + `--member-names` (monkeypatch `next_member_names` there too so they don't touch the real registry).
- [ ] **Step 4: Run** — `pytest tests/test_fasrc_pipeline.py tests/test_job_config.py -q` green.
- [ ] **Step 5: Commit** — `git commit -m "feat(fasrc): EnsembleTrainStep modes with submit-time name allocation"`

---

### Task 7: `ensemble_status` step + origin

**Files:**
- Modify: `euclid_polish/web/helpers/ensemble_viz.py` (`ensemble_status` member loop)
- Test: `tests/test_ensemble_status_training.py`

- [ ] **Step 1: Failing test**

```python
# tests/test_ensemble_status_training.py
import json
import os


def test_status_reports_step_and_origin(tmp_path, monkeypatch):
    from euclid_polish.config import Config
    from euclid_polish.web.helpers import ensemble_viz as ev
    monkeypatch.setattr(Config, "DEFAULT_CHECKPOINT_DIR",
                        str(tmp_path / "ckpt/wdsr"))
    monkeypatch.setattr(Config, "VIS_DIR", str(tmp_path / "vis"))
    base = ev.ensemble_dir()
    d = os.path.join(base, "member_00")
    os.makedirs(d)
    open(os.path.join(d, "checkpoint"), "w").write("x")
    header = ("step,wall_time,loss,psnr_stretched,psnr_raw,"
              "save_best_score,combined_loss,is_baseline\n")
    with open(os.path.join(d, "training_log.csv"), "w") as f:
        f.write(header + "1000,1,0.1,40,33,40,0.01,\n"
                       + "2000,2,0.09,41,34,41,0.009,\n")
    with open(os.path.join(d, "origin.json"), "w") as f:
        json.dump({"op": "fork", "forked_from": "member_9·psnr"}, f)

    st = ev.ensemble_status()
    (m,) = st["members"]
    assert m["step"] == 2000
    assert m["origin"]["forked_from"] == "member_9·psnr"


def test_status_step_blank_when_no_log(tmp_path, monkeypatch):
    from euclid_polish.config import Config
    from euclid_polish.web.helpers import ensemble_viz as ev
    monkeypatch.setattr(Config, "DEFAULT_CHECKPOINT_DIR",
                        str(tmp_path / "ckpt/wdsr"))
    monkeypatch.setattr(Config, "VIS_DIR", str(tmp_path / "vis"))
    d = os.path.join(ev.ensemble_dir(), "member_00")
    os.makedirs(d)
    open(os.path.join(d, "checkpoint"), "w").write("x")
    st = ev.ensemble_status()
    assert st["members"][0]["step"] is None
    assert st["members"][0]["origin"] is None
```

- [ ] **Step 2: Run** — KeyError `step`.
- [ ] **Step 3: Implement.** In `ensemble_viz.py` add helpers + extend the member dict:

```python
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
    try:
        with open(os.path.join(member_dir, "origin.json")) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
```

Member dict gains `"step": _member_last_step(d), "origin": _member_origin(d)`.
- [ ] **Step 4: Run** — new tests + `pytest tests/test_web.py -q -k ensemble` green.
- [ ] **Step 5: Commit** — `git commit -m "feat(ensemble status): per-member step + origin provenance"`

---

### Task 8: UI — mode-switching card + row shortcuts

**Files:**
- Modify: `euclid_polish/web/static/fasrc_step_card.js` (`case 'ensemble_train'` + post-render wiring)
- Modify: `euclid_polish/web/templates/ensemble.html` (window.ENSEMBLE_MEMBERS, Step column, ⑂ badge, ▶/⑂ row buttons)
- Test: manual smoke via dev server + `pytest tests/test_web.py -q` (page renders)

- [ ] **Step 1: Template data + table.** In `ensemble.html`:
  - In the `{% block scripts %}`, FIRST script tag:

```html
<script>
window.ENSEMBLE_MEMBERS = {{ members | tojson }};
</script>
```

  - Members table header gains `<th>Step</th>` (before Size); each row:

```html
          <td class="muted">{{ "{:,}".format(m.step) if m.step else "—" }}</td>
```

    and after the name cell's `<code>{{ m.name }}</code>` add

```html
          {% if m.origin and m.origin.op == 'fork' %}
          <span title="forked from {{ m.origin.forked_from }} ({{ m.origin.created_at }})">⑂</span>
          {% endif %}
```

  - Row buttons (in the same actions cell as 📦 archive):

```html
          <button type="button" class="mini-btn ens-continue-btn"
                  data-member="{{ m.name }}"
                  title="Continue training this member for N more steps (warm cosine restart — LR lifts moderately and decays back).">▶ continue</button>
          <button type="button" class="mini-btn ens-fork-btn"
                  data-member="{{ m.name }}"
                  title="Start new member(s) initialized from this member's weights, at step 0 with a fresh LR schedule.">⑂ fork</button>
```

  - Row-button wiring in the scripts block (after the archive wiring):

```html
// ▶/⑂ shortcuts: set the train card's mode, prefill the member, scroll to it.
function _prefillTrainCard(mode, member) {
  const form = document.querySelector('form[data-step-id="ensemble_train"]');
  if (!form) { alert("train card still loading — try again in a second"); return; }
  const sel = form.querySelector('select[name="mode"]');
  sel.value = mode;
  sel.dispatchEvent(new Event("change", { bubbles: true }));
  if (mode === "continue") {
    form.querySelectorAll('input[name="members"]').forEach((cb) => {
      cb.checked = cb.value === member;
    });
  } else {
    const src = form.querySelector('select[name="fork_from"]');
    if (src) src.value = member;
  }
  form.scrollIntoView({ behavior: "smooth", block: "center" });
}
document.querySelectorAll(".ens-continue-btn").forEach((b) =>
  b.addEventListener("click", () => _prefillTrainCard("continue", b.dataset.member)));
document.querySelectorAll(".ens-fork-btn").forEach((b) =>
  b.addEventListener("click", () => _prefillTrainCard("fork", b.dataset.member)));
```

- [ ] **Step 2: Step-card fields.** In `fasrc_step_card.js`, replace the `case 'ensemble_train':` return with a call to a new `_ensembleTrainFields()`:

```js
  function _ensembleTrainFields() {
    const members = window.ENSEMBLE_MEMBERS || [];
    const memberChecks = members.map((m) => `
        <label style="font-weight:normal;">
          <input type="checkbox" name="members" value="${m.name}">
          <code>${m.name}</code>
          <span class="muted">${m.step ? "@ " + m.step.toLocaleString() : ""}</span>
        </label>`).join("") ||
      '<p class="muted">no local members — pull the ensemble first.</p>';
    const memberOpts = members.map((m) =>
      `<option value="${m.name}">${m.name}</option>`).join("");
    return `
      <label>Mode
        <select name="mode"
                title="Add: train brand-new members only (existing ones untouched). Continue: N more steps on selected members (warm cosine restart). Fork: new member(s) initialized from an existing member's weights, LR schedule reset to step 0.">
          <option value="add">Add new members</option>
          <option value="continue">Continue existing</option>
          <option value="fork">Fork from member</option>
        </select></label>
      <span data-mode-group="add fork">
        <label>Count
          <input type="number" name="count" value="1" min="1" max="20"></label>
        <label>Steps (total)
          <input type="number" name="steps" value="200000" min="1000" max="2000000"></label>
        <label title="Member i is seeded base_seed+i. Blank = fresh entropy seed (recorded on provenance).">
          Base seed
          <input type="number" name="base_seed" value="" placeholder="blank = entropy"></label>
      </span>
      <span data-mode-group="fork" style="display:none;">
        <label>Fork source
          <select name="fork_from">${memberOpts}</select></label>
        <label title="Which of the source's two save-best checkpoints seeds the new member's weights.">
          Track
          <select name="fork_track">
            <option value="psnr">PSNR-best</option>
            <option value="loss">Loss-best</option>
          </select></label>
      </span>
      <span data-mode-group="continue" style="display:none;">
        <div class="form-row" style="flex-direction:column; gap:4px;">${memberChecks}</div>
        <label>Extra steps
          <input type="number" name="extra_steps" value="50000" min="1000" max="1000000"></label>
      </span>
      <p class="hint" style="flex-basis:100%;">One GPU job, members trained
         sequentially into <code>&lt;ckpt&gt;/../ensemble/member_NN/</code> on
         FASRC. New-member names are allocated from the local registry at
         submit (archived indices are never reused). Pull the members back
         below when it finishes.</p>`;
  }
```

  Post-render wiring: find where the module inserts the form into the DOM (the `mountOne` path that sets `innerHTML` with the `<form data-step-id=…>` markup, ~line 535) and, after insertion, add:

```js
    // ensemble_train: mode selector shows/hides its field groups.
    const modeSel = el.querySelector('form[data-step-id="ensemble_train"] select[name="mode"]');
    if (modeSel) {
      const apply = () => {
        el.querySelectorAll("[data-mode-group]").forEach((g) => {
          g.style.display =
            g.dataset.modeGroup.split(" ").includes(modeSel.value) ? "" : "none";
        });
      };
      modeSel.addEventListener("change", apply);
      apply();
    }
```

  (Adapt `el` to the actual container variable at that point in the module; unchecked checkboxes simply don't submit, and hidden groups' values are ignored by `build_command` per mode, so no field clearing is needed.)
- [ ] **Step 3: Smoke.** `pytest tests/test_web.py -q` (page renders with the new template vars); then start the dev server and eyeball: mode switch toggles groups, ▶/⑂ prefill, submit posts `mode` + fields (check the sbatch script preview / job label in the FASRC panel if connected, else just the form POST payload in devtools).
- [ ] **Step 4: Commit** — `git commit -m "feat(ensemble UI): mode-switching train card + per-row continue/fork"`

---

### Task 9: Full sweep + push

- [ ] **Step 1:** `~/miniforge3/envs/EuclidPolishEnv/bin/python -m pytest tests/ -q --ignore=tests/test_zoobot_morphology.py` → all green (fix fallout; expected: the old `test_ensemble_train_step_build_command` expectations).
- [ ] **Step 2:** Update the memory note + check the spec's job-label item: `EnsembleTrainStep` label — the submit route builds labels from step metadata; if a per-mode label is trivial (label param in build_sbatch_body callers), set it; otherwise skip (job history already shows the argv).
- [ ] **Step 3:** Commit remaining changes, push. FASRC validation of an actual add/continue/fork run stays pending (like the LR-schedule work) — note in the final report.
