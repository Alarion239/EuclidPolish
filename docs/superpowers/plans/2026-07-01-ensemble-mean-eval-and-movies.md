# Ensemble-mean evaluation + disagreement movies Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the ensemble mean the default evaluation model, and surface the ensemble disagreement "movie" in the evaluation viewer for all evaluated objects (real lens A/B/C, real-gal, syn-lens, syn-gal).

**Architecture:** Both grouped-eval SR sites funnel inference through `reconstruct(model, lr)` (`euclid_polish/training/inference.py`), which calls a raw keras model — so `EnsembleModel` cannot be passed there directly. Introduce one inference shim `sr_from_model(model, lr)` that returns `(lr_vis, sr, members)` — the ensemble mean plus the member stack when handed an ensemble (duck-typed via `member_arrays`), else `reconstruct` output with `members=None`. Where members are available, write per-object disagreement cubes (`std.fits`, `pca0..2.fits`, `disagreement.json`) next to `SR.fits`. `enforce_object_sizes` learns to crop the new cubes so they stay pixel-aligned with the (center-cropped) SR. The evaluation viewer advertises the `morph` tier + `pca_n` + `pca_amps` and serves the cubes; the client-side morph animation is already built. Model loading defaults to the ensemble with a graceful fallback to the single checkpoint.

**Tech Stack:** Python, numpy, TensorFlow (keras inference), astropy.io.fits, Flask (viewer routes), pytest.

**Alignment note (important):** The two SR paths crop differently, so disagreement cubes must be produced to match:
- **Real objects** (`reconstruct_cutout_at`): SR is written full-field and later **center-cropped** by `enforce_object_sizes`. Write `std`/`pca*` full-field too and add them to the crop list → all center-cropped together.
- **Synthetic** (`synthetic_runner`): SR is `crop_stamp`-ed at the source `(cx, cy)` **before** write. Crop each member the same way, then write cubes at stamp size (already `EVAL_HR_SIZE`, so `enforce_object_sizes` is a no-op on them).

---

### Task 1: Inference shim `sr_from_model` + ensemble-or-single loader

**Files:**
- Create: `euclid_polish/eval/ensemble_infer.py`
- Test: `tests/test_ensemble_infer.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_ensemble_infer.py`:

```python
from __future__ import annotations

import numpy as np

from euclid_polish.eval.ensemble_infer import sr_from_model


class _FakeEnsemble:
    """Duck-typed ensemble: member_arrays returns a fixed (M,H,W,C) stack."""
    def __init__(self, stack):
        self._stack = np.asarray(stack, dtype=np.float32)
    def member_arrays(self, lr_array):
        return self._stack


class _FakeSingle:
    """No member_arrays → routed through reconstruct; here we monkeypatch."""


def test_sr_from_model_ensemble_returns_mean_and_members():
    m = np.stack([np.full((6, 6, 4), 2.0, np.float32),
                  np.full((6, 6, 4), 4.0, np.float32)], axis=0)  # (2,6,6,4)
    lr = np.zeros((3, 3, 4), np.float32)
    lr_vis, sr, members = sr_from_model(_FakeEnsemble(m), lr)
    assert members is not None and members.shape == (2, 6, 6, 4)
    assert np.allclose(sr, 3.0)                       # mean of 2 and 4
    assert lr_vis.shape == (3, 3)                     # VIS plane of the LR cube


def test_sr_from_model_single_uses_reconstruct(monkeypatch):
    import euclid_polish.eval.ensemble_infer as ei
    lr = np.zeros((4, 4, 4), np.float32)
    fake_sr = np.ones((8, 8, 4), np.float32)
    monkeypatch.setattr(ei, "reconstruct",
                        lambda model, x: (np.asarray(x)[..., 0], fake_sr))
    lr_vis, sr, members = sr_from_model(_FakeSingle(), lr)
    assert members is None
    assert np.allclose(sr, fake_sr)
    assert lr_vis.shape == (4, 4)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_ensemble_infer.py -v`
Expected: FAIL with `ModuleNotFoundError: euclid_polish.eval.ensemble_infer`.

- [ ] **Step 3: Implement the module**

Create `euclid_polish/eval/ensemble_infer.py`:

```python
"""Inference shim that lets the grouped evaluator run either a single keras
model or the ensemble mean, and (for the ensemble) also return the member stack
so the disagreement cubes can be written for the movie viewer."""

from __future__ import annotations

from typing import Any

import numpy as np

from euclid_polish.config import Config
from euclid_polish.training.inference import reconstruct


def sr_from_model(model: Any, lr_cube: np.ndarray
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """``(lr_vis, sr, members)`` for one LR cube.

    When ``model`` exposes ``member_arrays`` (an :class:`EnsembleModel`), ``sr``
    is the ensemble mean and ``members`` is the ``(M, H, W, C)`` raw-electron
    stack. Otherwise ``model`` is a keras model run through :func:`reconstruct`
    and ``members`` is ``None``.
    """
    lr = np.asarray(lr_cube, dtype=np.float32)
    if hasattr(model, "member_arrays"):
        members = np.asarray(model.member_arrays(lr), dtype=np.float32)
        sr = members.mean(axis=0)
        lr_vis = lr[..., 0] if lr.ndim == 3 else lr
        return lr_vis, sr, members
    lr_vis, sr = reconstruct(model, lr)
    return lr_vis, sr, None


def load_eval_ensemble_or_single(checkpoint: str | None = None,
                                 num_res_blocks: int | None = None,
                                 *, log=None) -> Any:
    """Default eval model: the ensemble mean if a trained ensemble exists, else
    the single checkpoint. Logs which was chosen so a run is never ambiguous."""
    emit = log or (lambda m: None)
    from euclid_polish.ensemble import load_ensemble
    try:
        ens = load_ensemble(num_res_blocks=num_res_blocks
                            or Config.DEFAULT_NUM_RES_BLOCKS)
        if ens.n_members >= 1:
            emit(f"using ensemble mean ({ens.n_members} members)")
            return ens
        emit("ensemble present but empty; falling back to single model")
    except Exception as e:  # noqa: BLE001 — any load failure → single model
        emit(f"ensemble unavailable ({type(e).__name__}: {e}); using single model")
    from euclid_polish.eval.catalog_runner import load_eval_model
    emit(f"loading single model from {checkpoint or Config.DEFAULT_CHECKPOINT_DIR}")
    return load_eval_model(checkpoint, num_res_blocks)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_ensemble_infer.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add euclid_polish/eval/ensemble_infer.py tests/test_ensemble_infer.py
git commit -m "eval: sr_from_model shim + ensemble-or-single loader"
```

---

### Task 2: Disagreement-cube writer

**Files:**
- Create: `euclid_polish/eval/disagreement.py`
- Test: `tests/test_disagreement_cubes.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_disagreement_cubes.py`:

```python
from __future__ import annotations

import json
import os

import numpy as np
from astropy.io import fits

from euclid_polish.eval.disagreement import write_disagreement_cubes


def test_write_disagreement_cubes(tmp_path):
    rng = np.random.default_rng(0)
    members = rng.normal(10.0, 1.0, (5, 8, 8, 4)).astype(np.float32)  # (M,H,W,C)
    amps = write_disagreement_cubes(str(tmp_path), members, n_components=3)

    assert len(amps) == 3
    assert all(a >= 0 for a in amps)
    # std.fits present, channel-first (C,H,W), equals members std
    with fits.open(tmp_path / "std.fits") as h:
        std = np.asarray(h[0].data)
    assert std.shape == (4, 8, 8)
    assert np.allclose(np.moveaxis(std, 0, -1), members.std(axis=0), atol=1e-4)
    for i in range(3):
        assert (tmp_path / f"pca{i}.fits").is_file()
    meta = json.load(open(tmp_path / "disagreement.json"))
    assert meta["pca_n"] == 3 and len(meta["pca_amps"]) == 3


def test_write_disagreement_cubes_few_members(tmp_path):
    members = np.ones((1, 6, 6, 4), np.float32)          # M=1 → 0 pca comps
    amps = write_disagreement_cubes(str(tmp_path), members, n_components=3)
    assert amps == []
    assert (tmp_path / "std.fits").is_file()
    assert not (tmp_path / "pca0.fits").exists()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_disagreement_cubes.py -v`
Expected: FAIL with `ModuleNotFoundError: euclid_polish.eval.disagreement`.

- [ ] **Step 3: Implement the module**

Create `euclid_polish/eval/disagreement.py`:

```python
"""Write per-object ensemble-disagreement cubes for the evaluation movie viewer.

Given a ``(M, H, W, C)`` member stack, writes ``std.fits`` (per-pixel member
std), ``pca0..K.fits`` (the PCA eigen-images of the member residuals) and a
``disagreement.json`` sidecar ``{pca_n, pca_amps}``. FITS are channel-first
``(C, H, W)`` to match ``SR.fits`` so :func:`enforce_object_sizes` crops them
consistently."""

from __future__ import annotations

import json
import os

import numpy as np
from astropy.io import fits

from euclid_polish.ensemble import pca_field


def _write_cube_fits(path: str, hwc: np.ndarray) -> None:
    arr = np.asarray(hwc, dtype=np.float32)
    arr = np.moveaxis(arr, -1, 0) if arr.ndim == 3 else arr   # (C, H, W)
    hdr = fits.Header()
    hdr["BUNIT"] = "electron"
    fits.PrimaryHDU(np.ascontiguousarray(arr), header=hdr).writeto(
        path, overwrite=True, output_verify="silentfix")


def write_disagreement_cubes(obj_dir: str, members: np.ndarray,
                             *, n_components: int = 3) -> list[float]:
    """Write ``std.fits`` + ``pca*.fits`` + ``disagreement.json`` into
    ``obj_dir``. Returns the PCA amplitudes (population std along each component;
    empty when <2 members)."""
    mem = np.asarray(members, dtype=np.float32)
    os.makedirs(obj_dir, exist_ok=True)
    _write_cube_fits(os.path.join(obj_dir, "std.fits"), mem.std(axis=0))
    _mean, comps, amps = pca_field(mem, n_components=n_components)
    for i, comp in enumerate(comps):
        _write_cube_fits(os.path.join(obj_dir, f"pca{i}.fits"), comp)
    amps_l = [float(a) for a in amps]
    with open(os.path.join(obj_dir, "disagreement.json"), "w") as f:
        json.dump({"pca_n": int(len(comps)), "pca_amps": amps_l}, f)
    return amps_l
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_disagreement_cubes.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add euclid_polish/eval/disagreement.py tests/test_disagreement_cubes.py
git commit -m "eval: write_disagreement_cubes (std + pca + sidecar)"
```

---

### Task 3: Teach `enforce_object_sizes` to crop the disagreement cubes

**Files:**
- Modify: `euclid_polish/eval/catalog_runner.py:169-171` (the `plan` tuple)
- Test: `tests/test_enforce_object_sizes_disagreement.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_enforce_object_sizes_disagreement.py`:

```python
from __future__ import annotations

import numpy as np
from astropy.io import fits

from euclid_polish.eval.catalog_runner import (
    EVAL_HR_SIZE, EVAL_LR_SIZE, enforce_object_sizes,
)


def _wr(path, arr):
    fits.PrimaryHDU(np.ascontiguousarray(arr.astype(np.float32))).writeto(
        path, overwrite=True, output_verify="silentfix")


def test_enforce_crops_std_and_pca(tmp_path):
    big = EVAL_HR_SIZE + 6
    _wr(tmp_path / "original_stack.fits", np.zeros((4, EVAL_LR_SIZE + 4, EVAL_LR_SIZE + 4)))
    _wr(tmp_path / "SR.fits", np.zeros((4, big, big)))
    _wr(tmp_path / "std.fits", np.zeros((4, big, big)))
    _wr(tmp_path / "pca0.fits", np.zeros((4, big, big)))
    assert enforce_object_sizes(str(tmp_path)) is True
    for name in ("SR.fits", "std.fits", "pca0.fits"):
        with fits.open(tmp_path / name) as h:
            assert h[0].data.shape[-2:] == (EVAL_HR_SIZE, EVAL_HR_SIZE)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_enforce_object_sizes_disagreement.py -v`
Expected: FAIL — `std.fits`/`pca0.fits` are left at `big×big` (not in the crop plan yet).

- [ ] **Step 3: Extend the crop plan**

In `euclid_polish/eval/catalog_runner.py`, replace the `plan` tuple at lines 169-171:

```python
    plan = (("original_stack.fits", EVAL_LR_SIZE, True),
            ("SR.fits", EVAL_HR_SIZE, True),
            ("HR.fits", EVAL_HR_SIZE, False))
```

with:

```python
    plan = (("original_stack.fits", EVAL_LR_SIZE, True),
            ("SR.fits", EVAL_HR_SIZE, True),
            ("HR.fits", EVAL_HR_SIZE, False),
            ("std.fits", EVAL_HR_SIZE, False),
            ("pca0.fits", EVAL_HR_SIZE, False),
            ("pca1.fits", EVAL_HR_SIZE, False),
            ("pca2.fits", EVAL_HR_SIZE, False))
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_enforce_object_sizes_disagreement.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add euclid_polish/eval/catalog_runner.py tests/test_enforce_object_sizes_disagreement.py
git commit -m "eval: center-crop std/pca disagreement cubes with SR"
```

---

### Task 4: Default the grouped runner to the ensemble (with fallback)

**Files:**
- Modify: `euclid_polish/eval/grouped_runner.py:147-149`
- Test: `tests/test_grouped_runner_model_default.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_grouped_runner_model_default.py`:

```python
from __future__ import annotations

import euclid_polish.eval.ensemble_infer as ei


def test_loader_falls_back_to_single_when_no_ensemble(monkeypatch):
    calls = {}

    class _Empty:
        n_members = 0

    monkeypatch.setattr(ei, "load_ensemble", lambda **k: _Empty())
    monkeypatch.setattr(
        "euclid_polish.eval.catalog_runner.load_eval_model",
        lambda c, n: calls.setdefault("single", (c, n)) or "SINGLE_MODEL")
    out = ei.load_eval_ensemble_or_single("ckpt/x", 8, log=lambda m: None)
    assert out == "SINGLE_MODEL"
    assert "single" in calls


def test_loader_uses_ensemble_when_present(monkeypatch):
    class _Ens:
        n_members = 5

    monkeypatch.setattr(ei, "load_ensemble", lambda **k: _Ens())
    out = ei.load_eval_ensemble_or_single(log=lambda m: None)
    assert isinstance(out, _Ens)
```

(These test the loader — the single behavioural unit; `run_grouped_analysis` merely calls it.)

- [ ] **Step 2: Run the test to verify it passes for Task-1 code, then wire the runner**

Run: `pytest tests/test_grouped_runner_model_default.py -v`
Expected: PASS (the loader from Task 1 already satisfies these). If it fails, fix the loader.

- [ ] **Step 3: Wire the runner to the loader**

In `euclid_polish/eval/grouped_runner.py`, replace lines 147-149:

```python
    if needs_lens_model and model is None:
        _emit(f"loading model from {checkpoint}")
        model = catalog_runner.load_eval_model(checkpoint, num_res_blocks)
```

with:

```python
    if needs_lens_model and model is None:
        from euclid_polish.eval.ensemble_infer import load_eval_ensemble_or_single
        model = load_eval_ensemble_or_single(checkpoint, num_res_blocks, log=_emit)
```

- [ ] **Step 4: Run the suite for this area**

Run: `pytest tests/test_grouped_runner_model_default.py tests/test_ensemble_infer.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add euclid_polish/eval/grouped_runner.py tests/test_grouped_runner_model_default.py
git commit -m "eval: grouped runner defaults to ensemble mean (single-model fallback)"
```

---

### Task 5: Emit disagreement cubes on the real-object SR path

**Files:**
- Modify: `euclid_polish/web/helpers/jobs_impl.py:422` and after the SR write (~498)

- [ ] **Step 1: Route inference through the shim**

In `reconstruct_cutout_at`, replace line 422:

```python
    _, sr_data = reconstruct(model, lr_cube)
```

with:

```python
    from euclid_polish.eval.ensemble_infer import sr_from_model
    _, sr_data, _members = sr_from_model(model, lr_cube)
```

- [ ] **Step 2: Write the cubes after SR.fits is written**

Immediately after the SR write block (after line 498, `print(f"  ✓ saved SR  → {sr_fits_path}")`), add:

```python
    # Ensemble disagreement cubes (full-field; enforce_object_sizes center-crops
    # them alongside SR so they stay pixel-aligned). No-op for a single model.
    if _members is not None:
        from euclid_polish.eval.disagreement import write_disagreement_cubes
        try:
            write_disagreement_cubes(out_dir, _members)
            print("  ✓ saved disagreement cubes (std + pca)")
        except Exception as exc:  # noqa: BLE001 — never kill a run over the movie
            print(f"  [disagreement] cubes not written: {exc}")
```

- [ ] **Step 3: Guard against a NameError when the SR write path early-returns**

Confirm `_members` is always bound before use: it is assigned at Step 1 (line 422), which runs before the SR write. No further change needed.

- [ ] **Step 4: Smoke-check import + syntax**

Run: `python -c "import euclid_polish.web.helpers.jobs_impl"`
Expected: no error.

- [ ] **Step 5: Commit**

```bash
git add euclid_polish/web/helpers/jobs_impl.py
git commit -m "eval(real): write ensemble disagreement cubes beside SR.fits"
```

---

### Task 6: Emit disagreement cubes on the synthetic SR path (stamp-aligned)

**Files:**
- Modify: `euclid_polish/eval/synthetic_runner.py:249` and after the SR write (~295)

- [ ] **Step 1: Route inference through the shim, keep the members**

In `synthetic_runner.py`, replace line 249:

```python
                _, sr_data = reconstruct(model, lr_cube)
                sr_arr = np.asarray(sr_data, dtype=np.float32)            # (2H,2W,4)
```

with:

```python
                from euclid_polish.eval.ensemble_infer import sr_from_model
                _, sr_data, members_full = sr_from_model(model, lr_cube)
                sr_arr = np.asarray(sr_data, dtype=np.float32)            # (2H,2W,4)
```

Also change the import at line 140 (`from euclid_polish.training.inference import reconstruct`) — leave it in place; `reconstruct` is still used indirectly by the shim's single-model branch. No edit needed there.

- [ ] **Step 2: Crop each member to the source stamp and write cubes**

Immediately after the three `_wr(...)` FITS writes (after line 300, the `HR.fits` write), add:

```python
            # Ensemble disagreement cubes, cropped to the SAME source stamp as
            # SR.fits (crop_stamp at cx,cy) so the movie overlays exactly. No-op
            # for a single model. members_full is (M, 2H, 2W, C).
            if members_full is not None:
                from euclid_polish.eval.disagreement import write_disagreement_cubes
                try:
                    mem_st = np.stack([
                        np.stack([crop_stamp(mem[..., b], cx=cx, cy=cy, m=m)
                                  for b in range(mem.shape[-1])], axis=-1)
                        for mem in np.asarray(members_full, dtype=np.float32)
                    ], axis=0)                                   # (M, m, m, C)
                    write_disagreement_cubes(obj_dir, mem_st)
                except Exception as exc:  # noqa: BLE001
                    _emit(f"  [disagreement] {sub}: cubes not written: {exc}")
```

(`crop_stamp`, `cx`, `cy`, `m`, `obj_dir`, `_emit`, `np`, `sub` are all already in scope in this block — see lines 255-300.)

- [ ] **Step 3: Smoke-check import + syntax**

Run: `python -c "import euclid_polish.eval.synthetic_runner"`
Expected: no error.

- [ ] **Step 4: Commit**

```bash
git add euclid_polish/eval/synthetic_runner.py
git commit -m "eval(synthetic): write stamp-aligned ensemble disagreement cubes"
```

---

### Task 7: Expose the movie tier in the evaluation viewer

**Files:**
- Modify: `euclid_polish/web/helpers/viewer_data.py` — `_EVAL_TIER_FILES` (235-239), `_eval_objects` (244-282), `_eval_meta` (285-302), `_eval_cube` (305-323)
- Test: `tests/test_eval_viewer_morph.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_eval_viewer_morph.py`:

```python
from __future__ import annotations

import json

import numpy as np
from astropy.io import fits

import euclid_polish.web.helpers.viewer_data as vd


def _obj(dirpath, with_pca=True):
    def _wr(name, arr):
        fits.PrimaryHDU(np.ascontiguousarray(arr.astype(np.float32))).writeto(
            str(dirpath / name), overwrite=True, output_verify="silentfix")
    _wr("original_stack.fits", np.zeros((4, 8, 8)))
    _wr("SR.fits", np.zeros((4, 16, 16)))
    if with_pca:
        _wr("std.fits", np.zeros((4, 16, 16)))
        for i in range(3):
            _wr(f"pca{i}.fits", np.ones((4, 16, 16)) * (i + 1))
        json.dump({"pca_n": 3, "pca_amps": [0.3, 0.2, 0.1]},
                  open(dirpath / "disagreement.json", "w"))


def test_eval_meta_advertises_morph(tmp_path, monkeypatch):
    d = tmp_path / "obj_a"
    d.mkdir()
    _obj(d)
    monkeypatch.setattr(vd, "_eval_objects", lambda: [{
        "subdir": "obj_a", "label": "a", "grade": "A",
        "tiers": ["LR", "SR", "std"], "plens": {},
        "pca_n": 3, "pca_amps": [0.3, 0.2, 0.1]}])
    meta = vd._eval_meta({})
    assert any(t["key"] == "morph" for t in meta["tiers"])
    assert meta["pca_n"] == 3
    assert meta["pca_amps"] == [[0.3, 0.2, 0.1]]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_eval_viewer_morph.py -v`
Expected: FAIL — `_eval_meta` returns no `pca_n`/`morph`.

- [ ] **Step 3: Add `std` to the tier files**

Replace `_EVAL_TIER_FILES` (viewer_data.py:235-239):

```python
_EVAL_TIER_FILES = {
    "LR": "original_stack.fits",
    "SR": "SR.fits",
    "HR": "HR.fits",
}
```

with:

```python
_EVAL_TIER_FILES = {
    "LR": "original_stack.fits",
    "SR": "SR.fits",
    "HR": "HR.fits",
    "std": "std.fits",
}
```

- [ ] **Step 4: Read the disagreement sidecar in `_eval_objects`**

In `_eval_objects`, inside the `for r in rows:` loop, after `grade = (r.get("grade") or "").strip()` (line 270), add:

```python
        pca_n, pca_amps = 0, []
        dj = os.path.join(obj_dir, "disagreement.json")
        if os.path.isfile(dj):
            with contextlib.suppress(OSError, ValueError):
                with open(dj) as f:
                    meta = json.load(f)
                pca_n = int(meta.get("pca_n", 0) or 0)
                pca_amps = list(meta.get("pca_amps", []) or [])
```

and add `"pca_n": pca_n, "pca_amps": pca_amps,` to the appended dict (the `objs.append({...})` block at 275-281).

- [ ] **Step 5: Advertise `morph` + `pca_n` + `pca_amps` in `_eval_meta`**

Replace `_eval_meta` (viewer_data.py:285-302) with:

```python
def _eval_meta(params: dict[str, str]) -> dict[str, Any]:
    objs = _eval_objects()
    order = ["LR", "SR", "HR", "std"]
    seen = {t for o in objs for t in o["tiers"]}
    tiers = [{"key": k, "label": ("stdSR" if k == "std" else k)}
             for k in order if k in seen]
    pca_n = max((int(o.get("pca_n", 0) or 0) for o in objs), default=0)
    pca_amps = [list(o.get("pca_amps", []) or []) for o in objs]
    if pca_n > 0:
        tiers.append({"key": "morph", "label": "disagreement movie"})
    default = "SR" if any(t["key"] == "SR" for t in tiers) else (
        tiers[0]["key"] if tiers else "SR")
    return {
        "count": len(objs),
        "tiers": tiers,
        "default_tier": default,
        "band_names": list(BAND_NAMES),
        "pca_n": pca_n,
        "pca_amps": pca_amps,
        "objects": [{"label": o["label"], "grade": o["grade"],
                     "tiers": o["tiers"], "subdir": o["subdir"],
                     "plens": o["plens"]}
                    for o in objs],
    }
```

- [ ] **Step 6: Serve `pca0..2` cubes in `_eval_cube`**

Replace `_eval_cube` (viewer_data.py:305-323) with:

```python
def _eval_cube(index: int, tier: str, params: dict[str, str]):
    objs = _eval_objects()
    if index < 0 or index >= len(objs):
        raise ViewerError(404, "index out of range")
    obj = objs[index]
    root = os.path.abspath(Config.EVAL_RESULTS_DIR)
    asinh = float(Config.STRETCH_SCALE_E)
    if tier.startswith("pca") and tier[3:].isdigit():
        path = os.path.join(root, obj["subdir"], f"{tier}.fits")
        if not os.path.isfile(path):
            raise ViewerError(404, f"{tier} not available for this object")
    elif tier in _EVAL_TIER_FILES and tier in obj["tiers"]:
        path = os.path.join(root, obj["subdir"], _EVAL_TIER_FILES[tier])
    else:
        raise ViewerError(404, f"{tier} not available for this object")
    with fits.open(path, memmap=False) as hdul:
        data = hdul[0].data
        with contextlib.suppress(TypeError, ValueError):
            asinh = float(hdul[0].header.get("ASINH", asinh))
    cube = _as_hwc(data)
    info = {"label": f"{obj['label']} · {tier}", "asinh": asinh, "pixscale": 0.0}
    return cube, info
```

- [ ] **Step 7: Run the test to verify it passes**

Run: `pytest tests/test_eval_viewer_morph.py -v`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add euclid_polish/web/helpers/viewer_data.py tests/test_eval_viewer_morph.py
git commit -m "eval viewer: advertise + serve disagreement movie (std + pca + morph)"
```

---

### Task 8: Surface the model choice in the run-grouped route log

**Files:**
- Modify: `euclid_polish/web/routes/evaluation.py:257-289` (the run-grouped job)

- [ ] **Step 1: Confirm the default already flows**

`api_evaluation_run_grouped` calls `run_grouped_analysis(...)` with no `model=`, so `model` is `None` and Task 4's loader picks the ensemble-or-single. No code change is required for behaviour; the loader already logs the choice via `_emit`, which the route pipes to the job log (`cap.write`).

- [ ] **Step 2: Add a one-line banner to the job log**

In the `_run(cap)` closure (route lines ~272-287), add as the first line inside `_run`, before calling `grouped_runner.run_grouped_analysis`:

```python
        cap.write("model: ensemble mean if trained, else single checkpoint\n")
```

- [ ] **Step 3: Smoke-check import**

Run: `python -c "import euclid_polish.web.routes.evaluation"`
Expected: no error.

- [ ] **Step 4: Commit**

```bash
git add euclid_polish/web/routes/evaluation.py
git commit -m "eval route: log the model-selection policy for grouped runs"
```

---

### Task 9: Full suite + end-to-end manual verification

- [ ] **Step 1: Run the full test suite**

Run: `pytest tests/ -q`
Expected: PASS (no regressions).

- [ ] **Step 2: End-to-end grouped run against a trained ensemble**

With an ensemble present under `<ckpt>/ensemble/member_NN/`, trigger a small grouped run
(WebUI Evaluation tab → run grouped with n=1, or call `run_grouped_analysis` directly with
`out_dir` a temp dir, `n=1`). Confirm in the log: `using ensemble mean (N members)`.

- [ ] **Step 3: Confirm cubes + sidecar per object**

For a produced object dir, confirm `SR.fits`, `std.fits`, `pca0.fits`, `pca1.fits`,
`pca2.fits`, `disagreement.json` all exist and that `SR.fits`/`std.fits`/`pca*.fits`
share the same `EVAL_HR_SIZE×EVAL_HR_SIZE` spatial shape.

- [ ] **Step 4: Verify the viewer movie (lens + galaxy)**

Use the preview workflow: open the Evaluation page, select a lens object and a galaxy object,
pick the "disagreement movie" tier, confirm it animates (morph amplitude/speed sliders appear).
Confirm the `std` tier renders. Capture a screenshot for the user.

- [ ] **Step 5: Fallback smoke check**

Temporarily point `EUCLID_POLISH_CKPT_DIR` at a tree with no `ensemble/` (or rename it) and
run a grouped eval; confirm the log says it fell back to the single model and that no
disagreement cubes / morph tier appear (metrics still produced). Restore afterwards.

- [ ] **Step 6: Commit any final tweaks** (else nothing to do).

---

## Self-review notes

- Spec §2 coverage: ensemble mean is default ✅ (Task 4), written into `SR.fits` ✅ (Tasks 5/6 via `sr_from_model` mean), single-model fallback ✅ (Task 1 loader + Task 9 §5), per-object std/pca for all groups ✅ (Tasks 5/6 cover real + synthetic; the runner processes A/B/C/gal/syn-lens/syn-gal).
- Spec §3 coverage: `morph` tier + `pca_n` + `pca_amps` advertised ✅ (Task 7 `_eval_meta`), cubes served ✅ (`_eval_cube`), no JS change ✅ (client already animates), works for lenses + galaxies ✅ (per-object, group-agnostic; verified Task 9 §4).
- Alignment: real path center-cropped by `enforce_object_sizes` (Task 3 adds std/pca to the plan); synthetic path crops members with `crop_stamp` inline (Task 6) — both end aligned with `SR.fits`.
- Placeholder scan: no TBD/TODO; new-module code is complete; integration edits give exact before/after with in-scope variable names.
- Type consistency: `sr_from_model → (lr_vis, sr, members|None)` consumed identically in Tasks 5/6; `write_disagreement_cubes(obj_dir, members)` used in both; `disagreement.json` keys `pca_n`/`pca_amps` written in Task 2 and read in Task 7.
- Known cost (documented in spec): real-object path runs members once for the mean and reuses that same stack for cubes (single inference — `sr_from_model` returns both), so no double inference. Evaluation is ~M× slower than single-model, as expected.
