# 4-band Color Lens-finder Stamps + GPU/CPU Split — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split the fused lens-finder stamp build into a GPU SR-inference step + a CPU crop/render step, and render 4-band Lupton-asinh RGB stamps (cut LR 53 / SR-HR 106 px, upscaled to 424) instead of VIS-only grayscale.

**Architecture:** A new `lensfinder_sr_infer` GPU step runs the SR model over fields and persists `sr_{subset}.tfrecord` (4-band, resumable). The revised `lensfinder_build_stamps` CPU step reads `dirty_`/`sr_`/`hr_` records, cuts 4-band triplets, and renders Lupton RGB PNGs. No model is loaded in the CPU step.

**Tech Stack:** Python, TensorFlow (records), `astropy.visualization.make_lupton_rgb`, PIL, pytest. Tests run in `~/miniforge3/envs/EuclidPolishEnv/bin/python`.

**Spec:** `docs/superpowers/specs/2026-06-23-lensfinder-4band-color-stamps-design.md`

---

## File structure

- **Modify** `euclid_polish/lensfinder/stamps.py` — `cut_triplet` → 4-band recon-keyed cubes; add `render_stamp_rgb`; remove `recon_planes`, `lr_upsample_to_grid`, `render_stamp_png`.
- **Create** `scripts/lensfinder_sr_infer.py` — GPU SR inference; persists `sr_{subset}`; resumable.
- **Modify** `scripts/lensfinder_build_stamps.py` — read `sr_`, drop model, 4-band cut, RGB render, `--stamp-m` 106.
- **Modify** `euclid_polish/web/fasrc_pipeline.py` — add `LensfinderSRInferStep`; revise `LensfinderBuildStampsStep.build_command`.
- **Modify** `tests/test_lensfinder_stamps.py` — update triplet + render tests to the new API.
- **Create** `tests/test_lensfinder_sr_infer.py` — SR-infer core + resume tests.
- **Modify** `tests/test_lensfinder_fasrc.py` — new step + build-command changes.
- **Modify** `tests/test_fasrc_pipeline.py` — fix the 3 stale registry/job-name/GPU expectations.

`PY=~/miniforge3/envs/EuclidPolishEnv/bin/python` throughout.

---

## Task 1: 4-band `cut_triplet`; remove common-canvas helpers

**Files:**
- Modify: `euclid_polish/lensfinder/stamps.py:63-128`
- Test: `tests/test_lensfinder_stamps.py:45-71`

- [ ] **Step 1: Update the geometry tests to the new API**

Replace `class TestTripletGeometry` (`tests/test_lensfinder_stamps.py:45-71`) with:

```python
class TestTripletGeometry:
    def _field(self):
        lr_cube = np.random.default_rng(1).random((53, 53, 4)).astype(np.float32)
        sr_cube = np.random.default_rng(2).random((106, 106, 4)).astype(np.float32)
        hr_cube = np.random.default_rng(3).random((106, 106, 4)).astype(np.float32)
        return lr_cube, sr_cube, hr_cube

    def test_cut_triplet_keeps_four_bands_at_native_sizes(self):
        # Field big enough that a centered 106px stamp fits; LR is half-grid.
        lr_cube = np.random.default_rng(1).random((64, 64, 4)).astype(np.float32)
        sr_cube = np.random.default_rng(2).random((128, 128, 4)).astype(np.float32)
        hr_cube = np.random.default_rng(3).random((128, 128, 4)).astype(np.float32)
        t = st.cut_triplet(lr_cube, sr_cube, hr_cube, cx=64.0, cy=64.0, m=106)
        assert set(t) == {"lr", "sr", "hr"}
        assert t["lr"].shape == (53, 53, 4)      # LR half-grid, all 4 bands
        assert t["sr"].shape == (106, 106, 4)
        assert t["hr"].shape == (106, 106, 4)

    def test_cut_triplet_shares_crop_center(self):
        lr_cube = np.zeros((64, 64, 4), np.float32)
        sr_cube = np.zeros((128, 128, 4), np.float32)
        hr_cube = np.zeros((128, 128, 4), np.float32)
        sr_cube[64, 64, 0] = 9.0                 # mark HR-grid center
        hr_cube[64, 64, 0] = 9.0
        lr_cube[32, 32, 0] = 9.0                 # same point on LR half-grid
        t = st.cut_triplet(lr_cube, sr_cube, hr_cube, cx=64.0, cy=64.0, m=106)
        assert t["sr"][53, 53, 0] == 9.0         # center lands at stamp center
        assert t["hr"][53, 53, 0] == 9.0
        assert t["lr"][26, 26, 0] == 9.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `$PY -m pytest tests/test_lensfinder_stamps.py::TestTripletGeometry -q`
Expected: FAIL — old `cut_triplet` returns `lr_vis`/`sr_vis`/`hr_vis` 2-D planes.

- [ ] **Step 3: Replace `cut_triplet` and delete the common-canvas helpers**

In `euclid_polish/lensfinder/stamps.py`, replace `cut_triplet` (`:63-81`), `lr_upsample_to_grid` (`:84-92`), and `recon_planes` (`:118-128`) with a single new `cut_triplet` (keep `iter_field_sources` and `sample_galaxy_negatives` unchanged):

```python
def cut_triplet(lr_cube, sr_cube, hr_cube, *, cx: float, cy: float, m: int
                ) -> Dict[str, np.ndarray]:
    """Source-centered 4-band stamps, one per reconstruction.

    Returns ``{"lr": (m//2, m//2, C), "sr": (m, m, C), "hr": (m, m, C)}`` on
    their native grids — LR is half-resolution, so it crops at
    ``(cx/2, cy/2, m//2)`` while SR/HR crop at ``(cx, cy, m)`` (same sky FOV).
    All bands are kept (VIS, Y_E, J_E, H_E); the renderer composites them.
    """
    from euclid_polish.eval.synthetic_runner import crop_stamp

    def _cube_crop(cube, ccx, ccy, mm):
        cube = np.asarray(cube, dtype=np.float32)
        return np.stack([crop_stamp(cube[..., c], cx=ccx, cy=ccy, m=mm)
                         for c in range(cube.shape[-1])], axis=-1)

    return {
        "lr": _cube_crop(lr_cube, cx / 2.0, cy / 2.0, m // 2),
        "sr": _cube_crop(sr_cube, cx, cy, m),
        "hr": _cube_crop(hr_cube, cx, cy, m),
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `$PY -m pytest tests/test_lensfinder_stamps.py::TestTripletGeometry tests/test_lensfinder_stamps.py::TestSourceSelection -q`
Expected: PASS (source-selection tests untouched and still green).

- [ ] **Step 5: Commit**

```bash
git add euclid_polish/lensfinder/stamps.py tests/test_lensfinder_stamps.py
git commit -m "lensfinder: cut_triplet keeps 4 bands at native 53/106 grids"
```

---

## Task 2: `render_stamp_rgb` (Lupton asinh); remove `render_stamp_png`

**Files:**
- Modify: `euclid_polish/lensfinder/stamps.py:95-115`
- Test: `tests/test_lensfinder_stamps.py:74-81`

- [ ] **Step 1: Replace the render test**

Replace `class TestRender` (`tests/test_lensfinder_stamps.py:74-81`) with:

```python
class TestRender:
    def test_render_stamp_rgb_writes_424_rgb(self, tmp_path):
        from PIL import Image
        rng = np.random.default_rng(0)
        stamp4 = (rng.random((106, 106, 4)).astype(np.float32) * 500.0)
        out = str(tmp_path / "sr.png")
        st.render_stamp_rgb(stamp4, out, size=424)
        with Image.open(out) as im:
            assert im.size == (424, 424) and im.mode == "RGB"

    def test_render_stamp_rgb_handles_flat_stamp(self, tmp_path):
        # A uniform (zero-contrast) stamp must not crash or produce NaNs.
        stamp4 = np.full((53, 53, 4), 3.0, np.float32)
        out = str(tmp_path / "lr.png")
        st.render_stamp_rgb(stamp4, out, size=424)
        import os
        assert os.path.exists(out)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `$PY -m pytest tests/test_lensfinder_stamps.py::TestRender -q`
Expected: FAIL — `render_stamp_rgb` not defined.

- [ ] **Step 3: Add `render_stamp_rgb`, remove `render_stamp_png`**

In `euclid_polish/lensfinder/stamps.py`, delete `render_stamp_png` (`:95-115`) and add:

```python
def render_stamp_rgb(stamp4, out_png: str, *,
                     scale_r: float = 1.0, scale_g: float = 1.0,
                     scale_b: float = 1.0, stretch: float = 100.0,
                     Q: float = 8.0, size: int = 424) -> str:
    """Render a 4-band (H, W, 4) e- stamp to a Lupton-asinh RGB PNG.

    Band->channel matches Zoobot's GZ-DECaLS scheme (bluest->B, reddest->R):
    B=VIS (ch0), G=mean(Y_E, J_E) (ch1,2), R=H_E (ch3). ``scale_*`` are the
    per-band factors (GZ used 125/71/52 on fluxes); ``stretch``/``Q`` are the
    Lupton asinh parameters. Output is upscaled to ``size`` (bilinear)."""
    import os

    import numpy as np
    from astropy.visualization import make_lupton_rgb
    from PIL import Image

    a = np.asarray(stamp4, dtype=np.float32)
    if a.ndim != 3 or a.shape[-1] < 4:
        raise ValueError(f"render_stamp_rgb expects (H, W, >=4); got {a.shape}")
    b = a[..., 0] * scale_b
    g = 0.5 * (a[..., 1] + a[..., 2]) * scale_g
    r = a[..., 3] * scale_r
    rgb = make_lupton_rgb(r, g, b, stretch=stretch, Q=Q)   # (H, W, 3) uint8
    img = Image.fromarray(rgb, mode="RGB")
    if size and (img.width != size or img.height != size):
        img = img.resize((size, size), Image.BILINEAR)
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    img.save(out_png)
    return out_png
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `$PY -m pytest tests/test_lensfinder_stamps.py -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add euclid_polish/lensfinder/stamps.py tests/test_lensfinder_stamps.py
git commit -m "lensfinder: render_stamp_rgb (4-band Lupton-asinh) replaces grayscale"
```

---

## Task 3: SR-inference core (`run_sr_inference`) + resume

**Files:**
- Create: `scripts/lensfinder_sr_infer.py`
- Test: `tests/test_lensfinder_sr_infer.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_lensfinder_sr_infer.py`:

```python
"""SR-inference core in scripts/lensfinder_sr_infer.py: persist sr_{subset}
records (4-band) from dirty_{subset}, resumable by record-count."""

from __future__ import annotations

import importlib.util
import os

import numpy as np

from euclid_polish.config import Config
from euclid_polish.sky.tfrecord import (open_multiband_writer,
                                        read_multiband_skyimages, tfrecord_path)
from euclid_polish.sky.types import MultiBandSkyImage


def _load():
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "scripts", "lensfinder_sr_infer.py")
    spec = importlib.util.spec_from_file_location("lf_sr_infer", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sri = _load()


def _write_dirty(rdir, subset, n, shape=(16, 16, 4)):
    with open_multiband_writer(f"dirty_{subset}", records_dir=rdir) as w:
        for i in range(n):
            data = np.full(shape, float(i + 1), np.float32)
            w.write(MultiBandSkyImage(
                data=data, pixel_scale_arcsec=0.1,
                band_names=Config.LR_INPUT_BAND_NAMES, is_clean=False,
                index=i, subset=subset), index=i)


def test_run_sr_inference_writes_4band_records(tmp_path):
    rdir = str(tmp_path)
    _write_dirty(rdir, "train", 3)
    n = sri.run_sr_inference(rdir, "train", sr_fn=lambda lr: lr)   # identity SR
    assert n == 3
    out = read_multiband_skyimages(tfrecord_path(rdir, "sr_train"), num_images=10)
    assert len(out) == 3
    assert out[0].data.shape[-1] == 4              # 4-band preserved


def test_run_sr_inference_resume_skips_complete(tmp_path, monkeypatch):
    rdir = str(tmp_path)
    _write_dirty(rdir, "train", 3)
    sri.run_sr_inference(rdir, "train", sr_fn=lambda lr: lr)

    opened = []
    real = sri.open_multiband_writer
    monkeypatch.setattr(sri, "open_multiband_writer",
                        lambda name, **kw: opened.append(name) or real(name, **kw))
    n = sri.run_sr_inference(rdir, "train", sr_fn=lambda lr: lr)   # second run
    assert n == 3 and opened == []                 # skipped, nothing rewritten

    n2 = sri.run_sr_inference(rdir, "train", sr_fn=lambda lr: lr, force=True)
    assert n2 == 3 and "sr_train" in opened        # force regenerates


def test_count_records_truncated_is_none(tmp_path):
    rdir = str(tmp_path)
    _write_dirty(rdir, "train", 2)
    p = tfrecord_path(rdir, "dirty_train")
    with open(p, "r+b") as f:
        f.truncate(os.path.getsize(p) - 4)
    assert sri._count_records(p) is None
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `$PY -m pytest tests/test_lensfinder_sr_infer.py -q`
Expected: FAIL — module/function not defined.

- [ ] **Step 3: Create the script with the core**

Create `scripts/lensfinder_sr_infer.py`:

```python
#!/usr/bin/env python
"""Run SR over lens-finder fields and persist sr_{subset} records (main TF env, GPU).

Decoupled from stamp cutting: this GPU step reconstructs every field once and
writes the 4-band SR field to ``sr_{subset}.tfrecord``; the CPU
``lensfinder_build_stamps`` step then crops stamps from it. Resumable — a
subset whose sr_ record already has one example per input field is skipped.
"""

from __future__ import annotations

import argparse
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import tensorflow as tf

from euclid_polish.config import Config
from euclid_polish.observability.reporter import Reporter
from euclid_polish.sky.tfrecord import open_multiband_writer, tfrecord_path
from euclid_polish.sky.types import MultiBandSkyImage


def _count_records(path: str) -> int | None:
    """Examples in a TFRecord, or None if missing/truncated (mid-write kill)."""
    if not os.path.exists(path):
        return None
    try:
        return sum(1 for _ in tf.data.TFRecordDataset(path))
    except tf.errors.DataLossError:
        return None


def _sr_complete(records_dir: str, subset: str, n_fields: int) -> bool:
    """True iff sr_{subset} already has one example per input field."""
    return _count_records(tfrecord_path(records_dir, f"sr_{subset}")) == n_fields


def run_sr_inference(records_dir: str, subset: str, sr_fn, *,
                     force: bool = False, reporter=None) -> int:
    """Stream dirty_{subset} through ``sr_fn`` and write 4-band sr_{subset}.

    ``sr_fn(lr_cube_4band) -> sr_array_4band``. Returns the field count. Skips
    (without rewriting) a subset already complete unless ``force``."""
    in_path = tfrecord_path(records_dir, f"dirty_{subset}")
    n_fields = _count_records(in_path) or 0
    if n_fields == 0:
        return 0
    if not force and _sr_complete(records_dir, subset, n_fields):
        return n_fields
    ds = tf.data.TFRecordDataset(in_path)
    with open_multiband_writer(f"sr_{subset}", records_dir=records_dir) as w:
        for i, raw in enumerate(ds):
            img = MultiBandSkyImage.from_tfrecord(raw)
            sr = np.asarray(sr_fn(np.asarray(img.data, np.float32)), np.float32)
            idx = img.index if img.index is not None else i
            w.write(MultiBandSkyImage(
                data=sr, pixel_scale_arcsec=Config.DEFAULT_PIXEL_SCALE,
                band_names=Config.HR_TARGET_BAND_NAMES, is_clean=True,
                index=idx, subset=subset), index=idx)
            if reporter is not None:
                reporter.set_step(i + 1, n_fields, f"SR {subset} {i + 1}/{n_fields}")
    return n_fields
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `$PY -m pytest tests/test_lensfinder_sr_infer.py -q`
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/lensfinder_sr_infer.py tests/test_lensfinder_sr_infer.py
git commit -m "lensfinder: SR-inference core writes resumable 4-band sr_ records"
```

---

## Task 4: SR-inference CLI (`_parse_args` + `main`)

**Files:**
- Modify: `scripts/lensfinder_sr_infer.py`
- Test: `tests/test_lensfinder_sr_infer.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_lensfinder_sr_infer.py`:

```python
def test_parse_args_defaults():
    args = sri._parse_args(["--records-dir", "data/x"])
    assert args.records_dir == "data/x"
    assert args.subset_all is True          # both subsets by default
    assert args.num_res_blocks == Config.DEFAULT_NUM_RES_BLOCKS
    assert args.force is False
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `$PY -m pytest tests/test_lensfinder_sr_infer.py::test_parse_args_defaults -q`
Expected: FAIL — `_parse_args` not defined.

- [ ] **Step 3: Add `_parse_args` and `main`**

Append to `scripts/lensfinder_sr_infer.py`:

```python
def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--records-dir", required=True,
                   help="dir with dirty_{subset}.tfrecord; sr_{subset} written here")
    p.add_argument("--subset", default="", help="single subset; blank = train+validate")
    p.add_argument("--checkpoint", default=Config.DEFAULT_CHECKPOINT_DIR)
    p.add_argument("--num-res-blocks", type=int, default=Config.DEFAULT_NUM_RES_BLOCKS)
    p.add_argument("--force", action="store_true",
                   help="regenerate sr_ records even if already complete")
    args = p.parse_args(argv)
    args.subset_all = not args.subset
    return args


def main(argv=None) -> int:
    args = _parse_args(argv)
    reporter = Reporter.from_env()
    from euclid_polish.training.inference import (load_model_from_checkpoint,
                                                  reconstruct)

    reporter.set_stage(f"loading SR model from {args.checkpoint}")
    model = load_model_from_checkpoint(
        args.checkpoint, Config.DEFAULT_REBIN_FACTOR, args.num_res_blocks,
        nchan_out=Config.NUM_HR_CHANNELS)

    def sr_fn(lr_cube):
        _, sr = reconstruct(model, lr_cube)
        return np.asarray(sr, np.float32)

    subsets = ("train", "validate") if args.subset_all else (args.subset,)
    for subset in subsets:
        reporter.set_stage(f"SR inference {subset}")
        n = run_sr_inference(args.records_dir, subset, sr_fn,
                             force=args.force, reporter=reporter)
        print(f"  {subset}: {n} SR fields -> sr_{subset}.tfrecord")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `$PY -m pytest tests/test_lensfinder_sr_infer.py -q && $PY scripts/lensfinder_sr_infer.py --help`
Expected: tests PASS; `--help` prints with `--records-dir`, `--force`.

- [ ] **Step 5: Commit**

```bash
git add scripts/lensfinder_sr_infer.py tests/test_lensfinder_sr_infer.py
git commit -m "lensfinder: SR-infer CLI (model load + per-subset run)"
```

---

## Task 5: `LensfinderSRInferStep` (GPU FASRC step)

**Files:**
- Modify: `euclid_polish/web/fasrc_pipeline.py` (add class after `LensfinderBuildStampsStep`, ~`:1255`; register it)
- Test: `tests/test_lensfinder_fasrc.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_lensfinder_fasrc.py`:

```python
def test_sr_infer_step_is_gpu_main_env():
    s = REGISTRY.get("lensfinder_sr_infer")
    assert s.needs_gpu is True and s.defaults.partition == "gpu"
    assert s.defaults.n_gpus == 1
    assert s.conda_env is None                     # SR inference -> main TF env
    assert s.job_name == "lensfinder-sr-infer"
    cmd = s.build_command({"checkpoint": "ckpt/x", "num_res_blocks": 8})
    assert cmd[0] == "scripts/lensfinder_sr_infer.py"
    assert cmd[cmd.index("--checkpoint") + 1] == "ckpt/x"
    assert cmd[cmd.index("--num-res-blocks") + 1] == "8"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `$PY -m pytest tests/test_lensfinder_fasrc.py::test_sr_infer_step_is_gpu_main_env -q`
Expected: FAIL — `lensfinder_sr_infer` not registered.

- [ ] **Step 3: Add and register the step**

In `euclid_polish/web/fasrc_pipeline.py`, immediately after the `LensfinderBuildStampsStep` class (ends ~`:1255`), add:

```python
class LensfinderSRInferStep(FASRCPipelineStep):
    """Run SR over lens-finder fields, persist sr_{subset} records (GPU, main TF env).

    The GPU half of the (former) fused stamp build: one forward pass per field,
    written to ``sr_{subset}.tfrecord`` for the CPU ``lensfinder_build_stamps``
    step to crop. Resumable (skips a complete subset)."""

    def __init__(self) -> None:
        super().__init__(
            step_id="lensfinder_sr_infer",
            label="Lens-finder SR inference (GPU, PyTorch-free TF)",
            job_name="lensfinder-sr-infer",
            defaults=StepResources(
                partition="gpu", n_cpus=8, n_gpus=1,
                memory="48G", time_limit="6:00:00",
            ),
            needs_gpu=True,
        )

    def build_command(self, params: Dict[str, Any]) -> List[str]:
        cmd = [
            "scripts/lensfinder_sr_infer.py",
            "--records-dir", str(params.get("records_dir",
                                            "data/images/records_lensfinder")),
        ]
        sub = str(params.get("subset", "")).strip()
        if sub:
            cmd += ["--subset", sub]
        for key, flag in (("checkpoint", "--checkpoint"),
                          ("num_res_blocks", "--num-res-blocks")):
            v = params.get(key)
            if v not in (None, ""):
                cmd += [flag, str(v)]
        return cmd
```

Then find where the lensfinder steps are registered (search for `LensfinderBuildStampsStep()` in the `REGISTRY` construction near the bottom of the file) and add `LensfinderSRInferStep()` to that registration list, immediately before `LensfinderBuildStampsStep()`.

- [ ] **Step 4: Run the test to verify it passes**

Run: `$PY -m pytest tests/test_lensfinder_fasrc.py::test_sr_infer_step_is_gpu_main_env -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add euclid_polish/web/fasrc_pipeline.py tests/test_lensfinder_fasrc.py
git commit -m "lensfinder: register GPU LensfinderSRInferStep"
```

---

## Task 6: Revise `LensfinderBuildStampsStep.build_command`

**Files:**
- Modify: `euclid_polish/web/fasrc_pipeline.py:1199-1215`
- Test: `tests/test_lensfinder_fasrc.py:26-29` (and add one)

- [ ] **Step 1: Update the existing build-command test + add coverage**

Replace `test_build_command_honors_params` (`tests/test_lensfinder_fasrc.py:26-29`) with:

```python
def test_build_command_honors_params():
    s = REGISTRY.get("lensfinder_build_stamps")
    cmd = s.build_command({"stamp_m": 200, "max_fields": 50})
    assert "200" in cmd and "--max-fields" in cmd and "50" in cmd


def test_build_stamps_defaults_106_and_drops_checkpoint():
    s = REGISTRY.get("lensfinder_build_stamps")
    cmd = s.build_command({})
    assert cmd[cmd.index("--stamp-m") + 1] == "106"      # 424/4
    assert "--checkpoint" not in cmd                     # moved to sr_infer
    assert "--num-res-blocks" not in cmd
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `$PY -m pytest tests/test_lensfinder_fasrc.py -k build_stamps -q`
Expected: FAIL — default `--stamp-m` is 128 and `--checkpoint` is still emitted.

- [ ] **Step 3: Revise `build_command`**

Replace `LensfinderBuildStampsStep.build_command` (`euclid_polish/web/fasrc_pipeline.py:1199-1215`) with:

```python
    def build_command(self, params: Dict[str, Any]) -> List[str]:
        cmd = [
            "scripts/lensfinder_build_stamps.py",
            "--records-dir", str(params.get("records_dir",
                                            "data/images/records_lensfinder")),
            "--subset", str(params.get("subset", "train")),
            "--out-dir", str(params.get("out_dir", "data/lensfinder/stamps")),
            "--stamp-m", str(int(params.get("stamp_m", 106))),   # 424/4; LR = 53
            "--neg-per-lens", str(int(params.get("neg_per_lens", 2))),
        ]
        for key, flag in (("png_size", "--png-size"),
                          ("max_fields", "--max-fields"),
                          ("lupton_stretch", "--lupton-stretch"),
                          ("lupton_q", "--lupton-q")):
            v = params.get(key)
            if v not in (None, ""):
                cmd += [flag, str(v)]
        return cmd
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `$PY -m pytest tests/test_lensfinder_fasrc.py -k build_stamps -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add euclid_polish/web/fasrc_pipeline.py tests/test_lensfinder_fasrc.py
git commit -m "lensfinder: build-stamps card drops SR knobs, defaults stamp-m 106"
```

---

## Task 7: Rewrite `lensfinder_build_stamps.py` (read sr_, 4-band RGB)

**Files:**
- Modify: `scripts/lensfinder_build_stamps.py`
- Test: `tests/test_lensfinder_sr_infer.py` (integration helper reused) — add to a new test file `tests/test_lensfinder_build_stamps.py`

- [ ] **Step 1: Write the failing integration test**

Create `tests/test_lensfinder_build_stamps.py`:

```python
"""build-stamps reads dirty_/sr_/hr_ (no model) and writes 4-band Lupton RGB
stamps + catalog. Tiny synthetic records; no SR model, no torch."""

from __future__ import annotations

import importlib.util
import os

import numpy as np
from PIL import Image

from euclid_polish.config import Config
from euclid_polish.sky.source_catalog import SOURCE_COLS
from euclid_polish.sky.tfrecord import open_multiband_writer
from euclid_polish.sky.types import MultiBandSkyImage


def _load():
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "scripts", "lensfinder_build_stamps.py")
    spec = importlib.util.spec_from_file_location("lf_build", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bs = _load()


def _write_field(rdir, name, shape, bands):
    with open_multiband_writer(name, records_dir=rdir) as w:
        data = np.random.default_rng(abs(hash(name)) % 1000).random(shape).astype(
            np.float32) * 400.0
        w.write(MultiBandSkyImage(
            data=data, pixel_scale_arcsec=0.05, band_names=bands,
            is_clean=("dirty" not in name), index=0, subset="train"), index=0)


def _write_sources(rdir):
    rows = [
        {"field_index": 0, "type": "lens", "x_pix": 64.0, "y_pix": 64.0,
         "theta_E_arcsec": 1.2, "flux_vis_e": 500},
        {"field_index": 0, "type": "galaxy", "x_pix": 70.0, "y_pix": 60.0,
         "flux_vis_e": 300},
    ]
    with open(os.path.join(rdir, "sources_train.csv"), "w", newline="") as f:
        f.write(",".join(SOURCE_COLS) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(c, "")) for c in SOURCE_COLS) + "\n")


def test_build_stamps_writes_color_catalog(tmp_path):
    rdir = str(tmp_path / "rec")
    os.makedirs(rdir, exist_ok=True)
    _write_field(rdir, "dirty_train", (64, 64, 4), Config.LR_INPUT_BAND_NAMES)
    _write_field(rdir, "sr_train", (128, 128, 4), Config.HR_TARGET_BAND_NAMES)
    _write_field(rdir, "hr_train", (128, 128, 4), Config.HR_TARGET_BAND_NAMES)
    _write_sources(rdir)
    out = str(tmp_path / "stamps")

    rc = bs.main(["--records-dir", rdir, "--subset", "train", "--out-dir", out,
                  "--stamp-m", "106", "--png-size", "424"])
    assert rc == 0

    cat = os.path.join(out, "catalog.csv")
    assert os.path.exists(cat)
    text = open(cat).read().strip().splitlines()
    # 2 sources x 3 recons = 6 stamp rows + header.
    assert len(text) == 1 + 6
    # spot-check a rendered PNG is 424x424 RGB color.
    png = os.path.join(out, "train", "sr", "00000_lens_0.png")
    assert os.path.exists(png)
    with Image.open(png) as im:
        assert im.size == (424, 424) and im.mode == "RGB"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `$PY -m pytest tests/test_lensfinder_build_stamps.py -q`
Expected: FAIL — current `main` loads a model and uses the old VIS-only path.

- [ ] **Step 3: Rewrite the script body**

Replace the imports + `main` in `scripts/lensfinder_build_stamps.py` (the docstring/argparse header at `:1-61` stays except as noted). Update `_parse_args` to drop `--checkpoint`, `--num-res-blocks`, `--asinh-scale` and add Lupton args; change `--stamp-m` default to 106; add `--lupton-stretch`/`--lupton-q`/`--rgb-scale-r/g/b`. Concretely, replace the arg block (`:45-60`) with:

```python
    p.add_argument("--stamp-m", type=int, default=106,
                   help="HR-grid stamp size (even); LR stamp is half (424/4=106)")
    p.add_argument("--neg-per-lens", type=int, default=2,
                   help="galaxy negatives kept per lens in a field")
    p.add_argument("--edge-margin", type=float, default=0.0,
                   help="extra HR-px margin a source must keep from the border")
    p.add_argument("--png-size", type=int, default=424,
                   help="encoder input size; stamps are upscaled to this")
    p.add_argument("--lupton-stretch", type=float, default=Config.STRETCH_SCALE_E)
    p.add_argument("--lupton-q", type=float, default=8.0)
    p.add_argument("--rgb-scale-r", type=float, default=1.0)
    p.add_argument("--rgb-scale-g", type=float, default=1.0)
    p.add_argument("--rgb-scale-b", type=float, default=1.0)
    p.add_argument("--max-fields", type=int, default=0,
                   help="cap number of fields processed (0 = all)")
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--prefer-bright-neg", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)
```

Then replace `main` (`:64-163`) with:

```python
def main(argv=None) -> int:
    args = _parse_args(argv)
    reporter = Reporter.from_env()

    import numpy as np
    from tqdm import tqdm

    from euclid_polish.sky.source_catalog import read_sources
    from euclid_polish.sky.tfrecord import read_multiband_skyimages, tfrecord_path

    m = int(args.stamp_m)
    if m % 2:
        m += 1
    rng = np.random.default_rng(args.seed)
    rdir = args.records_dir

    src_csv = os.path.join(rdir, f"sources_{args.subset}.csv")
    by_field = read_sources(src_csv)
    if not by_field:
        print(f"no sources in {src_csv}")
        return 1

    window = args.max_fields or 1_000_000
    lr_by = {r.index: r for r in read_multiband_skyimages(
        tfrecord_path(rdir, f"dirty_{args.subset}"), num_images=window)}
    sr_by = {r.index: r for r in read_multiband_skyimages(
        tfrecord_path(rdir, f"sr_{args.subset}"), num_images=window)}
    hr_by = {r.index: r for r in read_multiband_skyimages(
        tfrecord_path(rdir, f"hr_{args.subset}"), num_images=window)}
    common = sorted(set(lr_by) & set(sr_by) & set(hr_by) & set(by_field))
    if args.max_fields:
        common = common[:args.max_fields]
    if not common:
        print("no fields with matching LR/SR/HR + sources")
        return 1
    field = int(np.asarray(hr_by[common[0]].data, np.float32).shape[0])
    print(f"{len(common)} fields, HR field {field}px, stamp {m}px HR (LR {m // 2})")

    reporter.set_stage(f"cutting 4-band stamps from {len(common)} fields")
    rows = []
    n_lens = n_gal = 0
    for i, idx in enumerate(tqdm(common, desc="fields")):
        reporter.set_step(i, len(common), f"field {idx}")
        lr_cube = np.asarray(lr_by[idx].data, dtype=np.float32)
        sr_cube = np.asarray(sr_by[idx].data, dtype=np.float32)
        hr_cube = np.asarray(hr_by[idx].data, dtype=np.float32)

        lenses = lf_stamps.iter_field_sources(
            by_field[idx], want_type="lens", field=field, m=m,
            edge_margin=args.edge_margin)
        galaxies = lf_stamps.iter_field_sources(
            by_field[idx], want_type="galaxy", field=field, m=m,
            edge_margin=args.edge_margin)
        negs = lf_stamps.sample_galaxy_negatives(
            galaxies, args.neg_per_lens * max(len(lenses), 1), rng=rng,
            prefer_bright=args.prefer_bright_neg)

        for stype, srcs in (("lens", lenses), ("galaxy", negs)):
            for j, s in enumerate(srcs):
                cx, cy = float(s["x_pix"]), float(s["y_pix"])
                triplet = lf_stamps.cut_triplet(lr_cube, sr_cube, hr_cube,
                                                cx=cx, cy=cy, m=m)
                base = f"{idx:05d}_{stype}_{j}"
                for recon, cube in triplet.items():
                    png = os.path.join(args.out_dir, args.subset, recon, base + ".png")
                    lf_stamps.render_stamp_rgb(
                        cube, png, scale_r=args.rgb_scale_r,
                        scale_g=args.rgb_scale_g, scale_b=args.rgb_scale_b,
                        stretch=args.lupton_stretch, Q=args.lupton_q,
                        size=args.png_size)
                    rows.append({
                        "id_str": f"{base}__{recon}",
                        "file_loc": png,
                        "is_lens": 1 if stype == "lens" else 0,
                        "theta_E_arcsec": (s.get("theta_E_arcsec", "")
                                           if stype == "lens" else ""),
                        "recon": recon,
                        "field_index": idx,
                        "src_x_pix": cx, "src_y_pix": cy,
                    })
                n_lens += stype == "lens"
                n_gal += stype == "galaxy"
        del sr_cube, hr_cube, lr_cube

    reporter.set_step(len(common), len(common), "done")
    lf_catalog.assign_splits(rows, val_frac=args.val_frac,
                             test_frac=args.test_frac, seed=args.seed)
    out_csv = os.path.join(args.out_dir, "catalog.csv")
    lf_catalog.write_catalog(out_csv, rows)
    reporter.metric({"fields": len(common), "lens": n_lens, "galaxy": n_gal,
                     "stamps": len(rows)})
    print(f"\n✓ {n_lens} lens + {n_gal} galaxy sources -> {len(rows)} stamps "
          f"({len(rows) // 3} per recon) -> {out_csv}")
    return 0
```

Also remove the now-unused `_PROJECT_ROOT`-adjacent imports of `load_model_from_checkpoint`/`reconstruct` if any remain in the module header (the original `main` imported them inside the function — confirm none are left at module scope).

- [ ] **Step 4: Run the test to verify it passes**

Run: `$PY -m pytest tests/test_lensfinder_build_stamps.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/lensfinder_build_stamps.py tests/test_lensfinder_build_stamps.py
git commit -m "lensfinder: build-stamps reads sr_ records, renders 4-band Lupton RGB"
```

---

## Task 8: Fix stale `test_fasrc_pipeline` expectations

**Files:**
- Modify: `tests/test_fasrc_pipeline.py:68-77`, `:389`, `:481-500`

- [ ] **Step 1: Update `test_all_steps_present`**

In `tests/test_fasrc_pipeline.py`, replace the assertion set in `test_all_steps_present` (`:68-77`) — add the four lensfinder ids:

```python
        assert ids == {
            "download", "extract_psf", "kernel", "tfrecords", "train",
            "euclid_sky_download", "euclid_roundtrip_tfrecords",
            "euclid_query", "euclid_verify_photometry",
            "download_euclid_cutouts", "extract_euclid_psf",
            "euclid_star_anchor_tfrecords",
            "download_tng_skirt",
            "tng_grid", "tng_stack", "poster_cutout",
            "synthetic_generate",
            "lensfinder_generate", "lensfinder_sr_infer",
            "lensfinder_build_stamps", "lensfinder_train",
        }
```

- [ ] **Step 2: Update the GPU-steps set**

Replace `:389` `assert gpu_steps == {"train"}` with:

```python
        assert gpu_steps == {"train", "lensfinder_sr_infer", "lensfinder_train"}
```

- [ ] **Step 3: Update the job-name map**

In `test_job_names_are_simple_and_stable`, add to the `expected` dict (`:481-500`), before the closing brace:

```python
            "lensfinder_generate":          "lensfinder-data",
            "lensfinder_sr_infer":          "lensfinder-sr-infer",
            "lensfinder_build_stamps":      "lensfinder-stamps",
            "lensfinder_train":             "lensfinder-train",
```

- [ ] **Step 4: Run the three tests to verify they pass**

Run: `$PY -m pytest "tests/test_fasrc_pipeline.py::TestRegistry::test_all_steps_present" "tests/test_fasrc_pipeline.py::TestRegistry::test_gpu_steps_are_the_expected_set" "tests/test_fasrc_pipeline.py::TestSbatchRendering::test_job_names_are_simple_and_stable" -q`
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_fasrc_pipeline.py
git commit -m "tests: register lensfinder steps in fasrc registry expectations"
```

---

## Task 9: Full-suite verification

**Files:** none (verification only)

- [ ] **Step 1: Run all touched + adjacent suites**

Run: `$PY -m pytest tests/test_lensfinder_stamps.py tests/test_lensfinder_sr_infer.py tests/test_lensfinder_build_stamps.py tests/test_lensfinder_fasrc.py tests/test_fasrc_pipeline.py -q`
Expected: all PASS.

- [ ] **Step 2: Smoke the two scripts' `--help`**

Run: `$PY scripts/lensfinder_sr_infer.py --help && $PY scripts/lensfinder_build_stamps.py --help`
Expected: both print usage; build-stamps shows `--stamp-m` (106) and `--lupton-stretch`, no `--checkpoint`.

- [ ] **Step 3: Push**

```bash
git push
```

---

## Self-review notes

- **Spec coverage:** split steps (Tasks 3-5 SR-infer GPU; Tasks 6-7 build-stamps CPU), 4-band SR persistence (Task 3, `HR_TARGET_BAND_NAMES`), Lupton RGB (Task 2), 53/106 sizing + upscale-to-424 (Tasks 1, 7, 6), resumable SR-infer (Tasks 3-4), registry/test fixes (Tasks 5, 6, 8). All mapped.
- **Type consistency:** `cut_triplet(lr_cube, sr_cube, hr_cube, *, cx, cy, m) -> {"lr","sr","hr"}` (recon-keyed 4-band cubes) is used identically in Task 1 test, Task 7 script. `render_stamp_rgb(stamp4, out_png, *, scale_r/g/b, stretch, Q, size)` matches between Task 2 and Task 7. `run_sr_inference(records_dir, subset, sr_fn, *, force, reporter)` matches Tasks 3-4. Step ids/job_names (`lensfinder_sr_infer`/`lensfinder-sr-infer`) match Tasks 5 and 8.
- **No placeholders:** every code step shows full code; commands have expected output.
- **Note for executor:** `make_lupton_rgb` is in `astropy.visualization`; the env (`EuclidPolishEnv`) has astropy + PIL. Lupton scale/stretch defaults are a starting point (spec flags a one-pass calibration on real stamps); they are CLI args so no code change is needed to tune.
```
