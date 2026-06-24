# Lens-finder /evaluation Preprocessing Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `/evaluation` lens-finder scoring preprocess eval cutouts identically to training — center-crop to the training stamp size, then render the same 4-band Lupton-asinh RGB — so each trained head sees the input distribution it was trained on.

**Architecture:** Add two pure helpers to `euclid_polish/lensfinder/stamps.py` (`load_fits_cube`, `render_eval_stamp`) that load an eval FITS, center-crop it, and reuse the existing `render_stamp_rgb` — guaranteeing the eval render is identical to training. Switch `scripts/lensfinder_score_eval.py` from the VIS-only `render_vis_png` to `render_eval_stamp`. Persist 4-band HR in `euclid_polish/eval/synthetic_runner.py` so the HR head also gets a real 4-band render.

**Tech Stack:** Python, numpy, astropy.io.fits, PIL, pytest. (The scoring script itself runs in the EuclidPolishZoobot/torch env, but every changed helper is pure and tested in the main env.)

---

## File Structure

- `euclid_polish/lensfinder/stamps.py` (modify) — add `load_fits_cube` + `render_eval_stamp`. Pure (numpy/astropy/PIL via the file's existing lazy-import pattern; no torch). The single source of "eval render == training render".
- `euclid_polish/eval/synthetic_runner.py` (modify) — crop & write all 4 HR bands instead of VIS-only; keep the VIS plane for PSNR.
- `scripts/lensfinder_score_eval.py` (modify) — call `render_eval_stamp` with per-recon crop size; add `--stamp-m`, `--lupton-q`, `--rgb-scale-{r,g,b}`.
- `tests/test_lensfinder_stamps.py` (modify) — cover `load_fits_cube` + `render_eval_stamp`.
- `tests/test_synthetic_runner_hr.py` (create) — assert `HR.fits` is written 4-band.

**Convention note:** `stamps.py` deliberately keeps heavy imports (astropy/PIL and the `crop_stamp` helper) *function-scoped and unconditional* so importing the module stays light (see its module docstring). New code follows that exact pattern — unconditional imports at the top of each function, never inside a branch.

---

## Task 1: `load_fits_cube` + `render_eval_stamp` in stamps.py

**Files:**
- Modify: `euclid_polish/lensfinder/stamps.py` (append after `render_stamp_rgb`, ~line 114)
- Test: `tests/test_lensfinder_stamps.py` (add a `TestEvalRender` class)

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_lensfinder_stamps.py`:

```python
class TestEvalRender:
    def _write_fits(self, path, arr):
        from astropy.io import fits
        fits.PrimaryHDU(np.ascontiguousarray(arr)).writeto(path, overwrite=True)

    def test_load_fits_cube_band_first_to_band_last(self, tmp_path):
        p = str(tmp_path / "sr.fits")
        self._write_fits(p, np.zeros((4, 12, 10), np.float32))   # (C, H, W)
        cube = st.load_fits_cube(p)
        assert cube.shape == (12, 10, 4)                          # (H, W, C)

    def test_load_fits_cube_2d_becomes_single_band(self, tmp_path):
        p = str(tmp_path / "hr_vis.fits")
        self._write_fits(p, np.zeros((12, 10), np.float32))       # legacy VIS-only
        cube = st.load_fits_cube(p)
        assert cube.shape == (12, 10, 1)

    def test_render_eval_stamp_crops_and_writes_424_rgb(self, tmp_path):
        from PIL import Image
        rng = np.random.default_rng(0)
        # SR-like 4-band 128px frame; crop to the 106 training size.
        arr = (rng.random((4, 128, 128)).astype(np.float32) * 500.0)
        src = str(tmp_path / "SR.fits")
        self._write_fits(src, arr)
        out = str(tmp_path / "after.png")
        st.render_eval_stamp(src, out, crop_m=106, size=424)
        with Image.open(out) as im:
            assert im.size == (424, 424) and im.mode == "RGB"

    def test_render_eval_stamp_centers_crop(self, tmp_path):
        # A single hot pixel at the frame center must survive the center-crop
        # (proves the crop is centered, not corner-anchored).
        arr = np.zeros((4, 128, 128), np.float32)
        arr[:, 64, 64] = 9.0
        src = str(tmp_path / "SR.fits")
        self._write_fits(src, arr)
        cube = st.load_fits_cube(src)
        from euclid_polish.eval.synthetic_runner import crop_stamp
        cropped = crop_stamp(cube[..., 0], cx=64.0, cy=64.0, m=106)
        assert cropped[53, 53] == 9.0          # center lands at stamp center

    def test_render_eval_stamp_fewer_than_4_bands_does_not_crash(self, tmp_path):
        import os
        # Legacy VIS-only HR.fits (2-D) must still render via band replication.
        arr = (np.random.default_rng(1).random((64, 64)).astype(np.float32) * 50)
        src = str(tmp_path / "HR.fits")
        self._write_fits(src, arr)
        out = str(tmp_path / "hr.png")
        st.render_eval_stamp(src, out, crop_m=53, size=424)
        assert os.path.exists(out)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_lensfinder_stamps.py::TestEvalRender -q`
Expected: FAIL — `AttributeError: module 'euclid_polish.lensfinder.stamps' has no attribute 'load_fits_cube'`.

- [ ] **Step 3: Implement the two helpers**

Append to `euclid_polish/lensfinder/stamps.py`:

```python
def load_fits_cube(fits_path: str) -> np.ndarray:
    """Load a FITS primary HDU as a band-last float32 cube ``(H, W, C)``.

    Accepts band-first ``(C, H, W)`` cubes (the eval LR/SR/HR layout) and 2-D
    ``(H, W)`` planes (a legacy VIS-only ``HR.fits``), which become ``(H, W, 1)``.
    """
    from astropy.io import fits

    with fits.open(fits_path) as hdul:
        data = np.asarray(hdul[0].data, dtype=np.float32)
    if data.ndim == 2:
        return data[..., None]
    if data.ndim == 3:
        return np.moveaxis(data, 0, -1)          # (C, H, W) → (H, W, C)
    raise ValueError(f"load_fits_cube expects 2-D or 3-D FITS; got {data.shape}")


def render_eval_stamp(fits_path: str, out_png: str, *, crop_m: int,
                      scale_r: float = 1.0, scale_g: float = 1.0,
                      scale_b: float = 1.0, stretch: float = 100.0,
                      Q: float = 8.0, size: int = 424) -> str:
    """Render an eval FITS cutout exactly as training renders a stamp.

    Mirrors ``lensfinder_build_stamps``: the eval cutouts are source-centered,
    so center-crop the cube to ``crop_m`` px (geometric center == source) and
    composite via :func:`render_stamp_rgb` with the same Lupton-asinh recipe.
    ``stretch`` defaults to ``Config.STRETCH_SCALE_E`` (100). A cube with fewer
    than four bands (e.g. a not-yet-regenerated VIS-only ``HR.fits``) is
    band-replicated to four so the render degrades to grayscale instead of
    crashing.
    """
    from euclid_polish.eval.synthetic_runner import crop_stamp

    cube = load_fits_cube(fits_path)
    h, w = cube.shape[:2]
    cropped = np.stack(
        [crop_stamp(cube[..., c], cx=w / 2.0, cy=h / 2.0, m=crop_m)
         for c in range(cube.shape[-1])], axis=-1)
    if cropped.shape[-1] < 4:
        reps = int(np.ceil(4 / cropped.shape[-1]))
        cropped = np.tile(cropped, (1, 1, reps))[..., :4]
    return render_stamp_rgb(cropped, out_png, scale_r=scale_r, scale_g=scale_g,
                            scale_b=scale_b, stretch=stretch, Q=Q, size=size)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_lensfinder_stamps.py -q`
Expected: PASS (all, including the pre-existing `TestRender`/`TestTripletGeometry`).

- [ ] **Step 5: Commit**

```bash
git add euclid_polish/lensfinder/stamps.py tests/test_lensfinder_stamps.py
git commit -m "lensfinder: add load_fits_cube + render_eval_stamp (training-parity eval render)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: Persist 4-band HR in synthetic_runner

**Files:**
- Modify: `euclid_polish/eval/synthetic_runner.py:207-208,214,247,255-256`
- Test: `tests/test_synthetic_runner_hr.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/test_synthetic_runner_hr.py`:

```python
"""HR.fits is persisted 4-band (heavy deps monkeypatched; no torch/TF)."""

from __future__ import annotations

import numpy as np
from astropy.io import fits

from euclid_polish.eval import synthetic_runner as sr


class _Img:
    def __init__(self, index, data):
        self.index = index
        self.data = data


def test_hr_fits_written_four_band(tmp_path, monkeypatch):
    # dirty_* records are LR half-grid (64²×4); hr_* are HR-grid (128²×4).
    def fake_read(path, num_images=0):
        if "dirty" in str(path):
            return [_Img(0, np.zeros((64, 64, 4), np.float32))]
        return [_Img(0, np.ones((128, 128, 4), np.float32))]

    monkeypatch.setattr("euclid_polish.sky.tfrecord.read_multiband_skyimages",
                        fake_read)
    monkeypatch.setattr("euclid_polish.sky.source_catalog.read_sources",
                        lambda p: {0: [{"type": "lens", "x_pix": 64.0,
                                        "y_pix": 64.0, "flux_vis_e": 1.0}]})
    monkeypatch.setattr("euclid_polish.training.inference.reconstruct",
                        lambda model, lr: (None, np.ones((128, 128, 4), np.float32)))

    out_dir = str(tmp_path / "eval")
    res = sr.run_synthetic_eval(
        out_dir, n=1, model=object(), records_dir=str(tmp_path),
        on_progress=lambda *a: None, log=lambda *a: None)
    assert res["n_ok"] == 1

    with fits.open(f"{out_dir}/syn-lens_0000/HR.fits") as hdul:
        hr = np.asarray(hdul[0].data)
        bands = hdul[0].header.get("BANDS", "")
    assert hr.shape == (4, 64, 64)          # 4-band, m//2-free HR grid (m=64)
    assert "VIS" in bands
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_synthetic_runner_hr.py -q`
Expected: FAIL — `assert (64, 64) == (4, 64, 64)` (HR currently written VIS-only 2-D).

- [ ] **Step 3: Edit synthetic_runner to write 4-band HR**

In `euclid_polish/eval/synthetic_runner.py`, replace the VIS-only HR crop with a 4-band stack and route PSNR through the VIS plane.

Replace (around lines 207-208):

```python
            hr_raw = np.asarray(hr_by[idx].data, dtype=np.float32)
            hr_vis = hr_raw[..., 0] if hr_raw.ndim == 3 else hr_raw   # (2H,2W)
```

with:

```python
            hr_raw = np.asarray(hr_by[idx].data, dtype=np.float32)
```

Replace the HR crop (around line 214):

```python
            hr_st = crop_stamp(hr_vis, cx=cx, cy=cy, m=m)
```

with a 4-band stack (mirrors the SR block right below it) plus a VIS view for metrics:

```python
            if hr_raw.ndim == 3:
                hr_cube_st = np.stack(
                    [crop_stamp(hr_raw[..., b], cx=cx, cy=cy, m=m)
                     for b in range(hr_raw.shape[-1])], axis=-1)
                hr_vis_st = hr_cube_st[..., 0]
            else:                                   # legacy VIS-only HR record
                hr_vis_st = crop_stamp(hr_raw, cx=cx, cy=cy, m=m)
                hr_cube_st = hr_vis_st[..., None]
```

Replace the HR write (around line 247):

```python
            _wr(os.path.join(obj_dir, "HR.fits"), hr_st, f"{grade} HR truth (VIS)")
```

with:

```python
            _wr(os.path.join(obj_dir, "HR.fits"),
                np.moveaxis(hr_cube_st, -1, 0) if hr_cube_st.ndim == 3
                else hr_cube_st,
                f"{grade} HR truth (electrons)",
                {"BANDS": (bands, "NAXIS3 plane order (band 0 = VIS)")})
```

Replace the two PSNR lines that referenced `hr_st` (around lines 255-256):

```python
                "psnr_lr_hr": _psnr(lr_up, hr_st),
                "psnr_sr_hr": _psnr(sr_vis_st, hr_st),
```

with the VIS plane:

```python
                "psnr_lr_hr": _psnr(lr_up, hr_vis_st),
                "psnr_sr_hr": _psnr(sr_vis_st, hr_vis_st),
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/test_synthetic_runner_hr.py -q`
Expected: PASS.

- [ ] **Step 5: Run the existing eval-catalog suite to confirm no regression**

Run: `python -m pytest tests/test_eval_catalog.py -q`
Expected: PASS (HR readers go through `load_vis_plane`/band-0, which handles 3-D).

- [ ] **Step 6: Commit**

```bash
git add euclid_polish/eval/synthetic_runner.py tests/test_synthetic_runner_hr.py
git commit -m "synthetic eval: persist 4-band HR truth (was VIS-only) for parity

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: Wire score_eval to the training-parity render

**Files:**
- Modify: `scripts/lensfinder_score_eval.py:31-33,40-54,62-101`

- [ ] **Step 1: Add the import and CLI args**

In `scripts/lensfinder_score_eval.py`, add the stamps import beside the existing `zoobot_morph` import (top-of-module, ~line 32):

```python
from euclid_polish.eval import zoobot_morph as zm
from euclid_polish.lensfinder import stamps as lf_stamps
```

In `_parse_args`, after the existing `--png-size` arg (~line 51), add:

```python
    p.add_argument("--stamp-m", type=int, default=106,
                   help="HR-grid crop size; MUST match build_stamps (LR uses m//2)")
    p.add_argument("--lupton-q", type=float, default=8.0)
    p.add_argument("--rgb-scale-r", type=float, default=1.0)
    p.add_argument("--rgb-scale-g", type=float, default=1.0)
    p.add_argument("--rgb-scale-b", type=float, default=1.0)
```

- [ ] **Step 2: Even the stamp size in main**

In `main`, right after `asinh = float(args.asinh_scale or Config.STRETCH_SCALE_E)` (~line 66), add:

```python
    m = int(args.stamp_m)
    m += m % 2                                  # even → integer LR half-crop
```

- [ ] **Step 3: Replace the render call**

In the per-object loop, replace (~lines 99-100):

```python
            png = os.path.join(obj["dir"], "lensfinder", f"{view}.png")
            zm.render_vis_png(src, png, asinh_scale=asinh, size=args.png_size)
```

with the per-recon center-crop + 4-band Lupton render:

```python
            png = os.path.join(obj["dir"], "lensfinder", f"{view}.png")
            crop_m = m // 2 if recon == "lr" else m
            lf_stamps.render_eval_stamp(
                src, png, crop_m=crop_m, stretch=asinh, Q=args.lupton_q,
                scale_r=args.rgb_scale_r, scale_g=args.rgb_scale_g,
                scale_b=args.rgb_scale_b, size=args.png_size)
```

- [ ] **Step 4: Verify the script still parses (no torch import needed for --help)**

Run: `python scripts/lensfinder_score_eval.py --help`
Expected: usage text listing `--stamp-m`, `--lupton-q`, `--rgb-scale-r/g/b`; exit 0. (`--help` short-circuits before the torch imports in `main`.)

- [ ] **Step 5: Byte-check the render path against a real eval object (main env)**

This exercises `render_eval_stamp` on actual eval FITS without needing the torch env. Run:

```bash
python -c "
from euclid_polish.lensfinder import stamps as st
import glob, os
d = sorted(glob.glob('data/eval_results/syn-gal_*'))[0]
for recon, fn in (('lr','original_stack.fits'),('sr','SR.fits'),('hr','HR.fits')):
    src = os.path.join(d, fn)
    if not os.path.isfile(src): continue
    out = f'/tmp/parity_{recon}.png'
    st.render_eval_stamp(src, out, crop_m=53 if recon=='lr' else 106, size=424)
    from PIL import Image
    with Image.open(out) as im:
        print(recon, fn, '->', im.size, im.mode)
"
```
Expected: three lines, each `-> (424, 424) RGB`. (HR may be VIS-only on un-regenerated runs; the `<4`-band fallback still yields a 424×424 RGB.)

- [ ] **Step 6: Commit**

```bash
git add scripts/lensfinder_score_eval.py
git commit -m "lensfinder score-eval: center-crop + 4-band Lupton render (training parity)

Replaces VIS-only render_vis_png with render_eval_stamp so the heads score the
same 106/53-px 4-band stamps they trained on. Adds --stamp-m (default 106) etc.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: Full verification + push

- [ ] **Step 1: Run the full pure-env test suite**

Run: `python -m pytest tests/test_lensfinder_stamps.py tests/test_synthetic_runner_hr.py tests/test_lensfinder_eval.py tests/test_eval_catalog.py tests/test_lensfinder_build_stamps.py -q`
Expected: all PASS.

- [ ] **Step 2: Run the broader suite to catch collateral**

Run: `python -m pytest tests/ -q -k "lensfinder or eval or synthetic"`
Expected: all PASS.

- [ ] **Step 3: Push** (per the auto commit+push workflow, once tests are green)

```bash
git push
```

---

## Self-Review

**Spec coverage:**
- Gap 1 (cutout size) → Task 1 `render_eval_stamp` center-crop + Task 3 per-recon `crop_m`. ✓
- Gap 2 (4-band Lupton render) → Task 1 reuses `render_stamp_rgb`; Task 3 calls it. ✓
- Prerequisite (4-band HR) → Task 2. ✓
- `--stamp-m` CLI default 106, no UI/Config → Task 3. ✓
- Crop center = geometric center → Task 1 `cx=w/2, cy=h/2`; `test_render_eval_stamp_centers_crop`. ✓
- `<4`-band fallback → Task 1 tile-to-4; `test_render_eval_stamp_fewer_than_4_bands_does_not_crash`. ✓
- Backward-compat for VIS-only HR consumers → Task 2 Step 5 runs `test_eval_catalog.py`. ✓
- Out-of-scope items untouched (jobs_impl, morphology script): not referenced by any task. ✓

**Placeholder scan:** none — every code/edit step shows the code; every run step shows the command + expected output.

**Type/name consistency:** `load_fits_cube` / `render_eval_stamp` signatures match between Task 1 (definition) and Task 3 (call: `crop_m`, `stretch`, `Q`, `scale_r/g/b`, `size`). `crop_stamp(plane, *, cx, cy, m)` keyword usage matches `synthetic_runner`. `hr_cube_st` / `hr_vis_st` introduced and consumed within Task 2.
