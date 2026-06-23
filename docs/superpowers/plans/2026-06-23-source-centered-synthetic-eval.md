# Source-centered Synthetic Eval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split the full-field `synthetic` eval group into two source-centered subgroups (`syn-lens`, `syn-gal`) of M×M postage stamps, persisting per-source position/type metadata as a sidecar CSV at generation.

**Architecture:** A new `source_catalog` module owns the sidecar schema (writer used by `run_pipeline.py` on FASRC, reader used by the eval on the Mac). `synthetic_runner.py` is rewritten to reconstruct a full validation field once, then crop centered stamps around the most-central fitting lens/galaxy. The grouped runner, plots, route, and template thread an editable stamp size `M` (default 64 HR px) and the two new group tags.

**Tech Stack:** Python, NumPy, astropy.io.fits, TensorFlow records, Flask, matplotlib, pytest.

Spec: [docs/superpowers/specs/2026-06-23-source-centered-synthetic-eval-design.md](../specs/2026-06-23-source-centered-synthetic-eval-design.md)

---

## File structure

- **Create** `euclid_polish/sky/source_catalog.py` — `SourceCatalogWriter`, `read_sources`, `concat_source_csvs`, `SOURCE_COLS`.
- **Modify** `scripts/run_pipeline.py` — write per-shard sidecar in the parallel worker + serial generate; concat shards in id order.
- **Rewrite** `euclid_polish/eval/synthetic_runner.py` — source-centered `syn-lens`/`syn-gal` cutouts; add `select_central_source`, `crop_stamp` pure helpers.
- **Modify** `euclid_polish/eval/grouped_runner.py` — thread `stamp_m`, update group set.
- **Modify** `euclid_polish/eval/zoobot_morph.py` — `GROUP_COLORS` (drop `synthetic`, add `syn-lens`/`syn-gal`); transformation PSNR panel keys on has-HR.
- **Modify** `euclid_polish/web/routes/evaluation.py` — read `stamp_m` from the form, pass through.
- **Modify** `euclid_polish/web/templates/evaluation.html` — editable "Synthetic stamp M (HR px)" input + JS.
- **Test** `tests/test_eval_catalog.py` (extend), `tests/test_source_catalog.py` (new).

Run tests with:
```bash
source /Users/alarion239/miniforge3/etc/profile.d/conda.sh && conda activate EuclidPolishEnv && EUCLID_POLISH_DISABLE_AUTO_SSH=1 python -m pytest <args>
```

---

## Task 1: Source-catalog module (writer + reader + concat)

**Files:**
- Create: `euclid_polish/sky/source_catalog.py`
- Test: `tests/test_source_catalog.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_source_catalog.py
import os
from euclid_polish.sky import source_catalog as sc


def _meta():
    return {
        "galaxies": [
            {"type": "galaxy", "render": "sersic", "x_pix": 100.0, "y_pix": 120.0,
             "z_phot": 0.7, "catalog_id": 5, "flux_e_per_band": [3000.0, 1, 2, 3]},
            {"type": "galaxy", "render": "tng", "x_pix": 40.0, "y_pix": 200.0,
             "z": float("nan"), "subhalo_id": 99, "flux_e_per_band": [800.0, 1, 2, 3]},
        ],
        "lenses": [
            {"type": "lens", "x_pix": 128.0, "y_pix": 130.0, "z_lens": 0.5,
             "z_source": 2.0, "theta_E_arcsec": 1.3, "lens_subhalo_id": "g7",
             "flux_e_per_band": [5000.0, 1, 2, 3]},
        ],
    }


def test_writer_then_reader_roundtrip(tmp_path):
    p = str(tmp_path / "sources_validate.csv")
    w = sc.SourceCatalogWriter(p)
    w.add_field(0, _meta())
    w.add_field(1, {"galaxies": [], "lenses": []})  # empty field still ok
    w.close()

    by_field = sc.read_sources(p)
    assert set(by_field) == {0}                       # field 1 contributed no rows
    rows = by_field[0]
    assert len(rows) == 3
    sersic = next(r for r in rows if r["render"] == "sersic")
    assert sersic["type"] == "galaxy" and sersic["x_pix"] == 100.0
    assert sersic["flux_vis_e"] == 3000.0 and sersic["z"] == 0.7
    lens = next(r for r in rows if r["type"] == "lens")
    assert lens["theta_E_arcsec"] == 1.3 and lens["subhalo_id"] == "g7"
    tng = next(r for r in rows if r["render"] == "tng")
    assert tng["subhalo_id"] == "99" and tng["z"] is None   # NaN -> None


def test_read_sources_missing_file(tmp_path):
    assert sc.read_sources(str(tmp_path / "nope.csv")) == {}


def test_concat_source_csvs_preserves_order(tmp_path):
    a = str(tmp_path / "a.csv"); b = str(tmp_path / "b.csv")
    wa = sc.SourceCatalogWriter(a); wa.add_field(0, _meta()); wa.close()
    wb = sc.SourceCatalogWriter(b); wb.add_field(1, _meta()); wb.close()
    out = str(tmp_path / "sources_validate.csv")
    sc.concat_source_csvs([a, b], out)
    by_field = sc.read_sources(out)
    assert set(by_field) == {0, 1}
    # single header line
    with open(out) as f:
        assert sum(1 for ln in f if ln.startswith("field_index,")) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `... python -m pytest tests/test_source_catalog.py -q`
Expected: FAIL (`ModuleNotFoundError: euclid_polish.sky.source_catalog`).

- [ ] **Step 3: Write the module**

```python
# euclid_polish/sky/source_catalog.py
"""Per-source sidecar catalog for synthetic fields.

``MultiBandSimulator.simulate_field`` knows every galaxy/lens it places, but the
TFRecord schema stores only pixels. This module persists that source list as a
CSV next to the records (``sources_<subset>.csv``) so the evaluation can crop
postage stamps centered on a known lens or galaxy. One row per galaxy and per
lens; stars are not recorded (we never center morphology on them).
"""

from __future__ import annotations

import csv
import math
import os
from typing import Any, Dict, List, Optional

SOURCE_COLS = ["field_index", "type", "render", "x_pix", "y_pix",
               "flux_vis_e", "z", "subhalo_id", "theta_E_arcsec"]


def _flux_vis(src: Dict[str, Any]):
    f = src.get("flux_e_per_band")
    return float(f[0]) if f else ""


def _z(src: Dict[str, Any]):
    z = src.get("z_phot", src.get("z_lens", src.get("z")))
    if z is None:
        return ""
    z = float(z)
    return "" if math.isnan(z) else z


def _galaxy_row(field_index: int, g: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "field_index": field_index, "type": "galaxy",
        "render": g.get("render", ""),
        "x_pix": float(g["x_pix"]), "y_pix": float(g["y_pix"]),
        "flux_vis_e": _flux_vis(g), "z": _z(g),
        "subhalo_id": g.get("subhalo_id", ""), "theta_E_arcsec": "",
    }


def _lens_row(field_index: int, l: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "field_index": field_index, "type": "lens", "render": "",
        "x_pix": float(l["x_pix"]), "y_pix": float(l["y_pix"]),
        "flux_vis_e": _flux_vis(l), "z": _z(l),
        "subhalo_id": l.get("lens_subhalo_id", ""),
        "theta_E_arcsec": float(l.get("theta_E_arcsec", "nan"))
        if l.get("theta_E_arcsec") is not None else "",
    }


class SourceCatalogWriter:
    """Append galaxy/lens rows to ``path`` as fields are generated."""

    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._f = open(path, "w", newline="")
        self._w = csv.DictWriter(self._f, fieldnames=SOURCE_COLS)
        self._w.writeheader()

    def add_field(self, field_index: int, meta: Dict[str, Any]) -> None:
        for g in meta.get("galaxies", []) or []:
            self._w.writerow(_galaxy_row(field_index, g))
        for l in meta.get("lenses", []) or []:
            self._w.writerow(_lens_row(field_index, l))

    def close(self) -> None:
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _parse(row: Dict[str, str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"type": row["type"], "render": row["render"],
                           "subhalo_id": row["subhalo_id"] or None}
    for k in ("field_index",):
        out[k] = int(row[k])
    for k in ("x_pix", "y_pix", "flux_vis_e", "z", "theta_E_arcsec"):
        v = row.get(k, "")
        out[k] = float(v) if v not in ("", None) else None
    return out


def read_sources(csv_path: str) -> Dict[int, List[Dict[str, Any]]]:
    """``field_index -> list[source dict]``; missing file -> ``{}``."""
    if not os.path.isfile(csv_path):
        return {}
    by_field: Dict[int, List[Dict[str, Any]]] = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            r = _parse(row)
            by_field.setdefault(r["field_index"], []).append(r)
    return by_field


def concat_source_csvs(part_paths: List[str], out_path: str) -> None:
    """Concatenate shard CSVs (in the given order) into one, single header."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="") as out:
        out.write(",".join(SOURCE_COLS) + "\r\n")
        for p in part_paths:
            if not os.path.isfile(p):
                continue
            with open(p, newline="") as f:
                next(f, None)                     # skip shard header
                for line in f:
                    out.write(line)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `... python -m pytest tests/test_source_catalog.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add euclid_polish/sky/source_catalog.py tests/test_source_catalog.py
git commit -m "sky: source_catalog sidecar (writer/reader/concat) for synthetic fields"
```

---

## Task 2: Persist sidecar in run_pipeline (parallel + serial)

**Files:**
- Modify: `scripts/run_pipeline.py` — serial `step_generate` (~228-237), parallel `_generate_convolve_range` (~356-375), shard merge (~491-499).

No new unit test (generation runs on FASRC with heavy assets); covered by Task 1's writer/reader/concat tests. Verify by import-smoke.

- [ ] **Step 1: Import the writer at top of `run_pipeline.py`**

Add near the other `from euclid_polish...` imports:
```python
from euclid_polish.sky.source_catalog import SourceCatalogWriter, concat_source_csvs
```

- [ ] **Step 2: Serial `step_generate` — write a sidecar for the subset**

Replace the `with open_multiband_writer(...) as w:` block body (lines ~228-237) so the loop also records sources:
```python
        with open_multiband_writer(f"clean_{subset}",
                                   records_dir=args.records_dir) as w, \
             SourceCatalogWriter(
                 tfrecord_path(args.records_dir, f"sources_{subset}")
                 .replace(".tfrecord", ".csv")) as sources:
            for i in tqdm(range(n), desc=f"  {subset}", unit="img"):
                sky, meta = sim.simulate_field(rng)
                sky.index = i
                sky.subset = subset
                w.write(sky, index=i)
                sources.add_field(i, meta)
                done += 1
                reporter.set_step(done, grand_total, f"generate {subset} {i + 1}/{n}")
            path, count = w.path, w.count
```

- [ ] **Step 3: Parallel `_generate_convolve_range` — per-shard sidecar**

In `_generate_convolve_range`, add a sidecar writer alongside the three TFRecord writers (lines ~356-370). Change the `with` to include it and record each field by its GLOBAL index `i`:
```python
    sources_part = tfrecord_path(records_dir,
                                 f"sources_{tag}").replace(".tfrecord", ".csv")
    with open_multiband_writer(f"clean_{tag}", records_dir=records_dir) as cw, \
         open_multiband_writer(f"hr_{tag}",    records_dir=records_dir) as hw, \
         open_multiband_writer(f"dirty_{tag}", records_dir=records_dir) as dw, \
         SourceCatalogWriter(sources_part) as sources:
        for local, i in enumerate(range(start, start + count), start=1):
            sky, meta = sim.simulate_field(rng)
            sky.index = i
            sky.subset = subset
            lr, hr = fwd.process(sky, rng=rng)
            hr.index = i
            hr.subset = subset
            lr.index = i
            lr.subset = subset
            cw.write(sky, index=i)
            hw.write(hr, index=i)
            dw.write(lr, index=i)
            sources.add_field(i, meta)
            now = time.perf_counter()
            if local == count or (now - last_emit) >= 2.0:
                reporter.set_worker_step(shard_id, local, count, subset)
                last_emit = now
```

- [ ] **Step 4: Parallel shard merge — concat sidecars in id order**

In `step_generate_and_convolve_parallel`, after the `for kind in ("clean", "hr", "dirty"):` concat loop (line ~493-499), add a sidecar concat using the same id-ordered bounds:
```python
        # Concatenate the per-shard source sidecars in the same id order.
        src_parts = [tfrecord_path(args.records_dir,
                                   f"sources_{subset}.part{sid:04d}")
                     .replace(".tfrecord", ".csv")
                     for sid, (s, e) in enumerate(bounds) if e > s]
        concat_source_csvs(src_parts, tfrecord_path(
            args.records_dir, f"sources_{subset}").replace(".tfrecord", ".csv"))
        for p in src_parts:
            if os.path.exists(p):
                os.remove(p)
```

- [ ] **Step 5: Smoke-import and commit**

Run: `... python -c "import ast,sys; ast.parse(open('scripts/run_pipeline.py').read()); print('ok')"`
Expected: `ok`
```bash
git add scripts/run_pipeline.py
git commit -m "run_pipeline: write source sidecar CSV alongside synthetic records"
```

---

## Task 3: Rewrite synthetic_runner for source-centered subgroups

**Files:**
- Rewrite: `euclid_polish/eval/synthetic_runner.py`
- Test: `tests/test_eval_catalog.py` (add `TestSyntheticCutouts`)

- [ ] **Step 1: Write failing tests for the pure helpers**

```python
# in tests/test_eval_catalog.py
class TestSyntheticCutouts:
    def test_select_central_source_picks_closest_fitting(self):
        from euclid_polish.eval import synthetic_runner as sr
        srcs = [
            {"type": "galaxy", "x_pix": 128.0, "y_pix": 128.0},  # center, fits
            {"type": "galaxy", "x_pix": 130.0, "y_pix": 131.0},  # near center
            {"type": "galaxy", "x_pix": 5.0,   "y_pix": 5.0},    # edge, rejected
            {"type": "lens",   "x_pix": 128.0, "y_pix": 128.0},  # wrong type
        ]
        pick = sr.select_central_source(srcs, "galaxy", field=256, m=64)
        assert pick is not None and pick["x_pix"] == 128.0

    def test_select_central_source_rejects_all_when_edge(self):
        from euclid_polish.eval import synthetic_runner as sr
        srcs = [{"type": "lens", "x_pix": 10.0, "y_pix": 10.0}]
        assert sr.select_central_source(srcs, "lens", field=256, m=64) is None

    def test_crop_stamp_hr_and_lr(self):
        import numpy as np
        from euclid_polish.eval import synthetic_runner as sr
        hr = np.arange(256 * 256, dtype=np.float32).reshape(256, 256)
        stamp = sr.crop_stamp(hr, cx=128.0, cy=100.0, m=64)
        assert stamp.shape == (64, 64)
        # center maps to (y-32 : y+32, x-32 : x+32)
        assert stamp[0, 0] == hr[68, 96]
        lr = np.arange(128 * 128, dtype=np.float32).reshape(128, 128)
        lstamp = sr.crop_stamp(lr, cx=64.0, cy=50.0, m=32)
        assert lstamp.shape == (32, 32)
```

- [ ] **Step 2: Run to verify it fails**

Run: `... python -m pytest tests/test_eval_catalog.py::TestSyntheticCutouts -q`
Expected: FAIL (`select_central_source` not defined).

- [ ] **Step 3: Rewrite `synthetic_runner.py`**

Full file:
```python
"""Source-centered synthetic evaluation (syn-lens / syn-gal).

Synthetic validation fields have HR truth, so we can measure whether SR moved a
source *toward* the truth. Unlike real lens cutouts these fields are crowded, so
instead of scoring the whole field (out of distribution for Zoobot) we crop M×M
HR-pixel postage stamps centered on one known source per field — a lens
(``syn-lens``) or a field galaxy (``syn-gal``) — using the sidecar source
catalog written at generation time. Every stamp is then a centered single object,
comparable to the real A/B/C lens cutouts. No network; needs the cached
``*_validate.tfrecord`` records + ``sources_validate.csv``.
"""

from __future__ import annotations

import math
import os
from typing import Any, Callable, Dict, List, Optional

from euclid_polish.config import Config


def default_records_dir() -> Optional[str]:
    """Local dir holding the validation TFRecords, or ``None`` if not present."""
    cand = [Config.RECORDS_DIR_V2]
    try:
        from euclid_polish.web.helpers.paths import _sky_records_local_dir
        cand.append(_sky_records_local_dir())
    except Exception:                           # noqa: BLE001 — optional
        pass
    for d in cand:
        if d and os.path.isfile(os.path.join(d, "dirty_validate.tfrecord")):
            return d
    return None


def _psnr(a, b) -> Optional[float]:
    """PSNR (dB) over the overlapping region, peak = Config.PSNR_PEAK_E."""
    import numpy as np
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    h = min(a.shape[0], b.shape[0]); w = min(a.shape[1], b.shape[1])
    if h == 0 or w == 0:
        return None
    rmse = float(np.sqrt(np.mean((a[:h, :w] - b[:h, :w]) ** 2)) + 1e-9)
    return float(20.0 * np.log10(float(Config.PSNR_PEAK_E) / rmse))


def select_central_source(sources, want_type: str, *, field: int, m: int):
    """Most-central source of ``want_type`` whose m×m box fits in a ``field``-px
    grid (>= m/2 from every edge), or ``None``. Distance to (field/2, field/2)."""
    half = m / 2.0
    c = field / 2.0
    best = None
    best_d = None
    for s in sources:
        if s.get("type") != want_type:
            continue
        x, y = float(s["x_pix"]), float(s["y_pix"])
        if x < half or x > field - half or y < half or y > field - half:
            continue
        d = (x - c) ** 2 + (y - c) ** 2
        if best_d is None or d < best_d:
            best, best_d = s, d
    return best


def crop_stamp(plane, *, cx: float, cy: float, m: int):
    """Crop an m×m stamp from a 2-D ``plane`` centered at (cx, cy) pixel coords."""
    import numpy as np
    x0 = int(round(cx)) - m // 2
    y0 = int(round(cy)) - m // 2
    return np.asarray(plane)[y0:y0 + m, x0:x0 + m]


# subgroups: (grade, source-type, HR-field-half-stays-true)
_SUBGROUPS = (("syn-lens", "lens"), ("syn-gal", "galaxy"))


def run_synthetic_eval(
    out_dir: str, n: int, *,
    model=None,
    records_dir: Optional[str] = None,
    checkpoint: Optional[str] = None,
    num_res_blocks: Optional[int] = None,
    asinh_scale: Optional[float] = None,
    stamp_m: int = 64,
    seed: int = 0,
    on_progress: Optional[Callable[[int, int, str], None]] = None,
    log: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Crop up to N syn-lens + N syn-gal source-centered stamps.

    Returns ``{"rows": [...], "n_ok", "n_skip", "groups": {...}}``. Requires the
    sidecar source catalog; if absent returns no rows and logs a clear message.
    """
    import numpy as np
    from astropy.io import fits

    from euclid_polish.sky.tfrecord import read_multiband_skyimages, tfrecord_path
    from euclid_polish.sky.source_catalog import read_sources
    from euclid_polish.training.inference import reconstruct
    from euclid_polish.eval.catalog_runner import load_eval_model

    def _emit(m): (log or print)(m)
    def _tick(i, total, lbl=""):
        if on_progress: on_progress(i, total, lbl)

    m = int(stamp_m)
    if m % 2:
        m += 1                                  # even → integer LR half-crop

    rdir = records_dir or default_records_dir()
    if rdir is None:
        raise FileNotFoundError(
            "validation records not found (dirty/hr_validate.tfrecord). Open the "
            "/inference page once to sync them, or set Config.RECORDS_DIR_V2.")
    src_csv = os.path.join(rdir, "sources_validate.csv")
    by_field = read_sources(src_csv)
    if not by_field:
        _emit("source catalog not found (sources_validate.csv) — regenerate the "
              "validation set with metadata; skipping syn-lens/syn-gal.")
        return {"rows": [], "n_ok": 0, "n_skip": 0, "groups": {}}

    # Read a window of records (enough to find N of each subgroup).
    window = max(n * 12, 60)
    lr_recs = read_multiband_skyimages(tfrecord_path(rdir, "dirty_validate"),
                                       num_images=window)
    hr_recs = read_multiband_skyimages(tfrecord_path(rdir, "hr_validate"),
                                       num_images=window)
    hr_by = {h.index: h for h in hr_recs}
    lr_by = {r.index: r for r in lr_recs}
    common = sorted(set(lr_by) & set(hr_by) & set(by_field))
    if not common:
        _emit("no validation fields with matching LR/HR + source catalog.")
        return {"rows": [], "n_ok": 0, "n_skip": 0, "groups": {}}

    # Determine the HR field size from a record (HR is 2× LR).
    hr0 = np.asarray(hr_recs[0].data, np.float32)
    field = hr0.shape[0]                         # e.g. 256

    # Assign fields to subgroups (each field used at most once). Seeded order.
    rng = np.random.default_rng(seed)
    order = list(common)
    rng.shuffle(order)
    plan: List[tuple] = []                       # (field_index, grade, source)
    used = set()
    for grade, stype in _SUBGROUPS:
        taken = 0
        for idx in order:
            if idx in used or taken >= n:
                continue
            pick = select_central_source(by_field[idx], stype, field=field, m=m)
            if pick is None:
                continue
            plan.append((idx, grade, pick))
            used.add(idx); taken += 1
        if taken < n:
            _emit(f"{grade}: only {taken}/{n} fields had a fitting {stype}.")

    if not plan:
        _emit("no fittable sources found in the window.")
        return {"rows": [], "n_ok": 0, "n_skip": 0, "groups": {}}

    if model is None:
        model = load_eval_model(checkpoint, num_res_blocks)
    scale_hdr = float(asinh_scale or Config.STRETCH_SCALE_E)
    bands = ",".join(Config.LR_INPUT_BAND_NAMES)

    rows: List[Dict[str, Any]] = []
    n_ok = n_skip = 0
    total = len(plan)
    for j, (idx, grade, src) in enumerate(plan):
        _tick(j, total, f"{grade} idx {idx}")
        sub = f"{grade}_{idx:04d}"
        obj_dir = os.path.join(out_dir, sub)
        rec: Dict[str, Any] = {
            "id": sub, "ra": "", "dec": "", "grade": grade,
            "ok": False, "error": "", "out_subdir": sub,
            "lr_total_e": "", "sr_total_e": "", "flux_ratio_sr_over_lr": "",
            "psnr_lr_hr": "", "psnr_sr_hr": "",
        }
        try:
            os.makedirs(obj_dir, exist_ok=True)
            lr_cube = np.asarray(lr_by[idx].data, dtype=np.float32)   # (H,W,4)
            hr_raw = np.asarray(hr_by[idx].data, dtype=np.float32)
            hr_vis = hr_raw[..., 0] if hr_raw.ndim == 3 else hr_raw   # (2H,2W)
            _, sr_data = reconstruct(model, lr_cube)
            sr_arr = np.asarray(sr_data, dtype=np.float32)            # (2H,2W,4)

            cx, cy = float(src["x_pix"]), float(src["y_pix"])
            # HR & SR live on the HR grid; LR cube is half-resolution.
            hr_st = crop_stamp(hr_vis, cx=cx, cy=cy, m=m)
            sr_cube_st = np.stack(
                [crop_stamp(sr_arr[..., b], cx=cx, cy=cy, m=m)
                 for b in range(sr_arr.shape[-1])], axis=-1) \
                if sr_arr.ndim == 3 else crop_stamp(sr_arr, cx=cx, cy=cy, m=m)
            lr_cube_st = np.stack(
                [crop_stamp(lr_cube[..., b], cx=cx / 2.0, cy=cy / 2.0, m=m // 2)
                 for b in range(lr_cube.shape[-1])], axis=-1)
            sr_vis_st = sr_cube_st[..., 0] if sr_cube_st.ndim == 3 else sr_cube_st
            lr_vis_st = lr_cube_st[..., 0]

            def _wr(path, arr, obj, extra=None):
                hdr = fits.Header()
                hdr["OBJECT"] = obj
                hdr["BUNIT"] = "electron"
                hdr["ASINH"] = (scale_hdr, "asinh knee for the local renderer")
                hdr["SRCX"] = (cx, "source x_pix in HR field")
                hdr["SRCY"] = (cy, "source y_pix in HR field")
                if extra:
                    for k, v in extra.items():
                        hdr[k] = v
                fits.PrimaryHDU(np.ascontiguousarray(arr), header=hdr).writeto(
                    path, overwrite=True, output_verify="silentfix")

            _wr(os.path.join(obj_dir, "original_stack.fits"),
                np.moveaxis(lr_cube_st, -1, 0), f"{grade} LR stamp (electrons)",
                {"BANDS": (bands, "NAXIS3 plane order (band 0 = VIS)")})
            _wr(os.path.join(obj_dir, "SR.fits"),
                np.moveaxis(sr_cube_st, -1, 0) if sr_cube_st.ndim == 3 else sr_cube_st,
                f"{grade} SR stamp (WDSR)",
                {"BANDS": (bands, "NAXIS3 plane order (band 0 = VIS)")})
            _wr(os.path.join(obj_dir, "HR.fits"), hr_st, f"{grade} HR truth (VIS)")

            lr_sum, sr_sum = float(np.sum(lr_vis_st)), float(np.sum(sr_vis_st))
            lr_up = np.repeat(np.repeat(lr_vis_st, 2, 0), 2, 1)
            rec.update({
                "ok": True,
                "lr_total_e": lr_sum, "sr_total_e": sr_sum,
                "flux_ratio_sr_over_lr": (sr_sum / lr_sum) if lr_sum else "",
                "psnr_lr_hr": _psnr(lr_up, hr_st),
                "psnr_sr_hr": _psnr(sr_vis_st, hr_st),
            })
            n_ok += 1
        except Exception as e:  # noqa: BLE001 — one bad field must not kill the run
            rec["error"] = f"{type(e).__name__}: {e}"
            _emit(f"  ! {sub} skipped: {rec['error']}")
            n_skip += 1
        rows.append(rec)

    groups: Dict[str, int] = {}
    for r in rows:
        if r.get("ok"):
            groups[r["grade"]] = groups.get(r["grade"], 0) + 1
    return {"rows": rows, "n_ok": n_ok, "n_skip": n_skip, "groups": groups}
```

- [ ] **Step 4: Run helper tests**

Run: `... python -m pytest tests/test_eval_catalog.py::TestSyntheticCutouts -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add euclid_polish/eval/synthetic_runner.py tests/test_eval_catalog.py
git commit -m "synthetic_runner: source-centered syn-lens/syn-gal stamps"
```

---

## Task 4: Grouped runner — thread stamp_m, update groups

**Files:**
- Modify: `euclid_polish/eval/grouped_runner.py`
- Test: `tests/test_eval_catalog.py` (extend an existing grouped test or add a no-sidecar degradation test)

- [ ] **Step 1: Write a failing degradation test**

```python
# in tests/test_eval_catalog.py (TestGroupedRunner or new)
def test_grouped_skips_synthetic_without_sidecar(self, tmp_path, monkeypatch):
    from euclid_polish.eval import grouped_runner, synthetic_runner
    # Force "no sidecar" by pointing records resolution at an empty dir.
    monkeypatch.setattr(synthetic_runner, "default_records_dir",
                        lambda: None)
    # Stub the lens path so no network: zero grades.
    out = str(tmp_path / "run")
    res = grouped_runner.run_grouped_analysis(
        out, n=1, grades=(), include_synthetic=True, stamp_m=64,
        log=lambda m: None)
    # No groups, but a manifest was written and nothing crashed.
    assert os.path.isfile(res["manifest"]) or res["n"] == 0
```

- [ ] **Step 2: Run to verify it fails**

Run: `... python -m pytest tests/test_eval_catalog.py -k grouped_skips_synthetic -q`
Expected: FAIL (`run_grouped_analysis() got an unexpected keyword argument 'stamp_m'`).

- [ ] **Step 3: Thread `stamp_m` through `run_grouped_analysis`**

In `grouped_runner.py`:
- Add `stamp_m: int = 64,` to the signature (after `asinh_scale`).
- In the `if include_synthetic:` block, pass it and tolerate the new return when the sidecar is missing:
```python
    if include_synthetic:
        _emit("synthetic subgroups (syn-lens / syn-gal)…")
        base = sum(len(r) for _, r in lens_plan)
        syn = synthetic_runner.run_synthetic_eval(
            out_dir, n, model=model, asinh_scale=asinh_scale, seed=seed,
            stamp_m=stamp_m,
            on_progress=(lambda i, t, lbl: on_progress(base + i, base + t, lbl))
            if on_progress else None,
            log=_emit)
        all_rows.extend(syn["rows"])
```
- Guard the `model` load: synthetic may be skipped, but A/B/C still need the model — the existing `total == 0` check stays. If `grades` is empty AND the sidecar is missing, `all_rows` is empty; the manifest writer below still runs and writes a header-only file. Ensure `total` no longer hard-assumes `+ n` for synthetic; replace the planning line:
```python
    total = sum(len(r) for _, r in lens_plan) + (2 * n if include_synthetic else 0)
```
- Update the returned `groups` dict to count whatever grades actually appear:
```python
    groups = {}
    for r in all_rows:
        if r.get("ok"):
            groups[r["grade"]] = groups.get(r["grade"], 0) + 1
    return {"out_dir": out_dir, "n": len(all_rows), "n_ok": n_ok,
            "manifest": manifest_path, "groups": groups}
```
- If `total == 0` but the model is still needed for an empty lens plan, keep the existing early-return. When `lens_plan` is empty but synthetic is requested, skip the model preload by loading lazily inside `run_synthetic_eval` (it already does `if model is None`). Change the model preload to only happen when there is lens work:
```python
    model = None
    if any(len(r) for _, r in lens_plan):
        _emit(f"loading model from {checkpoint}")
        model = catalog_runner.load_eval_model(checkpoint, num_res_blocks)
    os.makedirs(out_dir, exist_ok=True)
```
(`run_synthetic_eval` loads the model itself if `model is None`.)

- [ ] **Step 4: Run tests**

Run: `... python -m pytest tests/test_eval_catalog.py -k "grouped" -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add euclid_polish/eval/grouped_runner.py tests/test_eval_catalog.py
git commit -m "grouped_runner: thread stamp_m + syn-lens/syn-gal groups, lazy model"
```

---

## Task 5: Plot colors + transformation has-HR filter

**Files:**
- Modify: `euclid_polish/eval/zoobot_morph.py`
- Test: `tests/test_eval_catalog.py` (extend `test_morphology_summary_pca_with_hr_truth` grades + transformation test)

- [ ] **Step 1: Update `GROUP_COLORS`**

Replace:
```python
GROUP_COLORS = {"A": "#2a5db0", "B": "#2e8b57", "C": "#b8860b",
                "synthetic": "#b03a3a"}
```
with:
```python
GROUP_COLORS = {"A": "#2a5db0", "B": "#2e8b57", "C": "#b8860b",
                "syn-lens": "#b03a3a", "syn-gal": "#7a4fb0"}
```

- [ ] **Step 2: Transformation panel — filter on has-HR, not grade=="synthetic"**

In `render_transformation_summary`, replace the `syn = [...]` filter:
```python
    syn = [r for r in rows
           if _f(r, "psnr_lr_hr") is not None and _f(r, "psnr_sr_hr") is not None]
```
and color each point by its own group:
```python
        cols = [_group_color(r.get("grade", "")) for r in syn]
        ax[0].scatter(x, y, c=cols, s=40)
```
Keep the diagonal + title logic. (The "No synthetic group" empty-state text stays for when no HR rows exist.)

- [ ] **Step 3: Update the PCA test to use the new grades**

In `test_morphology_summary_pca_with_hr_truth`, change the manifest grades from `synthetic`/`A` to `syn-gal`/`syn-lens` so it exercises the new color keys:
```python
        f.write("id,grade,ok\nsyn0,syn-gal,True\nlensA,syn-lens,True\n")
```
(Rest of the test is unchanged; both now have HR stars via `has_hr=True`/`closer_to_ref`.)

- [ ] **Step 4: Run tests**

Run: `... python -m pytest tests/test_eval_catalog.py -k "morphology or transformation" -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add euclid_polish/eval/zoobot_morph.py tests/test_eval_catalog.py
git commit -m "zoobot_morph: syn-lens/syn-gal colors + has-HR transformation filter"
```

---

## Task 6: Route + template stamp_m wiring

**Files:**
- Modify: `euclid_polish/web/routes/evaluation.py` (~270-287)
- Modify: `euclid_polish/web/templates/evaluation.html` (~181-188, ~458-466)
- Test: `tests/test_eval_catalog.py` (route test)

- [ ] **Step 1: Write a failing route test**

```python
# in the route TestClass in tests/test_eval_catalog.py
def test_run_grouped_accepts_stamp_m(self, client, monkeypatch):
    captured = {}
    from euclid_polish.eval import grouped_runner
    monkeypatch.setattr(grouped_runner, "run_grouped_analysis",
                        lambda **kw: captured.update(kw) or {"groups": {}})
    # Run the job synchronously by stubbing the registry spawn.
    from euclid_polish.web import jobs as _jobs
    monkeypatch.setattr(_jobs.REGISTRY, "spawn",
                        lambda label, target: (target(_FakeCap()), "jid")[1])
    r = client.post("/api/evaluation/run-grouped",
                    data={"run_name": "t", "n": "2", "stamp_m": "96"})
    assert r.status_code == 200
    assert captured.get("stamp_m") == 96
```
Add a tiny `_FakeCap` helper near the top of the test module if not present:
```python
class _FakeCap:
    def tick(self, *a, **k): pass
    def write(self, *a, **k): pass
```

- [ ] **Step 2: Run to verify it fails**

Run: `... python -m pytest tests/test_eval_catalog.py -k stamp_m -q`
Expected: FAIL (`stamp_m` not in captured / KeyError).

- [ ] **Step 3: Read `stamp_m` in the route and pass it**

In `api_evaluation_run_grouped`, after parsing `cutout`:
```python
            stamp_m = int(f.get("stamp_m", 64) or 64)
            stamp_m = max(16, min(256, stamp_m + (stamp_m % 2)))  # even, bounded
```
(wrap in the same `try/except ValueError`). Then in `_run`:
```python
            return grouped_runner.run_grouped_analysis(
                out_dir=out_dir, n=n, cutout_size=cutout, stamp_m=stamp_m,
                include_synthetic=include_synth,
                on_progress=lambda i, t, lbl: cap.tick(i, t, lbl),
                log=lambda m: cap.write(m if m.endswith("\n") else m + "\n"))
```

- [ ] **Step 4: Add the template input + JS**

In `evaluation.html`, after the "Cutout size (VIS px)" label in the grouped form (~183), add:
```html
    <label>Synthetic stamp M (HR px)
      <input type="number" id="gpStampM" value="64" min="16" max="256" step="2"></label>
```
And in the `runGroupedBtn` click handler (~462), after `fd.set('cutout_size', ...)`:
```javascript
    fd.set('stamp_m', $('gpStampM').value);
```
Also update the helper caption (~176) to read: `— N each of A / B / C lenses + N syn-lens + N syn-gal (HR), one run`.

- [ ] **Step 5: Run tests + template smoke**

Run: `... python -m pytest tests/test_eval_catalog.py -k "stamp_m or run_grouped" -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add euclid_polish/web/routes/evaluation.py euclid_polish/web/templates/evaluation.html tests/test_eval_catalog.py
git commit -m "evaluation web: editable synthetic stamp M (HR px) wired to grouped run"
```

---

## Task 7: Full suite + push

- [ ] **Step 1: Run the whole suite**

Run: `... python -m pytest -q`
Expected: all pass (the `test_web_module_integrity` ordering flake passes in isolation; re-run it alone if it errors).

- [ ] **Step 2: Push**

```bash
git push
```

---

## Self-review notes

- **Spec coverage:** Task 1 = component A/B (sidecar writer/reader). Task 2 = generation persistence (parallel + serial). Task 3 = component C (source-centered eval, both subgroups, degradation). Task 4 = grouped wiring + lazy model + degradation. Task 5 = colors + has-HR transformation + PCA stars (already group-agnostic). Task 6 = route + template M. Task 7 = verify.
- **Coordinate mapping** matches the verified geometry: HR/SR cropped at HR coords m; LR cropped at half coords m/2.
- **Degradation**: no sidecar → `run_synthetic_eval` returns empty rows + logs; grouped runner still writes a manifest.
- **Type consistency**: `select_central_source`, `crop_stamp`, `run_synthetic_eval(..., stamp_m=)`, `read_sources`, `concat_source_csvs`, `SourceCatalogWriter.add_field` names are used identically across tasks.
