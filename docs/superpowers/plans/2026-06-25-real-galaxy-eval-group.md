# Real Galaxy Eval Group Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add real Euclid field galaxies (drawn from the same fields as the A/B/C lenses, 3× the grade-A count) as a true negative-control group in `/evaluation`, scored through the same SR + Zoobot + lens-finder pipeline, with a new real-data ROC panel.

**Architecture:** A new `galaxy_catalog` module queries `catalogue.mer_catalogue` (via `astroquery.esa.euclid.Euclid`) around the lens RA/Decs for clean, resolved, bigger-end galaxies and writes a normalized `id,ra,dec,grade="gal"` CSV — cached so cutouts download at most once. `grouped_runner` appends a `gal` group (rows fed through the existing grade-agnostic `eval_catalog_object` path). The analysis figure grows 2×2 → 2×3 with a real ROC (A/B/C vs gal).

**Tech Stack:** Python, astroquery (Euclid TAP/ADQL), astropy, matplotlib, Flask (WebUI), pytest. Env: `~/miniforge3/envs/EuclidPolishEnv/bin/python`.

**Run tests with:** `~/miniforge3/envs/EuclidPolishEnv/bin/python -m pytest <path> -q`

---

## File Structure

- **Create** `euclid_polish/euclid/galaxy_catalog.py` — archive-query galaxy catalog builder (column constants, ADQL builder, candidate parsing, seeded draw + incremental cache). One responsibility: produce the `gal` catalog CSV.
- **Create** `scripts/fetch_galaxy_catalog.py` — thin CLI over the module (mirrors `scripts/fetch_lens_catalog.py`).
- **Create** `tests/test_galaxy_catalog.py` — unit tests (archive mocked).
- **Modify** `euclid_polish/eval/grouped_runner.py` — add `_galaxy_plan` helper + `include_galaxies` param; append the `gal` group to the plan.
- **Modify** `euclid_polish/eval/lensfinder_eval.py` — add `gal` to `GROUPS`; figure 2×2 → 2×3 with the real ROC + real binary-score panels.
- **Modify** `euclid_polish/eval/zoobot_morph.py` — add `GROUP_COLORS["gal"]`.
- **Modify** `euclid_polish/web/routes/evaluation.py` — read the `galaxies` form flag → `include_galaxies`.
- **Modify** `euclid_polish/web/templates/evaluation.html` — "include real galaxies" checkbox + JS.
- **Modify** `tests/test_eval_catalog.py` — route flag test; `tests/test_lensfinder_eval.py` — gal-aware figure test.

---

## Task 1: galaxy_catalog module — constants + ADQL builder

**Files:**
- Create: `euclid_polish/euclid/galaxy_catalog.py`
- Test: `tests/test_galaxy_catalog.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_galaxy_catalog.py`:

```python
"""Unit tests for the real-galaxy eval catalog builder (archive mocked)."""
import csv
import math
import os

import pytest

from euclid_polish.euclid import galaxy_catalog as gc
from euclid_polish.config import Config


def test_diam_to_area_px_matches_circle():
    # diameter 5" at 0.1"/px → radius 25 px → area = pi*25^2.
    area = gc._diam_to_area_px(5.0)
    r_px = (5.0 / 2.0) / Config.VIS_PIXEL_SCALE_ARCSEC
    assert area == pytest.approx(math.pi * r_px * r_px)


def test_galaxy_adql_has_cuts():
    q = gc.galaxy_adql(10.0, -5.0, 0.05)
    assert "catalogue.mer_catalogue" in q
    assert f"{gc._POINTLIKE_COL} = 0" in q          # extended, not a star
    assert f"{gc._SPURIOUS_COL} = 0" in q           # not an artifact
    assert f"{gc._QUALITY_COL} = 0" in q            # clean detection
    assert f"{gc._SIZE_COL} BETWEEN" in q           # size window
    assert "CIRCLE('ICRS', 10.0, -5.0, 0.05)" in q  # the cone
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/miniforge3/envs/EuclidPolishEnv/bin/python -m pytest tests/test_galaxy_catalog.py -q`
Expected: FAIL — `ModuleNotFoundError: euclid_polish.euclid.galaxy_catalog`.

- [ ] **Step 3: Write minimal implementation**

Create `euclid_polish/euclid/galaxy_catalog.py`:

```python
"""Build a real-field-galaxy evaluation catalog by querying the Euclid archive.

Mirrors :mod:`euclid_polish.euclid.lens_catalog` in shape (one source of truth
shared by a CLI and the WebUI), but the "fetch" is a live ADQL cone query on
``catalogue.mer_catalogue`` around the strong-lens fields rather than a Zenodo
download. It selects clean, resolved, bigger-end galaxies — confidently *not*
gravitational lenses — and writes a normalized ``id,ra,dec,grade`` CSV with
``grade="gal"`` that the grouped eval runner consumes exactly like the A/B/C
lens catalog. The drawn set is cached (stable ids) so each galaxy's cutout is
downloaded at most once across runs.
"""
from __future__ import annotations

import csv
import math
import os
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

from astroquery.esa.euclid import Euclid

from euclid_polish.config import Config
from euclid_polish.euclid.auth import login
from euclid_polish.euclid.eval_catalog import read_eval_catalog
from euclid_polish.euclid.photometry import uJy_to_ab_mag
from euclid_polish.euclid.validator import angular_separation_arcsec

#: Group label for a real field galaxy (parallels the synthetic ``syn-gal``).
GAL_GRADE = "gal"

# MER catalogue (``catalogue.mer_catalogue``) column names. ``det_quality_flag``
# and the coord/flux columns are already used by StarCatalog (catalog.py); the
# morphology/classification columns below are the Euclid Q1 names — confirm them
# once against the live schema (see scripts/fetch_galaxy_catalog.py --probe), and
# if a name differs, change it here only (the ADQL is built from these).
_ID_COL = "object_id"
_RA_COL = "right_ascension"
_DEC_COL = "declination"
_SIZE_COL = "segmentation_area"      # px count in the segmentation map
_POINTLIKE_COL = "point_like_flag"   # 1 = point source (star), 0 = extended
_SPURIOUS_COL = "spurious_flag"      # 1 = artifact
_QUALITY_COL = "det_quality_flag"    # 0 = clean (no mask/blend/saturation/border)
_FLUX_COL = "flux_vis_psf"           # µJy — conservative brightness floor

# Selection defaults (tunable).
DIAM_LO_ARCSEC = 2.0                 # bigger-end lower bound
DIAM_HI_ARCSEC = 5.0                 # hard cap (must fit the 53 px LR stamp)
MAG_FLOOR = 23.0                     # keep galaxies brighter than this (VIS PSF mag)
LENS_EXCLUDE_ARCSEC = 10.0           # drop anything this close to a known lens


def default_out_csv() -> str:
    """Default normalized galaxy-catalog path under the configured data dir."""
    return os.path.join(Config.EVAL_CATALOG_DIR, "galaxy_catalog", "galaxies.csv")


def _diam_to_area_px(diam_arcsec: float) -> float:
    """Segmentation area (px) of a circular source of on-sky diameter ``diam``."""
    r_px = (diam_arcsec / 2.0) / Config.VIS_PIXEL_SCALE_ARCSEC
    return math.pi * r_px * r_px


def galaxy_adql(ra: float, dec: float, radius_deg: float) -> str:
    """ADQL cone query for clean, resolved, bigger-end galaxies at ``(ra, dec)``."""
    area_lo = _diam_to_area_px(DIAM_LO_ARCSEC)
    area_hi = _diam_to_area_px(DIAM_HI_ARCSEC)
    return f"""
    SELECT TOP 100000
        {_ID_COL}, {_RA_COL}, {_DEC_COL}, {_SIZE_COL}, {_FLUX_COL}
    FROM catalogue.mer_catalogue
    WHERE CONTAINS(
        POINT('ICRS', {_RA_COL}, {_DEC_COL}),
        CIRCLE('ICRS', {ra}, {dec}, {radius_deg})
    ) = 1
      AND {_POINTLIKE_COL} = 0
      AND {_SPURIOUS_COL} = 0
      AND {_QUALITY_COL} = 0
      AND {_SIZE_COL} BETWEEN {area_lo:.1f} AND {area_hi:.1f}
      AND {_FLUX_COL} IS NOT NULL
      AND {_FLUX_COL} > 0
    """
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/miniforge3/envs/EuclidPolishEnv/bin/python -m pytest tests/test_galaxy_catalog.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add euclid_polish/euclid/galaxy_catalog.py tests/test_galaxy_catalog.py
git commit -m "galaxy_catalog: ADQL builder + size mapping for field galaxies

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: galaxy_catalog — parse query results + brightness floor

**Files:**
- Modify: `euclid_polish/euclid/galaxy_catalog.py`
- Test: `tests/test_galaxy_catalog.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_galaxy_catalog.py`:

```python
def test_candidates_parse_and_mag_floor():
    rows = [
        # flux 50 µJy → mag ~19.65 → kept
        {"object_id": 1, "right_ascension": 10.01, "declination": -5.0,
         "segmentation_area": 800.0, "flux_vis_psf": 50.0},
        # flux 0.01 µJy → mag ~28.9 → too faint → dropped
        {"object_id": 2, "right_ascension": 10.02, "declination": -5.0,
         "segmentation_area": 800.0, "flux_vis_psf": 0.01},
        # non-finite flux → dropped
        {"object_id": 3, "right_ascension": 10.03, "declination": -5.0,
         "segmentation_area": 800.0, "flux_vis_psf": float("nan")},
    ]
    cands = gc._candidates_from_results(rows)
    assert [c["id"] for c in cands] == ["gal_1"]
    assert cands[0]["ra"] == 10.01 and cands[0]["dec"] == -5.0


def test_candidates_handle_none_results():
    assert gc._candidates_from_results(None) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/miniforge3/envs/EuclidPolishEnv/bin/python -m pytest tests/test_galaxy_catalog.py::test_candidates_parse_and_mag_floor -q`
Expected: FAIL — `AttributeError: module ... has no attribute '_candidates_from_results'`.

- [ ] **Step 3: Write minimal implementation**

Add to `euclid_polish/euclid/galaxy_catalog.py` (after `galaxy_adql`):

```python
def _unmask_float(value: Any) -> Optional[float]:
    """Float from a possibly-masked/None TAP cell, or None if not finite."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _candidates_from_results(results: Any,
                             mag_floor: float = MAG_FLOOR
                             ) -> List[Dict[str, Any]]:
    """Parse a TAP result into candidate galaxy dicts passing the brightness floor.

    Works with an astropy ``Table`` or any iterable of column-indexable rows
    (e.g. dicts in tests). Size/quality cuts are applied server-side in the
    ADQL; here we only drop non-finite or too-faint sources.
    """
    out: List[Dict[str, Any]] = []
    if results is None:
        return out
    for row in results:
        ra = _unmask_float(row[_RA_COL])
        dec = _unmask_float(row[_DEC_COL])
        flux = _unmask_float(row[_FLUX_COL])
        if ra is None or dec is None or flux is None or flux <= 0:
            continue
        mag = uJy_to_ab_mag(flux)
        if mag >= mag_floor:                       # too faint → skip
            continue
        out.append({"id": f"gal_{row[_ID_COL]}", "ra": ra, "dec": dec,
                    "mag_vis": mag})
    return out


def _run_query(query: str) -> Tuple[Any, str]:
    """Run a synchronous ADQL query; return ``(results_or_None, error)``."""
    try:
        job = Euclid.launch_job(query)
        return (job.get_results() if job is not None else None), ""
    except Exception as e:  # noqa: BLE001 — surfaced to the caller, never raised
        return None, f"{type(e).__name__}: {e}"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/miniforge3/envs/EuclidPolishEnv/bin/python -m pytest tests/test_galaxy_catalog.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add euclid_polish/euclid/galaxy_catalog.py tests/test_galaxy_catalog.py
git commit -m "galaxy_catalog: parse TAP results + VIS-mag brightness floor

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: galaxy_catalog — build() (query loop, lens-exclusion, seeded draw, cache)

**Files:**
- Modify: `euclid_polish/euclid/galaxy_catalog.py`
- Test: `tests/test_galaxy_catalog.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_galaxy_catalog.py`:

```python
class _FakeJob:
    def __init__(self, rows):
        self._rows = rows

    def get_results(self):
        return self._rows


def _fake_launch(query):
    # Field at ra=10 returns one good galaxy + one sitting ON the lens (excluded);
    # field at ra=20 returns two good galaxies.
    if "10.0," in query:
        return _FakeJob([
            {"object_id": 1, "right_ascension": 10.01, "declination": -5.0,
             "segmentation_area": 800.0, "flux_vis_psf": 50.0},
            {"object_id": 2, "right_ascension": 10.0, "declination": -5.0,
             "segmentation_area": 800.0, "flux_vis_psf": 50.0},
        ])
    return _FakeJob([
        {"object_id": 3, "right_ascension": 20.02, "declination": 30.0,
         "segmentation_area": 800.0, "flux_vis_psf": 50.0},
        {"object_id": 4, "right_ascension": 20.03, "declination": 30.01,
         "segmentation_area": 800.0, "flux_vis_psf": 50.0},
    ])


def _lens_csv(tmp_path):
    p = tmp_path / "lenses.csv"
    p.write_text("id,ra,dec,grade\nL1,10.0,-5.0,A\nL2,20.0,30.0,A\n")
    return str(p)


def test_build_draws_3n_and_excludes_lenses(monkeypatch, tmp_path):
    monkeypatch.setattr(gc, "login", lambda **k: True)
    monkeypatch.setattr(gc.Euclid, "launch_job", staticmethod(_fake_launch))
    out = tmp_path / "galaxies.csv"
    path, n = gc.build(str(out), n_galaxies=3, lens_catalog_path=_lens_csv(tmp_path),
                       seed=0, cone_radius_arcmin=3.0, oversample=4)
    rows = list(csv.DictReader(open(path)))
    ids = {r["id"] for r in rows}
    assert n == 3
    assert "gal_2" not in ids                       # within 10" of lens L1 → excluded
    assert ids == {"gal_1", "gal_3", "gal_4"}
    assert all(r["grade"] == "gal" for r in rows)


def test_build_requires_auth(monkeypatch, tmp_path):
    monkeypatch.setattr(gc, "login", lambda **k: False)
    with pytest.raises(RuntimeError):
        gc.build(str(tmp_path / "g.csv"), n_galaxies=3,
                 lens_catalog_path=_lens_csv(tmp_path), seed=0)


def test_build_reuses_cache_without_requery(monkeypatch, tmp_path):
    out = tmp_path / "galaxies.csv"
    out.write_text("id,ra,dec,grade\n"
                   "gal_1,10.0,-5.0,gal\ngal_2,11.0,-5.0,gal\ngal_3,12.0,-5.0,gal\n")
    called = {"login": False}
    monkeypatch.setattr(gc, "login",
                        lambda **k: called.__setitem__("login", True) or True)
    path, n = gc.build(str(out), n_galaxies=3,
                       lens_catalog_path="does-not-exist.csv", seed=0)
    assert n == 3 and called["login"] is False      # cache satisfied → no archive call


def test_build_seed_deterministic(monkeypatch, tmp_path):
    monkeypatch.setattr(gc, "login", lambda **k: True)
    monkeypatch.setattr(gc.Euclid, "launch_job", staticmethod(_fake_launch))
    a, _ = gc.build(str(tmp_path / "a.csv"), n_galaxies=2,
                    lens_catalog_path=_lens_csv(tmp_path), seed=7)
    b, _ = gc.build(str(tmp_path / "b.csv"), n_galaxies=2,
                    lens_catalog_path=_lens_csv(tmp_path), seed=7)
    ids_a = [r["id"] for r in csv.DictReader(open(a))]
    ids_b = [r["id"] for r in csv.DictReader(open(b))]
    assert ids_a == ids_b
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/miniforge3/envs/EuclidPolishEnv/bin/python -m pytest tests/test_galaxy_catalog.py::test_build_draws_3n_and_excludes_lenses -q`
Expected: FAIL — `AttributeError: ... has no attribute 'build'`.

- [ ] **Step 3: Write minimal implementation**

Add to `euclid_polish/euclid/galaxy_catalog.py`:

```python
def _near_any_lens(ra: float, dec: float, lenses: List[Dict[str, Any]],
                   radius_arcsec: float) -> bool:
    """True if (ra, dec) is within ``radius_arcsec`` of any lens position."""
    for lens in lenses:
        if angular_separation_arcsec(ra, dec, lens["ra"], lens["dec"]) < radius_arcsec:
            return True
    return False


def _read_cached(out_csv: str) -> List[Dict[str, Any]]:
    """Read a previously-written galaxy catalog (stable order), or []."""
    if not os.path.isfile(out_csv):
        return []
    rows: List[Dict[str, Any]] = []
    with open(out_csv, newline="") as f:
        for r in csv.DictReader(f):
            rows.append({"id": r["id"], "ra": float(r["ra"]),
                         "dec": float(r["dec"]), "grade": r.get("grade", GAL_GRADE)})
    return rows


def _write(out_csv: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "ra", "dec", "grade"])
        for r in rows:
            w.writerow([r["id"], r["ra"], r["dec"], r.get("grade", GAL_GRADE)])


def build(out_csv: Optional[str] = None, *, n_galaxies: int,
          lens_catalog_path: str, seed: int = 0,
          cone_radius_arcmin: float = 3.0, oversample: int = 4,
          regenerate: bool = False,
          log: Optional[Callable[[str], None]] = None) -> Tuple[str, int]:
    """Build (or top up) the galaxy catalog to ``n_galaxies`` rows; return ``(path, n)``.

    Drawn from the same fields as the lenses (cone queries around each lens
    RA/Dec). The CSV is cached: already-drawn galaxies are kept (stable ids →
    their cutouts are reused, never re-downloaded); only the shortfall is queried.
    """
    emit = log or (lambda m: None)
    out = out_csv or default_out_csv()

    cached = [] if regenerate else _read_cached(out)
    if len(cached) >= n_galaxies:
        return out, len(cached)                     # cache already satisfies the request

    if not login(allow_interactive=False):
        raise RuntimeError(
            "Euclid archive login required to build the galaxy catalog. Set "
            "EUCLID_USER/EUCLID_PASSWORD or a credentials file (same credentials "
            "the lens-cutout downloads use).")

    lenses = read_eval_catalog(lens_catalog_path)
    rng = random.Random(seed)
    fields = list(lenses)
    rng.shuffle(fields)

    pool: List[Dict[str, Any]] = []
    seen_ids = {r["id"] for r in cached}
    target_pool = oversample * n_galaxies
    radius_deg = cone_radius_arcmin / 60.0

    for fld in fields:
        if len(cached) + len(pool) >= target_pool:
            break
        results, err = _run_query(galaxy_adql(fld["ra"], fld["dec"], radius_deg))
        if err:
            emit(f"  galaxy query failed at ({fld['ra']:.4f}, {fld['dec']:.4f}): {err}")
            continue
        for cand in _candidates_from_results(results):
            if cand["id"] in seen_ids:
                continue
            if _near_any_lens(cand["ra"], cand["dec"], lenses, LENS_EXCLUDE_ARCSEC):
                continue
            seen_ids.add(cand["id"])
            pool.append(cand)

    rng.shuffle(pool)
    need = n_galaxies - len(cached)
    drawn = pool[:need]
    rows = cached + [{"id": d["id"], "ra": d["ra"], "dec": d["dec"],
                      "grade": GAL_GRADE} for d in drawn]
    if len(rows) < n_galaxies:
        emit(f"⚠ only {len(rows)} galaxies found (< {n_galaxies} requested); using all")
    _write(out, rows)
    return out, len(rows)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/miniforge3/envs/EuclidPolishEnv/bin/python -m pytest tests/test_galaxy_catalog.py -q`
Expected: PASS (8 passed).

- [ ] **Step 5: Commit**

```bash
git add euclid_polish/euclid/galaxy_catalog.py tests/test_galaxy_catalog.py
git commit -m "galaxy_catalog: build() — field queries, lens-exclusion, cached draw

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: CLI — scripts/fetch_galaxy_catalog.py

**Files:**
- Create: `scripts/fetch_galaxy_catalog.py`
- Test: `tests/test_galaxy_catalog.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_galaxy_catalog.py`:

```python
def test_cli_invokes_build(monkeypatch, tmp_path):
    import importlib
    cli = importlib.import_module("scripts.fetch_galaxy_catalog")
    captured = {}

    def fake_build(out_csv=None, *, n_galaxies, lens_catalog_path, seed=0, **kw):
        captured.update(n_galaxies=n_galaxies, lens=lens_catalog_path, out=out_csv)
        return (out_csv or "galaxies.csv"), n_galaxies

    monkeypatch.setattr(cli.galaxy_catalog, "build", fake_build)
    rc = cli.main(["--n", "6", "--lens", "lenses.csv", "--out", str(tmp_path / "g.csv")])
    assert rc == 0
    assert captured["n_galaxies"] == 6 and captured["lens"] == "lenses.csv"
```

(`scripts/` is importable — the repo root is on `sys.path` under pytest; other tests import `scripts.*` the same way.)

- [ ] **Step 2: Run test to verify it fails**

Run: `~/miniforge3/envs/EuclidPolishEnv/bin/python -m pytest tests/test_galaxy_catalog.py::test_cli_invokes_build -q`
Expected: FAIL — `ModuleNotFoundError: scripts.fetch_galaxy_catalog`.

- [ ] **Step 3: Write minimal implementation**

Create `scripts/fetch_galaxy_catalog.py`:

```python
#!/usr/bin/env python
"""Build a real-field-galaxy evaluation catalog from the Euclid archive.

Thin CLI over :mod:`euclid_polish.euclid.galaxy_catalog` (the shared build logic
also used by the grouped eval runner). Queries ``catalogue.mer_catalogue`` around
the strong-lens fields for clean, resolved, bigger-end galaxies and writes a
normalized ``id,ra,dec,grade`` CSV (``grade="gal"``). Needs Euclid archive
credentials (``EUCLID_USER``/``EUCLID_PASSWORD`` or a credentials file).

Usage::

    python scripts/fetch_galaxy_catalog.py --n 60 --lens path/to/lenses.csv
    python scripts/fetch_galaxy_catalog.py --n 60 --lens lenses.csv --out gals.csv
    python scripts/fetch_galaxy_catalog.py --n 60 --lens lenses.csv --regenerate
"""
from __future__ import annotations

import argparse
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from euclid_polish.euclid import galaxy_catalog


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, required=True,
                   help="number of galaxies to draw")
    p.add_argument("--lens", required=True,
                   help="normalized lens catalog CSV (id,ra,dec,grade) to sample fields from")
    p.add_argument("--out", default=None,
                   help="output CSV (default: Config.EVAL_CATALOG_DIR/galaxy_catalog/galaxies.csv)")
    p.add_argument("--seed", type=int, default=0, help="random seed for the draw")
    p.add_argument("--regenerate", action="store_true",
                   help="ignore any cached catalog and re-query")
    args = p.parse_args(argv)

    out_csv, n = galaxy_catalog.build(
        args.out, n_galaxies=args.n, lens_catalog_path=args.lens,
        seed=args.seed, regenerate=args.regenerate, log=print)
    print(f"  ✓ wrote {n} galaxies → {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/miniforge3/envs/EuclidPolishEnv/bin/python -m pytest tests/test_galaxy_catalog.py -q`
Expected: PASS (9 passed).

- [ ] **Step 5: Commit**

```bash
git add scripts/fetch_galaxy_catalog.py tests/test_galaxy_catalog.py
git commit -m "galaxy_catalog: CLI to build the galaxy catalog from the archive

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: grouped_runner — `_galaxy_plan` helper + `include_galaxies` wiring

**Files:**
- Modify: `euclid_polish/eval/grouped_runner.py`
- Test: `tests/test_eval_catalog.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_eval_catalog.py` (top-level functions, after the imports/existing tests):

```python
def test_galaxy_plan_counts_3x_grade_a(monkeypatch, tmp_path):
    import csv as _csv
    from euclid_polish.eval import grouped_runner
    from euclid_polish.euclid import galaxy_catalog
    calls = {}

    def fake_build(out_csv=None, *, n_galaxies, lens_catalog_path, seed=0, **kw):
        calls["n_galaxies"] = n_galaxies
        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
        with open(out_csv, "w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(["id", "ra", "dec", "grade"])
            for i in range(n_galaxies):
                w.writerow([f"gal_{i}", 10.0 + i * 1e-3, -5.0, "gal"])
        return out_csv, n_galaxies

    monkeypatch.setattr(galaxy_catalog, "build", fake_build)
    monkeypatch.setattr(galaxy_catalog, "default_out_csv",
                        lambda: str(tmp_path / "galaxies.csv"))
    lens_plan = [("A", [{}, {}, {}, {}]), ("B", [{}] * 10), ("C", [{}] * 10)]
    rows = grouped_runner._galaxy_plan(lens_plan, catalog="lenses.csv",
                                       seed=0, log=lambda m: None)
    assert calls["n_galaxies"] == 12               # 3 × 4 grade-A
    assert len(rows) == 12 and all(r["grade"] == "gal" for r in rows)


def test_galaxy_plan_graceful_when_build_fails(monkeypatch, tmp_path):
    from euclid_polish.eval import grouped_runner
    from euclid_polish.euclid import galaxy_catalog
    monkeypatch.setattr(galaxy_catalog, "default_out_csv",
                        lambda: str(tmp_path / "g.csv"))
    monkeypatch.setattr(galaxy_catalog, "build",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no auth")))
    rows = grouped_runner._galaxy_plan([("A", [{}, {}])], catalog="l.csv",
                                       seed=0, log=lambda m: None)
    assert rows == []                              # galaxies must not kill A/B/C
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/miniforge3/envs/EuclidPolishEnv/bin/python -m pytest tests/test_eval_catalog.py::test_galaxy_plan_counts_3x_grade_a -q`
Expected: FAIL — `AttributeError: module ... grouped_runner has no attribute '_galaxy_plan'`.

- [ ] **Step 3: Write minimal implementation**

In `euclid_polish/eval/grouped_runner.py`, add the import at the top of the module (with the other `from euclid_polish...` imports, around line 15-18):

```python
from euclid_polish.euclid import galaxy_catalog
```

Add `LENS_GRADES` constant is already present; below it, add the helper:

```python
def _galaxy_plan(lens_plan, *, catalog: str, seed: int,
                 log: Callable[[str], None]) -> List[Dict[str, Any]]:
    """Rows for the real-galaxy group: 3 × the realized grade-A lens count.

    Galaxies are drawn from the same fields as the lenses and cached, so each
    cutout is downloaded at most once. Any failure (no archive auth, sparse
    fields) logs and yields an empty plan — galaxies must never kill the A/B/C
    run.
    """
    n_a = next((len(rows) for g, rows in lens_plan if g == "A"), 0)
    n_gal = 3 * n_a
    if n_gal <= 0:
        return []
    try:
        gal_csv = galaxy_catalog.default_out_csv()
        galaxy_catalog.build(gal_csv, n_galaxies=n_gal,
                             lens_catalog_path=catalog, seed=seed, log=log)
        rows = read_eval_catalog(gal_csv, max_n=n_gal)
        log(f"galaxies: {len(rows)} (3 × {n_a} grade-A) from {gal_csv}")
        return rows
    except Exception as e:  # noqa: BLE001 — galaxies must not kill A/B/C
        log(f"galaxies skipped: {type(e).__name__}: {e}")
        return []
```

Note: `read_eval_catalog` is already imported at the top of `grouped_runner.py` (via `from euclid_polish.euclid.eval_catalog import read_eval_catalog`). If it is not, add that import at the top too.

Then wire it in. Change the signature of `run_grouped_analysis` to add the flag (in the keyword block, next to `include_synthetic`):

```python
    include_synthetic: bool = True,
    include_galaxies: bool = True,
```

And in the body, replace this block:

```python
    n_lens = sum(len(r) for _, r in lens_plan)
    total = n_lens + (2 * n if include_synthetic else 0)
```

with:

```python
    if include_galaxies:
        gal_rows = _galaxy_plan(lens_plan, catalog=catalog, seed=seed, log=_emit)
        if gal_rows:
            lens_plan.append(("gal", gal_rows))    # processed by the same A/B/C loop
    n_lens = sum(len(r) for _, r in lens_plan)     # now includes galaxies
    total = n_lens + (2 * n if include_synthetic else 0)
```

(The existing `for g, rows in lens_plan:` loop then downloads/reuses each galaxy via `can_reuse_eval_object` exactly like A/B/C — no new per-object code. `read_eval_catalog` needs `Any, Dict, List` which are already imported in this module.)

- [ ] **Step 4: Run test to verify it passes**

Run: `~/miniforge3/envs/EuclidPolishEnv/bin/python -m pytest tests/test_eval_catalog.py -q`
Expected: PASS (all existing + 2 new).

- [ ] **Step 5: Commit**

```bash
git add euclid_polish/eval/grouped_runner.py tests/test_eval_catalog.py
git commit -m "grouped_runner: add real-galaxy group (3 × grade-A), reuse cutouts

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 6: Display — add `gal` to GROUPS and GROUP_COLORS

**Files:**
- Modify: `euclid_polish/eval/lensfinder_eval.py:17`
- Modify: `euclid_polish/eval/zoobot_morph.py:258-259`
- Test: `tests/test_lensfinder_eval.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_lensfinder_eval.py`:

```python
def test_gal_group_registered():
    from euclid_polish.eval import lensfinder_eval
    from euclid_polish.eval.zoobot_morph import GROUP_COLORS, _group_color
    assert "gal" in lensfinder_eval.GROUPS
    assert "gal" in GROUP_COLORS
    # distinct from every other group's colour
    assert _group_color("gal") not in {GROUP_COLORS[g] for g in GROUP_COLORS if g != "gal"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/miniforge3/envs/EuclidPolishEnv/bin/python -m pytest tests/test_lensfinder_eval.py::test_gal_group_registered -q`
Expected: FAIL — `"gal" not in lensfinder_eval.GROUPS`.

- [ ] **Step 3: Write minimal implementation**

In `euclid_polish/eval/lensfinder_eval.py`, change line 17:

```python
GROUPS = ("A", "B", "C", "gal", "syn-lens", "syn-gal")
```

In `euclid_polish/eval/zoobot_morph.py`, change the `GROUP_COLORS` dict (lines 258-259):

```python
GROUP_COLORS = {"A": "#2a5db0", "B": "#2e8b57", "C": "#b8860b",
                "gal": "#17a2b8",
                "syn-lens": "#b03a3a", "syn-gal": "#7a4fb0"}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/miniforge3/envs/EuclidPolishEnv/bin/python -m pytest tests/test_lensfinder_eval.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add euclid_polish/eval/lensfinder_eval.py euclid_polish/eval/zoobot_morph.py tests/test_lensfinder_eval.py
git commit -m "eval: register real-galaxy group (gal) + its plot colour

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 7: Real ROC panel — figure 2×2 → 2×3 in render_lensfinder_summary

**Files:**
- Modify: `euclid_polish/eval/lensfinder_eval.py:84,101-127`
- Test: `tests/test_lensfinder_eval.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_lensfinder_eval.py`:

```python
def test_summary_renders_with_real_galaxies(tmp_path):
    from euclid_polish.eval import lensfinder_eval
    run = tmp_path / "run"
    run.mkdir()
    # A/B/C lenses (high SR P(lens)), real galaxies (low), + synthetic for the
    # synthetic ROC. Real ROC needs both real-lens positives and gal negatives.
    lines = ["id,grade,p_lens_lr,p_lens_sr,p_lens_hr"]
    for i in range(6):
        lines.append(f"A{i},A,0.6,0.9,")
        lines.append(f"B{i},B,0.5,0.8,")
        lines.append(f"C{i},C,0.4,0.7,")
        lines.append(f"G{i},gal,0.2,0.1,")
        lines.append(f"SL{i},syn-lens,0.5,0.85,0.95")
        lines.append(f"SG{i},syn-gal,0.2,0.15,0.05")
    (run / "lens_scores.csv").write_text("\n".join(lines) + "\n")
    out_png = run / "summary.png"
    res = lensfinder_eval.render_lensfinder_summary(str(run), str(out_png))
    assert res == str(out_png) and out_png.is_file() and out_png.stat().st_size > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/miniforge3/envs/EuclidPolishEnv/bin/python -m pytest tests/test_lensfinder_eval.py::test_summary_renders_with_real_galaxies -q`
Expected: PASS already (the existing 2×2 renderer tolerates `gal` rows) **or** FAIL only if the renderer errors. Either way, proceed — the goal of Step 3 is the new panels. (If it already passes, the assertion that the *real ROC* panel exists is added implicitly by the layout change; this test guards against the 2×3 change crashing.)

- [ ] **Step 3: Write minimal implementation**

In `euclid_polish/eval/lensfinder_eval.py`, change the subplots line (line 84):

```python
    fig, ax = plt.subplots(2, 3, figsize=(16.5, 9.0))
```

Immediately after the existing panel **(2) Synthetic ROC** block (the block ending at
`a.legend(fontsize=9, loc="lower right"); a.grid(alpha=0.2)` for the synthetic ROC,
around line 116), insert the new **real ROC** panel:

```python
    # (2b) Real ROC — A/B/C lenses (positive) vs real field galaxies (negative).
    # No HR: real objects have no ground truth. This is the payoff of having real
    # negatives — does SR improve real lens-vs-galaxy separability?
    a = ax[0, 2]
    real = [r for r in rows if r["grade"] in ("A", "B", "C", "gal")]
    is_lens = lambda r: 1 if r["grade"] in ("A", "B", "C") else 0
    if real and any(is_lens(r) for r in real) and any(not is_lens(r) for r in real):
        for recon, color in (("lr", "#888888"), ("sr", "#2a5db0")):
            pairs = [(r[recon], is_lens(r)) for r in real if math.isfinite(r[recon])]
            if len(pairs) >= 2 and len({p[1] for p in pairs}) == 2:
                s = [p[0] for p in pairs]; y = [p[1] for p in pairs]
                fpr, tpr, _ = lm.roc_curve(s, y)
                a.plot(fpr, tpr, color=color, lw=1.8,
                       label=f"{recon.upper()} (AUC {lm.auc(fpr, tpr):.3f})")
        a.plot([0, 1], [0, 1], ":", color="#bbb")
    a.set_xlabel("False positive rate"); a.set_ylabel("True positive rate")
    a.set_title("Real lens (A/B/C) vs galaxy — ROC")
    a.legend(fontsize=9, loc="lower right"); a.grid(alpha=0.2)
```

Then, after the existing panel **(4) P(lens) vs expert grade** block (the one ending
`a.grid(alpha=0.2)` near line 143), insert the new **real binary-score** panel:

```python
    # (4b) Real binary score separation — A/B/C lenses vs real galaxies (SR P(lens)).
    a = ax[1, 2]
    bins = np.linspace(0, 1, 26)
    lens_vals = _finite_pairs([r for r in rows if r["grade"] in ("A", "B", "C")], "sr")
    gal_vals = _finite_pairs([r for r in rows if r["grade"] == "gal"], "sr")
    if lens_vals:
        a.hist(lens_vals, bins=bins, color="#2a5db0", alpha=0.55, density=True,
               label=f"lenses A/B/C (n={len(lens_vals)})")
    if gal_vals:
        a.hist(gal_vals, bins=bins, color=_group_color("gal"), alpha=0.55, density=True,
               label=f"galaxies (n={len(gal_vals)})")
    a.set_xlabel("P(lens) — SR"); a.set_ylabel("density")
    a.set_title("Real: lens vs galaxy score"); a.legend(fontsize=8); a.grid(alpha=0.2)
```

(`_group_color`, `np`, `math`, `lm`, `_finite_pairs` are all already imported/defined
in this function or module. The two new axes are `ax[0,2]` and `ax[1,2]`; the four
existing panels keep their `ax[0,0] ax[0,1] ax[1,0] ax[1,1]` positions unchanged.)

- [ ] **Step 4: Run test to verify it passes**

Run: `~/miniforge3/envs/EuclidPolishEnv/bin/python -m pytest tests/test_lensfinder_eval.py -q`
Expected: PASS (all, including the new render test).

- [ ] **Step 5: Manually verify the figure (optional but recommended)**

Run:
```bash
~/miniforge3/envs/EuclidPolishEnv/bin/python -c "from euclid_polish.eval import lensfinder_eval; print(lensfinder_eval.render_lensfinder_summary('data/eval_results','/tmp/gal_summary.png'))"
```
Open `/tmp/gal_summary.png` — confirm the top row is [SR-vs-LR shift | synthetic ROC | real ROC] and the bottom row is [ridgeline (now with a `gal` band) | score vs grade | real lens-vs-galaxy score]. (Only meaningful once a run with `gal` objects exists.)

- [ ] **Step 6: Commit**

```bash
git add euclid_polish/eval/lensfinder_eval.py tests/test_lensfinder_eval.py
git commit -m "lensfinder_eval: add real ROC (A/B/C vs gal) + binary score panel

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 8: WebUI — "include real galaxies" toggle

**Files:**
- Modify: `euclid_polish/web/routes/evaluation.py:233-249`
- Modify: `euclid_polish/web/templates/evaluation.html:292-293,870-877`
- Test: `tests/test_eval_catalog.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_eval_catalog.py` (inside the same test class as `test_run_grouped_spawns_job`, mirroring it):

```python
    def test_run_grouped_passes_galaxies_flag(self, client, tmp_path, monkeypatch):
        from euclid_polish.eval import grouped_runner
        monkeypatch.setattr(Config, "EVAL_RESULTS_DIR", str(tmp_path / "res"))
        captured = {}
        monkeypatch.setattr(grouped_runner, "run_grouped_analysis",
                            lambda **k: captured.update(k) or {"n": 0})
        r = client.post("/api/evaluation/run-grouped",
                        data={"n": "3", "galaxies": "0"})
        assert r.status_code == 200
        assert captured["include_galaxies"] is False
        r = client.post("/api/evaluation/run-grouped",
                        data={"n": "3", "galaxies": "1"})
        assert captured["include_galaxies"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/miniforge3/envs/EuclidPolishEnv/bin/python -m pytest "tests/test_eval_catalog.py::TestEvaluationRoutes::test_run_grouped_passes_galaxies_flag" -q`
Expected: FAIL — `KeyError: 'include_galaxies'` (route doesn't pass it yet).
(If the class name differs, run the test by `-k test_run_grouped_passes_galaxies_flag`.)

- [ ] **Step 3: Write minimal implementation**

In `euclid_polish/web/routes/evaluation.py`, in `api_evaluation_run_grouped` (the
`include_synth = ...` area, ~line 238), add the galaxies flag parse right after it:

```python
        include_synth = str(f.get("synthetic", "1")).lower() in ("1", "true", "on", "yes")
        include_gal = str(f.get("galaxies", "1")).lower() in ("1", "true", "on", "yes")
```

and pass it into the call inside `_run` (~line 244-247):

```python
        def _run(cap):
            return grouped_runner.run_grouped_analysis(
                out_dir=out_dir, n=n, include_synthetic=include_synth,
                include_galaxies=include_gal,
                on_progress=lambda i, t, lbl: cap.tick(i, t, lbl),
                log=lambda m: cap.write(m if m.endswith("\n") else m + "\n"))
```

In `euclid_polish/web/templates/evaluation.html`, add the checkbox after the
`include synthetic` label (line 293):

```html
      <input type="checkbox" id="gpSynth" checked> include synthetic</label>
    <label class="checkbox-field" style="flex-direction:row; align-items:center; gap:6px;"
           title="3 × the grade-A count, drawn from the same fields — a real negative control">
      <input type="checkbox" id="gpGal" checked> include real galaxies</label>
```

and set the form field in the click handler (line 873):

```javascript
    fd.set('synthetic', $('gpSynth').checked ? '1' : '0');
    fd.set('galaxies', $('gpGal').checked ? '1' : '0');
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/miniforge3/envs/EuclidPolishEnv/bin/python -m pytest "tests/test_eval_catalog.py" -q`
Expected: PASS (all, including the new flag test).

- [ ] **Step 5: Commit**

```bash
git add euclid_polish/web/routes/evaluation.py euclid_polish/web/templates/evaluation.html tests/test_eval_catalog.py
git commit -m "evaluation WebUI: 'include real galaxies' toggle (default on)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 9: Full-suite regression + push

**Files:** none (verification only)

- [ ] **Step 1: Run the relevant suites**

Run:
```bash
~/miniforge3/envs/EuclidPolishEnv/bin/python -m pytest \
  tests/test_galaxy_catalog.py tests/test_eval_catalog.py \
  tests/test_lensfinder_eval.py tests/test_fasrc_integration.py -q
```
Expected: all PASS.

- [ ] **Step 2: Push**

```bash
git push
```
Expected: branch `main` updated on origin.

---

## Self-Review

**Spec coverage:**
- Unit 1 (galaxy_catalog module) → Tasks 1-3. ✓ (ADQL cuts, size mapping, auth, field-cone loop, lens-exclusion, seeded draw, incremental cache).
- CLI → Task 4. ✓
- Unit 2 (grouped_runner wiring, 3×A, reuse) → Task 5. ✓ (reuse is the existing `can_reuse_eval_object` path — galaxies flow through the unchanged A/B/C loop).
- Unit 3 (GROUPS/colors + real ROC) → Tasks 6-7. ✓ (ridgeline + SR-vs-LR scatter auto-include `gal` via GROUPS; real ROC + binary-score panels added).
- Unit 4 (WebUI/CLI toggle) → Tasks 4 + 8. ✓
- Caching requirement (downloads at most once) → Task 3 (catalog cache, stable ids) + Task 5 (per-object `can_reuse_eval_object`). ✓
- Error handling (auth missing raises; galaxies never kill A/B/C; shortfall logged) → Task 3 (`RuntimeError` on no auth, shortfall log) + Task 5 (`_galaxy_plan` try/except). ✓
- Testing (mocked archive; 3N count; seed determinism; lens-exclusion; figure renders) → Tasks 3, 5, 7. ✓

**Placeholder scan:** No TBD/TODO; every code step is complete. The MER column names are concrete constants with a documented one-line-change escape hatch (`--probe`/schema note) rather than a placeholder.

**Type consistency:** `build(out_csv, *, n_galaxies, lens_catalog_path, seed, cone_radius_arcmin, oversample, regenerate, log)` is called consistently in Task 4 (CLI), Task 5 (`_galaxy_plan`), and Task 8 (via the runner). `_galaxy_plan(lens_plan, *, catalog, seed, log)` and `galaxy_adql(ra, dec, radius_deg)` / `_candidates_from_results(results, mag_floor)` signatures match their tests. `grade` value `"gal"` is identical across the module, GROUPS, GROUP_COLORS, and the figure panels.

**Known residual risk (flagged, not a plan gap):** the exact `mer_catalogue` morphology/classification column names (`segmentation_area`, `point_like_flag`, `spurious_flag`) should be confirmed against the live schema with credentials; if any differs, change the single constant in `galaxy_catalog.py`. Unit tests don't depend on the real names (archive is mocked), so the plan is fully testable offline.
