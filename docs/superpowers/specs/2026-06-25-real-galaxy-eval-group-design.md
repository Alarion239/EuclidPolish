# Real field galaxies as a negative-control eval group

**Date:** 2026-06-25
**Status:** Approved design — ready for implementation plan

## Motivation

The `/evaluation` page currently scores three real strong-lens groups (A/B/C
expert grades) and two synthetic groups (syn-lens, syn-gal). It has **no real
negatives** — objects that are confidently *not* lenses. Synthetic galaxies are
a proxy, but they carry the synthesis pipeline's own biases. Adding **real field
galaxies** (≈99.999% not gravitational lenses) gives a true negative control
drawn from the same instrument and pipeline as the real lenses, enabling a
real-data lens-vs-galaxy comparison (a real ROC) instead of only a synthetic one.

Because real galaxies are "just Euclid LR cutouts", they flow through the exact
same per-object pipeline as A/B/C lenses (download → SR → size enforcement →
Zoobot morphology → lens-finder P(lens) → gallery). The only genuinely new work
is (1) a galaxy *catalog source* and (2) wiring a new group into the grouped
runner and the analysis figure.

## Key facts established during design

- The eval pipeline is **grade-agnostic and RA/Dec-driven**:
  `catalog_runner.eval_catalog_object(model, {id,ra,dec,grade}, …)` downloads a
  4-band cutout via `reconstruct_cutout_at(ra, dec, size)` and runs SR; the
  `grade` string is just a group label carried into `manifest.csv`. Everything
  downstream keys off that column.
- The Euclid archive is already queryable: `astroquery.esa.euclid.Euclid`
  ADQL cone queries on `catalogue.mer_catalogue`
  (`euclid_polish/euclid/catalog.py:431`, `_query_bright_stars`).
- Auth is handled centrally: `euclid_polish/euclid/auth.py` `login()` resolves
  `EUCLID_USER`/`EUCLID_PASSWORD` env → credentials file
  (`Config.DEFAULT_CREDENTIALS_FILE`) → interactive. Cutout downloads already
  depend on this, so galaxy queries reuse it.
- Canonical eval geometry: `EVAL_LR_SIZE = 53` VIS px (≈5.3″ at 0.1″/px),
  `EVAL_HR_SIZE = 106`. A galaxy with diameter ≤ 50 px (5″) fits the LR stamp;
  anything larger overflows and is dropped by `enforce_object_sizes`. The ≤5″
  cap is therefore both scientifically sensible and geometrically required.
- The lens catalog (`euclid_polish/euclid/lens_catalog.py`) normalizes the
  Zenodo Q1 discovery CSV to `id,ra,dec,grade,subset`; cutouts are re-fetched
  from the archive per RA/Dec.

## Design decisions (resolved with the user)

1. **Sky sampling:** draw galaxies from the **same fields as the A/B/C lenses**
   (cone-query around the lens RA/Decs) — matched depth/PSF/field, reuses the
   existing cone-query machinery, guaranteed inside the Q1 footprint.
2. **Selection:** **bigger-end, clean & resolved** galaxies (see cuts below).
3. **Real ROC panel:** **yes** — A/B/C (positive) vs real galaxies (negative),
   LR-vs-SR curves.
4. **Count:** **3N** real galaxies, where **N = the realized A-class lens count**
   in the run (`len(read_eval_catalog(catalog, grade="A", max_n=n))`), not the
   requested `n`.

## Architecture

### Unit 1 — `euclid_polish/euclid/galaxy_catalog.py` (new)

Purpose: produce a normalized galaxy catalog CSV (`id,ra,dec,grade="gal"`) by
querying the live MER catalogue around the lens fields. Mirrors `lens_catalog.py`
in shape (a single source of truth shared by a CLI script and the WebUI), but the
"fetch" is an archive query rather than a Zenodo download.

Public surface:

```
default_out_csv() -> str
    # <EVAL_CATALOG_DIR>/galaxy_catalog/galaxies.csv

build(out_csv=None, *, n_galaxies, lens_catalog_path=None, seed=0,
      cone_radius_arcmin=3.0, oversample=4, log=None) -> tuple[str, int]
    # Query + select + draw exactly n_galaxies; write normalized CSV; return (path, n).
```

Algorithm:

1. Ensure a Euclid session: `auth.login(allow_interactive=False)`; raise a clear
   error if not authenticated (same requirement as lens cutout downloads).
2. Read the lens catalog (A/B/C) → list of lens RA/Decs (the "fields").
3. Shuffle the fields with `seed`. For each field, ADQL cone-query
   `catalogue.mer_catalogue` (radius `cone_radius_arcmin`) for galaxies passing
   the cuts. Accumulate candidates (deduping by MER object id) until
   `len(candidates) >= oversample * n_galaxies`, then stop early.
4. Drop any candidate within **10″** of any lens-catalog entry (guarantees "not
   a known lens").
5. Seeded-random draw of exactly `n_galaxies` from the pool; write
   `id,ra,dec,grade` rows with `grade="gal"` and a stable `id` (e.g.
   `gal_<mer_object_id>`).

ADQL selection cuts (column names verified against the live MER schema at
implementation; primary choice + fallback noted):

- **Galaxy, not star:** `point_like_flag = 0` (fallback `point_like_prob < 0.2`).
- **Clean detection:** `det_quality_flag = 0` (the ePSF "clean" cut).
- **Not spurious:** `spurious_flag = 0` (fallback `spurious_prob < 0.2`).
- **Size (bigger-end, capped):** diameter in ~**2″–5″**. Implemented via
  `segmentation_area` thresholds (a roughly circular D-arcsec object has
  area ≈ π·(D/2 / 0.1″)² px²; 2″→~314 px², 5″→~1963 px²), or `semimajor_axis`
  in px if that proves the cleaner size proxy. Hard upper cap at 5″ so the
  stamp fits.
- **Brightness floor:** VIS magnitude bright enough to be well-resolved (≈
  mag_vis < 22; derived from `flux_vis_*` via the existing `uJy_to_ab_mag` /
  `Config.AB_ZP_UJY`). Tunable.

Returned columns selected by the query: at minimum `object_id`,
`right_ascension`, `declination`, plus the size/flux/flag columns needed for the
cuts.

### Unit 2 — `grouped_runner.run_grouped_analysis` wiring

- New parameter `include_galaxies: bool = True`.
- After building `lens_plan`, compute `n_a = len(rows for grade "A")`,
  `n_gal = 3 * n_a`. If `include_galaxies and n_gal > 0`: ensure the galaxy
  catalog exists (build/cache via `galaxy_catalog.build(n_galaxies=n_gal, …)`),
  read it, and append a `("gal", rows)` plan **alongside** the lens plan — the
  galaxy loop is identical to the A/B/C loop (`eval_catalog_object` → download +
  SR → `enforce_object_sizes` → `reuse_catalog_object`). No new per-object code.
- `total` (progress denominator) includes `n_gal`. Model load is gated the same
  way (`needs_lens_model` extended to galaxies that lack cached FITS).
- Caching: the galaxy catalog CSV is cached under `EVAL_CATALOG_DIR`; a re-run
  reuses it. A `regenerate_galaxies` flag (or deleting the CSV) forces a
  re-query. Per-object FITS reuse works exactly as for A/B/C.

### Unit 3 — Display

- `euclid_polish/eval/lensfinder_eval.py`: `GROUPS` →
  `("A","B","C","gal","syn-lens","syn-gal")`.
- `euclid_polish/eval/zoobot_morph.py`: add a distinct `GROUP_COLORS["gal"]`
  (e.g. a teal/orange not already used by A/B/C/syn groups).
- **Real ROC panel** in `render_lensfinder_summary`: positives = A/B/C with a
  finite score, negatives = `gal`; plot LR and SR ROC + AUC (no HR — real
  objects have no ground-truth HR). The figure grows **2×2 → 2×3**: keep the
  existing four panels, add the real ROC beside the synthetic ROC; the sixth
  cell holds an overflow panel (e.g. real-vs-synthetic AUC summary) or is left
  blank. The ridgeline score-distribution and the SR-vs-LR P(lens) scatter pick
  up `gal` automatically once it's in `GROUPS`.
- Gallery + cutout viewer: `gal` objects appear with no viewer changes; the
  P(lens) badge already works per object/tier.

### Unit 4 — WebUI / CLI

- The grouped-run endpoint (`/api/evaluation/run-grouped`) gains an
  `include_galaxies` form flag (default on). The count is automatic (3×A); no
  galaxy-count knob is exposed.
- A "Fetch galaxy catalog" affordance parallel to the existing lens-catalog
  fetch, or lazy build on first grouped run. A thin CLI
  (`scripts/fetch_galaxy_catalog.py`) mirrors `scripts/fetch_lens_catalog.py`.

## Data flow

```
lens_catalog (A/B/C RA/Decs)
   │  cone-query mer_catalogue (same fields)            ← Unit 1
   ▼
galaxy_catalog.csv  (id,ra,dec,grade=gal), 3N rows
   │  read_eval_catalog
   ▼
grouped_runner: A,B,C,gal lens-like loop + syn loop     ← Unit 2
   │  eval_catalog_object → reconstruct_cutout_at → SR → enforce_object_sizes
   ▼
manifest.csv (grade ∈ {A,B,C,gal,syn-lens,syn-gal})
   │  Zoobot morphology + lensfinder score (P(lens) lr/sr per object)
   ▼
gallery + analysis figure (ridgeline, P(lens) scatter, synthetic ROC,
                            NEW real ROC: A/B/C vs gal)  ← Unit 3
```

## Error handling

- Galaxy query needs auth: if `auth.login(allow_interactive=False)` fails, raise
  a clear "configure Euclid credentials" error before any query. The grouped run
  surfaces it like any other prep error, and (per existing pattern) galaxies must
  not kill the A/B/C run — a galaxy-catalog build failure logs and continues with
  the lens + synthetic groups (mirrors the synthetic try/except).
- A galaxy whose cutout download fails or whose stamp comes out below 53/106 is
  dropped exactly like a bad A/B/C object (captured in `manifest.error`).
- If the candidate pool is smaller than 3N (sparse fields), use all candidates
  and log the shortfall; never silently pad.

## Testing

- `tests/test_galaxy_catalog.py`: mock `Euclid.launch_job`/`launch_job_async`
  to return synthetic MER rows. Assert: ADQL string contains the galaxy/size/
  quality cuts; exactly 3N drawn; seed determinism (same seed → same draw);
  candidates within 10″ of a lens excluded; "accumulate until oversample"
  stops early; auth-missing raises.
- `tests/test_grouped_runner_galaxies.py` (or extend existing grouped test):
  tiny fake lens catalog + mocked `eval_catalog_object`/download → assert the
  `gal` group is planned at `3 × A-count` and rows land in the manifest with
  `grade="gal"`.
- `tests/test_lensfinder_eval.py`: extend with a manifest containing `gal` +
  A/B/C scores → assert the real ROC panel renders (figure has the new axes,
  AUC computed) and `gal` appears in the ridgeline.

## Defaults (tunable)

| Parameter | Default |
|---|---|
| Group label | `gal` |
| Galaxy count | `3 × N_A` (realized A-class count) |
| Cone radius | 3′ |
| Size window | 2″–5″ diameter (hard cap 5″ / 50 px) |
| VIS mag floor | ≲ 22 |
| Lens-exclusion radius | 10″ |
| Oversample factor | 4× |
| `include_galaxies` | on |

## Out of scope (YAGNI)

- No survey-wide random sampling (rejected in favor of matched fields).
- No new per-object download/SR code (galaxies reuse the A/B/C path verbatim).
- No galaxy-count UI knob (count is derived from A).
- No HR for real galaxies (no ground truth exists).
