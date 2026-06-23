# Source-centered synthetic evaluation (syn-lens / syn-gal)

**Date:** 2026-06-23
**Status:** approved (pending spec review)

## Problem

The `/evaluation` grouped analysis has four groups: real lens cutouts **A/B/C**
(LR+SR, no truth) and **synthetic** (LR→SR→HR, with truth). The synthetic group
feeds Zoobot the *whole* validation field. Zoobot is a single-galaxy classifier
with no detection/segmentation — it center-crops and globally pools one frame
into one vector, assuming a centered target galaxy. A crowded synthetic field
violates that assumption, so the synthetic Zoobot vector is an ill-defined
aggregate of the scene. Concretely:

- **Within** synthetic, the before/after/HR comparison is still valid (same
  field, identical framing → any vector shift is image-quality, not framing).
- **Across** groups it is confounded: synthetic points sit in a different region
  of feature space because they are whole fields vs. centered galaxies. The PCA
  shift map's group separation is then a framing artifact, not a science signal.

We want synthetic inputs that are genuine centered single-object postage stamps,
comparable to the A/B/C lens cutouts, and we want to distinguish synthetic
*lenses* from synthetic *field galaxies*.

## Goal

Replace the single full-field `synthetic` group with two **source-centered**
subgroups, each N postage stamps cropped M×M (HR pixels) around exactly one
source per field:

- **`syn-lens`** — centered on a synthetic strong-lens system.
- **`syn-gal`** — centered on a synthetic field galaxy.

Both carry HR truth, so both keep the SR-vs-HR PSNR recovery metric and the
Zoobot toward-HR shift. Because every stamp (A/B/C, syn-lens, syn-gal) is now a
centered single object, Zoobot inputs are finally comparable across all groups.

## Key finding that shapes the design

Source positions and types **exist at generation but are discarded**.
`MultiBandSimulator.simulate_field()`
([euclid_polish/sky/multiband_generator.py](../../../euclid_polish/sky/multiband_generator.py))
returns a `meta` dict whose `galaxies` / `lenses` lists carry per-source
`x_pix, y_pix, type, render`, redshift, per-band flux, TNG `subhalo_id`, and
(for lenses) `theta_E_arcsec, z_lens, z_source`. But the TFRecord schema in
[euclid_polish/sky/types.py](../../../euclid_polish/sky/types.py) serializes only
pixels + shape + band names + `is_clean`, and
[scripts/run_pipeline.py](../../../scripts/run_pipeline.py) captures `meta` as `_`
and drops it.

The generation assets (COSMOS2025 catalog, Euclid PSF FITS, TNG SKIRT stamps) are
**not present on the Mac** — they live on FASRC — so we cannot re-simulate fields
locally to recover positions. Therefore metadata must be **persisted at creation
on FASRC** and pulled down. A sidecar file is preferred over embedding in the
TFRecord (no `schema_version` bump, reader/writer untouched).

### Field geometry (verified)

- HR / clean field: **256 px @ 0.05″/pix** (12.8″ side, 0.0455 arcmin²).
- LR / dirty field: **128 px @ 0.10″/pix** (exact 2× downsample; same origin).
- Source `x_pix, y_pix` are HR-grid coordinates in [0, 255].
- Lens density **16.5/arcmin² → ~0.75 lenses per field** ⇒ supply for `syn-lens`
  is ample (validation set is ~400 MB, hundreds of fields). No dedicated
  lens-forced stream is needed.

## Decisions (from brainstorming)

- **M (stamp size):** editable in the WebUI, **default 64 HR px** (≈3.2″).
  Enforced even (the LR half-crop needs M/2 integer).
- **One most-central source per field**, accumulated across fields. Each field
  contributes **at most one stamp total** (a lens-bearing field used for
  `syn-lens` is not reused for `syn-gal`) → no shared-field background coupling.
- The full-field `synthetic` group is **replaced entirely** by the two subgroups.

## Components

### A. Source-catalog persistence — `euclid_polish/sky/source_catalog.py` (new) + `run_pipeline.py`

A small, well-bounded module owning the sidecar schema.

```python
SOURCE_COLS = ["field_index", "type", "render", "x_pix", "y_pix",
               "flux_vis_e", "z", "subhalo_id", "theta_E_arcsec"]

class SourceCatalogWriter:
    """Append per-source rows to <records_dir>/sources_<subset>.csv as fields
    are generated. One row per galaxy and per lens; stars are not recorded."""
    def __init__(self, path: str): ...
    def add_field(self, field_index: int, meta: dict) -> None:
        # iterate meta["galaxies"] (type="galaxy", render=sersic|tng,
        #   flux_vis_e=flux_e_per_band[0], z=z_phot or z, subhalo_id if tng)
        # and meta["lenses"] (type="lens", theta_E_arcsec set, z=z_lens)
    def close(self) -> None: ...
```

Per-type field mapping:

| source   | type     | render        | flux_vis_e          | z              | subhalo_id   | theta_E_arcsec |
|----------|----------|---------------|---------------------|----------------|--------------|----------------|
| sersic   | galaxy   | sersic        | flux_e_per_band[0]  | z_phot         | ""           | ""             |
| tng      | galaxy   | tng           | flux_e_per_band[0]  | z (NaN if off) | subhalo_id   | ""             |
| lens     | lens     | ""            | ""                  | z_lens         | lens_subhalo_id | theta_E_arcsec |

`run_pipeline.py` change: in the per-subset write loop, stop discarding `meta` —

```python
writer = SourceCatalogWriter(os.path.join(out_dir, f"sources_{subset}.csv"))
for i in tqdm(range(n), ...):
    sky, meta = sim.simulate_field(rng)
    sky.index = i; sky.subset = subset
    w.write(sky, index=i)
    writer.add_field(i, meta)
writer.close()
```

### B. Source-catalog reader — same module (pure, shared)

```python
def read_sources(csv_path: str) -> dict[int, list[dict]]:
    """field_index -> list of source dicts (typed: floats parsed, '' -> None).
    Missing file -> {} (caller degrades gracefully)."""
```

### C. Source-centered synthetic eval — rewrite `euclid_polish/eval/synthetic_runner.py`

New signature returns rows for both subgroups:

```python
def run_synthetic_eval(out_dir, n, *, model=None, records_dir=None,
                       checkpoint=None, num_res_blocks=None, asinh_scale=None,
                       stamp_m=64, seed=0, on_progress=None, log=None) -> dict:
    """Crop N syn-lens + N syn-gal source-centered M×M stamps from validation
    fields. Returns {"rows": [...], "n_ok", "n_skip", "groups": {...}}.
    Requires the sidecar source catalog; if absent, returns no rows and logs a
    clear message (caller runs A/B/C only)."""
```

Algorithm:

1. Resolve `records_dir` (existing `default_records_dir()`); resolve
   `sources_validate.csv` beside it. If the CSV is missing → log
   *"source catalog not found — regenerate the validation set with metadata;
   skipping syn-lens/syn-gal"*, return `{"rows": [], ...}`.
2. From `read_sources`, build per-field candidates: the most-central source of
   each wanted type whose M×M box fits (`M/2 ≤ x ≤ 256-M/2`, same for y),
   "most-central" = min Euclidean distance to (128, 128).
3. Seeded field order. Assign lens-bearing fields to `syn-lens` until N reached;
   assign *other* fields' central galaxies to `syn-gal` until N reached.
4. Per chosen (field, source): read `lr_cube` (128×4) + `hr_vis` (256) from the
   cached records, `reconstruct(model, lr_cube)` → `sr` (256×4) once, then crop:
   - HR & SR: `[y-M/2:y+M/2, x-M/2:x+M/2]` (HR coords).
   - LR cube: `[y//2-M//4:y//2+M//4, x//2-M//4:x//2+M//4]` (LR coords).
5. Write `original_stack.fits` (cropped LR cube), `SR.fits` (cropped SR cube),
   `HR.fits` (cropped HR VIS); compute `flux_ratio_sr_over_lr`, `psnr_lr_hr`
   (LR-stamp nearest-2×-up vs HR stamp), `psnr_sr_hr` on the **stamp**. Row
   `id = f"{grade}_{field_index:04d}"`, `grade ∈ {syn-lens, syn-gal}`,
   `out_subdir = id`. Reuse the existing `_psnr` and FITS-writing helpers.
6. One bad field must not kill the run (existing try/except per object).

### D. Wiring — `grouped_runner.py`, `zoobot_morph.py`, route, template

- **`zoobot_morph.py`:** `GROUP_COLORS` drops `synthetic`, adds
  `"syn-lens": "#b03a3a"` (crimson) and `"syn-gal": "#7a4fb0"` (violet).
- **`render_transformation_summary`:** the PSNR-recovery panel filters on
  *has HR* (rows with non-empty `psnr_sr_hr`) instead of `grade=="synthetic"`,
  coloured per subgroup; per-group flux + example strips already iterate groups.
- **`render_morphology_summary`:** the PCA HR-star logic already keys off
  `has_hr` / `closer_to_ref` (group-agnostic) — no change beyond the new colours.
- **`grouped_runner.py`:** thread `stamp_m` into `run_synthetic_eval`; the
  returned subgroup rows already carry `grade`, so the single-manifest assembly is
  unchanged. `groups` summary now reports `syn-lens` / `syn-gal`.
- **Route `/api/evaluation/run-grouped`:** read `stamp_m` (default 64, even,
  bounded) from the form, pass through.
- **Template `evaluation.html`:** add an editable *"Synthetic stamp M (HR px)"*
  number input (default 64) to the grouped-analysis form.

## Data flow

```
FASRC generation (run_pipeline.py)
  simulate_field → (sky, meta)
     ├─ TFRecord writer → {clean,hr,dirty}_validate.tfrecord
     └─ SourceCatalogWriter → sources_validate.csv      ← NEW
  rsync ↓ to Mac records cache (rides existing sync)

Mac eval (run_grouped_analysis)
  A/B/C  → catalog_runner (real cutouts, network)
  syn-*  → synthetic_runner
             read_sources(sources_validate.csv)
             pick most-central fitting lens / galaxy per field
             reconstruct full field → crop M×M stamps → FITS + metrics
  → one manifest.csv (grade ∈ {A,B,C,syn-lens,syn-gal})
  Zoobot (subprocess) → morphology_manifest.csv + zoobot_predictions.csv
  plots: transformation_summary.png + morphology_summary.png (PCA + HR stars)
```

## Error handling / degradation

- **No sidecar** (before FASRC regen): synthetic subgroups skipped with a clear
  log line; A/B/C run normally; manifest + transformation/morphology plots render
  from whatever groups are present.
- **Supply shortfall** (fewer than N fitting lenses/galaxies in the window):
  take what's available and log the shortfall; never abort.
- **Bad field / crop out of range:** per-object try/except records the error in
  the manifest row and continues.

## Testing

- `source_catalog`: writer→reader round-trip from a synthetic `meta` dict
  (galaxies + lenses + a NaN-z tng), typed parsing, missing-file → `{}`. Pure.
- `synthetic_runner` selection/crop: fabricate LR/HR planes + a source list;
  assert the most-central fitting source is chosen, the crop region and stamp
  shape are correct, edge sources (box overflows) are rejected, and stamp PSNR /
  flux-ratio are computed. Uses a stub model (identity-ish) — no real checkpoint.
- Degradation: `run_synthetic_eval` with no sidecar returns `{"rows": []}` and
  logs; `run_grouped_analysis` still writes an A/B/C-only manifest.
- WebUI: `/api/evaluation/run-grouped` accepts `stamp_m`; template contains the
  M input. Mirror existing route tests.

## Rollout dependency

`syn-lens` / `syn-gal` go live only **after the FASRC validation set is
regenerated** (to write `sources_validate.csv`) and synced down. A/B/C are
unaffected and work immediately. This regen rides along with the TFRecord regens
already pending (4-band, in-flight detector params, PSF background cleaning).

## Out of scope (YAGNI)

- Recording stars in the sidecar (not centered on for morphology).
- Per-galaxy source detection (photutils/SEP) on the fields — the generator
  already knows exact positions; detection would only add noise.
- A dedicated lens-forced validation stream — unnecessary given ~1 lens/field.
- Embedding metadata in the TFRecord schema — sidecar is sufficient.
