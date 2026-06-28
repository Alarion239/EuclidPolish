# Object-oriented pipeline API for EuclidPolish

## Context

The generate → forward-model → super-resolve → evaluate pipeline works, but the code that *invokes* it from the CLI, the WebUI, the eval runners, and the FASRC scripts is procedural glue: `cli/main.py` is ~1,750 lines whose generate/convolve/reconstruct menus are 80–110 lines each; `jobs_impl.reconstruct_cutout_at` is ~170 lines; `catalog_runner`/`grouped_runner` orchestrate by hand. The *engines* are fine — `MultiBandSimulator.simulate_field`, `MultiBandForward.process`, `training.inference.reconstruct` — but every caller re-implements load → loop → save and threads provenance by hand.

This change gives those engines a clean, documented, object-oriented face so the same work reads as a few lines, with provenance threaded automatically. Operators (simulator, forward model, trained model) are built once and carry their config/identity; typed cutouts flow through them.

The provenance/identity system (branch `feature/provenance-uuid`, on which this builds) already provides the `Cutout` hierarchy, `ProvId`/`ProvStore`/`Stamp`, and checkpoint identity — this work completes the encapsulation those classes started.

## Goals / non-goals

- **Goal:** a small set of well-documented classes where `Model`, the `Cutout` hierarchy, and the existing operator classes own the verbs, and CLI/WebUI/eval/scripts call a 3–6 line surface.
- **Goal:** provenance is automatic — the neat API never threads a `ProvStore`.
- **Non-goal:** rewriting the engines. `MultiBandSimulator`/`MultiBandForward`/`reconstruct` stay as the implementation; the classes wrap them.
- **Non-goal:** changing the physics, the TFRecord schema, or any on-disk format.

## The target surface

```python
sim     = MultiBandSimulator(catalog, gen_config)     # operator, built once
forward = MultiBandForward(psf_sets, fwd_config)       # operator, built once
model   = Model(checkpoint_dir)                        # operator, carries .id

hr = SyntheticHRCutout.generate(sim, rng=rng)          # generate
lr = hr.downsample(forward, rng=rng)                   # HR -> LR
sr = model.upsample(lr)                                # LR -> SR
sr.save_fits(out / "SR.fits")
sr.save_png(out / "eye.png", regime="eye")

# real Euclid path
sr = model.upsample(EuclidLRCutout.fetch(ra, dec, size))

# evaluation (SP2)
result = model.eval_catalog(catalog_path, out_dir, grades=["A", "B", "C"])
```

## Architecture

Two kinds of object:

- **Operators** — built once from config, reused across many cutouts: `MultiBandSimulator`, `MultiBandForward`, `Model`. They own the heavy work and (for `Model`) the identity.
- **Cutouts** — typed, immutable data handles that flow through operators and carry a `ProvId` + lineage: the existing `Cutout` tree (`HRCutout`/`LRCutout` → `SyntheticHRCutout`/`SyntheticLRCutout`/`EuclidLRCutout`/`SRCutout`).

Provenance threads automatically: every cutout-producing method defaults `store=None → default_store()` (guarded), mints the new cutout's id, and stamps parents — so callers never see a store.

## SP1 — the OO core (this spec's deliverable)

### `Model` (new — `euclid_polish/model.py`)

The public face of a trained checkpoint.

- `Model(checkpoint_dir, *, scale=2, num_res_blocks=Config.DEFAULT_NUM_RES_BLOCKS)` — loads weights via the existing `load_model_from_checkpoint` (which auto-infers `nchan_in/out`), and reads `model_id_of_checkpoint` into `self.id`.
- `id: ProvId | None` — the model's identity (or `None` for a legacy checkpoint).
- `upsample(lr: LRCutout, *, store=None) -> SRCutout` — wraps `reconstruct`; the SR carries `parents=(self.id, lr.id)` and is auto-stamped. Absorbs today's `LRCutout.super_resolve`.
- `upsample_array(arr: np.ndarray) -> np.ndarray` — the raw-array escape hatch for the few call sites that hold a bare ndarray.
- `eval_catalog(...)` / `eval_grouped(...)` — **declared here, implemented in SP2.**

Removing `super_resolve` from `LRCutout` takes `reconstruct` out of `cutout/base.py` — `Model` is the only importer of the *upsampling* engine. (`save_fits`/`save_png`/`fetch` still pull `astropy`/`plot_reconstruction`/the downloader at `cutout/base.py`'s module top — these are acyclic and expected, since those IO/render verbs now live on the cutout classes. The only **forbidden** import edge is `cutout → model`, enforced by a test.)

### `Cutout` hierarchy changes (`euclid_polish/cutout/`)

- `SyntheticHRCutout.generate(simulator, *, rng=None, store=None) -> SyntheticHRCutout` — classmethod wrapping `simulator.simulate_field(rng)`; symmetric with `downsample`'s forward operator.
- `SyntheticHRCutout.downsample(forward, *, rng=None, store=None) -> SyntheticLRCutout` — renames today's `convolve`; wraps `forward.process`.
- `EuclidLRCutout.fetch(ra, dec, size, *, bands=Config.LR_INPUT_BAND_NAMES, store=None) -> EuclidLRCutout` — absorbs the per-band archive download + ADU/s→e⁻ conversion currently inline in `jobs_impl` (~170 lines), so the real path needs no plumbing. Retains an injectable per-band fetcher for tests (today's `query(fetch_plane=…)` becomes that internal seam).
- `Cutout.save_fits(path, *, wcs_header=None)` — writes the FITS (scaled WCS for SR via `scaled_wcs_header`) with provenance cards, absorbing the writer from `jobs_impl`.
- `SRCutout.save_png(path, *, regime="eye", lr=None, hr=None)` — wraps `plot_reconstruction`.
- `write_tfrecord(records_dir, name)` — batch/stream convenience over the existing `to_tfrecord`.
- **Implicit store** on every cutout-producing method (`store=None → default_store()`), guarded so provenance never blocks the operation.

### `PSFSet.draw_random(rng) -> PSF`

A one-line convenience over the existing `sample_for_generation` (`draw_sample` + `apply_sample`), so the sampling verb reads naturally where a single random PSF is wanted.

## Decomposition (build order)

Each sub-project is its own plan → implementation cycle. Implementation is delegated to Sonnet subagents.

- **SP1 — OO core** (this spec): `Model`, `Cutout` `generate`/`downsample`/`fetch`/`save_fits`/`save_png`, implicit store, `PSFSet.draw_random`. No call-site behaviour changes yet — the existing procedural paths keep working.
- **SP2 — eval on `Model`:** implement `Model.eval_catalog` / `eval_grouped` absorbing `catalog_runner`/`grouped_runner` (incl. the new real-galaxy group + power-spectrum panels).
- **SP3 — CLI migration:** `cli/main.py` generate/convolve/reconstruct/train menus → thin OO calls.
- **SP4 — WebUI migration:** `jobs_impl.reconstruct_cutout_at` / `_job_generate_reconstruct` → `EuclidLRCutout.fetch` + `model.upsample` + `sr.save_*`.
- **SP5 — FASRC scripts:** `run_pipeline.py`, `infer_euclid_cutout.py`, `lensfinder_sr_infer.py`, etc.

## Provenance integration

Automatic and unchanged in mechanism — the new methods reuse the existing stamping: `generate`/`downsample`/`fetch` mint the new cutout's id and parent it correctly; `Model.upsample` parents on `(model.id, lr.id)`; `save_fits` writes the FITS provenance cards; all guarded.

## Testing strategy

- `Model`: load + `id` from a checkpoint with/without `provenance.json`; `upsample(lr)` returns an `SRCutout` with `parents=(model.id, lr.id)` (stub `reconstruct` to avoid a real model); `upsample_array` shape.
- `Cutout`: `generate` with a `TinyCosmosCatalog`-backed simulator; `downsample` (rename — migrate the existing convolve test); `fetch` with an injected fetcher; `save_fits`/`save_png` write readable files with provenance cards; implicit-store path mints ids without an explicit store.
- `PSFSet.draw_random` returns a `PSF`.
- Regression: the existing cutout / psf / inference / pipeline suites stay green; the moved `super_resolve` test becomes a `Model.upsample` test.

## Risks & migration safety

- **Behaviour-preserving by construction:** SP1 adds classes and methods; it does not change the engines or any call site. The procedural paths keep working until SP3–SP5 migrate them.
- **Import layering:** the ONLY forbidden edge is `cutout → model` (would cycle with `model → cutout`); enforced by a test. `cutout → training.inference` (for `save_png`'s `plot_reconstruction`) and `cutout → downloader` (for `fetch`) are acyclic and fine — those engines never import `cutout`. `Model` is the sole importer of `reconstruct`.
- **`save_png` complexity:** `plot_reconstruction` has three modes (real / synthetic-with-HR / colour). `save_png` wraps it without reimplementing; the mode is chosen from what the cutout carries.
- **Guarded provenance:** every implicit-store call is wrapped so a provenance failure degrades to an unstamped-but-correct artifact, never an error.
