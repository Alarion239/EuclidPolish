# Image / ImageSet — clean data atom + collection

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development for Phases 2–3. Steps use `- [ ]`.

**Goal:** Two well-defined classes — `Image` (one multi-band sky stamp) and `ImageSet` (a collection persisted as TFRecords) — sitting at the bottom of the import graph. They own ONLY self-contained operations (serialization, plotting, crop/rebin, metrics). Every transform that needs an operator (PSF, forward model, trained model, archive) lives on the operator, never on the image.

**Architecture:** `MultiBandSkyImage` is renamed/reshaped into `Image` and moved to a dedicated `euclid_polish/image/` package; the orchestrated-physics methods (`convolved_with`, `convolved_per_band`, `with_band_noise`) — which have **zero external callers** — are dropped. `Image` carries an optional provenance `Stamp` and a `role` tag (collapsing the old HR/LR/SR/Euclid subclasses into one class + enum). `ImageSet` owns the TFRecord stack I/O (moved out of `sky/tfrecord.py`). Back-compat is preserved by leaving `sky/types.py` and `sky/tfrecord.py` as re-export shims, so the ~50 existing call sites keep working untouched.

**Env:** worktree `/Users/alarion239/Desktop/EuclidPolish/.claude/worktrees/provenance`, branch `feature/provenance-uuid`. Tests: `~/miniforge3/envs/EuclidPolishEnv/bin/python -m pytest <path> -q`. HARD RULE: imports at module top.

---

## Import layering (the invariant this enforces)

```
provenance value-types (ProvId, Stamp — pure)
        ↑
   euclid_polish/image/  (Image, ImageSet, Role)  ← IO + plotting + crop/rebin + metrics
        ↑
   operators (Simulator, Forward, Model, Archive, PSFSet)  ← own ALL transforms
        ↑
   CLI / WebUI / eval / scripts
```

`image/` imports only third-party (numpy, tf, astropy, matplotlib) + `config` + `provenance`. It never imports an operator. Enforced by `tests/test_image_is_leaf.py`.

---

## Phase 1 — the clean foundation (THIS PHASE, controller-implemented)

New package `euclid_polish/image/`:
- `core.py` — `Image` (`@dataclass(StampCarrier)`) + `Role(str, Enum)`. Fields: `data, pixel_scale_arcsec, band_names, is_clean`, kw-only `role=Role.UNKNOWN`, `index, subset, metadata`, inherited kw-only `stamp`. Keeps: shape/num_channels, band accessors, `crop_array`/`centre_cropped_to`, `rebin_array`/`sum_rebinned`, measurements, `to_tfrecord`/`from_tfrecord` (now also round-trips `role`). Adds: `save_fits`/`from_fits` (PIXSCALE/BANDS/ROLE/IS_CLEAN + provenance cards), `plot`, `psnr_against`. Drops the dead physics methods.
- `tfio.py` — TFRecord stack functions moved from `sky/tfrecord.py` (`tfrecord_path`, `parse_record_graph_v2`, `write_multiband_skyimages`, `open_multiband_writer`, `read_multiband_skyimages`), referencing `Image`.
- `collection.py` — `ImageSet`: `from_images`, `write(records_dir, name)`, `read(path_or_glob)`/`__iter__`/`__len__`/`__getitem__`, `by_role`, `split(frac, rng)`, optional set-level `stamp`.
- `__init__.py` — exports `Image`, `ImageSet`, `Role`.

Shims (back-compat, no behaviour change):
- `sky/types.py` → `from euclid_polish.image.core import Image, Role; MultiBandSkyImage = Image`.
- `sky/tfrecord.py` → re-export the moved functions from `euclid_polish.image.tfio`.

Tests: `tests/test_image.py`, `tests/test_imageset.py`, `tests/test_image_is_leaf.py`. Whole suite stays green.

## Phase 2 — plotting engine relocation (subagents)

Move `plot_reconstruction` (and the asinh/RGB helpers it needs that aren't already leaves) into `image/plotting.py` so `ImageSet.plot_reconstruction(lr, sr, hr)` is self-contained; `training/inference.py` re-exports for back-compat; migrate the 4 callers (`cli/main.py`, `web/routes/evaluation.py`, `web/helpers/jobs_impl.py`, `cutout/base.py`).

## Phase 3 — operators own the verbs; remove the Cutout layer (subagents)

- `MultiBandSimulator.generate(rng) -> Image` (wraps `simulate_field`, stamps output).
- `MultiBandForward.apply(hr) -> Image` (today's `process`, stamps output parented on input + run).
- `EuclidArchive.fetch(ra, dec, size) -> Image` (absorbs `EuclidLRCutout._fetch_planes_from_archive`).
- `Model.upsample` already an operator verb — keep; reparent on `(model.id, lr.id)`.
- Rewire `cli/inference_ops.py`, the CLI menus, and the eval runners to operator calls; delete `euclid_polish/cutout/`.

---

## Verification

- Phase 1: `pytest tests/test_image.py tests/test_imageset.py tests/test_image_is_leaf.py tests/test_types_and_tfrecord.py tests/test_data_multiband.py -q` green, then full suite green.
- Each phase auto-commits on the branch and pushes.
