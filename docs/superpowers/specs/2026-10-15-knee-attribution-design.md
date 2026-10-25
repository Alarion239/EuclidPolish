# Knee attribution: which asinh knees a multi-knee model relies on, for which pixels

**Date:** 2026-10-15 · **Status:** approved in conversation (user: "build it", "work everywhere", "any multi-knee model", "a plot over all fields for what kind of pixels the knees are important")

## Goal

For any multi-knee model and any fields, measure how much each input knee contributes to each predicted pixel. Summarise that over all the fields by pixel kind (star core, galaxy outskirts, sky, saturated…) and by pixel brightness. Real tiles and synthetic test fields both work. Everything runs locally, with no FASRC use.

## Models

Any member whose input is stretched at several knees (`asinh_knees` in its origin):
- universal models (learned output knee: the network outputs electrons);
- single-output members with a fixed `output_knee` (the output is `sinh(clip(y, ±20))·k` electrons);
- multi-head members, through their served head (`inference.default_head_knee`).

Single-knee members are refused, with a reason.

## Fields

- **Synthetic:** the universal scores' test fields (`identity.indices` of the universal scores index), read from the local records: `dirty_test` (LR), `hr_test` (the starfull truth) and `clean_test` (the starless truth).
- **Real:** any `real_tiles` ref (`source/id`: poster, nexus, tile, archive, eval, pair…) through `real_tiles.get_tile(...).cube` (LR electrons, 4 bands).

## Per-pixel attribution (`euclid_polish/eval/knee_attribution.py`, pure numerics + TF)

For a sampled output pixel `p` (HR position, output band `b`) the network input is the stretched stack `s_{k,c} = asinh(x_c / q_k)` (knee-major, channel `k·C + c`). One backward pass gives `g = ∂y_b(p)/∂s`.

**Chain-rule split (exact).** The sensitivity to the LR electrons splits into knee terms:

`∂y/∂x_c(u) = Σ_k t_{k,c}(u)`, with `t_{k,c}(u) = g_{k,c}(u) / √(q_k² + x_c(u)²)`.

**Noise weighting (record version 2).**
- A term is per input electron, and the bands' electrons differ in scale. On the test fields the VIS sky noise is about 18 e⁻ and NISP's about 0.8 e⁻.
- Summed per electron, every output band looked about 58 % Y-driven, and the knee shares followed NISP's scale. That was a units artifact.
- So each input band's terms are weighted by its noise σ_c. `σ_c·|t_{k,c}|` is the change of `y` for a 1σ change of band `c` through knee `k`, comparable across bands.
- σ_c is the field's own robust spread of neighbouring-pixel differences, `1.4826·MAD(Δ)/√2` over both axes. Pairs touching a 0 (a blanked hole) or a non-finite pixel are left out. A band without a usable σ takes the median of the others.
- Version-1 records (per-electron weights) were discarded.

Reported per sample, with `a_{k,c} = σ_c Σ_u |t_{k,c}(u)|`:
- `knee_share[k] = Σ_c a_{k,c} / Σ a`
- `band_share[c] = Σ_k a_{k,c} / Σ a` (input band)
- `knee_share_by_input[c][k] = a_{k,c} / Σ_k a_{k,c}`: each input band's own knee split. It does not depend on σ, and `knee_share = Σ_c band_share_c · knee_share_by_input_c`. It is only in the aggregates; per sample it would multiply the record's size.
- `sensitivity = Σ a` (output e⁻ per input σ)
- **Local splits (record version 3):** `knee_share_local`, `band_share_local` and `knee_share_by_input_local`.
  - These are the same splits with `Σ_u` restricted to the 3×3 LR pixels under the sample (`LOCAL_RADIUS = 1`).
  - The receptive-field sums are mostly sky around any pixel. So they say what the prediction depends on overall, and barely change with the pixel's class or brightness.
  - The local ones say how the pixel's own neighbourhood is read.
  - Per sample, the record keeps `knee_share_local`; the rest is only in the aggregates.
- `ablation_share[k]`: replace knee `k`'s input channels with their window mean; `Δ_k = |y − y_k|`; `ablation_share = Δ / ΣΔ` (0s when ΣΔ = 0).

**Windows.**
- Each sample is computed on an LR window centred on its LR pixel. The side is `max(64, 2·(depth + 4))`, rounded up to an even number, which covers the receptive field: `depth` 3×3 blocks plus the entry and tail 3×3 plus the 5×5 skip.
- Field borders are zero-padded, which is what the full-field network sees with `same` padding.
- So a window's prediction and gradient at the sample equal the full field's.
- Windows are batched (default 64 per pass).

## Pixel classes

Per HR pixel and band, a class name and a brightness value (electrons) used for the brightness curves.

**Every field: the predicted-image bins.** `sky` < 2, `faint` 2–10, `mid` 10–52, `bright` 52–363, `core` > 363 e⁻ of the model's own SR in that band. These are the combiner diagnostic's edges, `_GATE_BRIGHTNESS_EDGES` in asinh(x/100) units.

**Synthetic fields: truth classes.**
- **Truth maps:**
  - galaxy flux = `clean_test`, blurred to the target PSF;
  - star flux = (`hr_test − clean_test`), blurred;
  - truth = galaxy + star.
- **Priority** (first match wins):
  1. `saturated`: the LR pixel under it is exactly 0 in that band while truth > 52 e⁻.
  2. `star core`: star flux > ½ truth and truth > 52.
  3. `star wing`: star flux > ½ truth and 2 < truth ≤ 52.
  4. `galaxy core`: truth > 52.
  5. `galaxy outskirts`: 2 < truth ≤ 52.
  6. `faint` (sub-noise): 0.01 < truth ≤ 2.
  7. `sky`: otherwise.
- The brightness value is the truth.

**Real fields:** the predicted-image bins, plus `blank` where the LR pixel is 0 or non-finite in that band. The brightness value is the SR.

**Sampling.** Per field and band, up to `per_class` (default 24) random pixels of each class present (seeded). Pixels within half a window of nothing special are fine, since the zero padding handles the borders.

## Aggregates (over all fields of a run, per model and output band)

- `by_class[band][class]`: mean `knee_share`, mean `ablation_share`, mean `band_share`, mean `knee_share_by_input` (C × K), the three `*_local` means, `n`.
- `by_brightness[band]`: half-decade log bins from 1e-3 to 1e7 e⁻ (bin edges stored), mean `knee_share`, `knee_share_by_input`, `knee_share_local` and `knee_share_by_input_local` per bin, and `n` per bin (bins with n = 0 are null).
- `cross_band` (and `cross_band_local`): a 4×4 matrix of mean `band_share`, rows = output band, cols = input band.

## Job and store

- **Module:** `euclid_polish/web/helpers/knee_attribution_runs.py`.
- **Job:** `job_knee_attribution(cap, *, models: list[str], synthetic: int, tiles: list[str], per_class: int, seed: int)`, registered as a local job, kind `knee-attribution`.
  - `models` are catalogue specs, `universal:member_NNN` or `member:member_NNN`.
  - `synthetic` is the number of test fields (0–100, the first N of the scoring indices).
  - `tiles` are real refs.
- **Record:** written atomically to `data/euclid_inference/knee_attribution/<id>.json`, id `YYYYmmdd-HHMMSS-xxxxxx`:

```json
{"version": 2, "id": "...", "created": "...", "finished": "...", "status": "done|failed|running|cancelled",
 "params": {"models": [...], "synthetic": 20, "tiles": [...], "per_class": 24, "seed": 0},
 "knee_grid_note": "...",
 "models": {"universal:member_222": {
     "knees": [0.001, ...], "bands": ["VIS","Y_E","J_E","H_E"], "depth": 16,
     "aggregate": {"synthetic": AGG, "real": AGG},
     "fields": [{"ref": "test/17" | "poster/target_181255", "kind": "synthetic|real",
                 "shape": [H2, W2], "samples": {"band": [...], "y": [...], "x": [...],
                 "class": [...], "value_e": [...], "knee_share": [[...K]...],
                 "ablation_share": [[...]...], "band_share": [[...4]...], "sensitivity": [...]},
                 "noise_sigma_e": [σ_VIS, σ_Y, σ_J, σ_H]}]
 }},
 "errors": {"<model>": "reason", "<model>|<field>": "reason"}}
```

  where `AGG = {"by_class": {...}, "by_brightness": {"edges_e": [...], "bands": {...}}, "cross_band": [[...]], "n_samples": N}`. Synthetic and real samples are aggregated separately, since their class vocabularies differ.

- **Field images for the map view:** `fields/<run-id>/<model-slug>/<ref-slug>.npz` holds the SR (float16, HR) and the LR (float16); the record links to them.

## API (`/api/knee-attribution`, JSON errors like `/api/experiments`)

- `GET /api/knee-attribution`: `{runs: [{id, created, finished, status, models, synthetic, tiles, per_class}]}`, newest first.
- `GET /api/knee-attribution/models`: `{models: [{spec, label, knees, depth, kind, available, reason}]}`, the multi-knee catalogue models.
- `GET /api/knee-attribution/<id>`: the record.
- `GET /api/knee-attribution/<id>/field/<model-slug>/<ref-slug>.png?band=VIS&layer=sr|lr`: the field image as asinh-scaled PNG.
- `POST /api/knee-attribution` (form `models`, `synthetic`, `tiles`, `per_class`, `seed`): `{ok, job_id, run_id}`. Validation errors return 400. Nothing starts on page load.

## Console view: Models › Knees (`/models/knees`)

1. **Run panel.** Pick models (multi-knee only; the newest universal models preselected), the number of synthetic test fields (default 20, up to 100), real tiles (a checkbox list from `GET /api/real/sources`), samples per class (24) and seed. A confirm then starts the job, with progress through the standard job hooks.
2. **"Which knees matter for which pixels"**, aggregated over all the run's fields: the user's main request.
   - **Knee share vs pixel brightness:** log x-axis, one line per knee, a band selector; a model selector, or models overlaid as small multiples.
   - **Class × knee stacked bars:** synthetic truth classes, and real brightness classes, in separate panels.
   - **Gradient vs ablation:** a per-class agreement table.
3. **Cross-band matrix:** a 4×4 heat table.
4. **Field explorer:** pick a field to show its SR (or LR) image with the sampled pixels as dots coloured by their dominant knee, the class shown on hover.

## Tests

- **Core:**
  - The chain-rule split sums to the direct ∂y/∂x computed through the stretch, on a tiny random WDSR.
  - A window gradient equals the full-field gradient for an interior pixel.
  - Shares sum to 1.
  - Ablation shares behave as defined.
  - Classes follow the stated priorities on hand-built arrays.
  - Aggregation means and bins are right.
- **Runs/API:** job and record round-trip with a tiny model, validation errors, listing, refusing a single-knee model, no job on GET.
- **Frontend:** model helpers, rendering the charts from a fixture, the run form posting the right fields.

## Pixel types (2026-10-25, record version 4)

**Request (user):** "approximate averaged types of pixels which correspond to different prevalence of the input knee"; effective knee + brightness-controlled types + stacked stamps, uniform sampling; run on the best model (249: 1σ loss floor + isotropy 2–64 px), then compare a second model.

**Record version 4.**
- `sampling`: `class` (as before), `uniform` (`per_band` random pixels per band) or `mixed` (both pooled, weights halved).
- Per sample: `weight` (the field pixels it stands for, so weighted sums count pixels without bias under any design), and `knee_share_own` / `knee_share_own_local` (the split of the sample's own input band, the one stable across retrains).
- Synthetic field images also keep the `truth`.

**Numerics:** `euclid_polish/eval/knee_pixel_types.py`.
- Effective knee: `log10 q̄ = Σ s_k log10 q_k`.
- Chain-rule baseline (`mechanical_share`): the share a network with the same gradient on every knee would get from the LR 3×3 block, `∝ Σ_u 1/√(q_k² + x_u²)`. The effective knee minus the baseline's is the network's own preference.
- Brightness-controlled groups: within each decade of brightness in σ, the bottom 20 % / middle / top 20 % by effective knee (or by the preference).
- Knee types: weighted k-means of the centred log-ratio of the shares, every brightness decade weighted alike; prevalences use the true weights.
- Stamps: aligned by a dihedral move (the scene's brighter side up), median-stacked, with an occupancy map (fraction > 1σ) and measures (nearest source, brightest pixel nearby, local maximum, slope).

**Script:** `scripts/knee_pixel_types.py` runs the attribution (no ablation) and writes the figures and `summary.json` to `<root>/pixel_types/<run>/<model>/`; `--compare A,B` pairs the same pixels across two runs of the same seed.
- Units are the band's LR noise σ.
- The real-tile scene and brightness are the LR smoothed 3×3, not the SR, because a model's holes in bright cores would pass for faint pixels.
- Samples whose LR block holds a blank (0) pixel are left out, since there `1/q` puts the share on the lowest knee whatever the network does.
