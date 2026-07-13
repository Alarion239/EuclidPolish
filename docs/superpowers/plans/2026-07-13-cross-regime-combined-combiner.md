# Cross-Regime Combined Combiner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two experimental cross-regime RBF combiners that share all-member predictions but fit and score the starfull and starless targets independently.

**Architecture:** Preserve `eval.combiner` mathematics and ordinary artifact paths. Add a shared `ensemble/combined/cubes[_validate]` prediction-cache lifecycle in `ensemble_viz`, persist each target model beneath its existing regime root, and pass an optional second reconstruction through existing evaluation and spectrum consumers. Both web surfaces expose the additive job and final card.

**Tech Stack:** Python/NumPy/TensorFlow, Flask jobs/routes, React/TypeScript, classic JavaScript, pytest, Ruff.

## Global Constraints

- Reuse `fit_combiner` and `FitBufferAccumulator`; do not add a different combiner model.
- Keep ordinary `combiner/`, viewer tier, and diagnostic point estimate unchanged.
- Require at least one active, checkpoint-present member from both regimes.
- Cache only all-member predictions under `ensemble/combined`; target artifacts stay per regime.
- Persist `combined_comb_<record>.npy`, never overwrite `comb_<record>.npy`.
- Omit only the combined curve/metric when its model or output is stale.

---

### Task 1: Generalize combiner persistence without changing ordinary paths

**Files:**
- Modify: `euclid_polish/eval/combiner.py`
- Test: `tests/test_combiner.py`

**Interfaces:**
- Produces: `save_combiner(comb, base_dir, artifact_dir=None)` and `load_combiner(base_dir, member_labels=None, artifact_dir=None)`.
- Compatibility: omitted `artifact_dir` resolves exactly to `<base_dir>/combiner`.

- [ ] Write a failing round-trip test that saves a second combiner to `combined_combiner`, then proves the ordinary combiner remains loadable from `combiner`.
- [ ] Run `pytest tests/test_combiner.py -q` and confirm the new test fails because the persistence API lacks `artifact_dir`.
- [ ] Implement `_combiner_dir(base_dir, artifact_dir=None)` with the default `"combiner"`; route both persistence functions through it without changing serialized arrays or JSON keys.
- [ ] Run `pytest tests/test_combiner.py -q` and confirm it passes.

### Task 2: Add the shared all-member cache and combined-model lifecycle

**Files:**
- Modify: `euclid_polish/web/helpers/ensemble_viz.py`
- Test: `tests/test_combined_combiner.py`

**Interfaces:**
- Produces: `job_combined_combiner_fit(cap, *, num_images, n_kernels, min_usage, starless)`.
- Produces: `compute_combined_combiner_payload(starless)` and `_apply_combined_combiner_to_test_cubes(starless)`.
- Consumes: canonical active registry order from `EnsembleModel(..., starless=None)` and `hr_*` or `clean_*` targets selected by `starless`.

- [ ] Write failing tests with two fake active members (one from each regime) that assert: both target fits receive the same `(N, M)` cached prediction matrices; their target buffers differ; and a second fit performs no member inference.
- [ ] Run `pytest tests/test_combined_combiner.py -q` and confirm failures name the missing combined lifecycle API.
- [ ] Implement a versioned, atomic-manifest cache at `ensemble/combined/cubes_validate` and `ensemble/combined/cubes`, keyed by subset, record indices, labels, checkpoint fingerprints, dirty fingerprint, HR shape, and bands; reuse prefixes and infer only missing records on identity match.
- [ ] Implement target-specific validation pairing, fit metadata (target kind, source regime tags, cache/target fingerprints), artifact persistence beneath `<regime>/combined_combiner`, and `combined_comb_<record>.npy` test application.
- [ ] Run `pytest tests/test_combined_combiner.py -q` and confirm it passes.

### Task 3: Integrate staleness, archive reconciliation, evaluation metrics, and spectra

**Files:**
- Modify: `euclid_polish/web/helpers/ensemble_viz.py`
- Modify: `euclid_polish/eval/power_spectrum.py`
- Test: `tests/test_combined_combiner.py`
- Test: `tests/test_power_spectrum.py`

**Interfaces:**
- Extends: `EnsembleSpectrumAccumulator.add(..., combined_combiner=None)`.
- Produces: `P_combined`, `r_combined`, `T_combined` alongside the unchanged ordinary `P_comb`, `r_comb`, `T_comb` keys.

- [ ] Write failing tests that a perfect optional combined reconstruction emits its own power/coherence series, that `T_combined=sqrt(P_combined/P_hr)`, and that a combined output does not replace `comb_` or diagnostics’ ordinary combiner input.
- [ ] Run the focused spectrum and combined tests; confirm failures are for absent optional combined fields.
- [ ] Add independent combined accumulators/plot transforms and fold current combined outputs into cached evaluation-payload reconstruction and static spectrum regeneration.
- [ ] Add model/output staleness checks and archive reconciliation: reindex an all-band-pruned column exactly; invalidate only combined targets that use an archived member; preserve existing ordinary reconciliation.
- [ ] Run `pytest tests/test_combined_combiner.py tests/test_power_spectrum.py -q` and confirm it passes.

### Task 4: Add additive routes and payload contract

**Files:**
- Modify: `euclid_polish/web/routes/ensemble.py`
- Modify: `tests/test_combiner_web.py`

**Interfaces:**
- Produces: `POST /ensemble/combined-combiner/fit` and `GET /ensemble/combined-combiner.json?mode=starfull|starless`.
- Contract: payload includes ordered labels, `starless` source tags, survivor masks, gate curves, stale/disabled reason, and combined test metric comparisons.

- [ ] Write failing route tests for each target mode, including a one-regime member-pool request that returns an honest unavailable payload/reason.
- [ ] Run `pytest tests/test_combiner_web.py -q` and confirm failures point to missing combined endpoints.
- [ ] Spawn only the selected target job; validate numeric controls as the ordinary route does; serve an unavailable/stale JSON payload rather than silently falling back.
- [ ] Run `pytest tests/test_combiner_web.py -q` and confirm it passes.

### Task 5: Render the final combined-combiner card and toggleable spectrum curve

**Files:**
- Modify: `euclid_polish/web/frontend/src/pages/Ensemble.tsx`
- Modify: `euclid_polish/web/static/ensemble_combiner.js`
- Modify: `euclid_polish/web/static/ensemble_evals.js`
- Modify: `euclid_polish/web/templates/ensemble.html`
- Test: `tests/test_combiner_web.py`

**Interfaces:**
- Consumes: `/ensemble/combined-combiner.json`, `evals.ps.r_combined`, and `evals.combined_combiner`.
- Produces: a final card after disagreement on both pages and a distinct independently toggleable `combined combiner` spectrum series.

- [ ] Write failing web/payload assertions that both pages retain ordinary card/viewer behavior while rendering the final combined card with disabled reasons and source-regime metadata.
- [ ] Run the focused web tests and confirm the new assertions fail before UI implementation.
- [ ] Reuse ordinary fit controls/progress and gate plots; add `by star regime` colouring, source labels, and the combined spectrum legend/chip/curve without altering ordinary curve keys or defaults.
- [ ] Run `pytest tests/test_combiner_web.py -q` and the frontend build; confirm both pass.

### Task 6: Verify the complete feature

**Files:**
- Verify only.

- [ ] Run focused Python tests: `pytest tests/test_combiner.py tests/test_combined_combiner.py tests/test_combiner_web.py tests/test_power_spectrum.py -q`.
- [ ] Run `ruff check euclid_polish tests` and `python -m compileall -q euclid_polish`.
- [ ] Run the frontend production build from `euclid_polish/web/frontend`.
- [ ] Run `git diff --check` and inspect the final diff against every global constraint.
