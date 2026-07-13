# Cross-Regime Combined Combiner Design

Date: 2026-07-13
Status: approved design

## Goal

Add a second, experimental combiner to each Ensemble regime without changing
the existing combiner or its model architecture.

The ordinary starfull combiner continues to fuse only starfull members and the
ordinary starless combiner continues to fuse only starless members. The new
combined combiners both consume the union of active starfull and starless
member predictions, but learn different targets:

```text
all starfull + starless members -> starfull HR target (`hr_*`)
all starfull + starless members -> starless target (`clean_*`)
```

This experiment tests whether the existing brightness gate discovers a useful
division of labor, such as favoring starless members for faint diffuse
structure and starfull members for stellar cores.

## Governing constraint: preserve the combiner architecture

The combined models use the existing `euclid_polish.eval.combiner` model and
fit path unchanged:

- one independent model per output band;
- the maximum asinh member prediction as the per-pixel brightness scalar;
- fixed RBF kernels over brightness;
- learned per-member logits followed by a softmax;
- a convex member mixture as the output;
- asinh-space L1 fitting on validation records; and
- the existing cumulative whole-member pruning rule.

There is no CNN, spatial feature extractor, neighborhood input, new loss,
non-convex residual, or target-specific change to the RBF model in this work.
Only the set of input members and the target record change.

A future experiment may use a CNN-conditioned RBF gate: a CNN could inspect a
local patch and condition the RBF mixture weights while the output remains a
convex member mixture. That is explicitly out of scope here.

## Existing behavior remains intact

The current regime-specific paths remain the production comparison baseline:

```text
starfull members -> ordinary starfull combiner -> hr target
starless members -> ordinary starless combiner -> clean target
```

Existing `combiner/` artifacts, routes, viewer tiers, evaluation metrics,
power-spectrum curves, and fitting controls retain their current behavior and
backward-compatible paths. The combined experiment is additive and never
replaces or silently promotes itself over an ordinary combiner.

In particular, the field viewer's `combiner` tier and the pixel-level
disagreement/error diagnostics continue to use the ordinary regime combiner.
The combined combiner is exposed through its bottom card, test metrics, and
power-spectrum curve only.

## Member pool

Each combined combiner consumes every registry-active PSNR-best member across
both regimes. Member order is the canonical active-registry order, filtered
only for checkpoint presence; it must match the order produced by
`EnsembleModel(..., starless=None)`.

Every member carries its source regime in the combined payload:

```json
{
  "label": "07·psnr",
  "starless": true
}
```

The existing loss, depth, knee, step, and test-PSNR metadata remain available.
Labels stay globally unique because ensemble member numbers are allocated from
one registry across both regimes.

A combined fit requires at least one active starfull member and at least one
active starless member. A one-regime union would duplicate an ordinary
combiner and therefore fails closed with a concrete UI explanation.

## Shared target-independent prediction cache

Model inference depends on the dirty LR input and member checkpoints, not on
whether the fit target is `hr` or `clean`. Both combined models therefore use
one shared all-member prediction cache:

```text
<VIS_DIR>/ensemble/combined/
  cubes_validate/
    viz_index.json
    member0_<record>.npy
    member1_<record>.npy
    ...
  cubes/
    viz_index.json
    member0_<record>.npy
    member1_<record>.npy
    ...
```

The validate bucket supplies fit buffers. The test bucket supplies automatic
post-fit scoring and the combined-combiner power-spectrum curve. These buckets
store member predictions only; they do not become another field viewer or
duplicate the ordinary regime mean, standard-deviation, and PCA artifacts.

Each manifest records:

- a cache schema/version;
- subset and covered record indices;
- the ordered all-member labels;
- checkpoint fingerprints for those labels;
- the dirty-record fingerprint; and
- the HR-grid shape and band order.

The cache is reusable only when all of those values match. A request for fewer
fields reuses the covered prefix. A request for additional fields infers only
missing record indices when the manifest identity still matches. A changed
dirty dataset, member order, active member set, checkpoint fingerprint, shape,
or band order invalidates the bucket.

The first combined fit may therefore perform all-member inference. The other
target's fit reuses exactly the same predictions and performs only target
loading, fit-buffer assembly, RBF optimization, and scoring.

## Target-specific fitting and persistence

The two combined models are independent artifacts under the existing regime
roots:

```text
<VIS_DIR>/ensemble/starfull/
  combined_combiner/
    combiner.npz
    combiner.json
  combined_combiner_evals.json

<VIS_DIR>/ensemble/starless/
  combined_combiner/
    combiner.npz
    combiner.json
  combined_combiner_evals.json
```

The ordinary `<regime>/combiner/` directories are untouched. Persistence may
generalize `save_combiner` and `load_combiner` to accept an explicit artifact
directory, but the serialized arrays, manifest schema, model types, and default
ordinary-combiner location remain backward compatible.

For a starfull fit, shared validation predictions are paired with
`hr_validate`; for a starless fit, the identical predictions are paired with
`clean_validate`. The target-specific model manifest records:

- target regime and target kind;
- ordered all-member labels and source regimes;
- the shared prediction-cache fingerprint;
- the target-record fingerprint;
- validate field count and indices; and
- the existing kernel, pruning, validation-loss, and fit metadata.

The job uses the existing `FitBufferAccumulator` and `fit_combiner` functions.
No combined-specific fitting math is introduced.

## Automatic test scoring

After fitting, the job applies the new model to the shared all-member test
stack for the target page's current evaluated record indices. It writes only
the target-specific combined output needed by existing evaluation consumers:

```text
<VIS_DIR>/ensemble/<regime>/cubes/combined_comb_<record>.npy
```

The ordinary `comb_<record>.npy` output is not overwritten. The regime cube
manifest separately records whether a current combined output exists and the
all-member identity that produced it.

Automatic scoring reports VIS stretched PSNR for:

- the combined combiner;
- the ordinary same-regime combiner, when current;
- the same-regime ensemble mean; and
- the best same-regime member.

The combined payload includes gains versus the ensemble mean and ordinary
combiner. Missing or stale ordinary-combiner results are shown as unavailable,
not treated as zero.

## Power-spectrum integration

`EnsembleSpectrumAccumulator` gains a second optional reconstruction input for
the combined combiner. It accumulates the same asinh-space auto-power and
HR-cross-power statistics already used for the ordinary combiner and emits:

```text
P_combined
r_combined
T_combined = sqrt(P_combined / P_hr)
```

The existing `P_comb`, `r_comb`, and `T_comb` keys retain their meaning. The
pure `ensemble_ps_plot_curves` transform exposes both curve families.

Both Ensemble pages show a separately colored and independently toggleable
`combined combiner` curve alongside:

- LR baseline;
- same-regime ensemble mean;
- ordinary same-regime combiner;
- individual members; and
- model-to-model cross-correlation.

The React plot, classic-page plot/JSON behavior, regenerated static spectrum,
and cached evaluation-payload rebuild all use the same two-combiner spectrum
data. If the combined model or its output is missing or stale, its curve is
omitted while every existing curve remains available.

## Web and job behavior

Each `/ensemble/starfull` and `/ensemble/starless` page adds a `Combined
combiner` card after the disagreement viewer, making it the final card on the
page. The classic Ensemble surface receives equivalent controls and status.

The card reuses the ordinary combiner presentation and controls:

- validate field count;
- RBF kernel count;
- minimum cumulative importance;
- local background-job progress and logs;
- fitted/stale status;
- validation L1;
- surviving-member count;
- member-importance bars; and
- per-band gate weight versus brightness.

It adds source-regime information to member rows and an optional `by star
regime` coloring mode. This makes the hypothesis directly inspectable: the
gate plot shows whether starless or starfull members dominate at different
brightness levels.

The fitting endpoint is additive:

```text
POST /ensemble/combined-combiner/fit
GET  /ensemble/combined-combiner.json?mode=starfull|starless
```

The existing ordinary-combiner endpoints are unchanged. A fit request applies
only to the active page's target model; it does not refit both targets. Shared
prediction caching makes fitting the second target cheap without coupling the
two model lifecycles.

The fit control is disabled with a truthful reason when:

- either source member pool is empty;
- dirty validation or the active target validation record is absent;
- the target page has no current test evaluation to score against; or
- required local records are unavailable.

Job progress names the phases explicitly: validate cache reuse/inference,
fit-buffer assembly, per-band RBF fitting, test cache reuse/inference, combined
output application, and metric/spectrum refresh.

## Staleness and archive behavior

A combined combiner becomes stale when any of these change:

- active all-member labels or their order;
- a contributing checkpoint fingerprint;
- the shared prediction-cache identity;
- the target validation records; or
- combiner serialization compatibility.

A target test-record change invalidates the combined test output and curve but
does not invalidate a model fitted on unchanged validation data. A target
validation-record change invalidates only that target's combined model. A dirty
input-record change invalidates the shared predictions and both target models.

Archiving a member reconciles both combined models using the existing exact
pruning rule:

- if the member is pruned in every band, remove its zero-weight column,
  reindex the model and shared caches, and preserve the output exactly;
- if the member contributes to either model, invalidate that affected model
  and its target-specific outputs; and
- independently apply the existing reconciliation to the member's ordinary
  regime combiner.

The UI may display a stale combined model and its learned gate for inspection,
but current metrics and power-spectrum curves never use stale output.

## Failure behavior

- Empty member pools fail before TensorFlow model loading.
- Missing or mismatched records report the exact required dirty/target path.
- A partial cache is not advertised as complete; successfully written fields
  remain reusable by record index.
- Cache writes use temporary files plus atomic replacement for manifests and
  per-field arrays.
- A failed first target fit leaves the ordinary combiner and the other target's
  combined model unchanged.
- Incompatible or old combined artifacts return an unavailable/stale payload
  rather than falling back to an ordinary combiner.
- Power-spectrum refresh failure does not delete a successfully fitted model;
  the job reports that fitting succeeded but test visualization needs a retry.

## Testing and verification

### Model and cache tests

- The combined path calls the existing `fit_combiner` with an all-member
  `(N, M)` input buffer and does not introduce another model implementation.
- Starfull and starless fits consume identical cached prediction matrices but
  different `hr` and `clean` targets.
- Saving/loading a combined artifact does not change ordinary-combiner paths or
  backward compatibility.
- The second target fit reuses a valid shared validation/test cache without
  member inference.
- A larger field request infers only missing record indices.
- Dirty-record, checkpoint, active-member, ordering, schema, shape, and band
  changes invalidate the appropriate shared cache.

### Lifecycle tests

- Target validation changes invalidate only the matching combined model.
- Target test changes invalidate only its scored output and curve.
- An archived pruned member is removed exactly without changing predictions.
- An archived contributing member invalidates the affected combined model.
- Ordinary regime combiners remain independently loadable and scoreable through
  every combined-model lifecycle operation.

### Evaluation and spectrum tests

- Automatic test application writes `combined_comb_*` without overwriting
  `comb_*`.
- Combined PSNR and gains versus mean/ordinary combiner use the same stretched
  VIS metric.
- The spectrum accumulator emits `P_combined` and `r_combined` independently of
  the ordinary combiner.
- `ensemble_ps_plot_curves` derives `T_combined` correctly.
- Cached payload rebuild and spectrum regeneration preserve both combiner
  curves.
- Missing/stale combined output omits only the combined curve.

### Web tests

- Both target modes spawn the new fit job with the correct target.
- Payload member order, metadata, source-regime tags, surviving masks, and gate
  curves stay aligned.
- Both React and classic pages render the final combined card and honest
  disabled states.
- The combined spectrum chip/legend/curve is distinct and independently
  toggleable.
- The ordinary combiner card, viewer tier, and diagnostic point estimate remain
  unchanged.

Verification uses the repository's focused Conda workflow: targeted backend
tests with writable cache directories and JIT/plugin isolation where needed,
Ruff, Python compilation, frontend build, and `git diff --check`.

## Out of scope

- Any change to the RBF combiner's mathematical architecture.
- A CNN, CNN-conditioned RBF gate, spatial routing, or learned convolutional
  combiner.
- Retraining starfull or starless base members.
- Replacing the ordinary combiner as the viewer or diagnostic point estimate.
- Adding a combined-combiner viewer tier or disagreement basis.
- Fitting both target models from one button click.
- Changing the scientific definition of the existing power-spectrum curves.
