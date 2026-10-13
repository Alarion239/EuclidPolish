# Universal multi-knee models as first-class models — design

Date: 2026-10-13. Status: approved by the user's request ("make these universal
multiple-knee models class 1 models … run their inference on the real fields and
compare to the combiner … compare all of them between each other … training
length and model depth … same-combined-training-time combiner … comparison based
on the training time they took; the combiner sums the time of the models it uses,
weighted by their max weight, not their mean weight").

## 1. Definitions

* **Universal model** — a single-image multi-knee member: its `origin.json` has a
  non-empty `asinh_knees` AND (`output_knee` is not None OR `learned_output_knee`
  is truthy). Learned-knee members (206+) have `learned_output_knee` as a DICT
  `{init_e, min_e, max_e}` and NO `output_knee` key; fixed-knee ones (195/196)
  have `output_knee: 10.0`. Member 197/198 (asinh_knees, no output knee: six
  heads) are NOT universal. In FASRC job-ledger `member_spec` entries the same
  models carry `asinh_knees` plus `learn_output_knee: true` (and still
  `output_knee: 10`, the init value) or plain `output_knee`.
  Python helper: `euclid_polish.ensemble_registry.is_universal_origin(origin: dict) -> bool`.
  TS helper: `isUniversal(m: KneeInfo)` in `workspaces/models/model.ts`.
* **Universal class (registry)** — a universal model pulled locally as a
  STANDALONE model: it is NOT an ensemble member (not in the production gate, the
  plain mean, Evaluate cubes, knee-PSNR identity, real-field cube cache, gate
  fits). Active members that happen to be universal (195/196) stay ensemble
  members; they are listed among the universal models too (class `member`).
* **Training time** of a model — Σ `duration_s` over the non-baseline rows of its
  `training_log.csv` (every row, rolled-back windows included; never deduped),
  in hours. Plus the hardware it ran on, from the job ledger.
* **Combiner training time** — Σ over the members the gate READS of
  `hours(member) × max_weight(member)`, where `max_weight` is the member's PEAK
  gate share (`eval.gate_members.member_peak_weights`: max over bands of the
  all-pixel, source-pixel and brightness-bin MEAN weights — the console's "peak"
  share, the "used by gate" rule). Also reported: weighting by the mean weight
  (mean over bands of the all-pixel share) and the unweighted sum. Default and
  headline: max weight.

## 2. Registry (`euclid_polish/ensemble_registry.py`) — owner: agent A

* `ckpt/ensemble_registry.json` gains a third key `"universal": [sorted member
  names]`. `load_registry(base_dir)` returns `{"active", "archived", "universal"}`.
  * Bootstrap (`on_disk - seen`) must treat universal names as seen: a universal
    dir is never auto-activated.
  * Universal entries are NEVER dropped because their dir is missing (mid-pull,
    mirror-deleted).
  * Whenever it writes, it writes all three keys (an old file without
    `universal` reads as `[]`; the key is not written until something changes,
    exactly like today).
  * A module-level `threading.Lock` guards every read-modify-write
    (`load_registry` writes, `add_universal_entry`, archive, restore).
* New functions:
  * `is_universal_origin(origin: Mapping) -> bool` (definition above).
  * `universal_names(base_dir) -> list[str]` (registered, sorted).
  * `universal_member_dirs(base_dir) -> list[str]` (registered ∩ dir exists ∩
    has checkpoint).
  * `universal_labels(base_dir) -> list[str]` (`"NNN·psnr"`).
  * `add_universal_entry(base_dir, name) -> str` — normalises with `member_name`,
    raises `ValueError` if the name is active, tombstoned or malformed;
    idempotent; returns the normalised name. Called BEFORE a pull's rsync.
* `next_member_names` counts universal names too.
* `archive_member_entry(base_dir, name)` accepts active OR universal names; the
  tombstone records `"class": "universal"` or `"member"` (old tombstones have no
  key = member). `restore_member_entry` restores into the recorded class.
* Unchanged: `active_member_dirs`, `active_labels`, `regime_labels` stay
  ensemble-only.

## 3. Loading (`euclid_polish/ensemble.py`) — owner: agent A

`EnsembleModel(base_dir, ..., labels=[...], include_universal=True)`: when
`include_universal` is true the candidate dirs are active ∪ universal member dirs
(each with a checkpoint). `include_universal=True` without `labels` raises
`ValueError("include_universal needs explicit labels")` — the default ensemble
(mean, Evaluate, gate fits, production) never sees universal models. The ensemble
-only rule holds: a universal model is an ensemble of one. `_select_member_dirs`
error text for an unknown label becomes "… is not an active ensemble member or a
universal model with a checkpoint" only when `include_universal` is on.

## 4. Model catalogue / real fields — owner: agent B

* New kind `"universal"`, prefix `UNIVERSAL_PREFIX = "universal:"`, spec
  `universal:member_NNN` (aliases `universal:NNN`, `universal:NNN·psnr`,
  `universal:member_NNN`; reuse `_MEMBER_DIGITS`). `spec_slug` →
  `universal-member_NNN`. `canonical_spec`'s error hint lists `universal:<name>`.
  Exact existing error strings (`unknown model spec '<spec>'`) are kept.
* `list_specs` appends one `ModelSpec` per REGISTERED universal model AFTER the
  gate variants (so `[:3]` stays production, mean, rbf): `kind="universal"`,
  `label="Universal NNN"`, `member_labels=reads=("NNN·psnr",)`, `available`
  iff it has a checkpoint and a STARFULL origin.json (`reason` otherwise), fingerprint
  `spec_fingerprint("universal", …) = _hash("universal", label, fp)`.
* `details` (cheap reads only: origin.json, training_log.csv; cache by
  `(dir, member_fingerprint)`; NO TensorFlow reads in `list_specs`):
  `{universal: true, class: "universal"|"member", num_res_blocks, steps,
  target_steps, batch_size, crops_per_field, knee: "learned"|"fixed",
  output_knee, jit_compile, mixed_precision, train_hours}`.
  Active universal members (`member:member_195/196`) get the SAME details with
  `class: "member"` (their kind stays `member`).
* `EnsembleMemberRunner` restores labels from active ∪ universal (builds the
  `EnsembleModel` with `include_universal=True` and explicit labels); predict for
  kind `universal` = `members.get(reads[0])` like `member`.
* `model_catalog.member_fingerprints` already resolves `<ensemble_dir>/member_NNN`
  — universal dirs live there too, so the member-SR cache and
  `experiments.purge_stale_member_sr_cache` keep current universal entries.
* `real_tiles._spec_kind`, `viewer_data` (`_real_default_specs` order: production,
  mean, rbf, member:, gate:, universal:, others; `_SPEC_NAMES`/tier labels),
  `nexus_plates.model_short` ("universal NNN"), `studies/numbers.py` (member label
  for universal specs too) learn the prefix.
* Tests: `tests/test_model_catalog.py`, `tests/test_experiments.py`,
  `tests/test_real_routes.py`, `tests/_real_fixtures.py` (a universal stub), plus
  any viewer/nexus tests touched. Report the API.md C9 text change in the final
  answer (agent C owns API.md).

## 5. Universal backend — owner: agent C

New module `euclid_polish/web/helpers/universal_models.py` (imports at module top;
no SSH on reads; atomic JSON writes via `ensemble_viz._atomic_json` or
`atomic_files`):

* `train_time(member_dir) -> dict` — `{hours, steps, rows, rollback_rows,
  first_wall, last_wall}` from `training_log.csv` (missing file → hours None);
  also reads a training_log.csv inside an archived tracking zip when given a zip.
* `ledger_jobs(name) -> list[dict]` — every `ensemble_train` ledger row naming the
  member (`member_names[i]` for add/fork, `members` for continue) via
  `fasrc_jobs.JOBLOG.history_for_step("ensemble_train")`; `hardware(name)` →
  `{partition: "gpu"|"gpu_test"|…|None, label: "A100"|"A100 MIG slice"|"mixed"|None,
  jobs: [jobid…]}`.
* `fasrc_universal_rows()` — universal models named in ledger add jobs (member
  spec universal) that are neither active, registered universal nor tombstoned:
  `{name, jobids, state, num_res_blocks, steps, knee, submitted_at, ended_at,
  partition, finished}` (finished = the newest job naming it is in a final state).
* Scores store `<VIS_DIR>/ensemble/starfull/universal/`: `<member_NNN>.npz`
  `{fields int[F], knees float[21], curves float[F,21,4]}` + `index.json`
  `{schema: 1, identity: {records_fp, subset: "test", indices, target_psf_fwhm_arcsec,
  knees}, models: {member_NNN: {fingerprint, integrated: [4], mean, n_fields,
  scored_at, seconds}}}`. Fields = the knee identity's `indices` when the
  Leaderboard payload exists, else `range(100)`; truth = `hr_test` blurred to
  0.066″ (`blur_target_array`), `stretched_truth`, `knee_psnr` — the Leaderboard's
  exact metric (see scratchpad prior art `score_member.py`). NEVER write into the
  cube buckets.
* Gate weights per gate (`production` and every `spatial_gate_<variant>/` dir):
  production from `spatial_gate_combiner_evals.json` (`gate_diagnostics` +
  `member_peak_weights`); variants from `<regime>/<variant dir>_evals.json`
  `{artifact_fp, gate_diagnostics}` computed with
  `ensemble_viz._spatial_gate_weight_diagnostic(load_spatial_gate(dir), starless=False)`
  when every read member has validate cubes (else unavailable + reason). Max =
  `gate_members.member_peak_weights`, mean = mean over bands of `usage`.
* Local job `job_universal_score(job, *, members=None, force=False)` (kind
  `universal-score`, `REGISTRY.spawn_exclusive`): scores every universal model
  (active universal + registered universal, or the given subset) whose entry is
  missing or stale (fingerprint / identity), loading each through
  `EnsembleModel(ensemble_dir, labels=[label], include_universal=True,
  starless=False)`; then computes missing variant gate diagnostics. Progress via
  `cap.tick`/the job API like the other ensemble jobs.
* Pull job `job_universal_pull(job, *, members)` (kind `universal-pull`,
  `@requires_fasrc` route): per name — refuse active/tombstoned/malformed; read the
  remote `origin.json` (ssh `cat`), refuse non-universal; `add_universal_entry`
  BEFORE the rsync; rsync `<remote ensemble>/<name>/` → `ckpt/ensemble/<name>/`
  (same helper as `job_ensemble_pull`); `prune_orphaned_checkpoints`. Result
  `{pulled: [...], refused: {name: reason}}`.
* Protect the ensemble: `job_ensemble_pull`'s "every changed member" (no explicit
  members) and its dry run exclude ledger-universal names that are not active or
  registered universal (report them as `universal_skipped`); registered universal
  names refresh in place (load_registry never activates them). The whole-ensemble
  mirror (`fasrc_mirror.Mirror._sync_once`) excludes ledger-universal names that
  are not registered (rsync `--exclude`).
* `member_detail(name)` serves registered universal models (row with class
  `universal`, no gate fields) so the Member inspector opens them.
* `members_payload` adds `universal: [registered names]`.
* Payload `universal_payload(starless=False)` for `GET /ensemble/universal.json`
  (local only; reads caches; never writes; answers on an empty ensemble):

```
{
  available: bool,                     // any universal model known
  knees: number[21], bands: ["VIS","Y_E","J_E","H_E"],
  integration: {from_e, to_e},
  identity: {...} | null,              // the scores store identity
  production: {integrated: number[4], mean: number, stale: boolean} | null,
  mean_model: {integrated: number[4], mean: number, stale: boolean} | null,
  models: UniversalRow[],
  combiners: CombinerRow[],
  fasrc: FasrcRow[],
  notes: string[]
}
UniversalRow = {
  name, label, spec,                   // spec "universal:member_211" | "member:member_196"
  class: "universal" | "member",
  num_res_blocks, steps, served_step: number|null, target_steps,
  batch_size, crops_per_field, knee: "learned"|"fixed", output_knee: number|null,
  learned_knees: number[]|null,        // from the scores store (recorded at scoring), never a TF read
  jit_compile: boolean, mixed_precision: boolean, bootstrap, seed,
  train_hours: number|null, train_rows: number,
  hardware: {partition, label, jobs: string[]},
  fingerprint: string|null,
  score: {integrated: number[4], mean: number, n_fields: number, stale: boolean, scored_at} | null,
  vs_production_db: number|null,
  created_at, commit
}
CombinerRow = {
  id: "production" | "gate:<variant>", name, label, spec, production: boolean,
  reads: string[], n_reads: number,
  integrated: number[4]|null, mean: number|null,
  score_source: "knee-psnr" | "compare" | null, stale: boolean,
  weights_source: "production" | "variant" | null, weights_reason: string|null,
  members: {label, name, hours: number|null, max_weight: number|null, mean_weight: number|null}[],
  hours: {max: number|null, mean: number|null, sum: number|null},
  missing_hours: string[],
  fit_hours: number|null
}
FasrcRow = {name, jobids: string[], state, num_res_blocks, steps, knee: "learned"|"fixed",
  submitted_at, ended_at, partition, finished: boolean}
```

  `production`/`mean_model` come from the knee-PSNR status (with its `stale`
  flag). Gate `integrated`: production from the knee payload; variants from
  their newest compare report knee row (`combiner_variants` already joins it).
* Routes in `routes/ensemble.py`: `GET /ensemble/universal.json`,
  `POST /ensemble/universal/score` (form `members`, `force`), `POST
  /ensemble/universal/pull` (`@requires_fasrc`, form `members` required),
  documented in `API.md` (Ensemble table + shapes + job kinds; also the C9 text
  agent B reports). `tests/test_fasrc_gate.py` marked list gets the pull.
* Tests: `tests/test_universal_models.py` (helpers, payload, jobs with fakes),
  route tests in `tests/test_ensemble_workspace_routes.py`, pull protection in
  `tests/test_ensemble_pull_skip.py`, mirror excludes in
  `tests/test_fasrc_mirror_sync.py`.

## 6. Console — Models › Universal tab — owner: agent D

* `spa_routes.json` models tabs: `[leaderboard, universal, members, train,
  combiner, diagnostics, images]` (second, for discoverability); `app/nav.ts` label
  "Universal" (+ description/keywords); `workspaces/models/index.tsx`; update
  `nav.test.ts`, `Shell.test.tsx`, `models.test.tsx`.
* `api.ts`: the payload types above, `url.universal()`, `useUniversal()`.
* Pure helpers in `workspaces/models/universal.ts` (+ `universal.test.ts`):
  rows shaping, combiner hours by weighting, time-matched combiner per model (the
  scored combiner nearest in log hours), series for the plots, depth colours.
* `tabs/Universal.tsx`:
  * Summary line (models, best, vs production gate with its max-weighted hours).
  * Callouts with confirmed job buttons: "Score N models on the 100 test fields"
    (POST score), "Pull N models from FASRC" (POST pull; disabled offline), stale
    production → link to Leaderboard/Evaluate. Opening the page never starts a job.
  * Toolbar (URL state): x axis Training time | Steps; y ∫PSNR mean | VIS | Y | J |
    H; combiner weighting Max weight (default) | Mean weight | Every member.
  * Plot "by training time": log-x hours, one scatter series per universal model
    coloured by depth (ring marker for MIG/gpu_test hardware), combiners as
    diamonds at their weighted hours (production in `C.comb`), production y guide.
  * Plot "by training length": log-x steps, a line per depth through that depth's
    models (dots), production y guide.
  * DataTable of models: model, class, depth, steps, batch, crops/field, knee,
    step (XLA/bf16), hardware, train h, ∫PSNR (+ per band, hidden by default), Δ vs
    production, Δ vs time-matched combiner (named), real holes % / median R̃ from
    the newest fingerprint-fresh Sky experiment holding the spec, status; inspect →
    member inspector; row action "Compare on real fields" → `/sky/compare?new=1&models=production,<spec>`.
  * Combiners table: gate, members read, Σ hours (all), max-weighted h,
    mean-weighted h, ∫PSNR, weights source.
  * "Compare all on real fields" button → `/sky/compare?new=1&models=production,<all local universal specs>`.
* Leaderboard: fix learned-knee facets (common.tsx `kneeColor`, Leaderboard
  `memberFacet` + dash: one-image = `output_knee != null || learned_output_knee`);
  add a "best universal model" row to the comparison table when the universal
  payload has a scored model (with real columns through its spec), and a link to
  the Universal tab.
* Members tab: `waitingOnFasrc` excludes universal jobs/members (they belong to
  the Universal tab); `members.json` `universal` names count as local.
* `jobs.ts`: JOB keys `universalScore`, `universalPull` with confirmed starters.
* Docs: `FOUNDATION.md` (§11.3 tab table, §11.9 page notes), README models tabs row.

## 7. Console — Sky model picker — owner: agent E

* `sky/results/api.ts` `ModelKind` += `"universal"`; typed universal `details`.
* `sky/results/model.ts`: `specRank` universal right after production (1; shift
  the rest), `specWords` ("universal 211"), `specShort` ("u211"), `groupModels`
  adds group `{id: "universal", label: "Universal models"}` right after production
  holding kind `universal` rows AND `member` rows whose `details.universal` is true
  (they also stay in Members), a `universalDetailText(m)` ("depth 16 · 1.0M steps ·
  learned knee · 12.7 h").
* `ModelPicker.tsx`: universal group uses the list layout with the detail line;
  quick picks "Production + universal" and "+ all universal".
* Fix any kind/prefix switch that would mis-handle universal (outputOrigin "1
  member", etc.). Tests in `results/model.test.ts`, `results/results.test.tsx`,
  `compare/*.test.*`, `testing/backend.tsx` fixture.

## 8. Not in scope (follow-ups)

Fitting new gates on member subsets to hit a time budget; promoting a universal
model into the ensemble; A100-equivalent normalisation of MIG hours (shown as
hardware labels instead).

## 9. Review fixes (2026-10-13, after the adversarial review)

Contract additions (all additive):

* **Fork recipes.** A fork inherits its source's knees. `ensemble_viz.ledger_member_recipes`
  resolves a fork job's knee keys (`asinh_knees`, `output_knee`, `learn_output_knee`,
  `num_res_blocks`) from `params.fork_from`: the local `<ensemble>/<src>/origin.json` when
  present (`learned_output_knee` → `learn_output_knee`), else the source's ledger recipe
  (recursively, cycle-guarded). The TS `universalJobMembers(job, knownUniversal)` treats a
  fork whose `fork_from` is a known universal name as universal.
* **Names that were ensemble members are never treated as FASRC-only universal models.**
  `unregistered_universal_names` / `fasrc_universal_rows` / `job_universal_pull` treat as
  known every name read by any local gate artifact (`spatial_gate_*/combiner.json`
  `member_labels`) and every name in a cube manifest (195/196 stay members even if their
  dirs vanish).
* **Explicit `/ensemble/pull`** drops unregistered universal names (reported in
  `universal_skipped`, "pull it on Models › Universal") unless the form sends
  `allow_universal=1`; when nothing else is left it answers 400.
* **Home loop** (`system_alerts.waiting_on_fasrc` / `members_stage`) treats registered
  universal names as local and keeps ledger-universal names out of the Members stage
  (they are mentioned as "N universal models on FASRC → Models › Universal", never stale).
* **Training curves** (`/ensemble/training-curves.json`): registered universal names count
  as local; remote entries of ledger-universal models carry `universal: true`.
* **Universal pull** checks the remote checkpoint before registering (same ssh call as the
  origin read: no checkpoint → refused "no checkpoint on FASRC", nothing registered); any
  rsync failure/exception after registering removes the registration this call added
  (`ensemble_registry.remove_universal_entry(base, name)`, which refuses non-universal
  names); per-name failures never abort the remaining names.
* **FasrcRow** gains `pullable: boolean` (newest job state COMPLETED or TIMEOUT) and
  `registered: boolean` (registered universal model with no local checkpoint — listed so it
  can be pulled again).
* **CombinerRow** gains `comparable: boolean` (its score is on the universal scores'
  fields/records/target: production → the payload-level rule; a compare-report score →
  the report's recorded identity `{records_fp, indices, target_psf_fwhm_arcsec}` equals the
  scores identity; legacy reports without one → false). Compare reports now record
  `identity` (from the test cube manifest) and `member_fps` (read members' recorded
  fingerprints); `_compare_stale` also marks a row stale when a read member's current
  fingerprint differs from the recorded one (legacy: when its checkpoint index mtime is
  newer than the report).
* **UniversalRow** gains `scored_hours`, `scored_steps` (train hours/steps recorded at
  scoring time; null for legacy entries) and keeps `served_step` for stale rows too.
* **Payload** gains `weights_pending: string[]` (combiner ids whose weights are missing but
  measurable now). The score job measures gate weights FIRST (validate cubes only), even
  when nothing needs scoring or the test records are missing.
* **member_detail** of a universal model: reference curves (mean/gate) only when
  comparable; exposes `comparable`; row carries `score_stale`. An archived universal model
  reports `class: "universal"` (from its tombstone).
* **Studies**: `matching_experiments` qualifies a record only through non-universal specs;
  `render._real_table` keeps universal rows (model name `spec_label`, never filtered by
  member labels).
* **Runner**: widening never appends an already-planned label.
* **Frontend**: Δ vs time-matched combiner only against comparable, non-stale, non-partial
  combiners (`missing_hours` non-empty = partial: hollow diamond, "≥ N h", excluded from
  time matching); stale scores drawn faded at `scored_hours`/`scored_steps` and their Δ
  cells marked stale; FASRC callout pulls `pullable` rows, failed ones listed with state and
  a per-row "Try pull", registered-but-missing ones offered "Pull again"; "Measure gate
  weights" when `weights_pending` is non-empty; depth colours skip the mean and combiner
  slots (`[1,3,4,5,6,7]`); inspector Δ vs plain mean only when comparable and current;
  restore wording by tombstone class; Members › Curves tags universal remote curves.
