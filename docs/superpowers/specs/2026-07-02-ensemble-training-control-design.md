# Ensemble Training Control: add / continue / fork

**Date:** 2026-07-02
**Status:** Approved

## Problem

Training control is rudimentary: `EnsembleModel.train(n_members=N)` blindly
loops `member_00..N-1` — it silently resumes members that exist and no-ops
members already at the step target. There is no way to (1) add new members
without touching the old ones, (2) deliberately continue training selected
members, or (3) branch a new member off an existing one with the LR schedule
reset to step 0.

Relevant mechanics (already in the trainer, reused as-is):

- `Trainer` auto-restores the latest checkpoint; `ckpt.step` persists;
  `steps` is an ABSOLUTE target (`while ckpt.step < steps`).
- The warmup→cosine LR is evaluated at the absolute step with
  `total_steps = steps` of the current run; warmup is a fixed early window,
  so a resumed run past warmup never re-warms.
- Weights-only restore from a checkpoint is an established pattern
  (`load_model_from_checkpoint`).
- Archived members are permanent registry tombstones — their names must
  NEVER be reused (the bootstrap would ignore a reincarnated name forever).

## Decision

Spec-driven sequential training with three modes on the existing script/step
(one FASRC GPU job per submission), explicit member-name allocation at submit
time, per-member `origin.json` provenance written by the job, and a single
mode-switching train card on /ensemble with per-row continue/fork shortcuts.

Chosen over a JSON `--spec` blob (opaque in job history, YAGNI) and over
three separate FASRC steps (registry/job-history clutter for 90%-shared
machinery).

## Design

### 1. Member-name allocation (registry)

- `ensemble_registry.next_member_names(base_dir, k) -> list[str]`: `k`
  consecutive fresh names starting at
  `max(index over active ∪ archived-tombstones ∪ member_* on disk) + 1`
  (`member_00` when none exist). Tombstoned indices are skipped forever.
- `EnsembleTrainStep.build_command` calls it at SUBMIT time and passes
  explicit `--member-names m,…` to the job — required because the remote
  ensemble dir still holds archived members' directories, so the job cannot
  infer safe indices from remote disk alone.
- Runtime collision guard in the script: if a passed name already exists on
  the FASRC filesystem (e.g. two queued add-jobs allocated the same name at
  submit time), shift to the next free on-disk index and log the shift.

### 2. Fork at the Model level

- `Model.__init__` gains `init_weights_from: str | None` (a checkpoint DIR:
  the source member's root for the psnr track or its `loss_best/` for the
  loss track).
- Behaviour: build fresh (seeded) as usual; if the member's OWN dir has no
  checkpoint and `init_weights_from` is set, copy weights from the source via
  the weights-only restore pattern (`load_model_from_checkpoint` +
  `set_weights`). Step stays 0, optimizer fresh, warmup→cosine starts over —
  "initialized as a previous model, LR schedule reset".
- Raise `ValueError` if the member's own dir already has a checkpoint
  (fork targets must be virgin) or the source has none.

### 3. Spec-driven ensemble training

- `MemberTrainSpec` dataclass in `euclid_polish/ensemble.py`:
  `name` (e.g. `member_09`), `seed` (int), `target_steps` (absolute),
  `init_from` (checkpoint dir or `None`), `run_steps` (steps this run will
  actually execute — for progress accounting; equals `target_steps` for
  add/fork, `extra_steps` for continue).
- `EnsembleModel.train_members(lr_path, hr_path, specs, **train_kwargs)`:
  sequential loop; per spec constructs
  `Model(member_dir, seed=spec.seed, init_weights_from=spec.init_from)` and
  calls `model.train(steps=spec.target_steps, …)` with the shared
  LR/plateau/callback kwargs. Existing `on_member`/callback plumbing is kept.
- Mode → specs:
  - **add**: `k` fresh names; seeds `base_seed + i` (entropy base when
    unset); `target_steps = steps`.
  - **continue**: for each selected member, read the authoritative `step`
    from its latest checkpoint (same reader the rollback path uses);
    `target_steps = current + extra_steps`. The cosine recomputes over the
    new total → warm cosine restart (approved; no re-warmup since the warmup
    window is long past). A member with no checkpoint fails fast with a
    clear message.
  - **fork**: `k` fresh names; each `init_from = <source>/<track dir>`;
    distinct seeds; `target_steps = steps`; step 0.
- Every member CREATED by a run (add/fork) gets `origin.json` in its dir:
  `{op, forked_from (name·track or null), seed, target_steps, created_at,
  commit}` — written by the job on FASRC, so it rsyncs down with the member
  and provenance survives without registry coupling. Continue does not
  rewrite it.
- Old `EnsembleModel.train(n_members=N, …)` becomes a thin wrapper that
  builds add-mode specs for `member_00..N-1` (back-compat for tests/manual
  use; the WebUI never calls it).

### 4. CLI (`scripts/train_ensemble.py`)

- `--mode {add,continue,fork}` (default `add`).
- add: `--count N` (replaces `--n-members`; keep `--n-members` as a
  deprecated alias for `--count` that also forces legacy 0..N-1 naming OFF —
  it now means "add N new members"), `--steps` (absolute), `--base-seed`.
- continue: `--members member_03,member_05`, `--extra-steps M`.
- fork: `--fork-from member_02`, `--fork-track {psnr,loss}` (default psnr),
  `--count K`, `--steps`, `--base-seed`.
- Common: `--member-names` (explicit allocation from the WebUI; when absent
  the script computes from on-disk max — manual-CLI fallback), all existing
  LR/plateau knobs, `--eval-images` post-train ensemble eval, staging,
  Reporter progress. The cumulative progress bar sums each spec's
  `run_steps` (continue contributes `extra`, not its absolute target).

### 5. FASRC step (`EnsembleTrainStep`)

- `build_command` reads `mode` + mode fields from the form and emits the
  flags above; for add/fork it calls `next_member_names()` and emits
  `--member-names`. Job label reflects the operation (e.g.
  `"ensemble-train: fork member_02 ×3"`); `job_name` stays `ensemble-train`.
- No new step ids; the generic submit/queue route is unchanged.

### 6. UI (/ensemble)

- `ensemble_status()` per-member additions: `step` (last row of the member's
  `training_log.csv`; blank when unreadable) and `origin` (parsed
  `origin.json` or `None`).
- Template injects `window.ENSEMBLE_MEMBERS = [{name, step, has_loss_best}]`
  for the step card.
- `fasrc_step_card.js` `ensemble_train` case: a **Mode** select
  (Add new / Continue existing / Fork from member) swapping three field
  sets: add = count/steps/base-seed; continue = member checkboxes (name +
  current step) / extra-steps; fork = source select / track select / count /
  steps / base-seed. Hidden inputs carry `mode` and the member selections.
- Members table: new *Step* column; ⑂ badge with origin tooltip on forked
  members; per-row **▶ continue** and **⑂ fork** buttons that set the card's
  mode, prefill/select that member, and scroll to the card.

### 7. Sync-back

Unchanged: new members appear on FASRC, arrive via ensemble pull / the
mirror, and the registry bootstrap activates them. Continue/fork read the
REMOTE member's checkpoint; if a member exists only locally the job fails
fast with a clear message (in practice all members are FASRC-trained).

### 8. Testing

- Registry: `next_member_names` skips tombstones and on-disk gaps; k>1
  consecutive.
- Model fork: weights equal source after init, trainer step 0, virgin-dir
  and missing-source guards (tiny-model fixtures, as in trainer tests).
- Spec building: CLI args → specs for all three modes; continue reads
  `current + extra` from a real checkpoint; collision shift.
- `build_command`: correct flags per mode; `--member-names` emitted for
  add/fork.
- `ensemble_status`: step column + origin parsing.

## Out of scope

- Parallel member training (multiple GPUs/jobs) — sequential single-job as
  today.
- Deleting/reverting a continue run (tracking backups cover rollback).
- Mixed-mode submissions (one job = one mode).
- Per-member LR-knob overrides (the global /config knobs apply to the run).
