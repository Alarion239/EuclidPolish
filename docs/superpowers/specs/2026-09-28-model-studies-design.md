# Model studies — frozen, whole-ensemble comparisons for the paper — design

Date: 2026-09-28. Status: decisions below approved by the user in chat; spec awaiting review.

## Why

The user will write a paper and needs figures that justify the chosen losses and knees. Today a
comparison lives only as long as its members: archiving a member zips its checkpoint, but the next
Evaluate rebuilds the cubes, knee-PSNR curves and leaderboard without it. Getting it back needs a
restore plus re-evaluation, or a time-travel second console. The user wants to "save the comparisons
and data of the model comparisons, and then work without this model, but with an option to later
retrieve this comparison and it be as interactive and capable of generating good plots for research".

## Decisions (user)

- A study freezes the **whole ensemble** (every active member, the production gate, the combiner
  comparison), not a hand-picked subset. Selections and groupings happen later, inside the study.
- A study may attach **at most 10 fields** (images).
- Freezing is a **dialog**: it shows what will be frozen, shows which fields are available with this
  setup, asks whether to proceed, then lets the user choose fields.
- Attached fields are stored **on holylabs only, fetched on demand** (the Mac has ~7 GiB free).
- Studies live in **Figures › Studies**; the freeze starts from **Models › Leaderboard** (and the palette).

## What a study is

An immutable record `<campaign>/studies/<study-id>/` in the active tracking campaign
(`euclid_polish/tracking`), mirrored to holylabs like every campaign. It never references a live
checkpoint, cube or cache: every value it shows is inside it (numbers) or in its holylabs field store.

`study.json` (the manifest):
- id (`YYYYMMDD-HHMMSS-<slug>`), name, note, created (UTC), code commit (+ dirty flag), regime;
- the ensemble snapshot: for every member, its label, name, full recipe (`origin.json`: loss, asinh
  knee or knees, output knee, knee loss, bootstrap, noise augmentation, ICNR, depth, seed, target
  steps, steps reached, status incl. TIMEOUT, created, commit, forked from) and checkpoint fingerprint;
- the production gate identity (kind, variant, member_labels, reads, mix space, artifact fingerprint,
  fit_meta summary) and every gate variant's summary;
- the records identity (`records_fp`, noise model, subset, field indices);
- content hashes (sha256) of every numbers file and of every field file on holylabs;
- `complete: true` only after everything is written (a partial study is never listed as usable).

`numbers/` (local and mirrored; a few MB):
- `members.csv`: one row per member, recipe columns + summary scores;
- `knee_psnr.json`: PSNR-vs-knee curves per member, band and field (the 11-knee grid 0.1–1e4 e⁻)
  for members, plain mean and production gate, **per field** so paired statistics are possible
  (computed from the cached test cubes at freeze time if the cache holds only aggregates);
- `integrated.csv`: knee-integrated PSNR per member × band × field;
- `training_curves.json`: validation PSNR/loss vs step per member (from the training logs already
  pulled);
- `gate.json`: held-out weight diagnostic (usage, source usage, by brightness) and the combiner
  comparison report (natural and blackout test fields);
- `real.json`: the real-tile experiment metrics of the runs that used this membership (holes %, flux
  R, per band), with their experiment ids.

Attached fields (holylabs only): for each chosen field, one compressed `.npz` per product —
every member's SR, plain mean, production gate, LR, HR (synthetic only), blackout mask (blackout
fields) — float32, lossless compression, plus `truth.json` (source catalogue) and `field.json`
(source, id, WCS/pixel scale, sizes, sha256 of each file). Written to
`<holylabs tracking root>/<campaign>/studies/<study-id>/fields/<field-id>/`.

## The freeze dialog

Opened by "Freeze study…" on Models › Leaderboard (and the palette). A read-only
`GET /api/studies/candidates?mode=` builds its contents; nothing is written until the last step.

1. **What will be frozen** — "37 members · production gate p20s1 · evaluated 2 h ago" with a
   status per numbers block (current / stale, with the reason, e.g. "evaluation predates members
   203–205"), the numbers size (a few MB) and a clear warning when anything is stale, with the
   choice to go back and re-evaluate. Buttons: *Choose fields…* / *Freeze without fields* / *Cancel*.
2. **Choose fields (0–10)** — a thumbnail gallery of every field that has SR for **all** members
   under this setup:
   - synthetic test fields (100; HR truth, so metrics can be recomputed at any knee);
   - blackout test fields (40);
   - real tiles with cached member SR for every member (the poster galaxy, NEXUS tiles from past
     comparisons).
   Each shows its size on holylabs; the counter reads "3 of 10 · 480 MB to upload". Fields missing
   a member's SR are listed but disabled with the reason. FASRC must be connected to attach fields
   (the dialog says so, and freezing without fields stays possible offline).
3. **Name and confirm** — name (required), note, summary of what will be written and uploaded;
   *Freeze* starts one cancellable local job.

The job: write the numbers locally → write the manifest (incomplete) → for each field, pack one
product at a time into a temp file, upload it to holylabs, verify its sha256 remotely, delete the
temp file (local disk never holds more than one product) → mark `complete` → mirror the campaign.
A failure leaves an incomplete study that the list shows as "incomplete — resume or delete".

## Figures › Studies

- **List**: name, created, members, gate, fields, note; open, rename note, delete (confirmed, also
  deletes its holylabs field store).
- **Study view** (reads only the study):
  - *Selection*: every member with its recipe; group or colour by loss, training knee(s), output
    knee, knee loss, depth, bootstrap, noise augmentation; pick subsets; name selections (saved in
    the study as a small sidecar, the frozen data itself never changes).
  - *Charts* (interactive, same kit as Models):
    1. PSNR vs knee per band, one curve per member or per group (median + band of members),
       plus mean and gate;
    2. integrated PSNR by loss family × training knee, one dot per member (seeds visible);
    3. paired per-field differences of a group or member against a reference (another group, the
       best member, the plain mean): mean Δ integrated PSNR with a 95 % paired bootstrap interval
       over fields (2,000 resamples, seed recorded), per band;
    4. the gate's weight per member and per family (all / sources / brightness bins);
    5. training curves (validation PSNR vs step), grouped;
    6. real-tile metrics per model when the study has them.
  - *Fields*: the attached fields in the full viewer (members, mean, gate, LR, HR, residuals,
    truth markers), fetched from holylabs on first open into a local cache with a size budget
    that keeps at least 5 GiB free (least-recently-used eviction; "Fetch field" is explicit, never
    on page open).
  - *Export*: each chart as PDF, PNG (300 dpi) and SVG rendered by the backend in the publication
    plate style, plus the CSV of exactly the numbers drawn; a "Log to notebook" entry citing the
    study id and hash.

## Rules

- A study is immutable once complete; only its note and named selections can change (sidecar).
- Opening Studies or a study never starts a job and never fetches a field; fetching and exporting
  are explicit.
- No numbers are invented: a chart whose data the study lacks says what is missing.
- Statistics follow the console's rule (readable, informative, nothing useless); paired intervals
  are the default way to state a difference.
- Archiving, retraining or deleting members never touches a study.

## Out of scope

Re-evaluating members from a study, restoring checkpoints (time travel remains the way to rerun old
code), and editing frozen data.

## Acceptance

- Freeze a study with 3 fields; archive a member; the study still shows that member in every chart
  and its fields in the viewer; exports reproduce the same numbers (CSV) and figures.
- The dialog lists only fields with SR for every member, blocks > 10, shows sizes, works offline
  without fields.
- The Mac's free space never drops below the 5 GiB margin while freezing or viewing.
- Tests for the manifest, packing/verification, bootstrap intervals, candidates and the dialog;
  typecheck, lint, vitest, pytest green; browser check in both themes.
