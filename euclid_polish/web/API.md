# EuclidPolish web console — HTTP API

The Flask backend (`euclid_polish/web/`) serves the React console (SPA) and
every endpoint it calls. This file is the reference for the frontend and for
every backend work package: **document each endpoint you add, change or
delete here**. `tests/test_api_docs.py` fails when the endpoint tables below
drift from `app.url_map` (a route missing or stale, different methods, or a
different FASRC gate mark).

Contracts C1–C5 referenced below are defined in
`docs/superpowers/plans/2026-09-25-webui-rework.md` §1.

## Conventions

### Security boundary (loopback, zero login)

- The server binds to loopback only (`validate_bind_host`: `127.0.0.1`, `::1`,
  `localhost`).
- **Host allowlist.** `app.config["TRUSTED_HOSTS"] = ["localhost",
  "127.0.0.1", "[::1]", "::1"]` (Werkzeug compares the `Host` header without
  its port). Any other `Host` — a DNS-rebinding page, a LAN name — gets
  **400** before any other hook or handler runs:
  `{"ok": false, "error": "untrusted Host header", "code": "untrusted_host"}`
  for `/api/*` or JSON requests, plain text otherwise.
- **Mutations are POST-only** (or PUT/PATCH/DELETE), never GET, so a
  cross-site `<img src>` cannot trigger them. Unsafe methods are refused with
  **403** `{"ok": false, "error": "cross-origin request rejected"}` when
  `Sec-Fetch-Site: cross-site` or an `Origin` that differs from the request's
  own origin is sent. Bodies are form-encoded (`application/x-www-form-urlencoded`
  or multipart) unless an endpoint says it accepts JSON.
- Hook order in `create_app`: Host allowlist → cross-origin mutation guard →
  SPA shell / redirects → FASRC gate → handler.

### Errors

- JSON endpoints report failures as `{"ok": false, "error": "<message>",
  "code"?: "<machine code>"}` with a 4xx/5xx status. Known codes:
  `fasrc_offline` (503, below) and `untrusted_host` (400).
- Some older endpoints still answer `{"error": ...}` without `ok`, or an HTML
  `abort()` page; new endpoints must use the shape above. Everything under
  `/viewer/` answers JSON `{"error": "<message>"}` with the status code
  (collection errors, unknown routes, unhandled exceptions — contract C6).
- **Prefix-wide JSON errors** (`euclid_polish/web/errors.py`): Flask keeps
  one handler per exception class, so a route module must **never** register
  its own `@app.errorhandler(HTTPException)` (it would silently replace
  another module's). Call `errors.json_errors_for(app, "/prefix/")` from the
  module's `register(app)` instead: the first call installs the app's single
  path-dispatching handler (`errors.json_http_error`), later calls add
  prefixes (idempotent). Under a registered prefix every HTTP error — routing
  404/405 and an unhandled exception's 500 included — is `{"error"}` with its
  status; other paths keep Flask's default. Registered today: `/viewer/`,
  `/api/real/`, `/api/models`, `/api/experiments` (`routes/real.py`) and
  `/api/sky/` (`routes/sky_atlas.py`).
  `tests/test_web_errors.py` fails if any other HTTPException handler exists.
- Other codes: `config_conflict` (409, `/api/config/save`), `refused_files`
  (409) / `no_selection` (400) (`/git/commit`), `confirm_required` (400,
  `/api/evaluation/sync`).

### FASRC gate (contract C4)

The console is **offline-first**: every endpoint works with FASRC
disconnected, except the handlers marked with
`@requires_fasrc` (`euclid_polish/web/fasrc_gate.py`; the **Gate** column
below says `fasrc`). While the shared SSH session is down a marked handler is
never entered; the request gets **503**

```json
{"ok": false, "error": "FASRC not connected", "code": "fasrc_offline"}
```

- Mark a handler that needs `STATE.ssh`, `remote.*`, a `fasrc_fetcher` pull,
  rsync or another SSH helper **and has no local fallback**. Decorator order
  does not matter (`@app.route(...)` then `@requires_fasrc`, or the reverse).
- Handlers that degrade gracefully offline (report `connected: false`, serve
  the local cache, or run a job that self-connects via `ensure_ssh_connected`
  and reports failure in the job) stay unmarked and are listed with their
  reason in `GRACEFUL` in `tests/test_fasrc_gate.py`. Two checks there keep
  the marks honest: an AST audit that follows every helper chain across all
  `euclid_polish` modules (calls *and* job targets handed to `spawn`) fails on
  any SSH-reaching handler that is neither marked nor listed, and a runtime
  sweep requests every argument-free GET offline with ssh/rsync/scp blocked
  (marked → the offline 503; unmarked → no SSH attempt, answers promptly).
- Always reachable offline: `GET /api/fasrc/status` (with `last_error`: the
  startup auto-connect error or the last failed connect, `null` after a
  successful connect or a manual disconnect), `GET/POST /api/fasrc/config`,
  `POST /api/fasrc/connect`, `POST /api/fasrc/disconnect`,
  `POST /api/connection/retry` (which answers its own 502 error, never the
  gate's), the jobs and version endpoints, and every local-data endpoint.
- Nothing redirects to a connection-error page any more: a GET of
  `/connection-error` 308s to `/settings/connections` (C1); the classic POST
  form is gone — connect with `POST /api/fasrc/connect` or
  `POST /api/connection/retry`.

### Local background jobs (contract C2)

Long-running work runs in the in-process job registry
(`euclid_polish/web/jobs.py`, `REGISTRY.spawn(label, target, kind=None)`).
An endpoint that starts one returns `{"ok": true, "job_id": "<8 hex>"}` (older
endpoints return just `{"job_id": ...}`); the client then polls
`GET /api/jobs/<job_id>` (or the list).

Job dict:

```jsonc
{
  "job_id": "3f9c2a1b", "label": "FASRC: update conda environment",
  "kind": "fasrc-env-update",          // free-form tag or null
  "status": "running",                 // running | done | failed | cancelled
  "started": 1790000000.0, "finished": null, "duration": 12.3,
  "error": null,                       // message + traceback when failed
  "cancellable": true,                 // running and no cancel requested yet
  "cancel_requested": false,
  "result": null,                      // done only: JSON-safe return value ≤ 64 KB, else null
  "log": "…last 4000 chars…",          // null in ?summary=1 listings
  "log_truncated": false,
  "progress": {"current": 3, "total": 10, "pct": 30.0, "label": "…",
               "stage_elapsed": 4.2, "rate_per_second": 0.7,
               "eta_seconds": 10.0, "updated_ago_seconds": 0.4}
}
```

- **Cancel** is cooperative: `POST /api/jobs/<id>/cancel` flags the job; it
  becomes `cancelled` when its target next calls `cap.tick(...)` (or a tqdm
  update inside `cap.tqdm_hook`), which raise `JobCancelled` (a
  `BaseException`, so a target's `except Exception` cannot swallow it).
  Long targets should tick (or call `cap.check_cancelled()`) regularly.
- The registry keeps at most **200 finished jobs** (oldest evicted); running
  jobs are never evicted. Jobs do not survive a server restart.
- Known kinds: `fasrc-env-update`, `tng-radii`, `real-tile` (cache a 25.6″
  tile, optionally + models), `real-experiment`, `jwst-discover`, `jwst-pair`
  (contract C9).

### Pages and redirects (contract C1)

`euclid_polish/web/spa_routes.json` is the single source of truth for page
URLs (`euclid_polish/web/spa_routes.py`: `load_manifest`, `is_page_path`,
`redirect_target`). A GET/HEAD of a **page path** — a workspace path with its
`:params` substituted, optionally followed by one `/<tab>` from its tabs —
serves `static/dist/index.html` (503 plain-text build hint when the bundle is
missing). A GET/HEAD of a **redirect source** answers **308** to its target
with the query string preserved; `/app/<rest>` → `/<rest>` with every
leading slash, backslash or control character of `<rest>` collapsed into one
`/` (so `/app//evil.example` → `/evil.example`, never the protocol-relative
`//evil.example`: no open redirect). Other methods
never get the shell or a redirect, and non-page paths that share a prefix
with a page (`/ensemble/status.json`, `/inspect/preview.png`) reach their own
handlers.

### Route modules

`euclid_polish/web/routes/__init__.py` exposes `MODULES`, the tuple of route
modules `create_app` registers (each has `register(app)`). Adding a route
group is one import + one tuple entry; `tests/test_route_registry.py` fails
when a `routes/*.py` file is not registered.

### FASRC step task parameters (contract C5)

Every registered FASRC step (`euclid_polish/web/fasrc_pipeline.py`) declares
`task_params`: the knobs its `build_command` reads, as
`{name, type, default, help, min?, max?, choices?, required?}` with `type` ∈
`int | float | str | bool | choice | json`. A `default` of `null` means
"unset" (the step's own fallback / no CLI flag). Resources (`partition`,
`n_cpus`, `n_gpus`, `memory`, `time_limit`) and the knobs `/config` injects
(`job_config.FASRC_STEP_PARAMS`: scene counts, densities, PSF warp, LR
schedule, `vis_pixels` …) are not task params.

- `GET /api/fasrc/steps/status` publishes `task_params` and `last_params` (the
  typed task params of the step's newest `COMPLETED` run in the job log;
  blank values → `null`; `null` when it never completed). Fresh-entropy
  seeds — `ensemble_train.base_seed`, `psf_rotation_pool.seed`,
  `poster_cutout.seed` ("blank = random") — are **always** `null` there: the
  job DB stores the number drawn at submit, and prefilling it would replay
  the previous run's seeds (members are seeded `base_seed + i`).
- `POST /api/fasrc/steps/<id>/submit` fills **absent** task params with their
  defaults and validates present ones (type, range, choices); the first
  invalid one is **400** `{ok:false, error:"<name>: …"}`. An explicitly
  **blank** value (empty or whitespace — what a generic form posts for a
  cleared field) is treated like an absent one and takes the default, except
  where blank has its own documented meaning: a param with a `null` default,
  and the `euclid_query` cuts `magnitude_min`, `magnitude_limit`, `snr_min`
  ("blank = no cut"); those are stored as `""` (unset). A `required` param
  refuses a blank value. So a blank `num_stars` asks for 10,000 stars, never
  200. Before submitting **or queueing**, the route renders the job exactly
  as promotion will (prepare, payload staging, command — no SSH); a spec that
  cannot be built is **400** `{ok:false,error}` and is never enqueued (a
  build failure at promotion would halt the whole queue). The job DB/history
  keep the posted form strings plus the values a step resolves at submit
  (member names, a drawn base seed, the array width, filled defaults …).
  `euclid_query` defaults to the last real catalogue run:
  `num_stars=10000, magnitude_min=18, magnitude_limit=19, snr_min=50`.
- `ensemble_train` also takes the multi-knee knobs `asinh_knees` (CSV of
  e⁻), `output_knee`, `knee_loss` (`plain|balanced`) and `evaluate_every`.

### Viewer collections (contract C6)

Collections (`helpers/viewer_data.py`): `sky`, `cutouts`, `evaluation`,
`ensemble`, `archive-fields`, `real-field`, `jwst-euclid`, `nexus-field`,
`psfs`, `real` (contract C9, see *Real tiles* below).

- **Objects.** Every `meta.objects[i]` has a stable string `id` (`sky` /
  `ensemble`: `"<subset>:<record index>"`; `cutouts`: star id; `evaluation`:
  the object sub-directory; `archive-fields`: sample id; `real-field`:
  `"<field_id>/<tile:03d>"`; `nexus-field`: `"<field_id>/<tile:04d>"`;
  `jwst-euclid`: pair id; `psfs`: `"cluster-NNN"`) and `ra`/`dec` (deg) when
  the object is on the sky and its position is finite (every real
  collection; `cutouts` from the star's `stars.csv` row). Synthetic `sky` /
  `ensemble` records have none. `archive-fields` objects carry the
  position-derived `field` and the manifest's `stored_field`.
- **`?id=` lookup.** `GET /viewer/cube/<collection>?id=<id>&tier=…` serves the
  object whose meta `id` matches (same collection params) instead of a
  positional index; `GET /viewer/meta/<collection>?id=<id>` adds `index` (its
  position). Unknown id → 404 `{"error": "unknown object id: <id>"}`; the
  cube route without `id` → 400. Every cube also sends `X-Cube-Index` (the
  resolved position, exposed).
- **Units.** Tiers may carry `unit` (`"e-"`, `"MJy/sr"`, `"ADU/s"`, `"arb"`)
  in the meta; each cube repeats it as `X-Cube-Unit` (PSF kernels, PCA
  eigen-images and JWST colour composites are `arb`; raw star cutouts are
  `ADU/s`; NEXUS/JWST native tiers are `MJy/sr`).
- **WCS.** `X-Cube-WCS` is compact JSON of the celestial WCS of *that tier's*
  pixel grid — `CTYPE1/2, CRVAL1/2, CRPIX1/2, CD1_1, CD1_2, CD2_1, CD2_2`
  (always the CD form), FITS 1-based convention, axis 1 = column (x), row 0
  of the cube = FITS y = 1. Present for every real tier: `evaluation`
  (`original_stack.fits`), `archive-fields` (VIS HDU), `real-field`
  (`original_stack.fits` shifted by the tile offset), `nexus-field` /
  `jwst-euclid` (the tile / product FITS), `cutouts` (VIS cutout). SR-grid
  tiers (SR, std, PCs, members, combiners) are the LR WCS magnified ×2:
  `CD/2`, `CRPIX → 2·CRPIX − 0.5`. Synthetic tiers (records, HR, PSFs) have
  none. Both headers are listed in `Access-Control-Expose-Headers`. The WCS
  is per cube only: it differs per object *and* tier, and computing it for
  every object at meta time would open every FITS, so `meta.tiers` carries
  no `wcs` (spec §9.4 "+ meta" is deliberately not implemented — a client
  learns a tier's WCS from the first cube it loads).
- **Channels.** FITS cubes are read channel-first, `.npy`/records
  channel-last, so cubes with more than four channels (multi-knee heads) are
  served as `(H, W, C)` with `C` channels (`X-Cube-Bands` = `ch0…` when the
  channels are not the four Euclid bands).
- **Ensemble.** `mode` defaults to `starfull`. Tier `sr` is the production
  combiner (`ACTIVE_COMBINER_KINDS[0]`, the spatial gate; label
  `"SR · production gate"`), offered when it is baked or loads for the cached
  membership; `mean` ("Mean of members") is the cached ensemble mean; the other
  active combiners (RBF kinds) stay as extra tiers when loadable. With a
  member subset (`?members=`), `sr` and `mean` both serve the subset mean.
  `meta.morph_base_tier = "mean"` names the disagreement movie's centre (the
  tier the `pcaN` components are about; a client animates
  `morph_base_tier + Σ amp·pcaK`, falling back to `sr` when the key is
  absent — the SPA viewer engine, `frontend/src/viewer/`, does exactly this);
  `meta.production_combiner` and `meta.regime` are informative.
- **Evaluation movie centre.** `evaluation` has no `mean` tier and no
  `morph_base_tier`, so its movie centres on `SR`. For results written since
  `eval/ensemble_infer.py` switched to the production combiner, `SR.fits` is
  the gate output while `pca*.fits` (`eval/disagreement.py`) are components
  about the member **mean** — a small, known approximation until the eval
  writer also persists the member mean (then expose it as `mean` and set
  `morph_base_tier`).

### Real tiles, model catalogue, experiments (contract C9)

`routes/real.py` over `helpers/{real_tiles,model_catalog,experiments,real_metrics}.py`.
Everything is local (the tile download talks to the public Euclid archive from
a local job): **nothing is FASRC-gated**.

- **Real tile** = four-band Euclid LR (`(H, W, 4)` electrons, bands
  `VIS,Y_E,J_E,H_E`) on a celestial grid, addressed `source/id` (ids never
  contain `/` or `,`). Sources: `nexus` (445 NEXUS × Euclid 255² tiles, JWST
  F200W; id `f200w-NNNN` = source index), `tile` (user-cached 25.6″ tiles,
  `data/euclid_inference/real_tiles/<id>/`: `lr_e.npy` (256, 256, 4) float32,
  `lr.fits` (4, 256, 256) with the VIS WCS, `raw/<band>.fits`,
  `manifest.json`; id `ra…_dec…`), `field` (every legacy 100-tile real field;
  id `<field id>-NNN`), `archive` (220 archive samples, ADU/s → e⁻ via
  MAGZERO; id `NNN`), `eval` (real evaluation objects; id = `out_subdir`),
  `poster` (`poster/*_results.fits`; WCS **constructed** north-up TAN from
  `RA/DEC/PIXSCALE`), `pair` (saved JWST × Euclid pairs; four-band once the
  pair's LR input exists, VIS-only before). Field labels are
  position-derived (`q1_field_for`).
- **Tile entry** (list rows and the card's base):
  `{source, id, ref:"source/id", label, ra, dec, field, shape:[H,W], pixscale,
  bands, model_ready, tiers:["lr", "jwst"?], has_jwst, polygon:[[ra,dec]×4],
  extras{…source specific: legacy_sr, grade, field_id, position_name…}}`. `extras.legacy_sr` is the SR the pre-C9 production
  pipeline wrote for the tile (a *legacy record*, below; `null` when none):
  NEXUS whole-field inference `tiles/starfull_combiner_NNNN.fits`, pair
  inference `starfull_inference/starfull_combiner.fits`, the poster's `SR_*`
  HDUs, the evaluation `SR.fits` (no recorded model: listed, never a tier).
  Every legacy SR that names a C9 spec (NEXUS / pair `inference` + pair
  `model_inference`, poster) is one of the tile's model outputs (the `models`
  rows below); the entry keeps them internally (`extras.legacy_outputs`, not
  serialised).
- **Model specs** (`GET /api/models`): `production` (the production spatial
  gate `spatial_gate_combiner/` fitted for the CURRENT STARFULL members —
  unavailable otherwise, no silent fallback), `mean` (all active STARFULL
  members), `member:member_<N>` (each active member; aliases `member:170`,
  `member:170·psnr`), `gate:<variant>` (every `spatial_gate_<variant>/` beside
  the production artifact, applied with **its own** member labels via
  `eval.spatial_gate.load_spatial_gate`; unavailable, with `reason`, unless all
  its members are active STARFULL members), `rbf` (the RBF combiner, own
  labels). Members always run through `EnsembleModel` by label. A spec's
  `fingerprint` hashes the member checkpoint fingerprints it reads plus (for
  combiners) the artifact's `combiner.json`+`combiner.npz`.
- **Output store**: one SR per (tile, spec) at
  `data/euclid_inference/experiments/outputs/<source>/<id>/<slug>.fits`
  (`(4, 2H, 2W)` electrons, WCS = LR WCS ×2: `CD/2`, `CRPIX → 2·CRPIX − 0.5`)
  + `<slug>.json` `{spec, label, fingerprint, member_labels,
  member_fingerprints, combiner_kind, combiner_fingerprint, lr_sha, shape,
  created, experiment_id, metrics}`; `slug` = spec with `:` → `-`. An output
  is `current` while its fingerprint equals the spec's now, `stale` when not,
  `unavailable` when the spec cannot run now. Member SRs are cached per tile
  under `experiments/cache/<source>/<id>/member_<N>.npy` keyed by checkpoint
  fingerprint + LR hash — **bounded**: ≤ 4 GiB over all tiles
  (`experiments.MEMBER_CACHE_BUDGET_BYTES`), least-recently-used entries
  evicted, and no cache write while the data disk keeps less than
  `MIN_FREE_BYTES` (5 GiB) + the budget free (the SR is then used from memory
  for that tile only).
- **Model outputs of a tile** = the output store merged with the tile's
  **legacy records** (read in place, never copied; the store wins for a spec
  unless only the legacy SR is current): `{spec, slug, kind, label,
  fingerprint, member_labels, member_fingerprints, member_count,
  combiner_kind, combiner_fingerprint, identity{combiner_kind,
  combiner_fingerprint, member_fingerprints}|null, shape, file, path,
  created, origin: "nexus-field"|"pair"|"poster", legacy: true,
  experiment_id: null, lr_sha: null}`. Its spec is the recorded `spec`
  (records since C9) else the `combiner_kind` (`spatial_gate` → `production`,
  `raw_incremental_minmeanmax_rbf` → `rbf`, poster `mean_explicit_members` →
  `mean`); its fingerprint is the recorded `spec_fingerprint`, else rebuilt
  from the recorded identity with the catalogue formula
  (`model_catalog.spec_fingerprint`), so its state is decided exactly like a
  store output's (the poster records no identity: never `current`). A legacy
  SR is always served on the LR WCS ×2 (`CRPIX → 2·CRPIX − 0.5`), never its
  file's own header (older NEXUS SR files carry `2·CRPIX − 1`). The 445 cached
  NEXUS SRs (RBF era) therefore appear as `m:rbf` (current while the RBF
  artifact + members are unchanged) and the tiles' `production_state` is
  `stale` until the production gate runs on them.
- **Production state** of a tile (`production_state`): `current` (its
  `production` output — store or legacy — carries the production fingerprint
  now), `stale` (that output is older, or only a legacy SR of the production
  pipeline exists, e.g. the RBF-era NEXUS SRs), `missing`.
- **Metrics** (`real_metrics.tile_metrics`, per band; version 1):
  `hole_pct` = % of the SR pixels under the brightest 1 % of LR pixels with
  `SR < 0.5 × LR/4` (`hole_pct_100sigma`, `n_bright_100sigma_px`: the same over
  the bright-1 % pixels that are also > 100 σ — on faint tiles the top 1 % is
  noise); enclosed-flux R around bright (> 100 σ, σ = 1.4826·MAD,
  background = median), locally dominant (brightest within ±1.5″) LR peaks
  that pass the central-pixel-fraction cut `F(1px)/F(3×3) ≤ 0.25` (VIS) /
  `≤ 0.14` (NISP): per peak `R = min over odd boxes 3–17 LR px (0.3–1.7″) of
  F_SR/F_LR` (background-subtracted); `n_peaks, n_artifacts, n_edge,
  pct_R_lt_0p8, pct_R_lt_0p5, median_R, min_R`; `flux_ratio = ΣSR/ΣLR`,
  `lr_flux_e, sr_flux_e, background_e, sigma_e, n_bright_px`; undefined →
  `null`. `peaks[band] = [[x, y, peak_e, cpf, R], …]` (LR px, ≤ 200). Gate
  specs add `gate_core_weights[band] = [[member label, mean weight], …]` (top
  3 over the brightest-1 % SR pixels). `summary` pools bands; experiment
  `summary[spec]` pools tiles (`real_metrics.aggregate`: holes weighted by
  bright pixels, R over the union of peaks).
- **Experiment record** (`GET /api/experiments/<id>`, id
  `YYYYMMDD-HHMMSS-xxxxxx`): `{version, id, label, created, finished,
  duration_s, status: running|done|failed|cancelled, job_id, tiles:["src/id"],
  models, skipped:{spec: reason}, fingerprints:{spec}, model_labels,
  definitions, results:{"src/id":{spec:{state: computed|reused, fingerprint,
  file, metrics:{per_band, summary, gate_core_weights?}}}}, summary:{spec:
  aggregate}, errors:{"src/id|spec": msg}, counts:{members_computed,
  members_reused, members_not_cached, members_evicted, outputs_computed,
  outputs_reused}}` — written progressively. One experiment computes at a
  time; a second waits (cancellable). A current output is `reused` without a
  model run: a store output with current metrics as is; a store output
  without metrics, or a current legacy SR, is scored (a legacy SR is then
  copied into the store with its metrics, `from_legacy` = its path). The
  member runner restores only the registry prefix holding the members of the
  specs that will actually run — a spec already current on every tile adds
  none — (`EnsembleModel(n_members=…)`; there is no arbitrary-subset option,
  and a request past the prefix reloads uncapped once).
- **Viewer collection `real`**: `GET /viewer/meta/real?source=<source>&models=<spec,…>`
  (default models: every spec with an output in that source). Tiers `lr`
  (e⁻, LR WCS), `jwst` (native, MJy/sr, its own WCS; `?jwst_band=` picks a
  pair filter) when any tile has JWST, and `m:<spec>` (e⁻, SR WCS; label gets
  ` · legacy` for a legacy SR and ` · stale` when not current; cube meta adds
  `model_state`, `legacy`; 404 "has not been run" when missing). Objects:
  `{id, label, ra, dec, field, ref, tiers, model_states:{spec: state},
  legacy_models:[spec…], model_ready}`; default models = every spec with an
  output (store or legacy) in the source; `?id=` works as for every
  collection.

| Methods | Path | Gate | Notes |
|---|---|---|---|
| GET, POST | `/api/experiments` |  | GET: `{experiments:[{id, label, created, finished, status, job_id, tiles, models, skipped, summary, errors, counts}]}` newest first. POST: start an experiment (local job `real-experiment`): `tiles` = comma list of `source/id`, `models` = comma list of specs, `label?`. `{ok, job_id, experiment_id, tiles, models (runnable), skipped:{spec: reason}}`. 400 bad/unknown spec (`unknown model spec 'gate:x'`) or no runnable model; 404 unknown tile; 409 tile without four-band LR; **507** `{ok:false, code:"insufficient_storage", needed_bytes, free_bytes}` when the new outputs (`(4, 2H, 2W)` float32 per (tile, spec) not yet in the store) would leave < 5 GiB free. Job result: `{experiment_id, status, tiles, models, errors, counts}`. |
| GET | `/api/experiments/<experiment_id>` |  | The experiment record (above); 404 unknown. |
| GET | `/api/models` |  | `{regime:"starfull", production_kind, members:[labels], models:[{spec, kind, label, slug, members, member_names, reads, n_members, available, reason, fingerprint, member_fingerprints, combiner_kind, combiner_fingerprint, details{mix_space, use_lr, width, active_members, fitted_at, artifact_dir, loss, loss_knees_e, steps, …}}]}` — order: production, mean, rbf, members, gate variants. |
| GET | `/api/real/<source>` |  | `{source, label, description, count, tiles:[tile entry + models:{spec:{state, legacy, label, fingerprint, created, experiment_id, file, origin, summary}} + production_state]}` (models = the tile's merged outputs). 404 unknown source. |
| GET | `/api/real/<source>/<identifier>` |  | Card: tile entry + `models:{spec:{state, legacy, label, kind, fingerprint, created, experiment_id, file, member_labels, combiner_kind, lr_sha, shape, origin, metrics{per_band, summary, gate_core_weights?}, image_url}}` (store + legacy outputs), `production_state`, `legacy` (= `extras.legacy_sr`: the pre-C9 production-pipeline SR record or `null`), `runnable_models`, `experiments:[ids]`, `disk{tile_bytes, output_bytes, cache_bytes (member-SR cache), legacy_bytes (NEXUS / pair legacy SR files), total_bytes}`, `q1_tile` (the containing Q1 tile), `image_urls{tier: url}` (one per `m:<spec>` with an output), `viewer{collection:"real", params{source}, id}`. |
| POST | `/api/real/<source>/<identifier>/delete-outputs` |  | Delete the tile's cached model outputs and member-SR cache (never the LR itself): `{ok, ref, removed:[paths], removed_count, cache_bytes_freed}`. |
| GET | `/api/real/<source>/<identifier>/image.fits` |  | 2-D float32 FITS of one plane with its celestial WCS (CD form) for sky overlays: `tier` = `lr` (default) \| `jwst` \| `m:<spec>` (store output or legacy SR; SR WCS = LR WCS ×2), `band` = `VIS` (default) \| `Y_E` \| `J_E` \| `H_E` (JWST: its filter, default the first). Header `BUNIT` (`electron` / `MJy/sr`), `TIER`, `BAND`, `REALTILE` (+ `LEGACYSR` for a legacy SR). 400 bad tier/band; 404 missing tier. |
| GET | `/api/real/sources` |  | `{sources:[{id, label, description, count, model_ready, has_jwst, ready, reason}]}` for `nexus, tile, field, archive, eval, poster, pair`. |
| POST | `/api/real/tiles` |  | Cache a 25.6″ four-band tile at `ra`, `dec` (local job `real-tile`; Euclid archive, not FASRC): Q1 coverage is checked first against the committed MER polygons (400 `{ok:false, code:"outside_q1"}`; 400 `{ok:false, code:"unobserved_q1", tile}` when every containing tile is `rejected` — measured unobserved by the noise campaign — unless `force=1`); each band is cut from the Q1 tile whose polygon CONTAINS the point (observed tiles first, deepest inside), VIS cropped to the exact 256² grid, NISP registered onto its WCS, electrons via MAGZERO. Optional `run=production,mean` then runs an experiment on it. `{ok, job_id, id, ref:"tile/<id>", experiment_id|null}`; job result `{tile, ref, experiment?}`. |

### Sky atlas (contract C9, `routes/sky_atlas.py`)

`helpers/sky_atlas.py`; every layer is built from local files (memoised on
their mtimes, ≤ 30 s) and works offline. Layer groups: `coverage`
(`q1-tiles` 352 MER polygons with `levels_e`/`rejected`/`state`, `q1-fields`
cones, `nexus-footprint` hull of the NEXUS tile cells), `results`
(`nexus-tiles`, `real-tiles`, `real-fields`, `poster`, `pairs`,
`archive-fields` — polygons with `state` = production state; `eval-objects`,
`experiments` — points), `catalogues` (`lens-candidates`, `galaxies`,
`stars` (the FASRC-mirror `stars.csv`, 43k rows `[ra, dec, mag, flags]`,
`flags` bit `2^b` = valid cutout in band `b` ∈ VIS,Y_E,J_E,H_E), `psf-clusters`,
`noise-positions`, `population-cones`, `gaia-fields`, `jwst-mast`).

Feature shapes (`GET /api/sky/layer/<id>`, always with `id, label, group, kind, count`):
`points` → `{columns:[…], rows:[[ra, dec, …]], inspect:{kind, prefix, id_column}}`
(row inspector entity `{kind}:{prefix}{row[id_column] or row index}`);
`polygons` → `{features:[{id, polygon:[[ra,dec],…], props, inspect:{kind, id}}]}`;
`circles` → `{features:[{id, ra, dec, radius_deg, props, inspect}]}`. Real
tiles inspect as `realtile:<source>/<id>`, others as `source:<layer>/<id>`.

| Methods | Path | Gate | Notes |
|---|---|---|---|
| GET | `/api/sky/at` |  | What covers `ra`, `dec`: `{ra, dec, field, in_q1, q1_observed (a containing tile is not rejected), q1_verdict: "observed"\|"unobserved"\|"outside", q1_tiles:[tile (incl. rejected) + margin_arcsec] (observed tiles first, each deepest first), best_tile, real_tiles:[{source, id, ref, label, has_jwst, state, inspect}], nexus, pairs, jwst:[discovered footprints containing the point], jwst_discovered}`. 400 bad/missing coordinates. |
| POST | `/api/sky/jwst/discover` |  | JWST × Euclid discovery (local job `jwst-discover`; MAST, not FASRC): wraps `scripts/find_jwst_euclid_overlap.py` (the same MAST cone query and cache in `data/jwst_euclid_overlap/mast/` — `jwst_euclid._mast_rows_for_scope`, astroquery imported at module top — plus the script's direct-imaging + public filters and CSV rows) with the exact test MAST `s_region` ∩ committed Q1 polygon done locally. Scope: `fields` (comma list: `EDF-N,EDF-S,EDF-F,LDN1641`) and/or `region=ra,dec,radius_deg` (≤ 10°); both empty = all Q1; `refresh=1` ignores the MAST cache. Results MERGE into `overlap.csv`/`overlap.json` (pairing input) and `jwst_footprints.json`. `{ok, job_id, tile_count, fields, region}`; job result = the discovery manifest + `footprint_count`. |
| GET | `/api/sky/jwst/footprints` |  | Cached MAST footprints near `ra`, `dec` within `r` deg (default 0.5, ≤ 5): `{ra, dec, r, ready, updated_utc, fields, count, truncated, footprints:[{obs_id, instrument, filters, target, proposal_id, exptime_s, ra, dec, polygons:[[[ra,dec],…]], status, euclid_tiles, fields}]}` (nearest first, ≤ 2000). |
| POST | `/api/sky/jwst/pair` |  | Download + align one JWST × Euclid pair, then build its four-band LR input (local job `jwst-pair`): `obs_id` (a discovered MAST observation → its location group) or `ra`, `dec` (inside one of the NEXUS × Euclid tile cells — their polygons, not the hull → a NEXUS cutout, `filter` F200W\|F444W; else the nearest discovered location within max(size/2, 15″)); `size_arcsec` (1–120, default 30); optional `run=<specs>`. 404 `code:"not_discovered"` when nothing covers it. `{ok, job_id, pair_id, ref:"pair/<id>", mode:"nexus"\|"archive"}`. The Euclid VIS box comes from the committed Q1 tiles containing the point (observed first, deepest inside first; the archive INTERSECTS search only outside every committed polygon) and must be ≥ 99.5 % observed, else the job fails (no partial pair is published); the pair manifest records `euclid_vis_tile_index` and `euclid_selection{method: q1_polygon\|archive_intersects, tile, coverage, tried}`. The pair becomes real source `pair`. |
| GET | `/api/sky/layer/<layer_id>` |  | One layer's features (shapes above); 404 unknown layer. |
| GET | `/api/sky/layers` |  | `{groups, layers:[{id, label, group, kind, count, bbox{ra_min, ra_max, dec_min, dec_max}\|null, style{color?, color_by?, colors?, shape?, size?, opacity?}, ready, reason, fill_action{method, url, label, requires_fasrc?}\|null, description, url}]}`. |

### Removed with the classic console (WP-B1b)

The Jinja templates, classic static JS/CSS and these routes are gone (the SPA
serves every page URL; see C1): page handlers `/`, `/catalog`, `/sky`,
`/config`, `/cutouts`, `/cutouts/<band>`, `/ensemble`, `/fasrc`, `/git`,
`/inference`, `/training`, `/psfs`, `/tng`, `/tracking`, `/visualization`,
`/inspect`, `/evaluation`, `/connection-error` (form); unreferenced
`/ensemble/render`, `/ensemble/eval-plot/<plot>.png`, `/view/star-cutout`,
`/api/jwst-euclid/saved`, `/api/jwst-euclid/nexus/options`,
`/api/jwst-euclid/field/<id>/<kind>`, `/api/fasrc/eta`, `/api/fasrc/jobs`,
`/api/fasrc/mirror/start|stop`, `/api/fasrc/runs/ckpt-bundle.tar`;
superseded `/ensemble/power-spectrum.png` (client-side from `evals.json`),
`/api/euclid-psf/preview` (viewer `psfs`), `/api/sky/totals`,
the PNG renderer of `/eval-files/<path>` (viewer `evaluation`; the route
keeps only its FITS download),
`/api/fasrc/runs/training-plot.png` (`training-curve.json`),
`/api/fasrc/training-status`, `/api/fasrc/log/<jobid>` (`/api/fasrc/runs/log`),
`/api/fasrc/stages/<jobid>`, `/api/fasrc/submit`
(`/api/fasrc/steps/<id>/submit`; a spec it left in the local queue still
promotes as `synthetic_generate`), `/star-cutout/inspect`, `/sky/inspect`,
`/sky/fits`. `FasrcConfig` no longer carries science knobs (`n_train`,
`n_valid`, `n_test`, `image_size`, `batch_size`, `steps`).

## Endpoints

Gate `fasrc` = `@requires_fasrc` (503 `fasrc_offline` while disconnected).
"Local job" = returns a `job_id` for the jobs API above. Paths use Flask rule
syntax. Flask's own `/static/<path:filename>` is omitted.

### Platform (`app.py`)

| Methods | Path | Gate | Notes |
|---|---|---|---|
| POST | `/api/connection/retry` |  | Retry the startup auto-connect. `{ok:true}` or 502 `{ok:false,error}`; the error is also kept as `last_error`. Works offline (never gated). |

### Jobs, version, files, inspector (`routes/files.py`)

| Methods | Path | Gate | Notes |
|---|---|---|---|
| GET | `/api/inspect` |  | FITS inspector payload (header rows, HDU info) for `?fits=<project-relative path>`. |
| GET | `/api/jobs` |  | Local background jobs, newest first (C2). `?summary=1` → same with `log: null`. |
| GET | `/api/jobs/<job_id>` |  | One job dict with the full log tail (C2); JSON 404 `{ok:false,error}` when unknown. |
| POST | `/api/jobs/<job_id>/cancel` |  | Cooperative cancel (C2): `{ok:true}`; the job turns `cancelled` at its next `cap.tick`. 404 unknown, 409 already finished. |
| GET | `/api/status` |  | Local status summary `{catalog, psfs, tfrecords, checkpoints}` — cache-only and cheap: no SSH, no rsync (`catalog.cached: true`, the last synchronised `stars.csv`). |
| POST | `/api/status/refresh-catalog` | fasrc | Explicitly re-pull the FASRC `stars.csv` (forced rsync): `{ok:true, catalog}`. |
| GET | `/api/version` |  | Server version (C3): boot vs HEAD commit, `behind`, `dirty`, `started_at`, `pid`, `dist{built_at,index_hash}`. Safe to poll: the git probes are read-only (`--no-optional-locks`), so they never take `.git/index.lock` from a concurrent commit. |
| GET | `/fasrc/file/download` | fasrc | Fetch one FASRC file (cached) and send it. |
| GET | `/fasrc/file/inspect` | fasrc | Fetch one FASRC file (`?remote_path=`, cached) and 302 to `/inspect?fits=<project-relative path>`; a failed fetch is JSON 502 `{ok:false,error}`. |
| GET | `/inference-files/<path:relpath>` |  | Serve FITS/PNG from `data/euclid_inference/` (jailed). |
| GET | `/inspect/download` |  | Download the inspected FITS (`?fits=`), jailed to project data roots. |
| GET | `/inspect/preview.png` |  | PNG preview of the inspected FITS (`?fits=`, `size` 16–2048). |
| GET | `/vis/<path:relpath>` |  | Serve a PNG from `data/vis/` (jailed; 403 outside, 404 missing). |

### FASRC connection and cluster (`routes/fasrc.py`)

| Methods | Path | Gate | Notes |
|---|---|---|---|
| POST | `/api/fasrc/bootstrap-data` | fasrc | Re-create the netscratch → holylabs data symlinks on FASRC. |
| POST | `/api/fasrc/cancel` | fasrc | `scancel` one SLURM job (`jobid`). |
| GET, POST | `/api/fasrc/config` |  | GET the FASRC connection settings; POST (form) patches them. Works offline. |
| POST | `/api/fasrc/connect` |  | Open the SSH ControlMaster from the saved settings. `{ok:true,status}` or 400 `{ok:false,error,status}`; failures are kept as `last_error`. Works offline. |
| GET | `/api/fasrc/current-submission` | fasrc | Newest PENDING/RUNNING submission (`current: {job, status, array, accounting}` or `null`) reconciled against `squeue`, the local `queue`, `stale`, and `live`: **every** PENDING/RUNNING job (newest first, same row shape as `current.job`: DB row + squeue `start_time/reason/nodes/time/time_limit`) — C5. Also advances the local submission queue. |
| GET | `/api/fasrc/data-listing` | fasrc | Sizes and entries of the FASRC data directories. |
| POST | `/api/fasrc/disconnect` |  | Close the session, stop the checkpoint mirror, clear `last_error`. |
| POST | `/api/fasrc/env-update` | fasrc | Start `yes \| mamba env update` on FASRC as a local job (`kind="fasrc-env-update"`): `{ok:true, job_id}`; the remote output streams into the job log; the job ends `done` with `result={exit_code, lines}` or `failed` on a non-zero exit. Cancellable: the remote command prints a filtered heartbeat every 2 s, so a cancel lands within one heartbeat even while mamba is silent; the remote side (no pty, so no SIGHUP) is then killed by its heartbeat watchdog when the closed channel makes a write fail. A cancel can leave the env half-updated; re-run to finish. POST-only (was a GET SSE stream). |
| POST | `/api/fasrc/git-pull` | fasrc | `git pull` on FASRC; `env_update_needed` when `environment.yml` changed (the UI then starts `POST /api/fasrc/env-update`). |
| GET | `/api/fasrc/git-status` | fasrc | Branch / HEAD / dirty state of the FASRC checkout. |
| GET | `/api/fasrc/jobs/<jobid>/status` |  | Structured status from one job's `.events` stream; an empty status offline (never gated). |
| GET | `/api/fasrc/mirror/status` |  | Checkpoint-mirror state (local). |
| POST | `/api/fasrc/mirror/trigger` | fasrc | One-shot checkpoint rsync FASRC → local mirror. |
| GET | `/api/fasrc/queue` | fasrc | The user's live `squeue` rows. |
| POST | `/api/fasrc/queue/clear` |  | Clear the local submission queue. |
| POST | `/api/fasrc/queue/remove` |  | Remove one queued submission (`id`). |
| POST | `/api/fasrc/refresh-accounting` | fasrc | Re-pull `sacct` for every finalised job and re-record it. |
| GET | `/api/fasrc/runs` | fasrc | Log files of recent runs on FASRC (reconciled against `squeue`). |
| GET | `/api/fasrc/runs/log` | fasrc | Tail of one FASRC log file (`path`, `lines`). |
| GET | `/api/fasrc/runs/training-curve.json` | fasrc | Per-step training records for one run's wall-time window (JSON, ≤ ~600 points). |
| GET | `/api/fasrc/status` |  | Connection state `{ssh_connected, connected_at, socket, last_error}` (C4). `last_error` = startup auto-connect error or last failed connect; null after a successful connect or manual disconnect. |
| GET, POST | `/api/fasrc/steps/<step_id>/history` |  | Per-step run history + best-match prefill suggestion (local job DB). |
| POST | `/api/fasrc/steps/<step_id>/submit` | fasrc | Submit (or queue behind the running job) one pipeline step: `confirm=yes`, resources (`n_cpus`, `n_gpus`, `memory`, `time_limit`; partition is fixed per step) and task params. Absent (and blank, unless blank means "unset") task params take the schema defaults; an invalid one, or a spec that cannot be rendered, is refused **400** `{ok:false,error}` before anything reaches FASRC or the queue (C5, see *FASRC step task parameters*). `{ok, jobid}` or `{ok, queued:true, queue}`. |
| GET | `/api/fasrc/steps/status` |  | `{ssh_connected, steps[], artifacts, remote_paths}`; each step: `step_id, label, needs_gpu, fixed_cpus, fixed_gpus, defaults` (resources), `task_params` (schema) and `last_params` (typed task params of the newest COMPLETED run, or `null`) — C5. Offline it skips the artifact probes (never gated). |

### Euclid archive auth (`routes/auth.py`)

| Methods | Path | Gate | Notes |
|---|---|---|---|
| POST | `/auth/login` |  | Log in to the Euclid archive (laptop-side session; `username`, `password`). |
| POST | `/auth/logout` |  | Log out of the Euclid archive session. |
| GET | `/auth/status` |  | `{authenticated, user}` of the laptop-side Euclid archive session. |
| POST | `/euclid-auth/save` | fasrc | Write Euclid archive credentials to `~/.euclid_credentials` on FASRC. |
| GET | `/euclid-auth/status` |  | Whether a credentials file exists on FASRC (username only); `connected:false` offline (never gated). |

### TNG (`routes/tng.py`)

| Methods | Path | Gate | Notes |
|---|---|---|---|
| GET | `/api/tng/radii/status` |  | Read-only (never starts a job): the last TNG radius-manifest validation from the local cache — the validator payload (`valid`, `expected_count`, `valid_count`, `reasons`, `checked_at`, `failed?`) + `cached`, `stale` (cache missing, > 1 h old, or a failure > 5 min old), `connected`, `refresh_job` (id of a running validation job, else null). Clients POST `/api/tng/radii/refresh` when `stale && connected && !refresh_job`. |
| POST | `/api/tng/radii/refresh` | fasrc | Re-validate the remote manifest now (local job, `kind="tng-radii"`, at most one at a time — a running one's id is returned): `{ok, job_id}`; the result lands in the status cache (failures too, with `failed: true`). |
| POST | `/tng-auth/save` | fasrc | Write the TNG API token to FASRC. |
| GET | `/tng-auth/status` |  | Whether a TNG token file exists on FASRC (presence + length only); `connected:false` offline. |
| GET | `/tng/histograms.png` |  | TNG property histograms of the downloaded galaxies (FASRC ids/API key optional). |
| GET | `/tng/result/grid.png` | fasrc | Latest `tng_grid` job artifact, pulled from FASRC. |
| GET | `/tng/result/stack.fits` | fasrc | Latest `tng_stack` job artifact (FITS download), pulled from FASRC. |

### Job config (`routes/config.py`)

| Methods | Path | Gate | Notes |
|---|---|---|---|
| GET | `/api/config` |  | The universal job config (`~/.euclid_polish/job_config.json`): `{ok, config, version}` — `version` is a content hash of the effective config. |
| POST | `/api/config/save` |  | Persist ONLY the posted job-config fields (unknown keys and blanks ignored) → `{ok, config, version, note}`. With `base_version` (the `version` the client loaded) a posted field changed server-side since then is refused **409** `{ok:false, code:"config_conflict", conflicts:{field:{base,current}}, config, version}`; fields the server did not change merge. An unknown `base_version` conflicts on every posted field whose value differs now. Without `base_version`: last write wins (legacy). |

### Local git (`routes/git.py`)

| Methods | Path | Gate | Notes |
|---|---|---|---|
| GET | `/api/git/diff` |  | Local working-tree diff (`?staged=1` for the index). |
| GET | `/api/git/status` |  | `{status, log}`: local repo status + last 15 commits. `status.files` is `[{xy, path, orig}]` from NUL-separated porcelain v1: `path` is the raw (unquoted) path — the **new** path of a rename, whose source is `orig` (else `null`); a wholly untracked directory is one `dir/` entry. Every listed `path` can be posted back to `/git/commit` unchanged. |
| POST | `/git/commit` |  | Commit in the local repo: `message` + `paths` (repeatable; one value may hold several newline-separated paths; files or directories, taken literally — commas, spaces and non-ASCII are part of a name, exactly as `/api/git/status` lists them) **or** `all=1` — never an implicit `git add -A`. Exactly the selection is committed (`git commit --only -- <selection>`, literal pathspecs): a file staged earlier but outside `paths` stays staged and is **not** committed; with `all=1` every changed file, staged or not, is the selection and the fully staged index is committed as is (this also concludes a merge; a `paths` commit is refused by git mid-merge); a staged rename brings its source path. Files > 10 MB and new (untracked or staged-new) `.fits/.npy/.zip/.jpg/.png` > 1 MB are refused **409** `{ok:false, code:"refused_files", refused:[{path,size,reason}]}` unless `force=1`. 400 `no_selection` without paths/all; 400 `nothing_selected` when no changed file matches. `{ok, stdout, committed:[paths]}`. |
| POST | `/git/fetch` |  | `git fetch` in the local repo. |
| POST | `/git/pull` |  | `git pull` in the local repo. |
| POST | `/git/push` |  | `git push` from the local repo. |

### Tracking / time-travel (`routes/tracking.py`)

| Methods | Path | Gate | Notes |
|---|---|---|---|
| POST | `/api/tracking/backup` |  | Back up a model / file into the active campaign; best-effort push. |
| POST | `/api/tracking/log` |  | Append to / replace the active campaign's log (`text`, `mode`). |
| POST | `/api/tracking/new` |  | Create a campaign (`title`, `description`). |
| POST | `/api/tracking/save` |  | Archive the active campaign; best-effort holylabs push (`sync`). |
| GET | `/api/tracking/state` |  | Tracking store state (active + archived campaigns with model backups, `ssh_connected`). |
| POST | `/api/tracking/sync` | fasrc | Push the tracking store to holylabs (400 with the sync error). |
| POST | `/api/tracking/timetravel/open` |  | Start a time-travel sandbox server for a backup (`short`). |
| POST | `/api/tracking/timetravel/remove` |  | Remove a time-travel sandbox (`short`). |
| POST | `/api/tracking/timetravel/restore` |  | Create a sandbox worktree at a backup's commit; the remote half (`remote=1`) is optional (never gated). |
| POST | `/api/tracking/timetravel/stop` |  | Stop a time-travel server (`short`). |

### Cutouts (`routes/cutouts.py`)

| Methods | Path | Gate | Notes |
|---|---|---|---|
| GET | `/api/cutouts/<band_name>/list.json` |  | Paginated per-band cutout filenames for the gallery. |
| GET | `/api/star-cutouts/totals` | fasrc | Count/size of stars valid in all four bands; re-pulls the FASRC catalog (no offline fallback yet). |
| GET | `/cutout-image/<band_name>/<path:filename>` |  | One cutout FITS rendered as PNG (`size` 16–2048). |

### PSFs (`routes/psfs.py`)

| Methods | Path | Gate | Notes |
|---|---|---|---|
| POST | `/api/euclid-psf/sync` | fasrc | Force a re-rsync of the four band ePSFs from FASRC. |
| POST | `/api/euclid-psf/sync-meta` | fasrc | Sync only the per-cluster ePSF metadata from FASRC. |

### Sky records, figures, diagnostics (`routes/views.py`)

| Methods | Path | Gate | Notes |
|---|---|---|---|
| POST | `/api/sky/generate-sr` |  | Run SR over the local dirty records (local job): STARFULL members through the production combiner (member mean when no current combiner loads); `overwrite=1` regenerates. |
| GET | `/api/sky/sr-status` |  | State of the records' SR tier (local). |
| POST | `/api/sky/sync` | fasrc | Rsync the synthetic TFRecord shards from FASRC into the local cache. |
| GET | `/api/vis/list.json` |  | The `data/vis/` PNG gallery (newest first). |
| GET | `/view/catalog` |  | Catalog diagnostic PNG (`?view=`) from the cached FASRC catalog. |
| GET | `/view/psf-clusters` |  | ePSF cluster sky-map PNG (local cache; 404 when absent). |
| GET | `/view/psfs` |  | ePSF panel PNG (`?band=`, local cache). |
| GET | `/view/training-log` |  | Training-log PNG of one checkpoint dir (default: first active member). |

### Noise (`routes/noise.py`)

| Methods | Path | Gate | Notes |
|---|---|---|---|
| GET | `/api/noise` |  | Q1 noise-level table payload (committed data). |

### Galaxy distributions (`routes/galaxy_distributions.py`)

| Methods | Path | Gate | Notes |
|---|---|---|---|
| GET | `/api/galaxy-distributions` |  | Galaxy population payload: distributions, availability, auth state, Q1 count/radius state (`?include_training=`). |
| POST | `/api/galaxy-distributions/activate` |  | Activate the fitted joint galaxy candidate (local job). |
| POST | `/api/galaxy-distributions/build` |  | Build the galaxy plot data (local job). |
| POST | `/api/galaxy-distributions/fit-q1-counts` |  | Fit the cached Q1 aperture counts (local job; 400 until queried). |
| GET | `/api/galaxy-distributions/joint-pair` |  | One pair-explorer grid (`x`, `y`). |
| POST | `/api/galaxy-distributions/query-q1-counts` |  | Query Q1 MER + PHZ counts (Euclid archive login required; local job). |
| POST | `/api/galaxy-distributions/refresh-population-cones` |  | Re-query the saved 24-cone population footprint (local job). |
| GET | `/view/galaxy-distribution-plate` |  | Download the four-panel galaxy population diagnostic. |
| GET | `/view/population-atlas` |  | Download the reviewed Euclid brightness–radius fit. |

### Star distribution (`routes/star_distribution.py`)

| Methods | Path | Gate | Notes |
|---|---|---|---|
| GET | `/api/star-distribution` |  | Stellar population payload: colour sample, calibration, distribution, availability. |
| POST | `/api/star-distribution/activate` |  | Activate the fitted stellar candidate (local job). |
| POST | `/api/star-distribution/fit` |  | Fit the stellar prior from the cached colour sample (local job). |
| POST | `/api/star-distribution/query` |  | Query stars (MER + PHZ + Gaia; Euclid archive login required; local job). |
| GET | `/view/star-population-calibration` |  | Gaia–Euclid stellar-prior diagnostic PNG. |

### Population comparison (`routes/population_comparison.py`)

| Methods | Path | Gate | Notes |
|---|---|---|---|
| GET | `/api/population-comparison` |  | Pixel-statistics comparison payload (`?include_training=`). |
| POST | `/api/population-comparison/build` |  | Build the local-field comparison (local job). |
| POST | `/api/population-comparison/sync-training-catalog` |  | Pull `sources_train.csv` from FASRC in a local job that self-connects (`ensure_ssh_connected`) and reports failure in the job (never gated). |

### Archive fields (`routes/archive_fields.py`)

| Methods | Path | Gate | Notes |
|---|---|---|---|
| GET | `/api/archive-fields` |  | Archive-field collection availability (local). `fields` / `comparison_fields` count samples by the **position-derived** Q1 field (`q1_field_for`; the stored EDF-F/EDF-S labels were swapped); `stored_fields` keeps the manifest labels. |
| POST | `/api/archive-fields/sync` |  | Sync the archive-field collection from FASRC in a local job that self-connects and reports failure (never gated); 409 while one runs. |

### Real-field inference (`routes/model.py`)

| Methods | Path | Gate | Notes |
|---|---|---|---|
| GET | `/api/inference/diagnostics.json` |  | Diagnostics of the latest cached real field. |
| GET | `/api/inference/field.json` |  | Latest cached real field + field size. |
| POST | `/inference/cache-real-field` |  | Cache a real Euclid field at (`ra`, `dec`) (local job, Euclid archive). |
| POST | `/inference/refresh-combiners` |  | Apply the newest STARFULL combiner to cached fields (local job). |

### Ensemble (`routes/ensemble.py`)

| Methods | Path | Gate | Notes |
|---|---|---|---|
| POST | `/ensemble/archive-member` |  | Retire one member: zip → tracking campaign, registry tombstone, member dir deleted, cube cache purged. |
| GET | `/ensemble/combiner.json` |  | The Combiner card's dataset for a regime (``?mode=``): per-band effective-weight curves, survivors, val loss and per-member meta (loss/depth/PSNR — the facets the gate plot colors by). |
| POST | `/ensemble/combiner/fit` |  | Fit the combiner for the requested star regime locally on the validate split. |
| GET | `/ensemble/evals.json` |  | The Evaluations card's dataset: power-spectrum curves, diagnostic histograms, calibration stats and per-member loss/depth meta. |
| POST | `/ensemble/evaluate` |  | Evaluate the ensemble on local test records (local job; `num_images`, `mode` = `starfull` (default) or `starless`). |
| POST | `/ensemble/knee-psnr` |  | Compute PSNR-vs-knee curves (local job; `mode`). |
| GET | `/ensemble/knee-psnr.json` |  | PSNR-vs-knee curves + integrated PSNR for every model of a regime (``?mode=``), flagged ``stale`` when the cubes or combiners changed. |
| POST | `/ensemble/member-psnr` |  | Refresh the members table's test PSNRs (asinh space). |
| GET | `/ensemble/pixel-trace.json` |  | Back-trace a diagnostic heatmap cell to real image stamps. |
| POST | `/ensemble/pull` | fasrc | Download the trained members from FASRC (local job). |
| GET | `/ensemble/status.json` |  | Members table + summary payload; `?mode=` picks the regime's eval summary + staleness (default `starfull`). Every `/ensemble/*` route defaults `mode` to STARFULL. |
| GET | `/ensemble/training-curves.json` |  | `{members:[{name, psnr:[[step,dB]…], loss_series:[[step,loss]…], loss_norm, loss (= loss_norm, compat), blocks, test_psnr, asinh_knee, starless}]}` — registry-active members only (rollback-deduped). |

### Evaluation (`routes/evaluation.py`)

| Methods | Path | Gate | Notes |
|---|---|---|---|
| GET | `/api/evaluation/angular-power-spectrum` |  | Render + serve the per-band HR-vs-SR angular power-spectrum PNG. |
| POST | `/api/evaluation/fetch-catalog` |  | Download + normalize the Euclid Q1 strong-lens catalog (Zenodo). |
| POST | `/api/evaluation/query-galaxies` |  | Query + cache the real-galaxy eval catalog as its own LOCAL step. |
| POST | `/api/evaluation/rerender` |  | Drop a run's cached eye/solar PNGs so they re-render from the FITS. |
| POST | `/api/evaluation/run-grouped` |  | Prepare the unified grouped dataset LOCALLY (A/B/C + synthetic) with the STARFULL members through the production combiner (member mean when no current combiner loads). |
| GET | `/api/evaluation/runs` |  | Summary of one evaluation run (`?run=`). |
| POST | `/api/evaluation/sync` | fasrc | Pull `<data_dir>/eval_results` from FASRC (`rsync --delete-after`, which also deletes local results the cluster lacks): requires `confirm=1`, else **400** `confirm_required`. |
| GET | `/api/evaluation/transformation` |  | Render + serve the run-level SR-transformation summary PNG. |
| GET | `/eval-files/<path:relpath>` |  | Download one per-object `.fits` under `eval_results/` (attachment, `application/fits`). Jailed: 403 `{ok:false,error}` outside the tree; 404 for anything that is not an existing FITS (the classic PNG renderer is gone). |

### JWST × Euclid (`routes/jwst_euclid.py`)

| Methods | Path | Gate | Notes |
|---|---|---|---|
| POST | `/api/jwst-euclid/download` |  | Download + align one JWST × Euclid pair (local job; MAST/Euclid archive, not FASRC). The Euclid VIS side is the archive tile covering the JWST position. |
| POST | `/api/jwst-euclid/download-all` |  | Download every remaining pair (local job). |
| GET | `/api/jwst-euclid/field.json` |  | Manifest of one saved paired field (`?id=`). |
| GET | `/api/jwst-euclid/field/<identifier>/download/<kind>` |  | Download one paired-field FITS asset. |
| GET | `/api/jwst-euclid/fields` |  | Cached JWST × Euclid location groups + status. |
| POST | `/api/jwst-euclid/infer` |  | Run the production model on a saved pair (local job): builds / reuses the pair's four-band LR input (`starfull_inference/euclid_lr_vis_y_j_h.fits`, bands cut from the containing Q1 tile), writes `starfull_inference/starfull_combiner.fits` (SR WCS = LR ×2). |
| POST | `/api/jwst-euclid/nexus/download` |  | Download one NEXUS tile at (`ra`, `dec`) (local job); Euclid VIS from the Q1 tile whose polygon contains the point. |
| POST | `/api/jwst-euclid/nexus/download-field` |  | Cache a NEXUS mosaic + four-band Euclid coverage (local job). Every band is cut from the Q1 MER tile whose polygon contains the tile centre (committed `q1_mer_tiles.json`; never the nearest tile centre); the manifest records each tile's `polygon` (VIS grid corners), `euclid_tile_index` and the mosaic grid `footprint`. |
| GET | `/api/jwst-euclid/nexus/fields` |  | Cached NEXUS fields. |
| POST | `/api/jwst-euclid/nexus/infer` |  | Run the production model (C9 spec `production`: the spatial gate fitted for the current STARFULL members; no RBF fallback) on the stale NEXUS tiles (local job). SR WCS = LR WCS ×2 (`CRPIX → 2·CRPIX − 0.5`). The helper `run_starfull_nexus_field_inference(…, tiles=, spec=)` also takes a tile subset and any model spec (non-production specs write the C9 output store; production SRs stay in `tiles/` and are served by C9 as legacy outputs — `m:production` on `/api/real/nexus/*` and the `real` viewer); run subsets/specs through `POST /api/experiments`. This route does not yet accept `tiles`/`spec` form fields (not WP-B2's file). |
| POST | `/api/jwst-euclid/scan-coverage` |  | Scan Euclid VIS coverage of the cached JWST rows (local job). |

### Viewer (`routes/viewer.py`)

| Methods | Path | Gate | Notes |
|---|---|---|---|
| GET | `/viewer/cube/<collection>` |  | The same cube addressed by object id: `?id=<meta object id>&tier=…` (404 unknown id, 400 without `id`). |
| GET | `/viewer/cube/<collection>/<int:index>` |  | Raw little-endian float32 `(H, W, C)` cube of one object/tier (`?tier=` + collection params) with `X-Cube-*` headers, incl. `X-Cube-WCS` / `X-Cube-Unit` / `X-Cube-Index` (C6, see *Viewer collections*). JSON `{error}` on failure. |
| GET | `/viewer/meta/<collection>` |  | Collection metadata: `count, tiers[{key,label,unit?,hidden?,disabled?}], default_tier, band_names, objects[{id, label, ra?, dec?, tiers?…}]`, colour constants; `?id=` adds `index`; `no-cache`. JSON `{error}` on failure (C6). |
| GET, POST | `/viewer/results` |  | GET: saved viewer results. POST: save a crop/result (JSON or form). |
| GET | `/viewer/results/<result_id>` |  | One saved result summary. |
| GET | `/viewer/results/<result_id>/panel.png` |  | PNG panel of one saved result (`tier`, `mode`). |
| GET | `/viewer/results/grid.<output_format>` |  | Publication grid of saved results (`result`, `row`, `dpi`). |

### Poster cutout (`routes/poster.py`)

| Methods | Path | Gate | Notes |
|---|---|---|---|
| GET | `/poster/result/cutout.fits` | fasrc | Latest `poster_cutout` FITS (download), pulled from FASRC. |
| GET | `/poster/result/cutout.png` | fasrc | Latest `poster_cutout` preview PNG, pulled from FASRC. |
