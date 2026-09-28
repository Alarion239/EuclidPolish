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
- **Work-doing GETs refuse cross-site requests.** A `<img src>` on a hostile
  page carries our `Host`, so the few GETs that still do work for the user —
  `?fresh=1` re-renders and the FASRC file pulls behind links — honour it only
  for a same-origin request (`security.is_same_origin_request`:
  `Sec-Fetch-Site` `same-origin`/`none` or absent, and a matching `Origin` when
  sent; `same-site` is another localhost port and is refused). Their POST
  forms are the state-changing way (`POST /api/evaluation/angular-power-spectrum`,
  `POST /api/fasrc/file/fetch`). A cross-site GET of a figure/payload that
  is not cached yet gets **404** `{"ok": false, "error": "not rendered yet …"}`
  instead of a render (`security.refuse_cross_site_cache_fill`: the
  evaluation PNGs, `/ensemble/evals.json`, which it also serves as cached
  without the diagnostics upgrade). Queue promotion is never a GET side
  effect (server-side ticker, see `/api/fasrc/current-submission`).
- **Security headers on every response** (errors and refusals included):
  `X-Frame-Options: DENY`, `Content-Security-Policy: frame-ancestors 'none'`
  (a framed console would make the frame's POSTs same-origin),
  `X-Content-Type-Options: nosniff`, `Referrer-Policy: same-origin`
  (`security.register_security_headers`).
- Hook order in `create_app`: Host allowlist → cross-origin mutation guard →
  SPA shell / redirects → FASRC gate → handler; after the handler the static
  caching and then the security headers.

### Static assets

- Content-hashed Vite chunks under `/static/dist/assets/` (`<name>-<hash>.<ext>`)
  are served `Cache-Control: public, max-age=31536000, immutable`; the SPA shell
  (`index.html`, answered for page paths) and unhashed files keep `no-cache`.
- Text assets (JS, CSS, JSON, SVG, WASM) ≥ 1 KB are gzip-compressed when the
  request accepts `gzip` (`Content-Encoding: gzip`, `Vary: Accept-Encoding`,
  ETag `"<etag>-gzip"`, a matching `If-None-Match` → 304); compressed bytes are
  memoised per file (path, mtime, size). Range requests are sent uncompressed
  (`static_assets.py`).

### Errors

- JSON endpoints report failures as `{"ok": false, "error": "<message>",
  "code"?: "<machine code>"}` with a 4xx/5xx status. Known codes:
  `fasrc_offline` (503, below), `untrusted_host` (400) and `busy` (409: a
  single-run job of that kind is already running for another request; the
  body carries the running `job_id`).
- **Every HTTP error under `/api/`** (routing 404/405, `abort()`, an unhandled
  exception's 500) is JSON `{"ok": false, "error": "<description>"}` with its
  status — `create_app` registers the whole `/api/` prefix. A few older
  handlers still answer `{"error": ...}` without `ok` on purpose-built
  failures; new endpoints must use the shape above. Everything under
  `/viewer/` is JSON the same way (collection errors, unknown routes,
  unhandled exceptions — contract C6).
- **Integer query arguments** go through `errors.int_arg`: a malformed value
  (`limit=abc`, `skip=1.5`) is a 400 `{ok:false, error}` — never a silent
  fallback to the default. Page offsets / limits are clamped to their range.
- **Single-run jobs** (`jobs.start_exclusive`): posting the same request while
  its job runs re-attaches — 200 `{ok, job_id, already_running: true}`; a
  different request of a kind that must not overlap is 409
  `{ok:false, code:"busy", error, job_id}`.
- **Prefix-wide JSON errors** (`euclid_polish/web/errors.py`): Flask keeps
  one handler per exception class, so a route module must **never** register
  its own `@app.errorhandler(HTTPException)` (it would silently replace
  another module's). Call `errors.json_errors_for(app, "/prefix/")` from the
  module's `register(app)` instead: the first call installs the app's single
  path-dispatching handler (`errors.json_http_error`), later calls add
  prefixes (idempotent). Under a registered prefix every HTTP error — routing
  404/405 and an unhandled exception's 500 included — is `{ok: false, error}`
  with its status; other paths keep Flask's default. Registered today: `/api/`
  (`app.py`; covers every `/api/*` below), `/viewer/`,
  `/api/real/`, `/api/models`, `/api/experiments` (`routes/real.py`),
  `/api/sky/` (`routes/sky_atlas.py`), `/api/system` (`routes/system.py`),
  `/api/inspect`, `/inspect/` (`routes/files.py`), `/api/evaluation/`,
  `/eval-files/` (`routes/evaluation.py`), `/api/figures/` (`routes/figures.py`),
  `/ensemble/` (`routes/ensemble.py`), `/api/provenance` (`routes/provenance.py`) and
  `/view/` (`routes/galaxy_distributions.py`; the plates answer a reason such as "format
  must be png, pdf or svg" or which fit is missing).
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
  `/connection-error` 308s to `/system/connections` (C1); the classic POST
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
- Known kinds: `fasrc-env-update`, `fasrc-accounting` (reconcile / re-pull
  sacct), `fasrc-mirror` (pull ensemble checkpoints), `tng-radii`, `real-tile` (cache a 25.6″
  tile, optionally + models), `real-experiment`, `jwst-discover`, `jwst-pair`
  (contract C9), `system-disk-usage` (`routes/system.py`), `nexus-inference`
  (`routes/jwst_euclid.py`), `figure-nexus-plates` (`routes/figures.py`), `ensemble-band-evals` (`routes/ensemble.py`), `sky-sync`,
  `sky-generate-sr`, `psf-sync`, `tng-properties`, `tng-result` (the Synthetic and Models record routes).

### Pages and redirects (contract C1)

`euclid_polish/web/spa_routes.json` (version 2, the "Loop console" rail:
`/`, `/synthetic/<tab>`, `/models/:mode/<tab>`, `/sky/<tab>`, `/figures/<tab>`,
`/files`, `/runs/<tab>`, `/notebook/<tab>`, `/system/<tab>`) is the single
source of truth for page URLs (`euclid_polish/web/spa_routes.py`:
`load_manifest`, `is_page_path`, `redirect_target`; the SPA's
`app/manifest.ts` mirrors it). A GET/HEAD of a **page path** — a workspace
path with its `:params` substituted, optionally followed by one `/<tab>` from
its tabs — serves `static/dist/index.html` (503 plain-text build hint when the
bundle is missing). A GET/HEAD of a **redirect source** answers **308** to its
target. Redirects are tried in this order, first match wins:

- `redirectRules` (query-aware): `from` is a path pattern whose `:name`
  segments bind one segment each, restricted to `params[name]`; `query`
  requires keys (`"*"` any value, a string, or a list of allowed values). The
  target is `to` with `:name` substituted; its query is the original pairs
  (order kept) after `drop`, `rename`, `map` (value per key), `prefix` and
  `set` (replace or append), form-encoded like the browser's
  `URLSearchParams` (e.g. `/ensemble/starless/curves?layout=time` →
  `/runs/history?step=ensemble_train`, `/sky/results?source=tile` →
  `/sky/targets?set=cached`).
- `redirects` (exact path, trailing slash ignored): the query string is
  appended untouched (`/inspect?fits=a.fits` → `/files?fits=a.fits`).
- `/app/<rest>` → `/<rest>` with every leading slash, backslash or control
  character of `<rest>` collapsed into one `/` (so `/app//evil.example` →
  `/evil.example`, never the protocol-relative `//evil.example`: no open
  redirect).

No target is itself a redirect, and `spa_redirect_cases.json` (next to the
manifest) pins Flask and the SPA to byte-identical targets for every old
URL. Other methods never get the shell or a redirect, and non-page paths that
share a prefix with a page or an old page (`/ensemble/status.json`,
`/inspect/preview.png`) reach their own handlers; backend URLs never move
with the pages.

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
`psfs`, `real` (contract C9, see *Real tiles* below), `fits` (any
inspectable FITS file, see *Files workspace* below), `study` (a model study's
attached fields, see *Model studies* below).

- **Study.** `?study=<id>`; objects = the study's uploaded fields (`id` = the
  field id, `kind`, `fetched`, `bytes`, `tiers`), tiers `lr`, `sr` (the
  production gate), `mean`, `hr`, `bhr` (blurred on the fly with the field's
  target FWHM), `mask` (blackout holes, `arb`) and `member<i>` (hidden; the
  study's member order, `meta.member_labels`). Served from the fetched-field
  cache only: an object is `fetched` once its core products are, and lists
  `fetched_members` (member indices); an unfetched field's cube is 404
  "fetch the field first", an unfetched member tier 404 "fetch member <N>
  first" (the viewer never fetches). Real-tile fields carry `X-Cube-WCS`
  (SR grid = LR ×2).

- **Objects.** Every `meta.objects[i]` has a stable string `id` (`sky` /
  `ensemble`: `"<subset>:<record index>"`; `cutouts`: star id; `evaluation`:
  the object sub-directory; `archive-fields`: sample id; `real-field`:
  `"<field_id>/<tile:03d>"`; `nexus-field`: `"<field_id>/<tile:04d>"`;
  `jwst-euclid`: pair id; `psfs`: `"cluster-NNN"`) and `ra`/`dec` (deg) when
  the object is on the sky and its position is finite (every real
  collection; `cutouts` from the star's `stars.csv` row, plus its `mag` —
  the navigator reads the synchronised FASRC mirror only, no SSH). Synthetic
  `sky` / `ensemble` records have none. The `sky` collection's tiers are
  `dirty` (LR), `hr`, `bhr`, `clean` ("Clean (starless)", the starless scene
  — its own tier, never HR) and `sr` (listed only once the production SR was
  generated over the split, Models › Images); each carries a one-sentence
  `hint` (what the tier is: the chip's tooltip; any collection's tier may);
  records are read by position through a
  header-scanned TFRecord offset index (`helpers/sky_records.py`), O(1) per
  cube. `archive-fields` objects carry the
  position-derived `field` and the manifest's `stored_field`.
- **`?id=` lookup.** `GET /viewer/cube/<collection>?id=<id>&tier=…` serves the
  object whose meta `id` matches (same collection params) instead of a
  positional index; `GET /viewer/meta/<collection>?id=<id>` adds `index` (its
  position). Unknown id → 404 `{"error": "unknown object id: <id>"}`; the
  cube route without `id` → 400. Every cube also sends `X-Cube-Index` (the
  resolved position, exposed).
- **Units.** Tiers may carry `unit` (`"e-"`, `"MJy/sr"`, `"ADU/s"`, `"arb"`)
  in the meta; each cube repeats it as `X-Cube-Unit` (PSF kernels, PCA
  eigen-images and JWST colour composites are `arb`; NEXUS/JWST native tiers
  are `MJy/sr`). The archive star cutouts (`cutouts`, ADU/s on disk) are
  served in electrons over each band's stack via their `MAGZERO` (label
  `… · e- via MAGZERO`), so the console's absolute e⁻ transfer shows them; a
  cutout without `MAGZERO` stays `ADU/s` with an `X-Cube-Display-Scale`.
  The meta's `unit` is the collection's usual unit (it does not open every
  cutout, so `cutouts` always says `e-`); a cube's `X-Cube-Unit` is the
  authority for that cube, and the viewer reads it first.
- **White point.** `meta.color.default_asinh` (K0; white at 30·K0 e⁻) is
  `Config.STRETCH_SCALE_E` for every collection except a bright `fits` file:
  when the selected plane's 99.99th percentile (e⁻) exceeds the default
  white (3000 e⁻), K0 = that percentile ÷ 30, shared by every tier of the
  file (the poster galaxy's core, 1.0 × 10⁵ e⁻, keeps its structure).
- **Labels.** Tier labels are plain words: the part before the first ` · `
  (or ` (`) is the chip ("LR VIS · HDU 1", "Mean · 30 starfull members",
  "JWST (native)"); regime names are lower case ("starfull").
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
- **Evaluation movie centre.** `eval/disagreement.py` persists the member
  mean (`mean.fits`, the centre the `pca*.fits` components are about) beside
  `SR.fits` (the production-gate output). The `evaluation` collection lists it
  as tier `mean` ("Mean of members", on the objects that have it) and sets
  `meta.morph_base_tier = "mean"` once any object has it; an object written
  before `mean.fits` existed answers `tier=mean` with its `SR` (label
  "SR (member mean not persisted)" — the old approximation). Tier order:
  `LR, SR, mean, HR, BHR, std` (+ `morph`).
- **Evaluation geometry.** `eval/catalog_runner.enforce_object_sizes` crops
  every object FITS to the canonical 53² LR / 106² SR-grid stamp keeping the
  sky under each pixel: each file's `CRPIX1/2` moves by its crop offset and a
  2×-grid plane exactly twice the LR stack is cut at twice the LR offset, so
  the SR tiers stay the LR WCS ×2. Objects cropped before this fix keep a WCS
  off by their crop offset (≤ 1 LR px) until regenerated.

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
  gate `spatial_gate_combiner/`, available while every member it READS — its
  `active_members`, all of them when unpruned — is an active STARFULL member
  with a checkpoint; members registered after its fit only add a `note`
  ("N member(s) joined after this fit; refit to consider them") and
  `details.joined_after_fit`; unavailable otherwise, no silent fallback),
  `mean` (all active STARFULL members), `member:member_<N>` (each active
  member; aliases `member:170`, `member:170·psnr`), `gate:<variant>` (every
  `spatial_gate_<variant>/` beside the production artifact, applied with
  **its own** member labels via `eval.spatial_gate.load_spatial_gate`;
  unavailable, with `reason`, unless every member it reads is an active
  STARFULL member), `rbf` (the RBF combiner, own labels, all of them
  needed). Members always run through `EnsembleModel` by label — exactly the
  members a job needs (`EnsembleModel(labels=…)`), never a registry prefix. A
  spec's `fingerprint` hashes the checkpoint fingerprints of the members it
  READS (a pruned gate: archiving or retraining a member it ignores leaves
  its outputs current) plus (for combiners) the artifact's
  `combiner.json`+`combiner.npz`.
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
  (e⁻, LR WCS), `jwst` ("JWST (native)", MJy/sr, its own WCS; `?jwst_band=`
  picks a pair filter) when any tile has JWST, and `m:<spec>` (e⁻, SR WCS;
  the tier label names what is served: the catalogue's model, or — when
  every output of the spec in the source is a legacy SR that is not current
  — the kind, member count(s) and "legacy" (`RBF (10 or 20 members,
  legacy)`, `Mean (4 members, legacy)`), or the catalogue label with
  `(a legacy SR on n of m tiles)` when only some are; the cube label gets
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
| GET | `/api/models` |  | `{regime:"starfull", production_kind, members:[labels], models:[{spec, kind, label, slug, members, member_names, reads, n_members, n_fitted, available, reason, note (soft remark, e.g. members joined after the fit)|null, fingerprint, member_fingerprints, combiner_kind, combiner_fingerprint, details{mix_space, use_lr, width, active_members, fitted_at, artifact_dir, loss, loss_knees_e, steps, complete, used_threshold, …}}]}` — order: production, mean, rbf, members, gate variants. `members` = the members the spec was built for (its staleness key), `reads` = the members it actually runs; `n_members` = `len(reads)` (a pruned gate: 6 of the 20 it was fitted with), `n_fitted` = `len(members)`. |
| GET | `/api/real/<source>` |  | `{source, label, description, count, tiles:[tile entry + models:{spec:{state, legacy, label, fingerprint, created, experiment_id, file, origin, summary, flux_ratio?{band: ΣSR/ΣLR}}} + production_state]}` (models = the tile's merged outputs; `flux_ratio` only on a scored output, so Sky › Targets plots and sorts the flux ratio without opening every card). 404 unknown source. |
| GET | `/api/real/<source>/<identifier>` |  | Card: tile entry + `models:{spec:{state, legacy, label, kind, fingerprint, created, experiment_id, file, member_labels, combiner_kind, lr_sha, shape, origin, metrics{per_band, summary, gate_core_weights?}, image_url}}` (store + legacy outputs), `production_state`, `legacy` (= `extras.legacy_sr`: the pre-C9 production-pipeline SR record or `null`), `runnable_models`, `experiments:[ids]`, `disk{tile_bytes, output_bytes, cache_bytes (member-SR cache), legacy_bytes (NEXUS / pair legacy SR files), total_bytes}`, `q1_tile` (the containing Q1 tile), `image_urls{tier: url}` (one per `m:<spec>` with an output), `files{lr?, sr?, "m:<spec>"?: project-relative FITS path}` (what the card's "Open in Files" opens — `/files?fits=<path>`; only files inside the inspectable roots; `sr` is a catalogue object's own evaluation SR), `viewer{collection:"real", params{source}, id}`. |
| POST | `/api/real/<source>/<identifier>/delete-outputs` |  | Delete the tile's cached model outputs and member-SR cache (never the LR itself): `{ok, ref, removed:[paths], removed_count, cache_bytes_freed}`. |
| GET | `/api/real/<source>/<identifier>/image.fits` |  | 2-D float32 FITS of one plane with its celestial WCS (CD form) for sky overlays: `tier` = `lr` (default) \| `jwst` \| `m:<spec>` (store output or legacy SR; SR WCS = LR WCS ×2), `band` = `VIS` (default) \| `Y_E` \| `J_E` \| `H_E` (JWST: its filter, default the first). Header `BUNIT` (`electron` / `MJy/sr`), `TIER`, `BAND`, `REALTILE` (+ `LEGACYSR` for a legacy SR). 400 bad tier/band; 404 missing tier. |
| GET | `/api/real/sources` |  | `{sources:[{id, label, description, count, model_ready, has_jwst, ready, reason}]}` for `nexus, tile, field, archive, eval, poster, pair`. |
| POST | `/api/real/tiles` |  | Cache a 25.6″ four-band tile at `ra`, `dec` (local job `real-tile`; Euclid archive, not FASRC): Q1 coverage is checked first against the committed MER polygons (400 `{ok:false, code:"outside_q1"}`; 400 `{ok:false, code:"unobserved_q1", tile}` when every containing tile is `rejected` — measured unobserved by the noise campaign — unless `force=1`); each band is cut from the Q1 tile whose polygon CONTAINS the point (observed tiles first, deepest inside), VIS cropped to the exact 256² grid, NISP registered onto its WCS, electrons via MAGZERO. Optional `run=production,mean` then runs an experiment on it. `{ok, job_id, id, ref:"tile/<id>", experiment_id|null}`; job result `{tile, ref, experiment?}`. One job per tile id: the same position again while it caches re-attaches (`already_running: true`, `experiment_id: null`); other positions cache concurrently. |

### Sky atlas (contract C9, `routes/sky_atlas.py`)

`helpers/sky_atlas.py`; every layer is built from local files (memoised on
their mtimes, ≤ 30 s) and works offline. Layer groups, in the Layers panel's
order (`GROUPS`): `real` — the real tiles SR runs on (`nexus-tiles`,
`real-tiles`, `real-fields`, `poster`, `pairs` — polygons with `state` =
production state; `experiments`, the comparison tiles — points), `targets` —
the science targets (`eval-objects`, the reconstructed lens candidates and Q1
galaxies, coloured by flux SR/LR; `lens-candidates`; `galaxies`), `inputs` — the
real data behind the synthetic scenes, each owned by a Synthetic tab (`stars`,
the FASRC-mirror `stars.csv`, 43k rows `[ra, dec, mag, flags]`, `flags` bit
`2^b` = valid cutout in band `b` ∈ VIS,Y_E,J_E,H_E; `psf-clusters`;
`noise-positions`; `population-cones`; `archive-fields`) and `coverage`
(`q1-tiles` 352 MER polygons with `levels_e`/`rejected`/`state`, `q1-fields`
cones, `nexus-footprint` hull of the NEXUS tile cells, `jwst-mast`). Each layer
row names its `home` — `{path, label}`, the tab that owns (and fills) its data,
e.g. `stars` → `/synthetic/psf?view=catalogue` "Synthetic › PSF",
`nexus-tiles` → `/sky/targets?set=nexus` "Sky › Targets".

Feature shapes (`GET /api/sky/layer/<id>`, always with `id, label, group, kind, count`):
`points` → `{columns:[…], rows:[[ra, dec, …]], inspect:{kind, prefix, id_column}}`
(row inspector entity `{kind}:{prefix}{row[id_column] or row index}`);
`polygons` → `{features:[{id, polygon:[[ra,dec],…], props, inspect:{kind, id}}]}`;
`circles` → `{features:[{id, ra, dec, radius_deg, props, inspect}]}`. Real
tiles inspect as `realtile:<source>/<id>`, others as `source:<layer>/<id>`.

| Methods | Path | Gate | Notes |
|---|---|---|---|
| GET | `/api/sky/at` |  | (Memoised: the real-tile listings are cached on the on-disk stamp of what they read — manifests and tile directories (for evaluation objects: `manifest.csv` and each `<sub>/` directory) — never on a timer, so a warm click answers in ~10 ms however long the server idled; any change is seen on the next request.) What covers `ra`, `dec`: `{ra, dec, field, in_q1, q1_observed (a containing tile is not rejected), q1_verdict: "observed"\|"unobserved"\|"outside", q1_tiles:[tile (incl. rejected) + margin_arcsec] (observed tiles first, each deepest first), best_tile, real_tiles:[{source, id, ref, label, has_jwst, state, inspect}], nexus, pairs, jwst:[discovered footprints containing the point], jwst_discovered}`. 400 bad/missing coordinates. |
| POST | `/api/sky/jwst/discover` |  | JWST × Euclid discovery (local job `jwst-discover`; MAST, not FASRC): wraps `scripts/find_jwst_euclid_overlap.py` (the same MAST cone query and cache in `data/jwst_euclid_overlap/mast/` — `jwst_euclid._mast_rows_for_scope`, astroquery imported at module top — plus the script's direct-imaging + public filters and CSV rows) with the exact test MAST `s_region` ∩ committed Q1 polygon done locally. Scope: `fields` (comma list: `EDF-N,EDF-S,EDF-F,LDN1641`) and/or `region=ra,dec,radius_deg` (≤ 10°); both empty = all Q1; `refresh=1` ignores the MAST cache. Results MERGE into `overlap.csv`/`overlap.json` (pairing input) and `jwst_footprints.json`. `{ok, job_id, tile_count, fields, region}`; job result = the discovery manifest + `footprint_count`. |
| GET | `/api/sky/jwst/footprints` |  | Cached MAST footprints near `ra`, `dec` within `r` deg (default 0.5, ≤ 5): `{ra, dec, r, ready, updated_utc, fields, count, truncated, footprints:[{obs_id, instrument, filters, target, proposal_id, exptime_s, ra, dec, polygons:[[[ra,dec],…]], status, euclid_tiles, fields}]}` (nearest first, ≤ 2000). |
| POST | `/api/sky/jwst/pair` |  | Download + align one JWST × Euclid pair, then build its four-band LR input (local job `jwst-pair`): `obs_id` (a discovered MAST observation → its location group) or `ra`, `dec` (inside one of the NEXUS × Euclid tile cells — their polygons, not the hull → a NEXUS cutout, `filter` F200W\|F444W; else the nearest discovered location within max(size/2, 15″)); `size_arcsec` (1–120, default 30); optional `run=<specs>`. 404 `code:"not_discovered"` when nothing covers it. `{ok, job_id, pair_id, ref:"pair/<id>", mode:"nexus"\|"archive"}`. NEXUS mode is a single-run job of kind `nexus-mosaic` (shared with `/api/jwst-euclid/nexus/download` and `/download-field`): the same pair again re-attaches (`already_running`), any other NEXUS mosaic job running → 409 `code:"busy"`. The crop streams only the mosaic rows it spans (`helpers/fits_plane.py`; tens of MB, never the 4.46 GB plane). The Euclid VIS box comes from the committed Q1 tiles containing the point (observed first, deepest inside first; the archive INTERSECTS search only outside every committed polygon) and must be ≥ 99.5 % observed, else the job fails (no partial pair is published); the pair manifest records `euclid_vis_tile_index` and `euclid_selection{method: q1_polygon\|archive_intersects, tile, coverage, tried}`. The pair becomes real source `pair`. |
| GET | `/api/sky/layer/<layer_id>` |  | One layer's features (shapes above); 404 unknown layer. |
| GET | `/api/sky/layers` |  | `{groups, layers:[{id, label, group, kind, count, bbox{ra_min, ra_max, dec_min, dec_max}\|null, style{color?, color_by?, colors?, shape?, size?, opacity?}, ready, reason, fill_action{method, url, label, requires_fasrc?, self_connects?, sync?}\|null, description, home{path, label}\|null, url}]}` (`groups` = the group ids in panel order). `requires_fasrc`: needs the console's FASRC connection; `self_connects`: the job opens its own FASRC connection (works from an offline console); `sync`: the POST answers the result directly (no `job_id`) — refresh right away. Layers are memoised on the stamp of their inputs (for results layers: the source listing, `outputs/<source>/<id>/` directories and the production fingerprint), never on a timer. |

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

Later, with no SPA (source or bundle) reference: the matplotlib PNGs
`/view/catalog`, `/view/psfs`, `/view/psf-clusters`, `/view/training-log`
(and their renderers, `helpers/sky_render.py`), `/inference/cache-real-field`
(real tiles are cached by `POST /api/real/tiles`), `/api/jwst-euclid/fields`,
`/api/jwst-euclid/field.json` and `/api/jwst-euclid/scan-coverage`
(`tests/test_legacy_removed.py` pins them).

### Files workspace (`routes/files.py`, `helpers/{paths,fits_inspect,fits_render}.py`)

- **Roots** (`paths.inspect_roots()`, computed per request): evaluation
  results, Euclid inference, JWST × Euclid, Euclid sky, star cutouts, band
  PSFs, viewer results, population comparison, TNG SKIRT, `data/vis`, the
  FASRC cache, synthetic records, the repo's `poster/` and `output/`, and
  `tracking/`. Every path (`?fits=`, `?dir=`, the viewer's `path`) must
  resolve — symlinks expanded — inside one of them (403 otherwise). Browse
  entries resolving outside are dropped; hidden names are skipped. FITS names:
  `.fits .fit .fts` (+ `.gz`), `.fz`. All `/api/inspect*` and `/inspect/*`
  errors are JSON `{error}` with the status.
- **HDU summary** (headers only; a 1 GB mosaic lists at once): `index, name,
  ver, kind, type ("image"|"vector"|"table"|"empty"|"other"), shape (numpy
  order), dtype (from BITPIX), scaling{bscale,bzero}?, ndim, planes,
  plane_axes, bunit (inherited from the primary), wcs{ctype, ra, dec,
  pixscale_arcsec, width_arcsec, height_arcsec, fov_deg, corners,
  constructed}, bands (a 3-D cube's `BANDS` card, or the four Euclid bands
  for an unlabelled 4-plane cube: `bands_assumed`), band (2-D: `FILTER` or a
  `…VIS/Y_E/J_E/H_E` name; bare NISP letters `Y`/`J`/`H`, `…_Y`, `…NISP_Y`
  and a `FILTER` of `NISP_Y`/`J` map to `Y_E`/`J_E`/`H_E`), compressed,
  size_bytes, viewable, reason`;
  tables add `columns, nrows, ncols`. A header without a celestial WCS whose
  primary has `RA`/`DEC` + a `PIXSCALE` gets a constructed north-up TAN
  (`constructed: true`, the poster convention). `band_groups`: 2-D HDUs named
  `<prefix><band>` covering the four bands with one shape →
  `{id: "b:<prefix>", label, hdus, bands, shape, wcs, bunit}`. A `.gz` over
  256 MB lists only its primary (`scan_truncated`).
- **Track** (`POST /api/tracking/backup kind=fits|image path=`): the path check
  (`helpers/paths._resolve_trackable_file`) answers 400 / 403 / 404 with a JSON
  `{ok: false, error}` body even though `/api/tracking` is not a JSON-error prefix.
- **Planes** are memory-mapped (never whole-file reads) and binned for display
  (auto: longer side ≤ 2048; never above 4096): a block mean up to 4096² source
  pixels, a strided sample beyond. Served pixel `j` is centred on source pixel
  `bin·j + offset` and the served WCS is adjusted to match. `BSCALE`/`BZERO`
  are applied, `BLANK` → NaN. A gzip / tile-compressed plane over 512 MB is 413.
- **Viewer collection `fits`**: params `path` (required), `hdu` (an image HDU
  index or `b:<prefix>`; default the first band group, else the first viewable
  image), `stack`
  (`bands` default | `planes`), `bin` (1–256, default auto), `render`
  (`log`). Objects = the selected HDU's planes (`id` `p<k>`, label `VIS` /
  `plane k` / `[i, j]`; one object for a band cube or band group); tiers =
  every viewable image HDU (`h<index>`, label `<name in words> · HDU <index>`:
  "LR VIS · HDU 1", "SR Y · HDU 6"; beyond 12 hidden) + every band group
  (`b:<prefix>`, "LR colour · VIS Y J H"), so HDUs compare side by side. A
  bright file moves `meta.color.default_asinh` (White point, above). A
  non-selected HDU serves its own plane `k` (a 2-D HDU its only plane).
  `meta.fits = {path, hdu, planes, planes_truncated, stacked}`. Cubes carry
  `X-Cube-WCS` (binned), `X-Cube-Unit` (from `BUNIT`), channel names (bands,
  else the HDU name) and, for units other than e⁻, `X-Cube-Display-Scale`
  (robust bright end → white; values stay native). An archive-rate band
  (`BUNIT` ADU/s + `MAGZERO`) displays as its electrons instead (display
  scale = the MAGZERO factor, readout native); a band group of such HDUs is
  served in electrons (`e-`, label `… · e- via MAGZERO`) so its colour
  composites are physical. Without any WCS or `PIXSCALE` the pixel scale is
  assumed to be the VIS 0.1″. A plane is picked with the viewer's object id
  (`initialId="p<k>"`, `?id=p<k>`); the Files page maps its one-shot
  `?slice=` (flat index, `i,j` multi-index or band name) to it.

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
| GET | `/api/inspect` |  | FITS inspector payload for `?fits=<project-relative path>` (headers only, no pixels): `{file{basename,size,size_kb,mtime,compressed}, hdus:[HDU summary + cards], band_groups, scan_truncated, stamp, rel, root, roots, allowed_roots}` (see *Files workspace* below). |
| GET | `/api/inspect/browse` |  | File browser: no `dir` → the existing roots as entries; `dir` → its sub-directories + FITS files (`{dir, root, crumbs, entries, other, truncated}`); `q` → a bounded FITS-name search under `dir` (or every root). 403 outside the roots, 404 missing. |
| GET | `/api/inspect/image/stats` |  | Statistics + histogram of one plane (`hdu`, `plane`) at full resolution (`sampled` = stride above 4096² px); a 1-D HDU returns its `series` instead. |
| GET | `/api/inspect/table` |  | One page of a table HDU: `hdu`, `offset`, `limit` (1–2000, default 200), `sort` (column), `desc`. `{total, columns[{name,format,unit,dim,null,kind}], rows, row_index}`. |
| GET | `/api/inspect/table/stats` |  | Per-column statistics of a table HDU (numeric: moments, percentiles, histogram, nulls; text: unique + top values; bool: counts). |
| GET | `/api/inspect/provenance` |  | `{stamp, sidecars[{file,id,kind,current,record}], related[{role,id,file,kind,record}], stale_sidecars}` for a FITS file. |
| GET | `/api/jobs` |  | Local background jobs, newest first (C2). `?summary=1` → same with `log: null`. |
| GET | `/api/jobs/<job_id>` |  | One job dict with the full log tail (C2); JSON 404 `{ok:false,error}` when unknown. |
| POST | `/api/jobs/<job_id>/cancel` |  | Cooperative cancel (C2): `{ok:true}`; the job turns `cancelled` at its next `cap.tick`. 404 unknown, 409 already finished. |
| GET | `/api/status` |  | Local status summary `{catalog, psfs, tfrecords, checkpoints}` — cache-only and cheap: no SSH, no rsync (`catalog.cached: true`, the last synchronised `stars.csv`). |
| POST | `/api/status/refresh-catalog` | fasrc | Explicitly re-pull the FASRC `stars.csv` (forced rsync): `{ok:true, catalog}`. |
| GET | `/api/version` |  | Server version (C3): boot and HEAD commits (informational), `behind` = a backend `euclid_polish/**/*.py` module the server loaded changed on disk since it was loaded (stat, then content hash; `web/frontend` and `web/static` excluded; checked at most every 10 s), `changed_files` (newest first, the first 8, repo-relative), `changed_count` and `changed_digest` (short hash of the whole changed set, null when none — a stable dismissal key), `dirty`, `started_at`, `pid`, `dist{built_at,index_hash,entry}` (`entry` = the build's `<script type="module" src>`). A module first seen after boot is checked against its `.pyc` source stamp, so an edit between its lazy import and the first check is still reported. Committing the code the server already runs is not `behind`. Safe to poll: the git probes are read-only (`--no-optional-locks`), so they never take `.git/index.lock` from a concurrent commit. |
| POST | `/api/fasrc/file/fetch` | fasrc | Pull one FASRC file (`remote_path`) into the local cache (the POST form of the two link GETs below): `{ok, path (project-relative), inspect_url:"/files?fits=…", download_url}`; 400 without `remote_path`; a failed fetch is JSON 502 `{ok:false,error}`. |
| GET | `/fasrc/file/download` | fasrc | Fetch one FASRC file (cached) and send it. A cross-site request is refused (403 JSON): the SPA's link is same-origin. |
| GET | `/fasrc/file/inspect` | fasrc | Fetch one FASRC file (`?remote_path=`, cached) and 302 to `/files?fits=<project-relative path>`; a failed fetch is JSON 502 `{ok:false,error}`; a cross-site request is refused (403 JSON). |
| GET | `/inference-files/<path:relpath>` |  | Serve FITS/PNG from `data/euclid_inference/` (jailed). |
| GET | `/inspect/download` |  | Download the inspected FITS (`?fits=`), jailed to project data roots. |
| GET | `/inspect/preview.png` |  | PNG thumbnail of one plane of the inspected FITS (`?fits=`, `hdu`, `plane`; default the first image HDU's first plane), longer side `size` px (16–2048), aspect kept, data-adaptive stretch. |
| GET | `/vis/<path:relpath>` |  | Serve a PNG from `data/vis/` (jailed; 403 outside, 404 missing). |

### FASRC connection and cluster (`routes/fasrc.py`)

Decisions (W-Ops, 2026-09-26): the checkpoint mirror is **manual only** (a
confirmed `rsync --delete-after` job; its periodic poller was started by the
deleted training-status poll and is gone); `FasrcConfig` carries no sbatch
defaults (each step owns its resources — the unused `partition/n_gpus/n_cpus/
memory/time_limit` fields were removed; old keys in `fasrc.json` are
ignored); the sqlite step-counter ETA helpers are removed (a SLURM job's
progress comes from its `.events` stream, `/api/fasrc/jobs/<jobid>/status`).

| Methods | Path | Gate | Notes |
|---|---|---|---|
| POST | `/api/fasrc/bootstrap-data` | fasrc | Re-create the netscratch → holylabs data symlinks on FASRC. |
| GET | `/api/fasrc/files` | fasrc | Remote file browser. No `dir` → the allowed roots (data dir, checkpoint dir, `<repo>/logs`); `dir` → `{dir, crumbs[{name,path}], entries[{name, path, type: dir\|file\|link\|other, size\|null, mtime, inspectable}], truncated}` (dirs first, ≤ 2000 entries). A `dir` outside the roots (or with `..`) is **403**. Inspect/download a file with `/fasrc/file/inspect` / `/fasrc/file/download` (`?remote_path=`). |
| GET | `/api/fasrc/history` |  | Run history across **every** step: the local job ledger (CSV) joined with the DB, newest first, each row compacted (`params` parsed, with `params_omitted` `{key: size}` for embedded payload blobs such as `_star_prior_json`; the raw `params_json` column is dropped so the params ship once), plus `db_state` and `state_display` (`fasrc_jobs.display_state`: sacct's final ledger verdict, else the DB state when live or finalised — a stale RUNNING/PENDING ledger snapshot never outranks a DB CANCELLED/DONE — else the speculative state). Filters `step` (comma list), `state` (comma list of display states, or `unresolved` = `fasrc_jobs.is_unresolved`: a blank/UNKNOWN/DONE ledger, or a stale live ledger state the DB has finalised, and not live in the DB), `q` (jobid/label/step); `offset`/`limit` (≤ 2000, default 500). `{total, offset, limit, rows, facets{steps, states}, unresolved}`. Local, works offline. |
| POST | `/api/fasrc/cancel` | fasrc | `scancel` one SLURM job (`jobid`). |
| GET, POST | `/api/fasrc/config` |  | GET the FASRC connection settings; POST (form) patches them. Works offline. |
| POST | `/api/fasrc/connect` |  | Open the SSH ControlMaster from the saved settings. `{ok:true,status}` or 400 `{ok:false,error,status}`; failures are kept as `last_error`. Works offline. |
| GET | `/api/fasrc/current-submission` | fasrc | Newest PENDING/RUNNING submission (`current: {job, status, array, accounting}` or `null`) reconciled against `squeue`, the local `queue`, `stale`, and `live`: **every** PENDING/RUNNING job (newest first, same row shape as `current.job`: DB row + squeue `start_time/reason/nodes/time/time_limit`) — C5. Read-only for the queue: it reconciles the job DB against `squeue` but never promotes (a GET must not `sbatch`); the server-side queue ticker (`fasrc_queue.TICKER`, every 20 s while items are queued; started by the server and poked by the queueing / resume POSTs) reconciles and promotes; it keeps reconciling after the queue empties until the last promoted job ends, so `active_jobid` clears — or the queue halts when that job failed. The time-travel sandbox launcher starts it too (`start_background_services`). |
| GET | `/api/fasrc/data-listing` | fasrc | Sizes and entries of the FASRC data directories. |
| POST | `/api/fasrc/disconnect` |  | Close the session and clear `last_error`. |
| POST | `/api/fasrc/env-update` | fasrc | Start `yes \| mamba env update` on FASRC as a local job (`kind="fasrc-env-update"`): `{ok:true, job_id}`; the remote output streams into the job log; the job ends `done` with `result={exit_code, lines}` or `failed` on a non-zero exit. Cancellable: the remote command prints a filtered heartbeat every 2 s, so a cancel lands within one heartbeat even while mamba is silent; the remote side (no pty, so no SIGHUP) is then killed by its heartbeat watchdog when the closed channel makes a write fail. A cancel can leave the env half-updated; re-run to finish. POST-only (was a GET SSE stream). |
| POST | `/api/fasrc/git-pull` | fasrc | `git pull` on FASRC; `env_update_needed` when `environment.yml` changed (the UI then starts `POST /api/fasrc/env-update`). |
| GET | `/api/fasrc/git-status` | fasrc | The FASRC checkout after a `git fetch`: `{repo, branch, ahead, behind (vs its upstream), head (full hash), local_head (this laptop's HEAD), relation {relation: same\|remote_behind\|remote_ahead\|diverged\|unknown, ahead, behind} (FASRC HEAD vs local HEAD; unknown when the local repo lacks that commit), dirty, dirty_files (≤ 50 porcelain lines), last{hash,subject,relative}}`. |
| GET | `/api/fasrc/jobs/<jobid>/status` |  | Structured status from one job's `.events` stream; an empty status offline (never gated). |
| GET | `/api/fasrc/mirror/status` |  | The last checkpoint pull `{last_run_at, last_rc, last_error, last_stdout, remote_dir, local_dir, job_id}` (`job_id` of a pull running now, else null). Manual only: there is no periodic mirror. Local. |
| POST | `/api/fasrc/mirror/trigger` | fasrc | Pull the remote ensemble checkpoints into the local mirror as a local job (`kind="fasrc-mirror"`): `{ok, job_id}` (a running pull's id with `already_running`). `rsync --delete-after` removes local files the FASRC copy lacks, so `confirm=1` is required (else 400 `code:"confirm_required"`). |
| GET | `/api/fasrc/queue` | fasrc | The user's live `squeue` rows. |
| POST | `/api/fasrc/queue/clear` |  | Clear the local submission queue. |
| POST | `/api/fasrc/queue/remove` |  | Remove one queued submission (`id`): `{ok, queue}`. |
| GET | `/api/fasrc/queue/state` |  | The local submission queue `{ok, queue{count, names, items[{id, label, step, queued_at, position}], active_jobid, halted, halted_reason}}` (the stored specs stay server-side). Local, works offline (`current-submission` carries the same block but needs SSH). |
| POST | `/api/fasrc/queue/resume` |  | Clear a halt so the queue continues past the job that stopped it (a failed active job leaves the lane; a running one stays): `{ok, queue}`. The server-side queue ticker promotes the head item right after (it skips while FASRC is offline). Local. |
| POST | `/api/fasrc/refresh-accounting` | fasrc | Re-pull Jobstats + sacct as a local job (`kind="fasrc-accounting"`, cancellable between jobs): `scope=unresolved` (default) reconciles the jobs whose outcome is unknown (ledger state blank/UNKNOWN/DONE or a stale RUNNING/PENDING snapshot the DB has finalised — the history's `unresolved`; live jobs skipped) and writes sacct's verdict into the ledger **and**, when final, over a speculative DB state (an authoritative DB state is never overwritten; a still-live sacct verdict resolves nothing); `scope=all` re-pulls every finished job (any recorded non-live ledger state plus the stale live ones; backfill). `{ok, job_id}` (a running one's id with `already_running`); result `{ok, updated, total, scope, resolved{jobid: state}}`. 400 on another scope. |
| GET | `/api/fasrc/runs` | fasrc | Log files of recent runs on FASRC (reconciled against `squeue`). A console-submitted run's `state` follows the same rule as the history's `state_display` (its raw DB state is `db_state`), so a job reads the same in Logs and History. |
| GET | `/api/fasrc/runs/log` | fasrc | One FASRC log file under `<repo>/<logs_subdir>/` ending `.out`/`.err` (`path`). `page`/`page_size` → a window counted from the end `{total_lines, start_line, end_line, has_older, has_newer, content}` (page 0 = newest; the SPA's follow mode re-polls it); `grep=<text>` → case-insensitive fixed-string search over the whole file `{matches[{line, text}] (≤ 500), truncated}` (400 when blank); neither → the legacy tail (`lines`). |
| GET | `/api/fasrc/runs/training-curve.json` | fasrc | Per-step training records for one run's wall-time window (JSON, ≤ ~600 points). |
| GET | `/api/fasrc/status` |  | Connection state `{ssh_connected, connected_at, socket, last_error}` (C4). `last_error` = startup auto-connect error or last failed connect; null after a successful connect or manual disconnect. |
| GET, POST | `/api/fasrc/steps/<step_id>/history` |  | Per-step run history (newest first, every state) + best-match prefill suggestion `match` (local job ledger). Rows are compacted like `/api/fasrc/history` (`params` without the embedded payload blobs, listed in `params_omitted`; no `params_json`). |
| POST | `/api/fasrc/steps/<step_id>/submit` | fasrc | Submit (or queue behind the running job) one pipeline step: `confirm=yes`, resources (`n_cpus`, `n_gpus`, `memory`, `time_limit`; partition is fixed per step) and task params. Absent (and blank, unless blank means "unset") task params take the schema defaults; an invalid one, or a spec that cannot be rendered, is refused **400** `{ok:false,error}` before anything reaches FASRC or the queue (C5, see *FASRC step task parameters*). `{ok, jobid}` or `{ok, queued:true, queue}`. |
| GET | `/api/fasrc/steps/status` |  | `{ssh_connected, steps[], artifacts, remote_paths}`; each step: `step_id, label, needs_gpu, fixed_cpus, fixed_gpus, defaults` (resources), `task_params` (schema) and `last_params` (typed task params of the newest COMPLETED run, or `null`) — C5 — and `outputs[{key, path, exists}]` (the remote artifacts the step is known to write; `exists` null offline). Offline it skips the artifact probes (never gated). |

### Euclid archive auth (`routes/auth.py`)

| Methods | Path | Gate | Notes |
|---|---|---|---|
| POST | `/auth/login` |  | Log in to the ONE laptop-side Euclid archive session (`username`, `password`; the password is never stored). `{ok:true, user}` (the session's details: `/auth/status`); 400 missing field; 500 `{ok:false,error}` (the archive's refusal) — a failed login keeps the previous state. |
| POST | `/auth/logout` |  | Log out of the laptop-side session: `{ok:true}`. |
| GET | `/auth/status` |  | The laptop-side Euclid archive session every local archive query reads (System › Connections): `{authenticated, user, logged_in_at (ISO)\|null, used_by:[{id, label, to}]}` — `used_by` names the console features that need it (Synthetic › Galaxies / Stars / Fields, Sky › Targets). |
| POST | `/euclid-auth/save` | fasrc | Write Euclid archive credentials to `~/.euclid_credentials` on FASRC. |
| GET | `/euclid-auth/status` |  | Whether a credentials file exists on FASRC (username only); `connected:false` offline (never gated). |

### System and Home health checks (`routes/system.py`)

All local (never gated); JSON errors under `/api/system` (`errors.json_errors_for`).

| Methods | Path | Gate | Notes |
|---|---|---|---|
| GET | `/api/system` |  | System › Code and System › Storage facts, cheap (nothing is walked): `{python{version, implementation, executable}, platform{system, release, machine, platform}, packages{flask, werkzeug, numpy, scipy, astropy, tensorflow, photutils: version\|null}, node (node --version, when Node is on the server PATH)\|null, pid, cwd, data_dir, noise_model, disk{path, total_bytes, free_bytes, used_bytes, used_fraction, level: ok\|warn\|bad\|unknown, warn_below_bytes (25 GiB), bad_below_bytes (10 GiB), warn_used_fraction (0.95)}, roots{items:[{id ("data/<dir>", "ckpt", "tracking", "poster", "output"), label, path, group, bytes, files, exists}] (biggest first), computed_at\|null, total_bytes, stale (never measured or > 6 h old), ttl_s, refresh_job\|null, experiments{cache_bytes, outputs_bytes}}, experiments{cache_budget_bytes, min_free_bytes, cache_bytes, outputs_bytes, measured_at}}`. The disk usage is the last measurement (memory, else `~/.euclid_polish/system_disk_usage.json`); clients POST the refresh when `stale && !refresh_job`. |
| GET | `/api/system/production` |  | Home's production numbers without the ~8 s `/ensemble/status.json`: `{eval_summary: {the scalar keys of the STARFULL eval_summary.json — ensemble_psnr, mean_member_psnr, ensemble_gain_db, spatial_gate_combiner_psnr, spatial_gate_combiner_vs_mean_db, spatial_gate_combiner_vs_best_member_db, n_scored, …; lists/dicts dropped}\|null, evaluated_at (ISO, file mtime)\|null, stale (the `evaluation` check below is not ok: members or test records changed), stale_reason\|null, members (active STARFULL regime labels), starless_members (active starless)}`. |
| POST | `/api/system/disk-usage/refresh` |  | Measure every data root (apparent size, symlinks not followed) + the experiments' member cache and outputs in a local job (`kind="system-disk-usage"`, one at a time — a running one's id is returned): `{ok, job_id}`; job result `{computed_at, roots, total_bytes}`. |
| GET | `/api/system/alerts` |  | The Home health checks, memoised 30 s (`?fresh=1` recomputes): `{computed_at, ttl_s, counts{bad, warn, ok, unknown}, checks:[{id, label, state: ok\|warn\|bad\|unknown, title, detail\|null, to (SPA path)\|null, action?{label, method, url, params, confirm}, facts?{…}}], alerts:[the warn/bad checks, bad first]}`. Checks: `disk` (free space, thresholds above), `real-sr` (production state of NEXUS / cached / legacy-field / poster / pair tiles from the sky layers: warn when any is `stale`; `facts{current, stale, missing, sources[]}`), `combiner` (the production spec of `/api/models`: warn with its `reason` when it does not fit the STARFULL members), `evaluation` (STARFULL `eval_summary.json` vs the active members and vs the test records' `records_fp`; action = evaluate), `knee` (`/ensemble/knee-psnr.json` missing or `stale`; action = compute), `records-noise` (each local `dirty_*.tfrecord`'s generation run noise model, from its provenance stamp, vs `Config.NOISE_MODEL`: `bad` on a mismatch, `unknown` when the run is not in the local provenance store), `tracking` (the newest `## <ISO>` heading of `tracking/current/log.md` vs the evaluation, production-gate fit, knee curves and experiment records written after it). A check that raises reads `unknown` with its error. |
| GET | `/api/system/loop` |  | The staleness service (`helpers/system_alerts.py`): one verdict per Loop stage, read by Home's Loop strip and System › Lineage, memoised 60 s (`?fresh=1` recomputes it and the alerts it reads): `{computed_at, ttl_s, stages:[{id: priors\|records\|members\|evaluation\|gate\|real-sr\|figures, label, state: current\|stale\|blocked\|unknown\|loading, reason (one short line), detail\|null (the tooltip), to (the SPA tab whose confirmed button fixes it)}] (loop order), counts{current, stale, blocked, unknown}, errors{<source>: message}}`. It reads the existing read-only checks — the Synthetic › Status overview, the STARFULL roster and the local `ensemble_train` log, the `evaluation` / `knee` / `combiner` / `real-sr` / `records-noise` alerts, the newest NEXUS plate render and the galaxy-plot cache; a source that raises is named in `errors` and its stage reads `unknown`. Read-only: never starts a job. |

### TNG (`routes/tng.py`)

| Methods | Path | Gate | Notes |
|---|---|---|---|
| GET | `/api/tng/radii/status` |  | Read-only (never starts a job): the last TNG radius-manifest validation from the local cache — the validator payload (`valid`, `expected_count`, `valid_count`, `reasons`, `checked_at`, `failed?`) + `cached`, `stale` (cache missing, > 1 h old, or a failure > 5 min old), `connected`, `refresh_job` (id of a running validation job, else null). Clients POST `/api/tng/radii/refresh` when `stale && connected && !refresh_job`. |
| POST | `/api/tng/radii/refresh` | fasrc | Re-validate the remote manifest now (local job, `kind="tng-radii"`, at most one at a time — a running one's id is returned): `{ok, job_id}`; the result lands in the status cache (failures too, with `failed: true`). |
| POST | `/tng-auth/save` | fasrc | Write the TNG API token to FASRC. |
| GET | `/tng-auth/status` |  | Whether a TNG token file exists on FASRC (presence + length only); `connected:false` offline. |
| GET | `/api/tng/properties` |  | Synthetic › Galaxies TNG-template explorer, local CSVs only (`helpers/tng_explorer.py`; no SSH, no TNG API, no writes; memoised per file state): `{present, files{properties, atlas: {present, name, rows, mtime, size_bytes}}, atlas_meta (the atlas CSV's `.meta.json`)\|null, columns: [id, sfr (M☉/yr), mass_stars (M☉), m_halo (total bound mass, M☉), reff (group-catalogue stellar half-mass radius, kpc), re_kpc (mean measured VIS R_e over the viewpoints), re_kpc_min, re_kpc_max, n_orient, local (SKIRT FITS frames of that galaxy under data/tng_skirt/<id>/)], rows, orientations{<id>: [[orientation, native_re_px, native_re_kpc], …]}, summary{n, n_quenched (SFR = 0), n_missing_sfr, n_in_atlas, n_local}}`. Galaxies measured in the atlas but missing from `tng_properties.csv` use the atlas copy of their properties. |
| POST | `/api/tng/properties/refresh` | fasrc | Re-query the TNG API for downloaded galaxies missing from `tng_properties.csv` (local job `kind="tng-properties"`, one at a time): ids from the FASRC `.done` markers, the token from `$TNG_API_KEY` or the FASRC token file (in memory only). `{ok, job_id}`; result `{n_ids, n_resolved}`. Replaces the removed `GET /tng/histograms.png` (which wrote the cache from a GET). |
| GET | `/api/tng/results` |  | Which `tng_grid` / `tng_stack` results were pulled (local): `{grid: {present, pulled_at, size_bytes}, stack: {…}, pull_job}`. |
| POST | `/api/tng/result/pull` | fasrc | Pull the latest job result(s) from FASRC (`kind=grid\|stack\|all`, default all; local job `kind="tng-result"`, force-fetched; the stack with the large ePSF cap). A new grid image is archived into `data/vis/tng/` (Figures gallery). `{ok, job_id}`; result `{grid?: {ok, size_bytes\|error}, stack?: …}` (the job fails when nothing was pulled). 400 bad kind. |
| GET | `/tng/result/grid.png` |  | The last pulled `tng_grid` image — cache-only (never pulls, never writes); 404 `{ok:false, error}` until pulled. |
| GET | `/tng/result/stack.fits` |  | The last pulled `tng_stack` FITS as a download (`TNG_stack.fits`) — cache-only; 404 until pulled. |

### Job config (`routes/config.py`)

| Methods | Path | Gate | Notes |
|---|---|---|---|
| GET | `/api/config` |  | The universal job config (`~/.euclid_polish/job_config.json`): `{ok, config, version, defaults, types, used_by, steps}` — `version` is a content hash of the effective config; `defaults` every field's default (a reset = saving the default), `types` `{field: "int"\|"float"\|"str"}`, `used_by` `{field: [step_id…]}` (the FASRC steps its value is injected into, `job_config.FASRC_STEP_PARAMS` inverted; local-only knobs such as `asinh_scale` are absent), `steps` `{step_id: label}`. |
| POST | `/api/config/save` |  | Persist ONLY the posted job-config fields (unknown keys and blanks ignored) → `{ok, config, version, note}`. With `base_version` (the `version` the client loaded) a posted field changed server-side since then is refused **409** `{ok:false, code:"config_conflict", conflicts:{field:{base,current}}, config, version}`; fields the server did not change merge. An unknown `base_version` conflicts on every posted field whose value differs now. Without `base_version`: last write wins (legacy). |

### Local git (`routes/git.py`)

| Methods | Path | Gate | Notes |
|---|---|---|---|
| GET | `/api/git/commit/<rev>` |  | One commit (`rev` = 4–40 hex): `{ok, full, hash, author, email, date, subject, body, stat, patch (≤ 60 k chars), truncated}`; 400 for anything else. |
| GET | `/api/git/diff` |  | Local diff `{diff, staged, path}`: `?staged=1` for the index, `?path=` for one file/dir (taken literally; an untracked file's unstaged diff is its content; a path outside the repo gives ''). |
| GET | `/api/git/log` |  | One page of the history: `?skip=&limit=` (1–500, default 50; a non-integer is 400) → `{commits[{hash, full, author, date, relative, subject}], total, skip, limit, has_more}`. |
| GET | `/api/git/status` |  | `{status, log}`: local repo status + last 15 commits. `status.files` is `[{xy, path, orig, staged, unstaged, untracked, size, guard}]` from NUL-separated porcelain v1: `path` is the raw (unquoted) path — the **new** path of a rename, whose source is `orig` (else `null`); a wholly untracked directory is one `dir/` entry. `staged`/`unstaged` split the index and worktree columns; `guard` is the reason `/git/commit` would refuse the file without `force` (`file > 10 MB`, `untracked binary > 1 MB`) or `null`. Every listed `path` can be posted back to `/git/commit`, `/git/stage` and `/git/unstage` unchanged. |
| POST | `/git/commit` |  | Commit in the local repo: `message` + `paths` (repeatable; one value may hold several newline-separated paths; files or directories, taken literally — commas, spaces and non-ASCII are part of a name, exactly as `/api/git/status` lists them) **or** `all=1` — never an implicit `git add -A`. Exactly the selection is committed (`git commit --only -- <selection>`, literal pathspecs): a file staged earlier but outside `paths` stays staged and is **not** committed; with `all=1` every changed file, staged or not, is the selection and the fully staged index is committed as is (this also concludes a merge; a `paths` commit is refused by git mid-merge); a staged rename brings its source path. Files > 10 MB and new (untracked or staged-new) `.fits/.npy/.zip/.jpg/.png` > 1 MB are refused **409** `{ok:false, code:"refused_files", refused:[{path,size,reason}]}` unless `force=1`. 400 `no_selection` without paths/all; 400 `nothing_selected` when no changed file matches. `{ok, stdout, committed:[paths]}`. |
| POST | `/git/fetch` |  | `git fetch` in the local repo. |
| POST | `/git/pull` |  | `git pull` in the local repo. |
| POST | `/git/push` |  | `git push` from the local repo. |
| POST | `/git/stage` |  | `git add -A` exactly the changed files covered by `paths` (repeatable / newline-separated, literal): `{ok, staged}`; 400 `no_selection` / `nothing_selected`. |
| POST | `/git/unstage` |  | `git restore --staged` `paths` (the working-tree edits stay): `{ok, unstaged}`; 400 without paths or outside the repo. |

### Tracking / time-travel (`routes/tracking.py`)

| Methods | Path | Gate | Notes |
|---|---|---|---|
| POST | `/api/tracking/backup` |  | Back up a model (`kind=model`, `ckpt_dir` under the checkpoint root) / FITS (`kind=fits`, `path`) / image (`kind=image`) into the active campaign with `comment`, optional `name`: `{ok, record, warning (dirty commit), sync}`; best-effort push. |
| GET | `/api/tracking/campaign/<name>` |  | One campaign (`current` or an archived campaign's dir): `{ok, dir, active, metadata, backups{models, fits, images}, log_md, jobs_count}`; 404 unknown. |
| GET | `/api/tracking/jobs` |  | One page of a campaign's FASRC job records, newest first, **without** embedded payload blobs (`params` compacted, `params_omitted` `{key: size}`): `campaign` = `current` (default), an archived dir or `unassigned`; `q` (jobid/label/step/time); `offset`/`limit` (≤ 500, default 50) → `{ok, campaign, total, offset, limit, jobs}`; `ids=1` answers every job id of the campaign instead, unpaged, deduplicated, newest first (`{ok, campaign, total, jobids}`: Runs › History's campaign filter); 404 unknown campaign. |
| POST | `/api/tracking/log` |  | Append to / replace the active campaign's log (`text`, `mode`). |
| POST | `/api/tracking/new` |  | Create a campaign (`title`, `description`). |
| POST | `/api/tracking/save` |  | Archive the active campaign; best-effort holylabs push (`sync`). |
| GET | `/api/tracking/state` |  | Tracking store state: `{active, archived (each with its model backups), backups, jobs_count, unassigned_count, log_md, remote_dir, tracking_dir, ssh_connected, sandboxes}`. The job records are paged by `/api/tracking/jobs` (they embed ~200 KB of calibration JSON each). |
| POST | `/api/tracking/sync` | fasrc | Push the tracking store to holylabs (400 with the sync error). |
| POST | `/api/tracking/timetravel/open` |  | Start a time-travel sandbox server for a backup (`short`: the sandbox id — a hex commit short hash of an existing sandbox, else 400 JSON, for open / stop / remove alike). |
| POST | `/api/tracking/timetravel/remove` |  | Remove a time-travel sandbox (`short`; validated as for `open` — an empty or `.` id used to name the whole time-travel root). |
| POST | `/api/tracking/timetravel/restore` |  | Create a sandbox worktree at a campaign's (`campaign` = `current` or an archived dir) or one model backup's (`model`; a retired `.zip` restores its commit without seeding a checkpoint) commit and start its second console; the remote half (`remote=1`: FASRC worktree + netscratch sandbox) is optional (never gated). `{ok, short, url, spawn, remote, commit, warning}`. |
| POST | `/api/tracking/timetravel/stop` |  | Stop a time-travel server (`short`). |

### Provenance (`routes/provenance.py`, `helpers/provenance_index.py`)

A read-only lineage browser (System › Lineage) over every provenance record
found locally: `data/_prov` (flat), `<id8>.<kind>.json` sidecars under the
data dir, and checkpoint stamps (`<ckpt root>/[*/]*/provenance.json`, shown as
`checkpointartifact` entries named by their member). Upstream edges are
`parents`, `produced_by` and a process's `inputs`. The **verdict** of an
`srcutoutartifact` / `inferencerun` compares its model(s) (model-kind
parents/inputs, one hop through the producing run) with the active members'
ids: `current`, `stale`, or `unknown` (no model recorded — every legacy
product). The index is cached ~5 min (rebuilt early when `data/_prov`
changes). Local; JSON errors under `/api/provenance` (`errors.json_errors_for`).

| Methods | Path | Gate | Notes |
|---|---|---|---|
| GET | `/api/provenance/record/<pid>` |  | One record (`pid` = 8 hex; 400 otherwise, 404 unknown): `{entry (listing row), record (the stored JSON), upstream[{id, role: parent\|produced_by\|input, exists, kind, label, member}], downstream[…role child] (≤ 300), ancestors{total, items[… depth]}, descendants{total, items}, models[{id, member}], current_models, inspect_path (FITS path for /files, the SPA Files page; /inspect 308s there; else null)}`. |
| GET | `/api/provenance/records` |  | Search, newest first: `q` (whitespace tokens ANDed over id/kind/label/path/git/member/config type/status), `kind` (comma list), `verdict` (current\|stale\|unknown; 400 otherwise), `source` (prov\|sidecar\|checkpoint), `offset`/`limit` (≤ 1000, default 200) → `{total, offset, limit, records[{id, kind, category, source, file, created_at, status, path, format, label, git, dirty, config_type, seed, produced_by, parents, inputs, outputs, ra, dec, member, verdict, models, n_upstream, n_downstream}]}`. |
| POST | `/api/provenance/rebuild` |  | Re-scan every root now (synchronous); answers the new summary. |
| GET | `/api/provenance/summary` |  | `{total, counts{kinds, verdicts}, roots[{path, role: index\|data\|checkpoints, records}], current_models[{id, member, regime, dir}], truncated (the walk hit its 500 k-file cap), duplicates, built_at, build_seconds, building}`. The GETs never wait for a re-scan: after ~5 min (or when `data/_prov` changes) the previous index keeps being served while one background thread rebuilds it (`building: true`); only the first build blocks. |

### Cutouts (`routes/cutouts.py`)

| Methods | Path | Gate | Notes |
|---|---|---|---|
| GET | `/api/catalog/stars` |  | Synthetic › PSF star-catalogue explorer over the **FASRC-mirror** `stars.csv` (the synchronised `data/_fasrc_cache/…/euclid_stars/stars.csv`, 43 k stars) — never the stale 200-row `data/euclid_stars` copy; cache-only (no SSH), memoised per file state; the explicit pull is `POST /api/status/refresh-catalog`. `{present, source: "fasrc-mirror", path (remote), local_path, size_bytes, mtime (= last pull), age_s, columns: [id, ra, dec, mag, flux_uJy, fluxerr_uJy, field (EDF-N\|EDF-S\|EDF-F from the position, "" outside), b_VIS, b_Y_E, b_J_E, b_H_E, nav], rows, bands, sizes (cutout sides seen, ascending), bits{valid: 1, corrupted: 2, failed: 4, size_shift: 3}, summary{total, valid, corrupted, failed, pending (a star's overall state = its best band; the four sum to total), valid_all4, navigator{size, count}, mag_min, mag_max}, band_stats:[{band, valid, corrupted, failed, pending (exclusive: valid > corrupted > failed > pending, each row sums to total), by_size{<size>: valid count}}]}`. Band code `b_*`: bit 0 = a cutout at some size passed validation, bit 1 = downloaded but rejected (NaN/Inf, all-zero or constant, unopenable), bit 2 = download failed (no mosaic tile / bad coordinates), bit `3+i` = valid at `sizes[i]`; no bit = pending. `nav` = in the cutouts navigator (valid in all four bands at the navigator size). `present:false` with empty rows when the mirror was never pulled. |
| GET | `/api/cutouts/<band_name>/list.json` |  | Paginated per-band cutout files of the local cache (`page`, `per_page` 12–240, default 60): `{band, files, items:[{file, id, size, ra, dec, mag}] (star from the mirror; null when unknown), total, page, n_pages, per_page, output_dir}`. `output_dir` (default `Config.DEFAULT_OUTPUT_DIR`) must resolve inside the inspectable data roots (the Files jail), else 403 JSON. |
| GET | `/api/star-cutouts/totals` |  | Count/size of the stars valid in all four bands (the cutouts navigator) from the synchronised mirror — cache-only, works offline: `{count, size, cached: true, catalog{present, path, mtime, age_s}}`. |
| GET | `/cutout-image/<band_name>/<path:filename>` |  | One cutout FITS rendered as PNG (`size` 16–2048); `stretch=band` (default: the band's shared stretch) or `stretch=star` (the cutout's own min–max under a soft asinh: the Synthetic › PSF cutout gallery; any other value is 400); `output_dir` jailed like the listing (403 JSON outside the roots). |

### PSFs (`routes/psfs.py`)

| Methods | Path | Gate | Notes |
|---|---|---|---|
| GET | `/api/euclid-psf/inventory` |  | Synthetic › PSF ePSF inventory, local cache only (headers read, memoised per file state): `{bands:[{name, fwhm (config Gaussian fallback, ″), oversampling, epsf_pixel_scale, state: "empirical"\|"no_empirical"\|"not_cached", empirical, path?, size_bytes?, synced_at?, n_psf?, shape?, pixel_scale?, measured_fwhm?, error?, last_sync{ok, error?, missing_remote?, checked_at}\|null}], clusters:[{index, id: "cluster-NNN", ra, dec, n_stars, fwhm_by_band{band: ″\|null}}], clusters_source: "metadata"\|"vis_headers"\|null, clusters_meta{present, synced_at}, last_sync, generation{subset, kind, run, created, psf_kinds{band: "empirical"\|"gaussian"}}\|null}`. `generation` is what the last generation run recorded in its records' provenance (the newest local split whose pulled `<id>.skytfrecordartifact.json` sidecar carries `psf_kinds`; `null` for records generated before the stamp or none synced). `no_empirical` = the last sync found no ePSF for that band on FASRC (generation uses the Gaussian fallback); `not_cached` = not synchronised yet or the last sync failed otherwise (`error`). The per-band outcomes live in `<FASRC_CACHE_DIR>/euclid_psf_sync.json`. |
| POST | `/api/euclid-psf/sync` | fasrc | Force a re-rsync of the four band ePSFs (large ePSF cap) + the cluster metadata from FASRC — a local job (`kind="psf-sync"`, one at a time; a running sync's id comes back with `already_running`): `{ok, job_id}`; result `{ok, files{band: {ok, remote_path, size_bytes, error?, missing_remote?}}, clusters_meta}`; each band's outcome is recorded for the inventory. |
| POST | `/api/euclid-psf/sync-meta` | fasrc | Metadata-only sync (job, `kind="psf-sync"`): dump the per-cluster centroids, star counts and per-band FWHM (`fwhm_arcsec` = VIS, `fwhm_by_band`) from the ePSF headers on the login node, rsync the kilobyte JSON: `{ok, job_id}`; result `{ok, n_clusters, local_path}`. |

### Sky records, figures, diagnostics (`routes/views.py`)

| Methods | Path | Gate | Notes |
|---|---|---|---|
| POST | `/api/sky/generate-sr` |  | Run the production SR over the local dirty records (local job `kind="sky-generate-sr"`, one at a time): the production gate, restoring and running only the STARFULL members it reads (a pruned gate: its `active_members`), valid while every one of them is active (members that joined after the fit are logged as a note); the plain mean of every active STARFULL member, logged as a warning, when no current gate loads. `subsets` (comma list; default every split with dirty records; 400 unknown / no records), `overwrite=1` (deletes the split's cubes first; without it a split that has SR is skipped). Writes `<vis>/sky_sr/sr_<split>_NNNN.npy` + `sr_<split>.json` (model identity `{member_labels (the gate's full fitted list), combiner_kind, combiner_fingerprint, run_labels (the members that ran)}`, `model_label`, the input records' size/mtime, `count`, `generated_at`). `{ok, job_id, subsets, overwrite}`; result `{generated{split: n}, skipped, model, identity}`. 400 without records or active members. |
| GET | `/api/sky/records/source` |  | Every column of one truth source (`subset`, `index` = record position, `row` = its position among the record's sources): `{subset, field_index, row, source (the compact row), values{column: number\|string\|null; JSON trace columns decoded}}`; 404 `{ok:false, error}` when absent, 400 bad arguments. |
| GET | `/api/sky/records/sources` |  | Truth sources from the generator's `sources_<subset>.csv` (local). With `index` (record position = the CSV's `field_index`): `{subset, field_index, present, sources:[{row, type: galaxy\|star\|lens\|other, render, x_pix, y_pix (HR pixels, 0-based, pixel centres at integers), off_field, flux_vis_e, flux_y_e, flux_j_e, flux_h_e, mag_vis (a star's sampled magnitude, a galaxy's achieved 2FWHM magnitude, else its target), mag_y_e, mag_j_e, mag_h_e, target_vis_mag, z, re_arcsec, theta_E_arcsec, orientation, temperature_k, subhalo_id, source_subhalo_id, sfr_class}], counts{galaxy, star, lens, other, off_field}, geometry}`; without it the split census `{subset, present, fields:[{field_index, galaxy, star, lens, other, off_field, n, brightest_star_mag, brightest_galaxy_mag, total_vis_e}], geometry}`. `geometry{hr{height, width, pixscale}\|null, lr{…}\|null}` from the first hr (else clean) / dirty record. 400 bad subset/index. |
| GET | `/api/sky/sr-status` |  | Synthetic › Records state (local, headers only): `{records, checkpoint, can_generate, subsets (splits with dirty records), sr{split: n cubes}, records_dir, splits{split: {files{dirty\|hr\|clean: {name, size_bytes, mtime, count (null = truncated/corrupt)}\|null, sources: {name, size_bytes, mtime}\|null}, count, present, sr{state: current\|stale\|partial\|missing\|unknown, reasons[], count, records_count, manifest\|null}}}, model (the identity an SR run would load now)\|null, sync_job, generate_job}`. `unknown` = SR cubes without a manifest (made before model tracking). The records half of `stale` compares a content fingerprint (SHA-1 over every frame header + payload CRC, recorded in the manifest's `records.fingerprint`), never the mtime: a re-sync of unchanged records keeps the SR `current`; a legacy manifest without a fingerprint compares the size only. |
| POST | `/api/sky/sync` | fasrc | Rsync the synthetic records from FASRC into the local cache as a local job (`kind="sky-sync"`, one at a time — a running sync's id comes back with `already_running`). `subsets` (comma list; default `test,validate`, `include_train=1` adds `train`), `kinds` (comma list of `dirty,hr,clean,sources`; default all). Each pulled record file's provenance sidecar (`<id>.skytfrecordartifact.json`, beside it on FASRC) is pulled too, best-effort (`files.<key>.provenance`). 5 GB pull cap; the requested files are protected from the cache LRU during the sync. `{ok, job_id, subsets, files}`; result `{ok, files{<kind>_<split>: {ok, size_bytes, error?}}, subsets, include_train}` (the job fails when nothing was pulled). 400 bad subsets/kinds. |
| GET | `/api/vis/list.json` |  | The `data/vis/` PNG gallery (newest first). |

### Synthetic status (`routes/realism.py`, `helpers/realism_overview.py`)

| Methods | Path | Gate | Notes |
|---|---|---|---|
| GET | `/api/realism/overview` |  | Read-only readiness of every synthetic-realism prior (Synthetic › Status): `{computed_at, gate{step: "synthetic_generate", ready, blockers:[{id, message}], message (the step's own first refusal)\|null, to}, items:[{id, label, state: ok\|warn\|bad\|unknown, title, detail\|null, to (SPA path)\|null, action{label, method: "POST", url, params, confirm\|null, requires_fasrc (a `@requires_fasrc` endpoint: the UI disables it offline), self_connects (a local job that connects itself and reports failure: enabled offline), requires_login}\|null, facts{…}}], counts{ok, warn, bad, unknown}, authenticated, training{available, population_fields, population_fields_with_training, sync (action)}}`. Items, in order: `galaxy-model` (active / candidate / unfitted joint galaxy model; action = activate), `star-prior` (same for the stellar prior), `tng-radii` (the last remote TNG radius-manifest validation cached by `POST /api/tng/radii/refresh`; `facts.stale` past its TTL), `noise-model` (`Config.NOISE_MODEL` and the committed noise-level table), `records-noise` (the Home `records-noise` check: local TFRecords' generation-run noise model vs `Config.NOISE_MODEL`), `galaxy-plots` (the galaxy plot artifact's schema/input freshness; action = build), `comparison-cache` (field-statistics cache freshness; action = build), `archive-fields` (multipoint archive reference; action = sync), `training-catalog` (`sources_train.csv`; action = the one training sync). The gate lists EVERY blocker `SyntheticGenerateStep.prepare_params` would raise (a parity test holds them together). Nothing is written. |

### Noise (`routes/noise.py`)

| Methods | Path | Gate | Notes |
|---|---|---|---|
| GET | `/api/noise` |  | Q1 noise-level table payload (committed data). |
| GET | `/api/noise/positions/<tile>` |  | One measured position (the `noisepos` inspector): `{tile, field, ra, dec, bands, levels_e{band: e⁻}, sub_levels_e{band: [16 levels]}\|null, grid_side\|null, steps{band: {step, scatter, seam}\|null}, step_threshold, uniformity_threshold, noise_model}` — `step` is the largest straight-line depth step of the band's 4×4 sub-tile grid, `seam` when it passes both thresholds. 404 `{error}` for an unknown tile. |

### Galaxy distributions (`routes/galaxy_distributions.py`)

| Methods | Path | Gate | Notes |
|---|---|---|---|
| GET | `/api/galaxy-distributions` |  | Galaxy population payload: distributions, availability, auth state, Q1 count/radius state (`?include_training=`). Read-only: a stale plot artifact is reported (`stale`), rebuilt only by `POST …/build`. |
| POST | `/api/galaxy-distributions/activate` |  | Activate the fitted joint galaxy candidate (local job). |
| POST | `/api/galaxy-distributions/build` |  | Build the galaxy plot data (local job). |
| POST | `/api/galaxy-distributions/fit-q1-counts` |  | Fit the cached Q1 aperture counts (local job; 400 until queried). |
| POST | `/api/galaxy-distributions/fit` |  | Refit the galaxy prior from the cached Q1 brackets: the VIS 2FWHM line, the joint size + colour/SFR candidate, then the plots (local job; no archive query or login; 400 in words until every aperture checkpoint and Rₑ bracket is cached). |
| GET | `/api/galaxy-distributions/joint-pair` |  | One pair-explorer grid (`x`, `y`). |
| POST | `/api/galaxy-distributions/query-q1-counts` |  | Query Q1 MER + PHZ counts (Euclid archive login required; local job). |
| POST | `/api/galaxy-distributions/refresh-population-cones` |  | Re-query the saved 24-cone population footprint (local job). |
| GET | `/view/galaxy-distribution-plate` |  | Download the four-panel galaxy population diagnostic. |
| GET | `/view/population-atlas` |  | Download the reviewed Euclid brightness–radius fit. |

### Star distribution (`routes/star_distribution.py`)

| Methods | Path | Gate | Notes |
|---|---|---|---|
| GET | `/api/star-distribution` |  | Stellar population payload: colour sample, calibration, distribution, availability (`?include_training=`). `distribution` holds the Gaia–Euclid colour-sample counts (`matched_stars`, `high_quality_stars`, `pointlike_over_0_9`), `density_comparison` (the VIS panel with Q1 PHZ / point sources, the law and the generated stars, and six Euclid colour panels; no Gaia series anywhere: no `gaia`, `gaia_x`, `gaia_fit`, `fit_ranges.gaia` or `gaia_native_g_count` since 2026-09-28) and `gaia_sampling`; the Gaia colour, CMD and projection views are deleted (no `colors`, `gaia_cmd`, `euclid_projection`). Read-only: a missing or stale plot cache is computed in memory (memoised per calibration + source signature); only the fit job writes `star_distribution*.json`. |
| POST | `/api/star-distribution/activate` |  | Activate the fitted stellar candidate (local job). |
| POST | `/api/star-distribution/fit` |  | Fit the stellar prior from the cached colour sample (local job); then persists both plot variants (with and without the training catalogue). |
| POST | `/api/star-distribution/query` |  | Query stars (MER + PHZ + Gaia; Euclid archive login required; local job). |
| GET | `/view/star-population-calibration` |  | Stellar calibration plate (`?format=` png, pdf or svg; `?dpi=`; `?inline=`): the density panel (Q1 PHZ VIS counts and the Q1-normalised law; no Gaia series — only `paper_figures/build_figures.py` asks `render_star_population_calibration(include_gaia=True)` for the native Gaia G_AB counts) and the VIS − Y, Y − J, J − H colour PDFs. |

### Population comparison (`routes/population_comparison.py`)

| Methods | Path | Gate | Notes |
|---|---|---|---|
| GET | `/api/population-comparison` |  | Pixel-statistics comparison payload (`?include_training=`). |
| POST | `/api/population-comparison/build` |  | Build the local-field comparison (local job). |
| POST | `/api/population-comparison/sync-training-catalog` |  | Pull `sources_train.csv` from FASRC in a local job that self-connects (`ensure_ssh_connected`) and reports failure in the job (never gated), then refresh the population census. `rebuild=1` (the Synthetic header's one sync action) also rebuilds the galaxy plots in the same job so their training variant exists; result `{path, size_bytes, galaxy_plots_version\|null}`. |

### Archive fields (`routes/archive_fields.py`)

| Methods | Path | Gate | Notes |
|---|---|---|---|
| GET | `/api/archive-fields` |  | Archive-field collection availability (local). `fields` / `comparison_fields` count samples by the **position-derived** Q1 field (`q1_field_for`; the stored EDF-F/EDF-S labels were swapped); `stored_fields` keeps the manifest labels. |
| POST | `/api/archive-fields/sync` |  | Sync the archive-field collection from FASRC in a local job that self-connects and reports failure (never gated); 409 while one runs. |

### Real-field inference (`routes/model.py`)

| Methods | Path | Gate | Notes |
|---|---|---|---|
| GET | `/api/inference/diagnostics.json` |  | Diagnostics of the latest cached real field (`{diagnostics: {version, member_labels (the members that ran), member_scope ("gate" = the production gate's read members, "all"), n_ensemble_members, model_power{k, r_pairs, r_cross, pixel_scale_arcsec}, std_brightness{x_edges, y_edges, counts, x_label, y_label}, combiners{kind: occupancy}} \| null}`; shown by Models › Diagnostics (real field) beside `/ensemble/evals.json?mode=starfull`). |
| GET | `/api/inference/field.json` |  | Latest cached real field + field size. The field manifest's `member_labels` is the whole active STARFULL membership (member cubes are indexed by position in it); `run_members` / `run_member_labels` are the members that ran and `member_scope` is `"gate"` (only the production gate's read members, the default) or `"all"`; mean, std, PCA and diagnostics cover the members that ran. Manifests from before scopes lack these keys and ran every member. |
| POST | `/inference/refresh-combiners` |  | Apply the newest STARFULL combiner to cached fields (local job). Runs only the members the production gate reads; `all_members=1` runs every active member (full member diagnostics). Member cubes already cached are reused and never deleted. |

### Ensemble (`routes/ensemble.py`)

The Models workspace (`/models/:mode/<tab>`, was `/ensemble/:mode/<tab>`; its data endpoints keep the `/ensemble/` prefix). Every route
takes `mode` = `starfull` (default) | `starless` (query or form). Errors under
`/ensemble/` are JSON `{error}` (`errors.json_errors_for`); the new endpoints
answer `{ok:false, error}` with 400 on a bad knob. Everything is local except
`/ensemble/pull` (FASRC). Local jobs return `{job_id}` (C2); kinds
`ensemble-compare`, `gate-fit`, `gate-promote`, `member-restore`.

- **Eval summary** (`<vis>/ensemble/<regime>/eval_summary.json`, both the
  full evaluation and the rebuild from cached cubes): every headline number is
  the **VIS asinh PSNR** (knee `psnr_knee_e` = `Config.STRETCH_SCALE_E`,
  `psnr_metric: "vis_asinh"`) over `n_scored` test fields: `ensemble_psnr`
  (plain mean of the members), `mean_member_psnr`, `best_member_psnr` +
  `best_member_label`, `per_member_vis_psnr`, `ensemble_vs_mean_member_db`,
  `ensemble_vs_best_member_db`, **`ensemble_gain_db` = the gain over the MEAN
  member** (one meaning everywhere; `EnsembleModel.evaluate` defines it the
  same way and adds `ensemble_vs_best_member_db`). Per combiner kind:
  `<kind>_combiner_psnr`, `…_vs_mean_db` (over the ensemble mean),
  `…_vs_best_member_db`, `…_vs_mean_member_db`. The production combiner is the
  spatial gate (`spatial_gate_*` keys); the bare `combiner_psnr` /
  `combiner_vs_mean_db` keys are the RBF's (kept for older readers). A full
  evaluation keeps `EnsembleModel`'s raw-electron numbers under `*_raw_e`.
- **Member row** (`members.json`, `member/<name>.json` `row`): `{name, label
  ("NN·psnr"), starless, regime, origin (origin.json), op, forked_from, loss,
  blocks, asinh_knee, asinh_knees, output_knee, knee_loss, noise_aug,
  bootstrap, icnr, seed, commit, created_at, noise_model, step, target_steps,
  fraction, status: complete|timeout|running|unknown, timeout (below target
  after its job ended), job{jobid, state, submitted_at, ended_at,
  elapsed_seconds, req_time_limit, gpu_util_mean, mode}|null (the newest
  ensemble_train submission that created/continued it, from the local job
  log), psnr (cached test PSNR), psnr_rank, knee_integrated{VIS,Y_E,J_E,H_E,
  mean}|null + knee_rank (from the knee payload), gate_usage{band} /
  gate_usage_source{band} (production gate, all / source pixels) |null,
  gate_usage_peak{value, band, bin (brightness bin, "sources", or null =
  all pixels)}|null (the member's largest share of the production gate's
  weight over bands, all/source pixels and every brightness bin — the
  "used by the gate" rule's peak), used_by_gate (the production gate reads
  it, so production SR runs it)|null (no gate payload),
  coherence{overall, sr}|null, has_loss_best, size_mb}`.
- **Variant row** (`combiners.json` `variants[]`; spatial-gate variants only —
  the legacy RBF has no row): `{name (directory), kind: gate, spec
  (production | gate:<x>), production, backup
  (spatial_gate_backup_*), member_labels, reads, n_members, n_reads, pruned,
  mix_space, use_lr, width, fitted_at, fingerprint, membership{current
  (every member it READS is active), missing (fitted members not active),
  missing_reads (read members not active), extra (active members that joined
  after the fit)}, promotion{ok, reason} (whether promote would install it:
  refused while a fit writes it (`.fitting.json` marker of a live process),
  when `fit_meta.complete` is not true, or — without `force` — when a
  member it reads is not active), applies_to_test_cubes (a gate: the test
  cubes hold every member it reads), fit
  {steps, steps_run, complete, batch_size, crop, learning_rate, loss,
  loss_knees_e, blackout_fields, fit_seconds, seed, eval_every, variant,
  fitted_via, members_requested, used_threshold (the `--members used` rule's
  peak-weight threshold, when that picked the members), promoted_from, …,
  train_field_count, holdout_field_count},
  selected, baseline, history[{step, loss, train_loss, vis_psnr, band_psnr,
  integrated_psnr}], test{source:"compare", report, band_psnr,
  blackout_band_psnr}|null, knee{source: "knee"|"compare", integrated[band],
  psnr[knee][band]}|null, eval{psnr, vs_mean_db, vs_best_member_db}
  (production only)}`.
- **Compare report** (`scripts/fit_spatial_gate.py compare`, now
  `eval/spatial_gate_compare.py`; schema 2): `{id, created, regime, members,
  bands, brightness_names, methods ["mean", "gate:<dir>"…] (reports written
  before 2026-09-27 may also hold "rbf"),
  method_labels, method_members, gates{method: selected}, n_fields{natural,
  blackout}, groups{natural|blackout: {method|member:<label>: {band_psnr[4],
  bin_mse[5], halo_mse[4], hole_mse[4]}}}, usage{gate: {labels, all_pixels,
  source_pixels}}, knee{knees, n_fields, methods{method: {psnr[knee][band],
  integrated[band]}}} (natural fields; same grid as knee-psnr.json),
  timing_s, members_needed, member_inference_s_per_field, blackout_seed,
  gates_requested}`. A variant fitted for a subset of the cube members is
  applied to exactly its members.

| Methods | Path | Gate | Notes |
|---|---|---|---|
| POST | `/ensemble/archive-member` |  | Retire one member (`member`: any member spelling): zip → tracking campaign, registry tombstone, member dir deleted, cube cache purged. The name is validated and must be ACTIVE before the job starts (400 `{ok:false, error}`); `{ok, job_id}`. |
| GET | `/ensemble/combiners.json` |  | Variant registry: `{regime, production (dir), active_members, cube_members, variants:[variant row], compare:{id, created, methods, n_fields}\|null, reports:[{id, created, methods, n_fields, gates_requested}]}`. |
| POST | `/ensemble/combiners/compare` |  | Compare job: `gates` (comma list of `spatial_gate_*` dirs or `gate:<x>`; default every variant that applies to the test cubes, backups excluded), `blackout_fields` (0–400, default 40), `seed`, `knee` (default 1). The legacy RBF is never scored (`include_rbf` is ignored). Writes `spatial_gate_comparisons/<id>.json` + the latest `spatial_gate_comparison.json`. Job result `{report_id, methods, n_fields}`. |
| GET | `/ensemble/combiners/compare.json` |  | One compare report (`?report=<id>`, default the latest); 404 before any compare. |
| POST | `/ensemble/combiners/fit` |  | Fit a NAMED gate variant (job, TensorFlow): `out_name` (`spatial_gate_<x>` or `<x>`; never `spatial_gate_combiner`, never `spatial_gate_backup_*`, never `spatial_gate_comparison*` or a name ending `.json` / `_evals` (compare reports and eval sidecars share the prefix); an existing VARIANT needs `overwrite=1`, any other existing entry is refused), `mix_space` (`linear` default \| `asinh`), `loss_knees` (`all` = 11 knees 0.1–1e4 e⁻ default \| `band` \| comma list), `use_lr`, `width` (32), `steps` (2000), `batch_size` (8), `crop` (192), `learning_rate` (2e-3), `eval_every` (250), `holdout` (15), `blackout_fields` (40), `seed` (0), `members` (subset → pruned gate), `num_images` (validate fields, 100), `target_psf_fwhm_arcsec`, `compare_after` (default 1: compare with production afterwards). The validate member cubes are re-inferred when stale. `{ok, job_id, variant}`; job result `{variant, n_members, selected, report_id}`. |
| POST | `/ensemble/combiners/promote` |  | Promote job: `variant` (dir or `gate:<x>`), `force` (required when the variant reads a member that is not active; members that joined after its fit need no force). A variant a fit is still writing, or whose `fit_meta.complete` is not true, is always refused (the job fails with the reason). Backs the current production gate up to `spatial_gate_backup_<UTC stamp>` (promote a backup to roll back), swaps the variant in, then refreshes the gate payload and — when the variant fits the cached test cubes — re-applies it and rebuilds the eval summary + knee curves (no member inference). Job result `{promoted, backup, test_rescored, summary}`. |
| GET | `/ensemble/evals.json` |  | The Diagnostics dataset: power spectrum (+ T(k)), coherence, std-vs-error, std-vs-brightness, calibration (`z_edges, pdf, stats{cover1..3, sigma_z}, field_std, field_rmse`), per-member meta (no RBF combiner-axes block since the 2026-09-27 rework; an older cached payload may still carry `combiner_feature_error`). 404 JSON before an evaluation. `?fresh=1` recomputes from the cached cubes for a same-origin request only; a cross-site request gets the cached payload as it is (no diagnostics upgrade) or 404 when none exists. Every payload names its `band` and `guides{band, psf_fwhm, read_noise, lr_scale, theta_min, vis_fwhm, rn_vis}`. `?band=Y_E\|J_E\|H_E` serves that band's payload (same schema, no member-pair spectra, plus `identity` and `stale`: computed for an earlier evaluation) from `ensemble_evals_<band>.json` — cache only, 404 in words until computed; 400 for another band. |
| POST | `/ensemble/evals/bands` |  | Compute the Y, J and H diagnostics from the cached test cubes (local job `ensemble-band-evals`, several minutes, no model inference; one at a time: a running one's id comes back with `already_running`) → `{ok, job_id, already_running}`; 400 in words without a cached evaluation. Evaluate refreshes them too, unless every band is current. |
| POST | `/ensemble/evaluate` |  | Evaluate the ensemble on local test records (local job; `num_images`, `mode`, `force=1` re-infers even when an identical evaluation is cached, `target_psf_fwhm_arcsec`). Archives queued since the last evaluation are applied first from the cached cubes: the production spatial gate is left byte-identical while every member it reads stays active (archiving a member it does not read keeps its fingerprint, so its outputs stay current), and is moved to `spatial_gate_backup_<UTC stamp>` (promotable again once the member is restored) when a member it reads was archived. The spatial gate is applied and scored on the test cubes by label (members that joined after its fit are skipped); the RBF needs exactly the active members. |
| POST | `/ensemble/knee-psnr` |  | Compute PSNR-vs-knee curves (local job; `mode`). |
| GET | `/ensemble/knee-psnr.json` |  | PSNR-vs-knee curves + integrated PSNR for every model of a regime (``?mode=``), flagged ``stale`` when the cubes or combiners changed. |
| GET | `/ensemble/member/<name>.json` |  | One member (`member_196`, `196`, `196·psnr`): `{name, label, active, archived (tombstone row)\|null, regime, row (member row)\|null, curves{psnr, band_psnr, loss_series, train_loss, gnorm, gnorm_max, step_time}\|null, knee{knees, bands, stale, models:[this member, the mean, the combiners]}\|null, gate{stale, bands, brightness_names, usage, usage_source, by_brightness, uniform}\|null}`. 400 bad name, 404 unknown. |
| POST | `/ensemble/member-psnr` |  | Refresh the members table's test PSNRs (asinh space). |
| GET | `/ensemble/members.json` |  | Members tab: `{regime, members:[member row] (active, this regime), other_regime_members, archived:[{name, archived_at, zip, commit, zip_found, zip_path, campaign, size_bytes}] newest first, knee{available, stale, n_fields}, gate{available, stale, n_members}, psnr_fields, vis_psnr{metric, knee_e, n_scored}\|null, eval_subset}`. Two per-member test PSNRs, never mixed: row `psnr` (+`psnr_rank`) = the member-PSNR cache, joint 4-band asinh `psnr_stretched` over `psnr_fields` test fields; row `vis_psnr` = the last evaluation's headline metric (VIS asinh, `eval_summary.per_member_vis_psnr`, or `per_member_psnr_stretched` of a cube-recomputed summary), the Overview "Best member" number. |
| GET | `/ensemble/overview.json` |  | Models › Leaderboard's status and verdict lines: `{regime, active_members, n_members, records_dir, eval_subset, test_present, evaluated_at, summary (eval_summary.json)\|null, headline{metric, knee_e, n_scored, production{psnr, vs_mean_db, vs_best_member_db}, mean{psnr, vs_mean_member_db}, best_member{psnr, label, mean_member_psnr}, knee{available, stale, n_fields, integration, production, production_bands, mean, best_member, best_member_label}}, checks:[{id, ok, tone, title, detail, action}] (eval vs members / records / production gate, gate vs members, knee, pending archives), production_gate{available, n_members, mix_space, fitted_at, promoted_from}}`. |
| GET | `/ensemble/pixel-trace.json` |  | Back-trace a diagnostic heatmap cell to real image stamps (`?mode=`, `diag` = `std_err` or `bright_std`, `model`, `i`, `j`, `band` = VIS (default) or Y_E/J_E/H_E: that band's back-tracing sidecar and per-pixel numbers; any other `diag` is 404, another band 400). |
| POST | `/ensemble/pull` | fasrc | Download changed members from FASRC (local job): `members` (comma list) limits it to those; `dry_run=1` only probes — job result `{dry_run, changed, tombstoned_skipped}`; else `{local, n_members, changed, up_to_date, requested, psnr}`. |
| POST | `/ensemble/restore-member` |  | Restore an archived member (`member`) from its tracking zip (searched in the active and every archived campaign; zip-slip refused): unzip into the ensemble dir, tombstone → active. Job result `{member, zip, regime}`. |
| GET | `/ensemble/status.json` |  | Members table + summary payload; `?mode=` picks the regime's eval summary + staleness (default `starfull`). Home reads it; the workspace uses `members.json`. |
| POST | `/ensemble/train/preview` |  | What an `ensemble_train` submit with this form would run, without FASRC: `{ok, mode, member_names (allocated from the local registry, tombstones never reused), count, array{tasks, max_parallel}\|null, command[argv], command_text, base_seed (null = drawn at submit), star_prior, params}`; 400 `{ok:false, error}` for a form the submit would refuse. The star regime is the workspace's: the Train tab sends the run-wide `starless=1` for add/fork from the starless workspace and never a per-member `starless` (a fork keeps its source's regime, continue each member's recorded one). |
| GET | `/ensemble/training-curves.json` |  | `{members:[{name, label, starless, psnr, band_psnr{VIS,Y_E,J_E,H_E}, loss_series, loss (= loss_series, deprecated), train_loss, gnorm, gnorm_max, step_time (s / 1000 steps), loss_norm, blocks, asinh_knee, asinh_knees, output_knee, knee_loss, target_steps, test_psnr}]}` — `[[step, value]…]` series, registry-active members only (rollback-deduped). |
| GET | `/ensemble/training-jobs.json` |  | `{jobs:[{jobid, state, submitted_at, started_at, ended_at, elapsed_seconds, req_time_limit, req_memory, req_cpus, req_gpus, partition, gpu_util_mean, mode, member_names, steps, continue_basis, target_steps, extra_steps, params}]}` — every ensemble_train submission in the local job log, newest first (the Train tab's presets / clone). |

### Evaluation (`routes/evaluation.py`)

| Methods | Path | Gate | Notes |
|---|---|---|---|
| GET, POST | `/api/evaluation/angular-power-spectrum` |  | The per-band HR-vs-SR angular power-spectrum PNG: served from its cache (`<eval_results>/angular_power_spectrum.png`), rendered when missing. `POST` re-renders → `{ok, rendered}`; `GET ?fresh=1` re-renders only for a same-origin request (a cross-site one gets the cache, or 404 when nothing is rendered yet — it never renders). 404 `{ok:false, error}` until the validation records and their SR cube exist. Every render also writes the curves to `angular_power_spectrum.json` beside the PNG. |
| GET | `/api/evaluation/angular-power-spectrum.json` |  | The angular power spectrum as curves (Models › Diagnostics › Recovery draws them): `{subset, n_fields, field_n, pixel_scale, lr_scale, theta_max, band_names, bands{<band>: {psf_fwhm, linear\|asinh: {theta, T, T_lo, T_hi, r, r_lo, r_hi, count}}}}` (θ = 1/2k in arcsec; per-field median and 16–84%; `null` for empty bins). Cache only: 404 in words until a render (`POST /api/evaluation/angular-power-spectrum`) wrote it. |
| POST | `/api/evaluation/fetch-catalog` |  | Download + normalize the Euclid Q1 strong-lens catalog (Zenodo). |
| GET | `/api/evaluation/objects/<object_id>` |  | One object's provenance card (`object_id` = `out_subdir`; `?run=` as for runs): the enriched manifest row (below) + `row` (raw), `members` (its `members.json`: `member_labels`, `combiner_kind`, `combiner_fingerprint`), `current` (the model an evaluation would load now), `disagreement` (`disagreement.json`), `files[{name, bytes, mtime}]`, `provenance` (the SR's `*.srcutoutartifact.json` sidecars, newest first: `id, created_at, produced_by, git, dirty, descriptors, file`), `sr_header` (provenance/geometry cards of `SR.fits`), `downloads{tier: /eval-files/…}`, `viewer{collection:"evaluation", id}`. 400 bad id, 404 unknown object. |
| POST | `/api/evaluation/query-galaxies` |  | Query + cache the real-galaxy eval catalog as its own LOCAL step. |
| POST | `/api/evaluation/rerender` |  | Drop a run's cached eye/solar PNGs so they re-render from the FITS. |
| POST | `/api/evaluation/run-grouped` |  | Prepare the unified grouped dataset LOCALLY (A/B/C + synthetic) with the STARFULL members through the production combiner (member mean when no current combiner loads). Objects are reused only while their `members.json` records the same STARFULL members AND production combiner (kind + artifact fingerprint); synthetic stamps reuse the ensemble page's cached STARFULL member stacks through the production combiner. |
| GET | `/api/evaluation/runs` |  | Summary of one evaluation run (`?run=` a sub-directory; default the shared store): `{name, run, n, n_ok, mtime, columns (the manifest columns), rows, current{n_members, member_labels, combiner_kind, combiner_fingerprint}, counts{current, stale, unknown} (ok rows), groups{grade: n} (ok rows)}`. Each row is the manifest row + `kind` (`lens`\|`galaxy`\|`synthetic`), `field` (position-derived Q1 field or `null`), `viewer_id` (the `evaluation` collection object id = the row's `out_subdir`, which the manifest `id` may be sanitised into), `realtile` (`eval/<out_subdir>` for a real object, else `null`), `tiers` (object FITS present: `LR, SR, mean, HR, BHR, std`), and the SR's model `state` against `current` — `current` (same STARFULL members + production combiner), `stale` (`state_reason`: membership or combiner changed, or no combiner recorded), `unknown` (no `members.json`), `null` for a failed row — with the recorded `n_members`, `combiner_kind`. 400 bad run name, 404 missing run (JSON). |
| POST | `/api/evaluation/sync` | fasrc | Pull `<data_dir>/eval_results` from FASRC (`rsync --delete-after`, which also deletes local results the cluster lacks): requires `confirm=1`, else **400** `confirm_required`. |
| GET, POST | `/api/evaluation/transformation` |  | The run-level SR-transformation summary PNG: served from `<run>/transformation_summary.png`, rendered when missing. `POST` re-renders → `{ok, rendered}`; `GET ?fresh=1` re-renders only for a same-origin request (a cross-site GET never renders: cache or 404). 404 `{ok:false, error}` without synthetic objects. |
| GET | `/eval-files/<path:relpath>` |  | Download one per-object `.fits` under `eval_results/` (attachment, `application/fits`). Jailed: 403 `{ok:false,error}` outside the tree; 404 for anything that is not an existing FITS (the classic PNG renderer is gone). |

### JWST × Euclid (`routes/jwst_euclid.py`)

| Methods | Path | Gate | Notes |
|---|---|---|---|
| POST | `/api/jwst-euclid/download` |  | Download + align one JWST × Euclid pair (local job; MAST/Euclid archive, not FASRC). The Euclid VIS side is the archive tile covering the JWST position. |
| POST | `/api/jwst-euclid/download-all` |  | Download every remaining pair (local job). |
| GET | `/api/jwst-euclid/field/<identifier>/download/<kind>` |  | Download one paired-field FITS asset. |
| POST | `/api/jwst-euclid/infer` |  | Run the production model on a saved pair (local job): builds / reuses the pair's four-band LR input (`starfull_inference/euclid_lr_vis_y_j_h.fits`, bands cut from the containing Q1 tile), writes `starfull_inference/starfull_combiner.fits` (SR WCS = LR ×2). |
| POST | `/api/jwst-euclid/nexus/download` |  | Download one NEXUS tile at (`ra`, `dec`) (local job, kind `nexus-mosaic`: one at a time — the same pair re-attaches with `already_running`, another NEXUS mosaic job running → 409 `code:"busy"`); Euclid VIS from the Q1 tile whose polygon contains the point. `{ok, job_id, field_id}`. The mosaic crop streams only its own rows. |
| POST | `/api/jwst-euclid/nexus/download-field` |  | Cache a NEXUS mosaic + four-band Euclid coverage (local job). Every band is cut from the Q1 MER tile whose polygon contains the tile centre (committed `q1_mer_tiles.json`; never the nearest tile centre); the manifest records each tile's `polygon` (VIS grid corners), `euclid_tile_index` and the mosaic grid `footprint`. Single-run (kind `nexus-mosaic`, key = filter): `{ok, job_id}`, `already_running` for the same filter, 409 `code:"busy"` while another NEXUS mosaic job runs. The tile scan reads one full-width row strip at a time (never the whole plane). |
| GET | `/api/jwst-euclid/nexus/fields` |  | Cached NEXUS fields. |
| POST | `/api/jwst-euclid/nexus/infer` |  | Run ONE model spec on a saved NEXUS field (local job, `kind="nexus-inference"`): `field_id`, optional `tiles` (comma list of real-tile ids `f200w-NNNN` or source indices; blank = every tile) and `spec` (a C9 spec, default `production`: the spatial gate fitted for the current STARFULL members; no RBF fallback). `production` replaces the stale per-tile SRs in `tiles/` (served by C9 as legacy outputs — `m:production` on `/api/real/nexus/*` and the `real` viewer); any other spec writes the C9 output store. SR WCS = LR WCS ×2 (`CRPIX → 2·CRPIX − 0.5`). `{ok, job_id, field_id, tiles (list or null), spec}`; **400** `{ok:false,error}` for an unknown tile, a malformed / unknown spec, more than one spec, or a spec that cannot run now (its `reason`); 404 unknown field. |

### Viewer (`routes/viewer.py`)

| Methods | Path | Gate | Notes |
|---|---|---|---|
| GET | `/viewer/cube/<collection>` |  | The same cube addressed by object id: `?id=<meta object id>&tier=…` (404 unknown id, 400 without `id`). |
| GET | `/viewer/cube/<collection>/<int:index>` |  | Raw little-endian float32 `(H, W, C)` cube of one object/tier (`?tier=` + collection params) with `X-Cube-*` headers, incl. `X-Cube-WCS` / `X-Cube-Unit` / `X-Cube-Index` (C6, see *Viewer collections*). JSON `{error}` on failure. |
| GET | `/viewer/meta/<collection>` |  | Collection metadata: `count, tiers[{key,label,unit?,hidden?,disabled?,hint?}], default_tier, band_names, objects[{id, label, ra?, dec?, tiers?…}]`, colour constants; `?id=` adds `index`; `no-cache`. JSON `{error}` on failure (C6). |
| GET, POST | `/viewer/results` |  | GET: `{schema_version, axis_defaults, limits{max_results, max_rows}, supported{logical_tiers, modes, transfer, dpi}, results:[summary]}` newest first. POST: save a crop/result (JSON or form; *Saved viewer results* below) → 201 `{id, result_id, result: summary}`. |
| GET | `/viewer/results/<result_id>` |  | `{result: summary}`. |
| GET | `/viewer/results/<result_id>/<logical>.fits` |  | Download one saved crop (`dirty`, `sr`, `hr`, `jwst`; checksum-verified; attachment). 400 bad tier, 404 absent, 409 checksum mismatch. |
| POST | `/viewer/results/<result_id>/delete` |  | Delete one saved result (and drop it from every grid layout): `{ok, id}`; 404 unknown. |
| GET | `/viewer/results/<result_id>/panel.png` |  | PNG panel of one saved result: `tier` + `mode` (`VIS` \| `Y_E` \| `J_E` \| `H_E` — one band in grey —, `VIS_H` — VIS azure + H_E amber — or `native`, a JWST tile's one band; 400 for any other mode; both empty = the result's `thumbnail` recipe), `size` (8–2048 px thumbnail side; downsampled, never upsampled). Content-addressed `ETag` (`If-None-Match` → 304 without a render). |
| POST | `/viewer/results/<result_id>/rename` |  | Set the user `label` (JSON or form; ≤ 120 chars, no control characters; empty = back to the default label): `{ok, result: summary}`. The id does not change. |
| GET | `/viewer/results/grid.<output_format>` |  | Publication grid of saved results (`result`, `row`, `dpi`, `missing`): results are the columns, recipes the rows, square cells on A4 with the row titles sized to their text (horizontal beside the cells when the page has the width, else rotated), the table centred across the page and hung from the top, titles 7–10 pt. `missing=refuse` (default) answers 400 unless every result supports every row; `missing=blank` draws every available cell and a grey "Not available" cell in place of each missing one (400 when no cell is available). |
| GET, POST | `/viewer/grid-layouts` |  | Named figure-grid layouts (`<results root>/grid_layouts.json`). GET: `{layouts:[{id "gl-<12 hex>", name, results:[ids], rows:["tier:mode"], regime, created_utc, updated_utc}]}` newest first. POST (JSON or form; `results`/`rows` lists or comma strings): `{name, results, rows, regime?, id?}` — updates `id`, else the layout with the same name (any case), else creates one (201): `{ok, layout, created}`. 400 unknown result / bad recipe / > 12 results / > 16 rows; 409 at 200 layouts. |
| POST | `/viewer/grid-layouts/<layout_id>/delete` |  | Delete one layout: `{ok, id}`; 404 unknown. |

#### Saved viewer results (`helpers/viewer_results.py`)

- **Bundle** `<data>/viewer_results/<id>/` (`EUCLID_POLISH_RESULTS_DIR` overrides the root):
  `manifest.json` + one float32 FITS per logical tier (`dirty.fits`, `sr.fits`, `hr.fits`,
  `jwst.fits`, band on axis 3). The id `vr-<24 hex>` hashes the source, selection, crop bounds
  and FITS checksums, so re-saving the same crop is idempotent.
- **Saving** (`POST /viewer/results`): `{collection, index, tiers (≤ 4), params, selection:{u, v,
  angular_side_arcsec | relative_side + relative_fallback_safe, source_tier?}, display}`. Tier
  aliases: `lr`/`real`/`original…` → `dirty`, `sr`, `hr`, `jwst`, and the `real` collection's
  `m:<spec>` → `sr` (one per result); `real` params `source`, `models` (comma list of specs),
  `jwst_band`. **Sky-matched crops:** when tiers carry a C6 WCS the centre is the selection's
  `(u, v)` on `source_tier` (else the first tier with a WCS), mapped to every other tier through
  both WCSs (continuous pixel x ↔ 0-based centre x − 0.5, as the viewer's lens); tiers without a
  WCS use the normalised `(u, v)`. Each saved FITS keeps its crop's celestial WCS (`CRPIX` shifted
  by the crop offset, `WCSKEEP = T`).
- **Summary** (list/get/save): `{id, created_utc, label (user label, else default_label),
  default_label, regime: real|synthetic, source{collection, regime, index, params, object{label,
  id?, ref?, field?, ra?, dec?, tiers?…}, viewer_tiers}, selection, logical_tiers, bands,
  pixscale_arcsec, recipes:["tier:mode"], recipe_options:[{tier, mode, key, label}], thumbnail
  ("tier:mode"|null), files{logical:{filename, shape_hwc, bands, pixscale_arcsec, source_tier,
  source_label, display_scale, direct_rgb, transfer_group, wcs}}, bytes, inspect_paths{logical:
  project-relative FITS path for /files}, display, wcs_preserved (every tier has a WCS),
  wcs_tiers, center{ra, dec}|null}`. Bundles saved before these fields list with `wcs: false`,
  `center: null`.

### Poster cutout (`routes/poster.py`)

| Methods | Path | Gate | Notes |
|---|---|---|---|
| GET | `/poster/result/cutout.fits` |  | The last pulled `poster_cutout` FITS (attachment; local FASRC-cache copy, works offline). 404 `{ok:false, error}` before the first pull. |
| GET | `/poster/result/cutout.png` |  | The last pulled `poster_cutout` preview PNG (local copy). 404 before the first pull. |
| POST | `/poster/result/pull` | fasrc | Pull the latest `poster_cutout` PNG + FITS from `$DATA/_poster/` (force, bypassing the fetch TTL) and archive a changed PNG into `data/vis/poster/poster_cutout_<ts>.png`: `{ok, png{size, mtime, pulled_at}\|null, fits…, archived (path)\|null, errors{kind: msg}}`; 404 when neither exists remotely. |
| GET | `/poster/result/export` |  | The pulled scene PNG for print: `?format=` png, pdf or svg, `?dpi=` 150, 300 or 600 (the dpi sets the printed size — PNG `pHYs`, a PDF page, an SVG in inches; the pixels stay the node's render). Attachment `poster_cutout_<dpi>dpi.<format>`; 404 before a pull, 400 otherwise. |
| GET | `/poster/result/status` |  | Local state of the last pull: `{ok, available, png{size, mtime, pulled_at}\|null, fits…, archive_dir}`. |

### Figures (`routes/figures.py`, `helpers/nexus_plates.py`)

NEXUS × Euclid comparison plates — Euclid LR | SR of one model spec | native NEXUS, one PNG
per tile plus a contact sheet — rendered through the C9 `real` collection (`source=nexus`,
tiers `lr`, `m:<spec>`, `jwst`), so a plate shows the viewer's arrays. `band` = `VIS` \| `Y_E`
\| `J_E` \| `H_E` (per-panel asinh stretch) or `temp` (the viewer's Temp colour for LR/SR,
NEXUS grey). Runs live in `output/nexus_comparisons/<tag>/`
(`EUCLID_POLISH_NEXUS_PLATES_DIR` overrides the root): `nexus_tile<NNN>_<band>__<model
slug>.png`, `nexus_tiles_<band>__<model slug>.png` and `plates.json` (`{version, renders:[{band,
model, model_label, model_short, model_fingerprint, model_available, field_id, filter, created,
sheet, tiles:[{index, id, ref, ra_deg, dec_deg, file, model_state, legacy, sr_label}], source,
collection}]}`, one render per (band, model)). Runs written by the old script (no model slug,
`provenance.json`) list as `legacy` renders. `scripts/render_nexus_comparisons.py` is a CLI over
the same helper.

| Methods | Path | Gate | Notes |
|---|---|---|---|
| GET, POST | `/api/figures/nexus-plates` |  | GET: `{root, runs:[{tag, updated, renders:[render record \| legacy {band, model:null, legacy:true, model_label, sheet, tiles…} (a legacy tile's `id`/`ref` are resolved from its NEXUS tile number, the run's field first; `null` when no cached tile has that number)], files:[{name, size, kind: tile\|sheet, band, tile_index, model_slug}]}], bands, defaults{tiles, band, max_tiles}}` newest first. POST (form or JSON): `tiles` (comma list of NEXUS tile numbers, `f200w-NNNN` ids or `nexus/…` refs; ≤ 24), `band` (default `VIS`), `model` (a C9 spec, default `production`), `tag` (default `<model slug>-<YYYYMMDD>`) → local job `figure-nexus-plates` `{ok, job_id, tag, band, model, tiles}`; validated before the job: 400 bad band / tag / spec, or `{ok:false, error, missing:[ids]}` when the spec has no output on some tiles (run it in Sky › Compare), 404 unknown tile. Job result `{tag, band, model, sheet, tiles}`. The contact sheet's dpi drops below 200 for long runs so its canvas stays ≤ 10 Mpx. |
| GET | `/api/figures/nexus-plates/<tag>/<name>` |  | One plate PNG of a run (`?download=1` attachment; `?thumb=<px>` a JPEG preview, longest side 64–1600 px, memoised). 404 outside the run's plate files. |
| GET | `/api/figures/nexus-plates/<tag>/export` |  | One rendered run re-drawn for download from the cached tile outputs (read-only): its contact sheet, or `?tile=<N>` that tile's plate, for the render `?band=` + `?model=`, at `?dpi=` 150, 300 or 600 as `?format=` png, pdf or svg. A long sheet's dpi drops to stay under 40 Mpx (`X-Plate-Dpi` and the file name carry the dpi used). 404 for an unknown run / render / tile or a legacy run without `plates.json`; 409 when the model's outputs changed since the render (render the run again); 400 for another format or dpi. |
| POST | `/api/figures/nexus-plates/<tag>/delete` |  | Delete one run directory: `{ok, tag}`; 404 unknown. |
| GET | `/api/figures/real-sr` |  | The newest cached production SRs of real tiles (the C9 output store only; Home's thumbnail strip): `{total, items:[{ref, source, id, source_label, label, created, state: current\|stale\|unavailable, thumb}]}` newest first; `?limit=` 0–24 (default 6; 400 when not an integer). Read-only: never runs a model. |
| GET | `/api/figures/real-sr/<source>/<identifier>.jpg` |  | Colour JPEG preview (the viewer's Temp rendering) of one cached production SR, longest side `?size=` 64–640 (default 240), memoised per file state; 404 when the tile has no cached production SR, 400 for a bad id or size. |

### Model studies (`routes/studies.py`, `euclid_polish/studies/`)

A study freezes the **whole** ensemble of one regime (every active member, the production gate,
the combiner comparison) into an immutable record that outlives the members: archiving,
retraining or deleting a member never touches it. Local store
`<Config.TRACKING_DIR>/studies/<id>/` (outside every tracking campaign): `study.json` (the
manifest: `id` = `YYYYMMDD-HHMMSS-<slug>`, `name`, `note`, `created`, `commit`, `regime`, the
ensemble snapshot `ensemble.members[{label, name, origin (origin.json), fingerprint, step,
target_steps, status, timeout, loss, asinh_knee(s), output_knee, knee_loss, blocks, bootstrap,
noise_aug, icnr, seed, …}]`, the production `gate` identity `{name (promoted_from), dir, kind,
member_labels, reads, mix_space, use_lr, width, fingerprint, fit, fitted_at}`, `records`
`{records_fp, subset, indices, target, target_psf_fwhm_arcsec, generation, noise_models}`,
`evaluation{evaluated_at, checks, headline}`, `blocks`, `warnings` (stale blocks at freeze),
`identity{labels, fingerprints, records_fp, gate_fingerprint}`, `numbers{file: {sha256,
bytes}}`, `fields[{fid, kind, ref, label, state: pending|uploaded, remote_dir,
products{name: {sha256, bytes, shape?, dtype?}}, bytes, gate}]`, `remote{field_root,
numbers_dir}`, `complete`, `completed`, `error` while incomplete), `numbers/` (read-only once
complete: `members.csv`, `knee_psnr.json` = `{knees (KNEE_GRID_E, 21), bands, fields
[record index], dropped_fields [evaluated fields left out: a missing cube or target record; also in the manifest `warnings`], models[{id, kind: member|mean|gate|combiner, label}], psnr[model][field][knee]
[band]}` — per field, the Leaderboard loop, `integrated.csv` = `model, kind, field, band,
integrated_psnr`, `training_curves.json`, `gate.json` = `{production, diagnostic (held-out
usage / source usage / by brightness), variants, compare (latest report or null),
compare_note}`, `real.json` = `{experiments[{id, label, created, tiles, specs{spec: {label,
member_label, fingerprint, summary, per_tile}}}], note}`, `thumbs/<fid>.jpg`) and the only
mutable sidecars `note.json`, `selections.json`. Attached fields (≤ 10) live on holylabs only:
`<parent of the remote tracking dir>/study_fields/<id>/<fid>/` — a sibling of the directory the
tracking mirror pushes (plain `rsync -az`, never `--delete`), so a push can never remove them —
one `<product>.npz` (key `data`, float32 lossless; `mask` uint8) per product: `member_<N>`,
`mean`, `gate`, `lr`, `hr` (synthetic), `mask` (blackout holes), plus `truth.json` and
`field.json` (identity, labels, pixel scales, WCS, target blur, product sha256s). Field ids:
`test-NNNNN`, `blackout-NNNNN` (record index), `real-<source>-<tile id>`. A freeze packs one
product at a time, uploads it, checks the remote `sha256sum` and deletes the temp file; the
numbers directory is mirrored to `<remote tracking dir>/studies/<id>/` after completion
(best-effort). Every local write of a freeze or a fetch reserves its bytes first, and each disk-margin check subtracts the other study jobs' reservations, so a freeze and a fetch cannot jointly leave < 5 GiB free. Fetching is per product: "Fetch field" brings the core products (`field.json`,
`truth.json`, `lr`, `hr`, `mean`, `gate`, `mask` — those the field has); member SRs are fetched
explicitly, one by one or "every member that fits". Fetched products are cached in
`<Config.VIS_DIR>/study_fields/<id>/<fid>/` (≤ 2 GiB over every study, accounted and evicted
per product file, least recently used first (by local use: a pulled product is stamped when it arrives, since rsync keeps the holylabs mtime), never a product of the field being fetched; a fetch
never leaves < 5 GiB free; one fetch at a time; every product's sha256 checked against the
manifest). Errors are JSON
`{ok:false, error}`; a margin violation is **507** `{code:"insufficient_storage", needed_bytes,
free_bytes}`; an offline request needing fields is 503 `fasrc_offline`. Opening or reading
never starts a job or fetches a field.

| Methods | Path | Gate | Notes |
|---|---|---|---|
| GET, POST | `/api/studies` |  | GET: `{ok, studies:[{id, name, created, completed, regime, members, gate, fields, field_ids, note, commit, state: complete\|incomplete, reason, numbers_bytes, fields_bytes}], root, max_fields: 10, freezing:{job_id, study_id}\|null}` newest first. POST (form or JSON): `name` (required), `note`, `mode` (`starfull` default \| `starless`), `fields` (comma list or JSON list of field ids, ≤ 10, each `available` in the candidates) → validates (400 no name / > 10 / unknown field; 409 stale test cubes, unavailable field or another freeze running `code:"busy"`; 503 `fasrc_offline` with fields while disconnected; 507 disk margin), creates the incomplete study and starts local job `study-freeze` → `{ok, job_id, study_id, fields, upload_bytes (upper bound)}`; a freeze that loses the race to another answers 409 `busy` with `study_id: null` and leaves no study. Job: numbers → fields (progress per product) → `complete` → mirror; result `{study_id, complete, fields, fields_bytes, numbers_bytes, mirror{ok, remote_dir, error?}}`. A failure (sha mismatch, cancel, …) leaves the study incomplete with `error`. |
| GET | `/api/studies/candidates` |  | Read-only freeze-dialog contents for `?mode=`: `{ok, regime, ensemble{members, n_members, gate{available, state, name, reads, mix_space, fitted_at}, evaluated_at, blocks[{id: members\|test_cubes\|gate\|gate_diagnostic\|compare\|training_curves\|real, title, state: current\|stale\|missing, detail}], stale[ids], numbers_bytes (≈)}, fields[{fid, kind: test\|blackout\|real, ref, label, available, reason, bytes, core_bytes, largest_product_bytes, bytes_upper_bound: true, thumb_url}], max_fields, can_freeze, blocking (why not), fasrc_connected, fields_note}`. Sizes are uncompressed upper bounds (the study manifest then records the compressed bytes). The test cubes are `current` only when they hold exactly the active membership, on the current records, with the checkpoints and the production gate the evaluation recorded (`eval_summary.json` `eval_identity.member_fps` / `combiner_fps.spatial_gate`; else `stale`, e.g. "member 203's checkpoint changed since the evaluation", "the production gate changed since the evaluation"). A field is available only with SR for every active member (current test cubes; blackout cubes built for those members and not older than any member's checkpoint on this machine, `max(mtime, ctime)` of its index — else "… delete cubes_blackout/blackout_index.json and run a combiner comparison to rebuild them"; real tiles, STARFULL only, with a current cached member SR — checkpoint fingerprint + LR hash — for every member) and when its core products and its largest product fit the 2 GiB fetch cache. 400 bad mode. |
| GET | `/api/studies/candidates/thumb/<fid>.jpg` |  | Colour JPEG of a candidate field (test: baked gate else mean; blackout: stamped LR; real: stored production SR else LR), `?size=` 32–480 (default 160), `?mode=`. 400 bad id, 404 no cube. |
| GET | `/api/studies/<study_id>` |  | `{ok, study (summary), manifest, manifest_sha256, note, selections, fields[{fid, kind, ref, label, state, bytes (compressed, all products), estimated_bytes (upper bound at freeze), core_bytes, member_bytes{member_<N>: bytes}, gate, fetched (core products cached), cached_products, members_fetched, products, thumb_url, viewer{collection:"study", params{study}, id}}], charts, group_fields, citation, numbers{knee_psnr, training_curves, gate, real}}` (`numbers` only for a complete study; `?numbers=0` omits it). 404 unknown. |
| POST | `/api/studies/<study_id>/resume` |  | Continue an incomplete study (same `study-freeze` job; the numbers are kept, pending fields uploaded) → `{ok, job_id, study_id, fields, upload_bytes}`. 409 already complete or the ensemble changed since it started (members, checkpoints, records or gate; delete and freeze again); 503 `fasrc_offline` with pending fields while disconnected. |
| POST | `/api/studies/<study_id>/note` |  | `note` (≤ 4000 chars) → `{ok, id, note}` (the `note.json` sidecar; the manifest keeps the frozen note). |
| POST | `/api/studies/<study_id>/selections` |  | JSON `{selections:[{name, members?:[label], group?:<recipe field>, note?}]}` (or form `selections` = that JSON list; names unique, ≤ 100) replaces the named selections → `{ok, id, selections}`. 400 malformed, a member that is not in the study, or an unknown group field. |
| POST | `/api/studies/<study_id>/delete` |  | `confirm=1` required (400 `confirm_required`). Deletes the local study, its fetched-field cache and (guarded `rm -rf`: absolute, depth ≥ 4, `/study_fields/` or `/studies/` in the path, the study id last) its holylabs field store and numbers mirror → `{ok, id, remote:[status lines]}`. 409 while its freeze or a fetch of one of its fields runs, or offline for a study with fields unless `local_only=1`. With `local_only=1` only the local study (and its fetch cache) is deleted and the holylabs copies are always kept (connected or not), reported as `NOT deleted — kept on FASRC (local_only)`. |
| POST | `/api/studies/<study_id>/fields/<fid>/fetch` | fasrc | Fetch products of one uploaded field from holylabs into the local cache (local job `study-fetch`; ONE fetch at a time: the same request re-attaches `already_running`, another answers 409 `busy`). `products` = `core` (default: the core products) \| `members` (every member SR that still fits, in member order; the rest are `skipped`) \| a comma list of product names (`member_170,…`) → `{ok, job_id, study_id, fid, products, bytes}`; job result `{study_id, fid, fetched, skipped, cached, bytes}`. Cached products are not transferred again. The job fails when a product would exceed the 2 GiB cache (with this field's products kept) or leave < 5 GiB free, or a sha256 does not match. 400 unknown product, 404 unknown field. |
| GET | `/api/studies/<study_id>/figure/<name>` |  | `<name>` = a chart (`knee`, `integrated`, `paired`, `gate`, `training`, `real`) → the figure in the plate style, `?format=` png (default) \| pdf \| svg, `?dpi=` 120–600 (default 300); `<chart>.csv` (or `?format=csv`) → exactly the plotted numbers. Selection: `?members=` (comma labels), `?group=` (recipe field: loss, training_knee, asinh_knee, output_knee, knee_loss, blocks, bootstrap, noise_aug, icnr, status, op), `?selection=<saved name>`; `paired`: `?reference=` mean (default) \| gate \| best \| `<label>` \| `group:<name>`, `?seed=`, `?resamples=` (default 2000; 95 % paired bootstrap over fields; 409 when a band has fewer than 2 fields with both values); `gate`: `?source=` all \| sources \| a brightness bin; `training`: `?metric=` psnr \| VIS \| Y_E \| J_E \| H_E \| loss; `real`: `?experiment=`. `?download=1` attachment `study-<id>-<chart>[-<dpi>dpi].<ext>`. 404 unknown chart / missing data (the error names what the study lacks), 409 incomplete study, 400 bad format / selection. |
| GET | `/api/studies/<study_id>/numbers/<path:name>` |  | One frozen numbers file (`members.csv`, `knee_psnr.json`, …, `thumbs/<fid>.jpg`); `?download=1` attachment. 404 unknown. |
