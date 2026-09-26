# EuclidPolish web console rework — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: use superpowers:subagent-driven-development (or
> superpowers:executing-plans) and superpowers:test-driven-development. Steps use `- [ ]`.
> Spec: `docs/superpowers/specs/2026-09-25-webui-rework-design.md` (read it first; it is the
> authority for behaviour). This plan is the authority for **file ownership, contracts,
> sequencing and acceptance**.

**Goal:** rebuild the EuclidPolish console as 9 URL-addressable workspaces on a new shell
(palette, job tray, inspector, display panel), a TS viewer engine with pixel/WCS inspection,
and an Aladin-Lite Sky workspace for experimenting with real results; delete legacy.

**Architecture:** Flask backend (`euclid_polish/web/`) + React 18/Vite/TS SPA
(`euclid_polish/web/frontend/`, committed build in `static/dist/`). One committed route manifest
(`euclid_polish/web/spa_routes.json`) drives both Flask page serving and the SPA router.

**Tech stack:** Flask 3.1, astropy, numpy; React 18, react-router 6.30 (data router),
TanStack Query 5, zustand 5, Radix UI, cmdk, sonner, tinykeys, react-resizable-panels,
@tanstack/react-virtual, aladin-lite 3.8.2 (lazy), vitest, eslint.

**Environment:**
- Python: `/Users/alarion239/miniforge3/envs/EuclidPolishEnv/bin/python` (prefix commands with
  `KMP_DUPLICATE_LIB_OK=TRUE`). Tests: `EUCLID_POLISH_DISABLE_AUTO_SSH=1 NUMBA_DISABLE_JIT=1
  MPLBACKEND=Agg PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 KMP_DUPLICATE_LIB_OK=TRUE <python> -m pytest -q`
  (baseline 2026-09-25: 2348 passed, 3 skipped, ~140 s). Lint: `<python> -m ruff check .`.
- Node v26 / npm 11 in `euclid_polish/web/frontend`: `npm run typecheck`, `npm test`,
  `npm run lint`. **Only the orchestrator runs `npm run build`** (it writes the committed
  `static/dist`).
- A dev server may be running on :9777 (`.claude/launch.json` "euclid-web"); agents must not
  start/stop it.

---

## 0. Rules for every work package (WP)

1. **Ownership.** Edit only the files your WP owns (listed per WP). Backend WPs own
   `euclid_polish/**` except `euclid_polish/web/frontend/**`, plus `tests/**`, `scripts/**`,
   `.github/**`. Frontend WPs own paths under `euclid_polish/web/frontend/` as listed.
   Full-stack workspace WPs (phase 3) own their workspace folder + the backend modules listed.
2. **No git commits, no pushes, no `npm run build`, no server restarts.** The orchestrator
   integrates, builds and commits at each phase gate.
3. **No new npm dependencies** except in WP-F (foundation) and WP-V (viewer); ask the
   orchestrator in your final report if you need one.
4. **Imports at module top** in every Python file you create or touch (user rule). No
   function-scoped or conditional imports.
5. **TDD**: write the failing test, see it fail, implement, see it pass. Keep the full pytest
   suite and `npm test` green at the end of your WP.
6. **Offline-first**: local-data endpoints must work with FASRC disconnected.
7. **Document** every endpoint you add or change in `euclid_polish/web/API.md` (backend) and every
   shared frontend API in `euclid_polish/web/frontend/src/FOUNDATION.md` (frontend).
8. **Report** at the end: files changed, tests added, commands run with results, open issues,
   and any contract deviation (with justification).

---

## 1. Contracts (fixed; do not change without the orchestrator)

### C1 — Route manifest `euclid_polish/web/spa_routes.json` (committed)
`{version, workspaces:[{id,label,path,params?,defaultParams?,tabs[],defaultTab?}], redirects}`.
A **page path** is a workspace `path` (params substituted from their allowed values) optionally
followed by `/<tab>` with `tab ∈ tabs`. `/inspect` and `/` have no tabs. Redirects are exact
path → target (308 server-side, `<Navigate replace>` client-side), preserving the query string;
`/app/<rest>` → `/<rest>` (308). Non-page paths (e.g. `/ensemble/status.json`,
`/inspect/preview.png`) must NOT match.

### C2 — Local jobs API
- `GET /api/jobs` → list (newest first) of job dicts: existing keys (`job_id, label, status,
  started, finished, duration, error, log, log_truncated, progress{current,total,pct,label,
  stage_elapsed,rate_per_second,eta_seconds,updated_ago_seconds}`) plus `kind` (str|null),
  `cancellable` (bool), `result` (JSON-safe value ≤ 64 KB, else null). `status ∈ {running,
  done, failed, cancelled}`.
- `GET /api/jobs?summary=1` → same, with `log` = null.
- `GET /api/jobs/<id>` → one job dict (full log tail).
- `POST /api/jobs/<id>/cancel` → `{ok:true}`; the job becomes `cancelled` when its target next
  calls `cap.tick(...)` (raises `JobCancelled`); `{ok:false,error}` 404/409 otherwise.
- The registry keeps at most 200 finished jobs (oldest evicted).
- `REGISTRY.spawn(label, target, kind=None)`.

### C3 — `GET /api/version`
`{boot_commit, boot_short, head_commit, head_short, behind: bool, dirty: bool, started_at (ISO),
pid, dist: {built_at (ISO)|null, index_hash|null}}` — `behind` is `boot_commit != head_commit`.

### C4 — FASRC connection
`GET /api/fasrc/status` adds `last_error` (str|null; includes the startup error). When FASRC is
disconnected, any endpoint that needs SSH returns **503** `{"ok": false, "error": "FASRC not
connected", "code": "fasrc_offline"}`; every other endpoint works. `GET/POST /api/fasrc/config`,
`POST /api/fasrc/connect`, `POST /api/connection/retry` work offline.

### C5 — FASRC steps
`GET /api/fasrc/steps/status` → each `steps[i]` adds `task_params: [{name, type:
"int"|"float"|"str"|"bool"|"choice"|"json", default, min?, max?, choices?, help, required?}]`
and `last_params: {…}|null` (task params of the newest successful run of that step).
`POST /api/fasrc/steps/<id>/submit` fills missing task params from the schema defaults and
rejects invalid ones (400 `{error}`). `euclid_query` schema defaults: `num_stars=10000,
magnitude_min=18, magnitude_limit=19, snr_min=50`. `GET /api/fasrc/current-submission` adds
`live: [row…]` (every PENDING/RUNNING job, same row shape as the existing single job).

### C6 — Viewer backend
- Errors: `/viewer/*` returns JSON `{error}` with the status code (no HTML aborts).
- Meta: every `objects[i]` has `id` (stable string) and, when known, `ra`, `dec` (deg). Every
  tier may carry `unit` (`"e-"`, `"MJy/sr"`, `"ADU/s"`, `"arb"`).
- Cube headers add `X-Cube-WCS` (compact JSON of FITS celestial WCS keywords for *that tier's*
  pixel grid: `CTYPE1, CTYPE2, CRVAL1, CRVAL2, CRPIX1, CRPIX2` and either `CD1_1, CD1_2, CD2_1,
  CD2_2` or `PC*`/`CDELT*`; FITS 1-based pixel convention, axis 1 = column/x) when known, and
  `X-Cube-Unit`. Both are listed in `Access-Control-Expose-Headers`.
- Ensemble collection: default `mode=starfull`; tier `sr` is the production combiner
  (`ACTIVE_COMBINER_KINDS[0]`, the spatial gate) with label "SR · production gate"; a new
  `mean` tier ("Mean of members"); RBF kinds stay as extra tiers when loadable.
- Cubes with > 4 channels are served without being misread (HWC with `c` = channel count).

### C7 — Display settings store (frontend `src/state/display.ts`, created by WP-F, consumed by WP-V)
```ts
export type ColorMode = "VIS" | "Y_E" | "J_E" | "H_E" | "lupton" | "temp" | "rgb" | "native";
export type Stretch = "asinh-abs" | "linear" | "log" | "sqrt" | "asinh-auto" | "zscale";
export type Colormap = "gray" | "viridis" | "magma" | "inferno" | "cividis" | "rdbu";
export type TransferGroup = { knee: number; gain: number; black: number };
export type DisplaySettings = {
  color: ColorMode;
  rgb: [string, string, string];          // band names for R, G, B in "rgb" mode
  stretch: Stretch;                       // default "asinh-abs" (locked default)
  groups: Record<string, TransferGroup>;  // "default" | "euclid" | "jwst"; knee e- (default 100), gain 1, black 0
  colormap: Colormap;                     // default "gray"
  residualColormap: Colormap;             // default "rdbu"
  invert: boolean;
  nanColor: string;                       // CSS colour, default "#ff00ff"
  linked: boolean;                        // true: all viewers follow the global settings
  wheel: "zoom-when-focused" | "always-zoom" | "scroll";
};
export const useDisplay: UseBoundStore<…>;  // zustand; persisted (try/catch localStorage)
// actions: set(patch), setGroup(name, patch), reset()
```

### C8 — Foundation API (WP-F implements; documented in `src/FOUNDATION.md`)
- `api/client.ts`: `apiGet<T>(url, {signal?})`, `apiPost<T>(url, data: Record|FormData|object,
  {json?: boolean})`, `class ApiError {status; message; code?; body}`. Compat re-exports in
  `src/api.ts` (`getJSON`, `postForm`).
- `api/query.ts`: `queryClient`; `useResource<T>(url|null, deps?, {ttl?, poll?})` →
  `{data, loading, error: ApiError|null, reload}` (compat: `error` truthy on failure); in-flight
  dedupe and shared subscribers; `invalidate(prefix)`.
- `api/jobs.ts`: `useJob()` (compat `{job, error, busy, run(url, data, {onDone}), reset}`;
  jobs survive navigation — started jobs are registered in the global feed), `useJobsFeed()`
  (all local + SLURM jobs), `cancelJob(id)`.
- `hooks/useUrlState.ts`: `useUrlState<T>(key, defaultValue, {parse?, serialize?, replace?})`.
- `hooks/useShortcut.ts`: `useShortcut(combo, handler, {scope?, description})`.
- `app/palette.ts`: `usePageActions(actions: {id, label, group?, keywords?, shortcut?, run}[])`.
- `app/inspector.ts`: `registerInspector(kind, Component<{id: string}>, {title?})`,
  `openInspector({kind, id})`, `closeInspector()`, URL param `?inspect=kind:id`.
- `ui/`: the kit listed in spec §5 (names exactly: `Button, IconButton, Tooltip, Popover,
  Dialog, confirm, Menu, ContextMenu, Tabs, Segmented, Switch, Checkbox, Slider, RangeSlider,
  NumberField, Input, Select, Field, Card, CardHead, CardBody, Section, Badge, Chip, Stat, Kpi,
  DefList, Callout, EmptyState, Skeleton, ProgressBar, LogView, JsonTree, CopyButton, Kbd,
  DataTable, toast`) plus compat exports of every current `ui/index.tsx` name (`Page, PageHead,
  Empty, Spinner, Table, LogTail, Gallery, PngFigure, ConnBadge, Textarea, …`).
- `charts/Plot.tsx` v2: existing props unchanged + `yScale: "linear"|"log"`, `syncKey?`,
  `tooltip?: boolean` (default true), `zoom?: boolean` (default true), `legendToggle`,
  `exportName?`; `Legend` interactive.
- `format.ts`, `ticks.ts` (see spec §5).
- Workspace contract: `src/workspaces/<id>/index.tsx` default-exports the workspace component;
  it renders `WorkspaceTabs` (router-linked) and lazy tab modules
  `src/workspaces/<id>/tabs/<Tab>.tsx`.

### C9 — Real results, models, experiments, sky atlas (WP-B2 implements; spec §9.1–9.3)
Endpoints (exact paths): `GET /api/real/sources`, `GET /api/real/<source>`,
`GET /api/real/<source>/<id>`, `GET /api/real/<source>/<id>/image.fits?tier=&band=`,
`POST /api/real/tiles` (`ra`, `dec`, optional `run=production,mean`), `POST
/api/real/<source>/<id>/delete-outputs`, `GET /api/models`, `POST /api/experiments`
(`tiles` = comma list of `source/id`, `models` = comma list of specs), `GET /api/experiments`,
`GET /api/experiments/<id>`, `GET /api/sky/layers`, `GET /api/sky/layer/<id>`,
`GET /api/sky/at?ra&dec`, `POST /api/sky/jwst/discover`, `GET /api/sky/jwst/footprints?ra&dec&r`,
`POST /api/sky/jwst/pair`. Viewer collection `real` (params `source`, `models`) with tiers `lr`,
`jwst` (if any) and `m:<spec>`. Model specs: `production`, `mean`, `member:<name>`,
`gate:<variant>`, `rbf`. Response shapes are documented by WP-B2 in `API.md`.

---

## 2. Phase 1 (parallel): WP-B1 backend platform ∥ WP-F frontend foundation

### WP-B1a — Backend platform
**Owns:** `euclid_polish/web/{app.py, security.py, jobs.py, spa_routes.py (new), fasrc_gate.py
(new), version.py (new), remote.py, API.md (new)}`, `routes/__init__.py`, `routes/files.py`
(jobs/version routes only), `routes/fasrc.py` (status/config/connect only), the gate marks in
every `routes/*.py`, `.github/workflows/quality.yml`, tests for all of the above.

- [ ] **T0 (first!) Decouple pytest from frontend source.** WP-F restructures `frontend/src` in
  parallel. Before anything else, rewrite every test that reads `frontend/src/**` or
  `static/dist/assets/**` contents (`test_noise_tab.py:126`, `test_galaxy_distributions.py:1012-1102`
  SPA asserts, `test_population_comparison.py:1078+`, `test_star_distribution.py:317,408`,
  `test_galaxy_corner.py:90,182`, `frontend/test`-mirroring asserts in `test_web.py` except the
  `cutout_viewer.js` source test) as backend behaviour tests (API contract / payload shape) or
  delete the SPA-source assertion. From then on no pytest may read `frontend/src`.
- [ ] **T1 SPA page matcher.** Create `spa_routes.py` with `load_manifest(path=None)`,
  `is_page_path(path) -> bool`, `redirect_target(path, query) -> str|None`. Tests
  `tests/test_spa_routes.py`: every workspace path and `path/<tab>` is a page (`/`, `/sky`,
  `/sky/atlas`, `/ensemble/starfull`, `/ensemble/starless/train`, `/inspect`, `/settings/about`);
  not pages: `/ensemble/status.json`, `/ensemble/foo`, `/sky/unknown`, `/inspect/preview.png`,
  `/api/jobs`, `/static/x.js`; redirects (every entry, query preserved, `/app/x?y=1` →
  `/x?y=1`). Run: `pytest tests/test_spa_routes.py -v` → fail, implement, pass.
- [ ] **T2 Serve the SPA from the manifest.** In `app.py` replace `_REACT_PAGE_PATHS`,
  `_DEPRECATED_PAGE_PATHS`, `_redirect_deprecated_app_prefix` and the page branch of
  `react_console` with: GET/HEAD redirect (308) when `redirect_target` matches; serve
  `static/dist/index.html` when `is_page_path`. Keep the "not built" 503 hint. Tests in
  `tests/test_web.py`/new: `/sky/atlas` → 200 `id="root"`; `/config` → 308 `/settings/config`;
  `/ensemble/status.json` still JSON.
- [ ] **T3 Route module registry.** `routes/__init__.py` exposes `MODULES` (tuple of modules
  with `register(app)`); `create_app` loops over it. Adding a module = one line there.
- [ ] **T4 FASRC gate by route.** `fasrc_gate.py`: `requires_fasrc(view)` sets
  `view._requires_fasrc = True`; `register_fasrc_gate(app)` adds a `before_request` that looks up
  `app.view_functions.get(request.endpoint)` and returns C4's 503 JSON when marked and not
  connected. Remove `_enforce_ssh_gate`, `_ALWAYS_REACHABLE_PREFIXES` and the redirect to
  `/connection-error`. Audit **every** route in `routes/*.py` and `app.py`: mark handlers that
  call `STATE.ssh`, `remote.*`, `fasrc_fetcher` pulls, rsync or SSH helpers without a local
  fallback. Tests `tests/test_fasrc_gate.py` with `STATE.ssh=None`: local endpoints respond
  non-503 (`/api/git/status`, `/inspect/preview.png?fits=…` (fixture), `/vis/<png>` fixture,
  `/api/jobs`, `/api/version`, `/api/fasrc/status`, `/api/fasrc/config`,
  `/api/connection/retry` (returns its own error, not the gate's), `/api/noise`); a sample of
  marked endpoints return the 503 JSON; an AST test that every handler whose body references
  `STATE.ssh`/`.run(`/`rsync` is either marked or listed in an explicit allowlist of
  graceful-degradation handlers in the test.
- [ ] **T5 Security.** `app.config["TRUSTED_HOSTS"] = ["localhost", "127.0.0.1", "[::1]",
  "::1"]` (verify Werkzeug 3.1 semantics with a test: `Host: evil.example` → 400;
  `localhost:9777` → OK). Move `GET /api/fasrc/env-update` to `POST` returning a local job
  (`kind="fasrc-env-update"`) that streams the remote output into the job log.
- [ ] **T6 Jobs (C2).** Add `kind`, `cancellable`, `result` (JSON-safe, size-capped) to
  `to_dict`; `JobCancelled`; `Job.cancel()` flag; `_LogCapture.tick` raises when flagged;
  `cancelled` status; eviction (200 finished); `?summary=1`; `POST /api/jobs/<id>/cancel`.
  Tests in `tests/test_jobs.py`: cancel mid-run, result passthrough, eviction, summary strips log.
- [ ] **T7 Version (C3).** `version.py` computes boot commit at import (git rev-parse via
  subprocess at module level of `create_app`, cached), HEAD on request (cheap `git rev-parse`),
  dirty flag, dist info from `static/dist/index.html` mtime/hash. Route `GET /api/version`. Test.
- [ ] **T8 Connection (C4).** `STATE.public_status()` / status route adds `last_error`
  (startup error or last connect failure). `/api/connection/retry` and `/api/fasrc/config`
  unmarked. Tests.
- [ ] **T9 CI.** Add a `frontend` job to `.github/workflows/quality.yml`: setup-node 22,
  `npm ci`, `npm run typecheck`, `npm run lint`, `npm test`, `npm run build`,
  `git diff --exit-code -- euclid_polish/web/static/dist` (build freshness).
- [ ] **T10 API.md.** Create `euclid_polish/web/API.md`: conventions (JSON errors, jobs, gate,
  POST-only mutations, Host allowlist) + a table of every endpoint after WP-B1a.
- [ ] **Acceptance:** full pytest green; `ruff check .` clean.

### WP-B1b — Legacy removal, fixes, viewer backend, step schemas (runs after WP-B1a)
**Owns:** `euclid_polish/web/**` (except frontend and the WP-B1a platform files, which it may
touch only for deletions), `euclid_polish/eval/ensemble_infer.py`, `euclid_polish/tracking/**`
(payload fix only), `scripts/fasrc_download_euclid_sky_cutouts.py` (EDF constant),
`euclid_polish/sky/observation/q1_fields.py` (new), `tests/**`.

- [ ] **T1 Legacy deletion (spec §10).** Delete templates, classic static JS/CSS (NOT
  `static/cutout_viewer.js` — WP-V removes it), dead page handlers and modules, unreferenced and
  superseded endpoints, orphan functions, `FasrcConfig` science fields. Replace `url_for` uses
  of deleted endpoints with literal paths. `fasrc_file_inspect` returns JSON 502 instead of a
  template. Keep `helpers/tng_prior.py`, the FASRC-mirror `stars.csv` path, and
  `reconstruct_cutout_at`. Wire-not-delete list stays. Port the `/api/fasrc/submit` queue tests
  to `/api/fasrc/steps/<id>/submit` before deleting it.
- [ ] **T2 Test rewrites.** Every test that pinned deleted code (`test_web` template/route
  tests, `test_inspect_fits:152`, `test_fasrc_fetcher:274-312`, `test_fasrc_integration`
  superseded endpoints, `test_psf_preview`, `test_eval_catalog /eval-files`,
  `test_archive_fields*` `/api/archive-fields` if removed) is rewritten as a backend behaviour
  test or deleted with its endpoint. **No test may read `frontend/src` or `static/dist/assets`
  contents** (frontend behaviour is tested by vitest). `tests/test_web.py`'s
  `cutout_viewer.js` source test stays until WP-V.
- [ ] **T3 Viewer backend (C6)** in `routes/viewer.py`, `helpers/viewer_data.py`: JSON errors,
  ids/ra/dec for every collection, tier units, `X-Cube-WCS`/`X-Cube-Unit` for every real
  collection (archive-fields per-band HDU WCS; real-field from `original_stack.fits` per tile
  with CRPIX offsets; nexus-field LR/SR/JWST FITS headers; evaluation `original_stack.fits` /
  `SR.fits`; SR grids = LR WCS scaled ×2 via `CD/2` and CRPIX adjusted), ensemble
  `sr`=production gate + `mean` tier + starfull default, >4-channel cubes. Move
  function-scoped imports in `viewer_data.py` to module top. Tests `tests/test_viewer_backend.py`
  with small FITS fixtures (WCS round trip: pixel (0,0) centre → RA/Dec equals the FITS WCS
  value within 1e-9 deg).
- [ ] **T4 Step task_params (C5)** in `euclid_polish/web/fasrc_pipeline.py` (+ step classes):
  a `TaskParam` dataclass and per-step declarations matching what each `build_command` reads;
  `last_params` from the jobs DB; submit fills/validates. Tests: every registered step's
  `build_command` accepts its schema defaults; `euclid_query` submit with no params produces a
  command containing `--num-stars 10000` (or the step's actual flag names) and never 200.
  `current-submission` `live` list.
- [ ] **T5 Fixes (spec §11, backend part):** `routes/ensemble.py` starless defaults → starfull;
  `eval/ensemble_infer.py` uses STARFULL members and the production combiner (fallback: mean when
  no combiner loads) — tests with a stub ensemble; `/api/evaluation/sync` requires `confirm=1`
  (400 otherwise); Git commit takes explicit `paths` (list) or `all=1`, refuses files > 10 MB or
  untracked binaries (`.fits`, `.npy`, `.zip`, `.jpg`, `.png` > 1 MB) unless `force=1`, returns
  the refused list (409); `/api/config` returns `version` (hash of the file) and
  `/api/config/save` accepts only posted fields plus `base_version` (409 on mismatch for any
  posted field changed server-side); tracking sandbox payload `source` → object plus
  `source_label` string; `/api/status` no forced rsync (cheap), add `POST
  /api/status/refresh-catalog` (gated) for the explicit pull; `/api/tng/radii/status` returns
  the cached result immediately and refreshes via a job (`POST /api/tng/radii/refresh`);
  `helpers/ensemble_viz.py:497` training-curve loss overwrite; EDF-F/EDF-S: create
  `euclid_polish/sky/observation/q1_fields.py` (`Q1_FIELDS` centres + `q1_field_for(ra, dec)`),
  use it in the 4 duplicate modules and fix the swapped constant in
  `scripts/fasrc_download_euclid_sky_cutouts.py`; archive-field / vis-noise payloads expose the
  position-derived `field` (and keep `stored_field`). Publication-figure heat bar unit label
  comes from the tier unit.
- [ ] **Acceptance:** full pytest green; ruff clean; `grep -rn "render_template" euclid_polish/web`
  empty; `ls euclid_polish/web/templates` gone; API.md updated.

### WP-F — Frontend foundation (parallel with WP-B1a/B1b)
**Owns:** `euclid_polish/web/frontend/**` except `src/pages/**` page bodies (it may edit
imports/wiring in pages only to keep them compiling) and except `src/viewer/**` (WP-V).
Executed as three sequential agents: **WP-F1** = T1–T4 + `format.ts`/`ticks.ts` (from T6) +
FOUNDATION.md skeleton; **WP-F2** = T5 + T6 (UI kit v2, DataTable, Plot v2); **WP-F3** = T7–T9
(shell, workspace shells with adapters, final FOUNDATION.md). Each WP is followed by a review
agent and, when the review finds issues, a fix agent.

- [ ] **T1 Tooling.** Add dependencies (spec §5; exact versions), vitest + testing-library +
  happy-dom, eslint config (typescript-eslint, react-hooks), scripts `typecheck`, `lint`,
  `test` (vitest run), `build` (`tsc --noEmit && vite build`). Port the 5 node tests to vitest.
  Fix the two existing tsc errors (`StarDistribution.tsx:411`, `TrainMembers.tsx:123`).
  Vite: import `../spa_routes.json` (C1), `manualChunks` (react, vendor), dev proxy of every
  non-page prefix to `FLASK_ORIGIN || http://localhost:9777` (derive page prefixes from C1),
  `base` only for build, `chunkSizeWarningLimit` for the lazy aladin chunk.
- [ ] **T2 Theme tokens** (`src/theme/tokens.css`, spec §5): full contract light + dark +
  system; band/loss/type/z/motion tokens; define every previously undefined token and `.muted`;
  AA contrast for `--text-faint`, `--warn`, `--good`. Theme pref `light|dark|system`.
- [ ] **T3 Data layer (C8 api/query/jobs).** TanStack Query client, `apiGet/apiPost/ApiError`,
  `useResource` compat with dedupe/subscribers/visibility-aware polling, `useJob` compat
  registering into a global jobs store, `useJobsFeed` polling `/api/jobs?summary=1` (2 s active /
  15 s idle / paused hidden) and `/api/fasrc/current-submission`, `cancelJob`. Unit tests with a
  mocked fetch.
- [ ] **T4 Stores.** `state/display.ts` exactly per C7; `state/inspector.ts`, `state/prefs.ts`,
  `state/selection.ts`; persistence guarded by try/catch. Tests.
- [ ] **T5 UI kit v2** (C8 names) on Radix; compat exports for every existing `ui/index.tsx`
  name so current pages compile; `DataTable` (virtualized, sort, filter, select, keyboard, CSV).
  Tests for DataTable sorting/filtering/selection and `confirm()`.
- [ ] **T6 Charts.** `Plot` v2 per C8 (tooltip, legend toggle, zoom/pan/reset, log y, syncKey,
  export PNG/CSV, draw only on input change); `format.ts`, `ticks.ts` with tests.
- [ ] **T7 Shell.** Data router from C1 (`app/routes.ts` builds routes from `spa_routes.json` +
  a `workspaceComponents` map of lazy imports), redirects, `ScrollRestoration`,
  `document.title`, per-route ErrorBoundary (retry, copy details), Suspense skeletons; `Rail`
  (icons, collapse, drawer < 900 px, badges), `TopBar` (breadcrumbs, palette, connection badge,
  job tray, display button, theme toggle, version banner from `/api/version`),
  `CommandPalette` (routes + tabs + page actions + global commands), shortcuts + `?` sheet,
  `Inspector` panel (resizable, `?inspect=`), `JobTray` (list, cancel, open log in inspector,
  toasts on finish/fail), `DisplayPanel` (edits C7 store; image section; "Sky" section left as
  an extension point for WP-W-Sky).
- [ ] **T8 Workspace shells with legacy adapters.** Create `src/workspaces/<id>/index.tsx` for all
  9 workspaces with router-linked tabs. Until phase 3 replaces them, each tab renders the
  corresponding existing page component (e.g. `realism/noise` → `pages/Noise`), `sky/atlas` →
  `pages/JwstEuclid`, `sky/results` → `pages/Inference`, `sky/catalog-eval` →
  `pages/Evaluation`, `ensemble/:mode/<tab>` → `pages/Ensemble` (all tabs) and `train` →
  `pages/TrainMembers`, `data/records` → `pages/Sky`, `ops/fasrc` → `pages/Fasrc`, etc.
  `home` → a minimal dashboard (KPIs from existing endpoints + `/api/version`),
  `settings/appearance` and `settings/about` → new small pages. Tabs with no legacy page render
  an EmptyState "arrives in phase 3". Remove `registry.ts`, `App.tsx` special cases,
  `Placeholder` lane text, rail legacy CSS, dead `Toolbar*`, `qs`.
- [ ] **T9 FOUNDATION.md** documenting C7/C8 with examples (how to build a workspace tab, open
  the inspector, register palette actions, use DataTable/Plot/useUrlState/useJob/confirm/toast).
- [ ] **Acceptance:** `npm run typecheck`, `npm run lint`, `npm test` green; every old page still
  reachable through its new URL; orchestrator build succeeds.

**Phase-1 gate (orchestrator):** merge nothing (same checkout); run full pytest, ruff, npm
typecheck/lint/test, `npm run build`; restart the preview server; browser-check every
workspace/tab in light+dark for crashes; commit + push.

---

## 3. Phase 2 (parallel): WP-V viewer engine v2 ∥ WP-B2 sky/real backend

### WP-V — Viewer engine v2
**Owns:** `frontend/src/viewer/**` (new), `frontend/src/legacy.tsx` (becomes a compat wrapper),
`frontend/src/viewer.css` (moved), `euclid_polish/web/static/cutout_viewer.js` (delete at the
end), `scripts/check_viewer_parity.mjs`, `scripts/_viewer_parity_ref.py`,
`tests/test_web.py` (only the `cutout_viewer.js` source test → removed/replaced).

- [ ] **T1 Golden parity data.** Extend `_viewer_parity_ref.py` to write
  `frontend/src/viewer/__fixtures__/color_golden.json` (per band constants, primitives, temp
  chain, lupton, gray, gray-log, direct-rgb, displayScale, magnitude for small synthetic cubes)
  computed from the CURRENT `cutout_viewer.js` via node (so the port is checked against the old
  engine) and from `visualization/color.py` where the old engine claims parity.
- [ ] **T2 Port pure modules** (`color.ts`, `selection.ts`, `cube.ts`, `wcs.ts`, `movie.ts`,
  `export.ts`) with vitest suites reproducing the goldens (tolerances as in the old parity
  script) + WCS tests (TAN/SIN, CD and PC+CDELT, round trip) + LRU/abort tests.
- [ ] **T3 React engine** `ImageViewer` (+ `TierGrid`, `Frame`, `Lens`, `Toolbar`, `Nav`,
  `ReadoutBar`, `HistogramPanel`, `ProfilePanel`) implementing every old feature and the new ones
  (spec §6), bound to `useDisplay` (C7) with per-viewer override, URL state (`useUrlState` with a
  viewer id prefix), keyboard scoped to the focused viewer, pointer events, JSON error display.
  Public API: `<ImageViewer collection params? tiers? initialIndex? id? urlKey? onState?
  onReady?(api) toolbar?="full"|"compact"|"none" className?/>` and `ViewerApi` (superset of the old
  one: `goTo, goToId, setTiers, setView, setParams, setMorphMembers, getState, exportFigure,
  saveCropToResults, reload, zoomTo(ra,dec,fovArcsec), getReadout`).
- [ ] **T4 Compat.** `legacy.tsx` `CutoutViewer` and `loadColorEngine` re-implemented on top of
  `ImageViewer`/`color.ts` with the same props, so all existing pages work unchanged; delete
  `static/cutout_viewer.js`; replace the source-text test in `tests/test_web.py`.
- [ ] **T5 README** `src/viewer/README.md` (API, URL keys, display binding, how residual/blink/
  histogram/profile work, how to add a collection tier).
- [ ] **Acceptance:** typecheck/lint/test green; parity suite green; orchestrator browser check of
  every page that mounts a viewer (records, cutouts, psfs, visual, sky results/NEXUS, catalog-eval,
  ensemble disagreement) incl. pan/zoom/readout/RA-Dec on NEXUS and archive-fields.

### WP-B2 — Real tile store, model catalogue, experiments, sky atlas backend (C9)
**Owns:** new `euclid_polish/web/helpers/{real_tiles.py, model_catalog.py, experiments.py,
real_metrics.py, sky_atlas.py}`, new `routes/{real.py, sky_atlas.py}`, `routes/__init__.py`
(add modules), `helpers/viewer_data.py` (add the `real` collection only), `helpers/jwst_euclid.py`
(tile-subset + model-spec NEXUS inference, footprint polygon in the manifest, discovery job
wrapper; move its function-scoped imports to module top), `helpers/real_field.py` (list all
fields), `euclid_polish/sky/observation/q1_mer_tiles.json` (new, committed), tests.

- [ ] **T1 Q1 tile polygons.** Build `q1_mer_tiles.json` from
  `data/population_comparison/mer_noise_levels_64px/q1_vis_mosaics.csv` (352 rows: `tile`,
  `ra`, `dec`, `polygon` [[ra,dec]…], `field` via `q1_field_for`), joined with
  `mer_noise_levels.json` levels where present; a loader + spherical point-in-polygon; tests
  (NEXUS centre ∈ tile 102158584; JADES (53.16,−27.78) ∈ 102044185; COSMOS ∉ Q1).
- [ ] **T2 Real tile store** (`real_tiles.py`): sources `nexus, tile, field, archive, eval,
  poster, pair`; `list_tiles(source)`, `get_tile(source, id) → RealTile` (LR electrons HWC float32
  + FITS WCS header + extras), unit conversions (ADU/s→e⁻ via MAGZERO as elsewhere),
  poster WCS construction; `cache_tile(ra, dec, progress)` for 25.6″ four-band tiles (256² VIS
  grid, NISP registered onto the VIS WCS reusing the NEXUS tile machinery), Q1 coverage check
  first; tests with synthetic FITS fixtures and a monkeypatched downloader.
- [ ] **T3 Model catalogue** (`model_catalog.py`): enumerate specs (C9) with labels,
  fingerprints (member fingerprints; combiner artifact fingerprint), required members,
  availability; `predict(spec, lr_e, members_cache) → SR`; tests with stub members/combiners.
- [ ] **T4 Metrics** (`real_metrics.py`): hole % (brightest 1% LR pixels where SR < 0.5 ×
  LR/4 per SR pixel), enclosed-flux R over bright (>100σ, robust σ), locally dominant (±1.5″)
  peaks with boxes 0.3–1.7″ after the central-pixel-fraction artifact cut (VIS ≤ 0.25, NISP ≤
  0.14), % R<0.8, % R<0.5, median R, total SR/LR flux ratio per band; tests on synthetic
  Gaussians/squares with known answers (flux-conserving SR → R≈1, 0% holes; zeroed core → 100%
  holes in that region).
- [ ] **T5 Experiments** (`experiments.py`, `routes/real.py`): job computing needed member SRs
  once per tile (cache keyed by member fingerprint under
  `data/euclid_inference/experiments/cache/`), applying specs, saving outputs + metrics + an
  experiment record; list/get endpoints; delete outputs; tests with stubs.
- [ ] **T6 `real` viewer collection** in `viewer_data.py` (tiers `lr`, `jwst`, `m:<spec>` with
  WCS/units per C6).
- [ ] **T7 Sky atlas** (`sky_atlas.py`, `routes/sky_atlas.py`): layer catalogue + features for
  every layer in spec §7.1 "Coverage / Real results / Catalogues" (compact arrays; large layers
  like stars as `[ra, dec, mag, flags]` rows; population cones as circles), `at`, JWST discovery
  job (wrap `scripts/find_jwst_euclid_overlap.py` discover with MAST s_region into
  `data/jwst_euclid_overlap/`), footprints cone query, pair download (existing
  `download_and_align_pair` / `download_nexus_pair`) as a `pair` real source. Tests for layer
  shapes with fixtures.
- [ ] **T8 image.fits** endpoint (2-D slice with WCS for LR/SR/JWST of any real tile).
- [ ] **Acceptance:** full pytest green; ruff clean; `API.md` updated with every C9 shape;
  `curl` smoke of each endpoint against local data documented in the report.

**Phase-2 gate:** as phase 1 + browser check of the viewer features; commit + push.

---

## 4. Phase 3 (parallel): workspaces (full-stack, one owner each)

Each WP owns `frontend/src/workspaces/<id>/**` and the backend modules listed; it replaces the
legacy adapters from WP-F T8 with the spec §7/§8 design, deletes the old `src/pages/<X>.tsx`
files it absorbs (and their CSS), and adds vitest tests for its pure logic plus pytest for its
backend changes. Shared UI goes through the foundation; a missing foundation primitive is
reported, not forked. Every tab: URL state, palette actions, inspector kinds for its entities,
EmptyState/Skeleton/error states, light+dark.

| WP | Workspace | Absorbs (pages) | Backend owned |
|---|---|---|---|
| W-SkyAtlas | `sky/` (index + `atlas` tab + `src/sky/**` Aladin engine + DisplayPanel "Sky" section) | JwstEuclid | none (uses C9) |
| W-SkyResults | `sky/tabs/{Results,Experiments,CatalogEval}.tsx` + inspector kinds `realtile`, `experiment` | Inference, Evaluation | `routes/evaluation.py`, `routes/model.py` |
| W-Ensemble | `ensemble/` | Ensemble, SpatialGateCard, KneePsnrPanel, TrainMembers | `routes/ensemble.py`, `helpers/ensemble_viz.py` (combiner variants list/compare/fit knobs/promote, members joined payload, calibration, curves), `scripts/fit_spatial_gate.py` (import compare logic into a helper) |
| W-Realism | `realism/` | Noise, GalaxyDistributions, GalaxyCorner, JointPairExplorer, StarDistribution, PopulationComparison, SyntheticReal, archiveFields, galaxyFwhm | `routes/{noise,galaxy_distributions,star_distribution,population_comparison,archive_fields}.py` (readiness overview endpoint) |
| W-Data | `data/` | Sky (records), Catalog, Cutouts, Psfs, Tng | `routes/{views,cutouts,psfs,tng,auth}.py` (stars JSON endpoint, clean tier/truth overlay data, sync-as-job, TNG property JSON) |
| W-Figures | `figures/` | Visualization, figure-grid | `routes/{viewer,poster}.py` (results rename/delete, thumbnails) |
| W-Inspect | `inspect/` | Inspect | `routes/files.py` + `helpers/{fits_render,paths}.py` (roots, HDU slices, table HDU rows, stats) |
| W-Ops | `ops/` | Fasrc, fasrc.tsx pieces, Tracking, Git | `routes/{fasrc,tracking,git}.py`, `git_ops.py`, new `routes/provenance.py` |
| W-Settings+Home | `settings/`, `home/` | Config | `routes/config.py`, `routes/auth.py` (single session status) |

**Phase-3 gate:** orchestrator integration: full test matrix, build, browser sweep of every
tab, adversarial review workflow (per workspace: correctness, UX, a11y, perf), fix loop; commit
+ push; memory update.
