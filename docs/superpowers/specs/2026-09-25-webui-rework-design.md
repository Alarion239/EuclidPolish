# EuclidPolish web console rework — design

Date: 2026-09-25. Status: approved by the user (navigation → workspaces; sky engine → Aladin
Lite + online HiPS; experiments → all four, with **25.6″ real tiles** instead of 256″ fields;
legacy → delete all dead code).

Source material: a 10-agent read-only map of the console (foundation, viewer engine, three page
groups, backend endpoint inventory, real sky data, current research work, sky tech research,
completeness critic). Facts quoted below were verified against the code on 2026-09-25.

---

## 1. Goals

1. A neat, fast console with comfortable controls: collapsible navigation, a command palette,
   keyboard shortcuts, a global job tray, toasts, resizable panels, URL-addressable state
   everywhere (every view is a shareable link).
2. "Inspect virtually anything": a global **Inspector** side panel for members, tiles, fields,
   sources, jobs, FITS files and figures; an image viewer with pixel readout, RA/Dec, pan/zoom,
   residuals, histograms and profiles; charts with hover readouts, zoom and export.
3. A global **Display (colour) panel** that drives every image viewer consistently, plus an
   appearance panel (light / dark / system theme, accent, density).
4. A **Sky** workspace: a celestial sphere (Aladin Lite v3) showing Euclid and JWST coverage and
   every REAL local result, where the user can experiment directly: compare models on real
   tiles with real-data metrics, cache a new 25.6″ four-band tile anywhere in Q1, discover JWST
   observations and pair them with Euclid, and overlay LR/SR/JWST pixels on the sky.
5. Remove redundancy and legacy; add the functions the current research needs (multi-knee
   members, combiner variants/compare/promote, knee-PSNR leaderboard, real-data benchmark,
   schema-driven FASRC steps, provenance/staleness everywhere).

Non-goals: changing the science pipelines' behaviour (except the explicit bug fixes in §11),
moving the science code, migrating off Flask/React/Vite.

## 2. Locked constraints

- React 18 + Vite + TypeScript SPA in `euclid_polish/web/frontend/`, building into the
  **committed** `euclid_polish/web/static/dist/` (time-travel sandboxes serve the committed dist
  and never run npm). Keep committing the build.
- Light theme is the default; dark and system are options. All colours come from CSS tokens.
- Viewer colour math keeps parity with `euclid_polish/visualization/color.py` and the locked
  **absolute asinh** transfer (black / knee / white; `K0 = meta.color.default_asinh`, white
  30·K0) as the DEFAULT (`docs/superpowers/specs/2026-06-24-unified-cutout-viewer-design.md`).
  New stretches/colormaps are opt-in.
- Imports at module scope only (user rule; conditional imports caused a production
  `UnboundLocalError`). Applies to every new/touched Python file.
- Ensemble-only model: never load `Model` directly; a single model is an ensemble of one.
- STARFULL is the default regime everywhere (starless is opt-in).
- Offline-first: every local-data page must work with FASRC disconnected.
- No experimental lanes, HST, round-trip, two-stage chain, lens-finder, redshift scheme or BerHu
  surfaces (all deleted/deprecated; do not resurrect).
- Tests green before each commit; commit on `main` and push (user's standing rule).

## 3. Information architecture

Nine workspaces. The route manifest (`src/app/routes.ts`) is the single source of truth for the
router, the rail, the command palette and (via a generated JSON) Flask's SPA page matcher.

| Workspace | Route | Tabs (URL segment) | Replaces |
|---|---|---|---|
| Home | `/` | — | redirect to Ensemble |
| Sky | `/sky` | `atlas` (default), `results`, `experiments`, `catalog-eval` | JWST × Euclid, Inference, Evaluation |
| Ensemble | `/ensemble/:mode` (`starfull`\|`starless`) | `overview`, `members`, `curves`, `knee`, `diagnostics`, `combiners`, `disagreement`, `train` | Ensemble, Train members |
| Realism | `/realism` | `overview`, `noise`, `galaxies`, `stars`, `pixels`, `visual` | Noise, Galaxy distributions, Star distribution, Field statistics, Synthetic–Real |
| Data | `/data` | `records`, `catalog`, `cutouts`, `psfs`, `tng` | Sky (TFRecords), Catalog, Cutouts, PSFs, TNG |
| Figures | `/figures` | `grid`, `plates`, `results` | Visualization |
| Inspect | `/inspect` | — (`?path=&hdu=&slice=`) | FITS inspector |
| Ops | `/ops` | `jobs`, `fasrc`, `tracking`, `git`, `provenance` | FASRC, Tracking, Git |
| Settings | `/settings` | `config`, `connections`, `appearance`, `about` | Config, connection-error |

Tab routes are `/<workspace>/<tab>`; Ensemble is `/ensemble/:mode/<tab>`. Old URLs redirect
(client-side and 308 server-side): `/config→/settings/config`, `/catalog→/data/catalog`,
`/psfs→/data/psfs`, `/cutouts→/data/cutouts`, `/tng→/data/tng`, `/noise→/realism/noise`,
`/galaxy-distributions→/realism/galaxies`, `/star-distribution→/realism/stars`,
`/population-comparison→/realism/pixels`, `/synthetic-real→/realism/visual`,
`/jwst-euclid→/sky`, `/inference→/sky/results`, `/evaluation→/sky/catalog-eval`,
`/train-members→/ensemble/starfull/train`, `/tracking→/ops/tracking`, `/visualization→/figures`,
`/fasrc→/ops/fasrc`, `/git→/ops/git`, `/connection-error→/settings/connections`.
The old `/sky` (training TFRecords) now lives at `/data/records`; `/sky` is the atlas.

## 4. Shell

- **Rail** (left): brand, 9 workspace entries with icons, collapsible to icons (state persisted),
  becomes a drawer below 900 px. Nav badges (e.g. running jobs on Ops, stale alerts on Home).
- **Top bar**: breadcrumbs (workspace › tab › entity), ⌘K palette trigger, FASRC connection
  badge (click → Settings/Connections), job tray button with running count, Display panel
  button, theme toggle, "server behind HEAD — restart" banner when `/api/version` says so.
- **Command palette** (`cmdk`): every route/tab, every action registered by the current page,
  plus global commands: go to RA/Dec or object name (Sky), open NEXUS tile N, open member N,
  open FITS path, run job X, toggle theme, open Display panel. Fuzzy search.
- **Keyboard**: a global shortcut registry (`tinykeys`) with a `?` cheat sheet. Page shortcuts
  register/unregister on mount. Viewer shortcuts are scoped to the focused viewer.
- **Inspector** (right, `react-resizable-panels`, collapsible, width persisted): renders any
  entity `{kind, id}` pushed into the inspector store, e.g. `member:member_196`,
  `tile:nexus/123`, `realtile:<id>`, `source:<layer>/<id>`, `job:local/<id>`,
  `job:slurm/<jobid>`, `fits:<path>`, `figure:<id>`. Reflected in the URL as `?inspect=kind:id`
  so inspection is shareable. Pin + back/forward history inside the panel.
- **Job tray**: popover listing local jobs (`/api/jobs?summary=1`) and SLURM jobs
  (`/api/fasrc/current-submission`), progress, ETA, cancel, open log (Inspector), toasts on
  finish/fail. One shared poll (2 s while anything runs, 15 s idle, paused when the tab is hidden).
- **Scroll reset** and `document.title` per route. Per-route ErrorBoundary with retry and "copy
  details". Suspense skeletons for lazy routes.

## 5. Frontend foundation

Dependencies (pin exact versions in package.json; React 18 compatible):
`@tanstack/react-query` v5, `zustand` v5, `cmdk` v1, `sonner` v2, `tinykeys`,
`react-resizable-panels`, `@tanstack/react-virtual` v3, Radix primitives (`@radix-ui/react-dialog`,
`-popover`, `-tooltip`, `-dropdown-menu`, `-slider`, `-switch`, `-tabs`, `-toggle-group`,
`-context-menu`, `-scroll-area`), `aladin-lite@3.8.2` (exact; lazy chunk only). Dev: `vitest`,
`@testing-library/react`, `happy-dom`, `eslint` + `typescript-eslint` + `eslint-plugin-react-hooks`.

Structure (`src/`):
- `app/` — `routes.ts` (manifest), `App.tsx` (data router `createBrowserRouter`, lazy routes,
  `ScrollRestoration`), `Shell.tsx`, `Rail.tsx`, `TopBar.tsx`, `CommandPalette.tsx`,
  `JobTray.tsx`, `Inspector.tsx` (+ `inspectors/` registry), `DisplayPanel.tsx`.
- `api/` — `client.ts` (`apiGet`, `apiPost` (form or JSON), typed `ApiError{status, message,
  body}`; the server's `{error}` text is preserved; 503 "FASRC not connected" is distinguishable),
  `query.ts` (QueryClient; `useResource(url, opts)` compatibility wrapper returning
  `{data, loading, error, reload}` over TanStack Query with in-flight dedupe, shared subscribers,
  visibility-aware polling, retry for idempotent GETs), `jobs.ts` (`useJob`, `useJobsFeed`),
  per-domain typed endpoint modules.
- `state/` — zustand stores: `display` (colour panel), `inspector`, `prefs` (theme, accent,
  density, rail collapsed), `selection` (cross-view selection: e.g. selected members/tiles).
  Persisted slices use try/catch-guarded localStorage.
- `hooks/` — `useUrlState` (typed search-param state with defaults and parsers),
  `useShortcut`, `usePageActions` (register palette actions).
- `ui/` — the kit rebuilt on Radix: Button (variants, sizes, loading, icon, asChild link),
  IconButton, Tooltip, Popover, Dialog/ConfirmDialog (replaces `window.confirm`), Menu,
  ContextMenu, Tabs (router-linked), Segmented, Switch, Checkbox, Slider (single/range, log
  scale), NumberField, Input, Select (searchable, multi), Field (label + hint popover), Card,
  Section, Badge, Chip, Stat/KPI, DefList, Callout, EmptyState, Skeleton, ProgressBar, LogView
  (search, follow, copy), JsonTree, CopyButton, Kbd, DataTable (see below). Focus-visible rings,
  ARIA roles, `type="button"` everywhere.
- `ui/DataTable` — TanStack-Virtual-backed table: column defs, sort (multi), text filter,
  column visibility, row selection (checkbox, shift-range), keyboard row navigation, row click →
  inspector, CSV export, sticky header, thousands of rows.
- `charts/` — `Plot` v2 (keeps the existing canvas renderer and props) adding: hover crosshair +
  tooltip with nearest-series readout, legend click-to-toggle and hover-highlight, box-zoom /
  wheel-zoom / pan / reset, log **y** axis, linked cursors (`syncKey`), PNG/SVG-free CSV + PNG
  export, ARIA label; `draw` only when inputs change. Shared `format.ts` (number, SI, bytes,
  duration, magnitude, coordinates sexagesimal/degrees, dates) and `ticks.ts` (linear, log,
  decade, magnitude) replace the 15 per-page tick helpers.
- `theme/` — tokens (`tokens.css`): surfaces, ink, accent, status, series, categorical,
  **band colours** (`--band-vis/-y/-j/-h`), **loss colours** (l1/l2/l3/mse), type scale
  (`--fs-*`), spacing, radius, z-index, motion, shadows; light + dark (+ system via
  `prefers-color-scheme`). Fix every undefined token (`--line --muted --mono --panel --bg
  --surface --surface-subtle --warning-text --radius-sm --r-1 --r-2 --danger --s8 --guide
  --text-sm`) and the missing `.muted` class; raise `--text-faint`, light `--warn`/`--good`
  contrast to WCAG AA.
- `viewer/` — viewer engine v2 (§6).
- `sky/` — Aladin wrapper (§7).
- `workspaces/<name>/` — one folder per workspace; each tab is its own lazy module; page CSS is
  co-located and scoped by a workspace class (no cross-page CSS dependencies).

Build: `"build": "tsc --noEmit && vite build"`; `"test": "vitest run"` (existing node tests
ported); `"lint": "eslint ."`. Vite `manualChunks`: vendor, aladin (lazy). Fix the dev server
(proxy every non-page prefix to `FLASK_ORIGIN || http://localhost:9777`, keep Origin,
`base` only in build). Build also emits `static/dist/spa-routes.json` (page path patterns) for
Flask.

## 6. Viewer engine v2 (`src/viewer/`)

Port `static/cutout_viewer.js` (2,640 lines) into typed modules inside the bundle and delete the
static file. The backend wire format stays (Float32 HWC cubes + `X-Cube-*` headers), extended.

Modules: `color.ts` (primitives, `prepareCore`, `transferCore`, Lupton, temp, direct-RGB,
gray-log — ported verbatim; new stretches and colormaps added as separate code paths),
`cube.ts` (fetch with AbortController, header parsing, JSON error surfacing, LRU by bytes),
`selection.ts` (lens/crop geometry, angular matching across pixel scales, receptive-field tags),
`wcs.ts` (TAN/SIN pixel↔sky from header keywords; CD or PC+CDELT), `movie.ts` (PCA morph,
worker-friendly), `export.ts` (PNG, publication figure, webm), React components `ImageViewer`,
`TierGrid`, `Frame`, `Lens`, `Toolbar`, `Nav`, `ReadoutBar`, `HistogramPanel`, `ProfilePanel`.

Features (all old features kept: tiers, hidden tiers, layouts, colour modes q–y keys, knee/gain
per transfer group, magnifier lens synchronized across tiers, freeze + save crop to results,
PNG/figure/video export, movie with member subset, BHR FWHM slider, JWST band chips, prefetch,
magnitude overlay with ±σ):
- **Pan/zoom** of the main frames (wheel, drag, double-click reset, keyboard +/−), synchronized
  across tiers in angular coordinates; the lens stays available (hold Alt or toggle).
- **Pixel readout** bar: x/y (per tier pixel grid), RA/Dec (sexagesimal + degrees) when WCS is
  known, value per band with units (e⁻, MJy/sr, ADU/s), for every visible tier at the same sky
  position; synchronized crosshair.
- **Residual tiers** computed client-side: A−B, A/B, (A−B)/σ for any pair of loaded tiers with
  matching grids (resampled by integer factor when needed), shown with a diverging colormap.
- **Blink** (cycle 2+ tiers in one frame) and **swipe** comparison.
- **Histogram panel** of the visible region with draggable black/white points (opt-in manual
  cuts), and **line/radial profile** panel (shift-drag a line; click a point for radial).
- **Display panel binding**: colour mode, custom band→RGB mapping, stretch (absolute asinh
  default; linear, log, sqrt, asinh-auto, zscale/percentile opt-in), knee, gain, black point,
  colormap (gray, viridis, magma, inferno, cividis, RdBu diverging for residuals), invert, NaN
  colour. "Link all viewers" (global) vs per-viewer override.
- **URL state** per viewer instance (`v.<id>.i`, tier list, zoom/center, colour) so a view is
  shareable; `?id=` lookup instead of positional index where the backend supports it.
- Touch/pointer events, focusable canvas, ARIA labels, no page-scroll trap (wheel zoom only when
  the viewer is focused or Ctrl/⌘ is held; plain wheel scrolls the page unless the lens is
  active — make this a setting).
- Errors from the backend are shown verbatim (not "not synced yet").

Parity: `scripts/check_viewer_parity.mjs` + `_viewer_parity_ref.py` extended to emit golden
values for every colour mode; a vitest suite consumes them. `tests/test_web.py`'s literal source
assertions on `cutout_viewer.js` are replaced by behavioural tests.

## 7. Sky workspace (`/sky`)

### 7.1 Atlas tab (the celestial sphere)

Aladin Lite v3.8.2 wrapped in `src/sky/engine.ts`: lazy `import("aladin-lite")`, `await A.init`
(a promise property), ONE instance created on first visit and kept alive in the shell (hidden
with `visibility:hidden` when leaving the route — no `destroy()` exists and WebGL contexts
leak), `log:false`, single `al.on(...)` per event fanned out through our own emitter (Aladin has
no `off`), theme applied on app theme change, logo/credits kept.

Layout: left **Layers** panel, centre sky, right Inspector; bottom status bar (cursor RA/Dec in
degrees + sexagesimal, galactic toggle, FoV, projection, pixel value of the base HiPS via
`readPixel`).

Layers panel (each with visibility, opacity, colour, count, legend, "zoom to"):
- **Background HiPS** (radio): Euclid Q1 colour (CDS `…/Euclid/Q1/CDS_P_Euclid_Q1_color`),
  Euclid Q1 VIS / NISP Y J H (FITS tiles; colormap/stretch/cuts preset from `hips_pixel_cut`),
  ESA Euclid VIS, DSS2 colour, 2MASS colour, PanSTARRS DR1, unWISE, DESI LS DR10. Per-layer
  colour controls (colormap, stretch, cuts, gamma, saturation, reverse) in the Display panel's
  "Sky" section.
- **Overlay HiPS** (opacity): JWST NIRCam/NIRISS/MIRI (ESA ESASky), CDS JWST F115W…F444W.
- **Coverage**: Euclid Q1 MOC (CDS `Moc.fits`), JWST MOC (MocServer union of ESAVO/P/JWST/*),
  Q1 MER tile polygons (352, from the committed `euclid_polish/sky/observation/q1_mer_tiles.json`,
  coloured by noise depth where known, rejected/unobserved tiles marked), deep-field circles.
- **Real results** (local; work offline): NEXUS F200W footprint + 445 tiles (coloured by SR
  state: current / stale / missing), cached 25.6″ real tiles, legacy real fields (all, not only
  latest), poster target(s), JWST×Euclid pairs, archive fields (220; EDF-F/S labels corrected
  from position), eval objects (583 real, coloured by `flux_ratio_sr_over_lr` or grade),
  experiment runs.
- **Catalogues**: Q1 lens candidates (2,584, by grade), stars (FASRC-cache `stars.csv`,
  43,401, by magnitude; never the stale 200-row local copy), PSF clusters (30, by FWHM), noise
  sample positions (294 by level per band), population cones (24), Gaia fields (3), JWST MAST
  footprints (after discovery; polygons fetched per view when zoomed in).
- Level of detail: tiles < ~2 px render as centroid markers / MOC at low zoom and polygons when
  zoomed in (Aladin skips sub-pixel polygons).

Interactions: click a feature → Inspector shows its card (metadata, staleness, provenance,
LR/SR/JWST mini viewer, actions); hover tooltip; rectangle/circle/polygon selection
(`al.select`) → selection list with bulk actions; right-click on empty sky → "cache a 25.6″
tile here", "what covers this point", "copy coordinates", "open in ESASky/SIMBAD"; palette
"go to <name|RA Dec>" (Sesame via Aladin `gotoObject`), "zoom to EDF-N/EDF-S/EDF-F/NEXUS";
URL `?ra&dec&fov&proj&base&ov&layers&sel&inspect`.

**Overlay tiles on the sky**: from a tile card, "Overlay LR / SR(<model>) / JWST" adds an
`A.image` layer from `GET /api/real/<source>/<id>/image.fits?tier=&band=` (2-D slice with
celestial WCS) with opacity, blink between overlays, and a colour matching the Display panel;
also available for a whole selection (e.g. all NEXUS tiles' SR as a mosaic at high zoom).

### 7.2 Real results tab

One DataTable over every real tile source (§9.1): source, id, RA/Dec, field, available tiers,
models computed (with staleness vs the current production model), metrics, disk size. Row →
Inspector; bulk select → run models, compute metrics, delete cached outputs. Replaces the
Inference page (legacy real field is listed with all its 100 sub-tiles).

### 7.3 Experiments tab (model comparison on real tiles)

Pick tiles (from the atlas selection or the results table) and models (§9.2 model catalogue:
production gate, mean, any active member, any gate variant, RBF) → a local job runs and caches
every (tile, model) SR → a comparison view: ImageViewer with LR + one tier per model (+ JWST
when available, + residual tiers), and a metrics table/plots per model: per-band **hole %**
(brightest 1% of LR pixels where SR < 0.5 × LR flux per SR pixel), **enclosed-flux R**
statistics over bright (>100σ) locally dominant peaks with boxes 0.3–1.7″ after the
central-pixel-fraction artifact cut (VIS ≤ 0.25 / NISP ≤ 0.14 sources vs ≥ 0.26 / ≥ 0.19
artifacts): % peaks with R < 0.8 and R < 0.5, median R; total SR/LR flux ratio per band; gate
weight maps for gate models when available. Runs are persisted and listed (history), each
logged to tracking with one click. Definitions follow the 2026-09-23/25 analysis recorded in
memory (`project_combiner_real_galaxy_holes`).

### 7.4 Catalog-eval tab

The old Evaluation page reorganised: reconstruction browser over `data/eval_results` (now with
provenance/staleness badges and RA/Dec, row → atlas), runs table (DataTable with all manifest
columns, filters), query galaxies (uses the single Euclid session from Settings), fetch lens
catalogue, grouped analysis job (now STARFULL + production combiner, not the mixed-regime plain
mean), FASRC sync (requires a confirm dialog because it runs `rsync --delete-after`),
transformation / power-spectrum figures rendered on demand (no `fresh=1` on every mount).

## 8. Other workspaces

### 8.1 Home
KPI tiles: FASRC connection, server version (behind HEAD?), running jobs, production model
(active STARFULL members, production combiner kind, knee-integrated PSNR headline, gain over
mean and best member), staleness alerts (real SR products vs production model; combiner vs
membership; evals vs records; TFRecords vs `NOISE_MODEL`; tracking log last entry), recent
jobs, quick actions (evaluate, fit gate, open sky), small sky thumbnail linking to `/sky`.

### 8.2 Ensemble (`/ensemble/:mode/<tab>`)
- **overview**: headline numbers with explicit definitions (production combiner PSNR, plain
  mean, best member, knee-integrated), run actions (evaluate with `force` option; refresh
  member PSNR; pull with member picker), staleness banners.
- **members**: ONE unified DataTable joining status + `origin.json` (loss, depth, knee(s) —
  multi-knee members shown as "multi ×6 → 10" not "100e", output knee, knee loss, noise aug,
  bootstrap, ICNR, seed, commit, forked_from), steps vs `target_steps` (progress; TIMEOUT
  flag), per-band knee-integrated PSNR, gate usage, coherence; filter/sort; multi-select →
  continue / fork / archive (confirm) / show in disagreement / compare curves; row → member
  Inspector (origin, per-band curves, loss (fix the loss-series overwrite at
  `helpers/ensemble_viz.py:497`), knee curve, gate usage). Archived members table (name, date,
  zip, commit).
- **curves**: training curves (per-band PSNR, loss, gnorm, step time) with hover identification,
  colour by loss/depth/knee/multi-knee, linked cursor.
- **knee**: knee-PSNR curves + sortable leaderboard (integrated per band + mean, rank delta),
  selectable integration range, CSV export, compute job. No 100 e⁻ reference line.
- **diagnostics**: power spectrum, coherence, std-vs-error, combiner axes, std-vs-brightness,
  calibration (z-pdf, coverage, per-field σ vs RMSE — currently computed but never shown),
  transfer function T(k); pixel trace on heat-cell click.
- **combiners**: variant registry (every `spatial_gate_*` dir + RBF) with `fit_meta`,
  held-out loss curves overlaid, knee-integrated PSNR per variant, real-data benchmark summary
  (from Sky experiments), **compare** job (wraps `scripts/fit_spatial_gate.py compare` logic),
  **fit** with full knobs (mix space, loss knees, LR input, width, steps, lr, crop, members
  picker from the table, out-name) writing a named variant — never overwriting production —
  and **promote** (backs up the current production artifact automatically, confirm dialog).
  RBF cards demoted (stale; the frozen-RBF card is removed since it was never fitted).
- **disagreement**: viewer v2 over the ensemble collection (production gate is the `sr` tier;
  plain mean is its own tier) + member picker driving the subset movie.
- **train**: TrainMembers with presets ("repeat last batch recipe"), multi-knee knobs
  (`asinh_knees`, `output_knee`, `knee_loss`), `fork_track`, `base_seed`, `evaluate_every`,
  member picker for continue/fork, preview of new member names and the final command, clone from
  a past job; geometry knobs owned here (Config shows them read-only with a link).

### 8.3 Realism (`/realism/<tab>`)
Shared header: include-training toggle and one training-catalogue sync action. **overview**:
readiness of each prior (galaxy joint model, star prior, TNG radius manifest, noise model
version, comparison-cache freshness) and the `synthetic_generate` gate status. **noise**,
**galaxies** (galaxy distributions; the three joint views kept but sharing one chart kit),
**stars**, **pixels** (field statistics; `source_detection` finally shown), **visual**
(synthetic–real with the viewer's own toolbar available). All hard-coded colours → tokens;
all tick/format helpers → shared modules; dead CSS removed.

### 8.4 Data (`/data/<tab>`)
**records** (training TFRecords viewer with a `clean` tier and a truth-source overlay from
`sources_*.csv`; generate SR with overwrite; sync as a background job with progress;
`synthetic_generate` step with parameters), **catalog** (star catalogue explorer: DataTable over
43k stars with search/filters, magnitude histogram, per-band validity explained, positions link
to the atlas; `euclid_query` step WITH its task parameters — fixes the 200-star overwrite),
**cutouts** (viewer + gallery), **psfs** (viewer, cluster table, "not cached" vs "no empirical
PSF" distinguished, syncs as jobs), **tng** (token, atlas download, radii (async, cached),
grid/stack, interactive property explorer from the local CSVs).

### 8.5 Figures (`/figures/<tab>`)
**grid** (FigureGridBuilder + thumbnails from `panel.png`, rename/delete saved results, saved
layouts), **plates** (population atlas, stellar calibration, galaxy plate, NEXUS plates),
**results** (saved viewer results with WCS badge). Duplicate cards (catalog views, PSF clusters,
FITS path box, training-curve PNG, link-only builder cards) removed.

### 8.6 Inspect (`/inspect`)
File browser over all allowed roots (add `eval_results`, `viewer_results`,
`jwst_euclid_overlap`, `euclid_sky`, repo `poster/`, `output/`, `tracking/`), HDU list, header
table with search, image HDUs of any dimensionality in ImageViewer (band/slice picker, pixel
probe with WCS, stats, histogram), table HDUs as a DataTable (first N rows, column stats),
provenance sidecar, "show on sky" (when WCS), download, "track" (backup to tracking).

### 8.7 Ops (`/ops/<tab>`)
**jobs** (local job centre: all jobs with logs, cancel, results), **fasrc** (current: ALL live
jobs; queue with per-item remove; history across all steps with filters and reconcile of
UNKNOWN states via `refresh-accounting`; logs with search + live tail; storage with remote file
browser inspect/download; FASRC git with local-vs-remote HEAD comparison and env update as a
POST job with streamed output), **tracking** (fixed sandbox render crash; backups; 📌 track from
anywhere; per-model and remote time travel; markdown log; confirmations; archived-campaign
detail; paginated jobs without embedded JSON blobs), **git** (per-file staging and diff viewer,
large/untracked-file guard — commit never runs `git add -A` blindly, push confirmation),
**provenance** (lineage browser over `data/_prov` + sidecars: search, ancestors/descendants,
stale check).

### 8.8 Settings (`/settings/<tab>`)
**config** (JobConfig; saves only dirty fields with an ETag/version check; shows defaults, reset,
"used by <step>" chips from `FASRC_STEP_PARAMS`; hints in popovers), **connections** (FASRC SSH
settings editor via `/api/fasrc/config`, connect/test showing the real error, one Euclid laptop
session, FASRC-side Euclid credentials, TNG token), **appearance** (theme light/dark/system,
accent, density, Display-panel defaults, viewer wheel behaviour), **about** (server boot commit
vs HEAD, dist build, Python/Node versions, local disk usage per data root).

## 9. Backend

### 9.1 Real tile store (`helpers/real_tiles.py`, `routes/real.py`)
A unified abstraction over every real product: `RealTile{source, id, ra, dec, lr_e (H,W,4
electrons), wcs_header, extras (jwst, …)}` with sources:
`nexus` (445 tiles; LR FITS), `tile` (NEW: user-cached 25.6″ four-band tiles, 256² VIS grid, NISP
registered to the VIS WCS like the NEXUS pipeline; stored under
`data/euclid_inference/real_tiles/<id>/`), `field` (legacy 100-tile real fields; all manifests),
`archive` (220 archive fields; ADU/s→e⁻ via MAGZERO), `eval` (eval objects), `poster` (poster
FITS; TAN WCS built from RA/DEC/PIXSCALE), `pair` (JWST×Euclid pairs).
Endpoints: `GET /api/real/sources`, `GET /api/real/<source>` (list with coords + state),
`GET /api/real/<source>/<id>` (card: metadata, models computed + staleness, metrics, files),
`GET /api/real/<source>/<id>/image.fits?tier=&band=` (2-D with WCS),
`POST /api/real/tiles {ra,dec}` (job: Q1 coverage check against local polygons, download
4 bands at 25.6″, register, save LR + WCS, optionally run production + mean),
`DELETE`-style `POST /api/real/<source>/<id>/delete-outputs`.

### 9.2 Model catalogue and experiments (`helpers/model_catalog.py`, `helpers/experiments.py`)
`GET /api/models` lists runnable model specs with labels, fingerprints, member requirements and
availability: `production` (active spatial gate), `mean` (active STARFULL members), `member:<name>`
(each active member), `gate:<variant>` (each `spatial_gate_*` dir whose members are all present),
`rbf`. `POST /api/experiments {tiles:[source/id…], models:[spec…]}` → job: computes the union of
needed member SRs per tile once (cached per member fingerprint), applies combiners, caches every
(tile, spec) output under `data/euclid_inference/experiments/`, computes metrics (§7.3),
writes an experiment record; `GET /api/experiments`, `GET /api/experiments/<id>`. Viewer
collection `real` serves any (source, id) with tiers `lr`, `jwst` (when present), and one tier
per computed model spec (`m:<spec>`), all with WCS. NEXUS whole-field inference keeps working
but accepts a tile subset and a model spec.

### 9.3 Sky atlas (`helpers/sky_atlas.py`, `routes/sky_atlas.py`)
`GET /api/sky/layers` (catalogue: id, kind, count, bbox, style defaults, readiness, fill action),
`GET /api/sky/layer/<id>` (compact features: points `[ra,dec,props]`, polygons `[[ra,dec]…]`,
circles; each with `inspect` `{kind,id}` link), `GET /api/sky/at?ra&dec` (what covers a point:
Q1 tile via spherical point-in-polygon, NEXUS tile, real tiles, pairs),
`POST /api/sky/jwst/discover {region|fields}` (job wrapping `find_jwst_euclid_overlap` with MAST
`s_region`, cached), `GET /api/sky/jwst/footprints?ra&dec&r` (cached MAST polygons in a cone),
`POST /api/sky/jwst/pair {obs_id|ra,dec,…}` (download + align a JWST×Euclid pair, then
optionally run models). Commit `euclid_polish/sky/observation/q1_mer_tiles.json` (352 IRSA
polygons + field label) so coverage works offline. Field labels derive from positions (fixes the
EDF-F/EDF-S swap at display level; the swapped constant in
`scripts/fasrc_download_euclid_sky_cutouts.py` is corrected too).

### 9.4 Viewer backend
JSON errors (`{error}` with status) instead of `abort(code)`; per-object `id`, `ra`, `dec` for
every collection; per-tier WCS (`X-Cube-WCS` JSON header + meta) and units (`X-Cube-Unit`);
`?id=` lookup; ensemble collection: `sr` = production combiner, `mean` tier added, STARFULL
default; >4-channel cubes handled (multi-knee heads) rather than misread.

### 9.5 Platform
- **SPA serving**: Flask reads `static/dist/spa-routes.json` and serves `index.html` for GET/HEAD
  page paths matching those patterns; old page URLs 308 to the new ones. `_REACT_PAGE_PATHS` and
  `_DEPRECATED_PAGE_PATHS` removed. Parity test between the manifest and Flask.
- **FASRC gate**: replace the prefix allowlist with explicit per-route `requires_fasrc`
  (decorator/registry); everything else is local and works offline; SSH-needing handlers return
  JSON 503 `{error:"FASRC not connected"}`. `/api/connection/retry` and `/api/fasrc/config` are
  reachable offline; `/api/fasrc/status` includes `last_error` / startup error.
- **Security**: Host allowlist (`TRUSTED_HOSTS` localhost/127.0.0.1/[::1]); every mutating GET
  becomes POST (e.g. env-update → POST job); keep the Origin guard.
- **Jobs**: cooperative cancel (`POST /api/jobs/<id>/cancel`), result passthrough (small JSON
  results in `to_dict`), eviction (keep the newest N finished), `GET /api/jobs?summary=1`
  without logs, optional per-kind concurrency guard for TensorFlow-heavy jobs.
- **Version**: `GET /api/version` {boot commit, HEAD, behind/dirty, dist build hash, started_at}.
- **FASRC steps**: each registered step declares `task_params` (name, type, default, min/max,
  choices, help) in `/api/fasrc/steps/status`; the SPA renders them generically, prefilled from
  the last successful run's params (`history.match`), with "clone run"; `euclid_query` defaults
  are the last real run (10,000 stars, mag 18–19, snr 50), never 200. The ignored partition
  field is either honoured or hidden. `current-submission` returns all live jobs.
- **Legacy removal** (§10).

## 10. Legacy removal

Delete: all 21 templates; `static/{ensemble_combiner,ensemble_evals,ensemble_train_curves,
fasrc_step_card,job_status}.js`, `static/style.css`, and `static/cutout_viewer.js` (after the TS
port); dead page handlers (`catalog.py` and `sky.py` modules; `config_page`, `cutouts_page`,
`cutouts_gallery`, `ensemble_page`, `fasrc_page`, `git_page`, `inference_page`,
`training_page`, `psfs_page`, `tng_page`, `tracking_page`, `visualization_page`,
`inspect_fits_page`, `evaluation_page` + `_list_catalogs`, `index`, the POST branch of
`connection_error_page`; `_cutout_layout_status`); unreferenced endpoints (`/ensemble/render` +
`job_ensemble_render`, `/ensemble/eval-plot/*`, `/view/star-cutout`, `/api/jwst-euclid/saved`,
`/nexus/options`, `/field/<id>/<kind>` PNG, `/api/fasrc/eta`, `/api/fasrc/jobs`,
`/api/fasrc/mirror/start|stop`, `/api/fasrc/runs/ckpt-bundle.tar`); superseded endpoints
(`/ensemble/power-spectrum.png`, `/api/euclid-psf/preview`, `/api/sky/totals`,
`/eval-files/*` PNG renderer, `/api/fasrc/runs/training-plot.png`,
`/api/fasrc/training-status`, `/api/fasrc/log/<jobid>` SSE, `/api/fasrc/stages/<jobid>`, legacy
`/api/fasrc/submit` (port its queue tests to the step submit first), `/star-cutout/inspect`,
`/sky/inspect`, `/sky/fits`, GET `/api/fasrc/env-update`); orphan functions
(`jobs_impl._job_generate_reconstruct`, `_forward_model_sr_residual`,
`_job_reconstruct_euclid_cutout`; `_jwst_filter_tint`, `_pair_asinh`, `_arrays_to_fits_bytes`;
the dead calibration lanes in `helpers/population_calibration.py`; `FasrcConfig` science
fields); SPA dead code (`PredictiveAxesCard`, dead memos/state in Ensemble, `Toolbar*`, `qs`,
rail legacy classes, removed step ids in `taskColumnsFor`, BerHu colours, dead page CSS).
Wire instead of delete: `/api/fasrc/queue/remove`, `/api/fasrc/refresh-accounting`,
`/api/tracking/backup`, `/api/fasrc/config`, `/api/jwst-euclid/nexus/download`,
`/api/jwst-euclid/field/<id>/download/<kind>`, `/inference-files` (as real-tile downloads),
`/viewer/results/<id>` + `panel.png`, `/poster/result/*`.
KEEP (verified live): `helpers/tng_prior.py` (used by population comparison), the FASRC-mirror
`stars.csv` path, `reconstruct_cutout_at` (used by `eval/catalog_runner.py`).
Tests that pin removed code or SPA source text are rewritten as behaviour tests
(`test_noise_tab`, `test_galaxy_distributions`, `test_population_comparison`,
`test_star_distribution`, `test_galaxy_corner`, `test_web` viewer-source and route tests,
`test_inspect_fits`, `test_fasrc_fetcher`, `test_fasrc_integration`, `test_psf_preview`,
`test_eval_catalog`, `test_archive_fields*`).

## 11. Bug fixes included

`euclid_query` 200-star default overwrite; Tracking sandbox `source` dict crash; `/connection-error`
placeholder; offline-gated local endpoints; `useResource` deps ignored while cached; the two
`tsc` errors; JWST carousel remount; RA/Dec hemisphere letters; silent 500-row caps; starless
defaults (`routes/ensemble.py`, `viewer_data`); mixed-regime evaluation/Generate SR;
`kneeTag(null)` → "100e"; loss-series overwrite in training curves; Config lost-update;
`git add -A`; unconfirmed `rsync --delete-after`; `/api/status` forced rsync on every call;
TNG radii status blocking page loads; EDF-F/EDF-S label swap; publication heat bar labelling
JWST as e⁻; viewer wheel scroll trap; five idle `useTrackedJob` pollers; CI missing the frontend.

## 12. Testing and verification

- Frontend: `tsc --noEmit`, eslint, vitest (colour parity goldens, WCS math, selection geometry,
  format/ticks, stores, DataTable, route manifest), build.
- Backend: pytest (full suite), new tests for real-tile store (synthetic fixtures), model
  catalogue, experiments metrics (hole %, enclosed-flux R on synthetic arrays with known
  answers), sky layers, SPA route matcher, FASRC gate decorator, jobs cancel/eviction, Host
  allowlist, step task_params.
- CI: add a Node job (npm ci, typecheck, lint, test, build, fail if `static/dist` differs from
  the committed build).
- Browser: every workspace/tab in light and dark, no ErrorBoundary, no console errors, key
  interactions (palette, inspector, display panel, viewer pan/zoom/readout, sky layers and
  click-through, job tray) verified in the in-app browser with screenshots.

## 13. Phasing

1. Backend platform (routing, gate, security, jobs, version, steps schema, viewer backend,
   legacy deletion + test rewrites) ∥ frontend foundation (shell, stores, data layer, UI kit,
   charts, theme) — in parallel, isolated.
2. Backend sky atlas + real tile store + model catalogue + experiments ∥ viewer engine v2.
3. Workspaces (parallel, one owner per workspace folder).
4. Integration, full verification, adversarial review, fixes; commit + push per phase.
