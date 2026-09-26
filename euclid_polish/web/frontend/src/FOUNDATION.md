# EuclidPolish SPA foundation

This is the shared frontend API that every workspace builds on: contracts C1, C7 and C8 of
`docs/superpowers/plans/2026-09-25-webui-rework.md`. The spec (behaviour) is
`docs/superpowers/specs/2026-09-25-webui-rework-design.md`.

> **Status:** the foundation is complete. WP-F1 (tooling, tokens, data layer, stores,
> `format.ts`, `ticks.ts`), WP-F2 (UI kit v2, DataTable, Plot v2: §9) and WP-F3 (data router,
> shell, command palette, shortcuts, inspector registry, job tray, Display panel and the nine
> workspace shells with legacy adapters: §10–§11) are in. Phase 3 replaces the legacy adapters
> workspace by workspace (§11.4).

Rules that apply everywhere:

- Every colour comes from a CSS token (`theme/tokens.css`). Never hard-code a hex value in a
  component.
- GET through `useResource` (or `apiGet`). POST through `apiPost` or `useJob`. Never call raw
  `fetch` from a page.
- Put shareable view state in the URL (`useUrlState`). Use `localStorage` only through
  `state/storage.ts` (every access is wrapped in try/catch).
- A shared primitive that is missing gets reported to the orchestrator. Don't fork a local copy.

---

## 1. Layout of `src/`

| Path | What it holds |
|---|---|
| `api/client.ts` | `apiGet`, `apiPost`, `ApiError`, `isFasrcOffline`, `toFormData` |
| `api/query.ts` | `queryClient`, `useResource`, `invalidate`, `prefetchResource`, `get/setResourceData` |
| `api/jobs.ts` | `useJob`, `useJobsFeed`, `useTrackedJob`, `cancelJob`, `cancelSlurmJob`, `useJobsStore` |
| `app/manifest.ts` | Typed route manifest (C1) and page matcher (mirrors `spa_routes.py`) |
| `app/devProxy.ts` | Decides which server answers a dev-server request (Vite / SPA / Flask) |
| `app/routes.ts` | `buildRoutes()`: the data router's route table from the manifest; `workspaceComponents` (lazy workspace imports) |
| `app/App.tsx` | `createBrowserRouter` + `<RouterProvider>` (main.tsx renders it inside `QueryClientProvider`) |
| `app/nav.ts` | Workspace metadata (icon, description, go-key, tab labels) and path helpers: `pagePath`, `landingPath`, `describePath`, `pageTitle`, `allPages` |
| `app/workspace.tsx` | The workspace contract: `defineTabs`, `<Workspace>`, `<WorkspaceTabs>`, `<PendingTab>`, `TabSkeleton` |
| `app/Shell.tsx` | Root layout: rail, top bar, stage, inspector, palette, ? sheet, Display panel, global shortcuts, `UiProvider` |
| `app/Rail.tsx`, `TopBar.tsx`, `JobTray.tsx`, `InspectorPanel.tsx`, `CommandPalette.tsx`, `ShortcutSheet.tsx`, `DisplayPanel.tsx`, `GlobalShortcuts.tsx` | The shell's parts |
| `app/inspector.ts` | Inspector registry (`registerInspector`, `openInspector`, `closeInspector`) and the `?inspect=` sync |
| `app/inspectors/` | Built-in inspector kinds (`job`) |
| `app/palette.ts` | `usePageActions` (palette actions) and `paletteSuggestions` |
| `app/displaySections.ts` | `registerDisplaySection` (the Display panel's extension point) |
| `app/shellStore.ts` | `useShellUi`: open/closed palette, ? sheet, Display panel, job tray, rail drawer |
| `app/status.ts` | `useVersion` (C3), `useFasrcStatus` (C4) |
| `app/ErrorBoundary.tsx`, `NotFound.tsx`, `useStageScroll.ts` | Per-tab error boundary, 404 page, stage scroll management |
| `state/display.ts` | Display (colour) settings store (C7) |
| `state/prefs.ts` | Theme, accent, density, rail collapsed, inspector width |
| `state/inspector.ts` | Inspector target, history and pins |
| `state/selection.ts` | Cross-view selection sets |
| `state/storage.ts` | try/catch-guarded `localStorage` (`safeJSONStorage`, `readStorage`, `writeStorage`) |
| `hooks/` | `useUrlState`, `useShortcut` / `bindShortcut`, `useMediaQuery`, `usePolling`, plus re-exports of `useResource` / `invalidate` |
| `format.ts` | Number, SI, bytes, duration, magnitude, RA/Dec and date formatting; `parseSkyCoord` |
| `ticks.ts` | Linear, log, decade and magnitude ticks; `extent`, `paddedDomain`, `unionDomain` |
| `colors.ts` | Canvas colour readers: `C.*`, `categorical`, `LOSS_COLOR`, `bandColor`, `viridis` |
| `theme/` | `tokens.css` (the token contract), `base.css` (element styles, `.muted`, `.eyebrow`, `.sr-only`), `index.css` (entry point) |
| `ui/` | UI kit v2 on Radix (§9): controls, overlays, `confirm`, `toast`, display primitives, `DataTable`, `LogView`, `JsonTree`, `JobProgress`, `Icon`, download/clipboard helpers, `UiProvider` |
| `charts/` | `Plot` v2, `Legend`, `useLegend` (§9.4); the pure maths is in `plotModel.ts` |
| `viewer/` | Viewer engine v2 *(WP-V)* |
| `workspaces/<id>/` | One folder per workspace: `index.tsx` + lazy `tabs/<Tab>.tsx` (§11) |
| `pages/` | The pre-rework pages, rendered by the workspace tabs as legacy adapters until phase 3 |

These compatibility modules keep the pre-rework pages compiling. New code should not import
from them:

| Old import | Now |
|---|---|
| `../api` (`getJSON`, `postForm`) | `api/client.ts`. `getJSON` still resolves `null` on any failure; `postForm` = `apiPost` |
| `../hooks` (`useResource`, `usePolling`, `invalidateCache`) | `api/query.ts`, `hooks/index.ts` |
| `../jobs` (`useJob`, `useTrackedJob`, `JobProgressView`) | `api/jobs.ts`. The panel is `JobProgress` in `ui/`; `JobProgressView` is its compat name |
| `../theme` (`useThemeValue`, `readTheme`) | `state/prefs.ts` (`usePrefs`, `useResolvedTheme`) |

---

## 2. Tooling

Run all of these from `euclid_polish/web/frontend`:

| Command | Does |
|---|---|
| `npm run typecheck` | `tsc --noEmit` over `src` (`tsconfig.json`), `test/` + `vitest.config.ts` (`tsconfig.test.json`), and `vite.config.ts` (`tsconfig.node.json`) |
| `npm run lint` | ESLint flat config (`eslint.config.js`): typescript-eslint + react-hooks. It must report 0 errors. Legacy `src/pages/**` only warns on dead locals |
| `npm test` | `vitest run` (happy-dom + Testing Library; setup in `test/setup.ts`) |
| `npm run build` | `tsc --noEmit && vite build` into the **committed** `../static/dist`. **Only the orchestrator runs this.** To check a build without touching dist, run `npx vite build --outDir <tmp> --emptyOutDir` |
| `npm run dev` | Vite on :5173 |

How the dev server works (`vite.config.ts`):

- Vite serves page paths from the manifest, and legacy page URLs, as `index.html`. The SPA then
  redirects legacy URLs client-side.
- Everything else goes to `FLASK_ORIGIN || http://localhost:9777`. The Host header is kept
  (`changeOrigin: false`), so Flask's same-origin POST guard accepts dev mutations.
- `base` is `/` in dev and `/static/dist/` in build.

Chunking:

- `manualChunks` splits the bundle into `react` (react, react-dom, router), `vendor` (all other
  `node_modules`) and `aladin` (lazy; only the Sky atlas imports it).
- `chunkSizeWarningLimit` is 2600 kB, sized for the aladin chunk.

How to write tests:

- Put a test next to the module it covers (`src/**/*.test.ts[x]`). `test/` holds the suites
  ported from the old node:test runner.
- Mock `fetch` with `vi.stubGlobal("fetch", …)`. Call `queryClient.clear()` between tests.
  `focusManager.setFocused(false)` simulates a hidden tab. To simulate the browser going
  offline, call `queryClient.mount()` (as `QueryClientProvider` does), dispatch a window
  `offline` event, and restore with `online` + `queryClient.unmount()` afterwards.
- To test a persisted store's first load (hydration), seed `localStorage`, call
  `vi.resetModules()`, then `await import("./store")` for a fresh module instance.
- happy-dom `localStorage` is cleared after every test (`test/setup.ts`).
- pytest never reads `frontend/src`. The frontend's behaviour is tested here.

Dependencies are pinned to exact versions in `package.json`. Only WP-F and WP-V add
dependencies.

---

## 3. Route manifest (C1): `app/manifest.ts`

```ts
import { MANIFEST, matchPage, isPagePath, redirectTarget, workspace, workspacePaths, pagePaths } from "./app/manifest";
matchPage("/ensemble/starless/knee") // {workspace:"ensemble", params:{mode:"starless"}, tab:"knee", base:"/ensemble/starless"}
isPagePath("/ensemble/status.json")  // false (data URL sharing a page prefix)
redirectTarget("/config", "?x=1")     // "/settings/config?x=1"; "/app/<rest>" → "/<rest>"
redirectTarget("/app//evil.example")  // "/evil.example" (never protocol-relative)
```

The manifest file is `euclid_polish/web/spa_routes.json`, and nobody edits it without the
orchestrator. `manifest.test.ts` mirrors `tests/test_spa_routes.py`, so Flask and the SPA agree on
what counts as a page. The router (`app/routes.ts`, §10.1), the rail, the breadcrumbs, the palette
and `document.title` are all built from `MANIFEST` + `app/nav.ts`.

The `/app/<rest>` redirect collapses every leading `/`, `\`, space or control character of
`<rest>` into one `/`, as `spa_routes._same_host_path` does. `//host` and `/\host` are
protocol-relative to a browser, so the target can never leave the host. Use `redirectTarget` for
the client-side `<Navigate replace>`, and do not rebuild the target by hand.

---

## 4. Data layer (C8)

### 4.1 `api/client.ts`

```ts
const data = await apiGet<Status>("/ensemble/status.json", { signal });
await apiPost("/api/config/save", { knee: 100, skip: null });            // form-encoded; null/undefined dropped
await apiPost("/api/experiments", { tiles, models }, { json: true });   // JSON body
try { … } catch (e) {
  if (e instanceof ApiError) e.status; e.message /* server {error} text */; e.code; e.body;
  if (isFasrcOffline(e)) { /* 503 {code:"fasrc_offline"} (C4) */ }
}
```

- A network failure is an `ApiError` with `status: 0`. An abort is rethrown untouched.
- A 200 that isn't JSON is an `ApiError` (`apiGet`).
- `apiPost` returns a 200 `{error}` body as-is. `useJob` reports it.

### 4.2 `api/query.ts`: `useResource`

```ts
const { data, loading, error, reload, fetching, staleError, updatedAt } =
  useResource<Evals>(url /* null or "" = idle */, [dep1, dep2], { ttl: 60_000, poll: 5_000 });
```

- A falsy URL (`null`, `undefined` or `""`) is idle: no request, `{data: null, loading: false,
  error: null}`. Idle hooks share no cache entry with any URL.
- Offline-first: the client uses `networkMode: "always"` (queries and mutations). Every request
  goes to the local Flask server, so a browser `offline` event (a Wi-Fi drop, which is exactly
  when FASRC disconnects) never pauses fetches, polls or retries.

- The shared TanStack Query cache is keyed by URL. Concurrent requests are deduped and every
  subscriber sees the same result.
- A cached copy is served until `ttl` runs out (default 5 min). After that it revalidates in the
  background while still showing the stale copy.
- A change in `deps` forces a refetch even while the URL is cached and fresh. The old hook
  ignored `deps` here.
- `deps` are compared **by value**, not by identity, so an inline `[{mode}]`,
  `[data?.x ?? []]` or `[asArray(y)]` rebuilt on every render cannot start a refetch loop:
  - primitives compare with `Object.is` (`NaN` equals itself, and `null` → `undefined` counts
    as a change);
  - arrays, plain objects and `Map` values compare element by element, recursively;
  - `Map` keys and `Set` members compare with `has`, and `Date`s compare by time;
  - functions are ignored, because a callback is never a refetch trigger;
  - class instances and typed arrays compare by identity. Keep those stable (`useMemo`), or pass
    the primitive that actually changes (an id, a counter).

  Prefer primitives: a counter bumped after a submit, a mode string or an id. Every refetch
  triggered by `deps` cancels the request in flight and starts a new one.
- `poll` pauses while the tab is hidden and refreshes on return.
- It retries network failures and 5xx up to 2 times. It never retries 4xx, a 501 or a 503
  (the FASRC gate). Each of those is fetched exactly once (tested).
- When the last subscriber of a URL unmounts, its in-flight request is aborted (the fetch
  `signal` fires). Another subscriber still mounted keeps it alive.
- `error` is an `ApiError | null`, and it is set only when there is no data. The compat
  truthiness still holds. A failed refresh while data is shown lands in `staleError` instead.
- After a mutation, call `invalidate("/ensemble/")` to refetch every mounted resource under that
  URL prefix. `invalidate()` with no argument refetches everything.
- `main.tsx` provides `QueryClientProvider`, so components may also call `useQuery` directly with
  `queryClient`.

### 4.3 `api/jobs.ts`: local and SLURM jobs (C2)

```ts
const ev = useJob("ensemble:evaluate");           // key optional: re-attach after navigation
ev.run("/api/ensemble/evaluate", { force: 1 }, { onDone: (j) => j.status === "done" && invalidate("/ensemble/") });
<JobProgress job={ev.job} error={ev.error} />       // from "../ui" (Cancel button for cancellable jobs)
ev.busy; ev.reset(); await ev.cancel();

const feed = useJobsFeed();   // {jobs, running, slurm, slurmQueue, fasrcOffline, runningCount, started, refresh}
await cancelJob(id);          // {ok} | {ok:false, error}; the job becomes "cancelled" at its next tick
await cancelSlurmJob(jobid);  // POST /api/fasrc/cancel; {ok} | {ok:false, error} (a body refusal or the C4 503)
const fit = useTrackedJob("combiner: fit starfull");   // attach by label (feed-based, no idle poller)
```

- `run` POSTs a form. A `{job_id}` reply is polled at `/api/jobs/<id>` with a 0.5 → 2 s backoff
  until it ends (skipped while the tab is hidden), and `onDone` fires once. A non-job reply
  calls `onDone` at once.
- Every started job is registered in `useJobsStore` (`started`), so it stays in the job tray
  after navigating away.
- The feed runs one shared poll per source:
  - local `/api/jobs?summary=1`: every 2 s while anything runs, 15 s when idle;
  - SLURM `/api/fasrc/current-submission`: every 10 s while a job is live, 30 s when idle,
    60 s while FASRC is offline.

  Both pause while the tab is hidden. The cadences are in `JOBS_FEED_TIMING`.
- FASRC offline (C4) shows up as `fasrcOffline: true` with an empty `slurm` list. It is not an
  error.
- SLURM rows come from `live` (C5, WP-B1b) when present, else from the single `current.job`.
- The store merges snapshots, with two guarantees: a summary never erases a known log, and a
  late "running" snapshot never regresses a finished job.

### 4.4 Endpoint modules

Per-domain typed endpoint modules (`api/<domain>.ts`) arrive with their workspaces in phase 3.

---

## 5. Stores (`state/`)

Every store is zustand v5. The persisted ones use `safeJSONStorage`, so a storage failure never
breaks the UI; they sanitise on load.

### 5.1 Display settings (C7): `state/display.ts`

```ts
const { color, stretch, groups, colormap, set, setGroup, reset } = useDisplay();
useDisplay.getState().set({ color: "lupton", invert: true });
useDisplay.getState().setGroup("euclid", { knee: 250 });            // knee in e⁻
const effective = mergeDisplay(useDisplay.getState(), viewerOverride); // per-viewer override (linked:false)
transferFor(effective, "jwst");                                    // {knee, gain, black}, falls back to "default"
```

- The types are exactly those of C7: `ColorMode`, `Stretch`, `Colormap`, `TransferGroup`,
  `DisplaySettings`.
- Defaults:
  - `color`: `VIS`
  - `rgb`: `[H_E, J_E, VIS]`
  - `stretch`: **`asinh-abs`** (the locked default)
  - `groups`: `default`, `euclid` and `jwst`, each `{knee: 100, gain: 1, black: 0}`
  - `colormap`: `gray`
  - `residualColormap`: `rdbu`
  - `invert`: `false`
  - `nanColor`: `#ff00ff`
  - `linked`: `true`
  - `wheel`: `zoom-when-focused`
- It persists to `"ep-display"`. The option lists are exported as `COLOR_MODES`, `STRETCHES`,
  `COLORMAPS` and `WHEEL_MODES`, and the slider ranges as `KNEE_RANGE` and `GAIN_RANGE`.
- `DEFAULT_DISPLAY` is deeply frozen (the groups record, each group and the `rgb` tuple), so a
  stray write throws instead of changing every later `reset()`. `reset()` and
  `sanitizeDisplay()` return fresh, editable copies.
- On load the saved value is sanitised field by field; with nothing saved the store keeps its
  initial (default) state.

### 5.2 Prefs: `state/prefs.ts`

```ts
const theme = useResolvedTheme();                 // "light" | "dark" (follows "system" + OS changes)
usePrefs.getState().setTheme("system");            // "light" | "dark" | "system"
usePrefs.getState().set({ accent: "teal", density: "compact" });
usePrefs.getState().toggleTheme(); usePrefs.getState().toggleRail();
```

- It persists to `"ep-prefs"`. When nothing is saved yet, the store starts from the pre-rework
  `"ep-theme"` key (`light` | `dark`), the same value the `index.html` pre-paint script applies,
  so there is no flash (tested at store level, through a fresh module load). Saved prefs win
  over the legacy key.
- `bindPrefsToDocument()` runs once in `main.tsx`. It writes these attributes to `<html>`:
  - `data-theme`: the resolved theme;
  - `data-theme-pref`;
  - `data-accent`: one of `blue`, `violet`, `teal`, `amber`, `rose`;
  - `data-density`: `comfortable` or `compact`.
- The pre-paint script in `index.html` applies the same attributes before first paint.
  `state/prePaint.test.ts` runs that script against seeded storage and a stubbed
  `prefers-color-scheme`, then checks it writes exactly what `bindPrefsToDocument()` writes
  after a fresh start from the same storage. The cases cover every theme, accent and density,
  invalid and corrupt values, the legacy key and throwing storage or `matchMedia`. Adding a
  preset to `ACCENTS` without updating the script's regex fails that test.
- The fields are `inspectorWidth` (clamped to 260–1200 px, default 380) and `railCollapsed`.

### 5.3 Inspector state: `state/inspector.ts`

```ts
useInspector.getState().show({ kind: "member", id: "member_196" });  // opens; pushes history
useInspector.getState().goBack(); goForward(); hide(); clear(); togglePin();
parseInspectParam("tile:nexus/123")  // {kind:"tile", id:"nexus/123"} (splits at the first colon)
formatInspectParam({ kind, id })     // "kind:id" for ?inspect=
```

This module is state only: the current target, the open flag, back/forward history (capped at 50)
and pins (persisted to `"ep-inspector"`). The kind → component registry, the `?inspect=` URL sync
and the panel itself are in `app/inspector.ts` / `app/InspectorPanel.tsx` (§10.4).

The store is the source of truth, and some callers write it directly (`DataTable`'s `inspect`
prop calls `useInspector.getState().show(target)`). The shell's `?inspect=` sync runs both ways,
so a row opened that way is shareable and survives a reload.

### 5.4 Selection: `state/selection.ts`

```ts
const ids = useSelected("member");
useSelection.getState().toggle("member", "member_196");
useSelection.getState().select("tile", ["nexus/1", "nexus/2"]);  // add / remove / clear(scope?)
```

- Each scope holds an ordered, de-duplicated id list (`member`, `tile`, `star`, …). The ids are
  the same ones the inspector uses.
- The lists are `readonly string[]` and are frozen at runtime. Copy a list (`[...ids]`) before
  editing it, and write changes back through the actions.
- It is session-only: nothing is persisted.

---

## 6. URL state: `hooks/useUrlState.ts`

```ts
const [tab, setTab] = useUrlState("tab", "overview");
const [i, setI] = useUrlState("v.main.i", 0);                    // number codec
const [tiers, setTiers] = useUrlState<string[]>("tiers", ["lr", "sr"]); // comma list
const [box, setBox] = useUrlState("box", { x: 0, y: 0 });         // JSON codec
useUrlState("k", def, { parse, serialize, replace: false });     // custom codec / push history
```

- The codec is inferred from the default value.
- A value equal to the default is removed from the URL.
- Unparseable values read as the default.
- Writing the value the hook already reads does nothing, and in push mode it adds no duplicate
  history entry. "The value it reads" is compared by serialization, so this also covers an
  explicit default such as `?a=0`, an unparseable param that reads as the default, and a
  non-canonical spelling such as `t=lr,sr`. A real change still writes.
- Several setters called in one tick are coalesced. Each setter builds on the search the
  previous one wrote, and the tick records **at most one** new history entry. One Back undoes
  the whole tick, **whatever order the setters ran in**:
  - With only replace-mode setters (the default), the tick edits the current entry in place
    and keeps its history `state`.
  - With a push-mode (`replace: false`) setter, all of the tick's changes land in one pushed
    entry. The first push-mode setter restores the original entry, undoing earlier in-place
    writes of the tick, and then pushes. Later setters in the tick replace the pushed entry.

  This holds under `BrowserRouter`/`MemoryRouter` and under a data router
  (`createBrowserRouter`) without loaders. Both orders are tested under both routers.
- Other params keep their exact spelling. Only the key's own segment is rewritten, so
  `?inspect=member:m_1` is never re-encoded to `member%3Am_1`. The key keeps its position,
  duplicates of it collapse into one, and the hash is kept.

Keyboard shortcuts: `hooks/useShortcut.ts` (§10.3).

---

## 7. Formatting and ticks

### 7.1 `format.ts`

Every formatter returns `"—"` for null, NaN or ±∞. The output is locale-independent.

| Function | Example |
|---|---|
| `formatNumber` | `formatNumber(3.14159, {digits: 2})` → `"3.14"`, `formatNumber(2.5e-5)` → `"2.5e-5"`, `{signed, unit}` |
| `formatCount` | `43401` → `"43,401"` |
| `formatPercent` | `0.1234` → `"12.3%"` |
| `formatSI` | `(2.5e6, {unit: "e⁻"})` → `"2.5 Me⁻"` |
| `formatBytes` | `1536` → `"1.5 KB"` |
| `formatDuration` | `185` → `"3m 05s"` |
| `formatMagnitude` | `(19.234, {sigma: 0.05})` → `"19.23 ± 0.05"` |
| `superscript`, `formatPow10` | `-3` → `"10⁻³"` |
| `formatRA` | hours: `"17h49m41.50s"`, or `{style: "colon"}` |
| `formatDec` | always signed: `"+64°53′14.3″"` |
| `formatDeg` | decimal degrees |
| `formatRaDec` | `mode`: `sexagesimal`, `degrees` or `both` |
| `parseSkyCoord(text)` | degrees or sexagesimal "RA Dec" → `{ra, dec}` degrees, or null |
| `formatDate`, `formatDateTime`, `formatRelative`, `parseTimestamp` | accept epoch s, epoch ms, ISO or `Date`; `{utc}` |

### 7.2 `ticks.ts`

Every generator returns `Tick[] = {v, label}`, the shape `Plot` takes.

| Function | Gives |
|---|---|
| `linearTicks([lo, hi], {count})`, `linearTickValues`, `niceStep`, `formatTick` | linear ticks on a 1-2-5 step |
| `logTicks([lo, hi], {space: "value"\|"log10", maxTicks})` | 1-2-5 mantissas over a few decades, decades over wide ranges |
| `decadeTicks` | integer decades labelled 10ⁿ |
| `magnitudeTicks([lo, hi], {invert})` | whole, half or tenth magnitudes |
| `extent`, `paddedDomain(values, {pad, minSpan, includeZero})`, `unionDomain` | domains |

`space: "log10"` is for axes whose data were already transformed with log10: the positions are
exponents and the labels are physical values.

Pathological domains never hang and never give NaN or ±∞ positions. This covers spans that
overflow to ∞, subnormal spans, steps that overflow, offsets past 2⁵³ steps, and a
non-positive or fractional `decadeTicks` `step`. Every generator loops over a bounded counter
capped at `MAX_TICKS` (1000). When the arithmetic breaks down, it falls back to the domain's
end points. `magnitudeTicks` switches to a nice step when the span is beyond 10-mag steps.
`paddedDomain` stays finite.

---

## 8. Theme tokens: `theme/tokens.css`

The contract is enforced by `theme/tokens.test.ts`. It covers:

- surfaces (`--bg-0/1`, `--surface-1/2/3`, `--border*`, `--overlay`, `--tooltip-*`);
- ink (`--text`, `--text-dim`, `--text-faint`);
- accent;
- status (`--good/--warn/--bad/--info` + `-soft`);
- chart series (`--series-*`), categorical (`--cat-0…7`), losses (`--loss-l1/l2/l3/mse`) and
  **bands** (`--band-vis/-y/-j/-h`);
- type (`--fs-2xs…3xl`, `--lh-*`, `--fw-*`, `--font-sans/mono`);
- space (`--s1…s8`) and radius (`--r-*`);
- shadows (`--shadow-1…3`) and focus (`--ring`);
- z-index (`--z-*`) and motion (`--dur-*`, `--ease-*`; zeroed under reduced motion);
- layout (`--rail-w*`, `--topbar-h`, `--inspector-w`, `--ctl-h`, `--row-h`, `--card-pad`).

The theme selectors:

- Light is `:root` (the default). Dark is `:root[data-theme="dark"]`.
- Accent presets use `[data-accent]`. Compact density uses `[data-density="compact"]`.

The test also enforces WCAG AA (≥ 4.5:1) in both themes:

- `--text`, `--text-dim` and `--text-faint` on `--surface-1/-2/-3` and `--bg-0/-1`;
- `--good`, `--warn`, `--bad` and `--info`:
  - as text on every opaque surface (`--surface-1/-2/-3`, `--bg-0/-1`);
  - on their `-soft` tint over each of those surfaces. This covers badges on cards, on the
    rail (`--bg-1`) and on inputs or active rows (`--surface-3`). The tightest pairing is the
    tint over `--surface-3`, at 4.65–4.89;
- `--accent`, for the default blue and for every `[data-accent]` preset:
  - as text on every opaque surface (`--surface-1/-2/-3`, `--bg-0/-1`);
  - with `--accent-press`, on the `--accent-soft` tint over `--surface-1/-2` and `--bg-0/-1`
    (the selected chip, tab or rail item);
- `--on-accent` on a filled `--accent`.

Keyboard focus shows a `--ring` box-shadow. Box-shadows are not painted in forced-colors mode
(Windows High Contrast), so `base.css` also sets a transparent outline, which forced colours
repaint. A `@media (forced-colors: active)` rule then forces a `CanvasText` outline on every
`:focus-visible`, including components that set `outline: none`.

A second test fails when any `var(--x)` used in `src` has no definition (variables that Radix
sets at runtime, `--radix-*`, are exempt). Names that older CSS
used without defining (`--line`, `--muted`, `--mono`, `--panel`, `--bg`, `--surface`, …) are
aliases now. The `.muted` class is defined in `base.css`.

To read a token in canvas code, use the `colors.ts` readers (`C.mean`, `categorical(i)`,
`LOSS_COLOR.l1`, `bandColor("VIS")`). To redraw on a theme flip, add `useResolvedTheme()` to the
figure's dependencies. A theme or accent flip also re-renders the active workspace tab and the
inspector content (`useTokenRerender`, §11.1), so a token read during render (`color: C.muted`
in JSX) picks up the new theme; memoised values and effects still need the dependency.

---

## 9. UI kit v2, DataTable and charts

Import everything from `"../ui"` (the barrel `ui/index.tsx`; it also loads the kit CSS) and
charts from `"../charts/Plot"`. Never import a part module (`ui/controls`, …) directly.

The C8 names are all exported: `Button, IconButton, Tooltip, Popover, Dialog, confirm, Menu,
ContextMenu, Tabs, Segmented, Switch, Checkbox, Slider, RangeSlider, NumberField, Input, Select,
Field, Card, CardHead, CardBody, Section, Badge, Chip, Stat, Kpi, DefList, Callout, EmptyState,
Skeleton, ProgressBar, LogView, JsonTree, CopyButton, Kbd, DataTable, toast`. So are the v1
names the old pages use: `Page, PageHead, Empty, Spinner, Table`/`Column`, `LogTail, Gallery,
PngFigure, ConnBadge, Textarea` and `JobProgressView` (= `JobProgress`). The dead v1
`Toolbar`, `ToolbarLabel` and `ToolbarSep` are gone. Extras: `MultiSelect`, `Toaster`,
`UiProvider`, `ConfirmHost`, `TooltipProvider`, `DialogClose`, `PopoverClose`, `Icon`,
`copyText`, `downloadText`, `downloadBlob`, `safeFileName`, `cx`, `useFieldAria`.

Conventions for every component:

- Change handlers take the value, not the DOM event: `onChange(value)`.
- Buttons are `type="button"` unless you pass `type="submit"`.
- Every control forwards its `ref` and accepts `aria-label`, so it can be a Tooltip or Popover
  trigger.
- Focus shows the shared `--ring`. Colours come from tokens only.
- `tone` is one of `neutral | good | warn | bad | info | accent`.

### 9.1 Hosts: `UiProvider`

`main.tsx` wraps the app in `<UiProvider>`. It mounts three things: the shared tooltip
provider, the toast stack (`<Toaster/>`, which follows the resolved theme) and the `confirm()`
dialog host. Components also work without it:

- A `Tooltip` outside the provider brings its own provider.
- The first `confirm()` with no host mounts one into `document.body`.

Tests can therefore render a component alone.

**Placement.** `UiProvider` is mounted exactly once, by the shell (`app/Shell.tsx`), INSIDE the
data router's root layout route and inside `<QueryClientProvider>` (main.tsx), so toast, confirm
and tooltip content can use a router `<Link>`, `useNavigate` or a query hook (e.g. the job toasts'
"Log" action opens the inspector). Never mount a second one, and never wrap `<RouterProvider>` in
it. The automatic `confirm()` host (no `UiProvider` mounted:
tests, isolated widgets) is a separate React root with no app context, so keep `confirm()` titles
and messages context-free (text and plain elements), and prefer callbacks to context in toast
content.

### 9.2 Components

**Actions**

```tsx
<Button variant="primary" icon="download" loading={job.busy} onClick={run}>Evaluate</Button>
<Button href="/api/x.csv" download size="sm">CSV</Button>             // an <a> with button styles (external / download)
<Button asChild variant="ghost"><Link to="/sky">Open sky</Link></Button> // router link
<IconButton icon="reset" label="Reset zoom" onClick={reset} />         // label = aria-label + tooltip
<IconButton icon="columns" label="Columns" pressed={open} />           // aria-pressed toggle
```

- `Button` variants are `default`, `primary`, `ghost`, `subtle` and `danger`; sizes are `sm`,
  `md` and `lg`.
- `loading` shows a spinner, sets `aria-busy` and ignores clicks, in all three modes (a
  `<button>`, `href` → `<a>`, `asChild`). With `asChild` the guard also blocks a `<Link>`'s
  navigation.
- The `ref` points at the rendered element (the `<a>` or the child in the link modes). Other
  props (`data-*`, `aria-*`, handlers) are forwarded in every mode; `href` mode drops the
  button-only attributes (`type`, `form*`, `name`, `value`).
- `icon` and `iconRight` take an `IconName` or any node.
- Icons come from `ui/icons.tsx` (inline SVG, `currentColor`). Add a path there when you need a
  new glyph.

**Overlays** (Radix; portalled; Escape closes; focus is trapped and then restored)

```tsx
<Tooltip content="Knee-integrated PSNR"><span tabIndex={0}>ΣPSNR</span></Tooltip>
<Popover trigger={<Button size="sm">Filters</Button>} label="Filters">…</Popover>
<Dialog open={open} onOpenChange={setOpen} title="Fit combiner" description="…" size="lg"
  footer={<><Button variant="ghost" onClick={close}>Cancel</Button><Button variant="primary">Fit</Button></>}>
  …body…
</Dialog>
<Menu label="Member actions" trigger={<IconButton icon="more" label="Actions" />} items={[
  { label: "Continue", shortcut: "C", onSelect: cont },
  { type: "checkbox", label: "Show archived", checked, onCheckedChange: setChecked },
  { type: "separator" },
  { type: "sub", label: "Fork", items: [...] },
  { label: "Archive", tone: "danger", onSelect: archive },
]} />
<ContextMenu items={[{ label: "Copy coordinates", onSelect: copy }]}><div>…target…</div></ContextMenu>
```

A `Tooltip` or `Popover` trigger must be ONE element that forwards refs. Kit controls and DOM
elements do. `Dialog` takes `trigger` for uncontrolled use.

**`confirm()` replaces `window.confirm`**

```ts
if (!(await confirm({ title: "Archive 3 members?", message: "They move to tracking.",
                      tone: "danger", confirmLabel: "Archive" }))) return;
await confirm("Push to origin?");                                      // string shorthand
await confirm({ title: "Sync with --delete-after?", requireText: "sync" }); // type-to-confirm
```

- It resolves `true` on confirm and `false` on Cancel, Escape or a click outside.
- Concurrent requests queue and are shown one at a time.
- `tone: "danger"` focuses Cancel first; otherwise Confirm has the initial focus.
- The dialog is a `role="alertdialog"`; `message` is its accessible description
  (`aria-describedby`), so the question and its consequence are announced together.
- `resetConfirm()` cancels everything pending (use it in test teardown).

`Dialog` takes `role="alertdialog"` too, and renders `description` as the accessible
description (a `<div>`, so rich content nests validly).

**Toasts**

```ts
toast("Saved"); toast.success("Evaluation finished"); toast.error(err.message);
toast.warning("Stale"); toast.info("FYI"); toast.promise(p, { loading: "…", success: "done", error: "failed" });
```

**Form controls**

```tsx
<Field label="Knee" hint="Asinh knee in e⁻ (popover)" description="0.1–1e4" error={err}>
  <Input value={q} onChange={setQ} icon="search" clearable onEnter={submit} />
</Field>
<NumberField label="Stars" value={n} onChange={setN} min={1} unit="rows" hint="per tile" />  // onChange gets the raw string
<Select value={loss} onChange={setLoss} options={[{ value: "l1", label: "L1" }]} />  // native <select>
<Select searchable value={member} onChange={setMember} options={members} />       // filterable list
<MultiSelect value={bands} onChange={setBands} options={BANDS} />                  // searchable checklist
<Checkbox checked={on} onChange={setOn} indeterminate={some}>Include training</Checkbox>
<Switch checked={linked} onChange={setLinked}>Link viewers</Switch>
<Segmented value={mode} onChange={setMode} aria-label="Regime" options={[{ value: "starfull", label: "starfull" }, …]} />
<Slider value={knee} onChange={setKnee} onCommit={save} min={0.1} max={1e4} scale="log" showValue />
<RangeSlider value={[lo, hi]} onChange={setRange} min={14} max={26} step={0.1} />
```

- `Field` renders `<span class="ui-field">` holding a `<label class="ui-field__main">` (the
  label text and the control) and, below it, the `description` and the `error`. The label
  wraps the control, so any child is associated with it, raw elements too. The `hint` "?"
  opens a popover and does not toggle the control. `inline` puts the label beside the control.
- `description` (inline help) and `error` sit OUTSIDE the label, so they are never part of the
  control's accessible name. They are its description (`aria-describedby`: the caller's own
  ids first, then the description, then the error), and `error` also sets `aria-invalid`. Kit
  `Input`, `Textarea`, `Select`, `MultiSelect` and `NumberField` read this from context. A
  single raw `<input>`, `<select>` or `<textarea>` child is wired directly. A custom control
  spreads `useFieldAria(ownDescribedBy?, ownInvalid?)` (from the barrel) onto its focusable
  element. `Popover` and `Dialog` content resets the context, so a control inside an overlay
  opened from a Field is not described by that Field's error. The error is `role="alert"`.
  The markup inside the label never depends on `error`, so an error appearing while the user
  types does not remount (or blur) the control.
- `NumberField`'s `hint` is the `description` of its Field.
- `Select` shows an unknown current value as a disabled option (`placeholder` or the value),
  rather than silently showing the first option.
- A searchable `Select` or a `MultiSelect` is a button named "label + current value"
  (`aria-labelledby` over the label and the value text, e.g. "Member member_196"), so the
  value is announced. The label is its `aria-label`, or the surrounding `Field`'s label. A
  plain `aria-label` or `<label>` alone would replace the button's text.
- On a log `Slider`, `min` and `max` must be > 0 and differ. The track has 1000 steps, and
  values snap to 3 significant digits. A range log cannot map (e.g. from data) falls back to a
  linear track with one `console.warn` instead of crashing the page. The pure mapping is
  `effectiveScale` / `toSliderPos` / `fromSliderPos` (`ui/scale.ts`).
- `Tabs` comes in two modes:
  - with `onChange`, ARIA tabs (arrow keys; pass the active panel as `children` to link it —
    without `children` the tabs carry no `aria-controls`, so nothing points at a missing panel);
  - with `to` on the items, a `<nav>` of router links with `aria-current="page"`, for
    router-linked tabs.

  Both modes take `variant="pill" | "line"` and a `badge` per tab.

**Display**

```tsx
<Card><CardHead eyebrow="Ensemble" title="Members" sub="26 active" right={<Button>…</Button>} /><CardBody>…</CardBody></Card>
<Section title="Calibration" sub="z-pdf" collapsible defaultOpen={false}>…</Section>
<Badge tone="good" dot>current</Badge>   <Chip on={shown} dot={color} onClick={toggle}>mean</Chip>
<Stat k="members" v="26" hint="active STARFULL" />   <Kpi label="Production PSNR" value="44.12" unit="dB" delta="+0.21" deltaTone="good" onClick={open} />
<Kpi label="Members" value="26" to="/ensemble/starfull/members" />   <Kpi label="Report" value="PDF" href="/api/report.pdf" />
<DefList items={[["loss", "l1"], cond && ["knee", "100"]]} />            // falsy rows are skipped
<Callout tone="warn" title="Stale" action={<Button size="sm">Refresh</Button>}>…</Callout>
<EmptyState icon="table" title="No experiments yet" action={<Button>Run one</Button>}>Pick tiles…</EmptyState>
<Skeleton lines={3} />  <Spinner label="Loading members" />  <ProgressBar value={3} max={10} label="3/10" />
<Kbd keys="mod+k" />  <CopyButton value={() => `${ra} ${dec}`} label="Copy coordinates" />
<LogView text={job.log} title="evaluate" exportName="evaluate" />       // search, follow, wrap, copy, download
<JsonTree data={origin} expandDepth={1} />                              // collapsible, paged, copy value/path
<JobProgress job={ev.job} error={ev.error} />                           // + Cancel when job.cancellable
```

- `Kpi` is an action tile when `to`, `href` or `onClick` is set:
  - `to` is an SPA route, rendered as a router `<Link>`. There is no page reload, so the jobs
    store, the query cache and the inspector survive (C8).
  - `href` is for external or download URLs only, as a plain `<a>`. An app path there reloads
    the whole document.
  - `onClick` alone renders a `<button>`; with `to` or `href` it also runs on click.

  In an action tile the `hint` icon is hover-only (nothing focusable nests inside the link or
  button), and the hint text becomes the tile's accessible description. A static tile's hint
  is a focusable "About this figure" button.
- `Callout` with tone `bad` or `warn` has `role="alert"`; the other tones are a polite status.
- A collapsible `Section` does not render its body while closed (its children do no work), so
  the toggle has `aria-controls` only while open.
- `JsonTree` is nested plain lists of disclosure buttons (`aria-expanded`, `aria-controls` while
  open): Tab + Enter/Space. It deliberately has no ARIA `tree` roles, which would promise
  arrow-key navigation.
- `ProgressBar` is a real `role="progressbar"`. `value={null}` makes it indeterminate.
- `LogView` search is case-insensitive:
  - Enter / Shift+Enter step through the matches;
  - "only matching lines" filters the log.

  Follow keeps the view at the end while you are at the end. Scrolling up pauses it, and "Jump
  to end" resumes it.
- `LogTail` (compat) is the same follow logic with no toolbar.

### 9.3 `DataTable`

```tsx
const COLUMNS: DataColumn<Member>[] = [                     // define once (module scope or useMemo)
  { id: "name", header: "Member" },
  { id: "psnr", header: "PSNR", numeric: true, cell: (r) => r.psnr?.toFixed(2) ?? "—" },
  { id: "loss", header: "Loss", accessor: (r) => r.origin.loss, cell: (r) => <Badge>{r.origin.loss}</Badge> },
  { id: "seed", header: "Seed", hidden: true },               // off by default, in the Columns menu
  { id: "actions", header: "", sortable: false, filterable: false, csv: false, hideable: false, cell: … },
];
<DataTable rows={members} columns={COLUMNS} rowKey={(r) => r.name} aria-label="Members"
  selectable selected={sel} onSelectedChange={(keys, rows) => setSel(keys)}
  inspect={(r) => ({ kind: "member", id: r.name })}          // row click / Enter → inspector
  exportName="members" urlKey="m" height={520} toolbar={<Button size="sm">Fork</Button>} />
```

**Columns.** A column is identified by `id`. `accessor` defaults to `row[id]` and supplies the
value used for sort, filter, CSV and the default cell. `cell` renders the cell. The other
fields:

- `headerText`: the plain-text name, used in CSV, the filter prefix and ARIA;
- `sortFn`;
- `filterText`;
- `csv`: a function, or `false` to leave the column out;
- `align` or `numeric`;
- `width`: px or any CSS width. Default widths are estimated from the header and the first 200
  rows, so virtual scrolling never reflows the table;
- `hidden`, `hideable`, `className`.

**Sort.** Click a header for asc → desc → none. Shift-click adds a secondary key. The sort is
natural (`member_2` < `member_10`), case-insensitive and stable, and empty values always go
last. Use `sort`/`onSortChange` to control it, or `defaultSort` to start from one.

**Filter.** One search box. Tokens are ANDed:

| Token | Matches |
|---|---|
| `text` | substring in any column |
| `"two words"` | the quoted phrase |
| `-text`, `!text` | rows without the text |
| `-0.3`, `-1e3` | the negative number as text (a `-` before a number is not a negation; use `!0.3`) |
| `col:text` | substring in one column |
| `col=v` | equal (numeric when both sides parse as numbers) |
| `col>n`, `col>=n`, `col<n`, `col<=n` | numeric comparison |

`col` is a column `id`, or its header with spaces removed. An unknown prefix is searched as
plain text. The toolbar shows "n of m rows". Use `filter`/`onFilterChange` to control it, or
`defaultFilter` to start from one.

**Selection** (`selectable`):

- Checkbox click toggles. Shift-click selects or deselects the range from the anchor, in the
  current sorted and filtered order.
- The anchor is the row last clicked or activated (a plain click or Enter, including rows that
  open the inspector or call `onRowClick`), ⌘/Ctrl-clicked, or toggled (checkbox, Space).
  Moving the cursor with the arrow keys alone does not move it.
- A shift range (Shift-click on a row, Shift + navigation keys) is rebuilt from the anchor on
  every move, on top of the selection that existed when the range started: moving back shrinks
  it. Moving the anchor starts a new range, so e.g. click r1, Shift-click r3, click r6,
  Shift-click r8 selects r1–r3 and r6–r8.
- Selected keys whose row has left `rows` (a data refresh) are ignored for the checkboxes and
  the "N selected" count, and dropped from the reported keys on the next change.
- The header checkbox covers the filtered rows and shows the mixed state.
- On a row: ⌘/Ctrl-click toggles, Shift-click extends.
- `onSelectedChange(keys, rows)` lists the keys in view order, then any selected keys that are
  filtered out; `keys` and `rows` always have the same length.
- It is controlled with `selected`, or starts from `defaultSelected`.

**Keyboard** (focus the grid):

| Key | Action |
|---|---|
| ↑ ↓, PageUp, PageDown, Home, End | move the row cursor (`aria-activedescendant`) |
| Shift + those keys | extend the selection |
| Space | toggle the cursor row |
| Enter | activate the cursor row (`onRowClick` / `inspect`) |
| ⌘/Ctrl-A | select all filtered rows |
| Esc | clear the selection |

**Row activation.** A plain click or Enter calls `onRowClick(row, index)`, where `index` is the
row's index in `rows`, and/or opens the inspector with `inspect(row)` (it writes the inspector
store; the shell mirrors it to `?inspect=`, see §5.3). The inspected row is
highlighted (`data-active`); `activeKey` overrides that. With neither prop, a click on a
selectable table toggles the row. Clicks on buttons, links and inputs inside cells never
activate the row.

**Other props:**

- `exportName` adds a CSV button, over the visible columns:
  - with no selection, it exports the filtered, sorted rows shown;
  - with a selection, it exports EVERY selected row (the button reads "CSV (n)" with the same
    n), including selected rows the filter hides, in the current sort order. The tooltip says
    how many of them are hidden.

  The CSV follows RFC 4180, and spreadsheet formulas in text cells are defused. The pure
  function is `toCSV`.
- `urlKey` keeps the filter and sort in the URL as `<key>.q` and `<key>.sort` (`-psnr,name`).
  It needs a router.
- The header is sticky. `height` is the viewport's max-height (default 560). `height="auto"`
  means no inner scroll and no virtualisation.
- Rows are virtualised above `virtualize` rows (default 150; `true`/`false` force it). They are
  measured, and `rowHeight` is only the estimate.
- `empty` sets the no-rows message. A filter that matches nothing shows "Clear filter".
  `loading` shows a skeleton.
- `dense`, `hideToolbar`, `rowClassName`, `caption`, `searchable`, `filterPlaceholder`,
  `columnVisibility`/`onColumnVisibilityChange`.

Memoize `columns`: sorting and filtering re-run when `rows`, `columns`, `sort` or `filter`
change. Rows must be distinct objects, or unique primitives, with unique keys. The pure model
(`parseFilter`, `filterRows`, `sortRows`, `nextSort`, `toCSV`, `rangeKeys`, `estimateWidths`,
`parseSort`/`serializeSort`) is in `ui/tableModel.ts` and unit-tested.

The v1 `Table` (compat) stays for small static tables. Its clickable rows are now reachable by
keyboard: Tab, then Enter or Space.

### 9.4 Charts: `Plot` v2, `Legend`, `useLegend`

Every v1 prop is unchanged: `xDomain, yDomain, xScale, xTicks, yTicks, xLabel, yLabel, title,
series, bands, guides, heat, onHeatClick, onPlotClick, highlight, height, aspect`. The v1 look
is unchanged too. New props:

```tsx
<Plot {...v1props}
  yScale="log"                       // log y (domain > 0; non-positive values are gaps)
  tooltip                            // default true: crosshair + nearest series + the lines at that x
  zoom zoomAxes="x"                  // default true / "xy"
  syncKey="train-curves"             // linked crosshair across plots with the same key
  legend="auto" legendToggle         // built-in legend (items or "auto" from named series); click toggles
  exportName="knee-psnr"             // PNG + CSV buttons (top-right, on hover/focus)
  xFormat={(v) => `${v} e⁻`} yFormat={(v) => v.toFixed(2)}
  aria-label="Knee PSNR, VIS"
  view={view} onViewChange={setView} // optional controlled zoom ({x, y} domains or null)
  hidden={hidden} onHiddenChange={setHidden} emphasis={key} />
```

Series take two new optional fields:

- `name`: the tooltip and auto-legend name (default: `label`). `label` is still the in-plot text
  drawn on the line.
- `key`: the visibility group (default `name`, then `label`, then `#index`). Series sharing a
  key toggle together, e.g. the 26 member lines under one "members" legend entry.

**Tooltip.** The header is the x value; the rows are the line/histogram series at that x, in
series order, the hovered (nearest) one marked `data-nearest` and bold. With more than 8 visible
lines the rows are the hovered series plus the lines whose value lies nearest the cursor, and a
"+N more" line counts the rest (so hovering the ensemble mean over a cloud of 350 pair curves
still names the mean). A scatter hit shows that point; a heat plot shows the cell count.

Mouse and keyboard controls. A plain wheel scrolls the page: a mouse click on the chart does not
change that, and neither does holding Ctrl/⌘ for a Ctrl-wheel.

| Input | Action |
|---|---|
| hover | crosshair and tooltip |
| drag | box-zoom |
| Shift-drag | pan |
| wheel with Ctrl/⌘ held, or while the plot has keyboard focus | zoom about the cursor |
| double-click, `0`, Esc | reset the zoom |
| `+`, `−` (plot focused) | zoom |
| Shift+arrows | pan |
| ← → | step the value readout, announced in a live region (gaps — null, or ≤ 0 on a log axis — are skipped) |
| ↑ ↓ | switch series |

The plot has keyboard focus when it was tabbed to, or when one of its own keys (`+ = − _ 0`,
Esc, the arrows) was pressed on it since the last click. Bare modifiers (Ctrl, ⌘, Shift, Alt)
and keys the plot ignores (Space, PageDown, letters, …) do not count. Ctrl/⌘/Alt key
combinations are left to the browser: Ctrl/⌘ + `=` `−` `0` zoom the page, not the plot.

A click without a drag still fires `onPlotClick` / `onHeatClick`, mapped through the current
zoom and any log axis. Heat plots show the hovered cell's count in the tooltip. When zoomed,
the caller's ticks are kept if at least 3 remain in view; otherwise ticks are generated with
`ticks.ts`, labelled by `xFormat`/`yFormat`. An uncontrolled zoom resets when
`xDomain`/`yDomain` change.

The plot redraws only when the drawn inputs change. The inputs are compared structurally, and
handlers are ignored, so `series={[{ x: data.x, y: data.y, color: C.mean }]}` rebuilt on every
render costs nothing. Two function-valued inputs are drawn:

- `heat.color` is compared by identity: a new closure redraws (a tint change shows at once), so
  memoise it when the parent re-renders often.
- `xFormat`/`yFormat` label the generated ticks of a zoomed axis; those ticks are compared by
  their labels, so an inline formatter is free and a changed one redraws.

The plot also redraws on a resize (one `ResizeObserver` for the plot's life), on a theme flip
(`useResolvedTheme`) and on a zoom, visibility or emphasis change. Hover never repaints the
canvas: the crosshair, marker, zoom box and tooltip are DOM overlays. Fonts are read from
`--font-mono` / `--font-sans`.

The chart is a keyboard-focusable `role="figure"` named by `aria-label ?? title`, with a
visually hidden text summary (series count and axis ranges). The canvas is `aria-hidden`.

**Legend.** A plot has a legend only when `legend` is given (`"auto"` builds it from the series
that have a `name`); `legendToggle={false}` makes it static. `<Legend items>` alone is the v1
static legend. To connect an external legend to one or more plots:

```tsx
const lg = useLegend();
<Plot {...props} {...lg.plotProps} />                       // hidden + onHiddenChange + emphasis
<Legend items={[{ label: "mean", color: C.mean }, …]} {...lg.legendProps} />  // toggle + hover-emphasis
```

A legend item's `key` defaults to its `label` and matches `Series.key ?? name ?? label`.
Hovering or focusing an entry dims the other series.

The pure maths is in `charts/plotModel.ts`, unit-tested: `axis`, `zoomDomain`, `panDomain`,
`clampView`, `nearestPoint`, `readoutAt`, `tooltipReadout`, `drawable`, `stepDrawable`, `viewTicks`,
`seriesToCSV`, `sameInputs`, `sameValue`. The CSV export uses the DataTable cell encoder (`csvCell` from
`ui/tableModel.ts`: RFC 4180, spreadsheet formulas in series names defused). The types
are in `charts/types.ts`, re-exported by `Plot.tsx`.

### 9.5 Testing kit components

- happy-dom has no layout and no canvas: `getContext` returns null. The Plot still computes its
  geometry and handles pointer events at the default 640×320 size.
- A virtualised `DataTable` needs a viewport height. Stub
  `HTMLElement.prototype.offsetHeight` for `.ui-dt__scroll`, as `DataTable.test.tsx` does.
- Radix menus open on `pointerdown` (`fireEvent.pointerDown(trigger, { button: 0 })`). Tabs
  activate on `mouseDown`.
- A Plot tooltip is `aria-hidden`: query inside it with `{ hidden: true }`.
- happy-dom's `WheelEvent` drops `ctrlKey`/`clientX`. Define them on the event by hand.
- Plot keyboard focus: `fig.focus()` with no preceding `pointerDown` is keyboard focus (wheel
  zooms); `pointerDown` on the canvas then `focus()` is a mouse click (wheel scrolls). To
  switch a clicked plot to keyboard mode, press one of its keys (`ArrowRight`, `+`); a
  modifier such as `{ key: "Shift" }` does not.
- Accessible names: a Field's error and description are `aria-describedby` targets, not part
  of the control's name. Assert them through the ids in `aria-describedby` (`kit.test.tsx` has
  a `description()` helper).
- `confirm()` dialogs are `role="alertdialog"`: query them with `findByRole("alertdialog")`.
- Tokens: `theme/tokens.test.ts` accepts runtime Radix variables (`--radix-*`) as defined.

## 10. Shell, router, palette, shortcuts, inspector, jobs, display

### 10.1 Router: `app/routes.ts`, `app/App.tsx`

```ts
const router = createBrowserRouter(buildRoutes({ layout: Shell }), { future: ROUTER_FUTURE });
// tests: createMemoryRouter(buildRoutes({ components: fakes, layout? }), { initialEntries: [url], future: ROUTER_FUTURE })
```

`buildRoutes()` turns the manifest into one pathless root route (the shell, `errorElement`
`RouteError`) with these children:

| Route | Renders |
|---|---|
| `/` | the home workspace |
| `<workspace path>/*` (`/sky/*`, `/ensemble/:mode/*`, `/inspect/*`, …) | the lazy workspace component (`workspaceComponents[id]`) inside a Suspense skeleton |
| every `manifest.redirects` key, and `/app/*` | `RedirectRoute`: `<Navigate replace>` to `redirectTarget(pathname, search)` + the hash |
| `*` | `NotFound` |

The router only picks the workspace. The workspace validates the rest of the path with the
manifest matcher (`<Workspace>`, §11), so `/sky/unknown`, `/ensemble/foo` and
`/ensemble/starfull/knee/x` show Not found — the same URLs Flask 404s. A bare workspace path
(`/sky`, `/ensemble/starless`) redirects to its default tab (or the workspace's `redirectTab`)
with query and hash kept.

Adding a workspace = a manifest entry (orchestrator) + an `app/nav.ts` entry + one line in
`workspaceComponents` + the folder. `routes.test.tsx`, `nav.test.ts` and
`workspaces/workspaces.test.ts` fail while one of them is missing.

`ROUTER_FUTURE` opts into every v7 future flag (and `<RouterProvider future={{
v7_startTransition: true }}>`); use the same flags in tests to keep them quiet.

### 10.2 Shell: `app/Shell.tsx`

The root layout: rail | top bar, version banner, stage (`<main id="main" class="stage">`, the
scroll container holding `<Outlet/>`) and the docked inspector (a `react-resizable-panels`
panel; its width is `prefs.inspectorWidth`, saved after a drag). Below 900 px
(`NARROW_QUERY`) the rail is a drawer (top-bar menu button) and the inspector a bottom sheet.

The shell mounts, exactly once: `UiProvider` (§9.1), `useInspectorUrlSync`, `useJobToasts`, the
global shortcuts, the command palette, the ? sheet, the Display panel, a skip link, and:

- `document.title` = `pageTitle(pathname)`, e.g. "Members · Ensemble (starless) · EuclidPolish";
- stage scrolling (`useStageScroll`): a new pathname scrolls to the top, back/forward restores
  that history entry's position, and a query-only change (a `useUrlState` write, `?inspect=`)
  keeps the scroll. This replaces react-router's `<ScrollRestoration>`, which only drives the
  window's scroll, not an inner scroll container.

Overlays open through `useShellUi` (`app/shellStore.ts`) from anywhere:

```ts
import { openPalette, openDisplayPanel, openShortcutSheet, openJobTray, useShellUi } from "../app/shellStore";
useShellUi.getState().openOnly("display");   // "palette" | "shortcuts" | "display" | "tray" | "drawer"
```

Top bar, left to right: menu (narrow only), breadcrumbs (workspace [· params] › tab › the
inspected entity's title), the ⌘K search button, the FASRC badge (`/api/fasrc/status`; offline
shows the real `last_error` in its tooltip; links to Settings › Connections), the job tray, the
Display button, the theme toggle and the ? sheet button. The version banner under it appears when
`/api/version` says `behind` (dismissable per HEAD commit). The rail shows the nine workspaces
(icons from `nav.ts`), a running-jobs badge on Ops, a "server behind HEAD" badge on Settings,
and the collapse toggle (`prefs.railCollapsed`; collapsed items get tooltips).

`app/status.ts` shares the two status resources (one cache entry each):

```ts
const v = useVersion().data;        // C3: {boot_short, head_short, behind, dirty, started_at, pid, dist}
const f = useFasrcStatus().data;    // C4: {ssh_connected, connected_at, socket, last_error}
```

### 10.3 Keyboard shortcuts: `hooks/useShortcut.ts`

```ts
useShortcut("g s", () => navigate("/sky/atlas"), { description: "Go to Sky", scope: "Navigation" });
useShortcut("$mod+Enter", submit, { description: "Submit", allowInInputs: true });
useShortcut("ArrowRight", next, { description: "Next object", scope: "Viewer", target: viewerRef });
const off = bindShortcut("Shift+E", run, { description: "Evaluate" });   // imperative; off() unbinds
<Kbd keys={comboParts("$mod+k")[0]} />                                     // display
```

- Combos are tinykeys syntax: `$mod` (⌘ / Ctrl), `Shift+?`, `[Shift]+?` (optional modifier), a
  space between the presses of a sequence (`g s`).
- A shortcut does not fire while typing in an input, textarea, select or contenteditable, nor
  with focus inside a modal dialog, unless `allowInInputs`. It also skips an event another handler
  already `preventDefault()`ed: the legacy image viewer consumes its keys (q–y, arrows, space, S)
  on `document`, which runs before these window listeners.
- The handler is read from a ref (no rebinding on re-render). It `preventDefault()`s unless it
  returns `false` (return `false` when it did nothing, so the key keeps its default).
- `target` (an element or a ref) scopes it to that element, e.g. the focused viewer. `enabled:
  false` unbinds. `hidden: true` keeps it out of the ? sheet.
- Every bound shortcut is listed in `useShortcutRegistry` (the ? sheet reads it, grouped by
  `scope`, default "Global").

Global shortcuts (`app/GlobalShortcuts.tsx`): `$mod+k` palette (also in fields), `?` this
sheet, `g h/s/e/r/d/f/i/o/,` go to Home/Sky/Ensemble/Realism/Data/Figures/Inspect/Ops/Settings,
`[` collapse the rail, `]` show/hide the inspector, `Shift+D` Display panel, `Shift+J` jobs,
`Shift+T` toggle the theme. Escape inside the inspector closes it. Pages must not rebind these.

### 10.4 Inspector: `app/inspector.ts`, `app/InspectorPanel.tsx`

```tsx
// a workspace module, at module scope (runs when the workspace loads):
registerInspector("member", MemberInspector, { title: (id) => `Member ${id}` });   // returns unregister
function MemberInspector({ id }: { id: string }) { … }   // lazy data via useResource

openInspector({ kind: "member", id: "member_196" });    // from anywhere
closeInspector();
<DataTable inspect={(r) => ({ kind: "member", id: r.name })} … />   // row click → inspector
const href = inspectHref({ kind: "tile", id: "nexus/12" }, location); // "/sky/atlas?…&inspect=tile%3Anexus%2F12"
```

- The panel shows the registered component for `current.kind` (a later registration of a kind
  wins until unregistered), with back/forward (the store's history), pin (pinned targets are chips
  at the top), copy link and close. Its content has its own error boundary and Suspense.
- An unregistered kind shows a "no inspector yet" card with the target; the URL keeps it, so
  the link works once the owning workspace registers the kind.
- `?inspect=kind:id` sync (`useInspectorUrlSync`, mounted by the shell): on load and on
  back/forward the URL wins; a navigation carrying the param opens it; a navigation without it
  keeps the open inspector and re-adds the param (inspection survives moving between workspaces);
  store changes are written in place (history replace), a malformed param is dropped.
- Built-in kind `job` (`app/inspectors/JobInspector.tsx`): `job:local/<job_id>` (status, cancel,
  full searchable log, JSON result) and `job:slurm/<jobid>` (the FASRC live monitor).
- Kinds planned by the spec for phase 3: `member`, `tile` (`nexus/<n>`), `realtile`, `source`,
  `experiment`, `fits`, `figure`. The palette already opens `member:member_<n>` and
  `tile:nexus/<n>`, but **nobody registers those kinds yet**: until W-Ensemble registers
  `member` and W-SkyAtlas registers `tile`, they open the "no inspector yet" card.
- The panel re-renders its content on a theme or accent flip (`useTokenRerender`, §11.1).
- A `job:local/<id>` inspector polls the job every 2 s while it runs (status from the detail,
  else the jobs feed) and stops once it has finished, on a 404 (a stale shared link after a
  server restart) or on an error with nothing to show.

### 10.5 Command palette: `app/palette.ts`, `app/CommandPalette.tsx`

```ts
usePageActions([
  { id: "evaluate", label: "Evaluate on test set", group: "Ensemble", keywords: ["psnr"],
    shortcut: "Shift+E", disabled: !ready, run: () => evalJob.run("/ensemble/evaluate", {…}) },
]);
```

- A page registers its actions while mounted. They are listed first, under their `group`
  (default "This page"). `run` is read from a ref, so inline closures are free; the registration
  changes only when an id, label, group, keyword, shortcut or `disabled` changes. A `shortcut` is
  bound for as long as the page is mounted (and listed in the ? sheet); a disabled action ignores
  it.
- The palette also lists every page (`allPages()`: each workspace × tab × ensemble regime) and
  the global commands (theme light/dark/system/toggle, Display panel, jobs, "Run a FASRC step…"
  (→ `/ops/fasrc`), shortcuts, rail, close inspector, refresh all data, copy a link to this
  view).
- **Not done in phase 1 (phase-3 owners):** no page registers `usePageActions` yet (the legacy
  pages predate it), so on real pages the page-actions group is empty until the phase-3 tabs
  register theirs (evaluate, fit gate, pull, fly to…). "Run job X" is only the route to Ops ›
  FASRC; per-step commands ("Submit <step>") belong to W-Ops. `member N` / `nexus N` need the
  `member` / `tile` inspector kinds (§10.4).
- Free text adds `paletteSuggestions(text, parseSkyCoord)` on top:

  | Typed | Offers |
  |---|---|
  | `269.27 66.1`, `17:57:04 +66:06:00` | `/sky/atlas?ra=<deg>&dec=<deg>` |
  | `member 196`, `member_196` | inspector `member:member_196` |
  | `nexus 12`, `tile 12` | inspector `tile:nexus/12` |
  | `data/…/x.fits` | `/inspect?fits=<path>` |
  | any other text | `/sky/atlas?goto=<text>` (the atlas resolves names with Sesame) |

  The suggestions are always listed (cmdk `forceMount`) but cmdk does not count them, so the
  palette shows "Nothing matches" only when there are no suggestions either.

  **URL contract for W-SkyAtlas:** the atlas tab reads `?ra&dec` (degrees; centre the view) and
  `?goto=` (a name or coordinate string for `gotoObject`).

### 10.6 Jobs in the shell: `app/JobTray.tsx`

- The tray button shows `useJobsFeed().runningCount` (running local + live SLURM). Its popover
  lists local jobs (running first, the newest 8) and live SLURM jobs, with progress, ETA,
  cancel (SLURM cancels ask with `confirm()`), and a click on a job opens its log in the
  inspector. FASRC offline (the C4 503) reads "FASRC offline", never an error.
- `useJobToasts()` (the shell) toasts every local job this session saw running when it ends:
  success, error (first line of the error, 10 s) or cancelled, each with a "Log" action.
- Reusable parts: `<JobList jobs limit? empty? onOpen?>`, `<JobRow job>`, `<SlurmRow job>`,
  `orderJobs(jobs, limit)` (Home and Ops › Jobs use them).

### 10.7 Display panel: `app/DisplayPanel.tsx`, `app/displaySections.ts`

The Display panel (top-bar button, `Shift+D`, palette) edits the C7 store (`useDisplay`, §5.1):
colour mode, custom RGB bands, stretch, colormap, residual colormap, NaN colour, invert, and the
knee / gain / black point of the `default`, `euclid` and `jwst` transfer groups; "Viewers":
link all viewers, mouse-wheel behaviour; "Reset to defaults". A workspace adds its own section:

```ts
useEffect(() => registerDisplaySection({ id: "sky", title: "Sky", order: 10, Component: SkyDisplay }), []);
```

Sections render after the built-in ones by `order` (default 100); registering an existing `id`
replaces it. **Extension point for W-SkyAtlas:** the "Sky" section (base HiPS colormap, stretch,
cuts…) is added this way.

### 10.8 Errors and not found: `app/ErrorBoundary.tsx`, `app/NotFound.tsx`

- Every tab renders inside `<ErrorBoundary resetKey={pathname} label="Workspace › Tab">`: a crash
  shows a contained card with Retry, Reload page and Copy details (message, URL, time, stacks);
  the rest of the console keeps working and navigating away resets it. A failed lazy chunk (the
  bundle was rebuilt under an open page) offers only Reload. `RouteError` is the root route's
  `errorElement`.
- `<NotFound/>`: the path, a link to the workspace the path starts with, Home, and the ⌘K hint.

## 11. Workspaces

### 11.1 The contract (C8)

Each workspace is `src/workspaces/<id>/`:

```tsx
// src/workspaces/realism/index.tsx
import { Workspace, defineTabs } from "../../app/workspace";

export const TABS = defineTabs("realism", {           // exactly the manifest's tabs
  overview: { load: () => import("./tabs/Overview") },
  noise:    { load: () => import("./tabs/Noise"), badge: "v5" },   // label from nav.ts unless `label`
  …
});

export default function RealismWorkspace() {
  return <Workspace id="realism" tabs={TABS} />;       // aside={…}: controls right of the tabs
}
```

- `defineTabs` runs once at module scope (it creates the `React.lazy` components) and warns
  about a tab the manifest does not list. `workspaces.test.ts` checks every workspace declares
  exactly its manifest tabs and that each tab module loads to a component.
- `<Workspace>` renders the router-linked tab strip (`<WorkspaceTabs>`: kit `Tabs` with `to`,
  `aria-current="page"`, the links keep `?inspect=`) and the active tab in its error boundary and
  a `TabSkeleton` Suspense fallback. A tabless workspace (home, inspect) passes `children`.
- A bare workspace path redirects to the manifest default tab, or to `redirectTab` when that is
  one of the workspace's tabs (the ensemble workspace passes the last tab it showed, so a regime
  switch to the bare `/ensemble/<mode>` keeps the tab).
- A theme or accent flip re-renders the active tab (and `children`, which `<Workspace>` clones
  for that reason): the route elements above a workspace are static, and the legacy pages read
  colour tokens during render. `bindPrefsToDocument` has already updated `<html data-theme>`
  when that render runs. `useTokenRerender()` (exported from `app/workspace.tsx`) gives the same
  behaviour to other hosts; the inspector panel uses it.
- A tab module default-exports its component and owns its URL state (`useUrlState`), palette
  actions (`usePageActions`), inspector kinds (`registerInspector`), and its empty/loading/error
  states. Co-locate its CSS in the workspace folder, scoped by a workspace class (`.ws--<id>` is
  on the workspace root).
- `<PendingTab workspace tab links?>`: the "arrives in phase 3" EmptyState, with links to related
  pages that exist.
- Tab labels, descriptions, icons and the go-key live in `app/nav.ts` (`WORKSPACE_META`).

### 11.2 How to build a tab (example)

```tsx
// src/workspaces/ensemble/tabs/Members.tsx
import { useState } from "react";
import { useResource, invalidate } from "../../../api/query";
import { useJob } from "../../../api/jobs";
import { useLocation } from "react-router-dom";
import { matchPage } from "../../../app/manifest";
import { usePageActions } from "../../../app/palette";
import { useUrlState } from "../../../hooks/useUrlState";
import { Button, Card, CardBody, CardHead, DataTable, JobProgress, Page, PageHead, confirm, toast, type DataColumn } from "../../../ui";

const COLUMNS: DataColumn<Member>[] = [
  { id: "name", header: "Member" },
  { id: "psnr", header: "PSNR", numeric: true, cell: (m) => m.psnr?.toFixed(2) ?? "—" },
];

export default function Members() {
  const mode = matchPage(useLocation().pathname)?.params.mode ?? "starfull";   // the :mode path param
  const [colorBy, setColorBy] = useUrlState("color", "loss");     // shareable view state (?color=)
  const status = useResource<Status>(`/ensemble/status.json?mode=${mode}`, [mode], { ttl: 60_000 });
  const archive = useJob("ensemble:archive");                     // survives navigation (job tray)
  const [sel, setSel] = useState<string[]>([]);
  usePageActions([{ id: "refresh", label: "Refresh members", group: "Ensemble", run: status.reload }]);

  async function archiveSelected() {
    if (!(await confirm({ title: `Archive ${sel.length} members?`, tone: "danger", confirmLabel: "Archive" }))) return;
    await archive.run("/ensemble/archive-member", { member: sel.join(",") }, {
      onDone: (j) => { invalidate("/ensemble/"); if (j.status === "done") toast.success("Archived"); },
    });
  }

  return (
    <Page>
      <PageHead eyebrow="ensemble · members" title="Members" />
      <Card>
        <CardHead title="Members" right={<Button disabled={!sel.length} onClick={archiveSelected}>Archive</Button>} />
        <CardBody>
          <DataTable rows={status.data?.members ?? []} columns={COLUMNS} rowKey={(m) => m.name}
            selectable selected={sel} onSelectedChange={setSel} urlKey="m"
            inspect={(m) => ({ kind: "member", id: m.name })} exportName="members" loading={status.loading} />
          <JobProgress job={archive.job} error={archive.error} />
        </CardBody>
      </Card>
    </Page>
  );
}
```

Charts: `import Plot, { Legend, useLegend } from "../../../charts/Plot"` (§9.4). Formatting:
`format.ts` / `ticks.ts` (§7), never a local tick helper.

### 11.3 Where every tab points in phase 1 (legacy adapters)

Each adapter tab module is one line, `export { default } from "../../../pages/<Page>"`.

| Tab | Renders |
|---|---|
| `/` | Home dashboard (new: FASRC, server version, running jobs, the STARFULL ensemble headline — the production spatial gate's PSNR when the eval summary scored it, else the labelled plain mean with its gain over the mean member —, local data, recent jobs, links) |
| `sky/atlas` · `results` · `catalog-eval` | `pages/JwstEuclid` · `pages/Inference` · `pages/Evaluation` |
| `sky/experiments` | PendingTab |
| `ensemble/:mode/{overview,members,curves,knee,diagnostics,combiners,disagreement}` | `pages/Ensemble` (the whole old page on each tab) |
| `ensemble/:mode/train` | `pages/TrainMembers` |
| `realism/noise` · `galaxies` · `stars` · `pixels` · `visual` | `pages/Noise` · `GalaxyDistributions` · `StarDistribution` · `PopulationComparison` · `SyntheticReal` |
| `realism/overview` | PendingTab (links to the other tabs) |
| `data/records` · `catalog` · `cutouts` · `psfs` · `tng` | `pages/Sky` · `Catalog` · `Cutouts` · `Psfs` · `Tng` |
| `figures/grid` · `plates` | `pages/figure-grid/FigureGridBuilder` (in a Page) · `pages/Visualization` |
| `figures/results` | PendingTab |
| `inspect` | `pages/Inspect` (`?fits=<project-relative path>`) |
| `ops/fasrc` · `tracking` · `git` | `pages/Fasrc` · `Tracking` · `Git` |
| `ops/jobs` | new: the local job list (`JobList`) |
| `ops/provenance` | PendingTab |
| `settings/config` | `pages/Config` |
| `settings/connections` | new, minimal: FASRC state with `last_error`, connect / retry / disconnect |
| `settings/appearance` | new: theme light/dark/system, accent, density, rail, inspector width, viewer wheel, Display defaults |
| `settings/about` | new: boot commit vs HEAD, dirty tree, started, pid, dist build, bundle mode + React version |

The ensemble workspace adds a starfull/starless switch beside its tabs that keeps the tab.

Known adapter limitations (the page bodies are not the shell's to edit; fixed when the phase-3
owner splits the page):

- **Two regime switches on every ensemble tab** (W-Ensemble): the workspace switch and the old
  page's own `Segmented` in `pages/Ensemble.tsx`, which navigates to the bare
  `/ensemble/<mode>`. Both keep the tab now (the bare path returns to the last tab, §11.1);
  W-Ensemble removes the page's switch.
- **The figure-grid builder shows on both `figures/grid` and `figures/plates`** (W-Figures):
  `pages/Visualization` embeds `<FigureGridBuilder/>` under its plates. Until Visualization is
  split, `plates` is "plates + the builder" and `grid` is the builder alone.
- Every ensemble tab except `train` shows the whole old Ensemble page (see the table).

### 11.4 Replacing an adapter in phase 3

1. Write the tab in `src/workspaces/<id>/tabs/<Tab>.tsx` (your folder) on the foundation.
2. Keep the tab ids and the module path; `workspaces.test.ts` keeps the contract.
3. Delete the absorbed `src/pages/<X>.tsx` (and its CSS / `ui/pages-compat.css` block) when no
   other tab imports it, and remove its row from the `ADAPTERS` list in `workspaces.test.ts`.
4. A missing shared primitive goes to the orchestrator; do not fork one.
