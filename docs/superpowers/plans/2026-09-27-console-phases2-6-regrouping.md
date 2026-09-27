# Console Phases 2–6 — Regrouping: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-home every console page into the approved "Loop console" grouping (Home · Synthetic · Models · Sky · Figures · Files · Runs · Notebook · System), merge the overlapping pages, and redirect every old URL (query parameters included) to its new home.

**Architecture:** First a single foundation task moves the *routing*: the route manifest v2 (`euclid_polish/web/spa_routes.json`), a query-aware redirect-rule matcher implemented identically in Flask (`spa_routes.py`) and the SPA (`app/manifest.ts`) and pinned by one shared case file, new workspace shells whose tabs load the existing components, and every internal page link. The console works end to end after the foundation, with old content in new places. Then five content teams work in parallel on disjoint folders to merge pages (Sky, Synthetic, Models, Files/Runs/Notebook/System, Home + Figures). An integration pass checks every walkthrough and every old URL.

**Tech Stack:** Flask (`euclid_polish/web`), React 18 + TypeScript + Vite (`euclid_polish/web/frontend`), vitest, pytest.

**Spec:** `docs/superpowers/specs/2026-09-27-console-regrouping-design.md` (approved by the user 2026-09-27, with Files as its own rail entry and the user's statistics rule). Phase 1 (statistics kit, deletions) is done: `ff5d23a`.

**Standing rules (every task):** opening a page never starts a job; nothing sticks over images; sentence case; colours from tokens; the statistics rule (necessary numbers visible — `SummaryLine`, `FactsList`, tables, legend/control counts, `Caption`; `Details` only for provenance); Python imports at module top; backend API URLs (`/api/*`, `/ensemble/*.json`, `/viewer/*`, …) never change — only PAGE paths move; never write `euclid_polish/web/static/dist` except in the final ship step; never touch `data/**`.

---

## New route manifest (v2)

| id | path | tabs (default first) |
|---|---|---|
| home | `/` | — |
| synthetic | `/synthetic` | status, records, galaxies, stars, noise, psf, fields |
| models | `/models/:mode` (mode ∈ starfull, starless; default starfull) | leaderboard, members, train, combiner, diagnostics, images |
| sky | `/sky` | atlas, targets, compare |
| figures | `/figures` | plates, sheet |
| files | `/files` | — |
| runs | `/runs` | live, history, steps |
| notebook | `/notebook` | log, backups, sandboxes |
| system | `/system` | connections, config, lineage, code, storage, appearance |

Rail order: Home, Synthetic, Models, Sky, Figures, Files, Runs, Notebook, System. The models path keeps the param name `mode` (the code's `useMode()`), so `/ensemble/:mode/…` rules substitute `:mode` directly.

Tab labels (unique across the rail): Status, Records, Galaxies, Stars, Noise, PSF, Fields · Leaderboard, Members, Train, Combiner, Diagnostics, Images · Atlas, Targets, Compare · Plates, Sheet · Live, History, Steps · Log, Backups, Sandboxes · Connections, Config, Lineage, Code, Storage, Appearance.

## Redirect rules (query-aware)

`spa_routes.json` gains `"redirectRules": [...]`, evaluated in order BEFORE the exact-path `redirects` map; the first match wins. A rule:

```json
{ "from": "/ensemble/:mode/curves", "query": {"layout": "time"},
  "to": "/runs/history", "drop": ["layout"], "set": {"step": "ensemble_train"} }
```

- `from`: a path pattern; `:name` matches one non-empty segment and binds it; trailing slash ignored.
- `query` (optional): every key must be present; the value is `"*"` (any value), a string (equal) or a list (one of).
- Target path: `to` with every `:name` substituted.
- Target query, in this order: start from the original pairs (order kept); remove `drop` keys; `rename` keys (`{"source": "set"}`); `map` values per key (`{"set": {"tile": "cached"}}`, unmapped values kept); `set` keys (replace the value, or append).
- Encoding (identical in both languages): WHATWG form encoding — keep `A–Z a–z 0–9 * - . _`, space → `+`, every other byte as `%XX` upper-case hex.
- A fragment cannot reach the server: targets never use `#…`; use a query key (`?section=census`).

The old→new table (from the spec) that the rules must implement:

| Old | New |
|---|---|
| `/` | `/` |
| `/app` | `/` |
| `/sky` | `/sky/atlas` |
| `/sky/atlas` | `/sky/atlas` |
| `/sky/atlas?obs=1` | `/sky/atlas?obs=1` |
| `/jwst-euclid` | `/sky/atlas?obs=1` |
| `/sky/results` | `/sky/targets` |
| `/sky/results?source=nexus|tile|field|eval|poster|pair` | `/sky/targets?set=nexus|cached|legacy|lenses,galaxies|poster|pairs` |
| `/sky/results?source=archive` | `/synthetic/fields?ref=1` |
| `/sky/results?inspect=tile:<ref>` | `/sky/targets?inspect=tile:<ref>` |
| `/sky/results?diag=1&fd=cross|brightness` | `/models/starfull/diagnostics?d=real-field&fd=cross|brightness` |
| `/sky/results?diag=1&fd=occupancy` | `/models/starfull/diagnostics?d=real-field (RBF occupancy deleted)` |
| `/inference` | `/sky/targets` |
| `/sky/experiments` | `/sky/compare` |
| `/sky/experiments?exp=<id>&scope=&metric=` | `/sky/compare?exp=<id>&scope=&metric=` |
| `/sky/catalog-eval` | `/sky/targets?set=lenses,galaxies` |
| `/sky/catalog-eval?g=lensA|lensB|lensC|galaxies` | `/sky/targets?set=lenses&g=A|B|C or ?set=galaxies` |
| `/sky/catalog-eval?g=syn-lens|syn-gal` | `/models/starfull/images?set=stamps&g=syn-lens|syn-gal` |
| `/sky/catalog-eval?figs=1` | `/models/starfull/diagnostics?d=recovery` |
| `/evaluation` | `/sky/targets?set=lenses,galaxies` |
| `/ensemble` | `/models/starfull/leaderboard` |
| `/ensemble/:mode` | `/models/:mode/leaderboard` |
| `/ensemble/:mode/overview` | `/models/:mode/leaderboard` |
| `/ensemble/:mode/knee (view, band, range, color params)` | `/models/:mode/leaderboard (same params)` |
| `/training` | `/models/starfull/leaderboard` |
| `/ensemble/:mode/members` | `/models/:mode/members` |
| `/ensemble/:mode/curves` | `/models/:mode/members?view=curves` |
| `/ensemble/:mode/curves?layout=time` | `/runs/history?step=ensemble_train` |
| `/ensemble/:mode/train` | `/models/:mode/train` |
| `/train-members` | `/models/starfull/train` |
| `/ensemble/:mode/diagnostics` | `/models/:mode/diagnostics` |
| `/ensemble/:mode/diagnostics?d=spectrum|transfer|coherence` | `/models/:mode/diagnostics?d=spectrum|transfer|coherence` |
| `/ensemble/:mode/diagnostics?d=stderr|brightness|calibration` | `/models/:mode/diagnostics?d=spread` |
| `/ensemble/:mode/diagnostics?d=axes` | `/models/:mode/diagnostics?d=spread (axes deleted)` |
| `/ensemble/:mode/combiners` | `/models/:mode/combiner` |
| `/ensemble/:mode/disagreement` | `/models/:mode/images` |
| `/realism` | `/synthetic/status` |
| `/realism/overview` | `/synthetic/status` |
| `/realism/noise` | `/synthetic/noise` |
| `/noise` | `/synthetic/noise` |
| `/realism/galaxies` | `/synthetic/galaxies` |
| `/realism/galaxies?view=distributions|relations|joint` | `/synthetic/galaxies?view=distributions|relations|joint` |
| `/realism/galaxies?view=model` | `/synthetic/galaxies?prior=1 (query in ?how=1)` |
| `/realism/galaxies?view=figure` | `/figures/plates?plate=galaxies` |
| `/galaxy-distributions` | `/synthetic/galaxies` |
| `/realism/stars` | `/synthetic/stars` |
| `/realism/stars?view=density` | `/synthetic/stars` |
| `/realism/stars?view=colours` | `/synthetic/stars (view deleted)` |
| `/realism/stars?view=gaia` | `/synthetic/stars (view deleted)` |
| `/realism/stars?view=prior` | `/synthetic/stars?prior=1 (query in ?how=1)` |
| `/star-distribution` | `/synthetic/stars` |
| `/realism/pixels` | `/synthetic/fields?view=stats` |
| `/realism/pixels?view=pixels` | `/synthetic/fields?view=stats` |
| `/realism/pixels?view=detection` | `/synthetic/fields?view=detection` |
| `/realism/pixels?view=census` | `/synthetic/records?section=census` |
| `/realism/pixels?view=inputs` | `/runs/steps?stage=noise-fields` |
| `/population-comparison` | `/synthetic/fields?view=stats` |
| `/realism/visual` | `/synthetic/fields?view=look` |
| `/synthetic-real` | `/synthetic/fields?view=look` |
| `/data` | `/synthetic/records` |
| `/data/records` | `/synthetic/records` |
| `/data/catalog` | `/synthetic/psf?view=catalogue` |
| `/catalog` | `/synthetic/psf?view=catalogue` |
| `/data/cutouts` | `/synthetic/psf?view=cutouts` |
| `/cutouts, /cutouts/VIS, /cutouts/Y_E, /cutouts/J_E, /cutouts/H_E` | `/synthetic/psf?view=cutouts&band=<band>` |
| `/data/psfs` | `/synthetic/psf?view=epsf` |
| `/psfs` | `/synthetic/psf?view=epsf` |
| `/data/tng` | `/synthetic/galaxies?view=templates` |
| `/tng` | `/synthetic/galaxies?view=templates` |
| `/figures` | `/figures/plates` |
| `/visualization` | `/figures/plates` |
| `/figures/plates` | `/figures/plates` |
| `/figures/grid` | `/figures/sheet` |
| `/figures/results` | `/figures/sheet?pool=gallery` |
| `/inspect (?fits=, ?hdu=, ?view=)` | `/files (same params)` |
| `/ops` | `/runs/live` |
| `/ops/jobs` | `/runs/live` |
| `/ops/jobs?job=<id>` | `/runs/history?run=local:<id>` |
| `/ops/fasrc` | `/runs/live` |
| `/fasrc` | `/runs/live` |
| `/ops/fasrc?view=live|queue` | `/runs/live` |
| `/ops/fasrc?view=steps&step=<s>` | `/runs/steps?step=<s>` |
| `/ops/fasrc?view=history` | `/runs/history` |
| `/ops/fasrc?view=logs&job=&task=` | `/runs/history?run=<job>&logs=1&task=` |
| `/ops/fasrc?view=storage` | `/system/storage?side=fasrc` |
| `/ops/fasrc?view=git` | `/system/code?side=fasrc` |
| `/ops/tracking` | `/notebook/log` |
| `/tracking` | `/notebook/log` |
| `/ops/tracking?view=backups` | `/notebook/backups` |
| `/ops/tracking?view=archive` | `/notebook/backups?show=campaigns` |
| `/ops/tracking?view=sandboxes` | `/notebook/sandboxes` |
| `/ops/tracking?view=jobs&jcamp=<c>` | `/runs/history?campaign=<c>` |
| `/ops/git` | `/system/code` |
| `/git` | `/system/code` |
| `/ops/provenance` | `/system/lineage` |
| `/settings` | `/system/connections` |
| `/settings/config` | `/system/config` |
| `/config` | `/system/config` |
| `/settings/connections` | `/system/connections` |
| `/connection-error` | `/system/connections` |
| `/settings/appearance` | `/system/appearance` |
| `/settings/about` | `/system/code (disk and data roots → /system/storage)` |

(`/inspect` → `/files`, not `/system/files`: the user kept Files in the rail.)

---

### Task F1: Shared redirect cases

**Files:** Create `euclid_polish/web/spa_redirect_cases.json`.

- [ ] Write one case per table row above (and the query variants), e.g.

```json
[
  {"from": "/realism/stars?view=colours&training=1", "to": "/synthetic/stars?training=1"},
  {"from": "/ensemble/starless/curves?layout=time&x=2", "to": "/runs/history?x=2&step=ensemble_train"},
  {"from": "/ensemble/starfull/curves", "to": "/models/starfull/members?view=curves"},
  {"from": "/sky/results?source=tile&inspect=tile%3Aposter%2Fp1", "to": "/sky/targets?set=cached&inspect=tile%3Aposter%2Fp1"},
  {"from": "/inspect?fits=poster%2Fx.fits", "to": "/files?fits=poster%2Fx.fits"},
  {"from": "/realism/pixels?view=census", "to": "/synthetic/records?section=census"},
  {"from": "/sky/atlas", "to": null}
]
```

`"to": null` means "not a redirect" (a current page).

### Task F2: Flask matcher

**Files:** Modify `euclid_polish/web/spa_routes.py`; Test `tests/test_spa_routes.py`.

- [ ] **Failing test** (append):

```python
import json
from pathlib import Path
import pytest
from euclid_polish.web import spa_routes

CASES = json.loads((Path(spa_routes.__file__).parent / "spa_redirect_cases.json").read_text())

@pytest.mark.parametrize("case", CASES, ids=[c["from"] for c in CASES])
def test_redirect_cases(case):
    path, _, query = case["from"].partition("?")
    assert spa_routes.redirect_target(path, query) == case["to"]
```

- [ ] **Implement** in `spa_routes.py` (module top imports: `from urllib.parse import parse_qsl`):

```python
_FORM_SAFE = frozenset(b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789*-._")


def _form_encode(text: str) -> str:
    """WHATWG application/x-www-form-urlencoded, byte for byte like URLSearchParams."""
    out = []
    for byte in text.encode("utf-8"):
        if byte in _FORM_SAFE:
            out.append(chr(byte))
        elif byte == 0x20:
            out.append("+")
        else:
            out.append(f"%{byte:02X}")
    return "".join(out)


def _match_path(pattern: str, path: str) -> dict[str, str] | None:
    want, got = _normalise(pattern).split("/"), _normalise(path).split("/")
    if len(want) != len(got):
        return None
    bound: dict[str, str] = {}
    for w, g in zip(want, got, strict=True):
        if w.startswith(":"):
            if not g:
                return None
            bound[w[1:]] = g
        elif w != g:
            return None
    return bound


def _query_matches(cond: Mapping[str, Any] | None, pairs: list[tuple[str, str]]) -> bool:
    if not cond:
        return True
    have = dict(pairs)
    for key, want in cond.items():
        if key not in have:
            return False
        if want == "*":
            continue
        options = want if isinstance(want, list) else [want]
        if have[key] not in options:
            return False
    return True


def _apply_rule(rule: Mapping[str, Any], bound: dict[str, str], pairs: list[tuple[str, str]]) -> str:
    target = str(rule["to"])
    for name, value in bound.items():
        target = target.replace(f":{name}", value)
    drop = set(rule.get("drop") or [])
    rename = rule.get("rename") or {}
    mapping = rule.get("map") or {}
    out = [(rename.get(k, k), v) for k, v in pairs if k not in drop]
    out = [(k, (mapping.get(k) or {}).get(v, v)) for k, v in out]
    for key, value in (rule.get("set") or {}).items():
        if any(k == key for k, _ in out):
            out = [(k, str(value) if k == key else v) for k, v in out]
        else:
            out.append((key, str(value)))
    if not out:
        return target
    return target + "?" + "&".join(f"{_form_encode(k)}={_form_encode(v)}" for k, v in out)
```

and in `redirect_target`, before the exact map:

```python
    text = query.decode("utf-8", "replace") if isinstance(query, bytes) else (query or "")
    pairs = parse_qsl(text, keep_blank_values=True)
    for rule in manifest.get("redirectRules") or []:
        bound = _match_path(str(rule["from"]), path)
        if bound is not None and _query_matches(rule.get("query"), pairs):
            return _apply_rule(rule, bound, pairs)
```

(Exact-map redirects keep today's behaviour: the original query string appended untouched.)

- [ ] Run `$PY -m pytest -q tests/test_spa_routes.py` → PASS.

### Task F3: SPA matcher

**Files:** Modify `euclid_polish/web/frontend/src/app/manifest.ts`; Test `app/manifest.test.ts`.

- [ ] **Failing test**: import `../../../spa_redirect_cases.json`; for each case, `redirectTarget(path, query)` equals `case.to`.
- [ ] **Implement** the same algorithm in TypeScript (`matchPath`, `queryMatches`, `applyRule`; encode with `URLSearchParams` semantics — build the string by hand with the same safe set so it is byte-identical to Python). `RouteManifest` gains `redirectRules?: RedirectRule[]` with a typed `RedirectRule`.
- [ ] `npx vitest run src/app/manifest.test.ts` → PASS.

### Task F4: Manifest v2 and workspace shells

**Files:** `euclid_polish/web/spa_routes.json`; `frontend/src/app/routes.ts`, `app/nav.ts` (labels, descriptions, rail order, icons), `app/Rail.tsx` if it hard-codes order; new `frontend/src/workspaces/{synthetic,models,files,runs,notebook,system}/index.tsx`; the old `workspaces/{realism,data,ensemble,inspect,ops,settings}/index.tsx` stop being routed (their folders stay as component libraries until the content teams move files); tests `app/routes.test.tsx`, `app/nav.test.ts`, `app/manifest.test.ts`, `workspaces/workspaces.test.ts`, `tests/test_spa_routes.py`.

- [ ] Rewrite `workspaces` in `spa_routes.json` to the v2 table; add every rule; keep the exact `redirects` that still apply and point them at v2 paths (no redirect may land on another redirect: add a test that resolves every rule/exact target and asserts it is a page path).
- [ ] New shells load EXISTING tab components (`defineTabs("models", { leaderboard: { load: () => import("../ensemble/tabs/Overview") }, … })`) so every page renders at its new path before any merge. Merged tabs load the primary old component for now (Leaderboard → Overview, Members → Members, PSF → Catalog, Fields → Pixels, Targets → Results, Compare → Experiments, Status → realism Overview, Sheet → figures Grid, Live → ops Jobs, History → ops Fasrc(view=history), Steps → ops Fasrc(view=steps), Log → ops Tracking, Backups/Sandboxes → ops Tracking views, Lineage → ops Provenance, Code → ops Git, Storage → settings About, Files → inspect).
- [ ] Inspector registrations (`app/Shell.tsx` side-effect imports) keep working; inspector "open page" links point at v2 paths.
- [ ] Palette entries and page titles follow the new labels.
- [ ] `npm run typecheck && npx vitest run src/app src/workspaces/workspaces.test.ts` and `$PY -m pytest -q tests/test_spa_routes.py` → PASS.

### Task F5: Internal page links

**Files:** every `frontend/src/**` file that builds a PAGE URL (grep for `"/sky/results`, `"/sky/experiments`, `"/sky/catalog-eval`, `"/realism/`, `"/data/`, `"/ops/`, `"/settings/`, `"/inspect`, `"/figures/grid`, `"/figures/results`, and page links under `"/ensemble/` — NOT API calls such as `/ensemble/members.json`, `/ensemble/combiners/…`, `/ensemble/train/preview`), plus backend strings that link to pages (alerts in `helpers/system*.py`, tracking notes, API.md).

- [ ] Replace each with the v2 path (use `pagePath()` from `app/nav.ts` where the code already does).
- [ ] Add a guard test `src/app/noOldPageLinks.test.ts` that scans `src/**/*.ts(x)` (not tests) for the old PAGE prefixes and fails on any hit (allow-list the API prefixes).
- [ ] Full checks: `npm run typecheck && npx eslint . && npx vitest run`, `$PY -m pytest -q`, `ruff`. Browser: every v2 tab renders; ten old URLs redirect (Flask 308 on a full load; SPA on an in-app link).

---

## Content teams (parallel, disjoint folders; each after the foundation)

Each team: implement its tabs to the outlines below (the spec's approved page outlines), then spec review, quality review, fix. Each team owns ONLY its folders; it may import (read-only) from other folders; anything else goes to open issues.

### Team S — Sky (`workspaces/sky/**`, `src/sky/**`; backend `routes/real.py`, `routes/evaluation.py`, `helpers/real_tiles.py`, `helpers/sky_atlas.py`, `helpers/experiments.py` presentation fields)

- **`/sky/atlas`**
  - Absorbs: Absorbs /sky/atlas and the JwstObservations dialog (?obs=1). Default framing is the Q1 deep fields, or the last inspected tile, instead of all-sky MOL. Background becomes a Select (Euclid colour/VIS/Y/J/H + more). Layer groups become 'Real tiles' (NEXUS×Euclid, cached, poster, JWST pairs, comparison tiles), 'Targets' (lens candidates, evaluation galaxies), 'Scene inputs' (PSF stars, PSF clusters, MER noise samples, population cones, archive reference fields; each links to its Synthetic tab) and 'Coverage'. The Gaia fields layer is deleted. Positional acquisition stays: right-click cache a tile here, what covers this point, the go-to box, and the JWST menu (discover, cache NEXUS mosaic, pair downloads, all confirmed). Fill actions leave the layer popovers. The pixel readout is hidden on RGB HiPS.
  - Outline: AtlasToolbar (layers toggle, go-to, quick jumps, Select menu, JWST menu with confirmed discover/cache/download, PNG) → LayersPanel: Background Select, JWST imagery, Pixel overlays, groups Real tiles / Targets / Scene inputs / Coverage (muted counts, links to owning tabs, no fill buttons) → Aladin stage framed on the Q1 deep fields or the last tile, right-click cache tile / what covers this point → SelectionPanel with bulk Compare models / Run production → StatusBar (pixel value hidden on RGB HiPS).
- **`/sky/compare`**
  - Absorbs: Absorbs /sky/experiments. Subtitle: 'models on real tiles, no truth'. It opens on the newest comparison: a viewer with Δm per model vs LR, then a pivot table (models × VIS/Y/J/H for one metric at a time: holes %, holes>100σ, flux R̃, R<0.8), with a dot-per-model band strip. The long CSV stays for export. Then History, with a headline-result column (worst-band holes of production vs mean). 'New comparison' is a drawer with target-set chips (poster, NEXUS selection, lenses, galaxies, pasted refs) and a ModelPicker, Run confirmed. The metric definitions live once, in this tab's info popover, and Targets links to it. This is the only producer of the real-holes numbers that Leaderboard, Combiner and Home read. 'Log to notebook' is here.
  - Outline: Head: comparison picker, scope Select pooled/per tile, info popover with the ONE metric-definitions text, Log to notebook, more → viewer LR / JWST / models with a Δm-per-model footer → metric Select + band dot strip (dot per model per band) → pivot FactsTable models × VIS/Y/J/H for that metric, CSV export of the long form → Gate core weights (per-tile scope) → Run details (collapsed) → History table with a headline-result column → New comparison drawer: set chips, paste refs, ModelPicker, label, Run (confirmed).
- **`/sky/targets`**
  - Absorbs: Absorbs /sky/results and the real half of /sky/catalog-eval, organised by science target instead of by store. Target chips show real labels with counts: NEXUS × JWST 445, Poster galaxy 4, Lens candidates 293, Q1 galaxies 290, Cached tiles 2, with Legacy field 100 under 'more'. Each set leads with one sentence ('Lens candidates: all 293 reconstructions predate the current gate · median flux SR/LR 0.67'), then a flux-ratio strip plot per set, then the table sorted by flux ratio. One state vocabulary everywhere: current / stale / missing, plus 'made by <model>' in words. The Holes and R̃ columns show only for scored rows. The primary action is 'Run production on stale' (confirmed). A Sources menu holds grouped analysis, fetch Q1 lens catalogue and query galaxies (login), all confirmed. The tile card (?inspect=tile:) is reordered: viewer first (footer Δm vs LR, warn-toned when |Δm|>0.1), then one status sentence, then 2 headline metrics, then 'Compare models on this tile', 'Open in Files', and a collapsed Details (grid, Q1 tile, disk). 'Delete model outputs' sits alone in a danger zone behind a typed confirmation.
  - Outline: Toolbar: target-set chips with real labels and counts (NEXUS × JWST, Poster galaxy, Lens candidates, Q1 galaxies, Cached tiles, more: Legacy field), state Segmented current/stale/missing with counts, 'Run production on stale' (primary, confirmed), Sources menu (grouped analysis, fetch lens catalogue, query galaxies, cache tile, all confirmed), metric-definitions link → per-set sentence → flux SR/LR strip plot per group with medians → DataTable sorted by flux ratio (Holes/R̃ only for scored rows, 'made by <model>') → tile card on ?inspect: viewer with Δm footer, status sentence, 2 headline metrics, Compare models on this tile, Open in Files, FITS menu, collapsed Details, danger zone with typed-confirmed Delete outputs.

Also: the "RBF occupancy" field-diagnostics view is deleted (its data no longer exists); the Real-results summary strip and duplicate counts go (spec statistics rework); the eval-store adapter maps real tiles (production_state) and eval objects (current/stale/unknown/failed) onto one current/stale/missing vocabulary with "made by <model>". `FieldDiagnostics.tsx`/`diagnostics.ts` are handed to Team M (do not edit them).

### Team Y — Synthetic (`workspaces/synthetic/**`, moving files out of `workspaces/realism/**` and `workspaces/data/**`; backend `routes/realism.py`, `star_distribution.py`, `galaxy_distributions.py`, `noise.py`, `population_comparison.py`, `cutouts.py`, `psfs.py`, `tng.py`)

- **`/synthetic/fields`**
  - Absorbs: Absorbs /realism/pixels (pixels, detection) and /realism/visual. Title: 'Synthetic vs real LR fields'. Views: look (default; the locked-transfer side-by-side viewers from Visual, with the real-lane count fixed to match 220), stats (the per-band scale-similarity score row as the verdict, then the 7 figures and the median-metrics table) and detection (detections and negative islands per field; completeness in a caption, since the real side has no truth). Sample sizes go on the sample chips ('synthetic LR · 200 fields', 'real LR · 220 fields / 44 pointings'). A stale cache shows the last result with a stale badge and an explicit Measure button, and never auto-runs. The 'Real reference' drawer holds the archive-field sync and archive_field_sample, which were duplicated on Pixels › Inputs and Visual.
  - Outline: VerdictLine from the per-band scale-similarity score row ('VIS overlap 0.93 (0.90–0.95), power syn/real 1.05', or the last result with a stale badge + Measure button) → view Segmented look/stats/detection → look: shared transfer row, two viewers real archive LR | synthetic dirty LR → stats: sample chips with sizes, band chips, 7 figures, median-metrics table → detection: detections and negative islands per field, completeness caption → Real-reference drawer: 220 fields / 44 pointings, sync, archive_field_sample.
- **`/synthetic/galaxies`**
  - Absorbs: Absorbs /realism/galaxies (distributions, relations, joint, model) and /data/tng. Views: distributions (default; keeps the trust boxes, and the generation ceiling folds into the turnover box with its unit), relations, joint (model draws forward-noised so contour widths compare with raw Q1 colours; per-cell n on hover), and templates (the old TNG page: property explorer with SFR=0 galaxies on a floor strip, histogram as the explorer's marginal, galaxy table, template thumbnails from the pulled grid, and one radius-manifest state). The Prior drawer (?prior=1) holds fit and activate, the model laws as a short Details list (the headline is integrated density 152 arcmin⁻²; slopes and scatters to 2 significant figures; SFR coverage as a sentence), and the flux-ratio-variance fit diagnostic moved out of relations. The 'How this is produced' drawer holds the Q1 MER+PHZ query and the re-query of cones (login required; progress only while running), download_tng_skirt, measure_tng_radii, tng_grid, tng_stack and Refresh properties. The figure view moves to Figures › Plates.
  - Outline: Toolbar: view Segmented distributions/relations/joint/templates, Prior drawer, How-produced drawer, download menu → distributions: trust boxes (turnover with units and the ceiling folded in, 5σ limit) above the wide brightness panel, then size, colours and radius shape, each with a caption → relations: radius and FWHM relations with a slope/scatter caption at 2 significant figures → joint: corner plot (forward-noised model draws, n on hover), pair explorer, mag × radius map with a caption → templates: property explorer (SFR=0 floor strip) with the histogram as its marginal, galaxy table, template thumbnails, one radius-manifest state → Prior drawer: VerdictLine '152 galaxies arcmin⁻²', model laws as Details, colour-forest variance diagnostic, Fit/Activate (confirmed) → How-produced drawer: Q1 MER+PHZ query and re-query cones (login, JobProgress only while running), TNG steps.
- **`/synthetic/noise`**
  - Absorbs: Absorbs /realism/noise (realism/tabs/Noise.tsx). It starts with the 3-sentence caption 'How a scene gets its noise'. Then the per-band level histograms: field legend above the first row, subtitle '294 Q1 positions in EDF-N/S/F', and median · p5–p95 in each caption. Then band pairs, with the Pair Segmented in that card's header, the r badge, and the 4×4 table on demand. Then depth steps and seams. A new card compares realised noise, synthetic vs real background σ per band (moved from the Pixels 'background vs robust noise' figure), so this tab is a realism check. 'Level quantiles' is deleted. 'Measured positions' becomes an on-demand table, since the atlas layer is the map. Footer: 'NOISE_MODEL v5 · Q1_R1 · retrieved 2026-09-19 · mer_noise_levels.json'. The drawer holds vis_noise_sample and the MER noise downloader.
  - Outline: 3-sentence caption 'How a scene gets its noise' → Sky noise level per band: field legend on top, subtitle '294 Q1 positions in EDF-N/S/F', 4 histograms with median · p5–p95 captions → Realised noise, synthetic vs real (new): background σ per band, with the VerdictLine for the tab → Band pairs card with the Pair Segmented in its header, r badge, 4×4 table on demand → Depth steps / seams → Measured positions (on demand, link to the atlas layer) → footer 'NOISE_MODEL v5 · Q1_R1 · retrieved 2026-09-19' → How-produced drawer: vis_noise_sample, MER downloader, scene-scale jitter switch.
- **`/synthetic/psf`**
  - Absorbs: Absorbs /data/catalog, /data/cutouts and /data/psfs as one chain: real Euclid stars, then their cutouts, then ePSFs. View catalogue (default) opens with 'Real Euclid Q1 stars the empirical PSFs are built from: 43,401 stars, 17,917 usable (valid in all 4 bands at 511 px)', then filters, the star table and the magnitude histogram. View cutouts shows the viewer with the target star marked and its magnitude kept separate from the whole-cutout magnitude, a gallery with a per-star stretch, and per-band validity as one stacked bar per band. View epsf shows the kernel viewer with FWHM, ePSF vs Gaussian fallback per band, a cluster FWHM map, the warp preview with a tooltip, and 'Used by the last generation run: empirical / Gaussian fallback' from records provenance. The drawer holds euclid_query, euclid_verify_photometry, download_euclid_cutouts, extract_euclid_psf, psf_rotation_pool, 'Pull stars.csv' and 'Sync PSF cluster metadata' (moved from the atlas layer popovers).
  - Outline: Header sentence 'Real Euclid Q1 stars the empirical PSFs are built from: 43,401 stars, 17,917 usable (4 bands, 511 px)' → view Segmented catalogue/cutouts/epsf → catalogue: filter toolbar (field select with counts, coverage, band, VIS RangeSlider), star DataTable, magnitude histogram with a caption explaining the magnitude windows → cutouts: viewer (target marked, 'star VIS 17.80 · whole cutout 16.44 AB'), per-band validity stacked bars, gallery with per-star stretch → epsf: 'Used by the last generation run: empirical/Gaussian' line, per-band kernel viewer with FWHM, FactsTable ePSF vs Gaussian FWHM per band, cluster FWHM map, warp preview → How-produced drawer: euclid_query, euclid_verify_photometry, download_euclid_cutouts, extract_euclid_psf, psf_rotation_pool, Pull stars.csv, Sync cluster metadata (all confirmed).
- **`/synthetic/records`**
  - Absorbs: Absorbs /data/records (data/tabs/Records.tsx) and /realism/pixels?view=census. Split switch (test/validate; train disabled with a tooltip). Viewer with tiers LR, HR, Blurred HR, Clean, each with a tooltip, and HR shown at matched surface brightness. Truth-marker type chips act as legend and filter. Then the truth-source table, then the census: a Σ VIS / brightest-star histogram above the census table (click an outlier to open that record), and a 3-row 'sources per arcmin²: generated · prior · Q1' table (galaxies, stars, lenses). A 'Generate and sync' drawer holds the synthetic_generate step, Sync, and read-only generation knobs linked to System › Config. The SR tier and 'Generate SR' move to Models › Images. Status badges appear only on a problem.
  - Outline: Toolbar: split Segmented with counts (train disabled with a tooltip), truth overlay Off/HR/All + type chips with counts (legend and filter), a problem-only badge, a 'Generate and sync' drawer button → ImageViewer LR/HR/Blurred HR/Clean with tier tooltips, matched HR stretch, footer mags → 'Truth sources of record N' table → Census: Σ VIS and brightest-star histograms (click → record), 'sources per arcmin²' FactsTable (kind · generated · prior · Q1), census DataTable → drawer: synthetic_generate step, Sync popover, read-only generation knobs with an 'edit in Config' link.
- **`/synthetic/stars`**
  - Absorbs: Absorbs /realism/stars?view=density and ?view=prior. The colours and gaia views are deleted. One view: a verdict line, then the VIS density panel (Q1 point sources, Q1 PHZ, model law, generated stars) with the trusted fit window shaded, then the six colour panels as normalised colour PDFs with correct legend names and forward-noised model colours. The Prior drawer holds Fit and Activate, a one-sentence fit sample, and the Gaia colour-field table collapsed as 'inputs', because the Gaia–Euclid colour sample is still what the prior fits colours on. The 'How this is produced' drawer holds the confirmed 'Query stars · MER + PHZ + Gaia' action and its JobProgress. A cue says the training toggle changes only the generated curve.
  - Outline: VerdictLine 'generated 5.03 vs prior 5.08 arcmin⁻² (−1%), trusted window VIS 17–23.5' → legend with sample sizes → wide VIS density panel with the trusted window shaded → 6 colour panels as normalised PDFs (fixed legend names, forward-noised model) → Caption (63.1 deg² · 3,456 Gaia-matched stars in 3 fields) → Prior drawer: fit sample sentence, Fit/Activate (confirmed), Gaia colour-field table collapsed → How-produced drawer: 'Query stars · MER + PHZ + Gaia' (confirmed, login) with query-result Details (≈520k point sources, ≈403k PHZ stars).
- **`/synthetic/status`**
  - Absorbs: Landing tab. Absorbs /realism/overview (realism/tabs/Overview.tsx; readiness from helpers/realism_overview.py). First comes the synthetic_generate gate: 'Ready' or 'Blocked by N', with a confirmed 'Generate validate+test on FASRC' button. Below it are two groups of rows. 'Blocks generation': Galaxies, Stars, Noise, PSF (a new row), TNG radii, Saturation rule, Training catalogue. 'Diagnostic caches': galaxy plots, field statistics. Each row has a state dot, one realism verdict number (e.g. 'stars: generated 5.03 vs prior 5.08 arcmin⁻²'), a 'records built with this?' tick, and a link to its tab. The single copies of 'Validate TNG radii on FASRC' and 'Rebuild field statistics' live here, both confirmed. Fingerprint hashes move to the row inspector. The ineffective training toggle is removed (header.tsx TRAINING_TABS).
  - Outline: Gate line 'Ready to generate' or 'Blocked by N' + confirmed 'Generate validate+test on FASRC' (and the train split) → group 'Blocks generation': rows Galaxies, Stars, Noise, PSF, TNG radii, Saturation rule, Training catalogue. Each row has a dot, a title, one verdict number with a unit (e.g. 'generated 5.03 vs prior 5.08 arcmin⁻²'), a 'records built with this?' tick, a fix button where one exists (Validate TNG radii, Sync training catalogue) and a link to its tab → group 'Diagnostic caches': galaxy plots, field statistics (Rebuild, confirmed) → row inspector with fingerprints.

The "Generate SR over local records" action moves to Models › Images (Team M); Records keeps its SR tier for viewing.

### Team M — Models (`workspaces/models/**`, moving files out of `workspaces/ensemble/**`; `workspaces/sky/results/FieldDiagnostics.tsx` + `diagnostics.ts` (moved to models); backend `routes/ensemble.py`, `helpers/ensemble_viz.py` presentation fields)

- **`/models/:regime/combiner`**
  - Absorbs: Absorbs /ensemble/:mode/combiners. Variants fitted for the current membership show by default; others are behind a 'history' chip, like the backups. The RBF row and 'include RBF' are deleted. Columns: members, mix, held-out, ∫PSNR, and real holes VIS·Y·J·H % with a header link to the Sky › Compare run they came from. Gate share per member, sortable, with 'Open in Members with this selection' for pruning. Actions: Fit variant…, Compare…, and Promote (confirmed). Held-out curves appear only for comparable losses, with unique colours. The compare report card shows only when a report exists.
  - Outline: Action bar: Fit variant…, Compare…, Report select, 'Real holes from' run select (with a link), history chip, Log to notebook → job progress while running → variants DataTable (current membership only by default): Variant, Members, Mix, Held-out, ∫PSNR, Real holes VIS·Y·J·H %, Fitted, Gate-share link, row menu (Inspect, Promote confirmed, Compare with production) → Gate share per member bar list + 'Open in Members with this selection' → Held-out curves (comparable losses only) → compare report (only when present).
- **`/models/:regime/diagnostics`**
  - Absorbs: Absorbs /ensemble/:mode/diagnostics, /sky/results?diag=1 (FieldDiagnostics.tsx) and the /sky/catalog-eval Figures. Every section gets a band switch. Sections: spectrum r(k) and transfer T(k), with x focused on θ<0.5″ and guide labels that no longer collide; coherence as a sorted dot plot; spread, which merges σ vs error, σ vs brightness and calibration, gives the one-sentence answer ('cross-member σ is not an error bar'), shows a 3-row coverage table, and fixes the clipped yDomain; real-field, the legacy real field beside its synthetic twin (r(d), σ vs brightness), noting 14 vs 30 members; and recovery, with SR→HR recovery and the angular power spectrum redrawn as interactive plots. d=axes and RBF occupancy are deleted.
  - Outline: Section Segmented spectrum/transfer/coherence/spread/real-field/recovery + band switch → spectrum and transfer: r(k)/T(k) focused on θ<0.5″ with non-overlapping guides → coherence: sorted horizontal dot plot → spread: one-sentence answer, σ-vs-error and σ-vs-brightness heat maps with PixelTrace, z-pdf with a correct y-domain and coverage FactsTable → real-field: legacy real field vs synthetic twin (r(d), σ vs brightness), caption '14 real vs 30 synthetic members' → recovery: interactive SR→HR recovery and angular power spectrum.
- **`/models/:regime/images`**
  - Absorbs: Absorbs /ensemble/:mode/disagreement, the synthetic groups of /sky/catalog-eval (Syn lens 30, Syn gal 300) and the SR tier plus 'Generate SR over local records' from /data/records. A set chip picks test fields (default) or source-centred stamps. Tiers: LR, SR (production), mean, member stills, disagreement movie, HR, BHR. The member picker exists once, as a side panel (the tab-strip popover is removed, so the nav stops reshaping). A per-set caption gives PSNR vs HR, and the footer shows Δm vs LR.
  - Outline: Set chip test fields / stamps (syn-lens, syn-gal) + Generate SR over local records (confirmed) → ImageViewer LR | SR | HR with tiers mean, member stills, disagreement movie, BHR, footer with Δm vs LR → per-set caption (PSNR vs HR) → side panel member picker (search, loss chips, Top 5 by ∫PSNR, links to the Leaderboard).
- **`/models/:regime/leaderboard`**
  - Absorbs: Absorbs /ensemble/:mode/overview and /ensemble/:mode/knee, plus the Home KPI copies. From the top: one status line ('All current', or each failing staleness check with its confirmed fix: Evaluate N fields, Member PSNR, Knee PSNR). Then the verdict line. Then a 3-row FactsTable (production gate / plain mean / best member) with columns ∫PSNR, ∫VIS/Y/J/H and real holes worst band plus flux R̃, where the real columns carry their Sky › Compare run date and model, or 'no real benchmark for this membership'. Then the knee curves with the integration slider in the toolbar. Then the full leaderboard table, the one ranking that Members, Combiner, Images and Home link to; Test VIS @100 e⁻ and Test 4b are hidden columns. The TIMEOUT card becomes one alert line: '5 stopped short · Continue'.
  - Outline: Regime switch in the workspace header → status line ('All current', or failing checks with confirmed Evaluate / Member PSNR / Knee PSNR) → VerdictLine → Caption → FactsTable gate / plain mean / best member × ∫PSNR, ∫VIS/Y/J/H, real holes worst band + flux R̃ (with run date or 'no real benchmark') → alert line for TIMEOUTs → toolbar: view vs mean/absolute, band, colour, integration RangeSlider, Log to notebook → PSNR-vs-knee curves per band → leaderboard DataTable (Test VIS and Test 4b hidden), row → member or combiner inspector.
- **`/models/:regime/members`**
  - Absorbs: Absorbs /ensemble/:mode/members and /ensemble/:mode/curves. Views: roster (default columns Member, Status, Steps, recipe loss·knee, ∫PSNR, Gate share as a sortable bar; per-band, Test 4b and telemetry hidden), curves (coloured by training knee by default, loss faceted by loss type, gradient norm; wall time moves to Runs), and archived (tombstones + Restore). A visible selection toolbar holds Continue, Fork, Curves and Images, with Archive set apart and typed-confirmed for more than 3 members. A 'Pull from FASRC…' dialog merges the Overview Pull dialog and FASRC Storage 'Pull checkpoints'; a banner appears only when new members are waiting.
  - Outline: Pull-from-FASRC banner (only when new members are waiting) → view Segmented roster/curves/archived → roster: filter + TIMEOUT/multi-knee chips, a visible selection toolbar (Continue, Fork, Curves, Images | Archive set apart), DataTable Member, Status, Steps, recipe, ∫PSNR, Gate share bar → curves: Show segmented, colour default knee, 3 plots (validation PSNR, loss faceted by type, grad norm), 70k guide → archived: tombstone table + Restore (confirmed).
- **`/models/:regime/train`**
  - Absorbs: Absorbs /ensemble/:mode/train. A 'Running batch' strip at the top reads Runs › Live. Then Add / Continue / Fork, Repeat last batch, and Clone. Member rows follow. Run knobs are split into 'Scheduling' and 'Forward model'; the forward-model values mirror System › Config read-only, with an edit link. The regime is an explicit field defaulting to :regime and is repeated in the button label ('Submit 4 STARFULL members to SLURM'). There is a command preview, and submitting needs confirmation.
  - Outline: Running-batch strip (from Runs › Live) → bar: Add/Continue/Fork, Repeat last batch, Clone select, Recipe reset → member rows editor (or Continue picker / Fork source) → Scheduling group (models at once, seed, evaluate every, batch) → Forward model group (read-only values from Config with an edit link) → Resources (CPUs, memory, time) → command preview + Copy → 'Submit N STARFULL members to SLURM' (confirm).

### Team O — Files, Runs, Notebook, System (`workspaces/{files,runs,notebook,system}/**`, moving files out of `workspaces/{inspect,ops,settings}/**`; backend `routes/files.py`, `routes/fasrc.py` presentation fields, `routes/tracking.py`, `routes/git.py`, `routes/system.py`, `routes/provenance.py`)

- **`/files`**
  - Absorbs: Absorbs /inspect. It is reachable from the palette, a top-bar 'Open file' icon and every FITS action, including a new 'Open in Files' on the tile card. The file-name column is wider. The Statistics, Histogram, Sky and Preview cards become one per-frame stats table (rows = shown frames; median, σ(MAD), p99, Σ flux; a non-finite row only when non-finite pixels exist) plus the histogram, whose clip line becomes a tooltip. The WCS becomes one caption line and the file facts one muted line in the file bar. Preview is deleted.
  - Outline: Start: path input + FileBrowser with roots labelled by stage and a wider name column → file bar (crumbs, name, one muted facts line, Download, Track, Show on sky, more) → view bar (HDU select, Image/Plot/Table/Header/Provenance, bin, render) → ImageViewer → per-frame stats FactsTable (median, σ(MAD), p99, Σ flux; non-finite row only if present) + histogram (clip counts in a tooltip) + WCS caption line → HDUs table.
- **`/notebook/backups`**
  - Absorbs: Absorbs /ops/tracking?view=backups and ?view=archive (archived campaigns as a filter): model, FITS and image backups, each with ⏱ time travel.
  - Outline: Filter chips models / FITS / images / archived campaigns with counts → backups table with ⏱ time-travel per row.
- **`/notebook/log`**
  - Absorbs: Absorbs /ops/tracking?view=notebook. The campaign bar holds New campaign, Back up… and Push; Save snapshot is secondary. Append entry, edit whole file, jump to day. Every 'Log to notebook' button (Leaderboard, Combiner, Sky › Compare, the Home 'no entry since' alert) lands here with a prefilled entry.
  - Outline: Campaign bar (active campaign, commit, Back up…, Push, New campaign; Save snapshot secondary) → Append entry card (prefilled when arriving from a 'Log to notebook' button) → rendered notebook with caption '110 entries, 2026-07-02 to 2026-09-21', sort, jump to day.
- **`/notebook/sandboxes`**
  - Absorbs: Absorbs /ops/tracking?view=sandboxes: the running time-travel servers.
  - Outline: Table of running time-travel sandbox servers with open and stop.
- **`/runs/history`**
  - Absorbs: Absorbs /ops/fasrc?view=history and ?view=logs, finished local jobs from /ops/jobs, and /ops/tracking?view=jobs (as a Campaign filter). Logs open as the side panel of the selected run. Clone and Logs are labelled buttons. The GPU column is hidden for CPU steps. The ensemble_train rows show wall time per 1k steps (moved from Curves).
  - Outline: Filters: step, state, campaign, local/SLURM → run ledger DataTable (state, elapsed, CPU, memory, GPU hidden for CPU steps, params) with labelled Clone and Logs buttons → side panel: log viewer (out/err, page, follow, search) and, for ensemble_train, wall time per 1k steps.
- **`/runs/live`**
  - Absorbs: Absorbs /ops/jobs (running), /ops/fasrc?view=live and ?view=queue, and the JobTray target. One list of local and SLURM jobs. Job labels name the members (199–202). SlurmMonitor task cards show 'step 10,650 / 70,000 (15%)' once, plus GPU/CPU %. Ledger facts appear only after completion. The rail badge, JobTray and Home all land here, so none dead-ends. No 'FASRC offline' or 'No job running' flashes before the status loads.
  - Outline: Scope chips all/local/SLURM with counts → DataTable of running and queued jobs (label names members) → side SlurmMonitor: per-task cards with 'step x / 70,000 (15%)' and GPU/CPU %, Cancel (confirmed) → Queue section (fail-stop queue, remove, resume) → ledger facts only after completion. Status loads before any 'offline' message.
- **`/runs/steps`**
  - Absorbs: Absorbs /ops/fasrc?view=steps and /realism/pixels?view=inputs. The FASRC step catalogue is always a visible list, grouped by stage: Reference data, Noise and fields, Generation, Training, Figures. Each step links to the tab whose drawer embeds it; for example ensemble_train goes to /models/starfull/train.
  - Outline: Stage groups Reference data / Noise and fields / Generation / Training / Figures, always a visible list → selected StepCard (params, resources, Queue confirmed, previous runs) → 'Home tab' link on every step to the tab whose drawer embeds it.
- **`/system/appearance`**
  - Absorbs: Absorbs /settings/appearance: theme, accent, density, layout. The Images DefList becomes one line with a link to the Display panel.
  - Outline: Theme card (Light/Dark/System, accent, density, preview) → Layout card (collapse rail, inspector width, reset) → one line 'Images: VIS, absolute asinh, knee 100 e⁻' with Open Display panel.
- **`/system/code`**
  - Absorbs: Absorbs /ops/git, /ops/fasrc?view=git and the server facts of /settings/about. One question: are this laptop, the running server and the FASRC checkout on the same commit? Local changes are shown as words (untracked/modified), with stage/commit and Push as primary only when ahead. The FASRC side has git pull and Update env. Server boot commit, restart-needed and runtime versions sit in Details.
  - Outline: One sentence 'Laptop, server and FASRC are on 3e8b270' (or which one differs) → Local side: changes table (untracked/modified in words), stage/unstage, commit message, Commit, Fetch/Pull, Push primary only when ahead, diff side card → FASRC side: checkout HEAD vs local, git pull, Update env → Details: server boot commit, restart-needed callout, runtime versions.
- **`/system/config`**
  - Absorbs: Absorbs /settings/config and remains the one editor of job_config.json, keeping the 409 conflict flow. Each group header links to the tab where its effect is judged (Synthetic › Records, Synthetic › PSF, Models › Train); those tabs show 'N knobs changed · Edit' back. Star density is read-only, labelled 'set by the active stellar prior (5.08 arcmin⁻²)'. Galaxy density is rounded. The dead Display group is deleted.
  - Outline: Toolbar: filter, group select, 'Changed from default · N' chip, 'Unsaved · N' chip, Discard, Save, reload → grouped cards, each header linking to the tab that judges its effect → per-field default shown and rounded, star density read-only 'set by the active stellar prior' → Set every field to default (confirmed).
- **`/system/connections`**
  - Absorbs: Absorbs /settings/connections: the FASRC session with a 'Connected 16 min ago' caption, SSH settings, Euclid archive login, FASRC-side Euclid credentials and the TNG token. Single-column stacked cards, which fixes the empty third track in settings.css:48.
  - Outline: Single column: FASRC card (caption 'Connected 16 min ago', Test, Disconnect confirmed, SSH settings collapsed, socket in Details) → Euclid archive · this laptop (login, 'used by' chips) → Euclid credentials · FASRC → TNG token · FASRC.
- **`/system/lineage`**
  - Absorbs: Absorbs /ops/provenance. It is a lookup tool: search a product and see its lineage side card. The KPI strip is replaced by one callout ('98% of records carry no model id; verdicts are not meaningful yet'). Verdicts come from the same staleness service as the Home Loop.
  - Outline: Callout '98% of records carry no model id; verdicts are not meaningful yet' → toolbar: search, kind, verdict Segmented with counts (fed by the same staleness service as Home), source select, 'indexed N min ago', Rebuild → records DataTable → lineage side card.
- **`/system/storage`**
  - Absorbs: Absorbs the /settings/about disk card and data-root table, /ops/fasrc?view=storage (remote du as a sortable table, remote browser, Re-link data), and the Catalog eval More actions 'Sync results from FASRC' and 'Drop cached PNGs'. One threshold sets both the badge tone and the Home alert.
  - Outline: Local disk bar with one threshold driving tone and the Home alert, and caption '68 GiB of 427 GiB used is ours' → data-root table (Measure now) → FASRC remote du as a sortable table, remote browser, Re-link data → maintenance: Sync eval results from FASRC and Drop cached PNGs (both confirmed).

### Team H — Home and Figures (`workspaces/home/**`, `workspaces/figures/**`; backend `routes/figures.py`, `helpers/system_alerts*`)

- **`/`** — PageLead (no badge) → VerdictLine for the production model with real holes from the latest Compare run → Caption (members · gate fitted · test fields · knees) → Loop strip: 7 chips Priors / Records / Members / Evaluation / Gate / Real SR / Figures, each with a state dot and one reason, amber chips linking to their fix tab → 'Running now' line (local + SLURM, TIMEOUT alert) → strip of up to 6 cached thumbnails (latest real SR tiles, latest plates) opening the tile card or plate. No KPI tiles and no job launchers. Disk and FASRC warnings appear on the strip only when there is a problem.

- **`/figures/plates`**
  - Absorbs: Absorbs /figures/plates and /realism/galaxies?view=figure. The segments are named after the plate titles: Galaxy population calibration, Galaxy distributions, Stellar population calibration (rebuilt from the density and colour-PDF panels, with the Gaia colour / G_AB panels removed), NEXUS comparison and Synthetic poster scene. The NEXUS model defaults to the production spatial gate, not RBF, and the 'Unknown tiles' message waits until the tile list loads. Each plate has a caption line: 'made with <model> · current/stale · rendered N d ago'. The Rendered PNGs gallery stays collapsed.
  - Outline: Plate Segmented named after the plate titles → per plate: title, 'made with <model> · current/stale · rendered N d ago' caption, dpi, PNG/PDF/SVG, source-tab link → NEXUS: render form (production gate default, tiles, colour, tag, Render confirmed), run browser with a one-line caption, contact sheet → Synthetic poster scene: pulled PNG, Pull latest (confirmed), step in a drawer → Rendered PNGs (collapsed).
- **`/figures/sheet`**
  - Absorbs: Absorbs /figures/grid and /figures/results. The left panel is the saved-crop pool (table/gallery, rename, delete, open source viewer, show on sky, open in Files). The right side has rows = recipes, now with Y and J bands and a fixed swatch legend, a live A4 preview, and PNG/PDF. A hint explains how to add a crop (freeze in any viewer, press S). The caps show only when reached.
  - Outline: Toolbar: template select, save layout, real/synthetic, dpi, PNG/PDF → left: saved-crop pool (table/gallery, find, rename, delete, show on sky, open in Files) with an 'add a crop' hint → rows panel (recipe × band incl. Y and J, reorder) → right: live A4 preview with a correct swatch legend; caps shown only when reached.

---

### Integration (after all teams)

- [ ] Delete the emptied old workspace folders (`realism`, `data`, `ensemble`, `inspect`, `ops`, `settings`) once nothing imports them; update FOUNDATION.md §11 (one subsection per new workspace) and API.md page references.
- [ ] Every old URL in `spa_redirect_cases.json` redirects correctly in the browser (full load and in-app).
- [ ] Walk the spec's walkthroughs (submit a batch, monitor and pull, evaluate→refit→compare→promote, check a model on real bright objects, change a prior and regenerate, browse records, find stale products, produce plates) at 1280×800 and 720×720, light and dark.
- [ ] `npm run typecheck && npx eslint . && npx vitest run`; `$PY -m pytest -q`; `ruff check euclid_polish tests scripts`; `npm run build`; commit and push.

## Self-review

- Spec coverage: grouping → F4; old→new table → F1–F3 + F4; internal links → F5; each workspace's tabs and outlines → Teams S, Y, M, O, H; deletions not done in phase 1 (atlas Gaia layer done; RBF occupancy, Overview run-action bar duplicate, Catalog-eval PSNR column on real rows, Provenance KPI → done in phase 1) → teams; FOUNDATION/API → Integration.
- Placeholders: page outlines come verbatim from the approved spec; the foundation matcher code is complete in Python; the TS port must pass the same case file.
- Consistency: models param stays `mode`; Files path is `/files`; fragment targets use `?section=`.
