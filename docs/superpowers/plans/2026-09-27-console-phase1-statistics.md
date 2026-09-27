# Console Phase 1 — Statistics Rework and Requested Deletions: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace every stat-card grid (`Stat`, `StatStrip`, `Kpi`) with readable, informative presentations that keep all necessary numbers and drop the useless ones, and delete the views the user called physically meaningless (Realism › Stars *colours* and *gaia*) together with everything that existed only for them.

**Architecture:** Five small kit components in `ui/` (a summary sentence, a facts list, a caption, a collapsed details block, and an approximate-count formatter) become the only way pages present statistics. Each of the 9 workspace files that use cards is rewritten against the page targets below. The deletions remove frontend views, their payload keys, the atlas layer and the plate panels. The regrouping (spec phases 2–6) is NOT part of this plan: pages stay at their current routes.

**Tech Stack:** React 18 + TypeScript + Vite (`euclid_polish/web/frontend`), vitest + Testing Library, Flask + pytest backend, matplotlib publication figures.

**Spec:** `docs/superpowers/specs/2026-09-27-console-regrouping-design.md` (sections *Statistics* and *Deletions*). The user's rule for statistics: *"Make it in a format that is readable and informative. Do not show useless numbers, but show all the necessary information in an accessible and nice way."*

**Standing rules (apply to every task):** sentence case, no ALL-CAPS labels; colours only from `src/theme/tokens.css`, working in light and dark; opening a page never starts a job; nothing sticks over images; Python imports at module top; run commands from `/Users/alarion239/Desktop/EuclidPolish/euclid_polish/web/frontend` (frontend) or the repo root (backend) with absolute `cd`; Python is `/Users/alarion239/miniforge3/envs/EuclidPolishEnv/bin/python`.

---

## File structure

| File | Responsibility |
|---|---|
| `src/format.ts` (modify) | `formatApprox(n)` — ≈3 significant figures with k/M suffix for expected/modelled counts |
| `src/ui/facts.tsx` (create) | `SummaryLine`, `Num`, `FactsList`, `Caption`, `Details` |
| `src/ui/facts.css` (create) | Their styles (tokens only; body font; tabular numerals) |
| `src/ui/index.tsx` (modify) | Export the new components; stop exporting `Stat`, `Kpi` at the end |
| `src/ui/display.tsx`, `src/ui/ui.css` (modify) | Remove `Stat`, `Kpi` and their CSS (last task) |
| `src/workspaces/realism/common.tsx` (modify) | Remove `StatStrip` |
| `src/workspaces/realism/tabs/Stars.tsx`, `realism/stars/model.ts` (modify) | Delete the colours/gaia views; rewrite density and prior statistics |
| `src/workspaces/realism/galaxies/Workflow.tsx` (modify) | Rewrite the galaxy model/query statistics |
| `src/workspaces/realism/tabs/Pixels.tsx`, `realism/tabs/Noise.tsx` (modify) | Rewrite; delete the noise level-quantile tables |
| `src/workspaces/home/Dashboard.tsx` (modify) | Replace the seven KPI tiles |
| `src/workspaces/ensemble/tabs/Overview.tsx`, `ensemble/tabs/Diagnostics.tsx` (modify) | Replace KPI tiles; delete the RBF `d=axes` view |
| `src/workspaces/ops/tabs/Provenance.tsx` (modify) | Replace the KPI strip with one callout |
| `euclid_polish/web/routes/star_distribution.py` + its helper (modify) | Drop the `colors`, `gaia_cmd`, `euclid_projection` payload keys (keep the Gaia–Euclid colour sample the prior fits on) |
| `euclid_polish/web/helpers/sky_atlas.py` (modify) | Remove the `gaia-fields` layer |
| `euclid_polish/eval/publication_figures.py` (or wherever `render_star_population_calibration` lives) (modify) | Remove the Gaia colour and G_AB projection panels |
| `euclid_polish/eval/ensemble_diagnostics.py`, `routes/ensemble.py` (modify) | Remove the RBF combiner-axes diagnostic and unused RBF routes |
| `src/FOUNDATION.md` §9 (modify) | Document the statistics components and rule |

---

### Task 1: `formatApprox` for expected counts

**Files:** Modify `src/format.ts`; Test `src/format.test.ts`

- [ ] **Step 1: Write the failing test** (append to `src/format.test.ts`)

```ts
import { formatApprox } from "./format";

describe("formatApprox", () => {
  it("rounds expected counts to 3 significant figures with a suffix", () => {
    expect(formatApprox(403069.7)).toBe("≈403k");
    expect(formatApprox(519611.8)).toBe("≈520k");
    expect(formatApprox(536780)).toBe("≈537k");
    expect(formatApprox(16_600_000)).toBe("≈16.6M");
    expect(formatApprox(1201.5)).toBe("≈1,200");
    expect(formatApprox(152.4)).toBe("≈152");
  });
  it("keeps small values and handles missing ones", () => {
    expect(formatApprox(5.084)).toBe("≈5.08");
    expect(formatApprox(null)).toBe("—");
    expect(formatApprox(Number.NaN)).toBe("—");
  });
  it("can drop the approximation sign", () => {
    expect(formatApprox(403069.7, { sign: false })).toBe("403k");
  });
});
```

- [ ] **Step 2: Run it and see it fail**

Run: `cd /Users/alarion239/Desktop/EuclidPolish/euclid_polish/web/frontend && npx vitest run src/format.test.ts`
Expected: FAIL — `formatApprox` is not exported.

- [ ] **Step 3: Implement** (add to `src/format.ts`)

```ts
/** An expected or modelled count at ≈3 significant figures ("≈403k",
 *  "≈16.6M", "≈1,200", "≈5.08"). Measured counts use formatCount instead. */
export function formatApprox(v: number | null | undefined, { sign = true }: { sign?: boolean } = {}): string {
  if (v == null || !Number.isFinite(v)) return "—";
  const a = Math.abs(v);
  const pre = sign ? "≈" : "";
  const sig3 = (x: number) => Number(x.toPrecision(3));
  if (a >= 1e6) return `${pre}${sig3(v / 1e6).toLocaleString("en-US")}M`;
  if (a >= 1e4) return `${pre}${sig3(v / 1e3).toLocaleString("en-US")}k`;
  return `${pre}${sig3(v).toLocaleString("en-US")}`;
}
```

- [ ] **Step 4: Run the test and see it pass** — same command, expected PASS.

- [ ] **Step 5: Commit** — `git add src/format.ts src/format.test.ts && git commit -m "Add formatApprox for expected counts"`

---

### Task 2: The statistics components

**Files:** Create `src/ui/facts.tsx`, `src/ui/facts.css`, `src/ui/facts.test.tsx`; Modify `src/ui/index.tsx`

- [ ] **Step 1: Write the failing tests** (`src/ui/facts.test.tsx`)

```tsx
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { Caption, Details, FactsList, Num, SummaryLine } from "./facts";

describe("statistics components", () => {
  it("SummaryLine is one sentence with bold numbers", () => {
    render(<SummaryLine>Generated <Num>5.03</Num> vs prior <Num>5.08</Num> arcmin⁻² (<Num tone="warn">−1%</Num>)</SummaryLine>);
    const p = screen.getByText(/Generated/);
    expect(p.tagName).toBe("P");
    expect(p.querySelectorAll("strong")).toHaveLength(3);
    expect(p.querySelector("[data-tone=warn]")?.textContent).toBe("−1%");
  });
  it("FactsList keeps label, value and unit on one row and skips empty facts", () => {
    render(<FactsList title="Sample" facts={[
      { label: "Q1 area", value: "63.1", unit: "deg²" },
      null,
      { label: "Matched stars", value: "3,456", hint: "Gaia × Euclid in 3 fields" },
    ]} />);
    expect(screen.getByRole("heading", { name: "Sample" })).toBeTruthy();
    const rows = screen.getAllByRole("group");
    expect(rows).toHaveLength(2);
    expect(rows[0].textContent).toBe("Q1 area63.1deg²");
    expect(screen.getByText("Matched stars").closest("[data-hint]")).toBeTruthy();
  });
  it("Caption and Details render quietly", () => {
    render(<><Caption>NOISE_MODEL v5 · Q1_R1</Caption><Details summary="Provenance">sha 1a2b</Details></>);
    expect(screen.getByText("NOISE_MODEL v5 · Q1_R1").className).toContain("ui-caption");
    const d = screen.getByText("Provenance").closest("details")!;
    expect(d.open).toBe(false);
  });
});
```

- [ ] **Step 2: Run and see them fail** — `npx vitest run src/ui/facts.test.tsx` → FAIL (module not found).

- [ ] **Step 3: Implement `src/ui/facts.tsx`**

```tsx
/* Statistics presentation (spec 2026-09-27, "Statistics: readable,
   informative, nothing useless"). A page states its answer in one
   SummaryLine, lists the supporting numbers it needs in a FactsList (label,
   value and unit on one readable row), qualifies figures with a Caption, and
   collapses provenance-only numbers in Details. No tiles. */
import type { ReactNode } from "react";
import { cx } from "./cx";
import { Tooltip } from "./overlays";
import type { Tone } from "./types";
import "./facts.css";

/** A number inside a SummaryLine: bold, tabular, optionally toned. */
export function Num({ children, tone }: { children: ReactNode; tone?: Tone }) {
  return <strong className="ui-num" data-tone={tone && tone !== "neutral" ? tone : undefined}>{children}</strong>;
}

/** The page's answer as one body-size sentence (at most one per page). */
export function SummaryLine({ children, className }: { children: ReactNode; className?: string }) {
  return <p className={cx("ui-summary", className)}>{children}</p>;
}

export type Fact = { label: ReactNode; value: ReactNode; unit?: ReactNode; hint?: ReactNode; tone?: Tone };

/** Necessary supporting numbers, one row each: label left, value + unit right. */
export function FactsList({ title, facts, className }: {
  title?: ReactNode; facts: (Fact | null | false | undefined)[]; className?: string;
}) {
  const rows = facts.filter((f): f is Fact => !!f);
  if (!rows.length) return null;
  return (
    <section className={cx("ui-facts", className)}>
      {title != null && <h3 className="ui-facts__title">{title}</h3>}
      <dl className="ui-facts__list">
        {rows.map((f, i) => {
          const label = <dt className="ui-facts__label" data-hint={f.hint != null || undefined} tabIndex={f.hint != null ? 0 : undefined}>{f.label}</dt>;
          return (
            <div key={i} role="group" className="ui-facts__row" data-tone={f.tone && f.tone !== "neutral" ? f.tone : undefined}>
              {f.hint != null ? <Tooltip content={f.hint}>{label}</Tooltip> : label}
              <dd className="ui-facts__value"><span className="ui-facts__v">{f.value}</span>{f.unit != null && <span className="ui-facts__u">{f.unit}</span>}</dd>
            </div>
          );
        })}
      </dl>
    </section>
  );
}

/** One muted line under the figure it qualifies (area, release, version). */
export function Caption({ children, className }: { children: ReactNode; className?: string }) {
  return <p className={cx("ui-caption", className)}>{children}</p>;
}

/** Provenance-level numbers (fingerprints, bins, bytes), collapsed. */
export function Details({ summary, children, className }: { summary: ReactNode; children: ReactNode; className?: string }) {
  return (
    <details className={cx("ui-details", className)}>
      <summary>{summary}</summary>
      <div className="ui-details__body">{children}</div>
    </details>
  );
}
```

(If `cx`, `Tooltip` or `Tone` live under other module names, import them the way `src/ui/display.tsx` does; match its import lines exactly.)

- [ ] **Step 4: Implement `src/ui/facts.css`**

```css
.ui-summary { margin: 0 0 var(--s3); font: var(--fw-regular) var(--fs-md)/1.5 var(--font-sans); color: var(--text); max-width: 80ch; }
.ui-num { font-weight: var(--fw-semibold); font-variant-numeric: tabular-nums; white-space: nowrap; }
.ui-num[data-tone="warn"] { color: var(--warn); }
.ui-num[data-tone="bad"] { color: var(--bad); }
.ui-num[data-tone="good"] { color: var(--good); }
.ui-facts { display: grid; gap: var(--s1); min-width: 0; }
.ui-facts__title { margin: 0; font: var(--fw-semibold) var(--fs-sm)/1.3 var(--font-sans); color: var(--text-dim); }
.ui-facts__list { display: grid; grid-template-columns: repeat(auto-fill, minmax(16rem, 1fr)); gap: 0 var(--s4); margin: 0; }
.ui-facts__row { display: flex; align-items: baseline; justify-content: space-between; gap: var(--s3); padding: 5px 0; border-bottom: 1px solid var(--border); min-width: 0; }
.ui-facts__label { font: var(--fw-regular) var(--fs-sm)/1.4 var(--font-sans); color: var(--text-dim); min-width: 0; }
.ui-facts__label[data-hint] { text-decoration: underline dotted var(--border-strong); text-underline-offset: 3px; cursor: help; }
.ui-facts__value { margin: 0; display: inline-flex; align-items: baseline; gap: 4px; white-space: nowrap; }
.ui-facts__v { font: var(--fw-medium) var(--fs-sm)/1.4 var(--font-sans); font-variant-numeric: tabular-nums; color: var(--text); }
.ui-facts__u { font-size: var(--fs-xs); color: var(--text-dim); }
.ui-facts__row[data-tone="warn"] .ui-facts__v { color: var(--warn); }
.ui-facts__row[data-tone="bad"] .ui-facts__v { color: var(--bad); }
.ui-caption { margin: var(--s1) 0 0; font-size: var(--fs-xs); line-height: 1.45; color: var(--text-dim); }
.ui-details { font-size: var(--fs-sm); color: var(--text-dim); }
.ui-details > summary { cursor: pointer; width: fit-content; }
.ui-details__body { padding: var(--s2) 0 0; }
```

(Use the token names that exist in `src/theme/tokens.css`; if `--fs-md` or `--fw-regular` do not exist, use the nearest ones the kit already uses for body text.)

- [ ] **Step 5: Export** — in `src/ui/index.tsx` add `export { Caption, Details, FactsList, Num, SummaryLine, type Fact } from "./facts";`

- [ ] **Step 6: Run tests** — `npx vitest run src/ui/facts.test.tsx` → PASS; `npm run typecheck` → clean.

- [ ] **Step 7: Commit** — `git add src/ui/facts.* src/ui/index.tsx && git commit -m "Add the statistics components (summary line, facts list, caption, details)"`

---

### Task 3: Delete the Stars *colours* and *gaia* views (frontend + payload)

**Files:** Modify `src/workspaces/realism/tabs/Stars.tsx`, `src/workspaces/realism/stars/model.ts`, `src/workspaces/realism/api.ts` (payload types), the realism tests; backend `euclid_polish/web/routes/star_distribution.py` and the helper that builds the payload (find with `grep -rn "gaia_cmd\|euclid_projection" euclid_polish`), `tests/test_star_distribution*.py`.

- [ ] **Step 1: Write the failing tests**
  - Frontend (`src/workspaces/realism/realism.test.tsx` or the Stars test file): `STAR_VIEWS.map(v => v.value)` equals `["density", "prior"]`; rendering `/realism/stars?view=colours` shows the density view (unknown views fall back to the default).
  - Backend: `GET /api/star-distribution` payload has none of the keys `colors`, `gaia_cmd`, `euclid_projection`, and still has the colour sample the prior fits on (assert the existing colour-PDF/density keys are present).
- [ ] **Step 2: Run** — `npx vitest run src/workspaces/realism` and `$PY -m pytest tests -q -k star_distribution` → FAIL.
- [ ] **Step 3: Implement**
  - `STAR_VIEWS` becomes `[{ value: "density", label: "Density" }, { value: "prior", label: "Prior" }]`.
  - Delete the `Colours` and `Gaia` components, `GAIA_LAYERS`, the `colours`/`gaia` branches, and the model helpers `COLOR_ORDER`, `correlationSeries`, `PROJECTION_ORDER`, `projectionSeries`, `cmdSeries` (and their tests) unless another module imports them (`grep -rn` first).
  - In the density panel drop the "native Gaia G_AB" and "Gaia shared-slope fit" series and legend items, *unless* the fitted magnitude law uses them (read `densitySeries`/`fitGuides` and the backend fit: the spec says Q1 PHZ counts set the magnitude density and the Gaia sample supplies colours only; keep whatever the fit consumes and state it in the commit message).
  - Backend: stop computing and returning `colors`, `gaia_cmd`, `euclid_projection`; keep the Gaia–Euclid colour sample and its fit inputs.
- [ ] **Step 4: Run** — same commands → PASS; `npm run typecheck` clean.
- [ ] **Step 5: Commit** — `git commit -m "Delete the Stars colours and gaia views"`

---

### Task 4: Delete what existed only for those views

**Files:** `euclid_polish/web/helpers/sky_atlas.py` (the `LayerSpec("gaia-fields"…)` and its builder), the frontend `SkyLink`/`atlasHref` uses of `gaia-fields` (`grep -rn "gaia-fields" src`), the stellar calibration plate (`grep -rn "render_star_population_calibration" euclid_polish`), and their tests.

- [ ] **Step 1: Failing tests** — `GET /api/sky/layers` has no `gaia-fields` layer; the stellar calibration plate renders without the Gaia colour and G_AB projection panels (assert its panel count or titles in the existing figure test; add one if none exists, rendering with a tiny fixture payload into `tmp_path`).
- [ ] **Step 2: Run** — `$PY -m pytest tests -q -k "sky_atlas or publication or star_population"` → FAIL.
- [ ] **Step 3: Implement** — remove the layer spec and builder; remove the frontend links to it; rebuild the plate from the density and colour-PDF panels only (keep its size/typography conventions).
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** — `git commit -m "Remove the Gaia-fields atlas layer and the Gaia plate panels"`

---

### Task 5: Stars page statistics

**Files:** `src/workspaces/realism/tabs/Stars.tsx` (density ~lines 76–86, prior ~222–229 and ~250–257), test file.

Target (density view), top to bottom:
1. `SummaryLine`: "Generated **{syn density}** vs prior **{model density}** arcmin⁻² (**{Δ%}**), trusted window VIS **{lo}–{hi}**" — `Num tone="warn"` when |Δ| > 5%. Densities at 3 significant figures; the window from the fitted guides (`fitGuides`).
2. Legend labels carry the sample sizes: "Q1 PHZ (VIS) · ≈403k", "Q1 point sources (VIS) · ≈520k", "generated test + validation stars · 6,040 in 1,201 arcmin²", "Euclid four-band · 3,456" where that series exists (`formatApprox` for expected counts, `formatCount` for measured).
3. `Caption` under the plots: "Q1 footprint 63.1 deg² · colours from 3,456 Gaia-matched stars in 3 fixed Q1 fields" (area in deg² from arcmin² ÷ 3600, 1 decimal).
4. No other numbers: the Gaia field area, native Gaia count and projection count are deleted with the views.

Target (prior view):
- A `FactsList title="Query result"`: Objects selected ≈537k · Point sources ≈520k · PHZ stars ≈403k · Footprint 63.1 deg².
- A fit sentence (`SummaryLine`): "Colours fitted on **2,398** stars with S/N ≥ 5 in all bands, of **3,456** matched".
- The per-field Gaia table inside `Details summary="Gaia colour fields"`.
- Deleted: VIS bin width, bin count, the probability threshold tile, the fixed-field count tile and the 3,462 variant.

- [ ] **Step 1: Failing test** — render the density view with the existing fixture; expect the summary sentence text, the legend label containing "≈403k", no element with class `ui-stat`; render the prior view and expect a `FactsList` with the four rows and a closed `<details>` titled "Gaia colour fields".
- [ ] **Step 2: Run** → FAIL. **Step 3:** implement with the kit from Task 2. **Step 4:** Run → PASS.
- [ ] **Step 5: Commit** — `git commit -m "Rewrite the Stars statistics as a summary line, legend counts and facts"`

---

### Task 6: Galaxies statistics (`realism/galaxies/Workflow.tsx`, 19 uses)

Targets:
- Query block (~69–78): `Caption` "v{n} · {cones} cones · {area} arcmin² · {rows} rows · VIS {lo}–{hi}". Checkpoints, brackets and passes appear only inside `JobProgress` while a query runs. PHZ weights and bin width are deleted.
- Model block (~131–142): `SummaryLine` "**{density}** galaxies arcmin⁻² at scene depth"; `FactsList title="Model"`: Radius slope {−0.15} dex/mag · Radius scatter {0.23} dex · Colour forest {83,583} rows · SFR known for {33}% of the weight · Rₑ resolved for {97.5}%. The plateau stays as a guide on the brightness plot (not a fact). The "brightness" string tile and "50 trees" are deleted.
- `SourceLedger` (~33–42): three sentences, each naming what its area covers ("Q1 query: 140,085 rows over 1,885 arcmin² (24 cones)").
- Marginals trust boxes: keep two; fold the generation ceiling into the turnover box as "generator plateau {30} arcmin⁻² mag⁻¹ = Q1 peak" (with its unit).

- [ ] Steps as in Task 5 (failing test on fixture text and no `ui-stat`, implement, pass, commit "Rewrite the galaxy statistics").

---

### Task 7: Pixels and Noise statistics

Pixels (`realism/tabs/Pixels.tsx`):
- Promote the score row (~170–181) to a `SummaryLine`: "VIS overlap **{0.93}** ({0.90}–{0.95}), power syn/real **{1.05}**".
- `FieldCensus` (~339–347): sample sizes on the sample chips ("synthetic LR · 200 fields", "real Euclid LR · 220 fields / 44 pointings"); geometry as a `Caption` "fields 255 × 255 px at 0.1″ (0.18 arcmin²)" — check the real field size in the payload and print what it says; "built" moves into the cache badge tooltip.
- Census tiles (~304–310): a comparison `Table` kind × generated · prior · Q1 (galaxies, stars, lenses) — not gated on the pixel cache.

Noise (`realism/tabs/Noise.tsx`):
- Delete the `StatStrip` (~308–314) and both level-quantile tables (~221–254).
- Histogram card subtitle: "{294} Q1 positions in EDF-N/S/F"; keep the per-panel "median · p5–p95" captions; footer `Caption`: "NOISE_MODEL v{5} · {Q1_R1} · retrieved {2026-09-19}"; coverage gaps ("{50} of {344} tiles without coverage") go in the card's info popover.

- [ ] Steps as in Task 5, one commit per file ("Rewrite the pixel statistics", "Rewrite the noise statistics").

---

### Task 8: Home tiles (`home/Dashboard.tsx`, 7 `Kpi`)

Target (phase 1 — the loop strip arrives in phase 6):
- `SummaryLine`: "Production gate **{60.97}** dB integrated PSNR, **{+1.01}** dB over the best member (#{196}) and **{+1.86}** dB over the plain mean · {30} members, evaluated {2 d ago}" — read from the same payload the tiles read today.
- A "Running now" line when jobs run ("{members 199–202} on FASRC · {15}%"), linking to the jobs page; nothing when idle.
- FASRC, server and disk appear only as `Callout tone="warn"` lines when broken (disconnected, server behind/changed code, disk below the existing threshold); no tiles when healthy.
- The "2 alerts" badge stays only if the alerts list is below it on the page; otherwise delete.

- [ ] Steps as in Task 5 (failing test: summary text from the fixture, no `ui-kpi`, the FASRC callout only when `ssh_connected: false`), commit "Replace the Home tiles with a summary line and problem-only notes".

---

### Task 9: Ensemble Overview and Diagnostics; RBF leftovers

Overview (`ensemble/tabs/Overview.tsx` ~66–93):
- One status line (all current, or each failing staleness check with its confirmed fix button — reuse the existing check list).
- `SummaryLine` as on Home for this regime.
- A comparison `Table` with rows production gate / plain mean / best member and columns ∫PSNR, ∫VIS, ∫Y, ∫J, ∫H (dB, 2 decimals), so "best member" is tied to its metric.
- `Caption`: "{30} members · evaluated {2 d ago}". TIMEOUT badges become one alert line ("{5} members stopped at the time limit: 178, 179, …").

Diagnostics (`ensemble/tabs/Diagnostics.tsx`):
- Calibration (~301–307): sentence "Cross-member σ is not an error bar: bright-field RMSE ≈ {10}× σ" + a 3-row `Table` |z| < 1 / 2 / 3 × observed vs Gaussian (68.3 / 95.4 / 99.7 %).
- Delete `d=axes` ("Combiner axes") and the RBF `MODEL_LABEL` entries (~32–35); backend `euclid_polish/eval/ensemble_diagnostics.py:226-275` and its route; the RBF "stale kind" row in Combiners, the "include the RBF" compare option, and the unused routes `/ensemble/combiner/fit` and `/ensemble/combiner.json` (confirm with `grep -rn` in `src/` that nothing calls them).

- [ ] Steps as in Task 5 (frontend + `$PY -m pytest tests -q -k "ensemble_diagnostics or combiner"`), commits "Rewrite the ensemble overview statistics" and "Remove the RBF combiner-axes diagnostic and routes".

---

### Task 10: Provenance strip (`ops/tabs/Provenance.tsx` ~85–93, ~116)

- Replace the five tiles with one `Callout tone="info"`: "{98}% of records carry no model id, so current/stale verdicts are not meaningful yet" (computed from the counts); verdict counts only on the verdict `Segmented` options; one line "showing {1,000} of {11,345}".
- [ ] Steps as in Task 5; commit "Replace the provenance tiles with one callout".

---

### Task 11: Remove the card components

**Files:** `src/ui/display.tsx`, `src/ui/ui.css`, `src/ui/index.tsx`, `src/workspaces/realism/common.tsx` (`StatStrip`), `src/workspaces/realism/realism.css` (`.rl-stats`), `src/workspaces/ensemble/ensemble.css` (`.ens-kpis`), `src/theme/chrome.test.ts` (or a new `src/ui/noCards.test.ts`).

- [ ] **Step 1: Failing guard test** (`src/ui/noCards.test.ts`)

```ts
import { readFileSync, readdirSync, statSync } from "node:fs";
import { join } from "node:path";
import { expect, it } from "vitest";

const SRC = join(__dirname, "..");
function files(dir: string): string[] {
  return readdirSync(dir).flatMap((n) => {
    const p = join(dir, n);
    return statSync(p).isDirectory() ? files(p) : /\.(tsx?|css)$/.test(n) ? [p] : [];
  });
}
it("no page presents statistics as cards", () => {
  const offenders = files(SRC).filter((p) => !p.endsWith("noCards.test.ts"))
    .filter((p) => /<(Stat|StatStrip|Kpi)[\s>]|ui-stat|ui-kpi|rl-stats|ens-kpis/.test(readFileSync(p, "utf8")));
  expect(offenders).toEqual([]);
});
```

- [ ] **Step 2: Run** → FAIL while any use remains. **Step 3:** delete `Stat`, `Kpi`, `StatStrip`, their CSS and exports. **Step 4:** `npm run typecheck && npx eslint . && npx vitest run` → all green.
- [ ] **Step 5: Commit** — `git commit -m "Remove the stat-card components"`

---

### Task 12: Document, verify, ship

- [ ] Update `src/FOUNDATION.md` §9 with the statistics rule (the 10 points in the spec's *Statistics* section, adjusted: necessary numbers visible in a FactsList; Details only for provenance) and the component API.
- [ ] `cd /Users/alarion239/Desktop/EuclidPolish/euclid_polish/web/frontend && npm run typecheck && npx eslint . && npx vitest run` — all pass.
- [ ] `cd /Users/alarion239/Desktop/EuclidPolish && $PY -m pytest -q && $PY -m ruff check euclid_polish tests scripts` — all pass.
- [ ] Browser check (own Vite, own tab) at 1280×800 and 720×720, light and dark: /realism/stars (density, prior; ?view=colours and ?view=gaia fall back to density), /realism/galaxies, /realism/pixels, /realism/noise, /, /ensemble/starfull/overview, /ensemble/starfull/diagnostics, /ops/provenance. No unit wraps onto its own line; no number with false decimals.
- [ ] `npm run build` (writes the committed `static/dist`), then commit and push: "Present statistics as readable sentences, facts and tables".

---

## Self-review

- Spec coverage: the Statistics section (rule + components) → Tasks 1, 2, 11, 12; the user-requested deletions (Stars views, Gaia layer, plate panels) → Tasks 3, 4; the RBF leftovers and d=axes → Task 9; page reworks for every file that uses cards today → Tasks 5–10. Page reworks for pages WITHOUT cards (Sky results strip, Catalog eval, Cutouts badge, TNG, Records badges, Inspect cards, Runs badges, About) move with their workspaces in phases 2–6.
- Placeholders: page-specific values in braces are read from the payloads the pages already fetch; each is named with its current source line.
- Consistency: components are `SummaryLine`, `Num`, `FactsList` (`Fact`), `Caption`, `Details`, and `formatApprox` everywhere.
