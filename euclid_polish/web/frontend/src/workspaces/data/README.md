# Data workspace (`/data/*`)

Records, Catalog, Cutouts, PSFs and TNG (spec §8.4), plus the `star`, `truth`,
`psf` and `tng` inspector cards. Image pages follow the image-first pass
(`docs/superpowers/specs/2026-09-27-image-first-viewer-design.md`).

| File | Holds |
|---|---|
| `index.tsx` | the workspace (`defineTabs`) |
| `register.ts` | the inspector kinds; loaded eagerly by the shell, so it imports only `ids.ts` |
| `ids.ts` | viewer object / inspector ids (`recordObjectId`, `truthId`, `clusterObjectId`, parsers); re-exported by `model.ts` |
| `api.ts` | endpoint URLs, response types, shared resources |
| `model.ts` | pure logic (catalogue decoding + filters, histograms, marker geometry, truth-type toggles, gallery tiles, PSF band badges, status wording, TNG explorer) |
| `viewerFit.ts` | pure: `frameFit` — the frame grid that fits under the viewer's top; `besideWidth` — the viewer width that leaves a side panel room |
| `ViewerStage.tsx` | the image slot: measures the viewer, fits its frames (CSS on `.cv-frames`), optional side panel |
| `common.tsx` | `DataBar` (+ `BarActions`, `Spacer`, `BarGroup`), freshness badges, load states, the job starter |
| `inspectors.tsx` | the inspector cards |
| `data.css` | everything, scoped by `.dt-*` |

## Page layout

- **Toolbar** (`DataBar`): ONE plain row under the tab strip — no card around
  it, never pinned (it scrolls with the page). What the page shows first (the
  split, the current star, the band states), then its state as badges whose
  detail is in the tooltip, then the actions at the right in `<BarActions>`.
  With `compactable`, when the row does not fit with every label the action
  buttons go icon-only (their `aria-label` / `title` / tooltip stay) instead
  of wrapping; the check lifts `data-compact` while it measures, so it does
  not flip-flop. In the labelled form the action buttons are words only.
  A long filter bar (Catalog) wraps.
- **Caption row** (Records, PSFs; `.dt-caption`): at most one line right above
  the viewer that says what the image shows — the record and its truth-source
  overlay (Off / HR / All tiers, a chip per present type with its marker key
  and count), the PSF cluster and its live warps. Segmented controls whose
  labels are words or band names take `className="dt-seg-text"` (the page's
  sans text as written: "All tiers", "VIS", matching the viewer's buttons;
  the kit's Segmented is lowercase mono).
- **The viewer** in a `<ViewerStage>`, then everything else below it (job
  progress, tables, census, galleries when there is no room beside, FASRC
  steps in collapsed sections).

## `ViewerStage` and the fit under the viewer's top

The viewer sizes its square frames to the whole stage height minus its own
bar and readout (`viewer/fit.ts`). The tab strip, toolbar and caption above it
would push a height-limited frame row (one frame — Cutouts, PSFs, blink /
swipe, a single tier — or any big pane) below the fold. `ViewerStage`
measures the table's top in the stage, its chrome (bar + readout, and the
Display dock when it is stacked under the frames) and the width the table
spends beside the frames (the Display dock), and `frameFit` returns the grid
the viewer would lay out over the height left under its top — the same
`fitFrames` choice (auto / one row / grid / stack) over the same full width —
or null when the viewer's own fit already fits (width-limited) or less than
`MIN_CAP_SIDE` (240 px) is left (a viewer far down a page scrolls into view).
The fit is applied to the frame grid only (`--dt-cols` × `--dt-side` on
`.cv-frames`, never in focus mode): **the viewer stays full width**, so its
bar keeps the rows it has and the smaller frames centre on the light table.
(Earlier the viewer itself was narrowed; its bar then wrapped onto more rows
and the frames shrank again — a blink frame came out smaller than the two-up
frames.) Measured at 792 × 720, Records blink: 449 px (was 389, 337 with the
banner); at 1280 × 800: 559 px with a one-row bar (was 529 with two rows).

`aside`: with a side panel, `besideWidth` narrows the viewer to its fitted
frames (+ the dock beside them), never below the bar's two-row labelled
width (its widest row, measured with the icon-only attribute lifted), when
that leaves the panel `asideMin`; else the panel goes below. `asideFill`
makes the panel exactly the viewer's height with its own list scrolling
inside (the Cutouts thumbnails, the PSF cluster table); a render function
receives `{ beside, height }` (the PSF table shows a narrow column set beside
the viewer). It re-measures on resize of the row, the viewer's table and
grid, every ancestor up to the stage (rows above the viewer, the shell's
banner) and the stage, and when the viewer swaps its placeholder, frames or
focus mode; never in focus mode. It measures in `requestAnimationFrame`, so a
hidden browser tab updates on its next frame.

This belongs in the viewer's own fit (subtract the viewer's offset in the
stage); once the viewer does that, `frameFit` finds nothing to change.

## Per page

- **Records**: toolbar = split (with record counts), one Files badge (the four
  local files in its tip), the SR state, the noise-model check, Sync / Generate
  SR / ⋯. Caption = the record + truth sources: where they are drawn (Off /
  HR / All tiers) and one toggle chip per type the record has — every type
  starts shown, a click hides that type's markers and table rows, a second
  click shows them (the URL keeps the hidden types, `hide=star`; the old
  solo list `st=` is ignored). The markers are drawn a little lighter than
  the viewer's default so the faint HR light shows through. Viewer (LR, HR by
  default).
  Below: job progress, the record's sources, the split's census, the
  synthetic_generate step (collapsed).
- **Cutouts**: toolbar = the star in the viewer (field, VIS mag, Details,
  On sky; its position and copy are in the viewer's readout), the navigator
  size (freshness in the tip; a badge only when stale), Catalogue. The viewer
  uses an auto stretch per viewer (`display={{ stretch: "asinh-auto" }}`: the
  cutouts are ADU/s, where the console's absolute e⁻ knee shows black). The
  cached-cutout gallery (one band, 96 files ≈ 48 stars per page; ONE tile per
  star — the cache holds a file per cutout size — showing the navigator's
  size when cached, the other sizes in its tooltip) sits beside it on the light
  table surround: white-on-black thumbnails (the server's gray_r PNG,
  inverted in CSS) at 256 px for a ~128 px cell, the viewer's star outlined
  (`aria-current`), a click shows that star. The download step and the
  archive login are in the collapsed section.
- **PSFs**: toolbar = one badge per band state naming its bands ("VIS:
  empirical", "J, H: not cached", "All bands: not cached"; the hint and any
  sync error in the tip) so the labelled actions fit at ~720 px, last sync,
  Sync ePSFs / Metadata only / Clusters on sky.
  Caption = the cluster + live warps. The cluster table beside the viewer
  (# / Stars / VIS ″; a row shows that cluster and opens its card), full
  width below it when there is no room (# / RA / Dec / Stars / VIS ″ … H ″:
  short headers sized for the kit's uppercase mono header, the full names in
  `headerText`). Below: the band table (in a pane ≤ 1000 px the kernel and
  file columns start hidden, reachable from its column menu, so it fits
  without scrolling), the extraction steps.
- **Catalog**: filters (field — "Outside the deep fields" for the stars
  beyond the three Q1 cones —, cutout coverage, band + status, VIS range),
  then one summary line of words (definitions in tips; the navigator count
  links to Cutouts), the star table (column widths sum to ~714 px so it fits
  a ~720 px pane unclipped; the band dots' column is "Bands"), then the
  magnitude distribution and the
  per-band validity. Ranges use `extent()` (ticks.ts), never
  `Math.min(...values)`, which overflows the stack past ~120k values.
- **TNG**: toolbar = token, counts, property freshness, Refresh properties;
  the explorer + distribution, the galaxy table, the radii check (never
  started by a page visit), the atlas download and the grid / stack jobs.

## Tests

`model.test.ts` (pure logic, incl. 200k-value ranges and the status wording),
`viewerFit.test.ts` (the fit: width-limited, a blink frame never smaller than
the two-up frames, the floor, auto re-choosing columns, fixed layouts; the
beside width and its bar floor), `ViewerStage.test.tsx` (the DOM wiring:
full-width viewer, beside / below, the compact bar, loading, focus),
`model.test.ts` also covers the truth-type toggles, the one-tile-per-star
gallery and the PSF band badges, `data.test.tsx`
(every tab against a mocked backend with a mocked viewer: toolbar content,
the viewer before every table, the caption, the gallery → viewer, the PSF
clusters, the TNG radii never auto-started, a 130k-star catalogue).
