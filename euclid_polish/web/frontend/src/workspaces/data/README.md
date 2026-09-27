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
| `common.tsx` | `DataBar` (+ `BarActions`, `Spacer`, `BarGroup`), freshness badges, load states, the job starter |
| `inspectors.tsx` | the inspector cards |
| `data.css` | everything, scoped by `.dt-*` |

## Page layout

- **Toolbar** (`DataBar`): ONE plain row under the tab strip — no card around
  it, never pinned (it scrolls with the page). What the page shows first (the
  split, the current star, the band states), then its state as badges whose
  detail is in the tooltip, then the actions at the right in `<BarActions>`.
  With `compactable`, when the row does not fit with every label it steps
  through compact levels instead of wrapping (so the viewer starts a row
  higher): level 1, the action buttons go icon-only (their `aria-label` /
  `title` / tooltip stay); a page may define more levels, styled by
  `data-compact="<n>"` — Records (`compactable={3}`) shows its status badges
  as their dots at 2 and 3, except the split's worst state, which keeps its
  word (`model.ts worstToneIndex`: bad > warn > info > neutral > good, the
  first on a tie; "No SR" on the validate split). The other words are
  clipped out of sight, not removed, so they stay the badges' text for
  screen readers and head the tooltip. At 3 the truth-source overlay is one
  "Sources: HR" menu button. The bar measures its one-row width at every level (`model.ts
  compactLevelFor`) with the attribute applied and restored, so it does not
  flip-flop. In the labelled form the action buttons are words only. A long
  filter bar (Catalog) wraps.
- **Caption** (`.dt-caption`): what the image shows — Records' truth-source
  overlay (Off / HR / All tiers, a chip per present type with its marker key
  and count) is a group INSIDE the toolbar row; PSFs keeps one caption line
  (the cluster and its live warps). Labels are the kit's UI face as written
  (sentence case; the kit's Segmented, Chip and Badge are no longer
  lowercase mono, so there are no local font overrides).
- **The viewer**, then everything else below it (job progress, tables,
  census, the PSF cluster list, FASRC steps in collapsed sections). The
  viewer fits its first frame row under its own top by itself
  (`viewer/fit.ts` `heightUnderTop`: the tab strip, toolbar, caption and
  banner above it are subtracted, and it refits when they change), so the
  page just places it. Cutouts puts its gallery beside the viewer on a wide
  page (`.dt-vsplit`, a container query at 880 px: the panel is exactly as
  tall as the viewer and its thumbnails scroll inside), below it otherwise.

## The fit under the viewer's top

The page-side fit (`ViewerStage` / `viewerFit.ts`, 2026-09-27) is gone: it
existed only because the viewer sized its frames to the whole stage height,
ignoring the rows above it. The viewer now subtracts its own offset in the
stage (`viewer/README.md` "Fit sizing"), so Records, Cutouts and PSFs render
the viewer directly.

## Per page

- **Records**: ONE toolbar row = split (with record counts), the truth
  sources (below), one Files badge (the four local files in its tip), the SR
  state, the noise-model check, Sync / Generate SR / ⋯ — compacted rather
  than wrapped (above): at 1024 × 768 and 720 × 720 the viewer starts 85 px
  below the stage top (was 110 px, 146 px with the version banner, with the
  bar on two rows). Truth sources: where they are drawn (Off /
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
  size (freshness in the tip; a badge only when stale), Catalogue. The
  `cutouts` collection serves each band in electrons over its stack (the
  cutout's `MAGZERO`, viewer_data `_cutouts_cube`), so the viewer uses the
  console's own absolute e⁻ transfer — no page-side stretch. The
  cached-cutout gallery (one band, 96 files ≈ 48 stars per page; ONE tile per
  star — the cache holds a file per cutout size — showing the navigator's
  size when cached, the other sizes in its tooltip) sits beside it on the light
  table surround: white-on-black thumbnails (the server's gray_r PNG,
  inverted in CSS) at 256 px for a ~128 px cell, the viewer's star outlined
  (`aria-current`), a click shows that star. The download step and the
  archive login are in the collapsed section.
- **PSFs**: toolbar = one badge per band state naming its bands ("VIS:
  empirical", "J, H: not cached"; every band in one state is a sentence:
  "Not cached in any band"; the hint and any sync error in the tip) so the
  labelled actions fit at ~720 px, last sync, Sync ePSFs / Metadata only /
  Clusters on sky.
  Caption = the cluster + live warps. The cluster table beside the viewer
  (# / Stars / VIS (″); a row shows that cluster and opens its card), full
  width below it when there is no room (# / RA / Dec / Stars / VIS (″) … H
  (″); the full names, "VIS FWHM (arcsec)", in `headerText`). Below: the band
  table (Band / State / "ePSF FWHM (″)" / "Gaussian FWHM (″)" / Clusters /
  Kernel / File / Synced; in a narrow pane the kernel, file, synced and
  cluster-count columns drop out in that order — kit `priority`, its column
  menu shows them again — so it fits without scrolling), the extraction
  steps.
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

`model.test.ts` (pure logic, incl. 200k-value ranges and the status wording;
the fit under the viewer's top is tested with the viewer, `viewer/fit.test.ts`),
`model.test.ts` also covers the truth-type toggles, the one-tile-per-star
gallery, the PSF band badges, the toolbar's compact level and the badge that
keeps its word, `data.test.tsx`
(every tab against a mocked backend with a mocked viewer: toolbar content,
the viewer before every table, the caption, the compact overlay menu and
dot badges (the worst one worded), the gallery → viewer (no page-side stretch), the PSF clusters,
the TNG radii never auto-started, a 130k-star catalogue). The `cutouts`
electrons are tested on the backend (`tests/test_viewer_backend.py`), with
a cutout without MAGZERO whose `X-Cube-Unit` (ADU/s) wins over the meta's `e-`.
