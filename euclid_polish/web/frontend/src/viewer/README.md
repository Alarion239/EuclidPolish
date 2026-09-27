# Viewer engine v2 (`src/viewer/`)

The image viewer of the console: raw Float32 cubes from `/viewer/cube/*`
rendered in the browser, with pan/zoom, a pixel readout with RA/Dec, residual
tiers, blink and swipe, histograms, profiles, the magnifier lens, the
disagreement movie and PNG / publication-figure / video export. It replaces
the pre-rework `static/cutout_viewer.js` (deleted; last version at commit
`0ad56d9`). Specs: `docs/superpowers/specs/2026-09-25-webui-rework-design.md` §6,
and the image-first pass `docs/superpowers/specs/2026-09-27-image-first-viewer-design.md`
(the look, the bar, fit sizing, focus mode).

```tsx
import { ImageViewer, type ViewerApi } from "../../viewer";

<ImageViewer collection="nexus-field" params={{ field }} tiers={["lr", "sr", "jwst"]}
  urlKey="nexus" toolbar="full" onReady={(api) => (ref.current = api)}
  onState={(s) => setIndex(s.index)} />
```

## Props

| Prop | Meaning |
|---|---|
| `collection` | `/viewer/meta/<collection>` (C6): `sky`, `cutouts`, `evaluation`, `ensemble`, `archive-fields`, `real-field`, `jwst-euclid`, `nexus-field`, `psfs`, … |
| `params` | Collection query params (`subset`, `mode`, `field`, …). A change remounts the engine. |
| `tiers` | Initial tiers (default: `meta.default_tier`). |
| `initialIndex` / `initialId` | Initial object (a meta `id` wins; unknown ids fall back to the server's `?id=` lookup). |
| `urlKey` | URL-state prefix `v.<urlKey>.*` (off when absent). |
| `toolbar` | The control bar: `"full"` (default); `"compact"` (no tools or export menus, and a basic Display dock with knee + brightness only: tiers, bands, Display, compare, a lens toggle, zoom, navigation, layout, focus); `"none"` (no bar, or with `nav` a bar of navigation, export and focus only). |
| `nav` | Navigation (◀ position / count ▶, run-through) in the bar (default `true`). The export menu is in every full bar, with or without `nav`. |
| `display` | Per-viewer display override (wins over the Display panel). |
| `onState(s)` | Every state change: the old `getState()` keys (`index, tier, tiers, color, layout, knee, gain, transfers, params, selection`) + `id, view, tool, compare`. |
| `onReady(api)` | The `ViewerApi` once mounted, `null` on teardown. |

## `ViewerApi`

`goTo(i)`, `goToId(id)`, `setTiers(keys)`, `setView({color, knee, gain})`
(per-viewer override of every transfer group; remembered before the meta
loads), `setParams(patch)` (refresh the visible cubes in place, e.g. the PSF
warp seed), `setMorphMembers(csv | null)`, `getIndex()`, `isReady()`,
`getState()`, `exportFigure()`, `savePng()`, `saveCropToResults()`, `reload()`,
`zoomTo(ra, dec, fovArcsec)`, `resetView()`, `getReadout()`, `destroy()`.

## Look: one light table

The viewer is ONE neutral dark block in both themes (`.cv-table`): the
control bar, the frames edge to edge with 2 px gaps (no per-frame border,
radius or shadow), and a one-line readout of reserved height (28 px, no
layout jump). The surround is neutral (`#161719`, untinted: colour judgement
on astronomy images needs it); the kit controls on it read a scoped dark
palette (AA: text ≥ 12.8, dim ≥ 7.2, faint ≥ 5.1, accent ≥ 5.8 on every
table surface; the user's accent preset in its dark-theme value). The
popovers the bar opens are portalled and follow the app theme; the Display
dock is part of the table.

```
┌ what ─ LR HR BHR ▾ │ VIS Y J H Lupton Temp ┐ ┌ how ─ ⚙ Display │ ▭▭ ◉ ⇹ │ ⌖ ·· − + ▢ │ ◀ 12 / 100 ▶ ▷ │ ▦ ⬇ ⛶ ┐
├─ LR ─────────────────────┬─ HR ─────────────────────┬─ Display (dock, when open) ─┐   frames: the tier's name; on hover the cube label + magnitude
│                          │                          │ knee ──●── 100 e⁻   ✕      │
└──────────────────────────┴──────────────────────────┴────────────────────────────┘
  x 204 y 346  17h53m30.9s +65°05′55″  VIS  LR 76.9 e⁻  HR 0.23 e⁻            record 12
```

- **Bar** (`Bar.tsx`, `BarMenus.tsx`, pure model `barModel.ts`). ONE row
  when it fits; else two — what is shown (tier chips + the tier menu, the
  band / colour choice) above how it is shown (Display, compare, tools ▾,
  zoom, navigation, layout ▾, export ▾, focus). `barLayout` decides from the
  measured group widths: one row with the Display text, else one row
  icon-only, else two rows (icon-only when the second row needs it); a row
  still too wide even icon-only (a ~300 px viewer beside the inspector, a
  phone) WRAPS onto another line — no control is ever cut off or scrolled
  out of sight (a barModel test sweeps the widths). Measured: 37 px for one
  row, 67 px for two. The result lives in the viewer store (`bar`), so the
  frame grid refits in the same layout pass. Before the meta arrives the
  bar reserves the rows it had last time for the collection
  (`localStorage` `euclid-polish.viewer.bar-rows`, `reservedBarRows`), else
  two rows below `ONE_ROW_MIN_WIDTH` (760 px), so the frames do not jump.
- **Navigation**: ◀ position / count ▶ and the run-through. The position is
  1-based over the object count ("1 / 100"; a typed number is 1-based,
  `navPosition` / `parsePosition`); labels and URLs keep the 0-based index,
  which the tooltip names.
- **Tier chips**: every tier that is not `hidden` (at most `MAX_TIER_CHIPS`,
  plus any selected one), short names (`shortTierLabel`: "SR · production
  gate" → "SR"; the full label in the tooltip / accessible name). The tier
  menu lists every tier (hidden member tiers under "More tiers") with its
  extras on its entry: the BHR target FWHM slider on BHR, the JWST band on
  JWST, the movie amplitude / speed on the movie.
- **Bands**: the collection's `band_names` (NISP shown as Y J H), then
  Lupton and Temp, with their Q–Y keys; none for a log collection (PSFs) or
  when every shown tier is a single plane (a native one-filter image).
- **Display dock** (`DisplayDock`; the bar's Display toggle, Esc or ✕
  closes it): docked BESIDE the frames on the light table (272 px; 244 px
  basic), so the image stays in sight while the stretch changes — the
  frames refit to the width left (the grid measures its own width). In a
  viewer narrower than 620 px (a container query on `.cv-table`) the dock
  goes UNDER the frames (≤ 240 px, scrolling) and the fit subtracts its
  height. It scrolls inside the frames' height and never makes the table
  taller. Top first: "Follow the Display panel" (the link), stretch; per
  transfer group knee / brightness / black point, each in the group's
  display unit (`groupUnit`: a JWST group reads MJy/sr = transfer ÷
  `X-Cube-Display-Scale`; Euclid reads e⁻) with a typed value beside each
  slider; then colormap, residual colormap, invert, NaN colour, and the
  histogram behind a "Histogram and cuts" disclosure. `basic` (compact
  bars): the link, knee and brightness only. The knee slider is log over
  `KNEE_SLIDER_RANGE` = 0.1–10⁴ (the research knee grid); the defaults stay
  absolute asinh at knee 100 e⁻. Keys typed in the dock stay with its
  controls (a slider's arrows do not change the object); Esc closes it.
- **Tools popover**: pan / magnifier lens, the profile panel (also in focus mode), residual tiers
  (A − B, log₂ A/B, (A − B)/σ), the run-through and blink intervals.
- **Export menu**: PNG, publication figure, video, save the crop to results (S).
- **Frame labels**: the tier's short name; hovering a frame shows the cube
  label + magnitude instead (also its accessible name); the readout carries
  the magnitudes while idle. Band names read as the chips do everywhere
  (frame label, readout, histogram: "Y", not "Y_E"; `bandLabel`).
- **Readout**: hovering — x y, RA Dec (copy), the shown band, every visible
  tier's value with its unit; idle — the object's position, each tier's
  magnitude, the zoom and field; at the right the save status or the
  object's label. The full per-band values are in the line's tooltip and in
  `getReadout()`.

### Fit sizing (`fit.ts`)

Every frame is a square: side = min(width per column, height per row), where
the height is the stage viewport (the nearest scrolling ancestor, i.e.
`main.stage`, else the window; the focus-mode surround in focus mode) minus
the viewer's own bar and readout and an 8 px margin (`availableHeight`) —
and the Display dock when it sits under the frames, the profile panel in
focus mode. The
layout (the layout menu, persisted in `localStorage`
`euclid-polish.viewer.layout`; the old `…cutout-viewer.layout` "two-rows"
migrates to Grid):

| Layout | Columns |
|---|---|
| Auto (default) | the count that gives the largest side (fewer columns only when ≥ 1 px larger) |
| One row | every frame side by side |
| Grid | ⌈√n⌉ |
| Stack | one |

The height may shrink a side to `MIN_FRAME_SIDE` (160 px; a very short
window scrolls), the width never. Blink / swipe are one frame. The
publication figure follows the on-screen rows (`figureLayout`), and a saved
crop's `display.layout` keeps the old two values ("one-row" | "two-rows").

### Focus mode

⛶ or `F`: the viewer covers the stage below the app top bar (and the
inspector) on the neutral surround, the frames re-fit to it, and it keeps
the keyboard while the pointer is elsewhere; `F`, `Esc` or the button
return. "Full screen" in the layout menu enters focus mode and asks the
browser for full screen (on the document, so the portalled menus stay
visible); leaving full screen leaves focus mode. The profile panel stays
under the table on the surround (the frames fit above it; it takes at most
40 % of the height) and the Display dock beside the frames. Esc unfreezes
the lens / clears the profile first, then leaves focus mode, then closes the
dock. The page keeps the viewer's
height while it is lifted out (no jump behind it).

## URL state (`urlKey`)

| Param | Holds |
|---|---|
| `v.<k>.id` | the object id (`meta.objects[i].id`) |
| `v.<k>.i` | the index, only when the object has no id |
| `v.<k>.t` | tiers (comma list; absent = the default tier) |
| `v.<k>.r` | residual tiers (`res:<op>:<a>:<b>`) |
| `v.<k>.z` | the view: `u,v,<side ″>` or `u,v,r<relative side>`, `u,v` on the first frame's grid |
| `v.<k>.c` | the colour when this viewer overrides the Display panel |

Writes are replace-mode and debounced; every mounted viewer's writes are
flushed in one tick (`useUrlState` coalesces a tick; separate ticks could
overwrite each other). The default state is not written: visiting a page
leaves its URL alone. `id` / `i` appear once the object differs from the
mount-time one (`initialIndex` / `initialId`), `t` once the tiers differ
from the mount-time `tiers` prop (else `meta.default_tier`), `c` once the
colour override differs from the one in place when the meta arrived (the
`display` prop or a page's `setView` from `onReady`); a key the URL already
carried stays written.

## Display binding (C7)

Effective settings = `mergeDisplay(useDisplay, viewer override)`:

- **Linked** (the Display panel's "Link all viewers" and the viewer's link
  toggle): a toolbar, keyboard or histogram edit writes the Display panel, so
  every linked viewer follows.
- **Overridden fields** (`setView`, the `display` prop, `v.<k>.c`) stay
  per-viewer; edits of those fields stay per-viewer.
- **Unlinking** a viewer copies the current settings into its override.
- **Behaviour change from the old engine:** the old engine's colour was per
  viewer. Now, while a viewer is linked (the default), its Q–Y colour keys and
  colour chips set the Display panel's colour, so every linked viewer on the
  page (and on later pages) follows — the C7 "Link all viewers" semantics.
  Unlink the viewer (toolbar link toggle) or turn linking off in the Display
  panel for per-viewer colours. The JWST "temperature" band chip is the
  exception: it is a choice about that viewer's JWST frame, so its Temp colour
  is a per-viewer override, dropped again when another band is chosen.

Transfer groups: `meta.transfer_groups` ∩ {euclid, jwst} each get a knee /
brightness pair; otherwise every cube uses `default`. The locked default is
absolute asinh with knee = K0 = `meta.color.default_asinh` and white at 30·K0
e⁻ (`color.ts`, bit-identical to the old engine). Opt-in paths live in
`render.ts`: stretches `linear | sqrt | log` (same black/white anchors),
`asinh-auto | zscale` (limits from the frame), black point, colormaps (gray,
viridis, magma, inferno, cividis, RdBu), invert. NaN pixels always take the
Display panel's NaN colour.

## Interaction

| Input | Action |
|---|---|
| wheel | zoom about the cursor when the viewer is focused or ⌘/Ctrl is held (`display.wheel`: `zoom-when-focused` default, `always-zoom`, `scroll`); otherwise the page scrolls. With the lens active: lens zoom. Horizontal wheel: brightness |
| drag (zoomed) / two-finger pinch | pan / zoom (pointer events) |
| double-click, `0` | fit (whole image) |
| `+` `−` | zoom |
| lens tool, `L`, or hold Alt | magnifier lens; click freezes the matched crop, click again unfreezes |
| shift-drag | line profile |
| click (profile panel open) or shift-click | radial profile |
| `Q W E R T Y` | VIS, Y, J, H, Lupton, Temp (old keys) |
| `←` `→`, Space | previous / next, run through |
| `S` (or Shift+S) | save the frozen crop to results (kept while frozen) |
| `B` | blink |
| `F` | focus mode (the images fill the page); `F` or Esc returns |
| Esc | unfreeze, clear the profile; else leave focus mode; else close the Display dock |

Keys answer only in the most recently hovered / focused viewer (several can
be mounted); a viewer in focus mode keeps them. A key typed right after a
plain `g` is left to the shell's `g <x>` navigation (`sequencePending`), and
Space / Enter on a focused bar control press that control. As in the old engine, Shift+letter acts as the letter (Shift+E =
J, Shift+S = save), except for a combo the shell owns (Shift+D/J/T) or a page
has bound with `useShortcut` (the viewer's `document` listener runs first,
so it checks the shortcut registry and leaves those alone). The keys are
listed in the ? sheet.

## How the features work

- **Tiers and frames.** Each selected tier is a `Frame`: its cube is rendered
  at natural resolution into an offscreen canvas, then drawn (no smoothing)
  into the visible canvas through the shared view (`selection.frameLayout`:
  the whole image contained, or the view's square crop).
- **Pan/zoom and the lens share one geometry** (`selection.ts`): a centre on
  the tier it was made on (`sourceTier`, normalised) plus a side in arcsec,
  resolved per tier by its pixel scale, so LR 0.1″, SR 0.05″ and JWST 0.03″
  frames show the same sky. The centre reaches every other tier through both
  tiers' WCS (`controller.selectionOn` / `cropOf` / `layoutOf`, the same
  mapping as the readout), so footprints that differ slightly (NEXUS: up to
  ~2.4 JWST px) still line up; without a WCS the normalised centre is shared.
- **Pointer → image**: frame CSS coordinates are measured from the frame's
  padding box (`selection.contentBoxOrigin`: the border box moved in by the
  frame's border, `clientLeft`/`clientTop` — 0 on the light table, 1 px in
  the pointer tests), where the canvas and overlays are drawn and
  `clientWidth` measures; the lens popups are placed from the same origin.
- **Readout.** The pointer's continuous image position on the hovered tier →
  RA/Dec through its `X-Cube-WCS` (`wcs.ts`, TAN/SIN, CD or PC+CDELT,
  checked against astropy) → every other tier's pixel through its own WCS
  (or by normalised position without one). Values are native units
  (`X-Cube-Unit`, else the meta tier `unit`), crosshairs on every frame.
- **Residual tiers** (`residual.ts`): `A − B`, `log₂(A/B)`, `(A − B)/σ` for two
  loaded tiers; a coarser tier is upsampled by an integer factor onto the finer
  grid, ÷f² (flux per pixel conserved); σ is the `std` tier when present, else
  the robust σ of the difference. Channels pair by band name (`X-Cube-Bands`;
  by position only without names); tiers with no common band or different
  units (unit from `X-Cube-Unit`, else the meta tier) are refused with the
  reason shown in the frame (`residualMismatch`). Values are native, so the
  display scale plays no part. Shown with the residual colormap: `A − B`
  on asinh with knee = its σ(MAD) and white at its 99.5th |value| percentile,
  ratios ±2 (log₂), significance ±5σ. The image gain does not apply.
- **Blink / swipe**: every frame stays mounted and stacked; blink cycles the
  visible one, swipe clips the second over the first.
- **Histogram** (`HistogramPanel`, in the Display dock behind its disclosure): the visible
  region of one tier/band, native units, log counts. Its black/white handles edit the frame's transfer
  group (black point; gain so that white = black + 30·K0/gain) — only for the
  absolute stretches.
- **Profiles** (`ProfilePanel`): line / radial profiles on every visible
  tier, mapped through the sky; distances in arcsec when every tier has a
  pixel scale; optional per-arcsec² normalisation. Tiers in different units
  get separate plots (one y-axis per unit, labelled band [unit]). A line
  released outside the frame ends on the image edge.
- **Movie** (`movie.ts`): `morph_base_tier` (default `sr`) + Σ amp·sin·PC over
  48 pre-prepared frames; each tick re-runs only the transfer; neighbours are
  cached in the background inside a 1.4 GB LRU.
- **Magnitudes**: whole-cube AB magnitude of the shown band for e⁻ tiers
  (`zeropoint_ab_e_total − 2.5 log₁₀ Σ`), ± (2.5/ln 10)·Σσ/ΣSR on SR when a
  `std` tier exists.
- **Exports** (`export.ts`): PNG of the frames as shown; the publication
  plate (the frozen crop, else the current pan/zoom view, else the whole
  image → matched crops) with a unit-aware heat bar (a JWST panel reads
  MJy/sr, knee and ticks ÷ display scale; residual panels get a diverging bar
  in their own scale). The bar follows the frame's display
  (`controller.heatbarInfo`): ticks placed by its stretch (asinh with the
  black point; linear / sqrt / log from black to black + 30·K0/gain;
  asinh-auto / zscale from the frame's own limits, `render.frameAutoStats`),
  the stretch named in the text, and the gradient drawn from its colormap
  (reversed when inverted; colour composites keep a luminance bar); webm video.
- **Transport** (`cube.ts`): one shared byte-bounded LRU (400 MB) keyed by
  collection + tier + index + params, in-flight dedupe, per-consumer abort.
  `/viewer/meta` goes through the app's query cache (`api/query.ts`, keyed by
  URL): concurrent readers — the page's own `useResource` of the same URL,
  two viewers — share ONE request, and a remount within `META_FRESH_MS`
  (15 s) reuses it. The collection's cached cubes are dropped only when its
  meta changed (`noteMeta`: object identity, then a content hash), on an
  explicit `reload()`, or on a params refresh (`setParams`) — never merely
  because a viewer mounted, so returning to a view downloads nothing.
  Server errors (`{error}`) are shown verbatim, with `missing_tier_labels`
  as the hint.

## Adding a collection tier

1. Backend (`helpers/viewer_data.py`): add the tier to the collection meta
   (`{key, label, unit}`) and serve it from `get_cube` with `info` keys `label`,
   `pixscale`, `unit`, `wcs` (the tier's own grid), and when relevant
   `transfer_group`, `display_scale`, `direct_rgb`, `amp`/`var`. Document it in
   `euclid_polish/web/API.md` (C6).
2. Nothing in the viewer changes: the tier appears as a chip; the readout,
   residuals and profiles use its unit and WCS. Only a tier that must not be
   saved as a science crop needs no change either — saving allows the tiers in
   `RESULT_SAVEABLE_TIERS` (`controller.ts`).

## Tests

- `color.test.ts` holds `color.ts` to the old engine (golden fixture
  `__fixtures__/color_golden.json`, 25 cases, bit-identical Float32 planes and
  RGBA bytes) and to `visualization/color.py`. Regenerate the fixture with
  `python scripts/_viewer_parity_ref.py --write-golden --engine <old js>`
  (`git show 0ad56d9:euclid_polish/web/static/cutout_viewer.js > /tmp/cv.js`).
- `python scripts/_viewer_parity_ref.py | node scripts/check_viewer_parity.mjs`
  checks `color.ts` against `color.py` from Node: the unrounded float chain
  at 1e-4 (as the pre-rework script did) and the rendered RGBA bytes.
- `wcs.test.ts` (astropy fixture), `selection`, `cube` (incl. the shared
  meta request and `noteMeta`), `movie`, `render`, `stats`, `residual`,
  `readout`, `export`, `fit` (fit sizing, layouts, the old-key migration)
  and `barModel` (chips, bands, units, number formatting, one / two rows /
  wrapping with no control out of sight, the reserved bar height, the
  1-based counter) unit suites; `viewer.test.tsx` drives the controller,
  `<ImageViewer>` and `<ProfilePanel>` against a mocked backend (incl. a
  stubbed 1 px frame border for the pointer and lens geometry, WCS-matched
  crops, the figure heat bar, URL defaults, Shift keys, the full / compact /
  none bars, Q–Y / arrows / Space / `g` sequences, focus mode, the knee range
  and units in the Display dock, the dock beside the frames (keys stay with
  its sliders, Esc closes it), the basic compact dock, export without
  navigation, single-plane band chips, and a remount
  that issues no request). Tests clear `queryClient` (the meta) and
  `resetMetaNotes()` between cases.
