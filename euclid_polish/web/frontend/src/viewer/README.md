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
| `tiers` | Initial tiers (default: `meta.default_tier`), filtered by the initial object's own `tiers` once it is resolved. A viewer narrower than 480 px (the inspector panel, the bottom sheet) opens with at most the first two, unless the URL names its tiers. |
| `initialIndex` / `initialId` | Initial object (a meta `id` wins; unknown ids fall back to the server's `?id=` lookup). Resolved BEFORE the tiers are filtered. |
| `urlKey` | URL-state prefix `v.<urlKey>.*` (off when absent). |
| `toolbar` | The control bar: `"full"` (default); `"compact"` (no tools or export menus, and a basic Display row with knee + brightness: tiers, bands, Display, compare, a lens toggle, zoom, navigation, layout, Open large); `"none"` (no bar, or with `nav` a bar of navigation, export and Open large only). |
| `nav` | Navigation (◀ position / count ▶, run-through) in the bar (default `true`); only a navigating viewer prefetches its neighbours. The export menu is in every full bar, with or without `nav`. |
| `display` | Per-viewer display override (wins over the Display panel). |
| `onState(s)` | Every state change: the old `getState()` keys (`index, tier, tiers, color, layout, knee, gain, transfers, params, selection`) + `id, view, tool, compare`. |
| `onReady(api)` | The `ViewerApi` once mounted, `null` on teardown. |

## `ViewerApi`

`goTo(i)`, `goToId(id)`, `setTiers(keys)`, `setView({color, knee, gain})`
(per-viewer override of every transfer group; remembered before the meta
loads), `setParams(patch)` (refresh the visible cubes in place, e.g. the PSF
warp seed), `setMorphMembers(csv | null)`, `getIndex()`, `isReady()`,
`getState()`, `exportFigure()`, `savePng()`, `saveCropToResults()`, `reload()`,
`zoomTo(ra, dec, fovArcsec)`, `resetView()`, `zoomBy(factor)` (one zoom
step, landing on an integer device-pixel magnification; out past the
smallest view fits the image — the + / − keys), `setTool("lens" | "profile"
| "none")` (the magnifier, the profile panel, or pan and zoom), `setFocus(on)`,
`getReadout()`, `destroy()`. Realism › Visual's shared row drives both of its
viewers through `zoomBy` and `setTool`.

## Look: one light table

The viewer is ONE neutral dark block in both themes (`.cv-table`): the
control bar, the Display row when open, the frames edge to edge with 2 px
gaps (no per-frame border, radius or shadow), and a readout of reserved
height (one 28 px line; 2–4 lines in a narrow viewer — no layout jump). The
surround is neutral (`#161719`, untinted: colour judgement on astronomy
images needs it); the kit controls on it read a scoped dark palette (AA:
text ≥ 12.8, dim ≥ 7.2, faint ≥ 5.1, accent ≥ 5.8 on every table surface;
the user's accent preset in its dark-theme value). The popovers the bar and
the Display row open are portalled and follow the app theme.

```
┌ what ─ LR HR BHR ▾ │ VIS Y J H Lupton Temp ┐ ┌ how ─ ⚙ Display │ ▭▭ ◉ ⇹ │ ⌖ ·· − + ▢ │ ◀ 12 / 100 ▶ ▷ │ ▦ ⬇ ⛶ ┐
├ Display row (when open) ─ Knee ──●── 100 e⁻   Brightness ──●── 1×   Stretch [Asinh ▾]   More display settings ▾  ✕ ┤
├─ LR ─────────────────────┬─ HR ─────────────────────┐   frames: the tier's name; on hover the cube label + magnitude
│                          │                          │
└──────────────────────────┴──────────────────────────┘
  x 204 y 346  17h53m30.9s +65°05′55″  VIS  LR 76.9 e⁻  HR 0.23 e⁻            record 12
```

- **Bar** (`Bar.tsx`, `BarMenus.tsx`, pure model `barModel.ts`). ONE row
  when it fits; else two — what is shown (tier chips + the tier menu, the
  band / colour choice) above how it is shown (Display, compare, tools ▾,
  zoom, navigation, layout ▾, export ▾, Open large). `barLayout` decides
  from the measured group widths (each group carries `data-g`): one row with
  the Display text, else one row icon-only, else two rows (icon-only when
  the second row needs it). A narrow viewer (the 380 px inspector panel, the
  bottom sheet: 300–480 px) keeps TWO rows: the first collapses its band
  chips into one select, then its tier chips to the selected ones (the tier
  menu ▾ has every tier); the second moves its rarely used groups into a
  More menu (⋯) — export, then layout, tools, compare, zoom — until it fits.
  Display, the navigation and "Open large" never move. Only below ~300 px
  does a row still wrap; no control is ever cut off or scrolled out of sight
  (barModel tests sweep 300–480 px). A group in More, or collapsed, keeps
  the width it had in the bar (no flip-flop). Measured: 37 px for one row,
  67 px for two (the NEXUS tile card at 228 px and Disagreement at 356 px:
  67 px, was 97–127 px). The result lives in the viewer store (`bar`), so
  the frame grid refits in the same layout pass. Before the meta arrives the
  bar reserves the rows it had last time for the collection (`localStorage`
  `euclid-polish.viewer.bar-rows`, `reservedBarRows`), else two rows below
  `ONE_ROW_MIN_WIDTH` (760 px), so the frames do not jump.
- **Open large** (⛶, key `F`): always a visible icon button at the end of
  the bar (also "Open large (focus mode)" in the layout menu): focus mode.
- **Navigation**: ◀ position / count ▶ and the run-through. The position is
  1-based over the object count ("1 / 100"; a typed number is 1-based,
  `navPosition` / `parsePosition`); labels and URLs keep the 0-based index,
  which the tooltip names.
- **Tier chips**: every tier that is not `hidden` (at most `MAX_TIER_CHIPS`,
  plus any selected one), short names (`shortTierLabel`: "SR · production
  gate" → "SR"; the full label in the tooltip / accessible name). A tier the
  current object does not have is DIMMED and not clickable (never struck
  through); its tooltip and accessible name say why — the backend's
  `missing_tier_labels[key]`, else "Not available for this object". The
  tier menu lists every tier (hidden member tiers under "More tiers") with
  its extras on its entry: the BHR target FWHM slider on BHR, the JWST band
  on JWST, the movie amplitude / speed on the movie.
- **Bands**: the collection's `band_names` (NISP shown as Y J H), then
  Lupton and Temp, with their Q–Y keys; none for a log collection (PSFs) or
  when every shown tier is a single plane (a native one-filter image). One
  select in a narrow bar.
- **Display row** (`DisplayRow`; the bar's Display toggle, Esc or ✕ closes
  it): ONE quick row under the bar with the three controls touched most —
  knee, brightness and stretch (one transfer group at a time, Euclid / JWST,
  when the collection has two; `basic` in compact bars: knee and brightness)
  — each slider with a typed value in the group's display unit (`groupUnit`:
  a JWST group reads MJy/sr = transfer ÷ `X-Cube-Display-Scale`; Euclid reads
  e⁻; "e⁻ per 0.1″ px" while surface brightness is matched). It never sits
  beside the frames and never scrolls: the fit subtracts its height (one
  line when it fits, else it wraps onto a second: 69 px at 728 px), so
  width-limited frames keep their size (Records at 1024 × 768: 363 px with
  the row open, was 227 px beside the old dock). "More display settings"
  opens the rest in a popover that is wide and short so it covers as little
  of the frames as it can (up to 600 px wide, two columns once it has 540
  px; never scrolls): the black point and a reset per group, colormap,
  residual colormap, invert and NaN colour; then "Match surface brightness
  across pixel scales", "Use the page-wide display settings" (the link),
  "Histogram and cuts" and the Display panel. The histogram is a page of its
  own in the popover ("Back" returns to the settings), so it replaces the
  settings instead of growing the popover. The knee slider is
  log over `KNEE_SLIDER_RANGE` = 0.1–10⁴ (the research knee grid); the
  defaults stay absolute asinh at knee 100 e⁻. Keys typed in the row stay
  with its controls (a slider's arrows do not change the object); Esc closes
  it.
- **Tools popover**: pan / magnifier lens, the profile panel (also in focus mode), residual tiers
  (A − B, log₂ A/B, (A − B)/σ), the run-through and blink intervals.
- **Export menu**: PNG, publication figure, video, save the crop to results (S).
- **Frame labels**: the tier's short name; hovering a frame shows the cube
  label + magnitude instead (also its accessible name); the readout carries
  the magnitudes while idle. Band names read as the chips do everywhere
  (frame label, readout, histogram: "Y", not "Y_E"; `bandLabel`).
- **No coverage**: a tier whose cube has (almost) no finite value — under
  `MIN_COVERAGE` (1 %) AND fewer than `EMPTY_MAX_FINITE` (1000) values (a
  JWST cutout outside the mosaic: 58 of 722 500 pixels on the NEXUS tile
  `f200w-0000`; `cubeIsEmpty`) — draws the neutral surround with a quiet
  centred caption, "No JWST data here" (the tier's short label), and the
  readout says "no data" for it. A sparse tier — under 1 % but at least 1000
  values, a real corner of data on a big cutout (`cubeIsSparse`) — is
  painted, with a quiet "Little JWST data here" at its foot that lets the
  pointer through. Partial coverage keeps the image and paints its NaN
  pixels in the NaN colour (a neutral dark grey, `#404040`).
- **Readout**: hovering — x y, RA Dec (copy), the shown band, every visible
  tier's value with its unit; idle — the object's position, each tier's
  magnitude, the zoom and field; at the right the save status or the
  object's label. Below ~560 px of viewer width (`READOUT_WRAP_WIDTH`), or
  when one line would cut values off, it reserves 2–4 lines
  (`readoutLines`, decided before any hover from worst-case widths): the
  position first, then the per-tier values, each tier's name, value and
  unit kept together (nowrap) and the line broken only between tiers; the
  object's label gives way first. On one line the values never shrink (the
  tile card at 228 px: 4 lines, 200 of 200 px used, was 258 of 581). Each
  line is 22 px (18 px of text plus the 4 px between lines, carried by the
  line itself, so the empty break line costs nothing); the height is
  1 + 6 + N · 22 px (51, 73, 95), and the content always fits it (tile
  inspector at 1024 × 768: 44 / 44 px with two lines, 66 / 66 with three;
  was 39 / 44, clipping the last line). The full per-band values are in the
  line's tooltip and in `getReadout()`.

### Fit sizing (`fit.ts`)

Every frame is a square: side = min(width per column, height per row). The
height is the stage viewport (the nearest scrolling ancestor, i.e.
`main.stage` or the inspector body, else the window; the focus-mode surround
in focus mode) BELOW THE VIEWER'S OWN TOP — `heightUnderTop`: viewport −
(viewerRoot.top − stage.top + stage.scrollTop) − the viewer's own chrome
(every row of the light table but the frames: the bar, the Display row, the
readout; in focus mode also the profile panel) − an 8 px margin — clamped by
`MIN_FRAME_SIDE` (160 px). Only a viewer that starts below the first screen
(not even a minimum frame in sight under its top) fits the whole viewport:
it is scrolled to. So the first frame row AND the readout are in sight
without scrolling, whatever sits above the viewer (tab strip, toolbar,
caption, the version banner). What sits above it is watched cheaply: a
ResizeObserver on every element that precedes the viewer or an ancestor (up
to the scroll box), and a childList MutationObserver on each ancestor — a
banner dismissed or a toolbar that wraps refits it (Disagreement at 720 ×
720: dismissing the banner moved the viewer up 52 px and the frames grew
228 → 254 px, the readout still ending at 711 of 720). The page-side fit
workarounds (Data `ViewerStage` / `viewerFit.ts`, Sky `FitBox` /
`fitMath.ts`) are gone. The layout (the layout menu, persisted in
`localStorage` `euclid-polish.viewer.layout`; the old `…cutout-viewer.layout`
"two-rows" migrates to Grid):

| Layout | Columns |
|---|---|
| Auto (default) | among the arrangements whose rows all fit the height (a side raised to the minimum can push a row below the fold), each scored by its side less 6 % per empty cell (`EMPTY_CELL_PENALTY`: a 2 + 1 grid must beat one row by more than that), one row when its score is within 2 % of the best (`ONE_ROW_TOLERANCE`), else the best; near-ties (`NEAR_TIE`, 2 %) go to the fewest empty cells, then to more columns |
| One row | every frame side by side |
| Grid | ⌈√n⌉ |
| Stack | one |

The height may shrink a side to `MIN_FRAME_SIDE` (a very short window
scrolls), the width never. Blink / swipe are one frame. The publication
figure follows the on-screen rows (`figureLayout`), and a saved crop's
`display.layout` keeps the old two values ("one-row" | "two-rows").

Measured (1024 × 768 unless noted; frame top / side / readout bottom, the
stage ends at 768): Disagreement 173 / 278 (2 + 1) / 759, 241 px in one row
with the Display row open; Records 277 / 363 / 668 (737 with the Display
row); Visual 232 / 363 / 645; Experiments 210 / 241 / 502 (one row: a 2 + 1 grid of 249 px lost to the empty-cell cost); Catalog eval
224 / 363 / 615; Inspect 264 / 371 / 663; Cutouts at 1280 × 800 222 / 542 /
792 (was 768 past a 752 stage); the NEXUS tile card in the inspector 175 /
228 (two tiers) / 727; the atlas tile at 720 × 720 (bottom sheet) 257 /
343 / 628.

### Pixel-exact scale (`draw.ts`)

In the fit (the whole image) the drawn image is snapped to the largest
integer multiple of native pixels, in DEVICE pixels, that fits the cell —
floor(side · dpr / w) · w / dpr for the finest tier that is magnified
(`snappedDrawSide`) — ONE side for every frame of the grid, so coarser
tiers on an integer fraction of its grid (LR 0.1″ next to HR 0.05″) are
integer magnifications too and blink / swipe / side by side line up; it is
centred on whole device pixels. A snap that would keep less than
`SNAP_MIN_FILL` (80 %) of the cell keeps the full cell instead (a 1.97× fit
snaps to 1×: half the side — Disagreement at 1024: 128 of 252 px; Records:
255 of 363 px), drawn "sharp-bilinear" (nearest neighbour to ⌈scale⌉ in a
scratch canvas, then bilinear down: square pixels of equal size, ≤ 1 device
px of blend at their edges). "Pixel-exact fit" in the layout menu
(`localStorage` `euclid-polish.viewer.pixel-exact`) always snaps. Below 1×
(downsampling) the frame is smoothed (bilinear, high quality). User zoom
(wheel, pinch) is continuous; the zoom steps (+ / −, the bar's buttons,
`zoomBy`) land on integer device-pixel magnifications of the finest tier
(`zoomPreset`: 1 2 3 4 6 8 12 16 24 32 48 64, the one nearest the target,
at least one step; out past the smallest zoomed view: the fit). Above 64×
(a small image such as a ~20 px ePSF in a big frame, whose fit is already
past it) the steps go on continuously by the step factor up to
`VIEW_MAX_ZOOM`, so + never stalls.

Decision pending (the lead): the spec asks to always snap; the default
keeps the 80 % rule because always snapping shrank the evidence pages to
50–70 % of their cell. Flip it by making "Pixel-exact fit" default on
(`controller.ts` reads `euclid-polish.viewer.pixel-exact`). Markers,
the readout, the crosshair, the lens, profiles, swipe and the PNG / video
export all follow the drawn rectangle (`controller.layoutOf`).

### Focus mode

⛶ or `F`: the viewer covers the stage below the app top bar (and the
inspector) on the neutral surround, the frames re-fit to it, and it keeps
the keyboard while the pointer is elsewhere; `F`, `Esc` or the button
return. "Full screen" in the layout menu enters focus mode and asks the
browser for full screen (on the document, so the portalled menus stay
visible); leaving full screen leaves focus mode. The profile panel stays
under the table on the surround (the frames fit above it; it takes at most
40 % of the height) and the Display row under the bar. Esc unfreezes
the lens / clears the profile first, then leaves focus mode, then closes the
Display row. The page keeps the viewer's height while it is lifted out (no
jump behind it). "Open large" (⛶) is always a visible button in the bar.

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
carried stays written. The tiers are compared with the ones the viewer
settled on at its first load (`controller.initialTiers`: the page's tiers as
the initial object has them, a narrow viewer's first two), and a tier set the
object has none of (every tier disabled) is never written.

## Display binding (C7)

Effective settings = `mergeDisplay(useDisplay, viewer override)` (the
switch is "Use the page-wide display settings" in More display settings):

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
Display panel's NaN colour (default a neutral dark grey `#404040`; persisted
state v3 migrates an untouched old default, magenta `#ff00ff` or slate
`#5b6475`).

**Match surface brightness across pixel scales** (`area.ts`;
`DisplaySettings.matchSurfaceBrightness`, OFF by default — the user has not
decided the default): under the shared e⁻/pixel stretch, HR and SR at 0.05″
look about 4× dimmer than LR at 0.1″ at the same surface brightness (Records
record 0: galaxy centre LR 1323 e⁻ vs HR 433.7 e⁻ at VIS 21.48 vs 21.51 AB).
When on, each e⁻ tier's display values are scaled by (ref / pixscale)²
before the stretch, ref = the coarsest shown e⁻ tier (pixscale from
`X-Cube-Pixscale`) — implemented as knee, black point and white reference ÷
that factor. The readout, magnitudes, histogram values and exports stay
native e⁻ per pixel; the Display row names it ("Surface brightness
matched") and the knee reads "e⁻ per 0.1″ px". Off, every factor is 1 and
the colour output is bit-identical to before (the golden parity tests are
unchanged). MJy/sr tiers and single-scale collections are untouched.

## Interaction

| Input | Action |
|---|---|
| wheel | zoom about the cursor when the viewer is focused or ⌘/Ctrl is held (`display.wheel`: `zoom-when-focused` default, `always-zoom`, `scroll`); otherwise the page scrolls. With the lens active: lens zoom. Horizontal wheel: brightness |
| drag (zoomed) / two-finger pinch | pan / zoom (pointer events) |
| double-click, `0` | fit (whole image) |
| `+` `−` | zoom one step (integer device-pixel magnifications, `zoomBy`) |
| lens tool, `L`, or hold Alt | magnifier lens; click freezes the matched crop, click again unfreezes |
| shift-drag | line profile |
| click (profile panel open) or shift-click | radial profile |
| `Q W E R T Y` | VIS, Y, J, H, Lupton, Temp (old keys) |
| `←` `→`, Space | previous / next, run through |
| `S` (or Shift+S) | save the frozen crop to results (kept while frozen) |
| `B` | blink |
| `F` | Open large: focus mode (the images fill the page); `F` or Esc returns |
| Esc | unfreeze, clear the profile; else leave focus mode; else close the Display row |

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
  at natural resolution into an offscreen canvas, then drawn into the
  visible canvas through the shared view (`selection.frameLayout` via
  `controller.layoutOf`: the whole image at the grid's snapped side,
  centred, or the view's square crop filling the frame; `draw.ts` picks
  nearest, sharp-bilinear or smooth).
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
- **Histogram** (`HistogramPanel`, in More display settings behind its disclosure): the visible
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
- **Exports** (`export.ts`): PNG of the frames as shown (each frame's drawn
  image, its `crop`, placed as on screen: a snapped image leaves its
  surround out); the publication
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
  because a viewer mounted, so returning to a view downloads nothing. The
  neighbours (+1…+3, −1) are prefetched only by a navigating viewer
  (`nav`), each with its OWN tiers (`meta.objects[j].tiers`,
  `tierAvailAt`): a tile without JWST is never asked for it (no 404s).
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
  `readout` (incl. coverage and the reserved readout lines), `export` (incl.
  the cropped PNG composite), `draw` (the draw plan, the snapped side and
  its 80 % rule, the zoom presets), `area`, `fit` (fit sizing under an
  offset, the minimum and below-the-first-screen rules, the auto rules, the
  old-key migration) and `barModel` (chips, bands, units, number
  formatting, one / two rows, the narrow More menu and collapsed chips with
  a 300–480 px sweep, wrapping with no control out of sight, the reserved
  bar height, the 1-based counter) unit suites; `viewer.test.tsx` drives the controller,
  `<ImageViewer>` and `<ProfilePanel>` against a mocked backend (incl. a
  stubbed 1 px frame border for the pointer and lens geometry, WCS-matched
  crops, the figure heat bar, URL defaults, Shift keys, the full / compact /
  none bars, Q–Y / arrows / Space / `g` sequences, focus mode and Open
  large, the knee range and units in the Display row (one group at a time),
  the row under the bar with More display settings in a popover (keys stay
  with its sliders, Esc closes it), the basic compact row, surface-brightness
  matching (off = unchanged display params; on: knee ÷ 4 on SR, the readout
  native), the object resolved before the tiers, per-object neighbour
  prefetch and none without navigation, a narrow viewer's two tiers, the
  snapped layout and Pixel-exact fit, `zoomBy` / `setTool`, dimmed
  unavailable chips with their reason, no disabled-only tier set in the URL,
  export without navigation, single-plane band chips, and a remount that
  issues no request). Tests clear `queryClient` (the meta) and
  `resetMetaNotes()` between cases.
