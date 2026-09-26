# Viewer engine v2 (`src/viewer/`)

The image viewer of the console: raw Float32 cubes from `/viewer/cube/*`
rendered in the browser, with pan/zoom, a pixel readout with RA/Dec, residual
tiers, blink and swipe, histograms, profiles, the magnifier lens, the
disagreement movie and PNG / publication-figure / video export. It replaces
the pre-rework `static/cutout_viewer.js` (deleted; last version at commit
`0ad56d9`). Spec: `docs/superpowers/specs/2026-09-25-webui-rework-design.md` §6.

```tsx
import { ImageViewer, type ViewerApi } from "../../viewer";

<ImageViewer collection="nexus-field" params={{ field }} tiers={["lr", "sr", "jwst"]}
  urlKey="nexus" toolbar="full" onReady={(api) => (ref.current = api)}
  onState={(s) => setIndex(s.index)} />
```

Pre-rework pages keep `CutoutViewer` / `loadColorEngine` from `src/legacy.tsx`,
thin wrappers over this module (same props; `urlKey` defaults to the
collection).

## Props

| Prop | Meaning |
|---|---|
| `collection` | `/viewer/meta/<collection>` (C6): `sky`, `cutouts`, `evaluation`, `ensemble`, `archive-fields`, `real-field`, `jwst-euclid`, `nexus-field`, `psfs`, … |
| `params` | Collection query params (`subset`, `mode`, `field`, …). A change remounts the engine. |
| `tiers` | Initial tiers (default: `meta.default_tier`). |
| `initialIndex` / `initialId` | Initial object (a meta `id` wins; unknown ids fall back to the server's `?id=` lookup). |
| `urlKey` | URL-state prefix `v.<urlKey>.*` (off when absent). |
| `toolbar` | `"full"` (default), `"compact"` (tiers, colour, pan/lens, zoom), `"none"`. |
| `nav` | Navigation / export row (default `true`). |
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
| Esc | unfreeze, clear the profile |

Keys answer only in the most recently hovered / focused viewer (several can
be mounted). As in the old engine, Shift+letter acts as the letter (Shift+E =
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
  1 px border), where the canvas and overlays are drawn and `clientWidth`
  measures; the lens popups are placed from the same origin.
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
- **Histogram** (`HistogramPanel`): the visible region of one tier/band,
  native units, log counts. Its black/white handles edit the frame's transfer
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
  collection + tier + index + params, in-flight dedupe, per-consumer abort;
  a meta load drops the collection's cubes. Server errors (`{error}`) are shown
  verbatim, with `missing_tier_labels` as the hint.

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
- `wcs.test.ts` (astropy fixture), `selection`, `cube`, `movie`, `render`,
  `stats`, `residual`, `readout`, `export` unit suites; `viewer.test.tsx`
  drives the controller, `<ImageViewer>`, `<ProfilePanel>` and the legacy
  wrapper against a mocked backend (incl. a stubbed 1 px frame border for the
  pointer and lens geometry, WCS-matched crops, the figure heat bar, URL
  defaults, Shift keys).
