# JWST on the Euclid f_ν scale (viewer setting + plate parameter)

Date: 2026-10-03 · Status: approved (user, 2026-10-03)

## Problem

Every Euclid ↔ JWST comparison is shown on unrelated brightness scales:

- **Viewer.** A JWST frame is scaled by its own brightest pixels
  (`viewer_data._robust_display_scale`: the 99.5th percentile of its positive
  pixels is mapped to 3000, the Euclid white level at knee 100). The factor
  changes from tile to tile — 1443 to 7162 on NEXUS tiles 40/42/70/178 — while
  the physical conversion is the constant 3387 (VIS, 0.1″). JWST has its own
  transfer group (knee, brightness, black point), unrelated to Euclid's.
- **Figures › Plates** (`nexus_plates.py`, `scripts/render_nexus_comparisons.py`):
  Euclid LR | SR | NEXUS panels are each stretched on their own
  (`_asinh_display`: p90 softening, p0.5/p99.5 limits).

The user wants flux density per unit frequency (f_ν, the AB system), not
band-integrated flux, compared on one scale, as a parameter, so that the
Euclid scale translates into the JWST scale automatically.

Measured (tile `f200w-0040`, 99.5th percentile in MJy/sr): LR VIS 0.151, Y
0.297, J 0.352, H 0.459; NEXUS F200W 0.468. On the f_ν scale Euclid H and
F200W agree, as they should; VIS is ~3× fainter (red galaxies).

## Decisions (user, 2026-10-03)

1. **JWST follows the shown band.** Euclid frames keep today's per-band
   electron scale. Only JWST is converted: its scale is the Euclid scale
   translated through the band on screen (VIS knee 100 e⁻ ↔ 0.030 MJy/sr,
   H knee 100 e⁻ ↔ 0.365 MJy/sr). Switching the Euclid band changes the JWST
   frame's brightness.
2. **On by default**, in the viewer and for new plate renders; a switch (and
   a plate parameter) returns to today's behaviour.

## The conversion (one source of truth)

`photometry.mjy_per_sr_to_electrons_factor(band, pixscale)` (exists; used for
the SKIRT mocks) — electrons over the band's stack, per pixel of side
`pixscale`, of a surface brightness of 1 MJy/sr:

```
E_b(p) = 1e12 · Ω(p) · 10^(0.4·(ZP_b − 23.90)),  Ω(p) = (p·π/648000)² sr
```

`ZP_b` = `band.sim_zeropoint_e` (the stack zero point, already served as
`zeropoint_ab_e_total`). At p = 0.1″: VIS 3386.9, Y 234.3, J 286.9, H 274.0.
`E_b(p) = E_b(1″) · p²` exactly, so the server sends one number per band and
the browser never re-derives the AB system:

- `viewer_data.color_constants()` gains, per band,
  `e_per_mjy_sr_arcsec2 = mjy_per_sr_to_electrons_factor(band, 1.0)`, and at
  the top level `lr_pixscale = Config.VIS_PIXEL_SCALE_ARCSEC` (0.1″, the
  fallback reference pixel below). Served with every viewer meta.

## Viewer

### Setting

`DisplaySettings.jwstFollowsEuclid: boolean`, **default true**
(`state/display.ts`: DEFAULTS, `sanitizeDisplay`, `settingsOf`; persisted
settings without the key get the default). Label: **"JWST on the Euclid scale
(f_ν)"**. Shown:

- in the viewer's *More display settings* next to "Match surface brightness
  across pixel scales", with a hint (on: "JWST follows Euclid VIS: 1 MJy/sr =
  3387 e⁻ per 0.1″ px"; off: "JWST is scaled to its own brightest pixels";
  no JWST frame shown: "No JWST frame is shown");
- in the page-wide Display panel (`app/DisplayPanel.tsx`) next to the
  surface-brightness switch.

Like every display field it follows the page-wide settings, or a per-viewer
override when the viewer is unlinked.

### What it does (new pure module `viewer/fnu.ts`)

A JWST frame **follows Euclid** when the setting is on, its transfer group is
`jwst`, its unit is MJy/sr (`X-Cube-Unit`, normalised), and the meta has an
Euclid transfer group. Then:

- **Band b** = the shown colour when it is a calibrated Euclid band (`VIS`,
  `Y_E`, `J_E`, `H_E` — present in `meta.color.bands`, not `display_only`);
  otherwise (`lupton`, `temp`, `rgb`, `native`) `VIS`, because the colour
  modes' knee is already VIS-equivalent (`prepareCore` factor =
  `abFluxNorm(VIS)`·…).
- **Reference pixel** = the coarsest shown Euclid frame in electrons (group
  `euclid`, unit e⁻): its pixel scale `p_ref` and its own display factor
  `D_ref` (served display scale × per-area factor; 1 for NEXUS / real tiles).
  With no Euclid e⁻ frame shown: `p_ref = meta.color.lr_pixscale`, `D_ref = 1`.
- **Display factor** of the JWST frame: `φ = E_b(1″) · p_ref² · D_ref`
  replaces its served robust display scale (in the controller: one more factor
  next to the per-area factor, `fnuFactor = φ / displayScale`, so knee, black
  point and white reference ÷ it — the same mechanism as `area.ts`; the
  prepared frame is not rebuilt).
- **Transfer group**: the frame uses the **Euclid** knee, brightness and black
  point (`controller.groupOf` returns `euclid` for it). The histogram's
  handles on a following JWST frame therefore edit the Euclid group.

Invariant: a JWST frame whose MJy/sr values equal an LR frame's band-b
electrons ÷ `E_b(p_ref)` renders the same pixels as that LR frame.

Not converted (they keep their own group and robust scale, and the frame's
hover label says "own scale"): JWST frames not in MJy/sr; the JWST colour and
temperature composites (unit `arb`, pairs only — none are cached now). The
auto stretches (`asinh-auto`, `zscale`) take their limits from each frame, so
the setting changes nothing visible there.

SR frames are untouched: with "Match surface brightness" off (its current
default) they stay per 0.05″ pixel; with it on they join LR's per-area scale
and therefore JWST's.

### Display row

While at least one shown JWST frame follows Euclid:

- the Euclid / JWST group switch hides the JWST option (only the Euclid
  sliders remain when every shown JWST frame follows);
- a note at the row's end, like "Surface brightness matched":
  **"JWST follows Euclid VIS · knee 0.030 MJy/sr"** (band label as the chips
  read it; the knee converted: `knee / φ` in MJy/sr, `formatSig`), its
  tooltip giving `1 MJy/sr = <E> e⁻ per <p_ref>″ px`.

### Unchanged

Readout, histogram values, magnitudes (JWST has none), residuals (refused
across units as today) stay native. The publication figure and PNG / video
exports draw the frames as shown; a JWST heatbar keeps native MJy/sr ticks
(its `scale` is the full factor φ). Saved results crops (`viewer_results`)
keep their recorded display scale (out of scope). With the setting off the
rendering is bit-identical to today.

## Plates (Figures › Plates, `nexus_plates.py`, the CLI)

New parameter **`scale`**: `fnu` (**default**, "Shared f_ν scale") or `panel`
("Each panel on its own", today's).

- **`fnu`, band b ∈ VIS/Y/J/H**: the LR panel is stretched exactly as today
  (`_asinh_display` on its own pixels: softening `s` = p90 of positive values,
  limits = p0.5/p99.5 of `asinh(x/s)`). The SR panel gets the same `s`, lo,
  hi after × `(p_LR/p_SR)²` (per unit area, as "Match surface brightness"
  does in the viewer), with the same clipping at 0. The NEXUS panel gets them after
  `mjy_per_sr_to_electrons(·, band b, p_LR)`. One stretch, three panels.
- **`fnu`, `temp`**: LR and SR in the viewer's Temp colour (`eye_rgb`, knee =
  the tier's asinh, 100), SR × `(p_LR/p_SR)²` first; NEXUS grey with the
  viewer's absolute asinh at that knee (white at 30×) after conversion
  through **VIS** (`mjy_per_sr_to_electrons(·, VIS, p_LR)`), as in the viewer.
- **`panel`**: unchanged.
- A JWST panel that is not in MJy/sr under `fnu` is an error
  (`PlateError(422)`, "render with scale=panel"), never a silent mis-scale.
  NEXUS mosaics are MJy/sr.

Records and surfaces:

- `render_plates(..., scale="fnu")`; `plates.json` render records gain
  `"scale"`; records without it read as `"panel"` (every earlier render). A
  run still holds one render per (band, model): re-rendering one with another
  scale replaces it (files keep today's names).
- `export_render` re-draws with the record's scale.
- `POST /api/figures/nexus-plates` accepts `scale` (default `fnu`; anything
  else → 400); `GET` adds `scales` and `defaults.scale`; the job title names
  the scale.
- Figures › Plates (`NexusPlates.tsx`): a "Scale" segmented control (Shared
  f_ν · Per panel, URL state `scale`) beside Band; the render caption names
  the scale ("shared f_ν scale (VIS)" / "per-panel stretch").
- `scripts/render_nexus_comparisons.py --scale fnu|panel` (default `fnu`).
- Docstrings that call NEXUS "not a photometric truth … each panel its own
  stretch" are updated (it is still not a truth for the SR — a different
  band — but it is now on a shared photometric scale).

## Testing

Python (`tests/test_photometry.py`, `tests/test_nexus_plates.py`,
`tests/test_viewer_backend.py`):

- `color_constants()` serves `e_per_mjy_sr_arcsec2` =
  `mjy_per_sr_to_electrons_factor(band, 1.0)` and `lr_pixscale`;
  `E_b(1″)·0.1²` equals `mjy_per_sr_to_electrons_factor(band, 0.1)`.
- Plates invariant: a fixture tile whose NEXUS cube = the LR plane ÷
  `E_b(0.1″)` (on the same grid) and whose SR = LR upsampled ÷ 4 gives three
  identical panel images under `fnu` (each band, and `temp` for LR vs SR);
  `panel` output unchanged; records carry `scale`; old records read as
  `panel`; export reuses the scale; POST validates `scale`; non-MJy/sr JWST
  under `fnu` → 422.

Frontend (vitest):

- `fnu.ts`: band choice (bands, colour modes → VIS), reference pixel
  (coarsest Euclid e⁻ frame, fallback `lr_pixscale`), factor golden values
  from the served constants (VIS 0.1″ = 3386.9).
- Engine invariant: with the setting on, a JWST frame holding LR ÷ E_b in
  MJy/sr renders the same ImageData as the LR frame, for VIS and for H; with
  it off, JWST rendering is bit-identical to today's (robust scale, own
  group); the existing colour parity tests are unchanged.
- Group routing (JWST follows `euclid`; composites and non-MJy/sr keep `jwst`),
  Display row note and group switch, More display settings / Display panel
  switches, persisted-state sanitising (missing key → true).
- Figures › Plates: the Scale control posts `scale`; captions name it.

Existing tests that assert the JWST group's independent knee are updated to
the new default (or turn the setting off where they test the old path).

## Docs

`viewer/README.md` (Display binding: the new setting, factors, notes),
`FOUNDATION.md` (C7 `DisplaySettings` field), `API.md` (meta `color` fields,
plates `scale`, render record `scale`), the `nexus_plates` and script
docstrings.
