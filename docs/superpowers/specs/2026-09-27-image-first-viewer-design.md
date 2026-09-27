# Image-first viewing — design

Date: 2026-09-27. Trigger: the user asked for "good visuals of all images and an easily
accessible interface of viewing them … intuitive and good UX".

## Problem (measured, 792×720 stage)

| Page | First frame starts at | Viewer controls | Frame visible without scroll |
|---|---|---|---|
| Data › Records | 486 px (67 % of the height) | 158 px, 4 rows | 67 % |
| Realism › Visual | 416 px | 90 px + 97 px nav | 68 % |
| Ensemble › Disagreement | 400 px | 184 px, 5 rows | 81 % |
| Data › Cutouts | 362 px | 128 px | 81 % |

Five rows of centred controls (tiers, target PSF, colour, knee/brightness, tools,
stretch/colormap) sit above the images; the navigation takes another row below; frames
are sized by width only, so no page shows a whole frame without scrolling.

## Principles

1. **The image is the interface.** On every image page the first frame starts within
   ~140 px of the stage top, and the frame row fits the remaining viewport height.
2. **One instrument, one strong surface.** The viewer is a single dark "light table"
   block in BOTH themes (neutral, not tinted — colour judgement on astronomy images
   needs a neutral surround): control bar, frames edge to edge with 2 px gaps, readout
   line. Everything around it stays calm; this is where the design spends its boldness.
3. **Progressive disclosure.** The bar shows only what is touched every minute (tiers,
   band/colour, compare mode, zoom, navigation, focus). Everything else is one click
   away in popovers: Display (stretch, knee, brightness, black point, colormap, invert,
   NaN colour, link, histogram), tier extras (BHR FWHM on the BHR entry, JWST band on the
   JWST entry), tools (lens, profile, residuals), export (PNG, figure, video, save crop).
4. **Sentence-case, plain words.** No ALL-CAPS mono eyebrows or `A · B · C` label strings
   in the viewer chrome; icon buttons have tooltips with their shortcut. Data values
   (readout, magnitudes) stay tabular mono — they are data.

## Layout

```
┌ viewer (dark) ─────────────────────────────────────────────────────────────────────┐
│ LR  HR  SR ▾2   │ VIS Y J H Lupton Temp │ ◐ Display ▾ │ ▭▭ side ◌ blink ⇹ swipe │ ⌖ ▾ │
│ − + ⤢ │ ◀  12 / 100  ▶ ▶▶ │ ⬇ ▾ │ ⛶                              (wraps below 760 px) │
├──────────────────────────────────────┬─────────────────────────────────────────────┤
│ LR  VIS 17.31 AB                     │ HR                                          │
│                                      │                                             │
│            (as large as fits)        │                                             │
└──────────────────────────────────────┴─────────────────────────────────────────────┘
│ x 204  y 346   17h53m30.9s +65°05′55″   LR 76.9 e⁻   HR 0.23 e⁻        record 12  │
└────────────────────────────────────────────────────────────────────────────────────┘
```

- **Fit sizing:** frame side = min(width per column, available height), where available
  height = stage viewport height − the viewer's own bar/readout − a small margin. The
  layout picks the column count that maximises the frame side for the number of visible
  tiers ("Auto"); the layout menu also offers One row / Grid / Stack.
- **Focus mode** (⛶, key `F`): the viewer covers the whole stage (below the app top bar)
  on the neutral dark surround; `Esc` returns. "Full screen" (browser Fullscreen API)
  sits in the same menu.
- **Readout:** one line, reserved height (no layout jump), values per visible tier with
  units; the current object's label at its right.

## Fixes folded in

- Knee sliders span 0.1 – 10⁴ e⁻ (the research knee grid) on a log scale; JWST transfer
  groups are labelled in their display unit (MJy/sr), not e⁻; a single-plane native tier
  shows no Euclid band chips.
- The shared cube cache survives remounts (returning to a view does not re-download).
- The "server is running older code" banner compares the backend `.py` files the server
  loaded against the files on disk, not commit ids (a commit of the code it already runs
  is not "older code").

## Acceptance

At 792×720 and 1280×800, light and dark: on Records, Cutouts, PSFs, Visual, Disagreement,
Sky results/experiments/catalog-eval, the atlas tile card and Inspect, the first frame is
fully visible without scrolling; all old viewer features and shortcuts still work;
typecheck, lint and tests pass.
