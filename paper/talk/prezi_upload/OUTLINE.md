# EuclidPOLISH: Prezi outline (build natively in Prezi)

Talk: Lars Hernquist's group (IllustrisTNG), Fri 4 Dec 2026, 15–20 min. Speaker: Aleksandr Belotserkovtsev
(with Tania Barone, Liam Connor). This is the backbone draft; text is minimal on purpose and will be rewritten.

## Principles

- One image per file; every image is placed on its own, never merged into a collage, so each can be zoomed into
  or replaced on its own.
- The camera moves between images: zoom out to a topic, then zoom in to one image at a time. Text sits around
  images, short (a title of ≤6 words, at most one line of ≤10 words under it).
- One plain dark background (#0f1218), no background photo. Light text (#e6e8eb), teal accent (#2fd4bf) for small
  labels, coral (#f08a6c) only for the "callback" images. Clean sans font.
- Images already carry their own labels (Euclid / POLISHed / JWST, band legend, scale bar). Do not crop them.

## Overview (the whole canvas, shown first and last)

Two rows read left to right, like the pipeline:

```
TOP ROW  "The real sky"        [0 Title] [1 Wide or sharp] [2 Euclid] [3 Lenses]  [6 The network]  [7 Results] [8 Limits]
BOTTOM ROW "The simulated sky"                           [4 The truth: TNG50] [5 Making Euclid images] ──┘
```

Topic 6 sits on the top row directly above the end of the bottom row (the trained network rises from it).
Path: 0 → 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → overview.

## Topics (images by file name in this folder)

### 0 · Title
- A01_title_barred_spiral_Euclid (large).
- Text: "EuclidPOLISH" / "Super-resolving Euclid with a network trained on TNG50 skies" /
  "Aleksandr Belotserkovtsev · Harvard College · with Tania Barone and Liam Connor".
- Note: this galaxy comes back at the end (callback).

### 1 · Wide or sharp
- A02_chart_wide_or_sharp, large.
- Zoom steps on the chart: (a) whole chart, (b) top-left cluster, text "Ground: wide, blurred by air",
  (c) bottom-right cluster, text "Space: sharp, tiny fields", (d) the Euclid point, text "Euclid: both".
- Two small text blocks next to the chart, each its own zoom stop:
  "Sharpen ground images? The blur is never known." and
  "Radio did it first: POLISH. The PSF is known exactly."

### 2 · Euclid
- Row of four PSFs: A03_PSF_VIS, A04_PSF_NISP_Y, A05_PSF_NISP_J, A06_PSF_NISP_H. Text: "Sharp and stable PSF".
- A07_PSF_VIS_wide_spikes beside the row (bigger).
- Small row: A08–A10 (same PSF in three sky regions). Text: "Same PSF across the sky".
- Text: "14,000 deg² · DR1 (12 Nov 2026): 1,900 deg², ~15,000 lens candidates".

### 3 · Lenses (callbacks)
- A11_lens_ring_Euclid, A12_NGC6505_ring_Euclid, A13_UltraDeepField_Euclid, each its own zoom stop.
- Text: "Compact lenses are missed" (near A11), "NGC 6505" (near A12),
  "Hubble's deepest field, as Euclid sees it" (near A13).

### 4 · The truth: TNG50  (bottom row)
- B01–B12: twelve TNG50-SKIRT galaxies, each its own image, in a loose 4 × 3 grid with gaps.
  Text: "TNG50-SKIRT · 1,154 galaxies · z = 0". Zoom into two or three of them.
- Sub-area "What we don't trust": C01_TNG_downside_dust_box, C02_TNG_downside_box_cut, C03_TNG_downside_slabs,
  C04_TNG_downside_particle_grain, each its own zoom stop, one-line labels:
  "Dust-box facets", "Cut by the box", "Slabs of light", "Particle grain".

### 5 · Making Euclid images  (bottom row)
- D01_forward_truth → D02_forward_PSF_pixels → D03_forward_noise_artifacts in a row with arrows.
  Text: "PSF · pixels · noise · artifacts".
- Quiz: E01–E08 in two rows of four (they carry letters A–H). Text: "Which four are real?"

### 6 · The network  (top row, above the end of the bottom row)
- F01_network_architecture, large. Zoom stop on its lower half (panel b, one residual block).
  Text: "15M steps · 70 A100-hours".
- Cat row: G01_cat_photo → G02_cat_truth → G03_cat_Euclid → G04_cat_POLISHed.
  Text: "What did it learn? Its prior is galaxies." Small credit under G01:
  "Photo: Alvesgaspar, Wikimedia Commons, CC BY-SA 3.0".

### 7 · Results (the payoff: every early Euclid image comes back POLISHed)
- 7a Quiz answered: H01–H08 POLISHed in the same 2 × 4 arrangement as the quiz; under B, C, E, H their truths
  H09–H12. Text: "A, D, F, G were real".
- 7b The opening galaxy: A01 (again) → I01_barred_spiral_POLISHed → I02_barred_spiral_JWST, one row.
  Text: "Same arm, same knots as JWST".
- 7c Stars: J01_star_Euclid → J02_star_POLISHed. Text: "Spikes are the telescope. The flux stays."
- 7d Lenses: A11 (again) → K01_lens_ring_POLISHed; A12 (again) → K02_NGC6505_ring_POLISHed_optional.
- 7e Deep field: A13 (again) → K03_UltraDeepField_POLISHed; zoom K04 → K05.
- 7f Against JWST: ten rows of three, L01–L10 (each row: Euclid | POLISHed | JWST, file names L##1 / L##2 / L##3).
  Text: "Against JWST, galaxy by galaxy". Zoom into two or three rows.
  (If the editor supports it, a nicer callback is to revisit topic 3 and fade the POLISHed image in on top of the
  Euclid one; optional.)

### 8 · Limits and a question
- Text only: "Faint flux −10% · faint pixel grid · bright cores" and, larger, in coral:
  "What would you change in our use of TNG?"
- Then zoom out to the overview.
