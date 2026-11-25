# EuclidPOLISH — Hernquist group, Fri 4 Dec 2026 — speaker script

Prezi: prezi.com/p/edit/bwmtjxdbnkqw (87 clicks, 11-25: four summary points). Backup of the pre-polish version:
prezi.com/p/edit/0nzro8jentfl. The same lines are in Prezi's Presenter view (frames/presenter_notes.py).
Target: ~16 min of talking, the last minute a live demo of the all-sky viewer, then questions.
`[n]` is the click that brings that view up. "→" means the click only fades something in. A card's first sentence is
already on screen when you arrive.

**The canvas is four summary points.** Each row of slides sits on one panel under one point; the talk follows the
paper's introduction left to right, top to bottom, and closes on the whole canvas, where the four points are the summary:

1. Euclid is the first wide survey with a sharp, known PSF *(history of surveys, lenses, POLISH, Euclid)*
2. Simulated TNG50 skies pass for real Euclid data *(mission, TNG50, artifacts, fake skies, the quiz)*
3. Trained only on them, POLISH resolves 2–3× finer detail, confirmed by JWST *(network, cat, lenses, JWST)*
4. Faint flux and bright cores remain; next: all of Euclid DR1 *(limitations, Next + QR)*

---

## Title (0:00–0:30)

**[0] The whole canvas.** "Four points; we'll come back to them at the end."
**[1] Title.** "Euclid is ESA's survey telescope: a third of the sky at nearly Hubble sharpness. Today: a network that
makes Euclid sharper still, trained only on TNG50 skies."

## Point 1 · Euclid is the first wide survey with a sharp, known PSF (0:30–6:30)

Open the row by saying the point. History of surveys:

**[2] Every survey trades area for sharpness.** On screen: "From the ground you can image the whole sky, but the
atmosphere blurs everything to about an arcsecond." [3] → "From space you're sharp, but the fields are tiny."
[4] Zoom on the chart: "Each dot is a survey. Up is sharper, right is wider. Nobody is in the top-right corner."

**[5] Each box is one survey's share of the sky.** "The big box is DSS2, photographic plates of the whole sky."
[6] → Pan-STARRS, three quarters. [7] → DES, an eighth.
[8] The 50″ window on the Hubble box: "Hubble's widest survey: this small." [9] → JWST's widest: smaller still.
[10] → JWST's resolution. [11]–[14] → walk back down: Hubble, DES, Pan-STARRS, DSS2. "Same sky, forty years of
telescopes. Area and sharpness never came together."

Our goal, lens finding:

**[15] Lenses.** On screen: "A ring's geometry weighs its galaxy, dark matter included." [16] → time delays measure
the Hubble constant. [17] → double-source lenses test cosmology. [18] → "Euclid should find ~170,000 lenses, most only
a few PSF widths across. Rare, and small: resolution decides which ones we can model."

Super-resolution:

**[19] Radio already inverts a known PSF.** On screen: "From the ground, the atmosphere's blur changes all night."
[20] → "In radio the PSF is set by the antenna layout and known exactly. That is why POLISH, from Liam's group, can
learn on simulations and work on real data."

Euclid arrives:

**[21] Euclid's PSF: sharp, stable and measured.** "Above the atmosphere, five times sharper than DES, nearly the same
across the sky: a radio-like, known PSF over 14,000 square degrees."
**[22]–[25] Back to the window:** [23] → Euclid in the same window. [24] → Hubble's box. [25] → JWST's box. "Nearly
Hubble-sharp, everywhere." **[26]–[27]** Euclid's box: "over 14,000 square degrees."
**[28]–[29] The chart:** [29] → "Euclid lands in the empty corner: 0.16″ over 14,000 deg². DR1, public since November,
already covers 1,900."

## Point 2 · Simulated TNG50 skies pass for real Euclid data (6:30–9:30)

Open the row by saying the point. Our mission:

**[30] The galaxy from the title.** "Our mission: extend POLISH to Euclid." [31] → "From this…" [32] → "…to THIS: the
same Euclid data, POLISHed." [33] → "And JWST, for comparison: the structure is real."

The truth, TNG50 galaxies:

**[34] 1,154 TNG50-SKIRT galaxies.** "Real physics and real morphology, which is your group's work. z ≈ 0, 100 pc per
pixel. We take only the morphology; colors come from Q1."
**[35] The simulated truth has its own artifacts.** "Dust-box facets, box cuts, slabs, grain. Dust-box and box-cut
galaxies are left out of training; slabs and grain we check for in the outputs."

Fake skies:

**[36] Each truth becomes a realistic Euclid image.** [37] → measured PSF and 0.1″ pixels. [38] → real Q1 noise,
artifacts and NISP holes. [39] → the recipe in one line.
**[40] Which four are real?** Give the room ~5 s. [41] → "Half are simulated. If you can't tell them apart, neither
can the network."

## Point 3 · Trained only on them, POLISH resolves 2–3× finer detail, confirmed by JWST (9:30–14:30)

Open the row by saying the point. Our network:

**[42] Super-resolution network.** "A residual CNN, VIS + Y + J + H in, the same four bands out on a 0.05″ grid. Arcsinh
inputs at eight knees; 15 million steps, about 70 A100-hours."
**[43]–[44] Back to the quiz:** "Now run it on all eight fields." [44] → "Real or simulated, the network treats them the
same way."
**[45] What did it learn? Priors on a cat.** [46]–[48] → truth, Euclid, POLISHed. [49]–[51] → face close-up: Euclid,
POLISHed, truth. "It restores fur it has never seen; it learned deconvolution, not galaxies."

Results:

**[52] Lenses, POLISHed: the six lenses from the first row.** Each lens: zoom, then → POLISHed. Pause on the Q1 ring
[53]–[54] and NGC 6505 [55]–[56]; take the four DR1 systems [57]–[64] faster. [65] back to the slide: all six, sharpened.
**[66]–[79] Checked galaxy by galaxy against JWST.** Four galaxies; each: zoom (Euclid) → JWST → POLISHed. Flip with
← → on the first one or two. [79] back to the slide: "Against JWST in H band, POLISHed resolves detail two to three
times finer than Euclid: about 0.2″ instead of 0.7″ on bright structure."

## Point 4 · Faint flux and bright cores remain; next: all of Euclid DR1 (14:30–16:30)

Open the row by saying the point.

**[80] Where it fails.** On screen: "Faint galaxies lose 2–8% of their flux." [81] the flux plot. [82]–[83] the
bright-nucleus close-up, then the holes outlined. [84]–[85] → "Very bright cores can come out with holes. Both trace
back to gaps in the training set."
**[86] Next.** "POLISH all of Euclid DR1, then the full survey. The viewer already runs on DR1. In my last minute, let
me show it." Switch to the browser tab with the viewer open (Euclid DR1, model 271):
https://cutout-production-538f5738.up.railway.app/#ra=228.827579&dec=45.009343&z=5.016&rel=dr1&m=271
Swipe Euclid ↔ POLISHed once, zoom into one galaxy, then come back.

## Close: the four points

**[87] The whole canvas.** Read the four points, one sentence each, then take questions. The QR code (bottom row)
opens the viewer.

---

### Delivery notes
- Open each row by saying its point; close by reading the four points on the whole canvas.
- Each slide title is the takeaway; say it first, then point at the evidence.
- Pause 2–3 s after each zoom so the room can orient before you talk (Prezi motion-sickness rule).
- Before the talk: open the viewer tab and let it load the first tiles (WebGPU warm-up), so the demo starts instantly.
- Practice out loud at least twice with a timer; the end-to-end target is 16 min before the demo.
