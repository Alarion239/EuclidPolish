# EuclidPOLISH — Hernquist group, Fri 4 Dec 2026 — speaker script

Prezi: prezi.com/p/edit/bwmtjxdbnkqw (78 clicks). Backups: prezi.com/p/edit/ccfvjh3y8k0b (before the 11-27 slide
comments), prezi.com/p/edit/tde1zyrwwauy (before the 11-25 picture
pass), prezi.com/p/edit/tubunespdpok (before the 11-25 review pass), prezi.com/p/edit/0nzro8jentfl (pre-polish).
The same lines are in Prezi's Presenter view (frames/presenter_notes.py).
Target: ~16 min of talking, the last minute a live demo of the all-sky viewer, then questions.
`[n]` is the click that brings that view up. "→" means the click only fades in an image. All of a slide's text is
on screen when you arrive.

**Slide numbers.** Every zoom stop has a number badge in its top-right corner (the 33 stops, numbered as in Prezi's
sidebar; frames/slide_badges.py). On a return the badge lists every visit, earlier ones grey, the current one bold:
title 1→30, chart 2→9, survey card 3→8, survey window 4→7→28 (bottom-right corner there), quiz 14→17,
lenses 18→21, JWST 22→27. The grey-out rides on an existing click (Prezi's multi-object fade), so it costs no click.
It happens while you are elsewhere, except the window's 7: that one updates with the first click there (Euclid |
DSS2 appears).

**The canvas.** Four left-aligned rows, each on a light panel under one summary point; the title slide and a big QR code
to the viewer sit on the right. The start view and the zoom-out at the end are the same frame: the summary.

1. Euclid is the first wide survey with a sharp, known PSF *(surveys, POLISH in radio, Euclid's PSF, lenses)*
2. Simulated TNG50 skies pass for real Euclid data *(TNG50, artifacts, fake skies, the quiz)*
3. Trained only on them, POLISH resolves 2–3× finer detail, confirmed by JWST *(network, cat, lenses, JWST)*
4. Faint flux and bright cores remain; next: all of Euclid DR1 *(limitations, Next)*

---

## Opening: from this to THIS (0:00–1:00)

**[0] The canvas.** "Four points; we'll come back to them at the end."
**[1] From this… to THIS.** Euclid on the left, POLISHed on the right, both on arrival. Pause. "The same Euclid data,
POLISHed. Remember this galaxy; we'll check it against JWST at the end." [2] → the title and names over the POLISHed
galaxy: "Euclid is ESA's survey telescope: a third of the sky at nearly Hubble sharpness. Today: a network that makes
Euclid sharper still, trained only on TNG50 skies."

## Point 1: Euclid is the first wide survey with a sharp, known PSF (1:00–6:00)

**[3] Every survey trades area for sharpness.** Say the point first. On screen: "Ground: wide but blurred. Space: sharp
but tiny." The chart: "Each dot is a survey. Up is sharper, right is wider. Nobody is in the top-right corner."

**[4] Each box is one survey's share of the sky.** "The big box is DSS2, photographic plates of the whole sky."
[5] → Pan-STARRS, three quarters. [6] → DES, an eighth.
[7] The 50″ window on the Hubble box: "Hubble's widest survey: this small." [8] → JWST's widest: smaller still, at
0.06″. Then walk back down: [9] → DES, 0.9″; [10] → DSS2, 2″. "Same sky, forty years of telescopes. Area and
sharpness never came together."

**[11] Radio already inverts a known PSF.** On screen: "Optical: the atmosphere blurs the PSF unpredictably. Radio: the
PSF is known exactly, so POLISH trains on simulations and works on real data." POLISH is from Liam's group.

**[12] Euclid's PSF: sharp, stable and measured.** No text on the slide: top row the ground (DES), VIS, NISP Y, J, H at
one scale and stretch; bottom row VIS in five sky regions. "Above the atmosphere: VIS 0.16″, five times sharper than
DES's 0.9″, nearly the same across the sky: a radio-like, known PSF over 14,000 square degrees."
**[13] Back to the window, now split:** Euclid on the left, every other survey on the right, one per click:
[14] → DSS2, 2″; [15] → Pan-STARRS, 1.1″; [16] → DES, 0.9″ ("Euclid beats every survey from the ground");
[17] → Hubble, 0.09″; [18] → JWST, 0.06″ ("only these two beat Euclid, and their fields are tiny").
**[19] Back to the sky share, split the same way:** Euclid's 14,000 deg² box on the left; [20] → DSS2, the whole sky;
[21] → Pan-STARRS, three quarters; [22] → DES, an eighth; [23] → Hubble, a dot; [24] → JWST, smaller still. Only the
all-sky surveys (DSS2, Pan-STARRS) cover more than Euclid.
**[25] The chart, whole card:** [26] → "Euclid lands in the empty corner: 0.16″ over 14,000 deg². DR1, public since
November, already covers 1,900."

**[27] Lenses: where resolution pays off.** All four lines on screen: ring geometry maps the lens galaxy's mass; time
delays measure the Hubble constant; double-source lenses test cosmology; "Euclid should find ~170,000 lenses, most only
a few PSF widths across. Resolution decides which ones we can model."

## Point 2: Simulated TNG50 skies pass for real Euclid data (6:00–9:00)

**[28] The truth: 1,154 TNG50-SKIRT galaxies.** Say the point first. "Real physics and real morphology, which is your
group's work. z ≈ 0, 100 pc per pixel. We take only the morphology; colors come from Q1." The 18 on the slide: spirals, bars, rings, edge-ons,
tidal tails, ellipticals.
**[29] The simulated truth has its own artifacts.** Pictures only. Top: dust-box facets (up to 70% of a galaxy's light);
bottom: box cuts, slabs, grain. "Dust-box and box-cut galaxies are left out of training; slabs and grain we check for in
the outputs."
**[30] Each truth becomes a realistic Euclid image.** Captions under each panel; [31] → measured PSF and 0.1″ pixels.
[32] → real Q1 noise and artifacts (each caption appears with its panel).
**[33] Which four are real?** Give the room ~5 s. [34] → "Half are simulated. If you can't tell them apart, neither
can the network."

## Point 3: Trained only on them, POLISH resolves 2–3× finer detail, confirmed by JWST (9:00–14:30)

**[35] One network: four bands in, a 2× finer grid out.** Say the point first. "A residual CNN, VIS + Y + J + H in, the
same four bands out on a 0.05″ grid. Arcsinh inputs at eight knees; 15 million steps, about 70 A100-hours."
**[36] It learned deconvolution, not galaxies.** "What did it learn? Put a cat through the same forward model."
[37]–[39] → truth, Euclid, POLISHed. [40]–[42] → face close-up: Euclid, POLISHed, truth. "It restores fur it has never
seen."
**[43]–[44] Back to the quiz:** [44] → all eight POLISHed: "Real or simulated, the network treats them the same way."
**[45] Lenses, POLISHed.** [46] → zoom the Q1 ring, [47] → POLISHed; [48] → NGC 6505, [49] → POLISHed. [50] back to the
slide, then the four DR1 systems flip in place: [51]–[54]. "All six, sharpened."
**[55] JWST confirms 2–3× finer detail.** Just the four galaxies on screen, 2×2; say it: "In H band, POLISHed resolves
detail down to ≈0.2″; Euclid, ≈0.7″" (the 0.16″ from before is the sharper VIS PSF). Each galaxy: zoom (Euclid) →
JWST → POLISHed: [56]–[58], [59]–[61], [62]–[64], [65]–[67] (the last: a clumpy irregular beside a red spiral). Flip
with ← → on the first one or two. [68] back to the slide.
**[69]–[71] And a whole field:** back to the 50″ window (it arrives on Euclid | JWST), [70] → POLISHed, [71] → Hubble's box on top: "the same
structure, where Hubble can check it." (← flips back.)

## Point 4: Faint flux and bright cores remain; next: all of Euclid DR1 (14:30–16:30)

**[72] Where it fails: two gaps in the training set.** Say the point first. Left: "Faint galaxies lose up to ~9% of their
flux: our training noise was too correlated." Right: "Very bright cores can get holes: training had no cores this
bright." [73] → POLISHed: the holes in the brightest core (no outline).
**[74] The title galaxy again.** "Remember this galaxy: the POLISHed one from the start." [75] → "And JWST, for comparison": Euclid, POLISHed and JWST side by side, one color stretch. Let it sit.
**[76] Next.** "POLISH all of Euclid DR1, then the full survey, with the same network."
**[77] The QR code.** "The viewer already runs on DR1: scan it. In my last minute, let me show it." Switch to the browser
tab with the viewer open (Euclid DR1, model 271):
https://cutout-production-538f5738.up.railway.app/#ra=228.827579&dec=45.009343&z=5.016&rel=dr1&m=271
Swipe Euclid ↔ POLISHed once, zoom into one galaxy, then come back.

## Close: the four points

**[78] The whole canvas.** Read the four points, one sentence each, then take questions. The QR code stays on screen.

---

### Delivery notes
- Open each row by saying its point; close by reading the four points on the whole canvas.
- Each slide title is the takeaway; say it first, then point at the evidence.
- Pause 2–3 s after each zoom so the room can orient before you talk (Prezi motion-sickness rule).
- Before the talk: open the viewer tab and let it load the first tiles (WebGPU warm-up), so the demo starts instantly.
- Practice out loud at least twice with a timer; the end-to-end target is 16 min before the demo.
