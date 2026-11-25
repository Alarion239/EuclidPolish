# EuclidPOLISH — Hernquist group, Fri 4 Dec 2026 — speaker script

Prezi: prezi.com/p/edit/bwmtjxdbnkqw (75 clicks). Backups: prezi.com/p/edit/tubunespdpok (before the 11-25 review
pass), prezi.com/p/edit/0nzro8jentfl (pre-polish).
The same lines are in Prezi's Presenter view (frames/presenter_notes.py).
Target: ~16 min of talking, the last minute a live demo of the all-sky viewer, then questions.
`[n]` is the click that brings that view up. "→" means the click only fades in an image. All of a slide's text is
on screen when you arrive.

**The canvas.** Four left-aligned rows, each on a light panel under one summary point; the title slide and a big QR code
to the viewer sit on the right. The start view and the zoom-out at the end are the same frame: the summary.

1. Euclid is the first wide survey with a sharp, known PSF *(surveys, POLISH in radio, Euclid's PSF, lenses)*
2. Simulated TNG50 skies pass for real Euclid data *(TNG50, artifacts, fake skies, the quiz)*
3. Trained only on them, POLISH resolves 2–3× finer detail, confirmed by JWST *(network, cat, lenses, JWST)*
4. Faint flux and bright cores remain; next: all of Euclid DR1 *(limitations, Next)*

---

## Opening: from this to THIS (0:00–1:00)

**[0] The canvas.** "Four points; we'll come back to them at the end."
**[1] Title.** "Euclid is ESA's survey telescope: a third of the sky at nearly Hubble sharpness. Today: a network that
makes Euclid sharper still, trained only on TNG50 skies." [2] → "From this…" [3] → "…to THIS." Pause. "The same Euclid
data, POLISHed. Remember this galaxy; we'll check it against JWST at the end."

## Point 1 · Euclid is the first wide survey with a sharp, known PSF (1:00–6:00)

**[4] Every survey trades area for sharpness.** Say the point first. On screen: "From the ground you can image the whole
sky, but the atmosphere blurs everything to about an arcsecond. From space you're sharp, but the fields are tiny." [5] The chart: "Each dot is a survey. Up is sharper, right is wider. Nobody is in the top-right corner."

**[6] Each box is one survey's share of the sky.** "The big box is DSS2, photographic plates of the whole sky."
[7] → Pan-STARRS, three quarters. [8] → DES, an eighth.
[9] The 50″ window on the Hubble box: "Hubble's widest survey: this small." [10] → JWST's widest: smaller still.
[11] → JWST's resolution, 0.06″. Then walk back down: [12] → DES, 0.9″; [13] → DSS2, 2″. "Same sky, forty years of
telescopes. Area and sharpness never came together."

**[14] Radio already inverts a known PSF.** On screen: "From the ground, the atmosphere's blur changes all night." "In radio the PSF is set by the antenna layout and known exactly. That is why POLISH, from Liam's group, can
learn on simulations and work on real data."

**[15] Euclid's PSF: sharp, stable and measured.** "Above the atmosphere: VIS 0.16″, five times sharper than DES's 0.9″,
nearly the same across the sky: a radio-like, known PSF over 14,000 square degrees."
**[16]–[19] Back to the window:** [17] → Euclid in the same window. [18] → Hubble's box. [19] → JWST's box. "Nearly
Hubble-sharp, everywhere." **[20]–[21]** Euclid's box: "over 14,000 square degrees."
**[22]–[23] The chart:** [23] → "Euclid lands in the empty corner: 0.16″ over 14,000 deg². DR1, public since November,
already covers 1,900."

**[24] Lenses: where resolution pays off.** All four lines on screen: ring geometry maps the lens galaxy's mass; time
delays measure the Hubble constant; double-source lenses test cosmology; "Euclid should find ~170,000 lenses, most only
a few PSF widths across. Resolution decides which ones we can model."

## Point 2 · Simulated TNG50 skies pass for real Euclid data (6:00–9:00)

**[25] The truth: 1,154 TNG50-SKIRT galaxies.** Say the point first. "Real physics and real morphology, which is your
group's work. z ≈ 0, 100 pc per pixel. We take only the morphology; colors come from Q1."
**[26] The simulated truth has its own artifacts.** "Dust-box and box-cut galaxies are left out of training; slabs and
grain we check for in the outputs."
**[27] Each truth becomes a realistic Euclid image.** Captions under each panel; [28] → measured PSF and 0.1″ pixels.
[29] → real Q1 noise, artifacts and NISP holes.
**[30] Which four are real?** Give the room ~5 s. [31] → "Half are simulated. If you can't tell them apart, neither
can the network."

## Point 3 · Trained only on them, POLISH resolves 2–3× finer detail, confirmed by JWST (9:00–14:30)

**[32] One network: four bands in, a 2× finer grid out.** Say the point first. "A residual CNN, VIS + Y + J + H in, the
same four bands out on a 0.05″ grid. Arcsinh inputs at eight knees; 15 million steps, about 70 A100-hours."
**[33] It learned deconvolution, not galaxies.** "What did it learn? Put a cat through the same forward model."
[34]–[36] → truth, Euclid, POLISHed. [37]–[39] → face close-up: Euclid, POLISHed, truth. "It restores fur it has never
seen."
**[40]–[41] Back to the quiz:** [41] → all eight POLISHed: "Real or simulated, the network treats them the same way."
**[42] Lenses, POLISHed.** [43] → zoom the Q1 ring, [44] → POLISHed; [45] → NGC 6505, [46] → POLISHed. [47] back to the
slide, then the four DR1 systems flip in place: [48]–[51]. "All six, sharpened."
**[52] JWST confirms 2–3× finer detail.** The line on screen: "In H band, POLISHed resolves detail down to ≈0.2″;
Euclid, ≈0.7″" (the 0.16″ from before is the sharper VIS PSF). Four galaxies; each: zoom (Euclid) → JWST → POLISHed:
[53]–[55], [56]–[58], [59]–[61], [62]–[64]. Flip with ← → on the first one or two. [65] back to the slide.
**[66]–[68] And a whole field:** back to the 50″ window, [67] → POLISHed, [68] → Hubble's box on top: "the same
structure, where Hubble can check it." (← flips back.)

## Point 4 · Faint flux and bright cores remain; next: all of Euclid DR1 (14:30–16:30)

**[69] Where it fails: two gaps in the training set.** Say the point first. Left: "Faint galaxies lose up to ~9% of their
flux: our training noise was more correlated than Euclid's." Right: "Very bright cores can come out with holes:
training had no bright, extended cores." [70] → the holes outlined.
**[71] The title galaxy again.** "Remember: from this… to THIS." [72] → "And JWST, for comparison." Let it sit.
**[73] Next.** "POLISH all of Euclid DR1, then the full survey, with the same network."
**[74] The QR code.** "The viewer already runs on DR1: scan it. In my last minute, let me show it." Switch to the browser
tab with the viewer open (Euclid DR1, model 271):
https://cutout-production-538f5738.up.railway.app/#ra=228.827579&dec=45.009343&z=5.016&rel=dr1&m=271
Swipe Euclid ↔ POLISHed once, zoom into one galaxy, then come back.

## Close: the four points

**[75] The whole canvas.** Read the four points, one sentence each, then take questions. The QR code stays on screen.

---

### Delivery notes
- Open each row by saying its point; close by reading the four points on the whole canvas.
- Each slide title is the takeaway; say it first, then point at the evidence.
- Pause 2–3 s after each zoom so the room can orient before you talk (Prezi motion-sickness rule).
- Before the talk: open the viewer tab and let it load the first tiles (WebGPU warm-up), so the demo starts instantly.
- Practice out loud at least twice with a timer; the end-to-end target is 16 min before the demo.
