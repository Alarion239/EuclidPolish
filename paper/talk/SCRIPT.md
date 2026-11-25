# EuclidPOLISH — Hernquist group, Fri 4 Dec 2026 — speaker script

Prezi: prezi.com/p/edit/bwmtjxdbnkqw (90 clicks). Backup of the pre-polish version: prezi.com/p/edit/0nzro8jentfl.
The same lines are in Prezi's Presenter view (frames/presenter_notes.py).
Target: ~16 min of talking, the last minute a live demo of the all-sky viewer, then questions.
`[n]` is the click that brings that view up. "→" means the click only fades something in. A card's first sentence is
already on screen when you arrive.

**The canvas.** Four left-aligned rows, each on a light panel under one summary point; the title slide and a big QR code
to the viewer sit on the right. The zoom-out at the end is the summary:

1. Euclid is the first wide survey with a sharp, known PSF *(surveys, lenses, POLISH in radio, Euclid's PSF)*
2. Simulated TNG50 skies pass for real Euclid data *(TNG50, artifacts, fake skies, the quiz)*
3. Trained only on them, POLISH resolves 2–3× finer detail, confirmed by JWST *(network, cat, lenses, JWST)*
4. Faint flux and bright cores remain; next: all of Euclid DR1 *(limitations, Next)*

---

## Opening: from this to THIS (0:00–1:00)

**[0] The canvas.** "Four points; we'll come back to them at the end."
**[1] Title.** "Euclid is ESA's survey telescope: a third of the sky at nearly Hubble sharpness. Today: a network that
makes Euclid sharper still, trained only on TNG50 skies." [2] → "From this…" [3] → "…to THIS." Pause. "The same Euclid
data, POLISHed. Remember this galaxy."

## Point 1 · Euclid is the first wide survey with a sharp, known PSF (1:00–6:00)

**[4] Every survey trades area for sharpness.** Say the point first. On screen: "From the ground you can image the whole
sky, but the atmosphere blurs everything to about an arcsecond." [5] → "From space you're sharp, but the fields are
tiny." [6] The chart: "Each dot is a survey. Up is sharper, right is wider. Nobody is in the top-right corner."

**[7] Each box is one survey's share of the sky.** "The big box is DSS2, photographic plates of the whole sky."
[8] → Pan-STARRS, three quarters. [9] → DES, an eighth.
[10] The 50″ window on the Hubble box: "Hubble's widest survey: this small." [11] → JWST's widest: smaller still.
[12] → JWST's resolution. [13]–[16] → walk back down: Hubble, DES, Pan-STARRS, DSS2. "Same sky, forty years of
telescopes. Area and sharpness never came together."

**[17] Radio already inverts a known PSF.** On screen: "From the ground, the atmosphere's blur changes all night."
[18] → "In radio the PSF is set by the antenna layout and known exactly. That is why POLISH, from Liam's group, can
learn on simulations and work on real data."

**[19] Euclid's PSF: sharp, stable and measured.** "Above the atmosphere, five times sharper than DES, nearly the same
across the sky: a radio-like, known PSF over 14,000 square degrees."
**[20]–[23] Back to the window:** [21] → Euclid in the same window. [22] → Hubble's box. [23] → JWST's box. "Nearly
Hubble-sharp, everywhere." **[24]–[25]** Euclid's box: "over 14,000 square degrees."
**[26]–[27] The chart:** [27] → "Euclid lands in the empty corner: 0.16″ over 14,000 deg². DR1, public since November,
already covers 1,900."

**[28] Lenses: where resolution pays off.** On screen: "A ring's geometry weighs its galaxy." [29] → time delays measure
the Hubble constant. [30] → double-source lenses test cosmology. [31] → "Euclid should find ~170,000 lenses, most only
a few PSF widths across. Resolution decides which ones we can model."

## Point 2 · Simulated TNG50 skies pass for real Euclid data (6:00–9:00)

**[32] The truth: 1,154 TNG50-SKIRT galaxies.** Say the point first. "Real physics and real morphology, which is your
group's work. z ≈ 0, 100 pc per pixel. We take only the morphology; colors come from Q1."
**[33] The simulated truth has its own artifacts.** "Dust-box and box-cut galaxies are left out of training; slabs and
grain we check for in the outputs."
**[34] Each truth becomes a realistic Euclid image.** [35] → measured PSF and 0.1″ pixels. [36] → real Q1 noise,
artifacts and NISP holes. [37] → the recipe in one line.
**[38] Which four are real?** Give the room ~5 s. [39] → "Half are simulated. If you can't tell them apart, neither
can the network."

## Point 3 · Trained only on them, POLISH resolves 2–3× finer detail, confirmed by JWST (9:00–14:30)

**[40] Super-resolution network.** Say the point first. "A residual CNN, VIS + Y + J + H in, the same four bands out on a
0.05″ grid. Arcsinh inputs at eight knees; 15 million steps, about 70 A100-hours."
**[41] What did it learn? Priors on a cat.** [42]–[44] → truth, Euclid, POLISHed. [45]–[47] → face close-up: Euclid,
POLISHed, truth. "It restores fur it has never seen; it learned deconvolution, not galaxies."
**[48]–[49] Back to the quiz:** [49] → all eight POLISHed: "Real or simulated, the network treats them the same way."
**[50] Lenses, POLISHed.** Each lens: zoom, then → POLISHed. Pause on the Q1 ring [51]–[52] and NGC 6505 [53]–[54];
take the four DR1 systems [55]–[62] faster. [63] back to the slide: all six, sharpened.
**[64]–[65] Back to the survey window:** [65] → "The same 50″ window, POLISHed." (← flips back.)
**[66]–[79] Checked galaxy by galaxy against JWST.** Four galaxies; each: zoom (Euclid) → JWST → POLISHed. Flip with
← → on the first one or two. [79] back to the slide: "Against JWST in H band, POLISHed resolves detail two to three
times finer than Euclid: about 0.2″ instead of 0.7″ on bright structure."

## Point 4 · Faint flux and bright cores remain; next: all of Euclid DR1 (14:30–16:30)

**[80] Where it fails.** Say the point first. On screen: "Faint galaxies lose 2–8% of their flux." [81] the flux plot.
[82]–[83] the bright-nucleus close-up, then the holes outlined. [84]–[85] → "Very bright cores can come out with
holes. Both trace back to gaps in the training set."
**[86] The title galaxy again.** "Remember: from this… to THIS." [87] → "And JWST, for comparison." Let it sit.
**[88] Next.** "POLISH all of Euclid DR1, then the full survey, with the same network."
**[89] The QR code.** "The viewer already runs on DR1: scan it. In my last minute, let me show it." Switch to the browser
tab with the viewer open (Euclid DR1, model 271):
https://cutout-production-538f5738.up.railway.app/#ra=228.827579&dec=45.009343&z=5.016&rel=dr1&m=271
Swipe Euclid ↔ POLISHed once, zoom into one galaxy, then come back.

## Close: the four points

**[90] The whole canvas.** Read the four points, one sentence each, then take questions. The QR code stays on screen.

---

### Delivery notes
- Open each row by saying its point; close by reading the four points on the whole canvas.
- Each slide title is the takeaway; say it first, then point at the evidence.
- Pause 2–3 s after each zoom so the room can orient before you talk (Prezi motion-sickness rule).
- Before the talk: open the viewer tab and let it load the first tiles (WebGPU warm-up), so the demo starts instantly.
- Practice out loud at least twice with a timer; the end-to-end target is 16 min before the demo.
