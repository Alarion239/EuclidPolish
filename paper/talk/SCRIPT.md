# EuclidPOLISH — Hernquist group, Fri 4 Dec 2026 — speaker script

Prezi: prezi.com/p/edit/bwmtjxdbnkqw (93 clicks, 11-25 layout). Backup of the pre-polish version:
prezi.com/p/edit/0nzro8jentfl. The same lines are in Prezi's Presenter view (frames/presenter_notes.py).
Target: ~16 min of talking, the last minute a live demo of the all-sky viewer, then questions.
`[n]` is the click that brings that view up. "→" means the click only fades something in.

**The canvas is the outline.** The zoom-out shows the title and the Next card on top and three rows below; each row's
header is one conclusion, and its cards are the evidence for it. The talk walks the rows in order, top to bottom, left
to right, and ends on the zoom-out, where the three headers are the summary.

1. Euclid: the first optical wide survey with a sharp, stable PSF
2. A network trained only on simulated TNG50 images
3. POLISHed images reveal what Euclid blurs

---

## Opening: from this to THIS (0:00–1:00)

**[0] Overview.** "Here is the whole talk on one page: three points, one per row. We'll come back here at the end."

**[1] A spiral galaxy as Euclid sees it.** "Euclid is ESA's survey telescope: a third of the sky at nearly Hubble
sharpness. Today: a network that makes Euclid sharper still, trained only on TNG50 skies."
[2] → "From this…" [3] → "…to THIS." Pause. "The same Euclid data, POLISHed. Remember this galaxy."

## Row 1 · Euclid: the first optical wide survey with a sharp, stable PSF (1:00–5:30)

**[4] Every survey trades area for sharpness.** Name the row's point first.
- [5] → "From the ground you can image the whole sky, but the atmosphere blurs everything to about an arcsecond."
- [6] → "From space you're sharp, but the fields are tiny."
- [7] Zoom on the chart. "Up is sharper, right is wider. The dashed line is the frontier. The box top-right, wide AND
  sharp, is empty."

**[8] Each box is one survey's share of the sky.** "The big box is DSS2, photographic plates of the whole sky."
[9] → "Pan-STARRS covers three quarters of it." [10] → "DES covers an eighth."
- [11] Zoom into a 50″ window on the Hubble box. "Hubble's widest survey: this small." [12] → "JWST's: smaller still."
- [13] → "And that is JWST's resolution." [14]–[17] → walk back down: Hubble, DES, Pan-STARRS, DSS2.
  "Same sky, forty years of telescopes. Area and sharpness never came together."

**[18] Radio already inverts a known PSF.** [19] → "From the ground, the atmosphere's blur changes all night."
[20] → "In radio the PSF is set by the antenna layout and known exactly. That is why POLISH, from Liam's group, can learn
on simulations and work on real data."

**[21] Euclid's PSF: sharp, stable and measured.** "Euclid flies above the atmosphere. Its PSF is five times sharper than
DES's and nearly the same across the sky: a radio-like, known PSF over 14,000 square degrees."

**[22]–[25] Back to the window:** [23] → Euclid in the same window. [24] → Hubble's box. [25] → JWST's box.
"Euclid is nearly Hubble-sharp, everywhere." **[26]–[27]** Euclid's box: "over 14,000 square degrees."
**[28]–[29] The chart again:** [29] → "Euclid lands inside the wide-and-sharp box: 0.16″ over 14,000 deg². Filled dot:
DR1, public since November; open dot: the full survey. The frontier moves."

## Row 2 · A network trained only on simulated TNG50 images (5:30–9:30)

**[30] The truth: 1,154 TNG50-SKIRT galaxies.** Name the row's point first. [31] → "z ≈ 0, 100 pc per pixel. Real
physics and real morphology, which is your group's work. We take only the morphology; colors come from Q1."

**[32] The simulated truth has its own artifacts.** "Dust-box facets, box cuts, slabs of light, particle grain.
Dust-box and box-cut galaxies are left out of training; slabs and grain we check for in the outputs."

**[33] Each truth becomes a realistic Euclid image.** [34] → measured PSF and 0.1″ pixels. [35] → real Q1 noise,
artifacts and NISP holes. [36] → the recipe in one line.

**[37] Super-resolution network.** "A residual CNN, VIS + Y + J + H in, the same four bands out on a 0.05″ grid. Arcsinh
inputs at eight knees let one network see faint outskirts and bright cores alike." [38] → "15 million steps, about 70
A100-hours."

**[39] Which four are real?** Give the room ~5 s. [40] → the answer: "Half are simulated. If you can't tell them apart,
neither can the network." [41] → all eight POLISHed: "Real or simulated, the network treats them the same way."

## Row 3 · POLISHed images reveal what Euclid blurs (9:30–15:00)

**[42] Back to the survey window.** Name the row's point first. [43] → "The same 50″ window, POLISHed." (← flips back.)

**[44] What did it learn? Priors on a cat.** [45]–[47] → truth, Euclid, POLISHed. [48]–[50] → face close-up: Euclid,
POLISHed, truth. "It restores fur it has never seen; it learned deconvolution, not galaxies."

**[51] Lenses: where resolution pays off.** [52]–[55] → one line per click; close on "Euclid should find ~170,000 lenses,
most only a few PSF widths across." Then each lens: zoom, then → POLISHed. Pause on the Q1 ring [56]–[57] and NGC 6505
[58]–[59]; take the four DR1 systems [60]–[67] faster. [68] back to the card: all six, sharpened.

**[69]–[82] Checked galaxy by galaxy against JWST.** Four galaxies; each: zoom (Euclid) → JWST → POLISHed. Flip back
and forth with ← → on the first one or two. [82] back to the card: "Against JWST in H band, POLISHed resolves detail two
to three times finer than Euclid: about 0.2″ instead of 0.7″ on bright structure."

**[83]–[89] Where it fails.** [84] → "Faint galaxies lose 2–8% of their flux." [85] the flux plot. [86]–[87] the
bright-nucleus close-up, then the holes outlined. [89] → "Very bright cores can come out with holes. Both trace back to
gaps in the training set."

## Close (15:00–17:00)

**[90] Back to the opening galaxy.** "Remember: from this… to THIS." [91] → "And this is JWST." Let it sit for a breath.
**[92] Next.** "POLISH all of Euclid DR1, then the full survey. The viewer already runs on DR1. In my last minute, let me
show it." Switch to the browser tab with the viewer already open (Euclid DR1, model 271):
https://cutout-production-538f5738.up.railway.app/#ra=228.827579&dec=45.009343&z=5.016&rel=dr1&m=271
Swipe Euclid ↔ POLISHed once, zoom into one galaxy, then come back.
**[93] The zoom-out.** Read the three row headers as the summary, one sentence each, then take questions. The QR code on
the Next card opens the same viewer.

---

### Delivery notes
- Open each row by saying its header; close the talk by reading the three headers on the zoom-out.
- Each card title is the takeaway; say it first, then point at the evidence.
- Pause 2–3 s after each zoom so the room can orient before you talk (Prezi motion-sickness rule).
- Fades are cheap; zooms are not. Inside the survey window and the JWST card, flip with ← → to compare.
- Before the talk: open the viewer tab and let it load the first tiles (WebGPU warm-up), so the demo starts instantly.
- Practice out loud at least twice with a timer; the end-to-end target is 16 min before the demo.
