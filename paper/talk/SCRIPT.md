# EuclidPOLISH — Hernquist group, Fri 4 Dec 2026 — speaker script

Prezi: prezi.com/p/edit/bwmtjxdbnkqw (99 clicks, after the 11-23 meeting pass). Backup of the pre-polish
version: prezi.com/p/edit/0nzro8jentfl. The same lines are in Prezi's Presenter view (frames/presenter_notes.py).
Target: ~16 min of talking, the last minute a live demo of the all-sky viewer, then questions.
`[n]` is the click that brings that view up. "→" means the click only fades something in.

---

## Opening: from this to THIS (0:00–1:00)

**[0] Overview.** "Here's the whole talk on one page; we'll come back to it."

**[1] A spiral galaxy as Euclid sees it.**
"Euclid is ESA's survey telescope: a third of the sky at nearly Hubble sharpness. Today: a network that makes
Euclid sharper still, trained only on TNG50 skies."
[2] → "From this…" [3] → "…to THIS." Pause. "The same Euclid data, POLISHed. Remember this galaxy."

## 1 · The problem (1:00–6:00)

**[4] Every survey trades area for sharpness.**
- [5] → "From the ground you can image the whole sky, but the atmosphere blurs everything to about an arcsecond."
- [6] → "From space you're sharp, but the fields are tiny."
- [7] Zoom on the chart. "Each dot is a survey. Up is sharper, right is wider. The dashed line is the frontier. The box
  top-right, wide AND sharp, is empty."

**[8] Each box is one survey's share of the sky.** The same patch near the Hubble Ultra Deep Field.
- "The big box is DSS2, photographic plates of the whole sky." [9] → "Pan-STARRS covers three quarters of it."
  [10] → "DES covers an eighth."
- [11] Zoom into a 50″ window; it opens on the Hubble box. "Hubble's widest survey: this small." [12] → "JWST's widest
  survey: smaller still."
- [13] → "And that is JWST's resolution." [14]–[17] → walk back down: Hubble, DES, Pan-STARRS, DSS2.
  "Same sky, forty years of telescopes. Area and sharpness never came together."

**[18] Radio already inverts a known PSF.**
- [19] → "From the ground, the atmosphere's blur changes all night and is never known exactly."
- [20] → "In radio the PSF is set by the antenna layout and is known exactly. That is why POLISH, from Liam's group, can
  learn on simulations and work on real data."

**[21] Euclid's PSF: sharp, stable and measured.** "Euclid flies above the atmosphere. Its PSF is five times sharper than
DES's and nearly the same across the sky: a radio-like, known PSF over 14,000 square degrees."

**[22]–[25] Back to the window.** [23] → "Here is Euclid in the same window." [24] → Hubble's box. [25] → JWST's box.
"Euclid is nearly Hubble-sharp, everywhere."
**[26]–[27] Euclid's box:** "and over 14,000 square degrees."
**[28]–[29] The chart again:** [29] → "Euclid lands inside the wide-and-sharp box: 0.16″ over 14,000 deg². The filled dot is
DR1, public since November; the open one is the full survey. The frontier moves."

**[30] Lenses: where resolution pays off.** [31]–[34] → one line per click. Close on [34]: "Euclid should find ~170,000
lenses, most only a few PSF widths across. Resolution decides which ones we can model."

## 2 · The method (6:00–9:30)

**[35] The truth: 1,154 TNG50-SKIRT galaxies.** [36] → "z ≈ 0, 100 pc per pixel. Real physics and real morphology, which is
your group's work. We take only the morphology; colors come from Q1."

**[37] The simulated truth has its own artifacts.** "Dust-box facets, galaxies cut by the box, slabs of light, particle
grain. Dust-box and box-cut galaxies are left out of training; slabs and grain we check for in the outputs."

**[38] Each truth becomes a realistic Euclid image.** [39] → measured PSF and 0.1″ pixels. [40] → real Q1 noise,
artifacts and NISP holes. [41] → the recipe in one line.

**[42] Which four are real?** Give the room ~5 s. [43] → the answer. "Half of these are simulated. If you can't tell them
apart, neither can the network. That's the test."

## 3 · The network and what it shows (9:30–15:00)

**[44] Super-resolution network.** "A residual CNN, VIS + Y + J + H in, the same four bands out on a 0.05″ grid. Arcsinh
inputs at eight knees let one network see faint outskirts and bright cores alike." [45] → "15 million steps, about 70
A100-hours."

**[46] What did it learn? Priors on a cat.** [47]–[49] → truth, Euclid, POLISHed. [50]–[52] → face close-up: Euclid,
POLISHed, truth. "It restores fur it has never seen; it learned deconvolution, not galaxies."

**[53]–[54] Quiz payoff.** [54] → all eight fields, POLISHed, in one click. "Real or simulated, the network treats them
the same way."

**[55]–[68] Lenses, POLISHed.** For each lens: zoom, then → POLISHed. Pause on the Q1 ring [56]–[57] and NGC 6505
[58]–[59]; take the four DR1 systems [60]–[67] faster. [68] back to the card: all six, sharpened.

**[69]–[70] Back to the survey window:** [70] → "The same 50″ window we saw at the start, POLISHed." (← flips back.)

**[71]–[84] Checked galaxy by galaxy against JWST.** Four galaxies; each: zoom (Euclid) → JWST → POLISHed. Flip back
and forth with ← → on the first one or two. [84] back to the card: "Against JWST in H band, POLISHed resolves detail two
to three times finer than Euclid: about 0.2″ instead of 0.7″ on bright structure."

**[85]–[91] Where it fails.** [86] → "Faint galaxies lose 2–8% of their flux." [87] the flux plot. [88]–[89] the
bright-nucleus close-up, then the holes outlined. [91] → "Very bright cores can come out with holes. Both trace back to
gaps in the training set."

## Close (15:00–17:00)

**[92] Back to the opening galaxy.** "Remember: from this… to THIS." [93] → "And this is JWST." Let it sit for a breath.
**[94] Summary.** [95]–[97] → the three takeaways, one per click; read each once. [98] → "Next: POLISH all of Euclid DR1,
then the full survey."
**[99] Overview:** "That's the whole story on one page. In my last minute, let me show you the all-sky viewer I made."
Switch to the browser tab with the viewer already open (Euclid DR1, model 271):
https://cutout-production-538f5738.up.railway.app/#ra=228.827579&dec=45.009343&z=5.016&rel=dr1&m=271
Swipe Euclid ↔ POLISHed once, zoom into one galaxy, then back to the overview for questions. The QR code on the summary
card opens the same view.

---

### Delivery notes
- Each title is the takeaway; say it first, then point at the evidence.
- Pause 2–3 s after each zoom so the room can orient before you talk (Prezi motion-sickness rule).
- Fades are cheap; zooms are not. Inside the survey-area window and the JWST card, flip with ← → to compare.
- Before the talk: open the viewer tab and let it load the first tiles (WebGPU warm-up), so the demo starts instantly.
- Practice out loud at least twice with a timer; the end-to-end target is 16 min before the demo.
- The free Prezi plan shows a small "Prezi" logo bottom-left; nothing important sits there.
