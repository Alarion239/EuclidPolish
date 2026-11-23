# EuclidPOLISH — Hernquist group, Fri 4 Dec 2026 — speaker script

Prezi: prezi.com/p/edit/bwmtjxdbnkqw (128 clicks). Backup of the 11-22 version: prezi.com/p/edit/0nzro8jentfl.
Target: 16–18 min of talking + questions. Times are cumulative guides, not rules.
`[n]` is the click that brings that view up. "→" means the click only fades something in.

---

## Opening (0:00–1:00)

**[0] Overview.** Don't talk over it for long: "Here's the whole talk on one page; we'll come back to it."

**[1] Hero: a barred spiral as Euclid sees it.**
"Euclid is ESA's survey telescope: a third of the sky at nearly Hubble sharpness. Today: how we teach a network to make Euclid sharper still, using only TNG50 skies."

**[2] → title fades.** "Remember this galaxy. We'll come back to it at the end."

## 1 · The problem (1:00–6:00)

**[3] Every survey trades area for sharpness.**
- [4] → "From the ground you can image the whole sky, but the atmosphere blurs everything to about an arcsecond."
- [5] → "From space you're sharp, but the fields are tiny."
- [6] Zoom on the chart. "Each dot is a survey. Up means sharper and right means wider. The dashed line is the frontier: nobody is both. The top-right corner is empty."

**[7] Each box is one survey's share of the sky.** The same patch around the Hubble Ultra Deep Field.
- "The big box is DSS2, photographic plates of the whole sky." [8] → "Pan-STARRS covers three quarters of it." [9] → "DES covers an eighth."
- [10] Zoom into a 50″ window. "Here is DES at that scale." [11] → "Hubble's widest survey: this small." [12] → "JWST: smaller still."
- [13]–[16] Walk back down: "That's JWST's resolution." Then Hubble, DES, Pan-STARRS, and finally DSS2. "Same sky, forty years of telescopes. Area and sharpness never came together."

**[17] Radio already inverts a known PSF.**
- [18] → "In optical, the atmosphere's blur is different every night."
- [19] → "In radio the PSF is set by the antenna layout and is known exactly. Every source carries the same starburst. That is why POLISH, from Liam's group, can train on simulations and work on real data."

**[20] Euclid's PSF: sharp, stable and measured.** "Euclid flies above the atmosphere. Its PSF is five times sharper than DES's and nearly the same across the sky. For the first time an optical survey has a radio-like, known PSF over 14,000 square degrees."

**[21]–[24] Back to the window: Euclid.** [22] → "Here is Euclid in the same window as DSS2." [23] → Hubble's box. [24] → JWST's box. "Euclid is nearly Hubble-sharp, everywhere."
**[25]–[26] Euclid's box:** "and over 14,000 square degrees."
**[27]–[28] The chart again:** "Euclid lands in the empty corner. The frontier moves."

**[29] Lenses: where resolution pays off.** [30]–[34] → one line per click. Close on [34]: "Euclid should find ~170,000 lenses, most only a few PSF widths across. Resolution decides which ones we can model."

## 2 · The method (6:00–10:00)

**[35] The truth: TNG50-SKIRT.** [36] → "1,154 galaxies at z ≈ 0, 100 pc per pixel. Real physics and real morphology, which is your group's work. We take only the morphology; colors come from Q1."

**[37] The simulated truth has its own artifacts.** "Dust-box facets, galaxies cut by the box, slabs of light, particle grain. A network trained on them could learn them as if they were real structure, so we keep the worst cases out and watch for the rest."

**[38] Each truth becomes a realistic Euclid image.** [39] → measured PSF and 0.1″ pixels. [40] → real Q1 noise, artifacts and NISP holes. [41] → the recipe in one line.

**[42] Which four are real?** Give the audience ~5 s. [43] → the answer. "Half of these are simulated. If you can't tell them apart, neither can the network. That's the test."

## 3 · The network, and what Euclid could not see (10:00–16:00)

**[44] One network: four bands, pixels twice as fine.** "A residual CNN, VIS + Y + J + H in, the same four bands out on a 0.05″ grid. Arcsinh inputs at eight knees let one network see faint outskirts and bright cores alike." [45] → "15 million steps, about 70 A100-hours."

**[46] What did it learn? A cat.** [47]–[49] → truth, Euclid, POLISHed. [50]–[52] → face close-up: Euclid, POLISHed, truth. "It restores fur it has never seen; it learned deconvolution, not galaxies."

**[53]–[61] Quiz payoff.** [54]–[61] → the POLISHed version of each field fades in. "Real or simulated, the network treats them the same way."

**[62]–[75] Lenses, POLISHed.** For each lens: zoom, then → POLISHed. Pause on the Q1 ring [64] and NGC 6505 [66]; take the four DR1 systems faster.

**[76] Stars become points; the deep field sharpens.** [77]–[78] star: Euclid → POLISHed. [79]–[80] deep field: Euclid → POLISHed.

**[81]–[111] Checked galaxy by galaxy against JWST.** Each galaxy: zoom (Euclid) → JWST → POLISHed. Use ← to flip back and forth on the first two or three galaxies, then go quickly. If short on time, skip through the last five (three clicks each).

**[112]–[113] Back to the survey patch:** [113] → "The same 50″ window, POLISHed."

**[114]–[119] Where it fails.** [115] → "Faint galaxies lose 2–8% of their flux." [116]–[118] the bright-nucleus close-up and the holes. [119] → "Very bright cores can come out with holes. Both trace back to gaps in the training set."

## Close (16:00–17:30)

**[120] The barred spiral again: Euclid.** "Remember this galaxy?" [121] → POLISHed. [122] → JWST. Let the JWST image sit for a breath.
[123]–[125] → the three takeaways, one per click; read each once.
**[126]–[128]** Overview: "That's the whole story on one page." Then take questions.

---

### Delivery notes
- Each title is the takeaway; say it first, then point at the evidence.
- Pause 2–3 s after each zoom so the room can orient before you talk (Prezi motion-sickness rule).
- Fades are cheap; zooms are not. Inside the survey-area window and the JWST card, flip with ← → to compare.
- The free Prezi plan shows a small "Prezi" logo bottom-left; nothing important sits there.
