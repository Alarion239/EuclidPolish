"""Presenter notes for the Prezi, one per path step (Prezi shows them in Presenter view).
NOTES[i] is the note for path action i, i.e. click i + 1 in SCRIPT.md. Writes ../presenter_notes.json, which is
pasted into the editor (action.updateNarrative / narrative.updateText) for both the path and the editor path.
Prezi's notes panel and Presenter view only show notes on zoom steps, so each zoom step's note also carries the
notes of the fades that follow it, one "→ " line per click (the fades keep their own one-line note too).
Keep in sync with SCRIPT.md when the path changes (75 actions since the 11-25 review pass; canvas = four summary points)."""
import json
import os

# 11-25: the narration order is the 11-23 one (title, surveys, radio, Euclid, lenses, TNG50, fake skies, quiz, network,
# cat, quiz payoff, lenses, survey window, JWST, limits, title again, Next, QR); the canvas groups the slides under four
# summary points (rows) with the title slide and a big QR card on the right. The closing view reads the four points.
# 11-25 (user): all the text points of a slide are on screen at once; only image reveals take a click (83 actions).
# 11-25 review pass: Lenses card moved to the end of row 1 (narration unchanged); walk-back is JWST, DES, DSS2; two lenses
# zoomed, the four DR1 ones flip on the slide; survey-window callback after the JWST check, with the Hubble box back on
# top; limits card has no zooms (75 actions).
# 11-25 picture pass: two-line survey labels from the start, so the JWST resolution click is gone (merged into the JWST
# box); the JWST card ends on a Euclid / JWST / POLISHed grid faded in after the zoom back (still 75 actions); the fourth
# JWST galaxy is the edge-on disk with two companions (was the inner-ring spiral); the PSF card compares DES with VIS.
# 11-25 (user): opening = From this... -> ...to THIS. -> title and names on the POLISHed frame (both fade OUT); JWST card
# = four images only (no result line, no grid; 74 actions); galaxy 4 = the clumpy irregular; the paper network figure is back.
# 11-27 (user): From this / to THIS on arrival (one fade-out less), no chart zoom, the chart return is the whole card; the
# window and sky-share returns step through Euclid | each survey side by side (5 clicks each); 78 actions.
N = [
    "From this... to THIS: the same Euclid data, POLISHed. Remember this galaxy; we'll check it against JWST at the end.",
    "EuclidPOLISH. Euclid is ESA's survey telescope: a third of the sky at nearly Hubble sharpness. Today: a network that makes Euclid sharper still, trained only on TNG50 skies.",
    "Point 1: Euclid is the first wide survey with a sharp, known PSF. Every survey trades area for sharpness. From the ground you can image the whole sky, but the atmosphere blurs everything to about an arcsecond. From space you're sharp, but the fields are tiny. Each dot is a survey: up is sharper, right is wider. Nobody is in the top-right corner: wide AND sharp.",
    "One patch near the Hubble Ultra Deep Field; each box's area is that survey's share of the sky. The big box: DSS2, photographic plates of the whole sky.",
    'Pan-STARRS covers three quarters of it.',
    'DES covers an eighth.',
    "Zoom into a 50-arcsecond window. The box is Hubble's widest survey: this small.",
    "JWST's widest survey: smaller still, at 0.06 arcsec. Now walk back down: what the ground sees in the same window.",
    'DES, the best wide survey from the ground: 0.9 arcsec.',
    'DSS2. Same sky, forty years of telescopes: area and sharpness never came together.',
    "Radio already inverts a known PSF. From the ground, the atmosphere blurs the PSF unpredictably. In radio the PSF is known exactly from the antenna layout; every source carries the same starburst. So POLISH, from Liam's group, trains on simulations and works on real data.",
    "Euclid flies above the atmosphere. Same scale, same stretch: on the left the ground (DES, 0.9 arcsec seeing), then Euclid's VIS PSF, 0.16 arcsec, five times sharper, and the three NISP bands. Below: VIS in five sky regions, nearly identical: a radio-like, known PSF over 14,000 square degrees.",
    'Back to the window: now Euclid on the left, every other survey on the right.',
    'DSS2, 2 arcsec.',
    'Pan-STARRS, 1.1.',
    'DES, 0.9: Euclid beats every survey from the ground.',
    'Hubble, 0.09: sharper than Euclid...',
    '...and JWST, 0.06. Only these two beat Euclid, and their fields are tiny.',
    "And the sky each one covers: Euclid's 14,000 square degrees on the left.",
    'DSS2: the whole sky.',
    'Pan-STARRS: three quarters of it. Only the all-sky surveys cover more than Euclid.',
    'DES: an eighth of the sky.',
    "Hubble's widest survey: a dot.",
    "JWST's: smaller still.",
    'Back to the chart.',
    'Euclid lands in the empty corner: 0.16 arcsec over 14,000 square degrees. DR1, public since November, already covers 1,900.',
    "Lenses: where resolution pays off. A ring's geometry weighs its galaxy, dark matter included. Time delays between the images measure the Hubble constant. Double-source lenses, like these DR1 systems, test cosmology. Euclid should find ~170,000 lenses, most only a few PSF widths across. Resolution decides which ones we can model.",
    "Point 2: simulated TNG50 skies pass for real Euclid data. The truth: 1,154 TNG50-SKIRT galaxies. Real physics and real morphology: your group's work. z ~ 0, 100 pc per pixel. We take only the morphology; colors come from Q1.",
    "The simulated truth has its own artifacts (nothing written on the slide). Top row: SKIRT's dust grid scatters light into flat facets and shadow rays, up to 70% of a galaxy's light. Bottom: galaxies cut by the periodic box, slabs of scattered light, star-particle grain. A network could learn these as real structure, so dust-box and box-cut galaxies are left out of training; slabs and grain we check for.",
    'Each truth becomes a realistic Euclid image. The recipe in one line.',
    'Measured Euclid PSFs and 0.1-arcsecond pixels.',
    'Real Q1 noise, artifacts and NISP holes.',
    'Which four are real? Give the room about five seconds.',
    "Half are simulated. If you can't tell them apart, neither can the network. That's the test.",
    'Point 3: trained only on them, POLISH resolves two to three times finer detail, confirmed by JWST. A residual CNN: VIS, Y, J, H in; the same four bands out on a 0.05-arcsecond grid. Arcsinh inputs at eight knees let one network see faint outskirts and bright cores alike. 15 million steps, about 70 A100-hours.',
    'What did it learn? Priors on a cat: put a cat through the same forward model.',
    'The truth.',
    'What Euclid would see.',
    'POLISHed.',
    'Close-up: Euclid.',
    'POLISHed.',
    'The truth. It restores fur it has never seen: it learned deconvolution, not galaxies.',
    'Back to the quiz: now run the network on all eight fields.',
    'All eight, POLISHed. Real or simulated, the network treats them the same way.',
    'Back to the lenses: POLISHed, the six from the first row.',
    'The Q1 grade-A ring, as Euclid sees it...',
    '...POLISHed: the ring and its knots separate.',
    "NGC 6505, Euclid's Einstein ring...",
    '...POLISHed.',
    'Back to the slide: the four DR1 systems, as Euclid sees them.',
    'A DR1 double-source lens, POLISHed.',
    'The next DR1 lens, POLISHed.',
    'POLISHed.',
    'POLISHed. All six, sharpened: arcs and knots separate.',
    'JWST confirms it, galaxy by galaxy (NEXUS field). In H band, POLISHed resolves detail down to about 0.2 arcsec; Euclid, about 0.7. (The 0.16 quoted earlier is the PSF of the sharper VIS band.) Each galaxy: Euclid, then JWST, then POLISHed; flip back and forth with the arrows.',
    'A tidally disturbed galaxy: Euclid.',
    'JWST.',
    'POLISHed: the tail and the dust lane come back.',
    'A barred spiral: Euclid.',
    'JWST.',
    'POLISHed: the bar and the arms.',
    'An interacting pair: Euclid.',
    'JWST.',
    'POLISHed.',
    'A clumpy irregular next to a red spiral: Euclid.',
    'JWST.',
    "POLISHed: the clumps, and the neighbor's spiral arms.",
    'All four, POLISHed: two to three times finer than Euclid (say the numbers: about 0.2 vs 0.7 arcsec in H).',
    'And a whole field: back to the 50-arcsecond window from the start.',
    'The same window, POLISHed.',
    'And Hubble in its box, over the POLISHed field: the same structure where Hubble can check it (left arrow flips back).',
    "Point 4: what remains, and what's next. Where it fails: two gaps in the training set. Faint galaxies lose up to about 9% of their flux, because our training noise was more correlated than Euclid's. Very bright cores can come out with holes, because training had no bright, extended cores. (If asked: retraining with the corrected noise model is under way.)",
    'POLISHed: holes in the brightest core.',
    'Remember this galaxy? The POLISHed one from the start.',
    'And JWST, for comparison: all three side by side, one color stretch. The structure is real.',
    'Next: POLISH all of Euclid DR1, then the full 14,000 square degrees, with the same network.',
    'The viewer already runs on DR1: scan the code to try it. In my last minute, let me show it (switch to the browser tab).',
    'The four points: one, Euclid is the first wide survey with a sharp, known PSF. Two, simulated TNG50 skies pass for real Euclid data. Three, trained only on them, POLISH resolves two to three times finer detail, confirmed by JWST. Four, faint flux and bright cores remain; next, all of Euclid DR1. Questions.',
]
ZOOM = set([0, 2, 3, 6, 10, 11, 12, 18, 24, 26, 27, 28, 29, 32, 34, 35, 42, 44, 45, 47, 49, 54, 55, 58, 61, 64, 67, 68, 71, 73, 75, 76, 77])
assert len(N) == 78, len(N)
# a zoom step's note also lists the fades that follow it ("→" lines): Prezi shows notes only on zoom steps
NOTES = []
for i, n in enumerate(N):
    if i in ZOOM:
        j, extra = i + 1, []
        while j < len(N) and j not in ZOOM:
            extra.append("→ " + N[j])
            j += 1
        NOTES.append("\n".join([n] + extra))
    else:
        NOTES.append(n)
out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "presenter_notes.json")
json.dump(NOTES, open(out, "w"), indent=0, ensure_ascii=False)
print(out, len(NOTES))
