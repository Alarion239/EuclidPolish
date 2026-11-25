"""Presenter notes for the Prezi, one per path step (Prezi shows them in Presenter view).
NOTES[i] is the note for path action i, i.e. click i + 1 in SCRIPT.md. Writes ../presenter_notes.json, which is
pasted into the editor (action.updateNarrative / narrative.updateText) for both the path and the editor path.
Prezi's notes panel and Presenter view only show notes on zoom steps, so each zoom step's note also carries the
notes of the fades that follow it, one "→ " line per click (the fades keep their own one-line note too).
Keep in sync with SCRIPT.md when the path changes (90 actions, 11-25: narration order as on 11-23, canvas = four summary points)."""
import json
import os

# 11-25: the narration order is the 11-23 one (title, surveys, radio, Euclid, lenses, TNG50, fake skies, quiz, network,
# cat, quiz payoff, lenses, survey window, JWST, limits, title again, Next, QR); the canvas groups the slides under four
# summary points (rows) with the title slide and a big QR card on the right. The closing view reads the four points.
N = [
    "Euclid is ESA's survey telescope: a third of the sky at nearly Hubble sharpness. Today: a network that makes Euclid sharper still, trained only on TNG50 skies.",
    "From this...",
    "...to THIS: the same Euclid data, POLISHed. Remember this galaxy; we'll come back to it.",
    "Point 1: Euclid is the first wide survey with a sharp, known PSF. Every survey trades area for sharpness. From the ground you can image the whole sky, but the atmosphere blurs everything to about an arcsecond.",
    "From space you're sharp, but the fields are tiny.",
    "Each dot is a survey. Up is sharper, right is wider. Nobody is in the top-right corner: wide AND sharp.",
    "One patch near the Hubble Ultra Deep Field; each box's area is that survey's share of the sky. The big box: DSS2, photographic plates of the whole sky.",
    "Pan-STARRS covers three quarters of it.",
    "DES covers an eighth.",
    "Zoom into a 50-arcsecond window. The box is Hubble's widest survey: this small.",
    "JWST's widest survey: smaller still.",
    "And that is JWST's resolution.",
    "Hubble's resolution.",
    "DES.",
    "Pan-STARRS.",
    "DSS2. Same sky, forty years of telescopes: area and sharpness never came together.",
    "Radio already inverts a known PSF. From the ground, the atmosphere's blur changes all night and is never known exactly.",
    "In radio the PSF is set by the antenna layout and known exactly; every source carries the same starburst. That is why POLISH, from Liam's group, learns on simulations and works on real data.",
    "Euclid flies above the atmosphere. Its PSF is five times sharper than DES's and nearly the same across the sky: a radio-like, known PSF over 14,000 square degrees.",
    "Back to the window.",
    "Here is Euclid in the same window.",
    "Hubble's box...",
    "...and JWST's. Euclid is nearly Hubble-sharp, everywhere.",
    "And Euclid's share of the sky...",
    "...14,000 square degrees.",
    "Back to the chart.",
    "Euclid lands in the empty corner: 0.16 arcsec over 14,000 square degrees. DR1, public since November, already covers 1,900.",
    "Lenses: where resolution pays off. A ring's geometry weighs its galaxy, dark matter included.",
    "Time delays between the images measure the Hubble constant.",
    "Double-source lenses, like these DR1 systems, test cosmology.",
    "Euclid should find ~170,000 lenses, most only a few PSF widths across. Resolution decides which ones we can model.",
    "Point 2: simulated TNG50 skies pass for real Euclid data. The truth: 1,154 TNG50-SKIRT galaxies. Real physics and real morphology: your group's work. z ~ 0, 100 pc per pixel. We take only the morphology; colors come from Q1.",
    "The simulated truth has its own artifacts: dust-box facets, box cuts, slabs of light, particle grain. A network could learn them as real structure, so dust-box and box-cut galaxies are left out of training; slabs and grain we check for.",
    "Each truth becomes a realistic Euclid image.",
    "Measured Euclid PSFs and 0.1-arcsecond pixels.",
    "Real Q1 noise, artifacts and NISP holes.",
    "The recipe in one line.",
    "Which four are real? Give the room about five seconds.",
    "Half are simulated. If you can't tell them apart, neither can the network. That's the test.",
    "Point 3: trained only on them, POLISH resolves two to three times finer detail, confirmed by JWST. A residual CNN: VIS, Y, J, H in; the same four bands out on a 0.05-arcsecond grid. Arcsinh inputs at eight knees let one network see faint outskirts and bright cores alike. 15 million steps, about 70 A100-hours.",
    "What did it learn? Priors on a cat: put a cat through the same forward model.",
    "The truth.",
    "What Euclid would see.",
    "POLISHed.",
    "Close-up: Euclid.",
    "POLISHed.",
    "The truth. It restores fur it has never seen: it learned deconvolution, not galaxies.",
    "Back to the quiz: now run the network on all eight fields.",
    "All eight, POLISHed. Real or simulated, the network treats them the same way.",
    "Back to the lenses: POLISHed, the six from the first row.",
    "The Q1 grade-A ring, as Euclid sees it...",
    "...POLISHed: the ring and its knots separate.",
    "NGC 6505, Euclid's Einstein ring...",
    "...POLISHed.",
    "A DR1 double-source lens: Euclid...",
    "...POLISHed.",
    "Next DR1 lens: Euclid...",
    "...POLISHed.",
    "Euclid...",
    "...POLISHed.",
    "Euclid...",
    "...POLISHed.",
    "All six, sharpened.",
    "Back to the survey window from the start.",
    "The same 50-arcsecond window, POLISHed. Compare it with the Hubble box (left arrow flips back).",
    "Checked galaxy by galaxy against JWST (NEXUS field). Each galaxy: Euclid, then JWST, then POLISHed; flip back and forth with the arrows.",
    "A tidally disturbed galaxy: Euclid.",
    "JWST.",
    "POLISHed: the tail and the dust lane come back.",
    "A barred spiral: Euclid.",
    "JWST.",
    "POLISHed: the bar and the arms.",
    "An interacting pair: Euclid.",
    "JWST.",
    "POLISHed.",
    "A spiral with an inner ring: Euclid.",
    "JWST.",
    "POLISHed: the ring.",
    "All four, POLISHed. Against JWST in H band, POLISHed resolves detail two to three times finer than Euclid: about 0.2 arcsec instead of 0.7 on bright structure.",
    "Point 4: what remains, and what's next. Where it fails. Faint galaxies lose 2-8% of their flux.",
    "Flux kept vs VIS magnitude: the loss grows toward the faint end.",
    "A very bright nucleus, as Euclid sees it (VIS)...",
    "...POLISHed, with the holes outlined.",
    "Two failure modes.",
    "Very bright cores can come out with holes. Both trace back to gaps in the training set.",
    "Remember this galaxy? From this... to THIS.",
    "And JWST, for comparison: the structure is real.",
    "Next: POLISH all of Euclid DR1, then the full 14,000 square degrees, with the same network.",
    "The viewer already runs on DR1: scan the code to try it. In my last minute, let me show it (switch to the browser tab).",
    "The four points: one, Euclid is the first wide survey with a sharp, known PSF. Two, simulated TNG50 skies pass for real Euclid data. Three, trained only on them, POLISH resolves two to three times finer detail, confirmed by JWST. Four, faint flux and bright cores remain; next, all of Euclid DR1. Questions."
]
ZOOM = set([0, 3, 5, 6, 9, 16, 18, 19, 23, 25, 27, 31, 32, 33, 37, 39, 40, 47, 49, 50, 52, 54, 56, 58, 60, 62, 63, 65, 66, 69, 72, 75, 78, 79, 80, 81, 83, 85, 87, 88, 89])
assert len(N) == 90, len(N)
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
