"""Presenter notes for the Prezi, one per path step (Prezi shows them in Presenter view).
NOTES[i] is the note for path action i, i.e. click i + 1 in SCRIPT.md. Writes ../presenter_notes.json, which is
pasted into the editor (action.updateNarrative / narrative.updateText) for both the path and the editor path.
Prezi's notes panel and Presenter view only show notes on zoom steps, so each zoom step's note also carries the
notes of the fades that follow it, one "→ " line per click (the fades keep their own one-line note too).
Keep in sync with SCRIPT.md when the path changes (122 actions on 11-23)."""
import json
import os

N = [
    # opening
    "Euclid is ESA's survey telescope: a third of the sky at nearly Hubble sharpness. Today: teaching a network to make Euclid sharper still, using only TNG50 skies.",
    "Remember this galaxy. We'll come back to it at the end.",
    # 1 · the problem
    "Every survey trades area for sharpness.",
    "From the ground you can image the whole sky, but the atmosphere blurs everything to about an arcsecond.",
    "From space you're sharp, but the fields are tiny.",
    "Each dot is a survey. Up is sharper, right is wider. The dashed line is the frontier: nobody is both. The top-right corner is empty.",
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
    "Radio already inverts a known PSF.",
    "From the ground, the atmosphere's blur changes all night and is never known exactly.",
    "In radio the PSF is set by the antenna layout and known exactly; every source carries the same starburst. That is why POLISH, from Liam's group, learns on simulations and works on real data.",
    "Euclid flies above the atmosphere. Its PSF is five times sharper than DES's and nearly the same across the sky: a radio-like, known PSF over 14,000 square degrees.",
    "Back to the window.",
    "Here is Euclid in the same window.",
    "Hubble's box...",
    "...and JWST's. Euclid is nearly Hubble-sharp, everywhere.",
    "And Euclid's share of the sky...",
    "...14,000 square degrees.",
    "Back to the chart.",
    "Euclid lands in the empty corner: the filled dot is DR1, public since November; the open one is the full survey. The frontier moves.",
    "Lenses: where resolution pays off.",
    "A ring's size weighs its galaxy, dark matter included.",
    "Bent arcs reveal dark-matter clumps that emit no light.",
    "Time delays between the images measure the Hubble constant.",
    "Double-source lenses, like these DR1 systems, test cosmology.",
    "Euclid should find ~170,000 lenses, most only a few PSF widths across. Resolution decides which ones we can model.",
    # 2 · the method
    "The truth: 1,154 TNG50-SKIRT galaxies. Real physics and real morphology: your group's work.",
    "z ~ 0, 100 pc per pixel. We take only the morphology; colors come from Q1.",
    "The simulated truth has its own artifacts: dust-box facets, box cuts, slabs of light, particle grain. A network could learn them as real structure, so dust-box and box-cut galaxies are left out of training; slabs and grain we check for.",
    "Each truth becomes a realistic Euclid image.",
    "Measured Euclid PSFs and 0.1-arcsecond pixels.",
    "Real Q1 noise, artifacts and NISP holes.",
    "The recipe in one line.",
    "Which four are real? Give the room about five seconds.",
    "Half are simulated. If you can't tell them apart, neither can the network. That's the test.",
    # 3 · the network and the results
    "A residual CNN: VIS, Y, J, H in; the same four bands out on a 0.05-arcsecond grid. Arcsinh inputs at eight knees let one network see faint outskirts and bright cores alike.",
    "15 million steps, about 70 A100-hours.",
    "What did it learn? Put a cat through the same forward model.",
    "The truth.",
    "What Euclid would see.",
    "POLISHed.",
    "Close-up: Euclid.",
    "POLISHed.",
    "The truth. It restores fur it has never seen: it learned deconvolution, not galaxies.",
    "Back to the quiz.",
    "All eight, POLISHed. Real or simulated, the network treats them the same way.",
    "Back to the lenses.",
    "The Q1 grade-A ring, as Euclid sees it...",
    "...POLISHed: the ring and its knots separate.",
    "NGC 6505, Euclid's Einstein ring...",
    "...POLISHed.",
    "A DR1 lens: Euclid...",
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
    "Stars become points; the deep field sharpens.",
    "A saturated star in Euclid...",
    "...POLISHed: a point.",
    "The Ultra Deep Field in Euclid...",
    "...POLISHed.",
    "Checked galaxy by galaxy against JWST (NEXUS field). Each galaxy: Euclid, then JWST, then POLISHed.",
]
for k in range(1, 11):
    N += [f"Galaxy {k} of 10: Euclid." + (" Flip back and forth on the first two or three." if k == 1 else ""),
          "JWST.", "POLISHed."]
N += [
    "All ten, POLISHed.",
    "Measured against JWST in H band, POLISHed resolves detail two to three times finer than Euclid: about 0.2 arcsec instead of 0.7 on bright structure.",
    "Where it fails.",
    "Faint galaxies lose 2-8% of their flux.",
    "Flux kept vs VIS magnitude: the loss grows toward the faint end.",
    "A very bright nucleus, as Euclid sees it (VIS)...",
    "...POLISHed, with the holes outlined.",
    "Two failure modes.",
    "Very bright cores can come out with holes. Both trace back to gaps in the training set.",
    # close
    "Remember this galaxy? Euclid.",
    "POLISHed.",
    "JWST. Let it sit for a breath.",
    "One: Euclid is the first wide survey with a sharp, stable PSF.",
    "Two: a network trained only on simulated TNG50 skies.",
    "Three: POLISHed images show what Euclid blurs, checked against JWST.",
    "The whole story on one page. Questions.",
]
assert len(N) == 122, len(N)
out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "presenter_notes.json")
json.dump(N, open(out, "w"), indent=0, ensure_ascii=False)
print(out, len(N))
