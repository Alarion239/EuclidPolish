# Talk plan: EuclidPOLISH at the Hernquist group (Fri 4 Dec 2026, 4 pm)

15–20 min, Prezi. The introduction carries the story: every Euclid image shown early is
"sealed" (small envelope mark in the corner) and gets super-resolved in the payoff.
Audience: the IllustrisTNG group. TNG50 is our ground truth, so they are the hook and the toughest critics.

Visual style:
- **Images = the JWST triples sent to Liam** (`output/jwst_triples_257/`, code in `code/`): full-bleed
  panels, bold white "Euclid / POLISHed / JWST / NIRCam" top-left, band legend in band colours
  bottom-left, coordinates, N/E compass, 1″ bar; one calibration (solar-SED f_ν, field white balance)
  and one Lupton asinh stretch shared across the panels of a comparison. Every callback is built
  with that code, so all payoff images look like one family.
- **Canvas = the weekly reports** (dark #0f1218, big teal section numbers, monospace teal eyebrow
  labels, captions under panels).
- New images I make for the talk (footprint map, HUDF-in-a-tile, ground-PSF animation, TNG panels,
  number cards) follow the same two rules.

---

## Arc and timing (≈ 18.5 min)

| Act | Minutes | Frames |
|---|---|---|
| 0. Cold open | 0:45 | 1 |
| 1. Introduction | 8:00 | 8 |
| 2. Method, with TNG honestly | 5:00 | 6 |
| 3. Payoff: callbacks | 4:00 | 6 |
| 4. Limits and a question for the room | 0:45 | 1 |

## Callbacks (images shown early, super-resolved later)

| # | Shown in | Image | Payoff | Status |
|---|---|---|---|---|
| C1 | Cold open | NEXUS tile f200w-0031, Euclid only (25.5″) | Euclid → POLISHed → JWST, then zoom into the tidal galaxy | **Exists** (`output/jwst_triples_257/00_*`, `01_*`) |
| C2 | "Always from the ground" | One spiral from the NEXUS triples (03 grand-design) as Pan-STARRS sees it, next to Euclid | Four-step ladder: ground → Euclid → POLISHed → JWST | Triple **exists**; Pan-STARRS cutout **needs a download** |
| C3 | "Space: sharp but tiny" | Hubble Ultra Deep Field drawn as a speck on one Euclid EDF-F tile (it is in Q1: tile 102044185) | POLISHed Euclid HUDF next to Hubble XDF | **Needs an SR run** + Hubble XDF cutout **download** |
| C4 | "Euclid is the first" | NGC 6505 Einstein ring (O'Riordan+25), in Q1 EDF-N, tile 102158889 | POLISHed ring | **Needs an SR run**; bright core risks holes, so test first |
| C5 | "Strong lenses" | Q1 grade-A complete ring 102042919, plus a quad and 4 more | 6-lens grid before/after | **Exists** (`output/lenses_ldn1641_257/best/`) |
| C6 | "Quiz: which is real?" | 4 real Q1 + 4 synthetic fields, shuffled | POLISH all 8; synthetic ones shown next to their TNG truth | Quiz **exists** (`paper_figures/generation_collages/euclid_vs_synthetic.png`); SRs **need a run** |
| C7 (stretch) | "DR1-Foundation, 5 days old" | Abell 3128 core from DR1 (cluster + arcs) | POLISHed core | **Needs DR1 access** (public, astroquery.esa.euclid); new data path |

Optional out-of-distribution pair for the cats slide: the Cat's Eye Nebula (NGC 6543, also in Q1 EDF-N, tile 102159486).

---

## Act 0. Cold open (0:45)

**F0. One Euclid tile.** Fly in from the full NEXUS tile (Euclid, colour) to the disturbed galaxy. Envelope mark: C1.
- Say: "This is one 25″ piece of Euclid. Remember it, we come back to it."
- Image: `output/jwst_triples_257/00_f200w-0031_full_tile.png`, Euclid panel only.

## Act 1. Introduction (8:00)

**F1. Why wide surveys.** Rare objects and statistics both need area.
- Image: survey footprints on the sky (SDSS → DES → Legacy → HSC → LSST), drawn by us on one all-sky projection.
- Line: the photographic surveys of the 1950s onward → today's billion-object catalogs.

**F2. Always from the ground** (C2).
- Same spiral: Pan-STARRS (≈ 1.1″ seeing) next to Euclid (0.13″ VIS PSF FWHM measured 0.147″).
- Line: best sites reach ≈ 0.6″, most ≥ 1″. The atmosphere, not the telescope, sets the resolution.

**F3. Space is sharp but tiny** (C3).
- Hubble/JWST image ≈ 10 arcmin² at a time; COSMOS, the largest Hubble mosaic, is 1.7 deg².
- Image: one Euclid tile (0.5° × 0.5°) with the HUDF as a speck inside it. Zoom into the speck.
- Line: space telescopes follow up; they don't survey.

**F4. Super-resolving the ground.** Luo+25 (SDSS/DECaLS → HSC), Akhaury+24.
- Image: an animation of a ground PSF changing night to night (we can draw it from seeing values).
- Line: the blur is never known, so the network has no fixed thing to invert, and nothing wide and sharp to check against.

**F5. Radio did it first: POLISH** (Liam).
- An interferometer's PSF is set by its antenna layout and known exactly.
- POLISH (Connor+22): WDSR network, trained only on simulations, beats CLEAN on PSNR/SSIM, deconvolves real VLA data, runs in seconds on a laptop.
- POLISH++ (Wu+26): wide fields in patches + arcsinh stretch; recovers simulated lenses with image separations near the PSF, ≈ 10× more lenses than CLEAN for DSA-2000.
- Image: POLISH figure (credit) or the DSA-2000 PSF.

**F6. Euclid is the first wide survey with a stable PSF** (C4).
- 14,000 deg² Wide Survey (≈ a third of the sky); VIS + 3 NISP bands.
- PSF: we measured it ourselves from 17,905 stars, 44 sky regions, 1023² kernels. Image: `paper_figures/drafts/psf_gallery.png` (4 bands, spikes).
- **DR1-Foundation came out on 12 Nov**: 1,900 deg², > 1 billion objects, ≈ 15,000 strong-lens candidates. Roman launched 30 Aug: > 5,000 deg² more.
- Image: NGC 6505's Einstein ring, a famous Euclid lens, which is in our Q1 data. Envelope: C4.

**F7. Why resolution matters: strong lenses** (C5).
- What lenses measure: deflector mass incl. dark matter, subhalos, H₀ from time delays, magnified faint sources.
- The compact ones (Einstein radius ≈ PSF) are missed → known lenses are biased to massive deflectors.
- Image: Q1 grade-A complete ring, Euclid only. Envelope: C5.

**F8. To train for this we need the truth.** Real Euclid has no sharp twin, so we simulate the sky.
- Segue: "and the best galaxies to simulate a sky with come from this room": TNG50.

## Act 2. Method, TNG honestly (5:00)

**F9. TNG50-SKIRT Atlas** (Baes+24, Kovačić+25 in Euclid bands).
- 1,154 z = 0 galaxies, 10^9.8–10^12 M☉, 5 projections each, 100 pc/px. 1,074 unique donors in training.
- Image: a grid of raw atlas galaxies (we have them locally in `data/tng_skirt`).

**F10. What we use, and what we don't trust** (say it plainly).
- **z = 0 only.** We tried placing them at z = 0.1–2.5 (Tolman dimming, spectral drift, compactness factor 1.3 because TNG sizes ran 1.24–1.41× large). We dropped it: now TNG gives **morphology only**; colours and SFR come from a quantile-regression forest trained on real Q1 rows; stamps are only ever shrunk.
- **Particle texture.** In faint galaxy light the truth has 22–25% "dropout" pixels; JWST sees 3.9% in real galaxies. The network learns to paint that clumpy texture on real sky (12.4% for the long run). Pre-blurring the truth cut it but cost 0.41 dB.
- **Atlas post-processing edges.** SKIRT's dust octree leaves flat square facets (up to 36% of VIS light in subhalo 117266, 4 of 342 galaxies affected); some galaxies are cut by the periodic box (21 excluded); hard stamp edges taught the network straight lines (now an exponential fade).
- Image: four small panels: clumpy faint light, a dust-box facet, a box-cut galaxy, a stamp edge, each with the fix.
- Tone: these are limits of how we use a z = 0 atlas, not complaints about TNG.

**F11. Building a Euclid sky.**
- Q1-calibrated galaxy counts (match ≤ 9%), stars, measured PSFs, noise from 294 Q1 noise maps, cosmic rays, dead pixels, saturation.
- Fun fact: for months our noise was too correlated because we averaged the dithers; Euclid median-combines (`COMBINET = MEDIAN`). Fixing it took flux recovery in galaxy outskirts from 0.60 back toward 1.
- Image: `paper_figures/generation_collages/hr_lr_pairs.png` (truth vs simulated Euclid).

**F12. Quiz: which four are real?** (C6)
- Show the 8 fields shuffled; the room votes; reveal.
- Image: `euclid_vs_synthetic.png`, re-cut into 8 unlabelled tiles.

**F13. The network.** Architecture figure; zoom into one residual block (Prezi zoom = the magnification cone).
- 8 arcsinh knees in, learned knee out, 16 residual blocks, shared trunk + per-band path.
- 15M steps, 70 A100-hours; 0.4–0.5 dB per doubling of training.
- Image: `images/model_diagram.pdf` → PNG.

**F14. What did it learn? Put cats through it.**
- Cats through the same forward model: the network loses to bicubic (e.g. 27.6 vs 28.8 dB). Its prior is galaxies, which is good news (it isn't a generic sharpener) and the reason TNG matters so much.
- Image: `output/cats_through_257/cats_vis_grid.png`, re-styled.

## Act 3. Payoff (4:00)

Fly back through the envelopes in reverse order of their importance, then end on the biggest picture.

**F15. C6 quiz fields, POLISHed**, with the truth for the 4 synthetic ones.
**F16. C2 ladder**: Pan-STARRS → Euclid → POLISHed → JWST, one galaxy.
**F17. C5 lenses**: 6 Q1 lenses before/after (complete rings, a quad).
**F18. C4 NGC 6505's ring, POLISHed.**
**F19. C3 HUDF**: POLISHed Euclid next to Hubble's deepest image.
**F20. C1 back to the first tile**: Euclid → POLISHed → JWST, then zoom out to the 10 NEXUS triples.
- One number: resolution measured against JWST with Fourier ring correlation in 4 band pairs.

## Act 4. Limits and a question (0:45)

**F21.**
- Open problems, plainly: faint flux on real sky is 0.98 of the catalog at 21–23 mag but 0.90 at 24.5–25 (synthetic: 0.999); a faint pixel-grid pattern in the long run; holes in very bright cores. All are sim-to-real: our synthetic test can't see them.
- Question for the room: "If you were building the training truth from TNG, what would you change: higher-z snapshots, smoother rendering of faint light, something else?"

---

## Fun facts bank (for speaker notes and Q&A)

- 1,292 commits over 8 months, about 5,060 tests.
- The "giants" bug: a missing flag filled every field with ≈ 20 arcsec-scale TNG galaxies (median R_e 1.10″ → 0.22″ once fixed).
- The VIS zeropoint was 0.42 mag off for a while; saturation was once computed per readout instead of per stack (stars saturated 1.5 mag too faint).
- Pixel-shuffle checkerboard, fixed by ICNR initialization and an isotropy penalty.
- The best synthetic model (62.81 dB) painted a pixel-axis grid on 26% of real-sky windows: the best score was not the best model on real sky.
- Ensembles of seed-only members "hallucinate together", so their spread is not an uncertainty; one universal multi-knee model replaced 32 gates (62.07 vs 61.13 dB).
- Prefetch autotuner OOM: host memory crept from 22 GB to > 32 GB over 1.8M steps.
- Forward model 653 → ≈ 240 ms per field; 16 CPUs feed 685 crops/s, 1.8× what an A100 eats.

## Likely questions from this audience

- **"Is the truth sharper than TNG50 resolves?"** TNG50 softening ≈ 0.3 kpc (verify); atlas renders at 100 pc/px; galaxies are only ever shrunk onto a 0.05″ grid. Have the numbers ready.
- **"Does the network learn a TNG prior?"** Yes (cats slide). The faint-light texture is the visible cost.
- **"Why not higher-z snapshots?"** The atlas is z = 0; the forest takes colours from Q1. Ask them.
- **"Uncertainties?"** Not from the ensemble (members are ~fully correlated).

## Production list

Rule: **every frame gets an image made for this talk.** Paper figures and the Liam reports are
sources of data and style, never pasted in. One script per frame under `paper/talk/frames/`,
writing 3840×2160 PNG (or a transparent panel sized to its Prezi frame), dark canvas, the
triples' panel style, no matplotlib chrome. Callback pairs are rendered as separate layers
(Euclid layer, POLISHed layer on top) so Prezi can fade one into the other at the same position.

| Frame | Image built for the talk | Data | Needs |
|---|---|---|---|
| F0 / F20 | NEXUS tile fly-in; Euclid layer, then POLISHed + JWST layers | local SR + JWST cache | — |
| F1 | Survey footprints on one all-sky map (SDSS, DES, Legacy, HSC, LSST, Euclid) | public footprint polygons | small downloads |
| F2 / F16 | Ground → Euclid → POLISHed → JWST ladder, one spiral | local + Pan-STARRS | **download approval** |
| F3 / F19 | One EDF-F tile with the HUDF as a speck; zoom layer; Hubble XDF next to POLISHed | Q1 tile + SR run + XDF | SR run, **download approval** |
| F4 | Ground-PSF animation: one galaxy under changing seeing (MP4) | synthetic | — |
| F5 | POLISH frame: DSA-2000-like PSF vs Euclid PSF, same scale | our PSF + a drawn interferometer PSF | credit POLISH figures |
| F6 | Euclid PSF in 4 bands, as a clean row; DR1 numbers card; NGC 6505 Euclid layer | local ePSFs, Q1 tile | SR run for the payoff |
| F7 / F17 | 6 Q1 lenses, Euclid layer + POLISHed layer | local SRs | restyle |
| F9 | TNG50-SKIRT atlas wall (≈ 40 galaxies, their native 100 pc/px) | `data/tng_skirt` | — |
| F10 | Four "plain downsides" panels: clumpy faint light vs JWST, a dust-octree facet, a box-cut galaxy, a hard stamp edge, each with its fix | local | — |
| F11 | Truth → simulated Euclid build-up (galaxies, + stars, + PSF, + noise, + artifacts) as layers | generator | — |
| F12 / F15 | Quiz: 8 shuffled fields; payoff POLISHed + truth for the synthetic ones | generator + SR run | SR run |
| F13 | Architecture figure as PNG + residual-block layer for the zoom | `architecture_gen.py` | — |
| F14 | Cats: photo → Euclid-like → POLISHed, in colour, 3 cats | `output/cats_through_257` | restyle |
| F21 | Limits: flux-vs-magnitude curve + grid-pattern example, in the deck style | battery outputs | — |
| stretch | DR1 Abell 3128 core, Euclid + POLISHed | DR1 | **DR1 access + approval** |
