# Training-data collages (paper)

Built by `build.py` (`sheets` = picking contact sheets, `final` = the figures) from `selection.json`.
Same colour recipe as the LR/POLISHed/JWST triples (`output/jwst_triples_257`): per-band sky subtraction
(σ-clipped median), solar-referenced calibration (`visualization.color.calibrate`), the triples' Euclid white
balance, R = H, G = (Y+J)/2, B = VIS, Lupton+04 asinh (hue kept), ×1.25 chroma. Width 7.13 in (AASTeX `figure*`).

## euclid_vs_synthetic.{png,pdf}
Top: four real Euclid Q1 MER fields (archive cutouts 023, 141, 066, 192; ADU/s → e⁻ via MAGZERO).
Bottom: four synthetic LR fields from the v6 test records (`data/records_v6_test/dirty_test`, fields 43, 35, 19, 4).
All 25.5″ (255 × 0.1″ px), one stretch for all eight panels (white = 99.7th percentile of the real panels'
intensity, knee = white/40), so populations, colours, PSF wings, spikes and noise compare directly.
Columns are matched loosely by content (bright galaxies, pair/edge-on, bright star, typical field).

## hr_lr_pairs.{png,pdf}
Six synthetic galaxies from the same test records: HR truth (`hr_test`, noise-free sky through the 0.066″ FWHM
target PSF, 0.05″ px) next to its forward-modelled Euclid LR (`dirty_test`: empirical Euclid ePSFs, 0.1″ pixels,
Q1 noise and artefacts), 6.4″ crops. One stretch per pair (white = 99.8th percentile of the HR crop, knee = white/20);
the LR is shown with its native 0.1″ pixels (flux per pixel ÷4 on the HR grid).
Objects: grand-design spiral (f46), tidally disturbed spiral (f79), spiral with a companion (f53),
dusty edge-on disk (f81), clumpy irregular (f35), red edge-on disk (f44).
