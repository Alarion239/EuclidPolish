# Presentation figures

EuclidPolish exports figures from the same cached arrays and display transfer
used by the WebUI. Export does not publish, refit, query an archive, or alter a
field.

The **Figures** page at `/visualization` is the central index. It previews the
galaxy and stellar population calibrations, catalog views, PSF clustering, and
the shared-scale four-band PSF plate. It also links to every interactive viewer
used to build LR/SR/HR, Euclid–JWST, lens, and selected-PSF comparisons.

## Reconstruction and PSF figures

Every shared image viewer has a **Figure** button. A reliable workflow is:

1. Select the tiers that belong in the comparison.
2. Move the pointer to the feature that should be magnified. Scroll vertically
   over the image to change the crop size, then click to select and freeze it.
3. Choose **Figure**. The selected region is projected onto every tier, and
   only those matched crops are exported. With no selection, full images are
   exported instead.

The output is a fixed high-resolution PNG with descriptive panel names outside
the image and physical scale bars when pixel scale is known. Each panel includes
its displayed band, asinh knee, and a heatbar whose ticks are labelled with the
pixel signal in electrons. Titles, parameters, and heatbar ticks use a large presentation type
scale that remains readable after the plate is placed on a slide. The plate
deliberately omits field identifiers and pixel dimensions; keep those in the
slide caption or manuscript caption.
**PNG** remains available for a quick screen-layout capture.

Suggested viewer presets:

| Figure | Page | Tiers |
| --- | --- | --- |
| selected real/synthetic field | Evaluation | LR, HR |
| synthetic reconstruction | Sky records | LR, SR, HR |
| real Euclid reconstruction | Inference | LR and available SR tiers |
| Euclid–JWST reference | Inference or JWST × Euclid | LR, SR, JWST |
| empirical PSFs | PSFs | VIS, Y_E, J_E, H_E |

Changing the index and repeating the workflow produces a consistent series of
roughly ten publication-ready figures without a separate plotting script.

## Population, catalog, and clustering figures

Field statistics provides one three-panel **Q1 brightness × staged TNG50
geometry atlas**. Download PNG for slides, PDF for LaTeX/Keynote placement, or
SVG for vector editing. Its first panel shows the Q1 MER+PHZ VIS 2FWHM raw
counts and the straight log-density law over 14–29 (with 28–29 marked as
extrapolation). Redshift and half-light-radius panels show
brightness-marginalized staged geometry; missing bins remain missing rather
than becoming zero.

The same page provides a four-panel **Q1 PHZ × Gaia DR3 × Euclid MER stellar population
calibration**: Q1 VIS and native Gaia G_AB counts with their shared-slope,
separate-intercept straight fits over 12–25, plus VIS−Y, Y−J, and J−H colour
checks. The plot keeps the fitted true-colour population, estimated true
colours of observed stars, estimated colours with simulated Euclid noise, and
raw Euclid catalogue colours visually distinct.

Catalog and PSF-cluster figures expose a **Download 300 dpi** action above the
rendered plot. The catalog toolbar chooses positions, magnitude distribution,
or per-band saturation before export. These exports, the population atlas, and
the four-band PSF panel share a presentation profile: 20 pt figure titles,
17 pt panel titles, 15 pt axis labels, 12.5 pt ticks, and 11.5 pt legends and
notes. Scientific units are carried by axis or colorbar labels, and categorical
status uses a color-blind-safe blue/orange pairing.
