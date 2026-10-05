# Presentation figures

EuclidPolish exports figures from the same cached arrays and display transfer
used by the WebUI. Export does not publish, refit, query an archive, or alter a
field.

The **Figures** workspace at `/figures` is the central index (the old
`/visualization` URL redirects to `/figures/plates`). Its Plates tab previews
the galaxy population calibration, galaxy distributions, and stellar
population calibration plates, the NEXUS comparison plates, and the synthetic
poster scene. Sheet builds the publication contact sheet from crops saved in
any viewer, each linking back to the viewer it came from; Studies holds frozen
whole-ensemble comparisons.

## Reconstruction and PSF figures

Every shared image viewer has a **Publication figure** item in its Export
menu. A reliable workflow is:

1. Select the tiers that belong in the comparison.
2. Turn on the magnifier lens (`L`, or hold Alt) and move the pointer to the
   feature that should be magnified. Scroll over the image to change the crop
   size, then click to freeze it.
3. Choose **Publication figure**. The frozen region is projected onto every
   tier, and only those matched crops are exported. With no frozen crop, the
   current pan/zoom view (else the whole image) is exported instead.

The output is a fixed high-resolution PNG with descriptive panel names outside
the image and physical scale bars when pixel scale is known. Each panel includes
its displayed band, its stretch (the asinh knee by default), and a heatbar whose
ticks are labelled with the pixel signal in the tier's unit (electrons for
Euclid tiers, MJy/sr for JWST). Titles, parameters, and heatbar ticks use a large presentation type
scale that remains readable after the plate is placed on a slide. The plate
deliberately omits field identifiers and pixel dimensions; keep those in the
slide caption or manuscript caption.
**Save the frames as PNG**, in the same menu, remains available for a quick
screen-layout capture.

Suggested viewer presets:

| Figure | Page | Tiers |
| --- | --- | --- |
| selected real/synthetic field | Sky › Targets (real), Models › Images (synthetic stamps) | LR, HR |
| synthetic reconstruction | Synthetic › Records | LR, SR, HR |
| real Euclid reconstruction | Sky › Targets | LR and available SR tiers |
| Euclid–JWST reference | Sky › Targets | LR, SR, JWST |
| empirical PSFs | Synthetic › PSF (ePSFs) | VIS, Y_E, J_E, H_E |

Changing the index and repeating the workflow produces a consistent series of
roughly ten publication-ready figures without a separate plotting script.

## Population, catalog, and clustering figures

Figures › Plates provides one three-panel **Galaxy population calibration**
plate. Download PNG for slides, PDF for LaTeX/Keynote placement, or SVG for
vector editing. Its first panel shows the Q1 MER+PHZ VIS 2FWHM raw counts, the
fitted main log-density law, and the continuous bright-bridge/main/flat
generation law over 14–29 (with 28–29 marked as extrapolation). The second
panel shows the circularized Sérsic \(R_e\) marginal and the third the
brightness–size relation; missing bins remain missing rather than becoming
zero.

The same tab provides a four-panel **Stellar population calibration** plate:
Q1 PHZ VIS counts with the Q1-normalized straight law over 12–25, plus VIS−Y,
Y−J, and J−H colour checks. The native Gaia G_AB counts and their shared-slope
fit are drawn only when the paper figure asks for them (`include_gaia=True`).
The plot keeps the fitted true-colour
population, estimated true colours of observed stars, estimated colours with
simulated Euclid noise, and raw Euclid catalogue colours visually distinct.

The catalogue and PSF-cluster views (Synthetic › PSF) are interactive charts
with their own PNG and CSV downloads. The calibration plates and the Studies
chart exports share a presentation profile: 20 pt figure titles, 17 pt panel
titles, 15 pt axis labels, 12.5 pt ticks, and 11.5 pt legends and notes.
Scientific units are carried by axis or colorbar labels.
