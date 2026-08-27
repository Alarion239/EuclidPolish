# EuclidPolish manuscript figures — first pass

This directory contains nine candidate manuscript figures assembled on
2026-08-12, plus separate synthetic and real-data reconstruction grids added on
2026-08-26. They are deliberately broader than a final paper set so that we can
decide together what belongs in the main text, supplement, or discard pile.

Run `MPLCONFIGDIR=/private/tmp/euclidpolish-matplotlib python3 paper_figures/build_figures.py`
from the repository root to rebuild every PNG. Set
`EUCLIDPOLISH_FIGURE_EXPORTS` if the presentation exports are not in
`/Users/alarion239/Downloads`.

Run
`MPLCONFIGDIR=/private/tmp/euclidpolish-matplotlib python3 paper_figures/build_sr_result_prototype.py`
to rebuild the tightly framed synthetic and real-data grids as PNG and PDF.
The synthetic grid uses local evaluation FITS. The real grid uses matched
field-16 VIS and H_E browser exports in `EUCLIDPOLISH_FIGURE_EXPORTS`, repeating
the available selections to fill five rows. Both grids use equal 4 mm horizontal
and vertical gutters, with the same 4 mm margin on every outer edge. The
synthetic and real-data page headings are intentionally omitted so that the
manuscript's LaTeX captions can supply them.

Run
`pdflatex -output-directory=tmp/pdfs paper_figures/sr_reconstruction_grids_a4.tex`
from the repository root to center the two trimmed PDFs on separate pages of
one LaTeX-ready A4 document.

Five alternative title-only layouts are declared in
`paper_figures/build_sr_template_variants.py`. Run it from the repository root
to write same-stem PNG previews and JSON provenance sidecars here, plus five
standalone PDFs under `output/pdf/`. Then run
`pdflatex -output-directory=output/pdf paper_figures/sr_template_variants_a4.tex`
to assemble `output/pdf/sr_template_variants_a4.pdf`. These variants leave the
existing Figure 10/11 grids untouched; every template has five rows, square
science panels, 4 mm gutters, and 4 mm padding around its title-only content.

The VIS-H_E false colour is a display product, not the network input or a
photometric colour measurement. The model reconstructs a four-plane
`(VIS, Y_E, J_E, H_E)` SR cube. After the per-band asinh display transfer, the
two selected output planes are mixed as
`R = H_E + 0.08 VIS`, `G = 0.54 (VIS + H_E)`, and
`B = VIS + 0.08 H_E`, then clipped and raised to the power `0.92`. Thus VIS is
cyan-blue and H_E is amber-red; the same mixing rule is used for Dirty, SR, and
HR panels. Each JSON sidecar records the exact transfer and source hashes.

The WebUI's **Figures** page also provides a result-grid assembler. Freeze a
matched region in any cutout viewer, then choose **Save crop to results** or
press **S**. The synchronized magnification windows keep their visible
positions when frozen. In the JWST x Euclid viewer, select native F200W before
saving a JWST comparison. The assembler uses saved crops as columns and
tier/band recipes as rows. Saved FITS cubes remain raw; the PNG/PDF renderer
applies its fixed absolute asinh transfer only while drawing the requested
grid.

| File | Proposed role | Current status |
| --- | --- | --- |
| `fig01_pipeline.png` | Six-panel overview of source construction, empirical PSFs, mock Euclid formation, and the network output. | Main-text candidate. It uses established poster assets; the caption must state that this is a schematic/example realization. |
| `fig02_galaxy_population_calibration.png` | Q1 MER+PHZ brightness law and Euclid-measured VIS Sérsic half-light-radius calibration used by the simulator. | Main-text methods candidate. Rendered from active, valid, reviewed calibration fingerprint `93c0d278252b…`. |
| `fig03_stellar_population_calibration.png` | Q1 PHZ × Gaia DR3 × Euclid MER stellar-count and colour calibration. | Methods or supplement candidate. Rendered from active, valid calibration fingerprint `6fbca51854bf…`. |
| `fig04_synthetic_lr_sr_hr_fields.png` | Two complete synthetic fields in Euclid input, super-resolved output, and known high-resolution truth. | Strong qualitative validation candidate. The displayed `HR` title means high-resolution synthetic truth. |
| `fig05_evaluation_morphology_gallery.png` | Four selected LR–SR temperature-composite examples with arc/ring-like morphology. | Draft only. The supplied presentation exports omit HR; re-export the matched HR tier before using this as a validation figure. |
| `fig06_nexus_widefield_comparisons.png` | Wide-field Euclid–SR–NEXUS comparisons for two real fields. | Main-text real-data candidate. NEXUS F200W is a registered external reference, not ground truth or proof of photometric cross-calibration. |
| `fig07_nexus_closeup_comparisons.png` | Matched close-ups of internal structure in two real fields. | Main-text or supplement candidate. Euclid temperature/VIS and JWST F200W are not identical bandpasses; the caption must avoid a pixel-truth claim. |
| `fig08_stress_and_limitations.png` | Saturated-star and compact-source stress cases. | Limitations candidate. Keep this separate from positive examples so the model's current failure modes are explicit. |
| `fig09_ensemble_diagnostics.png` | Spatial-frequency fidelity, member-disagreement versus error, and z-score calibration. | Quantitative-results candidate. Rendered from the current star-containing cache: 14 models and 100 test fields. |
| `fig10_synthetic_reconstruction_grid.png` / `.pdf` | Five-row table of Euclid-like VIS/H_E inputs, SR composites, and known HR truth. | Synthetic-results figure with column titles only and gutter-matched outer margins; the figure title belongs in the LaTeX caption. |
| `fig11_real_reconstruction_grid.png` / `.pdf` | Five-row table of Euclid VIS/H_E inputs, SR composites, and NEXUS F200W comparisons. | Real-data figure with gutter-matched outer margins. NEXUS remains an external reference, not ground truth. |

## Source policy

- Image panels are complete browser publication exports; composition only adds
  white space and panel/row labels. No astronomical pixels are repainted,
  denoised, sharpened, or generated.
- Figures 02, 03, and 09 are generated from repository caches rather than UI
  screenshots.
- NEXUS images are useful external morphological references in the Euclid Deep
  Field North overlap. They do not by themselves establish WCS, PSF,
  photometric, or bandpass equivalence.
- Figure 11's real VIS/H_E composites combine matched, display-stretched browser
  exports with VIS mapped to cyan and H_E mapped to amber. They are display-only
  false colour, not photometric colour measurements.
- Figure 05 is intentionally marked incomplete because LR–SR alone cannot
  validate synthetic reconstruction fidelity when an HR truth image exists.
