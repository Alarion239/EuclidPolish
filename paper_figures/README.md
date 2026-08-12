# EuclidPolish manuscript figures — first pass

This directory contains nine candidate manuscript figures assembled on
2026-08-12. They are deliberately broader than a final paper set so that we
can decide together what belongs in the main text, supplement, or discard pile.

Run `MPLCONFIGDIR=/private/tmp/euclidpolish-matplotlib python3 paper_figures/build_figures.py`
from the repository root to rebuild every PNG. Set
`EUCLIDPOLISH_FIGURE_EXPORTS` if the presentation exports are not in
`/Users/alarion239/Downloads`.

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

## Source policy

- Image panels are complete browser publication exports; composition only adds
  white space and panel/row labels. No astronomical pixels are repainted,
  denoised, sharpened, or generated.
- Figures 02, 03, and 09 are generated from repository caches rather than UI
  screenshots.
- NEXUS images are useful external morphological references in the Euclid Deep
  Field North overlap. They do not by themselves establish WCS, PSF,
  photometric, or bandpass equivalence.
- Figure 05 is intentionally marked incomplete because LR–SR alone cannot
  validate synthetic reconstruction fidelity when an HR truth image exists.
