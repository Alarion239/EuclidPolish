# Overleaf drop-in: Model subsection + star distributions

Upload this folder's contents to the Overleaf project root (or upload `overleaf_model_stars.zip`).

| File | Where it goes |
|---|---|
| `model_and_stars_options.tex` | Text. Star paragraphs go in *Distributions & Transformations* after the galaxy-colour paragraph; `\subsection{Model}` replaces *Training / Choices / Combiner model*. Option 1 of every paragraph compiles; Options 2 and 3 sit in `\iffalse … \fi`. |
| `architecture_figure.tex` | Replaces the draft's whole `figure*` block (TikZ unchanged except the ⊕ and knee box moved apart; new caption). |
| `references_new.bib` | Append to your `references.bib` (Yu2018, Shi2016, Kingma2015, Loshchilov2017, Odena2016). |
| `images/star_population.pdf` | Star figure (column width, 3.35 in). Replaces `images/fig03_stellar_population_calibration.png`. |
| `images/thumbs/lr_rgb.png`, `sr_rgb.png` | Thumbnails the architecture TikZ needs (`\mdthumbs = images/thumbs`). |
| `images/model_diagram.pdf` | The same architecture figure as a standalone PDF, if you'd rather `\includegraphics` it. |
| `images/star_population.png` | PNG preview of the star figure (not needed by LaTeX). |

Rebuild: `PYTHONPATH=. python paper_figures/build_star_figure.py` (star figure, active v7 star prior);
`latexmk -pdf paper_figures/model_diagram/model_diagram.tex` (standalone diagram).
