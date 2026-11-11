# Overleaf drop-in: Model subsection + star distributions

Upload this folder's contents to the Overleaf project root (or upload `overleaf_model_stars.zip`).

| File | Where it goes |
|---|---|
| `figure_snippets.tex` | Start here: FIND / REPLACE WITH snippets for each figure and text block, with line numbers in `POLISHingEuclid (2).tex`. Reference only; not meant to be compiled. |
| `model_and_stars_options.tex` | Text. Star paragraphs go in *Distributions & Transformations* after the galaxy-colour paragraph; `\subsection{Model}` replaces *Training / Choices / Combiner model*. Option 1 of every paragraph compiles; Options 2 and 3 sit in `\iffalse … \fi`. |
| `architecture_figure.tex` | The network figure (`figure*`: TikZ + caption). In the manuscript, replace the whole inline `figure*` block with `\input{architecture_figure}`. No blank lines inside the TikZ: with `linenumbers` they break the build. |
| `architecture_preamble.tex` | The diagram's packages, colours and macros. In the preamble, replace the `%%%% TIKZ PREAMBLE` block (down to the end of `\mdimstack`) with `\input{architecture_preamble}`. |
| `references_new.bib` | Append to your `references.bib` (Bovy2011, Yu2018, Shi2016, Kingma2015, Loshchilov2017, Odena2016). |
| `images/star_population.pdf` | Star figure (full page width, 7.1 in, one row of four panels; use `figure*`). Replaces `images/fig03_stellar_population_calibration.png`. |
| `images/thumbs/lr_rgb.png`, `sr_rgb.png` | Thumbnails the architecture TikZ needs (`\mdthumbs = images/thumbs`). |
| `images/model_diagram.pdf` | The same architecture figure as a standalone PDF, if you'd rather `\includegraphics` it. |
| `images/star_population.png` | PNG preview of the star figure (not needed by LaTeX). |

Rebuild: `PYTHONPATH=. python paper_figures/build_star_figure.py` (star figure, active v7 star prior);
`latexmk -pdf paper_figures/model_diagram/model_diagram.tex` (standalone diagram).
