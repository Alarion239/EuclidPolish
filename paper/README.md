# Paper: local LaTeX workspace

Open this folder in VS Code (`code paper`). LaTeX Workshop builds `src/POLISHingEuclid.tex`
on save with latexmk (same recipe as ~/Desktop/Notes) and writes the PDF to `pdfs/`.

- `src/aastex701.cls`, `src/aasjournalv7.bst`: AASTeX 7.0.1 from CTAN (TeX Live 2025 has only 6.31).
- Review comments use `todonotes`: `\ab{date}{text}` (Aleksandr's Overleaf comments), `\board{...}`
  (whiteboard outline), `\task{...}`. Set `\commentsfalse` in the preamble to hide them all.
- `src/references.bib` is a stand-in (poster bib + references_new.bib); 11 keys resolve only with
  the Overleaf references.bib.
- Figures under `src/images/` are gitignored copies from `paper_figures/`;
  `images/psfs_idx0_VIS-Y_E_VIS_figure.png` is a placeholder (the real one is only on Overleaf).
