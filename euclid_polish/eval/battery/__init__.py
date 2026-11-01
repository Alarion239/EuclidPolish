"""The model test battery: every model test of the 2026-10-29/30 comparison
of member_257 @ 15M with production 238, as one command
(``scripts/model_battery.py``), for any single models (``production``,
``universal:member_NNN``, ``member:NNN``, ``milestone:member_NNN@STEP``).

Tests (one module each; the first model is the baseline of every paired
comparison):

1. :mod:`.synthetic` — ∫PSNR per field and knee on the score's test fields,
   and the error direction of the spectra record;
2. :mod:`.grid` — AxisExcess grid share on real sky, synthetic fields and
   galaxy outskirts;
3. :mod:`.resolution` — ½-coherence resolution against JWST and P8–16 texture;
4. :mod:`.holes` — bright-core holes on 13 NEXUS galaxies and the poster;
5. / 6. :mod:`.flux` — Kron VIS SR/input against MER, and synthetic SR/truth;
7. :mod:`.outskirts` — outskirt light injection and the real-sky gain profile;
8. :mod:`.growth` — curves of growth of the NEXUS galaxies, SR against input;
9. :mod:`.figures` — the tests' figures and the synthetic / real-sky fields.

One model is in memory at a time; its SR outputs go to a cache every test
reads (:mod:`.common`), and per-model results to ``<out>/models/<spec>/`` so a
re-run (another model added, a summary changed) reuses what is stored. Nothing
under ``data/`` is written (the milestone loader's hard-linked ensemble dir
aside). :mod:`.report` writes the comparison table.
"""
