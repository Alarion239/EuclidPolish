# 2026-10-29/30 battery drivers (archive)

The scratch scripts of the test battery that compared member_257 at its 15M
milestone with production 238 on 2026-10-29/30, copied verbatim from the
temporary session scratchpads they ran in (paths inside them point there and no
longer resolve). They are kept as provenance only: the battery is now
`scripts/model_battery.py` → `euclid_polish/eval/battery/`, which reproduces
their numbers. Large data files were dropped; the small inputs the outskirt test
needs are in `euclid_polish/eval/battery/data/`.

| Folder | What it ran | Battery module |
|---|---|---|
| `tests257/grid/` | AxisExcess grid shares: `prep.py`, `catalog_masks.py`, `measure.py`, `summarize.py`, `final_tables.py`, `figure.py` | `grid.py`, `figures.py` |
| `tests257/realsky/` | JWST resolution (`run_jwst.sh` → `scripts/jwst_resolution.py`, `res_table_new.py`); holes (`holes_run.py`, `poster_figure.py`) | `resolution.py`, `holes.py`, `figures.py` |
| `realsky/` | earlier read-only summarisers (`holes.py`, `res_table.py`, `kron_bins.py`, `syn_bins.py`) | `holes.py`, `resolution.py`, `flux.py` |
| `tests257/flux/` | Kron VIS vs MER (`scripts/flux_conservation.py`, `kron_bins.py`, `kron_paired.py`); synthetic flux (`syn_flux.py`, `syn_bins.py`, `syn_paired.py`) | `flux.py` |
| `tests257/outskirts/` | outskirt injection (`inject_deck_pf.py`, `inject_hosts.py`, `summary.py`, `hosts_summary.py`), real-sky gain profile (`gain_run.py`, `gain_profile.py`, `features.py`), plots | `outskirts.py`, `figures.py` |
| `fields257/` | field figures (`render_synthetic.py`, `real_sky.py`) | `figures.py` |
| `underflux_20261028/` | the 10-28 under-flux investigation that made the outskirt inputs: `synth_run.py fields` (RC_syn, K_syn), `detect.py` (sources_syn, zp) | `outskirts.regenerate_inputs` |
| `kneedep_20261028/`, `lrcheck_20261028/` | the 10-28 synthetic-flux and deck runs the 10-29 ones were checked against | — |

`briefing_2026-10-29.md` is the status briefing of that day. Recorded results:
`*_out.txt`, `*.txt` and `*.json` next to the scripts, and the memory note
`project_long_run_257`.
