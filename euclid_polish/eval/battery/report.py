"""The battery's comparison table: one column per model, the first model the
baseline of every paired comparison; markdown for reading, JSON for tools.

:data:`REFERENCE` holds the 2026-10-29/30 battery's headline numbers for 238
(production then) and 257 @ 15M; a model with the same checkpoint gets a
reference check (each number within its last quoted digit).
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from euclid_polish.eval.battery.common import ModelInfo

#: The 2026-10-29/30 battery (member_257 at 15M against production 238).
REFERENCE = {
    "member_238": {
        "synthetic.mean": 62.13, "grid.real3": 2.4, "grid.syn3": 7.4, "grid.syn6": 3.2,
        "grid.real_out3": 13.7, "grid.syn_out3": 19.8, "resolution.r1.VIS": 1.4415,
        "resolution.r1.Y_E": 1.2632, "resolution.r1.J_E": 1.2487, "resolution.r1.H_E": 1.1128,
        "resolution.p816": 0.0635,
        "holes.nexus": 11.55, "holes.poster": 3.11, "kron.21-23": 0.981, "kron.23-24": 0.935,
        "kron.24-24.5": 0.914, "kron.24.5-25": 0.905, "synflux.19-23": 0.999, "synflux.23-24": 0.987,
        "synflux.24-25": 0.973, "synflux.25-26": 0.953, "outskirts.real": 0.58},
    "member_257@15000000": {
        "synthetic.mean": 62.82, "grid.real3": 2.1, "grid.syn3": 4.3, "grid.real_out3": 15.7,
        "resolution.p816": 0.0626, "holes.nexus": 11.48, "holes.poster": 2.59, "kron.24.5-25": 0.904,
        "synflux.25-26": 0.988, "outskirts.real": 0.58},
}
#: Quoted decimals of each reference number (the check's tolerance is half the last digit).
_DECIMALS = {"synthetic.mean": 2, "grid": 1, "resolution.r1": 4, "resolution.p816": 4, "holes": 2, "kron": 3,
             "synflux": 3, "outskirts": 2}


def _decimals(key: str) -> int:
    for prefix in sorted(_DECIMALS, key=len, reverse=True):
        if key.startswith(prefix):
            return _DECIMALS[prefix]
    return 3


def reference_id(info: ModelInfo) -> str:
    return info.member if info.step is None else f"{info.member}@{info.step}"


def _get(d, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def headline(s: dict, spec: str) -> dict[str, float | None]:
    """The model's numbers under :data:`REFERENCE`'s keys (percent for the grid)."""
    out: dict[str, float | None] = {}
    syn = _get(s, "synthetic", "models", spec)
    if syn:
        out["synthetic.mean"] = syn["mean"]
    grid = _get(s, "grid", "models", spec) or {}
    for key in ("real3", "syn3", "syn6", "real_out3", "syn_out3"):
        if grid.get(key) is not None:
            out[f"grid.{key}"] = 100 * grid[key]
    res = _get(s, "resolution", "models", spec)
    if res:
        for band, e in res["bands"].items():
            out[f"resolution.r1.{band}"] = e["r1"]
        out["resolution.p816"] = res["p816"]
    holes = _get(s, "holes", "models", spec) or {}
    for name in ("nexus", "poster"):
        if name in holes:
            out[f"holes.{name}"] = holes[name]["hole_pct_mean"]
    for name, cell in (_get(s, "kron", "models", spec) or {}).items():
        out[f"kron.{name}"] = cell["median"]
    for name, cell in (_get(s, "synflux", "models", spec, "sr_over_truth") or {}).items():
        out[f"synflux.{name}"] = cell["median"]
    hosts = _get(s, "outskirts", "hosts", "models", spec, "real NEXUS tiles")
    if hosts:
        out["outskirts.real"] = hosts["mean4"][_ring_index(s)]
    return out


def _ring_index(s: dict) -> int:
    rings = _get(s, "outskirts", "hosts", "rings") or []
    quoted = _get(s, "outskirts", "quoted_ring")
    return rings.index(quoted) if quoted in rings else 4


def matches(got: float | None, want: float, decimals: int) -> bool:
    """``got`` rounded to the reference's quoted decimals equals it."""
    return got is not None and abs(round(got, decimals) - want) <= 0.5 * 10 ** -decimals + 1e-12


def reference_check(s: dict, infos: Sequence[ModelInfo]) -> dict:
    """Per model with a :data:`REFERENCE` entry: ``{key: [reference, now, ok]}``."""
    out = {}
    for info in infos:
        ref = REFERENCE.get(reference_id(info))
        if not ref:
            continue
        now = headline(s, info.spec)
        rows = {}
        for key, want in ref.items():
            got = now.get(key)
            rows[key] = [want, got, matches(got, want, _decimals(key))]
        out[info.spec] = rows
    return out


# --------------------------------------------------------------------------- #
# formatting
# --------------------------------------------------------------------------- #

def pct(x, digits: int = 1) -> str:
    return "–" if x is None else f"{100 * x:.{digits}f}%"


def num(x, digits: int = 3) -> str:
    return "–" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{digits}f}"


def span(values, digits: int = 2) -> str:
    """``[lo, hi]`` as ``lo–hi``."""
    return "–" if not values else "–".join(num(v, digits) for v in values)


def hole_pct(cell: dict | None) -> str:
    value = None if not cell else cell.get("hole_pct_mean")
    return "–" if value is None else f"{value:.2f}%"


def triple(values) -> str:
    """``[lo, median, hi]`` as ``median [lo, hi]``."""
    return "–" if values is None else f"{values[1]:+.3f} [{values[0]:+.3f}, {values[2]:+.3f}]"


def ci(cell: dict | None, digits: int = 3, scale: float = 1.0, key: str = "ci68") -> str:
    if not cell or cell.get("median") is None:
        return "–"
    lo, hi = cell[key]
    return f"{scale * cell['median']:+.{digits}f} [{scale * lo:+.{digits}f}, {scale * hi:+.{digits}f}]"


def _band_list(values, digits: int) -> str:
    return " / ".join(num(v, digits) for v in values)


def table_rows(s: dict, infos: Sequence[ModelInfo]) -> list[tuple[str, str, list[str]]]:
    """``(section, metric, [cell per model])`` rows of the comparison table."""
    rows: list[tuple[str, str, list[str]]] = []
    specs = [i.spec for i in infos]

    def add(section, metric, fn):
        rows.append((section, metric, [fn(spec) for spec in specs]))

    syn = _get(s, "synthetic", "models") or {}
    vs_syn = _get(s, "synthetic", "vs_baseline") or {}
    if syn:
        sec = "Synthetic score"
        add(sec, "∫PSNR, mean of 4 bands (dB)", lambda p: num(_get(syn, p, "mean"), 2))
        add(sec, "∫PSNR VIS / Y / J / H (dB)",
            lambda p: _band_list(_get(syn, p, "integrated") or [None] * 4, 2))
        add(sec, "fields better than the baseline",
            lambda p: f"{vs_syn[p]['fields_better']}/{vs_syn[p]['n_fields']}" if p in vs_syn else "–")
        def knee_delta(p):
            if p not in vs_syn:
                return "–"
            lo, hi = vs_syn[p]["nisp_delta_range_db"]
            return f"{vs_syn[p]['vis_delta_at_lowest_knee_db']:+.2f}; {lo:+.2f}…{hi:+.2f}"

        add(sec, "Δ PSNR vs baseline: VIS at 0.1 e⁻; NISP range (dB)", knee_delta)
        add(sec, "error direction axis/diag, 2–3 px", lambda p: span(_get(syn, p, "axis_diag_2_3")))
        add(sec, "error direction axis/diag, 3–42 px", lambda p: span(_get(syn, p, "axis_diag_3_42")))
        add(sec, "learned output knee VIS / Y / J / H (e⁻)",
            lambda p: _band_list(_get(syn, p, "learned_knees") or [None] * 4, 3))
        add(sec, "score source", lambda p: _get(syn, p, "source") or "–")
    grid = _get(s, "grid", "models") or {}
    if grid:
        sec = "Grid (AxisExcess z ≥ 3, 4 bands pooled)"
        add(sec, "real sky (NEXUS clean tiles)", lambda p: pct(_get(grid, p, "real3")))
        add(sec, "synthetic test fields, z ≥ 3 (z ≥ 6)",
            lambda p: f"{pct(_get(grid, p, 'syn3'))} ({pct(_get(grid, p, 'syn6'))})")
        add(sec, "galaxy outskirts, real", lambda p: pct(_get(grid, p, "real_out3")))
        add(sec, "galaxy outskirts, synthetic", lambda p: pct(_get(grid, p, "syn_out3")))
    res = _get(s, "resolution", "models") or {}
    vs_res = _get(s, "resolution", "vs_baseline") or {}
    if res:
        sec = "JWST resolution (½-coherence, asinh knee at 1σ)"
        add(sec, "1σ raw VIS / Y / J / H (″)",
            lambda p: _band_list([_get(res, p, "bands", b, "r1") for b in ("VIS", "Y_E", "J_E", "H_E")], 4))
        add(sec, "vs baseline (mean of 4 bands)", lambda p: _get(vs_res, p, "r1", "words") or "–")
        add(sec, "P8–16 texture (mean of 4 bands)", lambda p: num(_get(res, p, "p816"), 4))
    holes = _get(s, "holes", "models") or {}
    if holes:
        sec = "Holes (bright-core pixels below ½ LR)"
        add(sec, "NEXUS, 13 galaxy tiles", lambda p: hole_pct(_get(holes, p, "nexus")))
        add(sec, "poster galaxy", lambda p: hole_pct(_get(holes, p, "poster")))
    kron = _get(s, "kron", "models") or {}
    if kron:
        sec = f"Kron VIS SR/input, isolated NEXUS galaxies (n = {_get(s, 'kron', 'n_isolated')})"
        for name in s["kron"]["ranges"]:
            add(sec, f"VIS {name} mag", lambda p, name=name: num(_get(kron, p, name, "median"), 3))
    synflux = _get(s, "synflux", "models") or {}
    if synflux:
        sec = f"Synthetic SR/truth, big Kron apertures (n = {_get(s, 'synflux', 'n_sources')})"
        for name in s["synflux"]["ranges"]:
            add(sec, f"truth VIS {name} mag",
                lambda p, name=name: num(_get(synflux, p, "sr_over_truth", name, "median"), 3))
    out = _get(s, "outskirts") or {}
    if _get(out, "hosts", "models") or _get(out, "deck", "models"):
        ring = out.get("quoted_ring", "2-2.5")
        i = _ring_index(s)
        sec = f"Outskirt injection, light recovered at {ring} r_Kron (host m > 19.5)"
        for test, host, label in (("hosts", "real NEXUS tiles", "real NEXUS tiles as hosts"),
                                  ("hosts", "synthetic test fields", "synthetic test fields as hosts"),
                                  ("deck", "real-like noise", "Gaussian host, real Euclid noise"),
                                  ("deck", "training-like noise", "Gaussian host, training (v5) noise")):
            add(sec, label, lambda p, test=test, host=host: num(
                (_get(out, test, "models", p, host, "mean4") or [None] * 8)[i], 2))
        gain = out.get("gain") or {}
        if gain:
            add(sec, f"real sky A_fit/LR at {ring} r_Kron",
                lambda p: num((_get(gain, p, "ratio_mean4") or [None] * 8)[i], 2))
            add(sec, "real sky blob gain at 1–1.5 r_Kron",
                lambda p: num((_get(gain, p, "gain_mean4") or [None] * 8)[2], 2))
    growth = _get(s, "growth", "models") or {}
    if growth:
        sec = (f"Curve of growth SR/input, NEXUS galaxies (n = {_get(s, 'growth', 'n_galaxies')}, "
               "circular ≤ 3″, median per galaxy)")
        mags = [r for r in s["growth"]["ranges"] if r != "all"]

        def cells(p, band, key, names, digits=3):
            return " / ".join(num(_get(growth, p, "bands", band, n, key, "median"), digits) for n in names)

        add(sec, "VIS F(<0.5″) / F(<1″) / F(<3″), all",
            lambda p: " / ".join(num(_get(growth, p, "bands", "VIS", "all", k, "median"), 3)
                                 for k in ("r0.5", "r1", "r3")))
        add(sec, f"VIS F(<3″), mag {' / '.join(mags)}", lambda p: cells(p, "VIS", "r3", mags))
        add(sec, f"VIS F(<3″) sky-matched, mag {' / '.join(mags)}", lambda p: cells(p, "VIS", "r3_sky", mags))
        add(sec, f"VIS at the Kron aperture, mag {' / '.join(mags)}", lambda p: cells(p, "VIS", "kron", mags))
        add(sec, "Y / J / H F(<3″), all (sky-matched)",
            lambda p: " / ".join(f"{cells(p, b, 'r3', ['all'])} ({cells(p, b, 'r3_sky', ['all'])})"
                                 for b in ("Y", "J", "H")))
        add(sec, "half-light radius SR/input VIS / Y / J / H",
            lambda p: " / ".join(cells(p, b, "r50", ["all"], 2) for b in ("VIS", "Y", "J", "H")))
        if any(_get(growth, p, "bands", "VIS", "all", "epsf_r3") for p in specs):
            seen = "SR ⊛ Euclid ePSF, rebinned"
            add(sec, f"{seen}: VIS F(<0.5″) / F(<1″) / F(<3″), all",
                lambda p: " / ".join(cells(p, "VIS", k, ["all"])
                                     for k in ("epsf_r0.5", "epsf_r1", "epsf_r3")))
            add(sec, f"{seen}: VIS F(<1″), mag {' / '.join(mags)}",
                lambda p: cells(p, "VIS", "epsf_r1", mags))
            add(sec, f"{seen}: Y / J / H F(<0.5″), all",
                lambda p: " / ".join(cells(p, b, "epsf_r0.5", ["all"]) for b in ("Y", "J", "H")))
            add(sec, f"{seen}: half-light radius/input VIS / Y / J / H",
                lambda p: " / ".join(cells(p, b, "epsf_r50", ["all"], 2) for b in ("VIS", "Y", "J", "H")))
        add(sec, "blank-sky SR − input VIS / Y / J / H (e⁻ per SR px)",
            lambda p: _band_list(_get(growth, p, "sky") or [None] * 4, 3))
    syn = _get(s, "growth", "synthetic") or {}
    if syn.get("models"):
        sec = (f"Curve of growth, synthetic test fields (n = {syn['n_sources']} sources, "
               f"{syn['n_fields']} fields, truth VIS mag)")
        mags = [r for r in s["growth"]["ranges"] if r != "all"]

        def syn_cells(p, kind, band, key, names):
            return " / ".join(num(_get(syn, "models", p, kind, band, n, key, "median"), 3) for n in names)

        add(sec, "SR ⊛ ePSF / input: VIS F(<0.5″) / F(<1″) / F(<3″), all",
            lambda p: " / ".join(syn_cells(p, "epsf", "VIS", k, ["all"]) for k in ("r0.5", "r1", "r3")))
        add(sec, f"SR ⊛ ePSF / input: VIS F(<1″), mag {' / '.join(mags)}",
            lambda p: syn_cells(p, "epsf", "VIS", "r1", mags))
        add(sec, "SR ⊛ ePSF / input: Y / J / H F(<0.5″), all",
            lambda p: " / ".join(syn_cells(p, "epsf", b, "r0.5", ["all"]) for b in ("Y", "J", "H")))
        add(sec, f"SR / truth: VIS F(<3″), mag {' / '.join(mags)}",
            lambda p: syn_cells(p, "truth", "VIS", "r3", mags))
        add(sec, "blank-sky SR − truth VIS / Y / J / H (e⁻ per SR px)",
            lambda p: _band_list(_get(syn, "models", p, "sky") or [None] * 4, 3))
    return rows


def paired_rows(s: dict, infos: Sequence[ModelInfo]) -> list[tuple[str, str, list[str]]]:
    """Paired differences against the baseline (same tiles / galaxies / fields)."""
    rows: list[tuple[str, str, list[str]]] = []
    specs = [i.spec for i in infos[1:]]

    def add(section, metric, fn):
        rows.append((section, metric, [fn(p) for p in specs]))

    grid = _get(s, "grid", "vs_baseline") or {}
    if grid:
        sec = "Grid z ≥ 3 share (points, 95 % CI)"
        for key, label in (("real", "real sky"), ("syn", "synthetic"), ("real_out", "outskirts real"),
                           ("syn_out", "outskirts synthetic")):
            add(sec, label, lambda p, key=key: ci(_get(grid, p, key), 1, 100.0, "ci95"))
    kron = _get(s, "kron", "vs_baseline") or {}
    if kron:
        sec = "Kron VIS SR/input (per galaxy, 68 % CI, share > 0)"
        for name in s["kron"]["ranges"]:
            add(sec, f"VIS {name} mag", lambda p, name=name: (
                ci(_get(kron, p, name)) + f" {100 * kron[p][name]['frac_positive']:.0f}%>0"
                if _get(kron, p, name, "frac_positive") is not None else "–"))
    synflux = _get(s, "synflux", "vs_baseline") or {}
    if synflux:
        sec = "Synthetic SR/truth (per source, 68 % CI)"
        for name in s["synflux"]["ranges"]:
            add(sec, f"truth VIS {name} mag", lambda p, name=name: ci(_get(synflux, p, name)))
    out = _get(s, "outskirts") or {}
    for test, hosts in (("hosts", ("real NEXUS tiles", "synthetic test fields")),
                        ("deck", ("real-like noise", "training-like noise"))):
        vs = _get(out, test, "vs_baseline") or {}
        if not vs:
            continue
        sec = f"Outskirt recovery, {test} (field bootstrap, 95 %)"
        for host in hosts:
            for ring in ("2-2.5", "2.5-3"):
                add(sec, f"{host}, {ring} r_Kron",
                    lambda p, vs=vs, host=host, ring=ring: triple(_get(vs, p, host, ring)))
    res = _get(s, "resolution", "vs_baseline") or {}
    if res:
        sec = "JWST resolution vs baseline"
        def ratio_words(p, key):
            words = _get(res, p, key, "words")
            if not words:
                return "–"
            return f"{words} (per band {', '.join(f'{v:.4f}' for v in res[p][key]['per_band'])})"

        for key, label in (("r1", "1σ raw"), ("nc1", "1σ noise-corrected"), ("r03", "0.3σ"), ("r3", "3σ")):
            add(sec, label, lambda p, key=key: ratio_words(p, key))
        add(sec, "P8–16 ratio", lambda p: num(_get(res, p, "p816_ratio"), 3))
    growth = _get(s, "growth", "vs_baseline") or {}
    if growth:
        sec = "Curve of growth F(<3″) SR/input (per galaxy, 68 % CI, share > 0)"

        def paired(p, band, name):
            cell = _get(growth, p, band, name)
            if not cell or cell.get("frac_positive") is None:
                return "–"
            return ci(cell) + f" {100 * cell['frac_positive']:.0f}%>0"

        for name in s["growth"]["ranges"]:
            add(sec, "VIS, all" if name == "all" else f"VIS {name} mag",
                lambda p, name=name: paired(p, "VIS", name))
        for band in ("Y", "J", "H"):
            add(sec, f"{band}, all", lambda p, band=band: paired(p, band, "all"))
    return rows


def _markdown_table(header: list[str], rows: list[tuple[str, str, list[str]]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    last = None
    for section, metric, cells in rows:
        if section != last:
            lines.append(f"| **{section}** |" + " |" * (len(header) - 1))
            last = section
        lines.append(f"| {metric} | " + " | ".join(cells) + " |")
    return lines


def markdown(s: dict, infos: Sequence[ModelInfo], meta: dict) -> str:
    names = [i.name for i in infos]
    lines = [f"# Model battery — {meta.get('created', '')}", "",
             "Models (the first is the baseline of every comparison):", ""]
    for info in infos:
        step = "" if info.step is None else f" @ step {info.step}"
        lines.append(f"- **{info.name}** — `{info.spec}` ({info.member}{step}, "
                     f"checkpoint `{info.fingerprint}`)")
    if meta.get("records_fp"):
        lines += ["", f"Synthetic test records: `{meta['records_fp']}`."]
    if meta.get("limits"):
        lines += ["", f"**Smoke run** ({meta['limits']}): numbers are not comparable with a full battery."]
    lines += ["", *_markdown_table(["metric", *names], table_rows(s, infos))]
    refs = _get(s, "grid", "references") or {}
    if refs:
        def ref(name, key):
            return pct(_get(refs, name, key))

        lines += ["", f"Grid references (no model): JWST F200W real {ref('jwst', 'real3')}, outskirts "
                  f"{ref('jwst', 'real_out3')}; Euclid input real {ref('lr', 'real3')}, synthetic "
                  f"{ref('lr', 'syn3')}; truth synthetic {ref('truth', 'syn3')} (outskirts "
                  f"{ref('truth', 'syn_out3')}); white noise {ref('white', 'real3')}."]
    syn_ref = _get(s, "growth", "synthetic", "reference", "VIS", "all") or {}
    if syn_ref:
        lines += ["", "Curve-of-growth method check (no model): truth ⊛ ePSF / input on the synthetic "
                  "fields, VIS F(<0.5″) / F(<1″) / F(<3″) = "
                  + " / ".join(num(_get(syn_ref, k, "median"), 3) for k in ("r0.5", "r1", "r3")) + "."]
    if len(infos) > 1:
        lines += ["", f"## Paired differences vs {infos[0].name}", "",
                  *_markdown_table(["comparison", *names[1:]], paired_rows(s, infos))]
    check = s.get("reference_check") or {}
    for spec, rows in check.items():
        # Numbers of tests that did not run are left out (battery.json keeps them).
        rows = {key: row for key, row in rows.items() if row[1] is not None}
        if not rows:
            continue
        n_ok = sum(1 for r in rows.values() if r[2])
        lines += ["", f"## Reference check: {spec} against the 2026-10-30 battery "
                  f"({n_ok}/{len(rows)} reproduce)", "",
                  "| number | 2026-10-30 | now | |", "|---|---|---|---|"]
        for key, (want, got, ok) in rows.items():
            lines.append(f"| {key} | {want:g} | {num(got, 4)} | {'✓' if ok else '✗'} |")
    if meta.get("figures"):
        lines += ["", "## Figures", ""] + [f"- `{f}`" for f in meta["figures"]]
    lines += ["", "Definitions: `euclid_polish/eval/battery/` (one module per test); per-test details are in "
              "`battery.json`; resolution changes read \"x% sharper/blurrier\" (mean over the 4 bands of the "
              "½-coherence ratio).", ""]
    return "\n".join(lines)
