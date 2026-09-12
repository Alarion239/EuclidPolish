"""Standalone scientific report; figures and results require no web application."""

from __future__ import annotations

import base64
import hashlib
import html
import json
from collections import Counter
from functools import partial
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from . import BANDS
from .archive import DPDD, digest, read_json, save_json, utc_now
from .numerics import amplitude_decision, clustered_interval


def number(value, digits=3):
    return f"{value:.{digits}g}" if isinstance(value, (int, float)) else "unavailable"


def interval_text(value):
    return "unavailable" if value is None else f"[{number(value[0])}, {number(value[1])}]"


def image_tag(path, alt):
    data = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f'<img loading="lazy" src="data:image/png;base64,{data}" alt="{html.escape(alt)}">'


def cached_figure(root, stem, records, render):
    """Resume report rendering only when input arrays and renderer are unchanged."""
    for row in records:
        if "arrays" in row and digest(root / row["arrays"]) != row["arrays_sha256"]:
            raise ValueError("Measurement arrays changed since measurement")
    fingerprint = hashlib.sha256(
        json.dumps(
            {
                "records": records,
                "renderer_sha256": digest(Path(__file__)),
                "matplotlib": matplotlib.__version__,
                "numpy": np.__version__,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    path = root / "figures" / (stem + ".png")
    record_path = path.with_suffix(".json")
    if path.exists() and record_path.exists():
        record = read_json(record_path)
        if record["fingerprint"] == fingerprint and record["png_sha256"] == digest(path):
            return path
    path = render()
    save_json(record_path, {"fingerprint": fingerprint, "png_sha256": digest(path)})
    return path


def panel(ax, array, title, mode="image"):
    a = np.asarray(array, float)
    values = a[np.isfinite(a)]
    if mode == "mask":
        im = ax.imshow(a, origin="lower", cmap="gray_r", vmin=0, vmax=1)
    elif mode == "z":
        im = ax.imshow(a, origin="lower", cmap="RdBu_r", vmin=-5, vmax=5)
    elif mode == "flags":
        im = ax.imshow(np.where(a > 0, np.log2(np.maximum(a, 1)) + 1, 0), origin="lower", cmap="magma")
    elif mode == "rms":
        logged = np.full(a.shape, np.nan)
        logged[a > 0] = np.log10(a[a > 0])
        finite = logged[np.isfinite(logged)]
        lo, hi = np.quantile(finite, [0.01, 0.99]) if finite.size else (0, 1)
        im = ax.imshow(logged, origin="lower", cmap="viridis", vmin=lo, vmax=hi)
    elif values.size:
        lo, hi = np.quantile(values, [0.01, 0.99])
        im = ax.imshow(a, origin="lower", cmap="gray" if mode == "image" else "viridis", vmin=lo, vmax=hi)
    else:
        im = ax.imshow(a, origin="lower", cmap="gray")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02)


def mer_figure(root, patch_id, rows):
    figure, axes = plt.subplots(4, 5, figsize=(15, 11), layout="constrained")
    for index, band in enumerate(BANDS):
        row = next((r for r in rows if r["band"] == band), None)
        if row is None or row["status"] != "measured":
            for ax in axes[index]:
                ax.axis("off")
            axes[index, 0].text(0.1, 0.5, f"{band}: unavailable", transform=axes[index, 0].transAxes)
            continue
        path = root / row["arrays"]
        if digest(path) != row["arrays_sha256"]:
            raise ValueError("MER measurement arrays changed since measurement")
        with np.load(path) as a:
            panel(axes[index, 0], a["science"], f"{band}: science")
            panel(
                axes[index, 1],
                a["official_rms"],
                f"log10(total RMS); median {number(row['official_rms_median'])}",
                "rms",
            )
            panel(axes[index, 2], a["flags"], "Flags: log2(value)+1", "flags")
            panel(
                axes[index, 3],
                a["legacy_mask"],
                f"Legacy excluded: {row['legacy_mask_fraction']:.1%}",
                "mask",
            )
            panel(
                axes[index, 4],
                np.where(a["legacy_mask"], np.nan, a["legacy_residual"]),
                f"Used residual; MAD {number(row['legacy_mad'])}",
            )
    figure.suptitle(
        f"{patch_id} · reference-stack-equivalent electrons/pixel\n"
        "Image display clipped at 1st/99th percentiles; measurements retain the full distributions.",
        fontsize=12,
    )
    path = root / "figures" / (patch_id + "_mer.png")
    figure.savefig(path, dpi=105)
    plt.close(figure)
    return path


def difference_figure(root, row):
    figure, axes = plt.subplots(2, 3, figsize=(12, 7), layout="constrained")
    path = root / row["arrays"]
    if digest(path) != row["arrays_sha256"]:
        raise ValueError("Difference arrays changed since measurement")
    with np.load(path) as a:
        panel(axes[0, 0], a["matched_a"], "Background-subtracted, matched A")
        panel(axes[0, 1], a["matched_b"], "Background-subtracted, matched B")
        panel(axes[0, 2], np.where(a["valid"], a["difference"], np.nan), "Difference A − B")
        panel(axes[1, 0], a["z"], "Z = difference / predicted RMS", "z")
        panel(axes[1, 1], a["valid"], "Valid transformation footprints", "mask")
        z = a["z"][np.isfinite(a["z"])]
        if z.size:
            counts, edges = np.histogram(z, bins=np.linspace(-8, 8, 81))
            axes[1, 2].stairs(counts / (z.size * np.diff(edges)), edges, color="#1a6c8e")
        x = np.linspace(-8, 8, 500)
        axes[1, 2].plot(x, np.exp(-x * x / 2) / np.sqrt(2 * np.pi), color="#b44b32", label="N(0,1)")
        axes[1, 2].set_yscale("log")
        axes[1, 2].set_xlabel("Z (display range ±8)")
        axes[1, 2].legend()
        axes[1, 2].set_title(
            f"All-tail statistics include pixels outside plot\nstd(Z)={number(row['statistics']['std_z'])}",
            fontsize=9,
        )
    figure.suptitle(
        f"{row['patch_id']} · {row['band']} · pair {row['pair_index']} · "
        f"{row['pixel_scale_arcsec']} arcsec/pixel"
    )
    path = root / "figures" / f"{row['patch_id']}_{row['band']}_pair{row['pair_index']}.png"
    figure.savefig(path, dpi=105)
    plt.close(figure)
    return path


def table(headers, rows):
    return (
        "<div class='table-wrap'><table><thead><tr>"
        + "".join(f"<th>{html.escape(h)}</th>" for h in headers)
        + "</tr></thead><tbody>"
        + "".join(
            "<tr>" + "".join(f"<td>{html.escape(str(cell))}</td>" for cell in row) + "</tr>" for row in rows
        )
        + "</tbody></table></div>"
    )


def exposure_summary(rows):
    summary = {}
    for band in BANDS:
        available = [
            r
            for r in rows
            if r["band"] == band and r["status"] == "measured" and r["stratum"] != "bright-star stress case"
        ]
        by_pointing = {}
        for row in available:
            by_pointing.setdefault(row["sample_id"], []).append(row)
        values = [float(np.mean([r["statistics"]["std_z"] for r in group])) for group in by_pointing.values()]
        ancestry = [
            sorted({e for r in group for e in r["input_exposure_ids"]}) for group in by_pointing.values()
        ]
        result = clustered_interval(values, ancestry)
        result["amplitude_target"] = amplitude_decision(result["ci95"])
        if band == "VIS":
            result["amplitude_target"] = "provisional: PSF grid coordinates unverified"
        result["pair_count"] = len(available)
        result["weighting"] = (
            "Average disjoint pairs within each pointing, then equal weight "
            "per pointing; cluster bootstrap by shared native exposure"
        )
        summary[band] = result
    return summary


def pointing_diagnostics(rows):
    """Descriptive associations on equally weighted pointings, never pixels."""
    records = []
    associations = {}
    for band in BANDS:
        groups = {}
        for row in rows:
            if (
                row["band"] == band
                and row["status"] == "measured"
                and row["stratum"] != "bright-star stress case"
            ):
                groups.setdefault(row["patch_id"], []).append(row)
        for patch_id, group in groups.items():
            brightness, correlations, apertures = [], [], []
            for row in group:
                stats = row["statistics"]
                strata = stats["brightness_strata"]
                if (
                    len(strata) >= 3
                    and min(strata[0]["count"], strata[2]["count"]) >= 10
                    and strata[0]["std_z"] > 0
                ):
                    brightness.append(strata[2]["std_z"] / strata[0]["std_z"])
                correlations.extend(
                    c["covariance_z"] / stats["std_z"] ** 2
                    for c in stats["spatial_covariance"]
                    if c["lag_yx"] in ([0, 1], [1, 0])
                    and c["pairs"] >= 64
                    and c["covariance_z"] is not None
                    and stats["std_z"] > 0
                )
                apertures.extend(
                    a["measured_sum_variance"] / a["mean_predicted_sum_variance"]
                    for a in stats["apertures"]
                    if a["square_side_pixels"] == 3
                    and a["count"] >= 8
                    and a["measured_sum_variance"] is not None
                    and a["mean_predicted_sum_variance"] > 0
                )
            records.append(
                {
                    "band": band,
                    "patch_id": patch_id,
                    "std_z": float(np.mean([r["statistics"]["std_z"] for r in group])),
                    "valid_fraction": float(np.mean([r["statistics"]["valid_fraction"] for r in group])),
                    "native_exposure_count": group[0]["native_exposure_count"],
                    "bright_to_faint_scatter": float(np.mean(brightness)) if brightness else None,
                    "lag_one_correlation": float(np.mean(correlations)) if correlations else None,
                    "aperture_variance_ratio_3px": float(np.mean(apertures)) if apertures else None,
                }
            )
        associations[band] = {}
        for metric in (
            "valid_fraction",
            "native_exposure_count",
            "bright_to_faint_scatter",
            "lag_one_correlation",
        ):
            pairs = [(r[metric], r["std_z"]) for r in records if r["band"] == band and r[metric] is not None]
            associations[band][metric] = {
                "pointings": len(pairs),
                "spearman_rho": float(spearmanr(*zip(*pairs, strict=True)).statistic)
                if len(pairs) >= 3 and len({x for x, _ in pairs}) > 1 and len({y for _, y in pairs}) > 1
                else None,
            }
    return {
        "pointings": records,
        "associations_with_std_z": associations,
        "interpretation": "Descriptive associations only, no causal attribution or significance claim; "
        "pair statistics averaged within each pointing.",
    }


def build_report(root):
    root = Path(root)
    manifest, summary = read_json(root / "manifest.json"), read_json(root / "summary.json")
    (root / "figures").mkdir(exist_ok=True)
    mer = summary["mer_measurements"]
    differences = summary.get("exposure_measurements", [])
    exp_summary = exposure_summary(differences)
    diagnostics = pointing_diagnostics(differences)
    summary["exposure_pointing_diagnostics"] = diagnostics
    if (root / "validation.json").exists():
        summary["implementation_validation"] = read_json(root / "validation.json")
    associations = {}
    for band in BANDS:
        central = [
            r for r in mer if r["band"] == band and r["kind"] == "central" and r["status"] == "measured"
        ]
        associations[band] = {}
        ratio = [r["mad_to_total_rms_ratio"] for r in central]
        for key in ("science_p99_native", "legacy_mask_fraction", "official_rms_median", "flagged_fraction"):
            values = [r[key] for r in central]
            associations[band][key] = (
                float(spearmanr(values, ratio).statistic) if len(set(values)) > 1 else None
            )
    summary["mer_descriptive_associations"] = {
        "spearman_rho": associations,
        "meaning": (
            "Associations with MAD/total-RMS ratio across central pointings; "
            "no independence/significance claim"
        ),
    }
    summary["exposure_pointing_summary"] = exp_summary
    stress_rms = []
    for row in mer:
        if row["kind"] == "extreme" and row["status"] == "measured":
            with np.load(root / row["arrays"]) as a:
                rms, flags = a["official_rms"], a["flags"]
                maximum = float(np.nanmax(rms))
                at_max = rms == maximum
                stress_rms.append(
                    {
                        "band": row["band"],
                        "median": row["official_rms_median"],
                        "maximum": maximum,
                        "fraction_at_maximum": float(at_max.mean()),
                        "fraction_at_maximum_with_invalid_bit_clear": float(
                            np.mean(((flags & 1) == 0)[at_max])
                        ),
                    }
                )
    summary["stress_rms_range_diagnostics"] = stress_rms
    cases = [r for r in mer if "saved_example" in r]
    complete_mer = len([r for r in mer if r["status"] == "measured"]) == 192
    audited = {(r["patch_id"], r["band"]) for r in differences}
    expected = {(s["patch_id"], b) for s in manifest.get("validation_selection", []) for b in BANDS}
    has_all_bands = all(any(r["status"] == "measured" and r["band"] == b for r in differences) for b in BANDS)
    summary["assessment_complete"] = bool(
        complete_mer
        and len(cases) == 4
        and all(r["saved_example"]["passed"] for r in cases)
        and len(expected) == 40
        and expected <= audited
        and has_all_bands
    )
    summary["completion_definition"] = (
        "Four-band measurements, all selected cases audited, examples "
        "reproduced, limits explicitly recorded; scientific agreement not "
        "required"
    )
    summary["report_generated_at"] = utc_now()
    summary["report_manifest_sha256"] = digest(root / "manifest.json")
    summary["implementation_sha256"] = {p.name: digest(p) for p in Path(__file__).parent.glob("*.py")}
    limitations = [
        "Official MER RMS maps represent total uncertainty, including "
        "source photon noise; they are not background-only measurements.",
        "Tile-to-observation associations are available. Exact per-pixel "
        "LayersStorage, rejection decisions, and MER contributing exposure"
        " lists were not supplied by the queried public metadata.",
        "Conditional Monte Carlo medians use actual retrieved exposure "
        "times, photometric scales and bilinear weights. They are not a "
        "verified reproduction of the MER stack.",
        "Native RMS products do not specify pixel covariance or covariance"
        " with the supplied background model. Predictions assume diagonal "
        "native covariance; transformation-induced covariance is retained.",
        "VIS PSF snapshot coordinates are absent from the released grid "
        "header. Equal-width grid-cell mapping is inferred; VIS matching "
        "results remain provisional.",
        "The NIR supplied PSF is constant per detector. Finite PSF "
        "support, sampling, spatial variation, and background/registration"
        " residuals can affect differences; residuals are not clipped "
        "away.",
        "The 44-pointing survey uses one central 256-pixel patch per "
        "pointing, plus four saved examples; it does not rerun all 4,400 "
        "original VIS tiles.",
        "Uncertainty intervals group pointings sharing exposures. Patch "
        "blocks and overlapping data do not count as independent "
        "observations. Small group counts limit precision.",
        "Retrieved bytes have SHA256 checksums verified on cache reuse. "
        "Authoritative full-source checksums were unavailable; multipart "
        "S3 ETags are recorded but not interpreted as checksums.",
        "Bright-star MER RMS maps contain repeated, extremely large values while the INVALID bit "
        "is clear. These sentinel-like values are retained, not removed using an invented threshold. "
        "Their presence limits interpretation of the official median and especially the mean-square RMS.",
    ]
    summary["limitations"] = limitations
    limitations.append(
        "NIR background headers can retain earlier astrometry. Subtraction uses matching native DET_ID "
        "and pixel indices, as defined by the product documentation; the sky-WCS discrepancy is recorded."
    )
    recommendations = [
        "Do not replace background noise with the median of a total-RMS "
        "map. Preserve the distinction between background scatter, source "
        "photon variance, and processing covariance.",
        "Retain source brightness, saturation/halo diagnostics, coverage, "
        "and aperture-sum variance in any future estimator; a single "
        "global MAD cannot establish these properties.",
        "Use the exposure-difference residuals and connected-pointing "
        "intervals to decide whether a variance model meets the "
        "predeclared 5% amplitude target. Keep unexplained residual "
        "structure visible.",
        "Obtain exact MER layer/rejection metadata and verified VIS PSF "
        "snapshot coordinates before using this diagnostic to certify a "
        "replacement generator against MER noise.",
        "Keep active calibrations and generation unchanged until a "
        "separate, reviewed generation-model change is supported by the "
        "measured covariance and source-noise evidence.",
    ]
    summary["recommendations"] = recommendations
    stress = next(
        (r for r in mer if r["kind"] == "extreme" and r["band"] == "VIS" and r["status"] == "measured"), None
    )
    if stress:
        recommendations.insert(
            0,
            (
                f"The saved bright-star VIS patch has legacy MAD {stress['legacy_mad']:.3f}, "
                f"versus median official total RMS {stress['official_rms_median']:.3f} "
                f"({stress['mad_to_total_rms_ratio']:.2f} times larger). The retained halo structure "
                "shows why the current masked statistic cannot alone establish a noise amplitude."
            ),
        )
    save_json(root / "summary.json", summary)
    parts = [
        "<!doctype html><html lang='en'><meta charset='utf-8'><meta "
        "name='viewport' content='width=device-width, initial-scale=1'>",
        "<title>Euclid Q1 · Four-band noise assessment</title><style>"
        "body{font:16px/1.55 system-ui,sans-serif;color:#172d38;background"
        ":#f8fafb;margin:0}main{max-width:1280px;margin:auto;padding:36px}"
        "h1{font-size:36px;line-height:1.15}h2{margin-top:42px}p{max-width"
        ":1000px}.note{border-left:4px solid #b96a28;padding:14px "
        "20px;background:#fff5e9}"
        "table{border-collapse:collapse;width:100%;font-size:14px;backgrou"
        "nd:white}td,th{padding:9px 12px;border-bottom:1px solid "
        "#dde5e9;text-align:left}"
        ".table-wrap{overflow:auto}th{background:#e8f0f4}details{margin:12"
        "px 0;background:white;border:1px solid "
        "#dae4e8;border-radius:6px;padding:12px}"
        "summary{cursor:pointer;font-weight:600}img{width:100%;height:auto"
        "}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}"
        "a{color:#126b91}code{background:#e8eff2;padding:2px "
        "5px}li{margin:8px "
        "0}.small{font-size:13px;color:#52636b}</style><main>",
        "<p class='small'>Euclid Q1_R1 · Independent diagnostic · VIS / Y "
        "/ J / H</p><h1>Four-band noise assessment</h1>",
        f"<p>Generated {html.escape(summary['report_generated_at'])}. "
        f"{len([r for r in mer if r['status'] == 'measured'])}/192 MER patch-band measurements; "
        f"{len([r for r in differences if r['status'] == 'measured'])} "
        "disjoint exposure-pair measurements.</p>",
        "<p class='note'>"
        + (
            "Assessment delivered with the documented limits below."
            if summary["assessment_complete"]
            else "Assessment incomplete: exposure acquisition/measurement or "
            "required validation remains outstanding."
        )
        + " The diagnostic target is 5% agreement in noise amplitude, "
        "assessed with 95% uncertainty intervals. Disagreement is a "
        "result; insufficient precision is reported explicitly.</p>",
        "<h2>Definitions and four-band comparison</h2><p>Legacy MAD is the"
        " current source-masked, plane-subtracted estimator, applied to "
        "the same patches. "
        "Official RMS is total pixel uncertainty. Their ratio compares "
        "different quantities and is not a pass/fail test of background "
        "noise. "
        "All displayed amplitudes use reference-stack-equivalent "
        "electrons/pixel; native values, product units and each conversion"
        " are retained in the numerical results.</p>",
    ]
    band_rows = []
    for band in BANDS:
        overall = summary["four_band_summary"][band]["all"]
        band_rows.append(
            [
                band,
                overall["legacy_mad"]["pointings"],
                number(overall["legacy_mad"]["mean"]),
                number(overall["official_rms_median"]["mean"]),
                number(overall["mad_to_total_rms_ratio"]["mean"]),
                overall["mad_to_total_rms_ratio"]["independent_components"],
                interval_text(overall["mad_to_total_rms_ratio"]["ci95"]),
            ]
        )
    parts.append(
        table(
            [
                "Band",
                "Pointings",
                "Mean legacy MAD",
                "Mean median total RMS",
                "Mean MAD / total RMS",
                "Ancestry groups",
                "Ratio 95% interval",
            ],
            band_rows,
        )
    )
    parts.append(
        "<details><summary>Field summaries and descriptive associations</summary><pre>"
        + html.escape(
            json.dumps(
                {
                    "fields": summary["four_band_summary"],
                    "associations": summary["mer_descriptive_associations"],
                },
                indent=2,
            )
        )
        + "</pre></details>"
    )
    parts += [
        "<h2>Independent exposure differences</h2><p>Supplied backgrounds "
        "are subtracted, WCS and photometric scales aligned, and PSFs "
        "cross-convolved. "
        "Δ=A−B; predicted VΔ=VA+VB; Z=Δ/√VΔ. No ordinary source mask or "
        "difference sigma clipping is applied. "
        "The stored sparse transformation operators represent C=L diag(v) "
        "Lᵀ, including bilinear and convolution covariance. "
        "Aperture predictions use the complete operator. Every pointing "
        "receives equal weight; shared native exposures connect bootstrap "
        "groups.</p>",
        table(
            [
                "Band",
                "Pointings",
                "Pairs",
                "Independent groups",
                "Mean std(Z)",
                "95% interval",
                "5% diagnostic",
            ],
            [
                [
                    b,
                    s["pointings"],
                    s["pair_count"],
                    s["independent_components"],
                    number(s["mean"]),
                    interval_text(s["ci95"]),
                    s["amplitude_target"],
                ]
                for b, s in exp_summary.items()
            ],
        ),
        "<p>Residual excess may include native pixel covariance and imperfect background, "
        "astrometric or PSF matching; it is not "
        "automatically a correction factor for the RMS maps. VIS remains provisional because "
        "the PSF snapshot coordinates could not be verified.</p>",
        "<h2>Coverage, source brightness and spatial correlation</h2><p>These comparisons retain "
        "residual structure. Bright/faint scatter compares the 90th–99th brightness percentiles "
        "with the lower half of the noisy pair mean, requiring at least ten pixels in each bin. "
        "Lag-one correlation averages horizontal and vertical covariance divided by the overall "
        "Z variance, with at least 64 pixel pairs per lag. The 3-pixel aperture variance ratio uses "
        "at least eight valid square apertures. Values are averaged within pointings. Unavailable "
        "values indicate inadequate coverage, not agreement.</p>",
        "<details><summary>Inspect the 36 pointing-band diagnostic summaries</summary>"
        + table(
            [
                "Band / pointing",
                "std(Z)",
                "Valid area",
                "Native exposures",
                "Bright/faint scatter",
                "Lag-one correlation",
                "3-pixel aperture variance ratio",
            ],
            [
                [
                    r["band"] + " / " + r["patch_id"],
                    number(r["std_z"]),
                    f"{r['valid_fraction']:.1%}",
                    r["native_exposure_count"],
                    number(r["bright_to_faint_scatter"]),
                    number(r["lag_one_correlation"]),
                    number(r["aperture_variance_ratio_3px"]),
                ]
                for r in diagnostics["pointings"]
            ],
        )
        + "</details>",
        "<p>Descriptive Spearman correlations with pointing-level std(Z) follow. Nine selected "
        "pointings per band are too few to establish causes; the selection itself spans official "
        "VIS RMS. Spatial correlation is expected after matching, so its presence alone does not "
        "demonstrate a failure. The separate bright-star example demonstrates a halo failure of "
        "the legacy statistic, but its exposure difference is unavailable.</p>",
        table(
            ["Band", "Valid area", "Exposure count", "Bright/faint scatter", "Lag-one correlation"],
            [
                [band]
                + [
                    number(values[key]["spearman_rho"]) + f" (n={values[key]['pointings']})"
                    for key in (
                        "valid_fraction",
                        "native_exposure_count",
                        "bright_to_faint_scatter",
                        "lag_one_correlation",
                    )
                ]
                for band, values in diagnostics["associations_with_std_z"].items()
            ],
        ),
        "<h2>Exposure validation sampling</h2>",
        table(
            ["Field", "Pointing", "Stratum", "Selection VIS total RMS"],
            [
                [s["field"], s["patch_id"], s["stratum"], number(s.get("selection_vis_rms"))]
                for s in manifest.get("validation_selection", [])
            ],
        ),
        "<p class='small'>Nine distinct pointings, three per field, ranked"
        " by official VIS RMS in the central patch. Sample 31 is reserved "
        "for the separate bright-star stress case.</p>",
        "<h2>Saved example reproduction</h2>",
        table(
            ["Example", "Saved MAD", "Recomputed MAD", "Same pixels", "Reproduced"],
            [
                [
                    r["patch_id"],
                    number(r["saved_example"]["saved_mad"], 12),
                    number(r["saved_example"]["recomputed_mad"], 12),
                    r["saved_example"]["pixels_match_saved"],
                    r["saved_example"]["passed"],
                ]
                for r in cases
            ],
        ),
        "<p>The bright-star official RMS maps also have repeated extreme values, shown below "
        "in the same comparison units. Their sentinel meaning is not verified by the available "
        "metadata. The official median retains these values; it is not an independently validated "
        "background-noise amplitude. RMS panels use log10 to make their large dynamic range visible.</p>",
        table(
            [
                "Stress-case band",
                "Median total RMS",
                "Maximum RMS",
                "Area at maximum",
                "Maximum pixels with INVALID clear",
            ],
            [
                [
                    r["band"],
                    number(r["median"]),
                    number(r["maximum"]),
                    f"{r['fraction_at_maximum']:.1%}",
                    f"{r['fraction_at_maximum_with_invalid_bit_clear']:.1%}",
                ]
                for r in stress_rms
            ],
        ),
        "<h2>Inspectable four-band MER panels</h2><p>Black legacy-mask "
        "pixels are excluded. Flags are shown separately. Each panel "
        "contains the identical crop in all layers.</p>",
    ]
    # Saved examples first; closed details keep the long report navigable.
    for patch in sorted(manifest["patches"], key=lambda p: (p["kind"] == "central", p["sample_id"])):
        rows = [r for r in mer if r["patch_id"] == patch["patch_id"]]
        path = cached_figure(
            root, patch["patch_id"] + "_mer", rows, partial(mer_figure, root, patch["patch_id"], rows)
        )
        label = html.escape(patch["field"] + " · " + patch["patch_id"] + " · " + patch["kind"])
        parts.append(
            f"<details><summary>{label}</summary>"
            + image_tag(path, patch["patch_id"] + " science, RMS, flags, legacy mask and residual")
            + "</details>"
        )
    parts.append("<h2>Differences, residual tails, and aperture variance</h2>")
    for row in differences:
        label = f"{row['patch_id']} · {row['band']} · {row.get('pair_index', '')} · {row['status']}"
        parts.append(f"<details><summary>{html.escape(label)}</summary>")
        if "arrays" in row:
            path = cached_figure(
                root,
                f"{row['patch_id']}_{row['band']}_pair{row['pair_index']}",
                [row],
                partial(difference_figure, root, row),
            )
            parts.append(image_tag(path, label))
            parts.append("<pre>" + html.escape(json.dumps(row["statistics"], indent=2)) + "</pre>")
        else:
            parts.append("<p>" + html.escape(row.get("reason", "Unavailable")) + "</p>")
        parts.append("</details>")
    parts += [
        "<h2>Propagated median predictions</h2><p>These are Monte Carlo "
        "predictions at 0.1 arcsec/pixel, distinct from direct "
        "exposure-difference measurements. "
        "Exact MER layer selection and rejection are unavailable, so these"
        " conditional predictions cannot certify the MER map.</p>",
        table(
            ["Patch", "Band", "Status", "Exposures", "Draws", "Limit"],
            [
                [
                    r["patch_id"],
                    r["band"],
                    r["status"],
                    r["native_exposure_count"],
                    r.get("draws", "—"),
                    r.get("limit", r.get("reason", "")),
                ]
                for r in summary.get("mer_predictions", [])
            ],
        ),
        "<h2>Missing data and limits</h2><ul>"
        + "".join("<li>" + html.escape(s) + "</li>" for s in limitations)
        + "</ul>",
    ]
    if (root / "exposures.json").exists():
        exposure_manifest = read_json(root / "exposures.json")
        parts.append(
            "<details><summary>Native exposure, background and PSF provenance</summary><pre>"
            + html.escape(json.dumps(exposure_manifest, indent=2))
            + "</pre></details>"
        )
        missing = [
            [pid, band, row.get("archive_metadata", {}).get("obs_publisher_did", "—"), row.get("reason", "")]
            for pid, p in exposure_manifest["pointings"].items()
            for band, rows in p["bands"].items()
            for row in rows.values()
            if row["status"] != "acquired"
        ]
        parts.append(
            table(["Patch", "Band", "Exposure", "Acquisition exclusion / unavailable reason"], missing)
        )
    parts += [
        "<h2>Provenance and reproducibility</h2><p>Run the independent CLI"
        " stages described in the repository README. "
        "The embedded JSON below includes numerical definitions, "
        "conversions, checksums, processing parameters, exclusions and "
        "sampling decisions. "
        "FITS byte ranges and query responses remain in the checksummed "
        "local cache; numerical arrays and PNGs accompany this HTML.</p>",
        "<details><summary>Controlled validation checks</summary><pre>"
        + html.escape(
            json.dumps(summary.get("implementation_validation", {"status": "not recorded"}), indent=2)
        )
        + "</pre></details>",
        "<details><summary>Measurement JSON</summary><pre>"
        + html.escape(json.dumps(summary, indent=2))
        + "</pre></details>",
        "<details><summary>Acquisition manifest and archive provenance</summary><pre>"
        + html.escape(json.dumps(manifest, indent=2))
        + "</pre></details>",
        "<p>Product definitions: <a href='" + DPDD + "'>MER</a>, "
        "<a href='https://euclid.esac.esa.int/dr/q1/dpdd/visdpd/dpcards/vi"
        "s_calibratedquadframe.html'>VIS exposures</a>, "
        "<a href='https://euclid.esac.esa.int/dr/q1/dpdd/nirdpd/dpcards/ni"
        "r_calibratedframe.html'>NIR exposures</a>; "
        "<a href='https://arxiv.org/html/2503.15305v2#S3.SS2'>MER processing §3.2</a>.</p>",
        "<p class='small'>This work has made use of the Quick Release (Q1)"
        " data from the Euclid mission of the European Space Agency (ESA),"
        " "
        "accessed through NASA/IPAC IRSA. <a "
        "href='https://doi.org/10.57780/esa-2853f3b'>Euclid Q1 data "
        "DOI</a>.</p>",
        "<h2>Recommendations for a later generation-model change</h2><ol>"
        + "".join("<li>" + html.escape(s) + "</li>" for s in recommendations)
        + "</ol>",
        "</main></html>",
    ]
    path = root / "report.html"
    path.write_text("\n".join(parts))
    save_json(
        root / "report_manifest.json",
        {
            "schema_version": 1,
            "generated_at": utc_now(),
            "report_sha256": digest(path),
            "summary_sha256": digest(root / "summary.json"),
            "figures": {p.name: digest(p) for p in (root / "figures").glob("*.png")},
            "measurement_status_counts": dict(Counter(r["status"] for r in differences)),
        },
    )
    return path
