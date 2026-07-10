"""Tests for the pixel-level ensemble disagreement diagnostics.

The key property: for a *calibrated* synthetic ensemble — members = truth +
independent N(0, σ) noise — the accumulator's derived statistics land on the
Gaussian expectations (z-score width ≈ 1, coverage ≈ 68/95/99.7%, binned
median |error| ≈ 0.674·σ of the *mean*), and the renderers write PNGs.
"""

from __future__ import annotations

import numpy as np

from euclid_polish.eval.ensemble_diagnostics import (
    EnsembleDiagnosticsAccumulator,
    render_calibration,
    render_std_vs_brightness,
    render_std_vs_error,
)


def _calibrated_ensemble(n=96, m=8, sigma=5.0, seed=0):
    """(hr, mean, members) where members = hr + independent N(0, sigma)."""
    rng = np.random.default_rng(seed)
    hr = np.abs(rng.normal(50.0, 20.0, (n, n)))
    members = hr[None] + rng.normal(0.0, sigma, (m, n, n))
    return hr, members.mean(0), members


def test_accumulator_counts_fields_and_members():
    acc = EnsembleDiagnosticsAccumulator()
    hr, mean, members = _calibrated_ensemble()
    acc.add(hr, mean, members)
    acc.add(hr, mean, members)
    assert acc.n_fields == 2
    assert acc.n_members == 8
    assert len(acc.field_mean_std) == len(acc.field_rmse) == 2
    assert acc.h_std_err.sum() == 2 * hr.size
    assert acc.h_bright_std.sum() == 2 * hr.size


def test_accumulator_rejects_bad_shapes():
    acc = EnsembleDiagnosticsAccumulator()
    hr, mean, members = _calibrated_ensemble(n=32)
    acc.add(hr, mean[:16, :16], members)          # mean shape mismatch
    acc.add(hr, mean, members[:1])                # single member: no std
    assert acc.n_fields == 0


def test_zscores_calibrated_ensemble_near_standard_normal():
    """members = truth + N(0, σ) ⇒ error of the mean = N(0, σ/√M); dividing by
    the member std (≈σ) gives z ~ N(0, 1/√M) — but the *calibration* question
    uses std of the MEAN, so here we only check the z pipeline is consistent:
    width ≈ 1/√M and coverage ≫ Gaussian (spread overestimates the mean's
    error, as expected for a member-std error bar)."""
    acc = EnsembleDiagnosticsAccumulator()
    for seed in range(3):
        hr, mean, members = _calibrated_ensemble(seed=seed)
        acc.add(hr, mean, members)
    st = acc.z_stats()
    m = acc.n_members
    assert abs(st["sigma_z"] - 1.0 / np.sqrt(m)) < 0.1
    assert st["cover1"] > 0.95        # |z|<1 ⊃ almost everything at σ_z≈0.35
    assert st["cover3"] > st["cover2"] >= st["cover1"]


def test_binned_median_error_tracks_gaussian_expectation():
    """Per std-bin median |mean − hr| ≈ 0.674·σ/√M for the synthetic case."""
    acc = EnsembleDiagnosticsAccumulator()
    hr, mean, members = _calibrated_ensemble(n=192, sigma=5.0)
    acc.add(hr, mean, members)
    cen, med = acc.binned_median_err()
    ok = np.isfinite(med)
    assert ok.any()
    # the populated std bins sit near σ=5; their median error near 0.674·5/√8
    i = int(np.nanargmax(acc.h_std_err.sum(axis=1) * ok))
    expected = 0.6745 * 5.0 / np.sqrt(acc.n_members)
    assert 0.5 * expected < med[i] < 2.0 * expected


def test_binned_std_percentiles_ordered():
    acc = EnsembleDiagnosticsAccumulator()
    hr, mean, members = _calibrated_ensemble()
    acc.add(hr, mean, members)
    p = acc.binned_std_percentiles()
    ok = np.isfinite(p["med"])
    assert ok.any()
    assert (p["lo"][ok] <= p["med"][ok]).all()
    assert (p["med"][ok] <= p["hi"][ok]).all()


def test_renderers_write_pngs(tmp_path):
    acc = EnsembleDiagnosticsAccumulator()
    for seed in range(2):
        hr, mean, members = _calibrated_ensemble(seed=seed)
        acc.add(hr, mean, members)
    for render, name in ((render_std_vs_error, "se.png"),
                         (render_std_vs_brightness, "sb.png"),
                         (render_calibration, "cal.png")):
        out = tmp_path / name
        assert render(str(out), acc) == str(out)
        assert out.is_file() and out.stat().st_size > 0


def test_renderers_empty_accumulator_return_none(tmp_path):
    acc = EnsembleDiagnosticsAccumulator()
    for render, name in ((render_std_vs_error, "se.png"),
                         (render_std_vs_brightness, "sb.png"),
                         (render_calibration, "cal.png")):
        out = tmp_path / name
        assert render(str(out), acc) is None
        assert not out.exists()


def test_to_payload_is_json_ready(tmp_path):
    import json

    acc = EnsembleDiagnosticsAccumulator()
    for seed in range(2):
        hr, mean, members = _calibrated_ensemble(seed=seed)
        acc.add(hr, mean, members)
    p = acc.to_payload()
    s = json.dumps(p)                      # no NaN anywhere → serializes
    assert "NaN" not in s
    assert p["n_fields"] == 2 and p["n_members"] == 8
    assert len(p["std_err"]["hist"]) == len(p["std_err"]["edges"]) - 1
    assert len(p["bright_std"]["hist"]) == len(p["bright_std"]["bright_edges"]) - 1
    assert len(p["calibration"]["pdf"]) == len(p["calibration"]["z_edges"]) - 1
    assert len(p["calibration"]["field_std"]) == 2
    # binned curves are log10 values with None gaps, same length as bins
    assert len(p["std_err"]["med_std"]) == len(p["std_err"]["hist"])
    stats = p["calibration"]["stats"]
    assert set(stats) >= {"cover1", "cover2", "cover3", "sigma_z"}


def test_to_payload_empty_accumulator_serializes():
    import json

    p = EnsembleDiagnosticsAccumulator().to_payload()
    json.dumps(p)
    assert p["n_fields"] == 0


def test_error_scored_against_combiner_when_given():
    """With a combiner passed, the std-vs-error histogram measures |combiner −
    HR|, not |mean − HR| — a combiner that nails the truth pushes the error mass
    to the floor bin, and the payload labels the axis 'combiner'."""
    hr, mean, members = _calibrated_ensemble(n=128, seed=1)

    acc_mean = EnsembleDiagnosticsAccumulator()
    acc_mean.add(hr, mean, members)
    assert acc_mean.pred_label() == "ensemble mean"
    assert acc_mean.to_payload()["std_err"]["pred"] == "ensemble mean"

    # A combiner equal to HR ⇒ zero error everywhere ⇒ all error mass in bin 0.
    acc_comb = EnsembleDiagnosticsAccumulator()
    acc_comb.add(hr, mean, members, combiner=hr.copy())
    assert acc_comb.pred_label() == "combiner"
    assert acc_comb.n_pred_combiner == acc_comb.n_fields == 1
    err_marginal = acc_comb.h_std_err.sum(axis=0)   # over σ → |err| distribution
    assert err_marginal[0] == hr.size               # everything in the floor bin
    assert acc_comb.to_payload()["std_err"]["pred"] == "combiner"


def test_mixed_combiner_run_labels_as_mean():
    """If only SOME fields carry a combiner, the label degrades to the mean —
    the histogram is a mix, so we don't over-claim it's the combiner's error."""
    hr, mean, members = _calibrated_ensemble()
    acc = EnsembleDiagnosticsAccumulator()
    acc.add(hr, mean, members, combiner=hr.copy())   # field 0: combiner
    acc.add(hr, mean, members)                        # field 1: mean only
    assert acc.n_pred_combiner == 1 and acc.n_fields == 2
    assert acc.pred_label() == "ensemble mean"


def test_samples_only_recorded_with_field_index():
    """Back-tracing reservoirs stay empty unless the caller opts a field in via
    ``field_index`` (only cached fields are reconstructable)."""
    acc = EnsembleDiagnosticsAccumulator()
    hr, mean, members = _calibrated_ensemble()
    acc.add(hr, mean, members)                       # no field_index → no samples
    assert not acc.se_samples and not acc.bs_samples
    acc.add(hr, mean, members, field_index=7)
    assert acc.se_samples and acc.bs_samples


def test_samples_point_at_the_cell_they_are_binned_into():
    """Every stored ``(field, y, x)`` sample must fall in the histogram cell it
    is keyed under — the click-to-inspect contract."""
    from euclid_polish.eval.ensemble_diagnostics import LOG_E_BINS

    acc = EnsembleDiagnosticsAccumulator()
    hr, mean, members = _calibrated_ensemble(n=128, seed=3)
    acc.add(hr, mean, members, field_index=0)
    std = members.std(0)
    err = np.abs(mean - hr)
    ls = np.log10(np.maximum(std, 1e-12))
    le = np.log10(np.maximum(err, 1e-12))
    for key, picks in acc.se_samples.items():
        i, j = key // LOG_E_BINS, key % LOG_E_BINS
        assert len(picks) <= acc.sample_k
        for field, y, x in picks:
            assert field == 0
            si = np.clip(np.searchsorted(acc.log_edges, ls[y, x], "right") - 1,
                         0, LOG_E_BINS - 1)
            ej = np.clip(np.searchsorted(acc.log_edges, le[y, x], "right") - 1,
                         0, LOG_E_BINS - 1)
            assert (si, ej) == (i, j)


def test_samples_span_multiple_fields():
    """Field-stratified reservoir: a popular cell draws examples from different
    fields, not five pixels of one."""
    acc = EnsembleDiagnosticsAccumulator()
    for f in range(6):
        hr, mean, members = _calibrated_ensemble(n=96, seed=f)
        acc.add(hr, mean, members, field_index=f)
    # the busiest std_err cell should reference >1 distinct field
    busiest = max(acc.se_samples.values(), key=len)
    assert len({field for field, _, _ in busiest}) >= 2


def test_samples_payload_is_json_ready():
    import json

    acc = EnsembleDiagnosticsAccumulator()
    for f in range(2):
        hr, mean, members = _calibrated_ensemble(seed=f)
        acc.add(hr, mean, members, field_index=f)
    p = acc.samples_payload()
    s = json.dumps(p)
    assert "NaN" not in s
    assert p["n_fields"] == 2 and p["sample_k"] == acc.sample_k
    assert set(p) >= {"std_err", "bright_std"}
    # keys are "i,j" and values are lists of [field, y, x] triples
    for cell, picks in p["std_err"].items():
        i, j = map(int, cell.split(","))
        assert 0 <= i < len(acc.log_edges) - 1
        for triple in picks:
            assert len(triple) == 3
