"""Tests for the HR-vs-SR angular power-spectrum math.

Validates the key diagnostic property: ``r(k)`` separates a *blur* (linear,
fully correlated → r≈1, T<1 at high k) from added *noise* (decorrelated →
r drops at high k), while identical images give r≈T≈1.
"""

from __future__ import annotations

import numpy as np
import pytest

from euclid_polish.eval.power_spectrum import (
    EnsembleSpectrumAccumulator,
    bin_powers,
    cross_power_2d,
    ensemble_ps_plot_curves,
    k_magnitude_2d,
    log_k_edges,
    normalized_log_scale_mean,
    pairwise_cross_correlation,
    ratios_from_powers,
    render_ensemble_power_spectrum,
)
from euclid_polish.eval.power_spectrum import (
    _SpectrumAccumulator as SpectrumAccumulator,
)

PIXEL_SCALE = 0.05  # arcsec/pixel (HR/SR grid)


def _smooth_field(n: int = 128, sigma: float = 3.0, seed: int = 0) -> np.ndarray:
    """A random field with power across scales (smoothed white noise)."""
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(seed)
    return gaussian_filter(rng.standard_normal((n, n)), sigma).astype(np.float64)


def _finite(arr, mask):
    a = arr[mask]
    return a[np.isfinite(a)]


def test_k_grid_reaches_nyquist():
    edges = log_k_edges(PIXEL_SCALE)
    assert edges[0] > 0
    assert np.isclose(edges[-1], 1.0 / (2.0 * PIXEL_SCALE))  # 10 cyc/arcsec
    kmag = k_magnitude_2d(64, PIXEL_SCALE)
    assert kmag.shape == (64, 64)
    assert kmag[0, 0] == 0.0  # DC mode


def test_identical_images_give_unity():
    a = _smooth_field()
    acc = SpectrumAccumulator(log_k_edges(PIXEL_SCALE))
    acc.add(a, a, PIXEL_SCALE)
    res = acc.finalize()
    m = res["count"] > 0
    assert np.allclose(_finite(res["T"], m), 1.0, atol=1e-9)
    assert np.allclose(_finite(res["r"], m), 1.0, atol=1e-9)


def test_blur_lowers_transfer_but_keeps_correlation():
    """A positive-MTF blur is fully correlated: r≈1, T falls at high k."""
    from scipy.ndimage import gaussian_filter

    a = _smooth_field(sigma=1.0)   # broadband power up through mid-k
    b = gaussian_filter(a, 1.5)    # extra positive-MTF blur
    acc = SpectrumAccumulator(log_k_edges(PIXEL_SCALE))
    acc.add(a, b, PIXEL_SCALE)     # default Tukey window suppresses edges
    res = acc.finalize()
    k, T, r = res["k"], res["T"], res["r"]
    m = res["count"] > 0
    lo = m & (k < 1.0)               # large scales
    mid = m & (k > 2.0) & (k < 5.0)  # small scales with real signal
    # correlation stays high (linear, deterministic blur)
    assert np.nanmedian(r[lo]) > 0.97
    assert np.nanmedian(r[mid]) > 0.85
    # transfer function clearly drops toward small scales
    assert np.nanmedian(T[lo]) > 0.9
    assert np.nanmedian(T[mid]) < 0.6 * np.nanmedian(T[lo])


def test_noise_decorrelates_at_high_k():
    """Independent noise added to SR drops r(k) at small scales."""
    a = _smooth_field()
    rng = np.random.default_rng(1)
    noise = rng.standard_normal(a.shape) * 0.5 * a.std()
    b = a + noise
    acc = SpectrumAccumulator(log_k_edges(PIXEL_SCALE))
    acc.add(a, b, PIXEL_SCALE, window=np.ones_like(a))
    res = acc.finalize()
    k, r = res["k"], res["r"]
    m = res["count"] > 0
    lo = m & (k < 1.0)
    hi = m & (k > 5.0)
    assert np.nanmedian(r[lo]) > 0.9          # real structure preserved
    assert np.nanmedian(r[hi]) < np.nanmedian(r[lo])  # high-k decorrelates
    assert np.nanmedian(r[hi]) < 0.7          # clearly invented at small scales


def test_cross_power_shapes_and_dc_removed():
    a = _smooth_field(64)
    b = _smooth_field(64, seed=2)
    p_aa, p_bb, p_ab = cross_power_2d(a, b)
    assert p_aa.shape == p_bb.shape == p_ab.shape == (64, 64)
    # mean subtraction zeroes the DC (0,0) mode
    assert p_aa[0, 0] == pytest.approx(0.0, abs=1e-6)


def test_accumulator_combines_mixed_stamp_sizes():
    """Stamps of different pixel sizes share the physical k-grid."""
    acc = SpectrumAccumulator(log_k_edges(PIXEL_SCALE))
    acc.add(_smooth_field(128), _smooth_field(128, seed=3), PIXEL_SCALE)
    acc.add(_smooth_field(106), _smooth_field(106, seed=4), PIXEL_SCALE)
    res = acc.finalize()
    assert acc.n_obj == 2
    assert np.isfinite(res["r"][res["count"] > 0]).all()


def test_pairwise_cross_correlation_identical_images_give_unity():
    a = _smooth_field()
    rows = pairwise_cross_correlation([a, a.copy(), a.copy()], PIXEL_SCALE,
                                      log_k_edges(PIXEL_SCALE))
    assert rows.shape[0] == 3          # (0,1), (0,2), (1,2)
    finite = rows[np.isfinite(rows)]
    assert finite.size and np.allclose(finite, 1.0, atol=1e-9)


def test_pairwise_cross_correlation_independent_noise_decorrelates():
    base = _smooth_field(sigma=1.0)
    rng = np.random.default_rng(7)
    members = [base + rng.standard_normal(base.shape) * base.std()
               for _ in range(3)]
    k_edges = log_k_edges(PIXEL_SCALE)
    rows = pairwise_cross_correlation(members, PIXEL_SCALE, k_edges,
                                      window=np.ones_like(base))
    kc = np.sqrt(k_edges[:-1] * k_edges[1:])
    med = np.nanmedian(rows, axis=0)
    lo, hi = kc < 1.0, kc > 5.0
    assert np.nanmedian(med[lo]) > np.nanmedian(med[hi])
    assert np.nanmedian(med[hi]) < 0.8   # independent noise splits the pair


def test_pairwise_cross_correlation_single_image_empty():
    rows = pairwise_cross_correlation([_smooth_field()], PIXEL_SCALE,
                                      log_k_edges(PIXEL_SCALE))
    assert rows.shape[0] == 0


def test_normalized_log_scale_mean_weights_frequency_octaves_equally():
    k = np.array([1.0, 2.0, 4.0])
    r = np.array([1.0, 0.5, 0.0])
    assert normalized_log_scale_mean(k, r, k_min=1.0, k_max=4.0) == pytest.approx(0.5)
    assert normalized_log_scale_mean(k, r, k_min=2.0, k_max=4.0) == pytest.approx(0.25)


def test_ensemble_coherence_scores_are_field_balanced():
    n = 64
    hr = _smooth_field(n, sigma=1.0)
    members = np.stack([hr.copy(), hr.copy()])
    acc = EnsembleSpectrumAccumulator(n, PIXEL_SCALE)
    acc.add(hr, hr, members, lr=hr)
    scores = acc.coherence_scores(sr_k_min=5.0)
    by_id = {row["id"]: row for row in scores["scores"]}
    assert by_id["ensemble_mean"]["overall"] == pytest.approx(1.0, abs=1e-6)
    assert by_id["ensemble_mean"]["sr"] == pytest.approx(1.0, abs=1e-6)
    assert by_id["lr_baseline"]["overall"] == pytest.approx(1.0, abs=1e-6)
    assert by_id["member_0"]["n_fields"] == 1


def test_ensemble_accumulator_emits_cross_model_curves():
    n = 64
    rng = np.random.default_rng(3)
    hr = _smooth_field(n, sigma=1.0)
    members = np.stack([hr + rng.standard_normal((n, n)) * 0.1
                        for _ in range(3)])
    acc = EnsembleSpectrumAccumulator(n, PIXEL_SCALE)
    acc.add(hr, members.mean(0), members)
    curves = acc.curves()
    assert curves["r_pairs"].shape == (3, acc.nbins)
    assert curves["r_cross"].shape == (acc.nbins,)
    r_cross = curves["r_cross"]
    m = np.isfinite(r_cross)
    assert m.any()
    # signal-dominated scales agree; the noise-dominated Nyquist bins decorrelate
    assert np.nanmedian(r_cross[m & (acc.k_cen < 5.0)]) > 0.9
    assert np.nanmin(r_cross[m]) < np.nanmax(r_cross[m])


def test_ensemble_spectrum_uses_displayed_raw_mean():
    """The primary r/T curves describe raw-mean inference output.

    The disagreement decomposition may use mean(asinh(member)), but the
    plotted ensemble line must be the displayed arithmetic raw-electron mean
    evaluated in the spectrum's asinh space.
    """
    n = 64
    hr = _smooth_field(n, sigma=1.0)
    members = np.stack([hr, 8.0 * hr + 1000.0])
    raw_mean = members.mean(axis=0)
    acc = EnsembleSpectrumAccumulator(n, PIXEL_SCALE)
    acc.add(hr, raw_mean, members)
    curves = acc.curves()

    ah = acc._asinh(hr)
    a_mean = acc._asinh(raw_mean)
    bh, bs, bx, bc = bin_powers(
        ah, a_mean, PIXEL_SCALE, acc.k_edges, acc.window)
    expected_t, expected_r = ratios_from_powers(bh, bs, bx, bc)
    finite = np.isfinite(expected_r) & np.isfinite(expected_t)
    assert np.allclose(curves["r"][finite], expected_r[finite])
    assert np.allclose(curves["T"][finite], expected_t[finite])


def test_ensemble_accumulator_combiner_series():
    """A combiner plane fed to add() yields its own P_comb/r_comb series; a
    perfect combiner (== HR) reads T_comb≈1 and r_comb≈1 on signal scales."""
    n = 64
    rng = np.random.default_rng(5)
    hr = _smooth_field(n, sigma=1.0)
    members = np.stack([hr + rng.standard_normal((n, n)) * 0.1 for _ in range(3)])
    acc = EnsembleSpectrumAccumulator(n, PIXEL_SCALE)
    acc.add(hr, members.mean(0), members, combiner=hr)        # perfect combiner
    assert acc.has_combiner
    curves = acc.curves()
    assert "P_comb" in curves and "r_comb" in curves
    plot = ensemble_ps_plot_curves(curves)
    assert plot["T_comb"].shape == plot["theta"].shape
    m = np.isfinite(plot["T_comb"]) & (acc.k_cen < 5.0)
    assert np.nanmedian(plot["T_comb"][m]) == pytest.approx(1.0, abs=0.05)
    assert np.nanmedian(plot["r_comb"][np.isfinite(plot["r_comb"])
                                       & (acc.k_cen < 5.0)]) > 0.95


def test_ensemble_accumulator_stats_rbf_series_is_independent():
    """The third combiner gets a separate curve rather than overwriting RBF."""
    n = 64
    rng = np.random.default_rng(15)
    hr = _smooth_field(n, sigma=1.0)
    members = np.stack([hr + rng.standard_normal((n, n)) * 0.1 for _ in range(3)])
    acc = EnsembleSpectrumAccumulator(n, PIXEL_SCALE)
    acc.add(hr, members.mean(0), members, combiner=members.mean(0),
            stats_rbf_combiner=hr)
    assert acc.has_combiner and acc.has_stats_rbf_combiner
    curves = acc.curves()
    assert "P_comb" in curves and "P_stats_rbf_comb" in curves
    plot = ensemble_ps_plot_curves(curves)
    m = np.isfinite(plot["T_stats_rbf_comb"]) & (acc.k_cen < 5.0)
    assert np.nanmedian(plot["T_stats_rbf_comb"][m]) == pytest.approx(1.0, abs=0.05)


def test_ensemble_accumulator_combined_combiner_series_is_independent():
    """The experimental reconstruction has its own power/coherence keys."""
    n = 64
    rng = np.random.default_rng(6)
    hr = _smooth_field(n, sigma=1.0)
    members = np.stack([hr + rng.standard_normal((n, n)) * 0.1 for _ in range(3)])
    acc = EnsembleSpectrumAccumulator(n, PIXEL_SCALE)
    acc.add(hr, members.mean(0), members, combiner=members.mean(0),
            combined_combiner=hr)
    curves = acc.curves()
    assert "P_comb" in curves and "P_combined" in curves
    plot = ensemble_ps_plot_curves(curves)
    m = np.isfinite(plot["T_combined"]) & (acc.k_cen < 5.0)
    assert np.nanmedian(plot["T_combined"][m]) == pytest.approx(1.0, abs=0.05)


def test_ensemble_ps_plot_curves_no_combiner_is_nan():
    """No combiner fed → T_comb/r_comb are present but all-NaN (the JS hides
    the series)."""
    k = np.array([0.5, 1.0])
    curves = {"k": k, "P_hr": np.array([1.0, 1.0]),
              "P_sr": np.array([1.0, 1.0]), "r": np.array([1.0, 1.0])}
    out = ensemble_ps_plot_curves(curves)
    assert out["T_comb"].shape == (2,) and np.isnan(out["T_comb"]).all()
    assert np.isnan(out["r_comb"]).all()


def test_ensemble_ps_plot_curves_derives_per_member_transfer():
    k = np.array([0.5, 1.0, 2.0])
    curves = {
        "k": k,
        "P_hr": np.array([4.0, 4.0, 4.0]),
        "P_sr": np.array([1.0, 1.0, 1.0]),
        "r": np.array([0.9, 0.8, 0.7]),
        "P_members": np.array([[4.0, 1.0, 0.25],
                               [16.0, 4.0, 1.0]]),
        "r_members": np.array([[0.90, 0.80, 0.70],
                               [0.85, 0.75, 0.65]]),
    }
    out = ensemble_ps_plot_curves(curves)
    assert np.allclose(out["theta"], 0.5 / k)
    assert np.allclose(out["T"], np.sqrt(np.array([0.25, 0.25, 0.25])))
    assert np.allclose(out["T_members"][0], np.sqrt(np.array([1.0, 0.25, 0.0625])))
    assert np.allclose(out["T_members"][1], np.sqrt(np.array([4.0, 1.0, 0.25])))
    assert np.allclose(out["r"], curves["r"])
    assert out["r_members"].shape == (2, 3)


def test_ensemble_ps_plot_curves_handles_no_members():
    k = np.array([0.5, 1.0])
    curves = {"k": k, "P_hr": np.array([1.0, 1.0]),
              "P_sr": np.array([1.0, 1.0]), "r": np.array([1.0, 1.0])}
    out = ensemble_ps_plot_curves(curves)
    assert out["T_members"].shape == (0, 2)
    assert out["r_members"].shape == (0, 2)


def test_render_ensemble_power_spectrum_writes_png(tmp_path):
    k = np.geomspace(0.2, 10.0, 24)
    rng = np.random.default_rng(0)
    curves = {
        "k": k,
        "P_hr": np.abs(rng.normal(1e3, 10, k.size)),
        "P_sr": np.abs(rng.normal(5e2, 10, k.size)),
        "P_disagree": np.abs(rng.normal(10, 1, k.size)),
        "r": np.clip(1.0 - k / 20.0, 0, 1),
        "T": np.clip(1.0 - k / 40.0, 0, 1),
        "P_members": np.abs(rng.normal(5e2, 20, (5, k.size))),
        "r_members": np.clip(
            1.0 - k[None, :] / 20.0 + rng.normal(0, 0.02, (5, k.size)), 0, 1),
        "r_pairs": np.clip(
            1.0 - k[None, :] / 15.0 + rng.normal(0, 0.02, (10, k.size)), 0, 1),
        "r_cross": np.clip(1.0 - k / 15.0, 0, 1),
    }
    out = tmp_path / "ps.png"
    res = render_ensemble_power_spectrum(str(out), curves, n_fields=12)
    assert res == str(out)
    assert out.is_file() and out.stat().st_size > 0


def test_render_ensemble_power_spectrum_empty_returns_none(tmp_path):
    out = tmp_path / "ps.png"
    assert render_ensemble_power_spectrum(
        str(out), {"k": np.array([]), "P_hr": np.array([])}) is None
    assert not out.exists()


def test_member_group_colors_by_loss_and_depth():
    from euclid_polish.eval.power_spectrum import (
        LOSS_LINE_COLORS,
        _member_group_colors,
    )
    meta = [{"loss": "l1", "blocks": 32}, {"loss": "L2", "blocks": 16},
            {"loss": None, "blocks": None}]
    by_loss = _member_group_colors(meta, "loss")
    assert by_loss[0] == ("L1", LOSS_LINE_COLORS["l1"])
    assert by_loss[1] == ("L2", LOSS_LINE_COLORS["l2"])   # case-insensitive
    assert by_loss[2][0] == "L1"                          # legacy → L1
    by_depth = _member_group_colors(meta, "depth")
    assert by_depth[0][0] == "32b" and by_depth[1][0] == "16b"
    assert by_depth[0][1] != by_depth[1][1]               # distinct colors
    assert by_depth[2][0] == "?"
    assert _member_group_colors(meta, None) is None
    assert _member_group_colors(None, "loss") is None


def test_render_ensemble_power_spectrum_with_loss_coloring(tmp_path):
    k = np.geomspace(0.2, 10.0, 24)
    rng = np.random.default_rng(1)
    curves = {
        "k": k,
        "P_hr": np.abs(rng.normal(1e3, 10, k.size)),
        "P_sr": np.abs(rng.normal(5e2, 10, k.size)),
        "r": np.clip(1.0 - k / 20.0, 0, 1),
        "P_members": np.abs(rng.normal(5e2, 20, (3, k.size))),
        "r_members": np.clip(
            1.0 - k[None, :] / 20.0 + rng.normal(0, 0.02, (3, k.size)), 0, 1),
    }
    meta = [{"loss": "l1", "blocks": 32}, {"loss": "l2", "blocks": 32},
            {"loss": "l3", "blocks": 16}]
    out = tmp_path / "ps_loss.png"
    assert render_ensemble_power_spectrum(
        str(out), curves, n_fields=5, member_meta=meta,
        color_by="loss") == str(out)
    assert out.stat().st_size > 0
    out2 = tmp_path / "ps_depth.png"
    assert render_ensemble_power_spectrum(
        str(out2), curves, n_fields=5, member_meta=meta,
        color_by="depth") == str(out2)
