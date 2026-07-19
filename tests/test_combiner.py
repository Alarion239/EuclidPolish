"""The per-band ensemble combiner (RBF brightness gate, convex mixture).

Operates directly on in-memory fit buffers (``{band: (X(N,M), y(N,))}`` in asinh
space) so no real checkpoints or TFRecords are needed. NOTE: these tests were
updated for the RBF-gate rewrite and have not been run in this session.
"""

from __future__ import annotations

import numpy as np
import pytest

from euclid_polish.eval.combiner import (
    MINMAX_RBF_GATE_KIND,
    STACKED_RBF_GATE_KIND,
    STATS_RBF_GATE_KIND,
    BandCombiner,
    Combiner,
    StatsRBFBandCombiner,
    build_fit_buffers_from_fields,
    combiner_artifact_fingerprint,
    fit_combiner,
    fit_stacked_combiner,
    load_combiner,
    member_weight_diagnostics,
    save_combiner,
)

STRETCH_E = 100.0


def _asinh(x):
    return np.arcsinh(np.asarray(x, np.float64) / STRETCH_E)


def _mean_l1(X, y):
    return float(np.mean(np.abs(X.mean(axis=1) - y)))


def _brightness_routed(n=8000, seed=0):
    """Two members with a brightness-dependent quality split:
    - member 0 (L1-like): matches the target at FAINT pixels, erases BRIGHT ones.
    - member 1 (L2-like): matches the target at BRIGHT pixels, noisy at faint.
    A good gate weights member 0 when faint and member 1 when bright.
    Returns asinh (X (n,2), y (n,))."""
    rng = np.random.default_rng(seed)
    y = _asinh(np.abs(rng.normal(30.0, 25.0, n)) ** 2)         # 0 .. ~6 asinh
    bright = y > np.median(y)
    x0 = np.where(bright, 0.0, y)                              # erases bright
    x1 = np.where(bright, y, y + rng.normal(0, 1.0, n))       # noisy at faint
    X = np.stack([x0, x1], axis=1).astype(np.float32)
    return X, y.astype(np.float32), bright


def test_weights_are_convex():
    X, y, _ = _brightness_routed()
    comb = fit_combiner({"VIS": (X, y)}, ["00", "01"], n_kernels=10, steps=300)
    w = comb.bands["VIS"].weights(X)
    assert w.shape == X.shape
    assert np.all(w >= -1e-6) and np.all(w <= 1 + 1e-6)
    np.testing.assert_allclose(w.sum(axis=1), 1.0, atol=1e-5)


def test_gate_routes_faint_to_l1_bright_to_l2():
    X, y, _ = _brightness_routed()
    comb = fit_combiner({"VIS": (X, y)}, ["00", "01"], n_kernels=12, steps=800)
    fit_l1 = float(np.mean(np.abs(comb.bands["VIS"].forward_asinh(X) - y)))
    assert fit_l1 < _mean_l1(X, y)                     # beats the naive mean
    eff = comb.effective_weights("VIS", n_levels=25)
    w = np.asarray(eff["jacobian"])                    # (L, 2), rows sum to 1
    lo = w[:5].mean(axis=0)                             # faintest levels
    hi = w[-5:].mean(axis=0)                            # brightest levels
    assert lo[0] > lo[1]                               # faint → member 0 (L1)
    assert hi[1] > hi[0]                               # bright → member 1 (L2)


def test_persist_load_roundtrip(tmp_path):
    X, y, _ = _brightness_routed()
    comb = fit_combiner({"VIS": (X, y)}, ["00", "01"], n_kernels=8, steps=200)
    save_combiner(comb, str(tmp_path))
    back = load_combiner(str(tmp_path), member_labels=["00", "01"])
    assert back is not None
    np.testing.assert_allclose(back.bands["VIS"].forward_asinh(X),
                               comb.bands["VIS"].forward_asinh(X),
                               rtol=1e-5, atol=1e-5)
    assert back.member_labels == ["00", "01"]
    assert back.n_kernels == comb.n_kernels


def test_load_returns_none_on_member_mismatch(tmp_path):
    X, y, _ = _brightness_routed()
    comb = fit_combiner({"VIS": (X, y)}, ["00", "01"], n_kernels=6, steps=100)
    save_combiner(comb, str(tmp_path))
    assert load_combiner(str(tmp_path), member_labels=["00", "02"]) is None
    assert load_combiner(str(tmp_path)) is not None


def test_separate_artifact_directory_preserves_ordinary_combiner(tmp_path):
    """An experimental combiner must not shadow the ordinary combiner path."""
    def make_combiner(n_kernels):
        band = BandCombiner(
            V=np.zeros((n_kernels, 2), np.float32), a=np.zeros(2, np.float32),
            centers=np.linspace(-1, 13, n_kernels, dtype=np.float32), sigma=1.0,
            surviving=np.ones(2, bool))
        return Combiner(member_labels=["00", "01"], n_kernels=n_kernels,
                        sigma_scale=1.0, min_usage=0.0, bands={"VIS": band},
                        band_names=("VIS",))

    ordinary = make_combiner(4)
    combined = make_combiner(5)

    save_combiner(ordinary, str(tmp_path))
    save_combiner(combined, str(tmp_path), artifact_dir="combined_combiner")

    assert load_combiner(str(tmp_path)).n_kernels == 4
    assert load_combiner(str(tmp_path), artifact_dir="combined_combiner").n_kernels == 5


def test_apply_field_shape_and_inverse_stretch():
    """apply_field returns (H,W,C) electrons. A convex mix of IDENTICAL members
    must recover those electrons exactly — checks the 100*sinh inverse."""
    M, H, W = 3, 8, 8
    K = 6
    centers = np.linspace(-1.0, 13.0, K).astype(np.float32)
    bands = {b: BandCombiner(V=np.zeros((K, M), np.float32),
                             a=np.zeros(M, np.float32), centers=centers,
                             sigma=2.0, surviving=np.ones(M, bool))
             for b in ("VIS", "Y_E", "J_E", "H_E")}
    comb = Combiner(member_labels=["00", "01", "02"], n_kernels=K, sigma_scale=1.0,
                    min_usage=0.0, bands=bands, band_names=("VIS", "Y_E", "J_E", "H_E"))
    rng = np.random.default_rng(5)
    field = np.abs(rng.normal(500, 200, (H, W, 4))).astype(np.float32)
    preds = np.stack([field, field, field], axis=0)        # identical members
    out = comb.apply_field(preds)
    assert out.shape == (H, W, 4)
    np.testing.assert_allclose(out, field, rtol=1e-4, atol=1e-2)


def test_legacy_min_usage_does_not_prune_during_fit():
    """Gate-frequency usage is no longer allowed to remove rare members."""
    X, y, _ = _brightness_routed()
    # member 1 is pure noise → the gate should barely use it.
    rng = np.random.default_rng(1)
    X2 = np.stack([X[:, 0], rng.normal(0, 1.0, len(y)).astype(np.float32)], axis=1)
    comb = fit_combiner({"VIS": (X2, y)}, ["00", "01"], n_kernels=10,
                        steps=500, min_usage=0.15)
    assert comb.bands["VIS"].surviving.tolist() == [True, True]
    assert comb.member_importance == {}


def test_member_weight_diagnostics_report_peak_and_integral_per_band():
    probabilities = np.asarray([0.8, 0.15, 0.05], np.float32)
    band = BandCombiner(
        V=np.zeros((1, 3), np.float32), a=np.log(probabilities),
        centers=np.zeros(1, np.float32), sigma=1.0,
        surviving=np.ones(3, bool))
    comb = Combiner(
        member_labels=["00", "01", "02"], n_kernels=1,
        sigma_scale=1.0, min_usage=0.0, bands={"VIS": band},
        band_names=("VIS",))
    X = np.zeros((32, 3), np.float32)
    peaks, integrals = member_weight_diagnostics(
        comb, {"VIS": (X, np.zeros(32, np.float32))}, chunk_rows=7)
    np.testing.assert_allclose(peaks["VIS"], probabilities, atol=1e-6)
    np.testing.assert_allclose(integrals["VIS"], probabilities, atol=1e-6)


def test_without_member_drops_pruned_column_exactly():
    """A PRUNED member can be removed with `without_member` without changing the
    fused output — the same result the combiner would give including it (weight
    0). This is what lets a combiner survive archiving a member it never used."""
    M, H, W, K = 3, 6, 6, 5
    names = ("VIS", "Y_E", "J_E", "H_E")
    centers = np.linspace(-1.0, 13.0, K).astype(np.float32)
    rng = np.random.default_rng(3)
    surv = np.array([True, False, True], bool)          # member 1 pruned in all bands
    bands = {b: BandCombiner(
        V=rng.normal(size=(K, M)).astype(np.float32),
        a=rng.normal(size=(M,)).astype(np.float32),
        centers=centers, sigma=2.0, surviving=surv.copy()) for b in names}
    comb = Combiner(member_labels=["00·a", "01·b", "02·c"], n_kernels=K,
                    sigma_scale=1.0, min_usage=0.1, bands=bands, band_names=names)

    assert comb.member_pruned(1) is True
    assert comb.member_pruned(0) is False and comb.member_pruned(2) is False

    small = comb.without_member(1)
    assert small.member_labels == ["00·a", "02·c"]
    assert all(small.bands[b].V.shape == (K, 2) for b in names)
    assert all(small.bands[b].surviving.tolist() == [True, True] for b in names)

    field = np.abs(rng.normal(400, 300, (M, H, W, 4))).astype(np.float32)
    out_full = comb.apply_field(field)                  # 3-member stack, member 1 unused
    out_small = small.apply_field(field[[0, 2]])        # 2-member reindexed stack
    np.testing.assert_allclose(out_full, out_small, rtol=1e-5, atol=1e-3)


def test_single_member_degenerate():
    rng = np.random.default_rng(6)
    y = _asinh(np.abs(rng.normal(40, 30, 4000)) ** 2).astype(np.float32)
    X = (y + rng.normal(0, 0.02, y.shape)).astype(np.float32)[:, None]   # (n,1)
    comb = fit_combiner({"VIS": (X, y)}, ["00"], n_kernels=6, steps=200)
    w = comb.bands["VIS"].weights(X)
    np.testing.assert_allclose(w, 1.0, atol=1e-6)       # only member → weight 1
    np.testing.assert_allclose(comb.bands["VIS"].forward_asinh(X), X[:, 0],
                               atol=1e-5)


def test_effective_weights_shape():
    X, y, _ = _brightness_routed()
    comb = fit_combiner({"VIS": (X, y)}, ["00", "01"], n_kernels=8, steps=150)
    eff = comb.effective_weights("VIS", n_levels=20)
    w = np.asarray(eff["jacobian"])
    assert w.shape == (20, 2)
    np.testing.assert_allclose(w.sum(axis=1), 1.0, atol=1e-5)
    assert np.asarray(eff["brightness_e"]).shape == (20,)


def test_stats_rbf_is_convex_uses_mean_std_centers_and_roundtrips(tmp_path):
    """The compact second regime has 2-D frozen RBF centres and a normal
    convex member mixture while retaining a compact hot path."""
    X, _y, _ = _brightness_routed(n=600, seed=19)
    # Hand-build a small trained-looking gate: the focused test is deliberately
    # TensorFlow-free so it verifies the hot-path representation and persistence
    # without opening the heavyweight training runtime.
    band = StatsRBFBandCombiner(
        V=np.array([[0.8, -0.8], [-0.6, 0.6], [0.4, -0.4], [0.0, 0.0]], np.float32),
        a=np.array([0.1, -0.1], np.float32),
        centers=np.array([[0.0, 0.0], [0.8, 0.1],
                          [2.0, 0.3], [3.5, 0.8]], np.float32),
        scales=np.array([1.0, 0.25], np.float32), sigma=1.0,
        surviving=np.ones(2, bool), std_floor=0.1,
    )
    comb = Combiner(member_labels=["00", "01"], n_kernels=4,
                    sigma_scale=1.0, min_usage=0.1, bands={"VIS": band},
                    band_names=("VIS",), kind=STATS_RBF_GATE_KIND)
    assert comb.kind == STATS_RBF_GATE_KIND
    assert comb.n_kernels == 4 and comb.min_usage == pytest.approx(0.1)
    assert band.centers.shape == (4, 2)
    assert band.scales.shape == (2,)
    assert band.std_floor == pytest.approx(0.1)
    weights = band.weights(X[:300])
    np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-5)
    assert np.all(weights >= 0.0)
    surface = band.weight_surface(n_mean=5, n_std=4)
    assert np.asarray(surface["mean_asinh"]).shape == (5,)
    assert np.asarray(surface["std_asinh"]).shape == (4,)
    assert float(surface["std_asinh"][0]) == 0.0
    assert float(surface["std_asinh"][-1]) > float(np.exp(band.centers[:, 1].max()) - band.std_floor)
    surface_w = np.asarray(surface["weights"])
    assert surface_w.shape == (4, 5, 2)
    np.testing.assert_allclose(surface_w.sum(axis=2), 1.0, atol=1e-5)

    save_combiner(comb, str(tmp_path), artifact_dir="stats_rbf_combiner")
    loaded = load_combiner(str(tmp_path), member_labels=["00", "01"],
                           artifact_dir="stats_rbf_combiner")
    assert loaded is not None and loaded.kind == STATS_RBF_GATE_KIND
    assert loaded.bands["VIS"].std_floor == pytest.approx(band.std_floor)
    np.testing.assert_allclose(loaded.bands["VIS"].forward_asinh(X[:300]),
                               band.forward_asinh(X[:300]), rtol=1e-5, atol=1e-5)


def test_minmax_rbf_surface_adapts_to_fitted_feature_range():
    band = StatsRBFBandCombiner(
        V=np.zeros((3, 2), np.float32), a=np.zeros(2, np.float32),
        centers=np.array([[-2.0, 4.0], [14.0, 21.0], [8.0, 16.0]], np.float32),
        scales=np.array([1.0, 2.0], np.float32), sigma=1.0,
        surviving=np.ones(2, bool), std_floor=0.1,
        feature_kind=MINMAX_RBF_GATE_KIND,
    )

    surface = band.weight_surface(n_mean=7, n_std=6)
    minimum = np.asarray(surface["mean_asinh"])
    maximum = np.asarray(surface["std_asinh"])
    assert minimum[0] < -2.0 and minimum[-1] > 14.0
    assert maximum[0] < 4.0 and maximum[-1] > 21.0
    assert surface["x_label"] == "min"
    assert surface["y_label"] == "max"


def test_stacked_rbf_is_convex_roundtrips_and_tracks_parent_artifacts(tmp_path):
    rng = np.random.default_rng(31)
    X = rng.normal(1.0, 0.45, (600, 2)).astype(np.float32)
    # Two deliberately distinct parent experts: mean+std routes to member 0,
    # while min+max routes to member 1. The stack therefore gets a meaningful
    # two-prediction input without involving model inference or TensorFlow.
    stats_band = StatsRBFBandCombiner(
        V=np.zeros((1, 2), np.float32), a=np.array([5.0, -5.0], np.float32),
        centers=np.array([[1.0, 0.0]], np.float32), scales=np.ones(2, np.float32),
        sigma=1.0, surviving=np.ones(2, bool), std_floor=0.1,
        feature_kind=STATS_RBF_GATE_KIND,
    )
    minmax_band = StatsRBFBandCombiner(
        V=np.zeros((1, 2), np.float32), a=np.array([-5.0, 5.0], np.float32),
        centers=np.array([[0.0, 2.0]], np.float32), scales=np.ones(2, np.float32),
        sigma=1.0, surviving=np.ones(2, bool), std_floor=0.1,
        feature_kind=MINMAX_RBF_GATE_KIND,
    )
    labels = ["00", "01"]
    stats = Combiner(labels, 1, 1.0, 0.0, {"VIS": stats_band},
                     band_names=("VIS",), kind=STATS_RBF_GATE_KIND)
    minmax = Combiner(labels, 1, 1.0, 0.0, {"VIS": minmax_band},
                      band_names=("VIS",), kind=MINMAX_RBF_GATE_KIND)
    # Truth switches between the two experts, so the fitted stack can learn a
    # useful local choice while remaining a convex interpolation everywhere.
    y = np.where(X.mean(axis=1) > 1.0, X[:, 0], X[:, 1]).astype(np.float32)
    buffers = {"VIS": (X, y)}
    stack = fit_stacked_combiner(
        buffers, labels, stats_combiner=stats, minmax_combiner=minmax,
        n_kernels=6, steps=80, batch=128, seed=4,
    )
    experts = stack.expert_asinh("VIS", X)
    out = stack.bands["VIS"].forward_asinh(experts)
    weights = stack.band_weights("VIS", X)
    np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-6)
    assert np.all(weights >= 0.0)
    assert np.all(out >= experts.min(axis=1) - 1e-6)
    assert np.all(out <= experts.max(axis=1) + 1e-6)
    np.testing.assert_allclose(
        stack.bands["VIS"].forward_asinh(np.array([[2.0, 2.0]])), 2.0,
        atol=1e-6,
    )

    save_combiner(stats, str(tmp_path), artifact_dir="stats_rbf_combiner")
    save_combiner(minmax, str(tmp_path), artifact_dir="minmax_rbf_combiner")
    stack.parent_fingerprints = {
        STATS_RBF_GATE_KIND: combiner_artifact_fingerprint(
            str(tmp_path), "stats_rbf_combiner"),
        MINMAX_RBF_GATE_KIND: combiner_artifact_fingerprint(
            str(tmp_path), "minmax_rbf_combiner"),
    }
    save_combiner(stack, str(tmp_path), artifact_dir="stacked_rbf_combiner")
    loaded = load_combiner(
        str(tmp_path), member_labels=labels, artifact_dir="stacked_rbf_combiner")
    assert loaded is not None and loaded.kind == STACKED_RBF_GATE_KIND
    np.testing.assert_allclose(loaded.band_weights("VIS", X[:50]),
                               stack.band_weights("VIS", X[:50]), atol=1e-6)

    # A separately refitted parent invalidates the derived stack instead of
    # silently combining a new parent with an old meta-gate.
    stats.bands["VIS"].a[0] += 0.5
    save_combiner(stats, str(tmp_path), artifact_dir="stats_rbf_combiner")
    assert load_combiner(str(tmp_path), member_labels=labels,
                         artifact_dir="stacked_rbf_combiner") is None


def test_stats_rbf_fit_keeps_all_members_and_persists_weight_diagnostics(monkeypatch, tmp_path):
    import euclid_polish.eval.combiner as combiner_mod

    X = np.tile(np.array([[2.0, 0.0]], np.float32), (200, 1))
    y = X[:, 0].copy()
    std_floor = 0.1

    def fake_fit(*_args, **_kwargs):
        # On the occupied point (mean=1, std=1), the RBF strongly selects
        # member 0. Away from it, the bias selects member 1; consequently the
        # former 1-D std=0 probe would make the opposite pruning decision.
        band = StatsRBFBandCombiner(
            V=np.array([[6.0, -6.0]], np.float32),
            a=np.array([-3.0, 3.0], np.float32),
            centers=np.array([[1.0, np.log(1.0 + std_floor)]], np.float32),
            scales=np.ones(2, np.float32), sigma=0.25,
            surviving=np.ones(2, bool), std_floor=std_floor,
        )
        return band, X, y

    monkeypatch.setattr(combiner_mod, "_fit_one_band_stats_rbf", fake_fit)
    buffers = {"VIS": (X, y)}
    unpruned = fit_combiner(buffers, ["00", "01"], n_kernels=2,
                            min_usage=0.4, model_kind=STATS_RBF_GATE_KIND)
    assert unpruned.bands["VIS"].surviving.tolist() == [True, True]
    assert unpruned.member_importance == {}
    peaks, integrals = member_weight_diagnostics(unpruned, buffers)
    unpruned.member_weight_peaks = peaks
    unpruned.member_weight_integrals = integrals
    save_combiner(unpruned, str(tmp_path), artifact_dir="stats_rbf_combiner")
    loaded = load_combiner(str(tmp_path), artifact_dir="stats_rbf_combiner")
    assert loaded is not None
    assert loaded.member_weight_peaks == peaks
    assert loaded.member_weight_integrals == integrals


def test_build_fit_buffers_stratified_balances_brightness():
    rng = np.random.default_rng(9)
    H = W = 60
    M = 2
    hr = np.abs(rng.normal(80.0, 400.0, (H, W, 4))).astype(np.float32)
    preds = np.stack([hr, hr + rng.normal(0, 5, (H, W, 4))], 0).astype(np.float32)
    buffers = build_fit_buffers_from_fields(
        iter([(preds, hr)]), ("VIS", "Y_E", "J_E", "H_E"),
        n_bright_bins=8, per_bin_per_field=40, seed=0)
    Xb, yb = buffers["VIS"]
    assert Xb.shape[1] == M and Xb.shape[0] == yb.shape[0]
    edges = np.linspace(-1.0, 12.0, 9)
    counts = np.bincount(np.clip(np.digitize(yb, edges) - 1, 0, 7), minlength=8)
    assert counts.max() <= 40
    assert int((counts > 0).sum()) >= 3
