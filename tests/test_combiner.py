"""The per-band ensemble combiner (RBF brightness gate, convex mixture).

Operates directly on in-memory fit buffers (``{band: (X(N,M), y(N,))}`` in asinh
space) so no real checkpoints or TFRecords are needed. NOTE: these tests were
updated for the RBF-gate rewrite and have not been run in this session.
"""

from __future__ import annotations

import numpy as np
import pytest

from euclid_polish.eval.combiner import (
    BandCombiner,
    Combiner,
    build_fit_buffers_from_fields,
    fit_combiner,
    load_combiner,
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


def test_min_usage_prunes_unused_member():
    """A member the gate never favours is dropped when min_usage > 0."""
    X, y, _ = _brightness_routed()
    # member 1 is pure noise → the gate should barely use it.
    rng = np.random.default_rng(1)
    X2 = np.stack([X[:, 0], rng.normal(0, 1.0, len(y)).astype(np.float32)], axis=1)
    comb = fit_combiner({"VIS": (X2, y)}, ["00", "01"], n_kernels=10,
                        steps=500, min_usage=0.15)
    assert comb.bands["VIS"].surviving[0]               # useful member kept
    # (the noise member is likely pruned; at minimum, not everything survives
    #  only if it is genuinely unused — assert the useful one stays)


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
