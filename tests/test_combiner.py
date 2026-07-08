"""The per-band ensemble combiner (tiny MLP, L1 loss, group-L1 pruning).

These tests operate directly on in-memory fit buffers (a dict
``{band: (X(N,M), y(N,))}`` in asinh space) so no real checkpoints or TFRecords
are needed — the combiner math, pruning, persistence, per-band independence and
inverse-stretch are all exercised on synthetic member stacks.
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
    """asinh-space L1 of the plain member mean vs the target."""
    return float(np.mean(np.abs(X.mean(axis=1) - y)))


def _one_band_good_vs_noise(n=6000, seed=0):
    """Two members: member 0 tracks the target, member 1 is pure noise.
    Returns asinh-space (X (n,2), y (n,))."""
    rng = np.random.default_rng(seed)
    y = _asinh(np.abs(rng.normal(30.0, 20.0, n)) ** 2)      # spread of brightness
    x0 = y + rng.normal(0, 0.02, n)                          # good member
    x1 = rng.normal(0, 1.0, n)                               # useless member
    X = np.stack([x0, x1], axis=1).astype(np.float32)
    return X, y.astype(np.float32)


def test_fit_reduces_l1_vs_mean():
    X, y = _one_band_good_vs_noise()
    comb = fit_combiner({"VIS": (X, y)}, ["00", "01"],
                        hidden=3, lam_group=1e-4, steps=400, seed=0)
    pred = comb.bands["VIS"].forward_asinh(X)
    fit_l1 = float(np.mean(np.abs(pred - y)))
    assert fit_l1 < _mean_l1(X, y)          # beats the naive mean
    assert fit_l1 < 0.2                      # actually tracks the good member


def test_group_l1_prunes_useless_member():
    """A member uncorrelated with the target should be pruned (survivor False)
    and read ~0 in the effective-weight probe."""
    X, y = _one_band_good_vs_noise()
    comb = fit_combiner({"VIS": (X, y)}, ["00", "01"],
                        hidden=3, lam_group=3e-2, steps=600, seed=1)
    surviving = comb.bands["VIS"].surviving
    assert bool(surviving[0]) is True        # good member kept
    assert bool(surviving[1]) is False       # noise member pruned
    eff = comb.effective_weights("VIS")
    jac = np.asarray(eff["jacobian"])         # (n_levels, M)
    assert np.abs(jac[:, 1]).max() < 0.1      # pruned member ~0 everywhere


def test_persist_load_roundtrip(tmp_path):
    X, y = _one_band_good_vs_noise()
    comb = fit_combiner({"VIS": (X, y)}, ["00", "01"],
                        hidden=3, lam_group=1e-4, steps=200, seed=2)
    save_combiner(comb, str(tmp_path))
    back = load_combiner(str(tmp_path), member_labels=["00", "01"])
    assert back is not None
    np.testing.assert_allclose(back.bands["VIS"].forward_asinh(X),
                               comb.bands["VIS"].forward_asinh(X),
                               rtol=1e-5, atol=1e-5)
    assert back.member_labels == ["00", "01"]
    assert back.hidden == comb.hidden
    assert back.lam_group == pytest.approx(comb.lam_group)


def test_load_returns_none_on_member_mismatch(tmp_path):
    X, y = _one_band_good_vs_noise()
    comb = fit_combiner({"VIS": (X, y)}, ["00", "01"], hidden=2, steps=100)
    save_combiner(comb, str(tmp_path))
    # active ensemble changed (a member was archived) → stale → None
    assert load_combiner(str(tmp_path), member_labels=["00", "02"]) is None
    # no filter → loads regardless
    assert load_combiner(str(tmp_path)) is not None


def test_per_band_independence():
    """Band 0 favours member A, band 1 favours member B → the two bands' fitted
    combiners must trust different members (each MLP sees only its own band)."""
    rng = np.random.default_rng(3)
    n = 6000
    yA = _asinh(np.abs(rng.normal(30, 20, n)) ** 2).astype(np.float32)
    yB = _asinh(np.abs(rng.normal(30, 20, n)) ** 2).astype(np.float32)
    # VIS: member A good, member B noise. H band: reversed.
    Xvis = np.stack([yA + rng.normal(0, .02, n), rng.normal(0, 1, n)], 1).astype(np.float32)
    Xh = np.stack([rng.normal(0, 1, n), yB + rng.normal(0, .02, n)], 1).astype(np.float32)
    comb = fit_combiner({"VIS": (Xvis, yA), "H_E": (Xh, yB)}, ["00", "01"],
                        hidden=3, lam_group=3e-2, steps=600, seed=4)
    assert bool(comb.bands["VIS"].surviving[0]) and not bool(comb.bands["VIS"].surviving[1])
    assert bool(comb.bands["H_E"].surviving[1]) and not bool(comb.bands["H_E"].surviving[0])


def test_apply_field_shape_and_inverse_stretch():
    """apply_field returns (H,W,C) electrons. A linear identity band-combiner
    (h = mean of asinh inputs, output = h) must recover electrons when all
    members agree — checks the 100*sinh inverse."""
    M, H, W = 3, 8, 8
    bands = {}
    for b in ("VIS", "Y_E", "J_E", "H_E"):
        # y = mean of the asinh member inputs, carried entirely by the skip;
        # the (zeroed) hidden part contributes nothing.
        bands[b] = BandCombiner(
            W1=np.zeros((M, 1), np.float32), b1=np.zeros(1, np.float32),
            W2=np.zeros(1, np.float32), b2=0.0,
            w_skip=np.full(M, 1.0 / M, np.float32), activation="linear",
            surviving=np.ones(M, bool))
    comb = Combiner(member_labels=["00", "01", "02"], hidden=1, lam_group=0.0,
                    bands=bands, band_names=("VIS", "Y_E", "J_E", "H_E"))
    rng = np.random.default_rng(5)
    # all members identical → mean of asinh == asinh(value) → recovers electrons
    field = np.abs(rng.normal(500, 200, (H, W, 4))).astype(np.float32)
    preds = np.stack([field, field, field], axis=0)        # (M,H,W,C)
    out = comb.apply_field(preds)
    assert out.shape == (H, W, 4)
    np.testing.assert_allclose(out, field, rtol=1e-4, atol=1e-2)


def test_single_member_degenerate():
    """M=1: fit runs, the lone member cannot be pruned, apply_field is a
    near-identity warp."""
    rng = np.random.default_rng(6)
    y = _asinh(np.abs(rng.normal(40, 30, 4000)) ** 2).astype(np.float32)
    X = (y + rng.normal(0, 0.02, y.shape)).astype(np.float32)[:, None]   # (n,1)
    comb = fit_combiner({"VIS": (X, y)}, ["00"], hidden=2, lam_group=1e-2,
                        steps=300, seed=7)
    assert comb.bands["VIS"].surviving.tolist() == [True]
    # monotone, finite reconstruction that correlates with the member
    pred = comb.bands["VIS"].forward_asinh(X)
    assert np.isfinite(pred).all()
    assert np.corrcoef(pred, y)[0, 1] > 0.99


def test_build_fit_buffers_stratified_balances_brightness():
    """The per-band buffer builder subsamples pixels stratified by brightness:
    no bin exceeds the per-bin quota, and several bins are represented — so
    faint pixels aren't drowned by the (dominant) sky."""
    rng = np.random.default_rng(9)
    H = W = 60
    M = 2
    hr = np.abs(rng.normal(80.0, 400.0, (H, W, 4))).astype(np.float32)
    preds = np.stack([hr, hr + rng.normal(0, 5, (H, W, 4))], 0).astype(np.float32)

    buffers = build_fit_buffers_from_fields(
        iter([(preds, hr)]), ("VIS", "Y_E", "J_E", "H_E"),
        n_bright_bins=8, per_bin_per_field=40, seed=0)

    X, y = buffers["VIS"]
    assert X.shape[1] == M and X.shape[0] == y.shape[0]
    edges = np.linspace(-1.0, 12.0, 9)
    counts = np.bincount(np.clip(np.digitize(y, edges) - 1, 0, 7),
                         minlength=8)
    assert counts.max() <= 40                     # quota respected per bin
    assert int((counts > 0).sum()) >= 3           # stratification happened


def test_effective_weights_shape_and_baseline():
    X, y = _one_band_good_vs_noise()
    comb = fit_combiner({"VIS": (X, y)}, ["00", "01"], hidden=3, steps=200)
    eff = comb.effective_weights("VIS", n_levels=20)
    assert np.asarray(eff["jacobian"]).shape == (20, 2)
    assert np.asarray(eff["brightness_e"]).shape == (20,)
    assert np.isfinite(eff["jacobian"]).all()
