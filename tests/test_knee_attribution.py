"""Knee attribution core: the exact chain-rule split of a multi-knee network's
sensitivity into knee terms, windows that reproduce the full field, the
shares, the pixel classes, the stratified sampling and the aggregates."""
from __future__ import annotations

import json

import numpy as np
import pytest
import tensorflow as tf

from euclid_polish.eval import knee_attribution as ka
from euclid_polish.model import Model
from euclid_polish.training.augmentation import asinh_stretch_multi_knee
from euclid_polish.training.models.wdsr import wdsr

KNEES = (0.01, 1.0, 100.0)


def _net(depth=1, seed=3):
    tf.keras.utils.set_random_seed(seed)
    model = wdsr(scale=2, num_res_blocks=depth, nchan_in=4 * len(KNEES), nchan_out=4,
                 input_knees=len(KNEES), learned_output_knee=10.0, eager_weight_norm_init=True)
    return ka.KneeNet(model=model, knees=KNEES, depth=depth, output="electrons")


def _field(n, seed=0):
    """LR electrons from sky noise (negative too) up to bright cores."""
    rng = np.random.default_rng(seed)
    lr = rng.lognormal(1.0, 1.5, (n, n, 4)) + rng.normal(0.0, 1.0, (n, n, 4))
    return lr.astype(np.float32)


def _full_field(net, lr, y, x, band):
    """Reference on the whole field: ``(dy/ds, dy/dx)`` of the predicted
    electrons at HR pixel ``(y, x)`` and ``band``."""
    xe = tf.constant(lr[None])
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(xe)
        s = asinh_stretch_multi_knee(xe, net.knees)
        value = net.predict_electrons(s)[0, y, x, band]
    g_s, g_x = tape.gradient(value, s), tape.gradient(value, xe)
    return g_s.numpy()[0], g_x.numpy()[0]


def _reference_shares(net, lr, y, x, band, sigma, radius=None):
    """``(knee_share, band_share, sensitivity, knee_share_by_input)`` on the
    whole field, each input band's terms weighted by its ``sigma``; with
    ``radius``, only the LR pixels within it of the sample's LR pixel."""
    g_s, _ = _full_field(net, lr, y, x, band)
    h, w, c = lr.shape
    q = np.asarray(net.knees, np.float64)[:, None]
    t = g_s.reshape(h, w, len(net.knees), c) / np.sqrt(q ** 2 + lr[:, :, None, :].astype(np.float64) ** 2)
    if radius is not None:
        rows, cols = np.abs(np.arange(h) - y // 2) <= radius, np.abs(np.arange(w) - x // 2) <= radius
        t = t * (rows[:, None] & cols[None, :])[:, :, None, None]
    a = np.abs(t).sum(axis=(0, 1)) * np.asarray(sigma, np.float64)[None, :]   # (K, C)
    by_input = a.T / a.T.sum(axis=1, keepdims=True)
    return a.sum(axis=1) / a.sum(), a.sum(axis=0) / a.sum(), a.sum(), by_input


def _samples(rows):
    band, y, x = (np.asarray(v) for v in zip(*rows, strict=True))
    return {"band": band.astype(np.int16), "y": y.astype(np.int32), "x": x.astype(np.int32),
            "code": np.zeros(len(rows), np.int16)}


# --------------------------------------------------------------------------- #
# constants
# --------------------------------------------------------------------------- #

def test_edges_come_from_the_gate_diagnostic():
    assert ka.PRED_CLASSES == ("sky", "faint", "mid", "bright", "core")
    assert ka.REAL_CLASSES == ka.PRED_CLASSES + ("blank",)
    np.testing.assert_allclose(ka.BRIGHTNESS_EDGES_E, 100.0 * np.sinh([0.02, 0.1, 0.5, 2.0]))
    np.testing.assert_allclose(ka.BRIGHTNESS_EDGES_E, (2.0, 10.0, 52.1, 362.7), rtol=2e-3)
    assert len(ka.LOG_EDGES_E) == 21
    np.testing.assert_allclose(ka.LOG_EDGES_E, 10.0 ** np.arange(-3.0, 7.01, 0.5))


def test_window_side_covers_the_receptive_field():
    assert ka.window_side(1) == 64
    assert ka.window_side(16) == 64
    assert ka.window_side(28) == 64
    assert ka.window_side(30) == 68
    assert ka.window_side(31) == 70


# --------------------------------------------------------------------------- #
# the chain-rule split and the windows
# --------------------------------------------------------------------------- #

def test_the_knee_terms_sum_to_the_direct_electron_gradient():
    net = _net()
    lr = _field(16)
    values, terms = ka._terms(net, tf.constant(lr[None]), np.array([13]), np.array([17]), np.array([2]))
    t = terms.numpy()[0]                                # (h, w, K, C)
    assert t.shape == (16, 16, len(KNEES), 4)
    _, direct = _full_field(net, lr, 13, 17, 2)
    assert np.abs(direct).max() > 0
    np.testing.assert_allclose(t.sum(axis=2), direct, rtol=1e-4, atol=1e-6 * np.abs(direct).max())
    assert (np.abs(t).sum(axis=(0, 1, 3)) > 0).all()   # every knee carries some of it
    want = net.predict_electrons(tf.constant(ka.stretch(lr[None], KNEES))).numpy()[0, 13, 17, 2]
    np.testing.assert_allclose(values.numpy()[0], want, rtol=1e-5)


def test_a_window_reproduces_the_full_field_everywhere():
    net = _net()
    lr = _field(96, seed=1)                             # larger than the 64 window
    rows = [(0, 100, 90), (3, 101, 91), (1, 1, 190), (2, 0, 0), (3, 191, 191), (0, 60, 2)]
    got = ka.attribute(net, lr, _samples(rows), ablation=False)
    assert "ablation_share" not in got
    sigma = ka.noise_sigma(lr)                          # the default weights
    for i, (band, y, x) in enumerate(rows):
        knee, bands, sens, by_input = _reference_shares(net, lr, y, x, band, sigma)
        np.testing.assert_allclose(got["knee_share"][i], knee, rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(got["band_share"][i], bands, rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(got["sensitivity"][i], sens, rtol=1e-4)
        np.testing.assert_allclose(got["knee_share_by_input"][i], by_input, rtol=1e-4, atol=1e-6)
        knee, bands, _, by_input = _reference_shares(net, lr, y, x, band, sigma, radius=ka.LOCAL_RADIUS)
        np.testing.assert_allclose(got["knee_share_local"][i], knee, rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(got["band_share_local"][i], bands, rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(got["knee_share_by_input_local"][i], by_input, rtol=1e-4, atol=1e-6)


def test_noise_sigma_is_each_bands_own_noise():
    rng = np.random.default_rng(5)
    n, truth = 200, np.array([18.0, 0.8, 1.0, 0.0])
    yy, xx = np.mgrid[:n, :n]
    galaxy = 500.0 * np.exp(-((yy - 100) ** 2 + (xx - 90) ** 2) / (2 * 6.0 ** 2))
    lr = galaxy[..., None] + rng.normal(0.0, 1.0, (n, n, 4)) * truth + 3.0
    lr[40:60, 40:60, :] = 0.0                           # a blanked hole is not noise
    lr[0, :5, 1] = np.nan
    lr[..., 3] = 0.0                                    # a dead band takes the others' median
    sigma = ka.noise_sigma(lr)
    assert sigma.dtype == np.float32 and sigma.shape == (4,)
    np.testing.assert_allclose(sigma[:3], truth[:3], rtol=0.05)
    assert sigma[3] == pytest.approx(float(np.median(sigma[:3])))
    assert (ka.noise_sigma(np.zeros((8, 8, 2))) == 1.0).all()
    with pytest.raises(ValueError, match="h, w, C"):
        ka.noise_sigma(np.zeros((8, 8)))


def test_bands_are_weighted_by_their_noise():
    """σ weights each input band's terms: the band shares move with σ, each
    input band's own knee split does not, and the knee share is the band-share
    weighted mix of the per-input splits."""
    net = _net()
    lr = _field(48, seed=6)
    samples = _samples([(0, 40, 41), (2, 10, 70), (3, 90, 5)])
    flat = ka.attribute(net, lr, samples, ablation=False, sigma_e=[1, 1, 1, 1])
    tilted = ka.attribute(net, lr, samples, ablation=False, sigma_e=[20, 1, 1, 1])
    np.testing.assert_allclose(tilted["knee_share_by_input"], flat["knee_share_by_input"], rtol=1e-5, atol=1e-7)
    b, b20 = flat["band_share"].astype(np.float64), tilted["band_share"].astype(np.float64)
    np.testing.assert_allclose(b20[:, 0], 20 * b[:, 0] / (20 * b[:, 0] + 1 - b[:, 0]), rtol=1e-4)
    for got in (flat, tilted):
        mix = np.einsum("nc,nck->nk", got["band_share"], got["knee_share_by_input"])
        np.testing.assert_allclose(mix, got["knee_share"], rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(got["knee_share_by_input"].sum(axis=2), 1.0, rtol=1e-5)
    with pytest.raises(ValueError, match="one σ per input band"):
        ka.attribute(net, lr, samples, sigma_e=[1, 1])


def test_shares_sum_to_one_and_are_non_negative():
    net = _net()
    lr = _field(40, seed=2)
    rng = np.random.default_rng(0)
    n = 37                                              # not a multiple of the batch
    samples = {"band": rng.integers(0, 4, n).astype(np.int16), "y": rng.integers(0, 80, n).astype(np.int32),
               "x": rng.integers(0, 80, n).astype(np.int32), "code": np.zeros(n, np.int16)}
    got = ka.attribute(net, lr, samples, batch=8)
    assert got["knee_share"].shape == (n, len(KNEES)) and got["knee_share"].dtype == np.float32
    assert got["band_share"].shape == (n, 4) and got["ablation_share"].shape == (n, len(KNEES))
    assert got["knee_share_by_input"].shape == got["knee_share_by_input_local"].shape == (n, 4, len(KNEES))
    assert got["knee_share_local"].shape == (n, len(KNEES)) and got["band_share_local"].shape == (n, 4)
    assert got["sensitivity"].shape == (n,)
    for key in ("knee_share", "band_share", "ablation_share", "knee_share_by_input",
                "knee_share_local", "band_share_local", "knee_share_by_input_local"):
        assert (got[key] >= 0).all()
        np.testing.assert_allclose(got[key].sum(axis=-1), 1.0, rtol=1e-5)
    assert (got["sensitivity"] > 0).all()
    once = ka.attribute(net, lr, samples, batch=64)    # batching does not change anything
    for key in got:
        np.testing.assert_allclose(once[key], got[key], rtol=1e-5, atol=1e-7)


def test_ablation_replaces_each_knee_with_its_window_mean():
    net = _net()
    lr = _field(80, seed=4)
    band, y, x = 1, 81, 77                              # LR (40, 38): the window is rows/cols 8..71 / 6..69
    got = ka.attribute(net, lr, _samples([(band, y, x)]))
    window = lr[40 - 32:40 + 32, 38 - 32:38 + 32]
    s = ka.stretch(window[None], KNEES)
    base = net.predict_electrons(tf.constant(s)).numpy()[0, y - 16, x - 12, band]
    deltas = []
    for k in range(len(KNEES)):
        ablated = s.copy()
        block = ablated[..., 4 * k:4 * k + 4]
        ablated[..., 4 * k:4 * k + 4] = block.mean(axis=(1, 2), keepdims=True)
        value = net.predict_electrons(tf.constant(ablated)).numpy()[0, y - 16, x - 12, band]
        deltas.append(abs(base - value))
    deltas = np.asarray(deltas)
    assert deltas.sum() > 0
    np.testing.assert_allclose(got["ablation_share"][0], deltas / deltas.sum(), rtol=1e-3, atol=1e-6)


def test_ablation_of_a_flat_field_is_all_zero():
    net = _net()
    lr = np.full((20, 20, 4), 7.0, np.float32)          # every knee block is its own mean
    got = ka.attribute(net, lr, _samples([(0, 10, 10), (2, 3, 30)]))
    assert np.all(got["ablation_share"] == 0.0)
    np.testing.assert_allclose(got["knee_share"].sum(axis=1), 1.0, rtol=1e-5)


def test_attribute_rejects_bad_input():
    net = _net()
    with pytest.raises(ValueError, match="bands"):
        ka.attribute(net, np.zeros((8, 8, 3), np.float32), _samples([(0, 1, 1)]))
    with pytest.raises(ValueError, match="outside"):
        ka.attribute(net, np.zeros((8, 8, 4), np.float32), _samples([(0, 16, 1)]))
    none = {"band": np.zeros(0, np.int16), "y": np.zeros(0, np.int32), "x": np.zeros(0, np.int32)}
    empty = ka.attribute(net, np.zeros((8, 8, 4), np.float32), none)
    assert empty["knee_share"].shape == (0, len(KNEES)) and empty["sensitivity"].shape == (0,)
    assert empty["ablation_share"].shape == (0, len(KNEES))


# --------------------------------------------------------------------------- #
# pixel classes
# --------------------------------------------------------------------------- #

def test_predicted_classes_bin_the_sr_at_the_gate_edges():
    edges = ka.BRIGHTNESS_EDGES_E
    sr = np.array([-1.0, 1.9, 5.0, 30.0, edges[2], 100.0, 1000.0, np.nan], np.float32)[None, :, None]
    codes, value = ka.predicted_classes(sr)
    assert codes.dtype == np.int8 and value.dtype == np.float32
    assert [ka.PRED_CLASSES[c] for c in codes[0, :, 0]] == [
        "sky", "sky", "faint", "mid", "bright", "bright", "core", "sky"]
    np.testing.assert_array_equal(value, sr)


def test_truth_classes_follow_the_priority_order():
    # (galaxy, star) per HR pixel, 2 rows x 8 columns, one band; LR is 1 x 4.
    pairs = [[(0, 900), (900, 0), (100, 300), (300, 100), (10, 20), (20, 10), (0.5, 0), (0.005, 0)],
             [(1, 1), (60, 0), (25, 25), (24.9, 25.1), (0, 0), (-3, 0), (0.011, 0), (40, 40)]]
    arr = np.asarray(pairs, np.float32)
    galaxy, star = arr[..., 0:1], arr[..., 1:2]
    lr = np.array([0.0, 5.0, 5.0, 5.0], np.float32)[None, :, None]
    codes, value = ka.truth_classes(galaxy, star, lr)
    assert codes.dtype == np.int8 and codes.shape == (2, 8, 1)
    names = [[ka.TRUTH_CLASSES[c] for c in row] for row in codes[..., 0]]
    assert names[0] == ["saturated", "saturated", "star core", "galaxy core", "star wing",
                        "galaxy outskirts", "faint", "sky"]
    # a star exactly half the truth is not a star; a faint zero-LR pixel is not saturated
    assert names[1] == ["faint", "saturated", "galaxy outskirts", "star wing", "sky", "sky",
                        "faint", "galaxy core"]
    np.testing.assert_allclose(value, galaxy + star)
    assert set(np.unique(codes)) == set(range(len(ka.TRUTH_CLASSES)))
    with pytest.raises(ValueError):
        ka.truth_classes(galaxy, star, np.zeros((1, 2, 1), np.float32))


def test_real_classes_mark_blank_lr_pixels_per_band():
    lr = np.array([[[0.0, 3.0], [7.0, np.nan]]], np.float32)          # (1, 2, 2)
    sr = np.empty((2, 4, 2), np.float32)
    sr[..., 0] = [1.0, 5.0, 20.0, 100.0]
    sr[..., 1] = [1000.0, 30.0, 5.0, 1.0]
    codes, value = ka.real_classes(sr, lr)
    names = [[ka.REAL_CLASSES[c] for c in codes[0, :, b]] for b in range(2)]
    assert names[0] == ["blank", "blank", "mid", "bright"]
    assert names[1] == ["core", "mid", "blank", "blank"]
    assert (codes[0] == codes[1]).all()
    np.testing.assert_array_equal(value, sr)


# --------------------------------------------------------------------------- #
# sampling and aggregation
# --------------------------------------------------------------------------- #

def test_sample_pixels_is_stratified_and_reproducible():
    codes = np.zeros((10, 12, 2), np.int8)
    codes[0, :5, 0] = 1
    codes[9, :2, 0] = 3
    codes[..., 1] = 2
    picked = ka.sample_pixels(codes, 4, np.random.default_rng(7))
    assert picked["band"].dtype == np.int16 and picked["code"].dtype == np.int16
    assert picked["y"].dtype == np.int32 and picked["x"].dtype == np.int32
    np.testing.assert_array_equal(codes[picked["y"], picked["x"], picked["band"]], picked["code"])
    counts = {(int(b), int(c)): 0 for b, c in zip(picked["band"], picked["code"], strict=True)}
    for b, c in zip(picked["band"], picked["code"], strict=True):
        counts[int(b), int(c)] += 1
    assert counts == {(0, 0): 4, (0, 1): 4, (0, 3): 2, (1, 2): 4}
    cells = set(zip(picked["band"].tolist(), picked["y"].tolist(), picked["x"].tolist(), strict=True))
    assert len(cells) == len(picked["band"])            # without replacement
    again = ka.sample_pixels(codes, 4, np.random.default_rng(7))
    for key in picked:
        np.testing.assert_array_equal(again[key], picked[key])
    other = ka.sample_pixels(codes, 4, np.random.default_rng(8))
    assert not all(np.array_equal(other[k], picked[k]) for k in ("y", "x"))


def test_aggregate_means_classes_and_brightness_bins():
    k3 = [[0.6, 0.3, 0.1], [0.2, 0.2, 0.6], [0.1, 0.8, 0.1]]
    part1 = {"band": np.array([0, 0, 1]), "code": np.array([0, 0, 2]),
             "value_e": np.array([5e-4, 1.0, 1e8]),
             "knee_share": np.array(k3, np.float32),
             "ablation_share": np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]], np.float32),
             "band_share": np.array([[1.0, 0, 0, 0], [0.5, 0.5, 0, 0], [0, 0, 1.0, 0]], np.float32)}
    part2 = {"band": np.array([0]), "code": np.array([1]), "value_e": np.array([1.5]),
             "knee_share": np.array([[0.4, 0.4, 0.2]], np.float32),
             "ablation_share": np.array([[0.5, 0.5, 0.0]], np.float32),
             "band_share": np.array([[0.0, 0.0, 0.0, 1.0]], np.float32)}
    empty = {key: value[:0] for key, value in part1.items()}
    names = ("c0", "c1", "c2")
    agg = ka.aggregate([part1, empty, part2], names, 4, KNEES)
    json.dumps(agg, allow_nan=False)
    assert agg["n_samples"] == 4
    by_class = agg["by_class"]
    assert set(by_class) == {"0", "1", "2", "3"} and by_class["2"] == {} and by_class["3"] == {}
    assert list(by_class["0"]) == ["c0", "c1"] and list(by_class["1"]) == ["c2"]
    c0 = by_class["0"]["c0"]
    assert c0["n"] == 2
    np.testing.assert_allclose(c0["knee_share"], [0.4, 0.25, 0.35])
    np.testing.assert_allclose(c0["ablation_share"], [0.5, 0.5, 0.0])
    np.testing.assert_allclose(c0["band_share"], [0.75, 0.25, 0.0, 0.0])
    np.testing.assert_allclose(by_class["0"]["c1"]["knee_share"], [0.4, 0.4, 0.2])
    assert by_class["1"]["c2"]["n"] == 1
    bright = agg["by_brightness"]
    np.testing.assert_allclose(bright["edges_e"], ka.LOG_EDGES_E)
    band0 = bright["bands"]["0"]
    assert len(band0["n"]) == len(band0["knee_share"]) == len(ka.LOG_EDGES_E) - 1
    assert band0["n"][0] == 1 and band0["n"][6] == 2 and sum(band0["n"]) == 3
    np.testing.assert_allclose(band0["knee_share"][0], k3[0])
    np.testing.assert_allclose(band0["knee_share"][6], [0.3, 0.3, 0.4])
    assert band0["knee_share"][1] is None and band0["n"][1] == 0
    band1 = bright["bands"]["1"]
    assert band1["n"][-1] == 1 and band1["knee_share"][-1] == pytest.approx(k3[2])
    assert bright["bands"]["3"]["n"] == [0] * 20 and bright["bands"]["3"]["knee_share"] == [None] * 20
    cross = agg["cross_band"]
    assert len(cross) == 4
    np.testing.assert_allclose(cross[0], [0.5, 1 / 6, 0.0, 1 / 3])
    np.testing.assert_allclose(cross[1], [0.0, 0.0, 1.0, 0.0])
    assert cross[2] is None and cross[3] is None


def test_aggregate_means_the_per_input_knee_split():
    split_a = np.array([[[1.0, 0, 0], [0, 1.0, 0]], [[0, 0, 1.0], [0.5, 0.5, 0]]], np.float32)
    part = {"band": np.array([0, 0]), "code": np.array([0, 0]), "value_e": np.array([1.0, 1.2]),
            "knee_share": np.full((2, 3), 1 / 3, np.float32),
            "ablation_share": np.full((2, 3), 1 / 3, np.float32),
            "band_share": np.full((2, 2), 0.5, np.float32), "knee_share_by_input": split_a}
    agg = ka.aggregate([part], ("c0",), 2, KNEES)
    json.dumps(agg, allow_nan=False)
    np.testing.assert_allclose(agg["by_class"]["0"]["c0"]["knee_share_by_input"],
                               [[0.5, 0, 0.5], [0.25, 0.75, 0]])
    bins = agg["by_brightness"]["bands"]["0"]["knee_share_by_input"]
    assert len(bins) == len(ka.LOG_EDGES_E) - 1 and bins[0] is None
    np.testing.assert_allclose(bins[6], [[0.5, 0, 0.5], [0.25, 0.75, 0]])
    assert agg["by_class"]["0"]["c0"]["knee_share_local"] is None and "cross_band_local" not in agg
    # A part without the split (an older record's samples) leaves it out.
    old = {key: value for key, value in part.items() if key != "knee_share_by_input"}
    agg = ka.aggregate([part, old], ("c0",), 2, KNEES)
    assert agg["by_class"]["0"]["c0"]["knee_share_by_input"] is None
    assert "knee_share_by_input" not in agg["by_brightness"]["bands"]["0"]


def test_aggregate_carries_the_local_splits():
    local = {"knee_share_local": np.array([[0.0, 1.0, 0.0], [0.0, 0.5, 0.5]], np.float32),
             "band_share_local": np.array([[1.0, 0.0], [0.0, 1.0]], np.float32),
             "knee_share_by_input_local": np.array([[[0, 1.0, 0], [0, 0, 1.0]], [[1.0, 0, 0], [0, 1.0, 0]]],
                                                   np.float32)}
    part = {"band": np.array([1, 1]), "code": np.array([0, 0]), "value_e": np.array([200.0, 400.0]),
            "knee_share": np.full((2, 3), 1 / 3, np.float32), "ablation_share": np.full((2, 3), 1 / 3, np.float32),
            "band_share": np.full((2, 2), 0.5, np.float32),
            "knee_share_by_input": np.full((2, 2, 3), 1 / 3, np.float32)} | local
    agg = ka.aggregate([part], ("c0",), 2, KNEES)
    json.dumps(agg, allow_nan=False)
    c0 = agg["by_class"]["1"]["c0"]
    np.testing.assert_allclose(c0["knee_share_local"], [0.0, 0.75, 0.25])
    np.testing.assert_allclose(c0["band_share_local"], [0.5, 0.5])
    np.testing.assert_allclose(c0["knee_share_by_input_local"], [[0.5, 0.5, 0], [0, 0.5, 0.5]])
    np.testing.assert_allclose(agg["cross_band_local"][1], [0.5, 0.5])
    assert agg["cross_band_local"][0] is None
    bins = agg["by_brightness"]["bands"]["1"]
    np.testing.assert_allclose(bins["knee_share_local"][10], [0.0, 1.0, 0.0])     # 200 e⁻: the 100–316 bin, 400 the next
    np.testing.assert_allclose(bins["knee_share_local"][11], [0.0, 0.5, 0.5])
    np.testing.assert_allclose(bins["knee_share_by_input_local"][11], [[1.0, 0, 0], [0, 1.0, 0]])


def test_aggregate_of_nothing_is_empty_but_well_formed():
    agg = ka.aggregate([], ("a",), 2, KNEES)
    json.dumps(agg, allow_nan=False)
    assert agg["n_samples"] == 0 and agg["by_class"] == {"0": {}, "1": {}}
    assert agg["cross_band"] == [None, None]
    assert agg["by_brightness"]["bands"]["1"]["n"] == [0] * 20


# --------------------------------------------------------------------------- #
# members
# --------------------------------------------------------------------------- #

def test_from_member_refuses_a_single_knee_model(tmp_path):
    member = Model(str(tmp_path / "m"), scale=2, num_res_blocks=1)
    with pytest.raises(ValueError, match="not a multi-knee model"):
        ka.KneeNet.from_member(member)


@pytest.mark.parametrize("kind", ["electrons", "stretched", "heads"])
def test_from_member_serves_the_members_own_image(tmp_path, kind):
    extra = {"electrons": {"output_knee": 10.0, "learn_output_knee": True},
             "stretched": {"output_knee": 10.0}, "heads": {}}[kind]
    member = Model(str(tmp_path / "m"), scale=2, num_res_blocks=1, asinh_knees=KNEES, seed=5, **extra)
    net = ka.KneeNet.from_member(member)
    assert net.output == kind and net.knees == KNEES
    assert (net.depth, net.scale, net.n_bands) == (1, 2, 4)
    if kind == "stretched":
        assert net.output_knee == 10.0
    if kind == "heads":
        assert net.head_index == KNEES.index(100.0)     # the served (default) head
    lr = _field(12, seed=6)
    want = member.upsample_array(lr)
    got = net.predict_electrons(tf.constant(ka.stretch(lr[None], KNEES))).numpy()[0]
    assert got.shape == want.shape == (24, 24, 4)
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-5 * np.abs(want).max())
    shares = ka.attribute(net, lr, _samples([(0, 5, 5), (3, 20, 11)]))
    np.testing.assert_allclose(shares["knee_share"].sum(axis=1), 1.0, rtol=1e-5)


def test_knee_net_validates_its_output_kind():
    model = _net().model
    with pytest.raises(ValueError, match="output"):
        ka.KneeNet(model=model, knees=KNEES, depth=1, output="pixels")
    with pytest.raises(ValueError, match="output_knee"):
        ka.KneeNet(model=model, knees=KNEES, depth=1, output="stretched")
    with pytest.raises(ValueError, match="head_index"):
        ka.KneeNet(model=model, knees=KNEES, depth=1, output="heads")
