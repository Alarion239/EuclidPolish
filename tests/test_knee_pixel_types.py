"""Pixel types by knee use: effective knee, brightness-controlled groups,
knee types (k-means of the share geometry), stamps, alignment and measures."""

from __future__ import annotations

import numpy as np
import pytest

from euclid_polish.eval import knee_pixel_types as kt

KNEES = (0.01, 1.0, 100.0, 1e4)


def test_effective_knee_and_knee_count():
    share = np.array([[0, 1, 0, 0], [0.5, 0, 0.5, 0], [0, 0, 0, 0], [0.25] * 4])
    log_q = kt.effective_knee(share, KNEES)
    np.testing.assert_allclose(log_q[[0, 1, 3]], [0.0, 0.0, 1.0])
    assert np.isnan(log_q[2])
    count = kt.knee_count(share)
    np.testing.assert_allclose(count[[0, 1, 3]], [1.0, 2.0, 4.0])
    assert np.isnan(count[2])
    # Rows that do not sum to 1 are normalised first.
    np.testing.assert_allclose(kt.effective_knee(2 * share[:2], KNEES), log_q[:2])


def test_mechanical_share_spreads_over_the_knees_below_the_block():
    # A block at 0: every term is 1/q, so the lowest knee dominates.
    zero = kt.mechanical_share(np.zeros((1, 9)), KNEES)
    np.testing.assert_allclose(zero[0], np.array([100.0, 1.0, 0.01, 1e-4]) / 101.0101, rtol=1e-6)
    # A bright block (1e3 e⁻): the knees far below share 1/x alike.
    bright = kt.mechanical_share(np.full((1, 9), 1e3), KNEES)[0]
    assert bright[0] == pytest.approx(bright[1], rel=1e-4)
    assert bright[3] < 0.1 * bright[0]
    # NaN pixels (beyond the field) are left out, rows sum to 1.
    partial = kt.mechanical_share(np.array([[0.0] * 4 + [np.nan] * 5]), KNEES)
    np.testing.assert_allclose(partial, zero)
    # The effective knee of the baseline rises with the block's brightness.
    levels = np.array([0.1, 1.0, 10.0, 100.0, 1e4])[:, None] * np.ones((1, 9))
    log_q = kt.effective_knee(kt.mechanical_share(levels, KNEES), KNEES)
    assert np.all(np.diff(log_q) > 0)


def test_weighted_quantile_follows_the_weights():
    values = np.array([1.0, 2.0, 3.0])
    assert kt.weighted_quantile(values, np.ones(3), [0.5])[0] == pytest.approx(2.0)
    assert kt.weighted_quantile(values, [1, 1, 100], [0.5])[0] == pytest.approx(3.0, abs=0.02)
    assert np.isnan(kt.weighted_quantile(values, np.zeros(3), [0.5])[0])


def test_brightness_bins_clip_at_the_ends_and_drop_non_finite():
    bins = kt.brightness_bin(np.array([-5.0, 0.0, 5e-3, 0.5, 2e7, np.nan]))
    assert bins.tolist() == [0, 0, 0, 2, 7, -1]


def test_residual_groups_split_each_brightness_bin_by_its_own_knee():
    rng = np.random.default_rng(0)
    # Two bins with different typical knees: the groups follow each bin's
    # own quantiles, not a global cut.
    value = np.concatenate([np.full(100, 0.5), np.full(100, 500.0)])
    log_q = np.concatenate([rng.normal(-1, 0.3, 100), rng.normal(2, 0.3, 100)])
    bins, groups, stats = kt.residual_groups(log_q, value, np.ones(200))
    for b in (2, 5):                                     # 0.1–1 σ and 100–1000 σ
        sel = bins == b
        frac = np.bincount(groups[sel], minlength=3) / sel.sum()
        np.testing.assert_allclose(frac, [0.2, 0.6, 0.2], atol=0.02)
        assert stats[b]["n"] == 100
    assert stats[2]["median"] == pytest.approx(-1, abs=0.15)
    assert stats[5]["median"] == pytest.approx(2, abs=0.15)
    assert stats[0]["n"] == 0 and stats[0]["median"] is None
    # A bin read through one knee only has no lower or higher pixels.
    _, flat, _ = kt.residual_groups(np.zeros(10), np.full(10, 3.0), np.ones(10))
    assert flat.tolist() == [1] * 10


def test_clr_is_centred_and_order_preserving():
    out = kt.clr(np.array([[0.7, 0.2, 0.1, 0.0]]))
    assert out.sum() == pytest.approx(0.0, abs=1e-12)
    assert out[0, 0] > out[0, 1] > out[0, 2] > out[0, 3]


def test_knee_types_find_planted_profiles_ordered_by_knee():
    rng = np.random.default_rng(1)
    profiles = np.array([[0.9, 0.1, 0, 0], [0, 0.1, 0.9, 0], [0, 0, 0.1, 0.9]])
    truth = rng.integers(0, 3, 300)
    share = np.abs(profiles[truth] + rng.normal(0, 0.02, (300, 4)))
    share /= share.sum(axis=1, keepdims=True)
    share[0] = 0.0                                       # no gradient: no type
    labels = kt.knee_types(share, np.ones(300), KNEES, 3, np.random.default_rng(2))
    assert labels[0] == -1
    # Types are numbered by effective knee, so they equal the planted order.
    np.testing.assert_array_equal(labels[1:], truth[1:])


def test_cut_stamps_centre_on_the_pixel_and_pad_with_nan():
    plane = np.arange(100, dtype=np.float32).reshape(10, 10)
    stamps = kt.cut_stamps(plane, [5, 0], [5, 9], 3)
    np.testing.assert_array_equal(stamps[0], plane[4:7, 4:7])
    assert stamps[1, 1, 1] == plane[0, 9]
    assert np.isnan(stamps[1, 0]).all() and np.isnan(stamps[1, :, 2]).all()
    with pytest.raises(ValueError, match="odd"):
        kt.cut_stamps(plane, [5], [5], 4)


def test_align_puts_the_brighter_side_up_then_right():
    base = np.zeros((7, 7), np.float32)
    base[3, 3] = 1.0
    base[3, 6] = 5.0                                     # a neighbour to the right
    moves = [base, base[::-1], base[:, ::-1], base.T, base.T[::-1], np.rot90(base), np.rot90(base, 2)]
    aligned = kt.align(np.stack(moves))
    for a in aligned:
        np.testing.assert_array_equal(a, aligned[0])
    assert aligned[0][6, 3] == 5.0                       # +row (up with origin="lower")
    # Other stamps of the same pixels follow the same moves.
    moves = kt.alignment(np.stack(moves))
    other = kt.apply_alignment(np.stack([base, base[::-1]]) * 2, moves[:2])
    np.testing.assert_array_equal(other[0], other[1])


def test_stack_is_a_nan_aware_median():
    stamps = np.array([[[1.0, np.nan]], [[3.0, np.nan]], [[2.0, np.nan]]])
    out = kt.stack(stamps)
    assert out[0, 0] == 2.0 and np.isnan(out[0, 1])
    assert kt.stack(np.zeros((0, 2, 2))).shape == (2, 2)


def test_stamp_features_tell_a_peak_from_a_flank():
    yy, xx = np.mgrid[:21, :21]
    blob = 100.0 * np.exp(-((yy - 10) ** 2 + (xx - 10) ** 2) / 8.0)
    flank = 100.0 * np.exp(-((yy - 10) ** 2 + (xx - 12) ** 2) / 8.0)   # peak 2 px to the right
    f = kt.stamp_features(np.stack([blob, flank]))
    assert f["local_max"].tolist() == [1.0, 0.0]
    assert f["peak_ratio"][0] == pytest.approx(1.0)
    assert f["peak_ratio"][1] == pytest.approx(np.exp(0.5))
    assert f["slope"][0] == pytest.approx(0.0, abs=1e-9) and f["slope"][1] > 10.0
    assert f["contrast"][0] == pytest.approx(100.0, rel=1e-3)
    assert f["nearest_source"].tolist() == [0.0, 0.0]
    assert f["stamp_peak"][1] == pytest.approx(100.0)
    far = np.zeros((21, 21))
    far[10, 17] = 5.0                                    # one source 7 px away
    lone = kt.stamp_features(np.stack([far, np.zeros((21, 21))]))["nearest_source"]
    assert lone[0] == pytest.approx(7.0)
    assert lone[1] > np.hypot(10, 10)                    # none: beyond the stamp


def test_occupancy_is_the_fraction_of_stamps_above_the_threshold():
    stamps = np.zeros((4, 3, 3))
    stamps[:3, 0, 0] = 2.0
    stamps[0, 1, 1] = np.nan
    occ = kt.occupancy(stamps)
    assert occ[0, 0] == pytest.approx(0.75) and occ[2, 2] == 0.0
    assert occ.dtype == np.float32


def test_balanced_weights_give_every_bin_the_same_total():
    out = kt.balanced_weights(np.array([0, 0, 0, 3, -1]), np.array([1.0, 1.0, 2.0, 9.0, 5.0]))
    np.testing.assert_allclose(out, [0.25, 0.25, 0.5, 1.0, 0.0])


def test_neighbourhood_max_grows_with_the_radius():
    stamp = np.zeros((1, 9, 9))
    stamp[0, 4, 4] = 1.0
    stamp[0, 4, 7] = 5.0                                 # 3 px from the centre
    out = kt.neighbourhood_max(stamp, (0, 2, 3, 4))
    assert out[0].tolist() == [1.0, 1.0, 5.0, 5.0]


def test_look_around_finds_the_radius_the_knee_follows():
    rng = np.random.default_rng(3)
    n = 400
    near = 10 ** rng.uniform(0, 3, n)                    # light within 4 px
    maxima = np.stack([np.ones(n), np.ones(n), near, near], axis=1)
    log_q = np.log10(near) + rng.normal(0, 0.1, n)       # the knee follows it
    out = kt.look_around(log_q, np.zeros(n, int), maxima)
    assert np.isnan(out[0][0]) and out[0][2] > 0.95
    assert kt.look_around(log_q[:10], np.zeros(10, int), maxima[:10]) == {}


def test_weighted_mix_counts_weight_not_samples():
    mix = kt.weighted_mix(np.array(["sky", "sky", "core"]), np.array([1.0, 1.0, 8.0]),
                          ["sky", "core", "wing"])
    assert mix == {"core": 0.8, "sky": 0.2}
