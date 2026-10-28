"""The flux-vs-magnitude plot (scripts/plot_flux_conservation.py): reports
measured in different runs merge on ``object_id`` whatever their row order,
the isolated sample is the input's own Kron pull, and the magnitude bins
take medians of what falls in them."""
from __future__ import annotations

import numpy as np

from scripts import plot_flux_conservation as plot


def _report(path, ids, mag, lr, models):
    np.savez(path, object_id=np.asarray(ids), vis_mag=np.asarray(mag, float),
             **{"lr:kron_vis": np.asarray(lr, float)},
             **{f"{spec}:kron_vis": np.asarray(v, float) for spec, v in models.items()},
             **{f"{spec}:ap4_vis": np.zeros(len(ids)) for spec in models})


def test_reports_merge_on_object_id_in_the_first_reports_order(tmp_path):
    _report(tmp_path / "a.npz", [10, 20, 30], [21.0, 22.0, 23.0], [1.0, 2.0, 3.0],
            {"universal:member_249": [1.1, 2.2, 3.3]})
    _report(tmp_path / "b.npz", [30, 10], [23.0, 21.0], [3.0, 1.0],
            {"universal:member_258": [2.7, 0.9]})
    ids, mag, lr, models = plot.load([str(tmp_path / "a.npz"), str(tmp_path / "b.npz")])
    assert list(ids) == [10, 20, 30]
    assert list(lr) == [1.0, 2.0, 3.0]
    assert list(models) == ["universal:member_249", "universal:member_258"]
    np.testing.assert_array_equal(models["universal:member_258"], [0.9, np.nan, 2.7])


def test_isolated_keeps_galaxies_the_input_measures_like_the_catalogue():
    keep = plot.isolated(np.array([100.0, 100.0, 100.0, 100.0]),
                         np.array([100.0 + 0.4 * 10, 100.0 + 0.6 * 10, 100.0, 0.0]),
                         np.array([10.0, 10.0, 0.0, 10.0]))
    assert list(keep) == [True, False, False, False]


def test_binned_median_per_bin_and_nan_where_empty():
    centres, medians = plot.binned_median(np.array([1.0, 3.0, 2.0, np.nan, 5.0]),
                                          np.array([21.1, 21.2, 21.3, 21.7, 22.4]),
                                          np.array([21.0, 21.5, 22.0, 22.5]))
    np.testing.assert_allclose(centres, [21.25, 21.75, 22.25])
    np.testing.assert_array_equal(medians, [2.0, np.nan, 5.0])


def test_default_labels_name_the_member_and_milestone_step():
    assert plot.default_label("universal:member_258") == "258"
    assert plot.default_label("milestone:member_257@4000000") == "257 @ 4M"
    assert plot.default_label("production") == "production"
