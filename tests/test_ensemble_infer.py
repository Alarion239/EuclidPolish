from __future__ import annotations

import numpy as np

from euclid_polish.eval.ensemble_infer import sr_from_model


class _FakeEnsemble:
    """Duck-typed ensemble: member_arrays returns a fixed (M,H,W,C) stack."""
    def __init__(self, stack):
        self._stack = np.asarray(stack, dtype=np.float32)

    @property
    def n_members(self):
        return int(self._stack.shape[0])

    def member_arrays(self, lr_array):
        return self._stack


def test_sr_from_model_ensemble_returns_mean_and_members():
    m = np.stack([np.full((6, 6, 4), 2.0, np.float32),
                  np.full((6, 6, 4), 4.0, np.float32)], axis=0)  # (2,6,6,4)
    lr = np.zeros((3, 3, 4), np.float32)
    lr_vis, sr, members = sr_from_model(_FakeEnsemble(m), lr)
    assert members is not None and members.shape == (2, 6, 6, 4)
    assert np.allclose(sr, 3.0)                       # mean of 2 and 4
    assert lr_vis.shape == (3, 3)                     # VIS plane of the LR cube


def test_sr_from_model_singleton_hides_members():
    m = np.full((1, 6, 6, 4), 5.0, np.float32)        # ensemble of 1
    lr_vis, sr, members = sr_from_model(_FakeEnsemble(m), np.zeros((3, 3, 4)))
    assert members is None                            # no all-zero std cubes
    assert np.allclose(sr, 5.0)
    assert lr_vis.shape == (3, 3)
