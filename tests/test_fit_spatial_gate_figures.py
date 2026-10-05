"""Comparison figure panels of ``scripts/fit_spatial_gate.py compare --figures``
(stub combiners and a synthetic field; the PNG writer is captured)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from euclid_polish.eval.spatial_gate_compare import Method
from scripts import fit_spatial_gate as cli

_H = _W = 8
_M = 3


class _Combiner:
    use_lr = False

    def apply_field(self, members, lr=None):
        return members.mean(0)

    def weights_field(self, members, lr=None):
        return np.full((_H, _W, 4, len(members)), 1.0 / len(members), np.float32)


def test_compare_figure_labels_the_rbf_panel_by_name(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    members = rng.uniform(1.0, 50.0, (_M, _H, _W, 4)).astype(np.float32)
    field = SimpleNamespace(index=3, target_e=members.mean(0), lr_e=None,
                            members_e=lambda: members)
    result = SimpleNamespace(
        labels=[f"{170 + i}·psnr" for i in range(_M)], report={}, source_lr=None,
        figure_candidates=[("natural", field, True)],
        methods={"rbf": Method("rbf", _Combiner(), [0, 1], "rbf", "RBF"),
                 "gate:spatial_gate_trial": Method("gate:spatial_gate_trial", _Combiner(),
                                                   [0, 1, 2], "gate", "spatial_gate_trial")})
    drawn = []
    monkeypatch.setattr(cli, "_crop_figure",
                        lambda path, title, truth, panels, *a: drawn.append([n for n, _ in panels]))

    cli._figures(result, SimpleNamespace(figures=str(tmp_path), max_figures=1))

    assert drawn == [["best member 170·psnr", "ensemble mean", "RBF combiner", "spatial gate"]]
