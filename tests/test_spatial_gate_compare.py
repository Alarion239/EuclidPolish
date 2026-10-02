"""spatial_gate_compare: the importable compare/fit core of
scripts/fit_spatial_gate.py (synthetic fields, stub combiners — no TF)."""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from euclid_polish.eval import spatial_gate_compare as sgc
from euclid_polish.eval.knee_psnr import KNEE_GRID_E
from euclid_polish.eval.spatial_gate_fit import GateField


class _Gate:
    """A stub gate that always picks its FIRST member (convex weights)."""

    use_lr = False
    fit_meta = {"selected": {"loss": 0.5}}

    def __init__(self, labels):
        self.member_labels = list(labels)

    def needed_member_indices(self):
        return [0]

    def weights_field(self, preds, lr=None):
        w = np.zeros(preds.shape[1:3] + (preds.shape[0], preds.shape[3]), np.float32)
        w[:, :, 0, :] = 1.0
        return w

    def apply_field(self, preds, lr=None):
        return np.asarray(preds[0], np.float32)


def _fields(tmp_path, n=3, members=3, size=16):
    rng = np.random.default_rng(0)
    out = []
    for idx in range(n):
        truth = (rng.random((size, size, 4)) * 50.0 + 1.0).astype(np.float32)
        paths = []
        for m in range(members):
            # member m is off by m e⁻: member 0 is exact
            p = tmp_path / f"member{m}_{idx:05d}.npy"
            np.save(p, truth + float(m))
            paths.append(str(p))
        lr = truth.reshape(size // 2, 2, size // 2, 2, 4).mean(axis=(1, 3))
        out.append(GateField(idx, paths, truth, lr.astype(np.float32)))
    return out


def test_member_positions_and_active_member_resolution():
    cube = ["169·psnr", "170·psnr", "171·psnr"]
    assert sgc.member_positions(["171·psnr", "169·psnr"], cube) == [2, 0]
    assert sgc.member_positions(["172·psnr"], cube) is None
    assert sgc.resolve_active_members(["170", "member_171"], cube) == [1, 2]
    assert sgc.resolve_active_members([], cube) is None
    with pytest.raises(ValueError, match="199"):
        sgc.resolve_active_members(["199"], cube)


def test_compare_scores_every_method_member_and_knee_curve(tmp_path):
    fields = _fields(tmp_path)
    labels = ["00·psnr", "01·psnr", "02·psnr"]
    # a subset gate over members 02 and 00 (in that order): it picks 02
    gate = sgc.Method("gate:spatial_gate_sub", _Gate(["02·psnr", "00·psnr"]), [2, 0], "gate", "sub")
    exact = sgc.Method("gate:spatial_gate_exact", _Gate(labels), [0, 1, 2], "gate", "exact")
    report, candidates = sgc.compare_methods(
        fields, [], labels, {gate.name: gate, exact.name: exact},
        source_lr={f.index: f.lr_e for f in fields})

    natural = report["groups"]["natural"]
    assert set(natural) == {"mean", gate.name, exact.name, "member:00·psnr",
                            "member:01·psnr", "member:02·psnr"}
    # the exact gate reproduces member 00 (perfect → capped PSNR), the subset
    # gate reproduces member 02 (the worst)
    assert natural[exact.name]["band_psnr"] == pytest.approx(natural["member:00·psnr"]["band_psnr"])
    assert natural[gate.name]["band_psnr"] == pytest.approx(natural["member:02·psnr"]["band_psnr"])
    assert report["method_members"][gate.name] == ["02·psnr", "00·psnr"]
    assert report["usage"][gate.name]["labels"] == ["02·psnr", "00·psnr"]
    assert report["usage"][gate.name]["all_pixels"][0] == pytest.approx([1.0] * 4)
    assert report["n_fields"] == {"natural": 3, "blackout": 0}
    assert sgc.best_member(report) == "member:00·psnr"
    knee = report["knee"]
    assert knee["knees"] == list(KNEE_GRID_E) and knee["n_fields"] == 3
    assert set(knee["methods"]) == {"mean", gate.name, exact.name}
    assert len(knee["methods"]["mean"]["psnr"]) == len(KNEE_GRID_E)
    assert len(knee["methods"]["mean"]["integrated"]) == 4
    assert knee["methods"][exact.name]["integrated"][0] > knee["methods"]["mean"]["integrated"][0]
    assert len(candidates) == 3
    assert "best single member by VIS: member:00·psnr" in sgc.format_report(report)


def test_compare_blackout_group_scores_holes(tmp_path):
    fields = _fields(tmp_path, n=2)
    labels = ["00·psnr", "01·psnr", "02·psnr"]
    stamped = []
    for f in fields:
        lr = f.lr_e.copy()
        lr[:2, :2] = 0.0
        stamped.append(GateField(f.index, f.member_paths, f.target_e, lr, tag="blackout"))
    report, _ = sgc.compare_methods(fields, stamped, labels, {},
                                    source_lr={f.index: f.lr_e for f in fields}, knee=False)
    assert report["n_fields"]["blackout"] == 2
    assert "knee" not in report
    hole = report["groups"]["blackout"]["member:01·psnr"]["hole_mse"]
    assert all(v > 0 for v in hole)


def test_compare_without_blackout_fields_is_strict_json(tmp_path):
    """A compare run with 0 blackout fields scores an EMPTY blackout group: its
    PSNRs are null, never NaN (a bare NaN token made /ensemble/combiners.json
    unparseable in the browser)."""
    fields = _fields(tmp_path, n=2)
    labels = ["00·psnr", "01·psnr", "02·psnr"]
    report, _ = sgc.compare_methods(fields, [], labels, {},
                                    source_lr={f.index: f.lr_e for f in fields}, knee=False)
    json.dumps(report, allow_nan=False)
    assert report["groups"]["blackout"]["mean"]["band_psnr"] == [None] * 4
    assert sgc.best_member(report, "blackout") is None
    assert sgc.best_member(report) == "member:00·psnr"
    text = sgc.format_report(report)
    assert "natural test fields" in text and "blackout test fields" not in text


def test_parse_loss_knees():
    assert sgc.parse_loss_knees(None) == sgc.ALL_KNEE_LOSS
    assert sgc.parse_loss_knees("all") == sgc.ALL_KNEE_LOSS
    assert sgc.parse_loss_knees("band") is None
    assert sgc.parse_loss_knees("100, 1,10") == (1.0, 10.0, 100.0)
    for bad in ("x", "0,1", "-3"):
        with pytest.raises(ValueError):
            sgc.parse_loss_knees(bad)


def test_fit_gate_variant_refuses_the_production_directory(tmp_path):
    with pytest.raises(ValueError, match="production"):
        sgc.fit_gate_variant([], [], out_dir=os.path.join(str(tmp_path), sgc.PRODUCTION_DIR))


def test_fit_gate_variant_saves_progressively_into_the_named_dir(tmp_path, monkeypatch):
    fields = _fields(tmp_path, n=4)
    labels = ["00·psnr", "01·psnr", "02·psnr"]
    saved = []

    def fake_fit(train, held, lbls, **kw):
        comb = type("C", (), {})()
        comb.fit_meta = {"complete": False}
        kw["checkpoint"](comb)
        comb.fit_meta["complete"] = True
        assert kw["active_members"] == [1]
        assert kw["mix_space"] == "asinh" and kw["loss_knees"] is None
        assert len(train) == 3 and len(held) == 1
        return comb

    monkeypatch.setattr(sgc, "fit_spatial_gate", fake_fit)
    monkeypatch.setattr(sgc, "save_spatial_gate", lambda comb, d: saved.append((d, dict(comb.fit_meta))))
    out = str(tmp_path / "spatial_gate_trial")
    comb = sgc.fit_gate_variant(fields, labels, out_dir=out, holdout=1, members=["01"],
                                loss_knees=None, mix_space="asinh", blackout_fields=0,
                                records_fp="fp", extra_meta={"variant": "trial"})
    assert [d for d, _ in saved] == [out, out]
    assert saved[-1][1]["variant"] == "trial" and saved[-1][1]["complete"] is True
    assert comb.records_fp == "fp" and comb.starfull is True
