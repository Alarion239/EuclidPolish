"""The "used by the gate" rule (eval/gate_members.py) and ``fit --members
used`` in scripts/fit_spatial_gate.py. tmp_path fixtures only."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from euclid_polish.eval import gate_members as gm
from euclid_polish.eval import spatial_gate_compare as sgc
from euclid_polish.eval.combiner import COMBINER_MODELS, combiner_artifact_fingerprint
from euclid_polish.eval.spatial_gate import SPATIAL_GATE_KIND
from scripts import fit_spatial_gate as cli

LABELS = ["170·psnr", "171·psnr", "172·psnr", "173·psnr"]
BANDS = ("VIS", "Y_E", "J_E", "H_E")


def _diagnostic(usage, source, bins):
    """Per-band arrays: ``usage`` / ``source`` (M,), ``bins`` (B, M)."""
    return {"available": True, "schema": 1,
            "usage": {b: list(usage) for b in BANDS},
            "usage_source": {b: list(source) for b in BANDS},
            "usage_by_brightness": {b: [list(row) for row in bins] for b in BANDS}}


def test_peak_is_the_max_over_all_pixels_sources_and_every_bin():
    """A core specialist with ~0 % over all pixels but half the weight in the
    core bin is kept; a member whose peak stays below 0.5 % everywhere is
    dropped (the rounded all-pixel mean would call both 0.0 %)."""
    usage = [0.97, 0.0004, 0.0296, 0.0001]
    source = [0.90, 0.0010, 0.0980, 0.0010]
    bins = [[0.99, 0.0002, 0.0098, 0.0001],        # sky
            [0.50, 0.4800, 0.0100, 0.0100]]        # core: 171 carries 48 %
    diag = _diagnostic(usage, source, bins)
    diag["usage"]["H_E"] = [0.9, 0.0, 0.0, 0.004]  # one band never above 0.4 %
    peaks = gm.member_peak_weights(diag, 4)
    assert peaks == pytest.approx([0.99, 0.48, 0.098, 0.01])
    choice = gm.used_members(diag, LABELS)
    assert choice.kept_labels == LABELS             # 173 peaks at 1 % in the core bin
    tight = gm.used_members(diag, LABELS, threshold=0.05)
    assert tight.kept_labels == ["170·psnr", "171·psnr", "172·psnr"]
    assert tight.dropped == (("173·psnr", pytest.approx(0.01)),)
    text = gm.format_choice(tight)
    assert "kept:    170 (99.00%), 171 (48.00%), 172 (9.80%)" in text
    assert "dropped: 173 (1.00%)" in text


def test_threshold_is_inclusive_and_pruned_members_are_dropped():
    usage = [0.995, 0.005, 0.0, 0.0]               # 172/173 pruned: zero weight
    diag = _diagnostic(usage, usage, [usage])
    choice = gm.used_members(diag, LABELS)
    assert choice.kept_labels == ["170·psnr", "171·psnr"]
    assert choice.dropped_labels == ["172·psnr", "173·psnr"]


def test_rule_refuses_unusable_diagnostics():
    with pytest.raises(ValueError, match="unavailable: no validation cube cache"):
        gm.used_members({"available": False, "reason": "no validation cube cache"}, LABELS)
    with pytest.raises(ValueError, match="expected 4"):
        gm.member_peak_weights(_diagnostic([1.0, 0.0], [1.0, 0.0], [[1.0, 0.0]]), 4)
    with pytest.raises(ValueError, match="no member reaches"):
        gm.used_members(_diagnostic([0.001] * 4, [0.001] * 4, [[0.001] * 4]), LABELS)


@pytest.mark.parametrize("raw, threshold", [
    ("used", 0.005), ("USED", 0.005), (" used:0.5% ", 0.005), ("used:2%", 0.02),
    ("used:0.01", 0.01), ("170,171", None), ("", None),
])
def test_parse_used_threshold(raw, threshold):
    got = gm.parse_used_threshold(raw)
    assert got == (pytest.approx(threshold) if threshold is not None else None)


@pytest.mark.parametrize("raw", ["used:", "used:abc", "used:0", "used:100%", "used:-1%"])
def test_parse_used_threshold_rejects_bad_values(raw):
    with pytest.raises(ValueError):
        gm.parse_used_threshold(raw)


def _production(regime, diagnostic, *, fresh=True):
    """A production gate artifact + its cached payload under ``regime``."""
    artifact = COMBINER_MODELS[SPATIAL_GATE_KIND].artifact_dir
    d = os.path.join(regime, artifact)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "combiner.json"), "w") as f:
        json.dump({"kind": "spatial_gate", "member_labels": LABELS}, f)
    np.savez(os.path.join(d, "combiner.npz"), w=np.zeros(2))
    fp = combiner_artifact_fingerprint(regime, artifact) if fresh else "old"
    with open(os.path.join(regime, f"{artifact}_evals.json"), "w") as f:
        json.dump({"member_labels": LABELS,
                   "gate_diagnostics": {**diagnostic, "artifact_fp": fp}}, f)


def test_cli_resolves_used_members_from_the_production_diagnostic(tmp_path, capsys):
    regime = str(tmp_path / "starfull")
    usage = [0.9, 0.0001, 0.0999, 0.0]
    _production(regime, _diagnostic(usage, usage, [[0.5, 0.49, 0.01, 0.0]]))
    assert cli.resolve_members("used", regime) == ["170", "171", "172"]
    out = capsys.readouterr().out
    assert "kept:    170 (90.00%), 171 (49.00%), 172 (9.99%)" in out
    assert "dropped: 173 (0.00%)" in out
    assert cli.resolve_members("used:10%", regime) == ["170", "171"]
    assert cli.resolve_members("170, 173", regime) == ["170", "173"]
    assert cli.resolve_members("", regime) == []


def test_cli_used_keeps_members_that_joined_after_the_production_fit(tmp_path, capsys):
    """The refit a "joined after this fit" note asks for must consider the
    new members: they have no weight evidence yet, so they are kept, and
    listed (the diagnostic only knows the production gate's members)."""
    regime = str(tmp_path / "starfull")
    usage = [0.9, 0.0001, 0.0999, 0.0]
    _production(regime, _diagnostic(usage, usage, [[0.5, 0.49, 0.01, 0.0]]))
    cubes = [*LABELS, "199·psnr", "200·psnr"]
    assert cli.resolve_members("used", regime, cubes) == ["170", "171", "172", "199", "200"]
    assert "new:     199, 200" in capsys.readouterr().out


def test_cli_refuses_a_diagnostic_of_another_artifact(tmp_path):
    regime = str(tmp_path / "starfull")
    usage = [1.0, 0.0, 0.0, 0.0]
    _production(regime, _diagnostic(usage, usage, [usage]), fresh=False)
    with pytest.raises(ValueError, match="not for the production gate"):
        cli.resolve_members("used", regime)
    with pytest.raises(ValueError, match="no cached production gate payload"):
        cli.resolve_members("used", str(tmp_path / "empty"))


def test_fit_marker_is_written_while_fitting_and_removed_after(tmp_path, monkeypatch):
    """fit_gate_variant marks its directory while the fit runs (the promote
    guard refuses it) and removes the marker however the fit ends."""
    out = str(tmp_path / "spatial_gate_trial")
    seen = {}

    def fake_fit(*_a, checkpoint, **_k):
        seen["marker"] = sgc.fit_in_progress(out)
        raise KeyboardInterrupt

    monkeypatch.setattr(sgc, "fit_spatial_gate", fake_fit)
    monkeypatch.setattr(sgc, "split_holdout", lambda fields, n, seed: (fields[:1], fields[1:]))
    with pytest.raises(KeyboardInterrupt):
        sgc.fit_gate_variant([object(), object()], ["1·psnr"], out_dir=out, blackout_fields=0)
    assert seen["marker"] and seen["marker"]["pid"] == os.getpid()
    assert not os.path.exists(os.path.join(out, sgc.FIT_MARKER))
    assert sgc.fit_in_progress(out) is None


def test_promotion_refusal_reasons(tmp_path):
    d = str(tmp_path / "spatial_gate_x")
    os.makedirs(d)
    assert sgc.promotion_refusal(d, {"fit_meta": {"complete": True}}) is None
    assert "step 3 of 9" in sgc.promotion_refusal(
        d, {"fit_meta": {"complete": False, "steps_run": 3, "steps": 9}})
    assert "no completion flag" in sgc.promotion_refusal(d, {"fit_meta": {}})
    with open(os.path.join(d, sgc.FIT_MARKER), "w") as f:
        json.dump({"pid": os.getpid(), "host": sgc.socket.gethostname(), "started": "t"}, f)
    assert "still being fitted" in sgc.promotion_refusal(d, {"fit_meta": {"complete": True}})
    with open(os.path.join(d, sgc.FIT_MARKER), "w") as f:
        json.dump({"pid": os.getpid(), "host": "another-host", "started": "t"}, f)
    assert "another-host" in sgc.promotion_refusal(d, {"fit_meta": {"complete": True}})


def test_a_dead_fit_marker_never_blocks_promotion_whatever_its_host(tmp_path):
    """Liveness is the pid alone: a laptop's DHCP host name changes with the
    network, so a crashed fit's marker must not block its variant forever."""
    d = str(tmp_path / "spatial_gate_x")
    os.makedirs(d)
    for marker in ({"pid": 2 ** 22 + 12345, "host": "another-host", "started": "t"},
                   {"host": "x"}, {"pid": -1}, {"pid": "nope"}):
        with open(os.path.join(d, sgc.FIT_MARKER), "w") as f:
            json.dump(marker, f)
        assert sgc.fit_in_progress(d) is None, marker
        assert sgc.promotion_refusal(d, {"fit_meta": {"complete": True}}) is None
    with open(os.path.join(d, sgc.FIT_MARKER), "w") as f:
        json.dump({"pid": os.getpid(), "host": "another-host", "started": "t"}, f)
    assert sgc.FIT_MARKER in sgc.promotion_refusal(d, {"fit_meta": {"complete": True}})
