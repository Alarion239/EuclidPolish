"""Blackout cube buckets (``spatial_gate_fit.build_blackout_fields``) and the
lazy member runner: cubes keyed by (member label, checkpoint fingerprint),
filled incrementally — a second build runs nothing, a continued or added
member runs alone, a departed one is dropped, and only a change of the
stamping identity (thresholds, seed, source records) re-runs everyone."""
from __future__ import annotations

import json

import numpy as np
import pytest

from euclid_polish.eval import spatial_gate_fit as sgf
from euclid_polish.eval.ensemble_cube_cache import member_cube_path
from euclid_polish.eval.spatial_gate_fit import GateField, LazyMemberRunner
from tests._ensemble_cube_cache_fixtures import FakeEnsemble, make_env, set_checkpoint


class _Runner:
    """Members ``label → value``: a member's cube is its value plus the LR sum."""

    def __init__(self, fingerprints):
        self.fingerprints = dict(fingerprints)
        self.calls: list[list[str]] = []

    def __call__(self, lr, labels):
        self.calls.append(list(labels))
        h, w, c = lr.shape
        return np.stack([np.full((2 * h, 2 * w, c), self.value(lb) + float(lr.sum()),
                                 np.float32) for lb in labels])

    def value(self, label):
        return float(len(self.fingerprints[label] or "")) + ord(label[0])

    def ran(self):
        return sorted(lb for call in self.calls for lb in call)


def _stamp(lr_e, rng, **_kw):
    out = np.array(lr_e, np.float32, copy=True)
    out[:2, :2] = 0.0
    return out


def _fields(n=4):
    out = []
    for index in range(n):
        lr = np.full((4, 4, 4), 1.0 + index, np.float32)
        out.append(GateField(index, [], np.zeros((8, 8, 4), np.float32), lr))
    return out


@pytest.fixture
def stamped(monkeypatch):
    monkeypatch.setattr(sgf, "stamp_blackouts", _stamp)


def _build(out_dir, labels, runner, *, seed=0, source="records-a", max_fields=3):
    return sgf.build_blackout_fields(_fields(), labels, runner, str(out_dir),
                                     max_fields=max_fields, seed=seed,
                                     source_fingerprint=source)


def _index(out_dir):
    return json.loads((out_dir / "blackout_index.json").read_text())


def test_blackout_cubes_fill_by_label_and_fingerprint(tmp_path, stamped):
    out_dir = tmp_path / "cubes_blackout"
    runner = _Runner({"a": "fa", "b": "fb", "c": "fc"})
    fields = _build(out_dir, ["a", "b", "c"], runner)
    assert len(fields) == 3 and runner.ran() == sorted(["a", "b", "c"] * 3)
    index = _index(out_dir)
    assert index["member_labels"] == ["a", "b", "c"]
    assert index["member_fps"] == {"a": "fa", "b": "fb", "c": "fc"}
    assert "member_labels" not in index["identity"]
    assert fields[0].member_paths == [member_cube_path(str(out_dir), lb, 0)
                                      for lb in ("a", "b", "c")]

    second = _Runner({"a": "fa", "b": "fb", "c": "fc"})
    again = _build(out_dir, ["a", "b", "c"], second)
    assert second.calls == []                                     # full cache hit
    assert [f.index for f in again] == [f.index for f in fields]
    np.testing.assert_array_equal(again[1].lr_e, fields[1].lr_e)


def test_a_continued_member_reruns_alone_and_a_departed_one_is_dropped(tmp_path, stamped):
    out_dir = tmp_path / "cubes_blackout"
    _build(out_dir, ["a", "b", "c"], _Runner({"a": "fa", "b": "fb", "c": "fc"}))
    continued = _Runner({"a": "fa", "b": "fb-continued", "c": "fc"})
    fields = _build(out_dir, ["a", "b", "c"], continued)
    assert continued.ran() == ["b", "b", "b"]
    cube = np.load(member_cube_path(str(out_dir), "b", 0))
    assert float(cube.flat[0]) == continued.value("b") + float(fields[0].lr_e.sum())

    joined = _Runner({"a": "fa", "b": "fb-continued", "c": "fc", "d": "fd"})
    _build(out_dir, ["a", "b", "c", "d"], joined)
    assert joined.ran() == ["d", "d", "d"]

    left = _Runner({"b": "fb-continued", "c": "fc", "d": "fd"})
    fields = _build(out_dir, ["b", "c", "d"], left)
    assert left.calls == []
    assert not list(out_dir.glob("member_a_*.npy"))
    assert _index(out_dir)["member_labels"] == ["b", "c", "d"]
    assert all(len(f.member_paths) == 3 for f in fields)


def test_a_stamping_identity_change_reruns_everyone(tmp_path, stamped):
    out_dir = tmp_path / "cubes_blackout"
    _build(out_dir, ["a", "b"], _Runner({"a": "fa", "b": "fb"}))
    for change in ({"seed": 1}, {"source": "records-b"}):
        runner = _Runner({"a": "fa", "b": "fb"})
        _build(out_dir, ["a", "b"], runner, **change)
        assert runner.ran() == sorted(["a", "b"] * 3), change


def test_a_positional_blackout_bucket_is_migrated_and_reinferred(tmp_path, stamped):
    """The old layout keyed files by position in ``identity.member_labels``
    and recorded no fingerprints: renamed, then re-inferred (unknown weights);
    files of positions beyond the labels are deleted."""
    out_dir = tmp_path / "cubes_blackout"
    out_dir.mkdir()
    identity = {"member_labels": ["a", "b"], "well_fractions": list(sgf.BLACKOUT_WELL_FRACTIONS),
                "seed": 0, "source": "records-a"}
    for index in range(3):
        np.save(out_dir / f"lr_{index:05d}.npy", _stamp(_fields()[index].lr_e, None))
        for pos in range(4):                                    # 2 orphan positions
            np.save(out_dir / f"member{pos}_{index:05d}.npy", np.zeros((8, 8, 4), np.float32))
    (out_dir / "blackout_index.json").write_text(
        json.dumps({"identity": identity, "indices": [0, 1, 2]}))
    runner = _Runner({"a": "fa", "b": "fb"})
    _build(out_dir, ["a", "b"], runner)
    assert runner.ran() == sorted(["a", "b"] * 3)
    assert not list(out_dir.glob("member[0-9]*_*.npy"))
    assert _index(out_dir)["member_fps"] == {"a": "fa", "b": "fb"}


def test_lazy_member_runner_loads_only_the_members_it_runs(tmp_path, monkeypatch):
    env = make_env(tmp_path, monkeypatch)
    runner = LazyMemberRunner(str(env["base"]), starless=False, labels=env["labels"])
    assert FakeEnsemble.loaded == []                       # nothing loaded up front
    lr = np.ones((4, 4, 4), np.float32)
    out = runner(lr, ["03·psnr", "01·psnr"])
    assert FakeEnsemble.loaded == [["03·psnr", "01·psnr"]]
    assert out.shape == (2, 8, 8, 4)
    assert float(out[0].flat[0]) > float(out[1].flat[0])        # 03 before 01, as asked
    runner(lr, ["01·psnr", "02·psnr"])
    assert FakeEnsemble.loaded == [["03·psnr", "01·psnr"], ["02·psnr"]]
    assert runner(lr).shape[0] == 3                              # default: every member
    assert len(FakeEnsemble.loaded) == 2
    set_checkpoint(env["base"], 2, step=9)
    assert runner.fingerprints["02·psnr"] != LazyMemberRunner(
        str(env["base"]), starless=False, labels=env["labels"]).fingerprints["02·psnr"]
    with pytest.raises(ValueError, match="99"):
        runner(lr, ["99·psnr"])
