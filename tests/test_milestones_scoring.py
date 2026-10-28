"""Measuring a long run's milestones (helpers/milestones.py): specs, the
pulled steps, a milestone materialized as an ensemble dir of its own (its
checkpoint, the member's metadata, the training log cut at its step), the
milestone score store, the JWST runner and the processor's notion of done."""
from __future__ import annotations

import csv
import json

import numpy as np
import pytest
import tensorflow as tf

from euclid_polish.web.helpers import milestones
from euclid_polish.web.helpers import universal_models as um


@pytest.fixture()
def root(tmp_path, monkeypatch):
    monkeypatch.setattr(milestones, "local_root", lambda: tmp_path / "milestones")
    monkeypatch.setattr(milestones, "scores_dir", lambda: tmp_path / "scores" / "milestones")
    return tmp_path / "milestones"


def _milestone(root, member, step, value):
    folder = root / member / "milestones" / milestones.step_dirname(step)
    ckpt = tf.train.Checkpoint(v=tf.Variable(float(value)))
    tf.train.CheckpointManager(ckpt, directory=str(folder), max_to_keep=1).save(checkpoint_number=step)


def test_specs_round_trip_and_reject_anything_else():
    assert milestones.spec_for("member_257", 200000) == "milestone:member_257@200000"
    assert milestones.parse_spec("milestone:member_257@200000") == ("member_257", 200000)
    for bad in ("universal:member_257", "milestone:member_x@1", "milestone:member_257", "milestone:../x@1"):
        with pytest.raises(ValueError):
            milestones.parse_spec(bad)


def test_a_milestone_materializes_as_an_ensemble_dir_of_its_own(root):
    member = "member_257"
    for step, value in ((200000, 2.0), (500000, 5.0)):
        _milestone(root, member, step, value)
    (root / member / "milestones" / "step_000000999").mkdir()      # an incomplete pull: no checkpoint
    assert milestones.local_steps(member) == [200000, 500000]
    (root / member / "origin.json").write_text(json.dumps({"asinh_knees": [0.1, 1.0]}))
    with open(root / member / "training_log.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["step", "duration_s"])
        writer.writeheader()
        writer.writerows([{"step": s, "duration_s": 80} for s in (100000, 200000, 300000, 500000)])
    ensemble = milestones.materialize(member, 200000)
    d = f"{ensemble}/{member}"
    assert tf.train.latest_checkpoint(d).endswith("ckpt-200000")
    reader = tf.train.load_checkpoint(tf.train.latest_checkpoint(d))
    assert reader.get_tensor("v/.ATTRIBUTES/VARIABLE_VALUE") == pytest.approx(2.0)
    assert json.loads(open(f"{d}/origin.json").read()) == {"asinh_knees": [0.1, 1.0]}
    assert [r["step"] for r in csv.DictReader(open(f"{d}/training_log.csv"))] == ["100000", "200000"]
    assert milestones.materialize(member, 200000) == ensemble                       # idempotent
    with pytest.raises(FileNotFoundError, match="pull it first"):
        milestones.materialize(member, 999)


def test_a_score_goes_to_the_milestone_store_not_the_universal_one(root, monkeypatch, tmp_path):
    member = "member_257"
    _milestone(root, member, 200000, 2.0)
    identity = {"indices": [1, 2], "records_fp": "fp", "target_psf_fwhm_arcsec": 0.066}
    monkeypatch.setattr(milestones.ev, "_sky_records_local_dir", lambda: str(tmp_path))
    monkeypatch.setattr(milestones.ev, "universal_score_identity", lambda: identity)
    seen = {}

    def fake_score(cap, name, d, ident, records_dir, *, done, total, ensemble_dir=None):
        seen.update(name=name, d=d, ensemble_dir=ensemble_dir)
        return ({"mean": 60.1, "integrated": [55.0, 64.0, 61.0, 60.4]}, np.zeros((2, 21, 4)), [1, 2],
                {"available": True})

    monkeypatch.setattr(um, "_score_model", fake_score)
    entry = milestones.score(None, member, 200000)
    assert entry["step"] == 200000 and entry["mean"] == 60.1
    assert seen["name"] == member and seen["d"].startswith(seen["ensemble_dir"])
    store = tmp_path / "scores" / "milestones"
    assert milestones.read_scores(member)["steps"]["200000"]["mean"] == 60.1
    assert (store / f"{member}@200000.npz").is_file() and (store / f"{member}@200000.spectra.json").is_file()
    # another identity starts the store afresh
    monkeypatch.setattr(milestones.ev, "universal_score_identity", lambda: {**identity, "records_fp": "new"})
    _milestone(root, member, 500000, 5.0)
    milestones.score(None, member, 500000)
    assert list(milestones.read_scores(member)["steps"]) == ["500000"]


def test_the_jwst_runner_runs_a_milestone(monkeypatch):
    from scripts import jwst_resolution

    class Fake:
        def member_arrays(self, lr):
            return [np.stack([lr[..., c].repeat(2, 0).repeat(2, 1) + c for c in range(4)], -1)]

    monkeypatch.setattr(jwst_resolution.milestones, "load", lambda member, step: Fake())
    run = jwst_resolution._runner("milestone:member_257@200000", "J_E")
    lr = np.ones((4, 4, 4), np.float32)
    assert run(lr).shape == (8, 8) and float(run(lr)[0, 0]) == 3.0


def test_the_processor_counts_a_milestone_done_once_scored_and_measured_in_every_band(root, tmp_path, monkeypatch):
    from scripts import process_milestones as pm

    member = "member_257"
    res = tmp_path / "res"
    res.mkdir()
    monkeypatch.setattr(milestones, "read_scores", lambda m: {"steps": {"200000": {}, "500000": {}}})
    for band in pm.BANDS:
        models = {"lr": {}, "milestone:member_257@200000": {}, "milestone:member_999@500000": {}}
        if band != "VIS":
            models["milestone:member_257@500000"] = {}
        (res / f"resolution_{band}.json").write_text(json.dumps({"models": models}))
    assert pm.done_steps(member, str(res)) == {200000}


def test_a_milestone_is_a_catalogue_spec_of_its_own(root):
    from euclid_polish.web.helpers import model_catalog as mc

    assert mc.canonical_spec("milestone:257@2000000") == "milestone:member_257@2000000"
    assert mc.canonical_spec(" milestone:member_257@0200000 ") == "milestone:member_257@200000"
    for bad in ("milestone:member_257", "milestone:x@1", "milestone:257@"):
        with pytest.raises(ValueError, match="milestone"):
            mc.canonical_spec(bad)
    spec = "milestone:member_257@200000"
    assert mc.spec_kind(spec) == "milestone" and mc.spec_slug(spec) == "milestone-member_257-200000"
    assert mc.spec_member_label(spec) == "257@200000" and mc.is_milestone_label("257@200000")
    assert not mc.is_milestone_label("257·psnr")
    assert mc.spec_label(spec) == "Milestone 257 @ 0.2M"
    assert mc.spec_label("milestone:member_257@15000000") == "Milestone 257 @ 15M"

    assert mc._milestone_specs() == []
    for step in (2000000, 200000):
        _milestone(root, "member_257", step, step / 1e6)
    (root / "member_257" / "origin.json").write_text(json.dumps({"target_steps": 15000000, "num_res_blocks": 16}))
    specs = mc._milestone_specs()
    assert [s.spec for s in specs] == ["milestone:member_257@200000", "milestone:member_257@2000000"]
    first = specs[0]
    assert first.kind == "milestone" and first.available and first.reads == ("257@200000",)
    assert first.fingerprint and first.fingerprint != specs[1].fingerprint
    assert first.details == {"milestone": True, "member": "member_257", "step": 200000,
                             "target_steps": 15000000, "num_res_blocks": 16}
    assert mc.is_single_model(first)
    assert mc.member_fingerprints(["257@200000"])["257@200000"].startswith("ckpt-200000:")


def test_the_runner_runs_a_milestone_from_its_own_ensemble_dir(root):
    from euclid_polish.web.helpers import model_catalog as mc

    _milestone(root, "member_257", 200000, 2.0)
    built = []

    class Fake:
        def __init__(self, base, *, labels, include_universal, starless):
            built.append((base, labels, include_universal, starless))

        def member_arrays(self, lr):
            return np.stack([np.full((8, 8, 4), 7.0, np.float32)])

    runner = mc.EnsembleMemberRunner(factory=Fake, labels=["257@200000"])
    out = runner.predict(np.ones((4, 4, 4), np.float32), "257@200000")
    runner.predict(np.ones((4, 4, 4), np.float32), "257@200000")
    assert out.shape == (8, 8, 4) and float(out[0, 0, 0]) == 7.0
    assert len(built) == 1                                    # loaded once, reused
    base, labels, include_universal, starless = built[0]
    assert base.endswith("ensembles/step_000200000") and labels == ["member_257"]
    assert include_universal and starless is False


def test_the_processor_compares_every_measured_milestone_on_the_poster(monkeypatch):
    from types import SimpleNamespace

    from scripts import process_milestones as pm

    monkeypatch.setattr(pm, "done_steps", lambda member, res_dir: {2000000, 200000})
    monkeypatch.setattr(pm.real_tiles, "list_entries",
                        lambda source: [SimpleNamespace(source=source, id="target_181255")])
    seen = {}

    def fake_run(tiles, specs, *, label, log):
        seen.update(tiles=tiles, specs=specs, label=label)
        return {"id": "x", "status": "done", "counts": {"outputs_computed": 1, "outputs_reused": 2}}

    monkeypatch.setattr(pm.experiments, "run_experiment", fake_run)
    pm.poster_comparison("member_257", ["universal:member_249"], "res")
    assert seen["tiles"] == [("poster", "target_181255")]
    assert seen["specs"] == ["universal:member_249", "milestone:member_257@200000", "milestone:member_257@2000000"]
    assert "2,000,000" in seen["label"]


def test_the_console_payload_joins_scores_and_jwst_scans(tmp_path, monkeypatch):
    scores, res = tmp_path / "scores", tmp_path / "res"
    scores.mkdir()
    res.mkdir()
    monkeypatch.setattr(milestones, "scores_dir", lambda: scores)
    monkeypatch.setattr(milestones, "resolution_dir", lambda: res)
    (scores / "member_257.json").write_text(json.dumps({"identity": {}, "steps": {
        "8000000": {"mean": 62.5, "integrated": [56.7, 66.6, 63.7, 62.9], "scored_hours": 37.3,
                    "learned_knees": [0.06, 0.01, 0.02, 0.02], "scored_at": "2026-10-27T10:00:00+00:00"},
        "200000": {"mean": 59.2, "integrated": [53.2, 63.2, 60.6, 59.7], "scored_hours": 0.9}}}))
    scan = [{"multiple": 1.0, "sb": 25.7, "resolution_coherence": {"px": 22.1, "arcsec": 1.105}},
            {"multiple": 3.0, "sb": 24.5, "resolution_coherence": None}]
    (res / "resolution_H_E.json").write_text(json.dumps({"meta": {"jwst_filter": "F200W", "euclid_noise_sb": 25.7},
        "models": {"milestone:member_257@8000000": {"knee_scan": scan}, "milestone:member_999@1": {"knee_scan": scan},
                   "production": {"knee_scan": scan}}}))
    out = milestones.report_payload()
    (m,) = out["members"]
    assert m["member"] == "member_257" and [r["step"] for r in m["steps"]] == [200000, 8000000]
    assert m["steps"][1]["spec"] == "milestone:member_257@8000000" and m["steps"][1]["hours"] == 37.3
    assert set(m["resolution"]) == {"H_E"}
    models = m["resolution"]["H_E"]["models"]
    assert set(models) == {"milestone:member_257@8000000", "production"}        # another run's milestones left out
    assert models["production"] == [{"multiple": 1.0, "sb": 25.7, "arcsec": 1.105}]
    assert m["resolution"]["H_E"]["jwst_filter"] == "F200W"
