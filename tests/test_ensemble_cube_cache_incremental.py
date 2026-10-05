"""The member-cube buckets fill incrementally, keyed by (member label,
checkpoint fingerprint): the test evaluation and the validate cubes the
combiner fits read run ONLY the members whose cubes are missing or were made
by another checkpoint; a departed member is dropped from the cache, and only
a records change re-infers everyone. Legacy positional buckets are migrated,
adopting a fingerprint only where the evaluation recorded it."""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from euclid_polish import ensemble_registry as er
from euclid_polish.config import Config
from euclid_polish.ensemble import member_fingerprint
from euclid_polish.eval import spatial_gate_compare as sgc
from euclid_polish.eval import spatial_gate_fit
from euclid_polish.eval.ensemble_cube_cache import member_cube_path
from euclid_polish.eval.spatial_gate_fit import LazyMemberRunner
from euclid_polish.web.helpers import ensemble_viz as ev
from tests._ensemble_cube_cache_fixtures import (
    N_FIELDS,
    Cap,
    FakeEnsemble,
    add_member,
    lr_of,
    make_env,
    member_dir,
    member_output,
    regenerate_records,
    run_counts,
    set_checkpoint,
)


@pytest.fixture
def env(tmp_path, monkeypatch):
    return make_env(tmp_path, monkeypatch)


def _manifest(directory):
    return json.loads((directory / "viz_index.json").read_text())


def _fps(env, labels):
    return {lb: member_fingerprint(str(member_dir(env["base"], int(lb.split("·")[0]))))
            for lb in labels}


def _evaluate(**kw):
    return ev.job_ensemble_evaluate(Cap(), num_images=N_FIELDS, starless=False, **kw)


def _validate():
    return ev._prepare_validate_cubes(Cap(), starless=False, num_images=N_FIELDS,
                                      target_fwhm=Config.TARGET_PSF_FWHM_ARCSEC)


def _mtimes(directory, label):
    return [os.stat(member_cube_path(str(directory), label, rec)).st_mtime_ns
            for rec in range(N_FIELDS)]


def _assert_mean_is_member_mean(directory, labels):
    for rec in range(N_FIELDS):
        stack = np.stack([np.load(member_cube_path(str(directory), lb, rec)) for lb in labels])
        np.testing.assert_allclose(np.load(directory / f"sr_{rec:05d}.npy"), stack.mean(0),
                                   rtol=1e-6)


# --------------------------------------------------------------------------- #
# test bucket (the console's Evaluate)
# --------------------------------------------------------------------------- #

def test_evaluate_fills_the_test_bucket_by_label_and_fingerprint(env):
    out = _evaluate()
    labels = env["labels"]
    assert run_counts() == dict.fromkeys(labels, N_FIELDS)
    man = _manifest(env["cubes"])
    assert man["member_labels"] == labels and man["member_fps"] == _fps(env, labels)
    assert sorted(man["indices"]) == list(range(N_FIELDS))
    for rec in range(N_FIELDS):
        cube = np.load(member_cube_path(str(env["cubes"]), "02·psnr", rec))
        np.testing.assert_array_equal(cube, member_output(lr_of(rec), 2, 1))
        assert not (env["cubes"] / f"member1_{rec:05d}.npy").exists()   # no positional files
    _assert_mean_is_member_mean(env["cubes"], labels)
    assert out["member_labels"] == labels and out["n_fields"] == N_FIELDS


def test_a_second_evaluation_is_a_full_cache_hit(env):
    _evaluate()
    FakeEnsemble.reset()
    assert _evaluate()["reused"] is True
    assert FakeEnsemble.loaded == [] and FakeEnsemble.runs == []
    # Even when the summary is gone (a real re-evaluation), every member cube
    # is current: the numbers are recomputed from the cache, no member runs.
    (env["regime"] / "eval_summary.json").unlink()
    out = _evaluate()
    assert out["reused"] is False and out["n_fields"] == N_FIELDS
    assert FakeEnsemble.loaded == [] and FakeEnsemble.runs == []


def test_a_continued_member_is_reinferred_alone(env):
    _evaluate()
    before = {lb: _mtimes(env["cubes"], lb) for lb in env["labels"]}
    FakeEnsemble.reset()
    set_checkpoint(env["base"], 2, step=2)                 # same label, new weights
    _evaluate()
    assert FakeEnsemble.loaded == [["02·psnr"]]
    assert run_counts() == {"02·psnr": N_FIELDS}
    for rec in range(N_FIELDS):
        np.testing.assert_array_equal(
            np.load(member_cube_path(str(env["cubes"]), "02·psnr", rec)),
            member_output(lr_of(rec), 2, 2))
    assert _mtimes(env["cubes"], "01·psnr") == before["01·psnr"]       # untouched
    assert _mtimes(env["cubes"], "03·psnr") == before["03·psnr"]
    assert _manifest(env["cubes"])["member_fps"]["02·psnr"] == _fps(env, ["02·psnr"])["02·psnr"]
    _assert_mean_is_member_mean(env["cubes"], env["labels"])


def test_an_added_member_is_inferred_alone_and_a_departed_one_dropped(env):
    _evaluate()
    FakeEnsemble.reset()
    added = add_member(env["base"], 4)
    _evaluate()
    assert FakeEnsemble.loaded == [[added]] and run_counts() == {added: N_FIELDS}
    assert _manifest(env["cubes"])["member_labels"] == [*env["labels"], added]

    FakeEnsemble.reset()
    er.archive_member_entry(str(env["base"]), "member_01", zip_path="models/x.zip",
                            commit=None)
    out = _evaluate()
    remaining = ["02·psnr", "03·psnr", added]
    assert FakeEnsemble.loaded == [] and FakeEnsemble.runs == []
    man = _manifest(env["cubes"])
    assert man["member_labels"] == remaining and "01·psnr" not in man["member_fps"]
    assert not list(env["cubes"].glob("member_01_*.npy"))
    _assert_mean_is_member_mean(env["cubes"], remaining)
    assert out["member_labels"] == remaining


def test_a_records_change_reinfers_every_member(env):
    _evaluate()
    FakeEnsemble.reset()
    regenerate_records(env["records"], "test")
    _evaluate()
    assert run_counts() == dict.fromkeys(env["labels"], N_FIELDS)
    np.testing.assert_array_equal(
        np.load(member_cube_path(str(env["cubes"]), "01·psnr", 0)),
        member_output(lr_of(0, seed=7), 1, 1))


def test_force_reinfers_every_member(env):
    _evaluate()
    FakeEnsemble.reset()
    _evaluate(force=True)
    assert run_counts() == dict.fromkeys(env["labels"], N_FIELDS)


def test_a_continued_member_refreshes_the_knee_curves(env):
    """The PSNR-vs-knee curves are keyed on the members' checkpoints too: a
    continued member's new cubes are scored, not the old curves reused."""
    _evaluate()
    path = ev._knee_psnr_path(False)
    with open(path) as handle:
        before = json.load(handle)
    set_checkpoint(env["base"], 2, step=2)
    _evaluate()
    with open(path) as handle:
        after = json.load(handle)
    assert after["identity"]["member_fps"] == _manifest(env["cubes"])["member_fps"]
    assert after["models"][1]["integrated"] != before["models"][1]["integrated"]
    assert after["models"][0]["integrated"] == before["models"][0]["integrated"]
    assert ev.knee_psnr_status(False)["stale"] is False


def test_an_evaluation_keeps_only_the_combiner_outputs_it_baked(env):
    """Member cubes persist across evaluations, a combiner's outputs do not:
    one the new evaluation no longer applies leaves no cube behind."""
    _evaluate()
    stale = env["cubes"] / "comb_spatial_gate_00001.npy"
    np.save(stale, np.zeros((2, 2, 4), np.float32))
    set_checkpoint(env["base"], 1, step=2)
    _evaluate()
    assert not stale.exists()
    assert _manifest(env["cubes"])["has_combiner_spatial_gate"] is False


def _interrupt_member_runs(monkeypatch, after):
    """Kill member inference once ``after`` member runs have happened."""
    real = FakeEnsemble.member_arrays

    def run(self, lr, indices=None):
        if len(FakeEnsemble.runs) >= after:
            raise KeyboardInterrupt("killed")
        return real(self, lr, indices)

    monkeypatch.setattr(FakeEnsemble, "member_arrays", run)
    return lambda: monkeypatch.setattr(FakeEnsemble, "member_arrays", real)


def test_an_interrupted_evaluation_is_never_reused_as_a_complete_one(env, monkeypatch):
    """An Evaluate stopped half-way keeps the member cubes it made, but no
    cache re-score (a combiner fit's test scoring, a promotion) turns the
    partial bucket into a summary or knee curves the next Evaluate reuses:
    that one infers the continued member on the fields the stopped run did
    not reach and scores every field."""
    _evaluate()
    set_checkpoint(env["base"], 2, step=2)
    FakeEnsemble.reset()
    restore = _interrupt_member_runs(monkeypatch, 1)        # field 0 only
    with pytest.raises(KeyboardInterrupt):
        _evaluate()
    restore()
    ev._reevaluate_from_cached_cubes(False, num_images=N_FIELDS)
    FakeEnsemble.reset()
    out = _evaluate()
    assert out["reused"] is False and run_counts() == {"02·psnr": N_FIELDS - 1}
    assert out["n_fields"] == N_FIELDS
    _assert_mean_is_member_mean(env["cubes"], env["labels"])
    assert ev.knee_psnr_status(False)["n_fields"] == N_FIELDS


def test_a_forced_evaluation_stopped_half_way_is_not_reused(env, monkeypatch):
    """``force`` empties the bucket first: once stopped, the previous run's
    summary must not stand in for cubes that are gone."""
    _evaluate()
    FakeEnsemble.reset()
    restore = _interrupt_member_runs(monkeypatch, 1)
    with pytest.raises(KeyboardInterrupt):
        _evaluate(force=True)
    restore()
    FakeEnsemble.reset()
    out = _evaluate()
    assert out["reused"] is False and out["n_fields"] == N_FIELDS
    assert run_counts() == dict.fromkeys(env["labels"], N_FIELDS)
    _assert_mean_is_member_mean(env["cubes"], env["labels"])


def test_a_stale_member_blocks_reuse_of_the_cached_summary(env):
    """A combiner refit re-scores the test cubes from the cache; its summary
    records the fingerprints the cubes were MADE with, so a member continued
    in the meantime still forces its re-inference on the next Evaluate."""
    _evaluate()
    set_checkpoint(env["base"], 3, step=5)
    summary = ev._reevaluate_from_cached_cubes(False, num_images=N_FIELDS)
    assert summary is not None
    FakeEnsemble.reset()
    out = _evaluate()
    assert out["reused"] is False and run_counts() == {"03·psnr": N_FIELDS}


# --------------------------------------------------------------------------- #
# validate bucket (combiner / gate fits)
# --------------------------------------------------------------------------- #

def test_validate_cubes_fill_incrementally(env):
    _base, _rdir, _fp, val_dir, indices, labels, _target = _validate()
    assert indices == list(range(N_FIELDS)) and labels == env["labels"]
    assert run_counts() == dict.fromkeys(labels, N_FIELDS)
    assert _manifest(env["validate"])["member_fps"] == _fps(env, labels)

    FakeEnsemble.reset()
    assert _validate()[4] == list(range(N_FIELDS))
    assert FakeEnsemble.loaded == [] and FakeEnsemble.runs == []       # full cache hit

    set_checkpoint(env["base"], 1, step=3)
    _validate()
    assert FakeEnsemble.loaded == [["01·psnr"]] and run_counts() == {"01·psnr": N_FIELDS}
    np.testing.assert_array_equal(
        np.load(member_cube_path(val_dir, "01·psnr", 2)),
        member_output(lr_of(2, seed=3), 1, 3))

    FakeEnsemble.reset()
    added = add_member(env["base"], 5)
    assert _validate()[5] == [*labels, added]
    assert run_counts() == {added: N_FIELDS}

    FakeEnsemble.reset()
    er.archive_member_entry(str(env["base"]), "member_02", zip_path="models/x.zip",
                            commit=None)
    remaining = ["01·psnr", "03·psnr", added]
    assert _validate()[5] == remaining
    assert FakeEnsemble.runs == []
    assert not list(env["validate"].glob("member_02_*.npy"))
    _assert_mean_is_member_mean(env["validate"], remaining)


def _interrupt_on_call(monkeypatch, number):
    """Kill the fill at the ``number``-th field whose aggregates it rewrites
    (after that field's member cubes are stored, before its ``sr_``)."""
    calls = {"n": 0}

    def save_lr(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == number:
            raise KeyboardInterrupt("killed")
        return real(*args, **kwargs)

    real = ev.save_cached_field_lr
    monkeypatch.setattr(ev, "save_cached_field_lr", save_lr)
    return lambda: monkeypatch.setattr(ev, "save_cached_field_lr", real)


@pytest.mark.parametrize("change", ["continued", "archived"])
def test_an_interrupted_validate_refill_leaves_no_stale_aggregates(env, monkeypatch, change):
    _validate()
    remaining = list(env["labels"])
    if change == "continued":
        set_checkpoint(env["base"], 2, step=2)
    else:
        er.archive_member_entry(str(env["base"]), "member_03", zip_path="models/x.zip",
                                commit=None)
        remaining.remove("03·psnr")
    restore = _interrupt_on_call(monkeypatch, 2 if change == "continued" else 1)
    with pytest.raises(KeyboardInterrupt):
        _validate()
    restore()
    _validate()
    _assert_mean_is_member_mean(env["validate"], remaining)


def test_validate_records_or_target_change_reinfers_everyone(env):
    _validate()
    FakeEnsemble.reset()
    regenerate_records(env["records"], "validate")
    _validate()
    assert run_counts() == dict.fromkeys(env["labels"], N_FIELDS)
    FakeEnsemble.reset()
    ev._prepare_validate_cubes(Cap(), starless=False, num_images=N_FIELDS,
                               target_fwhm=Config.TARGET_PSF_FWHM_ARCSEC * 2)
    assert run_counts() == dict.fromkeys(env["labels"], N_FIELDS)


def test_the_gate_fit_validate_blackouts_follow_a_continued_member(env, monkeypatch):
    """The blackout copies a gate fit trains on are re-inferred for a
    continued member too (they used to be keyed on the labels only)."""
    monkeypatch.setattr(ev, "SPATIAL_GATE_BLACKOUT_FIELDS", 2)

    def stamp(lr_e, rng, **_kw):
        out = np.array(lr_e, np.float32, copy=True)
        out[:2, :2] = 0.0
        return out

    monkeypatch.setattr(spatial_gate_fit, "stamp_blackouts", stamp)
    seen = []
    monkeypatch.setattr(ev, "fit_spatial_gate",
                        lambda train, held, labels, **kw: seen.append(
                            [f.member_paths for f in train if f.tag == "blackout"])
                        or _StubGate(labels))
    _validate_and_fit()
    FakeEnsemble.reset()
    set_checkpoint(env["base"], 3, step=4)
    _validate_and_fit()
    blackout_dir = env["regime"] / "cubes_validate_blackout"
    index = json.loads((blackout_dir / "blackout_index.json").read_text())
    assert index["member_fps"]["03·psnr"] == _fps(env, ["03·psnr"])["03·psnr"]
    # validate: 03 on every field; blackout copies: 03 on each of the 2 fields
    assert run_counts() == {"03·psnr": N_FIELDS + 2}
    assert all(os.path.basename(p).startswith("member_") for p in seen[-1][0])


def test_gate_compare_and_the_script_fit_refuse_cubes_of_an_old_checkpoint(env):
    """The script's fit fields and every compare read cubes as they are: a
    member continued since they were made is refused, naming it, until a
    console fit / Evaluate re-infers just that member."""
    _evaluate()
    _validate()
    set_checkpoint(env["base"], 2, step=6)
    runner = LazyMemberRunner(str(env["base"]), starless=False, labels=env["labels"])
    for run in (runner, None):              # the console compares without a runner too
        with pytest.raises(RuntimeError, match=r"member\(s\) 02 .*evaluate"):
            sgc.run_compare(regime_dir=str(env["regime"]), records_dir=str(env["records"]),
                            gates=[], runner=run, blackout_fields=0)
    with pytest.raises(RuntimeError, match=r"member\(s\) 02 .*combiner fit"):
        sgc.load_fit_fields(str(env["validate"]), str(env["records"]))
    FakeEnsemble.reset()
    _validate()
    fields, labels = sgc.load_fit_fields(str(env["validate"]), str(env["records"]))
    assert run_counts() == {"02·psnr": N_FIELDS}
    assert labels == env["labels"] and len(fields) == N_FIELDS


def test_the_script_fit_refuses_a_validate_bucket_a_stopped_refill_left_incomplete(
        env, monkeypatch):
    """A console refill stopped half-way leaves fields without the continued
    member's cube; the script's fit must not silently train on the rest."""
    _validate()
    set_checkpoint(env["base"], 2, step=2)
    restore = _interrupt_on_call(monkeypatch, 2)            # fields 0 and 1 refilled
    with pytest.raises(KeyboardInterrupt):
        _validate()
    restore()
    with pytest.raises(RuntimeError, match=r"1 field.*combiner fit"):
        sgc.load_fit_fields(str(env["validate"]), str(env["records"]))
    _validate()
    fields, _labels = sgc.load_fit_fields(str(env["validate"]), str(env["records"]))
    assert len(fields) == N_FIELDS


class _StubGate:
    def __init__(self, labels):
        self.member_labels = list(labels)
        self.fit_meta = {}
        self.records_fp = None
        self.starfull = True


def _validate_and_fit():
    (base, records_dir, records_fp, validate_dir, indices, labels,
     target) = _validate()
    return ev._fit_spatial_gate_on_validate(
        Cap(), base=base, labels=labels, indices=indices, records_dir=records_dir,
        records_fp=records_fp, validate_dir=validate_dir, starless=False, target=target,
        target_fwhm=Config.TARGET_PSF_FWHM_ARCSEC)


# --------------------------------------------------------------------------- #
# migration of positional buckets
# --------------------------------------------------------------------------- #

def _positional_bucket(directory, labels, *, subset, records_fp, extra=None):
    """A bucket as the code before label keying wrote it: member{i}_<rec>."""
    directory.mkdir(parents=True, exist_ok=True)
    for rec in range(N_FIELDS):
        lr = lr_of(rec, seed=0 if subset == "test" else 3)
        stack = [member_output(lr, int(lb.split("·")[0]), 1) for lb in labels]
        for i, cube in enumerate(stack):
            np.save(directory / f"member{i}_{rec:05d}.npy", cube)
        np.save(directory / f"sr_{rec:05d}.npy", np.mean(stack, axis=0))
    np.save(directory / f"member{len(labels)}_00000.npy", np.zeros(1))   # orphan position
    (directory / "viz_index.json").write_text(json.dumps({
        "subset": subset, "indices": list(range(N_FIELDS)), "member_labels": list(labels),
        "records_fp": records_fp, "target_psf_fwhm_arcsec": Config.TARGET_PSF_FWHM_ARCSEC,
        "pca_n": 3, **(extra or {})}))


def _write_summary(env, labels, *, recomputed: bool):
    ev._ensemble_regime_dir(False)
    identity = ev._eval_identity(str(env["base"]), str(env["records"]), "test",
                                 str(env["regime"]), starless=False, num_images=N_FIELDS)
    summary = {"member_labels": list(labels), "eval_identity": identity, "reused": False}
    if recomputed:
        summary["recomputed_from_cubes"] = True
    (env["regime"] / "eval_summary.json").write_text(json.dumps(summary))


def test_a_positional_test_bucket_adopts_the_fingerprints_its_evaluation_recorded(env):
    records_fp = ev._eval_records_fingerprint(str(env["records"]), "test")
    _positional_bucket(env["cubes"], env["labels"], subset="test", records_fp=records_fp)
    _write_summary(env, env["labels"], recomputed=False)
    out = _evaluate()
    assert out["reused"] is True
    assert FakeEnsemble.loaded == [] and FakeEnsemble.runs == []
    man = _manifest(env["cubes"])
    assert man["member_fps"] == _fps(env, env["labels"])
    assert not list(env["cubes"].glob("member[0-9]*_*.npy"))           # renamed or dropped
    np.testing.assert_array_equal(
        np.load(member_cube_path(str(env["cubes"]), "03·psnr", 1)),
        member_output(lr_of(1), 3, 1))


def test_a_positional_bucket_without_proven_fingerprints_is_reinferred(env):
    """A summary rebuilt from cubes (or none at all) stamped the checkpoints
    current THEN, not the ones that made the cubes: never trusted."""
    records_fp = ev._eval_records_fingerprint(str(env["records"]), "test")
    _positional_bucket(env["cubes"], env["labels"], subset="test", records_fp=records_fp)
    _write_summary(env, env["labels"], recomputed=True)
    _evaluate()
    assert run_counts() == dict.fromkeys(env["labels"], N_FIELDS)
    assert not list(env["cubes"].glob("member[0-9]*_*.npy"))


def test_a_positional_validate_bucket_is_reinferred(env):
    records_fp = ev._eval_records_fingerprint(str(env["records"]), "validate")
    _positional_bucket(env["validate"], env["labels"], subset="validate",
                       records_fp=records_fp)
    _validate()
    assert run_counts() == dict.fromkeys(env["labels"], N_FIELDS)
    assert not list(env["validate"].glob("member[0-9]*_*.npy"))
    assert _manifest(env["validate"])["member_fps"] == _fps(env, env["labels"])
