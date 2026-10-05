"""The label-keyed member-cube layout: file names, the migration of a
positional bucket, fingerprint-checked reuse, and every reader of the cube
buckets (viewer, gate fit/compare fields, knee curves, studies) on it."""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pytest

from euclid_polish.ensemble import member_fingerprints
from euclid_polish.eval import ensemble_cube_cache as ecc
from euclid_polish.eval.ensemble_cube_cache import (
    load_cached_member_stack,
    member_cube_key,
    member_cube_path,
    migrate_positional_bucket,
    missing_cached_members,
)
from euclid_polish.eval.spatial_gate_fit import load_cube_fields
from euclid_polish.studies import candidates
from euclid_polish.studies import fields as study_fields
from euclid_polish.web import remote
from euclid_polish.web.helpers import ensemble_viz as ev
from euclid_polish.web.helpers import viewer_data as vd
from tests._ensemble_cube_cache_fixtures import (
    N_FIELDS,
    Cap,
    lr_of,
    make_env,
    member_output,
    set_checkpoint,
)
from tests._studies_fixtures import LABELS as STUDY_LABELS
from tests._studies_fixtures import make_env as make_study_env
from tests._studies_fixtures import write_eval_summary


def test_member_cube_names_are_keyed_by_label():
    assert member_cube_key("196·psnr") == "196"
    assert member_cube_key("196·loss") == "196-loss"
    assert member_cube_key("a") == "a"
    assert member_cube_path("/c", "07·psnr", 12) == os.path.join("/c", "member_07_00012.npy")


def _positional(directory, labels, n_fields=2, extra_positions=1):
    os.makedirs(directory, exist_ok=True)
    for rec in range(n_fields):
        for pos in range(len(labels) + extra_positions):
            np.save(os.path.join(directory, f"member{pos}_{rec:05d}.npy"),
                    np.full((2, 2, 4), 10 * pos + rec, np.float32))
    with open(os.path.join(directory, "viz_index.json"), "w") as handle:
        json.dump({"subset": "test", "indices": list(range(n_fields)),
                   "member_labels": list(labels)}, handle)


def test_migration_renames_by_the_manifest_and_adopts_only_given_fingerprints(tmp_path):
    d = str(tmp_path / "cubes")
    _positional(d, ["170·psnr", "171·psnr"])
    man = migrate_positional_bucket(d, adopt={"171·psnr": "ckpt-3:1:2"})
    assert man["member_fps"] == {"170·psnr": None, "171·psnr": "ckpt-3:1:2"}
    assert float(np.load(member_cube_path(d, "171·psnr", 1)).flat[0]) == 11.0
    assert not [f for f in os.listdir(d) if f.startswith("member") and f[6].isdigit()]
    assert json.load(open(os.path.join(d, "viz_index.json")))["member_fps"] == man["member_fps"]
    # Idempotent: a label-keyed bucket is left as it is.
    assert migrate_positional_bucket(d) == man


def test_reuse_requires_the_recorded_fingerprint_to_be_current(tmp_path, monkeypatch):
    d = str(tmp_path / "cubes")
    _positional(d, ["170·psnr", "171·psnr"], extra_positions=0)
    current = {"170·psnr": "fp-170", "171·psnr": "fp-171"}
    # A positional bucket records no fingerprint: never served for real members.
    assert load_cached_member_stack(1, subset="test", cubes_dir=d, active=list(current),
                                    fingerprints=current) is None
    migrate_positional_bucket(d, adopt={"170·psnr": "fp-170", "171·psnr": "old"})
    out = load_cached_member_stack(1, subset="test", cubes_dir=d, active=["170·psnr"],
                                   fingerprints=current)
    assert out is not None and float(out[0].flat[0]) == 1.0
    assert load_cached_member_stack(1, subset="test", cubes_dir=d, active=list(current),
                                    fingerprints=current) is None
    assert missing_cached_members(1, subset="test", cubes_dir=d, active=list(current),
                                  fingerprints=current) == ["171·psnr"]
    # Default fingerprints: the members' checkpoints under the ensemble dir.
    monkeypatch.setattr(ecc, "member_fingerprints",
                        lambda base, labels: {lb: current.get(lb) for lb in labels})
    assert load_cached_member_stack(1, subset="test", cubes_dir=d,
                                    active=["170·psnr"]) is not None
    # Diagnostics may read whatever the bucket holds.
    assert load_cached_member_stack(1, subset="test", cubes_dir=d, active=list(current),
                                    require_current=False) is not None


# --------------------------------------------------------------------------- #
# readers on a bucket the evaluation filled
# --------------------------------------------------------------------------- #

@pytest.fixture
def evaluated(tmp_path, monkeypatch):
    env = make_env(tmp_path, monkeypatch)
    ev.job_ensemble_evaluate(Cap(), num_images=N_FIELDS, starless=False)
    return env


def test_viewer_serves_member_tiers_and_subset_movies_from_label_keyed_cubes(evaluated):
    meta = vd._ensemble_meta({"mode": "starfull"})
    assert meta["member_labels"] == evaluated["labels"]
    cube, info = vd._ensemble_cube(1, "member2", {})
    np.testing.assert_array_equal(cube, member_output(lr_of(1), 3, 1))
    assert "03·psnr" in info["label"]
    mean, _info = vd._ensemble_cube(0, "mean", {"members": "0,2"})
    np.testing.assert_allclose(mean, (member_output(lr_of(0), 1, 1)
                                      + member_output(lr_of(0), 3, 1)) / 2, rtol=1e-6)


def test_gate_fields_and_knee_curves_read_label_keyed_cubes(evaluated):
    fields, labels = load_cube_fields(str(evaluated["cubes"]), str(evaluated["records"]),
                                      "test", target_name="hr", target_fwhm_arcsec=0.066)
    assert labels == evaluated["labels"] and len(fields) == N_FIELDS
    assert fields[0].member_paths[1] == member_cube_path(str(evaluated["cubes"]), "02·psnr", 0)
    np.testing.assert_array_equal(fields[2].members_e()[0], member_output(lr_of(2), 1, 1))
    curves = ev.knee_psnr_fields(False)
    assert curves is not None and curves["fields"] == list(range(N_FIELDS))
    assert curves["curves"].shape[1] == len(labels) + 1          # members + mean


def test_cached_field_streams_skip_a_field_missing_a_member_cube(evaluated):
    """A fill stopped half-way leaves fields without some member's cube: the
    re-scoring from the cache skips them instead of stacking fewer members."""
    os.remove(member_cube_path(str(evaluated["cubes"]), "02·psnr", 1))
    recs = [rec for *_planes, rec in ev._iter_cached_fields(False)]
    assert recs == [0, 2]


def test_a_continued_member_is_not_reused_by_the_synthetic_cache_reader(evaluated):
    labels = evaluated["labels"]
    assert load_cached_member_stack(0, subset="test", active=labels) is not None
    set_checkpoint(evaluated["base"], 1, step=2)
    assert load_cached_member_stack(0, subset="test", active=labels) is None
    assert missing_cached_members(0, subset="test", active=labels) == ["01·psnr"]


def test_studies_read_label_keyed_test_and_blackout_cubes(tmp_path, monkeypatch):
    env = make_study_env(tmp_path, monkeypatch)
    monkeypatch.setattr(remote.STATE, "ssh", None)
    migrate_positional_bucket(str(env["cubes"]))
    migrate_positional_bucket(str(env["blackout"]), ecc.BLACKOUT_INDEX,
                              adopt=member_fingerprints(str(env["base"]), STUDY_LABELS))
    assert not list(env["cubes"].glob("member[0-9]*_*.npy"))
    fields = {f["fid"]: f for f in candidates.candidates(False)["fields"]}
    assert fields["test-00001"]["available"] and fields["blackout-00002"]["available"]
    os.remove(member_cube_path(str(env["cubes"]), STUDY_LABELS[2], 1))
    fields = {f["fid"]: f for f in candidates.candidates(False)["fields"]}
    assert not fields["test-00001"]["available"]
    assert "member(s) 03" in fields["test-00001"]["reason"]

    source = study_fields.open_field("blackout-00002", starless=False, labels=STUDY_LABELS)
    assert source.meta["blackout"]["member_labels"] == STUDY_LABELS
    products = dict(source.products())
    np.testing.assert_array_equal(
        products["member_02"], np.load(member_cube_path(str(env["blackout"]), "02·psnr", 2)))
    test_source = study_fields.open_field("test-00000", starless=False, labels=STUDY_LABELS)
    test_products = dict(test_source.products())
    np.testing.assert_array_equal(
        test_products["member_03"], np.load(member_cube_path(str(env["cubes"]), "03·psnr", 0)))


def _continue_study_member(base, number: int, step: int) -> None:
    directory = base / f"member_{number:02d}"
    (directory / f"ckpt-{step}.index").write_bytes(b"continued")
    (directory / "checkpoint").write_text(f'model_checkpoint_path: "ckpt-{step}"\n')


def test_studies_judge_label_keyed_blackout_cubes_by_their_fingerprints(tmp_path, monkeypatch):
    """A label-keyed blackout bucket records which checkpoint made each
    member's cubes: that decides, not file times (rsync keeps a pulled
    checkpoint's FASRC mtime; a cube can look older than the weights that made
    it)."""
    env = make_study_env(tmp_path, monkeypatch)
    monkeypatch.setattr(remote.STATE, "ssh", None)
    migrate_positional_bucket(str(env["blackout"]), ecc.BLACKOUT_INDEX,
                              adopt=member_fingerprints(str(env["base"]), STUDY_LABELS))
    past = time.time() - 10 * 86400
    for path in env["blackout"].glob("member_*.npy"):
        os.utime(path, (past, past))
    fields = {f["fid"]: f for f in candidates.candidates(False)["fields"]}
    assert fields["blackout-00002"]["available"], fields["blackout-00002"]["reason"]

    _continue_study_member(env["base"], 2, step=6)
    # Re-evaluated since: the test cubes are current again, the blackouts not.
    write_eval_summary(env["regime"], env["base"], STUDY_LABELS)
    fields = {f["fid"]: f for f in candidates.candidates(False)["fields"]}
    assert fields["test-00002"]["available"], fields["test-00002"]["reason"]
    reason = fields["blackout-00002"]["reason"]
    assert not fields["blackout-00002"]["available"]
    assert "member 02's blackout cubes were made by another checkpoint" in reason
    assert "run a combiner comparison" in reason and "delete" not in reason
