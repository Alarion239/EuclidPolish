"""The ``study`` viewer collection: a study's attached fields from the
fetched-field cache (LR, gate, mean, HR, BHR, holes, members hidden), 404
"fetch the field first" until fetched."""
from __future__ import annotations

import numpy as np
import pytest

from euclid_polish.studies import cache as study_cache
from euclid_polish.studies import fields, freeze
from euclid_polish.studies.store import StudyStore
from euclid_polish.web.helpers import experiments, viewer_data
from euclid_polish.web.helpers.viewer_data import ViewerError
from tests._studies_fixtures import LABELS, FakeRemote, _truth, make_env

REMOTE_TRACKING = "/n/holylabs/lab/me/EuclidPolish/tracking"


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    env = make_env(tmp_path, monkeypatch)
    monkeypatch.setattr(experiments, "free_bytes", lambda _path: 10 ** 15)
    ssh = FakeRemote(tmp_path / "remote")
    store = StudyStore()
    sid = freeze.start(store, name="Viewer", note="", starless=False,
                       fids=["test-00001", "blackout-00002"], connected=True)
    freeze.run(store, sid, ssh=ssh, remote_tracking_dir=REMOTE_TRACKING)
    return {**env, "sid": sid, "ssh": ssh, "store": store}


def _fetch(frozen, fid, products=None):
    manifest = frozen["store"].manifest(frozen["sid"])
    record = next(f for f in manifest["fields"] if f["fid"] == fid)
    fields.fetch_field(frozen["ssh"], frozen["sid"], record, record["remote_dir"],
                       study_cache.FieldCache(), products=products)


def test_meta_lists_the_attached_fields_and_tiers(frozen):
    meta = viewer_data.get_meta("study", {"study": frozen["sid"]})
    assert meta["count"] == 2 and meta["collection"] == "study"
    assert [o["id"] for o in meta["objects"]] == ["test-00001", "blackout-00002"]
    assert all(o["fetched"] is False for o in meta["objects"])
    keys = [t["key"] for t in meta["tiers"]]
    assert keys[:6] == ["lr", "sr", "mean", "hr", "bhr", "mask"]
    members = [t for t in meta["tiers"] if t["key"].startswith("member")]
    assert len(members) == 3 and all(t["hidden"] for t in members)
    assert meta["member_labels"] == LABELS and meta["default_tier"] == "sr"
    assert "mask" not in meta["objects"][0]["tiers"] and "mask" in meta["objects"][1]["tiers"]
    # Member tiers are listed per object only once fetched; the others are
    # dimmed by the viewer with the reason from ``missing_tier_labels``.
    assert not [t for t in meta["objects"][0]["tiers"] if t.startswith("member")]
    assert meta["missing_tier_labels"] == {"member0": "fetch member 01 first",
                                           "member1": "fetch member 02 first",
                                           "member2": "fetch member 03 first"}


def test_a_cube_needs_the_field_fetched_first(frozen):
    with pytest.raises(ViewerError) as err:
        viewer_data.get_cube("study", 0, "hr", {"study": frozen["sid"]})
    assert err.value.code == 404 and "fetch the field first" in str(err.value)


def test_fetched_cubes_serve_every_product(frozen):
    _fetch(frozen, "test-00001")
    params = {"study": frozen["sid"]}
    meta = viewer_data.get_meta("study", params)
    assert meta["objects"][0]["fetched"] is True
    hr, info = viewer_data.get_cube("study", 0, "hr", params)
    np.testing.assert_array_equal(hr, _truth(1))
    assert info["unit"] == "e-" and info["pixscale"] > 0
    gate, _ = viewer_data.get_cube("study", 0, "sr", params)
    np.testing.assert_array_equal(gate, np.load(frozen["cubes"] / "comb_spatial_gate_00001.npy"))
    with pytest.raises(ViewerError) as err:
        viewer_data.get_cube("study", 0, "member1", params)
    assert err.value.code == 404 and "fetch member 02 first" in str(err.value)
    _fetch(frozen, "test-00001", ["member_02"])
    meta = viewer_data.get_meta("study", params)
    assert meta["objects"][0]["fetched_members"] == [1]
    assert [t for t in meta["objects"][0]["tiers"] if t.startswith("member")] == ["member1"]
    member, info = viewer_data.get_cube("study", 0, "member1", params)
    np.testing.assert_array_equal(member, np.load(frozen["cubes"] / "member1_00001.npy"))
    assert "02·psnr" in info["label"]
    bhr, _ = viewer_data.get_cube("study", 0, "bhr", params)
    assert bhr.shape == hr.shape and not np.array_equal(bhr, hr)
    lr, info = viewer_data.get_cube("study", 0, "lr", params)
    assert lr.shape == (8, 8, 4)
    with pytest.raises(ViewerError):
        viewer_data.get_cube("study", 0, "mask", params)       # a natural field has no holes


def test_unknown_study_is_404():
    with pytest.raises(ViewerError) as err:
        viewer_data.get_meta("study", {"study": "20260101-000000-nope"})
    assert err.value.code == 404
