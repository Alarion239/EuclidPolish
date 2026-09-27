"""ensemble_registry: label/name helpers, the archived-member table (zip
lookup across tracking campaigns) and restoring a tombstone to active."""
from __future__ import annotations

import json
import os
import shutil

import pytest

from euclid_polish import ensemble_registry as er


def _mk_member(base, i, *, starless=None):
    d = os.path.join(base, f"member_{i:02d}")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "checkpoint"), "w") as f:
        f.write("x")
    if starless is not None:
        with open(os.path.join(d, "origin.json"), "w") as f:
            json.dump({"starless": starless}, f)
    return d


@pytest.mark.parametrize("raw, name", [
    ("196·psnr", "member_196"), ("196", "member_196"), ("member_196", "member_196"),
    (" 07 ", "member_07"), ("member_07·psnr", "member_07"), ("2", "member_02"),
])
def test_member_name_normalises_every_spelling(raw, name):
    assert er.member_name(raw) == name
    assert er.member_label(raw) == name.removeprefix("member_") + "·psnr"


@pytest.mark.parametrize("bad", ["", "abc", "member_", "12a", "../member_1", "196·loss"])
def test_member_name_rejects_garbage(bad):
    with pytest.raises(ValueError):
        er.member_name(bad)


def test_member_is_starless_lives_in_the_registry(tmp_path):
    base = str(tmp_path / "ensemble")
    assert er.member_is_starless(_mk_member(base, 0)) is False
    assert er.member_is_starless(_mk_member(base, 1, starless=True)) is True
    # regime_labels no longer needs a function-scoped import of ensemble.py
    assert er.regime_labels(base, True) == ["01·psnr"]
    assert er.regime_labels(base, False) == ["00·psnr"]


def _tracking(tmp_path):
    root = tmp_path / "tracking"
    (root / "current" / "models").mkdir(parents=True)
    (root / "current" / "metadata.json").write_text("{}")
    (root / "archive" / "old-campaign" / "models").mkdir(parents=True)
    (root / "archive" / "old-campaign" / "metadata.json").write_text("{}")
    return root


def test_archived_members_find_zip_in_current_or_archived_campaigns(tmp_path):
    base = str(tmp_path / "ensemble")
    _mk_member(base, 0)
    _mk_member(base, 1)
    _mk_member(base, 2)
    er.load_registry(base)
    root = _tracking(tmp_path)
    (root / "current" / "models" / "ensemble-member-01.zip").write_bytes(b"zz")
    (root / "archive" / "old-campaign" / "models" / "ensemble-member-02.zip").write_bytes(b"zzz")
    er.archive_member_entry(base, "member_01", zip_path="models/ensemble-member-01.zip", commit="abc")
    er.archive_member_entry(base, "member_02", zip_path="models/ensemble-member-02.zip", commit=None)
    er.archive_member_entry(base, "member_00", zip_path="models/missing.zip", commit=None)

    rows = {r["name"]: r for r in er.archived_members(base, str(root))}
    assert rows["member_01"]["zip_found"] is True
    assert rows["member_01"]["campaign"] == "current"
    assert rows["member_01"]["size_bytes"] == 2
    assert rows["member_02"]["campaign"] == "old-campaign"
    assert rows["member_02"]["zip_path"].endswith("ensemble-member-02.zip")
    assert rows["member_00"]["zip_found"] is False and rows["member_00"]["zip_path"] is None
    assert rows["member_01"]["commit"] == "abc"


def test_find_archived_zip_never_leaves_the_tracking_root(tmp_path):
    root = _tracking(tmp_path)
    secret = tmp_path / "secret.zip"
    secret.write_bytes(b"x")
    assert er.find_archived_zip({"zip": "../../secret.zip"}, str(root)) is None
    assert er.find_archived_zip({"zip": str(secret)}, str(root)) is None
    assert er.find_archived_zip({}, str(root)) is None


def test_restore_member_entry_moves_the_tombstone_back_to_active(tmp_path):
    base = str(tmp_path / "ensemble")
    _mk_member(base, 0)
    _mk_member(base, 1)
    er.load_registry(base)
    er.archive_member_entry(base, "member_01", zip_path="models/z.zip", commit=None)
    assert er.load_registry(base)["active"] == ["member_00"]

    reg = er.restore_member_entry(base, "member_01")
    assert reg["active"] == ["member_00", "member_01"]
    assert all(t["name"] != "member_01" for t in reg["archived"])
    assert er.load_registry(base)["active"] == ["member_00", "member_01"]


def test_restore_member_entry_refusals(tmp_path):
    base = str(tmp_path / "ensemble")
    _mk_member(base, 0)
    er.load_registry(base)
    with pytest.raises(ValueError, match="already active"):
        er.restore_member_entry(base, "member_00")
    with pytest.raises(ValueError, match="not archived"):
        er.restore_member_entry(base, "member_05")
    # the member dir must be back on disk first (else the bootstrap drops it)
    _mk_member(base, 3)
    er.load_registry(base)
    er.archive_member_entry(base, "member_03", zip_path="models/z.zip", commit=None)
    shutil.rmtree(os.path.join(base, "member_03"))
    with pytest.raises(ValueError, match="no checkpoint"):
        er.restore_member_entry(base, "member_03")
