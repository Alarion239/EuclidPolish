"""ensemble_registry: bootstrap, tombstones, labels, ensemble wiring."""
import os
import shutil

import pytest

from euclid_polish import ensemble_registry as er


def _mk_member(base, i, *, loss_best=False):
    d = os.path.join(base, f"member_{i:02d}")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "checkpoint"), "w") as f:
        f.write("x")
    if loss_best:
        lb = os.path.join(d, "loss_best")
        os.makedirs(lb, exist_ok=True)
        with open(os.path.join(lb, "checkpoint"), "w") as f:
            f.write("x")
    return d


def test_bootstrap_discovers_members_and_persists(tmp_path):
    base = str(tmp_path / "ensemble")
    _mk_member(base, 0)
    _mk_member(base, 1, loss_best=True)
    reg = er.load_registry(base)
    assert reg["active"] == ["member_00", "member_01"]
    assert reg["archived"] == []
    # persisted OUTSIDE the ensemble dir (mirror --delete-after safety)
    assert os.path.isfile(er.registry_path(base))
    assert not er.registry_path(base).startswith(base + os.sep)


def test_missing_dir_dropped_archived_never_reactivated(tmp_path):
    base = str(tmp_path / "ensemble")
    _mk_member(base, 0)
    _mk_member(base, 1)
    er.load_registry(base)
    er.archive_member_entry(base, "member_01", zip_path="z.zip", commit="abc")
    reg = er.load_registry(base)      # dir still on disk (mirror pulled it back)
    assert reg["active"] == ["member_00"]
    assert reg["archived"][0]["name"] == "member_01"
    # a vanished active member is dropped from active on load
    shutil.rmtree(os.path.join(base, "member_00"))
    assert er.load_registry(base)["active"] == []


def test_active_member_dirs_and_labels(tmp_path):
    base = str(tmp_path / "ensemble")
    _mk_member(base, 0, loss_best=True)
    _mk_member(base, 2)
    dirs = er.active_member_dirs(base)
    assert [os.path.basename(d) for d in dirs] == ["member_00", "member_02"]
    assert er.active_labels(base) == ["00·psnr", "00·loss", "02·psnr"]


def test_archive_unknown_member_raises(tmp_path):
    base = str(tmp_path / "ensemble")
    _mk_member(base, 0)
    er.load_registry(base)
    with pytest.raises(ValueError):
        er.archive_member_entry(base, "member_09", zip_path="z", commit=None)


def test_next_member_names_skips_tombstones_and_gaps(tmp_path):
    base = str(tmp_path / "ensemble")
    _mk_member(base, 0)
    _mk_member(base, 2)                       # gap at 1 stays a gap
    er.load_registry(base)
    er.archive_member_entry(base, "member_02", zip_path="z", commit=None)
    # active: member_00; archived: member_02 → next index is 3, never 1 or 2
    assert er.next_member_names(base, 2) == ["member_03", "member_04"]


def test_next_member_names_counts_unregistered_disk_dirs(tmp_path):
    base = str(tmp_path / "ensemble")
    _mk_member(base, 5)                       # on disk, not yet in registry
    assert er.next_member_names(base, 1) == ["member_06"]


def test_next_member_names_empty_ensemble(tmp_path):
    base = str(tmp_path / "ensemble")
    assert er.next_member_names(base, 2) == ["member_00", "member_01"]


def test_ensemble_available_respects_registry(tmp_path):
    from euclid_polish import ensemble as ens
    base = str(tmp_path / "ensemble")
    _mk_member(base, 0)
    assert ens.ensemble_available(base) is True
    er.load_registry(base)
    er.archive_member_entry(base, "member_00", zip_path="z", commit=None)
    assert ens.ensemble_available(base) is False   # dir on disk, but archived


def test_default_ensemble_dir_is_ckpt_sibling(monkeypatch, tmp_path):
    from euclid_polish import ensemble as ens
    from euclid_polish.config import Config
    monkeypatch.setattr(Config, "DEFAULT_CHECKPOINT_DIR",
                        str(tmp_path / "ckpt/wdsr"))
    assert ens.default_ensemble_dir() == str(tmp_path / "ckpt/ensemble")
