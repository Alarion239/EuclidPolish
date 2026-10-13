"""ensemble_registry: bootstrap, tombstones, labels, ensemble wiring, and the
universal (standalone model) class."""
import json
import os
import shutil
import threading
import time

import pytest

from euclid_polish import ensemble as ens
from euclid_polish import ensemble_registry as er
from euclid_polish.config import Config


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
    # PSNR-best only: loss_best/ stays on disk (fork source) but is not an
    # ensemble model, so it never appears in the fingerprint.
    assert er.active_labels(base) == ["00·psnr", "02·psnr"]


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
    base = str(tmp_path / "ensemble")
    _mk_member(base, 0)
    assert ens.ensemble_available(base) is True
    er.load_registry(base)
    er.archive_member_entry(base, "member_00", zip_path="z", commit=None)
    assert ens.ensemble_available(base) is False   # dir on disk, but archived


def test_default_ensemble_dir_is_ckpt_sibling(monkeypatch, tmp_path):
    monkeypatch.setattr(Config, "DEFAULT_CHECKPOINT_DIR",
                        str(tmp_path / "ckpt/wdsr"))
    assert ens.default_ensemble_dir() == str(tmp_path / "ckpt/ensemble")


# ---- universal (standalone) models ----------------------------------------- #

def _read_file(base):
    with open(er.registry_path(base)) as f:
        return json.load(f)


@pytest.mark.parametrize("origin, expected", [
    # fixed-knee single-image multi-knee members (195/196)
    ({"asinh_knees": [0.1, 1, 10], "output_knee": 10.0}, True),
    # learned-knee members (206+): a dict, NO output_knee key
    ({"asinh_knees": [0.1, 1, 10],
      "learned_output_knee": {"init_e": 10.0, "min_e": 0.1, "max_e": 1e4}}, True),
    # the FASRC ledger member_spec spelling of a learned knee
    ({"asinh_knees": [0.1, 1, 10], "learn_output_knee": True}, True),
    ({"asinh_knees": [0.1, 1, 10], "learn_output_knee": True, "output_knee": 10}, True),
    # six-head multi-knee members (197/198): one image per knee → not universal
    ({"asinh_knees": [0.1, 1, 10]}, False),
    ({"asinh_knees": [0.1, 1, 10], "output_knee": None, "learned_output_knee": None}, False),
    ({"asinh_knees": [0.1, 1, 10], "learned_output_knee": {}}, False),
    ({"asinh_knees": [0.1, 1, 10], "learn_output_knee": False}, False),
    # single-knee members, even with an output knee recorded
    ({"asinh_knee": 100.0}, False),
    ({"asinh_knees": [], "output_knee": 10.0}, False),
    ({"asinh_knees": None, "output_knee": 10.0}, False),
    ({"asinh_knees": "0.1,1", "output_knee": 10.0}, False),
    ({}, False),
    (None, False),
    ([], False),
])
def test_is_universal_origin(origin, expected):
    assert er.is_universal_origin(origin) is expected


def test_old_registry_reads_universal_as_empty_and_is_not_rewritten(tmp_path):
    base = str(tmp_path / "ensemble")
    _mk_member(base, 0)
    er.load_registry(base)
    before = _read_file(base)
    assert before == {"active": ["member_00"], "archived": [], "universal": []}
    # a registry written before the universal class existed
    with open(er.registry_path(base), "w") as f:
        json.dump({"active": ["member_00"], "archived": []}, f)
    mtime = os.path.getmtime(er.registry_path(base))
    reg = er.load_registry(base)
    assert reg == {"active": ["member_00"], "archived": [], "universal": []}
    # nothing changed → nothing written (the key appears on the next change)
    assert _read_file(base) == {"active": ["member_00"], "archived": []}
    assert os.path.getmtime(er.registry_path(base)) == mtime
    _mk_member(base, 1)                       # bootstrap → write all three keys
    assert _read_file(base) == {"active": ["member_00"], "archived": []}
    er.load_registry(base)
    assert _read_file(base) == {"active": ["member_00", "member_01"],
                                "archived": [], "universal": []}


def test_universal_dir_is_never_bootstrapped_active(tmp_path):
    base = str(tmp_path / "ensemble")
    _mk_member(base, 0)
    assert er.add_universal_entry(base, "211·psnr") == "member_211"
    _mk_member(base, 211)                     # the pull's rsync lands after registering
    reg = er.load_registry(base)
    assert reg["active"] == ["member_00"]
    assert reg["universal"] == ["member_211"]
    assert er.universal_names(base) == ["member_211"]
    assert [os.path.basename(d) for d in er.universal_member_dirs(base)] == ["member_211"]
    assert er.universal_labels(base) == ["211·psnr"]
    # the ensemble views never see it
    assert er.active_labels(base) == ["00·psnr"]
    assert er.regime_labels(base, False) == ["00·psnr"]
    assert [os.path.basename(d) for d in er.active_member_dirs(base)] == ["member_00"]


def test_universal_entry_survives_a_missing_dir(tmp_path):
    base = str(tmp_path / "ensemble")
    _mk_member(base, 0)
    er.add_universal_entry(base, "member_212")
    # mid-pull: registered, no dir yet → kept, but not loadable
    assert er.load_registry(base)["universal"] == ["member_212"]
    assert er.universal_names(base) == ["member_212"]
    assert er.universal_member_dirs(base) == []
    assert er.universal_labels(base) == []
    # a dir without a checkpoint is not loadable either
    os.makedirs(os.path.join(base, "member_212"))
    assert er.universal_member_dirs(base) == []
    _mk_member(base, 212)
    assert er.universal_labels(base) == ["212·psnr"]
    # the mirror deleted it again → the entry persists
    shutil.rmtree(os.path.join(base, "member_212"))
    assert er.load_registry(base)["universal"] == ["member_212"]
    assert _read_file(base)["universal"] == ["member_212"]


def test_every_write_carries_the_universal_key(tmp_path):
    base = str(tmp_path / "ensemble")
    _mk_member(base, 0)
    _mk_member(base, 1)
    er.add_universal_entry(base, "member_211")
    _mk_member(base, 2)                       # bootstrap write
    er.load_registry(base)
    assert _read_file(base)["universal"] == ["member_211"]
    shutil.rmtree(os.path.join(base, "member_02"))   # drop-vanished write
    er.load_registry(base)
    assert _read_file(base)["universal"] == ["member_211"]
    er.archive_member_entry(base, "member_01", zip_path="z", commit=None)
    assert _read_file(base)["universal"] == ["member_211"]
    er.restore_member_entry(base, "member_01")
    assert _read_file(base)["universal"] == ["member_211"]
    assert _read_file(base)["active"] == ["member_00", "member_01"]


def test_add_universal_entry_normalises_is_idempotent_and_refuses(tmp_path):
    base = str(tmp_path / "ensemble")
    _mk_member(base, 0)
    _mk_member(base, 3)
    er.load_registry(base)
    er.archive_member_entry(base, "member_03", zip_path="z", commit=None)
    assert er.add_universal_entry(base, "211") == "member_211"
    mtime = os.path.getmtime(er.registry_path(base))
    assert er.add_universal_entry(base, "member_211") == "member_211"
    assert er.add_universal_entry(base, "211·psnr") == "member_211"
    assert er.universal_names(base) == ["member_211"]
    assert os.path.getmtime(er.registry_path(base)) == mtime      # no-op: no write
    assert er.add_universal_entry(base, "9") == "member_09"
    assert er.universal_names(base) == ["member_09", "member_211"]
    with pytest.raises(ValueError, match="active ensemble member"):
        er.add_universal_entry(base, "00·psnr")
    with pytest.raises(ValueError, match="archived"):
        er.add_universal_entry(base, "member_03")
    for bad in ("", "abc", "../member_1", "196·loss"):
        with pytest.raises(ValueError):
            er.add_universal_entry(base, bad)
    reg = er.load_registry(base)
    assert reg["active"] == ["member_00"]
    assert reg["universal"] == ["member_09", "member_211"]
    assert [t["name"] for t in reg["archived"]] == ["member_03"]


def test_remove_universal_entry_unregisters_only_universal_names(tmp_path):
    """The rollback of a failed universal pull: it removes a registered
    universal name and refuses anything else (an active member, a tombstone,
    an unknown or malformed name), leaving the registry as it was."""
    base = str(tmp_path / "ensemble")
    _mk_member(base, 0)
    _mk_member(base, 3)
    er.load_registry(base)
    er.archive_member_entry(base, "member_03", zip_path="z", commit=None)
    er.add_universal_entry(base, "211")
    er.add_universal_entry(base, "212")
    assert er.remove_universal_entry(base, "211·psnr") == "member_211"
    assert er.universal_names(base) == ["member_212"]
    assert _read_file(base)["universal"] == ["member_212"]
    for name, needle in (("member_00", "not a registered universal"),
                         ("member_03", "not a registered universal"),
                         ("member_211", "not a registered universal"),
                         ("bad!", "not a member name")):
        with pytest.raises(ValueError, match=needle):
            er.remove_universal_entry(base, name)
    reg = er.load_registry(base)
    assert reg["active"] == ["member_00"] and reg["universal"] == ["member_212"]
    assert [t["name"] for t in reg["archived"]] == ["member_03"]


def test_a_name_holds_one_class_even_in_a_hand_edited_file(tmp_path):
    base = str(tmp_path / "ensemble")
    _mk_member(base, 0)
    _mk_member(base, 5)
    with open(er.registry_path(base), "w") as f:
        json.dump({"active": ["member_00"],
                   "archived": [{"name": "member_05"}],
                   "universal": ["member_05", "member_00", "member_211"]}, f)
    reg = er.load_registry(base)
    # tombstones are permanent and an ensemble member outranks a registration
    assert reg["active"] == ["member_00"]
    assert reg["universal"] == ["member_211"]
    assert _read_file(base)["universal"] == ["member_211"]


def test_next_member_names_counts_universal_names(tmp_path):
    base = str(tmp_path / "ensemble")
    _mk_member(base, 0)
    er.add_universal_entry(base, "member_218")      # registered, not pulled yet
    assert er.next_member_names(base, 2) == ["member_219", "member_220"]


def test_archive_records_the_class_on_the_tombstone(tmp_path):
    base = str(tmp_path / "ensemble")
    _mk_member(base, 0)
    _mk_member(base, 1)
    er.add_universal_entry(base, "member_211")
    _mk_member(base, 211)
    er.archive_member_entry(base, "member_01", zip_path="a.zip", commit="abc")
    reg = er.archive_member_entry(base, "member_211", zip_path="b.zip", commit=None)
    assert reg["active"] == ["member_00"] and reg["universal"] == []
    tombs = {t["name"]: t for t in er.load_registry(base)["archived"]}
    assert tombs["member_01"]["class"] == "member"
    assert tombs["member_211"]["class"] == "universal"
    assert tombs["member_211"]["zip"] == "b.zip"
    assert er.tombstone_class(tombs["member_211"]) == "universal"
    assert er.tombstone_class(tombs["member_01"]) == "member"
    assert er.tombstone_class({"name": "member_02"}) == "member"   # pre-class tombstone
    # a tombstoned universal dir reappearing on disk is never activated either
    assert er.load_registry(base)["active"] == ["member_00"]
    assert er.next_member_names(base, 1) == ["member_212"]


def test_archive_refuses_names_of_no_class(tmp_path):
    base = str(tmp_path / "ensemble")
    _mk_member(base, 0)
    er.load_registry(base)
    with pytest.raises(ValueError, match="universal model"):
        er.archive_member_entry(base, "member_09", zip_path="z", commit=None)


def test_registry_writes_are_serialised_across_threads(tmp_path, monkeypatch):
    base = str(tmp_path / "ensemble")
    _mk_member(base, 0)
    er.load_registry(base)
    real_read = er._read_json

    def slow_read(path):                      # widen the read → write window
        out = real_read(path)
        time.sleep(0.005)
        return out

    monkeypatch.setattr(er, "_read_json", slow_read)
    names = [f"member_{300 + i}" for i in range(16)]
    errors = []

    def add(n):
        try:
            er.add_universal_entry(base, n)
        except Exception as e:  # noqa: BLE001 — surfaced by the assert below
            errors.append(e)

    threads = [threading.Thread(target=add, args=(n,)) for n in names]
    threads += [threading.Thread(target=er.load_registry, args=(base,)) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert er.universal_names(base) == sorted(names)        # no lost update
    assert er.load_registry(base)["active"] == ["member_00"]
