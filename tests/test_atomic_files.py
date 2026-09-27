"""Atomic writes with collision-free temp names (``helpers/atomic_files``).

Web jobs are threads of one process: a ``.<name>.<pid>.tmp`` temp is shared
by two concurrent jobs of the same kind, so one job could replace the other's
half-written file. mkstemp names never collide."""

from __future__ import annotations

import json
import os

import pytest

from euclid_polish.web.helpers import atomic_files, fs_stamp


def test_two_temps_for_one_destination_never_collide(tmp_path):
    target = tmp_path / "lr.fits"
    with atomic_files.temporary_sibling(target, ".fits") as first, \
            atomic_files.temporary_sibling(target, ".fits") as second:
        assert first != second
        assert first.parent == second.parent == tmp_path
        assert first.name.startswith(".lr.fits.") and first.suffix == ".fits"
        assert str(os.getpid()) not in first.name.split(".")
    assert list(tmp_path.iterdir()) == []                  # unused temps are removed


def test_a_failed_write_leaves_neither_temp_nor_target(tmp_path):
    target = tmp_path / "manifest.json"
    with pytest.raises(RuntimeError), atomic_files.temporary_sibling(target) as temporary:
        temporary.write_text("{half", encoding="utf-8")
        raise RuntimeError("disk full")
    assert list(tmp_path.iterdir()) == []


def test_write_json_replaces_atomically(tmp_path):
    target = tmp_path / "sub" / "record.json"
    atomic_files.write_json(target, {"a": 1}, indent=2)
    atomic_files.write_json(target, {"a": 2}, indent=2)
    assert json.loads(target.read_text(encoding="utf-8")) == {"a": 2}
    assert [path.name for path in target.parent.iterdir()] == ["record.json"]


def test_tree_stamp_sees_nested_writes_but_not_deeper(tmp_path):
    (tmp_path / "field" / "tiles").mkdir(parents=True)
    before = fs_stamp.tree_stamp(tmp_path, 2)
    assert fs_stamp.tree_stamp(tmp_path, 2) == before
    (tmp_path / "field" / "tiles" / "sr.fits").write_bytes(b"x")    # moves tiles/ mtime
    after = fs_stamp.tree_stamp(tmp_path, 2)
    assert after != before
    assert fs_stamp.stat_key(tmp_path / "missing")[1:] == (None, None)


def _default_mode() -> int:
    umask = os.umask(0)
    os.umask(umask)
    return 0o666 & ~umask


def test_replaced_files_keep_the_umask_default_mode_not_mkstemp_0600(tmp_path):
    # mkstemp creates 0600 and os.replace keeps that inode: without a chmod
    # every FITS/sidecar/manifest would become owner-only (the tracking
    # mirror copies permissions to the group share).
    target = tmp_path / "record.json"
    atomic_files.write_json(target, {"a": 1})
    assert target.stat().st_mode & 0o777 == _default_mode()
    fits_target = tmp_path / "lr.fits"
    with atomic_files.temporary_sibling(fits_target, ".fits") as temporary:
        temporary.write_bytes(b"SIMPLE")
        os.replace(temporary, fits_target)
    assert fits_target.stat().st_mode & 0o777 == _default_mode()
