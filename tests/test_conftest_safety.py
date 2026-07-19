"""Regression checks for the test suite's live-data write boundary."""

from __future__ import annotations

from pathlib import Path

import pytest

from euclid_polish.config import Config


def test_write_below_live_data_dir_is_rejected():
    target = Path(Config.DATA_DIR) / ".pytest-must-not-create"

    with pytest.raises(AssertionError, match="live data path"):
        target.write_bytes(b"must never reach disk")

    assert not target.exists()


def test_directory_creation_below_live_data_dir_is_rejected():
    target = Path(Config.DATA_DIR) / ".pytest-must-not-create-dir"

    with pytest.raises(AssertionError, match="live data path"):
        target.mkdir()

    assert not target.exists()


def test_tmp_path_remains_writable(tmp_path):
    target = tmp_path / "allowed.txt"
    target.write_text("ok")
    assert target.read_text() == "ok"
