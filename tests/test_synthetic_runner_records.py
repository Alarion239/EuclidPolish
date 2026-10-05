"""Where the source-centred synthetic evaluation finds its records, and what it
says when they are missing (the held-out ``test`` split, else ``validate``)."""

from __future__ import annotations

import numpy as np
import pytest

from euclid_polish.config import Config
from euclid_polish.eval import synthetic_runner as sr


class _Img:
    def __init__(self, index, data):
        self.index = index
        self.data = data


@pytest.fixture
def no_records(tmp_path, monkeypatch):
    """Both record locations point at empty directories."""
    v2, synced = tmp_path / "v2", tmp_path / "synced"
    v2.mkdir()
    synced.mkdir()
    monkeypatch.setattr(Config, "RECORDS_DIR_V2", str(v2))
    monkeypatch.setattr(sr, "_sky_records_local_dir", lambda: str(synced))
    return v2, synced


@pytest.mark.parametrize("subset", ["test", "validate"])
def test_the_records_dir_is_found_by_either_eval_split(no_records, subset):
    _v2, synced = no_records
    (synced / f"dirty_{subset}.tfrecord").write_bytes(b"")
    assert sr.default_records_dir() == str(synced)


def test_missing_records_say_where_to_sync_them(no_records, tmp_path):
    assert sr.default_records_dir() is None
    with pytest.raises(FileNotFoundError) as err:
        sr.run_synthetic_eval(str(tmp_path / "out"), n=1, on_progress=lambda *a: None,
                              log=lambda *a: None)
    message = str(err.value)
    assert "dirty_test" in message and "dirty_validate" in message
    assert "Synthetic › Records" in message and "/inference" not in message


def test_no_matching_fields_names_the_split_it_read(tmp_path, monkeypatch):
    (tmp_path / "dirty_test.tfrecord").write_bytes(b"")          # the test split is present

    def fake_read(path, num_images=0):
        if "dirty" in str(path):
            return [_Img(0, np.zeros((64, 64, 4), np.float32))]
        return [_Img(1, np.zeros((128, 128, 4), np.float32))]   # no LR/HR index in common

    monkeypatch.setattr(sr, "read_images", fake_read)
    monkeypatch.setattr(sr, "read_sources", lambda path: {0: [], 1: []})
    logs: list[str] = []
    res = sr.run_synthetic_eval(str(tmp_path / "out"), n=1, records_dir=str(tmp_path),
                                on_progress=lambda *a: None, log=logs.append)
    assert res["rows"] == []
    assert "no test fields with matching LR/HR + source catalog." in logs
