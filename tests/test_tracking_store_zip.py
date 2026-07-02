"""TrackingStore.archive_model_zip: full-tree zip + meta sidecar."""
import os
import zipfile

import pytest

from euclid_polish.tracking.store import TrackingError, TrackingStore


def _store(tmp_path):
    s = TrackingStore(str(tmp_path / "tracking"))
    s.create_campaign("zip test")
    return s


def test_archive_model_zip_roundtrip(tmp_path):
    src = tmp_path / "member_00"
    (src / "loss_best").mkdir(parents=True)
    (src / "checkpoint").write_text("root")
    (src / "ckpt-5.index").write_text("idx")
    (src / "loss_best" / "checkpoint").write_text("lb")
    s = _store(tmp_path)
    meta = s.archive_model_zip(str(src), "ensemble-member_00", comment="bye")
    zpath = os.path.join(s.current_dir, "models", meta["name"])
    assert meta["name"].endswith(".zip") and os.path.isfile(zpath)
    with zipfile.ZipFile(zpath) as z:
        names = set(z.namelist())
    assert {"checkpoint", "ckpt-5.index",
            os.path.join("loss_best", "checkpoint")} <= names
    assert meta["kind"] == "model-zip" and meta["size_bytes"] > 0
    # listed alongside dir backups
    models = s.list_backups()["models"]
    assert any(m["name"] == meta["name"] for m in models)


def test_archive_model_zip_missing_src(tmp_path):
    s = _store(tmp_path)
    with pytest.raises(TrackingError):
        s.archive_model_zip(str(tmp_path / "nope"), "x")


def test_archive_model_zip_empty_src(tmp_path):
    src = tmp_path / "empty"
    src.mkdir()
    s = _store(tmp_path)
    with pytest.raises(TrackingError):
        s.archive_model_zip(str(src), "x")
