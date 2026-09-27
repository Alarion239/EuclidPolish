"""Data › Records helpers (``web/helpers/sky_records.py``): O(1) TFRecord
access through a header-scanned offset index, the truth-source sidecar of a
record (``sources_<subset>.csv``) and the SR tier's model identity/staleness.

Everything here is local: small synthetic TFRecords and CSVs in ``tmp_path``.
"""

from __future__ import annotations

import csv
import json
import os
import struct
import time

import numpy as np
import pytest

from euclid_polish.image import Image, Role
from euclid_polish.image.tfio import tfrecord_path, write_images
from euclid_polish.web.helpers import sky_records


def _images(n: int, size: int = 8, channels: int = 4, seed: int = 0) -> list[Image]:
    rng = np.random.default_rng(seed)
    return [
        Image(data=rng.normal(size=(size, size, channels)).astype(np.float32) + i,
              pixel_scale_arcsec=0.05, band_names=("VIS", "Y_E", "J_E", "H_E")[:channels],
              is_clean=True, role=Role.HR, index=i)
        for i in range(n)
    ]


def _write(tmp_path, name: str, n: int, size: int = 8, seed: int = 0) -> str:
    return write_images(_images(n, size, seed=seed), name, records_dir=str(tmp_path))


# ---------------------------------------------------------------------------
# TFRecord offset index
# ---------------------------------------------------------------------------

def test_masked_crc_matches_the_tensorflow_writer(tmp_path):
    path = _write(tmp_path, "hr_test", 1)
    with open(path, "rb") as handle:
        length, crc = struct.unpack("<QI", handle.read(12))
    assert sky_records.masked_crc32c(struct.pack("<Q", length)) == crc


def test_offsets_count_and_random_access(tmp_path):
    path = _write(tmp_path, "hr_test", 5)
    offsets, complete = sky_records.tfrecord_offsets(path)
    assert complete and len(offsets) == 5
    assert sky_records.record_count(path) == 5
    rec = sky_records.read_record(path, 3)
    assert rec.index == 3
    np.testing.assert_allclose(rec.data, _images(5)[3].data)
    with pytest.raises(IndexError):
        sky_records.read_record(path, 5)


def test_record_count_absent_garbage_and_truncated(tmp_path):
    assert sky_records.record_count(str(tmp_path / "nope.tfrecord")) == 0
    bad = tmp_path / "garbage.tfrecord"
    bad.write_bytes(b"\x00" * 1024 + b"not a real record" + b"\xff" * 1024)
    assert sky_records.record_count(str(bad)) is None

    path = _write(tmp_path, "dirty_test", 3)
    size = os.path.getsize(path)
    with open(path, "r+b") as handle:
        handle.truncate(size - 10)          # an interrupted rsync
    assert sky_records.record_count(path) is None
    # The intact records before the cut stay readable.
    assert sky_records.read_record(path, 1).index == 1


def test_offsets_are_cached_per_file_state(tmp_path, monkeypatch):
    path = _write(tmp_path, "hr_test", 2)
    sky_records.tfrecord_offsets(path)
    calls = []
    real = sky_records._scan_offsets
    monkeypatch.setattr(sky_records, "_scan_offsets", lambda p: calls.append(p) or real(p))
    sky_records.tfrecord_offsets(path)
    assert calls == []                              # unchanged file: cached
    _write(tmp_path, "hr_test", 3)                  # rewritten
    assert len(sky_records.tfrecord_offsets(path)[0]) == 3
    assert calls == [path]


# ---------------------------------------------------------------------------
# inventory
# ---------------------------------------------------------------------------

def _sources_csv(tmp_path, subset: str, rows: list[dict]) -> str:
    path = os.path.join(str(tmp_path), f"sources_{subset}.csv")
    cols = ["field_index", "type", "render", "x_pix", "y_pix", "off_field", "flux_vis_e",
            "flux_y_e", "flux_j_e", "flux_h_e", "z", "subhalo_id", "orientation",
            "theta_E_arcsec", "re_arcsec", "mag_vis", "mag_y_e", "mag_j_e", "mag_h_e",
            "target_vis_mag", "achieved_vis_2fwhm_mag", "tng_render_trace"]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in cols})
    return path


def test_inventory_lists_every_kind_per_subset(tmp_path):
    _write(tmp_path, "dirty_test", 2, size=4)
    _write(tmp_path, "hr_test", 2)
    _write(tmp_path, "clean_train", 3)
    _sources_csv(tmp_path, "test", [{"field_index": 0, "type": "star", "x_pix": 1, "y_pix": 2}])
    inv = sky_records.records_inventory(str(tmp_path))
    test = inv["test"]
    assert test["files"]["dirty"]["count"] == 2
    assert test["files"]["hr"]["count"] == 2
    assert test["files"]["clean"] is None
    assert test["files"]["sources"]["size_bytes"] > 0
    assert test["count"] == 2
    assert inv["train"]["files"]["clean"]["count"] == 3
    assert inv["train"]["count"] == 3
    assert inv["validate"]["count"] == 0
    assert inv["validate"]["present"] is False


# ---------------------------------------------------------------------------
# truth sources
# ---------------------------------------------------------------------------

ROWS = [
    {"field_index": 0, "type": "galaxy", "render": "tng", "x_pix": 10.5, "y_pix": 20.25,
     "off_field": 0, "flux_vis_e": 900.0, "flux_y_e": 180, "re_arcsec": 0.2,
     "subhalo_id": 658592, "orientation": 5, "achieved_vis_2fwhm_mag": 27.1,
     "tng_render_trace": '{"a":1}'},
    {"field_index": 0, "type": "star", "x_pix": 3, "y_pix": 4, "flux_vis_e": 5e5,
     "mag_vis": 18.2, "mag_y_e": 18.0, "mag_j_e": 17.9, "mag_h_e": 17.8},
    {"field_index": 1, "type": "lens", "render": "tng", "x_pix": 100, "y_pix": 90,
     "theta_E_arcsec": 1.2, "flux_vis_e": 1e4, "off_field": 1},
]


def test_record_sources_for_one_field(tmp_path):
    _sources_csv(tmp_path, "test", ROWS)
    out = sky_records.record_sources(str(tmp_path), "test", 0)
    assert out["field_index"] == 0
    assert [s["type"] for s in out["sources"]] == ["galaxy", "star"]
    galaxy, star = out["sources"]
    assert galaxy["row"] == 0 and star["row"] == 1
    assert galaxy["x_pix"] == 10.5 and galaxy["y_pix"] == 20.25
    assert galaxy["subhalo_id"] == "658592"
    assert galaxy["mag_vis"] == pytest.approx(27.1)         # the achieved 2FWHM magnitude
    assert star["mag_vis"] == pytest.approx(18.2)
    assert out["counts"] == {"galaxy": 1, "star": 1, "lens": 0, "other": 0, "off_field": 0}
    other = sky_records.record_sources(str(tmp_path), "test", 1)
    assert other["sources"][0]["off_field"] is True
    assert other["sources"][0]["theta_E_arcsec"] == 1.2
    assert sky_records.record_sources(str(tmp_path), "test", 7)["sources"] == []


def test_record_source_detail_has_every_column(tmp_path):
    _sources_csv(tmp_path, "test", ROWS)
    row = sky_records.record_source_detail(str(tmp_path), "test", 0, 0)
    assert row["values"]["render"] == "tng"
    assert row["values"]["tng_render_trace"] == {"a": 1}      # JSON column parsed
    assert row["values"]["theta_E_arcsec"] is None             # empty → null
    with pytest.raises(KeyError):
        sky_records.record_source_detail(str(tmp_path), "test", 0, 9)


def test_sources_summary_per_field(tmp_path):
    _sources_csv(tmp_path, "test", ROWS)
    summary = sky_records.sources_summary(str(tmp_path), "test")
    by_field = {f["field_index"]: f for f in summary["fields"]}
    assert by_field[0]["galaxy"] == 1 and by_field[0]["star"] == 1
    assert by_field[0]["brightest_star_mag"] == pytest.approx(18.2)
    assert by_field[1]["lens"] == 1 and by_field[1]["off_field"] == 1
    assert summary["present"] is True
    assert sky_records.sources_summary(str(tmp_path), "validate")["present"] is False


def test_sources_missing_file_is_empty_not_an_error(tmp_path):
    out = sky_records.record_sources(str(tmp_path), "validate", 0)
    assert out["present"] is False and out["sources"] == []


# ---------------------------------------------------------------------------
# SR identity + staleness
# ---------------------------------------------------------------------------

IDENTITY = {"member_labels": ["01·psnr", "02·psnr"], "combiner_kind": "spatial_gate",
            "combiner_fingerprint": "abc"}


@pytest.fixture
def sr_dir(tmp_path, monkeypatch):
    target = tmp_path / "sky_sr"
    monkeypatch.setattr(sky_records, "sky_sr_dir", lambda: str(target))
    return target


def _make_sr(subset: str, n: int) -> None:
    os.makedirs(sky_records.sky_sr_dir(), exist_ok=True)
    for i in range(n):
        np.save(sky_records.sr_path(subset, i), np.zeros((2, 2, 4), np.float32))


def test_sr_state_current_stale_missing_partial(tmp_path, sr_dir):
    records = _write(tmp_path, "dirty_test", 2, size=4)
    assert sky_records.sr_state("test", IDENTITY, records)["state"] == "missing"

    _make_sr("test", 2)
    sky_records.write_sr_manifest("test", IDENTITY, records, count=2, model_label="gate")
    state = sky_records.sr_state("test", IDENTITY, records)
    assert state["state"] == "current" and state["count"] == 2 and state["records_count"] == 2
    assert state["manifest"]["model_label"] == "gate"

    newer = {**IDENTITY, "member_labels": ["01·psnr", "03·psnr"]}
    stale = sky_records.sr_state("test", newer, records)
    assert stale["state"] == "stale"
    assert any("members" in r for r in stale["reasons"])

    time.sleep(0.01)
    _write(tmp_path, "dirty_test", 2, size=4, seed=1)          # regenerated records (same size)
    assert any("records" in r for r in sky_records.sr_state("test", IDENTITY, records)["reasons"])


def test_sr_state_ignores_a_resync_that_only_touches_the_mtime(tmp_path, sr_dir):
    """Every FASRC pull stamps os.utime on the local file: that alone must not
    mark the SR stale (it would push an unneeded TF regeneration)."""
    records = _write(tmp_path, "dirty_test", 2, size=4)
    _make_sr("test", 2)
    sky_records.write_sr_manifest("test", IDENTITY, records, count=2)
    st = os.stat(records)
    os.utime(records, ns=(st.st_atime_ns + 5_000_000_000, st.st_mtime_ns + 5_000_000_000))
    assert sky_records.sr_state("test", IDENTITY, records)["state"] == "current"
    _write(tmp_path, "dirty_test", 2, size=4)                  # a byte-identical re-pull
    assert sky_records.sr_state("test", IDENTITY, records)["state"] == "current"


def test_sr_state_legacy_manifest_without_fingerprint_compares_size_only(tmp_path, sr_dir):
    records = _write(tmp_path, "dirty_test", 2, size=4)
    _make_sr("test", 2)
    manifest = sky_records.write_sr_manifest("test", IDENTITY, records, count=2)
    legacy = {**manifest, "records": {k: v for k, v in manifest["records"].items() if k != "fingerprint"}}
    legacy["records"]["mtime_ns"] -= 10**9
    with open(sky_records.sr_manifest_path("test"), "w") as handle:
        json.dump(legacy, handle)
    assert sky_records.sr_state("test", IDENTITY, records)["state"] == "current"
    _write(tmp_path, "dirty_test", 3, size=4)                  # different size
    assert sky_records.sr_state("test", IDENTITY, records)["state"] == "stale"


def test_sr_state_without_manifest_is_unknown(tmp_path, sr_dir):
    records = _write(tmp_path, "dirty_test", 3, size=4)
    _make_sr("test", 3)
    state = sky_records.sr_state("test", IDENTITY, records)
    assert state["state"] == "unknown"                          # legacy SR: no identity recorded
    _make_sr("test", 1)
    sky_records.write_sr_manifest("test", IDENTITY, records, count=1)
    os.remove(sky_records.sr_path("test", 2))
    assert sky_records.sr_state("test", IDENTITY, records)["state"] == "partial"


def test_clear_sr_removes_only_that_subset(sr_dir):
    _make_sr("test", 2)
    _make_sr("validate", 1)
    sky_records.write_sr_manifest("test", IDENTITY, None, count=2)
    assert sky_records.clear_sr("test") == 2
    assert sky_records.sr_count("test") == 0
    assert sky_records.read_sr_manifest("test") is None
    assert sky_records.sr_count("validate") == 1


def test_tfrecord_path_helper_is_shared():
    assert tfrecord_path("/x", "dirty_test").endswith("dirty_test.tfrecord")
