"""purge_stale_bucket deletes exactly what a bucket's next writer would
discard: departed and continued members (with the aggregates their stack
made), unproven positional cubes, buckets made from other records, files of
unlisted members and — blackout buckets only — fields outside the manifest."""
from __future__ import annotations

import json
import os

import numpy as np

from euclid_polish.eval.ensemble_cube_cache import (
    BLACKOUT_INDEX,
    VIZ_INDEX,
    _remove_quietly,
    member_cube_path,
    purge_stale_bucket,
    write_bucket_manifest,
)

A, B = "01·psnr", "02·psnr"
FPS = {A: "ckpt-1:a", B: "ckpt-1:b"}
AGGREGATES = ("sr", "std", "pca0", "comb_spatial_gate")


def _bucket(tmp_path, *, labels=(A, B), fps=None, indices=(0, 1), name=VIZ_INDEX,
            records_fp="rec-1", positional=False):
    d = tmp_path / "cubes"
    d.mkdir()
    for position, label in enumerate(labels):
        for rec in indices:
            path = (d / f"member{position}_{rec:05d}.npy" if positional
                    else member_cube_path(str(d), label, rec))
            np.save(path, np.full(4, position, np.float32))
    for rec in indices:
        np.save(d / f"lr_{rec:05d}.npy", np.zeros(4, np.float32))
        for prefix in AGGREGATES:
            np.save(d / f"{prefix}_{rec:05d}.npy", np.zeros(4, np.float32))
    manifest = {"member_labels": list(labels), "indices": list(indices),
                "pca_amps": {"0": [1.0]}, "pca_var": {"0": [0.5]},
                "has_combiner_spatial_gate": True}
    if not positional:
        manifest["member_fps"] = dict(FPS if fps is None else fps)
    if name == VIZ_INDEX:
        manifest["records_fp"] = records_fp
    else:
        manifest["identity"] = {"seed": 0, "source": records_fp}
    write_bucket_manifest(str(d), manifest, name)
    return d


def _manifest(d, name=VIZ_INDEX):
    return json.loads((d / name).read_text())


def test_a_current_bucket_is_left_alone(tmp_path):
    d = _bucket(tmp_path)
    before = sorted(os.listdir(d))

    result = purge_stale_bucket(str(d), current=FPS, records_fp="rec-1")

    assert result.dropped == [] and result.wiped is None
    assert result.bytes_freed == 0 and result.files_deleted == 0
    assert sorted(os.listdir(d)) == before


def test_a_departed_member_loses_its_cubes_and_the_aggregates(tmp_path):
    d = _bucket(tmp_path)

    result = purge_stale_bucket(str(d), current={A: FPS[A]}, records_fp="rec-1")

    assert result.dropped == [B]
    assert not any(n.startswith("member_02_") for n in os.listdir(d))
    assert all((d / f"member_01_{rec:05d}.npy").is_file() for rec in (0, 1))
    assert all((d / f"lr_{rec:05d}.npy").is_file() for rec in (0, 1))
    assert not any(n.startswith(AGGREGATES) for n in os.listdir(d))
    manifest = _manifest(d)
    assert manifest["member_labels"] == [A]
    assert manifest["member_fps"] == {A: FPS[A]}
    assert manifest["has_combiner_spatial_gate"] is False
    assert manifest["pca_amps"] == {} and manifest["pca_var"] == {}
    assert result.files_deleted == 2 + 2 * len(AGGREGATES)
    assert result.bytes_freed > 0


def test_a_continued_member_is_dropped(tmp_path):
    d = _bucket(tmp_path)

    result = purge_stale_bucket(str(d), current={**FPS, B: "ckpt-2:b"}, records_fp="rec-1")

    assert result.dropped == [B]
    assert _manifest(d)["member_labels"] == [A]


def test_an_unproven_positional_bucket_drops_every_member(tmp_path):
    d = _bucket(tmp_path, positional=True)

    result = purge_stale_bucket(str(d), current=FPS, records_fp="rec-1")

    assert result.dropped == [A, B]
    assert sorted(os.listdir(d)) == ["lr_00000.npy", "lr_00001.npy", VIZ_INDEX]
    assert _manifest(d)["member_labels"] == []


def test_a_positional_bucket_keeps_members_whose_fingerprints_are_proven(tmp_path):
    d = _bucket(tmp_path, positional=True)

    result = purge_stale_bucket(str(d), current=FPS, records_fp="rec-1", adopt=FPS)

    assert result.dropped == []
    assert (d / "member_02_00001.npy").is_file()
    assert _manifest(d)["member_fps"] == FPS


def test_a_bucket_made_from_other_records_is_wiped(tmp_path):
    d = _bucket(tmp_path)

    result = purge_stale_bucket(str(d), current=FPS, records_fp="rec-2")

    assert result.wiped == "made from other records"
    assert not d.exists()
    assert result.files_deleted == 2 * 2 + 2 * (1 + len(AGGREGATES)) + 1


def test_unknown_current_records_never_wipe(tmp_path):
    d = _bucket(tmp_path)

    result = purge_stale_bucket(str(d), current=FPS, records_fp=None)

    assert result.wiped is None and d.is_dir()


def test_a_blackout_bucket_is_checked_against_its_stamping_source(tmp_path):
    d = _bucket(tmp_path, name=BLACKOUT_INDEX, records_fp=None)

    result = purge_stale_bucket(str(d), name=BLACKOUT_INDEX, current=FPS, records_fp="rec-1")

    assert result.wiped == "made from other records"


def test_cubes_without_a_manifest_are_wiped_but_an_empty_dir_is_not(tmp_path):
    d = tmp_path / "cubes"
    d.mkdir()
    assert purge_stale_bucket(str(d), current=FPS).wiped is None
    np.save(d / "member_01_00000.npy", np.zeros(4, np.float32))

    result = purge_stale_bucket(str(d), current=FPS)

    assert result.wiped == "no manifest" and not d.exists()


def test_files_of_unlisted_members_are_deleted(tmp_path):
    d = _bucket(tmp_path)
    np.save(d / "member_09_00000.npy", np.zeros(4, np.float32))
    np.save(d / "member3_00000.npy", np.zeros(4, np.float32))

    result = purge_stale_bucket(str(d), current=FPS, records_fp="rec-1")

    assert result.dropped == [] and result.files_deleted == 2
    assert (d / "sr_00000.npy").is_file()


def test_fields_outside_the_manifest_go_only_from_blackout_buckets(tmp_path):
    for name in (VIZ_INDEX, BLACKOUT_INDEX):
        root = tmp_path / name
        root.mkdir()
        d = _bucket(root, name=name, indices=(0,))
        np.save(d / "lr_00001.npy", np.zeros(4, np.float32))
        np.save(member_cube_path(str(d), A, 1), np.zeros(4, np.float32))

        purge_stale_bucket(str(d), name=name, current=FPS, records_fp="rec-1")

        kept = (d / "lr_00001.npy").is_file() and (d / "member_01_00001.npy").is_file()
        assert kept == (name == VIZ_INDEX)


def test_removing_a_missing_file_is_quiet(tmp_path):
    _remove_quietly(str(tmp_path / "gone.npy"))
