"""``fasrc_fetcher``: the dataset mirrors in the fetch cache are pinned.

The synced synthetic records (``images/records_v2``) and the star catalogue
mirror (``euclid_stars/stars.csv``) live in the generic LRU fetch cache. The
LRU must never evict them, and their bytes must not count against the cap, so
an ordinary pull (an ePSF stack, a FITS inspect) evicts only other pulls. The
cache's size bookkeeping is an in-memory index: a pull under the cap walks
nothing, an eviction walks the tree once.

Everything runs in tmp dirs with a fake SSH session; nothing touches FASRC or
the real ``data/_fasrc_cache``.
"""

from __future__ import annotations

import os

import pytest

from euclid_polish.config import Config
from euclid_polish.web import fasrc_config
from euclid_polish.web import fasrc_fetcher as ff
from euclid_polish.web import remote as remote_module
from euclid_polish.web.helpers import paths, purge_requests, sky_records, status
from euclid_polish.web.routes import views

DATA = "/n/netscratch/lab/EuclidPolish/data"
CFG = fasrc_config.FasrcConfig(
    data_dir=DATA,
    ckpt_dir="/n/netscratch/lab/EuclidPolish/ckpt/wdsr",
    repo_path="/n/holylabs/lab/EuclidPolish",
)
RECORDS = f"{DATA}/images/records_v2"
BACKUP = f"{DATA}/images/records_v2_local_backup_20260919"
STARS = f"{DATA}/euclid_stars/stars.csv"
CUTOUT = f"{DATA}/euclid_stars/cutouts/VIS/star_0001.fits"
PSF = f"{DATA}/euclid_psf/euclid_psf_VIS.fits"
CAP = 1000


class _FakeSSH:
    """Writes ``sizes[remote]`` bytes (default 100) where rsync would."""

    def __init__(self, sizes: dict[str, int]):
        self.sizes = sizes
        self.pulled: list[str] = []

    def is_connected(self) -> bool:
        return True

    def rsync_pull(self, remote_path, local_dir, **_kw):
        self.pulled.append(remote_path)
        os.makedirs(local_dir, exist_ok=True)
        with open(os.path.join(local_dir, os.path.basename(remote_path)), "wb") as fh:
            fh.write(b"x" * self.sizes.get(remote_path, 100))
        return 0, "", ""


@pytest.fixture
def ssh(tmp_path, monkeypatch):
    """A tmp cache with a 1000-byte cap, a fixed FASRC config and a fake SSH."""
    root = tmp_path / "cache"
    root.mkdir()
    monkeypatch.setattr(Config, "FASRC_CACHE_DIR", str(root))
    monkeypatch.setattr(Config.WebFetch, "MAX_CACHE_BYTES", CAP)
    monkeypatch.setattr(fasrc_config, "load", lambda: CFG)
    fake = _FakeSSH({})
    monkeypatch.setattr(ff, "_remote_size_bytes",
                        lambda p: (True, fake.sizes.get(p, 100), None))
    monkeypatch.setattr(remote_module.STATE, "ssh", fake)
    ff.reset_cache_index()
    yield fake
    ff.reset_cache_index()


def _plant(remote: str, size: int, mtime: float) -> str:
    """A file an earlier pull left in the cache, fetched at ``mtime``."""
    local = ff._local_path_for(remote)
    os.makedirs(os.path.dirname(local), exist_ok=True)
    with open(local, "wb") as fh:
        fh.write(b"p" * size)
    os.utime(local, (mtime, mtime))
    return local


def _disk_bytes(root: str) -> tuple[int, int]:
    """``(pinned, evictable)`` bytes on disk under ``root`` (no ``os.walk``)."""
    pinned = evictable = 0
    stack = [root]
    while stack:
        with os.scandir(stack.pop()) as it:
            for entry in it:
                if entry.is_dir(follow_symlinks=False):
                    stack.append(entry.path)
                elif ff.is_pinned(entry.path):
                    pinned += entry.stat().st_size
                else:
                    evictable += entry.stat().st_size
    return pinned, evictable


def _pull(remote: str, **kwargs) -> ff.FetchResult:
    result = ff.fetch_one_file(remote, force=True, max_bytes=10**6, **kwargs)
    assert result.ok, result.error
    return result


# ---------------------------------------------------------------------------
# What is pinned
# ---------------------------------------------------------------------------

def test_the_pins_follow_the_sync_helpers(ssh):
    """The pin list is derived from the helpers the syncs use, so it cannot
    drift: every file a records sync, the training-catalogue sync or the
    star-catalogue sync writes is pinned; other pulls are not."""
    records_dir = paths._sky_records_local_dir()
    assert records_dir == ff._local_path_for(RECORDS)
    targets = views.sync_targets(paths._sky_records_remote_dir(), list(sky_records.SUBSETS),
                                 list(views.SYNC_KINDS))
    assert targets and all(ff.is_pinned(ff._local_path_for(r)) for r in targets.values())
    sidecar = f"{paths._sky_records_remote_dir()}/{sky_records.artifact_sidecar_name('ab12cd34')}"
    assert ff.is_pinned(ff._local_path_for(sidecar))
    assert ff.is_pinned(ff._local_path_for(status._fasrc_catalog_remote_path()))
    for remote in (BACKUP + "/hr_test.tfrecord", CUTOUT, PSF, f"{DATA}/images/other.tfrecord",
                   f"{DATA}/euclid_stars/stars_old.csv"):
        assert not ff.is_pinned(ff._local_path_for(remote)), remote
    assert {pin["id"] for pin in ff.pinned_mirrors()} == {"records", "stars-catalog"}


# ---------------------------------------------------------------------------
# Eviction skips the pins and the cap counts evictable bytes only
# ---------------------------------------------------------------------------

def test_a_pull_over_the_cap_evicts_only_unpinned_files(ssh):
    hr = _plant(f"{RECORDS}/hr_validate.tfrecord", 600, 1_000_000)
    clean = _plant(f"{RECORDS}/clean_validate.tfrecord", 600, 1_000_001)
    sources = _plant(f"{RECORDS}/sources_validate.csv", 50, 1_000_002)
    stars = _plant(STARS, 100, 1_000_003)
    backup = _plant(f"{BACKUP}/hr_test.tfrecord", 500, 1_000_004)
    cutout = _plant(CUTOUT, 300, 1_000_005)
    ssh.sizes[PSF] = 400

    psf = _pull(PSF).local_path

    # 500 + 300 + 400 evictable > 1000: the oldest unpinned file (the backup
    # copy) goes; the older pinned records and the catalogue stay.
    assert not os.path.exists(backup)
    for path in (hr, clean, sources, stars, cutout, psf):
        assert os.path.isfile(path), path


def test_pinned_bytes_do_not_count_against_the_cap(ssh):
    hr = _plant(f"{RECORDS}/hr_test.tfrecord", 5000, 1_000_000)
    cutout = _plant(CUTOUT, 300, 1_000_001)
    ssh.sizes[PSF] = 400

    psf = _pull(PSF).local_path

    # 5700 B on disk but only 700 B evictable: under the 1000 B cap, so
    # nothing is evicted (pinned bytes would otherwise starve every pull).
    for path in (hr, cutout, psf):
        assert os.path.isfile(path), path


def test_direct_eviction_never_touches_pinned_files(ssh):
    pinned = [_plant(f"{RECORDS}/dirty_test.tfrecord", 200, 1_000_000), _plant(STARS, 30, 1_000_001)]
    loose = [_plant(CUTOUT, 300, 1_000_002), _plant(PSF, 70, 1_000_003)]

    assert ff._evict_lru_until_under(0) == 370

    assert all(os.path.isfile(p) for p in pinned)
    assert not any(os.path.exists(p) for p in loose)


def test_a_records_sync_keeps_every_shard_and_later_pulls_cannot_evict_them(ssh, monkeypatch):
    """The sky sync scenario: the test + validate shards together exceed the
    cap, and a later ePSF pull crosses it again; no shard is ever evicted."""
    monkeypatch.setattr(views.sky_records, "records_artifact_id", lambda _path: None)

    class _Cap:
        def tick(self, *_a):
            pass

        def write(self, _text):
            pass

    targets = views.sync_targets(paths._sky_records_remote_dir(), ["test", "validate"],
                                 list(views.SYNC_KINDS))
    for remote in targets.values():
        ssh.sizes[remote] = 400
    old_cutout = _plant(CUTOUT, 300, 1_000_000)

    result = views._job_sky_sync(_Cap(), targets, ["test", "validate"])

    assert all(entry["ok"] for entry in result["files"].values())
    assert purge_requests.read_pending()["reasons"][0].startswith("synced records ")
    shards = [ff._local_path_for(remote) for remote in targets.values()]
    assert all(os.path.isfile(p) for p in shards)
    assert os.path.isfile(old_cutout)        # 3200 B of shards cost the cap nothing

    ssh.sizes[PSF] = 900
    _pull(PSF)

    assert all(os.path.isfile(p) for p in shards)
    assert not os.path.exists(old_cutout)    # the only evictable file older than the PSF


# ---------------------------------------------------------------------------
# Size bookkeeping
# ---------------------------------------------------------------------------

def test_the_size_index_stays_correct_across_pulls_and_evictions(ssh, monkeypatch):
    root = os.path.realpath(Config.FASRC_CACHE_DIR)
    _plant(f"{RECORDS}/hr_test.tfrecord", 2000, 1_000_000)
    _plant(CUTOUT, 300, 1_000_001)
    walks: list[str] = []
    real_walk = os.walk

    def counting_walk(top, *args, **kwargs):
        walks.append(os.fspath(top))
        return real_walk(top, *args, **kwargs)

    monkeypatch.setattr(ff.os, "walk", counting_walk)

    # The first miss builds the index (one walk) …
    ssh.sizes[PSF] = 200
    _pull(PSF)
    assert len(walks) == 1
    assert ff._indexed_usage() == _disk_bytes(root) == (2000, 500)

    # … later misses under the cap walk nothing, and a re-pull of the same
    # file with a new size replaces its entry instead of adding to it.
    ssh.sizes[PSF] = 250
    _pull(PSF)
    ssh.sizes[f"{RECORDS}/hr_validate.tfrecord"] = 900
    _pull(f"{RECORDS}/hr_validate.tfrecord")
    assert len(walks) == 1
    assert ff._indexed_usage() == _disk_bytes(root) == (2900, 550)

    # A pull that crosses the cap walks once (to evict from what is really on
    # disk) and the index matches the disk after the eviction.
    other = f"{DATA}/euclid_psf/euclid_psf_Y.fits"
    ssh.sizes[other] = 600
    _pull(other)
    assert len(walks) == 2
    assert ff._indexed_usage() == _disk_bytes(root)
    assert _disk_bytes(root)[1] <= CAP
    assert not os.path.exists(ff._local_path_for(CUTOUT))


def test_a_cold_pull_that_crosses_the_cap_walks_once(ssh, monkeypatch):
    """The first pull of a process builds the index from disk; when that pull
    also crosses the cap, the eviction reuses the fresh index (one walk, not two)."""
    root = os.path.realpath(Config.FASRC_CACHE_DIR)
    old = _plant(CUTOUT, 900, 1_000_000)
    walks: list[str] = []
    real_walk = os.walk

    def counting_walk(top, *args, **kwargs):
        walks.append(os.fspath(top))
        return real_walk(top, *args, **kwargs)

    monkeypatch.setattr(ff.os, "walk", counting_walk)
    ssh.sizes[PSF] = 400

    _pull(PSF)

    assert len(walks) == 1
    assert not os.path.exists(old)
    assert ff._indexed_usage() == _disk_bytes(root) == (0, 400)


def test_a_file_deleted_behind_the_index_does_not_break_the_next_pull(ssh):
    gone = _plant(CUTOUT, 600, 1_000_000)
    _pull(PSF)                                 # warms the index with the cutout in it
    os.remove(gone)                            # removed by hand, not by the fetcher
    keep = _plant(f"{DATA}/euclid_psf/euclid_psf_J.fits", 300, 1_000_001)
    ssh.sizes[f"{DATA}/euclid_psf/euclid_psf_H.fits"] = 700

    result = _pull(f"{DATA}/euclid_psf/euclid_psf_H.fits")

    assert result.size_bytes == 700
    assert ff._indexed_usage() == _disk_bytes(os.path.realpath(Config.FASRC_CACHE_DIR))
    assert _disk_bytes(os.path.realpath(Config.FASRC_CACHE_DIR))[1] <= CAP
    assert not os.path.exists(keep)            # the oldest file left on disk made room


def test_cache_usage_reports_pinned_and_evictable_bytes(ssh):
    _plant(f"{RECORDS}/hr_test.tfrecord", 700, 1_000_000)
    _plant(STARS, 50, 1_000_001)
    _plant(CUTOUT, 200, 1_000_002)
    _plant(f"{BACKUP}/hr_test.tfrecord", 90, 1_000_003)

    usage = ff.cache_usage()

    assert usage["total_bytes"] == 1040 == ff.cache_size_bytes()
    assert usage["pinned_bytes"] == 750 and usage["evictable_bytes"] == 290
    assert usage["budget_bytes"] == CAP and usage["files"] == 4
    by_id = {pin["id"]: pin for pin in usage["pinned"]}
    assert by_id["records"]["bytes"] == 700 and by_id["records"]["files"] == 1
    assert by_id["stars-catalog"]["bytes"] == 50
    assert by_id["records"]["path"] == paths._sky_records_local_dir()
