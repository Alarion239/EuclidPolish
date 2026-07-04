"""Member-aware FASRC pull: a dry-run probe decides which members changed;
unchanged members are neither downloaded nor PSNR re-scored."""
from __future__ import annotations

import os


def _member(base, name, ckpt="ckpt-5"):
    d = os.path.join(base, name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "checkpoint"), "w") as f:
        f.write(f'model_checkpoint_path: "{ckpt}"\n')
    with open(os.path.join(d, f"{ckpt}.index"), "wb") as f:
        f.write(b"idx")
    return d


class _Cap:
    def tick(self, *a, **k):
        pass


class _FakeSSH:
    """Programmable rsync_pull: first call is the probe, later calls recorded."""

    def __init__(self, probe=(0, "", "")):
        self.probe = probe
        self.calls: list[tuple[str, tuple]] = []

    def is_connected(self):
        return True

    def rsync_pull(self, remote, local, extra_args=None, timeout=600):
        self.calls.append((remote, tuple(extra_args or [])))
        if extra_args and "--dry-run" in extra_args:
            return self.probe
        return (0, "", "")


def _setup(tmp_path, monkeypatch):
    from euclid_polish.config import Config
    from euclid_polish.web.helpers import ensemble_viz as ev
    from euclid_polish.web.remote import STATE
    monkeypatch.setattr(Config, "DEFAULT_CHECKPOINT_DIR",
                        str(tmp_path / "ckpt/wdsr"))
    monkeypatch.setattr(Config, "VIS_DIR", str(tmp_path / "vis"))
    monkeypatch.setattr(ev, "remote_ensemble_dir", lambda: "/remote/ensemble")
    # The PSNR refresh is covered by test_member_psnr_cache; stub it here.
    monkeypatch.setattr(ev, "job_member_psnr", lambda cap: {"stub": True})
    return ev, STATE


# --------------------------------------------------------------------------- #
# itemize parsing
# --------------------------------------------------------------------------- #

def test_itemize_content_changes_and_creations_count():
    from euclid_polish.web.helpers.ensemble_viz import changed_members_from_itemize
    out = "\n".join([
        ">f.st...... member_05/ckpt-73.index",       # size+time → changed
        ">f+++++++++ member_11/checkpoint",          # new file → changed
        "cd+++++++++ member_12/",                    # new dir → changed
        ">fc........ member_06/ckpt-1.index",        # checksum → changed
    ])
    assert changed_members_from_itemize(out) == {
        "member_05", "member_06", "member_11", "member_12"}


def test_itemize_openrsync_nine_char_flags():
    """macOS ships openrsync (protocol 29): 9-char flag strings, not 11 —
    exactly what the live probe emits locally. Verified against real output."""
    from euclid_polish.web.helpers.ensemble_viz import changed_members_from_itemize
    out = "\n".join([
        ".d..t.... member_14/",                      # dir time only → ignored
        ">f.st.... member_14/checkpoint",            # size+time → changed
        ">f+++++++ member_14/ckpt-9.index",          # new file → changed
        ">f..t.... member_15/provenance.json",       # time → changed
    ])
    assert changed_members_from_itemize(out) == {"member_14", "member_15"}


def test_itemize_attribute_noise_is_ignored():
    """Perm/owner-only lines are chronic on Linux→macOS pulls — if they counted
    as changes, every member would be re-downloaded on every pull."""
    from euclid_polish.web.helpers.ensemble_viz import changed_members_from_itemize
    out = "\n".join([
        ".f...p..... member_00/ckpt-81.index",       # perm-only, local-change dot
        ">f....og... member_01/ckpt-85.index",       # owner/group only
        ".d..t...... member_02",                     # dir time, dot prefix
        "not an itemize line",
        ">f.st...... other_dir/file",                # not a member
    ])
    assert changed_members_from_itemize(out) == set()


# --------------------------------------------------------------------------- #
# pull flow
# --------------------------------------------------------------------------- #

def test_unchanged_ensemble_downloads_nothing(tmp_path, monkeypatch):
    ev, STATE = _setup(tmp_path, monkeypatch)
    _member(ev.ensemble_dir(), "member_00")
    ssh = _FakeSSH(probe=(0, ".f...p..... member_00/ckpt-5.index\n", ""))
    monkeypatch.setattr(STATE, "ssh", ssh)

    out = ev.job_ensemble_pull(_Cap())
    assert out["changed"] == []
    assert out["n_members"] == 1
    # Exactly one rsync — the dry-run probe. No member downloads.
    assert len(ssh.calls) == 1 and "--dry-run" in ssh.calls[0][1]


def test_only_changed_members_are_downloaded(tmp_path, monkeypatch):
    ev, STATE = _setup(tmp_path, monkeypatch)
    _member(ev.ensemble_dir(), "member_00")
    _member(ev.ensemble_dir(), "member_01")
    ssh = _FakeSSH(probe=(0, ">f.st...... member_01/ckpt-9.index\n", ""))
    monkeypatch.setattr(STATE, "ssh", ssh)

    out = ev.job_ensemble_pull(_Cap())
    assert out["changed"] == ["member_01"]
    pulls = [c for c in ssh.calls if "--dry-run" not in c[1]]
    assert [r for r, _ in pulls] == ["/remote/ensemble/member_01/"]


def test_tombstoned_members_are_never_pulled(tmp_path, monkeypatch):
    """An archived member's FASRC leftover must not resurrect: it is excluded
    from the rsync and dropped from the changed set even if the probe (or a
    stale exclude) reports it."""
    from euclid_polish import ensemble_registry
    ev, STATE = _setup(tmp_path, monkeypatch)
    base = ev.ensemble_dir()
    _member(base, "member_00")
    ensemble_registry.load_registry(base)
    ensemble_registry.archive_member_entry(base, "member_09",
                                           zip_path="models/m09.zip",
                                           commit="abc")
    ssh = _FakeSSH(probe=(0, ">f.st.... member_09/checkpoint\n", ""))
    monkeypatch.setattr(STATE, "ssh", ssh)

    out = ev.job_ensemble_pull(_Cap())
    assert out["changed"] == []                      # tombstone wins
    probe_args = ssh.calls[0][1]
    assert "--exclude=/member_09/" in probe_args     # not even listed
    assert len(ssh.calls) == 1                       # and never downloaded


def test_probe_failure_falls_back_to_full_pull(tmp_path, monkeypatch):
    """A dead probe (transport error, no output) must NOT read as 'no changes'
    — it falls back to the old full-tree rsync."""
    ev, STATE = _setup(tmp_path, monkeypatch)
    _member(ev.ensemble_dir(), "member_00")
    ssh = _FakeSSH(probe=(255, "", "ssh: connection reset"))
    monkeypatch.setattr(STATE, "ssh", ssh)

    out = ev.job_ensemble_pull(_Cap())
    assert out["n_members"] == 1
    full = [c for c in ssh.calls if "--dry-run" not in c[1]]
    assert [r for r, _ in full] == ["/remote/ensemble/"]
