"""Member-aware FASRC pull: a dry-run probe decides which members changed;
unchanged members are neither downloaded nor PSNR re-scored."""
from __future__ import annotations

import json
import os

import pytest

from euclid_polish import ensemble_registry
from euclid_polish.config import Config
from euclid_polish.observability import JobRecord
from euclid_polish.web import fasrc_jobs
from euclid_polish.web.helpers import ensemble_viz as ev
from euclid_polish.web.helpers import purge_requests
from euclid_polish.web.helpers.ensemble_viz import changed_members_from_itemize
from euclid_polish.web.remote import STATE


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
    assert purge_requests.read_pending() is None        # nothing went stale
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
    assert purge_requests.read_pending()["reasons"] == ["pulled member_01"]
    pulls = [c for c in ssh.calls if "--dry-run" not in c[1]]
    assert [r for r, _ in pulls] == ["/remote/ensemble/member_01/"]


def test_tombstoned_members_are_never_pulled(tmp_path, monkeypatch):
    """An archived member's FASRC leftover must not resurrect: it is excluded
    from the rsync and dropped from the changed set even if the probe (or a
    stale exclude) reports it."""
    ev, STATE = _setup(tmp_path, monkeypatch)
    base = ev.ensemble_dir()
    _member(base, "member_00")
    _member(base, "member_09")
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


# --------------------------------------------------------------------------- #
# universal models stay out of the ensemble
# --------------------------------------------------------------------------- #

_UNIVERSAL_SPEC = {"asinh_knees": [0.1, 1, 10, 100, 1000, 10000], "output_knee": 10,
                   "learn_output_knee": True, "num_res_blocks": 16}


def _ledger_add(jobid, name, spec):
    fasrc_jobs.JOBLOG.record_submission(JobRecord(
        jobid=jobid, step_id="ensemble_train", label="Train",
        submitted_at=f"2026-10-1{jobid[-1]}T00:00:00Z", state="COMPLETED",
        params_json=json.dumps({"mode": "add", "member_names": name,
                                "member_spec": json.dumps([spec])})))


def _universal_world(tmp_path, monkeypatch):
    ev, STATE = _setup(tmp_path, monkeypatch)
    base = ev.ensemble_dir()
    ensemble_registry.add_universal_entry(base, "211")    # registered universal
    _member(base, "member_00")
    _member(base, "member_211")
    ensemble_registry.load_registry(base)
    _ledger_add("51", "member_211", _UNIVERSAL_SPEC)
    _ledger_add("52", "member_218", _UNIVERSAL_SPEC)      # FASRC-only universal
    _ledger_add("53", "member_219", {"asinh_knee": 0.3})  # an ordinary new member
    probe = ("".join(f">f.st.... {n}/checkpoint\n"
                     for n in ("member_00", "member_211", "member_218", "member_219")), )
    ssh = _FakeSSH(probe=(0, probe[0], ""))
    monkeypatch.setattr(STATE, "ssh", ssh)
    return ev, base, ssh


def test_every_member_pull_leaves_fasrc_only_universal_models_out(tmp_path, monkeypatch):
    """An every-changed-member pull (and its dry run) must not bootstrap a
    universal model the registry does not know into the ENSEMBLE: it is
    excluded from the rsync and reported; a registered universal model
    refreshes in place and stays standalone; an ordinary new member still
    arrives."""
    ev, base, ssh = _universal_world(tmp_path, monkeypatch)

    dry = ev.job_ensemble_pull(_Cap(), dry_run=True)
    assert dry["universal_skipped"] == ["member_218"]
    assert dry["changed"] == ["member_00", "member_211", "member_219"]
    assert "--exclude=/member_218/" in ssh.calls[0][1]

    out = ev.job_ensemble_pull(_Cap())
    assert out["universal_skipped"] == ["member_218"]
    pulls = [r for r, args in ssh.calls if "--dry-run" not in args]
    assert pulls == ["/remote/ensemble/member_00/", "/remote/ensemble/member_211/",
                     "/remote/ensemble/member_219/"]
    reg = ensemble_registry.load_registry(base)
    assert reg["universal"] == ["member_211"] and "member_211" not in reg["active"]


def test_an_explicit_pull_leaves_unregistered_universal_models_out(tmp_path, monkeypatch):
    """Typing a FASRC-only universal model into the ensemble Pull dialog must
    not make it an ensemble member: it is dropped (reported in
    ``universal_skipped``) and the other names still arrive; a list of only
    such names is refused."""
    ev, base, ssh = _universal_world(tmp_path, monkeypatch)
    out = ev.job_ensemble_pull(_Cap(), members=["218", "219"])
    assert out["universal_skipped"] == ["member_218"] and out["changed"] == ["member_219"]
    assert out["requested"] == ["member_218", "member_219"]
    assert "--exclude=/member_218/" in ssh.calls[0][1]
    pulls = [r for r, args in ssh.calls if "--dry-run" not in args]
    assert pulls == ["/remote/ensemble/member_219/"]
    with pytest.raises(RuntimeError, match="Models › Universal"):
        ev.job_ensemble_pull(_Cap(), members=["218"])
    assert "member_218" not in ensemble_registry.load_registry(base)["active"]


def test_an_explicit_pull_with_allow_universal_pulls_what_it_names(tmp_path, monkeypatch):
    ev, _base, ssh = _universal_world(tmp_path, monkeypatch)
    out = ev.job_ensemble_pull(_Cap(), members=["218"], allow_universal=True)
    assert out["universal_skipped"] == [] and out["changed"] == ["member_218"]
    assert "--exclude=/member_218/" not in ssh.calls[0][1]


def test_an_ensemble_member_a_gate_reads_is_pulled_back_as_a_member(tmp_path, monkeypatch):
    """195 is universal by recipe but an ensemble member the production gate
    reads: its dir vanished (dropped from active), so the every-member pull
    must restore it, not leave it out as a FASRC-only universal model."""
    ev, base, ssh = _universal_world(tmp_path, monkeypatch)
    _ledger_add("54", "member_195", _UNIVERSAL_SPEC)
    gate = os.path.join(str(tmp_path / "vis"), "ensemble", "starfull", "spatial_gate_combiner")
    os.makedirs(gate)
    with open(os.path.join(gate, "combiner.json"), "w") as f:
        json.dump({"kind": "spatial_gate", "member_labels": ["00·psnr", "195·psnr"]}, f)
    ssh.probe = (0, ssh.probe[1] + ">f.st.... member_195/checkpoint\n", "")
    out = ev.job_ensemble_pull(_Cap())
    assert out["universal_skipped"] == ["member_218"]
    assert "member_195" in out["changed"]


def test_probe_failure_fallback_keeps_universal_models_out(tmp_path, monkeypatch):
    ev, _base, ssh = _universal_world(tmp_path, monkeypatch)
    ssh.probe = (255, "", "ssh: connection reset")
    out = ev.job_ensemble_pull(_Cap())
    full = [args for r, args in ssh.calls if "--dry-run" not in args]
    assert full and "--exclude=/member_218/" in full[0]
    assert "member_218" not in out["changed"]
