"""Integration tests for the FASRC tab without touching real FASRC.

Strategy: swap the global :class:`SSHSession` in
``euclid_polish.web.remote.STATE`` for a :class:`LocalSSHSession` that
runs every command in a tmp directory with ``bash -c``. Put a fake
``sbatch`` shim on PATH that records what it was given and prints
``Submitted batch job 99999``.

This exercises the entire submission pipeline:

  * the Flask route receiving the form,
  * the script body being uploaded via the SSH ``cat <<'EOF'`` heredoc,
  * the chmod / sbatch invocations,
  * the sqlite write,
  * subsequent ``squeue`` parsing,
  * tailing the SLURM log via the SSE handler,
  * progress lines updating sqlite as they stream past.

No process leaves the box; no Bitwarden is touched.
"""

from __future__ import annotations

import json
import os
import re
import stat
import textwrap
import time
from pathlib import Path

import pytest

from euclid_polish.web import app as app_module
from euclid_polish.web import fasrc_config, fasrc_jobs
from euclid_polish.web.remote import STATE

from tests._local_ssh import LocalSSHSession


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_remote(tmp_path, monkeypatch):
    """Set up a fake FASRC root, sqlite, and SSH transport."""
    # 1. Fake remote FS rooted at tmp_path/remote.
    remote_root = tmp_path / "remote"
    repo = remote_root / "EuclidPolish"
    (repo / "logs" / "jobs").mkdir(parents=True)
    (repo / "scripts").mkdir(parents=True)
    # Stub run_pipeline.py — the script body will reference it but never run it.
    (repo / "scripts" / "run_pipeline.py").write_text("print('ok')\n")
    data_dir = remote_root / "data"
    ckpt_dir = remote_root / "ckpt"
    data_dir.mkdir(); ckpt_dir.mkdir()

    # 2. Fake `sbatch` and `mamba` shims so the local shell can run our script
    #    without SLURM or conda being installed. Each one logs its argv to a
    #    sidecar file the tests can later read.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    sbatch_log = bin_dir / "sbatch.argv"
    (bin_dir / "sbatch").write_text(textwrap.dedent(f"""\
        #!/usr/bin/env bash
        # record argv for inspection
        printf '%s\\n' "$@" > {sbatch_log}
        echo "Submitted batch job 99999"
    """))
    (bin_dir / "mamba").write_text("#!/usr/bin/env bash\nexit 0\n")
    (bin_dir / "module").write_text("#!/usr/bin/env bash\nexit 0\n")
    (bin_dir / "conda").write_text("#!/usr/bin/env bash\nexit 0\n")
    for shim in ("sbatch", "mamba", "module", "conda"):
        os.chmod(bin_dir / shim,
                 os.stat(bin_dir / shim).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    # 3. Point fasrc_config at a tmp file and persist matching paths.
    cfg_file = tmp_path / "fasrc.json"
    monkeypatch.setattr(fasrc_config, "CONFIG_PATH", str(cfg_file))
    monkeypatch.setattr(fasrc_config, "CONFIG_DIR",  str(tmp_path))
    cfg = fasrc_config.FasrcConfig(
        ssh_user="tester", ssh_host="localhost",
        bw_item="FASRC",
        repo_path=str(repo),
        conda_env_path=str(remote_root / "conda-env"),
        data_dir=str(data_dir),
        ckpt_dir=str(ckpt_dir),
        local_ckpt_mirror=str(tmp_path / "local_ckpt"),
        n_gpus=1, n_cpus=4, memory="8G", time_limit="01:00:00",
        n_train=10, n_valid=2, image_size=12, batch_size=2, steps=1000,
    )
    fasrc_config.save(cfg)

    # 4. Use a fresh sqlite db rooted in tmp_path.
    db = fasrc_jobs.JobDB(path=str(tmp_path / "jobs.db"))
    monkeypatch.setattr(fasrc_jobs, "DB", db)

    # 5. Swap the global SSH session for the local stand-in. The local
    #    session inherits the test's PATH override so our fake sbatch is found.
    sess = LocalSSHSession(
        cwd=str(remote_root),
        env={"PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"},
    )
    monkeypatch.setattr(STATE, "ssh", sess)
    monkeypatch.setattr(STATE, "connected_at", time.time())

    yield {
        "cfg": cfg, "remote_root": remote_root, "repo": repo,
        "data_dir": data_dir, "ckpt_dir": ckpt_dir,
        "bin_dir": bin_dir, "sbatch_log": sbatch_log, "db": db,
    }

    # Clean up the patched STATE so a later test doesn't see a stale session.
    STATE.ssh = None
    STATE.connected_at = None


@pytest.fixture
def client(fake_remote):
    app = app_module.create_app()
    app.testing = True
    return app.test_client()


# ---------------------------------------------------------------------------
# Submit flow
# ---------------------------------------------------------------------------

def test_submit_writes_sbatch_script_with_correct_contents(fake_remote, client):
    r = client.post("/api/fasrc/submit", data={
        "label": "integration",
        "n_gpus": 2, "n_cpus": 16, "memory": "64G", "time_limit": "06:00:00",
        "n_train": 100, "n_valid": 5, "image_size": 60, "batch_size": 4,
        "steps": 2000, "extra_flags": "--skip-generate",
    })
    assert r.status_code == 200, r.get_json()
    data = r.get_json()
    assert data["ok"] is True
    assert data["jobid"] == "99999"

    # The script should now live under <repo>/logs/jobs/.
    jobs_dir = fake_remote["repo"] / "logs" / "jobs"
    scripts  = sorted(jobs_dir.glob("*.sh"))
    assert len(scripts) == 1, f"expected one script, got {scripts}"
    body = scripts[0].read_text()

    # Resources made it into the SBATCH header.
    assert "#SBATCH --gres=gpu:2" in body
    assert "#SBATCH --cpus-per-task=16" in body
    assert "#SBATCH --mem=64G" in body
    assert "#SBATCH --time=06:00:00" in body
    # Training knobs reached the run_pipeline command.
    assert "--ntrain 100" in body
    assert "--nvalid 5" in body
    assert "--image-size 60" in body
    assert "--batch-size 4" in body
    assert "--steps 2000 --skip-generate" in body
    # Conda env path is the test's, not the developer default.
    assert str(fake_remote["cfg"].conda_env_path) in body
    assert str(fake_remote["cfg"].data_dir) in body
    assert str(fake_remote["cfg"].ckpt_dir) in body


def test_submit_invokes_sbatch_with_the_built_script(fake_remote, client):
    r = client.post("/api/fasrc/submit", data={
        "label": "y",
        "n_gpus": 1, "n_cpus": 8, "memory": "32G", "time_limit": "12:00:00",
        "n_train": 6400, "n_valid": 200, "image_size": 510, "batch_size": 16,
        "steps": 400000, "extra_flags": "",
    })
    assert r.status_code == 200
    # The shim wrote its argv list to sbatch.argv.
    argv = fake_remote["sbatch_log"].read_text().strip().splitlines()
    assert len(argv) == 1
    assert argv[0].startswith("logs/jobs/")
    assert argv[0].endswith(".sh")


def test_submit_records_job_in_sqlite(fake_remote, client):
    client.post("/api/fasrc/submit", data={
        "label": "remember me",
        "n_gpus": 1, "n_cpus": 8, "memory": "32G", "time_limit": "12:00:00",
        "n_train": 6400, "n_valid": 200, "image_size": 510, "batch_size": 16,
        "steps": 400000, "extra_flags": "",
    })
    row = fake_remote["db"].get("99999")
    assert row is not None
    assert row["label"] == "remember me"
    assert row["state"] == "PENDING"
    p = json.loads(row["params_json"])
    assert p["steps"] == 400000


def test_submit_refuses_when_disconnected(client, monkeypatch):
    monkeypatch.setattr(STATE, "ssh", None)
    r = client.post("/api/fasrc/submit", data={})
    assert r.status_code == 400
    assert "not connected" in r.get_json()["error"]


# ---------------------------------------------------------------------------
# Queue + cancel
# ---------------------------------------------------------------------------

def test_queue_endpoint_with_no_jobs(fake_remote, client, monkeypatch):
    # Override squeue → empty.
    bin_dir = fake_remote["bin_dir"]
    (bin_dir / "squeue").write_text("#!/usr/bin/env bash\nexit 0\n")
    os.chmod(bin_dir / "squeue", 0o755)
    r = client.get("/api/fasrc/queue")
    assert r.status_code == 200
    assert r.get_json()["rows"] == []


def test_queue_endpoint_parses_fake_squeue_output(fake_remote, client):
    bin_dir = fake_remote["bin_dir"]
    (bin_dir / "squeue").write_text(textwrap.dedent("""\
        #!/usr/bin/env bash
        printf '1001\teuclid-a\tRUNNING\t00:30:00\t12:00:00\t1\tNone\t2026-05-12T12:00:00\n'
        printf '1002\teuclid-b\tPENDING\t0:00\t12:00:00\t1\tResources\tN/A\n'
    """))
    os.chmod(bin_dir / "squeue", 0o755)
    r = client.get("/api/fasrc/queue")
    data = r.get_json()
    assert data["ok"] is True
    assert len(data["rows"]) == 2
    assert data["rows"][0]["state"] == "RUNNING"
    assert data["rows"][1]["jobid"] == "1002"


# ---------------------------------------------------------------------------
# Live log SSE
# ---------------------------------------------------------------------------

def test_log_stream_emits_lines_and_updates_progress(fake_remote, client):
    # First submit a job so the db knows about it + the log path.
    r = client.post("/api/fasrc/submit", data={
        "label": "stream test",
        "n_gpus": 1, "n_cpus": 8, "memory": "32G", "time_limit": "12:00:00",
        "n_train": 10, "n_valid": 2, "image_size": 12, "batch_size": 2,
        "steps": 1000, "extra_flags": "",
    })
    jobid = r.get_json()["jobid"]
    log_path = Path(fake_remote["db"].get(jobid)["log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Pre-seed the log file with a few lines so ``tail -F`` has something
    # to emit before we close the stream.
    log_path.write_text(textwrap.dedent("""\
        starting training
        step 100/1000 loss=4.2
        step 200/1000 loss=3.7
        step 300/1000 loss=3.1
    """))

    # The SSE handler is a generator — pull a few events with a timeout
    # cap on iterations so the test never hangs.
    resp = client.get(f"/api/fasrc/log/{jobid}?which=out", buffered=False)
    assert resp.status_code == 200
    chunks = []
    deadline = time.time() + 5
    for chunk in resp.response:
        chunks.append(chunk.decode() if isinstance(chunk, bytes) else chunk)
        if any("300/1000" in c for c in chunks) or time.time() > deadline:
            break
    resp.close()
    combined = "".join(chunks)
    assert "step 100/1000" in combined
    assert "step 300/1000" in combined

    # The line parser should have written into sqlite as side-effect.
    row = fake_remote["db"].get(jobid)
    assert row["progress_total"] == 1000
    assert row["progress_step"] >= 100  # latest line we saw


# ---------------------------------------------------------------------------
# Checkpoint auto-mirror
# ---------------------------------------------------------------------------

def test_mirror_trigger_rsyncs_remote_ckpts(fake_remote, client, tmp_path):
    # Pretend training wrote a checkpoint file into the "remote" ckpt dir.
    (fake_remote["ckpt_dir"] / "ckpt-12345.h5").write_bytes(b"hello world")
    (fake_remote["ckpt_dir"] / "training_log.jsonl").write_text('{"step": 1}\n')

    r = client.post("/api/fasrc/mirror/trigger")
    assert r.status_code == 200

    mirror = Path(fake_remote["cfg"].local_ckpt_mirror)
    assert (mirror / "ckpt-12345.h5").read_bytes() == b"hello world"
    assert (mirror / "training_log.jsonl").exists()


def test_mirror_status_reflects_last_sync(fake_remote, client):
    (fake_remote["ckpt_dir"] / "a.bin").write_bytes(b"x")
    client.post("/api/fasrc/mirror/trigger")
    s = client.get("/api/fasrc/mirror/status").get_json()
    assert s["last_run_at"] is not None
    assert s["last_rc"] == 0
    assert s["remote_dir"].endswith("/")
    assert s["local_dir"].endswith("local_ckpt")


# ---------------------------------------------------------------------------
# Listings + remote git
# ---------------------------------------------------------------------------

def _has_gnu_find() -> bool:
    """``find -printf`` is GNU-only; BSD/macOS find rejects the flag."""
    import subprocess
    r = subprocess.run(["find", "/tmp", "-maxdepth", "0", "-printf", "%p"],
                       capture_output=True)
    return r.returncode == 0


@pytest.mark.skipif(not _has_gnu_find(),
                    reason="route uses GNU `find -printf`; FASRC is Linux")
def test_data_listing_picks_up_tfrecord_files(fake_remote, client):
    (fake_remote["data_dir"] / "images" / "records_v2").mkdir(parents=True)
    (fake_remote["data_dir"] / "images" / "records_v2" / "clean_train.tfrecord").write_bytes(b"x" * 1024)
    (fake_remote["ckpt_dir"] / "training_log.jsonl").write_text("{}\n")
    r = client.get("/api/fasrc/data-listing")
    assert r.status_code == 200, r.get_json()
    data = r.get_json()
    assert data["ok"] is True
    paths = {t["path"] for t in data["tfrecords"]}
    assert any(p.endswith("clean_train.tfrecord") for p in paths)


def test_cancel_endpoint_marks_job_cancelled(fake_remote, client):
    # Plant a fake scancel and a queued job.
    bin_dir = fake_remote["bin_dir"]
    (bin_dir / "scancel").write_text("#!/usr/bin/env bash\nexit 0\n")
    os.chmod(bin_dir / "scancel", 0o755)
    fake_remote["db"].insert("77777", label="x", params={"steps": 100},
                             script_path=".", log_path=".", err_path=".")
    r = client.post("/api/fasrc/cancel", data={"jobid": "77777"})
    assert r.status_code == 200
    row = fake_remote["db"].get("77777")
    assert row["state"] == "CANCELLED"
    assert row["ended_at"] is not None
