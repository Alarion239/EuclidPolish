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
        "confirm": "yes",
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
    # Training knobs reached the run_pipeline command. Each argv token
    # is rendered on its own continuation line by the consolidated
    # builder, so we check the tokens individually instead of as a
    # joined ``--name value`` substring.
    for token in (
        "--ntrain", "100",
        "--nvalid", "5",
        "--image-size", "60",
        "--batch-size", "4",
        "--steps", "2000",
        "--skip-generate",
    ):
        assert token in body, f"missing argv token: {token!r}"
    # Conda env path is the test's, not the developer default.
    assert str(fake_remote["cfg"].conda_env_path) in body
    assert str(fake_remote["cfg"].data_dir) in body
    assert str(fake_remote["cfg"].ckpt_dir) in body


def test_submit_invokes_sbatch_with_the_built_script(fake_remote, client):
    r = client.post("/api/fasrc/submit", data={
        "confirm": "yes",
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


def test_second_submit_is_queued_while_first_runs(fake_remote, client):
    base = {"confirm": "yes",
            "n_gpus": 1, "n_cpus": 8, "memory": "32G", "time_limit": "12:00:00",
            "n_train": 100, "n_valid": 5, "image_size": 60, "batch_size": 4,
            "steps": 1000, "extra_flags": ""}
    # First submit goes straight to the cluster (lane is free).
    r1 = client.post("/api/fasrc/submit", data={**base, "label": "first"})
    d1 = r1.get_json()
    assert d1["ok"] is True and d1.get("jobid") == "99999"
    assert not d1.get("queued")

    # Second submit, while 99999 is PENDING in the DB → queued locally, no
    # new sbatch. The response carries the queue (list of names).
    r2 = client.post("/api/fasrc/submit", data={**base, "label": "second"})
    d2 = r2.get_json()
    assert d2["ok"] is True and d2.get("queued") is True
    assert d2["queue"]["names"] == ["second"]
    # Only one sbatch invocation happened (the queued one wasn't submitted).
    argv = fake_remote["sbatch_log"].read_text().strip()
    assert argv.count(".sh") <= 1


def test_submit_records_job_in_sqlite(fake_remote, client):
    client.post("/api/fasrc/submit", data={
        "confirm": "yes",
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
        printf '1001|euclid-a|RUNNING|00:30:00|12:00:00|1|None|2026-05-12T12:00:00\n'
        printf '1002|euclid-b|PENDING|0:00|12:00:00|1|Resources|N/A\n'
    """))
    os.chmod(bin_dir / "squeue", 0o755)
    r = client.get("/api/fasrc/queue")
    data = r.get_json()
    assert data["ok"] is True
    assert len(data["rows"]) == 2
    assert data["rows"][0]["state"] == "RUNNING"
    assert data["rows"][1]["jobid"] == "1002"


def test_training_status_running_returns_parsed_summary(fake_remote, client):
    """Plant a fake squeue + a Reporter *events* stream (NOT logs), then
    assert the route folds stage / progress / metrics / checkpoint straight
    from the events — no log parsing."""
    bin_dir = fake_remote["bin_dir"]
    # Squeue: one RUNNING job. The jobid matches what we'll plant in sqlite.
    (bin_dir / "squeue").write_text(textwrap.dedent("""\
        #!/usr/bin/env bash
        printf '88888|euclid-test|RUNNING|00:10:00|12:00:00|1|holygpu1|2026-05-13T01:00:00\n'
    """))
    os.chmod(bin_dir / "squeue", 0o755)

    cfg = fake_remote["cfg"]
    log_dir = Path(cfg.repo_path) / "logs" / "jobs"
    log_dir.mkdir(parents=True, exist_ok=True)
    out_path    = log_dir / "euclid-test.out"
    err_path    = log_dir / "euclid-test.err"
    events_path = log_dir / "euclid-test.events"
    fake_remote["db"].insert(
        "88888", label="test run", params={"steps": 400_000},
        script_path=str(log_dir / "euclid-test.sh"),
        log_path=str(out_path), err_path=str(err_path),
        events_path=str(events_path),
    )
    # The structured event stream the trainer writes via Reporter: one
    # stage, two step events (so rate/ETA is computable) and two metric
    # samples (the second wrote a checkpoint → saved=True).
    events = [
        {"ts": 100.0, "kind": "stage",  "value": "training 400000 steps"},
        {"ts": 110.0, "kind": "step",   "value": {"current": 11000, "total": 400000, "label": "train"}},
        {"ts": 110.0, "kind": "metric", "value": {"step": 11000, "total": 400000, "loss": 4.3, "psnr_stretched": 22.1, "psnr_raw": 19.7}},
        {"ts": 210.0, "kind": "step",   "value": {"current": 12000, "total": 400000, "label": "train"}},
        {"ts": 210.0, "kind": "metric", "value": {"step": 12000, "total": 400000, "loss": 4.21, "psnr_stretched": 22.314, "psnr_raw": 19.872, "saved": True}},
    ]
    events_path.write_text("".join(json.dumps(e) + "\n" for e in events))

    r = client.get("/api/fasrc/training-status")
    assert r.status_code == 200, r.get_json()
    data = r.get_json()
    assert data["ok"] is True
    assert data["running"] is True
    assert data["job"]["jobid"]  == "88888"
    assert data["job"]["label"]  == "test run"
    assert data["stage"] == "training 400000 steps"
    assert data["progress"]["current"] == 12000
    assert data["progress"]["total"]   == 400000
    assert data["latest_metrics"]["psnr_stretched"] == 22.314
    assert data["last_checkpoint"] == "step 12000"       # from the saved flag
    assert data["latest_validation"]["step"] == 12000
    assert len(data["validations"]) == 2
    assert data["eta_seconds"] is not None


def test_training_status_returns_running_false_when_queue_empty(fake_remote, client):
    bin_dir = fake_remote["bin_dir"]
    (bin_dir / "squeue").write_text("#!/usr/bin/env bash\nexit 0\n")
    os.chmod(bin_dir / "squeue", 0o755)
    r = client.get("/api/fasrc/training-status")
    assert r.status_code == 200
    data = r.get_json()
    assert data["ok"] is True
    assert data["running"] is False


def test_git_pull_flags_env_update_when_environment_yml_changed(fake_remote, client):
    """When the pull's diff includes ``environment.yml``, the response
    sets ``env_update_needed: True`` so the UI auto-runs mamba env update."""
    cfg = fake_remote["cfg"]
    # Fake `git` that pretends to pull successfully and lists one file.
    (fake_remote["bin_dir"] / "git").write_text(textwrap.dedent("""\
        #!/usr/bin/env bash
        case "$*" in
          'pull --ff-only') echo 'Already up to date.' ;;
          'diff --name-only ORIG_HEAD..HEAD') echo environment.yml ;;
        esac
    """))
    os.chmod(fake_remote["bin_dir"] / "git", 0o755)

    r = client.post("/api/fasrc/git-pull")
    data = r.get_json()
    assert data["ok"] is True
    assert data["env_update_needed"] is True
    assert "environment.yml" in data["changed_files"]


def test_git_pull_does_not_flag_env_update_for_unrelated_changes(fake_remote, client):
    (fake_remote["bin_dir"] / "git").write_text(textwrap.dedent("""\
        #!/usr/bin/env bash
        case "$*" in
          'pull --ff-only') echo 'Updating abc..def' ;;
          'diff --name-only ORIG_HEAD..HEAD') echo README.md ;;
        esac
    """))
    os.chmod(fake_remote["bin_dir"] / "git", 0o755)
    r = client.post("/api/fasrc/git-pull")
    data = r.get_json()
    assert data["env_update_needed"] is False
    assert data["changed_files"] == ["README.md"]


# ---------------------------------------------------------------------------
# Live log SSE
# ---------------------------------------------------------------------------

def test_log_stream_emits_lines(fake_remote, client):
    # First submit a job so the db knows about it + the log path.
    r = client.post("/api/fasrc/submit", data={
        "confirm": "yes",
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

    # The log stream is now a pure raw-log VIEWER: it emits lines but does
    # NOT scrape progress out of them. Progress comes from the Reporter
    # event stream (folded in JobStatus by /api/fasrc/training-status), so
    # the DB progress is untouched by merely tailing the log.
    row = fake_remote["db"].get(jobid)
    assert row["progress_step"] in (None, 0)


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


def test_stages_endpoint_returns_csv_rows_for_finished_job(fake_remote, client):
    """``/api/fasrc/stages/<jobid>`` reads the remote stages CSV that
    ``run_pipeline.py``'s StageTimer writes, and returns one JSON row
    per stage with parsed durations and the params at run time."""
    cfg = fake_remote["cfg"]
    rec_dir = Path(cfg.data_dir) / "images" / "records_v2"
    rec_dir.mkdir(parents=True, exist_ok=True)
    (rec_dir / "stages_12345.csv").write_text(textwrap.dedent("""\
        jobid,stage,started_at,ended_at,duration_seconds,params_dependent,n_train,n_valid,image_size,batch_size,steps
        12345,init,1000000.000,1000012.500,12.500,0,6400,200,510,16,400000
        12345,generate,1000012.500,1001000.000,987.500,1,6400,200,510,16,400000
        12345,convolve,1001000.000,1001400.000,400.000,1,6400,200,510,16,400000
        12345,train,1001400.000,1100000.000,98600.000,1,6400,200,510,16,400000
    """))
    r = client.get("/api/fasrc/stages/12345")
    assert r.status_code == 200
    data = r.get_json()
    assert data["ok"] is True
    rows = data["rows"]
    assert [r["stage"] for r in rows] == ["init", "generate", "convolve", "train"]
    assert rows[0]["params_dependent"] is False
    assert rows[1]["params_dependent"] is True
    assert rows[3]["duration_seconds"] == 98600.0
    assert rows[1]["n_train"] == "6400"


def test_stages_endpoint_returns_empty_when_csv_missing(fake_remote, client):
    """A job that hasn't reached the first stage marker yet doesn't have
    a stages CSV on disk. The endpoint should still succeed with an
    empty row list so the UI can render a 'pending' state."""
    r = client.get("/api/fasrc/stages/99999")
    assert r.status_code == 200
    data = r.get_json()
    assert data["ok"] is True
    assert data["rows"] == []
    assert "stages_99999.csv" in data["path"]


def test_stages_endpoint_rejects_disconnected(client, monkeypatch):
    monkeypatch.setattr(STATE, "ssh", None)
    r = client.get("/api/fasrc/stages/12345")
    assert r.status_code == 400


def test_bootstrap_data_creates_symlinks(fake_remote, client):
    """`ln -sfn` from the durable copy under ``<repo>/data/`` into the
    working ``data_dir`` on netscratch. Both sources exist in this test
    so the route should succeed and the two expected symlinks should
    appear under data_dir."""
    cfg = fake_remote["cfg"]
    repo_data = Path(cfg.repo_path) / "data"
    repo_psf = repo_data / "euclid_psf"
    repo_cosmos = repo_data / "COSMOS2025"
    repo_psf.mkdir(parents=True, exist_ok=True)
    repo_cosmos.mkdir(parents=True, exist_ok=True)
    (repo_psf / "psf_VIS.fits").write_bytes(b"x")
    (repo_cosmos / "cosmos2025.fits").write_bytes(b"y" * 64)

    r = client.post("/api/fasrc/bootstrap-data")
    assert r.status_code == 200, r.get_json()
    data = r.get_json()
    assert data["ok"] is True
    out = data["output"]
    assert "linked: euclid_psf -> "  in out
    assert "linked: COSMOS2025 -> "  in out

    data_root = Path(cfg.data_dir)
    assert (data_root / "euclid_psf").is_symlink()
    assert (data_root / "COSMOS2025").is_symlink()
    assert os.readlink(str(data_root / "euclid_psf"))  == str(repo_psf)
    assert os.readlink(str(data_root / "COSMOS2025")) == str(repo_cosmos)


def test_bootstrap_data_reports_missing_sources(fake_remote, client):
    """If the COSMOS dir hasn't been uploaded yet, surface a clear note
    rather than failing — the user just hasn't run Globus yet."""
    cfg = fake_remote["cfg"]
    repo_psf = Path(cfg.repo_path) / "data" / "euclid_psf"
    repo_psf.mkdir(parents=True, exist_ok=True)
    # Do NOT create the COSMOS dir.

    r = client.post("/api/fasrc/bootstrap-data")
    assert r.status_code == 200
    out = r.get_json()["output"]
    assert "linked: euclid_psf -> "  in out
    assert "MISSING source: "         in out
    assert "/data/COSMOS2025" in out


def test_bootstrap_data_refuses_when_disconnected(client, monkeypatch):
    monkeypatch.setattr(STATE, "ssh", None)
    r = client.post("/api/fasrc/bootstrap-data")
    assert r.status_code == 400


def test_env_update_streams_lines_and_terminates(fake_remote, client):
    """SSE env-update routes the `module load python + mamba env update`
    pipeline through the active SSH session. We stand in fake `module` /
    `mamba` shims that print recognisable lines, then assert the stream
    delivered them in order and closed with a ``done`` event."""
    bin_dir = fake_remote["bin_dir"]
    (bin_dir / "module").write_text(
        "#!/usr/bin/env bash\necho \"module $*\"\n"
    )
    (bin_dir / "mamba").write_text(textwrap.dedent("""\
        #!/usr/bin/env bash
        echo "mamba argv: $*"
        # Read+echo stdin so we can verify `yes |` is feeding it.
        head -3 || true
        echo "Proceed ([y]/n)? y"
        echo "Updated 0 packages."
    """))
    for shim in ("module", "mamba"):
        os.chmod(bin_dir / shim, 0o755)
    # Need a placeholder environment.yml inside the fake repo so the
    # `-f environment.yml` reference resolves under our local cwd.
    (fake_remote["repo"] / "environment.yml").write_text(
        "name: EuclidPolishEnv\nchannels:\n  - conda-forge\n"
        "dependencies:\n  - python=3.12\n",
    )

    r = client.get("/api/fasrc/env-update", buffered=False)
    assert r.status_code == 200
    chunks = []
    deadline = time.time() + 5
    for chunk in r.response:
        chunks.append(chunk.decode() if isinstance(chunk, bytes) else chunk)
        if any("done" in c for c in chunks) or time.time() > deadline:
            break
    r.close()
    combined = "".join(chunks)
    # Header lines from the route generator.
    assert "$ module load python" in combined
    # Output captured from the fake `module` and `mamba` shims.
    assert "module load python" in combined
    assert "mamba argv:" in combined
    assert "Updated 0 packages" in combined
    # And the route emitted the ``done`` event so the client can close.
    assert "event: done" in combined


def test_env_update_refuses_when_disconnected(client, monkeypatch):
    monkeypatch.setattr(STATE, "ssh", None)
    r = client.get("/api/fasrc/env-update")
    assert r.status_code == 400
    assert "not connected" in r.get_data(as_text=True)


def test_extend_time_invokes_scontrol(fake_remote, client):
    """Run a fake scontrol that records argv + echoes a TimeLimit line.

    Asserts the route formats ``+HH:MM:SS`` correctly, calls scontrol
    with the right shape, and surfaces the new TimeLimit to the UI.
    """
    bin_dir = fake_remote["bin_dir"]
    scontrol_log = bin_dir / "scontrol.argv"
    (bin_dir / "scontrol").write_text(textwrap.dedent(f"""\
        #!/usr/bin/env bash
        printf '%s\\n' "$@" >> {scontrol_log}
        case "$1" in
          update) exit 0 ;;
          show)   echo 'JobId=99999 TimeLimit=04:30:00 RunTime=00:30:00 ...' ;;
        esac
    """))
    os.chmod(bin_dir / "scontrol", 0o755)

    r = client.post("/api/fasrc/extend-time",
                    data={"jobid": "99999", "hours": "1.5"})
    assert r.status_code == 200, r.get_json()
    data = r.get_json()
    assert data["ok"] is True
    assert data["delta"] == "+01:30:00"
    assert "TimeLimit=" in data["scontrol"]

    # And scontrol was called with the right update spec.
    argv = scontrol_log.read_text()
    assert "update" in argv
    assert "job=99999" in argv
    assert "TimeLimit=+01:30:00" in argv


def test_extend_time_rejects_zero_or_negative(client, fake_remote):
    r = client.post("/api/fasrc/extend-time",
                    data={"jobid": "1", "hours": "0"})
    assert r.status_code == 400
    r = client.post("/api/fasrc/extend-time",
                    data={"jobid": "1", "hours": "-5"})
    assert r.status_code == 400


def test_extend_time_rejects_above_cap(client, fake_remote):
    r = client.post("/api/fasrc/extend-time",
                    data={"jobid": "1", "hours": "200"})
    assert r.status_code == 400


def test_extend_time_rejects_non_numeric_jobid(client, fake_remote):
    r = client.post("/api/fasrc/extend-time",
                    data={"jobid": "abc", "hours": "1"})
    assert r.status_code == 400


def test_extend_time_surfaces_scontrol_error(fake_remote, client):
    """If SLURM refuses (e.g. above partition MaxTime), the error text
    comes back instead of a generic 500."""
    bin_dir = fake_remote["bin_dir"]
    (bin_dir / "scontrol").write_text(textwrap.dedent("""\
        #!/usr/bin/env bash
        echo 'slurm_update error: Time limit exceeds maximum' >&2
        exit 1
    """))
    os.chmod(bin_dir / "scontrol", 0o755)
    r = client.post("/api/fasrc/extend-time",
                    data={"jobid": "1", "hours": "1"})
    assert r.status_code == 500
    assert "maximum" in r.get_json()["error"].lower()


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


# ---------------------------------------------------------------------------
# Evaluation results sync (eval_results/ is NOT covered by the ckpt mirror)
# ---------------------------------------------------------------------------

def test_evaluation_sync_pulls_results(fake_remote, client, tmp_path, monkeypatch):
    """POST /api/evaluation/sync rsyncs <data_dir>/eval_results → local gallery."""
    from euclid_polish.config import Config

    # Seed a finished run under the fake remote's eval_results dir.
    remote_run = fake_remote["data_dir"] / "eval_results" / "lenses"
    (remote_run / "obj1").mkdir(parents=True)
    (remote_run / "manifest.csv").write_text(
        "id,ra,dec,ok\nobj1,1.0,2.0,True\n")
    (remote_run / "obj1" / "eye.png").write_bytes(b"\x89PNGfake")

    # Point the local gallery at an empty tmp dir.
    local_eval = tmp_path / "local_eval"
    monkeypatch.setattr(Config, "EVAL_RESULTS_DIR", str(local_eval))

    r = client.post("/api/evaluation/sync")
    assert r.status_code == 200, r.get_json()
    j = r.get_json()
    assert j["ok"] is True
    # Files landed locally where the gallery reads them.
    assert (local_eval / "lenses" / "manifest.csv").is_file()
    assert (local_eval / "lenses" / "obj1" / "eye.png").is_file()
    # The response surfaces the now-visible runs so the UI can refresh.
    assert j["n_runs"] == 1
    assert any(run["name"] == "lenses" for run in j["runs"])


def test_evaluation_sync_refuses_when_disconnected(client, monkeypatch):
    monkeypatch.setattr(STATE, "ssh", None)
    r = client.post("/api/evaluation/sync")
    assert r.status_code == 400
    assert r.get_json()["ok"] is False


def test_evaluation_sync_heads_pulls_checkpoints(fake_remote, client, tmp_path,
                                                 monkeypatch):
    """POST /api/evaluation/sync-heads rsyncs <repo_path>/data/lensfinder/heads.

    The heads are written by ``lensfinder_train --out-dir data/lensfinder/heads``,
    a path relative to the SLURM workdir = the repo (holylabs), NOT the separate
    ``data_dir`` (netscratch). Only recons that carry a checkpoint are reported,
    so an untrained recon (hr) is skipped — matching the score step.
    """
    from euclid_polish.config import Config

    heads = fake_remote["repo"] / "data" / "lensfinder" / "heads"
    for recon in ("lr", "sr"):
        ck = heads / recon / "checkpoints"
        ck.mkdir(parents=True)
        (ck / "epoch=1.ckpt").write_bytes(b"fakeckpt")
    (heads / "hr").mkdir(parents=True)            # present but untrained → skipped

    local_data = tmp_path / "local_data"
    monkeypatch.setattr(Config, "DATA_DIR", str(local_data))

    r = client.post("/api/evaluation/sync-heads")
    assert r.status_code == 200, r.get_json()
    j = r.get_json()
    assert j["ok"] is True
    assert sorted(j["recons"]) == ["lr", "sr"]    # hr has no ckpt → not counted
    assert j["n_heads"] == 2
    assert (local_data / "lensfinder" / "heads" / "sr" / "checkpoints"
            / "epoch=1.ckpt").is_file()


def test_evaluation_sync_heads_empty_remote_is_not_error(fake_remote, client,
                                                         tmp_path, monkeypatch):
    """A missing remote heads dir (untrained) is a clean n_heads=0, not a 500."""
    from euclid_polish.config import Config

    # Deliberately do NOT create <data_dir>/lensfinder/heads on the fake remote.
    monkeypatch.setattr(Config, "DATA_DIR", str(tmp_path / "local_data"))

    r = client.post("/api/evaluation/sync-heads")
    assert r.status_code == 200, r.get_json()
    j = r.get_json()
    assert j["ok"] is True
    assert j["n_heads"] == 0 and j["recons"] == []


def test_evaluation_sync_heads_refuses_when_disconnected(client, monkeypatch):
    monkeypatch.setattr(STATE, "ssh", None)
    r = client.post("/api/evaluation/sync-heads")
    assert r.status_code == 400
    assert r.get_json()["ok"] is False
