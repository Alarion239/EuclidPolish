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
  * subsequent ``squeue`` parsing.

No process leaves the box; no Bitwarden is touched.
"""

from __future__ import annotations

import json
import os
import re
import signal
import stat
import subprocess
import textwrap
import time
from pathlib import Path

import pytest

from euclid_polish.config import Config
from euclid_polish.web import app as app_module
from euclid_polish.web import fasrc_config, fasrc_jobs, fasrc_queue, job_config
from euclid_polish.web.fasrc_gate import FASRC_OFFLINE_PAYLOAD
from euclid_polish.web.remote import STATE
from euclid_polish.web.routes import fasrc as fasrc_routes
from tests._local_ssh import LocalSSHSession

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _active_population_calibrations(monkeypatch):
    """Keep submission tests hermetic from the on-disk artifacts.

    ``prepare_params`` fails closed when the activated joint-galaxy artifact
    is stale (e.g. a pre-v15 fit) or no stellar calibration is active — as on
    a clean checkout (CI), where the gitignored data dir holds neither. These
    tests exercise the submission plumbing, not calibration validity.
    Individual tests may override.
    """
    joint = {
        "fingerprint": "j" * 64,
        "generation": {"surface_density_arcmin2": 123.0},
    }
    stars = {
        "fingerprint": "s" * 64,
        "population": {"density_arcmin2": 3.0},
    }
    monkeypatch.setattr(
        "euclid_polish.web.helpers.population_calibration.joint_galaxy_state",
        lambda: {"active": joint, "is_active": True},
    )
    monkeypatch.setattr(
        "euclid_polish.web.helpers.population_calibration.star_state",
        lambda: {"active": stars, "is_active": True},
    )


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
# Submit flow (synthetic generation through the generic step submit; the
# legacy ``/api/fasrc/submit`` route is gone)
# ---------------------------------------------------------------------------

SYNTH = "/api/fasrc/steps/synthetic_generate/submit"


@pytest.fixture
def job_config_file(tmp_path, monkeypatch):
    """A tmp ``job_config.json``: the step submit injects its scene counts."""
    path = tmp_path / "job_config.json"
    monkeypatch.setattr(job_config, "CONFIG_PATH", str(path))
    monkeypatch.setattr(job_config, "CONFIG_DIR", str(tmp_path))

    def write(**fields):
        path.write_text(json.dumps(fields))
    write(n_train=100, n_valid=5, n_test=3, hr_image_size=60)
    return write


def _synth_form(**extra):
    return {"confirm": "yes", "n_gpus": 0, "n_cpus": 8, "memory": "32G",
            "time_limit": "12:00:00", **extra}


def test_submit_writes_sbatch_script_with_correct_contents(
        fake_remote, client, job_config_file):
    r = client.post(SYNTH, data=_synth_form(
        label="integration", n_gpus=2, n_cpus=16, memory="64G",
        time_limit="06:00:00", extra_flags="--skip-generate"))
    assert r.status_code == 200, r.get_json()
    data = r.get_json()
    assert data["ok"] is True
    assert data["jobid"] == "99999"

    # The script should now live under <repo>/logs/jobs/.
    jobs_dir = fake_remote["repo"] / "logs" / "jobs"
    scripts  = sorted(jobs_dir.glob("*.sh"))
    assert len(scripts) == 1, f"expected one script, got {scripts}"
    body = scripts[0].read_text()
    payloads = sorted(jobs_dir.glob("*population.*.json"))
    assert len(payloads) == 2
    assert all(isinstance(json.loads(path.read_text()), dict) for path in payloads)

    # Resources made it into the SBATCH header.
    assert "#SBATCH --gres=gpu:2" in body
    assert "#SBATCH --cpus-per-task=16" in body
    assert "#SBATCH --mem=64G" in body
    assert "#SBATCH --time=06:00:00" in body
    # Generation knobs (from /config's job config) reached the run_pipeline
    # command. Each argv token is rendered on its own continuation line.
    for token in (
        "--ntrain", "100",
        "--nvalid", "5",
        "--ntest", "3",
        "--image-size", "60",
        "--skip-generate",
        "--skip-train",
        "--joint-galaxy-population-file",
        "--star-prior-file",
    ):
        assert token in body, f"missing argv token: {token!r}"
    assert "--joint-galaxy-population-json" not in body
    assert "--star-prior-json" not in body
    # Standalone generation: the training-only knobs are gone.
    assert "--batch-size" not in body
    assert "--steps" not in body
    # Conda env path is the test's, not the developer default.
    assert str(fake_remote["cfg"].conda_env_path) in body
    assert str(fake_remote["cfg"].data_dir) in body
    assert str(fake_remote["cfg"].ckpt_dir) in body


def test_submit_invokes_sbatch_with_the_built_script(
        fake_remote, client, job_config_file):
    r = client.post(SYNTH, data=_synth_form(label="y", extra_flags=""))
    assert r.status_code == 200, r.get_json()
    # The shim wrote its argv list to sbatch.argv.
    argv = fake_remote["sbatch_log"].read_text().strip().splitlines()
    assert len(argv) == 1
    assert argv[0].startswith("logs/jobs/")
    assert argv[0].endswith(".sh")


def test_second_submit_is_queued_while_first_runs(
        fake_remote, client, job_config_file):
    # First submit goes straight to the cluster (lane is free).
    r1 = client.post(SYNTH, data=_synth_form(label="first"))
    d1 = r1.get_json()
    assert d1["ok"] is True and d1.get("jobid") == "99999"
    assert not d1.get("queued")

    # Second submit, while 99999 is PENDING in the DB → queued locally, no
    # new sbatch. The response carries the queue (list of names).
    r2 = client.post(SYNTH, data=_synth_form(label="second"))
    d2 = r2.get_json()
    assert d2["ok"] is True and d2.get("queued") is True
    assert d2["queue"]["names"] == ["second"]
    # Only one sbatch invocation happened (the queued one wasn't submitted).
    argv = fake_remote["sbatch_log"].read_text().strip()
    assert argv.count(".sh") <= 1


def test_submit_records_job_in_sqlite(fake_remote, client, job_config_file):
    job_config_file(n_train=6400, n_valid=200, n_test=100, hr_image_size=510)
    client.post(SYNTH, data=_synth_form(label="remember me", extra_flags=""))
    row = fake_remote["db"].get("99999")
    assert row is not None
    assert row["label"] == "remember me"
    assert row["state"] == "PENDING"
    p = json.loads(row["params_json"])
    # Generation params are persisted; the decoupled training knobs are not.
    assert int(p["n_train"]) == 6400
    assert int(p["image_size"]) == 510
    assert p["step_id"] == "synthetic_generate"
    assert "steps" not in p
    assert "batch_size" not in p


def test_synthetic_submit_persists_prepared_calibration_identities(
    fake_remote, client, monkeypatch, job_config_file,
):
    """The submit ledger receives sidecar identities, not pre-render form data."""
    del fake_remote
    submitted: dict = {}

    def capture_submit(_ssh, *, cfg, built, label, params, step_id):
        del cfg, built, label
        submitted.update(params)
        return "99999", {"ok": True, "jobid": "99999", "step_id": step_id}

    monkeypatch.setattr(fasrc_jobs, "submit_sbatch_script", capture_submit)

    response = client.post(SYNTH, data=_synth_form(
        n_cpus=4, memory="8G", time_limit="01:00:00"))

    assert response.status_code == 200, response.get_json()
    assert submitted["_joint_galaxy_population_fingerprint"] == "j" * 64
    assert submitted["_star_prior_fingerprint"] == "s" * 64
    assert not any(key.startswith("_vis_noise") for key in submitted)


def test_submit_refuses_when_disconnected(client, monkeypatch):
    monkeypatch.setattr(STATE, "ssh", None)
    r = client.post(SYNTH, data={})
    assert r.status_code == 503
    assert r.get_json() == FASRC_OFFLINE_PAYLOAD


def test_legacy_submit_endpoint_is_gone(client):
    assert client.post("/api/fasrc/submit", data={}).status_code == 404


def test_queued_legacy_synthetic_spec_is_promoted_as_the_step(
        fake_remote, client, job_config_file):
    """A spec queued by the removed ``/api/fasrc/submit`` (``kind`` =
    ``synthetic``, no resource fields) still submits, as the
    ``synthetic_generate`` step with its default resources."""
    bin_dir = fake_remote["bin_dir"]
    (bin_dir / "squeue").write_text("#!/usr/bin/env bash\nexit 0\n")
    os.chmod(bin_dir / "squeue", 0o755)
    fasrc_queue.QUEUE.enqueue(
        {"kind": "synthetic", "step": "gen_convolve",
         "form": {"n_train": "7", "n_valid": "1", "n_test": "1",
                  "image_size": "60"}},
        "legacy")

    r = client.get("/api/fasrc/current-submission")

    assert r.status_code == 200, r.get_json()
    assert fasrc_queue.QUEUE.public()["count"] == 0
    assert not fasrc_queue.QUEUE.halted, fasrc_queue.QUEUE.halted_reason
    row = fake_remote["db"].get("99999")
    assert row is not None
    assert json.loads(row["params_json"])["step_id"] == "synthetic_generate"
    body = sorted((fake_remote["repo"] / "logs" / "jobs").glob("*.sh"))[0].read_text()
    assert "#SBATCH --cpus-per-task=16" in body       # the step default


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


def test_git_pull_flags_env_update_when_environment_yml_changed(fake_remote, client):
    """When the pull's diff includes ``environment.yml``, the response
    sets ``env_update_needed: True`` so the UI auto-runs mamba env update."""
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
# Checkpoint auto-mirror
# ---------------------------------------------------------------------------

def test_mirror_trigger_rsyncs_remote_ckpts(fake_remote, client, tmp_path):
    # Pretend ensemble training wrote a member on FASRC — since the
    # ensemble-only refactor the mirror pulls the remote ENSEMBLE dir
    # (sibling of ckpt_dir), so members land under member_NN/ locally.
    member = fake_remote["ckpt_dir"].parent / "ensemble" / "member_00"
    member.mkdir(parents=True)
    (member / "ckpt-12345.h5").write_bytes(b"hello world")
    (member / "training_log.jsonl").write_text('{"step": 1}\n')

    r = client.post("/api/fasrc/mirror/trigger")
    assert r.status_code == 200

    mirror = Path(fake_remote["cfg"].local_ckpt_mirror)
    assert (mirror / "member_00" / "ckpt-12345.h5").read_bytes() == b"hello world"
    assert (mirror / "member_00" / "training_log.jsonl").exists()


def test_mirror_status_reflects_last_sync(fake_remote, client):
    ens = fake_remote["ckpt_dir"].parent / "ensemble"
    ens.mkdir(exist_ok=True)
    (ens / "a.bin").write_bytes(b"x")
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
    assert r.status_code == 503
    assert r.get_json() == FASRC_OFFLINE_PAYLOAD


def _env_update_shims(fake_remote, *, mamba_exit: int = 0) -> None:
    bin_dir = fake_remote["bin_dir"]
    (bin_dir / "module").write_text(
        "#!/usr/bin/env bash\necho \"module $*\"\n"
    )
    (bin_dir / "mamba").write_text(textwrap.dedent(f"""\
        #!/usr/bin/env bash
        echo "mamba argv: $*"
        # Read+echo stdin so we can verify `yes |` is feeding it.
        head -3 || true
        echo "Proceed ([y]/n)? y"
        echo "Updated 0 packages."
        exit {mamba_exit}
    """))
    for shim in ("module", "mamba"):
        os.chmod(bin_dir / shim, 0o755)
    # Need a placeholder environment.yml inside the fake repo so the
    # `-f environment.yml` reference resolves under our local cwd.
    (fake_remote["repo"] / "environment.yml").write_text(
        "name: EuclidPolishEnv\nchannels:\n  - conda-forge\n"
        "dependencies:\n  - python=3.12\n",
    )


def _finished_job(client, job_id: str, timeout: float = 10.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").get_json()
        if job["status"] != "running":
            return job
        time.sleep(0.02)
    raise AssertionError(f"env-update job {job_id} never finished")


def test_env_update_runs_as_a_local_job_streaming_into_its_log(fake_remote, client):
    """``POST /api/fasrc/env-update`` routes the `module load python + mamba
    env update` pipeline through the active SSH session inside a local job
    (kind ``fasrc-env-update``) whose log receives the remote output line by
    line. We stand in fake `module` / `mamba` shims that print recognisable
    lines, then assert the log delivered them in order."""
    _env_update_shims(fake_remote)

    r = client.post("/api/fasrc/env-update")
    assert r.status_code == 200, r.get_json()
    body = r.get_json()
    assert body["ok"] is True

    job = _finished_job(client, body["job_id"])
    assert job["kind"] == "fasrc-env-update"
    assert job["status"] == "done", job["log"]
    log = job["log"]
    # Header lines from the job itself.
    assert "$ module load python" in log
    # Output captured from the fake `module` and `mamba` shims, in order.
    assert log.index("module load python") < log.index("mamba argv:")
    assert log.index("mamba argv:") < log.index("Updated 0 packages")
    assert "__EP_EXIT__" not in log
    assert job["result"] == {"exit_code": 0, "lines": job["result"]["lines"]}
    assert job["result"]["lines"] >= 3


def test_env_update_job_fails_when_the_remote_update_fails(fake_remote, client):
    _env_update_shims(fake_remote, mamba_exit=3)

    body = client.post("/api/fasrc/env-update").get_json()
    job = _finished_job(client, body["job_id"])

    assert job["status"] == "failed"
    assert "exit code 3" in job["error"]
    assert "Updated 0 packages" in job["log"]


def _write_mamba(fake_remote, body: str) -> None:
    mamba = fake_remote["bin_dir"] / "mamba"
    mamba.write_text("#!/usr/bin/env bash\n" + textwrap.dedent(body))
    os.chmod(mamba, 0o755)


def _silent_mamba(fake_remote, pid_file: Path) -> None:
    """A mamba that prints one line, then solves silently for 30 s."""
    _env_update_shims(fake_remote)
    _write_mamba(fake_remote, f"""\
        echo $$ > {pid_file}
        echo "Resolving environment"
        sleep 30
        echo "Updated 0 packages."
    """)


def _wait_for_log(client, job_id: str, needle: str, timeout: float = 10.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").get_json()
        if needle in (job["log"] or ""):
            return job
        assert job["status"] == "running", job
        time.sleep(0.02)
    raise AssertionError(f"{needle!r} never reached the log of job {job_id}")


def _alive(pid: int) -> bool:
    """True while ``pid`` runs (a zombie awaiting its reaper counts as gone)."""
    out = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)],
                         capture_output=True, text=True).stdout.strip()
    return bool(out) and not out.startswith("Z")


class _ChannelDropSession(LocalSSHSession):
    """Closing a stream only drops the channel, like the local ``ssh`` client
    exiting: the "remote" processes get no signal, only a closed stdout
    (no pty, so no SIGHUP either)."""

    def __init__(self, cwd: str, env: dict | None = None) -> None:
        super().__init__(cwd=cwd, env=env)
        self.procs: list[subprocess.Popen] = []

    def stream(self, cmd):
        proc = subprocess.Popen(
            ["bash", "-c", cmd], cwd=self.cwd, env=self._merged_env(),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, start_new_session=True,
        )
        self.procs.append(proc)
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                yield line.rstrip("\n")
        finally:
            proc.stdout.close()


def test_env_update_cancel_is_honoured_while_mamba_is_silent(
        fake_remote, client, monkeypatch, tmp_path):
    """A mamba solve can print nothing for minutes; the remote heartbeat
    keeps the job ticking so a cancel lands within one heartbeat, not at
    mamba's next output line."""
    monkeypatch.setattr(fasrc_routes, "_ENV_UPDATE_HEARTBEAT_S", 0.2)
    _silent_mamba(fake_remote, tmp_path / "mamba.pid")

    job_id = client.post("/api/fasrc/env-update").get_json()["job_id"]
    _wait_for_log(client, job_id, "Resolving environment")
    started = time.monotonic()
    assert client.post(f"/api/jobs/{job_id}/cancel").get_json() == {"ok": True}

    job = _finished_job(client, job_id, timeout=5.0)
    assert job["status"] == "cancelled"
    assert time.monotonic() - started < 5.0
    assert "Updated 0 packages" not in job["log"]
    assert "__EP_" not in job["log"]


def test_env_update_remote_side_dies_when_the_channel_closes(
        fake_remote, client, monkeypatch, tmp_path):
    """Cancel closes the stream; with no pty the remote processes get no
    signal, so the remote heartbeat watchdog notices its write failing and
    kills the remote process group (mamba included)."""
    monkeypatch.setattr(fasrc_routes, "_ENV_UPDATE_HEARTBEAT_S", 0.2)
    pid_file = tmp_path / "mamba.pid"
    _silent_mamba(fake_remote, pid_file)
    session = _ChannelDropSession(cwd=STATE.ssh.cwd, env=STATE.ssh.env)
    monkeypatch.setattr(STATE, "ssh", session)

    job_id = client.post("/api/fasrc/env-update").get_json()["job_id"]
    try:
        _wait_for_log(client, job_id, "Resolving environment")
        mamba_pid = int(pid_file.read_text())
        assert client.post(f"/api/jobs/{job_id}/cancel").get_json() == {"ok": True}
        assert _finished_job(client, job_id, timeout=5.0)["status"] == "cancelled"

        (remote_shell,) = session.procs
        remote_shell.wait(timeout=5)   # the whole remote group was killed
        deadline = time.time() + 5
        while _alive(mamba_pid) and time.time() < deadline:
            time.sleep(0.05)
        assert not _alive(mamba_pid)
    finally:
        for proc in session.procs:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=5)


def test_env_update_heartbeat_never_reaches_the_log(fake_remote, client, monkeypatch):
    """Heartbeat lines are filtered, even one glued to a partial output line
    (mamba's prompt has no newline)."""
    monkeypatch.setattr(fasrc_routes, "_ENV_UPDATE_HEARTBEAT_S", 0.1)
    _env_update_shims(fake_remote)
    _write_mamba(fake_remote, """\
        printf 'Proceed ([y]/n)? '
        sleep 0.5
        echo y
        echo "Updated 0 packages."
    """)

    job_id = client.post("/api/fasrc/env-update").get_json()["job_id"]
    job = _finished_job(client, job_id)

    assert job["status"] == "done", job["log"]
    assert "__EP_" not in job["log"]
    assert "Proceed ([y]/n)?" in job["log"]
    assert "Updated 0 packages" in job["log"]


def test_env_update_heartbeat_does_not_hold_the_stream_open(
        fake_remote, client, monkeypatch):
    """A quick update ends as soon as mamba does: the heartbeat (and its
    ``sleep``) must not keep the channel open for a full period."""
    monkeypatch.setattr(fasrc_routes, "_ENV_UPDATE_HEARTBEAT_S", 30)
    _env_update_shims(fake_remote)

    started = time.monotonic()
    job_id = client.post("/api/fasrc/env-update").get_json()["job_id"]
    job = _finished_job(client, job_id, timeout=10.0)

    assert job["status"] == "done", job["log"]
    assert time.monotonic() - started < 5.0


def test_env_update_refuses_when_disconnected(client, monkeypatch):
    monkeypatch.setattr(STATE, "ssh", None)
    r = client.post("/api/fasrc/env-update")
    assert r.status_code == 503
    assert r.get_json() == FASRC_OFFLINE_PAYLOAD


def test_env_update_is_not_reachable_with_get(client):
    assert client.get("/api/fasrc/env-update").status_code == 405


def test_extend_time_endpoint_remains_removed(client):
    """The retired mid-run time extender must not reappear as a dead API."""
    response = client.post(
        "/api/fasrc/extend-time", data={"jobid": "99999", "hours": "1"}
    )

    assert response.status_code == 404


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
    # Seed a finished run under the fake remote's eval_results dir.
    remote_run = fake_remote["data_dir"] / "eval_results" / "lenses"
    (remote_run / "obj1").mkdir(parents=True)
    (remote_run / "manifest.csv").write_text(
        "id,ra,dec,ok\nobj1,1.0,2.0,True\n")
    (remote_run / "obj1" / "eye.png").write_bytes(b"\x89PNGfake")

    # Point the local gallery at an empty tmp dir.
    local_eval = tmp_path / "local_eval"
    monkeypatch.setattr(Config, "EVAL_RESULTS_DIR", str(local_eval))

    r = client.post("/api/evaluation/sync", data={"confirm": "1"})
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
    assert r.status_code == 503
    assert r.get_json() == FASRC_OFFLINE_PAYLOAD
