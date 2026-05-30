"""Login-node remote-exec command builder (fasrc_jobs.run_remote_python).

The brightest-N catalog query runs on the FASRC login node over SSH (not a
SLURM job) so stars.csv lands on the shared netscratch $DATA_DIR. These
tests pin the shell command that ``build_remote_python_command`` emits.
"""

from __future__ import annotations

import shlex

from euclid_polish.web import fasrc_config, fasrc_jobs


def _cfg() -> fasrc_config.FasrcConfig:
    return fasrc_config.FasrcConfig(
        repo_path="/n/holylabs/REPO",
        data_dir="/n/netscratch/DATA",
        ckpt_dir="/n/netscratch/CKPT",
        conda_env_path="/n/holylabs/ENV",
    )


def test_command_is_login_shell_and_well_formed():
    cmd = fasrc_jobs.build_remote_python_command(
        _cfg(), ["scripts/query_brightest_stars.py", "--num-stars", "500"],
    )
    # Login shell so the user's conda init runs.
    assert cmd.startswith("bash -lc ")
    # The inner command is a single shell-quoted token.
    inner = shlex.split(cmd)[2]
    assert inner.startswith("cd /n/holylabs/REPO &&")
    # netscratch paths exported so the remote script reads/writes the
    # shared data dir, and the env is activated before python runs.
    assert "export EUCLID_POLISH_DATA_DIR=/n/netscratch/DATA" in inner
    assert "mamba activate /n/holylabs/ENV" in inner
    assert inner.rstrip().endswith(
        "python -u scripts/query_brightest_stars.py --num-stars 500"
    )


def test_argv_is_shell_quoted():
    # A pathological arg must not break out of the command.
    cmd = fasrc_jobs.build_remote_python_command(
        _cfg(), ["scripts/x.py", "--label", "a b; rm -rf /"],
    )
    inner = shlex.split(cmd)[2]
    # The dangerous token survives as one quoted argument, not as a
    # second command.
    assert "'a b; rm -rf /'" in inner


def test_run_remote_python_delegates_to_ssh():
    captured = {}

    class _FakeSSH:
        def run(self, cmd, timeout=60):
            captured["cmd"] = cmd
            captured["timeout"] = timeout
            return (0, "Added 5 stars", "")

    rc, out, err = fasrc_jobs.run_remote_python(
        _FakeSSH(), cfg=_cfg(), argv=["scripts/query_brightest_stars.py"],
        timeout=123,
    )
    assert rc == 0 and out == "Added 5 stars"
    assert captured["timeout"] == 123
    assert captured["cmd"].startswith("bash -lc ")
