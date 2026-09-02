"""Submit-time guard: a job whose entry script is missing from the FASRC
repo checkout fails at submit with an actionable message (instead of
starting on SLURM and dying on a cryptic python ENOENT)."""

from __future__ import annotations

from euclid_polish.web import fasrc_config
from euclid_polish.web.fasrc_jobs import submit_sbatch_script
from euclid_polish.web.fasrc_pipeline import REGISTRY, StepResources


class _FakeSSH:
    """Records commands; ``rc_map`` keys are substrings matched in order."""

    def __init__(self, stat_rc: int):
        self.stat_rc = stat_rc
        self.commands: list[str] = []

    def run(self, cmd: str, timeout: int = 0):
        self.commands.append(cmd)
        if cmd.startswith("stat "):
            return self.stat_rc, "", ""
        if cmd.startswith("mkdir -p "):
            return 0, "", ""
        raise AssertionError(f"unexpected command past the guard: {cmd!r}")


def _built(step_id="psf_rotation_pool", params=None):
    step = REGISTRY.get(step_id)
    return step.build_sbatch_body(
        params=params or {}, resources=step.defaults,
        cfg=fasrc_config.load(), label="test")


def test_render_emits_repo_relative_entry():
    built = _built()
    assert built["entry"] == "scripts/pregenerate_psf_rotations.py"


def test_missing_entry_fails_before_sbatch():
    ssh = _FakeSSH(stat_rc=1)
    slurm_id, payload = submit_sbatch_script(
        ssh, cfg=fasrc_config.load(), built=_built(), label="test",
        params={})
    assert slurm_id is None
    assert not payload["ok"]
    assert "git pull" in payload["error"]
    assert "pregenerate_psf_rotations.py" in payload["error"]
    # exactly one remote command ran (the stat) — no script write, no sbatch
    assert len(ssh.commands) == 1 and ssh.commands[0].startswith("stat ")


def test_large_payload_requires_streaming_ssh_writer():
    ssh = _FakeSSH(stat_rc=0)
    built = _built()
    built["payload_files"] = {
        "logs/jobs/frozen-population.json": "x" * 150_000,
    }

    slurm_id, payload = submit_sbatch_script(
        ssh, cfg=fasrc_config.load(), built=built, label="test", params={},
    )

    assert slurm_id is None
    assert not payload["ok"]
    assert "cannot stream large job payload" in payload["error"]
    assert not any("sbatch" in command for command in ssh.commands)
