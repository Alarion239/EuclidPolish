"""Unit tests for :class:`euclid_polish.observability.Reporter`.

The producer side of the structured-status pipeline. We exercise:

  * no-op behaviour when the events path is unset,
  * JSON line shape per event kind,
  * monotonic timestamps,
  * thread safety under concurrent writers,
  * stderr echo policy (stage / warn / error echo, step is silent).
"""

from __future__ import annotations

import json
import os
import threading

import pytest

from euclid_polish.observability import ENV_EVENTS_PATH, Reporter


@pytest.fixture
def events_file(tmp_path):
    return tmp_path / "job.events"


def _read_events(path) -> list[dict]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


# ---------------------------------------------------------------------------
# No-op when unset
# ---------------------------------------------------------------------------

class TestNoOp:

    def test_no_path_means_silent_writes(self, capfd):
        """A reporter constructed with no events path must accept every
        emit call without raising — that's the contract for running the
        same script outside of SLURM."""
        r = Reporter(events_path=None)
        r.set_stage("local run")
        r.set_step(1, 10)
        r.warn("benign")
        r.error("annoying")
        # The events file shouldn't have been touched (there is no path).
        # Stderr echo still happens — that's intentional, scripts run
        # interactively should still see warnings on the console.

    def test_from_env_missing_var_returns_noop(self, monkeypatch, capfd):
        monkeypatch.delenv(ENV_EVENTS_PATH, raising=False)
        r = Reporter.from_env()
        assert r.events_path is None
        r.set_stage("x")  # should not raise

    def test_from_env_with_events_silences_tqdm(self, monkeypatch, tmp_path):
        """silence_tqdm() sets TQDM_DISABLE=1; scripts that drive the WebUI
        progress bar via Reporter should call it explicitly before from_env().
        (monkeypatch.delenv restores the prior state at teardown so the flag
        never leaks into other tests.)"""
        monkeypatch.delenv("TQDM_DISABLE", raising=False)
        Reporter.silence_tqdm()
        assert os.environ.get("TQDM_DISABLE") == "1"

    def test_from_env_no_events_leaves_tqdm_untouched(self, monkeypatch):
        monkeypatch.delenv(ENV_EVENTS_PATH, raising=False)
        monkeypatch.delenv("TQDM_DISABLE", raising=False)
        Reporter.from_env()
        assert "TQDM_DISABLE" not in os.environ

    def test_from_env_respects_explicit_tqdm_disable(self, monkeypatch, tmp_path):
        """An explicit TQDM_DISABLE in the environment wins over the default."""
        monkeypatch.setenv(ENV_EVENTS_PATH, str(tmp_path / "j.events"))
        monkeypatch.setenv("TQDM_DISABLE", "0")
        Reporter.from_env()
        assert os.environ["TQDM_DISABLE"] == "0"


# ---------------------------------------------------------------------------
# Event shape
# ---------------------------------------------------------------------------

class TestEventShape:

    def test_set_stage_writes_one_jsonl_line(self, events_file, capfd):
        r = Reporter(events_path=str(events_file))
        r.set_stage("Downloading HLSP tiles")
        events = _read_events(events_file)
        assert len(events) == 1
        e = events[0]
        assert e["kind"] == "stage"
        assert e["value"] == "Downloading HLSP tiles"
        assert isinstance(e["ts"], float) and e["ts"] > 0

    def test_set_step_value_has_current_total_label(self, events_file, capfd):
        r = Reporter(events_path=str(events_file))
        r.set_step(3, 25, label="tile 3")
        e = _read_events(events_file)[0]
        assert e["kind"] == "step"
        assert e["value"] == {"current": 3, "total": 25, "label": "tile 3"}

    def test_set_step_label_defaults_to_empty(self, events_file, capfd):
        Reporter(events_path=str(events_file)).set_step(1, 5)
        e = _read_events(events_file)[0]
        assert e["value"]["label"] == ""

    def test_warn_and_error_are_strings(self, events_file, capfd):
        r = Reporter(events_path=str(events_file))
        r.warn("checksum mismatch")
        r.error("missing tile")
        kinds = [(e["kind"], e["value"]) for e in _read_events(events_file)]
        assert kinds == [("warn", "checksum mismatch"),
                         ("error", "missing tile")]

    def test_timestamps_are_monotonic_within_one_reporter(
        self, events_file, capfd,
    ):
        r = Reporter(events_path=str(events_file))
        for i in range(5):
            r.set_step(i, 5)
        events = _read_events(events_file)
        ts = [e["ts"] for e in events]
        assert ts == sorted(ts), f"timestamps not monotonic: {ts}"


# ---------------------------------------------------------------------------
# Stderr echo policy
# ---------------------------------------------------------------------------

class TestStderrEcho:

    def test_stage_echoes_to_stderr(self, events_file, capfd):
        Reporter(events_path=str(events_file)).set_stage("Phase 1")
        _, err = capfd.readouterr()
        assert "STAGE: Phase 1" in err

    def test_warn_echoes_with_prefix(self, events_file, capfd):
        Reporter(events_path=str(events_file)).warn("something off")
        _, err = capfd.readouterr()
        assert "WARN: something off" in err

    def test_error_echoes_with_prefix(self, events_file, capfd):
        Reporter(events_path=str(events_file)).error("kaboom")
        _, err = capfd.readouterr()
        assert "ERROR: kaboom" in err

    def test_step_echo_is_rate_limited(self, events_file, capfd):
        """``set_step`` echoes once on the first call (so a quick "did
        the loop start?" check works without a 30 s pause), then stays
        quiet until ``STEP_ECHO_INTERVAL_S`` has elapsed. A tight 100-
        iteration loop should produce exactly one ``STEP:`` line on
        stderr while still appending all 100 events to the JSONL.
        """
        r = Reporter(events_path=str(events_file))
        for i in range(100):
            r.set_step(i, 100, label="train")
        _, err = capfd.readouterr()
        # Exactly one echo (the first one) — the next is gated by
        # ``STEP_ECHO_INTERVAL_S`` (30 s) which the loop won't outrun.
        step_lines = [ln for ln in err.splitlines() if ln.startswith("STEP:")]
        assert len(step_lines) == 1
        # And the echoed line carries the count, percentage, and label.
        assert "0/100" in step_lines[0] or "1/100" in step_lines[0]
        assert "train" in step_lines[0]
        # The JSONL file still got every event — the rate-limit is on
        # the echo, not on the structured stream.
        with open(events_file) as f:
            lines = [ln for ln in f.read().splitlines() if ln.strip()]
        assert len(lines) == 100


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------

class TestConcurrency:

    def test_threaded_writers_do_not_interleave_lines(
        self, events_file, capfd,
    ):
        """Under threaded concurrent emits, every event must land as a
        single valid JSON line — no torn writes, no mixed-content
        lines."""
        r = Reporter(events_path=str(events_file))
        n_threads = 8
        per_thread = 50

        def worker(tag: int) -> None:
            for i in range(per_thread):
                r.warn(f"t{tag}-i{i}")

        threads = [threading.Thread(target=worker, args=(t,))
                   for t in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Every line should parse cleanly, and every (tag, i) pair we
        # emitted should be present exactly once.
        events = _read_events(events_file)
        assert len(events) == n_threads * per_thread
        seen = {e["value"] for e in events}
        expected = {f"t{t}-i{i}" for t in range(n_threads)
                    for i in range(per_thread)}
        assert seen == expected
