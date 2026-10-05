"""``GET /api/fasrc/steps/status`` builds every step's ``last_params`` from one
read of the job ledger rather than one ``history_for_step`` per step."""

from __future__ import annotations

import json

import pytest

from euclid_polish.observability import JobLog, JobRecord
from euclid_polish.web import fasrc_jobs
from euclid_polish.web.app import create_app
from euclid_polish.web.fasrc_pipeline import REGISTRY

_READS = ("get", "get_many", "list_all", "history_for_step", "history_by_step",
          "latest_match")


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    log = JobLog(str(tmp_path / "log.csv"))
    monkeypatch.setattr(fasrc_jobs, "JOBLOG", log)
    return log


def _count_reads(monkeypatch, log: JobLog) -> dict[str, int]:
    calls = dict.fromkeys(_READS, 0)
    for name in _READS:
        real = getattr(JobLog, name, None)
        if real is None:
            continue

        def spy(*args, _name=name, _real=real, **kwargs):
            calls[_name] += 1
            return _real(log, *args, **kwargs)

        monkeypatch.setattr(log, name, spy)
    return calls


def test_steps_status_reads_the_ledger_once(ledger, monkeypatch):
    ledger.record_submission(JobRecord(
        jobid="1", step_id="euclid_query", submitted_at="2026-05-26T10:00:00Z",
        params_json=json.dumps({"num_stars": "111"})))
    ledger.record_post_mortem("1", {"state": "COMPLETED"})
    ledger.record_submission(JobRecord(
        jobid="2", step_id="euclid_query", submitted_at="2026-05-26T12:00:00Z",
        params_json=json.dumps({"num_stars": "222"})))
    ledger.record_post_mortem("2", {"state": "COMPLETED"})
    ledger.record_submission(JobRecord(
        jobid="3", step_id="euclid_query", submitted_at="2026-05-26T13:00:00Z",
        params_json=json.dumps({"num_stars": "333"})))
    ledger.record_post_mortem("3", {"state": "FAILED"})
    calls = _count_reads(monkeypatch, ledger)

    app = create_app()
    app.config["TESTING"] = True
    body = app.test_client().get("/api/fasrc/steps/status").get_json()

    assert sum(calls.values()) == 1, calls
    by_id = {step["step_id"]: step for step in body["steps"]}
    assert set(by_id) == {step.step_id for step in REGISTRY.all()}
    # The newest COMPLETED run wins (the FAILED one is newer but skipped).
    assert by_id["euclid_query"]["last_params"] == {"num_stars": 222}
    assert by_id["tng_grid"]["last_params"] is None
