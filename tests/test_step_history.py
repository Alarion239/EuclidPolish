"""End-to-end behaviour for the history-driven defaults flow.

Covers:

  * ``StepResources.from_form_strict`` — rejects blank resource fields
    with a clear message; no silent fallback to step.defaults.
  * ``/api/fasrc/hst/<step_id>/history`` — returns the full history
    for a step and the latest exact-match row for prefill.
  * ``/api/fasrc/hst/<step_id>/submit`` — rejects when resources are
    blank (the new strict path).
"""

from __future__ import annotations

from typing import Tuple

import pytest

from euclid_polish.observability import JobLog, JobRecord
from euclid_polish.web import fasrc_jobs
from euclid_polish.web.app import create_app
from euclid_polish.web.fasrc_pipeline import StepResources


# ---------------------------------------------------------------------------
# StepResources.from_form_strict
# ---------------------------------------------------------------------------

class TestFromFormStrict:

    def test_happy_path_returns_resources(self):
        r = StepResources.from_form_strict({
            "partition": "shared", "n_cpus": "4", "n_gpus": "0",
            "memory": "8G", "time_limit": "1:00:00",
        })
        assert r == StepResources(
            partition="shared", n_cpus=4, n_gpus=0,
            memory="8G", time_limit="1:00:00",
        )

    def test_zero_gpus_explicitly_set_is_accepted(self):
        """``n_gpus=0`` is the common CPU case; blank ≠ zero."""
        r = StepResources.from_form_strict({
            "partition": "shared", "n_cpus": "4", "n_gpus": "0",
            "memory": "8G", "time_limit": "1:00:00",
        })
        assert r.n_gpus == 0

    @pytest.mark.parametrize("missing_key", [
        "partition", "n_cpus", "n_gpus", "memory", "time_limit",
    ])
    def test_each_field_required(self, missing_key):
        form = {
            "partition": "shared", "n_cpus": "4", "n_gpus": "0",
            "memory": "8G", "time_limit": "1:00:00",
        }
        form[missing_key] = ""
        with pytest.raises(ValueError) as exc:
            StepResources.from_form_strict(form)
        assert missing_key in str(exc.value)

    def test_all_blank_lists_every_missing_field(self):
        with pytest.raises(ValueError) as exc:
            StepResources.from_form_strict({})
        msg = str(exc.value)
        for f in ("partition", "n_cpus", "n_gpus", "memory", "time_limit"):
            assert f in msg

    def test_non_numeric_n_cpus_raises(self):
        with pytest.raises(ValueError):
            StepResources.from_form_strict({
                "partition": "shared", "n_cpus": "abc", "n_gpus": "0",
                "memory": "8G", "time_limit": "1:00:00",
            })


# ---------------------------------------------------------------------------
# Flask: /api/fasrc/hst/<step_id>/history
# ---------------------------------------------------------------------------

@pytest.fixture
def app(tmp_path, monkeypatch):
    """Flask app with an isolated CSV log so tests don't see each other."""
    log = JobLog(str(tmp_path / "log.csv"))
    monkeypatch.setattr(fasrc_jobs, "JOBLOG", log)
    flask_app = create_app()
    flask_app.testing = True
    yield flask_app, log


class TestHistoryEndpoint:

    def test_returns_history_and_match_via_post(self, app):
        flask_app, log = app
        # Pre-populate the log with two extract_psf runs at n_stars=200,
        # one of which COMPLETED, and one unrelated download row.
        log.record_submission(JobRecord(
            jobid="1", step_id="extract_psf",
            submitted_at="2026-05-26T10:00:00Z",
            params_json='{"n_stars":"200"}',
            req_memory="8G", req_time_limit="0:10:00",
        ))
        log.record_post_mortem("1", {"state": "COMPLETED"})
        log.record_submission(JobRecord(
            jobid="2", step_id="extract_psf",
            submitted_at="2026-05-26T12:00:00Z",
            params_json='{"n_stars":"200"}',
            req_memory="16G", req_time_limit="0:20:00",
        ))
        log.record_post_mortem("2", {"state": "FAILED"})
        log.record_submission(JobRecord(
            jobid="9", step_id="download",
            params_json='{"n_tiles":"5"}',
        ))

        client = flask_app.test_client()
        r = client.post("/api/fasrc/hst/extract_psf/history",
                        data={"n_stars": "200"})
        assert r.status_code == 200
        body = r.get_json()
        assert body["ok"] is True
        assert body["step_id"] == "extract_psf"
        # Full history for the step (newest first).
        ids = [row["jobid"] for row in body["history"]]
        assert ids == ["2", "1"]
        # Match is the COMPLETED run, not the failed-but-newer one.
        assert body["match"]["jobid"] == "1"
        # Canonical task-params JSON is returned so the client can show
        # exactly what we matched against.
        assert body["task_params_json"] == '{"n_stars":"200"}'

    def test_no_history_match_is_null(self, app):
        flask_app, log = app
        log.record_submission(JobRecord(
            jobid="1", step_id="extract_psf",
            params_json='{"n_stars":"200"}',
        ))
        client = flask_app.test_client()
        r = client.post("/api/fasrc/hst/extract_psf/history",
                        data={"n_stars": "999"})
        body = r.get_json()
        assert body["match"] is None
        # History still listed (so the UI can show "you ran this with
        # different params" hints).
        assert len(body["history"]) == 1

    def test_unknown_step_id_returns_404(self, app):
        flask_app, _ = app
        r = flask_app.test_client().post(
            "/api/fasrc/hst/not_a_step/history", data={},
        )
        assert r.status_code == 404

    def test_get_method_accepts_query_string(self, app):
        """Curl-friendly GET path used during debugging."""
        flask_app, log = app
        log.record_submission(JobRecord(
            jobid="1", step_id="extract_psf",
            params_json='{"n_stars":"200"}',
        ))
        log.record_post_mortem("1", {"state": "COMPLETED"})
        r = flask_app.test_client().get(
            "/api/fasrc/hst/extract_psf/history?n_stars=200",
        )
        assert r.status_code == 200
        assert r.get_json()["match"]["jobid"] == "1"

    def test_meta_keys_in_form_dont_break_match(self, app):
        """``confirm``, ``label``, ``preset`` are submit-time meta —
        the history endpoint must strip them before matching, otherwise
        the prefill would never hit for a submitted run."""
        flask_app, log = app
        log.record_submission(JobRecord(
            jobid="1", step_id="extract_psf",
            params_json='{"n_stars":"200"}',
        ))
        log.record_post_mortem("1", {"state": "COMPLETED"})
        r = flask_app.test_client().post(
            "/api/fasrc/hst/extract_psf/history",
            data={"n_stars": "200", "confirm": "yes",
                  "label": "annotation", "preset": "custom"},
        )
        assert r.get_json()["match"]["jobid"] == "1"


# ---------------------------------------------------------------------------
# /api/fasrc/hst/<step_id>/submit  — rejects blank resources
# ---------------------------------------------------------------------------

class TestSubmitRejectsBlankResources:

    def test_blank_resources_400(self, app):
        flask_app, _ = app
        r = flask_app.test_client().post(
            "/api/fasrc/hst/extract_psf/submit",
            data={"confirm": "yes", "n_stars": "200"},
        )
        assert r.status_code == 400
        body = r.get_json()
        assert body["ok"] is False
        assert "resource field" in body["error"]
        # The error message must name the fields the user needs to fill.
        for f in ("partition", "n_cpus", "n_gpus", "memory", "time_limit"):
            assert f in body["error"]
