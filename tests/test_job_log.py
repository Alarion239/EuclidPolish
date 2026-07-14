"""Unit tests for the CSV submission log."""

from __future__ import annotations

import csv
import time
from pathlib import Path

import pytest

from euclid_polish.observability import JobLog, JobRecord


def _rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Construction / schema
# ---------------------------------------------------------------------------

class TestConstruction:

    def test_creates_file_with_header(self, tmp_path):
        p = tmp_path / "log.csv"
        JobLog(str(p))
        assert p.exists()
        # Header row written, no data rows.
        with open(p, encoding="utf-8") as f:
            assert "jobid" in f.readline()

    def test_idempotent_construction_does_not_clobber(self, tmp_path):
        p = tmp_path / "log.csv"
        log = JobLog(str(p))
        log.record_submission(JobRecord(jobid="42"))
        # A fresh instance must NOT wipe the row.
        JobLog(str(p))
        assert _rows(p) and _rows(p)[0]["jobid"] == "42"


# ---------------------------------------------------------------------------
# Submission append
# ---------------------------------------------------------------------------

class TestRecordSubmission:

    def test_append_writes_all_columns(self, tmp_path):
        p = tmp_path / "log.csv"
        log = JobLog(str(p))
        log.record_submission(JobRecord(
            jobid="12345", step_id="extract_psf", label="step 2",
            partition="shared", req_cpus=1, req_gpus=0,
            req_memory="8G", req_time_limit="0:10:00",
            params_json='{"n_stars":200}',
            script_path="/repo/logs/hst_pipeline/x.sh",
            log_path="/repo/logs/hst_pipeline/x.out",
            err_path="/repo/logs/hst_pipeline/x.err",
            events_path="/repo/logs/hst_pipeline/x.events",
        ))
        rows = _rows(p)
        assert len(rows) == 1
        r = rows[0]
        assert r["jobid"] == "12345"
        assert r["step_id"] == "extract_psf"
        assert r["req_memory"] == "8G"
        assert r["params_json"] == '{"n_stars":200}'
        # ``submitted_at`` was auto-filled.
        assert r["submitted_at"]

    def test_submitted_at_preserved_if_provided(self, tmp_path):
        p = tmp_path / "log.csv"
        log = JobLog(str(p))
        log.record_submission(JobRecord(jobid="1",
                                        submitted_at="2026-05-26T12:00:00Z"))
        assert _rows(p)[0]["submitted_at"] == "2026-05-26T12:00:00Z"

    def test_two_submissions_two_rows(self, tmp_path):
        p = tmp_path / "log.csv"
        log = JobLog(str(p))
        log.record_submission(JobRecord(jobid="1"))
        log.record_submission(JobRecord(jobid="2"))
        assert [r["jobid"] for r in _rows(p)] == ["1", "2"]


# ---------------------------------------------------------------------------
# Post-mortem update
# ---------------------------------------------------------------------------

class TestRecordPostMortem:

    def test_accounting_attempts_are_throttled(self, tmp_path):
        log = JobLog(str(tmp_path / "log.csv"))
        log.record_submission(JobRecord(jobid="42"))
        assert log.accounting_attempt_due("42") is True
        assert log.mark_accounting_attempt("42", at=time.time()) is True
        assert log.accounting_attempt_due("42", min_interval=60.0) is False
        assert log.accounting_attempt_due("42", min_interval=0.0) is True

    def test_updates_existing_row(self, tmp_path):
        p = tmp_path / "log.csv"
        log = JobLog(str(p))
        log.record_submission(JobRecord(jobid="42", label="x"))
        ok = log.record_post_mortem("42", {
            "state":           "COMPLETED",
            "exit_code":       "0:0",
            "elapsed_seconds": 123.4,
            "max_rss_mb":      512.0,
            "alloc_cpus":      4,
        })
        assert ok is True
        r = log.get("42")
        assert r["state"] == "COMPLETED"
        assert r["exit_code"] == "0:0"
        assert r["elapsed_seconds"] == "123.4"
        assert r["max_rss_mb"] == "512.0"
        assert r["alloc_cpus"] == "4"
        # Untouched submission fields stay put.
        assert r["label"] == "x"

    def test_unknown_jobid_returns_false(self, tmp_path):
        log = JobLog(str(tmp_path / "log.csv"))
        log.record_submission(JobRecord(jobid="1"))
        assert log.record_post_mortem("999", {"state": "COMPLETED"}) is False

    def test_missing_keys_leave_cells_alone(self, tmp_path):
        log = JobLog(str(tmp_path / "log.csv"))
        log.record_submission(JobRecord(jobid="1"))
        log.record_post_mortem("1", {"state": "COMPLETED"})
        # exit_code was not in the stats dict — must remain blank, not get
        # set to "None"/"null".
        r = log.get("1")
        assert r["state"] == "COMPLETED"
        assert r["exit_code"] == ""

    def test_none_values_skipped(self, tmp_path):
        log = JobLog(str(tmp_path / "log.csv"))
        log.record_submission(JobRecord(jobid="1", req_cpus=4))
        log.record_post_mortem("1", {"alloc_cpus": None,
                                     "state": "COMPLETED"})
        r = log.get("1")
        # None means "no data, don't overwrite" — alloc_cpus stays blank.
        assert r["alloc_cpus"] == ""
        assert r["state"] == "COMPLETED"

    def test_only_target_row_changes(self, tmp_path):
        log = JobLog(str(tmp_path / "log.csv"))
        log.record_submission(JobRecord(jobid="1", label="one"))
        log.record_submission(JobRecord(jobid="2", label="two"))
        log.record_submission(JobRecord(jobid="3", label="three"))
        log.record_post_mortem("2", {"state": "FAILED"})
        rows = log.list_all()
        assert [r["jobid"] for r in rows] == ["1", "2", "3"]
        assert rows[0]["state"] == ""
        assert rows[1]["state"] == "FAILED"
        assert rows[2]["state"] == ""


# ---------------------------------------------------------------------------
# Schema migration
# ---------------------------------------------------------------------------

class TestSchemaMigration:

    def test_old_file_with_subset_of_columns_is_rewritten(self, tmp_path):
        """A CSV from a previous JobRecord version (fewer columns) must
        keep its rows but gain the new blank columns on next open."""
        p = tmp_path / "log.csv"
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["jobid", "label", "submitted_at"])
            w.writerow(["7", "old", "2026-01-01T00:00:00Z"])

        log = JobLog(str(p))
        rows = log.list_all()
        assert len(rows) == 1
        # New columns present, old data preserved.
        assert rows[0]["jobid"] == "7"
        assert rows[0]["label"] == "old"
        assert "exit_code" in rows[0]
        assert rows[0]["exit_code"] == ""


# ---------------------------------------------------------------------------
# Get / list
# ---------------------------------------------------------------------------

class TestQueries:

    def test_get_returns_none_for_missing(self, tmp_path):
        log = JobLog(str(tmp_path / "log.csv"))
        assert log.get("nope") is None

    def test_list_all_returns_in_submission_order(self, tmp_path):
        log = JobLog(str(tmp_path / "log.csv"))
        for i in range(5):
            log.record_submission(JobRecord(jobid=str(i)))
        assert [r["jobid"] for r in log.list_all()] == ["0", "1", "2", "3", "4"]


# ---------------------------------------------------------------------------
# Per-step history + match
# ---------------------------------------------------------------------------

class TestHistoryForStep:

    def test_filters_by_step_id_and_orders_newest_first(self, tmp_path):
        log = JobLog(str(tmp_path / "log.csv"))
        log.record_submission(JobRecord(
            jobid="1", step_id="extract_psf",
            submitted_at="2026-05-26T10:00:00Z",
        ))
        log.record_submission(JobRecord(
            jobid="2", step_id="extract_psf",
            submitted_at="2026-05-26T12:00:00Z",
        ))
        log.record_submission(JobRecord(
            jobid="3", step_id="download",
            submitted_at="2026-05-26T11:00:00Z",
        ))
        rows = log.history_for_step("extract_psf")
        # Only extract_psf rows, newest first.
        assert [r["jobid"] for r in rows] == ["2", "1"]

    def test_empty_when_no_runs_for_step(self, tmp_path):
        log = JobLog(str(tmp_path / "log.csv"))
        assert log.history_for_step("extract_psf") == []


class TestLatestMatch:

    def test_picks_completed_match_over_failed(self, tmp_path):
        """When task params match more than one past run, the most recent
        run that COMPLETED wins — never inherit a failed/OOM run's
        resources."""
        log = JobLog(str(tmp_path / "log.csv"))
        log.record_submission(JobRecord(
            jobid="1", step_id="extract_psf",
            submitted_at="2026-05-26T10:00:00Z",
            params_json='{"n_stars":200}',
        ))
        log.record_post_mortem("1", {"state": "COMPLETED"})
        log.record_submission(JobRecord(
            jobid="2", step_id="extract_psf",
            submitted_at="2026-05-26T12:00:00Z",
            params_json='{"n_stars":200}',
        ))
        log.record_post_mortem("2", {"state": "OUT_OF_MEMORY"})

        m = log.latest_match("extract_psf", '{"n_stars":200}')
        assert m is not None and m["jobid"] == "1"

    def test_falls_back_to_any_state_when_no_completed(self, tmp_path):
        """If no past run completed, the latest failed/cancelled one is
        better than ``None`` — gives the user a 'resubmit-as-is' hint."""
        log = JobLog(str(tmp_path / "log.csv"))
        log.record_submission(JobRecord(
            jobid="1", step_id="extract_psf",
            submitted_at="2026-05-26T10:00:00Z",
            params_json='{"n_stars":200}',
        ))
        log.record_post_mortem("1", {"state": "FAILED"})
        log.record_submission(JobRecord(
            jobid="2", step_id="extract_psf",
            submitted_at="2026-05-26T12:00:00Z",
            params_json='{"n_stars":200}',
        ))
        log.record_post_mortem("2", {"state": "OUT_OF_MEMORY"})

        m = log.latest_match("extract_psf", '{"n_stars":200}')
        assert m is not None and m["jobid"] == "2"

    def test_returns_none_for_no_matching_task_params(self, tmp_path):
        log = JobLog(str(tmp_path / "log.csv"))
        log.record_submission(JobRecord(
            jobid="1", step_id="extract_psf",
            params_json='{"n_stars":200}',
        ))
        # Different task params → no match.
        assert log.latest_match("extract_psf", '{"n_stars":500}') is None

    def test_returns_none_for_empty_params(self, tmp_path):
        """A step with no task params still records rows, but the match
        lookup must not return them under the empty-params key — that
        would leak resources across unrelated empty-param steps."""
        log = JobLog(str(tmp_path / "log.csv"))
        log.record_submission(JobRecord(
            jobid="1", step_id="kernel", params_json="",
        ))
        assert log.latest_match("kernel", "") is None

    def test_isolates_per_step(self, tmp_path):
        log = JobLog(str(tmp_path / "log.csv"))
        log.record_submission(JobRecord(
            jobid="1", step_id="extract_psf",
            params_json='{"n_stars":200}',
        ))
        log.record_post_mortem("1", {"state": "COMPLETED"})
        # Same params_json, different step.
        log.record_submission(JobRecord(
            jobid="2", step_id="download",
            params_json='{"n_stars":200}',
        ))
        m = log.latest_match("extract_psf", '{"n_stars":200}')
        assert m["jobid"] == "1"
        m = log.latest_match("download", '{"n_stars":200}')
        assert m["jobid"] == "2"
