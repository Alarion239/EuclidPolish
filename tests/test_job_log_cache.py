"""JobLog memoises the parsed ledger: repeated reads parse the CSV once, a
change on disk (another process, another JobLog) is picked up through the
file's identity/mtime/size, JobLog's own writes refresh the cache, and every
returned row is a copy the caller may mutate freely."""

from __future__ import annotations

import csv
import os

import pytest

from euclid_polish.observability import JobLog, JobRecord


@pytest.fixture
def parses(monkeypatch):
    """Count full CSV parses (``csv.DictReader`` constructions)."""
    seen = {"n": 0}
    real = csv.DictReader

    def counting(*args, **kwargs):
        seen["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(csv, "DictReader", counting)
    return seen


def _log_with_rows(path, n=3) -> JobLog:
    log = JobLog(str(path))
    for i in range(n):
        log.record_submission(JobRecord(
            jobid=str(i), step_id="extract_psf" if i % 2 == 0 else "download",
            label=f"job {i}", submitted_at=f"2026-05-26T1{i}:00:00Z",
            params_json='{"n_stars":200}'))
    return log


def _fresh_rows(path) -> list[dict[str, str]]:
    """What an independent reader (a fresh JobLog) sees on disk."""
    return JobLog(str(path)).list_all()


class TestRepeatedReadsHitTheCache:

    def test_reads_parse_the_file_once(self, tmp_path, parses):
        log = _log_with_rows(tmp_path / "log.csv")
        log.list_all()
        after_first = parses["n"]
        assert after_first <= 1
        for _ in range(5):
            assert log.get("1")["label"] == "job 1"
            assert len(log.list_all()) == 3
            assert [r["jobid"] for r in log.history_for_step("extract_psf")] == ["2", "0"]
            assert log.latest_match("download", '{"n_stars":200}')["jobid"] == "1"
            assert log.accounting_attempt_due("2") is True
        assert parses["n"] == after_first

    def test_construction_seeds_the_cache(self, tmp_path, parses):
        p = tmp_path / "log.csv"
        _log_with_rows(p)
        reopened = JobLog(str(p))
        before = parses["n"]
        reopened.list_all()
        reopened.get("0")
        assert parses["n"] == before


class TestChangesOnDiskArePickedUp:

    def test_another_joblog_append_is_seen(self, tmp_path):
        p = tmp_path / "log.csv"
        log = _log_with_rows(p)
        assert log.get("99") is None
        JobLog(str(p)).record_submission(JobRecord(jobid="99", label="other process"))
        assert log.get("99")["label"] == "other process"
        assert [r["jobid"] for r in log.list_all()] == ["0", "1", "2", "99"]

    def test_another_joblog_rewrite_is_seen(self, tmp_path):
        p = tmp_path / "log.csv"
        log = _log_with_rows(p)
        assert log.get("1")["state"] == ""
        JobLog(str(p)).record_post_mortem("1", {"state": "FAILED"})
        assert log.get("1")["state"] == "FAILED"
        assert log.history_for_step("download")[0]["state"] == "FAILED"

    def test_external_raw_append_is_seen(self, tmp_path):
        p = tmp_path / "log.csv"
        log = _log_with_rows(p)
        log.list_all()
        with open(p, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=JobLog.COLUMNS).writerow(
                {**dict.fromkeys(JobLog.COLUMNS, ""), "jobid": "7", "label": "by hand"})
        assert log.get("7")["label"] == "by hand"

    def test_same_size_in_place_edit_is_seen_through_mtime(self, tmp_path):
        """An in-place edit that keeps the inode and the byte size is still
        seen once the modification time moves."""
        p = tmp_path / "log.csv"
        log = _log_with_rows(p)
        log.record_post_mortem("1", {"state": "FAILED"})
        assert log.get("1")["state"] == "FAILED"
        st = os.stat(p)
        data = p.read_bytes()
        assert data.count(b"FAILED") == 1
        with open(p, "r+b") as f:
            f.write(data.replace(b"FAILED", b"PASSED"))
        os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
        assert os.stat(p).st_size == st.st_size
        assert os.stat(p).st_ino == st.st_ino
        assert log.get("1")["state"] == "PASSED"

    def test_deleted_then_recreated_file(self, tmp_path):
        p = tmp_path / "log.csv"
        log = _log_with_rows(p)
        assert len(log.list_all()) == 3
        os.remove(p)
        assert log.list_all() == []
        assert log.get("0") is None
        assert log.history_for_step("extract_psf") == []
        other = JobLog(str(p))
        other.record_submission(JobRecord(jobid="new"))
        assert [r["jobid"] for r in log.list_all()] == ["new"]


class TestOwnWritesRefreshTheCache:

    def test_post_mortem_is_visible_without_a_reparse(self, tmp_path, parses):
        p = tmp_path / "log.csv"
        log = _log_with_rows(p)
        log.list_all()
        before = parses["n"]
        assert log.record_post_mortem("1", {"state": "COMPLETED", "max_rss_mb": 512.0})
        assert log.mark_accounting_attempt("2", at=123.0)
        assert log.get("1")["state"] == "COMPLETED"
        assert log.get("1")["max_rss_mb"] == "512.0"
        assert log.get("2")["accounting_attempted_at"] == "123.0"
        assert parses["n"] == before
        assert log.list_all() == _fresh_rows(p)

    def test_submission_is_visible(self, tmp_path):
        p = tmp_path / "log.csv"
        log = _log_with_rows(p)
        log.list_all()
        log.record_submission(JobRecord(jobid="3", step_id="extract_psf",
                                        submitted_at="2026-05-26T20:00:00Z"))
        assert log.get("3") is not None
        assert [r["jobid"] for r in log.history_for_step("extract_psf")] == ["3", "2", "0"]
        assert log.list_all() == _fresh_rows(p)

    def test_cache_after_a_write_matches_a_fresh_parse_for_awkward_values(self, tmp_path):
        """The rows cached from JobLog's own rewrite equal what a reader parses
        back: embedded newlines/CR, quotes, commas, unicode, a large payload,
        and dict/list/bool/None stats values."""
        p = tmp_path / "log.csv"
        log = JobLog(str(p))
        log.record_submission(JobRecord(
            jobid="a", label='say "hi", then\nleave\r\nnow\rok',
            params_json='{"calibration":"' + ("x" * 200_000) + '"}'))
        log.record_submission(JobRecord(jobid="b", label="ünïcødé ✓"))
        log.record_submission(JobRecord(jobid="c"))
        log.list_all()
        log.record_post_mortem("b", {
            "state": "CANCELLED by 1000", "jobstats_notes_json": ["a", {"b": 1}],
            "jobstats_gpu_nodes_json": {"n1": [1.5, 2]}, "alloc_gpus": True,
            "exit_code": None, "not_a_column": "dropped", "elapsed_seconds": 1e-7,
        })
        log.record_post_mortem("c", {"label": ""})
        assert log.list_all() == _fresh_rows(p)
        assert log.get("b")["alloc_gpus"] == "true"
        assert log.get("b")["exit_code"] == ""
        assert "not_a_column" not in log.get("b")

    def test_failed_write_leaves_the_cache_intact(self, tmp_path, monkeypatch):
        p = tmp_path / "log.csv"
        log = _log_with_rows(p)
        before = log.list_all()

        def boom(*_args, **_kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(os, "replace", boom)
        with pytest.raises(OSError):
            log.record_post_mortem("1", {"state": "COMPLETED"})
        monkeypatch.undo()
        assert log.list_all() == before
        assert log.get("1")["state"] == ""


class TestReturnedRowsAreCopies:

    def test_mutating_results_does_not_touch_the_cache(self, tmp_path):
        p = tmp_path / "log.csv"
        log = _log_with_rows(p)
        pristine = _fresh_rows(p)

        rows = log.list_all()
        rows[0]["state"] = "MUTATED"
        rows.append({"jobid": "ghost"})
        log.get("1")["label"] = "MUTATED"
        log.history_for_step("extract_psf")[0]["label"] = "MUTATED"
        log.latest_match("download", '{"n_stars":200}')["label"] = "MUTATED"
        for row in log.get_many(["0", "2"]).values():
            row["label"] = "MUTATED"
        for group in log.history_by_step().values():
            group[0]["label"] = "MUTATED"
            group.clear()

        assert log.list_all() == pristine
        assert log.get("ghost") is None
        assert _fresh_rows(p) == pristine

    def test_post_mortem_after_a_caller_mutation_writes_clean_rows(self, tmp_path):
        p = tmp_path / "log.csv"
        log = _log_with_rows(p)
        log.get("0")["label"] = "MUTATED"
        log.record_post_mortem("1", {"state": "DONE"})
        assert _fresh_rows(p)[0]["label"] == "job 0"


class TestBulkReads:

    def test_get_many_matches_get(self, tmp_path):
        p = tmp_path / "log.csv"
        log = _log_with_rows(p)
        # A duplicated jobid: get() returns the first row, so get_many does too.
        log.record_submission(JobRecord(jobid="1", label="duplicate"))
        got = log.get_many(["1", "2", "missing"])
        assert set(got) == {"1", "2"}
        assert got["1"] == log.get("1")
        assert got["1"]["label"] == "job 1"
        assert got["2"] == log.get("2")
        assert log.get_many([]) == {}

    def test_history_by_step_matches_history_for_step(self, tmp_path):
        p = tmp_path / "log.csv"
        log = _log_with_rows(p, n=6)
        # Equal timestamps keep their file order, exactly as history_for_step.
        log.record_submission(JobRecord(jobid="tie-a", step_id="download",
                                        submitted_at="2026-05-26T15:00:00Z"))
        log.record_submission(JobRecord(jobid="tie-b", step_id="download",
                                        submitted_at="2026-05-26T15:00:00Z"))
        log.record_submission(JobRecord(jobid="no-step"))
        grouped = log.history_by_step()
        for step_id in ("extract_psf", "download", ""):
            assert grouped[step_id] == log.history_for_step(step_id)
        assert set(grouped) == {"extract_psf", "download", ""}
