"""Local git tab operations against a throw-away repo in tmp_path."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from euclid_polish.web import git_ops


def _run(cmd, cwd):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=True)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A fresh local git repo, ``cwd`` of the test set there."""
    monkeypatch.chdir(tmp_path)
    _run(["git", "init", "-b", "main"], cwd=tmp_path)
    _run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path)
    _run(["git", "config", "user.name",  "Test User"],        cwd=tmp_path)
    (tmp_path / "README.md").write_text("seed\n")
    _run(["git", "add", "README.md"], cwd=tmp_path)
    _run(["git", "commit", "-m", "initial"], cwd=tmp_path)
    return tmp_path


def test_status_reports_clean_repo(repo):
    s = git_ops.status()
    assert s["in_repo"] is True
    assert s["branch"] == "main"
    assert s["clean"] is True
    assert s["files"] == []
    assert s["last"]["subject"] == "initial"


def test_status_reports_dirty_files(repo):
    (repo / "untracked.txt").write_text("x")
    (repo / "README.md").write_text("modified\n")
    s = git_ops.status()
    assert s["clean"] is False
    paths = {f["path"] for f in s["files"]}
    assert "untracked.txt" in paths
    assert "README.md" in paths


def test_log_returns_chronological_list(repo):
    (repo / "a.txt").write_text("a")
    git_ops.commit("add a")
    (repo / "b.txt").write_text("b")
    git_ops.commit("add b")
    entries = git_ops.log(5)
    subjects = [e["subject"] for e in entries]
    assert subjects[:3] == ["add b", "add a", "initial"]


def test_commit_empty_message_rejected(repo):
    (repo / "x.txt").write_text("x")
    out = git_ops.commit("   ")
    assert out["ok"] is False
    assert "empty" in out["error"].lower() or "message" in out["error"].lower()


def test_commit_with_no_changes_is_a_failure(repo):
    out = git_ops.commit("nothing to do")
    assert out["ok"] is False  # git refuses to commit an empty index


def test_diff_truncates_at_limit(repo):
    big = "line\n" * 5_000
    (repo / "big.txt").write_text(big)
    # Diff isn't staged yet, so this hits ``git diff`` for untracked? No —
    # ``git diff`` only shows tracked files, so stage + commit + modify.
    git_ops.commit("seed big")
    (repo / "big.txt").write_text("line\n" * 6_000)
    d = git_ops.diff(max_chars=200)
    assert "truncated" in d


def test_push_pull_fetch_report_errors_without_remote(repo):
    # No remote configured — push/pull should fail cleanly.
    p = git_ops.push()
    assert p["ok"] is False
    assert p["error"]
    q = git_ops.pull()
    assert q["ok"] is False
    assert q["error"]
    f = git_ops.fetch()
    # ``git fetch`` with no remotes is a no-op on success on some git versions,
    # and an error on others — either way it shouldn't raise.
    assert "ok" in f
