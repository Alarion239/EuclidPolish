"""Local git tab operations against a throw-away repo in tmp_path."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from euclid_polish.web import git_ops
from euclid_polish.web.app import create_app


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
    git_ops.commit("add a", all_files=True)
    (repo / "b.txt").write_text("b")
    git_ops.commit("add b", all_files=True)
    entries = git_ops.log(5)
    subjects = [e["subject"] for e in entries]
    assert subjects[:3] == ["add b", "add a", "initial"]


def test_commit_empty_message_rejected(repo):
    (repo / "x.txt").write_text("x")
    out = git_ops.commit("   ", all_files=True)
    assert out["ok"] is False
    assert "empty" in out["error"].lower() or "message" in out["error"].lower()


def test_commit_with_no_changes_is_a_failure(repo):
    out = git_ops.commit("nothing to do", all_files=True)
    assert out["ok"] is False  # git refuses to commit an empty index


def test_diff_truncates_at_limit(repo):
    big = "line\n" * 5_000
    (repo / "big.txt").write_text(big)
    # Diff isn't staged yet, so this hits ``git diff`` for untracked? No —
    # ``git diff`` only shows tracked files, so stage + commit + modify.
    git_ops.commit("seed big", all_files=True)
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


# ---------------------------------------------------------------------------
# Commit selection + large/untracked-binary guard (never ``git add -A``
# blindly: the repo has 80 MB poster FITS lying around untracked)
# ---------------------------------------------------------------------------

def _tracked(repo) -> set[str]:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=repo,
                         capture_output=True, text=True, check=True).stdout
    return {path for path in out.split("\0") if path}


def test_commit_requires_explicit_paths_or_all(repo):
    (repo / "a.txt").write_text("a")
    out = git_ops.commit("no selection", None)
    assert out["ok"] is False and out["code"] == "no_selection"
    assert "a.txt" not in _tracked(repo)


def test_commit_stages_only_the_listed_paths(repo):
    (repo / "a.txt").write_text("a")
    (repo / "b.txt").write_text("b")
    out = git_ops.commit("just a", ["a.txt"])
    assert out["ok"] is True, out
    assert "a.txt" in _tracked(repo) and "b.txt" not in _tracked(repo)


def test_commit_all_stages_every_change_including_deletions(repo):
    (repo / "a.txt").write_text("a")
    (repo / "README.md").unlink()
    out = git_ops.commit("everything", all_files=True)
    assert out["ok"] is True, out
    assert _tracked(repo) == {"a.txt"}


def test_commit_refuses_untracked_binaries_and_huge_files(repo):
    (repo / "small.txt").write_text("fine")
    (repo / "field.fits").write_bytes(b"\0" * (1024 * 1024 + 1))
    (repo / "tiny.png").write_bytes(b"\x89PNG" + b"\0" * 100)    # small: allowed
    (repo / "data").mkdir()
    (repo / "data" / "huge.txt").write_bytes(b"x" * (10 * 1024 * 1024 + 1))
    out = git_ops.commit("oops", all_files=True)
    assert out["ok"] is False and out["code"] == "refused_files"
    refused = {item["path"]: item for item in out["refused"]}
    assert set(refused) == {"field.fits", "data/huge.txt"}
    assert refused["field.fits"]["reason"] == "untracked binary > 1 MB"
    assert refused["data/huge.txt"]["reason"] == "file > 10 MB"
    assert _tracked(repo) == {"README.md"}                 # nothing staged
    forced = git_ops.commit("on purpose", all_files=True, force=True)
    assert forced["ok"] is True
    assert {"field.fits", "data/huge.txt", "small.txt", "tiny.png"} <= _tracked(repo)


def _committed_files(repo) -> set[str]:
    out = subprocess.run(["git", "show", "--name-only", "--pretty=format:", "HEAD"],
                         cwd=repo, capture_output=True, text=True, check=True).stdout
    return set(out.split())


def test_commit_of_explicit_paths_leaves_other_staged_files_out(repo):
    (repo / "a.txt").write_text("a")
    _run(["git", "add", "a.txt"], cwd=repo)
    _run(["git", "commit", "-m", "seed a"], cwd=repo)
    (repo / "a.txt").write_text("a2")
    (repo / "big.fits").write_bytes(b"\0" * (2 * 1024 * 1024))
    _run(["git", "add", "big.fits"], cwd=repo)        # staged outside `paths`
    out = git_ops.commit("only a", ["a.txt"])
    assert out["ok"] is True, out
    assert out["committed"] == ["a.txt"]
    assert _committed_files(repo) == {"a.txt"}
    staged = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=repo,
                            capture_output=True, text=True, check=True).stdout.split()
    assert staged == ["big.fits"]                       # left staged, not committed


def test_commit_all_refuses_a_pre_staged_large_binary(repo):
    (repo / "small.txt").write_text("fine")
    (repo / "big.fits").write_bytes(b"\0" * (2 * 1024 * 1024))
    _run(["git", "add", "big.fits"], cwd=repo)        # staged new file ("A ")
    out = git_ops.commit("everything", all_files=True)
    assert out["ok"] is False and out["code"] == "refused_files"
    assert [item["path"] for item in out["refused"]] == ["big.fits"]
    assert out["refused"][0]["reason"] == "untracked binary > 1 MB"


def test_commit_of_a_renamed_path_commits_the_rename(repo):
    _run(["git", "mv", "README.md", "NOTES.md"], cwd=repo)
    out = git_ops.commit("rename", ["NOTES.md"])
    assert out["ok"] is True, out
    assert _tracked(repo) == {"NOTES.md"}
    status = subprocess.run(["git", "status", "--porcelain"], cwd=repo,
                            capture_output=True, text=True, check=True).stdout
    assert status == ""                                # deletion side committed too


def test_commit_route_contract(repo):
    app = create_app()
    app.config["TESTING"] = True
    client = app.test_client()
    (repo / "a.txt").write_text("a")
    (repo / "big.npy").write_bytes(b"\0" * (2 * 1024 * 1024))
    assert client.post("/git/commit", data={"message": "m"}).status_code == 400
    refused = client.post("/git/commit", data={"message": "m", "all": "1"})
    assert refused.status_code == 409
    assert [item["path"] for item in refused.get_json()["refused"]] == ["big.npy"]
    ok = client.post("/git/commit", data={"message": "m", "paths": ["a.txt"]})
    assert ok.status_code == 200, ok.get_json()
    assert "a.txt" in _tracked(repo) and "big.npy" not in _tracked(repo)


def test_commit_all_can_conclude_a_merge(repo):
    """``all=1`` commits the whole (fully staged) index, so it can conclude
    an in-progress merge — git refuses a pathspec commit there."""
    _run(["git", "checkout", "-b", "side"], cwd=repo)
    (repo / "README.md").write_text("side\n")
    _run(["git", "commit", "-am", "side"], cwd=repo)
    _run(["git", "checkout", "main"], cwd=repo)
    (repo / "README.md").write_text("main\n")
    _run(["git", "commit", "-am", "main"], cwd=repo)
    merge = subprocess.run(["git", "merge", "side"], cwd=repo,
                           capture_output=True, text=True)
    assert merge.returncode != 0                        # conflict
    (repo / "README.md").write_text("resolved\n")
    out = git_ops.commit("merge side", all_files=True)
    assert out["ok"] is True, out
    parents = subprocess.run(["git", "rev-list", "--parents", "-n", "1", "HEAD"],
                             cwd=repo, capture_output=True, text=True,
                             check=True).stdout.split()
    assert len(parents) == 3                            # a real merge commit


def test_status_lists_renames_and_odd_names_unquoted(repo):
    """``status()`` paths are the raw new paths (no ``old -> new``, no C
    quoting) and a rename carries its source as ``orig``."""
    _run(["git", "mv", "README.md", "NOTES.md"], cwd=repo)
    (repo / "sp ace.txt").write_text("s")
    (repo / "é.txt").write_text("e")
    (repo / "newdir").mkdir()
    (repo / "newdir" / "inner.txt").write_text("i")
    files = {f["path"]: f for f in git_ops.status()["files"]}
    assert set(files) == {"NOTES.md", "sp ace.txt", "é.txt", "newdir/"}
    assert files["NOTES.md"]["orig"] == "README.md"
    assert files["NOTES.md"]["xy"].startswith("R")
    assert files["sp ace.txt"]["xy"] == "??" and files["sp ace.txt"]["orig"] is None


def test_every_path_status_lists_can_be_committed_as_listed(repo):
    """Round trip: posting each ``/api/git/status`` path back to
    ``/git/commit`` unchanged commits exactly that file."""
    _run(["git", "mv", "README.md", "NOTES.md"], cwd=repo)
    (repo / "sp ace.txt").write_text("s")
    (repo / "é.txt").write_text("e")
    (repo / "a,b.txt").write_text("c")
    (repo / "newdir").mkdir()
    (repo / "newdir" / "inner.txt").write_text("i")
    app = create_app()
    app.config["TESTING"] = True
    client = app.test_client()
    listed = client.get("/api/git/status").get_json()
    paths = [f["path"] for f in listed["status"]["files"]]
    assert len(paths) == 5
    for path in paths:
        out = client.post("/git/commit", data={"message": f"add {path}",
                                                "paths": [path]})
        assert out.status_code == 200, (path, out.get_json())
    assert git_ops.status()["files"] == []
    assert _tracked(repo) == {"NOTES.md", "sp ace.txt", "é.txt", "a,b.txt",
                              "newdir/inner.txt"}
