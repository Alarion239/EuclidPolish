"""The experiment-tracking store — a file-based "lab notebook".

A *campaign* is a titled folder that collects everything you did during a
stretch of work: backed-up model checkpoints, FITS files, images, a
free-form markdown log, and a record of every FASRC job you launched.
Each campaign stamps the git commit it was created at (and the commit it
was saved at) into its metadata so results are always traceable to code.

Layout under ``Config.TRACKING_DIR`` (default ``./tracking``)::

    <root>/
      current/                 # the one active campaign
        metadata.json          # title, status, created/saved commit + time
        log.md                 # the markdown notebook (editable in the WebUI)
        fasrc_jobs.jsonl        # one line per submitted FASRC job
        models/  fits/  images/ # backups, each with a sidecar *.meta.json
      archive/
        <slug>/                # campaigns that have been "saved"

The store is deliberately git-agnostic for *storage*: the folder is
gitignored and mirrored to persistent FASRC holylabs storage instead
(see :mod:`euclid_polish.tracking.sync`). Git is used only to stamp the
commit a campaign was created / saved at.

All public mutating methods are guarded by a process-wide lock so the
multi-threaded Flask server can't race on unique-name allocation or
``jsonl`` appends.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import threading
from typing import Any, Dict, List, Optional

from euclid_polish.config import Config
from euclid_polish.observability.training_log import TrainingLog
from euclid_polish.provenance.gitinfo import capture_git as git_commit_info
from ._utils import _now_iso, _write_json, _read_json

# Project root = three levels up from this file (…/euclid_polish/tracking/).
# Used as the cwd for the git-commit stamp so it works regardless of where
# the caller (WebUI, script) launched from.
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

# One lock for every mutating op. Backups are I/O-bound and infrequent, so
# a single coarse lock is simpler and plenty fast.
_LOCK = threading.RLock()

# Which top-level files make a TensorFlow checkpoint dir restorable.
_CKPT_KEEP_EXACT = {"checkpoint", TrainingLog.FILENAME, "training_log.jsonl"}


class TrackingError(RuntimeError):
    """Raised for invalid tracking operations (no active campaign, etc.)."""


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _slugify(text: str, *, default: str = "campaign") -> str:
    """Filesystem-safe slug: lowercase, alnum + dashes, collapsed."""
    text = (text or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return text or default


def _unique_path(directory: str, name: str) -> str:
    """``directory/name``, or ``name-2`` / ``name-3`` … if taken.

    Splits a single extension off so ``foo.fits`` → ``foo-2.fits``.
    Works for files and directories alike.
    """
    base, ext = os.path.splitext(name)
    candidate = os.path.join(directory, name)
    i = 2
    while os.path.exists(candidate):
        candidate = os.path.join(directory, f"{base}-{i}{ext}")
        i += 1
    return candidate


def _commit_str(commit: Optional[Dict[str, Any]]) -> str:
    """One-line human rendering of a commit dict for markdown headers."""
    if not commit:
        return "(no git commit)"
    s = f"{commit.get('short', '?')} ({commit.get('branch', '?')})"
    if commit.get("dirty"):
        s += " +dirty"
    return s


def dirty_warning(commit: Optional[Dict[str, Any]]) -> Optional[str]:
    """Reproducibility warning for a just-stamped commit, or ``None`` if clean.

    We never auto-commit — capturing the exact state is the user's call.
    Time-travel checks out the recorded *commit*, so anything uncommitted
    won't be reproduced; this is the warning surfaced at backup/save time.
    """
    if commit is None:
        return ("not a git repository (or git unavailable) — this backup "
                "cannot be reproduced by time-travel.")
    if commit.get("dirty"):
        return ("working tree has uncommitted changes — they will NOT be "
                "captured. Time-travel restores commit "
                f"{commit.get('short', '?')} only. Commit everything you want "
                "included for this to be exactly reproducible.")
    return None


# ---------------------------------------------------------------------------
# the store
# ---------------------------------------------------------------------------

class TrackingStore:
    """File-based campaign store rooted at ``root`` (``Config.TRACKING_DIR``)."""

    def __init__(self, root: Optional[str] = None,
                 repo_root: str = _PROJECT_ROOT) -> None:
        self.root = os.path.abspath(root or Config.TRACKING_DIR)
        self.repo_root = repo_root
        self.current_dir = os.path.join(self.root, "current")
        self.archive_dir = os.path.join(self.root, "archive")

    # ----------------------------- campaigns ------------------------------

    def has_current(self) -> bool:
        return os.path.isfile(os.path.join(self.current_dir, "metadata.json"))

    def current(self) -> Optional[Dict[str, Any]]:
        """Active campaign metadata, or ``None`` if none is active."""
        return _read_json(os.path.join(self.current_dir, "metadata.json"))

    def _require_current(self) -> str:
        if not self.has_current():
            raise TrackingError(
                "no active tracking campaign — create one first "
                "(Tracking page → New campaign, or scripts/track.py new)."
            )
        return self.current_dir

    def create_campaign(self, title: str,
                        description: str = "") -> Dict[str, Any]:
        """Start a new active campaign. Fails if one is already active."""
        with _LOCK:
            if self.has_current():
                cur = self.current() or {}
                raise TrackingError(
                    f"campaign {cur.get('title', '?')!r} is already active; "
                    "save it before starting a new one."
                )
            title = (title or "").strip() or "untitled"
            commit = git_commit_info(self.repo_root)
            meta = {
                "title":          title,
                "slug":           _slugify(title),
                "status":         "active",
                "description":    description.strip(),
                "created_at":     _now_iso(),
                "created_commit": commit,
                "saved_at":       None,
                "saved_commit":   None,
            }
            for sub in ("models", "fits", "images"):
                os.makedirs(os.path.join(self.current_dir, sub), exist_ok=True)
            _write_json(os.path.join(self.current_dir, "metadata.json"), meta)
            # Seed the notebook + the (empty) job log.
            log = (
                f"# {title}\n\n"
                f"> Campaign created {meta['created_at']} at commit "
                f"{_commit_str(commit)}.\n\n"
            )
            if description.strip():
                log += description.strip() + "\n\n"
            log += "---\n"
            with open(os.path.join(self.current_dir, "log.md"), "w") as fp:
                fp.write(log)
            open(os.path.join(self.current_dir, "fasrc_jobs.jsonl"), "a").close()
            return meta

    def save_campaign(self) -> Dict[str, Any]:
        """Freeze the active campaign and move it to ``archive/<slug>``.

        Stamps the current HEAD commit as ``saved_commit`` so the archive
        records exactly where the code stood when work concluded. Returns
        ``{archive_path, metadata}``.
        """
        with _LOCK:
            self._require_current()
            meta = self.current() or {}
            meta["status"] = "saved"
            meta["saved_at"] = _now_iso()
            meta["saved_commit"] = git_commit_info(self.repo_root)
            _write_json(
                os.path.join(self.current_dir, "metadata.json"), meta,
            )
            os.makedirs(self.archive_dir, exist_ok=True)
            target = _unique_path(
                self.archive_dir, meta.get("slug") or "campaign",
            )
            shutil.move(self.current_dir, target)
            return {"archive_path": target, "metadata": meta}

    def list_campaigns(self) -> Dict[str, Any]:
        """``{active: metadata|None, archived: [...]}`` (archived newest first)."""
        archived: List[Dict[str, Any]] = []
        if os.path.isdir(self.archive_dir):
            for name in os.listdir(self.archive_dir):
                meta = _read_json(
                    os.path.join(self.archive_dir, name, "metadata.json"),
                )
                if meta:
                    meta = dict(meta)
                    meta["_dir"] = name
                    archived.append(meta)
        archived.sort(key=lambda m: m.get("saved_at") or "", reverse=True)
        return {"active": self.current(), "archived": archived}

    # ------------------------------- log ----------------------------------

    def read_log(self) -> str:
        try:
            with open(os.path.join(self.current_dir, "log.md")) as fp:
                return fp.read()
        except OSError:
            return ""

    def write_log(self, text: str) -> None:
        """Overwrite the whole notebook (Tracking-page editor)."""
        with _LOCK:
            self._require_current()
            with open(os.path.join(self.current_dir, "log.md"), "w") as fp:
                fp.write(text)

    def append_log(self, text: str) -> None:
        """Append a timestamped entry to the notebook."""
        with _LOCK:
            self._require_current()
            entry = f"\n## {_now_iso()}\n\n{text.strip()}\n"
            with open(os.path.join(self.current_dir, "log.md"), "a") as fp:
                fp.write(entry)

    # ----------------------------- backups --------------------------------

    def _record_meta(self, *, kind: str, name: str, comment: str,
                     source_path: str, files: List[str],
                     size_bytes: int) -> Dict[str, Any]:
        return {
            "name":        name,
            "kind":        kind,
            "comment":     (comment or "").strip(),
            "source_path": os.path.abspath(source_path),
            "files":       files,
            "size_bytes":  size_bytes,
            "created_at":  _now_iso(),
            "commit":      git_commit_info(self.repo_root),
        }

    def backup_fits(self, src_path: str, comment: str = "",
                   name: Optional[str] = None) -> Dict[str, Any]:
        return self._backup_file("fits", src_path, comment, name,
                                 default_ext=".fits")

    def backup_image(self, src_path: str, comment: str = "",
                    name: Optional[str] = None) -> Dict[str, Any]:
        return self._backup_file("images", src_path, comment, name,
                                 default_ext=".png")

    def _backup_file(self, subdir: str, src_path: str, comment: str,
                    name: Optional[str], *, default_ext: str) -> Dict[str, Any]:
        with _LOCK:
            self._require_current()
            if not os.path.isfile(src_path):
                raise TrackingError(f"source file not found: {src_path}")
            src_base = os.path.basename(src_path)
            src_ext = os.path.splitext(src_base)[1] or default_ext
            if name:
                stem = _slugify(os.path.splitext(name)[0], default="backup")
                stored = stem + (os.path.splitext(name)[1] or src_ext)
            else:
                stored = src_base
            dest_dir = os.path.join(self.current_dir, subdir)
            os.makedirs(dest_dir, exist_ok=True)
            dest = _unique_path(dest_dir, stored)
            shutil.copy2(src_path, dest)
            stored_name = os.path.basename(dest)
            meta = self._record_meta(
                kind=("fits" if subdir == "fits" else "image"),
                name=stored_name, comment=comment, source_path=src_path,
                files=[stored_name], size_bytes=os.path.getsize(dest),
            )
            _write_json(os.path.join(dest_dir, stored_name + ".meta.json"), meta)
            return meta

    def backup_model(self, ckpt_dir: Optional[str] = None, comment: str = "",
                    name: Optional[str] = None) -> Dict[str, Any]:
        """Copy a restorable snapshot of a TensorFlow checkpoint directory.

        Captures the ``checkpoint`` manifest, every ``ckpt-N.{index,data-*}``,
        the ``training_log.csv`` history, and the ``loss_best/`` sub-track.
        ``.bak`` rotations and stray files are skipped.
        """
        with _LOCK:
            self._require_current()
            ckpt_dir = ckpt_dir or Config.DEFAULT_CHECKPOINT_DIR
            if not os.path.isdir(ckpt_dir):
                raise TrackingError(f"checkpoint dir not found: {ckpt_dir}")
            stem = _slugify(name or os.path.basename(ckpt_dir.rstrip("/")),
                            default="model")
            dest_dir = _unique_path(
                os.path.join(self.current_dir, "models"), stem,
            )
            os.makedirs(dest_dir, exist_ok=True)
            files = _copy_checkpoint(ckpt_dir, dest_dir)
            if not files:
                shutil.rmtree(dest_dir, ignore_errors=True)
                raise TrackingError(
                    f"no checkpoint files found under {ckpt_dir}"
                )
            size = sum(
                os.path.getsize(os.path.join(dp, f))
                for dp, _d, fs in os.walk(dest_dir) for f in fs
            )
            meta = self._record_meta(
                kind="model", name=os.path.basename(dest_dir), comment=comment,
                source_path=ckpt_dir, files=files, size_bytes=size,
            )
            _write_json(os.path.join(dest_dir, "meta.json"), meta)
            return meta

    def list_backups(self) -> Dict[str, List[Dict[str, Any]]]:
        """Enumerate the active campaign's ``{models, fits, images}`` backups."""
        if not self.has_current():
            return {"models": [], "fits": [], "images": []}
        return self._backups_in(self.current_dir)

    def _backups_in(self, campaign_dir: str) -> Dict[str, List[Dict[str, Any]]]:
        out: Dict[str, List[Dict[str, Any]]] = {
            "models": [], "fits": [], "images": [],
        }
        # Models: one sub-dir each, meta.json inside.
        models_dir = os.path.join(campaign_dir, "models")
        if os.path.isdir(models_dir):
            for name in sorted(os.listdir(models_dir)):
                meta = _read_json(os.path.join(models_dir, name, "meta.json"))
                if meta:
                    out["models"].append(meta)
        # Files: sidecar *.meta.json next to each stored artifact.
        for sub in ("fits", "images"):
            d = os.path.join(campaign_dir, sub)
            if not os.path.isdir(d):
                continue
            for fn in sorted(os.listdir(d)):
                if fn.endswith(".meta.json"):
                    meta = _read_json(os.path.join(d, fn))
                    if meta:
                        out[sub].append(meta)
        for k in out:
            out[k].sort(key=lambda m: m.get("created_at") or "", reverse=True)
        return out

    # ---- campaign/backup resolution (used by time-travel) ----------------

    def campaign_dir(self, name: Optional[str]) -> str:
        """Abs dir for ``"current"`` or an archived campaign ``name``.

        ``name`` is basename-sanitised so it can't escape the archive dir.
        Raises :class:`TrackingError` if no such campaign exists.
        """
        if not name or name == "current":
            d = self.current_dir
        else:
            d = os.path.join(self.archive_dir, os.path.basename(name))
        if not os.path.isfile(os.path.join(d, "metadata.json")):
            raise TrackingError(f"no campaign {name!r}")
        return d

    def campaign_meta(self, name: Optional[str]) -> Optional[Dict[str, Any]]:
        return _read_json(os.path.join(self.campaign_dir(name), "metadata.json"))

    def backups_in(self, name: Optional[str]) -> Dict[str, List[Dict[str, Any]]]:
        return self._backups_in(self.campaign_dir(name))

    def model_backup_dir(self, name: Optional[str], model: str) -> str:
        """Abs dir of model backup ``model`` within campaign ``name``."""
        d = os.path.join(self.campaign_dir(name), "models",
                         os.path.basename(model))
        if not os.path.isdir(d):
            raise TrackingError(f"no model backup {model!r} in {name!r}")
        return d

    def model_backup_meta(self, name: Optional[str],
                          model: str) -> Optional[Dict[str, Any]]:
        return _read_json(os.path.join(self.model_backup_dir(name, model),
                                       "meta.json"))

    # --------------------------- FASRC jobs -------------------------------

    def log_fasrc_job(self, record: Dict[str, Any]) -> Optional[str]:
        """Append a submitted-job record to the active campaign's job log.

        Falls back to ``<root>/unassigned_fasrc_jobs.jsonl`` when no campaign
        is active, so a job is never silently dropped. Returns the file
        written, or ``None`` on failure.
        """
        with _LOCK:
            rec = dict(record)
            rec.setdefault("logged_at", _now_iso())
            rec.setdefault("commit", git_commit_info(self.repo_root))
            if self.has_current():
                path = os.path.join(self.current_dir, "fasrc_jobs.jsonl")
            else:
                os.makedirs(self.root, exist_ok=True)
                path = os.path.join(self.root, "unassigned_fasrc_jobs.jsonl")
            try:
                with open(path, "a") as fp:
                    fp.write(json.dumps(rec, sort_keys=True) + "\n")
                return path
            except OSError:
                return None

    def read_fasrc_jobs(self) -> List[Dict[str, Any]]:
        """Parse the active campaign's job log (newest first)."""
        if not self.has_current():
            return []
        path = os.path.join(self.current_dir, "fasrc_jobs.jsonl")
        out: List[Dict[str, Any]] = []
        try:
            with open(path) as fp:
                for line in fp:
                    line = line.strip()
                    if line:
                        try:
                            out.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        except OSError:
            return []
        out.reverse()
        return out


def _copy_checkpoint(ckpt_dir: str, dest_dir: str) -> List[str]:
    """Copy the restorable subset of a checkpoint dir; return relative paths."""
    copied: List[str] = []
    for name in sorted(os.listdir(ckpt_dir)):
        src = os.path.join(ckpt_dir, name)
        if os.path.isdir(src):
            # The loss-best sub-track is a full second CheckpointManager.
            if name == "loss_best":
                shutil.copytree(src, os.path.join(dest_dir, name))
                for dp, _d, fs in os.walk(os.path.join(dest_dir, name)):
                    for f in fs:
                        rel = os.path.relpath(os.path.join(dp, f), dest_dir)
                        copied.append(rel)
            continue
        keep = (
            name in _CKPT_KEEP_EXACT
            or (name.startswith("ckpt-")
                and (".index" in name or ".data-" in name))
        )
        if keep:
            shutil.copy2(src, os.path.join(dest_dir, name))
            copied.append(name)
    return copied


# Process-wide default store, lazily built so tests can monkeypatch
# Config.TRACKING_DIR before first use.
_DEFAULT: Optional[TrackingStore] = None


def default_store() -> TrackingStore:
    """Shared :class:`TrackingStore` rooted at ``Config.TRACKING_DIR``."""
    global _DEFAULT
    if _DEFAULT is None or _DEFAULT.root != os.path.abspath(Config.TRACKING_DIR):
        _DEFAULT = TrackingStore(Config.TRACKING_DIR)
    return _DEFAULT
