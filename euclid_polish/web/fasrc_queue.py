"""A local, fail-stop submission queue for FASRC jobs.

Only one cluster job runs at a time. When a job is submitted while
another is still active, the new one is *queued locally* (on the laptop,
never on the cluster) instead of being ``sbatch``'d immediately. When the
active job finishes:

  * **success** (sacct ``COMPLETED``) → the next queued job is submitted;
  * **failure** — anything else terminal, **including OOM**
    (``OUT_OF_MEMORY``), ``FAILED``, ``TIMEOUT``, ``CANCELLED``,
    ``NODE_FAIL`` … → the queue **halts** and nothing else is submitted.

The queue is persisted to ``~/.euclid_polish/fasrc_queue.json`` so it
survives a WebUI restart while a cluster job keeps running. Promotion is
driven by :meth:`JobQueue.tick`, called by the dashboard's poll routes
right after they reconcile the JobDB against ``squeue``.

This module owns *no* knowledge of how to build/submit a job: the caller
passes a ``submit_fn(spec) -> (slurm_id, payload)`` so the routing layer
keeps the sbatch-rendering logic.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

from euclid_polish.web import fasrc_config

QUEUE_PATH = os.path.join(fasrc_config.CONFIG_DIR, "fasrc_queue.json")

# sacct/DB states that mean "still going" (not yet an outcome).
_ACTIVE_STATES = frozenset({
    "RUNNING", "PENDING", "REQUEUED", "RESIZING", "SUSPENDED",
    "COMPLETING", "CONFIGURING",
})
# How long we wait for sacct to report an outcome after a job leaves
# squeue (state ``DONE``) before treating the silence as a failure — we
# must NOT promote on an unknown outcome (could have been an OOM).
_SACCT_GRACE_S = 1800.0


def job_outcome(jobid: str, db: Any, joblog: Any) -> Tuple[str, str]:
    """Classify a job as ``("success"|"failure"|"pending", state)``.

    Authoritative source is sacct (folded into the CSV ``joblog`` by the
    reconcile pass): ``COMPLETED`` → success; any other *terminal* state
    — ``FAILED`` / ``TIMEOUT`` / ``CANCELLED`` / ``OUT_OF_MEMORY`` /
    ``NODE_FAIL`` … → failure. Until sacct settles we fall back to the DB
    row; a job that left squeue (``DONE``) but has no sacct verdict yet is
    ``pending`` (we wait) until ``_SACCT_GRACE_S`` elapses, after which the
    silence is treated as a failure so we never submit past an
    unconfirmed result.
    """
    log = joblog.get(jobid) if joblog is not None else None
    s = ((log.get("state") if log else "") or "").strip().upper()
    if s == "COMPLETED":
        return ("success", s)
    if s and s not in _ACTIVE_STATES:
        return ("failure", s)          # FAILED / OUT_OF_MEMORY / TIMEOUT / …

    row = db.get(jobid) if db is not None else None
    ds = ((row.get("state") if row else "") or "").strip().upper()
    if ds in ("", "RUNNING", "PENDING") or ds in _ACTIVE_STATES:
        return ("pending", ds or "PENDING")
    if ds == "COMPLETED":
        return ("success", ds)
    if ds == "DONE":
        # Left squeue; wait for sacct, but don't wait forever.
        ended = (row or {}).get("ended_at") or 0
        if ended and (time.time() - float(ended)) > _SACCT_GRACE_S:
            return ("failure", "NO_SACCT")
        return ("pending", "DONE")
    # Anything else (FAILED/CANCELLED/TIMEOUT/UNKNOWN/OUT_OF_MEMORY/…).
    return ("failure", ds or "UNKNOWN")


class JobQueue:
    """Persistent, single-lane, fail-stop submission queue."""

    def __init__(self, path: str = QUEUE_PATH) -> None:
        self.path = path
        self._lock = threading.RLock()
        self.items: List[Dict[str, Any]] = []
        self.active_jobid: Optional[str] = None
        self.halted: bool = False
        self.halted_reason: Optional[str] = None
        self._load()

    # ------------------------------- persistence --------------------------

    def _load(self) -> None:
        try:
            with open(self.path) as fp:
                d = json.load(fp)
        except (OSError, json.JSONDecodeError):
            return
        self.items = d.get("items", []) or []
        self.active_jobid = d.get("active_jobid")
        self.halted = bool(d.get("halted", False))
        self.halted_reason = d.get("halted_reason")

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w") as fp:
            json.dump({
                "items":         self.items,
                "active_jobid":  self.active_jobid,
                "halted":        self.halted,
                "halted_reason": self.halted_reason,
            }, fp, indent=2)
        os.replace(tmp, self.path)

    # ------------------------------- queries ------------------------------

    def active_is_running(self, db: Any) -> bool:
        """True if a job is currently active (so a new submit must queue).

        Adopts any externally-running job as the active one so the queue
        respects jobs launched outside it (or before a restart).
        """
        with self._lock:
            aid = self.active_jobid
            if aid:
                row = db.get(aid)
                st = ((row.get("state") if row else "") or "").upper()
                if row is not None and (st in ("PENDING", "RUNNING")
                                        or st in _ACTIVE_STATES or st == ""):
                    return True
            for r in db.list_recent(10):
                st = (r.get("state") or "").upper()
                if st in ("PENDING", "RUNNING") or st in _ACTIVE_STATES:
                    self.active_jobid = r["jobid"]
                    self._save()
                    return True
            return False

    def public(self) -> Dict[str, Any]:
        """Jsonify-able snapshot for the UI (``names`` is the list of labels)."""
        with self._lock:
            return {
                "names":         [it.get("label", "(job)") for it in self.items],
                "items":         [{"id": it["id"], "label": it.get("label", "")}
                                  for it in self.items],
                "count":         len(self.items),
                "active_jobid":  self.active_jobid,
                "halted":        self.halted,
                "halted_reason": self.halted_reason,
            }

    # ------------------------------- mutations ----------------------------

    def enqueue(self, spec: Dict[str, Any], label: str) -> Dict[str, Any]:
        with self._lock:
            item = {"id": uuid.uuid4().hex[:8], "label": label,
                    "spec": spec, "queued_at": time.time()}
            self.items.append(item)
            self._save()
            return item

    def on_direct_submit(self, jobid: str) -> None:
        """Mark a just-submitted job as the active lane occupant."""
        with self._lock:
            self.active_jobid = jobid
            self._save()

    def resume(self) -> None:
        """Clear a halt (e.g. on an explicit fresh submit)."""
        with self._lock:
            self.halted = False
            self.halted_reason = None
            self._save()

    def clear(self) -> Dict[str, Any]:
        """Drop all queued items and clear any halt. Leaves active untouched."""
        with self._lock:
            self.items = []
            self.halted = False
            self.halted_reason = None
            self._save()
            return self.public()

    def remove(self, item_id: str) -> Dict[str, Any]:
        with self._lock:
            self.items = [it for it in self.items if it["id"] != item_id]
            self._save()
            return self.public()

    def _halt(self, reason: str) -> None:
        self.halted = True
        self.halted_reason = reason
        self._save()

    # ------------------------------- promotion ----------------------------

    def tick(self, db: Any, joblog: Any, ssh: Any,
             submit_fn: Callable[[Dict[str, Any]],
                                 Tuple[Optional[str], Dict[str, Any]]]) -> None:
        """Advance the queue: promote on success, halt on failure.

        Idempotent and cheap — safe to call on every dashboard poll. Does
        nothing while halted, while the active job is still pending, or
        when SSH is down (a promotion needs to ``sbatch``).
        """
        with self._lock:
            if self.halted:
                return
            if ssh is None or not getattr(ssh, "is_connected", lambda: False)():
                return
            aid = self.active_jobid
            if aid is not None:
                outcome, state = job_outcome(aid, db, joblog)
                if outcome == "pending":
                    return
                if outcome == "failure":
                    self._halt(f"job {aid} ended {state} — queue stopped; "
                               "nothing further will be submitted.")
                    return
                # success → fall through and promote the next item
            self._promote(submit_fn)

    def _promote(self,
                 submit_fn: Callable[[Dict[str, Any]],
                                     Tuple[Optional[str], Dict[str, Any]]]) -> None:
        if not self.items:
            self.active_jobid = None
            self._save()
            return
        item = self.items[0]
        try:
            slurm_id, payload = submit_fn(item["spec"])
        except Exception as e:                       # never let a submit crash the poll
            self._halt(f"submitting queued '{item.get('label')}' raised "
                       f"{type(e).__name__}: {e}")
            return
        if not slurm_id:
            self._halt(f"submitting queued '{item.get('label')}' failed: "
                       f"{payload.get('error', 'unknown error')}")
            return
        self.items.pop(0)
        self.active_jobid = slurm_id
        self._save()


#: Process-wide singleton; tests swap in their own via
#: ``monkeypatch.setattr(fasrc_queue, "QUEUE", JobQueue(tmp_path))``.
QUEUE = JobQueue()
