"""Ops › Provenance: a read-only lineage browser over ``data/_prov``, the
sidecars next to the data and the checkpoint stamps
(:mod:`euclid_polish.web.helpers.provenance_index`). Local, never gated."""

from __future__ import annotations

import re
from typing import Any

from flask import jsonify, request

from euclid_polish.web import errors
from euclid_polish.web.helpers import provenance_index as pi

_ID_RE = re.compile(r"^[0-9a-fA-F]{8}$")
_MAX_LIMIT = 1000
_DEFAULT_LIMIT = 200


def _int_arg(name: str, default: int, lo: int, hi: int) -> int:
    try:
        value = int(request.args.get(name, default))
    except (TypeError, ValueError):
        value = default
    return max(lo, min(value, hi))


def _inspect_path(entry: dict[str, Any]) -> str | None:
    """Project-relative path the Inspect workspace can open (FITS only)."""
    path = entry.get("path")
    if not isinstance(path, str) or not path.lower().endswith((".fits", ".fits.gz", ".fit")):
        return None
    return path[2:] if path.startswith("./") else path


def _summary(index: pi.ProvIndex) -> dict[str, Any]:
    return {"ok": True, "total": len(index.entries), "counts": index.counts(),
            "roots": index.roots, "current_models": index.current_models,
            "truncated": index.truncated, "duplicates": index.duplicates,
            "built_at": index.built_at, "build_seconds": index.build_seconds}


def register(app):
    errors.json_errors_for(app, "/api/provenance")

    @app.route("/api/provenance/summary")
    def api_provenance_summary():
        """Counts per kind and per verdict, the scanned roots and the current
        models (active members with a provenance id)."""
        return jsonify(_summary(pi.get_index()))

    @app.route("/api/provenance/records")
    def api_provenance_records():
        """Search the records: ``q`` (tokens ANDed over id/kind/label/path/
        git/member/config type), ``kind`` (comma list), ``verdict``
        (current|stale|unknown), ``source`` (prov|sidecar|checkpoint),
        ``offset``/``limit`` (≤ 1000). Newest first."""
        verdict = (request.args.get("verdict") or "").strip()
        if verdict and verdict not in pi.VERDICTS:
            return jsonify({"ok": False, "error": f"verdict must be one of {list(pi.VERDICTS)}"}), 400
        index = pi.get_index()
        hits = index.search(q=request.args.get("q") or "", kind=request.args.get("kind") or "",
                            verdict=verdict, source=(request.args.get("source") or "").strip())
        offset = _int_arg("offset", 0, 0, 10**9)
        limit = _int_arg("limit", _DEFAULT_LIMIT, 1, _MAX_LIMIT)
        rows = [index.row(e) for e in hits[offset:offset + limit]]
        return jsonify({"ok": True, "total": len(hits), "offset": offset, "limit": limit,
                        "records": rows})

    @app.route("/api/provenance/record/<pid>")
    def api_provenance_record(pid: str):
        """One record: the stored JSON, its listing row (verdict, models),
        direct upstream/downstream and the transitive ancestors/descendants
        (first 300 each, with their hop depth)."""
        if not _ID_RE.match(pid):
            return jsonify({"ok": False, "error": "a provenance id is 8 hex characters"}), 400
        index = pi.get_index()
        entry = index.get(pid)
        if entry is None:
            error = f"no provenance record {pid.lower()} in the local store"
            return jsonify({"ok": False, "error": error}), 404
        anc, n_anc = index.walk(pid, "ancestors")
        desc, n_desc = index.walk(pid, "descendants")
        row = index.row(entry)
        return jsonify({
            "ok": True, "entry": row, "record": pi.read_record(entry),
            "upstream": index.upstream(pid), "downstream": index.downstream(pid)[:pi.LINEAGE_CAP],
            "ancestors": {"total": n_anc, "items": anc},
            "descendants": {"total": n_desc, "items": desc},
            "models": row["models"], "current_models": index.current_models,
            "inspect_path": _inspect_path(entry),
        })

    @app.route("/api/provenance/rebuild", methods=["POST"])
    def api_provenance_rebuild():
        """Re-scan every root now (the index is otherwise cached ~5 min)."""
        return jsonify(_summary(pi.get_index(rebuild=True)))
