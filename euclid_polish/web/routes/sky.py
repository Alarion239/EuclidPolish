"""sky routes for the EuclidPolish web UI (extracted from app.py)."""
from __future__ import annotations

from typing import Any

from flask import render_template

from euclid_polish.web import fasrc_config
from euclid_polish.web.fasrc_fetcher import list_remote_dir


def _int_value(value: object) -> int:
    """Convert a JSON/form scalar to int without admitting arbitrary objects."""
    if isinstance(value, (str, bytes, bytearray, int, float)):
        return int(value)
    raise TypeError(f"expected an integer-like scalar, got {type(value).__name__}")


def register(app):

    # ---------------- Sky generation + forward ----------------
    @app.route("/sky")
    def sky_page():
        # Synthetic records are now generated on FASRC, so list the remote
        # records_v2 dir (one shallow ls) rather than the local disk. SSH is
        # up on this page (the connection gate is upstream); on failure we
        # just show "no records yet".
        cfg_loaded = fasrc_config.load()
        records_dir = f"{cfg_loaded.data_dir}/images/records_v2"
        tfrecords: list[dict[str, Any]] = []
        ok, entries, _ = list_remote_dir(
            records_dir, glob_pattern="*.tfrecord", max_entries=50,
        )
        if ok:
            for e in sorted(entries, key=lambda r: str(r.get("name", ""))):
                tfrecords.append({
                    "name":    e["name"],
                    "size_mb": f"{_int_value(e.get('size', 0)) / 1e6:.1f}",
                })
        return render_template("sky.html",
                               records_dir=records_dir,
                               tfrecords=tfrecords)

