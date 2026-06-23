"""Lens-finder routes — the SR-vs-LR lens-discovery pipeline (FASRC steps).

A thin page that mounts the three FASRC step cards (generate → build stamps →
train) in order. The cards self-load their metadata/history/submit UI from
``/api/fasrc/hst/status`` (see ``_step_card_mount.html``), so the route only
needs to render the template.
"""

from __future__ import annotations

from flask import render_template


def register(app):
    @app.route("/lensfinder")
    def lensfinder_page():
        return render_template("lensfinder.html")
