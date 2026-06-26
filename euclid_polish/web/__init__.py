"""
Localhost web interface for EuclidPolish.

Thin Flask layer over the existing modules — every page delegates to
``StarCatalog`` / ``psf_inventory`` / ``SkySimulator`` /
``ObservationSimulator`` rather than reimplementing pipeline logic. The
slow operations (generate, forward, extract-psf) run in background
threads tracked by :mod:`euclid_polish.web.jobs`; the UI polls the job
endpoints for live progress.

Bound to ``127.0.0.1`` by default. Not designed to be exposed to the
public internet — no auth, no rate limiting.

Entry points:
    python -m euclid_polish.web
    python scripts/serve.py
"""

from euclid_polish.web.app import create_app

__all__ = ["create_app"]
