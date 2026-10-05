"""
Localhost web interface for EuclidPolish.

Thin Flask layer over the existing modules — it serves the React console
and the API it calls, delegating to the pipeline packages rather than
reimplementing pipeline logic. Cluster work (generation, PSF extraction,
training) is submitted to FASRC as SLURM steps
(:mod:`euclid_polish.web.fasrc_pipeline`); the slow local operations
(syncs, SR generation, evaluations, combiner fits …) run in background
threads tracked by :mod:`euclid_polish.web.jobs`; the UI polls the job
endpoints for live progress.

Bound to ``127.0.0.1`` by default. Not designed to be exposed to the
public internet — no auth, no rate limiting.

Entry points:
    python -m euclid_polish.web
    python scripts/serve.py
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from euclid_polish.web.app import create_app

__all__ = ["create_app"]


def __getattr__(name: str):
    if name == "create_app":
        from euclid_polish.web.app import create_app

        return create_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
