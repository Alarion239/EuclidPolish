"""Route groups for the web UI; each module exposes ``register(app)``.

``MODULES`` is the registry :func:`euclid_polish.web.app.create_app` loops
over. Adding a route group is one line here (import + tuple entry);
``tests/test_route_registry.py`` fails when a ``routes/*.py`` file is missing
from the tuple. Handlers that need the FASRC SSH session are marked with
:func:`euclid_polish.web.fasrc_gate.requires_fasrc`; everything else must work
offline.
"""

from euclid_polish.web.routes import (
    archive_fields,
    auth,
    config,
    cutouts,
    ensemble,
    evaluation,
    fasrc,
    figures,
    files,
    galaxy_distributions,
    git,
    jwst_euclid,
    model,
    noise,
    population_comparison,
    poster,
    provenance,
    psfs,
    real,
    realism,
    sky_atlas,
    star_distribution,
    system,
    tng,
    tracking,
    viewer,
    views,
)

MODULES = (
    config,
    auth,
    cutouts,
    psfs,
    tng,
    poster,
    archive_fields,
    population_comparison,
    noise,
    galaxy_distributions,
    star_distribution,
    realism,
    model,
    ensemble,
    evaluation,
    views,
    jwst_euclid,
    real,
    sky_atlas,
    files,
    git,
    tracking,
    provenance,
    viewer,
    figures,
    fasrc,
    system,
)

__all__ = ["MODULES"]
