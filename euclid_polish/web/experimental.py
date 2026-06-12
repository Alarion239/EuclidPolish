"""Feature flags for experimental WebUI surfaces.

EXPERIMENTAL: alternative supervision lanes
-------------------------------------------
Besides the primary synthetic lane (|SR − scene| on simulated pairs), the
trainer supports three additional supervision lanes, each with its own data
pipeline and WebUI surfaces:

  * **HST lane**       — forward-modelled HST→Euclid pairs supervising
                         ``H ⊛ SR`` against the observed F814W image
                         (pages: HST tiles / HST cutouts / HST PSF /
                         HST Catalog; steps: ``download``, ``extract_psf``,
                         ``kernel``, ``tfrecords``).
  * **star-anchor lane** — real Euclid star cutouts with a sparse
                         delta-target at the catalog flux (step:
                         ``euclid_star_anchor_tfrecords``).
  * **round-trip lane** — real Euclid sky cutouts scored through the
                         forward operator (page: Round-trip; steps:
                         ``euclid_sky_download``,
                         ``euclid_roundtrip_tfrecords``).

These are **experimental features kept for future work and disabled for
now**: the current focus is the synthetic-only training path, so their
nav links, pages/routes, FASRC step cards, and training-form knobs are
hidden from the WebUI while the flag below is ``False``. The backend
(trainer lanes, generation scripts, data loaders) is intact — flipping
the flag back to ``True`` restores every surface.

Gated consumers:
  * ``web/app.py``               — injects ``experimental_lanes`` into the
                                   Jinja context (nav links, step-card
                                   mounts in templates).
  * ``web/routes/hst.py``        — HST tiles / cutouts / PSF pages + APIs.
  * ``web/routes/hstpairs.py``   — HST Catalog page + APIs.
  * ``web/routes/sky.py``        — Round-trip page + inspect endpoint.
  * ``web/routes/fasrc.py``      — step listing, submit guard, artifact
                                   probes.
  * ``web/static/fasrc_step_card.js`` — mirrors this flag as a JS const
                                   (lane knobs on the training card).
"""

from __future__ import annotations

EXPERIMENTAL_LANES_ENABLED: bool = False
