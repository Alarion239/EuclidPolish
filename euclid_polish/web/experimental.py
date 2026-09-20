"""Feature flags for experimental WebUI surfaces.

EXPERIMENTAL: the round-trip supervision lane
---------------------------------------------
Besides the primary synthetic lane (|SR − scene| on simulated pairs), the
project keeps one additional supervision lane as future work:

  * **round-trip lane** — real Euclid sky cutouts scored through the
                         forward operator (page: Round-trip; steps:
                         ``euclid_sky_download``,
                         ``euclid_roundtrip_tfrecords``).

This is an **experimental feature kept for future work and disabled for
now**: the current focus is the synthetic-only training path, so its
nav links, pages/routes and FASRC step cards are hidden from the WebUI
while the flag below is ``False``. The data-generation backend is
intact — flipping the flag back to ``True`` restores every surface.

Gated consumers:
  * ``web/app.py``               — injects ``experimental_lanes`` into the
                                   Jinja context (nav links, step-card
                                   mounts in templates).
  * ``web/routes/sky.py``        — Round-trip page + inspect endpoint.
  * ``web/routes/fasrc.py``      — step listing, submit guard, artifact
                                   probes.
"""

from __future__ import annotations

EXPERIMENTAL_LANES_ENABLED: bool = False
