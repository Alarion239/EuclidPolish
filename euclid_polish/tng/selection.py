"""Pick TNG galaxies for the image grid / stack by a property *mode*.

The grid/stack render jobs run on FASRC (they need the FITS pixels), but the
galaxy *properties* (stellar mass / SFR / effective radius) are cached **locally**
by the histogram render (``data/_tng_infographics/tng_properties.csv``). So the
WebUI selects the galaxy ids locally — by mode — and passes them to the FASRC
job via ``--ids``.

Modes: ``random``, or the most/least extreme in stellar mass, SFR, or effective
radius. The non-random modes use a **temperature-weighted** sample over the
ranked candidates, so the pick isn't a fixed top-N:

  * ``temperature = 0``  → the deterministic extreme (exact top-N).
  * higher temperature   → spreads the draw across more of the top, still
                           weighted toward the extreme (rank weight
                           ``exp(-rank / (temperature·SOFTNESS))``).

Re-submitting re-rolls the draw (a fresh RNG each call).
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np

from euclid_polish.config import Config
from euclid_polish.tng.properties import PROPERTIES_CSV, _read_cache

# mode → (property key, want the largest first?)
_MODE_PROPERTY = {
    "most_massive":       ("mass_stars", True),   # stellar mass
    "least_massive":      ("mass_stars", False),
    "most_star_forming":  ("sfr",        True),
    "least_star_forming": ("sfr",        False),
    "biggest_radius":     ("reff",       True),
    "smallest_radius":    ("reff",       False),
}
#: All selectable modes, ``random`` first.
MODES = ("random", *_MODE_PROPERTY.keys())
#: Rank weight scale at temperature = 1 (≈ how many of the top get real weight).
_SOFTNESS = 20.0


def local_properties_path() -> str:
    """Path of the local property cache written by the histogram render."""
    return os.path.join(Config.DATA_DIR, "_tng_infographics", PROPERTIES_CSV)


def _random(ids: List[str], n: int, rng) -> List[str]:
    k = min(n, len(ids))
    return [ids[i] for i in rng.choice(len(ids), size=k, replace=False)]


def select_ids(props: Dict[str, Dict[str, float]], mode: str, n: int, *,
               temperature: float = 0.3, rng=None) -> List[str]:
    """Choose ``n`` galaxy ids from ``props`` by ``mode``.

    Property modes rank by the relevant field (NaN values excluded) and draw a
    temperature-weighted sample over the ranking; ``random`` (or any unknown
    mode) draws uniformly. Returns ``[]`` only if ``props`` is empty.
    """
    rng = rng if rng is not None else np.random.default_rng()
    ids = list(props.keys())
    if not ids:
        return []
    if mode not in _MODE_PROPERTY:                       # random / unknown
        return _random(ids, n, rng)

    key, want_desc = _MODE_PROPERTY[mode]
    cand = [(g, float(props[g].get(key, np.nan))) for g in ids]
    cand = [(g, v) for g, v in cand if np.isfinite(v)]
    if not cand:                                         # no usable values
        return _random(ids, n, rng)
    cand.sort(key=lambda t: t[1], reverse=want_desc)     # best (extreme) first
    ordered = [g for g, _ in cand]
    N = len(ordered)
    take = min(n, N)
    if temperature <= 0:
        return ordered[:take]                            # deterministic extreme
    # Rank-exponential weights: weight drops with distance from the extreme;
    # a larger temperature flattens it so more of the top gets sampled.
    s = max(1e-6, float(temperature)) * _SOFTNESS
    w = np.exp(-np.arange(N) / s)
    w /= w.sum()
    idx = rng.choice(N, size=take, replace=False, p=w)
    return [ordered[int(i)] for i in idx]


def pick_by_mode(mode: str, n: int, *, temperature: float = 0.3, rng=None
                 ) -> List[str]:
    """Read the local property cache and select ids; ``[]`` if the cache is
    absent/empty (caller falls back to a node-side random pick)."""
    props = _read_cache(local_properties_path())
    return select_ids(props, mode, n, temperature=temperature, rng=rng)
