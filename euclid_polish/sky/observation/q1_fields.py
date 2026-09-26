"""The three Euclid Quick Release 1 (Q1) deep fields.

One committed table of field centres, shared by every population query and
sky-coverage helper. A field *label* is derived from a sky position with
:func:`q1_field_for` instead of trusting a stored string: the early archive
cutout script swapped EDF-F and EDF-S, so archive samples and noise tiles
written by it carry the wrong name.

EDF-F (Fornax) lies next to the Chandra Deep Field South (RA ≈ 53°, Dec ≈
−28°); EDF-S is further south (RA ≈ 61°, Dec ≈ −48°); EDF-N sits at the north
ecliptic pole. ``radius_deg`` is the cone radius the Q1 population queries
use around each centre.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Q1Field:
    """One Q1 deep field: centre (ICRS degrees) and query-cone radius."""

    name: str
    ra: float
    dec: float
    radius_deg: float


Q1_FIELDS: tuple[Q1Field, ...] = (
    Q1Field("EDF-N", 269.733, 66.018, 6.0),
    Q1Field("EDF-S", 61.241, -48.423, 6.0),
    Q1Field("EDF-F", 52.932, -28.088, 6.0),
)

#: ``(ra, dec, radius_deg)`` cones in the ``Q1_FIELDS`` order.
Q1_FIELD_CONES: tuple[tuple[float, float, float], ...] = tuple(
    (field.ra, field.dec, field.radius_deg) for field in Q1_FIELDS
)


def angular_separation_deg(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    """Great-circle separation of two ICRS positions (degrees, haversine)."""
    r1, d1, r2, d2 = map(math.radians, (ra1, dec1, ra2, dec2))
    a = (math.sin((d2 - d1) / 2.0) ** 2
         + math.cos(d1) * math.cos(d2) * math.sin((r2 - r1) / 2.0) ** 2)
    return math.degrees(2.0 * math.asin(math.sqrt(min(1.0, max(0.0, a)))))


def q1_field_for(ra: float, dec: float) -> str | None:
    """Name of the Q1 deep field whose cone contains ``(ra, dec)``, else ``None``.

    The nearest centre wins if cones ever overlap (they do not today).
    """
    best: tuple[float, str] | None = None
    for field in Q1_FIELDS:
        separation = angular_separation_deg(ra, dec, field.ra, field.dec)
        if separation <= field.radius_deg and (best is None or separation < best[0]):
            best = (separation, field.name)
    return None if best is None else best[1]


__all__ = [
    "Q1_FIELDS",
    "Q1_FIELD_CONES",
    "Q1Field",
    "angular_separation_deg",
    "q1_field_for",
]
