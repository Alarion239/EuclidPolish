"""Typed, dependency-light access to the local TNG property catalog."""

from __future__ import annotations

import csv
import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

type TNGCatalogSourceIdentity = tuple[str | None, int, int]


def _as_float(value: float | str | None) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


@dataclass(frozen=True, slots=True)
class TNGGalaxyProperties:
    """Physical group-catalog properties for one TNG subhalo.

    Missing or malformed catalog values are represented by ``nan``. A genuine
    zero SFR is retained as zero and can therefore be distinguished from a
    missing measurement through :attr:`zero_sfr`.
    """

    stellar_mass_msun: float
    sfr_msun_yr: float
    halo_mass_msun: float
    stellar_halfmass_radius_kpc: float

    def __post_init__(self) -> None:
        for name in (
            "stellar_mass_msun",
            "sfr_msun_yr",
            "halo_mass_msun",
            "stellar_halfmass_radius_kpc",
        ):
            object.__setattr__(self, name, _as_float(getattr(self, name)))

    @property
    def log_stellar_mass(self) -> float:
        """Return ``log10(M*/M_sun)``, or ``nan`` for an invalid mass."""
        mass = self.stellar_mass_msun
        if not math.isfinite(mass) or mass <= 0.0:
            return float("nan")
        return math.log10(mass)

    @property
    def zero_sfr(self) -> bool:
        """Whether the catalog records a genuine, finite zero SFR."""
        return math.isfinite(self.sfr_msun_yr) and self.sfr_msun_yr == 0.0

    @property
    def log_ssfr(self) -> float:
        """Return ``log10(SFR/M*)`` in yr^-1, or ``nan`` when undefined."""
        log_mass = self.log_stellar_mass
        sfr = self.sfr_msun_yr
        if not math.isfinite(log_mass) or not math.isfinite(sfr) or sfr <= 0.0:
            return float("nan")
        return math.log10(sfr) - log_mass


@dataclass(frozen=True, slots=True)
class TNGPropertyCatalog(Mapping[str, TNGGalaxyProperties]):
    """Immutable mapping from subhalo id to typed physical properties."""

    _rows: Mapping[str, TNGGalaxyProperties] = field(repr=False)
    source_identity: TNGCatalogSourceIdentity

    def __post_init__(self) -> None:
        rows: dict[str, TNGGalaxyProperties] = {}
        for raw_id, properties in self._rows.items():
            subhalo_id = str(raw_id).strip()
            if not subhalo_id:
                raise ValueError("TNG property catalog contains an empty subhalo id")
            if not isinstance(properties, TNGGalaxyProperties):
                raise TypeError(
                    "TNG property catalog rows must be TNGGalaxyProperties"
                )
            rows[subhalo_id] = properties
        object.__setattr__(self, "_rows", MappingProxyType(rows))

    @classmethod
    def read(cls, path: str | Path | None) -> TNGPropertyCatalog:
        """Read ``tng_properties.csv``; a missing path yields an empty catalog."""
        if path is None:
            return cls({}, (None, 0, 0))

        source = Path(path).expanduser().resolve()
        try:
            status = source.stat()
        except OSError:
            return cls({}, (str(source), 0, 0))

        rows: dict[str, TNGGalaxyProperties] = {}
        with source.open(newline="", encoding="utf-8") as handle:
            for raw in csv.DictReader(handle):
                subhalo_id = str(raw.get("id", "")).strip()
                if not subhalo_id:
                    continue
                if subhalo_id in rows:
                    raise ValueError(
                        f"duplicate TNG property row for subhalo {subhalo_id}"
                    )
                rows[subhalo_id] = TNGGalaxyProperties(
                    stellar_mass_msun=_as_float(raw.get("mass_stars")),
                    sfr_msun_yr=_as_float(raw.get("sfr")),
                    halo_mass_msun=_as_float(raw.get("m_halo")),
                    stellar_halfmass_radius_kpc=_as_float(raw.get("reff")),
                )
        return cls(
            rows,
            (str(source), int(status.st_size), int(status.st_mtime_ns)),
        )

    def __getitem__(self, subhalo_id: str) -> TNGGalaxyProperties:
        return self._rows[str(subhalo_id)]

    def __iter__(self) -> Iterator[str]:
        return iter(self._rows)

    def __len__(self) -> int:
        return len(self._rows)

    def __repr__(self) -> str:
        return (
            f"TNGPropertyCatalog(rows={len(self)}, "
            f"source={self.source_identity[0]!r})"
        )
