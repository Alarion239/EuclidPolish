"""The artifact contract: a small Protocol plus a dataclass mixin.

Existing typed classes (``MultiBandSkyImage``, ``PSF``, ``PSFSet``,
``DifferentialKernel``, ``CatalogObject``) join the provenance system by carrying
a :class:`~euclid_polish.provenance.records.Stamp` and declaring their on-disk
:class:`~euclid_polish.provenance.records.Format`. The :class:`StampCarrier`
mixin supplies the shared implementation; :class:`Persistable` is the public
structural contract.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import ClassVar, Optional, Protocol, runtime_checkable

from euclid_polish.provenance.records import Format, Stamp


@runtime_checkable
class Persistable(Protocol):
    """Anything that can be identified and stamped on disk."""

    PROV_FORMAT: ClassVar[Format]

    def prov_stamp(self) -> Optional[Stamp]: ...

    def with_stamp(self, stamp: Stamp) -> "Persistable": ...


@dataclass
class StampCarrier:
    """Mixin giving a dataclass an optional, copy-on-write provenance stamp.

    ``stamp`` is keyword-only so subclasses with their own required fields keep
    a clean constructor signature regardless of field order.
    """

    stamp: Optional[Stamp] = field(default=None, kw_only=True)

    def prov_stamp(self) -> Optional[Stamp]:
        return self.stamp

    def with_stamp(self, stamp: Stamp):
        """Return a copy carrying ``stamp`` (the original is unchanged)."""
        return dataclasses.replace(self, stamp=stamp)
