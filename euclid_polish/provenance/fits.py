"""Embed/extract a :class:`Stamp` in a FITS header.

Four cards carry the compact identity quartet (the full record lives in the
sidecar). Keys are ≤8 chars and values are 7-bit ASCII, as FITS requires —
8-hex ids satisfy both. Operates on any astropy ``Header`` (passed in), so this
module pulls no astropy import of its own.
"""

from __future__ import annotations

from typing import Optional

from euclid_polish.provenance.ids import ProvId
from euclid_polish.provenance.records import Stamp


def write_stamp_cards(header, stamp: Stamp) -> None:
    """Write a stamp's identity to ``header`` (PROVID/PRODBY/PROVPAR/PROVSCHM)."""
    header["PROVID"] = (str(stamp.id), "Provenance id (8-hex)")
    if stamp.produced_by is not None:
        header["PRODBY"] = (str(stamp.produced_by), "Producing run id")
    if stamp.parents:
        header["PROVPAR"] = (
            ",".join(str(p) for p in stamp.parents)[:68], "Parent ids (comma-sep)")
    header["PROVSCHM"] = (int(stamp.schema_version), "Provenance schema version")


def read_stamp_cards(header) -> Optional[Stamp]:
    """The stamp encoded in ``header``, or ``None`` if it carries no PROVID."""
    if "PROVID" not in header:
        return None
    parents = tuple(
        ProvId(p) for p in str(header.get("PROVPAR", "")).split(",") if p)
    produced_by = (
        ProvId(str(header["PRODBY"]).strip()) if "PRODBY" in header else None)
    return Stamp(
        id=ProvId(str(header["PROVID"]).strip()),
        produced_by=produced_by,
        parents=parents,
        schema_version=int(header.get("PROVSCHM", 3)),
    )
