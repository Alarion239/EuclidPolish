"""
Shared dataclasses for the HST template subsystem.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HSTCutoutMetadata:
    """Per-cutout metadata for an HST template on disk.

    Mirrors the per-row record persisted by :class:`HSTTemplateLibrary` so
    every cutout can be located, validated, and (re)rendered without
    re-querying MAST.

    Attributes
    ----------
    cosmos_id
        COSMOS2025 catalog ``id`` the cutout was downloaded for.
    ra_deg, dec_deg
        Requested sky position used at download time (degrees, ICRS).
    size_pix
        Native HST pixel side of the saved cutout (square).
    pixel_scale_arcsec
        Native HST pixel scale of the saved cutout (typically ~0.04″).
    filename
        Basename under :attr:`Config.HST_TEMPLATES_DIR` — full path is
        :func:`HSTTemplateLibrary.cutout_path`.
    filter_name
        HST filter the cutout was taken in (e.g. ``"F814W"``).
    bunit
        Pixel value units as reported by the cutout header
        (``ELECTRONS/S`` for ACS drizzled products).
    """

    cosmos_id:          int
    ra_deg:             float
    dec_deg:            float
    size_pix:           int
    pixel_scale_arcsec: float
    filename:           str
    filter_name:        str = "F814W"
    bunit:              str = "ELECTRONS/S"
