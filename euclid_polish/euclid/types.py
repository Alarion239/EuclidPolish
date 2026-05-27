"""Legacy import shim — the real :class:`PSF` lives in
:mod:`euclid_polish.psf`.

Before the PSF-package consolidation this module owned the ``PSF``
dataclass directly. Many call sites still import it from here::

    from euclid_polish.euclid.types import PSF

That keeps working — we re-export the new :class:`PSF` (plus the
``estimate_fwhm`` legacy helper which is now the standalone
:func:`euclid_polish.psf.estimate_fwhm_pixels_1d`).

New code should prefer::

    from euclid_polish.psf import PSF, load_hst_f814w_psf, load_euclid_band_psf
"""

from __future__ import annotations

from euclid_polish.psf import PSF
from euclid_polish.psf.measurements import (
    estimate_fwhm_pixels_1d as estimate_fwhm,
)

__all__ = ["PSF", "estimate_fwhm"]
