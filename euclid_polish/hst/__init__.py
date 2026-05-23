"""
HST template operations.

Real HST/ACS F814W cutouts of COSMOS-field galaxies are used as HR
morphology templates in the synthetic generator — see
:mod:`euclid_polish.sky.hst_templates` for the renderer that composites
them onto the simulator's HR canvas.

Public surface mirrors :mod:`euclid_polish.euclid`:

  * :class:`HSTCutoutDownloader` — server-side cutouts via MAST HAPcut.
  * :class:`HSTTemplateLibrary`  — on-disk template store + CSV index.
  * :class:`HSTCutoutMetadata`   — per-cutout dataclass.

Authentication is a no-op shim for symmetry; all relevant HST data
products (drizzled ACS F814W) are public-domain and need no login.
"""

from euclid_polish.hst.types import HSTCutoutMetadata
from euclid_polish.hst.downloader import HSTCutoutDownloader, HSTDownloadConfig
from euclid_polish.hst.catalog import HSTTemplateLibrary
from euclid_polish.hst import auth

__all__ = [
    "HSTCutoutMetadata",
    "HSTCutoutDownloader",
    "HSTDownloadConfig",
    "HSTTemplateLibrary",
    "auth",
]
