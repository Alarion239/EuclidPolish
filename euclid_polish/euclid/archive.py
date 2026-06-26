"""The Euclid archive as an operator: fetch a real cutout as an :class:`Image`.

    lr = EuclidArchive.fetch(ra, dec, size)   # -> Image (role 'real')

Absorbs the per-band archive download + ADU s⁻¹ → electron conversion that used
to live inline in the cutout layer. Lives in ``euclid/`` (not ``image/``) because
it needs the downloader + photometry — the image package stays a leaf.
"""

from __future__ import annotations

import os
import tempfile
from typing import Callable, Optional, Sequence

import numpy as np
from astropy.io import fits as _fits

from euclid_polish.config import Config
from euclid_polish.euclid.downloader import fetch_cutout_at
from euclid_polish.euclid.photometry import adu_per_s_to_electrons_factor
from euclid_polish.image import Image, Role
from euclid_polish.provenance.defaults import mint_id
from euclid_polish.provenance.records import Stamp

_LR_SCALE = Config.VIS_PIXEL_SCALE_ARCSEC   # 0.10 arcsec/pix


class EuclidArchive:
    """Operator face of the Euclid science archive."""

    @classmethod
    def fetch(
        cls,
        ra: float,
        dec: float,
        size: int,
        *,
        bands: Sequence[str] = Config.LR_INPUT_BAND_NAMES,
        store=None,
        fetch_plane: Optional[Callable] = None,
    ) -> Image:
        """Download a 4-band real Euclid cutout as an electron-unit :class:`Image`.

        Fetches each band from the archive (default) or via an injected
        ``fetch_plane`` (tests / offline), converts each ADU s⁻¹ image to
        electrons via the per-band ``MAGZERO`` header keyword, stacks to
        ``(H, W, len(bands))`` and tags the result ``role='real'`` with a freshly
        minted provenance stamp.

        Parameters
        ----------
        ra, dec : float
            ICRS coordinates in degrees.
        size : int
            Cutout side in VIS pixels (0.10″/pix reference grid).
        bands : sequence of str
            Band names (default ``Config.LR_INPUT_BAND_NAMES``).
        store : ProvStore, optional
            Provenance store; defaults to ``default_store()`` (guarded).
        fetch_plane : callable, optional
            ``(ra, dec, band_name, size) -> np.ndarray[float32]`` already in
            electrons. ``None`` uses the real archive download.
        """
        if fetch_plane is not None:
            planes = [
                np.asarray(fetch_plane(ra, dec, band_name, size), dtype=np.float32)
                for band_name in bands
            ]
        else:
            planes = cls._fetch_planes_from_archive(ra, dec, size, bands)

        data = np.stack(planes, axis=-1)
        return Image(
            data=data, pixel_scale_arcsec=_LR_SCALE, band_names=tuple(bands),
            is_clean=False, role=Role.REAL,
            stamp=Stamp(id=mint_id(store), schema_version=3))

    @classmethod
    def _fetch_planes_from_archive(
        cls, ra: float, dec: float, size: int, bands: Sequence[str],
    ) -> "list[np.ndarray]":
        """Real archive download path: fetch each band + MAGZERO → e⁻.

        Not exercised in tests (the ``fetch_plane`` injection bypasses it).
        """
        planes = []
        for band_name in bands:
            band = Config.get_band(band_name)
            with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as tf:
                tmp_path = tf.name
            try:
                ok, err = fetch_cutout_at(
                    ra=ra, dec=dec, band_name=band_name,
                    output_file=tmp_path, cutout_size_vis_pixels=size,
                )
                if not ok:
                    raise RuntimeError(
                        f"EuclidArchive.fetch: band {band_name} failed: {err}")
                with _fits.open(tmp_path) as hdul:
                    arr = hdul[0].data.astype(np.float32)
                    magzero = float(
                        hdul[0].header.get("MAGZERO", band.sim_zeropoint_e))
                factor = np.float32(adu_per_s_to_electrons_factor(magzero, band))
                planes.append((arr * factor).astype(np.float32))
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
        return planes
