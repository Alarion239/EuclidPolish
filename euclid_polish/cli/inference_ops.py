"""Testable, non-interactive inference operations the CLI menus call.

Pure functions over the OO surface (Model / Cutout) — no input()/questionary.
"""
from __future__ import annotations

import os
from typing import List, Optional

import numpy as np  # noqa: F401  (kept for callers/extension)

from euclid_polish.cutout.base import EuclidLRCutout, HRCutout, SyntheticLRCutout
from euclid_polish.provenance.defaults import default_store
from euclid_polish.image import Image


def _lr_cutout(img: Image, store) -> SyntheticLRCutout:
    # Reuse the record's embedded id when present (preserves lineage); else mint.
    stamp = img.prov_stamp()
    cid = stamp.id if stamp is not None else store.mint()
    return SyntheticLRCutout(image=img, id=cid)


def _hr_cutout(img: Image, store) -> HRCutout:
    stamp = img.prov_stamp()
    cid = stamp.id if stamp is not None else store.mint()
    return HRCutout(image=img, id=cid)


def reconstruct_and_render(
    lr_images: List[Image],
    model,
    out_dir: str,
    *,
    hr_images: Optional[List[Image]] = None,
    regime: str = "eye",
    store=None,
) -> List[str]:
    """Super-resolve each LR image with ``model`` and save a reconstruction PNG.

    Parameters
    ----------
    lr_images : list of Image
        The dirty LR inputs.
    model : Model
        A loaded :class:`~euclid_polish.model.Model`.
    out_dir : str
        Output directory (created if absent).
    hr_images : list of Image, optional
        Ground-truth HR targets (same length/order as ``lr_images``); when
        present the HR panel + residual metrics are rendered.
    regime : str
        Colour regime passed to ``save_png`` ("eye" or "calibrated").
    store : ProvStore, optional
        Provenance store; defaults to ``default_store()``.

    Returns
    -------
    list of str
        Paths of the written PNGs.
    """
    os.makedirs(out_dir, exist_ok=True)
    _store = store if store is not None else default_store()
    paths: List[str] = []
    for i, lr_img in enumerate(lr_images):
        lr_cut = _lr_cutout(lr_img, _store)
        sr = model.upsample(lr_cut, store=_store)
        hr_cut = (_hr_cutout(hr_images[i], _store)
                  if hr_images is not None and i < len(hr_images) else None)
        png = os.path.join(out_dir, f"reconstruction_{i:03d}.png")
        sr.save_png(png, regime=regime, lr=lr_cut, hr=hr_cut)
        paths.append(png)
    return paths


def fetch_and_superresolve(
    *,
    ra: float,
    dec: float,
    size: int,
    model,
    out_dir: str,
    regime: str = "eye",
    store=None,
    fetch_plane=None,
) -> tuple:
    """Fetch a real Euclid cutout at ``(ra, dec)``, super-resolve it, and save.

    Parameters
    ----------
    ra, dec : float
        ICRS coordinates in degrees.
    size : int
        Cutout side in VIS pixels (0.10"/pix grid).
    model : Model
        A loaded :class:`~euclid_polish.model.Model`.
    out_dir : str
        Output directory (created if absent).
    regime : str
        Colour regime for the PNG ("eye" or "calibrated").
    store : ProvStore, optional
        Provenance store; defaults to ``default_store()``.
    fetch_plane : callable, optional
        ``(ra, dec, band_name, size) -> np.ndarray`` (electrons) for
        tests/offline. ``None`` uses the real archive download.

    Returns
    -------
    tuple of (str, str)
        ``(sr_fits_path, sr_png_path)``.
    """
    os.makedirs(out_dir, exist_ok=True)
    _store = store if store is not None else default_store()
    lr = EuclidLRCutout.fetch(ra=ra, dec=dec, size=size,
                              store=_store, fetch_plane=fetch_plane)
    sr = model.upsample(lr, store=_store)
    fits_path = os.path.join(out_dir, "SR.fits")
    png_path = os.path.join(out_dir, "SR.png")
    sr.save_fits(fits_path)
    sr.save_png(png_path, regime=regime, lr=lr)
    return fits_path, png_path
