"""Testable, non-interactive inference operations the CLI menus call.

Pure functions over the OO operator surface — ``Model.upsample``,
``EuclidArchive.fetch``, ``Image``/``ImageSet`` — with no input()/questionary.
"""
from __future__ import annotations

import os
from typing import List, Optional

from euclid_polish.euclid.archive import EuclidArchive
from euclid_polish.image import Image, ImageSet, Role
from euclid_polish.visualization.reconstruction import plot_imageset


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
        Colour regime ("eye" or "calibrated").
    store : ProvStore, optional
        Provenance store threaded to ``model.upsample`` (defaults internally).

    Returns
    -------
    list of str
        Paths of the written PNGs.
    """
    os.makedirs(out_dir, exist_ok=True)
    paths: List[str] = []
    for i, lr_img in enumerate(lr_images):
        lr = lr_img.with_role(Role.LR)
        sr = model.upsample(lr, store=store)
        members = [lr, sr]
        if hr_images is not None and i < len(hr_images):
            members.append(hr_images[i].with_role(Role.HR))
        png = os.path.join(out_dir, f"reconstruction_{i:03d}.png")
        plot_imageset(ImageSet.from_images(members), png, regime=regime)
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
        Provenance store threaded to fetch + upsample (defaults internally).
    fetch_plane : callable, optional
        ``(ra, dec, band_name, size) -> np.ndarray`` (electrons) for
        tests/offline. ``None`` uses the real archive download.

    Returns
    -------
    tuple of (str, str)
        ``(sr_fits_path, sr_png_path)``.
    """
    os.makedirs(out_dir, exist_ok=True)
    lr = EuclidArchive.fetch(ra=ra, dec=dec, size=size,
                             store=store, fetch_plane=fetch_plane)
    sr = model.upsample(lr, store=store)
    fits_path = os.path.join(out_dir, "SR.fits")
    png_path = os.path.join(out_dir, "SR.png")
    sr.save_fits(fits_path)
    plot_imageset(ImageSet.from_images([lr, sr]), png_path, regime=regime)
    return fits_path, png_path
