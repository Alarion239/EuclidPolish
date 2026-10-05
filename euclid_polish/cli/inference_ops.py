"""Testable, non-interactive inference operations the CLI menus call.

Pure functions over the production model — the STARFULL ensemble plus its
production combiner from :func:`~euclid_polish.eval.ensemble_infer.load_eval_ensemble`,
predicted through :func:`~euclid_polish.eval.ensemble_infer.sr_from_model` —
``EuclidCatalog.fetch`` and ``Image``/``ImageSet``, with no input()/questionary.
"""
from __future__ import annotations

import os
from collections.abc import Callable, Sequence

import numpy as np

from euclid_polish.catalog import EuclidCatalog
from euclid_polish.config import Config
from euclid_polish.eval.ensemble_infer import sr_from_model
from euclid_polish.image import Image, ImageSet, Role
from euclid_polish.provenance.defaults import mint_id
from euclid_polish.provenance.records import Stamp
from euclid_polish.visualization.reconstruction import plot_imageset


def production_sr(model, lr: Image, *, store=None) -> Image:
    """The production super-resolution of ``lr`` as an SR :class:`Image`.

    ``model`` is the production model (an
    :class:`~euclid_polish.eval.ensemble_infer.EvalEnsemble`); the SR is its
    combiner applied to the members it runs (:func:`sr_from_model`), never a
    plain member average behind the combiner's back. The SR stamp's parent is
    ``lr``'s id when ``lr`` carries one; the new id is minted via ``store``
    (or the default store), degrading to an unstamped-but-correct artifact.
    """
    _lr_vis, sr, _members = sr_from_model(model, lr.data)
    sr = np.asarray(sr, dtype=np.float32)
    bands = (lr.band_names if sr.ndim == 3 and sr.shape[-1] == len(lr.band_names)
             else ("VIS",))
    parents = (lr.stamp.id,) if lr.stamp is not None else ()
    return Image(data=sr, pixel_scale_arcsec=Config.DEFAULT_PIXEL_SCALE,
                 band_names=bands, is_clean=True, role=Role.SR,
                 index=lr.index, subset=lr.subset,
                 stamp=Stamp(id=mint_id(store), parents=parents,
                             schema_version=3, subset=lr.subset))


def _psnr_db(a: np.ndarray, b: np.ndarray, peak: float) -> float:
    """PSNR (dB) over the overlapping region; ``inf`` for an exact match."""
    h, w = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
    mse = float(np.mean((np.asarray(a, np.float64)[:h, :w]
                         - np.asarray(b, np.float64)[:h, :w]) ** 2))
    return float("inf") if mse <= 0.0 else float(10.0 * np.log10(peak * peak / mse))


def evaluate_production_sr(
    model,
    lr_images: Sequence[Image],
    hr_images: Sequence[Image],
    *,
    on_progress: Callable[[int, int], None] | None = None,
) -> dict:
    """Score the production SR of each LR field against its HR target.

    Fields pair by record ``index``; an LR field without a target is skipped.
    Returns ``{"n_scored", "psnr_raw", "psnr_stretched"}`` — the per-field PSNR
    averaged over the scored fields, in raw electrons (peak
    ``Config.PSNR_PEAK_E``) and in the asinh space the members train in
    (knee ``Config.STRETCH_SCALE_E``, peak ``Config.PSNR_PEAK_STRETCHED``), the
    same definitions :meth:`~euclid_polish.ensemble.EnsembleModel.evaluate`
    reports per member.
    """
    hr_by_index = {h.index: h for h in hr_images}
    knee = float(Config.STRETCH_SCALE_E)
    raw: list[float] = []
    stretched: list[float] = []
    for i, lr in enumerate(lr_images):
        hr = hr_by_index.get(lr.index)
        if hr is not None:
            _lr_vis, sr, _members = sr_from_model(model, lr.data)
            sr = np.asarray(sr, np.float32)
            truth = np.asarray(hr.data, np.float32)
            raw.append(_psnr_db(sr, truth, float(Config.PSNR_PEAK_E)))
            stretched.append(_psnr_db(np.arcsinh(sr / knee), np.arcsinh(truth / knee),
                                      float(Config.PSNR_PEAK_STRETCHED)))
        if on_progress is not None:
            on_progress(i + 1, len(lr_images))
    return {"n_scored": len(raw),
            "psnr_raw": float(np.mean(raw)) if raw else float("nan"),
            "psnr_stretched": float(np.mean(stretched)) if stretched else float("nan")}


def reconstruct_and_render(
    lr_images: list[Image],
    model,
    out_dir: str,
    *,
    hr_images: list[Image] | None = None,
    regime: str = "eye",
    store=None,
) -> list[str]:
    """Super-resolve each LR image with ``model`` and save a reconstruction PNG.

    Parameters
    ----------
    lr_images : list of Image
        The dirty LR inputs.
    model : EvalEnsemble
        The production model (see :func:`production_sr`).
    out_dir : str
        Output directory (created if absent).
    hr_images : list of Image, optional
        Ground-truth HR targets (same length/order as ``lr_images``); when
        present the HR panel + residual metrics are rendered.
    regime : str
        Colour regime ("eye" or "calibrated").
    store : ProvStore, optional
        Provenance store the SR stamps are minted in (defaults internally).

    Returns
    -------
    list of str
        Paths of the written PNGs.
    """
    os.makedirs(out_dir, exist_ok=True)
    paths: list[str] = []
    for i, lr_img in enumerate(lr_images):
        lr = lr_img.with_role(Role.LR)
        sr = production_sr(model, lr, store=store)
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
    catalog=None,
) -> tuple:
    """Fetch a real Euclid cutout at ``(ra, dec)``, super-resolve it, and save.

    Parameters
    ----------
    ra, dec : float
        ICRS coordinates in degrees.
    size : int
        Cutout side in VIS pixels (0.10"/pix grid).
    model : EvalEnsemble
        The production model (see :func:`production_sr`).
    out_dir : str
        Output directory (created if absent).
    regime : str
        Colour regime for the PNG ("eye" or "calibrated").
    store : ProvStore, optional
        Provenance store threaded to the fetch and the SR stamp (defaults
        internally).
    catalog : EuclidCatalog, optional
        Authenticated client; defaults to an unauthenticated instance that
        reuses whatever astroquery session is active in the process.

    Returns
    -------
    tuple of (str, str)
        ``(sr_fits_path, sr_png_path)``.
    """
    os.makedirs(out_dir, exist_ok=True)
    cat = catalog or EuclidCatalog._unauthenticated()
    lr = cat.fetch(ra=ra, dec=dec, size=size, store=store)
    sr = production_sr(model, lr, store=store)
    fits_path = os.path.join(out_dir, "SR.fits")
    png_path = os.path.join(out_dir, "SR.png")
    sr.save_fits(fits_path)
    plot_imageset(ImageSet.from_images([lr, sr]), png_path, regime=regime)
    return fits_path, png_path
