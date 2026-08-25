"""fits_render helpers for the EuclidPolish web UI (extracted from app.py)."""
from __future__ import annotations

import io
import os
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.io import fits as _fits
from astropy.visualization import AsinhStretch, ImageNormalize, MinMaxInterval
from flask import abort
from PIL import Image

from euclid_polish.config import BandConfig, Config
from euclid_polish.psf import PSF
from euclid_polish.psf.psf_library import load_all_band_psfs
from euclid_polish.visualization.presentation_style import (
    AXIS_LABEL_SIZE,
    TICK_LABEL_SIZE,
    apply_presentation_figure,
)
from euclid_polish.web.helpers._const import _CUTOUT_FNAME_RE
from euclid_polish.web.helpers.status import _cached_fasrc_psf_dir


def _two_dimensional_image_data(hdu: object) -> np.ndarray | None:
    """Return data from a concrete image HDU when it is two-dimensional."""
    if not isinstance(hdu, (fits.PrimaryHDU, fits.ImageHDU, fits.CompImageHDU)):
        return None
    data = hdu.data
    if data is None or data.ndim != 2:
        return None
    return np.asarray(data)


def _resolve_cutout_path(band_name: str, filename: str,
                         output_dir: str) -> str:
    """Map ``(band, filename)`` → safe absolute FITS path.

    Refuses anything that doesn't pass the strict filename regex or that
    resolves outside the per-band cutout directory after symlink expansion.
    """
    if not _CUTOUT_FNAME_RE.match(filename):
        abort(400)
    try:
        Config.get_band(band_name)
    except Exception:
        abort(404)
    band_dir = Config.cutout_dir_for_band(
        band_name, root=os.path.join(output_dir, "cutouts"),
    )
    full = os.path.realpath(os.path.join(band_dir, filename))
    if not full.startswith(os.path.realpath(band_dir) + os.sep):
        abort(403)
    if not os.path.isfile(full):
        abort(404)
    return full


def _render_fits_to_png_adaptive(fits_path: str, size: int) -> bytes:
    """Render any 2-D FITS image to PNG with a data-adaptive stretch.

    The band-aware :func:`_render_fits_to_png` hardcodes an asinh knee
    (``band.asinh_stretch_scale_e``, ~1000 e⁻ by default) tuned for
    Euclid sky cutouts. That stretch is meaningless for files outside
    that domain — a unit-flux PSF sums to 1 over 511² pixels (values
    ~10⁻⁶–10⁻²), a differential kernel can have signed wings, dark
    frames hover near zero. Applying the cutout knee to those leaves
    `arcsinh(x / 1000) ≈ x / 1000` with a near-empty histogram, then
    the 1.0/99.7-percentile clip stretches numerical noise instead of
    real structure — what looked "weird" on /inspect for the diff
    kernel was exactly this mismatch.

    The /inspect route can't know what kind of file it's showing, so
    use ``MinMaxInterval + AsinhStretch(a=0.01)`` from
    astropy.visualization. MinMax keeps the full data range in view
    (a kernel's central peak sits 5 decades above the typical pixel —
    anything that trims outliers, like ZScale or a percentile clip,
    pushes the peak into saturation and the centre renders as a
    checkerboard instead of a smooth blob). The aggressive asinh knee
    (``a=0.01`` = transition at 1 % of the normalised range) compresses
    that dynamic range so the faint wings and the bright core are
    visible in the same frame.
    """

    with fits.open(fits_path, memmap=False) as hdul:
        data = None
        for hdu in hdul:
            image_data = _two_dimensional_image_data(hdu)
            if image_data is not None:
                data = np.asarray(image_data, dtype=np.float64)
                break
    if data is None:
        abort(415)

    finite = np.isfinite(data)
    data = np.zeros_like(data) if not finite.any() else np.where(finite, data, np.nanmedian(data[finite]))

    # ``clip=True`` guarantees the output stays in [0, 1] even when the
    # underlying stretch would push values outside (signed residuals
    # near the edges of the data range).
    norm = ImageNormalize(
        data, interval=MinMaxInterval(), stretch=AsinhStretch(a=0.01),
        clip=True,
    )
    normed = np.asarray(norm(data), dtype=np.float64)
    img8 = (255 * (1.0 - normed)).astype(np.uint8)   # gray_r: bright → dark
    img8 = np.flipud(img8)                            # FITS lower-origin → PIL
    pil = Image.fromarray(img8, mode="L")
    if size and size != pil.size[0]:
        pil = pil.resize((int(size), int(size)), Image.Resampling.NEAREST)
    buf = io.BytesIO()
    pil.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _render_fits_to_png(fits_path: str, band: BandConfig,
                        size: int | None = None) -> bytes:
    """Load a cutout FITS, apply per-band asinh stretch, return PNG bytes.

    - Asinh stretch knee comes from ``band.asinh_stretch_scale_e``.
    - vmin/vmax clip at the 1.0 / 99.7 percentiles of the stretched image.
    - Optional ``size`` resamples (nearest-neighbour) to a square thumbnail.

    Use this only for files whose pixel scale you know (Euclid sky
    cutouts, per-band thumbnails). For the universal /inspect view —
    where the file could be anything — use
    :func:`_render_fits_to_png_adaptive` instead, which derives the
    contrast window from the data and doesn't assume cutout units.
    """

    with fits.open(fits_path, memmap=False) as hdul:
        data = None
        for hdu in hdul:
            image_data = _two_dimensional_image_data(hdu)
            if image_data is not None:
                data = np.asarray(image_data, dtype=np.float32)
                break
    if data is None:
        abort(415)

    finite = np.isfinite(data)
    # Replace NaN/inf with the median of finite pixels so the stretch is well-defined.
    data = np.zeros_like(data) if not finite.any() else np.where(finite, data, np.nanmedian(data[finite]))

    stretched = np.arcsinh(data / float(band.asinh_stretch_scale_e))
    lo, hi = np.percentile(stretched, [1.0, 99.7])
    if hi <= lo:
        hi = lo + 1.0
    norm = np.clip((stretched - lo) / (hi - lo), 0.0, 1.0)
    # gray_r style: bright pixels → dark ink. Easier to read against white UI.
    img8 = (255 * (1.0 - norm)).astype(np.uint8)
    # FITS orientation: origin lower-left. PIL is origin upper-left → flip.
    img8 = np.flipud(img8)
    pil = Image.fromarray(img8, mode="L")
    if size is not None and size > 0 and size != pil.size[0]:
        pil = pil.resize((int(size), int(size)), Image.Resampling.NEAREST)
    buf = io.BytesIO()
    pil.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _list_band_cutouts(band_name: str, output_dir: str) -> list[str]:
    """Sorted list of cutout filenames present for ``band_name``."""
    band_dir = Config.cutout_dir_for_band(
        band_name, root=os.path.join(output_dir, "cutouts"),
    )
    if not os.path.isdir(band_dir):
        return []
    return sorted(
        f for f in os.listdir(band_dir)
        if f.lower().endswith(".fits") and _CUTOUT_FNAME_RE.match(f)
    )


def _read_fits_header_rows(path: str) -> list[dict[str, Any]]:
    """Return one row per HDU with its header laid out for the table view.

    Each row: ``{hdu_index, name, kind, shape, dtype, cards: [(key, value, comment), ...]}``.
    """
    rows: list[dict[str, Any]] = []
    with fits.open(path, memmap=False) as hdul:
        for i, hdu in enumerate(hdul):
            cards: list[tuple[str, str, str]] = []
            for card in hdu.header.cards:
                key  = str(card.keyword)
                val  = str(card.value)
                # Trim long string values — full text is visible in the
                # downloaded FITS, the UI just needs an at-a-glance view.
                if len(val) > 80:
                    val = val[:77] + "…"
                cmt  = str(card.comment)
                cards.append((key, val, cmt))
            shape = getattr(hdu.data, "shape", None) if hdu.is_image else None
            dtype = getattr(hdu.data, "dtype", None) if hdu.is_image else None
            rows.append({
                "hdu_index": i,
                "name":      hdu.name or f"HDU{i}",
                "kind":      type(hdu).__name__,
                "shape":     list(shape) if shape is not None else None,
                "dtype":     str(dtype) if dtype is not None else None,
                "cards":     cards,
            })
    return rows


def _fits_file_info(path: str) -> dict[str, Any]:
    """Lightweight ``ls``-style metadata for the inspector header card."""
    st = os.stat(path)
    return {
        "abspath":  path,
        "basename": os.path.basename(path),
        "size_kb":  round(st.st_size / 1024, 1),
        "mtime":    st.st_mtime,
    }


def _render_psf_panel_png(band: str | None, dpi: int = 110) -> bytes:
    """Render one band (or all four) on a log-stretch panel as PNG bytes."""
    matplotlib.use("Agg")

    # Render the FASRC-extracted PSFs (pulled to the local cache), not a
    # stale local copy. None → nothing on FASRC yet.
    psf_dir = _cached_fasrc_psf_dir()
    if not psf_dir:
        abort(404)
    psfs = load_all_band_psfs(psf_dir=psf_dir)
    if band and band != "all":
        if band not in psfs:
            abort(404)
        names = [band]
    else:
        names = [b.name for b in Config.BANDS if b.name in psfs]
    if not names:
        abort(404)
    n = len(names)
    logged = {
        name: np.log10(np.clip(psfs[name].data, 1e-8, None))
        for name in names
    }
    vmin = min(float(np.nanmin(data)) for data in logged.values())
    vmax = max(float(np.nanmax(data)) for data in logged.values())
    fig, axes = plt.subplots(
        1, n, figsize=(5.2 * n, 5.8), squeeze=False,
        layout="constrained",
    )
    image = None
    for ax, name in zip(axes[0], names, strict=False):
        p = psfs[name]
        image = ax.imshow(
            logged[name], cmap="viridis", origin="lower",
            interpolation="nearest", vmin=vmin, vmax=vmax,
        )
        ax.set_title(
            f"{name}\n{p.data.shape[0]}×{p.data.shape[1]} pixels"
            f" · {p.pixel_scale:.3f}″ pixel⁻¹",
            pad=12,
        )
        ax.set_xticks([]); ax.set_yticks([])
    assert image is not None
    colorbar = fig.colorbar(image, ax=axes[0].tolist(), shrink=0.82, pad=0.02)
    colorbar.set_label("log₁₀ normalized PSF intensity", fontsize=AXIS_LABEL_SIZE)
    colorbar.ax.tick_params(labelsize=TICK_LABEL_SIZE)
    apply_presentation_figure(fig)
    buf = io.BytesIO()
    fig.savefig(
        buf, dpi=max(72, min(int(dpi), 600)),
        bbox_inches="tight", format="png",
    )
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def _psf_preview_payload(band: str | None = None,
                         *, max_side: int = 256) -> dict[str, Any]:
    """Return bounded display arrays for browser-side PSF rendering.

    This reads only the already-synchronised FASRC cache. The React console
    owns colour mapping and canvas rendering; this helper avoids putting
    matplotlib or a multi-hundred-MB FITS transfer on the page request path.
    ``HDU0`` is the mean PSF for the multi-extension PSFSet format.
    """
    valid = {b.name for b in Config.BANDS}
    if band and band != "all" and band not in valid:
        abort(404)

    psf_dir = _cached_fasrc_psf_dir()
    if not psf_dir:
        return {
            "available": False,
            "source": "FASRC cache",
            "message": "No synchronised FASRC ePSFs are available.",
            "bands": [],
        }

    names = ([band] if band and band != "all" else
             [b.name for b in Config.BANDS])
    previews: list[dict[str, Any]] = []
    for name in names:
        cfg = Config.get_band(name)
        path = os.path.join(psf_dir, cfg.psf_fits_filename)
        if not os.path.isfile(path):
            continue
        psf = PSF.from_fits(path)
        data = np.asarray(psf.data, dtype=np.float32)
        finite = np.isfinite(data)
        if not finite.any():
            continue
        data = np.where(finite, data, 0.0)
        logged = np.log10(np.clip(data, 1e-12, None))
        original_shape = [int(v) for v in logged.shape]
        if max(logged.shape) > max_side:
            ys = np.linspace(0, logged.shape[0] - 1, max_side).astype(int)
            xs = np.linspace(0, logged.shape[1] - 1, max_side).astype(int)
            logged = logged[np.ix_(ys, xs)]
        previews.append({
            "name": name,
            "values": logged.astype(float).tolist(),
            "shape": original_shape,
            "pixel_scale": float(psf.pixel_scale),
            "fwhm": (float(psf.fwhm_arcsec)
                     if psf.fwhm_arcsec is not None else cfg.psf_fwhm_arcsec),
            "n_psf": int(fits.getheader(path, 0).get("NPSF", 1)),
        })

    return {
        "available": bool(previews),
        "source": "FASRC cache",
        "message": (None if previews else
                     "No synchronised FASRC ePSFs are available."),
        "bands": previews,
    }


def _arrays_to_fits_bytes(
    arrays,
    header_meta=None,
    primary_name: str | None = None,
) -> bytes:
    """Pack a dict of ``{name: array}`` into a multi-HDU FITS file.

    The first array goes into the PrimaryHDU; the rest become
    ``ImageHDU`` extensions with their dict key as ``EXTNAME``.
    ``header_meta`` keys are copied into every HDU's header
    (FITS keywords are uppercased and truncated to 8 chars).

    Used by the "Download FITS" sister endpoints of every renderer
    so the user can pull the raw linear array even when the view
    they're looking at is asinh-stretched, percentile-clipped, and
    colormapped for display.
    """
    header_meta = dict(header_meta or {})
    hdus = []
    items = list(arrays.items())
    if not items:
        raise ValueError("arrays dict is empty")
    for i, (name, arr) in enumerate(items):
        arr = np.asarray(arr, dtype=np.float32)
        hdu = _fits.PrimaryHDU(data=arr) if i == 0 else _fits.ImageHDU(data=arr)
        # EXTNAME on the primary HDU isn't strictly required but lots
        # of viewers (DS9, ginga) treat it as the human-readable label
        # so set it consistently.
        hdu.header["EXTNAME"] = name[:60]
        for k, v in header_meta.items():
            key = str(k).upper()[:8]
            if isinstance(v, (int, float, bool, str)):
                hdu.header[key] = v
        if primary_name and i == 0:
            hdu.header["EXTNAME"] = primary_name[:60]
        hdus.append(hdu)
    buf = io.BytesIO()
    _fits.HDUList(hdus).writeto(buf, overwrite=True)
    buf.seek(0)
    return buf.getvalue()
