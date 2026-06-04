"""sky_render helpers for the EuclidPolish web UI (extracted from app.py)."""
from __future__ import annotations

from astropy.io import fits
from euclid_polish.config import Config
from euclid_polish.euclid.catalog import StarCatalog
from euclid_polish.sky.tfrecord import read_multiband_skyimages
from euclid_polish.sky.tfrecord import tfrecord_path
from euclid_polish.visualization.color import calibrated_rgb_panel
from euclid_polish.visualization.methods import plot_star_positions
from flask import abort
from typing import Any
from typing import Dict
from typing import Optional
import io
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os


def _export_sky_record_fits(
    subset: str, kind: str, band: str, index: int,
    records_dir: Optional[str] = None,
) -> str:
    """Materialise one sky record as a single-band FITS file.

    Caches the result under ``data/vis/sky_fits/`` so repeat clicks on
    the same record don't re-read the TFRecord. Returns the absolute
    path to the saved file. ``records_dir`` defaults to the local
    ``RECORDS_DIR_V2``; the /sky viewer passes the FASRC cache dir.
    """

    if subset not in ("train", "validate"):
        abort(400)
    if kind not in ("clean", "dirty", "hr"):
        abort(400)
    band_names = list(Config.LR_INPUT_BAND_NAMES)
    if kind == "hr":
        if band != "VIS":
            abort(400)
        band_idx = 0
    else:
        if band not in band_names:
            abort(400)
        band_idx = band_names.index(band)
    try:
        idx = int(index)
    except (TypeError, ValueError):
        abort(400)
    if idx < 0:
        abort(400)

    name = f"{kind}_{subset}"
    src_path = tfrecord_path(records_dir or Config.RECORDS_DIR_V2, name)
    if not os.path.exists(src_path):
        abort(404)

    out_dir = os.path.realpath(os.path.join(Config.VIS_DIR, "sky_fits"))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{kind}_{subset}_{band}_{idx:04d}.fits")
    if os.path.isfile(out_path) and os.path.getsize(out_path) > 0:
        return out_path

    # Stream just enough records to reach ``idx`` — TFRecords don't have
    # random access so we read sequentially. Cheap for typical idx ≤ ~50.
    records = read_multiband_skyimages(src_path, num_images=idx + 1)
    if not records or idx >= len(records):
        abort(404)
    record = records[idx]
    data = record.data
    if data.ndim == 2:
        plane = data
    elif data.ndim == 3:
        if band_idx >= data.shape[-1]:
            abort(404)
        plane = data[..., band_idx]
    else:
        abort(415)

    hdu = fits.PrimaryHDU(np.ascontiguousarray(plane, dtype=np.float32))
    hdu.header["OBJECT"] = (f"EuclidPolish {kind} {band}", "kind + band")
    hdu.header["SUBSET"] = (subset, "TFRecord subset")
    hdu.header["IDX"]    = (idx, "record index within subset")
    hdu.header["BAND"]   = (band, "band name (VIS, Y_E, J_E, H_E)")
    hdu.header["KIND"]   = (kind, "clean | dirty | hr")
    hdu.header["BUNIT"]  = ("e-", "electrons (raw, sign preserved)")
    if kind == "clean":
        hdu.header["CDELT1"] = (-Config.DEFAULT_PIXEL_SCALE / 3600.0,
                                 "HR pixel scale (degrees)")
        hdu.header["CDELT2"] = ( Config.DEFAULT_PIXEL_SCALE / 3600.0,
                                 "HR pixel scale (degrees)")
    elif kind == "dirty":
        hdu.header["CDELT1"] = (-Config.VIS_PIXEL_SCALE_ARCSEC / 3600.0,
                                 "LR pixel scale (degrees)")
        hdu.header["CDELT2"] = ( Config.VIS_PIXEL_SCALE_ARCSEC / 3600.0,
                                 "LR pixel scale (degrees)")
    hdu.writeto(out_path, overwrite=True)
    return out_path


def _render_sky_record_png(subset: str, kind: str, band: str,
                           index: int,
                           records_dir: Optional[str] = None) -> bytes:
    """Render one image from the multi-band TFRecords.

    ``records_dir`` defaults to ``Config.RECORDS_DIR_V2`` (the locally-
    generated sky records). The HST Catalog passes its own dir so it
    can reuse the exact same renderer against the FASRC-cached HST
    TFRecords — they use the same MultiBandSkyImage schema, so the
    only thing that changes is where the file lives.

    ``subset`` ∈ {"train", "validate"},
    ``kind`` ∈ {"clean", "dirty", "hr"}
      • ``clean`` → 4-band HR clean record
      • ``dirty`` → 4-band LR dirty record (PSF + noise + artifacts)
      • ``hr``    → 1-band VIS HR target (the network's training output)
    ``band``:
      * one of ``Config.LR_INPUT_BAND_NAMES`` → grayscale asinh of that band
      * ``"color"`` → 4-band Lupton RGB (solar-balanced). Requires the
        record to carry all four bands — clean/dirty only. For the
        VIS-only ``hr`` record, ``color`` falls back to VIS grayscale.
    """
    matplotlib.use("Agg")

    if subset not in ("train", "validate"):
        abort(400)
    if kind not in ("clean", "dirty", "hr"):
        abort(400)
    if band not in Config.LR_INPUT_BAND_NAMES and band != "color":
        abort(400)
    name = f"{kind}_{subset}"
    path = tfrecord_path(records_dir or Config.RECORDS_DIR_V2, name)
    if not os.path.exists(path):
        abort(404)
    # Stream just enough records to reach ``index``.
    max_to_read = max(index + 1, 1)
    records = read_multiband_skyimages(path, num_images=max_to_read)
    if not records or index >= len(records):
        abort(404)
    img = records[min(index, len(records) - 1)]
    data = img.data

    if band == "color" and data.shape[-1] >= len(Config.LR_INPUT_BAND_NAMES):
        # 4-band solar-balanced RGB with per-channel [p1, p99.5]
        # normalisation — same dynamic-range convention as the
        # grayscale single-band panels in the rest of the sky tab.
        # ``clean`` records live on the HR grid (band-independent
        # asinh-stretch knee unspecified) — use the VIS knee as the
        # shared reference, matching how the rest of the UI treats HR.
        rgb = calibrated_rgb_panel(
            data, band_names=Config.LR_INPUT_BAND_NAMES,
            scheme="vis_nisp", reference="solar",
            stretch="asinh",
            asinh_scale_e=float(Config.BAND_VIS.asinh_stretch_scale_e),
        )
        fig, ax = plt.subplots(figsize=(6.5, 6.5))
        ax.imshow(np.clip(rgb, 0.0, 1.0), origin="lower",
                  interpolation="nearest")
        ax.set_title(
            f"{kind} {subset} · color (4-band, solar) · idx {img.index}  "
            f"({data.shape[0]}×{data.shape[1]} @ "
            f"{img.pixel_scale_arcsec:.3f}\"/pix)", fontsize=10,
        )
        ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, dpi=110, bbox_inches="tight", format="png")
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()

    # Grayscale single-band rendering. HR records carry shape (H, W, 1)
    # (VIS only), so any band selection falls back to channel 0.
    if data.shape[-1] == 1 or band == "color":
        plane = data[..., 0]
        band_name = "VIS"
    else:
        k = list(Config.LR_INPUT_BAND_NAMES).index(band)
        plane = data[..., k]
        band_name = band
    bcfg = Config.get_band(band_name)
    stretched = np.arcsinh(plane / float(bcfg.asinh_stretch_scale_e))
    lo, hi = np.percentile(stretched, [1.0, 99.7])
    if hi <= lo: hi = lo + 1.0
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.imshow(stretched, cmap="gray_r", origin="lower", vmin=lo, vmax=hi,
              interpolation="nearest")
    ax.set_title(f"{kind} {subset} · {band_name} · idx {img.index}  "
                 f"({data.shape[0]}×{data.shape[1]} @ "
                 f"{img.pixel_scale_arcsec:.3f}\"/pix)", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, dpi=110, bbox_inches="tight", format="png")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def _render_sky_record_pair_png(subset: str, band: str, index: int,
                                records_dir: Optional[str] = None) -> bytes:
    """Side-by-side triptych for one (clean, dirty, HR) pair.

    Pulls the three TFRecord files at the same ``index`` and renders
    them in a single figure: clean HR (4-band) ┃ dirty LR (4-band) ┃
    HR-VIS target (1-band).

    Display range = SHARED 99.7-percentile clip across **all three
    panels combined**. Sharing the clip across panels means "mid-grey
    here" corresponds to the same electron count as "mid-grey there",
    which is the invariant a side-by-side comparison needs. The
    single-panel kinds (clean / dirty / hr alone) keep the standard
    per-image clip from :func:`_render_sky_record_png`.

    ``band`` ∈ ``Config.LR_INPUT_BAND_NAMES`` ∪ {"color"}. With
    ``band="color"`` the clean + dirty panels use the 4-band Lupton
    RGB; the HR panel always uses VIS grayscale (it's a single-band
    record).
    """
    matplotlib.use("Agg")

    if subset not in ("train", "validate"):
        abort(400)
    if band not in Config.LR_INPUT_BAND_NAMES and band != "color":
        abort(400)

    base_dir = records_dir or Config.RECORDS_DIR_V2
    panels: Dict[str, Any] = {}
    for kind in ("clean", "dirty", "hr"):
        path = tfrecord_path(base_dir, f"{kind}_{subset}")
        if not os.path.exists(path):
            abort(404)
        records = read_multiband_skyimages(path, num_images=max(index + 1, 1))
        if not records or index >= len(records):
            abort(404)
        panels[kind] = records[min(index, len(records) - 1)]

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.4))

    # Pre-compute the SHARED 99.7-percentile clip across all three
    # grayscale panels. Skipped when band="color" because color panels
    # use the Lupton RGB renderer instead of asinh.
    def _grayscale_plane(img, band_name: str):
        data = img.data
        if data.shape[-1] == 1:
            plane = data[..., 0]
            bn = "VIS"
        else:
            k = list(Config.LR_INPUT_BAND_NAMES).index(band_name)
            plane = data[..., k]
            bn = band_name
        knee = float(Config.get_band(bn).asinh_stretch_scale_e)
        return np.arcsinh(plane / knee)

    shared_lo, shared_hi = 0.0, 1.0
    if band != "color":
        all_stretched = [_grayscale_plane(panels[k], band)
                         for k in ("clean", "dirty", "hr")]
        union = np.concatenate([s.ravel() for s in all_stretched])
        shared_lo, shared_hi = np.percentile(union, [1.0, 99.7])
        if shared_hi <= shared_lo:
            shared_hi = shared_lo + 1.0

    def _show_grayscale(ax, img, band_name: str, title: str) -> None:
        stretched = _grayscale_plane(img, band_name)
        ax.imshow(stretched, cmap="gray_r", origin="lower",
                  vmin=shared_lo, vmax=shared_hi,
                  interpolation="nearest")
        ax.set_title(
            f"{title}  ({img.data.shape[0]}×{img.data.shape[1]} @ "
            f"{img.pixel_scale_arcsec:.3f}\"/pix)", fontsize=10,
        )
        ax.set_xticks([]); ax.set_yticks([])

    def _show_color(ax, img, title: str) -> None:
        rgb = calibrated_rgb_panel(
            img.data, band_names=Config.LR_INPUT_BAND_NAMES,
            scheme="vis_nisp", reference="solar", stretch="asinh",
            asinh_scale_e=float(Config.BAND_VIS.asinh_stretch_scale_e),
        )
        ax.imshow(np.clip(rgb, 0.0, 1.0), origin="lower",
                  interpolation="nearest")
        ax.set_title(
            f"{title}  ({img.data.shape[0]}×{img.data.shape[1]} @ "
            f"{img.pixel_scale_arcsec:.3f}\"/pix)", fontsize=10,
        )
        ax.set_xticks([]); ax.set_yticks([])

    for ax, kind, title in [
        (axes[0], "clean", f"clean HR · idx {panels['clean'].index}"),
        (axes[1], "dirty", f"dirty LR · idx {panels['dirty'].index}"),
        (axes[2], "hr",    f"HR target · idx {panels['hr'].index}"),
    ]:
        img = panels[kind]
        # HR target is VIS-only — never has 4 bands, so always grayscale.
        if band == "color" and kind != "hr":
            _show_color(ax, img, title)
        else:
            _show_grayscale(ax, img, band, title)

    fig.suptitle(
        f"HST → Euclid pair · {subset} · idx {index} · band={band}",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, dpi=100, bbox_inches="tight", format="png")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def _render_catalog_view_png(view: str, output_dir: str) -> bytes:
    """Render a catalog visualization: positions or magnitude histogram."""
    matplotlib.use("Agg")

    cat = StarCatalog(output_dir)
    if not cat.exists():
        abort(404)
    data = cat.load()
    stars = data.get("stars", [])
    if not stars:
        abort(404)
    if view == "positions":
        fig = plot_star_positions(stars)
    elif view == "magnitudes":
        mags = [s.get("magnitude") for s in stars
                if s.get("magnitude") is not None]
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.hist(mags, bins=40, color="#2a5db0", edgecolor="white")
        ax.set_xlabel("VIS magnitude (AB)"); ax.set_ylabel("count")
        ax.set_title(f"Catalog mag distribution  "
                     f"(median = {float(np.median(mags)):.2f})")
        fig.tight_layout()
    else:
        abort(400)
    buf = io.BytesIO()
    fig.savefig(buf, dpi=110, bbox_inches="tight", format="png")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()
