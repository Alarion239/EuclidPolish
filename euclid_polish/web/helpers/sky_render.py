"""sky_render helpers for the EuclidPolish web UI (extracted from app.py)."""
from __future__ import annotations

from astropy.io import fits
from euclid_polish.config import Config
from euclid_polish.catalog.star_catalog import StarCatalog
from euclid_polish.image.tfio import read_images
from euclid_polish.image.tfio import tfrecord_path
from euclid_polish.visualization.methods import plot_star_positions
from flask import abort
from typing import List
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
    # ``hr`` records are 4-band since the VIS+NISP-output change, so any
    # band is selectable; a legacy 1-channel record (pre-change) simply
    # 404s below for bands beyond channel 0.
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
    records = read_images(src_path, num_images=idx + 1)
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


def _saturation_cutoff(valid_mags, invalid_mags, bins):
    """Suggested oversaturation cutoff: the faint edge of the bright regime
    where the invalid (saturated) fraction is still ≥ 50%.

    Stars brighter than the returned magnitude are predominantly rejected.
    Returns ``None`` when no bin crosses 50% invalid (no clear cutoff).
    """
    v_counts, _ = np.histogram(valid_mags, bins=bins)
    i_counts, _ = np.histogram(invalid_mags, bins=bins)
    cutoff = None
    for k in range(len(bins) - 1):
        total = v_counts[k] + i_counts[k]
        if total == 0:
            continue
        if i_counts[k] / total >= 0.5:
            # right (fainter) edge of this still-saturated bin
            cutoff = float(bins[k + 1])
    return cutoff


def _render_saturation_view(stars):
    """2×2 grid (one panel per band) of VIS-magnitude histograms split into
    valid (green) and invalid/saturated (red), with a suggested per-band
    oversaturation cutoff line."""
    band_names = [b.name for b in Config.BANDS]
    n = len(band_names)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 4.0 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, name in zip(axes, band_names):
        valid_mags, invalid_mags = [], []
        for s in stars:
            m = s.get("magnitude")
            if m is None or not np.isfinite(m):
                continue
            if StarCatalog.is_valid(s, band=name):
                valid_mags.append(float(m))
            elif StarCatalog.is_corrupted(s, band=name):
                invalid_mags.append(float(m))     # disjoint from valid

        combined = valid_mags + invalid_mags
        if not combined:
            ax.set_title(f"{name}")
            ax.text(0.5, 0.5, "no processed stars in this band",
                    ha="center", va="center", color="#888",
                    transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
            continue

        bins = np.linspace(min(combined), max(combined), 30)
        ax.hist([valid_mags, invalid_mags], bins=bins, stacked=True,
                color=["#2e9e4f", "#d1453b"],
                edgecolor="white", linewidth=0.4,
                label=[f"valid (N={len(valid_mags)})",
                       f"invalid/saturated (N={len(invalid_mags)})"])

        cutoff = _saturation_cutoff(valid_mags, invalid_mags, bins)
        if cutoff is not None:
            ax.axvline(cutoff, ls="--", lw=1.4, color="#222")
            ax.annotate(f"cutoff ≈ {cutoff:.2f}", xy=(cutoff, 0.96),
                        xycoords=("data", "axes fraction"),
                        xytext=(4, -4), textcoords="offset points",
                        ha="left", va="top", fontsize=9, color="#222")

        ax.set_title(name)
        ax.set_xlabel("VIS magnitude (AB)  · brighter ←")
        ax.set_ylabel("count")
        ax.legend(fontsize=8, loc="upper right")

    # Blank any unused panels (e.g. if BANDS isn't a multiple of ncols).
    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle("Per-band saturation — valid (green) vs invalid/saturated (red); "
                 "dashed line = suggested oversaturation cutoff", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def _great_circle_arcmin(ra1, dec1, ra2, dec2):
    """Great-circle separation in arcmin (haversine). Degrees in; ``ra2`` /
    ``dec2`` may be arrays (broadcast against scalar ``ra1`` / ``dec1``)."""
    r1, d1 = np.radians(ra1), np.radians(dec1)
    r2, d2 = np.radians(ra2), np.radians(dec2)
    a = (np.sin((d2 - d1) / 2.0) ** 2
         + np.cos(d1) * np.cos(d2) * np.sin((r2 - r1) / 2.0) ** 2)
    return np.degrees(2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))) * 60.0


def _render_psf_clusters_png(output_dir: Optional[str],
                             psf_path: Optional[str]) -> bytes:
    """Local sky map of the ePSF-extraction clusters + per-cluster diameter.

    ``psf_path`` is the VIS ePSF FITS to read cluster centroids from — pass
    the *same* synced copy the rest of the /psfs page reads (the FASRC cache),
    so the plot tracks a freshly-synced ePSF rather than a stale local file.
    The spatial clustering is shared across bands, so VIS centroids suffice.
    Membership + angular diameter are reconstructed by assigning each catalog
    star valid in all four bands to its nearest centroid — exactly the Voronoi
    assignment K-Means used at extraction. Far-apart fields get their own
    panel (zoomed local view, not full sky).

    Aborts 404 when the VIS ePSF FITS isn't on disk (PSFs not extracted /
    synced yet) — the page hides the panel in that case. A missing catalog
    degrades to a centroids-only map (diameters shown as "—").
    """
    matplotlib.use("Agg")

    if not psf_path or not os.path.isfile(psf_path):
        abort(404)
    cra, cdec = [], []
    with fits.open(psf_path) as hdul:
        for hdu in hdul[1:]:                       # HDU0 = mean PSF
            ra = hdu.header.get("RA")
            dec = hdu.header.get("DEC")
            if ra is not None and dec is not None:
                cra.append(float(ra)); cdec.append(float(dec))
    if not cra:
        abort(404)
    cra, cdec = np.asarray(cra), np.asarray(cdec)
    n_clusters = len(cra)

    # Catalog stars valid in all four bands → nearest-centroid assignment.
    sra, sdec = [], []
    cat = StarCatalog(output_dir) if output_dir else None
    if cat is not None and cat.exists():
        band_names = [b.name for b in Config.BANDS]
        for s in cat.load().get("stars", []):
            ra, dec = s.get("ra"), s.get("dec")
            if ra is None or dec is None or not (np.isfinite(ra) and np.isfinite(dec)):
                continue
            if all(StarCatalog.is_valid(s, band=bn) for bn in band_names):
                sra.append(float(ra)); sdec.append(float(dec))
    sra, sdec = np.asarray(sra), np.asarray(sdec)
    if len(sra):
        labels = np.array([int(np.argmin(_great_circle_arcmin(r, d, cra, cdec)))
                           for r, d in zip(sra, sdec)])
    else:
        labels = np.zeros(0, dtype=int)

    # Per-cluster member count + angular diameter (max pairwise separation).
    diam: List[Optional[float]] = [None] * n_clusters
    for k in range(n_clusters):
        mem = np.where(labels == k)[0]
        if len(mem) >= 2:
            mr, md = sra[mem], sdec[mem]
            diam[k] = max(float(_great_circle_arcmin(mr[i], md[i], mr, md).max())
                          for i in range(len(mem)))

    # Group centroids into spatially-disjoint regions (connected components
    # under a 2° link) so far-apart fields each get their own zoomed panel.
    LINK_ARCMIN = 120.0
    region = [-1] * n_clusters
    n_regions = 0
    for i in range(n_clusters):
        if region[i] != -1:
            continue
        stack, region[i] = [i], n_regions
        while stack:
            u = stack.pop()
            du = _great_circle_arcmin(cra[u], cdec[u], cra, cdec)
            for v in range(n_clusters):
                if region[v] == -1 and v != u and du[v] <= LINK_ARCMIN:
                    region[v] = n_regions
                    stack.append(v)
        n_regions += 1

    cmap = plt.get_cmap("tab20")
    ncols = min(n_regions, 2)
    nrows = (n_regions + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.5 * ncols, 5.4 * nrows),
                             squeeze=False)
    axes = axes.ravel()
    for r in range(n_regions):
        ax = axes[r]
        ks = [k for k in range(n_clusters) if region[k] == r]
        dec0 = float(np.mean(cdec[ks]))
        for k in ks:
            col = cmap(k % 20)
            mem = np.where(labels == k)[0]
            if len(mem):
                ax.scatter(sra[mem], sdec[mem], s=6, color=col, alpha=0.55,
                           linewidths=0)
            ax.scatter([cra[k]], [cdec[k]], marker="x", s=70, color="black",
                       linewidths=1.5, zorder=5)
            lbl = f"{k}\nØ{diam[k]:.1f}'" if diam[k] is not None else f"{k}\nØ—"
            ax.annotate(lbl, (cra[k], cdec[k]), textcoords="offset points",
                        xytext=(5, 4), fontsize=7, zorder=6)
        ax.set_xlabel("RA (deg)"); ax.set_ylabel("Dec (deg)")
        ax.set_aspect(1.0 / max(0.05, np.cos(np.radians(dec0))))
        ax.invert_xaxis()                          # RA increases to the left
        ax.set_title(f"region {r + 1}: {len(ks)} cluster(s)")
        ax.grid(alpha=0.2)
    for ax in axes[n_regions:]:
        ax.set_visible(False)

    ds = [d for d in diam if d is not None]
    summary = (f"{n_clusters} ePSF clusters in {n_regions} region(s)"
               + (f"; angular diameter {min(ds):.1f}–{max(ds):.1f}′ "
                  f"(median {float(np.median(ds)):.1f}′)"
                  if ds else "; download the catalog to get diameters"))
    fig.suptitle("PSF extraction clusters — " + summary, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    buf = io.BytesIO()
    fig.savefig(buf, dpi=110, bbox_inches="tight", format="png")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def _render_catalog_view_png(view: str, output_dir: str) -> bytes:
    """Render a catalog visualization: positions, magnitude histogram, or
    per-band saturation (valid vs invalid)."""
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
    elif view == "saturation":
        fig = _render_saturation_view(stars)
    else:
        abort(400)
    buf = io.BytesIO()
    fig.savefig(buf, dpi=110, bbox_inches="tight", format="png")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()
