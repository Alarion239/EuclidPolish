#!/usr/bin/env python
"""Extract the F814W ePSF from downloaded COSMOS HLSP tiles.

Scans each HLSP tile for bright unsaturated point sources, then runs
:class:`photutils.psf.EPSFBuilder` to construct a high-S/N empirical
PSF. Mirrors the Euclid VIS ePSF flow in
:mod:`euclid_polish.psf.psf_extractor`.

Output: ``$DATA_DIR/hst_psf/F814W.fits`` — a single FITS file with the
oversampled empirical PSF + provenance headers (n_stars used, tile
indices the stars came from, RMS reconstruction error).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings

import numpy as np
from astropy.io import fits
from astropy.nddata import NDData
from astropy.stats import sigma_clipped_stats
from astropy.table import Table
from photutils.detection import DAOStarFinder
from photutils.psf import EPSFBuilder, EPSFStar, EPSFStars, extract_stars

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from euclid_polish.config import Config
from euclid_polish.observability import Reporter

# HST PSF-extraction constants now live on Config.HST (see
# euclid_polish/config.py): HLSP_DIR_NAME, PSF_DIR_NAME, PSF_FILE_NAME,
# STARS_DIR_NAME, FALLBACK_PIX_SCALE_ARCSEC, EPSF_OVERSAMPLING,
# PSF_HALF_SIDE_PIX, MAX_STARS_PER_TILE, MAX_UNCOVERED_FRAC,
# MIN_STAMP_PEAK_SNR, MAX_PEAK_OFFCENTER_PX.


def _pixel_scale_from_header(header) -> float:
    """Read pixel scale (arcsec/pix) from a FITS WCS header.

    Tries CDELT first, then the CD matrix. Falls back to
    :data:`Config.HST.FALLBACK_PIX_SCALE_ARCSEC` only if neither is present.
    """
    if "CDELT1" in header:
        return abs(float(header["CDELT1"])) * 3600.0
    if "CD1_1" in header:
        cd11 = float(header["CD1_1"])
        cd12 = float(header.get("CD1_2", 0.0))
        return float(np.sqrt(cd11 ** 2 + cd12 ** 2)) * 3600.0
    return Config.HST.FALLBACK_PIX_SCALE_ARCSEC


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-stars", type=int, default=200,
                   help="Target number of stars to use (more = higher S/N, "
                        "slower). The actual number used is reported in "
                        "the FITS header.")
    p.add_argument("--half-side", type=int, default=Config.HST.PSF_HALF_SIDE_PIX,
                   help="Half-side of the FINAL ePSF in HLSP pixels; the "
                        "ePSF spans (2·half+1) px at the ~0.05\"/pix HLSP "
                        "scale (255 → 511², 511 → 1023²). 2·half+1 is always "
                        "odd, so the PSF stays centred on a pixel. Larger "
                        "captures more wings at higher per-tile I/O cost. "
                        "Changing it invalidates the cached star stamps "
                        "(different side) → forces a full tile re-scan.")
    p.add_argument("--extract-margin-frac", type=float, default=0.08,
                   help="Extract star stamps this fraction larger than "
                        "--half-side, then trim the extra border off the "
                        "built ePSF. EPSFBuilder's smoothing leaves edge "
                        "artifacts on the outermost pixels; the margin "
                        "pushes those into the trimmed region so the final "
                        "(2·half+1)² PSF has clean borders. Default 0.08 "
                        "(8%); 0 disables.")
    p.add_argument("--input-dir", default=None,
                   help="Directory of HLSP tiles. Defaults to $DATA_DIR/hst_hlsp/.")
    p.add_argument("--output-dir", default=None,
                   help="Where to write the ePSF FITS. Defaults to "
                        "$DATA_DIR/hst_psf/.")
    p.add_argument("--reuse-stars", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="If $DATA_DIR/hst_stars/ already contains per-star "
                        "FITS stamps of the right size from a prior run, "
                        "feed them straight into EPSFBuilder instead of "
                        "re-scanning every HLSP tile with DAOStarFinder + "
                        "extract_stars. The bottleneck on FASRC is the tile "
                        "I/O (HLSP tiles are ~500 MB each); reusing skips "
                        "it entirely when only the ePSF needs rebuilding "
                        "(e.g. trying a different EPSFBuilder setting). "
                        "Use --no-reuse-stars to force a full recut.")
    p.add_argument("--dry-run", action="store_true",
                   help="Report what would be done and exit.")
    return p.parse_args()


def _load_cached_star_stamps(
    stars_dir: str, *, half_side: int = Config.HST.PSF_HALF_SIDE_PIX,
) -> tuple:
    """Load per-star FITS stamps from a previous run, as ``EPSFStar`` objects.

    Returns ``(stamps, pix_scale, n_tiles_src)``. ``stamps`` is empty if
    ``stars_dir`` is missing, has no files matching the
    ``star_NNNN_{side}.fits`` naming, or every candidate has the wrong
    shape. ``pix_scale`` and ``n_tiles_src`` come from the first stamp's
    header (``PIXSCALE``, ``NTILESRC``) so the downstream ePSF FITS keeps
    the same provenance fields it would have had from a fresh extract.

    Only stamps whose side matches ``2*half_side + 1`` are kept — mixing
    sides would silently corrupt EPSFBuilder (it assumes a single grid).
    The ``cutout_center`` is set to the geometric centre; EPSFBuilder
    iteratively refines centroids, so this is a good-enough starting
    guess and matches what ``extract_stars`` would produce.
    """

    if not os.path.isdir(stars_dir):
        return [], Config.HST.FALLBACK_PIX_SCALE_ARCSEC, 0

    expected_side = 2 * half_side + 1
    suffix = f"_{expected_side}.fits"
    files = sorted(
        f for f in os.listdir(stars_dir)
        if f.startswith("star_") and f.endswith(suffix)
    )
    if not files:
        return [], Config.HST.FALLBACK_PIX_SCALE_ARCSEC, 0

    stamps: list = []
    pix_scale: float | None = None
    n_tiles_src: int = 0
    for fname in files:
        fpath = os.path.join(stars_dir, fname)
        try:
            with fits.open(fpath, memmap=False) as hdul:
                arr = np.asarray(hdul[0].data, dtype=np.float32)
                h = hdul[0].header
                if pix_scale is None:
                    pix_scale  = float(h.get("PIXSCALE", Config.HST.FALLBACK_PIX_SCALE_ARCSEC))
                    n_tiles_src = int(h.get("NTILESRC", 0))
        except Exception:
            continue
        if arr.shape != (expected_side, expected_side):
            continue
        # Apply the same clean-stamp cut to cached stamps, so re-running
        # with --reuse-stars after a bad run (which may have cached
        # seam/noise stamps) still yields a clean ePSF.
        if not _is_clean_star_stamp(arr):
            continue
        side = arr.shape[0]
        stamps.append(EPSFStar(data=arr, cutout_center=(side / 2.0, side / 2.0)))

    return stamps, (pix_scale if pix_scale is not None else Config.HST.FALLBACK_PIX_SCALE_ARCSEC), n_tiles_src


def _find_stars_in_tile(data: np.ndarray, *, max_n: int,
                        sigma: float = 5.0,
                        half_side: int = Config.HST.PSF_HALF_SIDE_PIX) -> Table:
    """Detect bright unsaturated point sources in one tile.

    Returns an astropy Table with at least ``x`` and ``y`` columns
    (renamed from DAOStarFinder's ``xcentroid``/``ycentroid``) so
    :func:`photutils.psf.extract_stars` accepts it directly. See
    :func:`_extract_stamps_from_tile` for the call site.
    """

    # Background from COVERED sky only. Drizzle fills no-coverage with 0;
    # including those zeros biases the σ-clipped stats, giving a wrong 50σ
    # threshold that lets noise through. Mask exact-0 / non-finite first.
    covered = np.isfinite(data) & (data != 0.0)
    if not covered.any():
        return None
    mean, median, std = sigma_clipped_stats(data[covered], sigma=3.0)
    if not np.isfinite(std) or std <= 0:
        return None
    # Threshold: 50× sigma — point-source-bright but not saturated.
    finder = DAOStarFinder(
        threshold=50 * std, fwhm=4.0,    # ~4 px FWHM at 0.05"/pix → 0.20"
        sharplo=0.4, sharphi=0.8,         # rejects extended / cosmic-ray
        roundlo=-0.4, roundhi=0.4,
    )
    sources = finder(data - median)
    if sources is None or len(sources) == 0:
        return None
    # Sort by peak brightness, take top max_n; reject ones too close to edge.
    sources.sort("peak")
    sources.reverse()
    H, W = data.shape
    border = half_side + 5
    keep = [
        (border < r["xcentroid"] < W - border
         and border < r["ycentroid"] < H - border)
        for r in sources
    ]
    sources = sources[keep]
    if max_n < len(sources):
        sources = sources[:max_n]
    # extract_stars wants ``x`` and ``y``, not ``xcentroid``/``ycentroid``.
    # Without this rename it raises a misleading "When inputting multiple
    # catalogs, each one must have a 'x' and 'y' column" error — internally
    # it always wraps a single catalog in a list before checking columns.
    sources["x"] = sources["xcentroid"]
    sources["y"] = sources["ycentroid"]
    return sources


def _is_clean_star_stamp(arr: np.ndarray) -> bool:
    """Accept only fully-covered, clearly-peaked, *isolated* star stamps.

    Rejects the failure modes large stamps expose:

      * **Coverage hole / tile seam** — a block of exact-0 (drizzle
        no-coverage fill) or NaN pixels. A clean sky stamp is continuous
        noise, so it has essentially no exact zeros.
      * **Noise-dominated** — the brightest pixel is barely above the
        local noise. S/N is measured per stamp with a robust MAD σ (the
        star is a tiny pixel fraction, so it doesn't inflate the MAD).
      * **Crowded / off-centre** — ``extract_stars`` centres the cutout
        on the detected star, so if the brightest pixel is NOT at the
        centre a brighter *neighbour* is in frame. EPSFBuilder would then
        recentre onto that neighbour and stack misaligned → a noisy,
        asymmetric ePSF core (the regression that prompted this). The big
        1023² stamps make this common, so require the global peak to sit
        within ``Config.HST.MAX_PEAK_OFFCENTER_PX`` of the stamp centre.
    """
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0 or not np.isfinite(arr).all():
        return False
    if float(np.mean(arr == 0.0)) > Config.HST.MAX_UNCOVERED_FRAC:
        return False
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med))) * 1.4826   # robust σ
    if mad <= 0:
        return False
    if (float(np.max(arr)) - med) < Config.HST.MIN_STAMP_PEAK_SNR * mad:
        return False
    py, px = np.unravel_index(int(np.argmax(arr)), arr.shape)
    cy = (arr.shape[0] - 1) / 2.0
    cx = (arr.shape[1] - 1) / 2.0
    return not (abs(py - cy) > Config.HST.MAX_PEAK_OFFCENTER_PX
                or abs(px - cx) > Config.HST.MAX_PEAK_OFFCENTER_PX)


def _extract_stamps_from_tile(
    data: np.ndarray, sources: Table, *, half_side: int = Config.HST.PSF_HALF_SIDE_PIX,
) -> list:
    """Pull ``2·half_side+1``-pixel stamps for each source in ``sources``.

    Thin wrapper around :func:`photutils.psf.extract_stars` so the
    column-rename invariant from :func:`_find_stars_in_tile` is enforced
    at the public boundary (and is testable in isolation without needing
    a real HST tile + DAOStarFinder).
    """
    if "x" not in sources.colnames or "y" not in sources.colnames:
        raise ValueError(
            "extract_stars requires 'x' and 'y' columns on the sources "
            "table — _find_stars_in_tile renames from xcentroid/ycentroid; "
            "did a caller supply a different table?"
        )
    nd = NDData(data=data)
    stamps = list(extract_stars(nd, sources, size=2 * half_side + 1))
    # Reject stamps straddling a tile seam / coverage hole or dominated by
    # noise — these otherwise poison the ePSF (the symptom that prompted
    # this check: "border" and "pure noise" cutouts at large half-side).
    return [st for st in stamps if _is_clean_star_stamp(np.asarray(st.data))]


def main() -> int:
    args = parse_args()
    # Extract from a margin-enlarged stamp, then trim the border (where
    # EPSFBuilder's smoothing leaves artifacts) back to the requested
    # --half-side. Config.HST.EPSF_OVERSAMPLING px of trim per native margin pixel.
    margin_px    = (max(1, int(round(args.half_side * args.extract_margin_frac)))
                    if args.extract_margin_frac > 0 else 0)
    half_extract = args.half_side + margin_px
    in_dir  = args.input_dir  or Config.HLSP_DIR
    out_dir = args.output_dir or Config.HST_PSF_DIR
    os.makedirs(out_dir, exist_ok=True)
    # Structured progress for the web UI. ``from_env()`` is a no-op
    # when ``EUCLID_POLISH_EVENTS_PATH`` isn't set (local dev), so the
    # ``reporter.*`` calls below are safe in every context.
    reporter = Reporter.from_env()

    print("=" * 64)
    print("  HST F814W ePSF extraction")
    print("=" * 64)
    print(f"  HLSP tile dir   = {in_dir}")
    print(f"  output dir      = {out_dir}")
    print(f"  target n_stars  = {args.n_stars}")
    print(f"  half_side       = {args.half_side} "
          f"(final ePSF → {2 * args.half_side + 1}²)")
    print(f"  extract margin  = {args.extract_margin_frac:.0%} "
          f"(+{margin_px}px → {2 * half_extract + 1}² stamps, border trimmed)")
    print()

    t0 = time.time()

    stars_dir = os.path.join(Config.DATA_DIR, Config.HST.STARS_DIR_NAME)
    star_stamps: list = []
    tiles_used: list = []
    pix_scale_observed: float = Config.HST.FALLBACK_PIX_SCALE_ARCSEC
    used_cache = False

    # ---- fast path: feed EPSFBuilder from a previous run's stamps ----
    # The expensive part of this script is the per-tile DAOStarFinder +
    # extract_stars pass (HLSP tiles are large, I/O-bound). If a prior
    # run already wrote per-star FITS into $DATA_DIR/hst_stars/ we can
    # skip straight to the ePSF build. ``--no-reuse-stars`` forces the
    # full tile scan (use it when you want to refresh the cached stamp
    # set with new HLSP tiles or new selection cuts).
    if args.reuse_stars:
        reporter.set_stage("Looking for cached star stamps")
        cached, cached_scale, cached_n_tiles = _load_cached_star_stamps(
            stars_dir, half_side=half_extract,
        )
        if cached:
            n_use = min(len(cached), args.n_stars)
            star_stamps        = cached[:n_use]
            pix_scale_observed = cached_scale
            # The output ePSF FITS records the contributing tile count
            # (NTILES). We don't have the original tile *names* in the
            # cache — only the count, preserved in the per-star FITS
            # NTILESRC header — so we materialise a placeholder list of
            # the right length so ``len(tiles_used)`` keeps working
            # downstream.
            tiles_used  = [None] * cached_n_tiles
            used_cache  = True
            print(f"[1/2] reusing {len(cached)} cached star stamps from "
                  f"{stars_dir}/  (target n_stars={args.n_stars} → "
                  f"using {n_use})")
            print(f"      pix scale (from cache header)  = "
                  f"{pix_scale_observed:.4f}\"/pix")
            print(f"      tile count (from cache header) = "
                  f"{cached_n_tiles}")
            if args.dry_run:
                print(f"\nDRY RUN — would skip tile scan and rebuild ePSF "
                      f"from {n_use} cached stamps")
                runtime = time.time() - t0
                print(f"\nRUNTIME_SECONDS={runtime:.1f}")
                return 0

    # ---- slow path: re-scan HLSP tiles ----
    if not used_cache:
        tiles = sorted(
            f for f in os.listdir(in_dir) if f.endswith(".fits")
            and f.startswith("hlsp_cosmos_hst_acs-wfc_mosaic")
        ) if os.path.isdir(in_dir) else []
        print(f"[1/3] {len(tiles)} HLSP tiles found")
        if not tiles:
            reporter.error(
                f"no HLSP tiles in {in_dir} — run the download step first.",
            )
            return 1
        reporter.set_stage(f"Scanning {len(tiles)} HLSP tiles for stars")

        if args.dry_run:
            print(f"\nDRY RUN — would scan {len(tiles)} tiles for "
                  f"~{args.n_stars} stars")
            runtime = time.time() - t0
            print(f"\nRUNTIME_SECONDS={runtime:.1f}")
            return 0

        # ---- collect stars across tiles until we hit the target count ----
        print("[2/3] scanning tiles for bright unsaturated point sources ...")

        # Cap per tile (not an even split across tiles): the loop already
        # stops once it has args.n_stars, so a couple of rich tiles can
        # satisfy the target without scanning the rest. ~100 keeps some
        # cross-tile diversity while slashing the number of 500 MB reads.
        stars_per_tile = min(args.n_stars, Config.HST.MAX_STARS_PER_TILE)

        for tile_idx, tname in enumerate(tiles):
            if len(star_stamps) >= args.n_stars:
                break
            tpath = os.path.join(in_dir, tname)
            print(f"      tile {tile_idx + 1}/{len(tiles)}: {tname}")
            reporter.set_step(tile_idx + 1, len(tiles), label=tname)
            with fits.open(tpath, memmap=True) as hdul:
                sci = next(
                    (e for e in hdul if e.is_image and e.data is not None), None,
                )
                if sci is None:
                    continue
                data = np.asarray(sci.data, dtype=np.float32)
                # Trust the first tile's WCS for pixel scale. All COSMOS HLSP
                # tiles in one product release share the same drizzle scale.
                if tile_idx == 0:
                    pix_scale_observed = _pixel_scale_from_header(sci.header)
                    print(f"      WCS pixel scale (this run) = "
                          f"{pix_scale_observed:.4f}\"/pix")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sources = _find_stars_in_tile(
                    data, max_n=stars_per_tile, half_side=half_extract)
            if sources is None or len(sources) == 0:
                print("        (no stars passed quality cuts)")
                continue
            print(f"        + {len(sources)} stars")
            try:
                stamps = _extract_stamps_from_tile(
                    data, sources, half_side=half_extract)
            except Exception as e:
                msg = (f"extract_stars failed on tile {tname}: "
                       f"{type(e).__name__}: {e}")
                print(f"        warn: {msg}")
                reporter.warn(msg)
                continue
            star_stamps.extend(stamps)
            tiles_used.append(tname)

        star_stamps = star_stamps[:args.n_stars]
        if not star_stamps:
            reporter.error("0 usable stars across all tiles")
            print("ERROR: 0 usable stars across all tiles")
            return 1
        print(f"      collected {len(star_stamps)} stars from "
              f"{len(tiles_used)} tiles")

        # ---- save each used star as a FITS so the UI can browse them ----
        os.makedirs(stars_dir, exist_ok=True)
        saved = 0
        for i, st in enumerate(star_stamps):
            try:
                stamp_arr = np.asarray(st.data, dtype=np.float32)
            except Exception:
                continue
            side = stamp_arr.shape[0]
            out_path = os.path.join(
                stars_dir, f"star_{i:04d}_{side}.fits",
            )
            hdu = fits.PrimaryHDU(stamp_arr)
            h = hdu.header
            h["OBJECT"]   = ("HST F814W star stamp",
                             "extracted for ePSF construction")
            h["FILTER"]   = ("F814W", "HST filter")
            h["INSTRUME"] = ("ACS/WFC", "HST instrument")
            h["PIXSCALE"] = (pix_scale_observed, "native HLSP pixel scale (arcsec)")
            h["STARIDX"]  = (i, "0-based index in this run")
            h["NTILESRC"] = (len(tiles_used), "tiles contributing to this run")
            h["BUNIT"]    = ("electrons / s", "ACS/WFC drizzled units")
            try:
                hdu.writeto(out_path, overwrite=True)
                saved += 1
            except Exception as e:
                print(f"        warn: failed to save {out_path}: {e}")
        print(f"      saved {saved} star stamps → {stars_dir}/")

    # ---- build the ePSF ----
    n_used = len(star_stamps)
    step_label = "[2/2]" if used_cache else "[3/3]"
    reporter.set_stage(f"Building ePSF from {n_used} stars")
    print(f"{step_label} running EPSFBuilder (oversampling = {Config.HST.EPSF_OVERSAMPLING}) ...")
    # ``fits`` is needed below for writing the ePSF; the slow path imports
    # it inside its branch, so reimport here for the cache path. (Top-level
    # imports stay light to keep the dry-run snappy.)
    builder = EPSFBuilder(
        oversampling=Config.HST.EPSF_OVERSAMPLING,
        maxiters=10,
        smoothing_kernel="quartic",
        progress_bar=False,
    )
    # ``_extract_stamps_from_tile`` returns a plain list so we can ``extend``
    # / slice it across multiple tiles. ``EPSFBuilder`` needs the proper
    # ``EPSFStars`` container though (it reads ``stars.n_stars``); wrap
    # right before the call so we get the best of both.
    epsf, _fitted_stars = builder(EPSFStars(star_stamps))
    psf_arr = np.asarray(epsf.data, dtype=np.float32)

    # Size the ePSF to EXACTLY the requested (2·half_side+1). EPSFBuilder
    # does NOT preserve the input stamp side (e.g. 1105² stamps → 1025²
    # ePSF here), so we cannot trim by ``margin_px`` — we centre-crop (or
    # zero-pad, if it came back smaller) straight to the target. Symmetric
    # on an odd array keeps the PSF centred. The margin still helped: it
    # gave EPSFBuilder room so its output ≥ target.
    target = 2 * args.half_side + 1
    M = psf_arr.shape[0]
    if target < M:
        off = (M - target) // 2
        psf_arr = psf_arr[off:off + target, off:off + target]
    elif target > M:
        lo = (target - M) // 2
        psf_arr = np.pad(psf_arr, ((lo, target - M - lo), (lo, target - M - lo)))
    if psf_arr.shape[0] != M:
        print(f"      ePSF sized to requested {target}² "
              f"(EPSFBuilder produced {M}²)")
    psf_arr = psf_arr / float(psf_arr.sum())   # unit flux (after resize)

    psf_pix_scale = pix_scale_observed / Config.HST.EPSF_OVERSAMPLING
    out_path = os.path.join(out_dir, Config.HST.PSF_FILE_NAME)
    hdu = fits.PrimaryHDU(psf_arr)
    h = hdu.header
    h["OBJECT"]   = ("HST F814W ePSF", "empirical PSF from COSMOS HLSP")
    h["FILTER"]   = ("F814W", "HST filter")
    h["INSTRUME"] = ("ACS/WFC", "HST instrument")
    h["NSTARS"]   = (n_used, "stars used in EPSFBuilder")
    h["NTILES"]   = (len(tiles_used), "HLSP tiles contributing stars")
    h["HALFSIDE"] = (args.half_side, "requested final ePSF half-side (px)")
    h["EXTRMARG"] = (margin_px, "extra border px extracted then trimmed")
    h["OVERSAMP"] = (Config.HST.EPSF_OVERSAMPLING, "oversampling factor relative to HLSP grid")
    h["PIXSCALE"] = (psf_pix_scale, "arcsec / pixel")
    h["TILESCAL"] = (pix_scale_observed, "source tile pixel scale (arcsec)")
    h["BUNIT"]    = ("", "unit flux (sums to 1)")
    hdu.writeto(out_path, overwrite=True)
    print(f"  wrote ePSF → {out_path}")
    print(f"    shape    = {psf_arr.shape}")
    print(f"    pix scale = {psf_pix_scale:.4f}\"/pix  "
          f"(tile {pix_scale_observed:.4f}\"/pix, oversample ×{Config.HST.EPSF_OVERSAMPLING})")
    print(f"    flux sum  = {psf_arr.sum():.6f}  (should be 1.0)")

    runtime = time.time() - t0
    print(f"\nRUNTIME_SECONDS={runtime:.1f}")
    print(f"N_STARS_USED={n_used}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
