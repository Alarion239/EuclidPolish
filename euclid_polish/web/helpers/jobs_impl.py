"""jobs_impl helpers for the EuclidPolish web UI (extracted from app.py)."""
from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any, cast

import numpy as np
from astropy.io import fits

from euclid_polish.catalog.downloader import fetch_cutout_at
from euclid_polish.config import Config
from euclid_polish.eval.disagreement import write_disagreement_cubes
from euclid_polish.eval.ensemble_infer import sr_from_model
from euclid_polish.eval.sr_provenance import write_sr_provenance
from euclid_polish.photometry import adu_per_s_to_electrons_factor
from euclid_polish.training.inference import (
    plot_reconstruction,
    scaled_wcs_header,
)


def reconstruct_cutout_at(
    model,
    ra: float,
    dec: float,
    cutout_size_vis_pixels: int,
    out_dir: str,
    *,
    asinh_scale: float | None = None,
    show_all_bands: bool = False,
    checkpoint_dir: str = "",
    render: bool = True,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    """Fetch a 4-band real Euclid cutout at ``(ra, dec)``, run SR, write outputs.

    This is the per-object body of the batch catalog evaluator
    (``eval/catalog_runner.py`` and ``scripts/fasrc_eval_catalog.py``). It fetches each band, converts the
    archive's ADU s⁻¹ to electrons-over-the-stack via the per-band ``MAGZERO``
    (so the model sees the same scale it trained on), stacks to ``(H, W, 4)``,
    runs ``reconstruct``, forward-models the SR for a self-consistency
    residual, and writes ``original_stack.fits`` + ``SR.fits`` (and, when
    ``render``, ``eye.png`` + ``solar.png``) into ``out_dir``.

    ``out_dir`` is created if absent and used as-is — callers that want a
    single overwrite slot must wipe it themselves. ``progress`` is an optional
    ``(done, total, label)`` callback (e.g. wrapping a job's ``cap.tick``).

    Returns a dict with the output paths, per-band info, and the
    forward-model residual metrics.
    """
    scale = Config.DEFAULT_REBIN_FACTOR
    band_names = Config.LR_INPUT_BAND_NAMES
    total = len(band_names) + 3
    os.makedirs(out_dir, exist_ok=True)

    def _tick(done: int, label: str) -> None:
        if progress is not None:
            progress(done, total, label)

    # Fetch each band; per-band MAGZERO from each header drives the
    # per-band ADU/s → electrons conversion so the model sees the same
    # calibration scale the simulator uses.
    bands_data: dict[str, np.ndarray] = {}
    bands_info: dict[str, dict[str, Any]] = {}
    vis_header = None
    for k, band_name in enumerate(band_names):
        _tick(k, f"loading {band_name} cutout")
        band = Config.get_band(band_name)
        outf = os.path.join(out_dir, f"{band_name}.fits")
        if os.path.isfile(outf) and os.path.getsize(outf) > 0:
            print(f"  {band_name}: reusing cached cutout → {outf}")
        else:
            _tick(k, f"downloading {band_name} cutout")
            ok, err = fetch_cutout_at(
                ra=ra, dec=dec, band_name=band_name, output_file=outf,
                cutout_size_vis_pixels=cutout_size_vis_pixels,
            )
            if not ok:
                raise RuntimeError(f"{band_name}: {err}")
        with fits.open(outf) as hdul:
            primary = cast(fits.PrimaryHDU, hdul[0])
            arr = np.asarray(primary.data, dtype=np.float32)
            header = primary.header
        if band_name == "VIS":
            vis_header = header.copy()
        magzero = float(cast(str | float, header.get(
            "MAGZERO", band.sim_zeropoint_e,
        )))
        # Single source of truth for archive ADU/s → electrons-over-stack.
        adu_to_e = adu_per_s_to_electrons_factor(magzero, band)
        data_e = (arr * adu_to_e).astype(np.float32)
        bands_data[band_name] = data_e
        bands_info[band_name] = {
            "shape":      data_e.shape,
            "magzero":    magzero,
            "adu_to_e":   adu_to_e,
            "pix_mean":   float(np.mean(data_e)),
            "pix_std":    float(np.std(data_e)),
            "fits_path":  outf,
        }
        print(f"  {band_name}: shape={data_e.shape}, MAGZERO={magzero:.3f}, "
              f"ADU/s→e⁻ factor={adu_to_e:.1f}")

    # All four cutouts must land on the same VIS-LR grid (the MER mosaic
    # pipeline delivers every band at 0.10″/pix). Anything else is a bug
    # in the archive query we should not silently paper over.
    shapes = {n: bands_data[n].shape for n in band_names}
    base_shape = shapes["VIS"]
    if any(s != base_shape for s in shapes.values()):
        raise RuntimeError(
            f"per-band shapes disagree: {shapes}; expected all bands at "
            "the same VIS LR grid (0.10″/pix)."
        )

    lr_cube = np.stack([bands_data[n] for n in band_names], axis=-1)  # (H,W,4)
    _tick(len(band_names), "running model")
    _, sr_data, members = sr_from_model(model, lr_cube)
    lr_vis = lr_cube[..., 0]

    # ESA cutout headers carry an EXTNAME that's invalid on a PrimaryHDU;
    # strip it (and let silentfix handle the rest) so the writes don't
    # trip FITS verification.
    def _clean_hdr(hdr):
        if hdr is None:
            return fits.Header()
        h = hdr.copy()
        for kbad in ("EXTNAME", "XTENSION"):
            if kbad in h:
                del h[kbad]
        return h

    # Stacked 4-band original (electrons): one image plane per band in
    # LR_INPUT_BAND_NAMES order (band 0 = VIS), carrying the VIS WCS so it
    # overlays the SR on-sky.
    stack = np.moveaxis(lr_cube, -1, 0).astype(np.float32)   # (4, H, W)
    stack_hdr = _clean_hdr(vis_header)
    stack_hdr["OBJECT"] = "Euclid LR stack (electrons)"
    stack_hdr["BUNIT"]  = "electron"
    stack_hdr["BANDS"]  = (",".join(band_names), "NAXIS3 plane order")
    stack_path = os.path.join(out_dir, "original_stack.fits")
    fits.PrimaryHDU(stack, header=stack_hdr).writeto(
        stack_path, overwrite=True, output_verify="silentfix")
    print(f"  ✓ saved stacked original → {stack_path}")

    # NOTE: there is deliberately NO forward-model self-consistency residual
    # for real cutouts. That comparison would push SR back through *our*
    # committed VIS PSF, but the true Euclid PSF is position-dependent and
    # unknown at an arbitrary (RA, Dec) — so a "predicted LR" and its residual
    # measure the PSF mismatch, not the reconstruction, and are misleading.
    # The pixel-level check that survives an unknown PSF is flux conservation
    # (total counts are invariant under a *normalised* PSF), computed below.
    _tick(len(band_names) + 1, "rendering")

    # Save SR with the 2× magnified VIS WCS so it overlays the stacked
    # original on-sky (0.05″/pix vs 0.10″/pix). A 4-band SR cube is
    # written one plane per band (NAXIS3), same convention as the
    # original_stack file.
    sr_fits_path = os.path.join(out_dir, "SR.fits")
    sr_hdr = (_clean_hdr(scaled_wcs_header(vis_header, scale))
              if vis_header is not None else fits.Header())
    sr_arr = np.asarray(sr_data, dtype=np.float32)
    sr_is_cube = sr_arr.ndim == 3
    if sr_is_cube:
        sr_arr = np.ascontiguousarray(np.moveaxis(sr_arr, -1, 0))
    sr_hdu = fits.PrimaryHDU(sr_arr, header=sr_hdr)
    sr_hdu.header["OBJECT"]   = ("Euclid SR (WDSR, 4-band)" if sr_is_cube
                                 else "Euclid SR (WDSR VIS)")
    if sr_is_cube:
        sr_hdu.header["BANDS"] = (",".join(band_names),
                                  "NAXIS3 plane order (band 0 = VIS)")
    sr_hdu.header["BUNIT"]    = "electron"
    sr_hdu.header["RA"]       = (float(ra),  "Input RA (deg)")
    sr_hdu.header["DEC"]      = (float(dec), "Input Dec (deg)")
    sr_hdu.header["CSIZE"]    = (int(cutout_size_vis_pixels),
                                 "Input VIS cutout size (px)")
    sr_hdu.header["CKPT"]     = (str(checkpoint_dir)[:60], "Checkpoint dir")
    sr_hdu.header["ASINH"]    = (float(asinh_scale or Config.STRETCH_SCALE_E),
                                 "asinh stretch knee used for plot")
    # Provenance: stamp the SR with its model lineage (best-effort, before write).
    try:
        write_sr_provenance(
            sr_hdu.header, checkpoint_dir=str(checkpoint_dir),
            sr_fits_path=sr_fits_path,
            descriptors={"ra": float(ra), "dec": float(dec),
                         "cutout_size": int(cutout_size_vis_pixels),
                         "bands": ",".join(band_names)},
        )
    except Exception as exc:
        print(f"  [provenance] SR.fits not stamped: {exc}")
    sr_hdu.writeto(sr_fits_path, overwrite=True, output_verify="silentfix")
    print(f"  ✓ saved SR  → {sr_fits_path}")
    # Ensemble disagreement cubes (full-field; enforce_object_sizes center-crops
    # them alongside SR so they stay pixel-aligned). No-op for a single model.
    if members is not None:
        try:
            write_disagreement_cubes(
                out_dir, members,
                member_labels=list(getattr(model, "member_labels", []) or []))
            print("  ✓ saved disagreement cubes (std + pca)")
        except Exception as exc:  # noqa: BLE001 — never kill a run over the movie
            print(f"  [disagreement] cubes not written: {exc}")

    # TWO colored LR → SR renders — the same figure once per color regime:
    # "eye" (physical blackbody-T colors, absolute) and "solar"
    # (solar-balanced adaptive windows).
    png_paths: list[str] = []
    if render:
        for regime, mode in (("eye", "eye"), ("solar", "calibrated")):
            out_path = os.path.join(out_dir, f"{regime}.png")
            plot_reconstruction(lr_vis, sr_data, hr_data=None,
                                output_path=out_path, lr_cube=lr_cube,
                                asinh_scale=asinh_scale,
                                show_all_bands=show_all_bands,
                                rgb_mode=mode)
            png_paths.append(out_path)
            print(f"  ✓ {out_path}")
    _tick(total, "saved outputs")

    # Flux conservation — the one pixel-level sanity check that doesn't depend
    # on the (unknown, position-dependent) true PSF: a normalised PSF + sum-
    # rebin conserves total counts, so Σ(forward(SR)) ≡ Σ(SR). We therefore
    # compare Σ(SR_VIS) to Σ(LR_VIS) directly — ratio ≈ 1 means the
    # deconvolution neither invented nor destroyed flux.
    _sr = np.asarray(sr_data)
    sr_vis = _sr[..., 0] if _sr.ndim == 3 else _sr
    lr_sum = float(np.sum(lr_vis))
    sr_sum = float(np.sum(sr_vis))
    metrics = {
        "lr_total_e":            lr_sum,
        "sr_total_e":            sr_sum,
        "flux_ratio_sr_over_lr": (sr_sum / lr_sum) if lr_sum != 0 else None,
    }

    return {
        "out_dir":      out_dir,
        "png_paths":    png_paths,
        "sr_fits_path": sr_fits_path,
        "stack_fits_path": stack_path,
        "ra":           float(ra),
        "dec":          float(dec),
        "cutout_size":  int(cutout_size_vis_pixels),
        "bands":        bands_info,
        "metrics":      metrics,
    }
