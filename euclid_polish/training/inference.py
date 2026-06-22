"""
Inference / reconstruction module for EuclidPolish.

Provides utilities for applying a trained WDSR model to low-resolution
images and visualizing the results.
"""

import os
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from astropy.wcs import WCS

from euclid_polish.config import Config
from euclid_polish.training.models.common import resolve_single
from euclid_polish.training.models.wdsr import wdsr
from euclid_polish.visualization.base import BaseVisualizer

from euclid_polish.visualization.color import lupton_rgb


def scaled_wcs_header(vis_header, scale: int):
    """Return a copy of ``vis_header`` with its WCS magnified by ``scale``.

    The SR image has ``scale``× more pixels per side at ``1/scale`` the
    pixel scale, so the WCS needs the standard magnify transform:
    ``CRPIX → scale·CRPIX − (scale−1)/2`` and the pixel-scale terms
    (CD matrix or CDELT) divided by ``scale``. This keeps the SR aligned
    on-sky with the original LR cutout. Falls back to the unscaled header
    when the WCS can't be parsed (the rescale is a bonus, never fatal).
    """
    hdr = vis_header.copy()
    try:
        w = WCS(vis_header)
        if not w.has_celestial:
            return hdr
        w = w.deepcopy()
        off = (scale - 1) / 2.0
        w.wcs.crpix = [c * scale - off for c in w.wcs.crpix]
        if w.wcs.has_cd():
            w.wcs.cd = w.wcs.cd / scale
        else:
            w.wcs.cdelt = [d / scale for d in w.wcs.cdelt]
        scaled = w.to_header()
        # Drop the old WCS keys before merging the scaled ones so stale
        # CD/CDELT/CRPIX don't linger.
        for key in list(hdr.keys()):
            if key.startswith(("CRPIX", "CRVAL", "CDELT", "CD1_", "CD2_",
                               "PC1_", "PC2_", "CTYPE", "CUNIT")):
                del hdr[key]
        hdr.update(scaled)
    except Exception as e:  # noqa: BLE001 — WCS is a bonus, never fatal
        hdr["WCSNOTE"] = f"SR WCS rescale skipped: {type(e).__name__}"
    return hdr


def infer_checkpoint_nchan_in(
    checkpoint_dir: str, skip_kernel_size: int = 5,
) -> Optional[int]:
    """Number of LR input channels a saved checkpoint's model expects, or None.

    The checkpoint *is* the source of truth for the architecture — a VIS-only
    run saves a 1-channel WDSR (``ckpt/wdsr-vis``), a normal run a 4-channel
    one (``ckpt/wdsr``). We read it straight from the stored weight shapes:
    excluding the ``skip_kernel_size``-sided skip-branch kernels (the 4-band
    model's PER-BAND skip convs have in-dim 1 regardless of ``nchan_in``, so
    they would poison a min over everything), the input conv is the only
    kernel whose in-channel dim is the LR channel count (1 or 4); every body
    kernel is ``num_filters`` (32) wide or wider, so the minimum in-channel
    dim over the remaining 4-D conv kernels is exactly ``nchan_in``.
    Returns None if no checkpoint or the shapes can't be read (caller falls
    back to its explicit value)."""
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    if latest is None:
        return None
    try:
        reader = tf.train.load_checkpoint(latest)
        shapes = reader.get_variable_to_shape_map()
    except Exception:    # pragma: no cover — unreadable ckpt → caller default
        return None
    in_dims = [shp[2] for shp in shapes.values()
               if len(shp) == 4
               and not (shp[0] == skip_kernel_size
                        and shp[1] == skip_kernel_size)]
    return int(min(in_dims)) if in_dims else None


def infer_checkpoint_nchan_out(
    checkpoint_dir: str, scale: int, nchan_in: int,
    skip_kernel_size: int = 5,
) -> Optional[int]:
    """Number of SR output channels a saved checkpoint's model emits, or None.

    Discriminates via the skip-branch conv kernels — the only
    ``skip_kernel_size``-sided (default 5×5) kernels in the model:

      * dense (legacy) skip — ONE kernel ``(5, 5, nchan_in, nchan_out·scale²)``
        → ``nchan_out = out_dim / scale²``. Covers the historical
        4-in/1-out checkpoints and every VIS-only (1→1) model.
      * per-band skip (the 4-band model) — per-band kernels
        ``(5, 5, 1, scale²)`` with ``nchan_in > 1`` → ``nchan_out = nchan_in``.

    Returns None if no checkpoint / unreadable / no 5×5 kernel found
    (caller falls back to its default).
    """
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    if latest is None:
        return None
    try:
        reader = tf.train.load_checkpoint(latest)
        shapes = reader.get_variable_to_shape_map()
    except Exception:    # pragma: no cover — unreadable ckpt → caller default
        return None
    skip_kernels = {
        tuple(shp) for shp in shapes.values()
        if len(shp) == 4 and shp[0] == skip_kernel_size
        and shp[1] == skip_kernel_size
    }
    for shp in skip_kernels:
        if shp[2] == nchan_in:                 # dense skip (legacy / 1-ch)
            return int(shp[3] // scale ** 2)
    if any(shp[2] == 1 for shp in skip_kernels):
        return int(nchan_in)                   # per-band skip: out == in
    return None


def load_model_from_checkpoint(
    checkpoint_dir: str,
    scale: int,
    num_res_blocks: int = Config.DEFAULT_NUM_RES_BLOCKS,
    nchan: int = None,           # back-compat alias for nchan_in
    nchan_in: int = None,
    nchan_out: int = None,
):
    """Build a WDSR model and restore weights from a TF checkpoint directory.

    ``nchan_in`` is normally inferred from the checkpoint itself (see
    :func:`infer_checkpoint_nchan_in`), so a VIS-only (1-channel) and a
    standard (4-channel) checkpoint both load correctly with no caller
    changes. An explicitly passed ``nchan_in`` is honoured only when the
    checkpoint can't be introspected; if it disagrees with the checkpoint, the
    checkpoint wins (building a mismatched model would silently leave the input
    layer at random init under ``expect_partial``).
    """
    if nchan is not None and nchan_in is None:
        nchan_in = nchan

    ckpt_nchan_in = infer_checkpoint_nchan_in(checkpoint_dir)
    if ckpt_nchan_in is not None:
        if nchan_in is not None and nchan_in != ckpt_nchan_in:
            print(f"  note: checkpoint has nchan_in={ckpt_nchan_in} "
                  f"(requested {nchan_in}); using the checkpoint's.")
        nchan_in = ckpt_nchan_in
    if nchan_in is None:
        nchan_in = 1

    # Output channels likewise come from the checkpoint: a 4-band
    # (VIS+NISP → VIS+NISP, per-band skip) model and a legacy 4-in/1-out
    # or VIS-only model all load with no caller changes. The skip-branch
    # structure differs between the two (per-band vs dense), so building
    # the wrong one would mis-restore — the checkpoint wins over any
    # explicitly passed value.
    ckpt_nchan_out = infer_checkpoint_nchan_out(
        checkpoint_dir, scale=scale, nchan_in=nchan_in)
    if ckpt_nchan_out is not None:
        if nchan_out is not None and nchan_out != ckpt_nchan_out:
            print(f"  note: checkpoint has nchan_out={ckpt_nchan_out} "
                  f"(requested {nchan_out}); using the checkpoint's.")
        nchan_out = ckpt_nchan_out
    if nchan_out is None:
        # New-style default: band k in → band k out.
        nchan_out = nchan_in

    model = wdsr(
        scale=scale, num_res_blocks=num_res_blocks,
        nchan_in=nchan_in, nchan_out=nchan_out,
    )
    checkpoint = tf.train.Checkpoint(model=model)
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    if latest is None:
        raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")
    checkpoint.restore(latest).expect_partial()
    print(f"Model restored from checkpoint at {latest} "
          f"(nchan_in={nchan_in}, nchan_out={nchan_out}).")
    return model


def load_model_from_weights(
    weights_path: str,
    scale: int,
    num_res_blocks: int = Config.DEFAULT_NUM_RES_BLOCKS,
    nchan: int = 1,
):
    """
    Build a WDSR model and load saved .h5 weights.

    Parameters
    ----------
    weights_path : str
        Path to a .h5 weights file.
    scale : int
        Super-resolution scale factor.
    num_res_blocks : int
        Number of residual blocks.
    nchan : int
        Number of image channels.

    Returns
    -------
    tf.keras.Model
    """
    model = wdsr(scale=scale, num_res_blocks=num_res_blocks, nchan=nchan)
    model.load_weights(weights_path)
    return model


def reconstruct(
    model,
    lr_input,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply super-resolution to a single LR image.

    Inputs and outputs are raw float32 electrons (over the stacked Euclid VIS
    integration). Internally this function

      1. asinh-stretches the input by Config.STRETCH_SCALE_E,
      2. runs the model (which operates entirely in stretched space),
      3. clips the stretched output for safety, and
      4. applies sinh × scale to recover electrons.

    Parameters
    ----------
    model : tf.keras.Model
        Trained WDSR model.
    lr_input : str or np.ndarray
        Either a ``.npy`` file path or a numpy array in raw electron units.

    Returns
    -------
    lr_data : ndarray, shape (H, W)
        The input LR image (2-D VIS plane, raw electrons).
    sr_data : ndarray, shape (H', W') or (H', W', C)
        The super-resolved output in raw electrons. 2-D for a
        single-output (VIS-only / legacy) model; a (H', W', 4) cube for
        the 4-band VIS+NISP model — channel 0 is VIS, so legacy callers
        can take ``sr[..., 0]`` (or pass the cube on for color panels).
    """
    if isinstance(lr_input, str):
        if lr_input.endswith(".npy"):
            lr_data = np.load(lr_input).astype(np.float32)
        else:
            raise ValueError(
                f"Unsupported file format: {lr_input}. "
                "Inputs must be .npy in raw electron units."
            )
    else:
        lr_data = np.asarray(lr_input, dtype=np.float32)

    # Normalize input to a (H, W, C) tensor for the model. Three shapes
    # we accept:
    #   2D (H, W)           — single-channel image
    #   3D (H, W, 1)        — single-channel with an explicit channel axis
    #   3D (H, W, C)        — multi-band cube (e.g. C=4 for VIS+Y_E+J_E+H_E)
    if lr_data.ndim == 2:
        lr_for_model = lr_data[:, :, None]              # → (H, W, 1)
    elif lr_data.ndim == 3:
        lr_for_model = lr_data                          # already (H, W, C)
    else:
        raise ValueError(
            f"reconstruct(): expected 2D or 3D input, got shape {lr_data.shape}"
        )

    # Match the cube to what the model expects. A VIS-only (1-channel) model
    # gets just VIS (channel 0); a 4-channel model gets the full stack. The
    # LR band order is [VIS, Y_E, J_E, H_E] (VIS first), so ``[..., :n_in]``
    # selects the right leading channels for either. Callers can therefore
    # always hand us the full 4-band cube and let the model decide.
    try:
        n_in = int(model.inputs[0].shape[-1])
    except (AttributeError, IndexError, TypeError):
        n_in = lr_for_model.shape[-1]
    c = lr_for_model.shape[-1]
    if c > n_in:
        lr_for_model = lr_for_model[..., :n_in]
    elif c < n_in:
        raise ValueError(
            f"reconstruct(): model expects {n_in} input channels but the LR "
            f"input has only {c}"
        )

    scale_e = float(Config.STRETCH_SCALE_E)

    # Stretch → model → unstretch. ``clip(±20)`` is a defensive guard against
    # an untrained / pathological model output: ``sinh(20) ≈ 2.4×10⁸``, well
    # above any realistic source, but finite in float32 (sinh overflows at ~89).
    lr_stretched = np.arcsinh(lr_for_model / scale_e).astype(np.float32)
    sr_stretched = resolve_single(model, tf.constant(lr_stretched)).numpy().astype(np.float32)
    sr_stretched = np.clip(sr_stretched, -20.0, 20.0)
    sr_data = (np.sinh(sr_stretched.astype(np.float64)) * scale_e).astype(np.float32)
    # Single-output models: squeeze the channel axis for the 2-D
    # visualizers downstream. The 4-band model keeps its (H, W, 4) cube —
    # callers slice channel 0 for VIS or feed the cube to color panels.
    if sr_data.ndim == 3 and sr_data.shape[-1] == 1:
        sr_data = sr_data[..., 0]
    # LR returned for display: if the caller passed a multi-channel cube,
    # show only the VIS channel (band 0) — that's what the HR target is.
    if lr_data.ndim == 3 and lr_data.shape[-1] > 1:
        lr_display = lr_data[..., 0]
    elif lr_data.ndim == 3 and lr_data.shape[-1] == 1:
        lr_display = lr_data[..., 0]
    else:
        lr_display = lr_data

    return lr_display, sr_data


def _image_stats(data: np.ndarray) -> dict:
    """Per-image statistics. Values are raw numerics — the stats panel
    auto-formats them consistently (``.4e`` for floats)."""
    finite = data[np.isfinite(data)]
    p1, p10, p50, p90, p99, p999 = np.percentile(finite, [1, 10, 50, 90, 99, 99.9])
    return {
        "Shape":           f"{data.shape[0]} × {data.shape[1]}",
        "Dtype":           str(data.dtype),
        "min":             float(finite.min()),
        "p1":              float(p1),
        "p10":             float(p10),
        "median":          float(p50),
        "p90":             float(p90),
        "p99":             float(p99),
        "p99.9":           float(p999),
        "max":             float(finite.max()),
        "mean":            float(np.mean(finite)),
        "std":             float(np.std(finite)),
        "% < 0":           f"{(data < 0).mean() * 100:>9.2f} %",
        "Total flux":      float(np.sum(finite)),
        "# bright (>p99)": int((finite > p99).sum()),
    }


def _residual_asinh_stats(residual_stretched: np.ndarray, residual_e: np.ndarray) -> dict:
    """Stats describing the residual structure in asinh space."""
    finite = residual_stretched[np.isfinite(residual_stretched)]
    p1, p50, p99 = np.percentile(finite, [1, 50, 99])
    abs_err = np.abs(finite)
    return {
        "mean (bias)":           float(np.mean(finite)),
        "median":                float(p50),
        "std (noise level)":     float(np.std(finite)),
        "p1":                    float(p1),
        "p99":                   float(p99),
        "median |Δ|":            float(np.median(abs_err)),
        "mean |Δ| (MAE)":        float(np.mean(abs_err)),
        "max |Δ|":               float(abs_err.max()),
        "# pixels |Δ|>3σ":       int((abs_err > 3 * np.std(finite)).sum()),
        "RMSE":                  float(np.sqrt(np.mean(finite ** 2))),
        "Median e⁻ equiv":       float(np.sinh(p50) * Config.STRETCH_SCALE_E),
    }


def _noise_floor_lr_std_e() -> float:
    """Expected per-LR-pixel noise std (e⁻) from the analytical noise model.

    Used as the denominator floor in relative-error panels — keeps the plot
    finite where signal → 0.
    """
    t_total = Config.EXPOSURE_TIME_S * Config.N_EXPOSURES
    pixel_area = Config.VIS_PIXEL_SCALE_ARCSEC ** 2
    sky_e   = Config.SKY_E_PER_S_PER_ARCSEC2 * pixel_area * t_total
    dark_e  = Config.DARK_E_PER_S_PER_PIX * t_total
    read_var = Config.READ_NOISE_E ** 2 * Config.N_EXPOSURES
    return float(np.sqrt(sky_e + dark_e + read_var))


def _residual_metrics(residual_stretched: np.ndarray,
                      residual_e: np.ndarray,
                      hr_stretched: np.ndarray,
                      hr_e: np.ndarray) -> dict:
    """PSNR / RMSE / MAE numbers for the reconstruction in both spaces.

    Peaks come from ``Config.PSNR_PEAK_*`` (mag-17 star).
    """
    eps = 1e-7
    rmse_str = float(np.sqrt(np.mean(residual_stretched ** 2)) + eps)
    rmse_raw = float(np.sqrt(np.mean(residual_e ** 2)) + eps)

    peak_str = float(Config.PSNR_PEAK_STRETCHED)
    peak_raw = float(Config.PSNR_PEAK_E)
    psnr_str = 20.0 * np.log10(peak_str / rmse_str)
    psnr_raw = 20.0 * np.log10(peak_raw / rmse_raw)

    return {
        "PSNR str (peak=mag17)":   f"{psnr_str:>9.3f} dB",
        "PSNR raw (peak=mag17)":   f"{psnr_raw:>9.3f} dB",
        "MAE (asinh)":             float(np.mean(np.abs(residual_stretched))),
        "MAE (raw e⁻)":           float(np.mean(np.abs(residual_e))),
        "RMSE (asinh)":            float(rmse_str),
        "RMSE (raw e⁻)":          float(rmse_raw),
        "Worst |Δ| (asinh)":       float(np.max(np.abs(residual_stretched))),
        "Worst |Δ| (raw e⁻)":      float(np.max(np.abs(residual_e))),
        "Best |Δ| (asinh)":        float(np.min(np.abs(residual_stretched))),
    }


def _safe_lupton_rgb(cube: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Build a Lupton RGB from a 4-band cube, swallowing any rendering
    failure so the rest of the plot still renders. Returns None if the
    cube is missing or doesn't have all four bands."""
    if cube is None or cube.ndim != 3 or cube.shape[-1] != len(Config.LR_INPUT_BAND_NAMES):
        return None
    try:
        return lupton_rgb(
            cube, band_names=Config.LR_INPUT_BAND_NAMES,
            scheme="vis_nisp", reference="solar", Q=8.0, stretch=1.0,
        )
    except Exception as e:    # pragma: no cover — color is best-effort
        print(f"  color composite skipped: {type(e).__name__}: {e}")
        return None


def plot_reconstruction(
    lr_data: np.ndarray,
    sr_data: np.ndarray,
    hr_data: Optional[np.ndarray] = None,
    output_path: str = "reconstruction.png",
    vmax: float | None = None,
    lr_cube: Optional[np.ndarray] = None,
    hr_cube: Optional[np.ndarray] = None,
    asinh_scale: float | None = None,
    show_all_bands: bool = False,
    predicted_dirty: Optional[np.ndarray] = None,
    residual: Optional[np.ndarray] = None,
    rgb_mode: str = "eye",
    dirty_hi_pct: float | None = None,
) -> None:
    """
    Visualize LR input, SR output, and (optionally) HR ground truth.

    Layout when HR is provided — 3 rows × 5 cols (4 rows when colour
    composites are added):

        Row 0 (color, optional):    LR color | (blank) | HR color | ...
        Row 1 (raw / linear):       LR raw | SR raw | HR raw | residual raw | rel-err raw
        Row 2 (asinh):              LR asinh | SR asinh | HR asinh | residual asinh | rel-err asinh
        Row 3 (stats):              LR stats | SR stats | HR stats | asinh-residual | PSNR

    All asinh panels share ``Config.STRETCH_SCALE_E`` (the network's
    training scale), so brightness is directly comparable to the loss.

    When HR is missing, falls back to a 2 × 3 LR/SR layout (raw + asinh
    + optional LR color).

    Color composites: SR and HR render in the regime picked by
    ``rgb_mode``:

      * ``"eye"`` (default) — the PHYSICAL mode (per-pixel blackbody
        color temperature → CIE Planckian-locus hue, absolute luminance
        keyed to the asinh knee — see
        :func:`euclid_polish.visualization.color.eye_rgb`). The
        transform is image-independent, so an SR-vs-HR hue difference is
        a reconstruction error, never a rendering artifact; a
        blackbody-T legend strip under the asinh HR (or SR) panel
        translates hue back to SED temperature.
      * ``"calibrated"`` — the solar-balanced adaptive mode (per-image
        [p1, p99.5] windows), the depth-adaptive rendering the /sky
        record viewer's "solar" chip uses.

    Inference jobs render the SAME figure once per regime so both
    color readings sit side-by-side in the gallery. The residual /
    metric panels stay on the VIS channel (channel 0) regardless, so
    the numbers remain comparable with historical single-band runs.
    """
    if rgb_mode not in ("eye", "calibrated"):
        raise ValueError(
            f"rgb_mode must be 'eye' or 'calibrated'; got {rgb_mode!r}"
        )
    # Asinh-stretch knee used in every "asinh" panel of this plot. The
    # caller can override per-run from the UI (especially useful on
    # real Euclid cutouts where the source-to-sky brightness ratio is
    # different from the simulator's). ``None`` → fall back to the
    # training scale ``Config.STRETCH_SCALE_E`` so default behaviour is
    # unchanged for everyone who doesn't pass the parameter.
    shared_scale = float(asinh_scale) if asinh_scale and asinh_scale > 0 \
                   else float(Config.STRETCH_SCALE_E)

    # 4-band SR (the VIS+NISP model): keep the cube for color panels and
    # use the VIS plane (channel 0) for every residual / metric panel.
    _nbands = len(Config.LR_INPUT_BAND_NAMES)
    sr_cube = (np.asarray(sr_data)
               if (np.asarray(sr_data).ndim == 3
                   and np.asarray(sr_data).shape[-1] == _nbands)
               else None)
    sr_vis = sr_cube[..., 0] if sr_cube is not None else np.asarray(sr_data)

    # Regime tag in the figure title so the eye / solar renders of the
    # same scene are distinguishable in the gallery.
    regime_note = ("" if sr_cube is None else
                   (" — eye color" if rgb_mode == "eye" else " — solar color"))

    def _add_sr_panel(vis_obj, stretch, temp_legend=False):
        """SR panel — color in the requested ``rgb_mode`` when the 4-band
        cube exists, else grayscale. The blackbody-T legend only applies
        to the eye regime."""
        if sr_cube is not None:
            vis_obj.add_rgb_scale_panel(
                sr_cube, stretch=stretch, asinh_scale=shared_scale,
                title_suffix="\nReconstruction (SR)",
                rgb_mode=rgb_mode,
                temp_legend=temp_legend and rgb_mode == "eye")
        else:
            kw = {"asinh_scale": shared_scale} if stretch == "asinh" else {}
            vis_obj.add_scale_panel(sr_vis, stretch=stretch,
                                    title_suffix="\nReconstruction (SR)",
                                    cmap="gray", **kw)

    if hr_data is None:
        # Real-Euclid inference flow: no HR truth available. Two layout
        # modes:
        #
        #   (a) ``show_all_bands=True`` (and a 4-band LR cube is
        #       supplied) → 2 × 5 grid showing each LR band as its own
        #       grayscale panel + SR, both linear and asinh. Useful for
        #       diagnosing per-band issues (saturation, NISP CR
        #       artefacts, persistence) the colour composite would
        #       blend together.
        #
        #   (b) default → 2 × 2 LR (colour or VIS gray) + SR (gray).
        nbands = len(Config.LR_INPUT_BAND_NAMES)
        has_cube = (lr_cube is not None and lr_cube.ndim == 3
                    and lr_cube.shape[-1] == nbands)

        if show_all_bands and has_cube:
            # Third row for the per-band SR planes when the model emits
            # all four bands.
            nrows = 3 if sr_cube is not None else 2
            vis = BaseVisualizer(rows=nrows, cols=nbands + 1,
                                 figsize=(6 * (nbands + 1), 6 * nrows),
                                 vmax=vmax)
            # Row 1 — linear per-band LR + SR (color when 4-band).
            for k, name in enumerate(Config.LR_INPUT_BAND_NAMES):
                vis.add_scale_panel(
                    lr_cube[..., k], stretch="linear",
                    title_suffix=f"\nDirty (LR · {name})",
                    cmap="gray",
                )
            _add_sr_panel(vis, "linear")
            # Row 2 — asinh per-band LR + SR (color when 4-band).
            for k, name in enumerate(Config.LR_INPUT_BAND_NAMES):
                vis.add_scale_panel(
                    lr_cube[..., k], stretch="asinh",
                    asinh_scale=shared_scale,
                    title_suffix=f"\nDirty (LR · {name})",
                    cmap="gray",
                )
            _add_sr_panel(vis, "asinh", temp_legend=True)
            # Row 3 — per-band SR planes (asinh), one per output band.
            if sr_cube is not None:
                for k, name in enumerate(Config.LR_INPUT_BAND_NAMES):
                    vis.add_scale_panel(
                        sr_cube[..., k], stretch="asinh",
                        asinh_scale=shared_scale,
                        title_suffix=f"\nSR · {name}",
                        cmap="gray",
                    )
            plt.suptitle("Super-Resolution Reconstruction (per-band view)"
                         + regime_note, fontsize=16)
            vis.save_figure(output_path)
            return

        # Default 2 × 2 layout: Dirty LR | SR, in linear and asinh rows.
        # Dirty LR is always rendered in VIS-only grayscale (a 4-band colour
        # composite is dominated by the much-noisier NISP channels and
        # visually underplays the VIS plane the model targets); SR renders in
        # the chosen colour regime. The LR panel uses an inset colorbar so it
        # does NOT shrink relative to the colorbar-less SR panel — LR and SR
        # then sit the same on-screen size, which is what makes them pleasant
        # to compare side by side.
        #
        # There is intentionally no "predicted LR" / forward-model residual
        # column for real cutouts: those would forward-model SR through *our*
        # committed VIS PSF, but the true Euclid PSF is position-dependent and
        # unknown at an arbitrary (RA, Dec), so the residual measures the PSF
        # mismatch rather than the reconstruction. (predicted_dirty / residual
        # are still accepted for the synthetic / known-PSF callers.)
        has_predicted = predicted_dirty is not None
        has_residual  = residual is not None
        cols = 2 + int(has_predicted) + int(has_residual)
        vis = BaseVisualizer(rows=2, cols=cols, figsize=(9 * cols, 18),
                             vmax=vmax)

        # Row 1 — linear. ``dirty_hi_pct`` lets the caller raise the upper clip
        # so a bright central galaxy stops saturating and its internal (lens)
        # structure shows.
        vis.add_scale_panel(lr_data, stretch="linear",
                            title_suffix="\nDirty (LR, VIS)",
                            cmap="gray", colorbar_inset=True,
                            hi_percentile=dirty_hi_pct)
        _add_sr_panel(vis, "linear")
        if has_predicted:
            vis.add_scale_panel(
                predicted_dirty, stretch="linear",
                title_suffix="\nVIS PSF ⨂ SR + 2× rebin\n(predicted LR)",
                cmap="gray",
            )
        if has_residual:
            vis.add_diverging_panel(
                residual, stretch="linear",
                title_suffix="\nResidual = Dirty − Predicted",
                colorbar_label="LR e⁻",
            )

        # Row 2 — asinh (loss-aligned stretch).
        vis.add_scale_panel(lr_data, stretch="asinh",
                            asinh_scale=shared_scale,
                            title_suffix="\nDirty (LR, VIS)",
                            cmap="gray", colorbar_inset=True,
                            hi_percentile=dirty_hi_pct)
        _add_sr_panel(vis, "asinh", temp_legend=True)
        if has_predicted:
            vis.add_scale_panel(
                predicted_dirty, stretch="asinh", asinh_scale=shared_scale,
                title_suffix="\nVIS PSF ⨂ SR + 2× rebin\n(predicted LR)",
                cmap="gray",
            )
        if has_residual:
            vis.add_diverging_panel(
                residual, stretch="asinh", asinh_scale=shared_scale,
                title_suffix="\nResidual = Dirty − Predicted",
            )
        plt.suptitle("Super-Resolution Reconstruction" + regime_note,
                     fontsize=16)
        vis.save_figure(output_path)
        return

    # Pre-compute residuals & noise floors (used as denominator clamps in
    # the rel-err panels so they don't blow up at sky-floor pixels).
    # Residual / metric panels compare the VIS planes (channel 0) so the
    # numbers stay comparable with historical single-band runs.
    hr_stretched       = np.arcsinh(hr_data / shared_scale).astype(np.float32)
    sr_stretched       = np.arcsinh(sr_vis / shared_scale).astype(np.float32)
    residual_e         = (hr_data - sr_vis).astype(np.float32)
    residual_stretched = (hr_stretched - sr_stretched).astype(np.float32)
    floor_e            = _noise_floor_lr_std_e()
    floor_str          = float(np.arcsinh(floor_e / shared_scale))

    # 3 × 5 layout. SR / HR cells render in colour when their 4-band
    # cubes are available — directly comparable composites since the
    # 4-band model emits every band.
    vis = BaseVisualizer(rows=3, cols=5, figsize=(45, 24), vmax=vmax)

    nbands = len(Config.LR_INPUT_BAND_NAMES)
    # LR (dirty) is always rendered VIS-only grayscale. NISP channels
    # are much noisier than VIS and a 4-band colour composite makes
    # the dirty image look worse than what the model actually sees —
    # the user wants to inspect the VIS channel the network targets.
    # HR (clean truth) keeps the 4-band colour composite since it's
    # noise-free and the colour informs galaxy-SED interpretation.
    hr_color = (hr_cube is not None and hr_cube.ndim == 3
                and hr_cube.shape[-1] == nbands)

    # ---- Row 1: linear (raw electrons) ----
    vis.add_scale_panel(lr_data, stretch="linear",
                        title_suffix="\nDirty (LR, VIS)",
                        cmap="gray")
    _add_sr_panel(vis, "linear")
    if hr_color:
        vis.add_rgb_scale_panel(hr_cube, stretch="linear",
                                title_suffix="\nTrue Sky (HR)",
                                rgb_mode=rgb_mode)
    else:
        vis.add_scale_panel(hr_data, stretch="linear",
                            title_suffix="\nTrue Sky (HR)")
    vis.add_diverging_panel(residual_e, stretch="linear",
                            title_suffix="\nResidual = HR − SR (raw e⁻)",
                            colorbar_label="Residual (e⁻)")
    vis.add_relative_error_panel(residual_e, hr_data, floor=floor_e,
                                 title_suffix="\nraw e⁻")

    # ---- Row 2: asinh (loss-aligned) ----
    vis.add_scale_panel(lr_data, stretch="asinh", asinh_scale=shared_scale,
                        title_suffix="\nDirty (LR, VIS)",
                        cmap="gray")
    _add_sr_panel(vis, "asinh")
    if hr_color:
        # Temperature legend on the asinh HR panel (eye regime only) —
        # one hue ↔ T_eff dictionary serves every eye panel.
        vis.add_rgb_scale_panel(hr_cube, stretch="asinh",
                                asinh_scale=shared_scale,
                                title_suffix="\nTrue Sky (HR)",
                                rgb_mode=rgb_mode,
                                temp_legend=(rgb_mode == "eye"))
    else:
        vis.add_scale_panel(hr_data, stretch="asinh", asinh_scale=shared_scale,
                            title_suffix="\nTrue Sky (HR)")
    vis.add_diverging_panel(residual_e, stretch="asinh", asinh_scale=shared_scale,
                            title_suffix="\nResidual = HR − SR")
    vis.add_relative_error_panel(residual_stretched, hr_stretched, floor=floor_str,
                                 title_suffix="\nasinh space")

    # ---- Row 3: stats ----
    vis.add_statistics_panel(lr_data, {
        "title": "LR (dirty input):",
        "stats": _image_stats(lr_data),
        "include_data_stats": False,
    })
    vis.add_statistics_panel(sr_vis, {
        "title": "SR (model output, VIS):",
        "stats": _image_stats(sr_vis),
        "include_data_stats": False,
    })
    vis.add_statistics_panel(hr_data, {
        "title": "HR (ground truth):",
        "stats": _image_stats(hr_data),
        "include_data_stats": False,
    })
    vis.add_statistics_panel(residual_stretched, {
        "title": "Residual (asinh space):",
        "stats": _residual_asinh_stats(residual_stretched, residual_e),
        "include_data_stats": False,
    })
    vis.add_statistics_panel(residual_e, {
        "title": "PSNR / error metrics:",
        "stats": _residual_metrics(residual_stretched, residual_e,
                                   hr_stretched, hr_data),
        "include_data_stats": False,
    })

    plt.suptitle("Super-Resolution Reconstruction" + regime_note,
                 fontsize=16)
    vis.save_figure(output_path)
