"""
Inference / reconstruction module for EuclidPolish.

Provides utilities for applying a trained WDSR model to low-resolution
images and visualizing the results.
"""


import re

import numpy as np
import tensorflow as tf
from astropy.wcs import WCS

from euclid_polish.config import Config
from euclid_polish.training.augmentation import asinh_stretch_lr, inverse_asinh_stretch_hr
from euclid_polish.training.models.common import resolve_single
from euclid_polish.training.models.wdsr import wdsr

# plot_reconstruction now lives in the visualization layer; re-exported here so
# ``from euclid_polish.training.inference import plot_reconstruction`` keeps working.
from euclid_polish.visualization.reconstruction import plot_reconstruction  # noqa: F401


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
) -> int | None:
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


def infer_checkpoint_num_res_blocks(checkpoint_dir: str,
                                    skip_kernel_size: int = 5) -> int | None:
    """Trunk depth (``num_res_blocks``) a saved checkpoint's model has, or None.

    WDSR's weighted layers are: 1 entry conv + 3 convs per res block + 1
    main-branch output conv + the skip conv(s) — and the skip convs are the
    only ``skip_kernel_size``-sided kernels in the model. Counting the
    distinct ``layer_with_weights-N`` indices under the checkpoint's
    ``model/`` prefix (which excludes optimizer slot duplicates) therefore
    gives ``L = 3·B + 2 + n_skip`` → ``B = (L - 2 - n_skip) / 3``.

    This is what lets members of DIFFERENT depths coexist in one ensemble:
    every load path asks the checkpoint how deep it is instead of trusting a
    global default. Returns None when there is no checkpoint, it is
    unreadable, or the layer count doesn't match the WDSR layout.
    """
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    if latest is None:
        return None
    try:
        reader = tf.train.load_checkpoint(latest)
        shapes = reader.get_variable_to_shape_map()
    except Exception:    # pragma: no cover — unreadable ckpt → caller default
        return None
    layer_re = re.compile(r"^model/layer_with_weights-(\d+)/")
    layers: set[int] = set()
    skip_layers: set[int] = set()
    for key, shp in shapes.items():
        m = layer_re.match(key)
        if m is None:
            continue
        idx = int(m.group(1))
        layers.add(idx)
        if (len(shp) == 4 and shp[0] == skip_kernel_size
                and shp[1] == skip_kernel_size):
            skip_layers.add(idx)
    if not layers or not skip_layers:
        return None
    trunk = len(layers) - 2 - len(skip_layers)
    if trunk <= 0 or trunk % 3:
        return None
    return trunk // 3


def checkpoint_step(checkpoint_dir: str) -> int | None:
    """The persisted ``ckpt.step`` of the latest checkpoint, or ``None``.

    Reads the step variable straight off the checkpoint file (no model
    build) — the authoritative "how far has this member trained" used by
    continue-mode target computation.
    """
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    if latest is None:
        return None
    try:
        reader = tf.train.load_checkpoint(latest)
        return int(reader.get_tensor("step/.ATTRIBUTES/VARIABLE_VALUE"))
    except Exception:            # unreadable / legacy layout → caller decides
        return None


def infer_checkpoint_nchan_out(
    checkpoint_dir: str, scale: int, nchan_in: int,
    skip_kernel_size: int = 5,
) -> int | None:
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

    # Trunk depth likewise comes from the checkpoint — members of different
    # depths (e.g. 32- and 64-block) coexist in one ensemble, so the caller's
    # value is only a fallback. A depth mismatch would restore a prefix of the
    # trunk and leave the rest at random init under ``expect_partial``.
    ckpt_blocks = infer_checkpoint_num_res_blocks(checkpoint_dir)
    if ckpt_blocks is not None:
        if ckpt_blocks != num_res_blocks:
            print(f"  note: checkpoint has num_res_blocks={ckpt_blocks} "
                  f"(requested {num_res_blocks}); using the checkpoint's.")
        num_res_blocks = ckpt_blocks

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
) -> tuple[np.ndarray, np.ndarray]:
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

    # Stretch → model → unstretch. ``clip(±20)`` is a defensive guard against
    # an untrained / pathological model output: ``sinh(20) ≈ 2.4×10⁸``, well
    # above any realistic source, but finite in float32 (sinh overflows at ~89).
    # Per-band asinh stretch mirrors the training pipeline (different scale per
    # band) rather than a single scalar Config.STRETCH_SCALE_E.
    lr_stretched = asinh_stretch_lr(tf.constant(lr_for_model)).numpy().astype(np.float32)
    sr_stretched = resolve_single(model, tf.constant(lr_stretched)).numpy().astype(np.float32)
    sr_stretched = np.clip(sr_stretched, -20.0, 20.0)
    sr_data = inverse_asinh_stretch_hr(
        tf.constant(sr_stretched)
    ).numpy().astype(np.float32)
    # Single-output models: squeeze the channel axis for the 2-D
    # visualizers downstream. The 4-band model keeps its (H, W, 4) cube —
    # callers slice channel 0 for VIS or feed the cube to color panels.
    if sr_data.ndim == 3 and sr_data.shape[-1] == 1:
        sr_data = sr_data[..., 0]
    # LR returned for display: if the caller passed a multi-channel cube,
    # show only the VIS channel (band 0) — that's what the HR target is.
    if lr_data.ndim == 3 and lr_data.shape[-1] > 1 or lr_data.ndim == 3 and lr_data.shape[-1] == 1:
        lr_display = lr_data[..., 0]
    else:
        lr_display = lr_data

    return lr_display, sr_data




