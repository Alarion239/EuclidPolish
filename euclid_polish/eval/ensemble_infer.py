"""Inference shim that lets the grouped evaluator run either a single keras
model or the ensemble mean, and (for the ensemble) also return the member stack
so the disagreement cubes can be written for the movie viewer."""

from __future__ import annotations

from typing import Any

import numpy as np

from euclid_polish.config import Config
from euclid_polish.training.inference import reconstruct


def sr_from_model(model: Any, lr_cube: np.ndarray
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """``(lr_vis, sr, members)`` for one LR cube.

    When ``model`` exposes ``member_arrays`` (an :class:`EnsembleModel`), ``sr``
    is the ensemble mean and ``members`` is the ``(M, H, W, C)`` raw-electron
    stack. Otherwise ``model`` is a keras model run through :func:`reconstruct`
    and ``members`` is ``None``.
    """
    lr = np.asarray(lr_cube, dtype=np.float32)
    if hasattr(model, "member_arrays"):
        members = np.asarray(model.member_arrays(lr), dtype=np.float32)
        sr = members.mean(axis=0)
        lr_vis = lr[..., 0] if lr.ndim == 3 else lr
        return lr_vis, sr, members
    lr_vis, sr = reconstruct(model, lr)
    return lr_vis, sr, None


def load_eval_ensemble_or_single(checkpoint: str | None = None,
                                 num_res_blocks: int | None = None,
                                 *, log=None) -> Any:
    """Default eval model: the ensemble mean if a trained ensemble exists, else
    the single checkpoint. Logs which was chosen so a run is never ambiguous."""
    emit = log or (lambda m: None)
    from euclid_polish.ensemble import load_ensemble
    try:
        ens = load_ensemble(num_res_blocks=num_res_blocks
                            or Config.DEFAULT_NUM_RES_BLOCKS)
        if ens.n_members >= 1:
            emit(f"using ensemble mean ({ens.n_members} members)")
            return ens
        emit("ensemble present but empty; falling back to single model")
    except Exception as e:  # noqa: BLE001 — any load failure -> single model
        emit(f"ensemble unavailable ({type(e).__name__}: {e}); using single model")
    from euclid_polish.eval.catalog_runner import load_eval_model
    emit(f"loading single model from {checkpoint or Config.DEFAULT_CHECKPOINT_DIR}")
    return load_eval_model(checkpoint, num_res_blocks)
