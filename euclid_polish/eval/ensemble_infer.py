"""Ensemble inference for the evaluators: mean SR + (multi-member) stack.

There is no single-model path anymore — a lone checkpoint is an ensemble of
one (``member_00``), so every evaluator loads through
:func:`load_eval_ensemble` and predicts through :func:`sr_from_model`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from euclid_polish.config import Config
from euclid_polish.ensemble import load_ensemble


def sr_from_model(model: Any, lr_cube: np.ndarray
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """``(lr_vis, sr, members)`` for one LR cube.

    ``sr`` is the ensemble mean. ``members`` is the ``(M, H, W, C)``
    raw-electron stack when the ensemble has >1 model (disagreement is
    meaningful), else ``None`` — so a 1-member ensemble writes no all-zero
    std/PCA cubes and evaluates exactly like the old single model.
    """
    lr = np.asarray(lr_cube, dtype=np.float32)
    members = np.asarray(model.member_arrays(lr), dtype=np.float32)
    sr = members.mean(axis=0)
    lr_vis = lr[..., 0] if lr.ndim == 3 else lr
    return lr_vis, sr, (members if int(model.n_members) > 1 else None)


def load_eval_ensemble(base_dir: str | None = None,
                       num_res_blocks: int | None = None, *,
                       log=None) -> Any:
    """THE eval model: the ensemble under ``base_dir`` (default location).

    Raises ``RuntimeError`` when the registry has no active members — there
    is no single-model fallback (a lone model is an ensemble of 1).
    """
    emit = log or (lambda m: None)
    ens = load_ensemble(base_dir, num_res_blocks=num_res_blocks
                        or Config.DEFAULT_NUM_RES_BLOCKS)
    if ens.n_members < 1:
        raise RuntimeError(
            "no active ensemble members — train one "
            "(scripts/train_ensemble.py --n-members 1 works) or pull members "
            "on the /ensemble page.")
    emit(f"using ensemble mean ({ens.n_members} models)")
    return ens
