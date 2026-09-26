"""Ensemble inference for the evaluators: STARFULL members + production combiner.

There is no single-model path — a lone checkpoint is an ensemble of one
(``member_00``), so every evaluator loads through :func:`load_eval_ensemble`
and predicts through :func:`sr_from_model`.

The evaluators (grouped run, catalog runner, the records' "Generate SR") used
to average every registry-active member of *both* star regimes. STARFULL is the
production regime (starless is opt-in), so they now load only STARFULL members
and reconstruct through the production combiner — ``ACTIVE_COMBINER_KINDS[0]``
(the spatial gate), fitted under ``<vis>/ensemble/starfull`` for exactly this
membership — falling back to the plain member mean when no current combiner
loads (none fitted yet, or fitted for a different membership).
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import numpy as np

from euclid_polish.config import Config
from euclid_polish.ensemble import EnsembleModel
from euclid_polish.ensemble_registry import default_ensemble_dir
from euclid_polish.eval.combiner import ACTIVE_COMBINER_KINDS, COMBINER_MODELS, load_combiner
from euclid_polish.image import Image, Role

#: The combiner production reconstructions use (the spatial gate).
PRODUCTION_COMBINER_KIND = ACTIVE_COMBINER_KINDS[0]


def starfull_regime_dir() -> str:
    """``<vis>/ensemble/starfull`` — where the STARFULL combiners are fitted."""
    return os.path.abspath(os.path.join(Config.VIS_DIR, "ensemble", "starfull"))


def load_production_combiner(member_labels: list[str],
                             combiner_dir: str | None = None) -> Any | None:
    """The fitted production combiner for exactly ``member_labels`` (their
    order included), or ``None`` when absent, stale or unreadable."""
    spec = COMBINER_MODELS[PRODUCTION_COMBINER_KIND]
    try:
        return load_combiner(combiner_dir or starfull_regime_dir(),
                             member_labels=list(member_labels),
                             artifact_dir=spec.artifact_dir)
    except Exception:  # noqa: BLE001 - a broken artifact means "use the mean"
        return None


class EvalEnsemble:
    """A STARFULL :class:`EnsembleModel` plus its production combiner.

    Delegates everything else to the ensemble (``n_members``,
    ``member_labels``, ``member_arrays`` …); :meth:`combine` turns a member
    stack into the production SR and :meth:`upsample_batch` is the drop-in
    for the records' SR job.
    """

    def __init__(self, ensemble: Any, combiner: Any | None,
                 combiner_kind: str | None) -> None:
        self._ensemble = ensemble
        self.combiner = combiner
        self.combiner_kind = combiner_kind if combiner is not None else None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._ensemble, name)

    @property
    def n_members(self) -> int:
        return int(self._ensemble.n_members)

    @property
    def member_labels(self) -> list[str]:
        return list(self._ensemble.member_labels)

    @property
    def label(self) -> str:
        """Human description for job logs and provenance notes."""
        if self.combiner is None:
            return f"member mean ({self.n_members} STARFULL models)"
        spec = COMBINER_MODELS[str(self.combiner_kind)]
        return f"{spec.label} over {self.n_members} STARFULL models"

    def member_arrays(self, lr_array: np.ndarray, *args: Any, **kwargs: Any) -> np.ndarray:
        return self._ensemble.member_arrays(lr_array, *args, **kwargs)

    def combine(self, members: np.ndarray, lr_cube: np.ndarray) -> np.ndarray:
        """Production SR of one field from its ``(M, H, W, C)`` member stack."""
        stack = np.asarray(members, dtype=np.float32)
        if self.combiner is None:
            return stack.mean(axis=0)
        lr = (np.asarray(lr_cube, np.float32)
              if getattr(self.combiner, "use_lr", False) else None)
        return np.asarray(self.combiner.apply_field(stack, lr=lr), dtype=np.float32)

    def upsample_batch(self, lr_images, *,
                       on_progress: Callable[[int, int, str], None] | None = None,
                       log: Callable[[str], None] | None = None) -> list[Image]:
        """Production SR :class:`Image` for every LR image, in order."""
        lr_list = list(lr_images)
        out: list[Image] = []
        for i, lr in enumerate(lr_list):
            _vis, sr, _members = sr_from_model(self, lr.data)
            bands = (lr.band_names if sr.ndim == 3
                     and sr.shape[-1] == len(lr.band_names) else ("VIS",))
            out.append(Image(data=np.asarray(sr, np.float32),
                             pixel_scale_arcsec=Config.DEFAULT_PIXEL_SCALE,
                             band_names=bands, is_clean=True, role=Role.SR,
                             index=lr.index, subset=lr.subset))
            if on_progress is not None:
                on_progress(i + 1, len(lr_list), f"field {lr.index}")
            if log is not None:
                log(f"field {lr.index}: {self.label}\n")
        return out


def sr_from_model(model: Any, lr_cube: np.ndarray
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """``(lr_vis, sr, members)`` for one LR cube.

    ``sr`` is the production reconstruction (:meth:`EvalEnsemble.combine`;
    the member mean for a model without a combiner). ``members`` is the
    ``(M, H, W, C)`` raw-electron stack when the ensemble has >1 model
    (disagreement is meaningful), else ``None`` — so a 1-member ensemble
    writes no all-zero std/PCA cubes and evaluates like a single model.
    """
    lr = np.asarray(lr_cube, dtype=np.float32)
    members = np.asarray(model.member_arrays(lr), dtype=np.float32)
    combine = getattr(model, "combine", None)
    sr = combine(members, lr) if callable(combine) else members.mean(axis=0)
    lr_vis = lr[..., 0] if lr.ndim == 3 else lr
    return lr_vis, sr, (members if int(model.n_members) > 1 else None)


def load_eval_ensemble(base_dir: str | None = None,
                       num_res_blocks: int | None = None, *,
                       log: Callable[[str], None] | None = None,
                       combiner_dir: str | None = None) -> EvalEnsemble:
    """THE eval model: the STARFULL members under ``base_dir`` (default
    location) with the production combiner fitted for them.

    Raises ``RuntimeError`` when the registry has no active STARFULL members —
    there is no single-model fallback (a lone model is an ensemble of 1).
    """
    emit = log or (lambda m: None)
    ens = EnsembleModel(base_dir or default_ensemble_dir(),
                        num_res_blocks=num_res_blocks or Config.DEFAULT_NUM_RES_BLOCKS,
                        starless=False)
    if ens.n_members < 1:
        raise RuntimeError(
            "no active STARFULL ensemble members — train one "
            "(scripts/train_ensemble.py --count 1 works) or pull members "
            "on the /ensemble page.")
    combiner = load_production_combiner(list(ens.member_labels), combiner_dir)
    model = EvalEnsemble(ens, combiner, PRODUCTION_COMBINER_KIND)
    emit(f"using {model.label}"
         + ("" if combiner is not None else
            " — no current production combiner for this membership"))
    return model


__all__ = [
    "PRODUCTION_COMBINER_KIND",
    "EvalEnsemble",
    "load_eval_ensemble",
    "load_production_combiner",
    "sr_from_model",
    "starfull_regime_dir",
]
