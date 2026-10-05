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
loads (none fitted yet, or a member it reads left the ensemble).

Only the members the production gate READS are restored and run
(:func:`production_plan`): a pruned gate (``active_members``) of 20 out of 30
fitted members loads 20 checkpoints. The gate is valid while every member it
reads is active — members that joined after the fit are a note, not a
reason to fall back. The model identity keeps the gate's FULL fitted list
(``member_labels``) so reuse checks and registry probes agree; what actually
ran is ``run_labels``.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from euclid_polish.config import Config
from euclid_polish.ensemble import EnsembleModel
from euclid_polish.ensemble_registry import default_ensemble_dir, regime_labels
from euclid_polish.eval.combiner import ACTIVE_COMBINER_KINDS, COMBINER_MODELS, load_combiner
from euclid_polish.image import Image, Role

_LOG = logging.getLogger(__name__)

#: The combiner production reconstructions use (the spatial gate).
PRODUCTION_COMBINER_KIND = ACTIVE_COMBINER_KINDS[0]


def starfull_regime_dir() -> str:
    """``<vis>/ensemble/starfull`` — where the STARFULL combiners are fitted."""
    return os.path.abspath(os.path.join(Config.VIS_DIR, "ensemble", "starfull"))


def load_production_combiner(active_labels: Sequence[str],
                             combiner_dir: str | None = None) -> Any | None:
    """The fitted production combiner when every member it READS is among
    ``active_labels`` (the active STARFULL members), else ``None`` (absent,
    a read member left, or unreadable). Members that joined after the fit do
    not make it stale: it never saw them and its math ignores them."""
    spec = COMBINER_MODELS[PRODUCTION_COMBINER_KIND]
    try:
        return load_combiner(combiner_dir or starfull_regime_dir(),
                             available_labels=list(active_labels),
                             artifact_dir=spec.artifact_dir)
    except Exception:  # noqa: BLE001 - a broken artifact means "use the mean"
        return None


def combiner_read_labels(combiner: Any, fallback: Sequence[str] = ()) -> list[str]:
    """The labels of the members ``combiner`` reads, in its fitted order
    (every fitted member for a combiner that cannot prune)."""
    fitted = [str(v) for v in (getattr(combiner, "member_labels", None) or fallback)]
    needed = getattr(combiner, "needed_member_indices", None)
    return [fitted[int(i)] for i in needed()] if callable(needed) else fitted


@dataclass(frozen=True)
class ProductionPlan:
    """What production SR runs for one active membership, resolved without
    loading any network.

    ``member_labels`` is the model identity: the gate's full fitted list (the
    active members for the plain mean). ``run_labels`` are the members whose
    SR is computed. ``joined`` are active members the gate was not fitted
    with (refit to consider them)."""

    member_labels: tuple[str, ...]
    run_labels: tuple[str, ...]
    combiner: Any | None
    joined: tuple[str, ...] = ()

    @property
    def combiner_kind(self) -> str | None:
        return PRODUCTION_COMBINER_KIND if self.combiner is not None else None


def production_plan(active_labels: Sequence[str],
                    combiner_dir: str | None = None) -> ProductionPlan:
    """The production gate and the members it needs for ``active_labels``;
    the member mean of all of them when no current gate loads."""
    active = tuple(str(v) for v in active_labels)
    combiner = load_production_combiner(active, combiner_dir) if active else None
    if combiner is None:
        return ProductionPlan(active, active, None)
    fitted = tuple(str(v) for v in (getattr(combiner, "member_labels", None) or active))
    return ProductionPlan(fitted, tuple(combiner_read_labels(combiner, fitted)), combiner,
                          tuple(label for label in active if label not in set(fitted)))


def combine_members(combiner: Any | None, members: np.ndarray,
                    lr_cube: np.ndarray) -> np.ndarray:
    """Production SR of one field from the ``(M, H, W, C)`` stack of the
    members ``combiner`` reads (the member mean without a combiner)."""
    stack = np.asarray(members, dtype=np.float32)
    if combiner is None:
        return stack.mean(axis=0)
    lr = (np.asarray(lr_cube, np.float32)
          if getattr(combiner, "use_lr", False) else None)
    return np.asarray(combiner.apply_field(stack, lr=lr), dtype=np.float32)


class EvalEnsemble:
    """A STARFULL :class:`EnsembleModel` plus its production combiner.

    The wrapped ensemble holds only the members the combiner reads
    (``run_labels``, ``n_run``); ``member_labels`` / ``n_members`` report the
    combiner's full fitted membership — the model identity the reuse checks
    and registry probes compare. Everything else delegates to the ensemble;
    :meth:`combine` turns a stack of the run members into the production SR
    and :meth:`upsample_batch` is the drop-in for the records' SR job.
    """

    def __init__(self, ensemble: Any, combiner: Any | None,
                 combiner_kind: str | None, *,
                 member_labels: Sequence[str] | None = None,
                 joined: Sequence[str] = ()) -> None:
        self._ensemble = ensemble
        self.combiner = combiner
        self.combiner_kind = combiner_kind if combiner is not None else None
        self._member_labels = None if member_labels is None else [str(v) for v in member_labels]
        self.joined = [str(v) for v in joined]

    def __getattr__(self, name: str) -> Any:
        return getattr(self._ensemble, name)

    @property
    def run_labels(self) -> list[str]:
        """The members this model actually runs (the combiner's reads)."""
        return list(self._ensemble.member_labels)

    @property
    def n_run(self) -> int:
        return int(self._ensemble.n_members)

    @property
    def member_labels(self) -> list[str]:
        return (list(self._member_labels) if self._member_labels is not None
                else self.run_labels)

    @property
    def n_members(self) -> int:
        return len(self.member_labels)

    @property
    def label(self) -> str:
        """Human description for job logs and provenance notes."""
        if self.combiner is None:
            return f"member mean ({self.n_run} STARFULL models)"
        spec = COMBINER_MODELS[str(self.combiner_kind)]
        if self.n_run != self.n_members:
            return (f"{spec.label} over {self.n_run} of {self.n_members} STARFULL "
                    "models (the members it reads)")
        return f"{spec.label} over {self.n_members} STARFULL models"

    def member_arrays(self, lr_array: np.ndarray, *args: Any, **kwargs: Any) -> np.ndarray:
        return self._ensemble.member_arrays(lr_array, *args, **kwargs)

    def combine(self, members: np.ndarray, lr_cube: np.ndarray) -> np.ndarray:
        """Production SR of one field from its run members' ``(M, H, W, C)`` stack."""
        return combine_members(self.combiner, members, lr_cube)

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
    ``(M, H, W, C)`` raw-electron stack of the members that RAN (the
    production gate's reads) when more than one ran (disagreement is
    meaningful), else ``None`` — so a 1-member run writes no all-zero
    std/PCA cubes and evaluates like a single model.
    """
    lr = np.asarray(lr_cube, dtype=np.float32)
    members = np.asarray(model.member_arrays(lr), dtype=np.float32)
    combine = getattr(model, "combine", None)
    sr = combine(members, lr) if callable(combine) else members.mean(axis=0)
    lr_vis = lr[..., 0] if lr.ndim == 3 else lr
    n_run = int(getattr(model, "n_run", model.n_members))
    return lr_vis, sr, (members if n_run > 1 else None)


_NO_MEMBERS = ("no active STARFULL ensemble members — train one "
               "(Models › Train, or scripts/train_ensemble.py --count 1) or "
               "pull members from FASRC in Models › Members.")


def load_eval_ensemble(base_dir: str | None = None,
                       num_res_blocks: int | None = None, *,
                       log: Callable[[str], None] | None = None,
                       combiner_dir: str | None = None) -> EvalEnsemble:
    """THE eval model: the production combiner fitted for the STARFULL
    members under ``base_dir`` (default location), restoring and running
    only the members it reads (:func:`production_plan`).

    Without a current production gate every active STARFULL member runs and
    the SR is their plain mean — logged as a warning, never silently.
    Raises ``RuntimeError`` when the registry has no active STARFULL members —
    there is no single-model fallback (a lone model is an ensemble of 1).
    """
    emit = log or (lambda m: None)
    base = base_dir or default_ensemble_dir()
    blocks = num_res_blocks or Config.DEFAULT_NUM_RES_BLOCKS
    active = list(regime_labels(base, False))
    if not active:
        raise RuntimeError(_NO_MEMBERS)
    plan = production_plan(active, combiner_dir)
    ens = None
    why = no_gate_reason(len(active))
    if plan.combiner is not None:
        try:
            ens = EnsembleModel(base, num_res_blocks=blocks, starless=False,
                                labels=list(plan.run_labels))
        except ValueError as exc:       # a member it reads has no checkpoint
            why = f"the members the production gate reads do not load ({exc})"
    if ens is None:
        ens = EnsembleModel(base, num_res_blocks=blocks, starless=False)
        if ens.n_members < 1:
            raise RuntimeError(_NO_MEMBERS)
        model = EvalEnsemble(ens, None, None)
        warn_mean_fallback(emit, model.n_run, why)
        return model
    model = EvalEnsemble(ens, plan.combiner, PRODUCTION_COMBINER_KIND,
                         member_labels=plan.member_labels, joined=plan.joined)
    emit(f"using {model.label}")
    if plan.joined:
        emit(f"note: {len(plan.joined)} member(s) joined after this fit "
             f"({', '.join(plan.joined[:6])}{'…' if len(plan.joined) > 6 else ''}); "
             "refit the gate to consider them")
    return model


def no_gate_reason(n_active: int) -> str:
    """Why production SR is the member mean when no gate loads."""
    return f"no current production gate for the {n_active} active STARFULL members"


def warn_mean_fallback(emit: Callable[[str], None], n: int, why: str) -> None:
    """Say — through ``emit`` and the ``logging`` warning channel, never
    silently — that production SR is the plain mean of ``n`` members."""
    message = (f"WARNING: production SR falls back to the plain member mean of "
               f"{n} STARFULL models — {why}")
    _LOG.warning(message)
    emit(message)


__all__ = [
    "PRODUCTION_COMBINER_KIND",
    "EvalEnsemble",
    "ProductionPlan",
    "combine_members",
    "combiner_read_labels",
    "load_eval_ensemble",
    "load_production_combiner",
    "no_gate_reason",
    "production_plan",
    "sr_from_model",
    "starfull_regime_dir",
    "warn_mean_fallback",
]
