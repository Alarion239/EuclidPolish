"""Which members a spatial gate needs: the "used by the gate" rule.

A gate's softmax never gives a member exactly 0 %, and the all-pixel mean
weight the console rounds to one decimal hides specialists: a member can
carry ~1 % of the weight over all pixels (sky dominates) yet half of it in
bright cores. So a member counts as *used* when its **peak** weight — the
maximum, per band, over the all-pixel mean, the source-pixel mean and the
mean in every brightness bin of the gate's held-out weight diagnostic
(``ensemble_viz._spatial_gate_weight_diagnostic``: ``usage``,
``usage_source``, ``usage_by_brightness``) — reaches a threshold, 0.5 % by
default. A pruned gate refit on exactly the used members
(``fit_spatial_gate.py fit --members used``) then lets production run only
those members.

The rule is a pure function of the diagnostic payload;
:func:`production_used_members` reads the production gate's cached one
(``<regime>/spatial_gate_combiner_evals.json``) — no networks, no writes.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from euclid_polish.eval.combiner import COMBINER_MODELS, combiner_artifact_fingerprint
from euclid_polish.eval.spatial_gate import SPATIAL_GATE_KIND

#: Default peak-weight threshold of the rule (0.5 %).
DEFAULT_USED_THRESHOLD = 0.005
#: The ``--members`` keyword that selects the used members.
USED_KEYWORD = "used"


def parse_used_threshold(raw: str) -> float | None:
    """``"used"`` → the default threshold, ``"used:0.5%"`` / ``"used:0.005"``
    → that threshold, anything else → ``None`` (an explicit member list).
    A malformed or out-of-range threshold raises :class:`ValueError`."""
    text = str(raw or "").strip().lower()
    if text == USED_KEYWORD:
        return DEFAULT_USED_THRESHOLD
    if not text.startswith(USED_KEYWORD + ":"):
        return None
    value = text[len(USED_KEYWORD) + 1:].strip()
    try:
        threshold = (float(value[:-1]) / 100.0 if value.endswith("%")
                     else float(value))
    except ValueError as exc:
        raise ValueError(f"bad used threshold {raw!r} (e.g. used:0.5%)") from exc
    if not np.isfinite(threshold) or not 0.0 < threshold < 1.0:
        raise ValueError(f"used threshold must lie in (0, 100 %): {raw!r}")
    return threshold


def member_peak_weights(diagnostic: Mapping[str, Any], n_members: int) -> list[float]:
    """Each member's peak weight in ``diagnostic`` — the maximum over bands
    of its all-pixel mean, source-pixel mean and every brightness-bin mean.

    Raises :class:`ValueError` when the diagnostic is unavailable or its
    arrays do not have ``n_members`` members."""
    if not diagnostic or not diagnostic.get("available"):
        reason = (diagnostic or {}).get("reason") or "no weight diagnostic"
        raise ValueError(f"the gate weight diagnostic is unavailable: {reason}")
    peak = np.zeros(int(n_members), np.float64)
    seen = False
    for key in ("usage", "usage_source"):
        for band, values in (diagnostic.get(key) or {}).items():
            arr = np.asarray(values, np.float64).reshape(-1)
            if arr.shape != peak.shape:
                raise ValueError(f"{key}[{band}] has {arr.size} members, "
                                 f"expected {n_members}")
            peak = np.maximum(peak, arr)
            seen = True
    for band, bins in (diagnostic.get("usage_by_brightness") or {}).items():
        arr = np.asarray(bins, np.float64)
        if arr.ndim != 2 or arr.shape[1] != peak.size:
            raise ValueError(f"usage_by_brightness[{band}] has shape {arr.shape}, "
                             f"expected (bins, {n_members})")
        peak = np.maximum(peak, arr.max(axis=0))
        seen = True
    if not seen:
        raise ValueError("the gate weight diagnostic holds no usage")
    return [float(v) for v in peak]


@dataclass(frozen=True)
class GateMemberChoice:
    """The rule's verdict: ``(label, peak weight)`` of each kept and each
    dropped member, in the gate's member order."""

    threshold: float
    kept: tuple[tuple[str, float], ...]
    dropped: tuple[tuple[str, float], ...]

    @property
    def kept_labels(self) -> list[str]:
        return [label for label, _ in self.kept]

    @property
    def dropped_labels(self) -> list[str]:
        return [label for label, _ in self.dropped]


def used_members(diagnostic: Mapping[str, Any], labels: Sequence[str], *,
                 threshold: float = DEFAULT_USED_THRESHOLD) -> GateMemberChoice:
    """Split ``labels`` (the gate's fitted members, in order) into the
    members the gate uses (peak weight ≥ ``threshold``) and the rest."""
    peaks = member_peak_weights(diagnostic, len(labels))
    pairs = list(zip([str(v) for v in labels], peaks, strict=True))
    kept = tuple(p for p in pairs if p[1] >= threshold)
    if not kept:
        raise ValueError(f"no member reaches {threshold:.2%} of the gate weight")
    return GateMemberChoice(float(threshold), kept,
                            tuple(p for p in pairs if p[1] < threshold))


def production_used_members(regime_dir: str, *,
                            threshold: float = DEFAULT_USED_THRESHOLD) -> GateMemberChoice:
    """The rule applied to the PRODUCTION gate's cached weight diagnostic.

    Raises :class:`ValueError` when the payload is missing, or was computed
    for another artifact than the production gate on disk now (open the
    Models › Combiner tab once to refresh it)."""
    artifact_dir = COMBINER_MODELS[SPATIAL_GATE_KIND].artifact_dir
    path = os.path.join(regime_dir, f"{artifact_dir}_evals.json")
    try:
        with open(path) as handle:
            payload = json.load(handle)
    except (OSError, ValueError) as exc:
        raise ValueError(f"no cached production gate payload at {path}") from exc
    diagnostic = payload.get("gate_diagnostics") or {}
    now = combiner_artifact_fingerprint(regime_dir, artifact_dir)
    if not now or diagnostic.get("artifact_fp") != now:
        raise ValueError("the cached gate weight diagnostic is not for the production "
                         "gate on disk (refit or promoted since) — open Models › "
                         "Combiner once to refresh it")
    return used_members(diagnostic, payload.get("member_labels") or [], threshold=threshold)


def format_choice(choice: GateMemberChoice) -> str:
    """Human summary: the kept and dropped members with their peak weights."""
    def row(pairs):
        return ", ".join(f"{label.split('·')[0]} ({peak:.2%})" for label, peak in pairs) or "—"
    return (f"members the gate uses (peak weight ≥ {choice.threshold:.2%}): "
            f"{len(choice.kept)}\n  kept:    {row(choice.kept)}\n"
            f"  dropped: {row(choice.dropped)}")


__all__ = [
    "DEFAULT_USED_THRESHOLD",
    "USED_KEYWORD",
    "GateMemberChoice",
    "format_choice",
    "member_peak_weights",
    "parse_used_threshold",
    "production_used_members",
    "used_members",
]
