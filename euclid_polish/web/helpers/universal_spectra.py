"""Models › Spectra: every local universal model's power spectrum against
the truth's, their coherence and transfer per period, and the direction
structure of its error (:mod:`~euclid_polish.eval.model_spectra`).

The records are made by the universal score job
(:func:`~euclid_polish.web.helpers.universal_models.job_universal_score`) in
the same pass over the same test fields as the scores, one
``<name>.spectra.json`` per model in the universal scores store. A record is
``current`` while its checkpoint fingerprint and test identity still hold,
``stale`` otherwise, ``missing`` when the model has none.

:func:`spectra_payload` (``GET /ensemble/universal/spectra.json``) and
:func:`spectra_map_payload` (``GET /ensemble/universal/spectra/<name>.json``)
only read: no TensorFlow, nothing written.
"""

from __future__ import annotations

from euclid_polish import ensemble_registry
from euclid_polish.config import Config
from euclid_polish.ensemble import member_fingerprint
from euclid_polish.eval.combiner import BAND_NAMES
from euclid_polish.eval.model_spectra import (
    DIRECTION_KNEE_E,
    DIRECTION_WINDOW,
    LONG_PERIOD_BANDS,
    LONG_WINDOW,
    PERIOD_BANDS,
    RADIAL_KNEE_E,
    RADIAL_WINDOW,
    SCHEMA,
    angle_centres,
    period_px,
)
from euclid_polish.web.helpers import ensemble_viz as ev
from euclid_polish.web.helpers.universal_models import _number, train_time, universal_targets


def spectra_status(record: dict | None, d: str, identity: dict) -> str:
    """``missing`` without a record; ``stale`` when its checkpoint or test
    identity changed (records that are not local cannot be checked);
    ``current`` otherwise."""
    if not record:
        return "missing"
    if (record.get("fingerprint") != member_fingerprint(d)
            or not ev.universal_identity_matches(record.get("identity"), identity)):
        return "stale"
    return "current"


def _model_entry(name: str, d: str, *, scores: dict, identity: dict) -> tuple[dict, dict | None]:
    """One model's summary row and its stored record."""
    record = ev.read_universal_spectra(name)
    status = spectra_status(record, d, identity)
    origin = ev._member_origin(d) or {}
    score = scores["models"].get(name) or {}
    available = bool(record and record.get("available"))
    radial = record.get("radial") if available else None
    direction = record.get("direction") if available else None
    served = ev._as_int(record.get("served_step")) if record else None
    knees = origin.get("asinh_knees")
    row = {
        "name": name, "label": ensemble_registry.member_label(name), "number": _number(name),
        "status": status,
        "computed_at": record.get("computed_at") if record else None,
        "n_fields": ev._as_int(record.get("n_fields")) if record else None,
        "mean_psnr": ev._finite(score.get("mean")),
        # the step of the checkpoint the spectra describe (else the log's)
        "steps": served if served is not None else train_time(d)["steps"],
        "served_step": served,
        "depth": ev._as_int(origin.get("num_res_blocks")),
        "knees": [ev._finite(k) for k in knees] if isinstance(knees, list) else None,
        "isotropy_penalty": ev._finite(origin.get("isotropy_penalty")) or 0.0,
        "radial": ({key: radial.get(key) for key in ("power_sr", "coherence", "transfer")}
                   if isinstance(radial, dict) else None),
        "axis_diag": direction.get("axis_diag") if isinstance(direction, dict) else None,
        "angular": direction.get("angular") if isinstance(direction, dict) else None,
        "reason": record.get("reason") if record and not available else None,
    }
    return row, record


def _truth_power(entries: list[tuple[dict, dict | None]]) -> list | None:
    """The truth's radial power (the same for every record of one identity):
    from a current record, else from any record that has it."""
    for want in ("current", "stale"):
        for row, record in entries:
            radial = (record or {}).get("radial")
            if row["status"] == want and isinstance(radial, dict) and radial.get("power_truth"):
                return radial["power_truth"]
    return None


def _entries() -> tuple[list[tuple[dict, dict | None]], dict]:
    identity = ev.universal_score_identity()
    scores = ev.read_universal_scores()
    entries = [_model_entry(name, d, scores=scores, identity=identity)
               for name, (_cls, d) in universal_targets(ev.ensemble_dir()).items()]
    return entries, identity


def spectra_payload() -> dict:
    """Models › Spectra (``GET /ensemble/universal/spectra.json``): the shared
    axes, the truth's power and every local universal model's row (status,
    recipe, radial spectra, axis/diagonal table and angular profiles; the
    2-D maps are served one model at a time by :func:`spectra_map_payload`).
    Reads cached files only."""
    entries, identity = _entries()
    models = [row for row, _record in entries]
    missing = [m["name"] for m in models if m["status"] == "missing"]
    stale = [m["name"] for m in models if m["status"] == "stale"]
    notes: list[str] = []
    if missing:
        notes.append(f"{len(missing)} model(s) without spectra: {', '.join(missing)} — "
                     "compute them (the score job with spectra)")
    if stale:
        notes.append(f"{len(stale)} spectra stale (checkpoint, test fields or records changed): "
                     + ", ".join(stale))
    if identity.get("records_fp") is None:
        notes.append("the test records are not local — spectra cannot be computed or checked "
                     "against them (sync them on Synthetic › Records)")
    return {
        "schema": SCHEMA,
        "records_local": identity.get("records_fp") is not None,
        "bands": list(BAND_NAMES),
        "period_bands": [list(b) for b in (*PERIOD_BANDS, *LONG_PERIOD_BANDS)],
        "angle_deg": angle_centres(),
        "period_px": period_px(),
        "pixel_scale_arcsec": float(Config.DEFAULT_PIXEL_SCALE),
        "radial": {"window_px": RADIAL_WINDOW, "stretch_knee_e": RADIAL_KNEE_E},
        "direction": {"window_px": DIRECTION_WINDOW, "long_window_px": LONG_WINDOW,
                      "stretch_knee_e": DIRECTION_KNEE_E},
        "truth_power": _truth_power(entries),
        "models": models,
        "missing": missing,
        "notes": notes,
    }


def spectra_map_payload(name: str) -> dict | None:
    """One model's 2-D error-direction map (``GET
    /ensemble/universal/spectra/<name>.json``) with its summary row;
    ``None`` when ``name`` is no local universal model or has no map."""
    targets = universal_targets(ev.ensemble_dir())
    if name not in targets:
        return None
    row, record = _model_entry(name, targets[name][1], scores=ev.read_universal_scores(),
                               identity=ev.universal_score_identity())
    if not record or not record.get("available") or not isinstance(record.get("map"), dict):
        return None
    return {**row, "bands": record.get("bands") or list(BAND_NAMES), "map": record["map"]}


__all__ = ["spectra_map_payload", "spectra_payload", "spectra_status"]
