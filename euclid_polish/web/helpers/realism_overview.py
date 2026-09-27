"""Readiness of every synthetic-realism prior, for ``GET /api/realism/overview``.

One cheap, read-only answer to "can I generate realistic synthetic fields
right now, and what is stale?": the galaxy joint model, the stellar prior,
the TNG radius manifest (last remote validation, from its local cache), the
noise model and the noise model of the local TFRecords, the galaxy-plot and
field-statistics caches, the multipoint archive reference, the training
source catalogue, and the ``synthetic_generate`` gate.

Every item has the Home health-check shape (``routes/system.py``):
``{id, label, state: ok|warn|bad|unknown, title, detail, to, action?, facts}``.
Nothing here writes: fixes are the POST jobs each item's ``action`` names.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from euclid_polish.config import Config
from euclid_polish.sky.observation.mer_noise_levels import TABLE_PATH
from euclid_polish.web import euclid_session
from euclid_polish.web.helpers import galaxy_distributions, population_comparison
from euclid_polish.web.helpers.population_calibration import joint_galaxy_state, star_state
from euclid_polish.web.helpers.q1_galaxy_counts import read_q1_galaxy_aperture_counts
from euclid_polish.web.helpers.q1_galaxy_radius_statistics import read_q1_galaxy_radius_statistics

#: The ``synthetic_generate`` refusals, verbatim from
#: ``fasrc_pipeline.SyntheticGenerateStep.prepare_params`` (a parity test
#: holds the two together).
GALAXY_BLOCKER = (
    "activate the Euclid VIS 2FWHM × Sérsic-R_e galaxy fit before generating fields"
)
GALAXY_DENSITY_BLOCKER = "empirical PHZ galaxy population has no finite density"
STAR_BLOCKER = (
    "activate a valid Gaia+Euclid stellar calibration before generating fields"
)

#: Last remote TNG radius-manifest validation, cached by ``routes/tng.py``
#: (``POST /api/tng/radii/refresh``). Same file and TTLs as there.
TNG_RADII_TTL_S = 3600.0
TNG_RADII_FAILED_TTL_S = 300.0

STATES = ("ok", "warn", "bad", "unknown")


def tng_radii_cache_path() -> Path:
    return Path(Config.DATA_DIR) / "_tng_infographics" / "tng_radius_manifest_status.json"


def _short(fingerprint: Any) -> str | None:
    return f"{str(fingerprint)[:12]}…" if fingerprint else None


def _density(payload: dict[str, Any] | None) -> float | None:
    try:
        value = float(((payload or {}).get("generation") or {})["surface_density_arcmin2"])
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _item(item_id: str, label: str, state: str, title: str, *, detail: str | None = None,
          to: str | None = None, action: dict[str, Any] | None = None,
          facts: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"id": item_id, "label": label, "state": state, "title": title, "detail": detail,
            "to": to, "action": action, "facts": facts or {}}


def _action(label: str, url: str, *, confirm: str | None = None,
            requires_fasrc: bool = False, self_connects: bool = False,
            requires_login: bool = False,
            params: dict[str, str] | None = None) -> dict[str, Any]:
    """A fix the UI can run. ``requires_fasrc``: the endpoint is gated
    (``@requires_fasrc``, 503 offline), so the UI disables it offline.
    ``self_connects``: a local job that opens the FASRC connection itself
    (``ensure_ssh_connected``) and reports a failure in the job, so the UI
    keeps it enabled offline."""
    return {"label": label, "method": "POST", "url": url, "params": params or {},
            "confirm": confirm, "requires_fasrc": requires_fasrc,
            "self_connects": self_connects, "requires_login": requires_login}


def _q1_progress() -> dict[str, Any]:
    out: dict[str, Any] = {}
    try:
        counts = read_q1_galaxy_aperture_counts()
        out["aperture_checkpoints"] = [counts.get("completed_queries"), counts.get("total_queries")]
    except ValueError:
        out["aperture_checkpoints"] = None
    try:
        radii = read_q1_galaxy_radius_statistics()
        out["radius_brackets"] = [radii.get("completed_queries"), radii.get("total_queries")]
    except ValueError:
        out["radius_brackets"] = None
    return out


def galaxy_item(state: dict[str, Any]) -> dict[str, Any]:
    candidate = state.get("candidate") or None
    active = state.get("active") or None
    facts = {
        "candidate_fingerprint": (candidate or {}).get("fingerprint"),
        "active_fingerprint": (active or {}).get("fingerprint"),
        "candidate_valid": bool((candidate or {}).get("valid")),
        "is_active": bool(state.get("is_active")),
        "version": (candidate or active or {}).get("version"),
        "surface_density_arcmin2": _density(active if state.get("is_active") else candidate),
        "color_rows": ((candidate or {}).get("color_sfr_model") or {}).get("row_count"),
        **_q1_progress(),
    }
    to = "/realism/galaxies"
    activate = _action(
        "Activate model", "/api/galaxy-distributions/activate",
        confirm="Activate this galaxy candidate for synthetic generation? "
                "It replaces the active galaxy model.",
    )
    if state.get("is_active"):
        density = facts["surface_density_arcmin2"]
        return _item("galaxy-model", "Galaxy joint model", "ok", "Galaxy model active",
                     detail=f"{_short(facts['active_fingerprint'])}"
                            + (f" · {density:.0f} galaxies arcmin⁻²" if density else ""),
                     to=to, facts=facts)
    if candidate and candidate.get("valid"):
        title = "A newer galaxy candidate is not active" if active else "Galaxy candidate ready, not active"
        return _item("galaxy-model", "Galaxy joint model", "warn", title,
                     detail=f"candidate {_short(candidate.get('fingerprint'))}", to=to,
                     action=activate, facts=facts)
    if candidate:
        return _item("galaxy-model", "Galaxy joint model", "bad", "Galaxy candidate failed validation",
                     detail="Re-run Query MER + PHZ to refit it.", to=to, facts=facts)
    return _item("galaxy-model", "Galaxy joint model", "bad", "Galaxy model not fitted",
                 detail="Query MER + PHZ on Galaxies to fit it.", to=to, facts=facts)


def star_item(state: dict[str, Any]) -> dict[str, Any]:
    candidate = state.get("candidate") or None
    active = state.get("active") or None
    warnings = list((candidate or {}).get("warnings") or [])
    population = (active if state.get("is_active") else candidate) or {}
    facts = {
        "candidate_fingerprint": (candidate or {}).get("fingerprint"),
        "active_fingerprint": (active or {}).get("fingerprint"),
        "candidate_valid": bool((candidate or {}).get("valid")),
        "is_active": bool(state.get("is_active")),
        "density_arcmin2": ((population.get("population") or {}).get("density_arcmin2")),
        "warnings": warnings,
    }
    to = "/realism/stars"
    if state.get("is_active"):
        density = facts["density_arcmin2"]
        return _item("star-prior", "Stellar prior", "ok", "Stellar prior active",
                     detail=f"{_short(facts['active_fingerprint'])}"
                            + (f" · {float(density):.3f} stars arcmin⁻²" if density is not None else ""),
                     to=to, facts=facts)
    if candidate and candidate.get("valid"):
        title = "A newer stellar candidate is not active" if active else "Stellar candidate ready, not active"
        return _item("star-prior", "Stellar prior", "warn", title,
                     detail=f"candidate {_short(candidate.get('fingerprint'))}", to=to,
                     action=_action("Activate stellar prior", "/api/star-distribution/activate",
                                    confirm="Activate this stellar candidate for synthetic "
                                            "generation? It replaces the active stellar prior."),
                     facts=facts)
    if candidate:
        return _item("star-prior", "Stellar prior", "bad", "Stellar candidate needs a refit",
                     detail=warnings[-1] if warnings else "The candidate failed validation.",
                     to=to, facts=facts)
    return _item("star-prior", "Stellar prior", "bad", "Stellar prior not fitted",
                 detail="Query stars, then fit the cached data on Stars.", to=to, facts=facts)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def tng_radii_item(now: float | None = None) -> dict[str, Any]:
    cached = _read_json(tng_radii_cache_path())
    now = time.time() if now is None else now
    refresh = _action("Validate on FASRC", "/api/tng/radii/refresh", requires_fasrc=True)
    to = "/data/tng"
    if cached is None:
        return _item("tng-radii", "TNG radius manifest", "unknown", "TNG radius manifest not validated",
                     detail="Validate the remote manifest once FASRC is connected.", to=to,
                     action=refresh, facts={"cached": False})
    try:
        checked_at = float(cached.get("checked_at") or 0.0)
    except (TypeError, ValueError):
        checked_at = 0.0
    ttl = TNG_RADII_FAILED_TTL_S if cached.get("failed") else TNG_RADII_TTL_S
    facts = {
        "cached": True,
        "valid": bool(cached.get("valid")),
        "valid_count": cached.get("valid_count"),
        "expected_count": cached.get("expected_count"),
        "failed_count": cached.get("failed_count"),
        "manifest_fingerprint": cached.get("manifest_fingerprint"),
        "checked_at": checked_at or None,
        "stale": now - checked_at > ttl,
        "reasons": list(cached.get("reasons") or []),
    }
    if cached.get("valid"):
        counts = (f"{cached.get('valid_count')}/{cached.get('expected_count')} radii valid"
                  if cached.get("expected_count") is not None else "valid")
        return _item("tng-radii", "TNG radius manifest", "ok", "TNG radius manifest valid",
                     detail=counts, to=to, action=refresh, facts=facts)
    reason = facts["reasons"][0] if facts["reasons"] else "validation failed"
    return _item("tng-radii", "TNG radius manifest", "warn" if cached.get("failed") else "bad",
                 "TNG radius check failed" if cached.get("failed") else "TNG radius manifest invalid",
                 detail=str(reason), to=to, action=refresh, facts=facts)


def noise_model_item() -> dict[str, Any]:
    table = _read_json(TABLE_PATH) or {}
    rows = table.get("rows") or []
    return _item(
        "noise-model", "Noise model", "ok" if rows else "bad",
        f"Noise model {Config.NOISE_MODEL.rsplit('-', 1)[-1]}" if rows else "Noise-level table missing",
        detail=(f"{len(rows)} measured Q1 positions · {table.get('release')}" if rows
                else f"{TABLE_PATH.name} is missing or empty"),
        to="/realism/noise",
        facts={"noise_model": Config.NOISE_MODEL, "positions": len(rows),
               "release": table.get("release"), "retrieved_last": table.get("retrieved_last")},
    )


def records_noise_item(check_records_noise: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    """The Home ``records-noise`` check (``routes/system.py``, passed in by
    ``routes/realism.py`` so this helper never imports a route module)."""
    try:
        check = check_records_noise()
    except Exception as exc:  # noqa: BLE001 — a failing probe reads "unknown", never a 500
        return _item("records-noise", "TFRecord noise model", "unknown",
                     "Could not read the local records", detail=str(exc), to="/data/records")
    return _item("records-noise", "TFRecord noise model", str(check.get("state") or "unknown"),
                 str(check.get("title") or ""), detail=check.get("detail"),
                 to=check.get("to") or "/data/records", facts=check.get("facts") or {})


def galaxy_plots_item() -> dict[str, Any]:
    state = galaxy_distributions.artifact_state()
    build = _action("Rebuild plots", "/api/galaxy-distributions/build")
    facts = dict(state)
    if not state["present"]:
        return _item("galaxy-plots", "Galaxy plot cache", "warn", "Galaxy plots not built",
                     detail="Build them from the cached Q1, generated and model data.",
                     to="/realism/galaxies", action=build, facts=facts)
    if state["stale"]:
        return _item("galaxy-plots", "Galaxy plot cache", "warn", "Galaxy plots need a rebuild",
                     detail=state["reason"], to="/realism/galaxies", action=build, facts=facts)
    return _item("galaxy-plots", "Galaxy plot cache", "ok", "Galaxy plots current",
                 to="/realism/galaxies", facts=facts)


def comparison_item(availability: dict[str, Any]) -> dict[str, Any]:
    cache = availability.get("comparison_cache") or {}
    synthetic = availability.get("synthetic") or {}
    real = availability.get("real") or {}
    fields = int(synthetic.get("fields") or 0) + int(real.get("fields") or 0)
    build = _action(
        "Rebuild statistics", "/api/population-comparison/build",
        confirm=f"Stream {fields} synthetic and real fields through the pixel and "
                "detection statistics? Reads TFRecords with TensorFlow; several minutes.",
    )
    facts = {key: cache.get(key) for key in ("present", "schema_current", "fresh", "reason")}
    facts.update({"synthetic_fields": synthetic.get("fields"), "real_fields": real.get("fields")})
    if cache.get("fresh"):
        return _item("comparison-cache", "Field-statistics cache", "ok", "Field statistics current",
                     detail=f"{synthetic.get('fields')} synthetic vs {real.get('fields')} real fields",
                     to="/realism/pixels", facts=facts)
    return _item("comparison-cache", "Field-statistics cache", "warn",
                 "Field statistics not built" if not cache.get("present") else "Field statistics are stale",
                 detail=cache.get("reason"), to="/realism/pixels",
                 action=build if real.get("ready") else None, facts=facts)


def archive_item(availability: dict[str, Any]) -> dict[str, Any]:
    real = availability.get("real") or {}
    sync = _action(
        "Sync from FASRC", "/api/archive-fields/sync",
        confirm="Pull the 220 four-band archive fields and their manifest from FASRC "
                "(SHA-256 checked; replaces the local collection)?",
        self_connects=True,
    )
    facts = {key: real.get(key) for key in ("fields", "independent_parents", "ready", "current",
                                           "collection_fingerprint")}
    if real.get("ready") and real.get("current"):
        return _item("archive-fields", "Archive reference", "ok", "Archive fields ready",
                     detail=f"{real.get('fields')} fields from {real.get('independent_parents')} pointings",
                     to="/realism/visual", facts=facts)
    return _item("archive-fields", "Archive reference", "warn",
                 "Archive fields changed upstream" if real.get("ready") else "Archive fields not ready",
                 detail=real.get("unavailable_reason"), to="/realism/visual", action=sync, facts=facts)


def training_item(availability: dict[str, Any]) -> dict[str, Any]:
    synthetic = availability.get("synthetic") or {}
    cached = bool(synthetic.get("train_source_catalog"))
    facts = {
        "cached": cached,
        "population_fields": synthetic.get("population_fields"),
        "population_fields_with_training": synthetic.get("population_fields_with_training"),
    }
    sync = training_sync_action()
    if cached:
        return _item("training-catalog", "Training source catalogue", "ok",
                     "Training catalogue cached",
                     detail=f"{synthetic.get('population_fields_with_training')} fields with training",
                     to="/realism/galaxies", action=sync, facts=facts)
    return _item("training-catalog", "Training source catalogue", "unknown",
                 "Training catalogue not synced",
                 detail="Optional: sources_train.csv adds the training split to the censuses.",
                 to="/realism/galaxies", action=sync, facts=facts)


def training_sync_action() -> dict[str, Any]:
    return _action(
        "Sync training catalogue", "/api/population-comparison/sync-training-catalog",
        confirm="Pull sources_train.csv from FASRC, refresh the population census and "
                "rebuild the galaxy plots?",
        self_connects=True, params={"rebuild": "1"},
    )


def generation_gate(galaxy: dict[str, Any], star: dict[str, Any]) -> dict[str, Any]:
    """Would ``synthetic_generate`` accept a submission now? Mirrors
    ``SyntheticGenerateStep.prepare_params`` and lists EVERY blocker (the
    step raises at the first one; ``message`` is that one)."""
    blockers: list[dict[str, str]] = []
    active = galaxy.get("active") or {}
    if not galaxy.get("is_active") or not active:
        blockers.append({"id": "galaxy-model", "message": GALAXY_BLOCKER})
    else:
        try:
            float(active["generation"]["surface_density_arcmin2"])
        except (KeyError, TypeError, ValueError):
            blockers.append({"id": "galaxy-model", "message": GALAXY_DENSITY_BLOCKER})
    if not star.get("is_active") or not star.get("active"):
        blockers.append({"id": "star-prior", "message": STAR_BLOCKER})
    return {
        "step": "synthetic_generate",
        "ready": not blockers,
        "blockers": blockers,
        "message": blockers[0]["message"] if blockers else None,
        "to": "/data/records",
    }


def overview_payload(
    *, check_records_noise: Callable[[], dict[str, Any]], now: float | None = None,
) -> dict[str, Any]:
    galaxy = joint_galaxy_state()
    star = star_state()
    availability = population_comparison.availability()
    items = [
        galaxy_item(galaxy),
        star_item(star),
        tng_radii_item(now),
        noise_model_item(),
        records_noise_item(check_records_noise),
        galaxy_plots_item(),
        comparison_item(availability),
        archive_item(availability),
        training_item(availability),
    ]
    counts = {state: sum(item["state"] == state for item in items) for state in STATES}
    synthetic = availability.get("synthetic") or {}
    return {
        "computed_at": datetime.now(UTC).isoformat(),
        "gate": generation_gate(galaxy, star),
        "items": items,
        "counts": counts,
        "authenticated": euclid_session.is_authenticated(),
        "training": {
            "available": bool(synthetic.get("train_source_catalog")),
            "population_fields": synthetic.get("population_fields"),
            "population_fields_with_training": synthetic.get("population_fields_with_training"),
            "sync": training_sync_action(),
        },
    }


__all__ = [
    "GALAXY_BLOCKER", "GALAXY_DENSITY_BLOCKER", "STAR_BLOCKER", "archive_item",
    "comparison_item", "galaxy_item", "galaxy_plots_item", "generation_gate",
    "noise_model_item", "overview_payload", "records_noise_item", "star_item",
    "tng_radii_cache_path", "tng_radii_item", "training_item", "training_sync_action",
]
