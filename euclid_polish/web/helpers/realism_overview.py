"""Readiness of every synthetic-realism prior, for ``GET /api/realism/overview``.

One cheap, read-only answer to "can I generate realistic synthetic fields
right now, and what is stale?" (Synthetic › Status): the ``synthetic_generate``
gate, then two groups of rows.

* ``generation`` — what a synthetic scene is drawn from: the galaxy joint
  model, the stellar prior, the noise model, the ePSFs, the TNG radius
  manifest (last remote validation, from its local cache), the saturation
  rule and the training source catalogue.
* ``diagnostic`` — the caches the realism checks read: the galaxy plots and
  the field statistics (whose input, the multipoint archive reference, is
  that row's problem when it is missing).

Every item has the Home health-check shape (``routes/system.py``):
``{id, label, state: ok|warn|bad|unknown, title, detail, to, action?, facts}``
plus ``group`` and ``records`` — whether the local records were built with
this ingredient (``current`` / ``predates`` / ``unknown``, from file times or
the records' provenance; ``None`` where nothing can tell). Nothing here
writes: fixes are the POST jobs each item's ``action`` names.
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
from euclid_polish.web import euclid_session, job_config
from euclid_polish.web.helpers import galaxy_distributions, population_comparison
from euclid_polish.web.helpers.paths import _sky_records_local_dir
from euclid_polish.web.helpers.population_calibration import (
    active_joint_galaxy_path,
    active_star_path,
    joint_galaxy_state,
    star_state,
)
from euclid_polish.web.helpers.q1_galaxy_counts import read_q1_galaxy_aperture_counts
from euclid_polish.web.helpers.q1_galaxy_radius_statistics import read_q1_galaxy_radius_statistics
from euclid_polish.web.helpers.status import psf_inventory_payload

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

#: The two row groups of Synthetic › Status.
GENERATION = "generation"
DIAGNOSTIC = "diagnostic"

#: The local splits whose dirty records the generation ticks read.
RECORD_SPLITS = ("test", "validate")


def tng_radii_cache_path() -> Path:
    return Path(Config.DATA_DIR) / "_tng_infographics" / "tng_radius_manifest_status.json"


def _density(payload: dict[str, Any] | None) -> float | None:
    try:
        value = float(((payload or {}).get("generation") or {})["surface_density_arcmin2"])
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _item(item_id: str, label: str, state: str, title: str, *, detail: str | None = None,
          to: str | None = None, action: dict[str, Any] | None = None,
          facts: dict[str, Any] | None = None, group: str = GENERATION,
          records: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"id": item_id, "label": label, "state": state, "title": title, "detail": detail,
            "to": to, "action": action, "facts": facts or {}, "group": group,
            "records": records}


def _iso(stamp: float | None) -> str | None:
    return datetime.fromtimestamp(stamp, UTC).isoformat(timespec="seconds") if stamp else None


def _mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def local_records_time(records_dir: str | None = None) -> float | None:
    """When the local synthetic records were generated: the oldest file time
    of the local ``dirty_<split>.tfrecord`` shards (the sync keeps the FASRC
    file times), or ``None`` without any."""
    root = Path(records_dir if records_dir is not None else _sky_records_local_dir())
    times = [t for t in (_mtime(root / f"dirty_{split}.tfrecord") for split in RECORD_SPLITS)
             if t is not None]
    return min(times) if times else None


def records_tick(prior_at: float | None, records_at: float | None, what: str) -> dict[str, Any]:
    """Were the local records built after this prior was activated?
    ``current`` / ``predates`` by file times; ``unknown`` without either."""
    base = {"records_at": _iso(records_at), "prior_at": _iso(prior_at)}
    if records_at is None:
        return {**base, "state": "unknown", "detail": "No local records to compare."}
    if prior_at is None:
        return {**base, "state": "unknown", "detail": f"No active {what} to compare."}
    if records_at >= prior_at:
        return {**base, "state": "current",
                "detail": f"The local records were generated after the {what} was activated."}
    return {**base, "state": "predates",
            "detail": f"The local records predate the {what}: regenerate them to use it."}


def noise_records_tick(check: dict[str, Any]) -> dict[str, Any]:
    """The records' noise model (the Home ``records-noise`` check) as a tick."""
    state = str(check.get("state") or "unknown")
    tick = {"ok": "current", "bad": "predates"}.get(state, "unknown")
    return {"state": tick, "detail": " ".join(
        str(part) for part in (check.get("title"), check.get("detail")) if part),
        "records_at": None, "prior_at": None}


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


def galaxy_item(state: dict[str, Any], records_at: float | None = None) -> dict[str, Any]:
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
    to = "/synthetic/galaxies"
    records = records_tick(_mtime(active_joint_galaxy_path()) if active else None,
                           records_at, "galaxy model")
    activate = _action(
        "Activate model", "/api/galaxy-distributions/activate",
        confirm="Activate this galaxy candidate for synthetic generation? "
                "It replaces the active galaxy model.",
    )

    def item(state_: str, title: str, **kw: Any) -> dict[str, Any]:
        return _item("galaxy-model", "Galaxies", state_, title, to=to, facts=facts,
                     records=records, **kw)

    if state.get("is_active"):
        # The density is the row's verdict; the fingerprints are the inspector's.
        return item("ok", "Galaxy model active")
    if candidate and candidate.get("valid"):
        title = "A newer galaxy candidate is not active" if active else "Galaxy candidate ready, not active"
        return item("warn", title, detail="Activate it to generate with it.", action=activate)
    if candidate:
        return item("bad", "Galaxy candidate failed validation",
                    detail="Re-run Query MER + PHZ to refit it.")
    return item("bad", "Galaxy model not fitted", detail="Query MER + PHZ on Galaxies to fit it.")


def star_item(state: dict[str, Any], records_at: float | None = None) -> dict[str, Any]:
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
    to = "/synthetic/stars"
    records = records_tick(_mtime(active_star_path()) if active else None,
                           records_at, "stellar prior")

    def item(state_: str, title: str, **kw: Any) -> dict[str, Any]:
        return _item("star-prior", "Stars", state_, title, to=to, facts=facts,
                     records=records, **kw)

    if state.get("is_active"):
        return item("ok", "Stellar prior active")
    if candidate and candidate.get("valid"):
        title = "A newer stellar candidate is not active" if active else "Stellar candidate ready, not active"
        return item("warn", title, detail="Activate it to generate with it.",
                    action=_action("Activate stellar prior", "/api/star-distribution/activate",
                                   confirm="Activate this stellar candidate for synthetic "
                                           "generation? It replaces the active stellar prior."))
    if candidate:
        return item("bad", "Stellar candidate needs a refit",
                    detail=warnings[-1] if warnings else "The candidate failed validation.")
    return item("bad", "Stellar prior not fitted",
                detail="Query stars, then fit the cached data on Stars.")


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def tng_radii_item(now: float | None = None) -> dict[str, Any]:
    cached = _read_json(tng_radii_cache_path())
    now = time.time() if now is None else now
    refresh = _action(
        "Validate TNG radii on FASRC", "/api/tng/radii/refresh", requires_fasrc=True,
        confirm="Runs scripts/validate_tng_radius_manifest.py on the cluster (up to a few "
                "minutes) and caches the result.",
    )
    to = "/synthetic/galaxies?view=templates"
    if cached is None:
        return _item("tng-radii", "TNG radii", "unknown", "TNG radius manifest not validated",
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
        return _item("tng-radii", "TNG radii", "ok", "TNG radius manifest valid",
                     detail=counts, to=to, action=refresh, facts=facts)
    reason = facts["reasons"][0] if facts["reasons"] else "validation failed"
    return _item("tng-radii", "TNG radii", "warn" if cached.get("failed") else "bad",
                 "TNG radius check failed" if cached.get("failed") else "TNG radius manifest invalid",
                 detail=str(reason), to=to, action=refresh, facts=facts)


def noise_model_item(check_records_noise: Callable[[], dict[str, Any]] | None = None) -> dict[str, Any]:
    """The noise-level table the generator draws from; its ``records`` tick is
    the Home ``records-noise`` check (``routes/system.py``, passed in by
    ``routes/realism.py`` so this helper never imports a route module)."""
    table = _read_json(TABLE_PATH) or {}
    rows = table.get("rows") or []
    records = None
    if check_records_noise is not None:
        try:
            records = noise_records_tick(check_records_noise())
        except Exception as exc:  # noqa: BLE001 — a failing probe reads "unknown", never a 500
            records = {"state": "unknown", "detail": f"Could not read the local records: {exc}",
                       "records_at": None, "prior_at": None}
    return _item(
        "noise-model", "Noise", "ok" if rows else "bad",
        f"Noise model {Config.NOISE_MODEL.rsplit('-', 1)[-1]}" if rows else "Noise-level table missing",
        detail=None if rows else f"{TABLE_PATH.name} is missing or empty",
        to="/synthetic/noise",
        facts={"noise_model": Config.NOISE_MODEL, "positions": len(rows),
               "release": table.get("release"), "retrieved_last": table.get("retrieved_last")},
        records=records,
    )


def psf_item(inventory: Callable[[], dict[str, Any]] = psf_inventory_payload) -> dict[str, Any]:
    """The ePSFs synthetic scenes are convolved with, from the local cache
    (``helpers/status.psf_inventory_payload``): empirical in every band, a
    Gaussian fallback where FASRC has none, or not synchronised here."""
    to = "/synthetic/psf?view=epsf"
    try:
        payload = inventory() or {}
        bands = list(payload.get("bands") or [])
    except Exception as exc:  # noqa: BLE001 — a failing probe reads "unknown", never a 500
        return _item("psf", "PSF", "unknown", "Could not read the ePSF cache", detail=str(exc), to=to)
    generation = payload.get("generation") or None
    names = [str(b.get("name", "")).replace("_E", "") for b in bands]
    by_state: dict[str, list[str]] = {}
    for name, band in zip(names, bands, strict=True):
        by_state.setdefault(str(band.get("state") or "not_cached"), []).append(name)
    facts = {
        "empirical": by_state.get("empirical", []),
        "gaussian_fallback": by_state.get("no_empirical", []),
        "not_cached": by_state.get("not_cached", []),
        "measured_fwhm_arcsec": {n: b.get("measured_fwhm") for n, b in zip(names, bands, strict=True)},
        "fallback_fwhm_arcsec": {n: b.get("fwhm") for n, b in zip(names, bands, strict=True)},
        # what the last generation run recorded in its records' provenance
        # (None: the local records predate the stamp, or none are synced)
        "records_psf_kinds": ({str(k).replace("_E", ""): v
                               for k, v in (generation.get("psf_kinds") or {}).items()}
                              if generation else None),
        "records_subset": generation.get("subset") if generation else None,
    }
    sync = _action(
        "Sync ePSFs", "/api/euclid-psf/sync", requires_fasrc=True,
        confirm="Force-pulls the four band ePSFs (the VIS stack is tens to hundreds of MB) "
                "and the cluster metadata.",
    )
    if bands and len(facts["empirical"]) == len(bands):
        return _item("psf", "PSF", "ok", "Empirical ePSF in every band", to=to, facts=facts)
    if facts["gaussian_fallback"]:
        return _item("psf", "PSF", "warn",
                     f"Gaussian fallback in {', '.join(facts['gaussian_fallback'])}",
                     detail="FASRC has no empirical ePSF for these bands.", to=to, facts=facts,
                     action=sync)
    return _item("psf", "PSF", "unknown", "ePSFs not synced to this machine",
                 detail="Generation on FASRC reads the FASRC ePSFs; sync them to see which.",
                 to=to, facts=facts, action=sync)


def saturation_item(config: Callable[[], Any] = job_config.load) -> dict[str, Any]:
    """The detector-saturation rule of the synthetic LR: the chance an
    above-well core is blacked out ramps from the configured base
    probability (just above the well) to ``SATURATION_MASK_PROB_BRIGHT``
    over ``SATURATION_MASK_RAMP_WELL_RATIOS`` × the well."""
    try:
        base = float(config().saturation_mask_prob)
    except Exception:  # noqa: BLE001 — an unreadable job config falls back to the code default
        base = float(Config.TRAIN_SATURATION_MASK_PROB)
    low, high = (float(v) for v in Config.SATURATION_MASK_RAMP_WELL_RATIOS)
    facts = {
        "base_probability": base,
        "bright_probability": float(Config.SATURATION_MASK_PROB_BRIGHT),
        "ramp_well_ratios": [low, high],
        "wells_e": {str(k).replace("_E", ""): float(v) for k, v in Config.STAR_SATURATION_WELL_E.items()},
        "extended_well_factor": float(Config.SATURATION_EXTENDED_WELL_FACTOR),
    }
    return _item("saturation", "Saturation rule", "ok", "Saturation rule", to="/system/config",
                 facts=facts)


def galaxy_plots_item() -> dict[str, Any]:
    state = galaxy_distributions.artifact_state()
    build = _action("Rebuild plots", "/api/galaxy-distributions/build")
    facts = dict(state)
    to = "/synthetic/galaxies"

    def item(state_: str, title: str, **kw: Any) -> dict[str, Any]:
        return _item("galaxy-plots", "Galaxy plots", state_, title, to=to, facts=facts,
                     group=DIAGNOSTIC, **kw)

    if not state["present"]:
        return item("warn", "Galaxy plots not built",
                    detail="Build them from the cached Q1, generated and model data.", action=build)
    if state["stale"]:
        return item("warn", "Galaxy plots need a rebuild", detail=state["reason"], action=build)
    return item("ok", "Galaxy plots current")


def _compared_real_fields(real: dict[str, Any]) -> int:
    """The real fields the statistics compare (the offset tiles; the centre
    tiles avoid bright stars), else every stored field."""
    return int(real.get("compared_fields") or real.get("fields") or 0)


def archive_sync_action() -> dict[str, Any]:
    """Pull the real reference (multipoint archive) fields from FASRC."""
    return _action(
        "Sync real reference fields", "/api/archive-fields/sync",
        confirm="Pull the 220 four-band archive fields and their manifest from FASRC "
                "(SHA-256 checked; replaces the local collection)?",
        self_connects=True,
    )


def comparison_item(availability: dict[str, Any]) -> dict[str, Any]:
    """The field statistics row. The real reference fields are its input, so
    a missing or upstream-changed archive collection is this row's problem
    (with the sync as its fix), not a row of its own."""
    cache = availability.get("comparison_cache") or {}
    synthetic = availability.get("synthetic") or {}
    real = availability.get("real") or {}
    compared = _compared_real_fields(real)
    fields = int(synthetic.get("fields") or 0) + compared
    build = _action(
        "Rebuild field statistics", "/api/population-comparison/build",
        confirm=f"Stream {fields} synthetic and real fields through the pixel and "
                "detection statistics? Reads TFRecords with TensorFlow; several minutes.",
    )
    facts = {key: cache.get(key) for key in ("present", "schema_current", "fresh", "reason")}
    facts.update({"synthetic_fields": synthetic.get("fields"), "real_fields": compared})
    facts.update({f"reference_{key}": real.get(key) for key in (
        "fields", "independent_parents", "ready", "current", "collection_fingerprint")})
    to = "/synthetic/fields?view=stats"
    if not (real.get("ready") and real.get("current")):
        return _item("comparison-cache", "Field statistics", "warn",
                     "Real reference fields changed upstream" if real.get("ready")
                     else "Real reference fields not synced",
                     detail=real.get("unavailable_reason"), to="/synthetic/fields?ref=1",
                     action=archive_sync_action(), facts=facts, group=DIAGNOSTIC)
    if cache.get("fresh"):
        return _item("comparison-cache", "Field statistics", "ok", "Field statistics current",
                     detail=f"{synthetic.get('fields')} synthetic vs {compared} real fields",
                     to=to, facts=facts, group=DIAGNOSTIC)
    return _item("comparison-cache", "Field statistics", "warn",
                 "Field statistics not built" if not cache.get("present") else "Field statistics are stale",
                 detail=cache.get("reason"), to=to, action=build, facts=facts, group=DIAGNOSTIC)


def training_item(availability: dict[str, Any]) -> dict[str, Any]:
    synthetic = availability.get("synthetic") or {}
    cached = bool(synthetic.get("train_source_catalog"))
    facts = {
        "cached": cached,
        "population_fields": synthetic.get("population_fields"),
        "population_fields_with_training": synthetic.get("population_fields_with_training"),
    }
    sync = training_sync_action()
    to = "/synthetic/galaxies"
    if cached:
        return _item("training-catalog", "Training catalogue", "ok", "Training catalogue cached",
                     to=to, action=sync, facts=facts)
    return _item("training-catalog", "Training catalogue", "unknown", "Training catalogue not synced",
                 detail="Optional: sources_train.csv adds the training split to the censuses.",
                 to=to, action=sync, facts=facts)


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
        "to": "/synthetic/records",
    }


def overview_payload(
    *, check_records_noise: Callable[[], dict[str, Any]], now: float | None = None,
    records_dir: str | None = None,
) -> dict[str, Any]:
    galaxy = joint_galaxy_state()
    star = star_state()
    availability = population_comparison.availability()
    records_at = local_records_time(records_dir)
    items = [
        galaxy_item(galaxy, records_at),
        star_item(star, records_at),
        noise_model_item(check_records_noise),
        psf_item(),
        tng_radii_item(now),
        saturation_item(),
        training_item(availability),
        galaxy_plots_item(),
        comparison_item(availability),
    ]
    counts = {state: sum(item["state"] == state for item in items) for state in STATES}
    synthetic = availability.get("synthetic") or {}
    return {
        "computed_at": datetime.now(UTC).isoformat(),
        "gate": generation_gate(galaxy, star),
        "items": items,
        "counts": counts,
        "records": {"generated_at": _iso(records_at), "splits": list(RECORD_SPLITS)},
        "authenticated": euclid_session.is_authenticated(),
        "training": {
            "available": bool(synthetic.get("train_source_catalog")),
            "population_fields": synthetic.get("population_fields"),
            "population_fields_with_training": synthetic.get("population_fields_with_training"),
            "sync": training_sync_action(),
        },
    }


__all__ = [
    "DIAGNOSTIC", "GALAXY_BLOCKER", "GALAXY_DENSITY_BLOCKER", "GENERATION", "STAR_BLOCKER",
    "archive_sync_action", "comparison_item", "galaxy_item", "galaxy_plots_item", "generation_gate",
    "local_records_time", "noise_model_item", "noise_records_tick", "overview_payload", "psf_item",
    "records_tick", "saturation_item", "star_item", "tng_radii_cache_path", "tng_radii_item",
    "training_item", "training_sync_action",
]
