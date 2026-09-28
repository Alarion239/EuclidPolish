"""The Loop: one staleness verdict per stage of the loop run after every batch.

Home's Loop strip and System › Lineage read the same verdicts from here
(spec 2026-09-27, Home). Seven stages, in loop order — Priors, Records,
Members, Evaluation, Gate, Real SR, Figures — each ``current | stale |
blocked | unknown`` with ONE short reason (it fits a chip in two lines), a
longer ``detail`` for the tooltip and ``to``, the tab whose confirmed button
fixes it.

There is no producer-stamped model identity on every product yet (spec,
Risks), so each stage reads the existing read-only checks:

* Priors, Records — :func:`realism_overview.overview_payload` (the Synthetic ›
  Status rows: the ``synthetic_generate`` gate, each ingredient's state and
  its "records built with this?" tick) and the ``records-noise`` alert;
* Members — the STARFULL roster (:func:`ensemble_viz.members_payload`) and the
  local ``ensemble_train`` job log (:func:`ensemble_viz.training_jobs`): what
  finished on FASRC and is not local yet (the Models › Members rule);
* Evaluation, Gate, Real SR — the ``evaluation``, ``knee``, ``combiner`` and
  ``real-sr`` checks of ``GET /api/system/alerts``;
* Figures — the newest NEXUS render (:func:`nexus_plates.list_runs`) against
  the catalogue's production fingerprint, and the ``galaxy-plots`` cache.

:func:`loop_stages` is pure (``tests/test_system_alerts.py``);
:func:`loop_payload` gathers the sources and memoises the answer for
:data:`LOOP_TTL_S`. The alerts payload and the records-noise check are
passed in by the route (``routes/system.py`` owns them), so this module
never imports a route. Everything is read-only: nothing here starts a job.
"""

from __future__ import annotations

import json
import re
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlencode

from euclid_polish.web.helpers import model_catalog, nexus_plates
from euclid_polish.web.helpers.ensemble_viz import members_payload, training_jobs
from euclid_polish.web.helpers.realism_overview import overview_payload

#: How long a computed Loop is served before it is recomputed (``?fresh=1`` skips it).
LOOP_TTL_S = 60.0
#: How long a finished batch counts as new (the Models › Members banner's window).
WAITING_DAYS = 14
_TERMINAL_WITH_CHECKPOINT = {"COMPLETED", "TIMEOUT"}
_TRUTHY = {"1", "true", "yes", "on"}

STAGE_LABELS = {
    "priors": "Priors",
    "records": "Records",
    "members": "Members",
    "evaluation": "Evaluation",
    "gate": "Gate",
    "real-sr": "Real SR",
    "figures": "Figures",
}
STAGE_ORDER = tuple(STAGE_LABELS)

#: What each blocker id of the synthetic_generate gate names.
_BLOCKER_TEXT = {"galaxy-model": "the galaxy model", "star-prior": "the stellar prior"}
#: What a records tick compares against.
_TICK_TEXT = {"galaxy-model": "galaxy model", "star-prior": "stellar prior", "noise-model": "noise model"}
#: The longest reason that still fits a chip in two lines; a longer one moves to the tooltip.
REASON_MAX = 44

_LOCK = threading.Lock()
_CACHE: tuple[float, dict[str, Any]] | None = None


# ── small helpers (the formats match the console's format.ts) ───────────────


def _lower_first(text: str) -> str:
    return text[:1].lower() + text[1:] if text else text


def _parse_time(value: Any) -> float | None:
    """Seconds since the epoch of an ISO string or a number; naive strings are local time."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return datetime.fromisoformat(value.strip().replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def relative(value: Any, now: float) -> str | None:
    """``"just now" | "N min ago" | "N h ago" | "N d ago"`` (format.ts formatRelative)."""
    at = _parse_time(value)
    if at is None:
        return None
    dt = at - now
    a = abs(dt)
    if a < 45:
        return "just now"
    if a < 45 * 60:
        n, unit = round(a / 60), "min"
    elif a < 22 * 3600:
        n, unit = round(a / 3600), "h"
    else:
        n, unit = round(a / 86400), "d"
    return f"{n} {unit} ago" if dt < 0 else f"in {n} {unit}"


def member_range(names: Iterable[str]) -> str | None:
    """``"members 199–202, 205"`` (homeModel.ts memberRange)."""
    nums = sorted(
        {n for n in (re.sub(r"^member_", "", str(x).strip()) for x in names) if n.isdigit()}, key=int
    )
    if not nums:
        return None
    parts: list[str] = []
    i = 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and int(nums[j + 1]) == int(nums[j]) + 1:
            j += 1
        if j - i >= 2:
            parts.append(f"{nums[i]}–{nums[j]}")
        else:
            parts.extend(nums[i : j + 1])
        i = j + 1
    return f"{'member' if len(nums) == 1 else 'members'} {', '.join(parts)}"


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if value == value and abs(value) != float("inf") else None


def _check(alerts: Mapping[str, Any] | None, check_id: str) -> Mapping[str, Any] | None:
    for check in (alerts or {}).get("checks") or []:
        if isinstance(check, Mapping) and check.get("id") == check_id:
            return check
    return None


def _bad(check: Mapping[str, Any] | None) -> bool:
    return bool(check) and check.get("state") in {"warn", "bad"}


def _stage(stage_id: str, state: str, reason: str, *, to: str, detail: str | None = None) -> dict[str, Any]:
    """One stage row (the chip: label, state dot, reason; the tooltip: detail)."""
    return {
        "id": stage_id,
        "label": STAGE_LABELS[stage_id],
        "state": state,
        "reason": reason,
        "detail": detail or None,
        "to": to,
    }


def _loading(stage_id: str, to: str) -> dict[str, Any]:
    return _stage(stage_id, "loading", "checking", to=to)


# ── the stages ──────────────────────────────────────────────────────────────


def priors_stage(o: Mapping[str, Any] | None) -> dict[str, Any]:
    to = "/synthetic/status"
    if o is None:
        return _loading("priors", to)
    gate = o.get("gate") or {}
    blockers = list(gate.get("blockers") or [])
    if gate and gate.get("ready") is False:
        names = ", ".join(_BLOCKER_TEXT.get(b.get("id"), str(b.get("id"))) for b in blockers)
        return _stage(
            "priors",
            "blocked",
            f"blocked by {len(blockers)}: {names}",
            to=to,
            detail="; ".join(str(b.get("message") or "") for b in blockers),
        )
    items = [i for i in (o.get("items") or []) if (i.get("group") or "generation") == "generation"]
    failing = [i for i in items if i.get("state") in {"warn", "bad"}]
    if failing:
        first = failing[0]
        more = f" (+{len(failing) - 1} more)" if len(failing) > 1 else ""
        full = f"{first.get('label')}: {_lower_first(str(first.get('title') or ''))}{more}"
        if len(full) <= REASON_MAX:
            reason, detail = full, first.get("detail")
        else:  # too long for the chip: its gist on the chip, the whole of it in the tooltip
            reason, detail = (
                f"{first.get('label')} needs a look{more}",
                f"{full}. {first.get('detail') or ''}".strip(),
            )
        return _stage("priors", "stale", reason, to=first.get("to") or to, detail=detail)
    if not items:
        return _stage("priors", "unknown", "no priors reported", to=to)
    galaxy = next((i for i in items if i.get("id") == "galaxy-model"), None)
    noise = next((i for i in items if i.get("id") == "noise-model"), None)
    version = ((galaxy or {}).get("facts") or {}).get("version")
    noise_v = re.search(r"-(v\d+)$", str(((noise or {}).get("facts") or {}).get("noise_model") or ""))
    parts = [
        (f"galaxies v{version}" if version is not None else "galaxies") if galaxy else None,
        "stars" if any(i.get("id") == "star-prior" for i in items) else None,
        (f"noise {noise_v.group(1)}" if noise_v else "noise") if noise else None,
    ]
    unchecked = [i for i in items if i.get("state") == "unknown"]
    detail = (
        "Not checked here: " + "; ".join(f"{i.get('label')}: {i.get('title')}" for i in unchecked) + "."
        if unchecked
        else None
    )
    return _stage(
        "priors", "current", " · ".join(p for p in parts if p) or "all active", to=to, detail=detail
    )


def records_stage(
    o: Mapping[str, Any] | None, alerts: Mapping[str, Any] | None, now: float
) -> dict[str, Any]:
    to = "/synthetic/records"
    noise = _check(alerts, "records-noise")
    if noise and noise.get("state") == "bad":
        return _stage(
            "records",
            "stale",
            "use an older noise model",
            to="/synthetic/status",
            detail=noise.get("detail") or noise.get("title"),
        )
    if o is None:
        return _loading("records", to)
    at = (o.get("records") or {}).get("generated_at")
    if not at:
        return _stage("records", "unknown", "no local records", to=to)
    ticks = [i for i in (o.get("items") or []) if (i.get("records") or {}).get("state")]
    predates = [i for i in ticks if i["records"]["state"] == "predates"]
    if predates:
        what = " and the ".join(
            _TICK_TEXT.get(i.get("id"), _lower_first(str(i.get("label") or i.get("id")))) for i in predates
        )
        return _stage(
            "records",
            "stale",
            f"predate the {what}",
            to="/synthetic/status",
            detail=" ".join(str(i["records"].get("detail")) for i in predates if i["records"].get("detail")),
        )
    unverified = [
        _TICK_TEXT.get(i.get("id"), str(i.get("label"))) for i in ticks if i["records"]["state"] == "unknown"
    ]
    when = relative(at, now)
    detail = "Generated after the active priors."
    if unverified:
        detail += f" Not verified here: the {' and the '.join(unverified)}."
    return _stage(
        "records",
        "current",
        f"built {when}, after the priors" if when else "built after the priors",
        to=to,
        detail=detail,
    )


def _job_regime(job: Mapping[str, Any]) -> str:
    """The regime a past training job trained in (run-wide flag or a per-member one)."""
    params = job.get("params") or {}
    if str(params.get("starless") or "").lower() in _TRUTHY:
        return "starless"
    spec = params.get("member_spec")
    if isinstance(spec, str):
        try:
            spec = json.loads(spec)
        except ValueError:
            spec = None
    if isinstance(spec, list) and any(isinstance(o, dict) and o.get("starless") is True for o in spec):
        return "starless"
    return "starfull"


def waiting_on_fasrc(jobs: Sequence[Mapping[str, Any]], members: Mapping[str, Any], now: float) -> list[str]:
    """STARFULL members that finished on FASRC and are not here yet: new members
    of add / fork batches that ended in the last :data:`WAITING_DAYS` and are
    neither active nor archived, and continued members whose job's target
    lies past their local step."""
    local = {r.get("name"): r for r in members.get("members") or []}
    archived = {r.get("name") for r in members.get("archived") or []}
    cutoff = now - WAITING_DAYS * 86400
    out: dict[str, None] = {}
    for job in jobs:
        if str(job.get("state") or "").upper() not in _TERMINAL_WITH_CHECKPOINT:
            continue
        ended = _parse_time(job.get("ended_at") or job.get("submitted_at"))
        if ended is not None and ended < cutoff:
            continue
        names = list(job.get("member_names") or [])
        if job.get("mode") == "continue":
            target = job.get("target_steps")
            for name in names:
                row = local.get(name)
                if row is not None and target is not None and (row.get("step") or 0) < target:
                    out[name] = None
            continue
        if _job_regime(job) != "starfull":
            continue
        for name in names:
            if name not in local and name not in archived:
                out[name] = None
    return list(out)


def members_stage(
    m: Mapping[str, Any] | None, jobs: Sequence[Mapping[str, Any]] | None, now: float
) -> dict[str, Any]:
    to = "/models/starfull/members"
    if m is None:
        return _loading("members", to)
    waiting = waiting_on_fasrc(jobs, m, now) if jobs else []
    if waiting:
        who = member_range(waiting) or ", ".join(waiting)
        return _stage(
            "members",
            "stale",
            f"{len(waiting)} new on FASRC",
            to=to,
            detail=f"{who} finished on FASRC and are not pulled: Pull from FASRC… on Models › Members.",
        )
    n = len(m.get("members") or [])
    return _stage(
        "members", "current" if n else "unknown", f"{n} active" if n else "no active members", to=to
    )


def evaluation_stage(alerts: Mapping[str, Any] | None, now: float) -> dict[str, Any]:
    to = "/models/starfull/leaderboard"
    if alerts is None:
        return _loading("evaluation", to)
    ev, knee = _check(alerts, "evaluation"), _check(alerts, "knee")
    if _bad(ev):
        return _stage(
            "evaluation",
            "stale",
            _lower_first(re.sub(r"^The evaluation ", "", ev["title"], flags=re.I)),
            to=to,
            detail=ev.get("detail"),
        )
    if _bad(knee):
        return _stage(
            "evaluation",
            "stale",
            _lower_first(re.sub(r"^PSNR-vs-knee ", "knee ", knee["title"], flags=re.I)),
            to=to,
            detail=knee.get("detail"),
        )
    if not ev or ev.get("state") == "unknown":
        return _stage("evaluation", "unknown", _lower_first(ev["title"]) if ev else "not checked", to=to)
    when = relative((ev.get("facts") or {}).get("evaluated_at"), now)
    return _stage("evaluation", "current", f"evaluated {when}" if when else "current", to=to)


def gate_stage(alerts: Mapping[str, Any] | None) -> dict[str, Any]:
    to = "/models/starfull/combiner"
    if alerts is None:
        return _loading("gate", to)
    c = _check(alerts, "combiner")
    if _bad(c):
        return _stage(
            "gate",
            "stale",
            _lower_first(re.sub(r"^The production gate ", "", c["title"], flags=re.I)),
            to=to,
            detail=c.get("detail"),
        )
    if not c or c.get("state") != "ok":
        return _stage("gate", "unknown", _lower_first(c["title"]) if c else "not checked", to=to)
    n = _num((c.get("facts") or {}).get("members"))
    return _stage(
        "gate",
        "current",
        f"fitted for the {int(n)} members" if n is not None else "fitted for the members",
        to=to,
    )


#: Real-tile store (the ``real-sr`` check's ``sources``) → its Sky › Targets set.
_TARGET_SETS = {"nexus": "nexus", "tile": "cached", "field": "legacy", "poster": "poster", "pair": "pairs"}
#: The order of the Sky › Targets set chips (so the link is the address Targets writes itself).
_TARGET_SET_ORDER = ("nexus", "poster", "cached", "legacy", "pairs")


def real_sr_stage(alerts: Mapping[str, Any] | None) -> dict[str, Any]:
    to = "/sky/targets"
    if alerts is None:
        return _loading("real-sr", to)
    c = _check(alerts, "real-sr")
    facts = (c or {}).get("facts") or {}
    stale = int(_num(facts.get("stale")) or 0)
    current = int(_num(facts.get("current")) or 0)
    if _bad(c) and stale:
        # Land on exactly the sets this check counted (Targets' "All" also
        # holds the catalogue sets, which it does not count).
        stale_sets = {
            _TARGET_SETS[str(s.get("source"))]
            for s in facts.get("sources") or []
            if isinstance(s, Mapping) and _num(s.get("stale")) and str(s.get("source")) in _TARGET_SETS
        }
        sets = [name for name in _TARGET_SET_ORDER if name in stale_sets]
        query = urlencode({"state": "stale", **({"set": ",".join(sets)} if sets else {})})
        return _stage(
            "real-sr", "stale", f"{stale} stale", to=f"/sky/targets?{query}", detail=c.get("detail")
        )
    if _bad(c):
        return _stage("real-sr", "stale", _lower_first(c["title"]), to=to, detail=c.get("detail"))
    if not c or c.get("state") == "unknown":
        return _stage("real-sr", "unknown", _lower_first(c["title"]) if c else "not checked", to=to)
    if current:
        return _stage("real-sr", "current", f"{current} current", to=to)
    return _stage("real-sr", "unknown", "none made yet", to=to)


def nexus_freshness(plates: Mapping[str, Any], catalog: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Is the newest NEXUS render made with today's production gate? ``{current, reason, detail}``."""
    runs = plates.get("runs") or []
    renders = (runs[0].get("renders") or []) if runs else []
    if not renders:
        return None
    render = renders[0]
    label = render.get("model_label") or render.get("model") or "a legacy SR"
    made_with = f"Made with {label}."
    again = " Render them again with production on Figures › Plates."
    if render.get("legacy") or not render.get("model"):
        return {"current": False, "reason": "NEXUS plates use a legacy SR", "detail": made_with + again}
    if render.get("model") != "production":
        return {"current": False, "reason": "NEXUS plates not from production", "detail": made_with + again}
    fingerprint = next(
        (m.get("fingerprint") for m in (catalog or {}).get("models") or [] if m.get("spec") == "production"),
        None,
    )
    if fingerprint and render.get("model_fingerprint") and render["model_fingerprint"] != fingerprint:
        return {
            "current": False,
            "reason": "NEXUS plates predate this fit",
            "detail": "Made with an earlier production gate fit." + again,
        }
    if any(t.get("model_state") and t.get("model_state") != "current" for t in render.get("tiles") or []):
        return {
            "current": False,
            "reason": "NEXUS plates use stale tiles",
            "detail": "Some tiles' production outputs are stale." + again,
        }
    return {"current": True, "reason": "NEXUS plates from production", "detail": made_with}


def figures_stage(
    plates: Mapping[str, Any] | None, catalog: Mapping[str, Any] | None, o: Mapping[str, Any] | None
) -> dict[str, Any]:
    to = "/figures/plates"
    if plates is None:
        return _loading("figures", to)
    nexus = nexus_freshness(plates, catalog)
    if nexus and not nexus["current"]:
        return _stage(
            "figures", "stale", nexus["reason"], to="/figures/plates?plate=nexus", detail=nexus["detail"]
        )
    plots = next((i for i in (o or {}).get("items") or [] if i.get("id") == "galaxy-plots"), None)
    if plots and plots.get("state") in {"warn", "bad"}:
        return _stage(
            "figures",
            "stale",
            "galaxy plots need a rebuild",
            to="/synthetic/status",
            detail="The Galaxy distributions plate reads them; rebuild them on Synthetic › Status.",
        )
    if not nexus:
        return _stage("figures", "unknown", "no NEXUS plates yet", to=to)
    return _stage("figures", "current", nexus["reason"], to=to, detail=nexus["detail"])


def loop_stages(
    *,
    alerts: Mapping[str, Any] | None,
    overview: Mapping[str, Any] | None,
    members: Mapping[str, Any] | None,
    jobs: Sequence[Mapping[str, Any]] | None,
    plates: Mapping[str, Any] | None,
    catalog: Mapping[str, Any] | None,
    now: float | None = None,
) -> list[dict[str, Any]]:
    """The seven stages, in loop order. A source passed as ``None`` has not
    answered: its stage reads ``loading`` ("checking"), never ``current``."""
    t = time.time() if now is None else now
    return [
        priors_stage(overview),
        records_stage(overview, alerts, t),
        members_stage(members, jobs, t),
        evaluation_stage(alerts, t),
        gate_stage(alerts),
        real_sr_stage(alerts),
        figures_stage(plates, catalog, overview),
    ]


# ── gathering ───────────────────────────────────────────────────────────────


def _safe(fn: Callable[[], Any], fallback: Any, errors: dict[str, str], name: str) -> Any:
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 - one broken source must not hide the other stages
        errors[name] = f"{type(exc).__name__}: {exc}"
        return fallback


def loop_payload(
    *,
    alerts: Callable[[], Mapping[str, Any]],
    check_records_noise: Callable[[], dict[str, Any]],
    fresh: bool = False,
) -> dict[str, Any]:
    """``{computed_at, ttl_s, stages:[…], counts, errors}``, memoised for
    :data:`LOOP_TTL_S`. A source that raises is read as empty (its stage
    reads ``unknown``) and named in ``errors``."""
    global _CACHE
    with _LOCK:
        cached = _CACHE
    if not fresh and cached is not None and time.monotonic() - cached[0] < LOOP_TTL_S:
        return cached[1]
    errors: dict[str, str] = {}
    alert_payload = _safe(alerts, {"checks": []}, errors, "alerts")
    overview = _safe(
        lambda: overview_payload(check_records_noise=check_records_noise), {}, errors, "overview"
    )
    members = _safe(lambda: members_payload(False), {}, errors, "members")
    jobs = _safe(training_jobs, [], errors, "jobs")
    plates = _safe(nexus_plates.list_runs, {"runs": []}, errors, "plates")
    catalog = _safe(model_catalog.catalog_payload, {}, errors, "catalog")
    stages = loop_stages(
        alerts=alert_payload, overview=overview, members=members, jobs=jobs, plates=plates, catalog=catalog
    )
    counts = dict.fromkeys(("current", "stale", "blocked", "unknown"), 0)
    for stage in stages:
        counts[stage["state"]] = counts.get(stage["state"], 0) + 1
    payload = {
        "computed_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "ttl_s": LOOP_TTL_S,
        "stages": stages,
        "counts": counts,
        "errors": errors,
    }
    with _LOCK:
        _CACHE = (time.monotonic(), payload)
    return payload


def clear_cache() -> None:
    """Forget the memoised Loop (tests; a producer that just changed a stage)."""
    global _CACHE
    with _LOCK:
        _CACHE = None
