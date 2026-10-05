"""What a study freeze would capture right now — read-only (``GET
/api/studies/candidates``).

:func:`candidates` answers the freeze dialog: the ensemble summary (members,
production gate, when it was evaluated) with a status per numbers block
(``current`` / ``stale`` / ``missing`` and why), the estimated numbers size,
whether a freeze is possible (the per-field knee curves need test cubes of
exactly the active membership) and every field that could be attached:

* ``test`` — the evaluation's cached test fields (HR truth);
* ``blackout`` — the combiner comparison's blackout copies of test fields
  (``cubes_blackout``; they hold only the stamped LR and the members, the
  mean and gate are computed at freeze);
* ``real`` — real tiles (STARFULL only) whose member-SR cache holds a current
  SR (checkpoint fingerprint and LR hash) for every active member.

A field is ``available`` only with SR for **all** active members; otherwise
it is listed with the ``reason``. Nothing here writes, runs a model or
touches FASRC (``fasrc_connected`` only reads the cached session state).
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import threading
from collections import OrderedDict
from collections.abc import Sequence
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image as PILImage

from euclid_polish.config import Config
from euclid_polish.eval.combiner import BAND_NAMES, COMBINER_MODELS
from euclid_polish.eval.knee_psnr import KNEE_GRID_E
from euclid_polish.eval.spatial_gate import SPATIAL_GATE_KIND, joined_after_fit, reads_available
from euclid_polish.image.tfio import tfrecord_path
from euclid_polish.studies.cache import FIELD_CACHE_BUDGET_BYTES
from euclid_polish.studies.store import MAX_FIELDS, check_field_id
from euclid_polish.visualization.color import eye_rgb
from euclid_polish.web.fasrc_gate import fasrc_connected
from euclid_polish.web.helpers import ensemble_viz as ev
from euclid_polish.web.helpers import experiments, model_catalog, real_tiles

KINDS = ("test", "blackout", "real")
PRODUCTION_DIR = COMBINER_MODELS[SPATIAL_GATE_KIND].artifact_dir
GATE_PREFIX = COMBINER_MODELS[SPATIAL_GATE_KIND].cube_prefix
#: Bytes per knee-curve number in knee_psnr.json (a rounded float + comma).
_JSON_NUMBER_BYTES = 9
_THUMB_DEFAULT = 160
_THUMB_MAX = 480
_LR_SHA_CACHE: OrderedDict[tuple, str] = OrderedDict()
_LR_SHA_LOCK = threading.Lock()
_CKPT_LINE = re.compile(r'model_checkpoint_path:\s*"(.+)"')


# ---------------------------------------------------------------------------
# field ids
# ---------------------------------------------------------------------------

def field_id(kind: str, ref: str) -> str:
    """``test`` + ``42`` → ``test-00042``; ``real`` + ``poster/x`` → ``real-poster-x``."""
    if kind in ("test", "blackout"):
        return check_field_id(f"{kind}-{int(ref):05d}")
    if kind == "real":
        source, _, identifier = str(ref).partition("/")
        return check_field_id(f"real-{source}-{identifier}")
    raise ValueError(f"unknown field kind {kind!r}")


def parse_field_id(fid: str) -> tuple[str, str]:
    """``(kind, ref)``: ref = the 5-digit record index or ``source/id``."""
    fid = check_field_id(fid)
    kind, _, rest = fid.partition("-")
    if kind == "real":
        source, _, identifier = rest.partition("-")
        if source not in real_tiles.SOURCES or not identifier:
            raise ValueError(f"bad field id {fid!r}")
        return kind, f"{source}/{identifier}"
    return kind, rest


# ---------------------------------------------------------------------------
# the live ensemble
# ---------------------------------------------------------------------------

def regime_slug(starless: bool) -> str:
    return "starless" if starless else "starfull"


def active_labels(starless: bool) -> list[str]:
    return list(ev._regime_labels(ev.ensemble_dir(), starless))


def _number_span(labels: Sequence[str]) -> str:
    """``["203·psnr", "204·psnr"]`` → ``203, 204`` (member numbers, ≤ 8 shown)."""
    names = [str(label).split("·")[0] for label in labels]
    text = ", ".join(names[:8])
    return text + (f" … (+{len(names) - 8})" if len(names) > 8 else "")


def _manifest(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _evaluation_provenance(regime: Path, active: Sequence[str]) -> list[str]:
    """Why the cubes are not the CURRENT weights: a member's checkpoint or the
    production gate changed since the evaluation (``eval_summary.json``
    ``eval_identity``; the gate follows the overview's eval-gate rule)."""
    summary = _manifest(regime / "eval_summary.json") or {}
    identity = summary.get("eval_identity") or {}
    recorded = [str(v) for v in summary.get("member_labels") or []]
    fps = identity.get("member_fps")
    problems = []
    if not isinstance(fps, list) or not recorded or len(fps) != len(recorded):
        problems.append("the evaluation recorded no checkpoint fingerprints of its members")
    else:
        then = dict(zip(recorded, fps, strict=True))
        now = model_catalog.member_fingerprints(active)
        changed = [lb for lb in active if lb in then and then[lb] != now.get(lb)]
        if len(changed) == 1:
            problems.append(f"member {_number_span(changed)}'s checkpoint changed since the "
                            "evaluation")
        elif changed:
            problems.append(f"the checkpoints of members {_number_span(changed)} changed "
                            "since the evaluation")
    gate_fp = ev._combiner_fingerprint(str(regime), SPATIAL_GATE_KIND)
    evaluated = (identity.get("combiner_fps") or {}).get(SPATIAL_GATE_KIND)
    if gate_fp and evaluated not in (None, gate_fp):
        problems.append("the production gate changed since the evaluation")
    return problems


def checkpoint_mtimes(labels: Sequence[str]) -> dict[str, float | None]:
    """When each member's served checkpoint index changed ON THIS MACHINE:
    ``max(mtime, ctime)`` — rsync keeps the FASRC mtime, so a checkpoint
    pulled after a blackout cube was built can carry an older mtime; its
    ctime is the pull. ``None`` without a checkpoint."""
    base = Path(ev.ensemble_dir())
    out: dict[str, float | None] = {}
    for label in labels:
        directory = base / model_catalog.member_name(label)
        out[str(label)] = None
        try:
            text = (directory / "checkpoint").read_text(encoding="utf-8")
        except OSError:
            continue
        match = _CKPT_LINE.search(text)
        if match:
            with contextlib.suppress(OSError):
                stat = (directory / f"{os.path.basename(match.group(1))}.index").stat()
                out[str(label)] = max(stat.st_mtime, stat.st_ctime)
    return out


def test_cubes_state(starless: bool, active: Sequence[str]) -> dict[str, Any]:
    """Whether the cached test cubes are the active membership's on the
    current records: ``{state, detail, manifest}``."""
    regime = Path(ev._regime_dir_ro(starless))
    manifest = _manifest(regime / "cubes" / "viz_index.json")
    if manifest is None:
        return {"state": "missing", "manifest": None,
                "detail": "No evaluation on the test set yet — Evaluate the ensemble first."}
    labels = [str(v) for v in manifest.get("member_labels") or []]
    missing = [lb for lb in active if lb not in labels]
    extra = [lb for lb in labels if lb not in active]
    problems = []
    if missing:
        problems.append(f"the evaluation predates members {_number_span(missing)}")
    if extra:
        problems.append(f"it includes members no longer active ({_number_span(extra)})")
    if not problems and labels != list(active):
        problems.append("its member order differs from the registry's")
    rdir = ev._sky_records_local_dir()
    subset = str(manifest.get("subset") or "")
    if not rdir or not os.path.isfile(tfrecord_path(rdir, f"{'clean' if starless else 'hr'}_{subset}")):
        problems.append("the test target records are not on this machine")
    elif manifest.get("records_fp") != ev._eval_records_fingerprint(rdir, subset,
                                                                     starless=starless):
        problems.append("the test records changed since the evaluation")
    problems += _evaluation_provenance(regime, active)
    n = len(manifest.get("indices") or [])
    if problems:
        text = "; ".join(problems)
        return {"state": "stale", "manifest": manifest,
                "detail": text[:1].upper() + text[1:] + " — re-evaluate to freeze."}
    return {"state": "current", "manifest": manifest,
            "detail": (f"{n} test fields × {len(labels)} members; per-field PSNR-vs-knee "
                       "curves are computed from them at freeze.")}


def production_gate(starless: bool, active: Sequence[str]) -> dict[str, Any]:
    """The production gate's identity and whether it fits the active members."""
    directory = Path(ev._regime_dir_ro(starless)) / PRODUCTION_DIR
    manifest = _manifest(directory / "combiner.json")
    if manifest is None:
        return {"available": False, "state": "missing", "name": None,
                "detail": "No production gate — fit a gate variant and promote it."}
    labels = [str(v) for v in manifest.get("member_labels") or []]
    chosen = manifest.get("active_members")
    reads = ([labels[int(i)] for i in chosen if int(i) < len(labels)]
             if isinstance(chosen, list) else labels)
    fit_meta = manifest.get("fit_meta") or {}
    name = fit_meta.get("promoted_from") or PRODUCTION_DIR
    joined = joined_after_fit(labels, list(active))
    missing = [lb for lb in reads if lb not in active]
    if not reads_available(reads, list(active)):
        state, detail = "stale", (f"It reads members that are not active "
                                  f"({_number_span(missing)}): its output cannot be frozen.")
    elif joined:
        state, detail = "stale", (f"Reads {len(reads)} members; {len(joined)} joined after "
                                  f"its fit ({_number_span(joined)}) — refit to consider them.")
    else:
        state, detail = "current", f"Reads {len(reads)} of the {len(active)} active members."
    return {"available": True, "state": state, "detail": detail, "name": str(name),
            "dir": PRODUCTION_DIR, "kind": manifest.get("kind"), "member_labels": labels,
            "reads": reads, "mix_space": manifest.get("mix_space", "asinh"),
            "use_lr": bool(manifest.get("use_lr")),
            "fitted_at": ev._iso_mtime(str(directory / "combiner.npz"))}


def matching_experiments(starless: bool) -> list[dict[str, Any]]:
    """Real-tile experiment records with at least one spec whose fingerprint
    is the current one (the specs of this membership); STARFULL only."""
    if starless:
        return []
    listed = experiments.list_experiments()
    if not listed:
        return []
    current = {spec: fp for spec, fp in model_catalog.current_fingerprints().items() if fp}
    out = []
    for item in listed:
        try:
            record = experiments.get_experiment(str(item.get("id")))
        except KeyError:
            continue
        specs = [spec for spec, fp in (record.get("fingerprints") or {}).items()
                 if fp and current.get(spec) == fp]
        if specs:
            out.append({**record, "matching_specs": specs})
    return out


# ---------------------------------------------------------------------------
# fields
# ---------------------------------------------------------------------------

def _file_bytes(path: Path) -> int:
    try:
        return int(path.stat().st_size)
    except OSError:
        return 0


def _sizes(member: int, n_members: int, core: int) -> dict[str, int]:
    """Upper-bound (uncompressed) sizes of a field: all products, the core
    set "Fetch field" brings, and the largest single product."""
    return {"bytes": int(member * n_members + core), "core_bytes": int(core),
            # Every product (a member, the mean, the gate, hr) is one SR plane.
            "largest_product_bytes": int(member)}


def _thumb_url(fid: str, starless: bool) -> str:
    return f"/api/studies/candidates/thumb/{fid}.jpg?mode={regime_slug(starless)}"


def _test_fields(starless: bool, active: Sequence[str], cubes: dict[str, Any]
                 ) -> list[dict[str, Any]]:
    manifest = cubes.get("manifest")
    if manifest is None:
        return []
    cubes_dir = Path(ev._regime_dir_ro(starless)) / "cubes"
    labels = [str(v) for v in manifest.get("member_labels") or []]
    subset = str(manifest.get("subset") or "test")
    out = []
    for rec in sorted(int(i) for i in manifest.get("indices") or []):
        tag = f"{rec:05d}"
        member_paths = [cubes_dir / f"member{i}_{tag}.npy" for i in range(len(labels))]
        absent = [labels[i] for i, p in enumerate(member_paths) if not p.is_file()]
        reason = None
        if cubes["state"] != "current":
            reason = cubes["detail"]
        elif absent:
            reason = f"no cached SR of member(s) {_number_span(absent)} in this field"
        elif not (cubes_dir / f"sr_{tag}.npy").is_file():
            reason = "the cached ensemble mean is missing"
        sr_bytes = _file_bytes(member_paths[0]) if member_paths else 0
        out.append({
            "fid": field_id("test", str(rec)), "kind": "test", "ref": str(rec),
            "label": f"{subset} · idx {rec}", "available": reason is None, "reason": reason,
            # mean + gate + hr on the SR grid, lr on the LR grid.
            **_sizes(sr_bytes, len(labels), 3 * sr_bytes + _file_bytes(cubes_dir / f"lr_{tag}.npy")),
            "thumb_url": _thumb_url(field_id("test", str(rec)), starless),
        })
    return out


def _blackout_fields(starless: bool, active: Sequence[str], cubes: dict[str, Any]
                     ) -> list[dict[str, Any]]:
    directory = Path(ev._regime_dir_ro(starless)) / "cubes_blackout"
    index = _manifest(directory / "blackout_index.json")
    if index is None:
        return []
    labels = [str(v) for v in (index.get("identity") or {}).get("member_labels") or []]
    missing = [lb for lb in active if lb not in labels]
    extra = [lb for lb in labels if lb not in active]
    shared = None
    if missing:
        shared = (f"the blackout cubes were built for {len(labels)} members; "
                  f"member(s) {_number_span(missing)} have none "
                  "(run a combiner comparison to rebuild them)")
    elif extra or labels != list(active):
        shared = "the blackout cubes hold members that are no longer active"
    elif cubes["state"] != "current":
        shared = "the test cubes (their HR truth) are stale: " + cubes["detail"]
    out = []
    trained = checkpoint_mtimes(labels) if shared is None else {}
    for rec in sorted(int(i) for i in index.get("indices") or []):
        tag = f"{rec:05d}"
        paths = [directory / f"member{i}_{tag}.npy" for i in range(len(labels))]
        absent = [labels[i] for i, p in enumerate(paths) if not p.is_file()]
        reason = shared
        if reason is None and absent:
            reason = f"no blackout SR of member(s) {_number_span(absent)} in this field"
        if reason is None:
            newer = [labels[i] for i, p in enumerate(paths)
                     if (trained.get(labels[i]) or 0.0) > p.stat().st_mtime]
            if newer:
                rebuild = ("delete cubes_blackout/blackout_index.json and run a combiner "
                           "comparison to rebuild them")
                reason = (f"member {_number_span(newer)}'s checkpoint is newer than its "
                          f"blackout cube ({rebuild})" if len(newer) == 1 else
                          f"the checkpoints of members {_number_span(newer)} are newer than "
                          f"their blackout cubes ({rebuild})")
        if reason is None and not (directory / f"lr_{tag}.npy").is_file():
            reason = "the stamped LR is missing"
        sr_bytes = _file_bytes(paths[0]) if paths else 0
        out.append({
            "fid": field_id("blackout", str(rec)), "kind": "blackout", "ref": str(rec),
            "label": f"blackout · idx {rec}", "available": reason is None, "reason": reason,
            # mean + gate + hr + a uint8 hole mask (¼ of a float plane) + the stamped LR.
            **_sizes(sr_bytes, len(labels), 3 * sr_bytes + sr_bytes // 4
                     + _file_bytes(directory / f"lr_{tag}.npy")),
            "thumb_url": _thumb_url(field_id("blackout", str(rec)), starless),
        })
    return out


def tile_lr_sha(entry: real_tiles.TileEntry) -> tuple[str, np.ndarray]:
    """``(lr_sha, model input)`` of a real tile — the key of its member-SR
    cache; memoised per listing stamp (the LR hash of an unchanged tile)."""
    tile = real_tiles.get_tile(entry.source, entry.id, entry=entry)
    lr = np.asarray(tile.lr_e, np.float32)
    lr_input = np.where(np.isfinite(lr), lr, 0.0).astype(np.float32)
    return model_catalog.array_sha(lr_input), lr_input


def _cached_lr_sha(entry: real_tiles.TileEntry) -> str:
    path = real_tiles.lr_path(entry)
    stamp = (entry.source, entry.id, str(path) if path else None,
             path.stat().st_mtime_ns if path else None, real_tiles.source_stamp(entry.source))
    with _LR_SHA_LOCK:
        if stamp in _LR_SHA_CACHE:
            _LR_SHA_CACHE.move_to_end(stamp)
            return _LR_SHA_CACHE[stamp]
    sha, _lr = tile_lr_sha(entry)
    with _LR_SHA_LOCK:
        _LR_SHA_CACHE[stamp] = sha
        while len(_LR_SHA_CACHE) > 64:
            _LR_SHA_CACHE.popitem(last=False)
    return sha


def missing_real_members(entry: real_tiles.TileEntry, active: Sequence[str]
                         ) -> tuple[list[str], dict[str, str | None]]:
    """Active members without a current cached SR on ``entry``."""
    fingerprints = model_catalog.member_fingerprints(active)
    members = experiments.CachedTileMembers(
        entry.source, entry.id, np.zeros((0,), np.float32), lr_sha=_cached_lr_sha(entry),
        runner=None, fingerprints=fingerprints)
    return [label for label in active if not members.cached(label)], fingerprints


def _real_fields(starless: bool, active: Sequence[str]) -> list[dict[str, Any]]:
    if starless:
        return []
    root = model_catalog.member_cache_root()
    if not root.is_dir():
        return []
    out = []
    for source_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if source_dir.name not in real_tiles.SOURCES:
            continue
        for tile_dir in sorted(p for p in source_dir.iterdir() if p.is_dir()):
            arrays = [p for p in tile_dir.glob("member_*.npy") if not p.name.startswith(".")]
            if not arrays:
                continue
            ref = f"{source_dir.name}/{tile_dir.name}"
            try:
                fid = field_id("real", ref)
            except ValueError:
                continue
            reason, label, entry_bytes = None, ref, 0
            try:
                entry = real_tiles.get_entry(source_dir.name, tile_dir.name)
                label = f"{source_dir.name} · {entry.label}"
                missing, _fps = missing_real_members(entry, active)
                if missing:
                    reason = (f"no current cached SR of member(s) {_number_span(missing)} "
                              "(run them on the tile in Sky › Compare)")
                side = entry.shape or (real_tiles.TILE_SIDE, real_tiles.TILE_SIDE)
                entry_bytes = int(side[0] * side[1] * len(entry.bands) * 4)
            except (real_tiles.RealTileError, OSError, ValueError, KeyError) as exc:
                reason = f"the tile is no longer readable ({exc})"
            member_bytes = max((_file_bytes(p) for p in arrays), default=0)
            out.append({
                "fid": fid, "kind": "real", "ref": ref, "label": label,
                "available": reason is None and bool(active), "reason": reason,
                # mean + gate on the SR grid, the LR.
                **_sizes(member_bytes, len(active), 2 * member_bytes + entry_bytes),
                "thumb_url": _thumb_url(fid, starless),
            })
    return out


# ---------------------------------------------------------------------------
# the dialog payload
# ---------------------------------------------------------------------------

def _numbers_bytes(n_members: int, n_fields: int, starless: bool,
                   active: Sequence[str]) -> int:
    """≈ size of the numbers files (knee curves dominate)."""
    n_models = n_members + 2
    bands = len(BAND_NAMES)
    knees = len(KNEE_GRID_E)
    knee = n_models * n_fields * knees * bands * _JSON_NUMBER_BYTES
    integrated = n_models * n_fields * bands * 48
    base = ev.ensemble_dir()
    curves = sum(_file_bytes(Path(base) / model_catalog.member_name(lb) / "training_log.csv")
                 for lb in active) // 2
    regime = Path(ev._regime_dir_ro(starless))
    gate = (_file_bytes(regime / COMBINER_MODELS[SPATIAL_GATE_KIND].payload_name) // 4
            + _file_bytes(regime / "spatial_gate_comparison.json") + 200_000)
    return int(knee + integrated + curves + gate + n_members * 600)


def ensemble_summary(starless: bool) -> dict[str, Any]:
    """The "what will be frozen" block of the dialog."""
    active = active_labels(starless)
    cubes = test_cubes_state(starless, active)
    gate = production_gate(starless, active)
    regime = Path(ev._regime_dir_ro(starless))
    blocks = [{"id": "members", "title": "Members",
               "state": "current" if active else "missing",
               "detail": (f"{len(active)} active {regime_slug(starless)} members, each with "
                          "its recipe (origin.json), steps and checkpoint fingerprint."
                          if active else "No active members in this regime.")},
              {"id": "test_cubes", "title": "Test-set curves", "state": cubes["state"],
               "detail": cubes["detail"]},
              {"id": "gate", "title": "Production gate", "state": gate["state"],
               "detail": gate["detail"]}]
    usage = ev._gate_usage(starless)
    blocks.append({"id": "gate_diagnostic", "title": "Gate weight diagnostic",
                   "state": ("missing" if not usage.get("available") else
                             "stale" if usage.get("stale") else "current"),
                   "detail": ("Held-out weight per member (all / sources / brightness)."
                              if usage.get("available") else
                              "No gate diagnostic payload — refit or re-evaluate the gate.")})
    report = ev._latest_compare(starless)
    if report is None:
        blocks.append({"id": "compare", "title": "Combiner comparison", "state": "missing",
                       "detail": ("No comparison report: the study keeps the weight "
                                  "diagnostic only (run Compare in Models › Combiner).")})
    else:
        compared = [str(v) for v in report.get("members") or []]
        same = compared == list(active)
        blocks.append({"id": "compare", "title": "Combiner comparison",
                       "state": "current" if same else "stale",
                       "detail": (f"Report {report.get('id') or ''} of {report.get('created')}"
                                  + ("." if same else
                                     f"; it compared {len(compared)} members, "
                                     f"{len(active)} are active now."))})
    logs = [lb for lb in active if os.path.isfile(os.path.join(
        ev.ensemble_dir(), model_catalog.member_name(lb), "training_log.csv"))]
    blocks.append({"id": "training_curves", "title": "Training curves",
                   "state": ("current" if len(logs) == len(active) and active else
                             "missing" if not logs else "stale"),
                   "detail": (f"{len(logs)} of {len(active)} members have a training log.")})
    real = matching_experiments(starless)
    blocks.append({"id": "real", "title": "Real-tile metrics",
                   "state": "current" if real else "missing",
                   "detail": (f"{len(real)} Sky › Compare run(s) used this membership."
                              if real else ("STARFULL only." if starless else
                                            "No Sky › Compare run used this membership."))})
    n_fields = len((cubes.get("manifest") or {}).get("indices") or [])
    return {
        "regime": regime_slug(starless), "members": active, "n_members": len(active),
        "gate": {k: gate.get(k) for k in ("available", "state", "name", "reads", "mix_space",
                                          "fitted_at")},
        "evaluated_at": ev._iso_mtime(str(regime / "eval_summary.json")),
        "blocks": blocks,
        "stale": [b["id"] for b in blocks if b["state"] != "current"],
        "numbers_bytes": _numbers_bytes(len(active), n_fields, starless, active),
        "_cubes": cubes,
    }


def candidates(starless: bool) -> dict[str, Any]:
    """The freeze dialog's contents (read-only)."""
    summary = ensemble_summary(starless)
    cubes = summary.pop("_cubes")
    active = summary["members"]
    blocking = None
    if not active:
        blocking = "No active members to freeze."
    elif cubes["state"] != "current":
        blocking = cubes["detail"]
    fields = (_test_fields(starless, active, cubes) + _blackout_fields(starless, active, cubes)
              + _real_fields(starless, active))
    for entry in fields:
        entry["bytes_upper_bound"] = True
        # A field must be viewable once fetched: its core products (and each
        # single product, e.g. one member SR) have to fit the fetch cache.
        too_big = max(int(entry["core_bytes"]), int(entry["largest_product_bytes"]))
        if entry["available"] and too_big > FIELD_CACHE_BUDGET_BYTES:
            entry["available"] = False
            entry["reason"] = (f"too large to fetch back: needs up to {too_big / 1024 ** 3:.1f} "
                               "GiB (its core products or one member SR), more than the "
                               f"{FIELD_CACHE_BUDGET_BYTES / 1024 ** 3:.0f} GiB field cache")
    connected = fasrc_connected()
    return {
        "regime": regime_slug(starless), "ensemble": summary, "fields": fields,
        "max_fields": MAX_FIELDS, "can_freeze": blocking is None, "blocking": blocking,
        "fasrc_connected": connected,
        "fields_note": (None if connected else
                        "FASRC is not connected: fields are stored on holylabs, so attaching "
                        "them needs the connection (freezing without fields works offline)."),
    }


# ---------------------------------------------------------------------------
# thumbnails
# ---------------------------------------------------------------------------

def _block_mean(cube: np.ndarray, side: int) -> np.ndarray:
    factor = max(1, min(cube.shape[0], cube.shape[1]) // max(1, 2 * side))
    if factor == 1:
        return cube
    h, w = (cube.shape[0] // factor) * factor, (cube.shape[1] // factor) * factor
    return cube[:h, :w].reshape(h // factor, factor, w // factor, factor, -1).mean(axis=(1, 3))


def render_thumbnail(cube: np.ndarray, side: int = _THUMB_DEFAULT) -> bytes:
    """A colour JPEG (the viewer's Temp rendering) of a 4-band cube."""
    side = max(32, min(int(side), _THUMB_MAX))
    data = np.nan_to_num(np.asarray(cube, np.float32))
    rgb = eye_rgb(_block_mean(data, side), tuple(Config.LR_INPUT_BAND_NAMES[:data.shape[-1]]),
                  asinh_scale_e=float(Config.STRETCH_SCALE_E))
    pixels = np.clip(np.asarray(rgb, np.float32)[..., :3], 0.0, 1.0)
    picture = PILImage.fromarray((pixels * 255.0 + 0.5).astype(np.uint8), "RGB")
    picture.thumbnail((side, side), PILImage.Resampling.LANCZOS)
    buffer = BytesIO()
    picture.save(buffer, format="JPEG", quality=85, optimize=True)
    return buffer.getvalue()


def thumbnail_cube(fid: str, starless: bool) -> np.ndarray:
    """The cheapest representative cube of a candidate field: the baked gate
    (else mean) of a test field, the stamped LR of a blackout field, the
    stored production SR (else the LR) of a real tile."""
    kind, ref = parse_field_id(fid)
    regime = Path(ev._regime_dir_ro(starless))
    if kind == "test":
        for name in (f"{GATE_PREFIX}_{ref}.npy", f"sr_{ref}.npy", f"lr_{ref}.npy"):
            if (regime / "cubes" / name).is_file():
                return np.load(regime / "cubes" / name)
        raise FileNotFoundError(f"no cached cube for {fid}")
    if kind == "blackout":
        path = regime / "cubes_blackout" / f"lr_{ref}.npy"
        if not path.is_file():
            raise FileNotFoundError(f"no stamped LR for {fid}")
        return np.load(path)
    source, _, identifier = ref.partition("/")
    entry = real_tiles.get_entry(source, identifier)
    try:
        cube, _header, _meta = real_tiles.load_output(entry, model_catalog.SPEC_PRODUCTION)
        return cube
    except FileNotFoundError:
        return real_tiles.get_tile(source, identifier, entry=entry).lr_e


def thumbnail(fid: str, starless: bool, side: int = _THUMB_DEFAULT) -> bytes:
    return render_thumbnail(thumbnail_cube(fid, starless), side)


__all__ = [
    "KINDS",
    "active_labels",
    "candidates",
    "ensemble_summary",
    "field_id",
    "matching_experiments",
    "missing_real_members",
    "parse_field_id",
    "production_gate",
    "regime_slug",
    "render_thumbnail",
    "test_cubes_state",
    "thumbnail",
    "thumbnail_cube",
    "tile_lr_sha",
]
