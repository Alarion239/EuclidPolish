"""Shared pieces of the model battery: model specs resolved once, one model in
memory at a time, the SR cache every test reads, the per-model result store and
the inputs (the NEXUS tiles, the synthetic test fields, the poster).

The SR cache keeps one ``.npy`` per (model checkpoint, input): the file name
carries a digest of the input array, so two tests that feed a model the same
tile share one inference, and a tile fed differently (another input) never
reads a stale output. Models are keyed by checkpoint (member, milestone step and
:func:`~euclid_polish.ensemble.member_fingerprint`), not by spec string:
``production`` and ``universal:member_NNN`` share outputs when they name the
same checkpoint, and a re-pointed ``production`` never reads the old one's.
"""
from __future__ import annotations

import gc
import glob
import hashlib
import json
import os
import re
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import tensorflow as tf

from euclid_polish.config import Config
from euclid_polish.ensemble import EnsembleModel, member_fingerprint
from euclid_polish.web.helpers import ensemble_viz as ev
from euclid_polish.web.helpers import milestones, model_catalog
from euclid_polish.web.helpers import real_tiles as rt
from euclid_polish.web.helpers import universal_models as um

#: Bump when a measurement changes: stored results of another version are redone.
VERSION = 1
BANDS = tuple(Config.LR_INPUT_BAND_NAMES)              # (VIS, Y_E, J_E, H_E)
SHORT = ("VIS", "Y", "J", "H")
JWST_OVERLAP = os.path.join(Config.DATA_DIR, "jwst_euclid_overlap")
#: Per-tile Euclid input + JWST on the SR grid (``scripts/jwst_resolution.py``).
RESOLUTION_CACHE = os.path.join(JWST_OVERLAP, "resolution_cache")
MER_CATALOGUE = os.path.join(JWST_OVERLAP, "flux_conservation", "mer_nexus.fits")
EXPERIMENT_RECORDS = os.path.join(Config.EUCLID_INFERENCE_DIR, "experiments", "records")


# --------------------------------------------------------------------------- #
# models
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ModelInfo:
    """One model of the battery, resolved from its spec."""

    spec: str           # canonical spec, as reports key it
    slug: str           # path-safe spec
    name: str           # short display name ("238", "257@15M", "production (257)")
    kind: str           # production | universal | member | milestone
    label: str          # the ensemble label it loads under
    member: str         # member_NNN
    step: int | None    # milestone step
    member_dir: str     # checkpoint dir
    fingerprint: str    # member_fingerprint of the checkpoint

    @property
    def cache_id(self) -> str:
        """Checkpoint identity of the SR cache (shared by specs naming it)."""
        digest = hashlib.sha1(self.fingerprint.encode()).hexdigest()[:8]
        return f"{self.member}{'' if self.step is None else f'@{self.step}'}-{digest}"


def step_text(step: int) -> str:
    """``15000000`` → ``15M``, ``200000`` → ``200k``."""
    return f"{step / 1e6:g}M" if step >= 1_000_000 else f"{step / 1e3:g}k"


def resolve(spec: str) -> ModelInfo:
    """A runnable single model (production, ``universal:``, ``member:`` or
    ``milestone:``); :class:`ValueError` for a combiner or an unavailable one."""
    canonical = model_catalog.canonical_spec(spec)
    slug = model_catalog.spec_slug(canonical)
    if canonical.startswith("milestone:"):
        member, step = milestones.parse_spec(canonical)
        member_dir = os.path.join(milestones.materialize(member, step), member)
        return ModelInfo(canonical, slug, f"{member.removeprefix('member_')}@{step_text(step)}",
                         "milestone", member, member, step, member_dir, member_fingerprint(member_dir))
    item = model_catalog.resolve_spec(canonical)
    if not item.available:
        raise ValueError(f"{canonical}: not available ({item.reason})")
    if len(item.reads) != 1:
        raise ValueError(f"{canonical}: a combiner of {len(item.reads)} models; "
                         "the battery runs single models")
    label = item.reads[0]
    member = model_catalog.member_name(label)
    member_dir = os.path.join(model_catalog.ensemble_dir(), member)
    number = member.removeprefix("member_")
    kind = model_catalog.spec_kind(canonical)
    name = f"production ({number})" if canonical == "production" else number
    return ModelInfo(canonical, slug, name, kind, label, member, None, member_dir,
                     member_fingerprint(member_dir))


class ModelRunner:
    """Runs one model at a time: asking for another drops the previous one
    first (16 GB laptops: one TF model in memory)."""

    def __init__(self) -> None:
        self._info: ModelInfo | None = None
        self._model: EnsembleModel | None = None

    def model(self, info: ModelInfo) -> EnsembleModel:
        if self._info != info:
            self.release()
            if info.kind == "milestone":
                self._model = milestones.load(info.member, info.step)
            else:
                self._model = EnsembleModel(model_catalog.ensemble_dir(), labels=[info.label],
                                            include_universal=True, starless=False)
            self._info = info
        return self._model

    def release(self) -> None:
        if self._model is not None:
            self._model, self._info = None, None
            tf.keras.backend.clear_session()
            gc.collect()

    def sr(self, info: ModelInfo, lr: np.ndarray) -> np.ndarray:
        """The model's SR of ``lr`` (electrons, ``(2h, 2w, 4)``), as every
        console path computes it (``member_arrays`` of an ensemble of one)."""
        stack = self.model(info).member_arrays(np.ascontiguousarray(lr, np.float32))
        return np.asarray(stack[0], np.float32)


# --------------------------------------------------------------------------- #
# the SR cache
# --------------------------------------------------------------------------- #

def input_digest(lr: np.ndarray) -> str:
    """12 hex digits naming an input array (its float32 bytes and shape)."""
    arr = np.ascontiguousarray(lr, np.float32)
    h = hashlib.sha1(repr(arr.shape).encode())
    h.update(arr.tobytes())
    return h.hexdigest()[:12]


def _atomic_save(path: str, array: np.ndarray) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as fh:
        np.save(fh, array)
    os.replace(tmp, path)


class SRCache:
    """``<root>/<model cache id>/<key>__<input digest>.npy``; computed on the
    first request (through ``runner``), read back after."""

    def __init__(self, root: str, runner: Callable[[ModelInfo, np.ndarray], np.ndarray]) -> None:
        self.root = root
        self.runner = runner
        self.computed = 0
        self.read = 0

    def folder(self, info: ModelInfo) -> str:
        return os.path.join(self.root, info.cache_id)

    def path(self, info: ModelInfo, key: str, lr: np.ndarray) -> str:
        return os.path.join(self.folder(info), f"{key}__{input_digest(lr)}.npy")

    def get(self, info: ModelInfo, key: str, lr: np.ndarray) -> np.ndarray:
        path = self.path(info, key, lr)
        if os.path.isfile(path):
            self.read += 1
            return np.load(path)
        sr = np.asarray(self.runner(info, lr), np.float32)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        _atomic_save(path, sr)
        self.computed += 1
        return sr

    def keys(self, info: ModelInfo) -> list[str]:
        return sorted(os.path.basename(p).rsplit("__", 1)[0]
                      for p in glob.glob(os.path.join(self.folder(info), "*__*.npy")))

    def prune(self, info: ModelInfo, keep: Callable[[str], bool]) -> int:
        """Delete the model's outputs whose key ``keep`` rejects; bytes freed."""
        freed = 0
        for path in glob.glob(os.path.join(self.folder(info), "*__*.npy")):
            if not keep(os.path.basename(path).rsplit("__", 1)[0]):
                freed += os.path.getsize(path)
                os.remove(path)
        return freed


def nexus_key(tile_id: str) -> str:
    return f"nexus-{tile_id}"


def field_key(index: int) -> str:
    return f"syn-{int(index):03d}"


def tile_key(ref: str) -> str:
    """``poster/target_181255`` → ``poster-target_181255``."""
    return ref.replace("/", "-")


# --------------------------------------------------------------------------- #
# the result store
# --------------------------------------------------------------------------- #

def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating | float):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(_jsonable(payload), fh, indent=1, allow_nan=False)
    os.replace(tmp, path)


#: Stages measured on the synthetic test records (their stamp names the records).
RECORD_STAGES = frozenset({"synthetic", "grid", "grid_prep", "grid_ref_lr_syn", "grid_ref_truth_syn",
                           "grid_ref_white_syn", "synflux_sources", "synflux", "outskirts_deck",
                           "outskirts_hosts", "growth_syn_geometry", "growth_syn"})


def depends_on_records(stage: str) -> bool:
    return stage in RECORD_STAGES or stage.startswith("outskirt_rc_")


class Store:
    """Stage results under ``<out>/models/<slug>/<stage>.json|.npz`` (one model)
    and ``<out>/shared/<stage>.json|.npz`` (no model). A stage counts as done
    while its JSON names the current battery :data:`VERSION`, the limits it ran
    under, (per model) the checkpoint fingerprint and (for a stage measured on
    the synthetic test records) the records' fingerprint — re-synced records
    (another noise model) are measured afresh."""

    def __init__(self, root: str, limits_tag: str = "", records_fp: str | None = "") -> None:
        self.root = root
        self.limits_tag = limits_tag
        self.records_fp = records_fp

    def folder(self, info: ModelInfo | None) -> str:
        return os.path.join(self.root, "shared" if info is None else os.path.join("models", info.slug))

    def _stamp(self, info: ModelInfo | None, stage: str) -> dict:
        return {"version": VERSION, "limits": self.limits_tag,
                **({"records": self.records_fp} if depends_on_records(stage) else {}),
                **({} if info is None else {"spec": info.spec, "fingerprint": info.fingerprint})}

    def read(self, info: ModelInfo | None, stage: str) -> dict | None:
        path = os.path.join(self.folder(info), f"{stage}.json")
        try:
            with open(path, encoding="utf-8") as fh:
                payload = json.load(fh)
        except (OSError, ValueError):
            return None
        if payload.get("_battery") != self._stamp(info, stage):
            return None
        if os.path.isfile(self.npz_path(info, stage)) or not payload.get("_arrays"):
            return payload
        return None

    def npz_path(self, info: ModelInfo | None, stage: str) -> str:
        return os.path.join(self.folder(info), f"{stage}.npz")

    def write(self, info: ModelInfo | None, stage: str, payload: Mapping[str, Any],
              arrays: Mapping[str, np.ndarray] | None = None) -> dict:
        os.makedirs(self.folder(info), exist_ok=True)
        if arrays:
            tmp = self.npz_path(info, stage) + ".tmp"
            with open(tmp, "wb") as fh:
                np.savez(fh, **arrays)
            os.replace(tmp, self.npz_path(info, stage))
        out = {**payload, "_battery": self._stamp(info, stage), "_arrays": bool(arrays),
               "_written": time.strftime("%Y-%m-%dT%H:%M:%S")}
        write_json(os.path.join(self.folder(info), f"{stage}.json"), out)
        return json.loads(json.dumps(_jsonable(out)))

    def arrays(self, info: ModelInfo | None, stage: str) -> dict[str, np.ndarray]:
        with np.load(self.npz_path(info, stage), allow_pickle=False) as data:
            return {k: data[k] for k in data.files}


# --------------------------------------------------------------------------- #
# inputs
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Limits:
    """Caps for a smoke run (``None`` = the full battery). Capped results are
    not comparable with full ones; the store keeps them apart."""

    tiles: int | None = None        # NEXUS tiles (resolution, grid, flux)
    fields: int | None = None       # synthetic test fields
    hosts: int | None = None        # outskirt injection fields (40)
    grad_tiles: int | None = None   # outskirt gain-profile tiles (45)

    @property
    def tag(self) -> str:
        values = {k: v for k, v in asdict(self).items() if v is not None}
        return ",".join(f"{k}={v}" for k, v in values.items())

    @classmethod
    def smoke(cls) -> Limits:
        return cls(tiles=6, fields=3, hosts=2, grad_tiles=1)


def nexus_entries(limit: int | None = None) -> list:
    """The NEXUS tiles with a JWST image (``scripts/jwst_resolution.py``)."""
    return [e for e in rt.list_entries("nexus") if e.extras.get("jwst_file")][:limit]


def nexus_lr(entry) -> np.ndarray:
    """A NEXUS tile's Euclid input ``(255, 255, 4)`` e⁻: the resolution cache's
    copy when it is there (bit-identical to ``real_tiles``)."""
    path = os.path.join(RESOLUTION_CACHE, f"{entry.id}.npz")
    if os.path.isfile(path):
        with np.load(path) as data:
            return np.asarray(data["lr"], np.float32)
    return np.asarray(rt.get_tile("nexus", entry.id, entry=entry).lr_e, np.float32)


def score_identity() -> dict:
    """The Leaderboard's test fields, records and target (the console score's)."""
    return ev.universal_score_identity()


def records_dir() -> str:
    return ev._sky_records_local_dir()


def test_fields(limit: int | None = None) -> Iterator[tuple[int, np.ndarray, object]]:
    """``(index, LR (255, 255, 4), HR record)`` of the score's test fields."""
    indices = list(score_identity()["indices"])[:limit]
    yield from um._test_pairs(records_dir(), indices)


def target_fwhm() -> float:
    return float(score_identity()["target_psf_fwhm_arcsec"])


def field_truth(target, fwhm: float) -> np.ndarray:
    """The score's truth: the HR record blurred to the target PSF."""
    return um._field_truth(target, fwhm)


def experiment_tiles(record: str) -> list[str]:
    """The tile refs (``nexus/f200w-0070``) of a Sky › Compare record."""
    with open(os.path.join(EXPERIMENT_RECORDS, record), encoding="utf-8") as fh:
        return list(json.load(fh)["tiles"])


def real_tile_input(ref: str) -> tuple[np.ndarray, np.ndarray]:
    """``(lr, model input)`` of a real tile ref: the input zeroes non-finite
    pixels (Sky › Compare's experiment code)."""
    source, ident = ref.split("/", 1)
    entry = rt.get_entry(source, ident)
    lr = rt.get_tile(source, ident, entry=entry).lr_e
    return lr, np.where(np.isfinite(lr), lr, 0.0).astype(np.float32)


def short(spec_or_name: str) -> str:
    """Compact text of a spec for figure titles."""
    match = re.fullmatch(r"milestone:member_(\d+)@(\d+)", spec_or_name)
    if match:
        return f"{match.group(1)}@{step_text(int(match.group(2)))}"
    return spec_or_name.removeprefix("universal:").removeprefix("member:").removeprefix("member_")


# --------------------------------------------------------------------------- #
# the run context
# --------------------------------------------------------------------------- #

@dataclass
class Battery:
    """What every test reads: the result store, the SR cache, the model runner,
    the limits, a logger, and the run's options."""

    store: Store
    cache: SRCache
    runner: ModelRunner
    limits: Limits
    log: Callable[[str], None]
    rescore: bool = False           # measure the synthetic score even when the console's is current
    force: bool = False             # redo stored stages (each once per run)
    fresh: set = field(default_factory=set)

    def stage(self, info: ModelInfo | None, name: str,
              compute: Callable[[], tuple[dict, dict | None]]) -> dict:
        """A stage's stored result, or ``compute()``'s ``(payload, arrays)``
        written first."""
        key = (None if info is None else info.slug, name)
        if key in self.fresh or not self.force:
            stored = self.store.read(info, name)
            if stored is not None:
                return stored
        started = time.monotonic()
        payload, arrays = compute()
        out = self.store.write(info, name, {**payload, "seconds": round(time.monotonic() - started, 1)},
                               arrays)
        self.fresh.add(key)
        self.log(f"  {name}{'' if info is None else ' · ' + info.name}: "
                 f"{time.monotonic() - started:.0f} s")
        return out
