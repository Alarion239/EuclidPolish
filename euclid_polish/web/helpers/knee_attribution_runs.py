"""Knee-attribution runs: which input knees a multi-knee model relies on, for
which pixels, over synthetic test fields and real tiles (spec
``docs/superpowers/specs/2026-10-15-knee-attribution-design.md``).

A run is a local job over models × fields. Per field it predicts the model's
SR, classes every HR pixel (truth classes on synthetic fields — star/galaxy
cores, wings, outskirts, saturated, sky — and predicted-brightness bins on real
tiles), samples pixels per class and band, and splits each sampled pixel's
sensitivity over the knees (:mod:`euclid_polish.eval.knee_attribution`),
each input band weighted by its noise σ. The record (``<root>/<id>.json``)
keeps every sample and the per-model aggregates over all fields (the
per-input-band knee split only in the aggregates); each field's SR and LR go
to ``<root>/fields/<id>/`` for the console's field explorer.
"""

from __future__ import annotations

import io
import json
import os
import re
import uuid
import zlib
from collections.abc import Iterator, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from euclid_polish import ensemble_registry
from euclid_polish.config import Config
from euclid_polish.ensemble import EnsembleModel
from euclid_polish.eval import knee_attribution as ka
from euclid_polish.image.collection import ImageSet
from euclid_polish.image.tfio import tfrecord_path
from euclid_polish.training.inference import infer_checkpoint_asinh_knees
from euclid_polish.training.target_blur import blur_target_array
from euclid_polish.web.helpers import atomic_files, milestones, model_catalog, real_tiles
from euclid_polish.web.helpers import ensemble_viz as ev
from euclid_polish.web.jobs import JobCancelled

#: 5: noise σ from lag-4 differences (the per-pixel σ of resampled NISP,
#: 2.6× the old lag-1 estimate); 4: uniform sampling, per-sample weights and
#: own-band knee splits, the synthetic truth in the field images, one
#: sampling stream per field; 3: adds the local (3×3 LR) splits;
#: 2: shares weighted by each input band's noise σ (1 weighted per electron).
RECORD_VERSION = 5
_RUN_ID = re.compile(r"^[0-9]{8}-[0-9]{6}-[0-9a-f]{6}$")
_SLUG = re.compile(r"[^A-Za-z0-9._-]+")
#: Most synthetic test fields a run takes (the universal scores' 100).
MAX_SYNTHETIC = 100
MAX_PER_CLASS = 256
MAX_PER_BAND = 8192
#: How a field's pixels are sampled: per class (``per_class`` of each class
#: present), uniformly (``per_band`` pixels per band, classes as common as
#: they are) or both pooled (``mixed``, weights halved).
SAMPLINGS = ("class", "uniform", "mixed")
#: Models a run accepts: multi-knee members, served as universal or member specs.
_KINDS = ("universal", "member", "milestone")


def root() -> Path:
    """``<inference>/knee_attribution`` (next to the experiments)."""
    return model_catalog.experiments_root().parent / "knee_attribution"


def new_run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]


def check_run_id(identifier: str) -> str:
    if not _RUN_ID.fullmatch(str(identifier or "")):
        raise KeyError(f"unknown knee-attribution run {identifier!r}")
    return str(identifier)


def slug(text: str) -> str:
    """A path-safe name for a model spec or a field ref."""
    return _SLUG.sub("__", str(text)).strip("_") or "x"


def _record_path(run_id: str) -> Path:
    return root() / f"{check_run_id(run_id)}.json"


def _field_path(run_id: str, model_spec: str, ref: str) -> Path:
    return root() / "fields" / check_run_id(run_id) / slug(model_spec) / f"{slug(ref)}.npz"


def _write(record: Mapping[str, Any]) -> None:
    atomic_files.write_json(_record_path(str(record["id"])), record, allow_nan=False)


def get_run(run_id: str) -> dict[str, Any]:
    """One run record (:class:`KeyError` when unknown)."""
    try:
        return json.loads(_record_path(run_id).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise KeyError(f"unknown knee-attribution run {run_id!r}") from exc


#: A record's summary by path, kept while its (mtime, size) holds: a mixed
#: run's record is tens of MB, so the list parses each one once.
_SUMMARIES: dict[Path, tuple[tuple[int, int], dict[str, Any]]] = {}


def _summary(path: Path) -> dict[str, Any] | None:
    try:
        stat = path.stat()
        key = (stat.st_mtime_ns, stat.st_size)
        hit = _SUMMARIES.get(path)
        if hit and hit[0] == key:
            return hit[1]
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    params = record.get("params") or {}
    row = {"id": record.get("id"), "created": record.get("created"),
           "finished": record.get("finished"), "status": record.get("status"),
           "models": params.get("models", []), "synthetic": params.get("synthetic", 0),
           "tiles": params.get("tiles", []), "per_class": params.get("per_class"),
           "sampling": params.get("sampling", "class"), "per_band": params.get("per_band"),
           "seed": params.get("seed"), "version": record.get("version")}
    _SUMMARIES[path] = (key, row)
    return row


def list_runs(limit: int = 200) -> list[dict[str, Any]]:
    """Run summaries, newest first (no samples; each record parsed once
    while it is unchanged)."""
    folder = root()
    if not folder.is_dir():
        return []
    rows = (_summary(path) for path in sorted(folder.glob("*.json"), reverse=True)[:limit])
    return [dict(row) for row in rows if row is not None]


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #

def candidate_models() -> list[dict[str, Any]]:
    """The catalogue's universal, member and milestone specs with their
    knees: available when the catalogue runs them and they read several
    knees (a milestone reads its member's knees)."""
    base = ev.ensemble_dir()
    out = []
    for item in model_catalog.list_specs():
        if item.kind not in _KINDS or len(item.reads) != 1:
            continue
        if item.kind == "milestone":
            member, _ = model_catalog.milestone_of(item.reads[0])
            knees = infer_checkpoint_asinh_knees(str(milestones.local_root() / member))
        else:
            name = ensemble_registry.member_name(item.reads[0])
            knees = infer_checkpoint_asinh_knees(os.path.join(base, name))
        details = dict(item.details or {})
        reason = item.reason
        if item.available and (not knees or len(knees) < 2):
            reason = "a single-knee model (nothing to split over knees)"
        out.append({"spec": item.spec, "label": item.label, "kind": item.kind,
                    "knees": [float(q) for q in knees] if knees else [],
                    "depth": details.get("num_res_blocks"),
                    "available": bool(item.available and knees and len(knees) >= 2),
                    "reason": None if (item.available and knees and len(knees) >= 2) else reason})
    return out


def _load(spec: str):
    """``(KneeNet, member Model)`` of one spec (the ensemble-only rule: a
    member loads as an ensemble of one)."""
    item = model_catalog.resolve_spec(spec)
    if item.kind not in _KINDS or len(item.reads) != 1:
        raise ValueError(f"{spec}: not a single multi-knee model")
    if item.kind == "milestone":                            # its own ensemble dir of one
        member_name, step = model_catalog.milestone_of(item.reads[0])
        model = EnsembleModel(milestones.materialize(member_name, step), labels=[member_name],
                              include_universal=True, starless=False)
    else:
        model = EnsembleModel(ev.ensemble_dir(), labels=[item.reads[0]],
                              include_universal=True, starless=False)
    member = model.members[0]
    return ka.KneeNet.from_member(member), member


# --------------------------------------------------------------------------- #
# Fields
# --------------------------------------------------------------------------- #

def synthetic_indices(n: int) -> list[int]:
    """The first ``n`` of the universal scores' test fields (else 0…n−1)."""
    try:
        identity = json.loads(Path(ev.universal_scores_dir(), "index.json")
                              .read_text(encoding="utf-8")).get("identity") or {}
    except (OSError, ValueError):
        identity = {}
    indices = [int(i) for i in identity.get("indices") or []] or list(range(MAX_SYNTHETIC))
    return indices[:max(0, int(n))]


def _synthetic_fields(indices: Sequence[int]
                      ) -> Iterator[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    """``(ref, lr_e, galaxy_e, star_e)`` per wanted test field, the truth
    components blurred to the target PSF: galaxy = the starless clean scene,
    star = the starfull HR target minus it (records in index order)."""
    if not indices:
        return
    records_dir = ev._sky_records_local_dir()
    last = max(indices) + 1
    streams = [iter(ImageSet.read(tfrecord_path(records_dir, kind), num_images=last))
               for kind in ("dirty_test", "hr_test", "clean_test")]
    heads = [next(s, None) for s in streams]
    fwhm = float(Config.TARGET_PSF_FWHM_ARCSEC)
    for index in sorted({int(i) for i in indices}):
        for k, stream in enumerate(streams):
            while heads[k] is not None and (heads[k].index or 0) < index:
                heads[k] = next(stream, None)
        if any(h is None or h.index != index for h in heads):
            continue
        lr, hr, clean = heads
        hr_e = np.asarray(hr.data, np.float32)
        clean_e = np.asarray(clean.data, np.float32)
        h = min(hr_e.shape[0], clean_e.shape[0])
        w = min(hr_e.shape[1], clean_e.shape[1])
        hr_e, clean_e = hr_e[:h, :w], clean_e[:h, :w]
        galaxy = np.asarray(blur_target_array(clean_e, fwhm,
                                              pixel_scale_arcsec=hr.pixel_scale_arcsec), np.float32)
        star = np.asarray(blur_target_array(np.clip(hr_e - clean_e, 0.0, None), fwhm,
                                            pixel_scale_arcsec=hr.pixel_scale_arcsec), np.float32)
        yield f"test/{index}", np.asarray(lr.data, np.float32), galaxy, star


def _real_lr(ref: str) -> np.ndarray:
    source, _, identifier = ref.partition("/")
    tile = real_tiles.get_tile(source, identifier)
    if not tile.entry.model_ready:
        raise ValueError(f"{ref}: not a 4-band tile")
    return np.asarray(tile.lr_e, np.float32)


# --------------------------------------------------------------------------- #
# Request
# --------------------------------------------------------------------------- #

def parse_request(form: Mapping[str, Any]) -> dict[str, Any]:
    """Validated run parameters from the POST form (:class:`ValueError`)."""
    specs = model_catalog.parse_specs(str(form.get("models", "") or ""))
    if not specs:
        raise ValueError("name at least one multi-knee model (models)")
    usable = {m["spec"]: m for m in candidate_models()}
    for spec in specs:
        row = usable.get(spec)
        if row is None:
            raise ValueError(f"{spec}: not a universal or member model")
        if not row["available"]:
            raise ValueError(f"{spec}: {row['reason'] or 'unavailable'}")

    def number(key: str, default: int, lo: int, hi: int) -> int:
        raw = str(form.get(key, "") or "").strip()
        try:
            value = int(raw) if raw else default
        except ValueError as exc:
            raise ValueError(f"{key} must be a whole number") from exc
        if not lo <= value <= hi:
            raise ValueError(f"{key} must lie in [{lo}, {hi}]")
        return value

    synthetic = number("synthetic", 20, 0, MAX_SYNTHETIC)
    per_class = number("per_class", 24, 1, MAX_PER_CLASS)
    per_band = number("per_band", 512, 1, MAX_PER_BAND)
    sampling = str(form.get("sampling", "") or "class").strip().lower()
    if sampling not in SAMPLINGS:
        raise ValueError(f"sampling must be one of {', '.join(SAMPLINGS)}")
    seed = number("seed", 0, 0, 2 ** 31 - 1)
    tiles = [f"{s}/{i}" for s, i in real_tiles.parse_refs(str(form.get("tiles", "") or ""))]
    for ref in tiles:
        source, _, identifier = ref.partition("/")
        try:
            real_tiles.get_entry(source, identifier)
        except Exception as exc:  # noqa: BLE001 — an unknown tile is a 400
            raise ValueError(f"{ref}: {exc}") from exc
    if synthetic == 0 and not tiles:
        raise ValueError("pick synthetic test fields or real tiles")
    ablation = str(form.get("ablation", "1")).strip().lower() not in ("0", "false", "no")
    return {"models": specs, "synthetic": synthetic, "tiles": tiles,
            "per_class": per_class, "seed": seed, "ablation": ablation,
            "sampling": sampling, "per_band": per_band}


# --------------------------------------------------------------------------- #
# The job
# --------------------------------------------------------------------------- #

def field_rng(seed: int, ref: str) -> np.random.Generator:
    """One field's sampling stream, from the seed and the field alone: every
    model of a run (and any run with the same seed) draws the same uniform
    pixels of a field, so models can be compared pixel by pixel."""
    return np.random.default_rng([int(seed), zlib.crc32(str(ref).encode("utf-8"))])


def _round(values: np.ndarray, digits: int) -> list:
    return np.round(np.asarray(values, np.float64), digits).tolist()


def _field_samples(net, lr_e: np.ndarray, codes: np.ndarray, values: np.ndarray,
                   class_names: Sequence[str], *, per_class: int, rng: np.random.Generator,
                   ablation: bool, sigma_e: np.ndarray, sampling: str = "class",
                   per_band: int = 512) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """One field's samples: ``(JSON row samples, numpy part for aggregate)``.
    The per-input knee splits (C × K per sample) and the local band shares
    only go to the part: they would multiply the record's size. Each row
    keeps its own band's knee split (``knee_share_own`` and
    ``knee_share_own_local``: the split of the input band it predicts, the
    one comparable across retrains) and its ``weight`` (field pixels it
    stands for)."""
    if sampling == "uniform":
        picked = ka.sample_uniform(codes, per_band, rng)
    elif sampling == "mixed":
        picked = ka.sample_mixed(codes, per_class, per_band, rng)
    else:
        picked = ka.sample_pixels(codes, per_class, rng)
    shares = ka.attribute(net, lr_e, picked, ablation=ablation, sigma_e=sigma_e)
    band, y, x = picked["band"], picked["y"], picked["x"]
    value_e = values[y, x, band].astype(np.float64)
    rows = np.arange(band.size)
    own = shares["knee_share_by_input"][rows, band]
    own_local = shares["knee_share_by_input_local"][rows, band]
    part = {"band": band, "code": picked["code"], "value_e": value_e,
            "knee_share": shares["knee_share"], "band_share": shares["band_share"],
            "knee_share_by_input": shares["knee_share_by_input"],
            "knee_share_local": shares["knee_share_local"],
            "band_share_local": shares["band_share_local"],
            "knee_share_by_input_local": shares["knee_share_by_input_local"],
            "ablation_share": shares.get("ablation_share",
                                         np.zeros_like(shares["knee_share"]))}
    row = {"band": band.astype(int).tolist(), "y": y.astype(int).tolist(),
           "x": x.astype(int).tolist(),
           "class": [class_names[int(c)] for c in picked["code"]],
           "value_e": _round(value_e, 4),
           "weight": _round(picked["weight"], 3),
           "knee_share": _round(shares["knee_share"], 4),
           "knee_share_own": _round(own, 4),
           "knee_share_own_local": _round(own_local, 4),
           "knee_share_local": _round(shares["knee_share_local"], 4),
           "ablation_share": _round(part["ablation_share"], 4),
           "band_share": _round(shares["band_share"], 4),
           "sensitivity": _round(shares["sensitivity"], 6)}
    return row, part


def _save_field(run_id: str, spec: str, ref: str, sr_e: np.ndarray, lr_e: np.ndarray,
                truth_e: np.ndarray | None = None) -> str:
    """The field's SR and LR (and a synthetic field's truth, for the pixel
    types' stamps) as float16 display copies."""
    path = _field_path(run_id, spec, ref)
    path.parent.mkdir(parents=True, exist_ok=True)
    limit = float(np.finfo(np.float16).max)                 # display copies: clip, never inf
    planes = {"sr": sr_e, "lr": lr_e} | ({} if truth_e is None else {"truth": truth_e})
    with atomic_files.temporary_sibling(path, suffix=".tmp.npz") as tmp:
        np.savez_compressed(tmp, **{k: np.clip(v, -limit, limit).astype(np.float16)
                                    for k, v in planes.items()})
        os.replace(tmp, path)
    return (f"/api/knee-attribution/{run_id}/field/{slug(spec)}/{slug(ref)}.png")


def _one_field(run_id: str, spec: str, net, member, ref: str, kind: str, lr_e: np.ndarray, *,
               galaxy: np.ndarray | None = None, star: np.ndarray | None = None,
               per_class: int, rng: np.random.Generator, ablation: bool,
               sampling: str = "class", per_band: int = 512
               ) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """One field of one model: ``(the record's field row, the aggregate
    part)``. Synthetic fields class by truth (``galaxy``/``star``), real ones
    by the predicted image."""
    # One clean LR for the SR, the classes and the attribution alike (a
    # non-finite pixel would turn the SR around it into NaN).
    lr_e = np.nan_to_num(np.asarray(lr_e, np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    sr_e = np.asarray(member.upsample_array(lr_e), np.float32)
    truth = None
    if kind == "synthetic":
        h, w = sr_e.shape[:2]
        codes, values = ka.truth_classes(galaxy[:h, :w], star[:h, :w], lr_e)
        names = ka.TRUTH_CLASSES
        truth = values
    else:
        codes, values = ka.real_classes(sr_e, lr_e)
        names = ka.REAL_CLASSES
    sigma_e = ka.noise_sigma(lr_e)
    row, part = _field_samples(net, lr_e, codes, values, names, per_class=per_class,
                               rng=rng, ablation=ablation, sigma_e=sigma_e,
                               sampling=sampling, per_band=per_band)
    image = _save_field(run_id, spec, ref, sr_e, lr_e, truth)
    return ({"ref": ref, "kind": kind, "shape": list(sr_e.shape[:2]), "image": image,
             "noise_sigma_e": _round(sigma_e, 4), "samples": row}, part)


def job_knee_attribution(cap, *, run_id: str, models: Sequence[str], synthetic: int,
                         tiles: Sequence[str], per_class: int, seed: int,
                         ablation: bool = True, sampling: str = "class",
                         per_band: int = 512) -> dict[str, Any]:
    """The job: every model over every field, the record rewritten after each
    model; ``{run_id, status, errors}``. A cancelled run keeps the models it
    finished, status ``cancelled``."""
    record: dict[str, Any] = {
        "version": RECORD_VERSION, "id": run_id,
        "created": datetime.now(UTC).isoformat(), "finished": None, "status": "running",
        "params": {"models": list(models), "synthetic": int(synthetic), "tiles": list(tiles),
                   "per_class": int(per_class), "seed": int(seed), "ablation": bool(ablation),
                   "sampling": str(sampling), "per_band": int(per_band)},
        "models": {}, "errors": {}}
    _write(record)
    try:
        _run_models(cap, record, run_id, models, synthetic_indices(synthetic), tiles,
                    per_class=per_class, seed=seed, ablation=ablation, sampling=sampling,
                    per_band=per_band)
    except (JobCancelled, KeyboardInterrupt):
        record["status"] = "cancelled"
        raise
    else:
        record["status"] = "done" if record["models"] else "failed"
    finally:
        record["finished"] = datetime.now(UTC).isoformat()
        _write(record)
    return {"run_id": run_id, "status": record["status"], "errors": record["errors"]}


def _run_models(cap, record: dict[str, Any], run_id: str, models: Sequence[str],
                indices: Sequence[int], tiles: Sequence[str], *, per_class: int, seed: int,
                ablation: bool, sampling: str = "class", per_band: int = 512) -> None:
    """Every model over every field into ``record`` (rewritten after each model)."""
    n_fields = len(indices) + len(tiles)
    total = max(1, len(models) * n_fields)
    done = 0
    for spec in models:
        try:
            net, member = _load(spec)
        except Exception as exc:  # noqa: BLE001 — one model never stops the others
            record["errors"][spec] = f"{type(exc).__name__}: {exc}"
            done += n_fields
            continue
        entry: dict[str, Any] = {"knees": list(net.knees), "bands": list(Config.LR_INPUT_BAND_NAMES),
                                 "depth": net.depth, "aggregate": {}, "fields": []}
        parts: dict[str, list[dict[str, np.ndarray]]] = {"synthetic": [], "real": []}

        for ref, lr_e, galaxy, star in _synthetic_fields(indices):
            cap.check_cancelled()
            cap.tick(done, total, f"{spec}: {ref}")
            try:
                field, part = _one_field(run_id, spec, net, member, ref, "synthetic", lr_e,
                                         galaxy=galaxy, star=star, per_class=per_class,
                                         rng=field_rng(seed, ref), ablation=ablation, sampling=sampling,
                                         per_band=per_band)
                entry["fields"].append(field)
                parts["synthetic"].append(part)
            except Exception as exc:  # noqa: BLE001 — one field never stops the run
                record["errors"][f"{spec}|{ref}"] = f"{type(exc).__name__}: {exc}"
            done += 1
        for ref in tiles:
            cap.check_cancelled()
            cap.tick(done, total, f"{spec}: {ref}")
            try:
                field, part = _one_field(run_id, spec, net, member, ref, "real", _real_lr(ref),
                                         per_class=per_class, rng=field_rng(seed, ref),
                                         ablation=ablation, sampling=sampling, per_band=per_band)
                entry["fields"].append(field)
                parts["real"].append(part)
            except Exception as exc:  # noqa: BLE001
                record["errors"][f"{spec}|{ref}"] = f"{type(exc).__name__}: {exc}"
            done += 1
        n_bands = len(entry["bands"])
        if parts["synthetic"]:
            entry["aggregate"]["synthetic"] = ka.aggregate(
                parts["synthetic"], ka.TRUTH_CLASSES, n_bands, net.knees)
        if parts["real"]:
            entry["aggregate"]["real"] = ka.aggregate(parts["real"], ka.REAL_CLASSES, n_bands, net.knees)
        record["models"][spec] = entry
        _write(record)
    cap.tick(total, total, "done")


# --------------------------------------------------------------------------- #
# Field images
# --------------------------------------------------------------------------- #

def field_png(run_id: str, model_slug: str, ref_slug: str, *, band: str = "VIS",
              layer: str = "sr") -> bytes:
    """A stored field as an 8-bit asinh-scaled PNG (rows as stored, so the
    samples' ``(y, x)`` index it directly; LR shown at its own size)."""
    path = root() / "fields" / check_run_id(run_id) / slug(model_slug) / f"{slug(ref_slug)}.npz"
    if not path.is_file():
        raise KeyError("no such field image")
    bands = list(Config.LR_INPUT_BAND_NAMES)
    if band not in bands:
        raise ValueError(f"band must be one of {bands}")
    if layer not in ("sr", "lr"):
        raise ValueError("layer must be sr or lr")
    with np.load(path) as stored:
        plane = np.asarray(stored[layer][..., bands.index(band)], np.float32)
    plane = np.nan_to_num(plane, nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = np.percentile(plane, [1.0, 99.7])
    scale = max(float(hi - lo), 1e-6)
    stretched = np.arcsinh(np.clip(plane - lo, 0.0, None) / (0.02 * scale)) / np.arcsinh(50.0)
    img8 = np.clip(stretched * 255.0, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(img8, mode="L").save(buf, format="PNG")
    return buf.getvalue()
