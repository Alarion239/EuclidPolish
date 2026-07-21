"""Production-parity ensemble evaluation for the isolated lens experiment.

The metric/cube contract intentionally matches ``ensemble_viz`` while every
input and output path is rooted below ``data/experiments/lens_isolation``.
"""

from __future__ import annotations

import base64
import glob
import json
import os
import shutil

import numpy as np

from euclid_polish.config import Config
from euclid_polish.ensemble import member_fingerprint
from euclid_polish.eval.combiner import RAW_INCREMENTAL_MINMEANMAX_RBF_KIND
from euclid_polish.eval.ensemble_diagnostics import EnsembleDiagnosticsAccumulator
from euclid_polish.eval.power_spectrum import EnsembleSpectrumAccumulator
from euclid_polish.experiments.lens_isolation.config import ExperimentPaths
from euclid_polish.experiments.lens_isolation.ensemble import LensIsolationEnsemble
from euclid_polish.image.tfio import read_images, tfrecord_path
from euclid_polish.model import _checkpoint_exists
from euclid_polish.training.inference import infer_checkpoint_num_res_blocks
from euclid_polish.web.helpers.ensemble_viz import (
    ENSEMBLE_PCA_COMPONENTS,
    ENSEMBLE_VIZ_FIELDS_MAX,
    MEMBER_PSNR_FIELDS,
    PIXEL_TRACE_HALF,
    PIXEL_TRACE_STAMPS,
    _cache_field_cubes,
    _CombinerMetricAcc,
    _evals_payload,
    _jsonable,
    _lr_cube_on_hr_grid,
    _lr_on_hr_grid,
    _vis,
)

REGIME = "lens-isolation"
TARGET_KIND = "lens"


def _paths() -> ExperimentPaths:
    return ExperimentPaths()


def ensemble_dir() -> str:
    return os.path.abspath(_paths().ensemble)


def records_dir() -> str:
    return os.path.abspath(_paths().records)


def output_dir() -> str:
    path = os.path.abspath(os.path.join(_paths().evaluation, "ensemble"))
    os.makedirs(path, exist_ok=True)
    return path


def cubes_dir(subset: str | None = None) -> str:
    return os.path.join(output_dir(), "cubes" if not subset else f"cubes_{subset}")


def _member_dirs() -> list[str]:
    return sorted(
        path for path in glob.glob(os.path.join(ensemble_dir(), "member_*"))
        if os.path.isdir(path) and _checkpoint_exists(path)
    )


def member_labels() -> list[str]:
    return [f"{os.path.basename(path).removeprefix('member_')}·psnr" for path in _member_dirs()]


def _read_json(path: str) -> dict | None:
    try:
        with open(path, encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else None
    except (OSError, ValueError, TypeError):
        return None


def _origin(path: str) -> dict:
    return _read_json(os.path.join(path, "origin.json")) or {}


def _last_step(path: str) -> int | None:
    log = os.path.join(path, "training_log.csv")
    try:
        with open(log, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            handle.seek(max(0, handle.tell() - 4096))
            lines = handle.read().decode(errors="replace").splitlines()
        return next((int(line.split(",", 1)[0]) for line in reversed(lines)
                     if line.split(",", 1)[0].isdigit()), None)
    except OSError:
        return None


def _record_fingerprint(subset: str) -> str | None:
    parts = []
    for kind in ("dirty", TARGET_KIND):
        path = tfrecord_path(records_dir(), f"{kind}_{subset}")
        try:
            stat = os.stat(path)
        except OSError:
            return None
        parts.append(f"{kind}:{stat.st_size}:{stat.st_mtime_ns}")
    return "|".join(parts)


def _combiner_fingerprint() -> str | None:
    try:
        stat = os.stat(os.path.join(output_dir(), "combiner", "combiner.npz"))
    except OSError:
        return None
    return f"{stat.st_size}:{stat.st_mtime_ns}"


def _identity(subset: str, num_images: int) -> dict:
    return {
        "records_fp": _record_fingerprint(subset),
        "subset": subset,
        "num_images": int(num_images),
        "regime": REGIME,
        "member_fps": [member_fingerprint(path) for path in _member_dirs()],
        "combiner_fp": _combiner_fingerprint(),
    }


def _summary() -> dict | None:
    return _read_json(os.path.join(output_dir(), "eval_summary.json"))


def _reusable(identity: dict) -> dict | None:
    summary = _summary()
    manifest = _read_json(os.path.join(cubes_dir(), "viz_index.json"))
    if not summary or not manifest:
        return None
    if summary.get("eval_identity") != identity:
        return None
    if manifest.get("records_fp") != identity.get("records_fp"):
        return None
    if list(manifest.get("member_labels") or []) != member_labels():
        return None
    return summary


def _split_status(subset: str) -> dict:
    metadata = _read_json(os.path.join(records_dir(), f"split_{subset}.json")) or {}
    files = {
        kind: os.path.isfile(tfrecord_path(records_dir(), f"{kind}_{subset}"))
        for kind in ("dirty", TARGET_KIND)
    }
    files["sources"] = os.path.isfile(os.path.join(records_dir(), f"sources_{subset}.csv"))
    return {"present": all(files.values()), "count": metadata.get("count"), "files": files}


def _member_meta(labels: list[str], summary: dict | None = None) -> list[dict]:
    psnr = {}
    if summary:
        psnr = dict(zip(
            [str(x) for x in summary.get("member_labels") or summary.get("per_member_labels") or []],
            summary.get("per_member_psnr_stretched") or [],
            strict=False,
        ))
    meta = []
    by_name = {os.path.basename(path): path for path in _member_dirs()}
    for label in labels:
        name = f"member_{str(label).split('·')[0]}"
        path = by_name.get(name, os.path.join(ensemble_dir(), name))
        origin = _origin(path)
        meta.append({
            "loss": origin.get("loss_norm") or "l1",
            "blocks": infer_checkpoint_num_res_blocks(path),
            "asinh_knee": origin.get("asinh_knee"),
            "step": _last_step(path),
            "psnr": psnr.get(label),
        })
    return meta


def status() -> dict:
    summary = _summary()
    labels = member_labels()
    stale = bool(summary) and list(
        summary.get("member_labels") or summary.get("per_member_labels") or []
    ) != labels
    psnr = {
        label: meta.get("psnr")
        for label, meta in zip(labels, _member_meta(labels, summary), strict=True)
    }
    members = []
    for rank, path in enumerate(_member_dirs(), 1):
        name = os.path.basename(path)
        label = f"{name.removeprefix('member_')}·psnr"
        origin = _origin(path)
        members.append({
            "name": name,
            "seed": origin.get("seed"),
            "step": _last_step(path),
            "blocks": infer_checkpoint_num_res_blocks(path),
            "loss": origin.get("loss_norm") or "l1",
            "asinh_knee": origin.get("asinh_knee"),
            "psnr": psnr.get(label),
            "psnr_rank": rank,
            "checkpoint": True,
            "source": origin.get("source"),
        })
    ranked = sorted((m for m in members if m["psnr"] is not None), key=lambda m: -m["psnr"])
    for rank, member in enumerate(ranked, 1):
        member["psnr_rank"] = rank
    splits = {subset: _split_status(subset) for subset in ("train", "validate", "test")}
    return {
        "base_dir": ensemble_dir(),
        "records_dir": records_dir(),
        "members": members,
        "n_members": len(members),
        "n_models": len(members),
        "records": {"present": any(item["present"] for item in splits.values()), "splits": splits,
                    "dataset": _read_json(os.path.join(records_dir(), "dataset.json"))},
        "eval_subset": "test",
        "test_present": splits["test"]["present"],
        "validate_present": splits["validate"]["present"],
        "train_present": splits["train"]["present"],
        "psnr_fields": MEMBER_PSNR_FIELDS,
        "evaluations_available": os.path.isfile(os.path.join(cubes_dir(), "viz_index.json")),
        "evaluations_ready": bool(summary and not stale),
        "eval_summary": summary,
        "eval_summary_stale": stale,
    }


def training_curves_payload() -> list[dict]:
    from euclid_polish.training.log_plot import ensemble_training_series

    summary = _summary()
    meta_by_name = {
        f"member_{str(label).split('·')[0]}": meta
        for label, meta in zip(member_labels(), _member_meta(member_labels(), summary), strict=True)
    }
    out = []
    for series in ensemble_training_series(ensemble_dir()):
        meta = meta_by_name.get(series.get("name"))
        if meta is None:
            continue
        series.update(meta)
        out.append(series)
    return out


def payload_path() -> str:
    return os.path.join(output_dir(), "ensemble_evals.json")


def combiner_payload_path() -> str:
    return os.path.join(output_dir(), "combiner_evals.json")


def _samples_path() -> str:
    return os.path.join(output_dir(), "ensemble_diag_samples.json")


def _write_samples(diag: EnsembleDiagnosticsAccumulator) -> None:
    with open(_samples_path(), "w") as handle:
        json.dump(diag.samples_payload(), handle)


def _iter_cached_fields(subset: str = "test"):
    bucket = cubes_dir(None if subset == "test" else subset)
    manifest = _read_json(os.path.join(bucket, "viz_index.json"))
    if not manifest or list(manifest.get("member_labels") or []) != member_labels():
        return
    if manifest.get("records_fp") != _record_fingerprint(subset):
        return
    indices = [int(i) for i in manifest.get("indices") or []]
    if not indices:
        return
    targets = {r.index: r for r in read_images(
        tfrecord_path(records_dir(), f"{TARGET_KIND}_{subset}"), num_images=max(indices) + 1)}
    dirty_path = tfrecord_path(records_dir(), f"dirty_{subset}")
    dirty = ({r.index: r for r in read_images(dirty_path, num_images=max(indices) + 1)}
             if os.path.isfile(dirty_path) else {})
    n_members = len(manifest.get("member_labels") or [])
    for rec in indices:
        sr_path = os.path.join(bucket, f"sr_{rec:05d}.npy")
        truth = targets.get(rec)
        member_paths = [os.path.join(bucket, f"member{i}_{rec:05d}.npy") for i in range(n_members)]
        if (
            truth is None
            or not os.path.isfile(sr_path)
            or not all(os.path.isfile(path) for path in member_paths)
        ):
            continue
        truth_v = _vis(np.asarray(truth.data, np.float32))
        comb_path = os.path.join(bucket, f"comb_{rec:05d}.npy")
        dirty_rec = dirty.get(rec)
        lr_v = (_lr_on_hr_grid(np.asarray(dirty_rec.data, np.float32), int(truth_v.shape[0]))
                if dirty_rec is not None else None)
        yield (
            truth_v,
            _vis(np.load(sr_path)),
            np.stack([_vis(np.load(path)) for path in member_paths]),
            _vis(np.load(comb_path)) if os.path.isfile(comb_path) else None,
            lr_v,
            rec,
        )


def compute_evaluation_payload() -> dict | None:
    spectrum = None
    diag = EnsembleDiagnosticsAccumulator()
    metrics = _CombinerMetricAcc()
    for truth, mean, members, combiner, lr, rec in _iter_cached_fields():
        if spectrum is None:
            spectrum = EnsembleSpectrumAccumulator(int(truth.shape[0]), float(Config.DEFAULT_PIXEL_SCALE))
        spectrum.add(
            truth, mean, members,
            model_combiners={RAW_INCREMENTAL_MINMEANMAX_RBF_KIND: combiner},
            lr=lr,
        )
        diag.add(truth, mean, members, combiner=combiner, field_index=rec)
        metrics.add(truth, mean, members, combiner)
    if diag.n_fields == 0:
        return None
    _write_samples(diag)
    manifest = _read_json(os.path.join(cubes_dir(), "viz_index.json")) or {}
    labels = [str(x) for x in manifest.get("member_labels") or []]
    curves = spectrum.curves() if spectrum is not None and float(spectrum.bc.sum()) > 0 else None
    payload = _evals_payload(curves, diag, labels, manifest.get("subset", "test"),
                             combiner=metrics.block(labels))
    payload["members"] = [
        {"label": label, **meta}
        for label, meta in zip(labels, _member_meta(labels, _summary()), strict=True)
    ]
    payload["regime"] = REGIME
    payload["target_kind"] = TARGET_KIND
    with open(payload_path(), "w") as handle:
        json.dump(payload, handle)
    return payload


def job_evaluate(cap, *, num_images: int = 100, force: bool = False) -> dict:
    subset = "test"
    if _record_fingerprint(subset) is None:
        raise RuntimeError("lens-isolation dirty_test + lens_test records are not synced")
    if not _member_dirs():
        raise RuntimeError("lens-isolation checkpoints are not synced")
    identity = _identity(subset, num_images)
    if not force and (cached := _reusable(identity)) is not None:
        cap.tick(0, 1, "cached evaluation found — rebuilding figures")
        compute_evaluation_payload()
        cap.tick(1, 1, "reused cached evaluation")
        return {**cached, "reused": True}

    bucket = cubes_dir()
    shutil.rmtree(bucket, ignore_errors=True)
    os.makedirs(bucket, exist_ok=True)
    ensemble = LensIsolationEnsemble(ensemble_dir())
    dirty = read_images(tfrecord_path(records_dir(), "dirty_test"), num_images=int(num_images))
    targets = read_images(tfrecord_path(records_dir(), "lens_test"), num_images=int(num_images))
    viz_cap = min(int(num_images), ENSEMBLE_VIZ_FIELDS_MAX)
    saved: list[int] = []
    pca_amps, pca_var = {}, {}
    spectrum = [None]
    diag = EnsembleDiagnosticsAccumulator()
    metrics = _CombinerMetricAcc()
    from euclid_polish.eval.combiner import load_combiner
    combiner_model = load_combiner(output_dir(), member_labels=ensemble.member_labels)

    def on_field(rec, lr, predictions, mean, std, truth):
        combined = None
        if truth is not None:
            truth_v, mean_v = _vis(truth), _vis(mean)
            members = np.asarray(predictions, np.float32)
            member_v = members[..., 0] if members.ndim == 4 else members
            if (
                combiner_model is not None
                and members.ndim == 4
                and members.shape[0] == len(combiner_model.member_labels)
            ):
                combined = combiner_model.apply_field(members)
            comb_v = _vis(combined) if combined is not None else None
            if spectrum[0] is None:
                spectrum[0] = EnsembleSpectrumAccumulator(
                    int(truth_v.shape[0]), float(Config.DEFAULT_PIXEL_SCALE)
                )
            spectrum[0].add(
                truth_v, mean_v, member_v,
                model_combiners={RAW_INCREMENTAL_MINMEANMAX_RBF_KIND: comb_v},
                lr=_lr_on_hr_grid(lr, int(truth_v.shape[0])),
            )
            diag.add(truth_v, mean_v, member_v, combiner=comb_v,
                     field_index=int(rec) if len(saved) < viz_cap else None)
            metrics.add(truth_v, mean_v, member_v, comb_v)
        if len(saved) >= viz_cap:
            return
        amps, variance = _cache_field_cubes(bucket, int(rec), predictions, mean, std)
        if combined is not None:
            np.save(os.path.join(bucket, f"comb_{int(rec):05d}.npy"), np.asarray(combined, np.float32))
        pca_amps[int(rec)], pca_var[int(rec)] = amps, variance
        saved.append(int(rec))

    result = ensemble.evaluate(
        dirty,
        targets,
        on_field=on_field,
        on_progress=lambda i, n, label: cap.tick(i, n, label),
    )
    labels = ensemble.member_labels
    with open(os.path.join(bucket, "viz_index.json"), "w") as handle:
        json.dump({
            "subset": subset,
            "indices": saved,
            "pca_n": ENSEMBLE_PCA_COMPONENTS,
            "pca_amps": pca_amps,
            "pca_var": pca_var,
            "member_labels": labels,
            "has_combiner": combiner_model is not None and metrics.n_comb > 0,
            "records_fp": _record_fingerprint(subset),
            "target_kind": TARGET_KIND,
        }, handle)
    curves = spectrum[0].curves() if spectrum[0] is not None and float(spectrum[0].bc.sum()) > 0 else None
    payload = _evals_payload(curves, diag, labels, subset, combiner=metrics.block(labels))
    payload["members"] = [
        {"label": label, **meta}
        for label, meta in zip(labels, _member_meta(labels, result), strict=True)
    ]
    payload["regime"], payload["target_kind"] = REGIME, TARGET_KIND
    with open(payload_path(), "w") as handle:
        json.dump(payload, handle)
    _write_samples(diag)
    if curves is not None:
        with open(os.path.join(output_dir(), "ensemble_power_spectrum.json"), "w") as handle:
            json.dump({key: _jsonable(value) for key, value in curves.items()}, handle)
    block = metrics.block(labels)
    if block and block.get("available"):
        result["combiner_psnr"] = block["psnr"]
        result["combiner_vs_mean_db"] = block["psnr"] - block["ensemble_mean_psnr"]
    result.update({
        "regime": REGIME,
        "target_kind": TARGET_KIND,
        "member_labels": labels,
        "eval_identity": identity,
        "reused": False,
        "viz_fields": len(saved),
    })
    with open(os.path.join(output_dir(), "eval_summary.json"), "w") as handle:
        json.dump(result, handle, indent=2)
    return result


def compute_combiner_payload() -> dict | None:
    from euclid_polish.eval.combiner import load_combiner
    combiner = load_combiner(output_dir())
    if combiner is None:
        return None
    labels = member_labels()
    payload = {
        "available": True,
        "stale": list(combiner.member_labels) != labels,
        "kind": combiner.kind,
        "regime": REGIME,
        "member_labels": list(combiner.member_labels),
        "members": [
            {"label": label, **meta}
            for label, meta in zip(combiner.member_labels,
                                   _member_meta(list(combiner.member_labels), _summary()), strict=True)
        ],
        "n_kernels": int(combiner.n_kernels),
        "min_usage": float(combiner.min_usage),
        "val_l1": combiner.val_l1,
        "band_names": list(combiner.band_names),
        "surviving": combiner.surviving_members(),
        "eff_weights": {},
    }
    for band in combiner.bands:
        weights = combiner.effective_weights(band)
        payload["eff_weights"][band] = {
            "brightness_asinh": _jsonable(weights["brightness_asinh"]),
            "brightness_e": _jsonable(weights["brightness_e"]),
            "jacobian": _jsonable(weights["jacobian"]),
        }
    with open(combiner_payload_path(), "w") as handle:
        json.dump(payload, handle)
    return payload


def _apply_combiner_to_test() -> bool:
    from euclid_polish.eval.combiner import load_combiner
    manifest_path = os.path.join(cubes_dir(), "viz_index.json")
    manifest = _read_json(manifest_path)
    if not manifest:
        return False
    labels = [str(x) for x in manifest.get("member_labels") or []]
    combiner = load_combiner(output_dir(), member_labels=labels)
    if combiner is None:
        return False
    applied = 0
    for rec in [int(i) for i in manifest.get("indices") or []]:
        paths = [os.path.join(cubes_dir(), f"member{i}_{rec:05d}.npy") for i in range(len(labels))]
        if not all(os.path.isfile(path) for path in paths):
            continue
        stack = np.stack([np.load(path) for path in paths])
        np.save(os.path.join(cubes_dir(), f"comb_{rec:05d}.npy"), combiner.apply_field(stack))
        applied += 1
    manifest["has_combiner"] = bool(applied)
    with open(manifest_path, "w") as handle:
        json.dump(manifest, handle)
    return bool(applied)


def _reevaluate_from_cubes(num_images: int = 100) -> dict | None:
    payload = compute_evaluation_payload()
    if payload is None:
        return None
    labels = member_labels()
    metrics = _CombinerMetricAcc()
    for truth, mean, members, combiner, _lr, _rec in _iter_cached_fields():
        metrics.add(truth, mean, members, combiner)
    block = metrics.block(labels)
    if block is None:
        return None
    per_member = (metrics.mem / metrics.n).tolist() if metrics.mem is not None else []
    mean_member = float(np.mean(per_member)) if per_member else None
    summary = dict(_summary() or {})
    summary.update({
        "regime": REGIME,
        "target_kind": TARGET_KIND,
        "member_labels": labels,
        "per_member_labels": labels,
        "per_member_psnr_stretched": [float(value) for value in per_member],
        "n_scored": int(metrics.n),
        "ensemble_psnr": float(metrics.mean / metrics.n),
        "mean_member_psnr": mean_member,
        "ensemble_gain_db": (float(metrics.mean / metrics.n - max(per_member)) if per_member else None),
        "eval_identity": _identity("test", num_images),
        "recomputed_from_cubes": True,
        "reused": False,
    })
    if block.get("available"):
        summary["combiner_psnr"] = block["psnr"]
        summary["combiner_vs_mean_db"] = block["psnr"] - block["ensemble_mean_psnr"]
    with open(os.path.join(output_dir(), "eval_summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def job_combiner_fit(cap, *, num_images: int = 100, n_kernels: int = 128,
                     min_usage: float = 0.0) -> dict:
    from euclid_polish.eval.combiner import BAND_NAMES, FitBufferAccumulator, fit_combiner, save_combiner
    from euclid_polish.eval.ensemble_cube_cache import load_cached_member_stack
    if _record_fingerprint("validate") is None:
        raise RuntimeError("lens-isolation dirty_validate + lens_validate records are not synced")
    labels = member_labels()
    if not labels:
        raise RuntimeError("lens-isolation checkpoints are not synced")
    bucket = cubes_dir("validate")
    manifest = _read_json(os.path.join(bucket, "viz_index.json"))
    reuse = bool(
        manifest
        and manifest.get("subset") == "validate"
        and manifest.get("records_fp") == _record_fingerprint("validate")
        and list(manifest.get("member_labels") or []) == labels
        and manifest.get("indices")
    )
    accumulator = FitBufferAccumulator(BAND_NAMES)
    if reuse:
        indices = [int(i) for i in manifest.get("indices") or []]
        targets = {r.index: r for r in read_images(
            tfrecord_path(records_dir(), "lens_validate"),
            num_images=max(indices) + 1,
        )}
        for position, rec in enumerate(indices, 1):
            stack = load_cached_member_stack(rec, subset="validate", cubes_dir=bucket, active=labels)
            truth = targets.get(rec)
            if stack is not None and truth is not None:
                accumulator.add(stack, np.asarray(truth.data, np.float32))
            cap.tick(position, len(indices), f"reuse field {rec}")
    else:
        shutil.rmtree(bucket, ignore_errors=True)
        os.makedirs(bucket, exist_ok=True)
        ensemble = LensIsolationEnsemble(ensemble_dir())
        dirty = read_images(tfrecord_path(records_dir(), "dirty_validate"), num_images=int(num_images))
        targets = read_images(tfrecord_path(records_dir(), "lens_validate"), num_images=int(num_images))
        saved = []

        def on_field(rec, _lr, predictions, mean, std, truth):
            _cache_field_cubes(bucket, int(rec), predictions, mean, std)
            saved.append(int(rec))
            if truth is not None:
                accumulator.add(predictions, truth)

        ensemble.evaluate(dirty, targets, on_field=on_field,
                          on_progress=lambda i, n, label: cap.tick(i, n, label))
        indices = saved
        with open(os.path.join(bucket, "viz_index.json"), "w") as handle:
            json.dump({"subset": "validate", "indices": saved,
                       "member_labels": labels, "records_fp": _record_fingerprint("validate"),
                       "pca_n": ENSEMBLE_PCA_COMPONENTS, "target_kind": TARGET_KIND}, handle)
    buffers = accumulator.buffer()
    if not np.asarray(buffers[0]).size:
        raise RuntimeError("no lens-isolation validation pixels collected")
    combiner = fit_combiner(buffers, labels, n_kernels=int(n_kernels), min_usage=float(min_usage))
    combiner.records_fp = _record_fingerprint("validate")
    combiner.starfull = False
    combiner.fit_meta = {"subset": "validate", "num_images": int(num_images),
                         "experiment": REGIME, "target_kind": TARGET_KIND}
    save_combiner(combiner, output_dir())
    compute_combiner_payload()
    cap.tick(0, 1, "scoring combiner on test cubes")
    scored = _apply_combiner_to_test()
    summary = _reevaluate_from_cubes() if scored else None
    cap.tick(1, 1, "done")
    result = {"n_members": len(labels), "n_kernels": int(n_kernels),
              "min_usage": float(min_usage), "val_l1": combiner.val_l1,
              "surviving": combiner.surviving_members(), "test_scored": summary is not None,
              "regime": REGIME}
    if summary:
        result["combiner_psnr"] = summary.get("combiner_psnr")
        result["combiner_vs_mean_db"] = summary.get("combiner_vs_mean_db")
    return result


def _b64_f32(array: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(array, dtype="<f4").tobytes()).decode("ascii")


def pixel_trace(diag: str, i: int, j: int, *, half: int = PIXEL_TRACE_HALF,
                max_stamps: int = PIXEL_TRACE_STAMPS) -> dict:
    size = 2 * int(half) + 1
    out = {"diag": diag, "i": int(i), "j": int(j), "half": int(half), "size": size,
           "bands": list(Config.LR_INPUT_BAND_NAMES), "stretch": float(Config.STRETCH_SCALE_E),
           "stamps": []}
    side = _read_json(_samples_path()) or {}
    picks = (side.get(diag) or {}).get(f"{int(i)},{int(j)}") or []
    if not picks:
        return out
    picks = [tuple(int(value) for value in pick) for pick in picks][:max_stamps]
    grouped: dict[int, list[tuple[int, int]]] = {}
    for rec, y, x in picks:
        grouped.setdefault(rec, []).append((y, x))
    max_index = max(grouped) + 1
    targets = {r.index: r for r in read_images(
        tfrecord_path(records_dir(), "lens_test"), num_images=max_index)}
    dirty = {r.index: r for r in read_images(
        tfrecord_path(records_dir(), "dirty_test"), num_images=max_index)}

    def crop(cube, y, x):
        cube = cube[..., None] if cube.ndim == 2 else cube
        height, width, channels = cube.shape
        window = np.zeros((size, size, channels), np.float32)
        y0, y1 = max(0, y - half), min(height, y + half + 1)
        x0, x1 = max(0, x - half), min(width, x + half + 1)
        window[y0 - (y - half):y1 - (y - half), x0 - (x - half):x1 - (x - half)] = cube[y0:y1, x0:x1]
        return window

    for rec, coords in grouped.items():
        target = targets.get(rec)
        dirty_rec = dirty.get(rec)
        sr_path = os.path.join(cubes_dir(), f"sr_{rec:05d}.npy")
        std_path = os.path.join(cubes_dir(), f"std_{rec:05d}.npy")
        combiner_path = os.path.join(cubes_dir(), f"comb_{rec:05d}.npy")
        if target is None or not os.path.isfile(sr_path) or not os.path.isfile(std_path):
            continue
        truth = np.asarray(target.data, np.float32)
        use_combiner = os.path.isfile(combiner_path)
        reconstruction = np.load(combiner_path if use_combiner else sr_path).astype(np.float32)
        std = _vis(np.load(std_path)).astype(np.float32)
        lr = (_lr_cube_on_hr_grid(np.asarray(dirty_rec.data, np.float32), truth.shape[0])
              if dirty_rec is not None else None)
        truth_v, reconstruction_v = _vis(truth), _vis(reconstruction)
        for y, x in coords:
            if not (0 <= y < truth.shape[0] and 0 <= x < truth.shape[1]):
                continue
            stamp = {
                "field": rec, "y": y, "x": x, "center": int(half),
                "sr_is_combiner": use_combiner,
                "hr": _b64_f32(crop(truth, y, x)),
                "sr": _b64_f32(crop(reconstruction, y, x)),
                "std": _b64_f32(crop(std, y, x)[..., 0]),
                "hr_val": float(truth_v[y, x]), "sr_val": float(reconstruction_v[y, x]),
                "std_val": float(std[y, x]),
                "err_val": abs(float(reconstruction_v[y, x] - truth_v[y, x])),
                "bright_asinh": float(np.arcsinh(truth_v[y, x] / Config.STRETCH_SCALE_E)),
            }
            if lr is not None:
                stamp["lr"] = _b64_f32(crop(lr, y, x))
            out["stamps"].append(stamp)
    return out
