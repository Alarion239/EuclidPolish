"""Persistent real-Euclid field inference for the legacy real fields (Models › Diagnostics, Sky tiles).

One archive request per band fetches a 2560-pixel VIS field.  The field is
then cut deterministically into a 10x10 grid of 256-pixel LR tiles.  Every
tile keeps its raw LR cube plus the STARFULL member, mean, disagreement and
available-combiner SR cubes, so opening the viewer never re-runs inference.

By default only the members the production gate reads are run (a pruned gate
skips the ones it gives no weight), and the member mean, disagreement, PCA and
diagnostics describe just those members (``member_scope`` ``"gate"``).
``all_members=True`` runs every active member (``"all"``), the full
member-diagnostic cache. Member cubes stay indexed by position in the whole
active membership, so switching scope reuses every cube already computed and
never deletes one.
"""
from __future__ import annotations

import contextlib
import json
import os
import warnings
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
from astropy.io import fits

from euclid_polish import ensemble_registry
from euclid_polish.config import Config
from euclid_polish.ensemble import (
    EnsembleModel,
    default_ensemble_dir,
    member_fingerprints,
    pca_field,
)
from euclid_polish.eval.combiner import COMBINER_MODELS, load_combiner
from euclid_polish.eval.ensemble_cube_cache import tree_usage
from euclid_polish.eval.ensemble_infer import (
    PRODUCTION_COMBINER_KIND,
    combiner_read_labels,
    load_production_combiner,
)
from euclid_polish.eval.power_spectrum import log_k_edges, pairwise_cross_correlation
from euclid_polish.photometry import adu_per_s_to_electrons_factor, header_magzero
from euclid_polish.web.helpers import jwst_euclid

FIELD_SIZE = 2560
TILE_SIZE = 256
GRID_SIDE = FIELD_SIZE // TILE_SIZE
_BRIGHTNESS_EDGES = np.linspace(-1.0, 13.0, 81)
_LOG_STD_EDGES = np.linspace(-6.0, 3.0, 73)
_MINMAX_EDGES = np.linspace(-1.0, 13.0, 81)
_POWER_K_EDGES = log_k_edges(Config.DEFAULT_PIXEL_SCALE, kmin=0.2, nbins=24)
_POWER_K_CENTERS = np.sqrt(_POWER_K_EDGES[:-1] * _POWER_K_EDGES[1:])
REAL_FIELD_DIAGNOSTICS_VERSION = 2
#: ``member_scope``: the production gate's read members, or every member.
SCOPE_GATE = "gate"
SCOPE_ALL = "all"


def field_id(ra: float, dec: float) -> str:
    """Stable, filesystem-safe identity for a field centre."""
    return f"ra{ra:010.5f}_dec{dec:+010.5f}".replace("+", "p").replace("-", "m")


def fields_root() -> Path:
    return Path(Config.EUCLID_INFERENCE_DIR) / "real_fields"


def field_dir(identifier: str) -> Path:
    return fields_root() / identifier


def manifest_path(identifier: str) -> Path:
    return field_dir(identifier) / "manifest.json"


def _read_manifest(identifier: str) -> dict[str, Any] | None:
    try:
        with manifest_path(identifier).open() as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def list_fields() -> list[dict[str, Any]]:
    """Every cached real field's manifest (not only the latest), newest first."""
    root = fields_root()
    candidates: list[tuple[float, str, dict[str, Any]]] = []
    if not root.is_dir():
        return []
    for path in root.iterdir():
        if not path.is_dir():
            continue
        manifest = _read_manifest(path.name)
        if manifest is not None:
            candidates.append((path.stat().st_mtime, path.name, manifest))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return [manifest for _mtime, _name, manifest in candidates]


def latest_field() -> dict[str, Any] | None:
    fields = list_fields()
    return fields[0] if fields else None


def _write_json(path: Path, value: dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    with tmp.open("w") as f:
        json.dump(value, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _diagnostic_accumulators(combiners: dict[str, Any], n_members: int) -> dict[str, Any]:
    return {
        "power_rows": [],
        "std_brightness": np.zeros((len(_BRIGHTNESS_EDGES) - 1,
                                    len(_LOG_STD_EDGES) - 1), np.int64),
        "combiner_counts": {
            kind: np.zeros((len(_MINMAX_EDGES) - 1,
                            len(_MINMAX_EDGES) - 1), np.int64)
            for kind in combiners
        },
    }


def _accumulate_diagnostics(acc: dict[str, Any], members: np.ndarray,
                            combiners: dict[str, Any]) -> None:
    """Collect between-member relations and real-pixel gate occupancy.

    Spectral relations use every tile and band; std density is deterministically
    subsampled to keep a 100-tile cache operation bounded, while combiner
    occupancy bins every pixel.
    """
    values = np.arcsinh(np.asarray(members, np.float32) / Config.STRETCH_SCALE_E)
    # Model relation curves: Fourier cross-correlation for every model pair,
    # computed independently in each band. This is explicitly model-vs-model
    # only: no HR enters. Tile/band curves are retained so the final payload
    # can use the same robust median-across-fields convention as evaluation.
    acc["power_rows"].extend(
        pairwise_cross_correlation(
            [values[i, :, :, band] for i in range(values.shape[0])],
            Config.DEFAULT_PIXEL_SCALE, _POWER_K_EDGES,
        )
        for band in range(values.shape[-1])
    )

    std_sample = values[:, ::4, ::4, :]
    brightness = std_sample.mean(axis=0).reshape(-1)
    spread = std_sample.std(axis=0).reshape(-1)
    acc["std_brightness"] += np.histogram2d(
        brightness, np.log10(np.maximum(spread, 1e-6)),
        bins=(_BRIGHTNESS_EDGES, _LOG_STD_EDGES),
    )[0].astype(np.int64)

    # Gate occupancy is the actual distribution of real pixels in each fitted
    # model's coordinate system, rather than an error plot (there is no HR).
    for kind, combiner in combiners.items():
        counts = acc["combiner_counts"][kind]
        for ci, _name in enumerate(combiner.band_names):
            band_values = values[..., ci].reshape(values.shape[0], -1).T
            counts += np.histogram2d(
                np.min(band_values, axis=1), np.max(band_values, axis=1),
                bins=(_MINMAX_EDGES, _MINMAX_EDGES),
            )[0].astype(np.int64)


def _diagnostic_payload(acc: dict[str, Any], labels: list[str],
                        combiners: dict[str, Any], *, scope: str = SCOPE_ALL,
                        n_ensemble: int | None = None) -> dict[str, Any]:
    power_rows = acc["power_rows"]
    n_pairs = len(labels) * (len(labels) - 1) // 2
    if power_rows:
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            pair_curves = np.nanmedian(np.stack(power_rows, axis=0), axis=0)
            median_curve = np.nanmedian(pair_curves, axis=0)
    else:
        pair_curves = np.full((n_pairs, len(_POWER_K_CENTERS)), np.nan)
        median_curve = np.full(len(_POWER_K_CENTERS), np.nan)

    def json_numbers(array: np.ndarray) -> list:
        return [None if not np.isfinite(float(value)) else float(value)
                for value in np.asarray(array).reshape(-1)]

    def json_rows(array: np.ndarray) -> list[list[float | None]]:
        return [json_numbers(row) for row in np.asarray(array)]

    combiner_payload: dict[str, Any] = {}
    for kind, counts in acc["combiner_counts"].items():
        combiner_payload[kind] = {
            "kind": kind, "mode": "heat", "x_edges": _MINMAX_EDGES.tolist(),
            "y_edges": _MINMAX_EDGES.tolist(), "counts": counts.tolist(),
            "x_label": "min member brightness (asinh)",
            "y_label": "max member brightness (asinh)",
            "pixel_count": int(counts.sum()),
        }
    return {
        "version": REAL_FIELD_DIAGNOSTICS_VERSION,
        "member_labels": labels,
        "member_scope": scope,
        "n_ensemble_members": len(labels) if n_ensemble is None else int(n_ensemble),
        "model_power": {
            "k": _POWER_K_CENTERS.tolist(),
            "r_pairs": json_rows(pair_curves),
            "r_cross": json_numbers(median_curve),
            "pair_indices": [[i, j] for i in range(len(labels))
                             for j in range(i + 1, len(labels))],
            "samples": len(power_rows),
            "pixel_scale_arcsec": float(Config.DEFAULT_PIXEL_SCALE),
        },
        "std_brightness": {"x_edges": _BRIGHTNESS_EDGES.tolist(), "y_edges": _LOG_STD_EDGES.tolist(),
                           "counts": acc["std_brightness"].tolist(),
                           "x_label": "mean brightness (asinh)", "y_label": "log10(member std)"},
        "combiners": combiner_payload,
    }


def _clean_header(header):
    out = header.copy()
    for key in ("EXTNAME", "XTENSION"):
        with contextlib.suppress(KeyError):
            del out[key]
    return out


def _center_crop_field(data: np.ndarray, header, *, size: int) -> tuple[np.ndarray, Any]:
    """Return the central ``size`` square and its WCS-adjusted header.

    The Euclid cutout service can round a pixel request upward (for example,
    2560 → 2571).  We retain the requested, tileable footprint rather than
    rejecting an otherwise valid archive response.
    """
    if data.ndim != 2 or data.shape[0] != data.shape[1]:
        raise RuntimeError(f"archive field must be square 2-D, got {data.shape}")
    side = int(data.shape[0])
    if side < size:
        raise RuntimeError(f"archive field is {data.shape}, smaller than {size}x{size}")
    offset = (side - size) // 2
    cropped = np.ascontiguousarray(data[offset:offset + size, offset:offset + size])
    adjusted = header.copy()
    # FITS CRPIX is one-indexed but represents a pixel coordinate; removing
    # ``offset`` leading pixels shifts the reference by that same amount.
    for key in ("CRPIX1", "CRPIX2"):
        if key in adjusted:
            adjusted[key] = float(adjusted[key]) - offset
    return cropped, adjusted


def _load_or_download_lr(ra: float, dec: float, root: Path,
                         tick: Callable[[int, int, str], None]) -> np.ndarray:
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    bands: list[np.ndarray] = []
    headers = []
    for index, name in enumerate(Config.LR_INPUT_BAND_NAMES):
        path = raw / f"{name}.fits"
        if not path.is_file() or path.stat().st_size == 0:
            tick(index, 4 + GRID_SIDE * GRID_SIDE, f"downloading {name} field")
            # From the Q1 MER tile whose polygon CONTAINS the position (the
            # nearest-centre archive lookup returned partial edge fields).
            ok, error = jwst_euclid.fetch_q1_cutout(
                ra=ra, dec=dec, band_name=name, output_file=str(path),
                cutout_size_vis_pixels=FIELD_SIZE)
            if not ok:
                raise RuntimeError(f"{name}: {error}")
        else:
            tick(index, 4 + GRID_SIDE * GRID_SIDE, f"reusing {name} field")
        with fits.open(path, memmap=False) as hdul:
            primary = cast(fits.PrimaryHDU, hdul[0])
            data = np.asarray(primary.data, np.float32)
            header = primary.header.copy()
        if data.shape != (FIELD_SIZE, FIELD_SIZE):
            print(f"  {name}: archive returned {data.shape}; center-cropping to "
                  f"{FIELD_SIZE}x{FIELD_SIZE}")
        data, header = _center_crop_field(data, header, size=FIELD_SIZE)
        band = Config.get_band(name)
        bands.append(data * adu_per_s_to_electrons_factor(
            header_magzero(header, source=f"{name} field"), band))
        headers.append(header)
    cube = np.stack(bands, axis=-1).astype(np.float32)
    stack = np.moveaxis(cube, -1, 0)
    fits.PrimaryHDU(stack, header=_clean_header(headers[0])).writeto(
        root / "original_stack.fits", overwrite=True, output_verify="silentfix")
    return cube


def _preserve_matching_member_cubes(
    cubes: Path,
    old_labels: list[str],
    new_labels: list[str],
    *,
    count: int,
    reusable: Callable[[str], bool] | None = None,
) -> tuple[Path, int]:
    """Hard-link reusable member tiles into a temporary remapping directory.
    ``reusable`` (default: every matching label) narrows which labels count."""
    staging = cubes / ".member_reuse"
    staging.mkdir(exist_ok=True)
    for stale in staging.iterdir():
        if stale.is_file():
            stale.unlink()
    old_indices = {str(label): index for index, label in enumerate(old_labels)}
    preserved = 0
    for new_index, label in enumerate(new_labels):
        old_index = old_indices.get(str(label))
        if old_index is None or (reusable is not None and not reusable(str(label))):
            continue
        for tile in range(max(0, int(count))):
            source = cubes / f"member{old_index}_{tile:03d}.npy"
            if not source.is_file():
                continue
            staged = staging / f"member{new_index}_{tile:03d}.npy"
            os.link(source, staged)
            preserved += 1
    return staging, preserved


def _restore_matching_member_cubes(cubes: Path, staging: Path) -> None:
    for staged in staging.glob("member*_*.npy"):
        os.replace(staged, cubes / staged.name)
    with contextlib.suppress(OSError):
        staging.rmdir()


def _run_member_indices(combiners: dict[str, Any], labels: list[str], *,
                        all_members: bool) -> tuple[list[int], str]:
    """``(positions in labels, scope)`` of the members a cache pass runs: the
    production gate's read members, or every member when asked (or when no
    production gate is current for this membership, so the mean is the
    fallback SR)."""
    production = combiners.get(PRODUCTION_COMBINER_KIND)
    if all_members or production is None:
        return list(range(len(labels))), SCOPE_ALL
    position = {label: i for i, label in enumerate(labels)}
    return sorted(position[label] for label in combiner_read_labels(production)), SCOPE_GATE


def _applicable_combiners(combiners: dict[str, Any],
                          run_labels: list[str]) -> dict[str, Any]:
    """The combiners whose every read member is in ``run_labels``."""
    have = set(run_labels)
    return {kind: comb for kind, comb in combiners.items()
            if set(combiner_read_labels(comb)) <= have}


def _combiner_stack(members: np.ndarray, run_labels: list[str], combiner) -> np.ndarray:
    """The rows of the ``run_labels`` stack that ``combiner`` reads, in its order."""
    rows = {label: row for row, label in enumerate(run_labels)}
    return members[[rows[label] for label in combiner_read_labels(combiner)]]


def _load_regime_combiners(labels: list[str]) -> dict[str, Any]:
    """Every combiner fitted for exactly ``labels``; the production gate
    whenever every member it reads is among them (members that joined after
    its fit do not make it stale)."""
    regime_dir = Path(Config.VIS_DIR) / "ensemble" / "starfull"
    combiners = {
        kind: comb
        for kind, spec in COMBINER_MODELS.items()
        if (comb := load_combiner(str(regime_dir), member_labels=labels,
                                  artifact_dir=spec.artifact_dir)) is not None
    }
    production = load_production_combiner(labels, str(regime_dir))
    if production is not None:
        combiners[PRODUCTION_COMBINER_KIND] = production
    return combiners


def _combiner_state(combiners: dict[str, Any]) -> dict[str, list[int]]:
    regime_dir = Path(Config.VIS_DIR) / "ensemble" / "starfull"
    state = {}
    for kind in combiners:
        stat = (regime_dir / COMBINER_MODELS[kind].artifact_dir / "combiner.npz").stat()
        state[kind] = [int(stat.st_mtime_ns), int(stat.st_size)]
    return state


def _write_tile_products(cubes: Path, tile: int, tile_lr: np.ndarray,
                         members: np.ndarray, run_labels: list[str],
                         combiners: dict[str, Any]) -> tuple[list[float], list[float]]:
    """Mean, std, PCA and combiner cubes of one tile from the stack of the
    ``run_labels`` members; returns the tile's PCA amplitudes and variances."""
    mean, pcs, amps, variance = pca_field(members)
    np.save(cubes / f"sr_{tile:03d}.npy", mean)
    np.save(cubes / f"std_{tile:03d}.npy", members.std(axis=0))
    for i, component in enumerate(pcs):
        np.save(cubes / f"pca{i}_{tile:03d}.npy", component)
    for kind, combiner in combiners.items():
        prefix = COMBINER_MODELS[kind].cube_prefix
        np.save(cubes / f"{prefix}_{tile:03d}.npy",
                combiner.apply_field(_combiner_stack(members, run_labels, combiner),
                                     lr=tile_lr))
    return [float(x) for x in amps], [float(x) for x in variance]


def _scope_fields(labels: list[str], run: list[int], scope: str) -> dict[str, Any]:
    return {"run_members": list(run),
            "run_member_labels": [labels[i] for i in run],
            "member_scope": scope,
            "pca_n": min(3, max(0, len(run) - 1))}


def cache_real_field(ra: float, dec: float, *,
                     progress: Callable[[int, int, str], None],
                     all_members: bool = False) -> dict[str, Any]:
    """Materialise the 100-tile STARFULL real-data cache, reusing raw data.

    Runs only the members the production gate reads unless ``all_members``;
    member cubes already on disk are reused and never deleted."""
    identifier = field_id(ra, dec)
    root = field_dir(identifier)
    cubes = root / "cubes"
    cubes.mkdir(parents=True, exist_ok=True)
    old = _read_manifest(identifier) or {}
    lr = _load_or_download_lr(ra, dec, root, progress)

    # STARFULL is intentional: real images contain stars, so this workspace
    # never mixes in the separate star-erasing regime.
    labels = ensemble_registry.regime_labels(default_ensemble_dir(), starless=False)
    if not labels:
        raise RuntimeError("no active STARFULL ensemble members")
    n_members = len(labels)
    old_labels = old.get("member_labels") or []
    fingerprints = member_fingerprints(default_ensemble_dir(), labels)
    old_fps = old.get("member_fps")

    def reusable(label: str) -> bool:
        # A cache without fingerprints cannot prove which checkpoint made it.
        return (isinstance(old_fps, dict) and label in old_fps
                and old_fps[label] == fingerprints.get(label))

    if old_labels != labels or not all(reusable(str(label)) for label in old_labels):
        # Preserve member cubes whose label AND checkpoint still match,
        # remapping their positional indices through hard links. All
        # aggregate/PCA/combiner products are membership-dependent and are
        # rebuilt below.
        staging, preserved = _preserve_matching_member_cubes(
            cubes,
            [str(label) for label in old_labels],
            labels,
            count=int(old.get("count", GRID_SIDE * GRID_SIDE)),
            reusable=reusable,
        )
        for path in cubes.glob("*.npy"):
            path.unlink()
        _restore_matching_member_cubes(cubes, staging)
        if preserved:
            print(f"  reused {preserved} matching cached member tiles")

    combiners = _load_regime_combiners(labels)
    combiner_state = _combiner_state(combiners)
    if old.get("combiner_state") != combiner_state:
        # A refit changes the derived prediction even though member cubes are
        # still valid.  Rebuild only the affected cheap fused cubes.
        for spec in COMBINER_MODELS.values():
            for path in cubes.glob(f"{spec.cube_prefix}_*.npy"):
                path.unlink()
    run, scope = _run_member_indices(combiners, labels, all_members=all_members)
    run_labels = [labels[i] for i in run]
    applied = _applicable_combiners(combiners, run_labels)
    count = GRID_SIDE * GRID_SIDE
    missing = sorted({i for tile in range(count) for i in run
                      if not (cubes / f"member{i}_{tile:03d}.npy").is_file()})
    models: dict[int, Any] = {}
    if missing:
        # Restore only the checkpoints some tile still lacks.
        try:
            ensemble = EnsembleModel(default_ensemble_dir(), starless=False,
                                     labels=[labels[i] for i in missing])
        except ValueError as exc:
            raise RuntimeError(
                "STARFULL membership changed during real-field refresh") from exc
        models = dict(zip(missing, ensemble.members, strict=True))
    print(f"  running {len(run)} of {n_members} STARFULL members ({scope}); "
          f"{len(missing)} need inference")
    pca_amps: dict[str, list[float]] = {}
    pca_var: dict[str, list[float]] = {}
    diagnostics = _diagnostic_accumulators(applied, len(run))
    for tile in range(count):
        row, col = divmod(tile, GRID_SIDE)
        ys, xs = row * TILE_SIZE, col * TILE_SIZE
        tile_lr = lr[ys:ys + TILE_SIZE, xs:xs + TILE_SIZE]
        np.save(cubes / f"lr_{tile:03d}.npy", tile_lr)
        member_paths = [cubes / f"member{i}_{tile:03d}.npy" for i in run]
        for position, path in zip(run, member_paths, strict=True):
            if not path.is_file():
                np.save(path, np.asarray(
                    models[position].upsample_array(tile_lr), np.float32))
        members = np.stack(
            [np.load(path) for path in member_paths]).astype(np.float32)
        pca_amps[str(tile)], pca_var[str(tile)] = _write_tile_products(
            cubes, tile, tile_lr, members, run_labels, applied)
        _accumulate_diagnostics(diagnostics, members, applied)
        progress(4 + tile + 1, 4 + count, f"caching tile {tile + 1}/{count}")

    manifest = {
        "field_id": identifier, "ra": float(ra), "dec": float(dec),
        "field_size": FIELD_SIZE, "tile_size": TILE_SIZE, "grid_side": GRID_SIDE,
        "count": count, "member_labels": labels,
        "member_fps": {label: fingerprints.get(label) for label in labels},
        "combiner_kinds": sorted(applied), "combiner_state": combiner_state,
        **_scope_fields(labels, run, scope),
        "pca_amps": pca_amps, "pca_var": pca_var,
    }
    _write_json(manifest_path(identifier), manifest)
    _write_json(root / "diagnostics.json", _diagnostic_payload(
        diagnostics, run_labels, applied, scope=scope, n_ensemble=n_members))
    return manifest


def refresh_real_field_combiners(
    identifier: str | None = None, *,
    progress: Callable[[int, int, str], None],
    all_members: bool = False,
) -> dict[str, Any]:
    """Reapply fitted STARFULL combiners to an existing real-field cache.

    Member cubes remain untouched; the mean, disagreement and PCA cubes are
    rebuilt only when the members in scope changed (a promoted gate reads
    others, or ``all_members`` flipped). This path loads one cached member
    stack at a time and never constructs the TensorFlow member networks,
    making post-fit real-star reevaluation both faster and substantially less
    memory-intensive than a full field recache. A member in scope without its
    cubes raises the "member cache is stale" error, so the caller runs
    :func:`cache_real_field` instead.
    """
    manifest = (_read_manifest(identifier) if identifier else latest_field())
    if manifest is None:
        raise RuntimeError("no cached real Euclid field")
    identifier = str(manifest["field_id"])
    root = field_dir(identifier)
    cubes = root / "cubes"
    labels = [str(label) for label in manifest.get("member_labels", [])]
    active_labels = ensemble_registry.regime_labels(
        default_ensemble_dir(), starless=False)
    if labels != active_labels:
        raise RuntimeError(
            "real-field member cache is stale; run the full field cache once")
    recorded_fps = manifest.get("member_fps")
    current_fps = member_fingerprints(default_ensemble_dir(), labels)
    if not isinstance(recorded_fps, dict) or any(
            recorded_fps.get(label) != current_fps.get(label) for label in labels):
        raise RuntimeError(
            "real-field member cache is stale (a member's checkpoint changed); "
            "run the full field cache once")

    combiners = _load_regime_combiners(labels)
    if not combiners:
        raise RuntimeError("no fitted STARFULL combiners")
    run, scope = _run_member_indices(combiners, labels, all_members=all_members)
    run_labels = [labels[i] for i in run]
    applied = _applicable_combiners(combiners, run_labels)
    # A cache from before member scopes held every member.
    recorded = [int(i) for i in manifest.get("run_members", range(len(labels)))]
    rebuild_members = recorded != run
    diagnostics = _diagnostic_accumulators(applied, len(run))
    count = int(manifest.get("count", 0))
    pca_amps: dict[str, list[float]] = {}
    pca_var: dict[str, list[float]] = {}
    for tile in range(count):
        paths = [cubes / f"member{i}_{tile:03d}.npy" for i in run]
        if not all(path.is_file() for path in paths):
            raise RuntimeError(
                "real-field member cache is stale: member cubes missing for "
                f"tile {tile + 1}")
        members = np.stack([np.load(path) for path in paths]).astype(np.float32)
        tile_lr = np.load(cubes / f"lr_{tile:03d}.npy")
        if rebuild_members:
            pca_amps[str(tile)], pca_var[str(tile)] = _write_tile_products(
                cubes, tile, tile_lr, members, run_labels, applied)
        else:
            for kind, combiner in applied.items():
                prefix = COMBINER_MODELS[kind].cube_prefix
                np.save(cubes / f"{prefix}_{tile:03d}.npy",
                        combiner.apply_field(
                            _combiner_stack(members, run_labels, combiner), lr=tile_lr))
        _accumulate_diagnostics(diagnostics, members, applied)
        progress(tile + 1, count, f"real-star combiner reevaluation {tile + 1}/{count}")

    manifest["combiner_kinds"] = sorted(applied)
    manifest["combiner_state"] = _combiner_state(combiners)
    manifest.update(_scope_fields(labels, run, scope))
    if rebuild_members:
        manifest["pca_amps"], manifest["pca_var"] = pca_amps, pca_var
    _write_json(manifest_path(identifier), manifest)
    _write_json(root / "diagnostics.json", _diagnostic_payload(
        diagnostics, run_labels, applied, scope=scope, n_ensemble=len(labels)))
    return manifest


def purge_stale_real_field(identifier: str,
                           current: Mapping[str, str | None]) -> dict[str, Any] | None:
    """Drop the cubes of the members a field's cache can no longer reuse.

    ``current`` maps the ACTIVE STARFULL labels to their checkpoint
    fingerprints now. A member that is not active, has no recorded
    fingerprint or was continued loses its cubes; the kept members are
    renumbered (the hard-link remap :func:`cache_real_field` uses) and every
    membership-dependent product (``sr_``, ``std_``, ``pcaN_``, combiner
    outputs) goes. ``lr_`` tiles, ``raw/`` and the stack stay, so the viewer
    shows the LR until the next refresh re-infers the dropped members.
    ``None`` when nothing was stale."""
    manifest = _read_manifest(identifier)
    if manifest is None:
        return None
    labels = [str(label) for label in manifest.get("member_labels", []) or []]
    recorded = manifest.get("member_fps")
    recorded = recorded if isinstance(recorded, dict) else {}
    keep = [label for label in labels
            if label in current and label in recorded and recorded[label] == current[label]]
    dropped = [label for label in labels if label not in keep]
    if not dropped:
        return None
    cubes = field_dir(identifier) / "cubes"
    before = tree_usage(str(cubes))
    staging, _preserved = _preserve_matching_member_cubes(
        cubes, labels, keep, count=int(manifest.get("count", GRID_SIDE * GRID_SIDE)))
    for path in cubes.glob("*.npy"):
        if not path.name.startswith("lr_"):
            path.unlink(missing_ok=True)
    _restore_matching_member_cubes(cubes, staging)
    old_run = [int(i) for i in manifest.get("run_members", range(len(labels)))]
    run = [keep.index(labels[i]) for i in old_run if i < len(labels) and labels[i] in keep]
    manifest.update({
        "member_labels": keep, "member_fps": {label: recorded[label] for label in keep},
        "run_members": run, "run_member_labels": [keep[i] for i in run],
        "combiner_kinds": [], "combiner_state": {}, "pca_amps": {}, "pca_var": {},
        "pca_n": 0,
    })
    _write_json(manifest_path(identifier), manifest)
    after = tree_usage(str(cubes))
    return {"field_id": identifier, "dropped": dropped,
            "bytes_freed": max(0, before[0] - after[0]),
            "files_deleted": max(0, before[1] - after[1])}


def purge_stale_real_fields(current: Mapping[str, str | None]) -> list[dict[str, Any]]:
    """:func:`purge_stale_real_field` over every cached field (what changed)."""
    root = fields_root()
    if not root.is_dir():
        return []
    out = []
    for path in sorted(root.iterdir()):
        if path.is_dir():
            purged = purge_stale_real_field(path.name, current)
            if purged is not None:
                out.append(purged)
    return out
