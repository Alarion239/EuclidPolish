"""Persistent real-Euclid field inference used by the Inference workspace.

One archive request per band fetches a 2560-pixel VIS field.  The field is
then cut deterministically into a 10x10 grid of 256-pixel LR tiles.  Every
tile keeps its raw LR cube plus the STARFULL member, mean, disagreement and
available-combiner SR cubes, so opening the viewer never re-runs inference.
"""
from __future__ import annotations

import contextlib
import json
import os
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

from euclid_polish import ensemble_registry
from euclid_polish.catalog.downloader import fetch_cutout_at
from euclid_polish.config import Config
from euclid_polish.ensemble import default_ensemble_dir, pca_field
from euclid_polish.eval.combiner import COMBINER_MODELS, load_combiner
from euclid_polish.eval.power_spectrum import log_k_edges, pairwise_cross_correlation
from euclid_polish.photometry import adu_per_s_to_electrons_factor

FIELD_SIZE = 2560
TILE_SIZE = 256
GRID_SIDE = FIELD_SIZE // TILE_SIZE
_BRIGHTNESS_EDGES = np.linspace(-1.0, 13.0, 81)
_LOG_STD_EDGES = np.linspace(-6.0, 3.0, 73)
_MINMAX_EDGES = np.linspace(-1.0, 13.0, 81)
_POWER_K_EDGES = log_k_edges(Config.DEFAULT_PIXEL_SCALE, kmin=0.2, nbins=24)
_POWER_K_CENTERS = np.sqrt(_POWER_K_EDGES[:-1] * _POWER_K_EDGES[1:])
REAL_FIELD_DIAGNOSTICS_VERSION = 2


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


def latest_field() -> dict[str, Any] | None:
    root = fields_root()
    candidates: list[tuple[float, dict[str, Any]]] = []
    if not root.is_dir():
        return None
    for path in root.iterdir():
        if not path.is_dir():
            continue
        manifest = _read_manifest(path.name)
        if manifest is not None:
            candidates.append((path.stat().st_mtime, manifest))
    return max(candidates, default=(0.0, None), key=lambda x: x[0])[1]


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
                        combiners: dict[str, Any]) -> dict[str, Any]:
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
            ok, error = fetch_cutout_at(
                ra=ra, dec=dec, band_name=name, output_file=str(path),
                cutout_size_vis_pixels=FIELD_SIZE)
            if not ok:
                raise RuntimeError(f"{name}: {error}")
        else:
            tick(index, 4 + GRID_SIDE * GRID_SIDE, f"reusing {name} field")
        with fits.open(path, memmap=False) as hdul:
            data = np.asarray(hdul[0].data, np.float32)
            header = hdul[0].header.copy()
        if data.shape != (FIELD_SIZE, FIELD_SIZE):
            print(f"  {name}: archive returned {data.shape}; center-cropping to "
                  f"{FIELD_SIZE}x{FIELD_SIZE}")
        data, header = _center_crop_field(data, header, size=FIELD_SIZE)
        band = Config.get_band(name)
        bands.append(data * adu_per_s_to_electrons_factor(
            float(header.get("MAGZERO", band.sim_zeropoint_e)), band))
        headers.append(header)
    cube = np.stack(bands, axis=-1).astype(np.float32)
    stack = np.moveaxis(cube, -1, 0)
    fits.PrimaryHDU(stack, header=_clean_header(headers[0])).writeto(
        root / "original_stack.fits", overwrite=True, output_verify="silentfix")
    return cube


def cache_real_field(ra: float, dec: float, *,
                     progress: Callable[[int, int, str], None]) -> dict[str, Any]:
    """Materialise the 100-tile STARFULL real-data cache, reusing raw data."""
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
    if old_labels != labels:
        # Raw archive data remains reusable, but every derived cube belongs to
        # the old membership and must not be presented as current inference.
        for path in cubes.glob("*.npy"):
            path.unlink()

    regime_dir = Path(Config.VIS_DIR) / "ensemble" / "starfull"
    combiners = {
        kind: comb
        for kind, spec in COMBINER_MODELS.items()
        if (comb := load_combiner(str(regime_dir), member_labels=labels,
                                  artifact_dir=spec.artifact_dir)) is not None
    }
    combiner_state = {}
    for kind in combiners:
        spec = COMBINER_MODELS[kind]
        path = regime_dir / spec.artifact_dir / "combiner.npz"
        stat = path.stat()
        combiner_state[kind] = [int(stat.st_mtime_ns), int(stat.st_size)]
    if old.get("combiner_state") != combiner_state:
        # A refit changes the derived prediction even though member cubes are
        # still valid.  Rebuild only the affected cheap fused cubes.
        for spec in COMBINER_MODELS.values():
            for path in cubes.glob(f"{spec.cube_prefix}_*.npy"):
                path.unlink()
    pca_amps: dict[str, list[float]] = {}
    pca_var: dict[str, list[float]] = {}
    diagnostics = _diagnostic_accumulators(combiners, n_members)
    ensemble = None
    for tile in range(GRID_SIDE * GRID_SIDE):
        row, col = divmod(tile, GRID_SIDE)
        ys, xs = row * TILE_SIZE, col * TILE_SIZE
        tile_lr = lr[ys:ys + TILE_SIZE, xs:xs + TILE_SIZE]
        np.save(cubes / f"lr_{tile:03d}.npy", tile_lr)
        member_paths = [cubes / f"member{i}_{tile:03d}.npy" for i in range(n_members)]
        if not all(path.is_file() for path in member_paths):
            if ensemble is None:
                from euclid_polish.ensemble import EnsembleModel
                ensemble = EnsembleModel(default_ensemble_dir(), starless=False)
                if list(ensemble.member_labels) != labels:
                    raise RuntimeError("STARFULL membership changed during real-field refresh")
            members = np.asarray(ensemble.member_arrays(tile_lr), np.float32)
            for i, member in enumerate(members):
                np.save(member_paths[i], member)
        else:
            members = np.stack([np.load(path) for path in member_paths]).astype(np.float32)
        mean, pcs, amps, variance = pca_field(members)
        np.save(cubes / f"sr_{tile:03d}.npy", mean)
        np.save(cubes / f"std_{tile:03d}.npy", members.std(axis=0))
        for i, component in enumerate(pcs):
            np.save(cubes / f"pca{i}_{tile:03d}.npy", component)
        for kind, combiner in combiners.items():
            prefix = COMBINER_MODELS[kind].cube_prefix
            np.save(cubes / f"{prefix}_{tile:03d}.npy", combiner.apply_field(members))
        _accumulate_diagnostics(diagnostics, members, combiners)
        pca_amps[str(tile)] = [float(x) for x in amps]
        pca_var[str(tile)] = [float(x) for x in variance]
        progress(4 + tile + 1, 4 + GRID_SIDE * GRID_SIDE,
                 f"caching tile {tile + 1}/100")

    manifest = {
        "field_id": identifier, "ra": float(ra), "dec": float(dec),
        "field_size": FIELD_SIZE, "tile_size": TILE_SIZE, "grid_side": GRID_SIDE,
        "count": GRID_SIDE * GRID_SIDE, "member_labels": labels,
        "combiner_kinds": sorted(combiners), "combiner_state": combiner_state,
        "pca_n": min(3, max(0, ensemble.n_members - 1)),
        "pca_amps": pca_amps, "pca_var": pca_var,
    }
    _write_json(manifest_path(identifier), manifest)
    _write_json(root / "diagnostics.json", _diagnostic_payload(diagnostics, labels, combiners))
    return manifest


def refresh_real_field_combiners(
    identifier: str | None = None, *,
    progress: Callable[[int, int, str], None],
) -> dict[str, Any]:
    """Reapply fitted STARFULL combiners to an existing real-field cache.

    Member, mean, disagreement, and PCA cubes remain untouched.  This path
    loads one cached member stack at a time and never constructs the TensorFlow
    member networks, making post-fit real-star reevaluation both faster and
    substantially less memory-intensive than a full field recache.
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

    regime_dir = Path(Config.VIS_DIR) / "ensemble" / "starfull"
    combiners = {
        kind: comb
        for kind, spec in COMBINER_MODELS.items()
        if (comb := load_combiner(
            str(regime_dir), member_labels=labels,
            artifact_dir=spec.artifact_dir)) is not None
    }
    if not combiners:
        raise RuntimeError("no fitted STARFULL combiners")
    diagnostics = _diagnostic_accumulators(combiners, len(labels))
    count = int(manifest.get("count", 0))
    for tile in range(count):
        paths = [cubes / f"member{i}_{tile:03d}.npy" for i in range(len(labels))]
        if not all(path.is_file() for path in paths):
            raise RuntimeError(f"real-field member cubes missing for tile {tile + 1}")
        members = np.stack([np.load(path) for path in paths]).astype(np.float32)
        for kind, combiner in combiners.items():
            prefix = COMBINER_MODELS[kind].cube_prefix
            np.save(cubes / f"{prefix}_{tile:03d}.npy",
                    combiner.apply_field(members))
        _accumulate_diagnostics(diagnostics, members, combiners)
        progress(tile + 1, count, f"real-star combiner reevaluation {tile + 1}/{count}")

    combiner_state = {}
    for kind in combiners:
        spec = COMBINER_MODELS[kind]
        path = regime_dir / spec.artifact_dir / "combiner.npz"
        stat = path.stat()
        combiner_state[kind] = [int(stat.st_mtime_ns), int(stat.st_size)]
    manifest["combiner_kinds"] = sorted(combiners)
    manifest["combiner_state"] = combiner_state
    _write_json(manifest_path(identifier), manifest)
    _write_json(root / "diagnostics.json",
                _diagnostic_payload(diagnostics, labels, combiners))
    return manifest
