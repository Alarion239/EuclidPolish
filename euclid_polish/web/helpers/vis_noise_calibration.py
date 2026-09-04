"""Empirical, parent-grouped calibration of delivered-MER VIS noise.

The sampler manifest is the statistical boundary: every retained cutout has
an exact parent mosaic identity.  Patches within a cutout improve the local
noise estimate, but only parent mosaics are independent fit/holdout units.
The active artifact is deliberately the strict runtime payload; the candidate
keeps source masks, per-parent statistics, validation gates, and provenance.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.io.fits.verify import VerifyError
from scipy.ndimage import binary_dilation, gaussian_filter
from scipy.signal import correlate2d, fftconvolve

from euclid_polish.config import Config
from euclid_polish.photometry import adu_per_s_to_electrons_factor

VIS_NOISE_CANDIDATE_VERSION = 1
VIS_NOISE_RUNTIME_VERSION = 1
VIS_NOISE_RUNTIME_KIND = "euclid_mer_vis_noise"
VIS_NOISE_ESTIMATOR_VERSION = "source-masked-plane-mad-pair-covariance-psd-v2"
DEFAULT_MANIFEST_NAME = "vis_noise_sampling_manifest.json"
VIS_NOISE_SAMPLING_SUBDIR = "vis_noise_samples"
DEFAULT_TILE_SIZE = 256
DEFAULT_MAX_LAG = 8
DEFAULT_KERNEL_SIDE = 9
_REQUIRED_Q1_FIELDS = ("EDF-N", "EDF-S", "EDF-F")
_FIELD_SCALE_PROBABILITIES = np.asarray([0.0, 0.16, 0.5, 0.84, 1.0])
_TOTAL_ELECTRON_UNITS = {
    "e-",
    "e-/pixel",
    "electron",
    "electron/pixel",
    "electrons",
    "electrons/pixel",
}
_ARCHIVE_RATE_UNITS = {
    "adu/s",
    "adu/sec",
    "count/s",
    "counts/s",
    "e-/s",
    "electron/s",
    "electrons/s",
}
_RUNTIME_KEYS = {
    "kind",
    "version",
    "coloring_kernel",
    "residual_scale",
    "field_scale_quantiles",
    "owns_field_scale",
    "source_release",
    "estimator_version",
    "fingerprint",
}


class _SampleIdentityError(ValueError):
    """A manifest row cannot be proven to describe its local FITS bundle."""


def vis_noise_candidate_path() -> Path:
    return (
        Path(Config.DATA_DIR)
        / "population_comparison"
        / "calibrations"
        / "vis_noise_candidate.json"
    )


def active_vis_noise_path() -> Path:
    return (
        Path(Config.DATA_DIR)
        / "population_comparison"
        / "calibrations"
        / "vis_noise_active.json"
    )


def default_sampling_manifest_path() -> Path:
    return (
        Path(Config.EUCLID_SKY_DIR)
        / VIS_NOISE_SAMPLING_SUBDIR
        / DEFAULT_MANIFEST_NAME
    )


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_sampling_manifest(manifest: Mapping[str, Any]) -> tuple[str, int]:
    if manifest.get("kind") != "euclid_vis_noise_sampling":
        raise ValueError("Not an euclid_vis_noise_sampling manifest")
    if type(manifest.get("version")) is not int or manifest["version"] != 1:
        raise ValueError("VIS-noise sampling manifest version must be 1")
    source_release = str(manifest.get("source_release") or "").strip()
    if not source_release:
        raise ValueError("VIS-noise sampling manifest has no source_release")
    archive = manifest.get("archive")
    if not isinstance(archive, Mapping):
        raise ValueError("VIS-noise sampling manifest has no archive provenance")
    if archive.get("product_type") != "DpdMerBksMosaic":
        raise ValueError("VIS-noise samples must come from DpdMerBksMosaic")
    if archive.get("requested_release_name") != source_release:
        raise ValueError("Manifest release_name provenance does not match source_release")
    selection = manifest.get("selection")
    if not isinstance(selection, Mapping):
        raise ValueError("VIS-noise sampling manifest has no selection provenance")
    try:
        vis_pixels = int(selection["cutout_size_vis_pixels"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Manifest has no valid VIS cutout pixel size") from exc
    if vis_pixels <= 0:
        raise ValueError("Manifest VIS cutout pixel size must be positive")
    samples = manifest.get("samples")
    if not isinstance(samples, list):
        raise ValueError("VIS-noise sampling manifest samples must be a list")
    seen_ids: set[int] = set()
    for sample in samples:
        if not isinstance(sample, Mapping):
            raise ValueError("VIS-noise sampling manifest contains a non-object sample")
        try:
            sample_id = int(sample["sample_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("VIS-noise sample has no valid sample_id") from exc
        if sample_id < 0:
            raise ValueError("VIS-noise sample_id must be non-negative")
        if sample_id in seen_ids:
            raise ValueError(f"Duplicate VIS-noise sample_id {sample_id}")
        seen_ids.add(sample_id)
        parent_id = str(sample.get("parent_id") or "").strip()
        parent = sample.get("parent")
        if not parent_id or not isinstance(parent, Mapping):
            raise ValueError(f"VIS-noise sample {sample_id} has no parent provenance")
        if str(parent.get("parent_id") or "") != parent_id:
            raise ValueError(f"VIS-noise sample {sample_id} parent identity disagrees")
        if str(parent.get("release_name") or "") != source_release:
            raise ValueError(f"VIS-noise sample {sample_id} release identity disagrees")
        if parent.get("product_type") != "DpdMerBksMosaic":
            raise ValueError(f"VIS-noise sample {sample_id} product type disagrees")
        if not str(parent.get("mosaic_product_oid") or "").strip():
            raise ValueError(f"VIS-noise sample {sample_id} has no mosaic product OID")
        if sample.get("field") not in _REQUIRED_Q1_FIELDS:
            raise ValueError(f"VIS-noise sample {sample_id} has an unknown Q1 field")
        for coordinate in ("ra", "dec"):
            try:
                value = float(sample[coordinate])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"VIS-noise sample {sample_id} has no valid {coordinate}"
                ) from exc
            if not np.isfinite(value):
                raise ValueError(f"VIS-noise sample {sample_id} has no valid {coordinate}")
        if sample.get("status") in {"written", "cached"}:
            actual_shape = sample.get("actual_shape")
            if not isinstance(actual_shape, (list, tuple)):
                raise ValueError(
                    f"Completed VIS-noise sample {sample_id} has no actual_shape"
                )
            try:
                dimensions = tuple(int(value) for value in actual_shape)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Completed VIS-noise sample {sample_id} has no actual_shape"
                ) from exc
            tolerance = int(Config.Matching.DOWNLOAD_SIZE_TOL_PIXELS)
            if len(dimensions) != 2 or any(
                abs(dimension - vis_pixels) > tolerance for dimension in dimensions
            ):
                raise ValueError(
                    f"Completed VIS-noise sample {sample_id} actual_shape is invalid"
                )
    return source_release, vis_pixels


def _q1_fields_have_fit_and_holdout(fields: Mapping[str, Any]) -> bool:
    return all(int(fields.get(field, 0)) >= 2 for field in _REQUIRED_Q1_FIELDS)


def _canonical_runtime_fingerprint(payload: Mapping[str, Any]) -> str:
    core = {key: payload[key] for key in sorted(_RUNTIME_KEYS - {"fingerprint"})}
    encoded = json.dumps(
        core,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def runtime_vis_noise_payload(
    payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Project a review candidate (or active file) to the strict runtime schema."""
    source: Mapping[str, Any] | None = payload
    if source is None:
        source = _read_json(active_vis_noise_path())
    if source is None:
        raise ValueError("No VIS noise calibration payload is available")
    nested = source.get("runtime") if isinstance(source, Mapping) else None
    raw = nested if isinstance(nested, Mapping) else source
    runtime = {key: raw[key] for key in _RUNTIME_KEYS if key in raw}
    if set(runtime) != _RUNTIME_KEYS:
        raise ValueError(
            "Incomplete VIS noise runtime payload: "
            f"missing {sorted(_RUNTIME_KEYS - set(runtime))}"
        )
    if runtime["fingerprint"] != _canonical_runtime_fingerprint(runtime):
        raise ValueError("VIS noise runtime fingerprint does not match its payload")
    # Keep the helper usable while preserving the strict dependency boundary:
    # the core class remains the final schema/normalization validator.
    from euclid_polish.sky.observation.noise_calibration import VISNoiseCalibration

    return VISNoiseCalibration.from_payload(runtime).to_payload()


def _robust_sigma(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return float("nan")
    median = float(np.median(finite))
    sigma = 1.4826 * float(np.median(np.abs(finite - median)))
    if np.isfinite(sigma) and sigma > 0.0:
        return sigma
    fallback = float(np.std(finite))
    return fallback if fallback > 0.0 else float("nan")


def _source_masked_residual(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Data-derived source mask and plane-only background residual.

    A smoothed image is used only to *detect* compact sources.  Noise
    measurement is performed on the original pixels after an iteratively
    clipped constant-plus-plane fit, preserving the delivered MER covariance
    and power on the 0.25--8 arcsec calibration range.
    """
    data = np.asarray(image, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError(f"VIS image must be 2-D, got {data.shape}")
    finite = np.isfinite(data)
    if np.count_nonzero(finite) < max(64, data.size // 4):
        raise ValueError("VIS patch has too few finite pixels")
    fill = float(np.median(data[finite]))
    work = np.where(finite, data, fill)
    smooth_sigma = max(3.0, min(12.0, min(data.shape) / 24.0))
    detection_background = gaussian_filter(work, smooth_sigma, mode="reflect")
    detection_residual = work - detection_background
    first_sigma = _robust_sigma(detection_residual[finite])
    if not np.isfinite(first_sigma) or first_sigma <= 0.0:
        raise ValueError("VIS patch has degenerate background variance")
    # A lower positive threshold catches source wings; the symmetric high
    # threshold removes cosmic rays, zeroed saturation cores, and bad pixels.
    seeds = (
        (detection_residual > 4.0 * first_sigma)
        | (np.abs(detection_residual) > 8.0 * first_sigma)
        | ~finite
    )
    mask = binary_dilation(seeds, iterations=4)
    usable = ~mask
    if np.count_nonzero(usable) < max(64, data.size // 5):
        raise ValueError("Source mask leaves too few background pixels")
    yy, xx = np.indices(data.shape, dtype=np.float64)
    xx = (xx - 0.5 * (data.shape[1] - 1)) / max(1.0, data.shape[1] - 1)
    yy = (yy - 0.5 * (data.shape[0] - 1)) / max(1.0, data.shape[0] - 1)
    design = np.column_stack((
        np.ones(np.count_nonzero(finite)), xx[finite], yy[finite],
    ))
    values = work[finite]
    fit_usable = usable[finite].copy()
    coefficients = np.asarray([float(np.median(values[fit_usable])), 0.0, 0.0])
    for _ in range(5):
        if np.count_nonzero(fit_usable) < 64:
            break
        coefficients, *_ = np.linalg.lstsq(
            design[fit_usable], values[fit_usable], rcond=None,
        )
        fit_residual = values - design @ coefficients
        centre = float(np.median(fit_residual[fit_usable]))
        sigma = _robust_sigma(fit_residual[fit_usable])
        if not np.isfinite(sigma) or sigma <= 0.0:
            break
        clipped = usable[finite] & (np.abs(fit_residual - centre) <= 5.0 * sigma)
        if np.array_equal(clipped, fit_usable):
            break
        fit_usable = clipped
    final_usable = np.zeros(data.shape, dtype=bool)
    final_usable[finite] = fit_usable
    mask = ~final_usable
    usable = final_usable
    plane = coefficients[0] + coefficients[1] * xx + coefficients[2] * yy
    residual = work - plane
    residual -= float(np.median(residual[usable]))
    return residual, mask


def _covariance_2d(
    residual: np.ndarray,
    mask: np.ndarray,
    *,
    max_lag: int,
) -> np.ndarray:
    usable = ~np.asarray(mask, dtype=bool)
    values = np.asarray(residual, dtype=np.float64)
    variance = float(np.mean(np.square(values[usable])))
    if not np.isfinite(variance) or variance <= 0.0:
        raise ValueError("Cannot normalize a degenerate covariance")
    side = 2 * int(max_lag) + 1
    covariance = np.full((side, side), np.nan, dtype=np.float64)
    height, width = values.shape
    for row, dy in enumerate(range(-max_lag, max_lag + 1)):
        ay = slice(max(0, dy), min(height, height + dy))
        by = slice(max(0, -dy), min(height, height - dy))
        for col, dx in enumerate(range(-max_lag, max_lag + 1)):
            ax = slice(max(0, dx), min(width, width + dx))
            bx = slice(max(0, -dx), min(width, width - dx))
            valid = usable[ay, ax] & usable[by, bx]
            if np.count_nonzero(valid) < 64:
                continue
            covariance[row, col] = float(np.mean(
                values[ay, ax][valid] * values[by, bx][valid]
            )) / variance
    covariance[max_lag, max_lag] = 1.0
    return covariance


def _radial_covariance(matrix: np.ndarray) -> np.ndarray:
    array = np.asarray(matrix, dtype=np.float64)
    centre_y, centre_x = np.asarray(array.shape) // 2
    yy, xx = np.indices(array.shape)
    radius = np.rint(np.hypot(yy - centre_y, xx - centre_x)).astype(int)
    maximum = min(centre_y, centre_x)
    return np.asarray([
        float(np.nanmean(array[radius == lag]))
        for lag in range(maximum + 1)
    ])


def _power_edges(tile_size: int, bins: int = 18) -> np.ndarray:
    minimum = 1.0 / max(16, int(tile_size))
    return np.geomspace(minimum, 0.5, int(bins) + 1)


def _normalized_psd(
    residual: np.ndarray,
    mask: np.ndarray,
    *,
    radial_edges: np.ndarray,
    bins_2d: int = 33,
) -> tuple[np.ndarray, np.ndarray]:
    """Pair-normalized lag-window PSD, corrected for source-mask geometry."""
    values = np.asarray(residual, dtype=np.float64)
    usable = ~np.asarray(mask, dtype=bool)
    height, width = values.shape
    centred = values - float(np.median(values[usable]))
    weighted = np.where(usable, centred, 0.0)
    weights = usable.astype(np.float64)
    numerator = fftconvolve(weighted, weighted[::-1, ::-1], mode="full")
    pair_count = fftconvolve(weights, weights[::-1, ::-1], mode="full")
    covariance = np.divide(
        numerator,
        pair_count,
        out=np.zeros_like(numerator),
        where=pair_count >= 64.0,
    )
    lag_y = np.arange(-(height - 1), height, dtype=np.float64)
    lag_x = np.arange(-(width - 1), width, dtype=np.float64)
    # A compact Bartlett lag window controls the variance of weakly supported
    # edge lags while retaining scales through half the tile width.
    window_y = np.clip(1.0 - np.abs(lag_y) / max(1.0, height / 2.0), 0.0, 1.0)
    window_x = np.clip(1.0 - np.abs(lag_x) / max(1.0, width / 2.0), 0.0, 1.0)
    covariance *= np.outer(window_y, window_x)
    power = np.fft.fftshift(
        np.fft.fft2(np.fft.ifftshift(covariance)).real,
    )
    power = np.maximum(power, 0.0)
    if not np.any(power > 0.0):
        raise ValueError("Mask-corrected VIS power spectrum is degenerate")
    fy = np.fft.fftshift(np.fft.fftfreq(power.shape[0]))
    fx = np.fft.fftshift(np.fft.fftfreq(power.shape[1]))
    ky, kx = np.meshgrid(fy, fx, indexing="ij")
    radius = np.hypot(kx, ky)
    radial_sum, _ = np.histogram(radius, bins=radial_edges, weights=power)
    radial_count, _ = np.histogram(radius, bins=radial_edges)
    radial = radial_sum / np.maximum(radial_count, 1)
    radial = np.maximum(radial, np.finfo(np.float64).tiny)
    radial /= float(np.sum(radial))
    edges_2d = np.linspace(-0.5, 0.5, int(bins_2d) + 1)
    histogram, _, _ = np.histogram2d(
        ky.ravel(), kx.ravel(), bins=(edges_2d, edges_2d),
        weights=power.ravel(),
    )
    total = float(histogram.sum())
    if total > 0.0:
        histogram /= total
    return radial, histogram


def _iter_tiles(image: np.ndarray, tile_size: int) -> list[np.ndarray]:
    data = np.asarray(image)
    height, width = data.shape
    side = min(int(tile_size), height, width)
    if side <= 0:
        raise ValueError("tile_size must be positive")
    if height < tile_size or width < tile_size:
        return [data]

    # SODA commonly returns one or two pixels more than requested.  Remove
    # only that tolerance-sized excess around the centre before partitioning;
    # otherwise an appended edge window would overlap its neighbour by 255 of
    # 256 pixels and heavily over-weight a border strip.
    tolerance = int(Config.Matching.DOWNLOAD_SIZE_TOL_PIXELS)

    def normalized_slice(length: int) -> slice:
        nearest_multiple = max(1, int(round(length / side))) * side
        excess = length - nearest_multiple
        if 0 < excess <= tolerance:
            start = excess // 2
            return slice(start, start + nearest_multiple)
        return slice(None)

    data = data[normalized_slice(height), normalized_slice(width)]
    height, width = data.shape

    def starts(length: int) -> list[int]:
        count = max(1, int(math.ceil(length / side)))
        if count == 1:
            return [0]
        return np.rint(np.linspace(0, length - side, count)).astype(int).tolist()

    return [
        data[y:y + side, x:x + side]
        for y in starts(height)
        for x in starts(width)
    ]


def _validate_fits_identity(
    header: fits.Header,
    image_shape: tuple[int, ...],
    sample: Mapping[str, Any],
    manifest: Mapping[str, Any],
    path: Path,
) -> None:
    source_release, vis_pixels = _validate_sampling_manifest(manifest)
    parent = sample.get("parent")
    if not isinstance(parent, Mapping):
        raise _SampleIdentityError(f"{path}: sample has no parent provenance")
    expected = {
        "POS_ID": int(sample["sample_id"]),
        "PARENT": str(sample["parent_id"]),
        "RELEASE": str(parent.get("release_name") or source_release),
        "VIS_PIX": vis_pixels,
        "MOSAIC_PRODUCT_OID": str(parent["mosaic_product_oid"]),
        "PRODTYPE": str(parent["product_type"]),
    }
    for key, wanted in expected.items():
        actual = header.get(key)
        if str(actual) != str(wanted):
            raise _SampleIdentityError(
                f"{path}: FITS {key}={actual!r} does not match manifest {wanted!r}"
            )
    for key, sample_key in (("RA", "ra"), ("DEC", "dec")):
        try:
            actual = float(str(header[key]))
            wanted = float(sample[sample_key])
        except (KeyError, TypeError, ValueError) as exc:
            raise _SampleIdentityError(
                f"{path}: FITS/manifest {key} coordinate is missing or invalid"
            ) from exc
        if not np.isclose(actual, wanted, rtol=0.0, atol=1e-8):
            raise _SampleIdentityError(
                f"{path}: FITS {key}={actual!r} does not match manifest {wanted!r}"
            )
    size_tolerance = int(Config.Matching.DOWNLOAD_SIZE_TOL_PIXELS)
    if len(image_shape) != 2 or any(
        abs(int(actual) - vis_pixels) > size_tolerance for actual in image_shape
    ):
        raise _SampleIdentityError(
            f"{path}: VIS shape {image_shape} does not match "
            f"VIS_PIX={vis_pixels} within tolerance {size_tolerance}"
        )
    actual_shape = sample.get("actual_shape")
    if actual_shape is not None:
        try:
            recorded_shape = tuple(int(value) for value in actual_shape)
        except (TypeError, ValueError) as exc:
            raise _SampleIdentityError(
                f"{path}: manifest actual_shape is invalid"
            ) from exc
        if recorded_shape != image_shape:
            raise _SampleIdentityError(
                f"{path}: VIS shape {image_shape} does not match recorded "
                f"actual_shape {recorded_shape}"
            )


def _load_vis_image(
    sample: Mapping[str, Any],
    manifest_dir: Path,
    manifest: Mapping[str, Any] | None = None,
) -> np.ndarray:
    raw_path = sample.get("output_path")
    if not raw_path:
        raise ValueError(f"sample {sample.get('sample_id')} has no output_path")
    path = Path(str(raw_path))
    if not path.is_absolute():
        path = manifest_dir / path
    if not path.is_file():
        # FASRC manifests contain absolute paths from the cluster.  After the
        # manifest and cutouts are synchronized locally, rebase the stable
        # bundle basename under the manifest's cutouts directory.
        rebased = manifest_dir / Config.EuclidSky.CUTOUTS_SUBDIR / path.name
        if rebased.is_file():
            path = rebased
    if not path.is_file():
        raise _SampleIdentityError(f"VIS sample bundle does not exist: {path}")
    try:
        with fits.open(path, memmap=False) as hdul:
            hdul.verify("exception")
            hdu: Any = None
            with np.errstate(all="ignore"):
                try:
                    hdu = hdul["VIS"]
                except (KeyError, IndexError):
                    if manifest is None:
                        for hdu_index in range(len(hdul)):
                            candidate: Any = hdul[hdu_index]
                            if candidate.data is not None:
                                hdu = candidate
                                break
            if hdu is None or hdu.data is None:
                raise _SampleIdentityError(f"{path} has no VIS image")
            data = np.asarray(hdu.data, dtype=np.float64)
            header = hdu.header.copy()
            primary_hdu: Any = hdul[0]
            primary_header = primary_hdu.header.copy()
    except _SampleIdentityError:
        raise
    except (OSError, IndexError, ValueError, VerifyError) as exc:
        raise _SampleIdentityError(f"VIS sample bundle is unreadable: {path}") from exc
    if manifest is not None:
        _validate_fits_identity(
            primary_header, data.shape, sample, manifest, path,
        )
    unit = str(sample.get("data_unit") or header.get("BUNIT") or "").strip().lower()
    unit = "".join(unit.split())
    if unit in _TOTAL_ELECTRON_UNITS:
        return data
    # Q1 MER cutouts commonly omit BUNIT but carry MAGZERO=24.6.  A finite
    # MAGZERO is the required archive-rate evidence in that one legacy case;
    # a non-empty, unknown unit is never guessed.
    if unit and unit not in _ARCHIVE_RATE_UNITS:
        raise _SampleIdentityError(f"{path}: unsupported VIS BUNIT {unit!r}")
    raw_zeropoint = header.get("MAGZERO")
    try:
        zeropoint = float(raw_zeropoint)
    except (TypeError, ValueError) as exc:
        raise _SampleIdentityError(
            f"{path}: archive rate image requires a finite MAGZERO"
        ) from exc
    if not np.isfinite(zeropoint):
        raise _SampleIdentityError(
            f"{path}: archive rate image requires a finite MAGZERO"
        )
    data *= adu_per_s_to_electrons_factor(zeropoint, Config.BAND_VIS)
    return data


def _sample_diagnostics(
    sample: Mapping[str, Any],
    *,
    manifest_dir: Path,
    tile_size: int,
    max_lag: int,
    radial_edges: np.ndarray,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    image = _load_vis_image(sample, manifest_dir, manifest)
    tiles = []
    failures = 0
    for tile in _iter_tiles(image, tile_size):
        try:
            residual, mask = _source_masked_residual(tile)
            usable = ~mask
            rms = _robust_sigma(residual[usable])
            covariance = _covariance_2d(residual, mask, max_lag=max_lag)
            radial_covariance = _radial_covariance(covariance)
            radial_power, power_2d = _normalized_psd(
                residual, mask, radial_edges=radial_edges,
            )
        except ValueError:
            failures += 1
            continue
        tiles.append({
            "rms_e": float(rms),
            "unmasked_pixels": int(np.count_nonzero(usable)),
            "total_pixels": int(mask.size),
            "covariance_2d": covariance,
            "lag": radial_covariance,
            "power": radial_power,
            "power_2d": power_2d,
        })
    if not tiles:
        raise ValueError(f"sample {sample.get('sample_id')} has no usable noise tiles")
    return {
        "parent_id": str(sample["parent_id"]),
        "field": str(sample.get("field") or "unknown"),
        "sample_id": int(sample.get("sample_id", -1)),
        "tiles": tiles,
        "failed_tiles": failures,
    }


def _nanmedian_stack(values: Sequence[np.ndarray]) -> np.ndarray:
    with np.errstate(all="ignore"):
        result = np.nanmedian(np.stack(values), axis=0)
    return np.asarray(result, dtype=np.float64)


def _parent_diagnostics(samples: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        grouped[str(sample["parent_id"])].append(sample)
    parents: list[dict[str, Any]] = []
    for parent_id in sorted(grouped):
        group = grouped[parent_id]
        tiles = [tile for sample in group for tile in sample["tiles"]]
        rms_values = np.asarray([tile["rms_e"] for tile in tiles], dtype=np.float64)
        unmasked = sum(int(tile["unmasked_pixels"]) for tile in tiles)
        total = sum(int(tile["total_pixels"]) for tile in tiles)
        parents.append({
            "parent_id": parent_id,
            "field": str(group[0]["field"]),
            "sample_count": len(group),
            "tile_count": len(tiles),
            "failed_tile_count": sum(int(sample["failed_tiles"]) for sample in group),
            "unmasked_pixels": unmasked,
            "masked_fraction": 1.0 - unmasked / total,
            "patch_rms_e": rms_values,
            "rms_e": float(np.median(rms_values)),
            "rms_p16_e": float(np.percentile(rms_values, 16)),
            "rms_p84_e": float(np.percentile(rms_values, 84)),
            "covariance_2d": _nanmedian_stack([
                tile["covariance_2d"] for tile in tiles
            ]),
            "lag": _nanmedian_stack([tile["lag"] for tile in tiles]),
            "power": _nanmedian_stack([tile["power"] for tile in tiles]),
            "power_2d": _nanmedian_stack([tile["power_2d"] for tile in tiles]),
        })
    return parents


def _weighted_quantiles(
    values: np.ndarray,
    weights: np.ndarray,
    probabilities: np.ndarray,
) -> np.ndarray:
    order = np.argsort(values)
    ordered = np.asarray(values, dtype=np.float64)[order]
    ordered_weights = np.asarray(weights, dtype=np.float64)[order]
    cumulative = np.cumsum(ordered_weights) - 0.5 * ordered_weights
    cumulative /= float(np.sum(ordered_weights))
    return np.interp(probabilities, cumulative, ordered, left=ordered[0], right=ordered[-1])


def _parent_weighted_rms(parents: Sequence[Mapping[str, Any]]) -> tuple[float, np.ndarray]:
    values = np.concatenate([
        np.asarray(parent["patch_rms_e"], dtype=np.float64) for parent in parents
    ])
    weights = np.concatenate([
        np.full(len(parent["patch_rms_e"]), 1.0 / len(parent["patch_rms_e"]))
        for parent in parents
    ])
    quantiles = _weighted_quantiles(values, weights, _FIELD_SCALE_PROBABILITIES)
    median = float(quantiles[2])
    factors = np.maximum(quantiles / median, np.finfo(np.float64).eps)
    factors[2] = 1.0
    factors = np.maximum.accumulate(factors)
    return median, factors


def _spectral_factor_kernel(covariance: np.ndarray, *, side: int) -> np.ndarray:
    if side < 1 or side % 2 == 0:
        raise ValueError("kernel side must be a positive odd integer")
    covariance = np.asarray(covariance, dtype=np.float64)
    covariance = np.nan_to_num(
        0.5 * (covariance + covariance[::-1, ::-1]), nan=0.0,
    )
    centre = tuple(np.asarray(covariance.shape) // 2)
    covariance[centre] = 1.0
    fft_side = 1
    while fft_side < max(64, 4 * max(covariance.shape)):
        fft_side *= 2
    embedded = np.zeros((fft_side, fft_side), dtype=np.float64)
    y0 = fft_side // 2 - covariance.shape[0] // 2
    x0 = fft_side // 2 - covariance.shape[1] // 2
    embedded[y0:y0 + covariance.shape[0], x0:x0 + covariance.shape[1]] = covariance
    spectrum = np.real(np.fft.fft2(np.fft.ifftshift(embedded)))
    spectrum = np.maximum(spectrum, 0.0)
    amplitude = np.sqrt(spectrum)
    full = np.fft.fftshift(np.fft.ifft2(amplitude).real)
    half = side // 2
    cy, cx = np.asarray(full.shape) // 2
    kernel = full[cy - half:cy + half + 1, cx - half:cx + half + 1]
    norm = float(np.linalg.norm(kernel))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("Empirical covariance has no finite spectral factor")
    return kernel / norm


def _kernel_covariance(kernel: np.ndarray, max_lag: int) -> np.ndarray:
    correlation = correlate2d(kernel, kernel, mode="full")
    cy, cx = np.asarray(correlation.shape) // 2
    output = np.zeros((2 * max_lag + 1, 2 * max_lag + 1), dtype=np.float64)
    for row, dy in enumerate(range(-max_lag, max_lag + 1)):
        for col, dx in enumerate(range(-max_lag, max_lag + 1)):
            yy, xx = cy + dy, cx + dx
            if 0 <= yy < correlation.shape[0] and 0 <= xx < correlation.shape[1]:
                output[row, col] = correlation[yy, xx]
    return output / float(correlation[cy, cx])


def _kernel_power(kernel: np.ndarray, radial_edges: np.ndarray, tile_size: int) -> np.ndarray:
    # The pair-corrected empirical covariance has ``2 * tile_size - 1`` lag
    # samples.  Evaluate the compact kernel on the same odd Fourier grid so
    # narrow low-frequency annuli cannot be empty in only the model curve.
    spectrum_side = 2 * int(tile_size) - 1
    canvas = np.zeros((spectrum_side, spectrum_side), dtype=np.float64)
    half = kernel.shape[0] // 2
    centre = spectrum_side // 2
    canvas[
        centre - half:centre + half + 1,
        centre - half:centre + half + 1,
    ] = kernel
    power = np.square(np.abs(np.fft.fft2(np.fft.ifftshift(canvas))))
    fy = np.fft.fftfreq(spectrum_side)
    fx = np.fft.fftfreq(spectrum_side)
    ky, kx = np.meshgrid(fy, fx, indexing="ij")
    radius = np.hypot(kx, ky)
    sums, _ = np.histogram(radius, bins=radial_edges, weights=power)
    counts, _ = np.histogram(radius, bins=radial_edges)
    result = sums / np.maximum(counts, 1)
    result = np.maximum(result, np.finfo(np.float64).tiny)
    return result / float(np.sum(result))


def _parent_split(
    parents: Sequence[Mapping[str, Any]], seed: int,
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    def stable_order(parent: Mapping[str, Any]) -> str:
        return hashlib.sha256(
            f"{int(seed)}:{parent['parent_id']}".encode()
        ).hexdigest()

    by_field: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for parent in parents:
        by_field[str(parent.get("field") or "unknown")].append(parent)
    train: list[Mapping[str, Any]] = []
    holdout: list[Mapping[str, Any]] = []
    for field in sorted(by_field):
        ordered = sorted(by_field[field], key=stable_order)
        if len(ordered) < 2:
            train.extend(ordered)
            continue
        holdout_count = max(1, int(round(0.2 * len(ordered))))
        holdout_count = min(holdout_count, len(ordered) - 1)
        holdout.extend(ordered[:holdout_count])
        train.extend(ordered[holdout_count:])
    if not holdout:
        # Three singleton fields cannot each be represented on both sides;
        # retain two train parents and choose one deterministic global holdout.
        ordered = sorted(train, key=stable_order)
        holdout = ordered[:1]
        train = ordered[1:]
    if len(train) < 2:
        raise ValueError("Parent holdout split must retain at least 2 fit parents")
    return sorted(train, key=stable_order), sorted(holdout, key=stable_order)


def _median_parent_array(
    parents: Sequence[Mapping[str, Any]], key: str,
) -> np.ndarray:
    return _nanmedian_stack([
        np.asarray(parent[key], dtype=np.float64) for parent in parents
    ])


def _fit_validation(
    train: Sequence[Mapping[str, Any]],
    holdout: Sequence[Mapping[str, Any]],
    *,
    radial_edges: np.ndarray,
    tile_size: int,
    max_lag: int,
    kernel_side: int,
) -> tuple[dict[str, Any], np.ndarray]:
    train_covariance = _median_parent_array(train, "covariance_2d")
    train_kernel = _spectral_factor_kernel(train_covariance, side=kernel_side)
    model_lag = _radial_covariance(_kernel_covariance(train_kernel, max_lag))
    real_lag = _median_parent_array(holdout, "lag")
    lag_rows = np.stack([np.asarray(parent["lag"]) for parent in holdout])
    lag_p16 = np.percentile(lag_rows, 16, axis=0)
    lag_p84 = np.percentile(lag_rows, 84, axis=0)
    lag_error = np.abs(model_lag - real_lag)
    within = (model_lag >= lag_p16) & (model_lag <= lag_p84)

    model_power = _kernel_power(train_kernel, radial_edges, tile_size)
    real_power = _median_parent_array(holdout, "power")
    real_power = np.maximum(real_power, np.finfo(np.float64).tiny)
    real_power /= float(np.sum(real_power))
    log_ratio = np.log10(model_power / real_power)

    train_rms, _ = _parent_weighted_rms(train)
    holdout_rms, _ = _parent_weighted_rms(holdout)
    ratio = train_rms / holdout_rms
    angular_scale = (
        Config.BAND_VIS.pixel_scale_lr_arcsec
        / np.sqrt(radial_edges[:-1] * radial_edges[1:])
    )

    def band_metrics(scale_mask: np.ndarray) -> dict[str, Any]:
        selected_model = model_power[scale_mask]
        selected_real = real_power[scale_mask]
        selected_log_ratio = log_ratio[scale_mask]
        if not len(selected_model):
            return {
                "bin_count": 0,
                "median_abs_log10_ratio": None,
                "p90_abs_log10_ratio": None,
                "shape_overlap": None,
                "variance_ratio": None,
            }
        model_fraction = float(np.sum(selected_model))
        real_fraction = float(np.sum(selected_real))
        model_shape = selected_model / model_fraction
        real_shape = selected_real / real_fraction
        abs_log_ratio = np.abs(selected_log_ratio)
        return {
            "bin_count": int(np.count_nonzero(scale_mask)),
            "median_abs_log10_ratio": float(np.median(abs_log_ratio)),
            "p90_abs_log10_ratio": float(np.percentile(abs_log_ratio, 90)),
            "shape_overlap": float(np.minimum(model_shape, real_shape).sum()),
            "variance_ratio": float(
                (train_rms * train_rms * model_fraction)
                / (holdout_rms * holdout_rms * real_fraction)
            ),
            "model_power_fraction": model_fraction,
            "real_power_fraction": real_fraction,
        }

    calibrated_scale = (angular_scale >= 0.25) & (angular_scale <= 8.0)
    primary_power_metrics = band_metrics(calibrated_scale)
    large_scale_metrics = band_metrics(angular_scale > 8.0)
    subpixel_scale_metrics = band_metrics(angular_scale < 0.25)
    parent_holdout_rms = np.asarray([parent["rms_e"] for parent in holdout])
    validation = {
        "grouped_by": "parent_id",
        "train_parent_ids": [str(parent["parent_id"]) for parent in train],
        "holdout_parent_ids": [str(parent["parent_id"]) for parent in holdout],
        "rms": {
            "model": train_rms,
            "real": holdout_rms,
            "ratio": ratio,
            "p16": float(np.percentile(parent_holdout_rms, 16)),
            "p50": float(np.percentile(parent_holdout_rms, 50)),
            "p84": float(np.percentile(parent_holdout_rms, 84)),
        },
        "lag": {
            "lags_pixels": list(range(len(real_lag))),
            "model": model_lag.tolist(),
            "real": real_lag.tolist(),
            "p16": lag_p16.tolist(),
            "p84": lag_p84.tolist(),
            "median_abs_error": float(np.nanmedian(lag_error)),
            "max_abs_error": float(np.nanmax(lag_error)),
            "within_interval_fraction": float(np.mean(within)),
        },
        "power": {
            "angular_scale_arcsec": angular_scale.tolist(),
            "model": model_power.tolist(),
            "real": real_power.tolist(),
            "log10_ratio": log_ratio.tolist(),
            "evaluation_range_arcsec": [0.25, 8.0],
            **primary_power_metrics,
            "large_scale_gt_8_arcsec": large_scale_metrics,
            "subpixel_lt_0p25_arcsec": subpixel_scale_metrics,
        },
    }
    return validation, train_kernel


def _quality_gates(validation: Mapping[str, Any]) -> dict[str, Any]:
    rms_ratio = float(validation["rms"]["ratio"])
    lag = validation["lag"]
    power = validation["power"]
    gates = {
        "rms_ratio_0p95_1p05": 0.95 <= rms_ratio <= 1.05,
        "lag_median_abs_error_le_0p03": float(lag["median_abs_error"]) <= 0.03,
        "lag_max_abs_error_le_0p07": float(lag["max_abs_error"]) <= 0.07,
        "lag_within_interval_fraction_ge_0p80": (
            float(lag["within_interval_fraction"]) >= 0.80
        ),
        "power_median_abs_log10_ratio_le_0p05": (
            float(power["median_abs_log10_ratio"]) <= 0.05
        ),
        "power_p90_abs_log10_ratio_le_0p10": (
            float(power["p90_abs_log10_ratio"]) <= 0.10
        ),
        "power_shape_overlap_ge_0p98": float(power["shape_overlap"]) >= 0.98,
        "power_variance_ratio_0p90_1p10": (
            0.90 <= float(power["variance_ratio"]) <= 1.10
        ),
    }
    return {**gates, "passed": all(gates.values())}


def _review_parent(parent: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "parent_id": str(parent["parent_id"]),
        "field": str(parent["field"]),
        "sample_count": int(parent["sample_count"]),
        "tile_count": int(parent["tile_count"]),
        "failed_tile_count": int(parent["failed_tile_count"]),
        "unmasked_pixels": int(parent["unmasked_pixels"]),
        "masked_fraction": float(parent["masked_fraction"]),
        "rms_e": float(parent["rms_e"]),
        "rms_p16_e": float(parent["rms_p16_e"]),
        "rms_p84_e": float(parent["rms_p84_e"]),
        "patch_rms_e": np.asarray(parent["patch_rms_e"]).tolist(),
    }


def fit_vis_noise_candidate(
    progress: Callable[[int, int, str], None] | None = None,
    *,
    manifest_file: str | os.PathLike[str] | None = None,
    tile_size: int = DEFAULT_TILE_SIZE,
    max_lag: int = DEFAULT_MAX_LAG,
    kernel_side: int = DEFAULT_KERNEL_SIDE,
    split_seed: int = 28012025,
) -> dict[str, Any]:
    """Fit, validate, fingerprint, and atomically persist a VIS-noise candidate."""
    manifest_path = Path(manifest_file or default_sampling_manifest_path())
    manifest = _read_json(manifest_path)
    if not manifest:
        raise ValueError(f"No readable VIS-noise sampling manifest at {manifest_path}")
    _validate_sampling_manifest(manifest)
    samples = [
        sample for sample in manifest.get("samples", [])
        if sample.get("status") in {"written", "cached"}
        and sample.get("parent_id")
    ]
    if not samples:
        raise ValueError("Sampling manifest has no completed VIS samples")
    radial_edges = _power_edges(int(tile_size))
    diagnostics: list[dict[str, Any]] = []
    failures: list[str] = []
    total = len(samples)
    for index, sample in enumerate(samples):
        if progress:
            progress(index, total + 1, f"VIS sample {index + 1}/{total}")
        try:
            diagnostics.append(_sample_diagnostics(
                sample,
                manifest_dir=manifest_path.parent,
                tile_size=int(tile_size),
                max_lag=int(max_lag),
                radial_edges=radial_edges,
                manifest=manifest,
            ))
        except _SampleIdentityError:
            # Never fit around a stale basename, mismatched parent, release,
            # location, or cutout size.  Those are provenance failures rather
            # than statistically rejectable noisy patches.
            raise
        except (OSError, ValueError) as exc:
            failures.append(f"sample {sample.get('sample_id')}: {exc}")
    parents = _parent_diagnostics(diagnostics)
    if len(parents) < 3:
        raise ValueError(
            "VIS noise calibration needs at least 3 independent parent mosaics; "
            f"found {len(parents)}"
        )
    train, holdout = _parent_split(parents, int(split_seed))
    validation, _train_kernel = _fit_validation(
        train,
        holdout,
        radial_edges=radial_edges,
        tile_size=int(tile_size),
        max_lag=int(max_lag),
        kernel_side=int(kernel_side),
    )
    gates = _quality_gates(validation)
    fields = Counter(str(parent["field"]) for parent in parents)
    gates["all_q1_fields_ge_2_parents"] = _q1_fields_have_fit_and_holdout(fields)
    gates["passed"] = all(
        bool(value) for name, value in gates.items() if name != "passed"
    )

    final_covariance = _median_parent_array(parents, "covariance_2d")
    final_kernel = _spectral_factor_kernel(final_covariance, side=int(kernel_side))
    residual_scale, field_scale_quantiles = _parent_weighted_rms(parents)
    runtime_core = {
        "kind": VIS_NOISE_RUNTIME_KIND,
        "version": VIS_NOISE_RUNTIME_VERSION,
        "coloring_kernel": final_kernel.tolist(),
        "residual_scale": residual_scale,
        "field_scale_quantiles": field_scale_quantiles.tolist(),
        "owns_field_scale": True,
        "source_release": str(manifest.get("source_release") or "unknown"),
        "estimator_version": VIS_NOISE_ESTIMATOR_VERSION,
    }
    runtime = {
        **runtime_core,
        "fingerprint": _canonical_runtime_fingerprint(runtime_core),
    }
    # Validate the final strict boundary before writing either artifact.
    runtime = runtime_vis_noise_payload({"runtime": runtime})

    total_unmasked = sum(int(parent["unmasked_pixels"]) for parent in parents)
    total_pixels = sum(
        int(round(parent["unmasked_pixels"] / (1.0 - parent["masked_fraction"])))
        for parent in parents
    )
    warnings = list(failures)
    if not _q1_fields_have_fit_and_holdout(fields):
        warnings.append(
            "Calibration requires at least 2 independent parents in each of "
            "EDF-N, EDF-S, and EDF-F"
        )
    if not gates["passed"]:
        warnings.append("Grouped parent holdout quality gates did not all pass")
    parent_rms = np.asarray([parent["rms_e"] for parent in parents])
    final_power = _median_parent_array(parents, "power")
    final_power /= float(np.sum(final_power))
    final_power_2d = _median_parent_array(parents, "power_2d")
    final_power_2d /= float(np.sum(final_power_2d))
    frequency_edges_2d = np.linspace(-0.5, 0.5, final_power_2d.shape[0] + 1)
    candidate = {
        "version": VIS_NOISE_CANDIDATE_VERSION,
        "kind": VIS_NOISE_RUNTIME_KIND,
        "fingerprint": runtime["fingerprint"],
        "fitted_at": datetime.now(UTC).isoformat(),
        "source_release": runtime["source_release"],
        "residual_scale": runtime["residual_scale"],
        "field_scale_quantiles": runtime["field_scale_quantiles"],
        "owns_field_scale": runtime["owns_field_scale"],
        "active": False,
        "valid": bool(gates["passed"] and len(parents) >= 3),
        "warnings": warnings,
        "runtime": runtime,
        "runtime_semantics": {
            "residual_scale": (
                "absolute target 1.4826*MAD RMS in electrons per 0.10 arcsec "
                "MER pixel after source masking"
            ),
            "field_scale_quantiles": {
                "probabilities": _FIELD_SCALE_PROBABILITIES.tolist(),
                "weighting": "equal total weight per parent mosaic across its patches",
            },
            "coloring_kernel": (
                f"{int(kernel_side)}x{int(kernel_side)} real-space "
                "L2-normalized spectral factor"
            ),
        },
        "sample_summary": {
            "independent_parent_count": len(parents),
            "sample_count": len(diagnostics),
            "tile_count": sum(int(parent["tile_count"]) for parent in parents),
            "unmasked_pixels": total_unmasked,
            "masked_fraction": 1.0 - total_unmasked / total_pixels,
            "train_parent_count": len(train),
            "holdout_parent_count": len(holdout),
            "fields": dict(sorted(fields.items())),
        },
        "fit": {
            "rms_e": {
                "p16": float(np.percentile(parent_rms, 16)),
                "p50": float(np.percentile(parent_rms, 50)),
                "p84": float(np.percentile(parent_rms, 84)),
                "minimum": float(np.min(parent_rms)),
                "maximum": float(np.max(parent_rms)),
                "parent_weighted_patch_median": residual_scale,
            },
            "parents": [_review_parent(parent) for parent in parents],
            "covariance_2d": {
                "lags_pixels": list(range(-int(max_lag), int(max_lag) + 1)),
                "matrix": final_covariance.tolist(),
            },
            "psd_2d": {
                "frequency_edges_cycles_per_pixel": frequency_edges_2d.tolist(),
                "frequency_edges_cycles_per_arcsec": (
                    frequency_edges_2d / Config.BAND_VIS.pixel_scale_lr_arcsec
                ).tolist(),
                "matrix": final_power_2d.tolist(),
            },
            "power": {
                "angular_scale_arcsec": (
                    Config.BAND_VIS.pixel_scale_lr_arcsec
                    / np.sqrt(radial_edges[:-1] * radial_edges[1:])
                ).tolist(),
                "normalized_psd": final_power.tolist(),
            },
        },
        "validation": validation,
        "quality_gates": gates,
        "provenance": {
            "sampling_manifest": str(manifest_path.resolve()),
            "sampling_manifest_sha256": _sha256(manifest_path),
            "source_release": str(manifest.get("source_release") or "unknown"),
            "selection": manifest.get("selection"),
            "support": manifest.get("support"),
            "archive": manifest.get("archive"),
            "split_seed": int(split_seed),
            "grouping_unit": "parent_id",
            "failed_samples": failures,
        },
    }
    _write_json(vis_noise_candidate_path(), candidate)
    if progress:
        progress(total + 1, total + 1, "VIS noise candidate fitted")
    return candidate


def activate_vis_noise_candidate() -> dict[str, Any]:
    """Atomically activate a valid candidate with at least three parents."""
    candidate = _read_json(vis_noise_candidate_path())
    if not candidate or not candidate.get("valid"):
        raise ValueError("No valid fitted VIS noise candidate is available")
    parent_count = int(
        (candidate.get("sample_summary") or {}).get("independent_parent_count", 0)
    )
    if parent_count < 3:
        raise ValueError("VIS noise activation requires at least 3 independent parents")
    fields = (candidate.get("sample_summary") or {}).get("fields") or {}
    if not isinstance(fields, Mapping) or not _q1_fields_have_fit_and_holdout(fields):
        raise ValueError(
            "VIS noise activation requires at least 2 independent parents "
            "in each Q1 deep field"
        )
    runtime = runtime_vis_noise_payload(candidate)
    _write_json(active_vis_noise_path(), runtime)
    return {**candidate, "active": True}


def _sampling_state(path: Path) -> dict[str, Any]:
    manifest = _read_json(path)
    if not manifest:
        return {
            "manifest_path": str(path),
            "exists": False,
            "planned_samples": 0,
            "completed_samples": 0,
            "failed_samples": 0,
            "independent_parent_count": 0,
        }
    samples = list(manifest.get("samples") or [])
    try:
        _validate_sampling_manifest(manifest)
        manifest_error = None
    except ValueError as exc:
        manifest_error = str(exc)
    completed = [
        sample for sample in samples
        if sample.get("status") in {"written", "cached"}
    ]
    parents_by_field: dict[str, set[str]] = defaultdict(set)
    for sample in completed:
        if sample.get("parent_id"):
            parents_by_field[str(sample.get("field") or "unknown")].add(
                str(sample["parent_id"])
            )
    fields = {
        field: len(parent_ids) for field, parent_ids in sorted(parents_by_field.items())
    }
    return {
        "manifest_path": str(path),
        "exists": True,
        "version": manifest.get("version"),
        "seed": manifest.get("seed"),
        "source_release": manifest.get("source_release"),
        "planned_samples": len(samples),
        "completed_samples": len(completed),
        "failed_samples": sum(sample.get("status") == "failed" for sample in samples),
        "independent_parent_count": len({
            str(sample["parent_id"]) for sample in completed if sample.get("parent_id")
        }),
        "fields": fields,
        "manifest_error": manifest_error,
    }


def vis_noise_state() -> dict[str, Any]:
    """Return review candidate, strict active state, and sampling readiness."""
    candidate = _read_json(vis_noise_candidate_path())
    active_raw = _read_json(active_vis_noise_path())
    try:
        active_runtime = (
            runtime_vis_noise_payload(active_raw) if active_raw is not None else None
        )
    except ValueError:
        active_runtime = None
    sampling = _sampling_state(default_sampling_manifest_path())
    unavailable_reason = None
    can_fit = bool(
        sampling["exists"]
        and not sampling.get("manifest_error")
        and sampling["completed_samples"] > 0
        and sampling["independent_parent_count"] >= 3
        and _q1_fields_have_fit_and_holdout(sampling.get("fields") or {})
    )
    if not sampling["exists"]:
        unavailable_reason = "No VIS-noise sampling manifest is available"
    elif sampling.get("manifest_error"):
        unavailable_reason = str(sampling["manifest_error"])
    elif sampling["independent_parent_count"] < 3:
        unavailable_reason = "At least 3 completed independent parent mosaics are required"
    elif not _q1_fields_have_fit_and_holdout(sampling.get("fields") or {}):
        unavailable_reason = (
            "At least 2 completed independent parents are required in each "
            "of EDF-N, EDF-S, and EDF-F"
        )
    candidate_is_active = bool(
        candidate
        and active_runtime
        and candidate.get("fingerprint") == active_runtime.get("fingerprint")
        and candidate.get("valid")
    )
    active = (
        {**candidate, "runtime": active_runtime, "active": True}
        if candidate_is_active and candidate
        else active_runtime
    )
    return {
        "candidate": candidate,
        "active": active,
        "is_active": active_runtime is not None,
        "candidate_is_active": candidate_is_active,
        "can_fit": can_fit,
        "unavailable_reason": unavailable_reason,
        "sampling": sampling,
    }


__all__ = [
    "active_vis_noise_path",
    "activate_vis_noise_candidate",
    "default_sampling_manifest_path",
    "fit_vis_noise_candidate",
    "runtime_vis_noise_payload",
    "vis_noise_candidate_path",
    "vis_noise_state",
]
