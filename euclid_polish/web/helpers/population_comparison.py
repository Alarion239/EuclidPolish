"""Cached real-Euclid versus synthetic-sky population diagnostics.

The expensive image pass is deliberately streaming: one 256-pixel field is
loaded, sampled, Fourier-transformed, and released before the next field.  The
result is a small JSON artifact consumed by the React comparison page.
"""
from __future__ import annotations

import csv
import json
import math
import os
import random
import tempfile
import warnings
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astroquery.esa.euclid import Euclid

from euclid_polish.config import Config
from euclid_polish.photometry import electrons_to_ab_mag, uJy_to_ab_mag
from euclid_polish.sky.generation.source_catalog import read_sources
from euclid_polish.web import job_config
from euclid_polish.web.helpers.paths import _sky_records_local_dir
from euclid_polish.web.helpers.tng_prior import (
    DetectionAccumulator,
    detection_payload,
    tng_prior_payload,
)

VERSION = 8
CATALOG_VERSION = 4
BANDS = ("VIS", "Y_E", "J_E", "H_E")
TILE_SIZE = 256
ANALYSIS_SIZE = 255
PIXEL_SCALE_ARCSEC = float(Config.VIS_PIXEL_SCALE_ARCSEC)
FIELD_AREA_ARCMIN2 = (TILE_SIZE * PIXEL_SCALE_ARCSEC / 60.0) ** 2
DEFAULT_CONE_RA = 267.4229
DEFAULT_CONE_DEC = 64.8873
DEFAULT_CONE_RADIUS_ARCMIN = math.sqrt((200 * FIELD_AREA_ARCMIN2) / math.pi)
MAX_CATALOG_ROWS = 50_000
SCALE_BOOTSTRAPS = 256
PIXEL_QUANTILES = (0.1, 1.0, 5.0, 16.0, 50.0, 84.0, 95.0, 99.0, 99.9)
FIELD_METRICS = (
    "mean",
    "median",
    "std",
    "robust_std",
    "p01",
    "p99",
    "zero_fraction",
    "negative_fraction",
)

_PARAM_META = {
    "objects_per_field": ("objects per field", "count"),
    "mag_vis": ("VIS magnitude", "AB mag"),
    "mag_y_e": ("Y_E magnitude", "AB mag"),
    "mag_j_e": ("J_E magnitude", "AB mag"),
    "mag_h_e": ("H_E magnitude", "AB mag"),
    "vis_y_color": ("VIS − Y colour", "AB mag"),
    "y_j_color": ("Y − J colour", "AB mag"),
    "j_h_color": ("J − H colour", "AB mag"),
    "flux_vis_e": ("VIS source flux", "e⁻ / stack"),
    "flux_vis_psf_uJy": ("VIS PSF flux", "µJy"),
    "fluxerr_vis_psf_uJy": ("VIS PSF flux error", "µJy"),
    "flux_vis_aper_uJy": ("VIS 3-FWHM aperture flux", "µJy"),
    "fluxerr_vis_aper_uJy": ("VIS 3-FWHM aperture flux error", "µJy"),
    "flux_y_aper_uJy": ("Y 3-FWHM aperture flux", "µJy"),
    "fluxerr_y_aper_uJy": ("Y 3-FWHM aperture flux error", "µJy"),
    "flux_j_aper_uJy": ("J 3-FWHM aperture flux", "µJy"),
    "fluxerr_j_aper_uJy": ("J 3-FWHM aperture flux error", "µJy"),
    "flux_h_aper_uJy": ("H 3-FWHM aperture flux", "µJy"),
    "fluxerr_h_aper_uJy": ("H 3-FWHM aperture flux error", "µJy"),
    "vis_snr": ("VIS PSF signal-to-noise", "ratio"),
    "aper_vis_snr": ("VIS aperture signal-to-noise", "ratio"),
    "aper_y_snr": ("Y aperture signal-to-noise", "ratio"),
    "aper_j_snr": ("J aperture signal-to-noise", "ratio"),
    "aper_h_snr": ("H aperture signal-to-noise", "ratio"),
    "point_like_prob": ("point-like probability", "probability"),
    "extended_prob": ("extended-source probability", "probability"),
    "spurious_prob": ("spurious-source probability", "probability"),
    "blended_prob": ("blended-source probability", "probability"),
    "segmentation_area": ("segmentation area", "VIS pixels"),
    "semimajor_axis": ("semi-major axis", "VIS pixels"),
    "ellipticity": ("ellipticity", "ratio"),
    "kron_radius": ("Kron radius", "VIS pixels"),
    "fwhm": ("photometry FWHM", "arcsec"),
    "mu_max": ("peak surface brightness", "mag / arcsec²"),
    "mumax_minus_mag": ("peak − total magnitude", "mag / arcsec²"),
    "gal_ebv": ("Galactic E(B−V)", "mag"),
    "gaia_match_quality": ("Gaia match quality", "score"),
    "gaia_matched": ("Gaia counterpart", "0 / 1"),
    "deblended": ("deblended source", "0 / 1"),
    "z": ("redshift", "z"),
    "re_arcsec": ("half-light radius", "arcsec"),
    "logmass": ("stellar mass", "log₁₀ M☉"),
    "mass_scale": ("TNG mass scale", "ratio"),
    "temperature_k": ("stellar temperature", "K"),
    "extinction_av": ("stellar extinction Aᵥ", "mag"),
}

_SHARED_PARAMETERS = {
    "mag_vis": ("VIS magnitude", "AB mag"),
    "mag_y_e": ("Y_E magnitude", "AB mag"),
    "mag_j_e": ("J_E magnitude", "AB mag"),
    "mag_h_e": ("H_E magnitude", "AB mag"),
    "vis_y_color": ("VIS − Y colour", "AB mag"),
    "y_j_color": ("Y − J colour", "AB mag"),
    "j_h_color": ("J − H colour", "AB mag"),
}


def cache_dir() -> Path:
    return Path(Config.DATA_DIR) / "population_comparison"


def comparison_path() -> Path:
    return cache_dir() / "comparison.json"


def euclid_catalog_path() -> Path:
    return cache_dir() / "euclid_population.csv"


def euclid_catalog_meta_path() -> Path:
    return cache_dir() / "euclid_population_meta.json"


def cosmos_euclid_fit_path() -> Path:
    return cache_dir() / "cosmos2025" / "cosmos_euclid_density_fit.json"


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open() as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def read_comparison() -> dict[str, Any] | None:
    payload = _read_json(comparison_path())
    return payload if payload and payload.get("version") == VERSION else None


def read_cosmos_euclid_fit() -> dict[str, Any] | None:
    payload = _read_json(cosmos_euclid_fit_path())
    return payload if payload and payload.get("version") == 1 else None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(temporary, path)


def _synthetic_paths(
    *,
    include_training: bool = False,
) -> tuple[list[Path], list[Path]]:
    root = Path(_sky_records_local_dir())
    records = [
        root / f"dirty_{subset}.tfrecord"
        for subset in ("test", "validate")
        if (root / f"dirty_{subset}.tfrecord").is_file()
    ]
    source_splits = ["test", "validate"]
    if include_training:
        source_splits.append("train")
    sources = [
        root / f"sources_{subset}.csv"
        for subset in source_splits
        if (root / f"sources_{subset}.csv").is_file()
    ]
    return records, sources


def _real_field_sources() -> tuple[list[Path], list[Path]]:
    inference = sorted(
        Path(Config.EUCLID_INFERENCE_DIR).glob("real_fields/*/raw/VIS.fits")
    )
    overlap = sorted(
        (Path(Config.DATA_DIR) / "jwst_euclid_overlap").glob(
            "nexus_fields/*/tiles/euclid_lr_vis_y_j_h_*.fits"
        )
    )
    return inference, overlap


def availability() -> dict[str, Any]:
    records, source_csvs = _synthetic_paths()
    _, source_csvs_with_training = _synthetic_paths(include_training=True)
    inference, overlap = _real_field_sources()
    synthetic_fields = sum(_count_tfrecord(path) for path in records)
    population_fields = _source_field_count(source_csvs)
    population_fields_with_training = _source_field_count(
        source_csvs_with_training
    )
    inference_fields = 100 * len(inference)
    real_fields = inference_fields + len(overlap)
    meta = _read_json(euclid_catalog_meta_path())
    catalog_current = (
        meta is not None and meta.get("catalog_version") == CATALOG_VERSION
    )
    catalog_usable = bool(
        catalog_current
        and meta.get("rows", 0) > 0
        and sum(
            int(meta.get("counts", {}).get(kind, 0))
            for kind in ("galaxy", "unknown")
        ) > 0
    )
    return {
        "synthetic": {
            "fields": synthetic_fields,
            "area_arcmin2": synthetic_fields * FIELD_AREA_ARCMIN2,
            "population_fields": population_fields,
            "population_area_arcmin2": population_fields * FIELD_AREA_ARCMIN2,
            "population_fields_with_training": population_fields_with_training,
            "population_area_arcmin2_with_training": (
                population_fields_with_training * FIELD_AREA_ARCMIN2
            ),
            "record_files": len(records),
            "source_catalogs": len(source_csvs_with_training),
            "train_source_catalog": (
                Path(_sky_records_local_dir()) / "sources_train.csv"
            ).is_file(),
        },
        "real": {
            "fields": real_fields,
            "area_arcmin2": real_fields * FIELD_AREA_ARCMIN2,
            "inference_fields": inference_fields,
            "jwst_overlap_fields": len(overlap),
        },
        "euclid_catalog": {
            "cached": euclid_catalog_path().is_file() and catalog_usable,
            "meta": meta if catalog_current else None,
        },
        "field_area_arcmin2": FIELD_AREA_ARCMIN2,
        "default_cone": {
            "ra": DEFAULT_CONE_RA,
            "dec": DEFAULT_CONE_DEC,
            "radius_arcmin": DEFAULT_CONE_RADIUS_ARCMIN,
            "area_arcmin2": math.pi * DEFAULT_CONE_RADIUS_ARCMIN**2,
        },
    }


def _count_tfrecord(path: Path) -> int:
    """Count framed TFRecord entries without importing TensorFlow."""
    if not path.is_file():
        return 0
    count = 0
    with path.open("rb") as handle:
        while True:
            length_bytes = handle.read(8)
            if not length_bytes:
                break
            if len(length_bytes) != 8:
                raise ValueError(f"truncated TFRecord length in {path}")
            length = int.from_bytes(length_bytes, "little", signed=False)
            handle.seek(4 + length + 4, os.SEEK_CUR)
            count += 1
    return count


def _synthetic_fields(paths: Iterable[Path]) -> Iterator[np.ndarray]:
    import tensorflow as tf

    from euclid_polish.image.core import Image

    for path in paths:
        for raw in tf.data.TFRecordDataset([str(path)]):
            yield np.asarray(Image.from_tfrecord(raw).data, dtype=np.float32)


def _center_crop(array: np.ndarray, size: int) -> np.ndarray:
    y0 = max(0, (array.shape[0] - size) // 2)
    x0 = max(0, (array.shape[1] - size) // 2)
    return array[y0:y0 + size, x0:x0 + size]


def _real_fields(inference: Iterable[Path],
                 overlap: Iterable[Path]) -> Iterator[np.ndarray]:
    for vis_path in inference:
        raw_dir = vis_path.parent
        hdus = [fits.open(raw_dir / f"{band}.fits", memmap=True) for band in BANDS]
        try:
            planes = [
                _center_crop(np.asarray(hdu[0].data, dtype=np.float32), 2560)
                for hdu in hdus
            ]
            for row in range(10):
                for col in range(10):
                    y0, x0 = row * TILE_SIZE, col * TILE_SIZE
                    yield np.stack(
                        [plane[y0:y0 + TILE_SIZE, x0:x0 + TILE_SIZE]
                         for plane in planes],
                        axis=-1,
                    )
        finally:
            for hdu in hdus:
                hdu.close()
    for path in overlap:
        with fits.open(path, memmap=True) as hdul:
            data = np.asarray(hdul[0].data, dtype=np.float32)
            if data.ndim == 3 and data.shape[0] == len(BANDS):
                data = np.moveaxis(data, 0, -1)
            yield data


def _normalise_field(field: np.ndarray) -> np.ndarray:
    data = np.asarray(field, dtype=np.float32)
    if data.ndim != 3:
        raise ValueError(f"expected H×W×4 field, got shape {data.shape}")
    if data.shape[-1] != len(BANDS) and data.shape[0] == len(BANDS):
        data = np.moveaxis(data, 0, -1)
    if data.shape[-1] != len(BANDS):
        raise ValueError(f"expected four bands, got shape {data.shape}")
    return _center_crop(data, ANALYSIS_SIZE)


def _power_geometry(size: int = ANALYSIS_SIZE) -> tuple[np.ndarray, np.ndarray]:
    fy = np.fft.fftfreq(size, d=PIXEL_SCALE_ARCSEC)
    fx = np.fft.rfftfreq(size, d=PIXEL_SCALE_ARCSEC)
    radius = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    positive = radius[radius > 0]
    edges = np.geomspace(max(1.0 / (size * PIXEL_SCALE_ARCSEC), positive.min()),
                         positive.max(), 25)
    return radius, edges


def _radial_average(values: np.ndarray, radius: np.ndarray,
                    edges: np.ndarray) -> np.ndarray:
    flat_r = radius.reshape(-1)
    flat_v = np.asarray(values).reshape(-1)
    which = np.digitize(flat_r, edges) - 1
    out = np.full(len(edges) - 1, np.nan, dtype=np.float64)
    for index in range(len(out)):
        selected = flat_v[which == index]
        selected = selected[np.isfinite(selected)]
        if selected.size:
            out[index] = float(np.mean(selected))
    return out


class _FieldAccumulator:
    def __init__(self) -> None:
        self.radius, self.k_edges = _power_geometry()
        self.samples: list[list[np.ndarray]] = [[] for _ in BANDS]
        self.power: list[list[np.ndarray]] = [[] for _ in BANDS]
        self.field_means: list[list[float]] = [[] for _ in BANDS]
        self.field_metrics: list[dict[str, list[float]]] = [
            {metric: [] for metric in FIELD_METRICS} for _ in BANDS
        ]
        self.band_correlations: dict[str, list[float]] = {
            f"{BANDS[left]}:{BANDS[right]}": []
            for left in range(len(BANDS))
            for right in range(left + 1, len(BANDS))
        }
        self.count = 0
        self.window = np.outer(np.hanning(ANALYSIS_SIZE),
                               np.hanning(ANALYSIS_SIZE)).astype(np.float32)

    def add(self, field: np.ndarray) -> None:
        data = _normalise_field(field)
        self.count += 1
        for band in range(len(BANDS)):
            plane = data[..., band]
            values = plane[np.isfinite(plane)]
            if not values.size:
                raise ValueError(f"field has no finite {BANDS[band]} pixels")
            sampled = plane[::8, ::8].reshape(-1)
            sampled = sampled[np.isfinite(sampled)]
            self.samples[band].append(sampled)
            median = float(np.median(values))
            mean = float(np.mean(values))
            metrics = {
                "mean": mean,
                "median": median,
                "std": float(np.std(values)),
                "robust_std": float(
                    1.4826 * np.median(np.abs(values - median))
                ),
                "p01": float(np.percentile(values, 1)),
                "p99": float(np.percentile(values, 99)),
                "zero_fraction": float(np.mean(values == 0)),
                "negative_fraction": float(np.mean(values < 0)),
            }
            for metric, value in metrics.items():
                self.field_metrics[band][metric].append(value)
            self.field_means[band].append(mean)
            filled = np.where(np.isfinite(plane), plane, median)
            centered = (filled - mean) * self.window
            fft = np.fft.rfft2(centered)
            self.power[band].append(
                _radial_average(np.abs(fft) ** 2, self.radius, self.k_edges)
            )
        sampled_bands = data[::4, ::4, :].reshape(-1, len(BANDS))
        for left in range(len(BANDS)):
            for right in range(left + 1, len(BANDS)):
                pair = sampled_bands[:, [left, right]]
                pair = pair[np.all(np.isfinite(pair), axis=1)]
                if pair.shape[0] < 3:
                    continue
                if np.std(pair[:, 0]) == 0 or np.std(pair[:, 1]) == 0:
                    continue
                value = float(np.corrcoef(pair[:, 0], pair[:, 1])[0, 1])
                if np.isfinite(value):
                    key = f"{BANDS[left]}:{BANDS[right]}"
                    self.band_correlations[key].append(value)


def _json_curve(values: np.ndarray) -> list[float | None]:
    return [float(value) if np.isfinite(value) else None
            for value in np.asarray(values).reshape(-1)]


def _normalise_scale_curve(values: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Return a unit-integral 2-D variance density over logarithmic k."""
    curve = np.asarray(values, dtype=np.float64)
    variance_density = k**2 * curve
    valid = np.isfinite(variance_density) & (variance_density >= 0) & (k > 0)
    out = np.full_like(variance_density, np.nan)
    if np.count_nonzero(valid) < 2:
        return out
    integral = float(np.trapezoid(variance_density[valid], np.log(k[valid])))
    if not np.isfinite(integral) or integral <= 0:
        return out
    out[valid] = variance_density[valid] / integral
    return out


def _scale_similarity(
    synthetic_rows: list[np.ndarray],
    real_rows: list[np.ndarray],
    k: np.ndarray,
    *,
    seed: int,
) -> dict[str, Any]:
    """Compare phase-free scale allocation and total fluctuation power."""
    synthetic = np.stack(synthetic_rows)
    real = np.stack(real_rows)
    synthetic_shapes = np.stack([
        _normalise_scale_curve(row, k) for row in synthetic
    ])
    real_shapes = np.stack([
        _normalise_scale_curve(row, k) for row in real
    ])

    def representative(rows: np.ndarray) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            median = np.nanmedian(rows, axis=0)
        return _normalise_scale_curve(median / k**2, k)

    def total_variance(row: np.ndarray) -> float:
        density = k**2 * row
        valid = np.isfinite(density) & (density >= 0) & (k > 0)
        if np.count_nonzero(valid) < 2:
            return math.nan
        return float(np.trapezoid(density[valid], np.log(k[valid])))

    synthetic_total = np.asarray(
        [total_variance(row) for row in synthetic], dtype=np.float64
    )
    real_total = np.asarray(
        [total_variance(row) for row in real], dtype=np.float64
    )

    def compare(
        synthetic_shape: np.ndarray,
        real_shape: np.ndarray,
        synthetic_variance: float,
        real_variance: float,
    ) -> tuple[np.ndarray, float, float]:
        valid = (
            np.isfinite(synthetic_shape)
            & np.isfinite(real_shape)
            & (synthetic_shape > 0)
            & (real_shape > 0)
        )
        ratio = np.full_like(k, np.nan)
        ratio[valid] = np.log10(
            synthetic_shape[valid] / real_shape[valid]
        )
        overlap = (
            float(np.trapezoid(
                np.sqrt(synthetic_shape[valid] * real_shape[valid]),
                np.log(k[valid]),
            ))
            if np.count_nonzero(valid) >= 2 else math.nan
        )
        variance_ratio = (
            synthetic_variance / real_variance
            if np.isfinite(synthetic_variance)
            and np.isfinite(real_variance)
            and real_variance > 0 else math.nan
        )
        return ratio, float(np.clip(overlap, 0.0, 1.0)), variance_ratio

    synthetic_shape = representative(synthetic_shapes)
    real_shape = representative(real_shapes)
    ratio, overlap, variance_ratio = compare(
        synthetic_shape,
        real_shape,
        float(np.nanmedian(synthetic_total)),
        float(np.nanmedian(real_total)),
    )

    rng = np.random.default_rng(seed)
    ratio_boot = np.full((SCALE_BOOTSTRAPS, len(k)), np.nan)
    overlap_boot = np.full(SCALE_BOOTSTRAPS, np.nan)
    variance_boot = np.full(SCALE_BOOTSTRAPS, np.nan)
    for index in range(SCALE_BOOTSTRAPS):
        synthetic_indices = rng.integers(
            0, len(synthetic_shapes), len(synthetic_shapes)
        )
        real_indices = rng.integers(0, len(real_shapes), len(real_shapes))
        boot_ratio, boot_overlap, boot_variance = compare(
            representative(synthetic_shapes[synthetic_indices]),
            representative(real_shapes[real_indices]),
            float(np.nanmedian(synthetic_total[synthetic_indices])),
            float(np.nanmedian(real_total[real_indices])),
        )
        ratio_boot[index] = boot_ratio
        overlap_boot[index] = boot_overlap
        variance_boot[index] = boot_variance

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ratio_p16 = np.nanpercentile(ratio_boot, 16, axis=0)
        ratio_p84 = np.nanpercentile(ratio_boot, 84, axis=0)
    return {
        "k": k.tolist(),
        "log_shape_ratio": {
            "median": _json_curve(ratio),
            "p16": _json_curve(ratio_p16),
            "p84": _json_curve(ratio_p84),
        },
        "overlap": {
            "median": overlap,
            "p16": float(np.nanpercentile(overlap_boot, 16)),
            "p84": float(np.nanpercentile(overlap_boot, 84)),
        },
        "variance_ratio": {
            "median": variance_ratio,
            "p16": float(np.nanpercentile(variance_boot, 16)),
            "p84": float(np.nanpercentile(variance_boot, 84)),
        },
        "x_label": "angular frequency (cycles / arcsec)",
        "y_label": "log₁₀ synthetic / real normalized scale power",
    }


def _field_payload(synthetic: _FieldAccumulator,
                   real: _FieldAccumulator) -> dict[str, Any]:
    centers_k = np.sqrt(synthetic.k_edges[:-1] * synthetic.k_edges[1:])
    histograms: dict[str, Any] = {}
    power: dict[str, Any] = {}
    scale_similarity: dict[str, Any] = {}
    quantiles: dict[str, Any] = {}
    relations: dict[str, Any] = {
        "mean_std": {},
        "median_robust_std": {},
    }

    for band_index, band in enumerate(BANDS):
        samples_s = np.concatenate(synthetic.samples[band_index])
        samples_r = np.concatenate(real.samples[band_index])
        combined = np.concatenate((samples_s, samples_r))
        # Use the complete shared range.  The former 0.2–99.8 percentile
        # window made the bulk of each distribution easier to see, but hid
        # the bright and negative tails that this comparison is meant to
        # expose.
        lo, hi = np.min(combined), np.max(combined)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = -1.0, 1.0
        edges = np.linspace(float(lo), float(hi), 65)

        def histogram_fraction(
            samples: np.ndarray,
            histogram_edges: np.ndarray = edges,
        ) -> np.ndarray:
            counts, _ = np.histogram(samples, bins=histogram_edges)
            total = max(int(counts.sum()), 1)
            return counts / total

        histograms[band] = {
            "x": ((edges[:-1] + edges[1:]) / 2).tolist(),
            "synthetic": histogram_fraction(samples_s).tolist(),
            "real": histogram_fraction(samples_r).tolist(),
            "zero_bin": (
                int(np.clip(
                    np.searchsorted(edges, 0.0, side="right") - 1,
                    0,
                    len(edges) - 2,
                ))
                if lo <= 0.0 <= hi else None
            ),
            "x_label": "pixel brightness (e⁻ / stack)",
            "y_label": "fraction of sampled pixels / bin",
            "range": [float(lo), float(hi)],
        }
        quantiles[band] = {
            "q": list(PIXEL_QUANTILES),
            "synthetic": np.percentile(
                samples_s, PIXEL_QUANTILES
            ).astype(float).tolist(),
            "real": np.percentile(
                samples_r, PIXEL_QUANTILES
            ).astype(float).tolist(),
            "x_label": "pixel percentile",
            "y_label": "pixel brightness (e⁻ / stack)",
        }

        def power_summary(rows: list[np.ndarray]) -> dict[str, Any]:
            stacked = np.stack(rows)
            with np.errstate(invalid="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                return {
                    "p16": _json_curve(np.nanpercentile(stacked, 16, axis=0)),
                    "median": _json_curve(np.nanmedian(stacked, axis=0)),
                    "p84": _json_curve(np.nanpercentile(stacked, 84, axis=0)),
                }

        power[band] = {
            "k": centers_k.tolist(),
            "synthetic": power_summary(synthetic.power[band_index]),
            "real": power_summary(real.power[band_index]),
            "x_label": "angular frequency (cycles / arcsec)",
            "y_label": "mean-subtracted power (e⁻²)",
        }
        scale_similarity[band] = _scale_similarity(
            synthetic.power[band_index],
            real.power[band_index],
            centers_k,
            seed=1701 + band_index,
        )
        relations["mean_std"][band] = {
            "synthetic": {
                "x": synthetic.field_metrics[band_index]["mean"],
                "y": synthetic.field_metrics[band_index]["std"],
            },
            "real": {
                "x": real.field_metrics[band_index]["mean"],
                "y": real.field_metrics[band_index]["std"],
            },
            "x_label": "field mean (e⁻ / pixel)",
            "y_label": "field standard deviation (e⁻ / pixel)",
        }
        relations["median_robust_std"][band] = {
            "synthetic": {
                "x": synthetic.field_metrics[band_index]["median"],
                "y": synthetic.field_metrics[band_index]["robust_std"],
            },
            "real": {
                "x": real.field_metrics[band_index]["median"],
                "y": real.field_metrics[band_index]["robust_std"],
            },
            "x_label": "field median (e⁻ / pixel)",
            "y_label": "robust noise, 1.4826 × MAD (e⁻ / pixel)",
        }

    def interval(values: list[float]) -> dict[str, float]:
        return {
            "median": float(np.median(values)),
            "p16": float(np.percentile(values, 16)),
            "p84": float(np.percentile(values, 84)),
        }

    def field_summary(acc: _FieldAccumulator) -> dict[str, Any]:
        return {
            band: {
                metric: interval(acc.field_metrics[index][metric])
                for metric in FIELD_METRICS
            }
            for index, band in enumerate(BANDS)
        }

    correlation_pairs = list(synthetic.band_correlations)
    band_correlation = {
        "pairs": [pair.replace("_E", "").replace(":", "–")
                  for pair in correlation_pairs],
        "synthetic": {
            key: [interval(synthetic.band_correlations[pair])[key]
                  for pair in correlation_pairs]
            for key in ("median", "p16", "p84")
        },
        "real": {
            key: [interval(real.band_correlations[pair])[key]
                  for pair in correlation_pairs]
            for key in ("median", "p16", "p84")
        },
        "x_label": "band pair",
        "y_label": "within-field pixel correlation",
    }

    return {
        "bands": list(BANDS),
        "histograms": histograms,
        "quantiles": quantiles,
        "power": power,
        "scale_similarity": scale_similarity,
        "relations": relations,
        "band_correlation": band_correlation,
        "summary": {
            "synthetic": field_summary(synthetic),
            "real": field_summary(real),
        },
    }


def _finite(value: Any) -> float | None:
    if np.ma.is_masked(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _source_field_count(paths: Iterable[Path]) -> int:
    total = 0
    for path in paths:
        fields: set[int] = set()
        with path.open(newline="") as handle:
            for raw in csv.DictReader(handle):
                fields.add(int(raw["field_index"]))
        total += len(fields)
    return total


def _derive_colours(row: dict[str, Any]) -> None:
    for colour, left, right in (
        ("vis_y_color", "mag_vis", "mag_y_e"),
        ("y_j_color", "mag_y_e", "mag_j_e"),
        ("j_h_color", "mag_j_e", "mag_h_e"),
    ):
        if row.get(colour) is not None:
            continue
        left_value = _finite(row.get(left))
        right_value = _finite(row.get(right))
        if left_value is not None and right_value is not None:
            row[colour] = left_value - right_value


def _read_synthetic_sources(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    offset = 0
    for path in paths:
        local_max = -1
        with path.open(newline="") as handle:
            for raw in csv.DictReader(handle):
                field_index = int(raw["field_index"]) + offset
                local_max = max(local_max, int(raw["field_index"]))
                source_type = str(raw.get("type", "unknown")).strip().lower()
                row: dict[str, Any] = {
                    # Lensed galaxies are part of the galaxy population here:
                    # there are too few synthetic lenses for a meaningful
                    # standalone distribution and MER has no matching lens
                    # classification.
                    "type": "galaxy" if source_type == "lens" else source_type,
                    "render": str(raw.get("render", "")).strip().lower(),
                    "field_index": field_index,
                }
                for key in _PARAM_META:
                    if key == "objects_per_field":
                        continue
                    row[key] = _finite(raw.get(key))
                for key in (
                    "galaxy_density_arcmin2",
                    "tng_density_arcmin2",
                    "tng_mf_alpha",
                ):
                    row[key] = _finite(raw.get(key))
                if row.get("mag_vis") is None and row.get("flux_vis_e"):
                    row["mag_vis"] = _finite(electrons_to_ab_mag(
                        row["flux_vis_e"], Config.BAND_VIS
                    ))
                _derive_colours(row)
                rows.append(row)
        offset += local_max + 1
    return rows


def _synthetic_dataset_tng_prior(rows: list[dict[str, Any]]) -> float:
    """Raw TNG budget that produced the displayed source truth.

    New source catalogs persist the value per TNG row. Existing cached
    catalogs predate that column and are known to use the legacy 60/arcmin²
    generator; never reinterpret those pixels with the live Config value.
    """
    saved = [
        value
        for row in rows
        if str(row.get("render")) == "tng"
        if (
            value := _finite(
                row.get("galaxy_density_arcmin2")
                or row.get("tng_density_arcmin2")
            )
        ) is not None
    ]
    if saved:
        return float(np.median(saved))
    return float(Config.TNG_LEGACY_DATASET_DENSITY_ARCMIN2)


def _read_euclid_sources() -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    path = euclid_catalog_path()
    meta = _read_json(euclid_catalog_meta_path())
    if (
        not path.is_file()
        or meta is None
        or meta.get("catalog_version") != CATALOG_VERSION
    ):
        return [], None
    rows: list[dict[str, Any]] = []
    with path.open(newline="") as handle:
        for raw in csv.DictReader(handle):
            row = {"type": raw.get("type", "unknown")}
            for key in _PARAM_META:
                if key != "objects_per_field":
                    row[key] = _finite(raw.get(key))
            _derive_colours(row)
            rows.append(row)
    return rows, meta


def _histogram(values: list[float]) -> dict[str, Any]:
    data = np.asarray(values, dtype=np.float64)
    data = data[np.isfinite(data)]
    if not data.size:
        return {"x": [], "density": [], "count": 0}
    lo, hi = np.percentile(data, [0.5, 99.5])
    if hi <= lo:
        lo, hi = float(data.min()) - 0.5, float(data.max()) + 0.5
    bins = min(36, max(8, int(np.sqrt(data.size))))
    edges = np.linspace(float(lo), float(hi), bins + 1)
    clipped = data[(data >= lo) & (data <= hi)]
    counts, _ = np.histogram(clipped, bins=edges)
    density = counts / max(int(counts.sum()), 1)
    return {
        "x": ((edges[:-1] + edges[1:]) / 2).tolist(),
        "density": density.tolist(),
        "count": int(data.size),
        "range": [float(lo), float(hi)],
    }


def _parameter_payload(rows: list[dict[str, Any]], area_arcmin2: float,
                       *, include_per_field: bool) -> dict[str, Any]:
    types = sorted({str(row.get("type", "unknown")) for row in rows})
    parameters: dict[str, Any] = {}
    for parameter, (label, unit) in _PARAM_META.items():
        series: dict[str, Any] = {}
        if parameter == "objects_per_field":
            if not include_per_field:
                continue
            field_ids = sorted({int(row["field_index"]) for row in rows})
            counts_by_type = {
                kind: dict.fromkeys(field_ids, 0)
                for kind in types
            }
            for row in rows:
                kind = str(row.get("type", "unknown"))
                counts_by_type[kind][int(row["field_index"])] += 1
            for kind in types:
                counts = [counts_by_type[kind][field_id]
                          for field_id in field_ids]
                series[kind] = _histogram(counts)
        else:
            for kind in types:
                values = [
                    value for row in rows if row.get("type") == kind
                    if (value := _finite(row.get(parameter))) is not None
                ]
                if values:
                    series[kind] = _histogram(values)
        if series:
            parameters[parameter] = {
                "label": label,
                "unit": unit,
                "series": series,
            }
    counts = {kind: sum(row.get("type") == kind for row in rows) for kind in types}
    return {
        "objects": len(rows),
        "counts": counts,
        "density_arcmin2": {
            kind: count / area_arcmin2 if area_arcmin2 > 0 else None
            for kind, count in counts.items()
        },
        "area_arcmin2": area_arcmin2,
        "parameters": parameters,
    }


def _shared_histogram(
    values: list[float],
    edges: np.ndarray,
    area_arcmin2: float,
) -> dict[str, Any]:
    data = np.asarray(values, dtype=np.float64)
    data = data[np.isfinite(data)]
    counts, _ = np.histogram(data, bins=edges)
    return {
        "x": ((edges[:-1] + edges[1:]) / 2).tolist(),
        "density": (
            counts / area_arcmin2
            if area_arcmin2 > 0 else np.zeros_like(counts, dtype=float)
        ).tolist(),
        "count": int(data.size),
        "range": [float(edges[0]), float(edges[-1])],
    }


def _shared_parameter_payload(
    synthetic_rows: list[dict[str, Any]],
    euclid_rows: list[dict[str, Any]],
    synthetic_area_arcmin2: float,
    euclid_area_arcmin2: float,
) -> dict[str, Any]:
    """Build only like-for-like observables on identical histogram bins."""

    def comparison_class(row: dict[str, Any], *, euclid: bool) -> str | None:
        kind = str(row.get("type", "unknown"))
        if kind == "star":
            return "star"
        if kind == "galaxy" or (euclid and kind == "unknown"):
            return "nonstellar"
        return None

    parameters: dict[str, Any] = {}
    for parameter, (label, unit) in _SHARED_PARAMETERS.items():
        values_by_class: dict[str, tuple[list[float], list[float]]] = {}
        for kind in ("nonstellar", "star"):
            synthetic_values = [
                value for row in synthetic_rows
                if comparison_class(row, euclid=False) == kind
                if (value := _finite(row.get(parameter))) is not None
            ]
            euclid_values = [
                value for row in euclid_rows
                if comparison_class(row, euclid=True) == kind
                if (value := _finite(row.get(parameter))) is not None
            ]
            if not synthetic_values or not euclid_values:
                continue
            values_by_class[kind] = (synthetic_values, euclid_values)
        if values_by_class:
            combined = np.asarray([
                value
                for pair in values_by_class.values()
                for values in pair
                for value in values
            ], dtype=np.float64)
            lo, hi = float(combined.min()), float(combined.max())
            if hi <= lo:
                lo, hi = lo - 0.5, hi + 0.5
            bins = min(36, max(8, int(np.sqrt(combined.size))))
            edges = np.linspace(float(lo), float(hi), bins + 1)
            classes = {
                kind: {
                    "synthetic": _shared_histogram(
                        synthetic_values, edges, synthetic_area_arcmin2
                    ),
                    "euclid": _shared_histogram(
                        euclid_values, edges, euclid_area_arcmin2
                    ),
                }
                for kind, (synthetic_values, euclid_values)
                in values_by_class.items()
            }
            parameters[parameter] = {
                "label": label,
                "unit": unit,
                "classes": classes,
            }
    return {
        "parameters": parameters,
        "class_labels": {
            "nonstellar": "galaxies / non-stellar candidates",
            "star": "stars",
        },
        "density_unit": "objects / arcmin² / bin",
    }


def _population_payload(source_csvs: Iterable[Path],
                        synthetic_field_count: int,
                        source_detection: dict[str, Any] | None = None,
                        *,
                        calibrate_tng_prior: bool = True,
                        ) -> dict[str, Any]:
    paths = list(source_csvs)
    synthetic_rows = _read_synthetic_sources(paths)
    euclid_rows, euclid_meta = _read_euclid_sources()
    dataset_tng_prior = _synthetic_dataset_tng_prior(synthetic_rows)
    population_fields = _source_field_count(paths) or synthetic_field_count
    synthetic_area = population_fields * FIELD_AREA_ARCMIN2
    euclid_area = float(euclid_meta.get("area_arcmin2", 0.0)) if euclid_meta else 0.0
    return {
        "synthetic": _parameter_payload(
            synthetic_rows, synthetic_area, include_per_field=True
        ),
        "synthetic_field_count": population_fields,
        "euclid": (
            _parameter_payload(euclid_rows, euclid_area, include_per_field=False)
            if euclid_rows and euclid_area > 0 else None
        ),
        # Expose the shared parameters as a selection-mismatched diagnostic.
        # This must remain separate from the matched-detection calibration:
        # synthetic sidecars are complete truth, while Euclid MER is selected.
        "shared": (
            _shared_parameter_payload(
                synthetic_rows,
                euclid_rows,
                synthetic_area,
                euclid_area,
            )
            if euclid_rows and euclid_area > 0 else None
        ),
        "tng_prior": (
            tng_prior_payload(
                synthetic_rows,
                euclid_rows,
                synthetic_area,
                euclid_area,
                FIELD_AREA_ARCMIN2,
                source_detection,
                dataset_prior=dataset_tng_prior,
                configured_prior=float(
                    job_config.load().galaxy_density_arcmin2
                ),
            )
            if calibrate_tng_prior and euclid_rows and euclid_area > 0
            else None
        ),
        "euclid_meta": euclid_meta,
    }


def _population_variants(
    synthetic_field_count: int,
    source_detection: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Current-prior population and optional legacy-training-inclusive view.

    Pixel statistics and TNG calibration always use regenerated test+validate
    fields. The training toggle changes only catalog histograms/counts because
    the cached training truth was produced with the legacy generator prior.
    """
    _, current_paths = _synthetic_paths()
    _, all_paths = _synthetic_paths(include_training=True)
    current = _population_payload(
        current_paths,
        synthetic_field_count,
        source_detection,
    )
    current["synthetic_splits"] = ["test", "validate"]
    current["training_included"] = False
    current["calibration_splits"] = ["test", "validate"]

    if all_paths == current_paths:
        with_training = dict(current)
        with_training["training_included"] = False
        return current, with_training

    with_training = _population_payload(
        all_paths,
        synthetic_field_count,
        source_detection,
        calibrate_tng_prior=False,
    )
    # Never reinterpret the legacy training catalog using the current raw
    # prior. The calibration remains the matched 200-field test+validate one.
    with_training["tng_prior"] = current["tng_prior"]
    with_training["synthetic_splits"] = ["train", "test", "validate"]
    with_training["training_included"] = True
    with_training["calibration_splits"] = ["test", "validate"]
    return current, with_training


def refresh_population_comparison() -> dict[str, Any] | None:
    """Refresh source-population statistics without rereading image fields."""
    payload = read_comparison()
    if payload is None:
        return None
    synthetic_field_count = int(
        payload.get("samples", {}).get("synthetic", {}).get("fields", 0)
    )
    population, population_with_training = _population_variants(
        synthetic_field_count,
        payload.get("fields", {}).get("source_detection"),
    )
    payload["population"] = population
    payload["population_with_training"] = population_with_training
    _write_json(comparison_path(), payload)
    return population


def build_comparison(
    progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    records, _ = _synthetic_paths()
    inference, overlap = _real_field_sources()
    synthetic_count = sum(_count_tfrecord(path) for path in records)
    real_count = 100 * len(inference) + len(overlap)
    if not records:
        raise FileNotFoundError("no local dirty test/validate TFRecords")
    if not real_count:
        raise FileNotFoundError("no cached Euclid inference or JWST-overlap fields")

    total = synthetic_count + real_count
    done = 0
    synthetic_acc = _FieldAccumulator()
    synthetic_detection = DetectionAccumulator()
    for record_path in records:
        source_path = record_path.with_name(
            record_path.name
            .replace("dirty_", "sources_")
            .replace(".tfrecord", ".csv")
        )
        truth_by_field = read_sources(str(source_path))
        for field_index, field in enumerate(_synthetic_fields([record_path])):
            normalized = _normalise_field(field)
            synthetic_acc.add(normalized)
            synthetic_detection.add(
                normalized[..., 0],
                truth_by_field.get(field_index, []),
            )
            done += 1
            if progress:
                progress(done, total, "synthetic LR fields + VIS detections")

    real_acc = _FieldAccumulator()
    real_detection = DetectionAccumulator()
    for field in _real_fields(inference, overlap):
        normalized = _normalise_field(field)
        real_acc.add(normalized)
        real_detection.add(normalized[..., 0])
        done += 1
        if progress:
            progress(done, total, "real Euclid LR fields + VIS detections")

    detections = detection_payload(synthetic_detection, real_detection)
    fields = _field_payload(synthetic_acc, real_acc)
    fields["source_detection"] = detections

    population, population_with_training = _population_variants(
        synthetic_acc.count,
        detections,
    )
    payload = {
        "version": VERSION,
        "geometry": {
            "tile_size": TILE_SIZE,
            "analysis_size": ANALYSIS_SIZE,
            "pixel_scale_arcsec": PIXEL_SCALE_ARCSEC,
            "field_area_arcmin2": FIELD_AREA_ARCMIN2,
        },
        "samples": {
            "synthetic": {
                "fields": synthetic_acc.count,
                "area_arcmin2": synthetic_acc.count * FIELD_AREA_ARCMIN2,
                "splits": ["test", "validate"],
            },
            "real": {
                "fields": real_acc.count,
                "area_arcmin2": real_acc.count * FIELD_AREA_ARCMIN2,
                "inference_fields": 100 * len(inference),
                "jwst_overlap_fields": len(overlap),
            },
        },
        "fields": fields,
        "population": population,
        "population_with_training": population_with_training,
    }
    _write_json(comparison_path(), payload)
    return payload


def query_euclid_population(
    ra: float,
    dec: float,
    radius_arcmin: float,
    *,
    limit: int = MAX_CATALOG_ROWS,
    relogin: Callable[[], bool] | None = None,
    _catalog_path: Path | None = None,
    _meta_path: Path | None = None,
    _require_nonstellar: bool = True,
) -> dict[str, Any]:
    """Query clean MER sources in a cone and cache generation-facing columns.

    MER's point-like probability is retained as a fractional membership value;
    hard flags remain available for provenance and single-cone summaries.
    """
    radius_deg = radius_arcmin / 60.0
    query = f"""
    SELECT TOP {int(limit)}
        object_id, right_ascension, declination,
        point_like_flag, point_like_prob, extended_flag, extended_prob,
        spurious_prob, blended_prob, deblended_flag,
        segmentation_area, semimajor_axis, ellipticity, kron_radius, fwhm,
        mu_max, mumax_minus_mag, gal_ebv, gaia_id, gaia_match_quality,
        flux_vis_psf, fluxerr_vis_psf,
        flux_vis_3fwhm_aper, fluxerr_vis_3fwhm_aper,
        flux_y_3fwhm_aper, fluxerr_y_3fwhm_aper,
        flux_j_3fwhm_aper, fluxerr_j_3fwhm_aper,
        flux_h_3fwhm_aper, fluxerr_h_3fwhm_aper
    FROM catalogue.mer_catalogue
    WHERE CONTAINS(
        POINT('ICRS', right_ascension, declination),
        CIRCLE('ICRS', {float(ra)}, {float(dec)}, {radius_deg})
    ) = 1
      AND det_quality_flag = 0
      AND flag_vis = 0
      AND (spurious_flag IS NULL OR spurious_flag = 0)
      AND flux_vis_psf IS NOT NULL
      AND flux_vis_psf > 0
    """
    job = Euclid.launch_job_async(query)
    if job is None and relogin is not None:
        try:
            session_refreshed = relogin()
        except Exception:  # noqa: BLE001 - the archive error is surfaced below
            session_refreshed = False
        if session_refreshed:
            job = Euclid.launch_job_async(query)
    if job is None:
        raise RuntimeError(
            "The Euclid archive query failed before returning a job. "
            "The previous population cache was preserved."
        )
    results = job.get_results()
    if results is None:
        raise RuntimeError(
            "The Euclid archive query returned no result table. "
            "The previous population cache was preserved."
        )
    rows: list[dict[str, Any]] = []

    def value(raw: Any, key: str) -> float | None:
        try:
            return _finite(raw[key])
        except (KeyError, IndexError, TypeError):
            return None

    def magnitude(flux: float | None) -> float | None:
        return uJy_to_ab_mag(flux) if flux is not None and flux > 0 else None

    def signal_to_noise(flux: float | None,
                        error: float | None) -> float | None:
        return (
            flux / error
            if flux is not None and error is not None and error > 0 else None
        )

    for raw in ([] if results is None else results):
        flux = value(raw, "flux_vis_psf")
        flux_error = value(raw, "fluxerr_vis_psf")
        if flux is None or flux <= 0:
            continue
        point_like = value(raw, "point_like_flag")
        extended = value(raw, "extended_flag")
        source_type = (
            "star" if point_like == 1
            else "galaxy" if extended == 1
            else "unknown"
        )
        aperture_fluxes = {
            band: value(raw, f"flux_{band}_3fwhm_aper")
            for band in ("vis", "y", "j", "h")
        }
        aperture_errors = {
            band: value(raw, f"fluxerr_{band}_3fwhm_aper")
            for band in ("vis", "y", "j", "h")
        }
        magnitudes = {
            band: magnitude(aperture_fluxes[band])
            for band in ("vis", "y", "j", "h")
        }
        try:
            gaia_text = str(raw["gaia_id"]).strip()
        except (KeyError, IndexError, TypeError):
            gaia_text = ""
        gaia_id = gaia_text if gaia_text and gaia_text not in {"--", "nan"} else None
        rows.append({
            "object_id": str(raw["object_id"]),
            "gaia_id": gaia_id,
            "type": source_type,
            "ra": value(raw, "right_ascension"),
            "dec": value(raw, "declination"),
            "mag_vis": magnitudes["vis"],
            "mag_y_e": magnitudes["y"],
            "mag_j_e": magnitudes["j"],
            "mag_h_e": magnitudes["h"],
            "vis_y_color": (
                magnitudes["vis"] - magnitudes["y"]
                if magnitudes["vis"] is not None and magnitudes["y"] is not None
                else None
            ),
            "y_j_color": (
                magnitudes["y"] - magnitudes["j"]
                if magnitudes["y"] is not None and magnitudes["j"] is not None
                else None
            ),
            "j_h_color": (
                magnitudes["j"] - magnitudes["h"]
                if magnitudes["j"] is not None and magnitudes["h"] is not None
                else None
            ),
            "flux_vis_psf_uJy": flux,
            "fluxerr_vis_psf_uJy": flux_error,
            **{
                f"flux_{band}_aper_uJy": aperture_fluxes[band]
                for band in ("vis", "y", "j", "h")
            },
            **{
                f"fluxerr_{band}_aper_uJy": aperture_errors[band]
                for band in ("vis", "y", "j", "h")
            },
            "vis_snr": signal_to_noise(flux, flux_error),
            **{
                f"aper_{band}_snr": signal_to_noise(
                    aperture_fluxes[band], aperture_errors[band]
                )
                for band in ("vis", "y", "j", "h")
            },
            "point_like_prob": value(raw, "point_like_prob"),
            "extended_prob": value(raw, "extended_prob"),
            "spurious_prob": value(raw, "spurious_prob"),
            "blended_prob": value(raw, "blended_prob"),
            "segmentation_area": value(raw, "segmentation_area"),
            "semimajor_axis": value(raw, "semimajor_axis"),
            "ellipticity": value(raw, "ellipticity"),
            "kron_radius": value(raw, "kron_radius"),
            "fwhm": value(raw, "fwhm"),
            "mu_max": value(raw, "mu_max"),
            "mumax_minus_mag": value(raw, "mumax_minus_mag"),
            "gal_ebv": value(raw, "gal_ebv"),
            "gaia_match_quality": value(raw, "gaia_match_quality"),
            "gaia_matched": 1.0 if gaia_id is not None else 0.0,
            "deblended": (
                1.0 if value(raw, "deblended_flag") == 1 else 0.0
            ),
        })

    usable_nonstellar = sum(
        row["type"] != "star" and row.get("mag_vis") is not None
        for row in rows
    )
    if _require_nonstellar and not usable_nonstellar:
        raise ValueError(
            "The Euclid archive returned no usable nonstellar sources for "
            f"the cone at ({ra:g}, {dec:g}). The previous population cache "
            "was preserved."
        )

    out = _catalog_path or euclid_catalog_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(".tmp")
    columns = ["object_id", "gaia_id", "type", "ra", "dec", *[
        key for key in _PARAM_META if key not in {
            "objects_per_field", "flux_vis_e", "z", "re_arcsec", "logmass",
            "mass_scale", "temperature_k", "extinction_av",
        }
    ]]
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, out)

    meta = {
        "catalog_version": CATALOG_VERSION,
        "ra": float(ra),
        "dec": float(dec),
        "radius_arcmin": float(radius_arcmin),
        "area_arcmin2": math.pi * float(radius_arcmin) ** 2,
        "rows": len(rows),
        "limit": int(limit),
        "limit_reached": len(rows) >= int(limit),
        "counts": {
            kind: sum(row["type"] == kind for row in rows)
            for kind in ("star", "galaxy", "unknown")
        },
        "classification": (
            "POINT_LIKE_PROB is fractional stellar membership; galaxy weight "
            "is 1 − POINT_LIKE_PROB; invalid probabilities are excluded"
        ),
        "classification_note": (
            "Euclid documents the point-like selector as high-purity but "
            "low-completeness; unknown rows must not be counted as galaxies."
        ),
        "photometry": (
            "3 FWHM PSF-matched aperture magnitudes, raw fluxes/errors, and "
            "colours; VIS PSF flux is retained separately"
        ),
        "probability_coverage": {
            "field": "point_like_prob",
            "valid_rows": sum(
                _finite(row.get("point_like_prob")) is not None
                and 0.0 <= float(row["point_like_prob"]) <= 1.0
                for row in rows
            ),
            "missing_or_invalid_rows": sum(
                _finite(row.get("point_like_prob")) is None
                or not 0.0 <= float(row["point_like_prob"]) <= 1.0
                for row in rows
            ),
        },
    }
    _write_json(_meta_path or euclid_catalog_meta_path(), meta)
    return meta


def select_star_cone_centers(
    *,
    count: int = 6,
    radius_arcmin: float = DEFAULT_CONE_RADIUS_ARCMIN,
    stars_csv: str | Path | None = None,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Randomly select non-overlapping, archive-known positions from stars.csv."""
    path = Path(stars_csv or Path(Config.DATA_DIR) / "euclid_stars" / "stars.csv")
    candidates: list[dict[str, Any]] = []
    seen_positions: set[tuple[float, float]] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                candidate = {
                    "star_id": str(row["id"]),
                    "ra": float(row["ra"]),
                    "dec": float(row["dec"]),
                    "magnitude": float(row["magnitude"]),
                }
            except (KeyError, TypeError, ValueError):
                continue
            position = (candidate["ra"], candidate["dec"])
            if position not in seen_positions:
                seen_positions.add(position)
                candidates.append(candidate)
    if not candidates:
        raise ValueError(f"No usable star positions in {path}")
    requested = int(count)
    if requested < 1:
        raise ValueError("count must be at least 1")
    if requested > len(candidates):
        raise ValueError(
            f"Requested {requested} cones but only {len(candidates)} "
            "unique star positions are available"
        )
    shuffled = list(candidates)
    random.Random(seed).shuffle(shuffled)

    def separation(a: dict[str, Any], b: dict[str, Any]) -> float:
        ra1, ra2 = math.radians(a["ra"]), math.radians(b["ra"])
        d1, d2 = math.radians(a["dec"]), math.radians(b["dec"])
        cosine = (
            math.sin(d1) * math.sin(d2)
            + math.cos(d1) * math.cos(d2) * math.cos(ra1 - ra2)
        )
        return math.degrees(
            math.acos(max(-1.0, min(1.0, cosine)))
        ) * 60.0

    minimum_separation = 2.0 * float(radius_arcmin)
    chosen: list[dict[str, Any]] = []
    for candidate in shuffled:
        if all(
            separation(candidate, previous) >= minimum_separation
            for previous in chosen
        ):
            chosen.append(candidate)
            if len(chosen) == requested:
                break
    if len(chosen) < requested:
        raise ValueError(
            f"Only {len(chosen)} non-overlapping cones of radius "
            f"{radius_arcmin:g} arcmin fit around the saved star positions"
        )
    return chosen


def query_euclid_population_multi(
    *, count: int = 6, radius_arcmin: float = DEFAULT_CONE_RADIUS_ARCMIN,
    selection_seed: int | None = None,
    centers: list[dict[str, Any]] | None = None,
    progress: Callable[[int, int, str], None] | None = None,
    relogin: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Query star-centred cones, deduplicate, and cache one census.

    ``centers`` is used by the schema-refresh path so that a failed refresh
    never silently changes the footprint being compared.
    """
    if centers is None:
        if selection_seed is None:
            selection_seed = random.SystemRandom().getrandbits(64)
        centers = select_star_cone_centers(
            count=count,
            radius_arcmin=radius_arcmin,
            seed=selection_seed,
        )
    else:
        centers = [
            {
                **center,
                "ra": float(center["ra"]),
                "dec": float(center["dec"]),
            }
            for center in centers
        ]
        if not centers:
            raise ValueError("same-footprint refresh has no saved centers")
        count = len(centers)
    combined: dict[str, dict[str, Any]] = {}
    cone_meta: list[dict[str, Any]] = []
    out = euclid_catalog_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".euclid_population.", dir=out.parent,
    ) as temporary_dir:
        cone_catalog = Path(temporary_dir) / "cone.csv"
        cone_meta_path = Path(temporary_dir) / "cone.json"
        for index, center in enumerate(centers):
            if progress:
                progress(
                    index, len(centers),
                    f"Euclid cone {index + 1}/{len(centers)}",
                )
            meta = query_euclid_population(
                center["ra"], center["dec"], radius_arcmin,
                relogin=relogin,
                _catalog_path=cone_catalog,
                _meta_path=cone_meta_path,
                _require_nonstellar=False,
            )
            with cone_catalog.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    row["cone_index"] = index
                    combined.setdefault(str(row["object_id"]), row)
            cone_meta.append({**center, "rows": meta["rows"]})

    rows = list(combined.values())
    if not any(
        row.get("type") != "star" and _finite(row.get("mag_vis")) is not None
        for row in rows
    ):
        raise ValueError(
            "The Euclid cones contained no usable nonstellar sources. "
            "The previous population cache was preserved."
        )
    temporary = out.with_suffix(".tmp")
    columns = list(rows[0]) if rows else ["object_id", "cone_index"]
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, out)
    meta = {
        "catalog_version": CATALOG_VERSION,
        "cone_count": len(centers),
        "cones": cone_meta,
        "selection_method": "random saved stars without replacement",
        "selection_seed": int(selection_seed),
        "minimum_center_separation_arcmin": 2.0 * float(radius_arcmin),
        "radius_arcmin": float(radius_arcmin),
        "area_arcmin2": len(centers) * math.pi * float(radius_arcmin) ** 2,
        "rows": len(rows),
        "counts": {
            kind: sum(row.get("type") == kind for row in rows)
            for kind in ("star", "galaxy", "unknown")
        },
        "classification": (
            "POINT_LIKE_PROB is fractional stellar membership; galaxy weight "
            "is 1 − POINT_LIKE_PROB; invalid probabilities are excluded"
        ),
        "classification_note": (
            "Aggregated from random non-overlapping cones centred on locally "
            "saved Euclid stars; object_id duplicates are removed."
        ),
        "photometry": (
            "3 FWHM PSF-matched aperture magnitudes, raw fluxes/errors, and "
            "colours; VIS PSF flux is retained separately"
        ),
        "probability_coverage": {
            "field": "point_like_prob",
            "valid_rows": sum(
                _finite(row.get("point_like_prob")) is not None
                and 0.0 <= float(row["point_like_prob"]) <= 1.0
                for row in rows
            ),
            "missing_or_invalid_rows": sum(
                _finite(row.get("point_like_prob")) is None
                or not 0.0 <= float(row["point_like_prob"]) <= 1.0
                for row in rows
            ),
        },
    }
    _write_json(euclid_catalog_meta_path(), meta)
    if progress:
        progress(len(centers), len(centers), "Euclid cones cached")
    return meta


def refresh_cached_euclid_population_multi(
    *, progress: Callable[[int, int, str], None] | None = None,
    relogin: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Refresh raw Euclid photometry at the exact saved cone centers."""
    saved = _read_json(euclid_catalog_meta_path())
    if not saved or not saved.get("cones"):
        raise ValueError("no saved multi-cone footprint is available")
    centers = list(saved["cones"])
    radius = float(saved.get("radius_arcmin") or 0.0)
    if radius <= 0.0:
        raise ValueError("saved cone metadata has no valid radius")
    return query_euclid_population_multi(
        count=len(centers),
        radius_arcmin=radius,
        selection_seed=saved.get("selection_seed"),
        centers=centers,
        progress=progress,
        relogin=relogin,
    )
