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
import warnings
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astroquery.esa.euclid import Euclid

from euclid_polish.config import Config
from euclid_polish.photometry import electrons_to_ab_mag, uJy_to_ab_mag
from euclid_polish.web.helpers.paths import _sky_records_local_dir

VERSION = 4
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

_PARAM_META = {
    "objects_per_field": ("objects per field", "count"),
    "mag_vis": ("VIS magnitude", "AB mag"),
    "mag_y_e": ("Y_E magnitude", "AB mag"),
    "mag_j_e": ("J_E magnitude", "AB mag"),
    "mag_h_e": ("H_E magnitude", "AB mag"),
    "flux_vis_e": ("VIS source flux", "e⁻ / stack"),
    "flux_vis_psf_uJy": ("VIS PSF flux", "µJy"),
    "fluxerr_vis_psf_uJy": ("VIS PSF flux error", "µJy"),
    "vis_snr": ("VIS PSF signal-to-noise", "ratio"),
    "segmentation_area": ("segmentation area", "VIS pixels"),
    "z": ("redshift", "z"),
    "re_arcsec": ("half-light radius", "arcsec"),
    "logmass": ("stellar mass", "log₁₀ M☉"),
    "mass_scale": ("TNG mass scale", "ratio"),
    "temperature_k": ("stellar temperature", "K"),
    "extinction_av": ("stellar extinction Aᵥ", "mag"),
}


def cache_dir() -> Path:
    return Path(Config.DATA_DIR) / "population_comparison"


def comparison_path() -> Path:
    return cache_dir() / "comparison.json"


def euclid_catalog_path() -> Path:
    return cache_dir() / "euclid_population.csv"


def euclid_catalog_meta_path() -> Path:
    return cache_dir() / "euclid_population_meta.json"


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open() as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def read_comparison() -> dict[str, Any] | None:
    payload = _read_json(comparison_path())
    return payload if payload and payload.get("version") == VERSION else None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(temporary, path)


def _synthetic_paths() -> tuple[list[Path], list[Path]]:
    root = Path(_sky_records_local_dir())
    records = [
        root / f"dirty_{subset}.tfrecord"
        for subset in ("test", "validate")
        if (root / f"dirty_{subset}.tfrecord").is_file()
    ]
    sources = [
        root / f"sources_{subset}.csv"
        for subset in ("test", "validate", "train")
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
    inference, overlap = _real_field_sources()
    synthetic_fields = sum(_count_tfrecord(path) for path in records)
    population_fields = _source_field_count(source_csvs)
    inference_fields = 100 * len(inference)
    real_fields = inference_fields + len(overlap)
    meta = _read_json(euclid_catalog_meta_path())
    return {
        "synthetic": {
            "fields": synthetic_fields,
            "area_arcmin2": synthetic_fields * FIELD_AREA_ARCMIN2,
            "population_fields": population_fields,
            "population_area_arcmin2": population_fields * FIELD_AREA_ARCMIN2,
            "record_files": len(records),
            "source_catalogs": len(source_csvs),
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
            "cached": euclid_catalog_path().is_file() and meta is not None,
            "meta": meta,
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
        self.count = 0
        self.window = np.outer(np.hanning(ANALYSIS_SIZE),
                               np.hanning(ANALYSIS_SIZE)).astype(np.float32)

    def add(self, field: np.ndarray) -> None:
        data = _normalise_field(field)
        finite = np.where(np.isfinite(data), data, 0.0)
        self.count += 1
        for band in range(len(BANDS)):
            plane = finite[..., band]
            self.samples[band].append(plane[::8, ::8].reshape(-1))
            self.field_means[band].append(float(np.mean(plane)))
            centered = (plane - np.mean(plane)) * self.window
            fft = np.fft.rfft2(centered)
            self.power[band].append(
                _radial_average(np.abs(fft) ** 2, self.radius, self.k_edges)
            )


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

    def field_summary(acc: _FieldAccumulator) -> dict[str, Any]:
        return {
            band: {
                "mean": float(np.mean(acc.field_means[index])),
                "median": float(np.median(acc.field_means[index])),
                "p16": float(np.percentile(acc.field_means[index], 16)),
                "p84": float(np.percentile(acc.field_means[index], 84)),
            }
            for index, band in enumerate(BANDS)
        }

    return {
        "bands": list(BANDS),
        "histograms": histograms,
        "power": power,
        "scale_similarity": scale_similarity,
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
                    "field_index": field_index,
                }
                for key in _PARAM_META:
                    if key == "objects_per_field":
                        continue
                    row[key] = _finite(raw.get(key))
                if row.get("mag_vis") is None and row.get("flux_vis_e"):
                    row["mag_vis"] = _finite(electrons_to_ab_mag(
                        row["flux_vis_e"], Config.BAND_VIS
                    ))
                rows.append(row)
        offset += local_max + 1
    return rows


def _read_euclid_sources() -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    path = euclid_catalog_path()
    if not path.is_file():
        return [], None
    rows: list[dict[str, Any]] = []
    with path.open(newline="") as handle:
        for raw in csv.DictReader(handle):
            row = {"type": raw.get("type", "unknown")}
            for key in _PARAM_META:
                if key != "objects_per_field":
                    row[key] = _finite(raw.get(key))
            rows.append(row)
    return rows, _read_json(euclid_catalog_meta_path())


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


def _population_payload(source_csvs: Iterable[Path],
                        synthetic_field_count: int) -> dict[str, Any]:
    paths = list(source_csvs)
    synthetic_rows = _read_synthetic_sources(paths)
    euclid_rows, euclid_meta = _read_euclid_sources()
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
        "euclid_meta": euclid_meta,
    }


def refresh_population_comparison() -> dict[str, Any] | None:
    """Refresh source-population statistics without rereading image fields."""
    payload = read_comparison()
    if payload is None:
        return None
    _, source_csvs = _synthetic_paths()
    synthetic_field_count = int(
        payload.get("samples", {}).get("synthetic", {}).get("fields", 0)
    )
    population = _population_payload(source_csvs, synthetic_field_count)
    payload["population"] = population
    _write_json(comparison_path(), payload)
    return population


def build_comparison(
    progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    records, source_csvs = _synthetic_paths()
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
    for field in _synthetic_fields(records):
        synthetic_acc.add(field)
        done += 1
        if progress:
            progress(done, total, "synthetic LR fields")

    real_acc = _FieldAccumulator()
    for field in _real_fields(inference, overlap):
        real_acc.add(field)
        done += 1
        if progress:
            progress(done, total, "real Euclid LR fields")

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
        "fields": _field_payload(synthetic_acc, real_acc),
        "population": _population_payload(source_csvs, synthetic_acc.count),
    }
    _write_json(comparison_path(), payload)
    return payload


def query_euclid_population(
    ra: float,
    dec: float,
    radius_arcmin: float,
    *,
    limit: int = MAX_CATALOG_ROWS,
) -> dict[str, Any]:
    """Query clean MER sources in a cone and cache every selected parameter."""
    radius_deg = radius_arcmin / 60.0
    query = f"""
    SELECT TOP {int(limit)}
        object_id, right_ascension, declination, point_like_flag,
        segmentation_area, flux_vis_psf, fluxerr_vis_psf
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
    results = job.get_results() if job is not None else []
    rows: list[dict[str, Any]] = []
    for raw in ([] if results is None else results):
        flux = _finite(raw["flux_vis_psf"])
        flux_error = _finite(raw["fluxerr_vis_psf"])
        if flux is None or flux <= 0:
            continue
        point_like = _finite(raw["point_like_flag"])
        rows.append({
            "object_id": str(raw["object_id"]),
            "type": "star" if point_like == 1 else "galaxy",
            "ra": _finite(raw["right_ascension"]),
            "dec": _finite(raw["declination"]),
            "mag_vis": uJy_to_ab_mag(flux),
            "flux_vis_psf_uJy": flux,
            "fluxerr_vis_psf_uJy": flux_error,
            "vis_snr": (
                flux / flux_error if flux_error is not None and flux_error > 0
                else None
            ),
            "segmentation_area": _finite(raw["segmentation_area"]),
        })

    out = euclid_catalog_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(".tmp")
    columns = [
        "object_id", "type", "ra", "dec", "mag_vis", "flux_vis_psf_uJy",
        "fluxerr_vis_psf_uJy", "vis_snr", "segmentation_area",
    ]
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, out)

    meta = {
        "ra": float(ra),
        "dec": float(dec),
        "radius_arcmin": float(radius_arcmin),
        "area_arcmin2": math.pi * float(radius_arcmin) ** 2,
        "rows": len(rows),
        "limit": int(limit),
        "limit_reached": len(rows) >= int(limit),
        "classification": (
            "point_like_flag = 1 → star; clean non-spurious remainder → galaxy"
        ),
    }
    _write_json(euclid_catalog_meta_path(), meta)
    return meta
