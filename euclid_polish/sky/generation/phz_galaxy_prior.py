"""Empirical Euclid PHZ, apparent-size, and VIS-Kron galaxy prior.

The prior intentionally has no latent luminosity or size-evolution model.  A
compact three-dimensional grid is built directly from the cached MER+PHZ
catalogue: each object contributes its fractional galaxy membership times its
redshift PDF at its measured circularized size and detection-band Kron
magnitude.  TNG supplies morphology and colours; this prior supplies only the
observed redshift, apparent size, and VIS brightness anchor.
"""
from __future__ import annotations

import base64
import csv
import hashlib
import json
import math
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from euclid_polish.config import Config
from euclid_polish.photometry import ab_mag_to_electrons, uJy_to_ab_mag

PHZ_EMPIRICAL_KIND = "phz_empirical_kron_tng_draw"
PHZ_EMPIRICAL_VERSION = 1


@dataclass(frozen=True)
class PhzTngDraw:
    """One observed-space target for a TNG morphology."""

    catalog_id: str
    mag_hst_f814w: float
    target_vis_mag: float
    target_vis_flux_e: float
    z: float
    logmass: float
    re_arcsec: float
    imputed_size: bool
    brightness_transfer: str
    mass_quantile: float = float("nan")
    ssfr_quantile: float = float("nan")
    activity_class: str = "unconditioned"
    logssfr: float = float("nan")
    physical_model_fingerprint: str = ""
    target_vis_estimator: str = "MER detection-band Kron flux"
    radius_estimator: str = "MER circularized SEMIMAJOR_AXIS proxy"


def _covering_edges(values: np.ndarray, step: float) -> np.ndarray:
    """Return regular bin edges that include every finite input value."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError("empirical population coordinates must be finite")
    lower = math.floor(float(np.min(values)) / step) * step
    upper = math.ceil(float(np.max(values)) / step) * step
    if upper <= float(np.max(values)) + 1e-12:
        upper += step
    edges = np.arange(lower, upper + 0.5 * step, step, dtype=np.float64)
    if edges.size < 2 or edges.size > 257:
        raise ValueError("empirical population coordinate range is unreasonable")
    return edges


def _encoded_density(density: np.ndarray) -> str:
    raw = np.asarray(density, dtype="<f4", order="C").tobytes(order="C")
    return base64.b64encode(zlib.compress(raw, level=9)).decode("ascii")


def _decoded_density(payload: dict[str, Any]) -> np.ndarray:
    grid = payload.get("grid") or {}
    try:
        shape = tuple(int(value) for value in grid["shape"])
        encoded = str(grid["density_zlib_base64"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("PHZ empirical density encoding is incomplete") from exc
    if len(shape) != 3 or any(value <= 0 for value in shape):
        raise ValueError("PHZ empirical density shape is invalid")
    try:
        raw = zlib.decompress(base64.b64decode(encoded, validate=True))
    except (ValueError, zlib.error) as exc:
        raise ValueError("PHZ empirical density encoding is invalid") from exc
    expected_bytes = int(np.prod(shape, dtype=np.int64)) * 4
    if len(raw) != expected_bytes:
        raise ValueError("PHZ empirical density byte count is invalid")
    return np.frombuffer(raw, dtype="<f4").reshape(shape).astype(np.float64)


def build_phz_galaxy_population_payload(
    catalog_path: str | Path,
    pdf_path: str | Path,
    meta_path: str | Path,
    *,
    magnitude_bin_width: float = 0.20,
    log_radius_bin_width: float = 0.04,
) -> dict[str, Any]:
    """Build compact weighted ``p(z, m_Kron, r_MER)`` from cached Euclid data.

    Only rows with a normalized PHZ PDF, positive Kron flux, a finite MER size,
    and a valid ``POINT_LIKE_PROB`` are used.  The weight is
    ``1 - POINT_LIKE_PROB``, matching the project's fractional galaxy policy.
    No magnitude or radius clipping is applied, so rare bright and large
    catalogue objects remain in the draw distribution.
    """
    catalog_path = Path(catalog_path)
    pdf_path = Path(pdf_path)
    meta_path = Path(meta_path)
    try:
        meta = json.loads(meta_path.read_text())
        area_arcmin2 = float(meta["area_arcmin2"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError("Euclid population metadata has no valid area") from exc
    if not np.isfinite(area_arcmin2) or area_arcmin2 <= 0.0:
        raise ValueError("Euclid population area must be positive")

    try:
        with np.load(pdf_path, allow_pickle=False) as data:
            pdf_ids = np.asarray(data["object_id"]).astype(str)
            probability = np.asarray(data["probability"], dtype=np.float64)
            z_edges = np.asarray(data["z_edges"], dtype=np.float64)
    except (OSError, KeyError, ValueError) as exc:
        raise ValueError("Euclid PHZ PDF cache is unavailable or malformed") from exc
    if probability.shape != (pdf_ids.size, z_edges.size - 1):
        raise ValueError("Euclid PHZ PDF arrays are not aligned")
    if (
        z_edges.ndim != 1 or z_edges.size < 2
        or not np.isfinite(z_edges).all() or np.any(np.diff(z_edges) <= 0.0)
        or not np.isfinite(probability).all() or np.any(probability < 0.0)
        or not np.allclose(probability.sum(axis=1), 1.0, rtol=1e-5, atol=1e-6)
    ):
        raise ValueError("Euclid PHZ PDFs or redshift edges are invalid")
    if pdf_ids.size != np.unique(pdf_ids).size:
        raise ValueError("Euclid PHZ PDF cache contains duplicate object IDs")
    pdf_index = {object_id: index for index, object_id in enumerate(pdf_ids)}

    selected_pdf_index: list[int] = []
    magnitude: list[float] = []
    log_radius: list[float] = []
    weight: list[float] = []
    catalog_rows = 0
    with catalog_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            catalog_rows += 1
            index = pdf_index.get(str(row.get("object_id") or ""))
            if index is None:
                continue
            try:
                point_like = float(row["point_like_prob"])
                spurious = float(row.get("spurious_prob") or 0.0)
                semimajor = float(row["semimajor_axis"])
                ellipticity = float(row["ellipticity"])
                kron_flux_uJy = float(row["flux_detection_total_uJy"])
            except (KeyError, TypeError, ValueError):
                continue
            if not (
                np.isfinite(point_like) and 0.0 <= point_like <= 1.0
                and np.isfinite(spurious) and spurious <= 0.5
                and np.isfinite(semimajor) and semimajor > 0.0
                and np.isfinite(ellipticity) and 0.0 <= ellipticity < 1.0
                and np.isfinite(kron_flux_uJy) and kron_flux_uJy > 0.0
            ):
                continue
            galaxy_weight = 1.0 - point_like
            if galaxy_weight <= 0.0:
                continue
            radius_arcsec = (
                Config.VIS_PIXEL_SCALE_ARCSEC
                * semimajor
                * math.sqrt(1.0 - ellipticity)
            )
            kron_magnitude = float(uJy_to_ab_mag(kron_flux_uJy))
            if not (
                np.isfinite(radius_arcsec) and radius_arcsec > 0.0
                and np.isfinite(kron_magnitude)
            ):
                continue
            selected_pdf_index.append(index)
            magnitude.append(kron_magnitude)
            log_radius.append(math.log10(radius_arcsec))
            weight.append(galaxy_weight)

    if not weight or float(np.sum(weight)) <= 0.0:
        raise ValueError("Euclid cache has no usable PHZ Kron galaxy rows")
    magnitude_array = np.asarray(magnitude, dtype=np.float64)
    log_radius_array = np.asarray(log_radius, dtype=np.float64)
    weight_array = np.asarray(weight, dtype=np.float64)
    magnitude_edges = _covering_edges(magnitude_array, magnitude_bin_width)
    log_radius_edges = _covering_edges(log_radius_array, log_radius_bin_width)
    magnitude_index = np.searchsorted(
        magnitude_edges, magnitude_array, side="right",
    ) - 1
    radius_index = np.searchsorted(
        log_radius_edges, log_radius_array, side="right",
    ) - 1
    if (
        np.any(magnitude_index < 0)
        or np.any(magnitude_index >= magnitude_edges.size - 1)
        or np.any(radius_index < 0)
        or np.any(radius_index >= log_radius_edges.size - 1)
    ):
        raise RuntimeError("empirical population binning lost an outlier")

    density = np.zeros(
        (
            z_edges.size - 1,
            magnitude_edges.size - 1,
            log_radius_edges.size - 1,
        ),
        dtype=np.float64,
    )
    for pdf_row, magnitude_bin, radius_bin, galaxy_weight in zip(
        np.asarray(selected_pdf_index, dtype=np.int64),
        magnitude_index,
        radius_index,
        weight_array,
        strict=True,
    ):
        density[:, magnitude_bin, radius_bin] += (
            galaxy_weight * probability[pdf_row]
        )
    density /= area_arcmin2
    density32 = np.asarray(density, dtype="<f4")
    realised_density = float(np.sum(density32, dtype=np.float64))

    identity = {
        "version": PHZ_EMPIRICAL_VERSION,
        "kind": PHZ_EMPIRICAL_KIND,
        "catalog_version": meta.get("catalog_version"),
        "area_arcmin2": area_arcmin2,
        "catalog_rows": catalog_rows,
        "selected_rows": len(weight),
        "z_edges": z_edges.tolist(),
        "kron_magnitude_edges": magnitude_edges.tolist(),
        "log_radius_edges": log_radius_edges.tolist(),
        "density_sha256": hashlib.sha256(density32.tobytes()).hexdigest(),
    }
    fingerprint = hashlib.sha256(json.dumps(
        identity, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    return {
        "version": PHZ_EMPIRICAL_VERSION,
        "kind": PHZ_EMPIRICAL_KIND,
        "valid": True,
        "active": True,
        "validated": False,
        "fingerprint": fingerprint,
        "source": {
            "catalog_version": meta.get("catalog_version"),
            "area_arcmin2": area_arcmin2,
            "catalog_rows": catalog_rows,
            "selected_rows": len(weight),
            "selected_galaxy_weight": float(np.sum(weight_array)),
        },
        "selection": {
            "classification_weight": "1 - POINT_LIKE_PROB",
            "maximum_spurious_probability": 0.5,
            "requires_normalized_phz_pdf": True,
            "magnitude_clipping": None,
            "radius_clipping": None,
        },
        "measurement_model": {
            "redshift": "per-object PHZ redshift PDF",
            "radius": (
                "VIS_PIXEL_SCALE_ARCSEC * SEMIMAJOR_AXIS * "
                "sqrt(1 - ELLIPTICITY); a detection proxy, not fitted R_e"
            ),
            "brightness": "MER FLUX_DETECTION_TOTAL detection-band Kron flux",
            "rendering_anchor": (
                "after apparent-size matching, set integrated clean TNG VIS "
                "flux equal to the sampled Kron flux; no post-PSF Kron "
                "remeasurement is claimed"
            ),
        },
        "generation": {
            "surface_density_arcmin2": realised_density,
            "position_process": "homogeneous_poisson",
            "morphology_assignment": "balanced_random_tng_atlas",
        },
        "grid": {
            "shape": list(density32.shape),
            "z_edges": z_edges.tolist(),
            "kron_magnitude_edges": magnitude_edges.tolist(),
            "log_radius_edges": log_radius_edges.tolist(),
            "density_unit": "objects / arcmin2 / cell",
            "density_encoding": "zlib+base64 little-endian float32 C-order",
            "density_sha256": identity["density_sha256"],
            "density_zlib_base64": _encoded_density(density32),
        },
    }


class PhzGalaxyPopulationPrior:
    """Sample the compact empirical PHZ/Kron/size grid."""

    morphology_mode = "balanced_random_tng_atlas"
    population_label = "phz_empirical_kron_v1"

    def __init__(self, payload: dict[str, Any]):
        if payload.get("version") != PHZ_EMPIRICAL_VERSION:
            raise ValueError("PHZ empirical population has an unsupported version")
        if payload.get("kind") != PHZ_EMPIRICAL_KIND:
            raise ValueError("PHZ empirical population has the wrong kind")
        if not payload.get("valid") or not payload.get("active"):
            raise ValueError("PHZ empirical population is not active and valid")
        self.fingerprint = str(payload.get("fingerprint") or "")
        if len(self.fingerprint) != 64:
            raise ValueError("PHZ empirical population fingerprint is invalid")
        grid = payload.get("grid") or {}
        self._z_edges = np.asarray(grid.get("z_edges"), dtype=np.float64)
        self._magnitude_edges = np.asarray(
            grid.get("kron_magnitude_edges"), dtype=np.float64,
        )
        self._log_radius_edges = np.asarray(
            grid.get("log_radius_edges"), dtype=np.float64,
        )
        density = _decoded_density(payload)
        density_digest = hashlib.sha256(
            np.asarray(density, dtype="<f4", order="C").tobytes(order="C")
        ).hexdigest()
        if density_digest != str(grid.get("density_sha256") or ""):
            raise ValueError("PHZ empirical density fingerprint is invalid")
        expected_shape = (
            self._z_edges.size - 1,
            self._magnitude_edges.size - 1,
            self._log_radius_edges.size - 1,
        )
        if density.shape != expected_shape:
            raise ValueError("PHZ empirical density and edges are not aligned")
        if (
            any(
                edges.ndim != 1 or edges.size < 2
                or not np.isfinite(edges).all() or np.any(np.diff(edges) <= 0.0)
                for edges in (
                    self._z_edges, self._magnitude_edges, self._log_radius_edges,
                )
            )
            or not np.isfinite(density).all()
            or np.any(density < 0.0)
            or float(np.sum(density)) <= 0.0
        ):
            raise ValueError("PHZ empirical grid contains invalid values")
        try:
            expected_density = float(
                payload["generation"]["surface_density_arcmin2"]
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("PHZ empirical population has no density") from exc
        realised_density = float(np.sum(density, dtype=np.float64))
        if not np.isclose(
            realised_density, expected_density, rtol=2e-6, atol=1e-9,
        ):
            raise ValueError("PHZ empirical population density is inconsistent")
        self.surface_density_arcmin2 = realised_density
        self._shape = density.shape
        self._cdf = np.cumsum(density.ravel())
        self._cdf /= self._cdf[-1]

    def __len__(self) -> int:
        return int(np.count_nonzero(np.diff(np.r_[0.0, self._cdf]) > 0.0))

    def sample(self, rng: np.random.Generator) -> PhzTngDraw:
        flat_index = min(
            int(np.searchsorted(self._cdf, rng.random(), side="right")),
            self._cdf.size - 1,
        )
        z_index, magnitude_index, radius_index = np.unravel_index(
            flat_index, self._shape,
        )
        redshift = float(rng.uniform(
            self._z_edges[z_index], self._z_edges[z_index + 1],
        ))
        kron_magnitude = float(rng.uniform(
            self._magnitude_edges[magnitude_index],
            self._magnitude_edges[magnitude_index + 1],
        ))
        log_radius = float(rng.uniform(
            self._log_radius_edges[radius_index],
            self._log_radius_edges[radius_index + 1],
        ))
        return PhzTngDraw(
            catalog_id=(
                f"phz:{self.fingerprint[:12]}:{z_index}:"
                f"{magnitude_index}:{radius_index}"
            ),
            mag_hst_f814w=float("nan"),
            target_vis_mag=kron_magnitude,
            target_vis_flux_e=float(ab_mag_to_electrons(
                kron_magnitude, Config.get_band("VIS"),
            )),
            z=redshift,
            logmass=float("nan"),
            re_arcsec=float(10.0**log_radius),
            imputed_size=False,
            brightness_transfer=(
                f"euclid_phz_kron_anchor:{self.fingerprint}:observed"
            ),
        )


def population_prior_from_payload(payload: dict[str, Any]):
    """Dispatch an embedded galaxy-population artifact by its explicit kind."""
    if payload.get("kind") == PHZ_EMPIRICAL_KIND:
        return PhzGalaxyPopulationPrior(payload)
    from euclid_polish.sky.generation.cosmos_tng_prior import (
        JointGalaxyPopulationPrior,
    )

    return JointGalaxyPopulationPrior(payload)
