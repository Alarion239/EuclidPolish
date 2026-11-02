"""Empirical Euclid stellar colours for synthetic point sources.

Colours come from Euclid alone: a star of VIS magnitude ``m`` adopts the
colours (VIS−Y, Y−J, J−H) of one real Q1 point source in the 0.5-mag bin
holding ``m``, drawn by its PHZ star probability, with measurement noise
removed by one-component extreme deconvolution within that bin (Bovy, Hogg &
Roweis 2011). This is the galaxy colour forest's resampling with magnitude as
the only conditioning variable, so the forest's leaves reduce to magnitude
bins. A bin with too little star weight (the saturated bright end, the faint
end where Q1 stops classifying point sources) uses its nearest populated bin.

Unlike the galaxy forest, the deconvolution works on the three colours
jointly and in magnitudes. Stars lie on a tight, compact locus in colour
space (Y−J and J−H barely move while VIS−Y spans a magnitude), and adjacent
colours share a band's error. In flux-ratio space the same locus is a long
ray, whose narrow Y−J direction a single Gaussian estimates badly. Magnitudes
are safe here: colour rows need S/N ≥ 5 in every band.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from euclid_polish.population._codec import array_sha256, decode_array, encode_array
from euclid_polish.population.magnitude_law import StraightMagnitudeLaw

STAR_COLOR_MODEL_KIND = "q1_point_source_color_resampling_v1"
#: Width of the magnitude bins that play the forest leaves' role.
STAR_COLOR_BIN_WIDTH_MAG = 0.5
#: Minimum PHZ star weight a bin needs to be sampled on its own.
STAR_COLOR_MIN_BIN_WEIGHT = 25.0
#: Modelled colours, in payload order.
STAR_COLOR_NAMES = ("vis_y", "y_j", "j_h")
#: Rows further than this many robust sigmas from the bin median in any
#: colour are left out of the bin's correlation (they are still drawn).
_COVARIANCE_CLIP_SIGMA = 4.0
#: 2.5 / ln 10: magnitude error per unit fractional flux error.
_MAG_PER_FRACTIONAL_FLUX = 2.5 / math.log(10.0)


def _psd(matrix: np.ndarray) -> np.ndarray:
    """Symmetric positive-semidefinite projection (negative eigenvalues → 0)."""
    symmetric = 0.5 * (matrix + matrix.T)
    values, vectors = np.linalg.eigh(symmetric)
    projected = (vectors * np.maximum(values, 0.0)) @ vectors.T
    return 0.5 * (projected + projected.T)


def color_noise_covariance(mag_sigma: np.ndarray) -> np.ndarray:
    """Noise covariance of (VIS−Y, Y−J, J−H) from per-band magnitude errors.

    ``mag_sigma`` has shape (..., 4) for VIS, Y, J, H. Adjacent colours share
    a band, so their errors anticorrelate.
    """
    v = np.asarray(mag_sigma, dtype=np.float64) ** 2
    zero = np.zeros_like(v[..., 0])
    return np.stack([
        np.stack([v[..., 0] + v[..., 1], -v[..., 1], zero], axis=-1),
        np.stack([-v[..., 1], v[..., 1] + v[..., 2], -v[..., 2]], axis=-1),
        np.stack([zero, -v[..., 2], v[..., 2] + v[..., 3]], axis=-1),
    ], axis=-2)


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values, kind="mergesort")
    cumulative = np.cumsum(weights[order])
    grid = (cumulative - 0.5 * weights[order]) / cumulative[-1]
    return float(np.interp(quantile, grid, values[order]))


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    return _weighted_quantile(values, weights, 0.5)


def bin_moments(
    color: np.ndarray,
    noise_covariance: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """(location, intrinsic covariance) of one bin's colours.

    Location is the weighted median of each colour. The observed covariance
    has each colour's weighted 16–84 half-width as its scale (as the galaxy
    forest uses) and the correlation of the rows within 4 robust sigmas in
    every colour. The noise covariance is the weighted mean over rows whose
    noise trace is below 25× the median (a few percent of MER rows report
    absurd errors, which must not swamp the noise term). Intrinsic =
    observed − noise, projected to positive semidefinite.
    """
    location = np.asarray(
        [_weighted_median(color[:, index], weights) for index in range(3)],
    )
    deviation = color - location
    scale = np.asarray([
        max(1.4826 * _weighted_median(np.abs(deviation[:, index]), weights), 1e-12)
        for index in range(3)
    ])
    kept = np.all(np.abs(deviation) <= _COVARIANCE_CLIP_SIGMA * scale, axis=1)
    if np.count_nonzero(kept) < 4:
        kept = np.ones(color.shape[0], dtype=bool)
    kept_weights = weights[kept]
    centred = color[kept] - np.average(color[kept], axis=0, weights=kept_weights)
    clipped = (centred * kept_weights[:, None]).T @ centred / np.sum(kept_weights)
    sigma = np.sqrt(np.maximum(np.diag(clipped), 1e-300))
    half_width = np.asarray([
        0.5 * (_weighted_quantile(color[:, index], weights, 0.84)
               - _weighted_quantile(color[:, index], weights, 0.16))
        for index in range(3)
    ])
    observed = clipped / np.outer(sigma, sigma) * np.outer(half_width, half_width)
    trace = np.trace(noise_covariance, axis1=1, axis2=2)
    sane = trace <= 25.0 * max(_weighted_median(trace, weights), 1e-300)
    noise = np.average(noise_covariance[sane], axis=0, weights=weights[sane])
    return location, _psd(observed - noise)


def deconvolved_draw(
    location: np.ndarray,
    intrinsic: np.ndarray,
    row_color: np.ndarray,
    row_noise: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Posterior draw of one row's true colours (analytic XD, all three jointly).

    The gain is the matrix K = Σ (Σ + C)⁻¹ for intrinsic covariance Σ and the
    row's noise covariance C: a precise row keeps its measurement (K → 1), a
    noise-dominated one falls back to the bin centre plus the intrinsic
    scatter (K → 0), along the locus rather than colour by colour.
    """
    gain = intrinsic @ np.linalg.pinv(intrinsic + row_noise, hermitian=True)
    mean = location + gain @ (row_color - location)
    covariance = _psd((np.eye(3) - gain) @ intrinsic)
    values, vectors = np.linalg.eigh(covariance)
    return mean + vectors @ (np.sqrt(np.maximum(values, 0.0)) * rng.standard_normal(3))


@dataclass(frozen=True)
class StellarSED:
    """One sampled star: four-band AB magnitudes at a supplied VIS magnitude."""

    magnitudes: dict[str, float]


def _populated_bin_sources(bin_weight: np.ndarray, min_weight: float) -> np.ndarray:
    """Each bin's own index if populated, else its nearest populated bin."""
    populated = np.flatnonzero(bin_weight >= min_weight)
    if not populated.size:
        raise ValueError("no stellar colour bin reaches the minimum star weight")
    index = np.arange(bin_weight.size)
    nearest = np.abs(index[:, None] - populated[None, :]).argmin(axis=1)
    return populated[nearest].astype(np.int32)


def magnitude_errors(flux: np.ndarray, flux_error: np.ndarray) -> np.ndarray:
    """Linearised AB magnitude errors 2.5/ln10 · σ_F / F."""
    return _MAG_PER_FRACTIONAL_FLUX * np.asarray(flux_error) / np.asarray(flux)


def build_star_color_model(
    magnitude: np.ndarray,
    color: np.ndarray,
    mag_sigma: np.ndarray,
    weight: np.ndarray,
    *,
    mag_bright: float,
    mag_faint: float,
    bin_width: float = STAR_COLOR_BIN_WIDTH_MAG,
    min_bin_weight: float = STAR_COLOR_MIN_BIN_WEIGHT,
) -> dict[str, Any]:
    """The ``color_model`` payload section from selected Q1 point-source rows.

    ``magnitude`` is each row's VIS PSF magnitude, ``color`` its measured
    (VIS−Y, Y−J, J−H) (shape (n, 3)), ``mag_sigma`` its VIS/Y/J/H magnitude
    errors (shape (n, 4); :func:`color_noise_covariance`), and ``weight`` its
    PHZ star probability.
    """
    magnitude = np.asarray(magnitude, dtype=np.float64)
    color = np.asarray(color, dtype=np.float64)
    mag_sigma = np.asarray(mag_sigma, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    count = magnitude.size
    if not (
        magnitude.ndim == 1 and color.shape == (count, 3)
        and mag_sigma.shape == (count, 4) and weight.shape == (count,)
    ):
        raise ValueError("stellar colour rows have inconsistent shapes")
    if not (
        np.all(np.isfinite(magnitude)) and np.all(np.isfinite(color))
        and np.all(np.isfinite(mag_sigma)) and np.all(mag_sigma >= 0.0)
        and np.all(np.isfinite(weight)) and np.all(weight > 0.0)
    ):
        raise ValueError("stellar colour rows must be finite with positive weights")
    bin_count = int(round((mag_faint - mag_bright) / bin_width))
    if bin_count < 1 or not math.isclose(
        mag_bright + bin_count * bin_width, mag_faint, abs_tol=1e-9,
    ):
        raise ValueError("stellar colour range must divide into whole bins")
    edges = mag_bright + bin_width * np.arange(bin_count + 1)
    row_bin = np.clip(
        np.searchsorted(edges, magnitude, side="right") - 1, 0, bin_count - 1,
    ).astype(np.int32)
    bin_weight = np.bincount(row_bin, weights=weight, minlength=bin_count)
    sources = _populated_bin_sources(bin_weight, min_bin_weight)
    return {
        "kind": STAR_COLOR_MODEL_KIND,
        "color_space": "ab_colors_vis_y_y_j_j_h",
        "deconvolution": "one-component extreme deconvolution of the three "
        "colours jointly within each 0.5-mag bin (full covariance)",
        "color_names": list(STAR_COLOR_NAMES),
        "magnitude_edges": edges.tolist(),
        "bin_weight": bin_weight.tolist(),
        "bin_source": sources.tolist(),
        "min_bin_weight": float(min_bin_weight),
        "row_count": int(count),
        "rows": {
            "bin_zlib_base64": encode_array(row_bin, "<i4"),
            "bin_sha256": array_sha256(row_bin, "<i4"),
            "color_zlib_base64": encode_array(color, "<f8"),
            "color_sha256": array_sha256(color, "<f8"),
            "mag_sigma_zlib_base64": encode_array(mag_sigma, "<f8"),
            "mag_sigma_sha256": array_sha256(mag_sigma, "<f8"),
            "weight_zlib_base64": encode_array(weight, "<f8"),
            "weight_sha256": array_sha256(weight, "<f8"),
        },
    }


@dataclass(frozen=True)
class StarColorModel:
    """Validated arrays of a ``q1_point_source_color_resampling_v1`` section."""

    magnitude_edges: np.ndarray
    bin_source: np.ndarray
    color: np.ndarray
    #: Per-row (3, 3) noise covariance of the colours.
    noise_covariance: np.ndarray
    weight: np.ndarray
    bin_rows: tuple[np.ndarray, ...]
    #: (location, intrinsic covariance) of each bin; None for an empty bin.
    bin_moments: tuple[tuple[np.ndarray, np.ndarray] | None, ...]

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> StarColorModel:
        if payload.get("kind") != STAR_COLOR_MODEL_KIND:
            raise ValueError("stellar colour model has the wrong kind")
        try:
            edges = np.asarray(payload["magnitude_edges"], dtype=np.float64)
            sources = np.asarray(payload["bin_source"], dtype=np.int64)
            count = int(payload["row_count"])
            rows = payload["rows"]
            row_bin = decode_array(
                rows["bin_zlib_base64"], (count,), "<i4",
                sha256=rows.get("bin_sha256"), name="stellar colour bins",
            )
            color = decode_array(
                rows["color_zlib_base64"], (count, 3), "<f8",
                sha256=rows.get("color_sha256"), name="stellar colours",
            )
            mag_sigma = decode_array(
                rows["mag_sigma_zlib_base64"], (count, 4), "<f8",
                sha256=rows.get("mag_sigma_sha256"),
                name="stellar magnitude errors",
            )
            weight = decode_array(
                rows["weight_zlib_base64"], (count,), "<f8",
                sha256=rows.get("weight_sha256"), name="stellar colour weights",
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("stellar colour model is malformed") from exc
        bins = edges.size - 1
        if not (
            edges.ndim == 1 and bins >= 1 and np.all(np.diff(edges) > 0.0)
            and sources.shape == (bins,)
            and np.all((sources >= 0) & (sources < bins))
            and np.all((row_bin >= 0) & (row_bin < bins))
            and np.all(np.isfinite(color))
            and np.all(np.isfinite(mag_sigma)) and np.all(mag_sigma >= 0.0)
            and np.all(np.isfinite(weight)) and np.all(weight > 0.0)
        ):
            raise ValueError("stellar colour model is invalid")
        bin_rows = tuple(np.flatnonzero(row_bin == index) for index in range(bins))
        if any(not bin_rows[int(source)].size for source in sources):
            raise ValueError("stellar colour model points at an empty bin")
        noise_covariance = color_noise_covariance(mag_sigma)
        moments = tuple(
            bin_moments(color[rows], noise_covariance[rows], weight[rows])
            if rows.size else None
            for rows in bin_rows
        )
        return cls(
            magnitude_edges=edges, bin_source=sources.astype(np.int32),
            color=color, noise_covariance=noise_covariance, weight=weight,
            bin_rows=bin_rows, bin_moments=moments,
        )

    def serving_bin(self, mag_vis: float) -> int:
        """Index of the populated bin that serves ``mag_vis``."""
        edges = self.magnitude_edges
        index = int(np.searchsorted(edges, mag_vis, side="right") - 1)
        index = max(0, min(index, edges.size - 2))
        return int(self.bin_source[index])

    def sample_colors(self, rng: np.random.Generator, mag_vis: float) -> np.ndarray:
        """True (VIS−Y, Y−J, J−H) of one star: a resampled, deconvolved Q1 row."""
        source = self.serving_bin(mag_vis)
        rows = self.bin_rows[source]
        location, intrinsic = self.bin_moments[source]
        weights = self.weight[rows]
        row = int(rows[int(rng.choice(rows.size, p=weights / np.sum(weights)))])
        return deconvolved_draw(
            location, intrinsic, self.color[row], self.noise_covariance[row], rng,
        )


@dataclass(frozen=True)
class EmpiricalStellarPrior:
    """Activated Q1 star-count law plus the Euclid colour resampler."""

    magnitude_law: StraightMagnitudeLaw
    color_model: StarColorModel

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> EmpiricalStellarPrior:
        distribution = payload.get("population", {}).get("magnitude_distribution", {})
        try:
            magnitude_law = StraightMagnitudeLaw.from_payload(distribution)
        except ValueError as exc:
            raise ValueError(
                "stellar prior requires a straight magnitude-count law"
            ) from exc
        try:
            color_model = StarColorModel.from_payload(payload.get("color_model") or {})
        except ValueError as exc:
            raise ValueError(
                "stellar prior requires a Euclid point-source colour model; "
                "refit the stellar population"
            ) from exc
        return cls(magnitude_law, color_model)

    def sample_magnitude(self, rng: np.random.Generator) -> float:
        """Sample VIS magnitude from the activated straight count law."""
        return self.magnitude_law.sample(rng)

    def sample(self, rng: np.random.Generator, mag_vis: float) -> StellarSED:
        vis_y, y_j, j_h = self.color_model.sample_colors(rng, mag_vis)
        y = float(mag_vis) - float(vis_y)
        j = y - float(y_j)
        return StellarSED(magnitudes={
            "VIS": float(mag_vis), "Y_E": y, "J_E": j, "H_E": j - float(j_h),
        })


def sample_stellar_sed(
    rng: np.random.Generator, mag_vis: float,
    prior: EmpiricalStellarPrior | None = None,
) -> StellarSED:
    """Draw one star from the required empirical calibration."""
    if prior is None:
        raise ValueError(
            "an active empirical stellar prior is required for star generation"
        )
    return prior.sample(rng, mag_vis)


def stellar_prior_density_arcmin2(payload: dict[str, Any]) -> float:
    """Star surface density (per arcmin²) of an activated stellar prior.

    Reads ``population.density_arcmin2``, the value FASRC generation uses for
    the fixed validate/test stars, and falls back to the integral of the
    prior's magnitude-count law when that field is missing.
    """
    population = payload.get("population") or {}
    density = population.get("density_arcmin2")
    if density is None:
        law = StraightMagnitudeLaw.from_payload(
            population.get("magnitude_distribution") or {},
        )
        density = law.integrated_density()
    value = float(density)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(
            f"stellar prior density must be finite and >= 0, got {value!r}"
        )
    return value
