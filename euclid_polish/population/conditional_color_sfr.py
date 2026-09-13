"""Empirical conditional colour+SFR sampler: forest leaf resampling.

The model is a conditional-distribution random forest in the quantile-forest
construction (Meinshausen 2006; Ćevid et al. 2022): trees partition the
(VIS 2FWHM magnitude, log10 R_e, R_e-resolved) feature space into regions of
homogeneous NISP/VIS colours and SFR, every leaf stores the *real* Euclid Q1
catalogue rows routed to it, and a draw resamples one catalogue row from the
weighted union of the query point's leaves across all trees. The joint of
(colours, SFR) inside a neighbourhood is therefore exactly empirical.

Measurement noise is removed by analytic one-component extreme deconvolution
(Bovy, Hogg & Roweis 2011) *within the drawn neighbourhood*: MER NISP fluxes
are forced photometry at the VIS position, so each row's flux ratio is
``true + N(0, var_row)`` with the per-row variance propagated from the
catalogue flux errors. Per draw, the neighbourhood's intrinsic variance is
its observed spread minus its reported noise variance (floored at zero) and
the drawn row's true ratio is sampled from its Gaussian posterior around the
neighbourhood location. Marginalized over rows this reproduces the intrinsic
distribution under the local-Gaussian model: bright rows come back
essentially raw (k → 1), while noise-dominated faint neighbourhoods collapse
toward the local colour relation — all the data can identify there. Negative
observed fluxes are informative and kept; the final ratio is floored at a
small positive value only against extreme posterior tails.

A few percent of MER rows report absurd flux errors (tens of µJy against a
~0.2 µJy typical σ), so every neighbourhood statistic is ROBUST: the location
is the weighted median, the observed spread the weighted 16–84 half-width,
and the noise level the weighted median of the reported variances. A junk-σ
row still enters a neighbourhood, but its own posterior gain k ≈ 0 collapses
it to the local relation, and it cannot poison the neighbourhood estimates.

Fitting lives in :mod:`euclid_polish.population.conditional_color_sfr_fit`
(the only place scikit-learn is imported); this module is pure NumPy so the
generation path never needs sklearn. Trees are stored as plain arrays — no
pickle — with SHA-256 digests pinning every array.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from euclid_polish.population._codec import decode_array

COLOR_SFR_MODEL_VERSION = 1
COLOR_SFR_MODEL_KIND = "conditional_color_sfr_forest"
#: Ensemble size; every tree contributes its leaf population to a draw.
COLOR_SFR_FOREST_TREES = 50
#: Minimum effective galaxy weight per leaf (sets the smoothing scale).
COLOR_SFR_MIN_LEAF_WEIGHT = 25.0
#: Seeded forest growth so refits on identical inputs are identical.
COLOR_SFR_FIT_SEED = 20260913
#: asinh softening (ratio units) for the split-criterion colour transform.
COLOR_SFR_ASINH_SOFTENING = 0.1
#: Posterior colour draws are floored at this NISP/VIS flux ratio.
COLOR_SFR_RATIO_FLOOR = 1e-4
#: Colour-training rows need this VIS 2FWHM S/N: below it the VIS flux sits
#: in the ratio DENOMINATOR at its own noise level, so the measured ratios
#: blow up non-Gaussianly and carry no colour information. Fainter generated
#: magnitudes terminal-pool at the deepest rows above this floor.
COLOR_SFR_VIS_SNR_FLOOR = 5.0
#: log10 sSFR below which a galaxy counts as quenched (project convention).
COLOR_SFR_ACTIVITY_THRESHOLD_LOGSSFR = -11.0
#: log10 sSFR at or above which the PHZ physical fit is pathological.
COLOR_SFR_PATHOLOGICAL_LOGSSFR = -8.2

COLOR_SFR_FEATURE_NAMES = ("vis_2fwhm_mag", "log10_re_arcsec", "re_resolved")
#: NISP bands whose 2FWHM aperture flux ratio to VIS is modelled.
COLOR_SFR_RATIO_BANDS = ("Y_E", "J_E", "H_E")
SFR_CLASS_QUENCHED = 0
SFR_CLASS_STAR_FORMING = 1
SFR_CLASS_UNKNOWN = 255
_SFR_CLASS_NAMES = {
    SFR_CLASS_QUENCHED: "quenched",
    SFR_CLASS_STAR_FORMING: "star_forming",
    SFR_CLASS_UNKNOWN: "unknown",
}
_SFR_CLASS_CODES = {
    "quenched": SFR_CLASS_QUENCHED,
    "star_forming": SFR_CLASS_STAR_FORMING,
}


def weighted_mid_quantiles(
    values: np.ndarray, weights: np.ndarray,
) -> np.ndarray:
    """Weighted, tie-aware empirical CDF midpoints in ``(0, 1)``.

    Equal values share one midpoint, so a point mass at a fit-grid floor
    (censored low SFRs) automatically becomes one shared bottom rank.
    """
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if (
        v.ndim != 1 or not v.size or w.shape != v.shape
        or not np.isfinite(v).all() or not np.isfinite(w).all()
        or np.any(w <= 0.0)
    ):
        raise ValueError(
            "weighted quantiles require finite values and positive weights"
        )
    unique, inverse = np.unique(v, return_inverse=True)
    mass = np.bincount(inverse, weights=w, minlength=unique.size)
    cumulative = np.cumsum(mass)
    mid = (cumulative - 0.5 * mass) / cumulative[-1]
    return mid[inverse]


def weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    quantiles: np.ndarray | tuple[float, ...] | float,
) -> np.ndarray:
    """Weighted quantiles at the cumulative-weight midpoints of each value."""
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    order = np.argsort(v, kind="mergesort")
    ordered_values = v[order]
    ordered_weights = w[order]
    cumulative = np.cumsum(ordered_weights)
    grid = (cumulative - 0.5 * ordered_weights) / cumulative[-1]
    return np.interp(
        np.asarray(quantiles, dtype=np.float64), grid, ordered_values,
    )


def ratios_to_colors(ratios: np.ndarray) -> tuple[float, float, float]:
    """AB colours (VIS−Y, Y−J, J−H) from positive NISP/VIS flux ratios."""
    r = np.asarray(ratios, dtype=np.float64)
    if r.shape != (3,) or not np.all(np.isfinite(r)) or np.any(r <= 0.0):
        raise ValueError("colour conversion requires three positive ratios")
    vis_minus_y = 2.5 * math.log10(r[0])
    y_minus_j = 2.5 * math.log10(r[1] / r[0])
    j_minus_h = 2.5 * math.log10(r[2] / r[1])
    return vis_minus_y, y_minus_j, j_minus_h


@dataclass(frozen=True)
class ColorSFRDraw:
    """One resampled catalogue neighbour with deconvolved colours."""

    ratio_y: float
    ratio_j: float
    ratio_h: float
    vis_minus_y: float
    y_minus_j: float
    j_minus_h: float
    log_sfr: float
    sfr_valid: bool
    sfr_borrowed: bool
    sfr_rank: float
    sfr_class: str
    neighborhood_rows: int
    pooling: str

    @property
    def ratios(self) -> tuple[float, float, float]:
        return (self.ratio_y, self.ratio_j, self.ratio_h)


def _validated_tree(
    tree: dict[str, Any], index: int, row_count: int,
) -> dict[str, np.ndarray]:
    try:
        node_count = int(tree["node_count"])
        feature = decode_array(
            tree["feature_zlib_base64"], (node_count,), "<i4",
            name=f"tree {index} feature",
        )
        threshold = decode_array(
            tree["threshold_zlib_base64"], (node_count,), "<f8",
            name=f"tree {index} threshold",
        )
        children_left = decode_array(
            tree["children_left_zlib_base64"], (node_count,), "<i4",
            name=f"tree {index} children_left",
        )
        children_right = decode_array(
            tree["children_right_zlib_base64"], (node_count,), "<i4",
            name=f"tree {index} children_right",
        )
        row_order = decode_array(
            tree["row_order_zlib_base64"], (row_count,), "<i4",
            name=f"tree {index} row_order",
        )
        node_row_start = decode_array(
            tree["node_row_start_zlib_base64"], (node_count + 1,), "<i4",
            name=f"tree {index} node_row_start",
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"colour+SFR forest tree {index} is malformed"
        ) from exc
    leaves = children_left == -1
    if not (
        node_count >= 1
        and np.array_equal(leaves, children_right == -1)
        and np.all(
            (children_left[~leaves] > 0) & (children_left[~leaves] < node_count)
        )
        and np.all(
            (children_right[~leaves] > 0)
            & (children_right[~leaves] < node_count)
        )
        and np.all(
            (feature[~leaves] >= 0)
            & (feature[~leaves] < len(COLOR_SFR_FEATURE_NAMES))
        )
        and np.all(np.isfinite(threshold[~leaves]))
        and np.array_equal(
            np.sort(np.asarray(row_order)), np.arange(row_count)
        )
        and node_row_start[0] == 0
        and node_row_start[-1] == row_count
        and np.all(np.diff(node_row_start) >= 0)
    ):
        raise ValueError(f"colour+SFR forest tree {index} is invalid")
    return {
        "feature": feature,
        "threshold": threshold,
        "children_left": children_left,
        "children_right": children_right,
        "row_order": row_order,
        "node_row_start": node_row_start,
    }


class ConditionalColorSFRSampler:
    """Load-time-validated forest sampler over real catalogue rows."""

    def __init__(self, payload: dict[str, Any]):
        if int(payload.get("version") or 0) != COLOR_SFR_MODEL_VERSION:
            raise ValueError("colour+SFR model has an unsupported version")
        if payload.get("kind") != COLOR_SFR_MODEL_KIND:
            raise ValueError("colour+SFR model has the wrong kind")
        self.calibration_fingerprint = str(
            payload.get("calibration_fingerprint") or ""
        )
        if len(self.calibration_fingerprint) != 64:
            raise ValueError("colour+SFR model fingerprint is invalid")
        self.catalog_version = int(payload.get("catalog_version") or 0)
        self.selection = str(payload.get("selection") or "")
        if not self.selection.strip():
            raise ValueError("colour+SFR model has no selection description")
        self.ratio_floor = float(
            payload.get("ratio_floor", COLOR_SFR_RATIO_FLOOR)
        )
        if not np.isfinite(self.ratio_floor) or self.ratio_floor <= 0.0:
            raise ValueError("colour+SFR ratio floor must be positive")

        row_count = int(payload.get("row_count") or 0)
        rows = payload.get("rows") or {}
        try:
            self._ratio = decode_array(
                rows["ratio_zlib_base64"], (row_count, 3), "<f4",
                sha256=rows.get("ratio_sha256"), name="colour ratios",
            ).astype(np.float64)
            self._ratio_var = decode_array(
                rows["ratio_var_zlib_base64"], (row_count, 3), "<f4",
                sha256=rows.get("ratio_var_sha256"),
                name="colour ratio variances",
            ).astype(np.float64)
            self._log_sfr = decode_array(
                rows["log_sfr_zlib_base64"], (row_count,), "<f4",
                sha256=rows.get("log_sfr_sha256"), name="log SFR",
            ).astype(np.float64)
            self._weight = decode_array(
                rows["weight_zlib_base64"], (row_count,), "<f4",
                sha256=rows.get("weight_sha256"), name="row weights",
            ).astype(np.float64)
            self._sfr_rank = decode_array(
                rows["sfr_rank_zlib_base64"], (row_count,), "<f4",
                sha256=rows.get("sfr_rank_sha256"), name="SFR ranks",
            ).astype(np.float64)
            self._sfr_valid = decode_array(
                rows["sfr_valid_zlib_base64"], (row_count,), "|u1",
                sha256=rows.get("sfr_valid_sha256"), name="SFR validity",
            ).astype(bool)
            self._sfr_class = decode_array(
                rows["sfr_class_zlib_base64"], (row_count,), "|u1",
                sha256=rows.get("sfr_class_sha256"), name="SFR classes",
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("colour+SFR row table is malformed") from exc
        valid = self._sfr_valid
        if not (
            row_count >= 2
            and np.all(np.isfinite(self._ratio))
            and np.all(np.isfinite(self._ratio_var))
            and np.all(self._ratio_var >= 0.0)
            and np.all(np.isfinite(self._weight))
            and np.all(self._weight > 0.0)
            and np.all(np.isfinite(self._log_sfr[valid]))
            and np.all(np.isfinite(self._sfr_rank[valid]))
            and np.all(
                (self._sfr_rank[valid] > 0.0) & (self._sfr_rank[valid] < 1.0)
            )
            and np.all(np.isin(
                self._sfr_class[valid],
                (SFR_CLASS_QUENCHED, SFR_CLASS_STAR_FORMING),
            ))
            and np.all(self._sfr_class[~valid] == SFR_CLASS_UNKNOWN)
            and bool(np.any(valid))
        ):
            raise ValueError("colour+SFR row table is invalid")

        imputation = payload.get("re_imputation") or {}
        try:
            self._impute_pivot = float(imputation["pivot_mag"])
            self._impute_intercept = float(
                imputation["intercept_log10_arcsec"]
            )
            self._impute_slope = float(
                imputation["slope_log10_arcsec_per_mag"]
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "colour+SFR radius imputation is malformed"
            ) from exc
        if not (
            np.isfinite(self._impute_pivot)
            and np.isfinite(self._impute_intercept)
            and np.isfinite(self._impute_slope)
        ):
            raise ValueError("colour+SFR radius imputation is invalid")

        trees = payload.get("trees")
        if not isinstance(trees, (list, tuple)) or not trees:
            raise ValueError("colour+SFR model has no trees")
        self._trees = [
            _validated_tree(tree, index, row_count)
            for index, tree in enumerate(trees)
        ]
        for tree in self._trees:
            starts = tree["node_row_start"]
            cumulative = np.concatenate((
                [0.0], np.cumsum(self._weight[tree["row_order"]]),
            ))
            tree["leaf_weight"] = cumulative[starts[1:]] - cumulative[starts[:-1]]

        self._sfr_pool_rows = np.flatnonzero(valid)
        pool_weight = self._weight[self._sfr_pool_rows]
        self._sfr_pool_probability = pool_weight / np.sum(pool_weight)
        self._class_pools: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for code in (SFR_CLASS_QUENCHED, SFR_CLASS_STAR_FORMING):
            members = np.flatnonzero(valid & (self._sfr_class == code))
            if members.size:
                class_weight = self._weight[members]
                self._class_pools[code] = (
                    members, class_weight / np.sum(class_weight),
                )
        self.row_count = row_count
        self.tree_count = len(self._trees)
        self._payload = payload

    def to_payload(self) -> dict[str, Any]:
        return self._payload

    # ------------------------------------------------------------------ #
    def _imputed_log_radius(self, magnitude: float) -> float:
        return (
            self._impute_intercept
            + self._impute_slope * (magnitude - self._impute_pivot)
        )

    def _leaf_node(self, tree: dict[str, np.ndarray], x: np.ndarray) -> int:
        feature = tree["feature"]
        threshold = tree["threshold"]
        left = tree["children_left"]
        right = tree["children_right"]
        node = 0
        while left[node] != -1:
            node = (
                int(left[node])
                if x[feature[node]] <= threshold[node]
                else int(right[node])
            )
        return node

    def _gather(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Union of the query's leaf populations, QRF-weighted per tree."""
        rows: list[np.ndarray] = []
        weights: list[np.ndarray] = []
        for tree in self._trees:
            node = self._leaf_node(tree, x)
            start = int(tree["node_row_start"][node])
            stop = int(tree["node_row_start"][node + 1])
            leaf_weight = float(tree["leaf_weight"][node])
            if stop <= start or leaf_weight <= 0.0:
                continue
            members = tree["row_order"][start:stop]
            rows.append(members)
            weights.append(self._weight[members] / leaf_weight)
        if not rows:
            raise ValueError("colour+SFR forest produced an empty neighbourhood")
        return np.concatenate(rows), np.concatenate(weights)

    def _neighborhood_moments(
        self,
        rows: np.ndarray,
        weights: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Robust per-band (location, intrinsic variance) of a neighbourhood.

        Location = weighted median; observed spread = weighted 16–84
        half-width; noise = weighted mean reported variance over rows below
        25× the median variance (legitimate error heterogeneity belongs in
        the noise term, but the few-percent junk-σ tail — hundreds of times
        the typical error — must not zero the intrinsic variance or drag
        the location).
        """
        location = np.empty(3, dtype=np.float64)
        intrinsic = np.empty(3, dtype=np.float64)
        for band in range(3):
            low, mid, high = weighted_quantile(
                self._ratio[rows, band], weights, (0.16, 0.5, 0.84),
            )
            observed_variance = (0.5 * (high - low)) ** 2
            variance_column = self._ratio_var[rows, band]
            median_variance = float(weighted_quantile(
                variance_column, weights, 0.5,
            ))
            sane = variance_column <= 25.0 * max(median_variance, 1e-300)
            noise_variance = float(np.average(
                variance_column[sane], weights=weights[sane],
            ))
            location[band] = mid
            intrinsic[band] = max(observed_variance - noise_variance, 0.0)
        return location, intrinsic

    def _deconvolved_ratios(
        self,
        rows: np.ndarray,
        weights: np.ndarray,
        row: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Posterior draw of the chosen row's true ratios (analytic XD)."""
        location, intrinsic = self._neighborhood_moments(rows, weights)
        row_variance = self._ratio_var[row]
        gain = np.divide(
            intrinsic,
            intrinsic + row_variance,
            out=np.zeros(3, dtype=np.float64),
            where=intrinsic + row_variance > 0.0,
        )
        mean = location + gain * (self._ratio[row] - location)
        spread = np.sqrt(np.maximum((1.0 - gain) * intrinsic, 0.0))
        draws = mean + spread * rng.standard_normal(3)
        return np.maximum(draws, self.ratio_floor)

    def _draw(
        self,
        rows: np.ndarray,
        weights: np.ndarray,
        rng: np.random.Generator,
        *,
        pooling: str,
    ) -> ColorSFRDraw:
        probability = weights / np.sum(weights)
        pick = int(rng.choice(rows.size, p=probability))
        row = int(rows[pick])
        ratios = self._deconvolved_ratios(rows, weights, row, rng)
        borrowed = False
        if self._sfr_valid[row]:
            sfr_row = row
        else:
            borrowed = True
            valid = self._sfr_valid[rows]
            if np.any(valid):
                valid_probability = weights[valid] / np.sum(weights[valid])
                sfr_row = int(rows[valid][int(
                    rng.choice(int(np.sum(valid)), p=valid_probability)
                )])
            else:
                pooling = f"{pooling}+global_sfr_pool"
                sfr_row = int(self._sfr_pool_rows[int(rng.choice(
                    self._sfr_pool_rows.size, p=self._sfr_pool_probability,
                ))])
        vis_minus_y, y_minus_j, j_minus_h = ratios_to_colors(ratios)
        return ColorSFRDraw(
            ratio_y=float(ratios[0]),
            ratio_j=float(ratios[1]),
            ratio_h=float(ratios[2]),
            vis_minus_y=vis_minus_y,
            y_minus_j=y_minus_j,
            j_minus_h=j_minus_h,
            log_sfr=float(self._log_sfr[sfr_row]),
            sfr_valid=bool(self._sfr_valid[row]),
            sfr_borrowed=borrowed,
            sfr_rank=float(self._sfr_rank[sfr_row]),
            sfr_class=_SFR_CLASS_NAMES[int(self._sfr_class[sfr_row])],
            neighborhood_rows=int(rows.size),
            pooling=pooling,
        )

    # ------------------------------------------------------------------ #
    def sample(
        self,
        magnitude: float,
        re_arcsec: float,
        rng: np.random.Generator,
    ) -> ColorSFRDraw:
        """Draw (colours, SFR) at a sampled VIS 2FWHM magnitude and radius."""
        mag = float(magnitude)
        radius = float(re_arcsec)
        if not (math.isfinite(mag) and math.isfinite(radius) and radius > 0.0):
            raise ValueError(
                "colour+SFR sampling requires a finite magnitude and radius"
            )
        x = np.asarray([mag, math.log10(radius), 1.0], dtype=np.float64)
        rows, weights = self._gather(x)
        return self._draw(rows, weights, rng, pooling="forest")

    def sample_class_conditioned(
        self,
        magnitude: float,
        sfr_class: str,
        rng: np.random.Generator,
    ) -> ColorSFRDraw:
        """Draw (colours, SFR) restricted to one activity class (lens path)."""
        mag = float(magnitude)
        if not math.isfinite(mag):
            raise ValueError("colour+SFR class sampling requires a magnitude")
        code = _SFR_CLASS_CODES.get(str(sfr_class))
        if code is None:
            raise ValueError(f"unknown SFR class {sfr_class!r}")
        x = np.asarray(
            [mag, self._imputed_log_radius(mag), 0.0], dtype=np.float64,
        )
        rows, weights = self._gather(x)
        selected = self._sfr_valid[rows] & (self._sfr_class[rows] == code)
        if np.any(selected):
            return self._draw(
                rows[selected], weights[selected], rng,
                pooling="forest_class",
            )
        pool = self._class_pools.get(code)
        if pool is None:
            raise ValueError(
                f"colour+SFR model has no {sfr_class!r} population"
            )
        return self._draw(
            pool[0], pool[1] * 1.0, rng, pooling="global_class_pool",
        )

    def mean_colors(self, magnitude: np.ndarray) -> np.ndarray:
        """Deterministic median-colour trend for diagnostics."""
        values = np.atleast_1d(np.asarray(magnitude, dtype=np.float64))
        result = np.empty((values.size, 3), dtype=np.float64)
        for index, mag in enumerate(values):
            x = np.asarray(
                [mag, self._imputed_log_radius(float(mag)), 1.0],
                dtype=np.float64,
            )
            rows, weights = self._gather(x)
            location, _ = self._neighborhood_moments(rows, weights)
            result[index] = ratios_to_colors(
                np.maximum(location, self.ratio_floor)
            )
        return result
