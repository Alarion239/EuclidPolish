"""Noise propagation with composed linear weights and explicit sample ancestry.

The sparse operator is the covariance representation: C = L diag(v) L.T.
We keep L and v instead of materializing a quadratic-size dense covariance.
Native pixel independence is a declared model assumption, not a measurement.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse


def robust_sigma(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    return float(1.482602218505602 * np.median(np.abs(values - np.median(values)))) if values.size else np.nan


def bilinear_operator(input_shape, x, y, valid=None, scale=1.0):
    """Map input pixels to coordinates x,y. Reject any incomplete footprint.

    `scale` includes the photometric and pixel-area factors, when applicable.
    Exact integer coordinates do not depend on zero-weight neighbors.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.shape != y.shape or not np.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid coordinates or multiplicative scale")
    h, w = input_shape
    finite = np.isfinite(x) & np.isfinite(y)
    xx, yy = np.where(finite, x, -10).ravel(), np.where(finite, y, -10).ravel()
    ix, iy = np.floor(xx).astype(int), np.floor(yy).astype(int)
    fx, fy = xx - ix, yy - iy
    rows, cols, weights = [], [], []
    covered = finite.ravel().copy()
    input_valid = np.ones(h * w, bool) if valid is None else np.asarray(valid, bool).ravel()
    if input_valid.size != h * w:
        raise ValueError("Validity shape does not match input")
    for dx, dy, weight in (
        (0, 0, (1 - fx) * (1 - fy)),
        (1, 0, fx * (1 - fy)),
        (0, 1, (1 - fx) * fy),
        (1, 1, fx * fy),
    ):
        cx, cy = ix + dx, iy + dy
        inside = (cx >= 0) & (cx < w) & (cy >= 0) & (cy < h)
        col = np.clip(cy, 0, h - 1) * w + np.clip(cx, 0, w - 1)
        nonzero = weight > 1e-12
        covered &= ~nonzero | (inside & input_valid[col])
        take = nonzero & inside
        rows.extend(np.flatnonzero(take))
        cols.extend(col[take])
        weights.extend(weight[take] * scale)
    matrix = sparse.csr_matrix((weights, (rows, cols)), shape=(xx.size, h * w))
    matrix = sparse.diags(covered.astype(float)) @ matrix
    return matrix.tocsr(), covered.reshape(x.shape)


def convolution_operator(shape, kernel, valid=None):
    """Flux-preserving finite PSF convolution, with strict footprint exclusion."""
    kernel = np.asarray(kernel, float)
    if (
        kernel.ndim != 2
        or any(n % 2 == 0 for n in kernel.shape)
        or not np.all(np.isfinite(kernel))
        or kernel.sum() <= 0
    ):
        raise ValueError("PSF must be finite, odd-sized, and have positive total flux")
    kernel = kernel / kernel.sum()
    h, w = shape
    yy, xx = np.indices(shape)
    rows, cols, weights = [], [], []
    covered = np.ones(shape, bool)
    input_valid = np.ones(shape, bool) if valid is None else np.asarray(valid, bool)
    for ky, kx in zip(*np.nonzero(kernel), strict=True):
        sy, sx = yy + kernel.shape[0] // 2 - ky, xx + kernel.shape[1] // 2 - kx
        inside = (sy >= 0) & (sy < h) & (sx >= 0) & (sx < w)
        sy, sx = np.clip(sy, 0, h - 1), np.clip(sx, 0, w - 1)
        covered &= inside & input_valid[sy, sx]
        take = np.flatnonzero(inside)
        rows.append(take)
        cols.append((sy * w + sx).ravel()[take])
        weights.append(np.full(take.size, kernel[ky, kx]))
    matrix = sparse.csr_matrix(
        (np.concatenate(weights), (np.concatenate(rows), np.concatenate(cols))), shape=(h * w, h * w)
    )
    return (sparse.diags(covered.ravel().astype(float)) @ matrix).tocsr(), covered


@dataclass
class PropagatedNoise:
    operator: sparse.csr_matrix
    native_variance: np.ndarray
    shape: tuple[int, int]

    def __post_init__(self):
        self.native_variance = np.asarray(self.native_variance, float).ravel()
        if (
            self.operator.shape != (int(np.prod(self.shape)), self.native_variance.size)
            or not np.all(np.isfinite(self.native_variance))
            or np.any(self.native_variance < 0)
        ):
            raise ValueError("Invalid variance or transformation dimensions")

    def diagonal(self):
        return np.asarray(self.operator.power(2) @ self.native_variance).reshape(self.shape)

    def aperture_variance(self, aperture):
        weights = np.asarray(self.operator.T @ np.asarray(aperture, float).ravel()).ravel()
        return float(np.dot(weights * weights, self.native_variance))

    def covariance(self, first, second):
        return float(
            (self.operator.getrow(first).multiply(self.operator.getrow(second)) @ self.native_variance)[0]
        )

    def draw(self, rng):
        return np.asarray(
            self.operator @ (rng.normal(size=self.native_variance.size) * np.sqrt(self.native_variance))
        ).reshape(self.shape)


def independent_components(ancestry):
    """Connected components of pointings sharing ANY native input exposure.

    Missing ancestry belongs to a single unknown component; it is never counted
    as a collection of independent observations.
    """
    sets = [set(a) if a else {"__unknown_ancestry__"} for a in ancestry]
    groups = []
    for index, inputs in enumerate(sets):
        overlapping = [g for g in groups if any(inputs & sets[j] for j in g)]
        merged = {index}
        for group in overlapping:
            merged.update(group)
            groups.remove(group)
        groups.append(merged)
    return [sorted(group) for group in groups]


def amplitude_decision(interval, target=0.05):
    if interval is None or not np.all(np.isfinite(interval)):
        return "insufficient precision"
    low, high = interval
    if low >= 1 - target and high <= 1 + target:
        return "within 5% target"
    if high < 1 - target or low > 1 + target:
        return "disagreement"
    return "insufficient precision"


def clustered_interval(pointing_values, ancestry, draws=2000, seed=39017):
    """Equal-pointing mean; resample whole connected exposure components."""
    values = np.asarray(pointing_values, float)
    if len(values) != len(ancestry) or not np.all(np.isfinite(values)):
        raise ValueError("Finite pointing values and matching ancestry required")
    groups = independent_components(ancestry)
    result = {
        "mean": float(np.mean(values)) if len(values) else None,
        "pointings": len(values),
        "independent_components": len(groups),
        "ci95": None,
    }
    if len(groups) < 3 or any(not a for a in ancestry):
        result["precision_status"] = "insufficient independent exposure ancestry or components"
        return result
    rng = np.random.default_rng(seed)
    means = [
        np.mean(values[np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])])
        for _ in range(draws)
    ]
    result["ci95"] = np.quantile(means, [0.025, 0.975]).tolist()
    result["precision_status"] = "cluster bootstrap; small component counts limit precision"
    return result


def difference_measurement(a, b, noise_a, noise_b, valid, ancestry_a, ancestry_b, block=32):
    """Unclipped differences, including normal source pixels and their Poisson noise."""
    if not ancestry_a or not ancestry_b or set(ancestry_a) & set(ancestry_b):
        raise ValueError("Differences require documented disjoint input exposures")
    difference = np.asarray(a, float) - np.asarray(b, float)
    variance = noise_a.diagonal() + noise_b.diagonal()
    valid = np.asarray(valid, bool) & np.isfinite(difference) & np.isfinite(variance) & (variance > 0)
    z = np.full(difference.shape, np.nan)
    z[valid] = difference[valid] / np.sqrt(variance[valid])
    values = z[valid]
    if values.size < 64:
        return {
            "status": "insufficient coverage",
            "n_valid": int(values.size),
            "valid_fraction": float(valid.mean()),
            "std_z": None,
            "amplitude_target": "insufficient precision",
            "reason": "Fewer than 64 valid difference pixels",
        }, {"difference": difference, "predicted_variance": variance, "z": z, "valid": valid}
    stats = {
        "n_valid": int(values.size),
        "valid_fraction": float(valid.mean()),
        "mean_z": float(np.mean(values)),
        "std_z": float(np.std(values, ddof=1)),
        "rms_z": float(np.sqrt(np.mean(values**2))),
        "mad_z": robust_sigma(values),
        "tail_abs_z_gt_3": float(np.mean(np.abs(values) > 3)),
        "tail_abs_z_gt_5": float(np.mean(np.abs(values) > 5)),
        "difference_std": float(np.std(difference[valid], ddof=1)),
        "predicted_difference_rms": float(np.sqrt(np.mean(variance[valid]))),
        "quantiles_z": np.quantile(values, [0.001, 0.01, 0.16, 0.5, 0.84, 0.99, 0.999]).tolist(),
    }
    # Spatial block bootstrap is a patch-level precision diagnostic, never the
    # independent-pointing sample count. Blocks must exceed the correlation scale.
    blocks = []
    for y in range(0, z.shape[0], block):
        for x in range(0, z.shape[1], block):
            zz = z[y : y + block, x : x + block]
            vv = zz[np.isfinite(zz)]
            if vv.size >= block * block * 0.8:
                blocks.append((float(np.sum(vv)), float(np.sum(vv**2)), vv.size))
    stats["block_size_pixels"] = block
    stats["block_count"] = len(blocks)
    stats["std_z_ci95"] = None
    if len(blocks) >= 8:
        rng = np.random.default_rng(90210)
        bs = np.asarray(blocks)
        draws = bs[rng.integers(0, len(bs), (1000, len(bs)))].sum(axis=1)
        std = np.sqrt(np.maximum(0, (draws[:, 1] - draws[:, 0] ** 2 / draws[:, 2]) / (draws[:, 2] - 1)))
        stats["std_z_ci95"] = np.quantile(std, [0.025, 0.975]).tolist()
    stats["amplitude_target"] = amplitude_decision(stats["std_z_ci95"])
    stats["ci_caveat"] = (
        "Spatial block bootstrap; conditional on blocks exceeding measured correlation length"
    )
    correlations = []
    for dy, dx in ((0, 1), (1, 0), (1, 1), (0, 2), (0, 4), (0, 8), (0, 16)):
        z1, z2 = z[: z.shape[0] - dy or None, : z.shape[1] - dx or None], z[dy:, dx:]
        keep = np.isfinite(z1) & np.isfinite(z2)
        correlations.append(
            {
                "lag_yx": [dy, dx],
                "pairs": int(keep.sum()),
                "covariance_z": float(np.mean((z1[keep] - values.mean()) * (z2[keep] - values.mean())))
                if keep.any()
                else None,
            }
        )
    stats["spatial_covariance"] = correlations
    apertures = []
    for size in (1, 3, 5, 9, 17):
        sums, predictions = [], []
        aperture = np.zeros(z.shape)
        for y in range(0, z.shape[0] - size + 1, max(size, 8)):
            for x in range(0, z.shape[1] - size + 1, max(size, 8)):
                if np.all(valid[y : y + size, x : x + size]):
                    aperture[y : y + size, x : x + size] = 1
                    sums.append(float(np.sum(difference[y : y + size, x : x + size])))
                    predictions.append(
                        noise_a.aperture_variance(aperture) + noise_b.aperture_variance(aperture)
                    )
                    aperture[y : y + size, x : x + size] = 0
        apertures.append(
            {
                "square_side_pixels": size,
                "count": len(sums),
                "measured_sum_variance": float(np.var(sums, ddof=1)) if len(sums) > 1 else None,
                "mean_predicted_sum_variance": float(np.mean(predictions)) if predictions else None,
            }
        )
    stats["apertures"] = apertures
    brightness = (np.asarray(a) + np.asarray(b)) / 2
    edges = np.quantile(brightness[valid], [0, 0.5, 0.9, 0.99, 1])
    strata = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        keep = valid & (brightness >= lo) & (brightness <= hi)
        v = z[keep]
        strata.append(
            {
                "brightness_range": [float(lo), float(hi)],
                "count": int(v.size),
                "std_z": float(np.std(v, ddof=1)) if v.size > 1 else None,
                "mean_z": float(v.mean()) if v.size else None,
            }
        )
    stats["brightness_strata"] = strata
    gy, gx = np.gradient(brightness)
    structure = {}
    for name, template in (("brightness", brightness), ("gradient_x", gx), ("gradient_y", gy)):
        keep = valid & np.isfinite(template)
        if keep.sum() > 64 and np.std(template[keep]) > 0 and np.std(z[keep]) > 0:
            structure[name + "_correlation_with_z"] = float(np.corrcoef(template[keep], z[keep])[0, 1])
        else:
            structure[name + "_correlation_with_z"] = None
    structure["interpretation"] = (
        "Diagnostic associations only; gradients/brightness are noisy. "
        "No fit is subtracted and no source or registration residual is "
        "absorbed into the variance prediction."
    )
    stats["residual_structure"] = structure
    stats["source_structure_caveat"] = (
        "Brightness strata use noisy pair mean; matching residuals are retained"
    )
    return stats, {"difference": difference, "predicted_variance": variance, "z": z, "valid": valid}


def median_prediction(noises, signals, validity, draws=512, seed=31071):
    """Monte Carlo of the actual median, after each exposure's complete operator.

    Inputs must already encode actual times, zero points, bilinear pixel-area
    resampling and native variances. No Gaussian median asymptotic shortcut.
    """
    if len(noises) != len(signals) or len(noises) != len(validity) or len(noises) < 2 or draws < 32:
        raise ValueError("At least two exposures and 32 Monte Carlo draws required")
    shape = noises[0].shape
    valid = np.logical_and.reduce(validity)
    if not valid.any() or any(n.shape != shape for n in noises):
        raise ValueError("No common coverage or inconsistent output shapes")
    rng = np.random.default_rng(seed)
    mean, m2 = np.zeros(shape), np.zeros(shape)
    aperture_positions = []
    for side in (1, 3, 5, 9, 17):
        positions = [
            (y, x)
            for y in range(0, shape[0] - side + 1, max(8, side))
            for x in range(0, shape[1] - side + 1, max(8, side))
            if np.all(valid[y : y + side, x : x + side])
        ][:64]
        aperture_positions.append((side, positions))
    aperture_draws = [np.zeros((draws, len(positions))) for _, positions in aperture_positions]
    lags = tuple(
        (dy, dx) for dy, dx in ((0, 1), (1, 0), (0, 2), (0, 4), (0, 8)) if dy < shape[0] and dx < shape[1]
    )
    cross_sums = np.zeros(len(lags))
    lag_masks = [valid[: shape[0] - dy, : shape[1] - dx] & valid[dy:, dx:] for dy, dx in lags]
    for i in range(draws):
        stack = np.stack([s + n.draw(rng) for s, n in zip(signals, noises, strict=True)])
        sample = np.median(stack, axis=0)
        delta = sample - mean
        mean += delta / (i + 1)
        m2 += delta * (sample - mean)
        for j, (dy, dx) in enumerate(lags):
            product = delta[: shape[0] - dy, : shape[1] - dx] * (sample - mean)[dy:, dx:]
            cross_sums[j] += product[lag_masks[j]].sum()
        for j, (side, positions) in enumerate(aperture_positions):
            aperture_draws[j][i] = [sample[y : y + side, x : x + side].sum() for y, x in positions]
    variance = m2 / (draws - 1)
    variance[~valid] = np.nan
    return {
        "kind": "propagated prediction, not an exposure-difference measurement",
        "draws": draws,
        "seed": seed,
        "combination": "MEDIAN",
        "native_covariance": "diagonal model",
        "draw_distribution": "Independent Gaussian native pixels conditional on supplied total RMS",
        "spatial_covariance": [
            {
                "lag_yx": list(lag),
                "valid_pairs": int(mask.sum()),
                "mean_covariance": float(total / ((draws - 1) * mask.sum())) if mask.any() else None,
            }
            for lag, mask, total in zip(lags, lag_masks, cross_sums, strict=True)
        ],
        "aperture_sum_variance": [
            {
                "square_side_pixels": side,
                "apertures": len(positions),
                "mean_variance": float(np.var(samples, axis=0, ddof=1).mean()) if positions else None,
            }
            for (side, positions), samples in zip(aperture_positions, aperture_draws, strict=True)
        ],
        "coverage_policy": "common coverage of supplied contributing exposures",
        "approx_mc_amplitude_relative_se": float(1 / np.sqrt(2 * (draws - 1))),
    }, variance
