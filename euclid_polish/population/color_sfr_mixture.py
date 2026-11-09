"""PROTOTYPE: joint Gaussian-mixture model of galaxy colours, SFR and mass.

An alternative to the colour+SFR forest (:mod:`conditional_color_sfr`). In
each VIS-magnitude bin, the joint distribution of

    x = (Y/VIS, J/VIS, H/VIS flux ratios, log SFR, log M*, log R_e, VIS mag)

is a K-component Gaussian mixture with FULL covariances, fitted by extreme
deconvolution (XD; Bovy, Hogg & Roweis 2011): every Q1 row enters with its
own noise covariance and its own set of observed dimensions, so

- measurement noise is removed jointly, correlations included: the three
  ratios share the VIS flux error, which XD models as a covariance;
- rows with no usable PHZ physical fit (two thirds of the weight) still
  constrain the colours, size and magnitude, and rows with an unusable
  Sérsic radius still constrain everything else (missing dimensions are
  marginalized, not imputed or borrowed);
- generation conditions the mixture on the drawn magnitude and size with the
  exact Gaussian conditional, so colour, SFR and mass come out jointly, with
  their correlations and their dependence on brightness and size.

NumPy plus scikit-learn k-means seeding. Not wired into generation: this
module exists to be compared against the forest.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans

#: Dimension order of the modelled vector.
MIXTURE_DIMENSIONS = ("ratio_y", "ratio_j", "ratio_h", "log_sfr", "log_mass", "log_re", "vis_mag")
RATIO_DIMS = (0, 1, 2)
SFR_DIM, MASS_DIM, RE_DIM, MAG_DIM = 3, 4, 5, 6
#: Log SFR is censored from below here before fitting (PHZ's unphysical tail).
MIXTURE_SFR_CENSOR = -5.0
#: Ratio draws are floored here (same floor as the forest).
MIXTURE_RATIO_FLOOR = 1e-4
#: log sSFR below this counts as quenched (project convention).
MIXTURE_QUENCHED_LOGSSFR = -11.0


def _logsumexp(a: np.ndarray, axis: int) -> np.ndarray:
    peak = np.max(a, axis=axis, keepdims=True)
    peak = np.where(np.isfinite(peak), peak, 0.0)
    return np.squeeze(peak, axis=axis) + np.log(np.sum(np.exp(a - peak), axis=axis))


def xd_fit(
    values: np.ndarray,
    noise: np.ndarray,
    observed: np.ndarray,
    weight: np.ndarray,
    n_components: int,
    *,
    rng: np.random.Generator,
    max_iter: int = 300,
    tol: float = 1e-6,
    regularization: float = 1e-6,
) -> dict:
    """Weighted extreme deconvolution with missing dimensions.

    ``values`` (n, d) with anything in unobserved entries; ``noise`` (n, d, d)
    per-row noise covariances (only the observed block is read);
    ``observed`` (n, d) bool; ``weight`` (n,) data weights. Returns
    ``{"alpha", "mean", "cov", "loglike", "iterations"}``.
    """
    n, d = values.shape
    values = np.where(observed, values, 0.0)
    # Group rows by missingness pattern so every group is a dense batch.
    patterns, inverse = np.unique(observed, axis=0, return_inverse=True)
    groups = [(patterns[g], np.flatnonzero(inverse.ravel() == g)) for g in range(len(patterns))]

    # Seed: k-means on fully observed rows (or all rows with mean-filled gaps).
    complete = np.all(observed, axis=1)
    seed_rows = complete if complete.sum() >= 10 * n_components else np.ones(n, bool)
    filled = values.copy()
    column_mean = np.array([
        np.average(values[observed[:, j], j], weights=weight[observed[:, j]]) for j in range(d)
    ])
    filled[~observed] = np.broadcast_to(column_mean, (n, d))[~observed]
    scale = np.std(filled[seed_rows], axis=0) + 1e-12
    km = KMeans(n_components, n_init=4, random_state=int(rng.integers(1 << 31))).fit(
        filled[seed_rows] / scale, sample_weight=weight[seed_rows],
    )
    mean = km.cluster_centers_ * scale
    cov = np.array([np.cov(filled[seed_rows].T, aweights=weight[seed_rows]) for _ in range(n_components)])
    alpha = np.full(n_components, 1.0 / n_components)

    previous = -np.inf
    eye = np.eye(d)
    total_weight = float(np.sum(weight))
    iterations = 0
    for _ in range(max_iter):
        iterations += 1
        log_resp = np.empty((n, n_components))
        b_all = np.empty((n_components, n, d))
        B_all = np.empty((n_components, n, d, d))
        for mask, rows in groups:
            o = np.flatnonzero(mask)
            y = values[np.ix_(rows, o)]
            S = noise[np.ix_(rows, o, o)]
            for k in range(n_components):
                Voo = cov[k][np.ix_(o, o)]
                Vxo = cov[k][:, o]
                T = Voo[None] + S
                resid = y - mean[k][o]
                L = np.linalg.cholesky(T)
                z = np.linalg.solve(L, resid[..., None])[..., 0]
                logdet = 2.0 * np.sum(np.log(np.diagonal(L, axis1=1, axis2=2)), axis=1)
                quad = np.sum(z * z, axis=1) + logdet + o.size * math.log(2 * math.pi)
                log_resp[rows, k] = math.log(alpha[k]) - 0.5 * quad
                Tinv_r = np.linalg.solve(T, resid[..., None])[..., 0]
                b_all[k, rows] = mean[k] + Tinv_r @ Vxo.T
                # B = V − V Rᵀ T⁻¹ R V
                VxoT = np.broadcast_to(Vxo.T, (rows.size, o.size, d))
                B_all[k, rows] = cov[k] - np.einsum("do,noe->nde", Vxo, np.linalg.solve(T, VxoT))
        norm = _logsumexp(log_resp, axis=1)
        loglike = float(np.sum(weight * norm) / total_weight)
        q = np.exp(log_resp - norm[:, None]) * weight[:, None]
        qk = np.sum(q, axis=0)
        alpha = np.maximum(qk / total_weight, 1e-12)
        for k in range(n_components):
            wk = q[:, k] / max(qk[k], 1e-300)
            mean[k] = wk @ b_all[k]
            diff = b_all[k] - mean[k]
            cov[k] = np.einsum("n,nd,ne->de", wk, diff, diff) + np.einsum("n,nde->de", wk, B_all[k])
            cov[k] = 0.5 * (cov[k] + cov[k].T) + regularization * eye
        if loglike - previous < tol * max(1.0, abs(loglike)):
            previous = loglike
            break
        previous = loglike
    return {"alpha": alpha, "mean": mean, "cov": cov, "loglike": previous, "iterations": iterations}


@dataclass
class MixtureBin:
    lo: float
    hi: float
    alpha: np.ndarray       # (K,)
    mean: np.ndarray        # (K, d)
    cov: np.ndarray         # (K, d, d)
    rows: int
    sfr_rows: int
    loglike: float


def mixture_rows(rows, flux_vis_snr: np.ndarray, log_mass: np.ndarray,
                 sfr_sigma: np.ndarray, mass_sigma: np.ndarray):
    """Values, noise covariances and observed masks for :func:`xd_fit`.

    ``rows`` is a :class:`ColorSFRRows`; the extra arrays align with it.
    Ratio noise: per-band variances plus the shared-VIS covariance
    r_a·r_b / SNR_VIS². PHZ SFR/mass noise from their 16–84 half-widths.
    """
    n = rows.weight.size
    d = len(MIXTURE_DIMENSIONS)
    values = np.zeros((n, d))
    observed = np.zeros((n, d), bool)
    noise = np.zeros((n, d, d))
    values[:, :3] = rows.ratio
    observed[:, :3] = True
    shared = rows.ratio[:, :, None] * rows.ratio[:, None, :] / flux_vis_snr[:, None, None] ** 2
    noise[:, :3, :3] = shared
    # The diagonal from the catalogue variance already includes the VIS term.
    idx = np.arange(3)
    noise[:, idx, idx] = rows.ratio_var
    valid = rows.sfr_valid & np.isfinite(log_mass)
    values[valid, SFR_DIM] = np.maximum(rows.log_sfr[valid], MIXTURE_SFR_CENSOR)
    values[valid, MASS_DIM] = log_mass[valid]
    observed[valid, SFR_DIM] = observed[valid, MASS_DIM] = True
    noise[valid, SFR_DIM, SFR_DIM] = np.clip(sfr_sigma[valid], 0.05, 3.0) ** 2
    noise[valid, MASS_DIM, MASS_DIM] = np.clip(mass_sigma[valid], 0.03, 2.0) ** 2
    values[rows.resolved, RE_DIM] = rows.log_radius[rows.resolved]
    observed[rows.resolved, RE_DIM] = True
    values[:, MAG_DIM] = rows.magnitude
    observed[:, MAG_DIM] = True
    # Small jitter on noise-free dimensions keeps T well conditioned.
    noise[:, RE_DIM, RE_DIM] += 1e-4
    noise[:, MAG_DIM, MAG_DIM] += 1e-4
    return values, noise, observed


def magnitude_bins(magnitude: np.ndarray, sfr_valid: np.ndarray, *,
                   width: float = 0.5, min_rows: int = 1500, min_sfr_rows: int = 300) -> np.ndarray:
    """0.5-mag edges, merged from the bright end (and the faint end) until
    every bin holds ``min_rows`` rows and ``min_sfr_rows`` SFR-valid rows."""
    lo = math.floor(float(np.min(magnitude)) / width) * width
    hi = math.ceil(float(np.max(magnitude)) / width) * width
    edges = list(np.arange(lo, hi + width / 2, width))

    def counts(a, b):
        s = (magnitude >= a) & (magnitude < b)
        return int(s.sum()), int((s & sfr_valid).sum())

    merged = [edges[0]]
    i = 1
    while i < len(edges):
        n_all, n_sfr = counts(merged[-1], edges[i])
        if (n_all >= min_rows and n_sfr >= min_sfr_rows) or i == len(edges) - 1:
            merged.append(edges[i])
        i += 1
    # The faint remainder may be thin: merge it into its neighbour.
    while len(merged) > 2:
        n_all, n_sfr = counts(merged[-2], merged[-1] + 1e-9)
        if n_all >= min_rows and n_sfr >= min_sfr_rows:
            break
        del merged[-2]
    return np.asarray(merged)


def fit_mixture_model(values, noise, observed, weight, magnitude, sfr_valid, *,
                      n_components: int = 3, seed: int = 20261108, **bin_kwargs) -> list[MixtureBin]:
    edges = magnitude_bins(magnitude, sfr_valid, **bin_kwargs)
    rng = np.random.default_rng(seed)
    bins = []
    for a, b in zip(edges[:-1], edges[1:], strict=True):
        last = b == edges[-1]
        s = (magnitude >= a) & ((magnitude <= b) if last else (magnitude < b))
        fit = xd_fit(values[s], noise[s], observed[s], weight[s], n_components, rng=rng)
        bins.append(MixtureBin(float(a), float(b), fit["alpha"], fit["mean"], fit["cov"],
                               int(s.sum()), int((s & sfr_valid).sum()), fit["loglike"]))
    return bins


class ColorSFRMixtureSampler:
    """Draw (ratios, log SFR, log M*) given VIS magnitude and R_e."""

    def __init__(self, bins: list[MixtureBin], sfr_reference: np.ndarray, sfr_reference_weight: np.ndarray):
        self.bins = bins
        self._lo = np.array([b.lo for b in bins])
        order = np.argsort(sfr_reference)
        self._sfr_sorted = np.asarray(sfr_reference)[order]
        w = np.asarray(sfr_reference_weight)[order]
        self._sfr_cdf = (np.cumsum(w) - 0.5 * w) / np.sum(w)

    def _bin(self, magnitude: float) -> MixtureBin:
        index = int(np.clip(np.searchsorted(self._lo, magnitude, side="right") - 1, 0, len(self.bins) - 1))
        return self.bins[index]

    def conditional(self, magnitude: float, log_re: float | None):
        """Mixture over the free dimensions given the conditioning ones.

        Magnitude is clamped to the bin's range: the within-bin trend is
        linear in magnitude and must not be extrapolated past the data.
        """
        b = self._bin(magnitude)
        mag = float(np.clip(magnitude, b.lo, b.hi))
        cond = [MAG_DIM] if log_re is None else [RE_DIM, MAG_DIM]
        x_c = np.array([mag] if log_re is None else [log_re, mag])
        free = [j for j in range(len(MIXTURE_DIMENSIONS)) if j not in (RE_DIM, MAG_DIM)]
        logw, means, covs = [], [], []
        for k in range(len(b.alpha)):
            m, V = b.mean[k], b.cov[k]
            Vcc = V[np.ix_(cond, cond)]
            Vfc = V[np.ix_(free, cond)]
            r = x_c - m[cond]
            sol = np.linalg.solve(Vcc, r)
            _, logdet = np.linalg.slogdet(Vcc)
            logw.append(math.log(b.alpha[k]) - 0.5 * (r @ sol + logdet))
            means.append(m[free] + Vfc @ sol)
            covs.append(V[np.ix_(free, free)] - Vfc @ np.linalg.solve(Vcc, Vfc.T))
        logw = np.array(logw)
        w = np.exp(logw - logw.max())
        return w / w.sum(), np.array(means), np.array(covs)

    def sample(self, magnitude: float, re_arcsec: float | None, rng: np.random.Generator, size: int = 1):
        """(size, 5) array: ratio_y, ratio_j, ratio_h, log_sfr, log_mass."""
        log_re = None if re_arcsec is None else math.log10(re_arcsec)
        w, means, covs = self.conditional(magnitude, log_re)
        comp = rng.choice(w.size, size=size, p=w)
        out = np.empty((size, means.shape[1]))
        for k in np.unique(comp):
            sel = comp == k
            out[sel] = rng.multivariate_normal(means[k], 0.5 * (covs[k] + covs[k].T), size=int(sel.sum()),
                                               method="eigh")
        out[:, :3] = np.maximum(out[:, :3], MIXTURE_RATIO_FLOOR)
        return out

    def sfr_rank(self, log_sfr: np.ndarray) -> np.ndarray:
        return np.interp(log_sfr, self._sfr_sorted, self._sfr_cdf)

    @staticmethod
    def quenched(draws: np.ndarray) -> np.ndarray:
        return (draws[:, 3] - draws[:, 4]) < MIXTURE_QUENCHED_LOGSSFR
