from __future__ import annotations

import numpy as np

from euclid_polish.population.color_sfr_mixture import xd_fit


def _truth():
    mean = np.array([[0.0, 0.0, 0.0], [3.0, 2.0, -1.0]])
    cov = np.array([
        [[1.0, 0.8, 0.3], [0.8, 1.0, 0.5], [0.3, 0.5, 0.6]],
        [[0.5, -0.3, 0.0], [-0.3, 0.8, 0.2], [0.0, 0.2, 0.4]],
    ])
    return np.array([0.6, 0.4]), mean, cov


def test_xd_recovers_correlated_mixture_through_noise_and_missing_dims():
    rng = np.random.default_rng(0)
    alpha, mean, cov = _truth()
    n = 6000
    comp = rng.choice(2, n, p=alpha)
    truth = np.stack([rng.multivariate_normal(mean[k], cov[k]) for k in comp])
    # Heteroscedastic, correlated noise comparable to the intrinsic spread.
    sigma = rng.uniform(0.2, 1.0, n)
    noise = np.einsum("n,de->nde", sigma**2, np.array([[1, 0.5, 0], [0.5, 1, 0], [0, 0, 1.0]]))
    observed = np.ones((n, 3), bool)
    observed[rng.random(n) < 0.5, 2] = False  # half the rows lack dimension 2
    values = truth + np.stack([rng.multivariate_normal(np.zeros(3), s) for s in noise])
    fit = xd_fit(values, noise, observed, np.ones(n), 2, rng=np.random.default_rng(1))
    order = np.argsort(fit["mean"][:, 0])
    np.testing.assert_allclose(fit["alpha"][order], alpha, atol=0.03)
    np.testing.assert_allclose(fit["mean"][order], mean, atol=0.12)
    np.testing.assert_allclose(fit["cov"][order], cov, atol=0.15)


def test_xd_without_noise_matches_plain_gaussian_moments():
    rng = np.random.default_rng(2)
    x = rng.multivariate_normal([1.0, -2.0], [[2.0, 1.2], [1.2, 1.5]], size=4000)
    fit = xd_fit(x, np.zeros((4000, 2, 2)) + 1e-9 * np.eye(2), np.ones((4000, 2), bool),
                 np.ones(4000), 1, rng=rng)
    np.testing.assert_allclose(fit["mean"][0], x.mean(0), atol=1e-6)
    np.testing.assert_allclose(fit["cov"][0], np.cov(x.T, bias=True), atol=1e-4)
