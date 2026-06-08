"""Realistic distribution of **apparent angular half-light radii** for galaxies
in a magnitude-limited sample — used to place injected TNG50 stamps at lifelike
sizes instead of a flat ×1/×2/×3/×4 downsample (which made every TNG galaxy a
giant blob).

Why a *physical forward model* rather than a hand-drawn curve
------------------------------------------------------------
The apparent size of a galaxy is its physical half-light radius divided by the
angular-diameter distance to its redshift, so the observed angular-size
distribution is the convolution

    p(θ) = ∫dz n(z) ∫dR P(R | z) δ(θ − R / d_A(z)).

We sample that convolution directly:

* **P(R)** — physical half-light radius is **log-normal** at fixed mass, the
  consensus of Shen et al. 2003 (SDSS, z≈0) and van der Wel et al. 2014
  (CANDELS), with scatter σ_lnR ≈ 0.3–0.5 dex; sizes shrink with redshift
  roughly as (1+z)^−0.75.
* **n(z)** — the standard magnitude-limited form n(z) ∝ z² exp(−(z/z₀)^β).
* **d_A(z)** — Planck15 angular-diameter distance.

The defaults are tuned so the result reproduces the **deep-field measurements**:
median apparent R_e ≈ 0.22″ (matching our COSMOS mag<25 selection and the
0.2–0.25″ medians reported for faint HST samples), with a **physically fat,
continuous tail** — only ~2% of galaxies exceed 1″, ~0.3% exceed 2″, ~0.1%
exceed 3″. So most injected galaxies are small (heavily downsampled) and big,
resolved galaxies are rare but genuinely present — no artificial spike.

The model is number-weighted (one draw per galaxy slot), which is what scene
generation needs.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
from astropy.cosmology import Planck15 as _COSMO
import astropy.units as u

_RAD_PER_ARCSEC = np.pi / 180.0 / 3600.0


class ApparentSizeModel:
    """Sampler for apparent angular half-light radius (arcsec).

    Parameters (all with literature-tuned defaults) describe the physical
    forward model; :meth:`sample` draws angular sizes from it.

    Parameters
    ----------
    r0_kpc, sigma_lnR, size_z_evolution
        Physical half-light radius is ``r0_kpc · (1+z)^−size_z_evolution``
        times a log-normal factor ``exp(N(0, sigma_lnR))``.
    nz_z0, nz_beta
        Redshift distribution ``n(z) ∝ z² exp(−(z/nz_z0)^nz_beta)``.
    theta_min_arcsec, theta_max_arcsec
        Hard clip on the returned angular size (guards the pathological tails).
    zmax, n_grid
        Redshift grid extent / resolution for the inverse-CDF + d_A spline
        (precomputed once at construction).
    """

    def __init__(
        self,
        *,
        r0_kpc: float = 2.2,
        sigma_lnR: float = 0.55,
        size_z_evolution: float = 0.75,
        nz_z0: float = 0.42,
        nz_beta: float = 1.4,
        theta_min_arcsec: float = 0.03,
        theta_max_arcsec: float = 12.0,
        zmax: float = 5.0,
        n_grid: int = 2048,
    ):
        if r0_kpc <= 0 or sigma_lnR < 0:
            raise ValueError("r0_kpc must be > 0 and sigma_lnR ≥ 0")
        if not (theta_min_arcsec < theta_max_arcsec):
            raise ValueError("theta_min must be < theta_max")
        self.r0_kpc = float(r0_kpc)
        self.sigma_lnR = float(sigma_lnR)
        self.size_z_evolution = float(size_z_evolution)
        self.nz_z0 = float(nz_z0)
        self.nz_beta = float(nz_beta)
        self.theta_min_arcsec = float(theta_min_arcsec)
        self.theta_max_arcsec = float(theta_max_arcsec)

        # Redshift grid: inverse-CDF table for n(z) + kpc-per-arcsec spline.
        zg = np.linspace(1e-3, float(zmax), int(n_grid))
        pz = zg ** 2 * np.exp(-((zg / self.nz_z0) ** self.nz_beta))
        cdf = np.cumsum(pz)
        cdf /= cdf[-1]
        self._zg = zg
        self._zcdf = cdf
        # Proper kpc subtended by 1 arcsec at each z (angular-diameter distance).
        self._kpc_per_arcsec = (
            _COSMO.angular_diameter_distance(zg).to(u.kpc).value * _RAD_PER_ARCSEC
        )

    # ------------------------------------------------------------------ #
    def sample(
        self, rng: np.random.Generator, size: Optional[int] = None
    ) -> Union[float, np.ndarray]:
        """Draw apparent half-light radii (arcsec).

        ``size=None`` returns a Python float; an int returns an ``(size,)``
        ndarray. Fully vectorised — drawing a batch is as cheap as one draw.
        """
        scalar = size is None
        n = 1 if scalar else int(size)
        # Redshift via inverse-CDF, physical size log-normal, then θ = R / d_A.
        z = np.interp(rng.random(n), self._zcdf, self._zg)
        r_phys = (
            self.r0_kpc
            * (1.0 + z) ** (-self.size_z_evolution)
            * np.exp(rng.normal(0.0, self.sigma_lnR, n))
        )
        kpc_per_arcsec = np.interp(z, self._zg, self._kpc_per_arcsec)
        theta = np.clip(
            r_phys / kpc_per_arcsec, self.theta_min_arcsec, self.theta_max_arcsec
        )
        return float(theta[0]) if scalar else theta


class CosmosSizeSampler:
    """Draw TNG target apparent half-light radii from the COSMOS catalog's *own*
    effective-radius distribution, so injected TNG galaxies match the Sérsic
    population **by construction** — no calibration, guaranteed agreement.

    COSMOS is a deep field, so it has essentially no big, resolved galaxies; a
    small, continuous **big tail** is mixed in (log-uniform over ``big_range``,
    probability ``big_fraction``) so the network still sees real large-galaxy
    structure occasionally.

    Parameters
    ----------
    effective_re_arcsec
        Per-galaxy circularized combined bulge+disk half-light radii (arcsec),
        e.g. ``catalog.effective_re_arcsec``.
    big_fraction
        Probability a draw comes from the big tail instead of the catalog.
    big_range
        ``(lo, hi)`` arcsec for the log-uniform big tail (lo overlaps the
        catalog's upper end so the combined distribution stays continuous).
    floor_arcsec
        Drop catalog radii at/below this (degenerate, sub-pixel) before sampling.
    """

    def __init__(
        self,
        effective_re_arcsec: np.ndarray,
        *,
        big_fraction: float = 0.05,
        big_range: tuple = (1.0, 4.0),
        floor_arcsec: float = 0.02,
    ):
        re = np.asarray(effective_re_arcsec, dtype=float)
        re = re[np.isfinite(re) & (re > float(floor_arcsec))]
        if re.size == 0:
            raise ValueError("no usable COSMOS effective radii to sample from")
        self._re = re
        self.big_fraction = float(np.clip(big_fraction, 0.0, 1.0))
        self.big_lo, self.big_hi = float(big_range[0]), float(big_range[1])
        if not (0.0 < self.big_lo <= self.big_hi):
            raise ValueError(f"invalid big_range {big_range}")

    def sample(self, rng: np.random.Generator, size: Optional[int] = None
               ) -> Union[float, np.ndarray]:
        scalar = size is None
        n = 1 if scalar else int(size)
        out = self._re[rng.integers(0, self._re.size, n)].astype(float)
        if self.big_fraction > 0.0:
            big = rng.random(n) < self.big_fraction
            nb = int(big.sum())
            if nb:
                out[big] = np.exp(rng.uniform(
                    np.log(self.big_lo), np.log(self.big_hi), nb))
        return float(out[0]) if scalar else out
