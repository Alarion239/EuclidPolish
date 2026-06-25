"""Angular (flat-sky 2D Fourier) power-spectrum comparison of HR vs SR.

For each band we measure, in the Fourier domain and averaged azimuthally:

- the **transfer function** ``T(k) = sqrt(P_SR(k) / P_HR(k))`` — whether SR
  carries the right amount of power at scale ``k`` (over-smoothing → T<1,
  hallucinated/excess power → T>1), and
- the **cross-correlation coefficient** ``r(k) = P_HR×SR(k) /
  sqrt(P_HR(k)·P_SR(k))`` — whether the power at ``k`` is the *real* structure
  (r→1) or decorrelated/invented (r→0).

``r(k)`` is the direct answer to "how well does the model resolve features",
especially in the super-resolution regime above the LR sampling limit
(5 cyc/arcsec on the 0.10" LR grid) up to the HR-grid Nyquist (10 cyc/arcsec on
the 0.05" SR/HR grid).

HR ground truth exists only for the synthetic objects (``syn-lens_*`` /
``syn-gal_*``); real A/B/C lenses have no HR, so this metric is synthetic-only —
the same scoping the lensfinder ROC panel uses. The eval set mixes two
generations: newer objects carry 4-band HR ``(4, H, W)``, older ones a VIS-only
HR ``(H, W)`` paired with a 4-band SR. We therefore accumulate per band into a
**fixed physical k-grid** (cycles/arcsec), which combines stamps of different
sizes correctly and lets each band draw on whatever objects provide it.

Pure numpy/scipy (matplotlib only inside the renderer, imported lazily) so this
stays importable in the no-torch evaluation environment.
"""

from __future__ import annotations

import glob
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from euclid_polish.config import Config

#: per-band line colours (blue→red, roughly by pivot wavelength)
BAND_COLORS: Dict[str, str] = {
    "VIS": "#3b6fb0", "Y_E": "#5aae61", "J_E": "#e08214", "H_E": "#d6604d",
}

#: LR sampling Nyquist on the 0.10" grid — the super-resolution boundary.
LR_NYQUIST_CYC_ARCSEC = 1.0 / (2.0 * Config.VIS_PIXEL_SCALE_ARCSEC)  # 5.0


# --------------------------------------------------------------------------- #
# Core math (unit-tested)                                                      #
# --------------------------------------------------------------------------- #
def tukey_window_2d(n: int, alpha: float = 0.25) -> np.ndarray:
    """Separable 2D Tukey window (``n×n``) to suppress finite-stamp leakage."""
    from scipy.signal.windows import tukey

    w = tukey(n, alpha)
    return np.outer(w, w)


def cross_power_2d(
    a: np.ndarray, b: np.ndarray, window: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """2D auto/cross power for two same-shape real images.

    Each image is mean-subtracted (kills the DC term) and multiplied by
    ``window`` before the FFT. Returns ``(P_aa, P_bb, P_ab)`` as unshifted 2D
    real arrays, where ``P_ab = Re(A · conj(B))``.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    if window is not None:
        a = (a - a.mean()) * window
        b = (b - b.mean()) * window
    else:
        a = a - a.mean()
        b = b - b.mean()
    fa = np.fft.fft2(a)
    fb = np.fft.fft2(b)
    p_aa = (fa * np.conj(fa)).real
    p_bb = (fb * np.conj(fb)).real
    p_ab = (fa * np.conj(fb)).real
    return p_aa, p_bb, p_ab


def k_magnitude_2d(n: int, pixel_scale_arcsec: float) -> np.ndarray:
    """``|k|`` (cycles/arcsec) for an ``n×n`` unshifted FFT grid."""
    f = np.fft.fftfreq(n, d=pixel_scale_arcsec)  # cycles / arcsec
    kx, ky = np.meshgrid(f, f, indexing="xy")
    return np.hypot(kx, ky)


def log_k_edges(
    pixel_scale_arcsec: float, kmin: float = 0.2, nbins: int = 24
) -> np.ndarray:
    """Log-spaced k-bin edges from ``kmin`` to the Nyquist ``1/(2·pixscale)``."""
    k_nyq = 1.0 / (2.0 * pixel_scale_arcsec)
    return np.geomspace(kmin, k_nyq, nbins + 1)


def bin_powers(
    hr: np.ndarray,
    sr: np.ndarray,
    pixel_scale_arcsec: float,
    k_edges: np.ndarray,
    window: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Azimuthally-binned ``(p_hr, p_sr, p_cross, count)`` for one HR/SR pair."""
    n = hr.shape[0]
    if window is None:
        window = tukey_window_2d(n)
    p_hr2, p_sr2, p_x2 = cross_power_2d(hr, sr, window)
    kmag = k_magnitude_2d(n, pixel_scale_arcsec).ravel()
    nb = len(k_edges) - 1
    idx = np.digitize(kmag, np.asarray(k_edges, float)) - 1
    keep = (idx >= 0) & (idx < nb)
    idx = idx[keep]
    bh = np.bincount(idx, weights=p_hr2.ravel()[keep], minlength=nb)
    bs = np.bincount(idx, weights=p_sr2.ravel()[keep], minlength=nb)
    bx = np.bincount(idx, weights=p_x2.ravel()[keep], minlength=nb)
    bc = np.bincount(idx, minlength=nb).astype(float)
    return bh, bs, bx, bc


def ratios_from_powers(
    bh: np.ndarray, bs: np.ndarray, bx: np.ndarray, bc: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """``(T, r)`` from binned powers; empty bins → NaN. r ∈ [-1, 1] per object."""
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.sqrt(bs / bh)
        r = bx / np.sqrt(bh * bs)
    empty = bc <= 0
    t[empty] = np.nan
    r[empty] = np.nan
    return t, r


class SpectrumAccumulator:
    """Raw-sum stacked auto/cross power on a fixed physical k-grid.

    Sums power across objects, the standard stacked-spectrum estimator. NOTE:
    because power scales as flux², a few extreme-flux objects dominate this sum,
    so the renderer prefers a per-object *median* (see ``BandStat``); this class
    backs the low-level math/unit tests and physical k-grid combination of
    different stamp sizes.
    """

    def __init__(self, k_edges: np.ndarray):
        self.k_edges = np.asarray(k_edges, dtype=np.float64)
        nb = len(self.k_edges) - 1
        self.sum_hr = np.zeros(nb)
        self.sum_sr = np.zeros(nb)
        self.sum_cross = np.zeros(nb)
        self.count = np.zeros(nb)
        self.n_obj = 0

    def add(
        self,
        hr: np.ndarray,
        sr: np.ndarray,
        pixel_scale_arcsec: float,
        window: Optional[np.ndarray] = None,
    ) -> None:
        """Add one HR/SR plane pair (same shape) to the ensemble."""
        bh, bs, bx, bc = bin_powers(hr, sr, pixel_scale_arcsec, self.k_edges, window)
        self.sum_hr += bh
        self.sum_sr += bs
        self.sum_cross += bx
        self.count += bc
        self.n_obj += 1

    def finalize(self) -> Dict[str, np.ndarray]:
        """Return ``{k, T, r, p_hr, p_sr, count}``; empty bins are NaN."""
        kc = np.sqrt(self.k_edges[:-1] * self.k_edges[1:])
        t, r = ratios_from_powers(
            self.sum_hr, self.sum_sr, self.sum_cross, self.count
        )
        return {
            "k": kc, "T": t, "r": r,
            "p_hr": self.sum_hr.copy(), "p_sr": self.sum_sr.copy(),
            "count": self.count.copy(),
        }


class BandStat:
    """Per-object T(k)/r(k) curves for one band, aggregated by median.

    Equal-weight across objects (robust to flux outliers): each object yields one
    bounded ``r_i(k) ∈ [-1, 1]`` and ``T_i(k)``; we report the median curve plus
    the 16–84% inter-object spread.
    """

    def __init__(self, k_edges: np.ndarray):
        self.k_edges = np.asarray(k_edges, dtype=np.float64)
        self.kc = np.sqrt(self.k_edges[:-1] * self.k_edges[1:])
        self._t_rows: List[np.ndarray] = []
        self._r_rows: List[np.ndarray] = []

    @property
    def n_obj(self) -> int:
        return len(self._t_rows)

    def add(
        self,
        hr: np.ndarray,
        sr: np.ndarray,
        pixel_scale_arcsec: float,
        window: Optional[np.ndarray] = None,
    ) -> None:
        bh, bs, bx, bc = bin_powers(hr, sr, pixel_scale_arcsec, self.k_edges, window)
        t, r = ratios_from_powers(bh, bs, bx, bc)
        self._t_rows.append(t)
        self._r_rows.append(r)

    def finalize(self) -> Dict[str, np.ndarray]:
        nb = len(self.kc)
        if not self._t_rows:
            nan = np.full(nb, np.nan)
            return {"k": self.kc, "T": nan, "r": nan, "T_lo": nan, "T_hi": nan,
                    "r_lo": nan, "r_hi": nan, "count": np.zeros(nb), "n_obj": 0}
        import warnings

        T = np.vstack(self._t_rows)
        R = np.vstack(self._r_rows)
        count = np.sum(np.isfinite(R), axis=0).astype(float)
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN low-k bins
            out = {
                "k": self.kc,
                "T": np.nanmedian(T, axis=0),
                "T_lo": np.nanpercentile(T, 16, axis=0),
                "T_hi": np.nanpercentile(T, 84, axis=0),
                "r": np.nanmedian(R, axis=0),
                "r_lo": np.nanpercentile(R, 16, axis=0),
                "r_hi": np.nanpercentile(R, 84, axis=0),
                "count": count,
                "n_obj": self.n_obj,
            }
        empty = count <= 0
        for key in ("T", "T_lo", "T_hi", "r", "r_lo", "r_hi"):
            out[key][empty] = np.nan
        return out


# --------------------------------------------------------------------------- #
# Data loading                                                                 #
# --------------------------------------------------------------------------- #
def _bands_from_cube(arr: np.ndarray) -> Dict[int, np.ndarray]:
    """Map a HR/SR FITS array to ``{band_index: 2D plane}``.

    Accepts ``(H, W)`` (VIS-only → band 0) or ``(C, H, W)`` (channel-first, the
    stored convention) and returns native-endian float64 planes.
    """
    a = np.asarray(arr)
    if a.ndim == 2:
        return {0: a.astype(np.float64)}
    if a.ndim == 3:
        # channel-first (C, H, W): C is the small axis.
        if a.shape[0] <= a.shape[-1] and a.shape[0] <= 8:
            return {c: a[c].astype(np.float64) for c in range(a.shape[0])}
        # channel-last fallback (H, W, C)
        return {c: a[..., c].astype(np.float64) for c in range(a.shape[-1])}
    raise ValueError(f"unexpected FITS ndim={a.ndim}, shape={a.shape}")


# --------------------------------------------------------------------------- #
# Renderer (matplotlib)                                                        #
# --------------------------------------------------------------------------- #
def render_power_spectrum_summary(run_dir: str, out_png: str) -> Optional[str]:
    """Render the per-band HR-vs-SR power-spectrum figure → ``out_png``.

    Two rows (linear electrons / asinh space) × two columns (transfer function
    ``T(k)`` and cross-correlation ``r(k)``), one curve per band, accumulated
    over every synthetic object that carries an ``HR.fits``. Returns the PNG
    path, or ``None`` if no HR/SR pairs are found.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from astropy.io import fits

    band_names = list(Config.LR_INPUT_BAND_NAMES)  # (VIS, Y_E, J_E, H_E)
    pixel_scale = float(Config.DEFAULT_PIXEL_SCALE)  # 0.05" HR/SR grid
    k_edges = log_k_edges(pixel_scale)
    spaces = ("linear", "asinh")

    # one per-object-median stat per (space, band)
    accs: Dict[Tuple[str, str], BandStat] = {
        (sp, b): BandStat(k_edges) for sp in spaces for b in band_names
    }
    stretch = float(Config.STRETCH_SCALE_E)

    obj_dirs = sorted(
        d for d in glob.glob(os.path.join(run_dir, "*"))
        if os.path.isfile(os.path.join(d, "HR.fits"))
        and os.path.isfile(os.path.join(d, "SR.fits"))
    )
    n_used = 0
    for d in obj_dirs:
        try:
            with fits.open(os.path.join(d, "HR.fits"), memmap=False) as h:
                hr_cube = _bands_from_cube(h[0].data)
            with fits.open(os.path.join(d, "SR.fits"), memmap=False) as h:
                sr_cube = _bands_from_cube(h[0].data)
        except Exception:
            continue
        used = False
        for bi, bname in enumerate(band_names):
            if bi not in hr_cube or bi not in sr_cube:
                continue
            hr = hr_cube[bi]
            sr = sr_cube[bi]
            if hr.shape != sr.shape:
                continue
            win = tukey_window_2d(hr.shape[0])
            accs[("linear", bname)].add(hr, sr, pixel_scale, win)
            accs[("asinh", bname)].add(
                np.arcsinh(hr / stretch), np.arcsinh(sr / stretch), pixel_scale, win
            )
            used = True
        n_used += int(used)

    if n_used == 0:
        return None

    # finalize once per (space, band)
    res = {(sp, b): accs[(sp, b)].finalize() for sp in spaces for b in band_names}

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.5))
    col_meta = (
        (0, "T", "transfer function  T(k)=√(P_SR/P_HR)", (0.0, 1.45)),
        (1, "r", "cross-correlation  r(k)=P_HR×SR/√(P_HR·P_SR)", (0.0, 1.05)),
    )
    for ri, sp in enumerate(spaces):
        for ci, key, short, ylim in col_meta:
            ax = axes[ri, ci]
            for bname in band_names:
                rb = res[(sp, bname)]
                color = BAND_COLORS.get(bname, "#444")
                ax.fill_between(
                    rb["k"], rb[key + "_lo"], rb[key + "_hi"],
                    color=color, alpha=0.12, lw=0,
                )
                ax.plot(rb["k"], rb[key], "-o", ms=3.0, lw=1.7,
                        color=color, label=bname)
            ax.axhline(1.0, ls=":", color="#888", lw=1.0)
            ax.axvline(
                LR_NYQUIST_CYC_ARCSEC, ls="--", color="#444", lw=1.1,
                label="LR Nyquist (5)" if (ri == 0 and ci == 0) else None,
            )
            # per-band PSF resolution proxy: k = 1 / FWHM
            for bname in band_names:
                fwhm = float(Config.get_band(bname).psf_fwhm_arcsec)
                ax.axvline(1.0 / fwhm, ls="-", lw=0.8, alpha=0.22,
                           color=BAND_COLORS.get(bname, "#444"))
            ax.set_xscale("log")
            ax.set_xlim(k_edges[0], k_edges[-1])
            ax.set_ylim(*ylim)
            ax.set_xlabel("k  [cycles / arcsec]")
            ax.set_ylabel(f"{key}(k)  [{sp}]")
            ax.grid(alpha=0.2)
            ax.set_title(f"{sp} — {short}")
    axes[0, 0].legend(fontsize=8, loc="lower left", ncol=2)

    n_vis = res[("linear", "VIS")]["n_obj"]
    n_4 = res[("linear", "H_E")]["n_obj"]
    fig.suptitle(
        "Angular power spectrum — HR vs SR (synthetic, per-object median ± 16–84%)\n"
        f"VIS from {n_vis} objects · NISP bands from {n_4} · "
        "thin verticals = 1/PSF-FWHM per band · dashed = LR Nyquist (5 cyc/arcsec)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_png
