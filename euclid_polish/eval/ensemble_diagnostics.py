"""Pixel-level ensemble disagreement diagnostics (VIS band, electrons).

Every plot here asks one question: **is the cross-member per-pixel std a usable
stand-in for the actual error of the ensemble mean** — the thing we cannot
measure on real Euclid images where there is no HR truth?

- **std vs error** — joint distribution of the member std σ(x) and the actual
  absolute error |mean − HR|(x) per pixel. If disagreement predicts error the
  binned median tracks the Gaussian guide 0.674·σ (half the |error| mass below
  it); a flat median means the members agree confidently on the *wrong* answer.
- **std vs brightness** — where the disagreement lives: σ(x) against the HR
  pixel value. Separates photon-noise-driven spread (rises with flux) from
  hallucination on faint structure (excess spread at low flux).
- **calibration** — (a) the z-score histogram z = (mean − HR)/σ against the
  standard normal: if the ensemble spread were a calibrated error bar, z ~
  N(0, 1) (in practice an M-member std underestimates the error, so |z| tails
  are heavy); (b) per-field mean σ vs per-field RMSE — can disagreement rank
  *fields* by reconstruction quality without truth?

The accumulators are pure numpy (unit-tested); the renderers lazily import
matplotlib. Fed one field at a time by the web helper, which streams the cached
per-field cubes written by the last "Evaluate on test set" run.
"""

from __future__ import annotations

import numpy as np

from euclid_polish.config import Config

#: log10(e⁻) histogram edges for std / |error| axes — wide enough for both the
#: ~0.05 e⁻ background disagreement and the ~1e3 e⁻ spread on bright stars.
LOG_E_RANGE = (-4.0, 4.0)
LOG_E_BINS = 96

#: z-score histogram edges (z = error / member std).
Z_RANGE = (-8.0, 8.0)
Z_BINS = 161


def _log10_clipped(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.log10(np.maximum(np.asarray(x, np.float64), eps))


class EnsembleDiagnosticsAccumulator:
    """Streaming pixel/field statistics for the three diagnostic figures.

    :meth:`add` takes one field's VIS planes in electrons: ``hr`` and ``mean``
    are ``(n, n)``, ``members`` is ``(M, n, n)``. The per-pixel std is
    recomputed from the members (population std, matching ``EnsembleModel``).
    """

    def __init__(self, *, stretch: float | None = None,
                 sample_k: int = 6) -> None:
        self.stretch = float(stretch if stretch is not None else
                             Config.STRETCH_SCALE_E)
        lo, hi = LOG_E_RANGE
        self.log_edges = np.linspace(lo, hi, LOG_E_BINS + 1)
        # brightness axis: asinh(HR/stretch) — handles the zero/negative sky.
        self.bright_edges = np.linspace(-1.0, 12.0, LOG_E_BINS + 1)
        self.z_edges = np.linspace(*Z_RANGE, Z_BINS + 1)
        nb = LOG_E_BINS
        self.h_std_err = np.zeros((nb, nb), np.float64)      # [std, |err|]
        self.h_bright_std = np.zeros((nb, nb), np.float64)   # [bright, std]
        self.h_z = np.zeros(Z_BINS, np.float64)
        self.n_z = 0                                          # z-scored pixels
        self.cover = np.zeros(3, np.float64)                  # |z|<1,2,3 counts
        self.field_mean_std: list[float] = []
        self.field_rmse: list[float] = []
        self.n_fields = 0
        self.n_members = 0

        # ---- pixel back-tracing reservoirs --------------------------------- #
        # For each 2D-histogram cell keep up to ``sample_k`` example pixel
        # locations ``(field, y, x)``, so a click on a heatmap cell can pull the
        # real image stamps that fell into it. Sampling is stratified by FIELD
        # (one representative pixel per field per occupied cell, reservoir-
        # replaced) — the examples then span different fields, which is what a
        # human wants when inspecting "what lives here". Keyed by ``i*NB + j``.
        self.sample_k = int(sample_k)
        self.se_samples: dict[int, list[tuple[int, int, int]]] = {}
        self.se_seen: dict[int, int] = {}
        self.bs_samples: dict[int, list[tuple[int, int, int]]] = {}
        self.bs_seen: dict[int, int] = {}
        # Deterministic given call order → reproducible sidecars + testable.
        self._rng = np.random.default_rng(0)

    def _reservoir_add_field(self, samples, seen, bin_ids, W, field_index):
        """Insert one random representative pixel per occupied cell of one
        field into ``samples`` (capacity ``sample_k`` per cell), reservoir-
        replacing across fields. ``bin_ids`` is the flattened ``i*NB + j`` cell
        key per pixel; pixel index → ``(y, x) = divmod(idx, W)``."""
        k = self.sample_k
        rng = self._rng
        order = np.argsort(bin_ids, kind="stable")
        sb = bin_ids[order]
        if sb.size == 0:
            return
        change = np.ones(sb.size, bool)
        change[1:] = sb[1:] != sb[:-1]
        starts = np.flatnonzero(change)
        ends = np.append(starts[1:], sb.size)
        for s0, s1 in zip(starts, ends):
            key = int(sb[s0])
            pick = int(order[s0 + int(rng.integers(s1 - s0))])
            y, x = divmod(pick, W)
            cnt = seen.get(key, 0) + 1
            seen[key] = cnt
            lst = samples.setdefault(key, [])
            if len(lst) < k:
                lst.append((int(field_index), int(y), int(x)))
            else:
                j = int(rng.integers(cnt))
                if j < k:
                    lst[j] = (int(field_index), int(y), int(x))

    def add(self, hr: np.ndarray, mean: np.ndarray,
            members: np.ndarray, *, field_index: int | None = None) -> None:
        hr = np.asarray(hr, np.float64)
        mean = np.asarray(mean, np.float64)
        members = np.asarray(members, np.float64)
        if (hr.ndim != 2 or mean.shape != hr.shape or members.ndim != 3
                or members.shape[1:] != hr.shape or len(members) < 2):
            return
        std = members.std(axis=0)
        err = mean - hr

        ls = _log10_clipped(std).ravel()
        le = _log10_clipped(np.abs(err)).ravel()
        lo, hi = self.log_edges[0], self.log_edges[-1]
        h, _, _ = np.histogram2d(np.clip(ls, lo, hi), np.clip(le, lo, hi),
                                 bins=(self.log_edges, self.log_edges))
        self.h_std_err += h

        bright = np.arcsinh(hr / self.stretch).ravel()
        blo, bhi = self.bright_edges[0], self.bright_edges[-1]
        h, _, _ = np.histogram2d(np.clip(bright, blo, bhi),
                                 np.clip(ls, lo, hi),
                                 bins=(self.bright_edges, self.log_edges))
        self.h_bright_std += h

        ok = std.ravel() > 0
        z = err.ravel()[ok] / std.ravel()[ok]
        self.h_z += np.histogram(np.clip(z, *Z_RANGE), bins=self.z_edges)[0]
        self.n_z += int(ok.sum())
        az = np.abs(z)
        self.cover += [(az < 1).sum(), (az < 2).sum(), (az < 3).sum()]

        self.field_mean_std.append(float(std.mean()))
        self.field_rmse.append(float(np.sqrt(np.mean(err ** 2))))
        self.n_fields += 1
        self.n_members = max(self.n_members, len(members))

        # Back-tracing reservoirs — only for fields whose cubes are cached (the
        # caller passes ``field_index``), so a sampled pixel is always
        # reconstructable into an image stamp on click.
        if field_index is not None:
            NB = LOG_E_BINS
            W = int(hr.shape[1])
            std_bin = np.clip(
                np.searchsorted(self.log_edges, ls, side="right") - 1, 0, NB - 1)
            err_bin = np.clip(
                np.searchsorted(self.log_edges, le, side="right") - 1, 0, NB - 1)
            bright_bin = np.clip(
                np.searchsorted(self.bright_edges, bright, side="right") - 1,
                0, NB - 1)
            self._reservoir_add_field(self.se_samples, self.se_seen,
                                      std_bin * NB + err_bin, W, field_index)
            self._reservoir_add_field(self.bs_samples, self.bs_seen,
                                      bright_bin * NB + std_bin, W, field_index)

    # ---- derived curves (pure, testable) ---------------------------------- #
    def binned_median_err(self) -> tuple[np.ndarray, np.ndarray]:
        """Per std-bin median |error| from the 2D histogram → ``(std, err)``
        in electrons (bin geometric centres); empty std-bins are NaN."""
        cen = 10.0 ** (0.5 * (self.log_edges[:-1] + self.log_edges[1:]))
        med = np.full(LOG_E_BINS, np.nan)
        for i, row in enumerate(self.h_std_err):
            tot = row.sum()
            if tot <= 0:
                continue
            cum = np.cumsum(row)
            med[i] = cen[int(np.searchsorted(cum, 0.5 * tot))]
        return cen, med

    def binned_std_percentiles(self) -> dict[str, np.ndarray]:
        """Median + 16/84% of std per brightness bin → arrays over the
        brightness axis (asinh units); empty bins are NaN."""
        cen = 10.0 ** (0.5 * (self.log_edges[:-1] + self.log_edges[1:]))
        bright = 0.5 * (self.bright_edges[:-1] + self.bright_edges[1:])
        out = {"bright": bright}
        for name, q in (("lo", 0.16), ("med", 0.5), ("hi", 0.84)):
            v = np.full(LOG_E_BINS, np.nan)
            for i, row in enumerate(self.h_bright_std):
                tot = row.sum()
                if tot > 0:
                    v[i] = cen[int(np.searchsorted(np.cumsum(row), q * tot))]
            out[name] = v
        return out

    def to_payload(self) -> dict:
        """JSON-ready dict of everything the frontend renderers need — the
        2D histograms, binned curves and calibration stats, with NaN → None
        (JSON has no NaN). Kilobytes, so the browser can redraw styling
        changes instantly without touching the cubes again."""
        def _l(a):
            return [None if not np.isfinite(v) else float(v)
                    for v in np.asarray(a, float)]

        cen, med = self.binned_median_err()
        pct = self.binned_std_percentiles()
        n_z = max(self.n_z, 1)
        width = self.z_edges[1] - self.z_edges[0]
        return {
            "n_fields": int(self.n_fields),
            "n_members": int(self.n_members),
            "std_err": {
                "edges": _l(self.log_edges),           # log10 e⁻, both axes
                "hist": self.h_std_err.astype(int).tolist(),
                "med_std": _l(np.log10(cen)),
                "med_err": _l(np.log10(med)),
            },
            "bright_std": {
                "bright_edges": _l(self.bright_edges),  # asinh(x/stretch)
                "std_edges": _l(self.log_edges),
                "hist": self.h_bright_std.astype(int).tolist(),
                "bright": _l(pct["bright"]),
                "lo": _l(np.log10(pct["lo"])),
                "med": _l(np.log10(pct["med"])),
                "hi": _l(np.log10(pct["hi"])),
                "stretch": float(self.stretch),
            },
            "calibration": {
                "z_edges": _l(self.z_edges),
                "pdf": _l(self.h_z / (n_z * width)),
                "stats": {k: (None if not np.isfinite(v) else float(v))
                          for k, v in self.z_stats().items()},
                "field_std": _l(self.field_mean_std),
                "field_rmse": _l(self.field_rmse),
            },
        }

    def samples_payload(self) -> dict:
        """Back-tracing sidecar: per diagnostic, a ``{"i,j": [[field, y, x], …]}``
        map of example pixel locations per histogram cell. Small (≤ ``sample_k``
        per occupied cell). A click on a heatmap cell reads this to fetch the
        real image stamps for that cell."""
        NB = LOG_E_BINS

        def enc(res: dict) -> dict:
            return {f"{key // NB},{key % NB}":
                    [[int(f), int(y), int(x)] for f, y, x in v]
                    for key, v in res.items()}

        return {
            "n_fields": int(self.n_fields),
            "n_members": int(self.n_members),
            "sample_k": int(self.sample_k),
            "std_err": enc(self.se_samples),
            "bright_std": enc(self.bs_samples),
        }

    def z_stats(self) -> dict[str, float]:
        """Coverage fractions + robust z width (Gaussian ⇒ 0.683/0.954/0.997,
        sigma_z = 1)."""
        n = max(self.n_z, 1)
        cen = 0.5 * (self.z_edges[:-1] + self.z_edges[1:])
        cum = np.cumsum(self.h_z)
        sig = np.nan
        if cum[-1] > 0:
            q16 = cen[int(np.searchsorted(cum, 0.16 * cum[-1]))]
            q84 = cen[int(np.searchsorted(cum, 0.84 * cum[-1]))]
            sig = 0.5 * (q84 - q16)
        return {"cover1": self.cover[0] / n, "cover2": self.cover[1] / n,
                "cover3": self.cover[2] / n, "sigma_z": float(sig)}


# --------------------------------------------------------------------------- #
# Renderers                                                                    #
# --------------------------------------------------------------------------- #
def _density_panel(ax, h2, x_edges, y_edges, *, xlog=True, ylog=True):
    import matplotlib.colors as mcolors

    h = np.where(h2 > 0, h2, np.nan).T          # imshow-style: y rows
    x = 10.0 ** x_edges if xlog else x_edges
    y = 10.0 ** y_edges if ylog else y_edges
    m = ax.pcolormesh(x, y, h, cmap="viridis",
                      norm=mcolors.LogNorm(vmin=1, vmax=max(np.nanmax(h), 2)))
    if xlog:
        ax.set_xscale("log")
    if ylog:
        ax.set_yscale("log")
    ax.grid(alpha=0.15)
    return m


def render_std_vs_error(out_png: str, acc: EnsembleDiagnosticsAccumulator,
                        ) -> str | None:
    """Per-pixel member std vs |ensemble mean − HR| (VIS, electrons)."""
    if acc.n_fields == 0 or acc.h_std_err.sum() <= 0:
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    m = _density_panel(ax, acc.h_std_err, acc.log_edges, acc.log_edges)
    cen, med = acc.binned_median_err()
    ax.plot(cen, med, "-o", ms=3, lw=2.0, color="#d6604d",
            label="median |error| per std bin")
    lim = 10.0 ** np.array(LOG_E_RANGE)
    ax.plot(lim, lim, ls="--", color="#333", lw=1.2, label="|error| = std")
    ax.plot(lim, 0.6745 * lim, ls=":", color="#333", lw=1.2,
            label="0.674·std (calibrated Gaussian)")
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xlabel("cross-member per-pixel std  σ  [e⁻]")
    ax.set_ylabel("actual error  |ensemble mean − HR|  [e⁻]")
    ax.set_title("Does disagreement predict error?  "
                 f"(VIS, {acc.n_fields} fields, {acc.n_members} members)")
    ax.legend(fontsize=8, loc="upper left")
    fig.colorbar(m, ax=ax, label="pixels")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_png


def render_std_vs_brightness(out_png: str,
                             acc: EnsembleDiagnosticsAccumulator,
                             ) -> str | None:
    """Per-pixel member std vs HR pixel brightness (VIS)."""
    if acc.n_fields == 0 or acc.h_bright_std.sum() <= 0:
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedFormatter, FixedLocator

    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    m = _density_panel(ax, acc.h_bright_std, acc.bright_edges, acc.log_edges,
                       xlog=False)
    p = acc.binned_std_percentiles()
    ax.fill_between(p["bright"], p["lo"], p["hi"], color="#d6604d",
                    alpha=0.18, lw=0, label="16–84%")
    ax.plot(p["bright"], p["med"], "-o", ms=3, lw=2.0, color="#d6604d",
            label="median std per brightness bin")
    # asinh axis, ticks labelled in electrons
    tick_e = np.array([0.0, 100, 1e3, 1e4, 1e5, 1e6])
    ax.xaxis.set_major_locator(FixedLocator(np.arcsinh(tick_e / acc.stretch)))
    ax.xaxis.set_major_formatter(FixedFormatter(
        ["0", "100", "10³", "10⁴", "10⁵", "10⁶"]))
    ax.set_xlim(acc.bright_edges[0], acc.bright_edges[-1])
    ax.set_ylim(*(10.0 ** np.array(LOG_E_RANGE)))
    ax.set_xlabel("HR pixel brightness  [e⁻]  (asinh-spaced axis)")
    ax.set_ylabel("cross-member per-pixel std  σ  [e⁻]")
    ax.set_title("Where does disagreement live?  "
                 f"(VIS, {acc.n_fields} fields, {acc.n_members} members)")
    ax.legend(fontsize=8, loc="upper left")
    fig.colorbar(m, ax=ax, label="pixels")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_png


def render_calibration(out_png: str, acc: EnsembleDiagnosticsAccumulator,
                       ) -> str | None:
    """z-score histogram vs N(0,1) + per-field mean-std vs RMSE scatter."""
    if acc.n_fields == 0 or acc.n_z == 0:
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_z, ax_f) = plt.subplots(1, 2, figsize=(13.0, 5.4))

    cen = 0.5 * (acc.z_edges[:-1] + acc.z_edges[1:])
    width = acc.z_edges[1] - acc.z_edges[0]
    pdf = acc.h_z / (acc.n_z * width)
    st = acc.z_stats()
    ax_z.step(cen, pdf, where="mid", color="#3b6fb0", lw=1.6,
              label="z = (mean − HR)/σ")
    zz = np.linspace(*Z_RANGE, 400)
    ax_z.plot(zz, np.exp(-0.5 * zz ** 2) / np.sqrt(2 * np.pi), ls="--",
              color="#333", lw=1.2, label="N(0, 1) — calibrated")
    ax_z.set_yscale("log")
    ax_z.set_ylim(max(pdf[pdf > 0].min() * 0.5, 1e-8), 2.0)
    ax_z.set_xlim(*Z_RANGE)
    ax_z.set_xlabel("z-score")
    ax_z.set_ylabel("pixel PDF")
    ax_z.set_title(
        "Is the member std a calibrated error bar?\n"
        f"σ(z) = {st['sigma_z']:.2f} (1 if calibrated) · coverage "
        f"|z|<1: {st['cover1'] * 100:.1f}% (68.3) · "
        f"|z|<2: {st['cover2'] * 100:.1f}% (95.4) · "
        f"|z|<3: {st['cover3'] * 100:.1f}% (99.7)", fontsize=10)
    ax_z.grid(alpha=0.2)
    ax_z.legend(fontsize=8)

    ms = np.asarray(acc.field_mean_std, float)
    rm = np.asarray(acc.field_rmse, float)
    ok = (ms > 0) & (rm > 0)
    rho = (np.corrcoef(np.log10(ms[ok]), np.log10(rm[ok]))[0, 1]
           if ok.sum() >= 2 else np.nan)
    ax_f.scatter(ms, rm, s=18, color="#3b6fb0", alpha=0.7, edgecolors="none")
    ax_f.set_xscale("log")
    ax_f.set_yscale("log")
    ax_f.set_xlabel("per-field mean member std  [e⁻]")
    ax_f.set_ylabel("per-field RMSE (mean vs HR)  [e⁻]")
    ax_f.set_title("Does disagreement rank fields by quality?\n"
                   f"{acc.n_fields} fields · log–log Pearson r = {rho:.2f}",
                   fontsize=10)
    ax_f.grid(alpha=0.2, which="both")

    fig.suptitle(f"Ensemble error calibration — VIS ({acc.n_members} members)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_png
