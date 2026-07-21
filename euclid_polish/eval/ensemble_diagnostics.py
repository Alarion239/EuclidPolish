"""Pixel-level ensemble disagreement diagnostics (VIS band, electrons).

Every plot here asks one question: **is the cross-member per-pixel std a usable
stand-in for the actual error of the ensemble mean** — the thing we cannot
measure on real Euclid images where there is no HR truth?

- **std vs error** — joint distribution of the member std σ(x) and the actual
  absolute error |mean − HR|(x) per pixel. If disagreement predicts error the
  binned median rises with σ (tracking the |error|=σ diagonal); a flat median
  means the members agree confidently on the *wrong* answer.
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

#: Initial log10(e⁻) histogram range for std / |error| axes.  The upper edge is
#: expanded from observed finite σ and error values as fields are streamed;
#: this lower bound and initial range keep ordinary diagnostics unchanged.
LOG_E_RANGE = (-4.0, 6.0)
LOG_E_BINS = 96

#: Combiner-input bins.  These are deliberately coarser than the error axis:
#: each occupied feature cell then has enough real pixels for a stable median
#: and useful click-to-inspect examples.
FEATURE_BINS = 48

#: z-score histogram edges (z = error / member std).
Z_RANGE = (-8.0, 8.0)
Z_BINS = 161


def _log10_clipped(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.log10(np.maximum(np.asarray(x, np.float64), eps))


def _rebin_axis(hist: np.ndarray, old_edges: np.ndarray,
                new_edges: np.ndarray, axis: int) -> np.ndarray:
    """Move an existing histogram onto expanded edges using bin centres.

    The accumulator is streaming, so an observed high-value field can arrive
    after lower-value fields have already been counted.  Only the upper range
    grows; centre-based remapping preserves those counts without retaining all
    pixels.
    """
    a = np.moveaxis(np.asarray(hist), axis, 0)
    out = np.zeros((len(new_edges) - 1, *a.shape[1:]), dtype=a.dtype)
    old_centres = 0.5 * (old_edges[:-1] + old_edges[1:])
    new_bins = np.clip(np.searchsorted(new_edges, old_centres, side="right") - 1,
                       0, len(new_edges) - 2)
    for old_bin, new_bin in enumerate(new_bins):
        out[new_bin] += a[old_bin]
    return np.moveaxis(out, 0, axis)


class EnsembleDiagnosticsAccumulator:
    """Streaming pixel/field statistics for the three diagnostic figures.

    :meth:`add` takes one field's VIS planes in electrons: ``hr`` and ``mean``
    are ``(n, n)``, ``members`` is ``(M, n, n)``. The per-pixel std is
    recomputed from the members (population std, matching ``EnsembleModel``).
    """

    def __init__(self, *, stretch: float | None = None,
                 sample_k: int = 10) -> None:
        self.stretch = float(stretch if stretch is not None else
                             Config.STRETCH_SCALE_E)
        lo, hi = LOG_E_RANGE
        self.log_edges = np.linspace(lo, hi, LOG_E_BINS + 1)
        # brightness axis: asinh(HR/stretch) — handles the zero/negative sky.
        self.bright_edges = np.linspace(-1.0, 12.0, LOG_E_BINS + 1)
        self.z_edges = np.linspace(*Z_RANGE, Z_BINS + 1)
        nb = LOG_E_BINS
        self.h_std_err = np.zeros((nb, nb), np.float64)      # [std, |err|]
        # Same disagreement axis, one error distribution per point estimate.
        # This is additive: the legacy ``h_std_err`` / calibration remain tied
        # to the displayed max-RBF output when available.
        self.h_std_err_models: dict[str, np.ndarray] = {}
        self.std_err_model_fields: dict[str, int] = {}
        # Per-combiner |error| conditioned on the coordinates that actually
        # drive its gate.  Values are histograms over [...feature bins, error]
        # so medians remain streaming/cache-friendly (no pixel arrays retained).
        self.h_combiner_feature_err: dict[str, dict[str, np.ndarray]] = {}
        self.combiner_feature_meta: dict[str, dict] = {}
        self.h_bright_std = np.zeros((nb, nb), np.float64)   # [bright, std]
        self.h_z = np.zeros(Z_BINS, np.float64)
        self.n_z = 0                                          # z-scored pixels
        self.cover = np.zeros(3, np.float64)                  # |z|<1,2,3 counts
        self.field_mean_std: list[float] = []
        self.field_rmse: list[float] = []
        self.n_fields = 0
        self.n_members = 0
        self.n_pred_combiner = 0   # fields whose error was scored vs the combiner

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
        self.se_model_samples: dict[str, dict[int, list[tuple[int, int, int]]]] = {}
        self.se_model_seen: dict[str, dict[int, int]] = {}
        self.cf_model_samples: dict[
            str, dict[str, dict[int, list[tuple[int, int, int]]]]
        ] = {}
        self.cf_model_seen: dict[str, dict[str, dict[int, int]]] = {}
        self.bs_samples: dict[int, list[tuple[int, int, int]]] = {}
        self.bs_seen: dict[int, int] = {}
        # Deterministic given call order → reproducible sidecars + testable.
        self._rng = np.random.default_rng(0)

    def _ensure_log_upper(self, values: tuple[np.ndarray, ...]) -> None:
        """Expand σ/error axes to contain the largest finite observed value."""
        finite_max = -np.inf
        for value in values:
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                a = _log10_clipped(np.asarray(value, np.float64))
            finite = a[np.isfinite(a)]
            if finite.size:
                finite_max = max(finite_max, float(np.max(finite)))
        old_edges = self.log_edges
        if not np.isfinite(finite_max) or finite_max <= old_edges[-1]:
            return

        # A small log-space margin keeps the largest occupied bin away from
        # the frame edge while remaining data-adaptive (rather than choosing
        # another arbitrary fixed detector-flux ceiling).
        new_hi = finite_max + max(0.10, 0.02 * abs(finite_max))
        new_edges = np.linspace(old_edges[0], new_hi, LOG_E_BINS + 1)
        self.h_std_err = _rebin_axis(self.h_std_err, old_edges, new_edges, 0)
        self.h_std_err = _rebin_axis(self.h_std_err, old_edges, new_edges, 1)
        for kind, hist in self.h_std_err_models.items():
            hist = _rebin_axis(hist, old_edges, new_edges, 0)
            self.h_std_err_models[kind] = _rebin_axis(hist, old_edges, new_edges, 1)
        self.h_bright_std = _rebin_axis(self.h_bright_std, old_edges, new_edges, 1)
        for axis_hists in self.h_combiner_feature_err.values():
            for kind, hist in axis_hists.items():
                axis_hists[kind] = _rebin_axis(hist, old_edges, new_edges, -1)

        self._remap_std_error_samples(old_edges, new_edges)
        self.log_edges = new_edges

    def _remap_cells(self, samples: dict[int, list[tuple[int, int, int]]],
                     seen: dict[int, int], old_edges: np.ndarray,
                     new_edges: np.ndarray, *, remap_x: bool = True) -> None:
        """Remap reservoir keys after the std/error grid expands."""
        nb = LOG_E_BINS
        old_centres = 0.5 * (old_edges[:-1] + old_edges[1:])
        new_bins = np.clip(np.searchsorted(new_edges, old_centres, side="right") - 1,
                           0, nb - 1)
        merged: dict[int, list[tuple[int, int, int]]] = {}
        merged_seen: dict[int, int] = {}
        for key, picks in samples.items():
            x, y = divmod(int(key), nb)
            nx = int(new_bins[x]) if remap_x else x
            ny = int(new_bins[y])
            dst = nx * nb + ny
            merged.setdefault(dst, []).extend(picks)
            merged_seen[dst] = merged_seen.get(dst, 0) + seen.get(key, len(picks))
        samples.clear()
        samples.update({key: picks[:self.sample_k] for key, picks in merged.items()})
        seen.clear()
        seen.update(merged_seen)

    def _remap_std_error_samples(self, old_edges: np.ndarray,
                                 new_edges: np.ndarray) -> None:
        self._remap_cells(self.se_samples, self.se_seen, old_edges, new_edges)
        for kind, samples in self.se_model_samples.items():
            self._remap_cells(samples, self.se_model_seen[kind],
                              old_edges, new_edges)
        # Brightness is the x axis here; only the σ (y) coordinate moves.
        self._remap_cells(self.bs_samples, self.bs_seen, old_edges, new_edges,
                          remap_x=False)

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
        for s0, s1 in zip(starts, ends, strict=True):
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

    def _axis_features(self, members: np.ndarray, axis_mode: str):
        """Return one model-independent projection and its fixed plot edges.

        Member values are transformed to the same asinh space used by the
        combiners.  Mean/std is shown in raw input coordinates; the model's RBF
        distance internally log-warps std, but the raw std axis is the quantity
        a scientist can read directly and is the input being conditioned on.
        """
        x = np.arcsinh(np.asarray(members, np.float64) / self.stretch)
        fb = FEATURE_BINS
        level_edges = np.linspace(-1.0, 13.0, fb + 1)
        if axis_mode == "mean_std":
            # Cross-member asinh spread is non-negative.  Five covers the
            # useful occupied range while clipping extreme star cores into the
            # final cell instead of letting them flatten the quiet regime.
            return ((np.mean(x, axis=0), np.std(x, axis=0)),
                    (level_edges, np.linspace(0.0, 5.0, fb + 1)),
                    ("mean member", "member std"))
        if axis_mode == "min_max":
            return ((np.min(x, axis=0), np.max(x, axis=0)),
                    (level_edges, level_edges),
                    ("min member", "max member"))
        return None

    def add(self, hr: np.ndarray, mean: np.ndarray,
            members: np.ndarray, *, combiner: np.ndarray | None = None,
            combiners: dict[str, np.ndarray | None] | None = None,
            field_index: int | None = None) -> None:
        hr = np.asarray(hr, np.float64)
        mean = np.asarray(mean, np.float64)
        members = np.asarray(members, np.float64)
        if (hr.ndim != 2 or mean.shape != hr.shape or members.ndim != 3
                or members.shape[1:] != hr.shape or len(members) < 2):
            return
        std = members.std(axis=0)
        # Evaluate every available fused point estimate against the same member
        # disagreement map. The primary legacy diagnostic remains max-RBF when
        # available, otherwise the ensemble mean.
        predictions: dict[str, np.ndarray] = {"ensemble_mean": mean}
        for kind, image in (combiners or {}).items():
            image = np.asarray(image, np.float64) if image is not None else None
            if image is not None and image.shape == hr.shape:
                predictions[str(kind)] = image
        if combiner is not None and "rbf_gate" not in predictions:
            image = np.asarray(combiner, np.float64)
            if image.shape == hr.shape:
                predictions["rbf_gate"] = image
        axis_features = {
            mode: self._axis_features(members, mode)
            for mode in ("mean_std", "min_max")
        }
        pred = mean
        if "rbf_gate" in predictions:
            pred = predictions["rbf_gate"]
            self.n_pred_combiner += 1
        err = pred - hr

        with np.errstate(over="ignore", invalid="ignore"):
            self._ensure_log_upper((std, np.abs(err), *(
                np.abs(image - hr) for image in predictions.values())))

        ls = _log10_clipped(std).ravel()
        le = _log10_clipped(np.abs(err)).ravel()
        lo, hi = self.log_edges[0], self.log_edges[-1]
        h, _, _ = np.histogram2d(np.clip(ls, lo, hi), np.clip(le, lo, hi),
                                 bins=(self.log_edges, self.log_edges))
        self.h_std_err += h
        model_log_errors: dict[str, np.ndarray] = {}
        for kind, point_estimate in predictions.items():
            model_err = _log10_clipped(np.abs(point_estimate - hr)).ravel()
            model_log_errors[kind] = model_err
            hist, _, _ = np.histogram2d(
                np.clip(ls, lo, hi), np.clip(model_err, lo, hi),
                bins=(self.log_edges, self.log_edges))
            self.h_std_err_models.setdefault(
                kind, np.zeros_like(self.h_std_err))[:] += hist
            self.std_err_model_fields[kind] = self.std_err_model_fields.get(kind, 0) + 1

            # Project every point estimate onto every requested coordinate
            # plane.  Model choice and plot geometry are intentionally
            # independent so their error surfaces can be compared directly.
            for axis_mode, feature_info in axis_features.items():
                if feature_info is None:
                    continue
                features, edges, axis_names = feature_info
                values = [np.clip(np.asarray(v).ravel(), e[0], e[-1])
                          for v, e in zip(features, edges, strict=True)]
                sample = np.column_stack((*values, np.clip(model_err, lo, hi)))
                hist, _ = np.histogramdd(sample, bins=(*edges, self.log_edges))
                axis_hists = self.h_combiner_feature_err.setdefault(axis_mode, {})
                axis_hists.setdefault(kind, np.zeros_like(hist))[:] += hist
                self.combiner_feature_meta[axis_mode] = {
                    "axis_names": axis_names, "edges": edges,
                }

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
            for kind, model_err in model_log_errors.items():
                model_err_bin = np.clip(
                    np.searchsorted(self.log_edges, model_err, side="right") - 1,
                    0, NB - 1)
                samples = self.se_model_samples.setdefault(kind, {})
                seen = self.se_model_seen.setdefault(kind, {})
                self._reservoir_add_field(samples, seen,
                                          std_bin * NB + model_err_bin,
                                          W, field_index)

                for axis_mode, feature_info in axis_features.items():
                    if feature_info is None:
                        continue
                    features, edges, _axis_names = feature_info
                    bins = [np.clip(np.searchsorted(edge, np.asarray(value).ravel(),
                                                    side="right") - 1,
                                    0, FEATURE_BINS - 1)
                            for value, edge in zip(features, edges, strict=True)]
                    # Both coordinate planes share the same compact
                    # i*FEATURE_BINS+j sidecar key convention.
                    cell_ids = (bins[0] * FEATURE_BINS +
                                (bins[1] if len(bins) == 2 else 0))
                    samples = self.cf_model_samples.setdefault(
                        axis_mode, {}).setdefault(kind, {})
                    seen = self.cf_model_seen.setdefault(
                        axis_mode, {}).setdefault(kind, {})
                    self._reservoir_add_field(samples, seen, cell_ids, W, field_index)
            self._reservoir_add_field(self.bs_samples, self.bs_seen,
                                      bright_bin * NB + std_bin, W, field_index)

    def pred_label(self) -> str:
        """Which point estimate's error the std-vs-error histogram measures —
        ``"combiner"`` only when EVERY field was scored against it, else
        ``"ensemble mean"`` (a mixed run degrades to the mean's label)."""
        return ("combiner" if self.n_fields > 0
                and self.n_pred_combiner == self.n_fields else "ensemble mean")

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

    def std_err_block(self, hist: np.ndarray, *, n_fields: int) -> dict:
        """Serialize one point estimate's error-vs-disagreement histogram."""
        cen = 10.0 ** (0.5 * (self.log_edges[:-1] + self.log_edges[1:]))
        med = np.full(LOG_E_BINS, np.nan)
        for i, row in enumerate(np.asarray(hist, np.float64)):
            total = row.sum()
            if total > 0:
                med[i] = cen[int(np.searchsorted(np.cumsum(row), .5 * total))]
        def _l(a):
            return [None if not np.isfinite(v) else float(v)
                    for v in np.asarray(a, float)]
        return {"edges": _l(self.log_edges),
                "hist": np.asarray(hist, int).tolist(),
                "med_std": _l(np.log10(cen)), "med_err": _l(np.log10(med)),
                "n_fields": int(n_fields)}

    def combiner_feature_error_blocks(self) -> dict:
        """Serialize every model error over both model-independent planes."""
        axes = {}
        all_medians = []
        error_centers = 0.5 * (self.log_edges[:-1] + self.log_edges[1:])
        for axis_mode, model_hists in self.h_combiner_feature_err.items():
            meta = self.combiner_feature_meta[axis_mode]
            models = {}
            for kind, hist in model_hists.items():
                totals = hist.sum(axis=-1)
                cumulative = np.cumsum(hist, axis=-1)
                median_idx = np.argmax(
                    cumulative >= (0.5 * totals)[..., None], axis=-1)
                med = error_centers[median_idx].astype(float)
                med[totals <= 0] = np.nan
                all_medians.extend(med[np.isfinite(med)].tolist())
                models[kind] = {
                    "median_log_error": np.where(
                        np.isfinite(med), med, None).tolist(),
                    "counts": totals.astype(int).tolist(),
                }
            axes[axis_mode] = {
                "axis_names": list(meta["axis_names"]),
                "edges": [[float(v) for v in edge] for edge in meta["edges"]],
                "models": models,
            }
        if all_medians:
            color_range = [float(np.floor(min(all_medians))),
                           float(np.ceil(max(all_medians)))]
            if color_range[1] <= color_range[0]:
                color_range[1] = color_range[0] + 1.0
        else:
            color_range = [float(self.log_edges[0]), float(self.log_edges[-1])]
        return {"axes": axes, "color_range": color_range,
                "error_unit": "log10_electrons"}

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
                "adaptive_range": True,
                "edges": _l(self.log_edges),           # log10 e⁻, both axes
                "hist": self.h_std_err.astype(int).tolist(),
                "med_std": _l(np.log10(cen)),
                "med_err": _l(np.log10(med)),
                "pred": self.pred_label(),             # "combiner" | "ensemble mean"
                "primary": ("rbf_gate" if self.pred_label() == "combiner"
                            else "ensemble_mean"),
                "models": {kind: self.std_err_block(hist, n_fields=self.std_err_model_fields.get(kind, 0))
                           for kind, hist in self.h_std_err_models.items()},
            },
            "combiner_feature_error": self.combiner_feature_error_blocks(),
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

        def enc(res: dict, width: int = NB) -> dict:
            return {f"{key // width},{key % width}":
                    [[int(f), int(y), int(x)] for f, y, x in v]
                    for key, v in res.items()}

        return {
            "n_fields": int(self.n_fields),
            "n_members": int(self.n_members),
            "sample_k": int(self.sample_k),
            "std_err": enc(self.se_samples),
            "std_err_models": {kind: enc(samples)
                               for kind, samples in self.se_model_samples.items()},
            "combiner_feature_error": {
                axis_mode: {kind: enc(samples, FEATURE_BINS)
                            for kind, samples in model_samples.items()}
                for axis_mode, model_samples in self.cf_model_samples.items()
            },
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

    pred = acc.pred_label()
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    m = _density_panel(ax, acc.h_std_err, acc.log_edges, acc.log_edges)
    cen, med = acc.binned_median_err()
    ax.plot(cen, med, "-o", ms=3, lw=2.0, color="#d6604d",
            label="median |error| per std bin")
    lim = 10.0 ** np.array([acc.log_edges[0], acc.log_edges[-1]])
    ax.plot(lim, lim, ls="--", color="#333", lw=1.2, label="|error| = std")
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xlabel("cross-member per-pixel std  σ  [e⁻]")
    ax.set_ylabel(f"actual error  |{pred} − HR|  [e⁻]")
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
    ax.set_ylim(*(10.0 ** np.array([acc.log_edges[0], acc.log_edges[-1]])))
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
