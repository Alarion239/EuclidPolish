"""Pixel types by knee use: which kinds of pixel a multi-knee model reads
through which knees, from a knee-attribution run's samples.

Each sample carries its own band's local knee split ``s_k`` (the 3×3 LR block
under the pixel, :mod:`euclid_polish.eval.knee_attribution`). Three views:

* **Effective knee** ``log10 q̄ = Σ_k s_k log10 q_k`` (electrons): the knee
  the pixel is read through, and ``exp(H(s))``, how many knees it uses.
* **Chain-rule baseline.** A knee term is ``g_k / √(q_k² + x²)``: even a
  network with the same gradient ``g`` on every knee would read a pixel
  through the knees below its block's brightness. :func:`mechanical_share`
  is that share, from the LR block alone; the effective knee minus the
  baseline's is what the network itself prefers.
* **Brightness-controlled groups.** The effective knee follows the pixel's
  own brightness, so within each decade of brightness (in noise σ of the
  band, :data:`SIGMA_EDGES`) the samples
  split by their effective knee into ``lower`` (the bottom
  :data:`GROUP_QUANTILES` of the bin, by weight), ``usual`` and ``higher``:
  what is different about a pixel read through a lower or higher knee than
  pixels as bright as it.
* **Knee types.** Weighted k-means of the centred log-ratio of ``s`` (the
  geometry of shares), per band, ordered by effective knee: the recurring
  knee profiles, whatever the brightness. Empty sky is most of any field, so
  the clustering weighs every brightness decade alike
  (:func:`balanced_weights`); prevalences use the true weights.

Each group or type becomes an "average pixel": the median of its stamps
(:func:`cut_stamps`), each turned by a dihedral move so its brighter side
points the same way (:func:`alignment`), where sources sit around it
(:func:`occupancy`) and a few stamp measures (:func:`stamp_features`). Pure numerics, no I/O or plotting
(``scripts/knee_pixel_types.py`` reads the run and draws).
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np

#: Brightness bins of the groups, in noise σ: decades from 1e-3 to 1e5 σ
#: (values below 1e-2 σ, empty sky included, go to the first bin, from 1e4 σ
#: on to the last).
SIGMA_EDGES: np.ndarray = np.logspace(-3.0, 5.0, 9)
#: The groups within a brightness bin, by effective knee.
GROUPS: tuple[str, ...] = ("lower", "usual", "higher")
#: Weighted quantiles of the effective knee splitting a bin into the groups.
GROUP_QUANTILES: tuple[float, float] = (0.2, 0.8)
#: Floor of a share before its log (a knee with share 0).
CLR_FLOOR = 1e-4
#: Radius (HR pixels) of the neighbourhood whose peak :func:`stamp_features`
#: compares with the centre.
PEAK_RADIUS = 4
#: Annulus (HR pixels) of the local background in :func:`stamp_features`.
ANNULUS = (6.0, 10.0)
#: A "source" pixel in :func:`occupancy` and :func:`stamp_features`: above
#: this many σ.
SOURCE_SIGMA = 1.0


def effective_knee(share: np.ndarray, knees: Sequence[float]) -> np.ndarray:
    """``Σ_k s_k log10 q_k`` per row of ``share`` ``(N, K)``, normalised by
    the row's sum; NaN where the row is all 0 (no gradient)."""
    s = np.asarray(share, np.float64)
    total = s.sum(axis=1)
    logq = np.log10(np.asarray(knees, np.float64))
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (s @ logq) / total
    return np.where(total > 0, out, np.nan)


def mechanical_share(blocks_e: np.ndarray, knees: Sequence[float]) -> np.ndarray:
    """The local knee share a network with the same gradient on every
    knee's channel would have: ``∝ Σ_u 1/√(q_k² + x_u²)`` over the LR
    block's pixels ``x_u`` (``blocks_e`` ``(N, U)`` electrons, NaN for a
    pixel beyond the field, left out) → ``(N, K)``, rows summing to 1."""
    x = np.asarray(blocks_e, np.float64)
    q = np.asarray(knees, np.float64)
    terms = 1.0 / np.sqrt(q[None, None, :] ** 2 + x[:, :, None] ** 2)        # (N, U, K)
    terms = np.where(np.isfinite(x)[:, :, None], terms, 0.0).sum(axis=1)
    return terms / terms.sum(axis=1, keepdims=True)


def knee_count(share: np.ndarray) -> np.ndarray:
    """Effective number of knees a row uses, ``exp(−Σ s log s)``: 1 for one
    knee, K for an even split; NaN for an all-0 row."""
    s = np.asarray(share, np.float64)
    total = s.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = s / total
        h = -np.where(p > 0, p * np.log(np.where(p > 0, p, 1.0)), 0.0).sum(axis=1)
    return np.where(total[:, 0] > 0, np.exp(h), np.nan)


def weighted_quantile(values: np.ndarray, weights: np.ndarray, qs: Sequence[float]) -> np.ndarray:
    """Quantiles of ``values`` under ``weights`` (the midpoint rule of the
    cumulative weight, interpolated); NaN when nothing has weight."""
    v = np.asarray(values, np.float64).reshape(-1)
    w = np.asarray(weights, np.float64).reshape(-1)
    keep = np.isfinite(v) & (w > 0)
    v, w = v[keep], w[keep]
    if not v.size:
        return np.full(len(qs), np.nan)
    order = np.argsort(v, kind="stable")
    v, w = v[order], w[order]
    cum = (np.cumsum(w) - 0.5 * w) / w.sum()
    return np.interp(np.asarray(qs, np.float64), cum, v)


def brightness_bin(value_e: np.ndarray, edges: np.ndarray = SIGMA_EDGES) -> np.ndarray:
    """The bin of each value (below the first edge → 0, from the last edge
    on → the last bin), −1 for a non-finite value."""
    v = np.asarray(value_e, np.float64)
    finite = np.isfinite(v)
    bins = np.clip(np.searchsorted(edges, np.where(finite, v, 0.0), side="right") - 1,
                   0, len(edges) - 2)
    return np.where(finite, bins, -1)


def residual_groups(log_q: np.ndarray, value_e: np.ndarray, weight: np.ndarray,
                    edges: np.ndarray = SIGMA_EDGES,
                    quantiles: tuple[float, float] = GROUP_QUANTILES
                    ) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Split samples (of one band) by effective knee within their brightness
    bin: ``(bins, groups, stats)``. ``groups`` indexes :data:`GROUPS` (−1
    for a sample without an effective knee or bin): below the bin's lower
    weighted quantile → ``lower``, above its upper one → ``higher``, else
    ``usual``. ``stats`` per bin: ``n``, ``weight`` and the weighted
    ``q20``/``median``/``q80`` of ``log_q`` (``None`` for an empty bin).
    Ties at a cut point stay ``usual``, so a bin read through one knee only
    has no lower or higher group."""
    log_q = np.asarray(log_q, np.float64)
    weight = np.asarray(weight, np.float64)
    bins = brightness_bin(value_e, edges)
    groups = np.full(log_q.size, -1, np.int64)
    stats: list[dict] = []
    for b in range(len(edges) - 1):
        sel = (bins == b) & np.isfinite(log_q)
        if not sel.any():
            stats.append({"n": 0, "weight": 0.0, "q20": None, "median": None, "q80": None})
            continue
        lo, med, hi = weighted_quantile(log_q[sel], weight[sel], (quantiles[0], 0.5, quantiles[1]))
        g = np.where(log_q[sel] < lo, 0, np.where(log_q[sel] > hi, 2, 1))
        groups[sel] = g
        stats.append({"n": int(sel.sum()), "weight": float(weight[sel].sum()),
                      "q20": float(lo), "median": float(med), "q80": float(hi)})
    return bins, groups, stats


def balanced_weights(bins: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Weights rescaled so every occupied bin carries the same total (each
    bin's weights keep their proportions); 0 for bin −1."""
    bins = np.asarray(bins)
    weight = np.asarray(weight, np.float64)
    out = np.zeros_like(weight)
    for b in np.unique(bins[bins >= 0]):
        sel = bins == b
        total = weight[sel].sum()
        if total > 0:
            out[sel] = weight[sel] / total
    return out


def clr(share: np.ndarray, floor: float = CLR_FLOOR) -> np.ndarray:
    """Centred log-ratio of each share row (floored at ``floor``, then
    renormalised): the Euclidean geometry for compositions."""
    s = np.maximum(np.asarray(share, np.float64), floor)
    s = s / s.sum(axis=1, keepdims=True)
    log = np.log(s)
    return log - log.mean(axis=1, keepdims=True)


def kmeans(points: np.ndarray, weights: np.ndarray, k: int, rng: np.random.Generator, *,
           restarts: int = 4, iterations: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Weighted k-means (k-means++ seeding, Lloyd steps), the best of
    ``restarts`` by weighted inertia: ``(labels (N,), centres (k, D))``.
    ``k`` is capped at the number of distinct points."""
    x = np.asarray(points, np.float64)
    w = np.asarray(weights, np.float64)
    n = x.shape[0]
    k = int(min(k, len(np.unique(x, axis=0)))) if n else 0
    if k < 1:
        return np.zeros(n, np.int64), np.zeros((0, x.shape[1] if x.ndim == 2 else 0))
    best = (np.inf, None, None)
    for _ in range(max(1, int(restarts))):
        centres = x[[rng.choice(n, p=w / w.sum())]]
        while len(centres) < k:
            d2 = ((x[:, None, :] - centres[None]) ** 2).sum(-1).min(axis=1) * w
            if d2.sum() <= 0:
                break
            centres = np.vstack([centres, x[rng.choice(n, p=d2 / d2.sum())]])
        for _ in range(int(iterations)):
            labels = ((x[:, None, :] - centres[None]) ** 2).sum(-1).argmin(axis=1)
            moved = centres.copy()
            for j in range(len(centres)):
                mine = labels == j
                if w[mine].sum() > 0:
                    moved[j] = np.average(x[mine], axis=0, weights=w[mine])
            if np.allclose(moved, centres):
                break
            centres = moved
        labels = ((x[:, None, :] - centres[None]) ** 2).sum(-1).argmin(axis=1)
        inertia = float((((x - centres[labels]) ** 2).sum(-1) * w).sum())
        if inertia < best[0]:
            best = (inertia, labels, centres)
    return best[1], best[2]


def knee_types(share: np.ndarray, weight: np.ndarray, knees: Sequence[float], k: int,
               rng: np.random.Generator) -> np.ndarray:
    """Type of each share row: k-means of :func:`clr`, renumbered by the
    types' weighted mean effective knee (type 0 reads through the lowest
    knee); −1 for an all-0 row."""
    share = np.asarray(share, np.float64)
    weight = np.asarray(weight, np.float64)
    ok = share.sum(axis=1) > 0
    labels = np.full(share.shape[0], -1, np.int64)
    if not ok.any():
        return labels
    found, _ = kmeans(clr(share[ok]), weight[ok], k, rng)
    log_q = effective_knee(share[ok], knees)
    present = np.unique(found)
    order = sorted(present, key=lambda j: np.average(log_q[found == j], weights=weight[ok][found == j]))
    renumber = {int(j): i for i, j in enumerate(order)}
    labels[ok] = [renumber[int(j)] for j in found]
    return labels


def cut_stamps(plane: np.ndarray, y: np.ndarray, x: np.ndarray, side: int) -> np.ndarray:
    """``side × side`` stamps (odd ``side``) of a 2-D ``plane`` centred on
    each ``(y, x)``, NaN beyond the plane: ``(N, side, side)`` float32."""
    if side % 2 != 1:
        raise ValueError(f"a stamp side must be odd, got {side}")
    plane = np.asarray(plane, np.float32)
    r = side // 2
    padded = np.pad(plane, r, constant_values=np.nan)
    y = np.asarray(y, np.int64).reshape(-1) + r
    x = np.asarray(x, np.int64).reshape(-1) + r
    offs = np.arange(-r, r + 1)
    return padded[(y[:, None] + offs)[:, :, None], (x[:, None] + offs)[:, None, :]]


def alignment(stamps: np.ndarray) -> np.ndarray:
    """The dihedral move of each square stamp that points the centroid of its
    excess over its median along +row, then +column: ``(N, 3)`` bool flags
    ``(transpose, flip rows, flip columns)``, applied in that order by
    :func:`apply_alignment`. With the brighter side of every stamp on the
    same side, a median stack keeps one-sided structure (an edge, a
    neighbour) instead of averaging it into a ring."""
    s = np.asarray(stamps, np.float64)
    n, h, w = s.shape
    if h != w:
        raise ValueError(f"stamps must be square, got {h}×{w}")
    yy, xx = np.mgrid[:h, :w]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # an all-NaN stamp stays put
        median = np.nanmedian(s.reshape(n, -1), axis=1)
    excess = np.nan_to_num(np.maximum(s - median[:, None, None], 0.0))
    dy = (excess * (yy - h // 2)).sum(axis=(1, 2))
    dx = (excess * (xx - w // 2)).sum(axis=(1, 2))
    transpose = np.abs(dx) > np.abs(dy)
    dy, dx = np.where(transpose, dx, dy), np.where(transpose, dy, dx)
    return np.stack([transpose, dy < 0, dx < 0], axis=1)


def apply_alignment(stamps: np.ndarray, moves: np.ndarray) -> np.ndarray:
    """Apply :func:`alignment`'s moves (to the stamps they came from, or to
    other stamps of the same pixels: the truth, the LR)."""
    s = np.asarray(stamps)
    moves = np.asarray(moves, bool)
    out = np.where(moves[:, 0, None, None], s.transpose(0, 2, 1), s)
    out = np.where(moves[:, 1, None, None], out[:, ::-1, :], out)
    return np.where(moves[:, 2, None, None], out[:, :, ::-1], out)


def align(stamps: np.ndarray) -> np.ndarray:
    """Each stamp turned by its own :func:`alignment`: brighter side along
    +row (up when plotted with ``origin="lower"``), then +column."""
    return apply_alignment(stamps, alignment(stamps))


def stack(stamps: np.ndarray) -> np.ndarray:
    """The median stamp (NaN-aware; NaN where every stamp is NaN)."""
    s = np.asarray(stamps, np.float32)
    if not s.shape[0]:
        return np.full(s.shape[1:], np.nan, np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # all-NaN pixels stay NaN
        return np.nanmedian(np.where(np.isfinite(s), s, np.nan), axis=0).astype(np.float32)


def occupancy(stamps: np.ndarray, threshold: float = SOURCE_SIGMA) -> np.ndarray:
    """Fraction of the stamps above ``threshold`` at each position (NaN
    positions left out): where sources sit around the pixels."""
    s = np.asarray(stamps, np.float64)
    finite = np.isfinite(s)
    with np.errstate(invalid="ignore", divide="ignore"):
        return ((np.where(finite, s, -np.inf) > threshold).sum(axis=0) / finite.sum(axis=0)
                ).astype(np.float32)


def stamp_features(stamps: np.ndarray, threshold: float = SOURCE_SIGMA) -> dict[str, np.ndarray]:
    """Measures of each stamp (in any unit, e.g. noise σ) around its centre:

    * ``centre``: the centre value;
    * ``peak_ratio``: the brightest value within :data:`PEAK_RADIUS` over
      ``max(centre, 1)``: above 1 on the flank of something brighter;
    * ``local_max``: 1 when the centre is at least each of its 8 neighbours;
    * ``slope``: the centre's gradient magnitude (central differences);
    * ``contrast``: centre minus the median of the :data:`ANNULUS`;
    * ``nearest_source``: distance (pixels) from the centre to the nearest
      pixel above ``threshold`` (0 when the centre is one; the stamp's
      half-diagonal + 1 when none is);
    * ``stamp_peak``: the stamp's brightest value.
    """
    s = np.asarray(stamps, np.float64)
    n, h, w = s.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[:h, :w]
    r = np.hypot(yy - cy, xx - cx)
    centre = s[:, cy, cx]
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)          # all-NaN stamps give NaN
        near = np.nanmax(np.where(r <= PEAK_RADIUS, s, np.nan), axis=(1, 2))
        neighbours = s[:, cy - 1:cy + 2, cx - 1:cx + 2].reshape(n, 9)
        local_max = (centre >= np.nanmax(np.delete(neighbours, 4, axis=1), axis=1)).astype(np.float64)
        gy = 0.5 * (s[:, cy + 1, cx] - s[:, cy - 1, cx])
        gx = 0.5 * (s[:, cy, cx + 1] - s[:, cy, cx - 1])
        ring = (r >= ANNULUS[0]) & (r <= ANNULUS[1])
        background = np.nanmedian(np.where(ring, s, np.nan).reshape(n, -1), axis=1)
        source = np.where(np.isfinite(s), s, -np.inf) > threshold
        nearest = np.where(source.any(axis=(1, 2)),
                           np.where(source, r[None], np.inf).min(axis=(1, 2)), r.max() + 1.0)
        return {"centre": centre, "peak_ratio": near / np.maximum(centre, 1.0),
                "local_max": local_max, "slope": np.hypot(gy, gx),
                "contrast": centre - background, "nearest_source": nearest,
                "stamp_peak": np.nanmax(s.reshape(n, -1), axis=1)}


def weighted_mix(labels: np.ndarray, weights: np.ndarray, names: Sequence[str]) -> dict[str, float]:
    """Weighted share of each label (names indexed by label), largest first."""
    labels = np.asarray(labels)
    weights = np.asarray(weights, np.float64)
    total = weights.sum()
    if total <= 0:
        return {}
    mix = {str(name): float(weights[labels == name].sum() / total) for name in names}
    return dict(sorted(((k, v) for k, v in mix.items() if v > 0), key=lambda kv: -kv[1]))
