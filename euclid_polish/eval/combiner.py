"""Deep-ensemble **combiner** — a tiny per-band brightness-gated mixture.

For the STARFULL regime the reconstruction is normally the naive per-pixel mean
of the ``M`` ensemble members. That discards their complementary strengths
(L1-trained members render faint galaxies cleanly but erase star cores;
L2-trained members keep the point sources but leave a speckle floor). This
module learns a tiny **per-band** model that fuses the members — leaning on the
right member at the right brightness.

Design (RBF brightness gate — replaced the earlier skip-MLP, which couldn't
route faint pixels to the L1 members: near zero it collapsed to a fixed linear
blend that inherited the L2 floor, and its tanh gate modulated strongest at
faint, i.e. the wrong direction):

* **Per-band gated convex mixture.** One model per output band (VIS, Y_E, J_E,
  H_E). At each pixel a **brightness scalar** ``b`` (the max over members, asinh)
  is expanded in ``K`` fixed **RBF kernels**, mapped to per-member logits and
  soft-maxed to weights ``w(b)`` (≥0, sum to 1); the output is the convex
  combination ``Σ wₘ(b)·xₘ``. Localized kernels give an ARBITRARY weight-vs-
  brightness profile — faint bins can put ~all weight on the L1 members (killing
  the L2 floor exactly, since it's a convex combination) while bright bins pick
  the star-reproducing members. The weight curves ARE the model (see
  :meth:`Combiner.effective_weights`). No saturation issue: a convex combo of
  members never has to extrapolate.
* **asinh space.** ``arcsinh(electrons / STRETCH_SCALE_E)`` in; output inverse-
  stretched with ``sinh(·)·scale`` after the same ±clip the inference path uses.
* **L1 loss** (asinh-space ``mean|·|``), fit LOCALLY on the ``validate`` split.
* **Optional post-hoc pruning.** After fitting, a member's **importance** (mean
  gate weight over the brightness sweep) is summed across the 4 bands — the total
  bar in the importance chart — and if it falls below ``min_usage`` the member is
  dropped from ALL bands (masked + renormalised); otherwise kept in all. Default
  0 (keep all). Whole-member + importance-based (not per-band, not pixel-
  frequency): it keeps rare-but-critical specialists (the L2 star members) and
  drops only members unimportant across the whole gate.
* **Starfull-only** (the ``hr_`` target — starless members merely erase stars).

Persistence: ``<dir>/combiner/combiner.npz`` + ``combiner.json``.
:func:`load_combiner` returns ``None`` when the saved member labels no longer
match the active ensemble (stale) or the format is incompatible.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np

from euclid_polish.config import Config

#: Per-band asinh knee (electrons). All HR bands use STRETCH_SCALE_E today, but
#: honour the per-band config so a future change tracks automatically.
_BAND_SCALE = {name: float(Config.get_band(name).asinh_stretch_scale_e)
               for name in Config.HR_TARGET_BAND_NAMES}

#: Clip on the asinh output before ``sinh`` (matches training/inference.py).
SINH_CLIP = float(getattr(Config, "SINH_STRETCH_CLIP", 20.0))

#: Brightness (asinh) sweep the RBF kernels tile — faint sky (~0) to a bright
#: star core (asinh of a very bright source).
GATE_LEVEL_RANGE = (-1.0, 13.0)
DEFAULT_N_KERNELS = 12
DEFAULT_SIGMA_SCALE = 1.0

BAND_NAMES = tuple(Config.HR_TARGET_BAND_NAMES)


def _band_scale(name: str) -> float:
    return _BAND_SCALE.get(name, float(Config.STRETCH_SCALE_E))


def _brightness(X: np.ndarray) -> np.ndarray:
    """Per-pixel brightness scalar from the member vector (asinh): the max over
    members — 'is there a bright source here'. Sharpens star (all-high) vs floor
    (one member's small speckle) far better than the mean."""
    return np.max(np.asarray(X, np.float64), axis=1)


def _rbf(b: np.ndarray, centers: np.ndarray, sigma: float) -> np.ndarray:
    z = (np.asarray(b, np.float64)[:, None] - np.asarray(centers)[None, :]) / sigma
    return np.exp(-0.5 * z * z)


def _softmax_masked(logits: np.ndarray, surviving: np.ndarray) -> np.ndarray:
    logits = np.where(surviving[None, :], logits, -1e9)
    logits = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

@dataclass
class BandCombiner:
    """One band's RBF brightness gate: ``(N, M) asinh members → (N,) asinh``.

    ``b = max_m x`` → ``φ = rbf(b)`` (K kernels) → ``w = softmax(φ·V + a)`` →
    ``y = Σ wₘ·xₘ`` (convex combination).
    """

    V: np.ndarray                  # (K, M)
    a: np.ndarray                  # (M,)
    centers: np.ndarray            # (K,)
    sigma: float
    surviving: np.ndarray          # (M,) bool

    def weights(self, X: np.ndarray) -> np.ndarray:
        """Per-pixel member weights ``(N, M)`` for a member stack ``(N, M)``.

        The brightness scalar is the max over the SURVIVING members only, so
        pruned members influence neither the gate nor the convex sum (weight 0):
        their SR is genuinely unused and need never be computed. With no pruning
        (all surviving) this is identical to the max over all members."""
        X = np.asarray(X, np.float64)
        surv = np.asarray(self.surviving, bool)
        b = np.max(X[:, surv] if surv.any() else X, axis=1)
        logits = _rbf(b, self.centers, self.sigma) @ self.V + self.a
        return _softmax_masked(logits, self.surviving)

    def forward_asinh(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, np.float64)
        return np.sum(self.weights(X) * X, axis=1)

    def weights_at(self, b_levels: np.ndarray) -> np.ndarray:
        """Gate weights ``(L, M)`` at given brightness levels (asinh)."""
        logits = _rbf(b_levels, self.centers, self.sigma) @ self.V + self.a
        return _softmax_masked(logits, self.surviving)


@dataclass
class Combiner:
    """The 4-band combiner + metadata to apply/persist/validate it."""

    member_labels: list[str]
    n_kernels: int
    sigma_scale: float
    min_usage: float
    bands: dict[str, BandCombiner]
    band_names: tuple[str, ...] = BAND_NAMES
    level_range: tuple[float, float] = GATE_LEVEL_RANGE
    records_fp: str | None = None
    starfull: bool = True
    val_l1: float | None = None
    kind: str = "rbf_gate"
    fit_meta: dict = field(default_factory=dict)

    # -- inference -- #
    def apply_field(self, preds: np.ndarray,
                    band_names: tuple[str, ...] | None = None) -> np.ndarray:
        """Combine a member stack ``(M,H,W,C)`` (electrons) → ``(H,W,C)``."""
        preds = np.asarray(preds, np.float32)
        if preds.ndim != 4:
            raise ValueError(f"expected (M,H,W,C) member stack, got {preds.shape}")
        m, h, w, c = preds.shape
        names = tuple(band_names) if band_names is not None else self.band_names
        out = np.empty((h, w, c), np.float32)
        for ci in range(c):
            name = names[ci]
            bc = self.bands[name]
            scale = _band_scale(name)
            x = np.arcsinh(preds[..., ci].reshape(m, h * w).T / scale)   # (HW, M)
            y = np.clip(bc.forward_asinh(x), -SINH_CLIP, SINH_CLIP)
            out[..., ci] = (np.sinh(y) * scale).reshape(h, w).astype(np.float32)
        return out

    def needed_member_indices(self) -> list[int]:
        """Indices of the members actually used by the gate (surviving in any
        band). Pruned members contribute nothing and don't need to be run."""
        m = len(self.member_labels)
        mask = np.zeros(m, bool)
        for bc in self.bands.values():
            s = np.asarray(bc.surviving, bool)
            mask[:len(s)] |= s
        return [int(i) for i in np.where(mask)[0]]

    def upsample(self, ens, lr_array: np.ndarray) -> np.ndarray:
        """Combine one LR field, running ONLY the surviving members — the pruned
        members' super-resolutions are never computed. Returns ``(H,W,C)``
        electrons. ``ens`` is an ``EnsembleModel`` whose member order matches
        this combiner's ``member_labels``."""
        keep = self.needed_member_indices()
        if not keep:
            raise RuntimeError("combiner has no surviving members")
        m = len(self.member_labels)
        preds = None
        for i in keep:
            sr = np.asarray(ens._models[i].upsample_array(lr_array), np.float32)
            if preds is None:
                preds = np.zeros((m, *sr.shape), np.float32)   # pruned slots stay 0
            preds[i] = sr
        return self.apply_field(preds)

    # -- interpretability -- #
    def effective_weights(self, band: str, *, n_levels: int = 25,
                          level_range: tuple[float, float] | None = None) -> dict:
        """The gate weight of each member vs pixel brightness — the model made
        visible. Weights are ≥0 and sum to 1 at every brightness."""
        bc = self.bands[band]
        lr = level_range or self.level_range
        levels = np.linspace(lr[0], lr[1], int(n_levels))
        scale = _band_scale(band)
        return {"brightness_asinh": levels,
                "brightness_e": np.sinh(levels) * scale,
                "jacobian": bc.weights_at(levels)}       # (L, M), rows sum to 1

    def surviving_members(self) -> dict[str, list[bool]]:
        return {b: self.bands[b].surviving.astype(bool).tolist()
                for b in self.bands}


# ---------------------------------------------------------------------------
# Fit-data assembly (unchanged — streaming, brightness-stratified)
# ---------------------------------------------------------------------------

class FitBufferAccumulator:
    """Streaming per-band fit-buffer builder. ``add(preds, hr)`` one field at a
    time — so the caller never retains the full (multi-GB) member stack — pixel-
    subsampling **stratified by brightness** (equal quota per asinh-brightness
    bin) so faint structure isn't drowned by the dominant sky. ``buffers()``
    returns ``{band: (X(N,M), y(N,))}`` in asinh space."""

    def __init__(self, band_names, *, max_rows: int = 3_000_000,
                 n_bright_bins: int = 8, per_bin_per_field: int = 2000,
                 level_range: tuple[float, float] = (-1.0, 12.0), seed: int = 0):
        self.band_names = tuple(band_names)
        self.max_rows = int(max_rows)
        self.n_bright_bins = int(n_bright_bins)
        self.per_bin_per_field = int(per_bin_per_field)
        self.edges = np.linspace(level_range[0], level_range[1],
                                 int(n_bright_bins) + 1)
        self._rng = np.random.default_rng(seed)
        self._X = {b: [] for b in self.band_names}
        self._y = {b: [] for b in self.band_names}
        self._n = {b: 0 for b in self.band_names}

    def add(self, preds: np.ndarray, hr: np.ndarray) -> None:
        preds = np.asarray(preds, np.float32)
        hr = np.asarray(hr, np.float32)
        m = preds.shape[0]
        for ci, name in enumerate(self.band_names):
            if self._n[name] >= self.max_rows:
                continue
            scale = _band_scale(name)
            xs = np.arcsinh(preds[..., ci].reshape(m, -1).T / scale)   # (P, M)
            ys = np.arcsinh(hr[..., ci].reshape(-1) / scale)           # (P,)
            bin_idx = np.clip(np.digitize(ys, self.edges) - 1,
                              0, self.n_bright_bins - 1)
            for b in range(self.n_bright_bins):
                if self._n[name] >= self.max_rows:
                    break
                sel = np.where(bin_idx == b)[0]
                if sel.size == 0:
                    continue
                take = int(min(self.per_bin_per_field, sel.size,
                               self.max_rows - self._n[name]))
                pick = self._rng.choice(sel, size=take, replace=False)
                self._X[name].append(xs[pick])
                self._y[name].append(ys[pick])
                self._n[name] += take

    def buffers(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name in self.band_names:
            if self._X[name]:
                out[name] = (np.concatenate(self._X[name]).astype(np.float32),
                             np.concatenate(self._y[name]).astype(np.float32))
            else:
                out[name] = (np.zeros((0, 0), np.float32), np.zeros((0,), np.float32))
        return out


def build_fit_buffers_from_fields(field_iter, band_names, **kw
                                  ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Convenience: drive a :class:`FitBufferAccumulator` from an iterator of
    ``(preds(M,H,W,C), hr(H,W,C))`` fields (electrons)."""
    acc = FitBufferAccumulator(band_names, **kw)
    for preds, hr in field_iter:
        acc.add(preds, hr)
    return acc.buffers()


def build_fit_buffers(base_dir: str, records_dir: str, *, subset: str = "validate",
                      num_images: int = 100, ens=None, on_progress=None,
                      **kw) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], list[str]]:
    """Records-backed convenience: run each ``validate`` field through the
    STARFULL ensemble and assemble the per-band fit buffers. Returns
    ``(buffers, member_labels)``."""
    from euclid_polish.ensemble import EnsembleModel
    from euclid_polish.image.collection import ImageSet
    from euclid_polish.image.tfio import tfrecord_path

    ens = ens or EnsembleModel(base_dir, starless=False)
    labels = list(ens.member_labels)
    lr_list = list(ImageSet.read(tfrecord_path(records_dir, f"dirty_{subset}"),
                                 num_images=num_images))
    hr_by = {h.index: h for h in ImageSet.read(
        tfrecord_path(records_dir, f"hr_{subset}"), num_images=num_images)}

    def _iter():
        n = len(lr_list)
        for i, lr in enumerate(lr_list):
            hr = hr_by.get(lr.index)
            if hr is None:
                continue
            preds = ens.member_arrays(lr.data)
            if on_progress is not None:
                on_progress(i + 1, n, f"field {lr.index}")
            yield preds, np.asarray(hr.data, np.float32)

    buffers = build_fit_buffers_from_fields(_iter(), BAND_NAMES, **kw)
    return buffers, labels


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------

def _fit_one_band(X: np.ndarray, y: np.ndarray, *, n_kernels: int,
                  sigma_scale: float, level_range: tuple[float, float],
                  steps: int, lr: float, batch: int, seed: int, holdout: float
                  ) -> tuple[BandCombiner, np.ndarray, np.ndarray]:
    import tensorflow as tf

    X = np.asarray(X, np.float32)
    y = np.asarray(y, np.float32)
    n, m = X.shape
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    X, y = X[order], y[order]
    n_val = int(n * holdout)
    Xtr, ytr = (X[n_val:], y[n_val:]) if n_val > 0 else (X, y)
    Xval, yval = (X[:n_val], y[:n_val]) if n_val > 0 else (X, y)

    centers = np.linspace(level_range[0], level_range[1],
                          int(n_kernels)).astype(np.float32)
    sigma = float((level_range[1] - level_range[0])
                  / max(int(n_kernels) - 1, 1) * float(sigma_scale))

    tf.random.set_seed(seed)
    V = tf.Variable(tf.zeros((int(n_kernels), m), tf.float32))   # start uniform
    a = tf.Variable(tf.zeros((m,), tf.float32))
    cen = tf.constant(centers)
    opt = tf.keras.optimizers.Adam(lr)

    def _weights(Xb):
        b = tf.reduce_max(Xb, axis=1, keepdims=True)            # (n, 1)
        phi = tf.exp(-0.5 * ((b - cen) / sigma) ** 2)           # (n, K)
        return tf.nn.softmax(tf.matmul(phi, V) + a, axis=1)     # (n, M)

    def _forward(Xb):
        return tf.reduce_sum(_weights(Xb) * Xb, axis=1)

    @tf.function
    def train_step(xb, yb):
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(tf.abs(_forward(xb) - yb))
        grads = tape.gradient(loss, [V, a])
        opt.apply_gradients(zip(grads, [V, a]))

    bs = int(min(batch, max(1, len(Xtr))))
    ds = (tf.data.Dataset.from_tensor_slices((Xtr, ytr))
          .shuffle(min(len(Xtr), 100_000), seed=seed, reshuffle_each_iteration=True)
          .batch(bs).repeat())
    it = iter(ds)

    def _val_l1():
        return float(np.mean(np.abs(_forward(tf.constant(Xval)).numpy() - yval)))

    eval_every = max(1, int(steps) // 20)
    patience = 5
    best = np.inf
    best_w = None
    stale = 0
    for s in range(int(steps)):
        xb, yb = next(it)
        train_step(xb, yb)
        if (s + 1) % eval_every == 0 or s == int(steps) - 1:
            v = _val_l1()
            if v < best - 1e-6:
                best, stale = v, 0
                best_w = [V.numpy().copy(), a.numpy().copy()]
            else:
                stale += 1
                if stale >= patience:
                    break
    if best_w is None:
        best_w = [V.numpy().copy(), a.numpy().copy()]
    Vn, an = best_w
    # No pruning here — the survivor mask is a CUMULATIVE (cross-band) decision
    # made once in fit_combiner. Return the holdout so the final (post-prune)
    # loss can be measured there.
    bc = BandCombiner(V=Vn.astype(np.float32), a=an.astype(np.float32),
                      centers=centers, sigma=sigma, surviving=np.ones(m, bool))
    return bc, Xval, yval


def fit_combiner(buffers, member_labels, *, n_kernels: int = DEFAULT_N_KERNELS,
                 sigma_scale: float = DEFAULT_SIGMA_SCALE, min_usage: float = 0.0,
                 level_range: tuple[float, float] = GATE_LEVEL_RANGE,
                 steps: int = 3000, lr: float = 1e-2, batch: int = 16384,
                 seed: int = 0, holdout: float = 0.1) -> Combiner:
    """Fit one per-band RBF brightness gate (convex mixture, L1 loss) for each
    band in ``buffers`` (``{band: (X(N,M), y(N,))}`` asinh space).

    Pruning (``min_usage`` > 0) is a **cumulative, whole-member** decision: a
    member's importance is summed across the 4 bands (the total bar in the
    importance chart) and, if below the threshold, the member is pruned in ALL
    bands; otherwise it is kept in all bands. So a member is either in the
    ensemble or not — never per-band."""
    bands: dict[str, BandCombiner] = {}
    holdouts: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, (X, y) in buffers.items():
        if np.asarray(X).size == 0:
            continue
        bc, Xval, yval = _fit_one_band(X, y, n_kernels=int(n_kernels),
                                       sigma_scale=float(sigma_scale),
                                       level_range=level_range, steps=int(steps),
                                       lr=float(lr), batch=int(batch),
                                       seed=int(seed), holdout=float(holdout))
        bands[name] = bc
        holdouts[name] = (Xval, yval)

    # Cumulative pruning: sum each member's per-band importance (mean gate weight
    # over the brightness sweep) across bands → one keep/drop mask applied to
    # every band. Renormalising over survivors only raises their weight, so each
    # survivor's cumulative importance ends up >= the threshold.
    if float(min_usage) > 0.0 and bands:
        m = next(iter(bands.values())).V.shape[1]
        levels = np.linspace(level_range[0], level_range[1], 25)
        cum = np.zeros(m)
        for bc in bands.values():
            cum += bc.weights_at(levels).mean(axis=0)
        surviving = cum >= float(min_usage)
        if not surviving.any():
            surviving = np.ones(m, bool)
        for bc in bands.values():
            bc.surviving = surviving.copy()

    vals = [float(np.mean(np.abs(bc.forward_asinh(holdouts[name][0]) - holdouts[name][1])))
            for name, bc in bands.items()]
    return Combiner(member_labels=list(member_labels), n_kernels=int(n_kernels),
                    sigma_scale=float(sigma_scale), min_usage=float(min_usage),
                    bands=bands, band_names=tuple(buffers.keys()),
                    level_range=tuple(level_range),
                    val_l1=(float(np.mean(vals)) if vals else None))


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _combiner_dir(base_dir: str) -> str:
    return os.path.join(base_dir, "combiner")


def save_combiner(comb: Combiner, base_dir: str) -> None:
    d = _combiner_dir(base_dir)
    os.makedirs(d, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    for name, bc in comb.bands.items():
        arrays[f"{name}__V"] = np.asarray(bc.V, np.float32)
        arrays[f"{name}__a"] = np.asarray(bc.a, np.float32)
        arrays[f"{name}__centers"] = np.asarray(bc.centers, np.float32)
        arrays[f"{name}__sigma"] = np.asarray([bc.sigma], np.float32)
        arrays[f"{name}__mask"] = np.asarray(bc.surviving, bool)
    np.savez_compressed(os.path.join(d, "combiner.npz"), **arrays)
    manifest = {
        "kind": comb.kind,
        "member_labels": list(comb.member_labels),
        "n_kernels": int(comb.n_kernels),
        "sigma_scale": float(comb.sigma_scale),
        "min_usage": float(comb.min_usage),
        "level_range": list(comb.level_range),
        "band_names": list(comb.band_names),
        "stretch_e": {b: _band_scale(b) for b in comb.band_names},
        "records_fp": comb.records_fp,
        "starfull": bool(comb.starfull),
        "val_l1": comb.val_l1,
        "surviving": comb.surviving_members(),
        "fit_meta": comb.fit_meta,
    }
    with open(os.path.join(d, "combiner.json"), "w") as f:
        json.dump(manifest, f, indent=2)


def load_combiner(base_dir: str, *, member_labels: list[str] | None = None
                  ) -> Combiner | None:
    """Load a persisted combiner, or ``None`` if absent, **stale** (its saved
    member labels no longer match ``member_labels``), or an incompatible/old
    format (e.g. a pre-RBF combiner)."""
    d = _combiner_dir(base_dir)
    jp, npzp = os.path.join(d, "combiner.json"), os.path.join(d, "combiner.npz")
    if not (os.path.exists(jp) and os.path.exists(npzp)):
        return None
    try:
        with open(jp) as f:
            man = json.load(f)
        if man.get("kind") != "rbf_gate":
            return None
        if member_labels is not None and list(man["member_labels"]) != list(member_labels):
            return None
        z = np.load(npzp)
        bands: dict[str, BandCombiner] = {}
        for name in man["band_names"]:
            bands[name] = BandCombiner(
                V=z[f"{name}__V"], a=z[f"{name}__a"],
                centers=z[f"{name}__centers"].reshape(-1),
                sigma=float(z[f"{name}__sigma"][0]),
                surviving=z[f"{name}__mask"])
        return Combiner(
            member_labels=list(man["member_labels"]),
            n_kernels=int(man.get("n_kernels", DEFAULT_N_KERNELS)),
            sigma_scale=float(man.get("sigma_scale", DEFAULT_SIGMA_SCALE)),
            min_usage=float(man.get("min_usage", 0.0)), bands=bands,
            band_names=tuple(man["band_names"]),
            level_range=tuple(man.get("level_range", GATE_LEVEL_RANGE)),
            records_fp=man.get("records_fp"), starfull=bool(man.get("starfull", True)),
            val_l1=man.get("val_l1"), fit_meta=man.get("fit_meta", {}))
    except (OSError, ValueError, KeyError, TypeError):
        return None
