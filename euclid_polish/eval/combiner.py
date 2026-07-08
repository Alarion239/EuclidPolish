"""Deep-ensemble **combiner** — a tiny per-band mixture of the member SR cubes.

For the STARFULL regime the reconstruction is normally the naive per-pixel mean
of the ``M`` ensemble members. That discards their complementary strengths
(L1-trained members render faint galaxies cleanly but erase star cores;
L2-trained members keep the point sources but leave speckle). This module learns
a tiny **per-band** model ``f: ℝ^M → ℝ`` applied at every pixel that fuses the
members — leaning on the right member at the right brightness.

Design (locked with the user):

* **Direct MLP with a linear skip, per band.** One small model per output band
  (VIS, Y_E, J_E, H_E): ``y = x·w_skip + b + MLP(x)`` where ``MLP`` is a single
  hidden layer of **2–4 units** (default 3, tanh). The linear skip carries the
  weighted-average bulk (so the model reproduces the wide asinh range even when a
  member is already good, which a bounded tanh cannot); the tiny hidden part is a
  brightness-dependent *correction*. Still a direct ``ℝ^M → ℝ`` map, and more
  interpretable — the skip weights are the base per-member mixture (see
  :meth:`Combiner.effective_weights`). Brightness-adaptivity is automatic: the
  input magnitudes *are* the brightness.
* **asinh space.** Everything runs in ``arcsinh(electrons / STRETCH_SCALE_E)``;
  the output is inverse-stretched with ``sinh(·)·scale`` after the same ±clip the
  inference path uses, so it can't overflow.
* **L1 loss** (asinh-space ``mean|·|``), fit LOCALLY on the ``validate`` split.
* **Group-L1 pruning.** An L1 penalty on each member's *whole input footprint*
  (its hidden column + its skip weight) drives unhelpful members to ~0; pruned
  members are hard-zeroed so they read exactly 0 in the effective-weight probe
  (survivors are reported).
* **Starfull-only** (the ``hr_`` target — starless members merely erase stars).

Persistence: ``<dir>/combiner/combiner.npz`` (weights) + ``combiner.json``
(metadata: member labels it was fit against, arch, λ, per-band survivors, the
validate records fingerprint). :func:`load_combiner` returns ``None`` when the
saved member labels no longer match the active ensemble (stale).
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

#: Survivor threshold: a member survives if its first-layer column norm exceeds
#: this fraction of the largest column norm in that band.
SURVIVOR_TAU = 0.05

BAND_NAMES = tuple(Config.HR_TARGET_BAND_NAMES)


def _band_scale(name: str) -> float:
    return _BAND_SCALE.get(name, float(Config.STRETCH_SCALE_E))


def _act_np(z: np.ndarray, activation: str) -> np.ndarray:
    if activation == "linear":
        return z
    if activation == "relu":
        return np.maximum(z, 0.0)
    return np.tanh(z)


def _act_deriv_np(z: np.ndarray, activation: str) -> np.ndarray:
    if activation == "linear":
        return np.ones_like(z)
    if activation == "relu":
        return (z > 0.0).astype(z.dtype)
    return 1.0 - np.tanh(z) ** 2


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

@dataclass
class BandCombiner:
    """One band's tiny skip-MLP: ``(N, M) asinh members → (N,) asinh combined``.

    ``y = X·w_skip + b2 + act(X·W1 + b1)·W2``.
    """

    W1: np.ndarray                 # (M, H)
    b1: np.ndarray                 # (H,)
    W2: np.ndarray                 # (H,)
    b2: float
    w_skip: np.ndarray             # (M,) linear per-member skip weights
    surviving: np.ndarray          # (M,) bool
    activation: str = "tanh"

    def forward_asinh(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, np.float64)
        h = _act_np(X @ self.W1 + self.b1, self.activation)   # (N, H)
        return X @ self.w_skip + self.b2 + h @ self.W2        # (N,)

    def group_norms(self) -> np.ndarray:
        """Per-member input footprint: ``sqrt(||W1[m]||² + w_skip[m]²)``."""
        w1 = np.asarray(self.W1, np.float64)
        ws = np.asarray(self.w_skip, np.float64)
        return np.sqrt(np.sum(w1 ** 2, axis=1) + ws ** 2)


@dataclass
class Combiner:
    """The 4-band combiner + the metadata needed to apply/persist/validate it."""

    member_labels: list[str]
    hidden: int
    lam_group: float
    bands: dict[str, BandCombiner]
    band_names: tuple[str, ...] = BAND_NAMES
    activation: str = "tanh"
    records_fp: str | None = None
    starfull: bool = True
    val_l1: float | None = None
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

    # -- interpretability -- #
    def effective_weights(self, band: str, *, n_levels: int = 25,
                          level_range: tuple[float, float] = (-1.0, 12.0)
                          ) -> dict:
        """Local Jacobian ``∂output/∂memberₘ`` over an all-members-equal
        brightness sweep — the "how much is member m trusted vs brightness"
        curve. Pruned members read ≈0 everywhere."""
        bc = self.bands[band]
        m = bc.W1.shape[0]
        levels = np.linspace(level_range[0], level_range[1], int(n_levels))
        X0 = np.repeat(levels[:, None], m, axis=1)             # (L, M)
        z = X0 @ bc.W1 + bc.b1                                 # (L, H)
        dact = _act_deriv_np(z, bc.activation)                 # (L, H)
        jac = (np.einsum("mh,lh,h->lm", bc.W1, dact, bc.W2)    # nonlinear part
               + bc.w_skip[None, :])                           # + linear skip
        scale = _band_scale(band)
        return {"brightness_asinh": levels,
                "brightness_e": np.sinh(levels) * scale,
                "jacobian": jac}

    def surviving_members(self) -> dict[str, list[bool]]:
        return {b: self.bands[b].surviving.astype(bool).tolist()
                for b in self.bands}


# ---------------------------------------------------------------------------
# Fit-data assembly
# ---------------------------------------------------------------------------

def build_fit_buffers_from_fields(field_iter, band_names, *,
                                  max_rows: int = 3_000_000,
                                  n_bright_bins: int = 8,
                                  per_bin_per_field: int = 2000,
                                  level_range: tuple[float, float] = (-1.0, 12.0),
                                  seed: int = 0) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Stream ``(preds(M,H,W,C), hr(H,W,C))`` fields (electrons) → per-band
    ``{band: (X(N,M), y(N,))}`` in asinh space, subsampling pixels **stratified
    by brightness** (equal quota per asinh-brightness bin) so faint structure
    isn't drowned by the dominant sky pixels."""
    rng = np.random.default_rng(seed)
    edges = np.linspace(level_range[0], level_range[1], int(n_bright_bins) + 1)
    Xacc: dict[str, list] = {b: [] for b in band_names}
    yacc: dict[str, list] = {b: [] for b in band_names}
    counts = {b: 0 for b in band_names}

    for preds, hr in field_iter:
        preds = np.asarray(preds, np.float32)
        hr = np.asarray(hr, np.float32)
        m = preds.shape[0]
        for ci, name in enumerate(band_names):
            if counts[name] >= max_rows:
                continue
            scale = _band_scale(name)
            xs = np.arcsinh(preds[..., ci].reshape(m, -1).T / scale)   # (P, M)
            ys = np.arcsinh(hr[..., ci].reshape(-1) / scale)           # (P,)
            bin_idx = np.clip(np.digitize(ys, edges) - 1, 0, int(n_bright_bins) - 1)
            for b in range(int(n_bright_bins)):
                if counts[name] >= max_rows:
                    break
                sel = np.where(bin_idx == b)[0]
                if sel.size == 0:
                    continue
                take = int(min(per_bin_per_field, sel.size, max_rows - counts[name]))
                pick = rng.choice(sel, size=take, replace=False)
                Xacc[name].append(xs[pick])
                yacc[name].append(ys[pick])
                counts[name] += take

    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in band_names:
        if Xacc[name]:
            out[name] = (np.concatenate(Xacc[name]).astype(np.float32),
                         np.concatenate(yacc[name]).astype(np.float32))
        else:
            out[name] = (np.zeros((0, 0), np.float32), np.zeros((0,), np.float32))
    return out


def build_fit_buffers(base_dir: str, records_dir: str, *, subset: str = "validate",
                      num_images: int = 100, ens=None, on_progress=None,
                      **kw) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], list[str]]:
    """Records-backed convenience: run each ``validate`` field through the
    STARFULL ensemble and assemble the per-band fit buffers. Returns
    ``(buffers, member_labels)``. Members are inferred fresh here; the fit job
    prefers reusing cached member cubes and calls
    :func:`build_fit_buffers_from_fields` directly."""
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

def _fit_one_band(X: np.ndarray, y: np.ndarray, *, hidden: int, activation: str,
                  lam_group: float, steps: int, lr: float, batch: int,
                  seed: int, holdout: float,
                  eps: float = 1e-12) -> tuple[BandCombiner, float]:
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

    tf.random.set_seed(seed)
    init = tf.keras.initializers.GlorotUniform(seed=seed)
    W1 = tf.Variable(init((m, hidden)), dtype=tf.float32)
    b1 = tf.Variable(tf.zeros((hidden,), tf.float32))
    W2 = tf.Variable(init((hidden, 1)), dtype=tf.float32)
    b2 = tf.Variable(tf.zeros((1,), tf.float32))
    # Linear skip, initialised to the plain mean (a sensible starting mixture).
    w_skip = tf.Variable(tf.fill((m, 1), 1.0 / float(m)), dtype=tf.float32)
    variables = [W1, b1, W2, b2, w_skip]
    opt = tf.keras.optimizers.Adam(lr)
    act = {"tanh": tf.nn.tanh, "relu": tf.nn.relu, "linear": tf.identity}[activation]
    lam = tf.constant(float(lam_group), tf.float32)

    def _forward(xb):
        h = act(tf.matmul(xb, W1) + b1)
        return tf.squeeze(tf.matmul(xb, w_skip) + tf.matmul(h, W2) + b2, axis=-1)

    @tf.function
    def train_step(xb, yb):
        with tf.GradientTape() as tape:
            data = tf.reduce_mean(tf.abs(_forward(xb) - yb))
            # Group-L1 over each member's whole input footprint (hidden col +
            # skip weight) → prunes entire members.
            group = tf.sqrt(tf.reduce_sum(W1 ** 2, axis=1)
                            + tf.squeeze(w_skip, axis=-1) ** 2 + eps)
            loss = data + lam * tf.reduce_sum(group)
        grads = tape.gradient(loss, variables)
        opt.apply_gradients(zip(grads, variables))

    bs = int(min(batch, max(1, len(Xtr))))
    ds = (tf.data.Dataset.from_tensor_slices((Xtr, ytr))
          .shuffle(min(len(Xtr), 100_000), seed=seed, reshuffle_each_iteration=True)
          .batch(bs).repeat())
    it = iter(ds)

    def _val_l1():
        return float(np.mean(np.abs(_forward(tf.constant(Xval)).numpy() - yval)))

    def _snapshot():
        return [W1.numpy().copy(), b1.numpy().copy(), W2.numpy().copy(),
                float(b2.numpy()[0]), w_skip.numpy().copy()]

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
                best_w = _snapshot()
            else:
                stale += 1
                if stale >= patience:
                    break
    if best_w is None:
        best_w = _snapshot()

    w1, b1n, w2, b2n, wskip = best_w
    w2 = np.asarray(w2, np.float32).reshape(-1)                 # (H,)
    wskip = np.asarray(wskip, np.float32).reshape(-1)           # (M,)
    group = np.sqrt(np.sum(w1.astype(np.float64) ** 2, axis=1) + wskip.astype(np.float64) ** 2)
    mx = float(group.max()) if group.size else 0.0
    surviving = (group > SURVIVOR_TAU * mx) if mx > 0 else np.ones(m, bool)
    # Hard-zero pruned members so they contribute exactly nothing.
    w1 = w1.copy()
    wskip = wskip.copy()
    w1[~surviving] = 0.0
    wskip[~surviving] = 0.0
    bc = BandCombiner(W1=w1.astype(np.float32), b1=np.asarray(b1n, np.float32),
                      W2=w2, b2=float(b2n), w_skip=wskip,
                      surviving=np.asarray(surviving, bool), activation=activation)
    val_l1 = float(np.mean(np.abs(bc.forward_asinh(Xval) - yval)))
    return bc, val_l1


def fit_combiner(buffers, member_labels, *, hidden: int = 3, activation: str = "tanh",
                 lam_group: float = 1e-3, steps: int = 2000, lr: float = 3e-3,
                 batch: int = 16384, seed: int = 0, holdout: float = 0.1) -> Combiner:
    """Fit one tiny per-band MLP (L1 loss + group-L1 pruning) for each band in
    ``buffers`` (``{band: (X(N,M), y(N,))}`` asinh space)."""
    bands: dict[str, BandCombiner] = {}
    vals: list[float] = []
    for name, (X, y) in buffers.items():
        if np.asarray(X).size == 0:
            continue
        bc, vl = _fit_one_band(X, y, hidden=int(hidden), activation=activation,
                               lam_group=float(lam_group), steps=int(steps),
                               lr=float(lr), batch=int(batch), seed=int(seed),
                               holdout=float(holdout))
        bands[name] = bc
        vals.append(vl)
    return Combiner(member_labels=list(member_labels), hidden=int(hidden),
                    lam_group=float(lam_group), bands=bands,
                    band_names=tuple(buffers.keys()), activation=activation,
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
        arrays[f"{name}__w1"] = np.asarray(bc.W1, np.float32)
        arrays[f"{name}__b1"] = np.asarray(bc.b1, np.float32)
        arrays[f"{name}__w2"] = np.asarray(bc.W2, np.float32)
        arrays[f"{name}__b2"] = np.asarray([bc.b2], np.float32)
        arrays[f"{name}__wskip"] = np.asarray(bc.w_skip, np.float32)
        arrays[f"{name}__mask"] = np.asarray(bc.surviving, bool)
    np.savez_compressed(os.path.join(d, "combiner.npz"), **arrays)
    manifest = {
        "member_labels": list(comb.member_labels),
        "hidden": int(comb.hidden),
        "lam_group": float(comb.lam_group),
        "activation": comb.activation,
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
    """Load a persisted combiner, or ``None`` if absent or **stale** (its saved
    member labels no longer match ``member_labels``, i.e. the ensemble changed)."""
    d = _combiner_dir(base_dir)
    jp, npzp = os.path.join(d, "combiner.json"), os.path.join(d, "combiner.npz")
    if not (os.path.exists(jp) and os.path.exists(npzp)):
        return None
    with open(jp) as f:
        man = json.load(f)
    if member_labels is not None and list(man["member_labels"]) != list(member_labels):
        return None
    z = np.load(npzp)
    bands: dict[str, BandCombiner] = {}
    for name in man["band_names"]:
        bands[name] = BandCombiner(
            W1=z[f"{name}__w1"], b1=z[f"{name}__b1"],
            W2=z[f"{name}__w2"].reshape(-1), b2=float(z[f"{name}__b2"][0]),
            w_skip=z[f"{name}__wskip"].reshape(-1),
            surviving=z[f"{name}__mask"], activation=man["activation"])
    return Combiner(
        member_labels=list(man["member_labels"]), hidden=int(man["hidden"]),
        lam_group=float(man["lam_group"]), bands=bands,
        band_names=tuple(man["band_names"]), activation=man["activation"],
        records_fp=man.get("records_fp"), starfull=bool(man.get("starfull", True)),
        val_l1=man.get("val_l1"), fit_meta=man.get("fit_meta", {}))
