"""Fit the spatial gating combiner on cached ensemble member cubes.

Training reads random crops straight from the memory-mapped member cubes, so
the fit never holds the whole validation stack in RAM. The TensorFlow graph
here mirrors :func:`euclid_polish.eval.spatial_gate.gate_logits` operation for
operation; the fitted parameters are handed to the NumPy
:class:`~euclid_polish.eval.spatial_gate.SpatialGateCombiner` for inference.

The gate averages the members in electrons or in per-band asinh space
(``mix_space``, see :mod:`euclid_polish.eval.spatial_gate`); the graph keeps
its output in band-knee asinh either way, so the losses below read the same.
The loss is squared error in per-band asinh space (the space the ensemble is
scored in), divided per field and band by the best member's error there: a
loss of 1.0 means "as good as the best single member", and every field and
band counts equally, as in the mean per-field PSNR the ensemble is scored by.
The best member is chosen on the natural (not blackout-augmented) fields. The
gate starts near that member (per band) and the checkpoint with the lowest
held-out loss is kept. Optionally the loss is evaluated at several asinh
knees and averaged (``loss_knees``), matching the knee-integrated PSNR
instead of favouring the brightnesses around the band knee.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import shutil
import tempfile
import time
import weakref
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import tensorflow as tf

from euclid_polish.config import Config
from euclid_polish.ensemble import EnsembleModel
from euclid_polish.eval.knee_psnr import integrated_psnr, knee_psnr
from euclid_polish.eval.spatial_gate import (
    BAND_NAMES,
    DILATIONS,
    MIX_ASINH,
    MIX_LINEAR,
    MIX_SPACES,
    PARAM_NAMES,
    SPATIAL_GATE_KIND,
    SpatialGateCombiner,
    band_scales,
    lr_features,
)
from euclid_polish.image.collection import ImageSet
from euclid_polish.image.tfio import tfrecord_path
from euclid_polish.sky.observation.saturation import (
    StarSaturationModel,
    apply_saturation_masking,
)
from euclid_polish.training.target_blur import blur_target_array

#: Synthetic blackouts are stamped on sources above this fraction of each
#: band's saturation well (VIS, Y, J, H). Real saturation is rare in the
#: synthetic fields, so the thresholds are lowered to create enough examples.
BLACKOUT_WELL_FRACTIONS = (0.03, 0.1, 0.1, 0.1)
ERROR_MAP_BLOCK_PX = 30
LOSS_BORDER_PX = 16
INITIAL_BEST_WEIGHT = 0.9
#: Knees (e⁻) of the all-knee loss: log-uniform, half a decade apart, over the
#: same 0.1–10⁴ e⁻ span as the integrated PSNR, so the loss is its stand-in.
ALL_KNEE_LOSS = tuple(float(v) for v in np.logspace(-1.0, 4.0, 11))
#: Knees the held-out integrated PSNR is monitored on (both loss modes).
MONITOR_KNEES = ALL_KNEE_LOSS
ProgressFn = Callable[[int, int, str], None]


@dataclass
class GateField:
    """One field: member cube paths plus its target and LR in memory."""

    index: int
    member_paths: list[str]
    target_e: np.ndarray            # (H, W, C) electrons, blurred target
    lr_e: np.ndarray                # (H/2, W/2, C) electrons
    tag: str = "natural"
    _lr_feats: np.ndarray | None = field(default=None, repr=False)

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.target_e.shape[0]), int(self.target_e.shape[1])

    def members_e(self, y0: int = 0, x0: int = 0,
                  size: int | None = None) -> np.ndarray:
        """``(M, h, w, C)`` electrons for a crop (or the whole field)."""
        h, w = self.shape
        y1 = h if size is None else y0 + size
        x1 = w if size is None else x0 + size
        return np.stack([np.asarray(np.load(p, mmap_mode="r")[y0:y1, x0:x1],
                                    np.float32) for p in self.member_paths])

    def lr_feats(self, scales: np.ndarray) -> np.ndarray:
        if self._lr_feats is None:
            self._lr_feats = lr_features(self.lr_e, scales)
        return self._lr_feats


class LazyMemberRunner:
    """Runs the ensemble members on an LR field; loads them on first use, so a
    fully cached blackout bucket costs no model loading."""

    def __init__(self, base_dir: str, *, starless: bool, labels: Sequence[str]):
        self.base_dir = base_dir
        self.starless = bool(starless)
        self.labels = [str(v) for v in labels]
        self.seconds: list[float] = []
        self._ensemble: EnsembleModel | None = None

    def __call__(self, lr: np.ndarray) -> np.ndarray:
        if self._ensemble is None:
            self._ensemble = EnsembleModel(self.base_dir, starless=self.starless)
            if list(self._ensemble.member_labels) != self.labels:
                raise RuntimeError("active members differ from the cached cubes")
        started = time.time()
        out = self._ensemble.member_arrays(lr)
        self.seconds.append(time.time() - started)
        return out


def split_holdout(fields: Sequence[GateField], n_holdout: int,
                  seed: int) -> tuple[list[GateField], list[GateField]]:
    """Deterministic field-level split into (train, held-out)."""
    n_holdout = min(max(1, int(n_holdout)), max(1, len(fields) - 1))
    order = np.random.default_rng(seed).permutation(len(fields))
    held = set(order[:n_holdout].tolist())
    return ([f for i, f in enumerate(fields) if i not in held],
            [f for i, f in enumerate(fields) if i in held])


def load_cube_fields(cubes_dir: str, records_dir: str, subset: str, *,
                     target_name: str, target_fwhm_arcsec: float,
                     indices: Sequence[int] | None = None,
                     progress: ProgressFn | None = None) -> tuple[list[GateField], list[str]]:
    """Pair a cube bucket's member files with its target and LR records."""
    with open(os.path.join(cubes_dir, "viz_index.json")) as handle:
        manifest = json.load(handle)
    labels = [str(v) for v in manifest.get("member_labels", [])]
    wanted = sorted(int(i) for i in (indices if indices is not None
                                     else manifest.get("indices", [])))
    if not wanted or not labels:
        return [], labels
    last = max(wanted) + 1
    targets = iter(ImageSet.read(
        tfrecord_path(records_dir, f"{target_name}_{subset}"), num_images=last))
    lrs = iter(ImageSet.read(
        tfrecord_path(records_dir, f"dirty_{subset}"), num_images=last))
    target_rec, lr_rec = next(targets, None), next(lrs, None)
    fields: list[GateField] = []
    for position, index in enumerate(wanted, 1):
        while target_rec is not None and target_rec.index < index:
            target_rec = next(targets, None)
        while lr_rec is not None and lr_rec.index < index:
            lr_rec = next(lrs, None)
        paths = [os.path.join(cubes_dir, f"member{m}_{index:05d}.npy")
                 for m in range(len(labels))]
        if (target_rec is None or target_rec.index != index or lr_rec is None
                or lr_rec.index != index
                or not all(os.path.isfile(p) for p in paths)):
            continue
        target = blur_target_array(
            np.asarray(target_rec.data, np.float32), target_fwhm_arcsec,
            pixel_scale_arcsec=target_rec.pixel_scale_arcsec)
        fields.append(GateField(index, paths, np.asarray(target, np.float32),
                                np.asarray(lr_rec.data, np.float32)))
        if progress is not None:
            progress(position, len(wanted), f"loading {subset} field {index}")
    return fields, labels


def stamp_blackouts(lr_e: np.ndarray, rng: np.random.Generator, *,
                    well_fractions: Sequence[float] = BLACKOUT_WELL_FRACTIONS,
                    band_names: Sequence[str] = BAND_NAMES) -> np.ndarray:
    """A copy of ``lr_e`` with MER-style blackouts on every source above
    ``well_fractions`` × the band's saturation well."""
    wells = {name: float(frac) * float(Config.STAR_SATURATION_WELL_E[name])
             for name, frac in zip(band_names, well_fractions, strict=True)}
    source = np.asarray(lr_e, np.float32)
    stamped = source.copy()
    apply_saturation_masking(stamped, StarSaturationModel(well_e=wells), rng,
                             band_names=tuple(band_names), trigger_4ch=source,
                             mask_probability=1.0)
    return stamped


def build_blackout_fields(fields: Sequence[GateField], member_labels: Sequence[str],
                          run_members: Callable[[np.ndarray], np.ndarray],
                          out_dir: str, *, max_fields: int, seed: int,
                          source_fingerprint: str | None = None,
                          well_fractions: Sequence[float] = BLACKOUT_WELL_FRACTIONS,
                          progress: ProgressFn | None = None) -> list[GateField]:
    """Stamp synthetic blackouts on field LRs, run the members on them, and
    cache the resulting member cubes under ``out_dir``. Fields whose stamping
    zeroes nothing new are skipped. A cache written with the same members,
    thresholds, seed and source records is reused without inference."""
    identity = {"member_labels": list(member_labels),
                "well_fractions": [float(v) for v in well_fractions],
                "seed": int(seed), "source": source_fingerprint}
    manifest_path = os.path.join(out_dir, "blackout_index.json")
    cached: dict = {}
    if os.path.isfile(manifest_path):
        with open(manifest_path) as handle:
            cached = json.load(handle)
        if cached.get("identity") != identity:
            cached = {}
    done = {int(i) for i in cached.get("indices", [])}
    os.makedirs(out_dir, exist_ok=True)
    by_index = {f.index: f for f in fields}
    out: list[GateField] = []
    for position, source in enumerate(fields, 1):
        if len(out) >= int(max_fields):
            break
        tag = f"{source.index:05d}"
        member_paths = [os.path.join(out_dir, f"member{m}_{tag}.npy")
                        for m in range(len(member_labels))]
        lr_path = os.path.join(out_dir, f"lr_{tag}.npy")
        if source.index in done and os.path.isfile(lr_path) and all(
                os.path.isfile(p) for p in member_paths):
            stamped = np.load(lr_path)
        else:
            rng = np.random.default_rng([int(seed), source.index])
            stamped = stamp_blackouts(source.lr_e, rng, well_fractions=well_fractions)
            if not np.any((stamped == 0) & (source.lr_e != 0)):
                continue
            members = np.asarray(run_members(stamped), np.float32)
            for path, member in zip(member_paths, members, strict=True):
                np.save(path, member)
            np.save(lr_path, stamped)
            done.add(source.index)
            with open(manifest_path, "w") as handle:
                json.dump({"identity": identity, "indices": sorted(done)}, handle)
        out.append(GateField(source.index, member_paths,
                             by_index[source.index].target_e, stamped, tag="blackout"))
        if progress is not None:
            progress(position, len(fields), f"blackout field {source.index}")
    return out


# --------------------------------------------------------------------------- #
# TensorFlow mirror of the NumPy forward pass
# --------------------------------------------------------------------------- #

def _conv1x1(x, w, b):
    return tf.tensordot(x, w, axes=[[3], [0]]) + b


def _conv3x3(x, w, b, d: int):
    padded = tf.pad(x, [[0, 0], [d, d], [d, d], [0, 0]], mode="SYMMETRIC")
    return tf.nn.convolution(padded, w, padding="VALID", dilations=d) + b


def _upsample2x(x):
    s = tf.shape(x)
    prev = tf.concat([x[:, :1], x[:, :-1]], axis=1)
    nxt = tf.concat([x[:, 1:], x[:, -1:]], axis=1)
    x = tf.reshape(tf.stack([0.75 * x + 0.25 * prev, 0.75 * x + 0.25 * nxt], axis=2),
                   [s[0], 2 * s[1], s[2], s[3]])
    prev = tf.concat([x[:, :, :1], x[:, :, :-1]], axis=2)
    nxt = tf.concat([x[:, :, 1:], x[:, :, -1:]], axis=2)
    return tf.reshape(tf.stack([0.75 * x + 0.25 * prev, 0.75 * x + 0.25 * nxt], axis=3),
                      [s[0], 2 * s[1], 2 * s[2], s[3]])


def tf_gate_logits(params: dict, members, lr_feats, use_lr: bool):
    """Batched ``(B, H, W, M*C)`` logits; mirrors ``spatial_gate.gate_logits``."""
    feat = tf.nn.relu(_conv1x1(members, params["w_in"], params["b_in"]))
    coarse = tf.nn.space_to_depth(feat, 2)
    if use_lr:
        coarse = tf.concat([coarse, lr_feats], axis=-1)
    hidden = tf.nn.relu(_conv1x1(coarse, params["w_merge"], params["b_merge"]))
    for d in DILATIONS:
        hidden = hidden + tf.nn.relu(
            _conv3x3(hidden, params[f"w_d{d}"], params[f"b_d{d}"], d))
    fused = tf.concat([feat, _upsample2x(hidden)], axis=-1)
    fused = tf.nn.relu(_conv1x1(fused, params["w_fuse"], params["b_fuse"]))
    return _conv1x1(fused, params["w_out"], params["b_out"])


def tf_mix(logits, members, n_members: int, n_bands: int, mix_space: str = MIX_ASINH):
    """Convex member mixture of the band-knee asinh ``members``, returned in
    band-knee asinh. ``"linear"`` averages in electrons: the band knee cancels,
    ``asinh(Σ w·sinh(x))``."""
    shape = tf.shape(logits)
    five = [shape[0], shape[1], shape[2], n_members, n_bands]
    weights = tf.nn.softmax(tf.reshape(logits, five), axis=3)
    members = tf.reshape(members, five)
    if mix_space == MIX_LINEAR:
        return tf.asinh(tf.reduce_sum(weights * tf.sinh(members), axis=3))
    return tf.reduce_sum(weights * members, axis=3)


def init_params(n_members: int, n_bands: int, width: int, use_lr: bool,
                best_member_per_band: Sequence[int], seed: int) -> dict[str, np.ndarray]:
    """Glorot hidden layers; zero output kernel with a bias that puts
    ``INITIAL_BEST_WEIGHT`` on each band's best member."""
    rng = np.random.default_rng(seed)

    def glorot(shape, fan_in, fan_out):
        limit = math.sqrt(6.0 / (fan_in + fan_out))
        return rng.uniform(-limit, limit, shape).astype(np.float32)

    mc, f = n_members * n_bands, int(width)
    merge_in = 4 * f + (n_bands + 1 if use_lr else 0)
    params = {"w_in": glorot((mc, f), mc, f), "b_in": np.zeros(f, np.float32),
              "w_merge": glorot((merge_in, f), merge_in, f),
              "b_merge": np.zeros(f, np.float32)}
    for d in DILATIONS:
        params[f"w_d{d}"] = glorot((3, 3, f, f), 9 * f, 9 * f)
        params[f"b_d{d}"] = np.zeros(f, np.float32)
    params["w_fuse"] = glorot((2 * f, f), 2 * f, f)
    params["b_fuse"] = np.zeros(f, np.float32)
    params["w_out"] = np.zeros((f, mc), np.float32)
    bias = np.zeros((n_members, n_bands), np.float32)
    if n_members > 1:
        logit = math.log(INITIAL_BEST_WEIGHT * (n_members - 1)
                         / (1.0 - INITIAL_BEST_WEIGHT))
        for c, m in enumerate(best_member_per_band):
            bias[int(m), c] = logit
    params["b_out"] = bias.reshape(mc)
    return {name: params[name] for name in PARAM_NAMES}


# --------------------------------------------------------------------------- #
# Fitting
# --------------------------------------------------------------------------- #

class _FeatureCache:
    """Training fields' member features ``(H, W, M*C)`` as float16 asinh on
    disk, so a training crop is one contiguous read instead of one strided
    read per member file (the difference between an input-bound and a
    compute-bound fit). The temporary directory is removed by ``cleanup`` or,
    failing that, when the cache is garbage collected."""

    def __init__(self) -> None:
        self.directory = tempfile.mkdtemp(prefix="spatial_gate_features_")
        self._finalizer = weakref.finalize(self, shutil.rmtree, self.directory, True)
        self.paths: list[str] = []

    def add(self, features: np.ndarray) -> None:
        path = os.path.join(self.directory, f"{len(self.paths):05d}.npy")
        np.save(path, np.asarray(features, np.float16))
        self.paths.append(path)

    def crop(self, i: int, y0: int, x0: int, size: int) -> np.ndarray:
        return np.asarray(np.load(self.paths[i], mmap_mode="r")
                          [y0:y0 + size, x0:x0 + size], np.float32)

    def cleanup(self) -> None:
        self._finalizer()


def _member_statistics(fields: Sequence[GateField], scales: np.ndarray,
                       progress: ProgressFn | None, cache: _FeatureCache):
    """Per-field ``(F, M, C)`` member MSE and block error maps (asinh); also
    fills ``cache`` with each field's member features."""
    field_mse: list[np.ndarray] = []
    maps: list[np.ndarray] = []
    b = ERROR_MAP_BLOCK_PX
    for position, f in enumerate(fields, 1):
        members = np.arcsinh(f.members_e() / scales.astype(np.float32))
        h, w = f.shape
        cache.add(members.transpose(1, 2, 0, 3).reshape(h, w, -1))
        target = np.arcsinh(f.target_e / scales.astype(np.float32))
        err2 = (members - target[None]) ** 2
        field_mse.append(err2.mean(axis=(1, 2), dtype=np.float64))
        hb, wb = h // b, w // b
        maps.append(err2[:, :hb * b, :wb * b]
                    .reshape(len(members), hb, b, wb, b, -1).sum(axis=(2, 4)))
        if progress is not None:
            progress(position, len(fields), f"member statistics field {f.index}")
    return np.stack(field_mse), maps


def _stretched_psnr(mse: float) -> float:
    peak = float(Config.PSNR_PEAK_STRETCHED)
    return float("inf") if mse <= 0 else float(10.0 * np.log10(peak * peak / mse))


def fit_spatial_gate(train_fields: Sequence[GateField],
                     holdout_fields: Sequence[GateField],
                     member_labels: Sequence[str], *,
                     band_names: Sequence[str] = BAND_NAMES,
                     width: int = 32, use_lr: bool = False,
                     steps: int = 2000, batch_size: int = 8, crop: int = 192,
                     learning_rate: float = 2e-3, warmup_steps: int = 100,
                     uniform_crop_fraction: float = 0.5,
                     eval_every: int = 250, seed: int = 0,
                     active_members: Sequence[int] | None = None,
                     loss_knees: Sequence[float] | None = None,
                     mix_space: str = MIX_ASINH,
                     progress: ProgressFn | None = None,
                     log: Callable[[str], None] | None = None,
                     checkpoint: Callable[[SpatialGateCombiner], None] | None = None,
                     ) -> SpatialGateCombiner:
    """Fit the gate; returns the checkpoint with the lowest held-out loss.

    ``active_members`` (positions in ``member_labels``) fits a pruned gate
    that reads only those members; it is still keyed to the whole ensemble.

    ``loss_knees`` scores the output at several asinh knees (e⁻) instead of
    only the band knee the members are mixed at: at each knee the error is
    taken relative to the reference member's on that field, then averaged —
    a stand-in for the knee-integrated PSNR (see :data:`ALL_KNEE_LOSS`).

    ``mix_space`` is where the members are averaged: ``"linear"``
    (electrons) or ``"asinh"`` (the band knee).

    ``checkpoint`` is called with the best gate so far every time the held-out
    loss improves (its ``fit_meta["complete"]`` is False), so a fit stopped
    early still leaves its best checkpoint behind."""
    if mix_space not in MIX_SPACES:
        raise ValueError(f"mix_space must be one of {MIX_SPACES}, got {mix_space!r}")
    all_labels = [str(v) for v in member_labels]
    active = (list(range(len(all_labels))) if active_members is None
              else sorted({int(i) for i in active_members}))
    if not active or active[-1] >= len(all_labels):
        raise ValueError("active_members must be positions within member_labels")
    if len(active) != len(all_labels):
        train_fields = [dataclasses.replace(f, member_paths=[f.member_paths[i] for i in active])
                        for f in train_fields]
        holdout_fields = [dataclasses.replace(f, member_paths=[f.member_paths[i] for i in active])
                          for f in holdout_fields]
    labels = [all_labels[i] for i in active]
    names = tuple(band_names)
    n_members, n_bands = len(labels), len(names)
    if not train_fields or not holdout_fields:
        raise ValueError("spatial gate needs training and held-out fields")
    if crop % 2 or crop <= 2 * LOSS_BORDER_PX:
        raise ValueError("crop must be even and wider than twice the loss border")
    scales = band_scales(names)
    say = log or (lambda _msg: None)
    started = time.time()

    cache = _FeatureCache()
    field_mse, error_maps = _member_statistics(train_fields, scales, progress, cache)
    # The reference member and the ranking use the natural fields only: the
    # blackout copies are extra training signal, not the scored distribution.
    natural = [i for i, f in enumerate(train_fields) if f.tag == "natural"]
    member_mse = field_mse[natural or list(range(len(train_fields)))].mean(axis=0)
    best_per_band = [int(np.argmin(member_mse[:, c])) for c in range(n_bands)]
    # Each field's loss is relative to its own best-member error, so every
    # field counts equally (the ensemble is scored by mean per-field PSNR).
    field_norm = np.maximum(np.stack([field_mse[:, m, c] for c, m in
                                      enumerate(best_per_band)], -1), 1e-12)
    say(f"best member per band: {[labels[m] for m in best_per_band]}")
    multi_knee = loss_knees is not None
    knees_e = np.asarray(loss_knees if multi_knee else [], np.float32)

    def reference_knee_mse(f: GateField) -> np.ndarray:
        """``(K, C)`` MSE of the reference member at each loss knee."""
        target = np.asarray(f.target_e, np.float32)
        refs = {m: np.load(f.member_paths[m]).astype(np.float32)
                for m in set(best_per_band)}
        out = np.empty((len(knees_e), n_bands), np.float64)
        for k, q in enumerate(knees_e):
            for c, m in enumerate(best_per_band):
                err = np.arcsinh(refs[m][..., c] / q) - np.arcsinh(target[..., c] / q)
                out[k, c] = np.mean(err * err, dtype=np.float64)
        return np.maximum(out, 1e-12)

    knee_norm = (np.stack([reference_knee_mse(f) for f in train_fields])
                 if multi_knee else None)                      # (F, K, C)

    # Crop sampler: half uniform, half centred where the best member's
    # relative error is (crops land where improvement is possible).
    block = ERROR_MAP_BLOCK_PX
    placement = []
    for i, emap in enumerate(error_maps):
        best = np.stack([emap[best_per_band[c], ..., c] for c in range(n_bands)], -1)
        weight = (best / field_norm[i]).sum(-1).reshape(-1)
        placement.append(weight / weight.sum() if weight.sum() > 0 else None)
    rng = np.random.default_rng(seed)
    lr_feats = [f.lr_feats(scales) for f in train_fields]
    targets = [np.arcsinh(f.target_e / scales.astype(np.float32)) for f in train_fields]

    def sample_crop(i: int) -> tuple[int, int]:
        h, w = train_fields[i].shape
        p = placement[i]
        if p is None or rng.random() < uniform_crop_fraction:
            y0, x0 = rng.integers(0, h - crop + 1), rng.integers(0, w - crop + 1)
        else:
            k = int(rng.choice(len(p), p=p))
            wb = w // block
            cy = (k // wb) * block + int(rng.integers(block))
            cx = (k % wb) * block + int(rng.integers(block))
            y0 = int(np.clip(cy - crop // 2, 0, h - crop))
            x0 = int(np.clip(cx - crop // 2, 0, w - crop))
        return int(y0) - int(y0) % 2, int(x0) - int(x0) % 2

    def batches():
        while True:
            xs, ts, ls, ns = [], [], [], []
            for _ in range(batch_size):
                i = int(rng.integers(len(train_fields)))
                y0, x0 = sample_crop(i)
                x = cache.crop(i, y0, x0, crop)
                t = targets[i][y0:y0 + crop, x0:x0 + crop]
                lf = lr_feats[i][y0 // 2:(y0 + crop) // 2, x0 // 2:(x0 + crop) // 2]
                k, flip = int(rng.integers(4)), bool(rng.random() < 0.5)
                arrays = [np.rot90(a, k, axes=(0, 1)) for a in (x, t, lf)]
                if flip:
                    arrays = [a[:, ::-1] for a in arrays]
                xs.append(arrays[0]); ts.append(arrays[1]); ls.append(arrays[2])
                ns.append(knee_norm[i] if multi_knee else field_norm[i])
            yield (np.ascontiguousarray(np.stack(xs), np.float32),
                   np.ascontiguousarray(np.stack(ts), np.float32),
                   np.ascontiguousarray(np.stack(ls), np.float32),
                   np.asarray(ns, np.float32))

    signature = (
        tf.TensorSpec((batch_size, crop, crop, n_members * n_bands), tf.float32),
        tf.TensorSpec((batch_size, crop, crop, n_bands), tf.float32),
        tf.TensorSpec((batch_size, crop // 2, crop // 2, n_bands + 1), tf.float32),
        tf.TensorSpec((batch_size, len(knees_e), n_bands) if multi_knee
                      else (batch_size, n_bands), tf.float32))
    dataset = tf.data.Dataset.from_generator(batches, output_signature=signature)
    iterator = iter(dataset.prefetch(4))

    initial = init_params(n_members, n_bands, width, use_lr, best_per_band, seed)
    params = {k: tf.Variable(v, name=k) for k, v in initial.items()}
    variables = [params[k] for k in PARAM_NAMES]
    first = [tf.Variable(tf.zeros_like(v)) for v in variables]
    second = [tf.Variable(tf.zeros_like(v)) for v in variables]
    step_var = tf.Variable(0, dtype=tf.int64)
    border = LOSS_BORDER_PX

    def schedule(step):
        s = tf.cast(step, tf.float32)
        warm = tf.minimum(1.0, (s + 1.0) / float(max(1, warmup_steps)))
        progress_frac = tf.clip_by_value(
            (s - warmup_steps) / float(max(1, steps - warmup_steps)), 0.0, 1.0)
        cosine = 0.01 + 0.99 * 0.5 * (1.0 + tf.cos(math.pi * progress_frac))
        return learning_rate * warm * cosine

    band_scale = tf.constant(scales, tf.float32)

    def knee_crop_mse(y, t):
        """``(B, K, C)`` crop MSE at every loss knee, from band-knee asinh."""
        y_e = tf.sinh(y) * band_scale
        t_e = tf.sinh(t) * band_scale
        per_knee = []
        for q in knees_e:
            err = tf.asinh(y_e / float(q)) - tf.asinh(t_e / float(q))
            per_knee.append(tf.reduce_mean(err * err, axis=[1, 2]))
        return tf.stack(per_knee, axis=1)

    @tf.function(reduce_retracing=True)
    def train_step(x, t, lf, norm):
        with tf.GradientTape() as tape:
            logits = tf_gate_logits(params, x, lf, use_lr)
            y = tf_mix(logits, x, n_members, n_bands, mix_space)
            y_in = y[:, border:-border, border:-border]
            t_in = t[:, border:-border, border:-border]
            if multi_knee:
                crop_mse = knee_crop_mse(y_in, t_in)                 # (B, K, C)
            else:
                err = y_in - t_in
                crop_mse = tf.reduce_mean(err * err, axis=[1, 2])    # (B, C)
            loss = tf.reduce_mean(crop_mse / norm)
        grads = tape.gradient(loss, variables)
        step_var.assign_add(1)
        lr_now = schedule(step_var)
        t1 = tf.cast(step_var, tf.float32)
        for v, g, m1, m2 in zip(variables, grads, first, second, strict=True):
            m1.assign(0.9 * m1 + 0.1 * g)
            m2.assign(0.999 * m2 + 0.001 * g * g)
            m1_hat = m1 / (1.0 - 0.9 ** t1)
            m2_hat = m2 / (1.0 - 0.999 ** t1)
            v.assign_sub(lr_now * m1_hat / (tf.sqrt(m2_hat) + 1e-8))
        return loss

    @tf.function(reduce_retracing=True)
    def predict(x, lf):
        return tf_mix(tf_gate_logits(params, x, lf, use_lr), x, n_members, n_bands,
                      mix_space)

    holdout = []
    for f in holdout_fields:
        members = np.arcsinh(f.members_e() / scales.astype(np.float32))
        target = np.arcsinh(f.target_e / scales.astype(np.float32))
        h, w = f.shape
        best_mse = np.asarray([np.mean((members[m, ..., c] - target[..., c]) ** 2)
                               for c, m in enumerate(best_per_band)])
        holdout.append((
            members.transpose(1, 2, 0, 3).reshape(1, h, w, -1).astype(np.float16),
            target, f.lr_feats(scales)[None], np.maximum(best_mse, 1e-12),
            reference_knee_mse(f) if multi_knee else None,
            np.asarray(f.target_e, np.float32)))

    def evaluate() -> dict:
        relative, vis_psnr, band_psnr, integrated = [], [], [], []
        for x, t, lf, best_mse, best_knee_mse, target_e in holdout:
            y = predict(tf.constant(x, tf.float32), tf.constant(lf, tf.float32))[0].numpy()
            mse = ((y - t) ** 2).mean(axis=(0, 1))
            y_e = np.sinh(y) * scales.astype(np.float32)
            curve = knee_psnr(y_e, target_e, knees=MONITOR_KNEES)       # (K, C)
            integrated.append(integrated_psnr(curve, MONITOR_KNEES))
            if multi_knee:
                knee_mse = np.stack([
                    np.mean((np.arcsinh(y_e / q) - np.arcsinh(target_e / q)) ** 2,
                            axis=(0, 1)) for q in knees_e])
                relative.append(knee_mse / best_knee_mse)
            else:
                relative.append(mse / best_mse)
            band_psnr.append([_stretched_psnr(float(v)) for v in mse])
            vis_psnr.append(band_psnr[-1][0])
        return {"loss": float(np.mean(relative)),
                "band_psnr": np.mean(band_psnr, axis=0).tolist(),
                "vis_psnr": float(np.mean(vis_psnr)),
                "integrated_psnr": np.mean(integrated, axis=0).tolist(),
                "vis_integrated_psnr": float(np.mean(integrated, axis=0)[0])}

    def build(step: int, *, complete: bool) -> SpatialGateCombiner:
        """The gate at the best checkpoint so far, as of ``step``."""
        return SpatialGateCombiner(
            member_labels=all_labels, params=dict(snapshot), width=int(width),
            use_lr=bool(use_lr), band_names=names, val_l1=None, kind=SPATIAL_GATE_KIND,
            mix_space=mix_space,
            active_members=None if len(active) == len(all_labels) else tuple(active),
            fit_meta={
                "model": f"convolutional member gate, convex {mix_space} mixture",
                "mix_space": mix_space,
                "active_member_labels": labels,
                "width": int(width), "use_lr": bool(use_lr),
                "dilations": list(DILATIONS),
                "loss": ("per-field relative asinh MSE: each field and band's MSE "
                         "over the best member's, averaged (1.0 = best member)"
                         + ("; at every loss knee, averaged over knees" if multi_knee else "")),
                "loss_knees_e": [float(q) for q in knees_e] if multi_knee else None,
                "best_member_per_band": [labels[m] for m in best_per_band],
                "member_train_mse": member_mse.tolist(),
                "steps": int(steps), "steps_run": int(step), "complete": bool(complete),
                "batch_size": int(batch_size), "crop": int(crop),
                "learning_rate": float(learning_rate),
                "uniform_crop_fraction": float(uniform_crop_fraction),
                "train_fields": [f.index for f in train_fields],
                "train_field_tags": [f.tag for f in train_fields],
                "holdout_fields": [f.index for f in holdout_fields],
                "baseline_holdout": baseline, "selected": dict(best),
                "history": list(history), "fit_seconds": float(time.time() - started),
            })

    snapshot = {k: v.numpy().copy() for k, v in params.items()}
    baseline = evaluate()
    best = dict(baseline, step=0)
    history = [dict(baseline, step=0, train_loss=None)]
    say(f"step 0 (best member per band): held-out loss {baseline['loss']:.4f} "
        f"VIS PSNR {baseline['vis_psnr']:.3f} dB, integrated "
        f"{baseline['vis_integrated_psnr']:.3f} dB")
    running = []
    for step in range(1, int(steps) + 1):
        x, t, lf, norm = next(iterator)
        running.append(float(train_step(x, t, lf, norm)))
        if step % int(eval_every) == 0 or step == int(steps):
            metrics = evaluate()
            row = dict(metrics, step=step, train_loss=float(np.mean(running)))
            history.append(row)
            running = []
            improved = metrics["loss"] < best["loss"]
            if improved:
                best = dict(metrics, step=step)
                snapshot = {k: v.numpy().copy() for k, v in params.items()}
            say(f"step {step}: train {row['train_loss']:.4f} held-out {metrics['loss']:.4f} "
                f"VIS PSNR {metrics['vis_psnr']:.3f} dB, integrated "
                f"{metrics['vis_integrated_psnr']:.3f} dB{'  *' if improved else ''} "
                f"({time.time() - started:.0f}s)")
            if improved and checkpoint is not None:
                checkpoint(build(step, complete=False))
        if progress is not None:
            progress(step, int(steps), f"spatial gate step {step}")

    cache.cleanup()
    return build(int(steps), complete=True)
