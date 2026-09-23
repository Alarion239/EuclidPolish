"""Spatial gating combiner: a small convolutional gate over ensemble members.

The gate looks at a neighbourhood of every member's super-resolution (and,
optionally, the LR input plus a mask of its zeroed pixels) and returns, per
pixel and per band, a softmax weight over members. The output is the weighted
mean of the member predictions, so it is convex by construction: every output
value lies between the smallest and largest member value at that pixel and
band. The mean is taken in electrons (``mix_space="linear"``: knee-free and
flux-conserving) or in per-band asinh space (``"asinh"``, the original gates:
arithmetic below the band knee, geometric above it). The gate always *sees*
the members in asinh space; that only compresses its input range.

Network (``F`` = ``width`` channels, ``M`` members, ``C`` bands)::

    members asinh (H, W, M*C)
      └─ 1x1 conv → F, relu                                   full resolution
          ├─ space-to-depth 2x2 (H/2, W/2, 4F)
          │   + [LR asinh (C), LR-zero mask (1)]              LR resolution
          │   └─ 1x1 conv → F, relu
          │       └─ 4 residual 3x3 convs, dilation 1, 2, 4, 8, relu
          │           └─ bilinear 2x upsample (half-pixel, edge clamped)
          └─ concat ──┘ → 1x1 conv → F, relu → 1x1 conv → M*C logits
    softmax over members per band → weighted mean (electrons or asinh) → electrons

The receptive field is about 62 HR pixels, wide enough to see a bright star's
halo from any pixel in it. Every operation is written out explicitly (symmetric
padding, a fixed upsampling stencil) so this NumPy forward pass reproduces the
TensorFlow training graph in :mod:`euclid_polish.eval.spatial_gate_fit`
exactly, and inference stays free of TensorFlow.
"""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import binary_dilation

from euclid_polish.config import Config

SPATIAL_GATE_KIND = "spatial_gate"
SPATIAL_GATE_SCHEMA = 1
DILATIONS = (1, 2, 4, 8)
#: Spaces the member predictions can be averaged in (see the module docstring).
MIX_LINEAR = "linear"
MIX_ASINH = "asinh"
MIX_SPACES = (MIX_LINEAR, MIX_ASINH)
LR_MASK_DILATION_PX = 2
BAND_NAMES = tuple(Config.HR_TARGET_BAND_NAMES)

#: Parameter names, in the order the fitter creates them.
PARAM_NAMES = (
    "w_in", "b_in", "w_merge", "b_merge",
    *(name for d in DILATIONS for name in (f"w_d{d}", f"b_d{d}")),
    "w_fuse", "b_fuse", "w_out", "b_out",
)


def band_scales(band_names=BAND_NAMES) -> np.ndarray:
    """Per-band asinh knee in electrons (the space the gate sees members in)."""
    return np.asarray([Config.get_band(name).asinh_stretch_scale_e
                       for name in band_names], np.float64)


def member_features(members_e: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """``(M, H, W, C)`` electrons → ``(H, W, M*C)`` asinh, member-major."""
    stack = np.asarray(members_e, np.float32)
    if stack.ndim != 4:
        raise ValueError(f"expected (M,H,W,C) members, got {stack.shape}")
    m, h, w, c = stack.shape
    asinh = np.arcsinh(stack / np.asarray(scales, np.float32)[None, None, None, :])
    return np.ascontiguousarray(asinh.transpose(1, 2, 0, 3).reshape(h, w, m * c))


def lr_features(lr_e: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """``(h, w, C)`` LR electrons → ``(h, w, C+1)``: asinh LR and a mask of
    pixels zeroed in any band (blackouts), dilated by a couple of pixels."""
    lr = np.asarray(lr_e, np.float32)
    if lr.ndim != 3:
        raise ValueError(f"expected (h,w,C) LR, got {lr.shape}")
    asinh = np.arcsinh(lr / np.asarray(scales, np.float32)[None, None, :])
    mask = binary_dilation(np.any(lr == 0.0, axis=-1),
                           iterations=LR_MASK_DILATION_PX)
    return np.concatenate((asinh, mask[..., None].astype(np.float32)), axis=-1)


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0, out=x)


def _conv1x1(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    h, wd, c = x.shape
    out = x.reshape(-1, c) @ w
    out += b
    return out.reshape(h, wd, w.shape[1])


def _conv3x3(x: np.ndarray, w: np.ndarray, b: np.ndarray, dilation: int) -> np.ndarray:
    """3x3 dilated conv with symmetric padding (``SYMMETRIC`` in TF)."""
    d = int(dilation)
    h, wd, c = x.shape
    padded = np.pad(x, ((d, d), (d, d), (0, 0)), mode="symmetric")
    out = np.empty((h * wd, w.shape[3]), np.float32)
    out[:] = b
    for ky in range(3):
        for kx in range(3):
            tap = padded[ky * d:ky * d + h, kx * d:kx * d + wd]
            out += tap.reshape(-1, c) @ w[ky, kx]
    return out.reshape(h, wd, w.shape[3])


def space_to_depth(x: np.ndarray) -> np.ndarray:
    """``(H, W, C)`` → ``(H/2, W/2, 4C)``, channel order ``(dy, dx, c)``."""
    h, w, c = x.shape
    return (x.reshape(h // 2, 2, w // 2, 2, c)
            .transpose(0, 2, 1, 3, 4).reshape(h // 2, w // 2, 4 * c))


def upsample2x(x: np.ndarray) -> np.ndarray:
    """Half-pixel bilinear 2x upsampling with edge clamping, separable."""
    def along(a: np.ndarray, axis: int) -> np.ndarray:
        n = a.shape[axis]
        prev = np.take(a, np.r_[0, np.arange(n - 1)], axis=axis)
        nxt = np.take(a, np.r_[np.arange(1, n), n - 1], axis=axis)
        even = 0.75 * a + 0.25 * prev
        odd = 0.75 * a + 0.25 * nxt
        out = np.stack((even, odd), axis=axis + 1)
        shape = list(a.shape)
        shape[axis] = 2 * n
        return out.reshape(shape)
    return along(along(x, 0), 1)


def gate_logits(params: dict, members_asinh: np.ndarray,
                lr_feats: np.ndarray | None) -> np.ndarray:
    """``(H, W, M*C)`` member logits; ``H`` and ``W`` must be even."""
    feat = _relu(_conv1x1(members_asinh, params["w_in"], params["b_in"]))
    coarse = space_to_depth(feat)
    if lr_feats is not None:
        coarse = np.concatenate((coarse, lr_feats), axis=-1)
    hidden = _relu(_conv1x1(coarse, params["w_merge"], params["b_merge"]))
    for d in DILATIONS:
        hidden = hidden + _relu(_conv3x3(hidden, params[f"w_d{d}"],
                                         params[f"b_d{d}"], d))
    fused = np.concatenate((feat, upsample2x(hidden)), axis=-1)
    fused = _relu(_conv1x1(fused, params["w_fuse"], params["b_fuse"]))
    return _conv1x1(fused, params["w_out"], params["b_out"])


def _pad_even(x: np.ndarray, axis_h: int, axis_w: int) -> np.ndarray:
    pad = [(0, 0)] * x.ndim
    pad[axis_h] = (0, x.shape[axis_h] % 2)
    pad[axis_w] = (0, x.shape[axis_w] % 2)
    return np.pad(x, pad, mode="edge") if any(p[1] for p in pad) else x


@dataclass
class SpatialGateCombiner:
    """Fitted convolutional member gate (see the module docstring).

    ``member_labels`` is the whole ensemble the gate was fitted for (the
    staleness key); ``active_members`` are the positions it actually reads
    (``None`` = all). A pruned gate needs only those members' SR, so callers
    can skip running the others (:meth:`needed_member_indices`)."""

    member_labels: list[str]
    params: dict[str, np.ndarray]
    width: int
    use_lr: bool
    band_names: tuple[str, ...] = BAND_NAMES
    active_members: tuple[int, ...] | None = None
    records_fp: str | None = None
    starfull: bool = True
    val_l1: float | None = None
    kind: str = SPATIAL_GATE_KIND
    fit_meta: dict = field(default_factory=dict)
    mix_space: str = MIX_ASINH

    def __post_init__(self) -> None:
        if self.mix_space not in MIX_SPACES:
            raise ValueError(f"mix_space must be one of {MIX_SPACES}, got {self.mix_space!r}")

    @property
    def n_kernels(self) -> int:
        """Parameter count (the RBF combiners report kernels here)."""
        return int(sum(np.asarray(v).size for v in self.params.values()))

    @property
    def weight_labels(self) -> list[str]:
        return list(self.member_labels)

    @property
    def active(self) -> list[int]:
        return (list(range(len(self.member_labels))) if self.active_members is None
                else [int(i) for i in self.active_members])

    def _prepare(self, preds: np.ndarray, lr: np.ndarray | None):
        """Accepts the whole ensemble's stack or just the active members'."""
        stack = np.asarray(preds, np.float32)
        if stack.ndim != 4:
            raise ValueError(f"expected (M,H,W,C) member stack, got {stack.shape}")
        active = self.active
        if stack.shape[0] == len(self.member_labels) and len(active) != len(self.member_labels):
            stack = stack[active]
        m, h, w, c = stack.shape
        if m != len(active) or c != len(self.band_names):
            raise ValueError(
                f"spatial gate expects {len(self.member_labels)} (or {len(active)} active) "
                f"members and {len(self.band_names)} bands, got {m} members and {c} bands")
        scales = band_scales(self.band_names)
        members = _pad_even(member_features(stack, scales), 0, 1)
        lr_feats = None
        if self.use_lr:
            if lr is None:
                raise ValueError("this spatial gate was fitted with the LR input; "
                                 "pass lr= to apply it")
            lr_arr = np.asarray(lr, np.float32)
            if lr_arr.ndim == 2:
                lr_arr = lr_arr[..., None]
            want = (members.shape[0] // 2, members.shape[1] // 2)
            if lr_arr.shape[-1] != c:
                raise ValueError(f"LR has {lr_arr.shape[-1]} bands, expected {c}")
            lr_arr = _pad_even(lr_arr, 0, 1)[:want[0], :want[1]]
            if lr_arr.shape[:2] != want:
                raise ValueError(
                    f"LR shape {lr_arr.shape[:2]} is not half the SR grid {want}")
            lr_feats = lr_features(lr_arr, scales)
        return stack, members, lr_feats

    def _weights(self, stack: np.ndarray, members: np.ndarray,
                 lr_feats: np.ndarray | None) -> np.ndarray:
        """``(H, W, M, C)`` softmax weights over the active members."""
        m, h, w, c = stack.shape
        logits = gate_logits(self.params, members, lr_feats)[:h, :w]
        logits = logits.reshape(h, w, m, c)
        logits -= logits.max(axis=2, keepdims=True)
        np.exp(logits, out=logits)
        logits /= logits.sum(axis=2, keepdims=True)
        return logits

    def weights_field(self, preds: np.ndarray, *,
                      lr: np.ndarray | None = None) -> np.ndarray:
        """``(H, W, M, C)`` convex weights over the whole ensemble (pruned
        members get zero) for one field."""
        stack, members, lr_feats = self._prepare(preds, lr)
        logits = self._weights(stack, members, lr_feats)
        m, h, w, c = stack.shape
        if m == len(self.member_labels):
            return logits
        full = np.zeros((h, w, len(self.member_labels), c), np.float32)
        full[:, :, self.active] = logits
        return full

    def apply_field(self, preds: np.ndarray,
                    band_names: tuple[str, ...] | None = None, *,
                    lr: np.ndarray | None = None) -> np.ndarray:
        """``(M, H, W, C)`` member electrons → ``(H, W, C)`` electrons."""
        if band_names is not None and tuple(band_names) != tuple(self.band_names):
            raise ValueError(f"spatial gate expects bands {self.band_names}")
        stack, members, lr_feats = self._prepare(preds, lr)
        weights = self._weights(stack, members, lr_feats)
        m, h, w, c = stack.shape
        if self.mix_space == MIX_LINEAR:
            return np.einsum("hwmc,mhwc->hwc", weights, stack,
                             optimize=True).astype(np.float32)
        mixed = np.einsum("hwmc,hwmc->hwc", weights,
                          members[:h, :w].reshape(h, w, m, c), optimize=True)
        return (np.sinh(mixed) * band_scales(self.band_names)[None, None, :]
                ).astype(np.float32)

    def upsample(self, ens, lr_array: np.ndarray) -> np.ndarray:
        """SR of one LR field, running only the members the gate reads."""
        return self.apply_field(
            ens.member_arrays(lr_array, indices=self.needed_member_indices()),
            lr=lr_array)

    def needed_member_indices(self) -> list[int]:
        return self.active

    def member_pruned(self, index: int) -> bool:
        return int(index) not in self.active

    def without_member(self, index: int) -> SpatialGateCombiner:
        """The same gate after a member it does not read leaves the ensemble."""
        index = int(index)
        if not self.member_pruned(index):
            raise ValueError("a spatial gate must be refitted when a member it reads leaves")
        labels = [lbl for i, lbl in enumerate(self.member_labels) if i != index]
        active = tuple(i - (i > index) for i in self.active)
        return dataclasses.replace(self, member_labels=labels, active_members=active)

    def surviving_members(self) -> dict[str, list[bool]]:
        active = set(self.active)
        return {"source": [i in active for i in range(len(self.member_labels))]}


def save_spatial_gate(comb: SpatialGateCombiner, directory: str) -> None:
    """Write ``combiner.json`` + ``combiner.npz`` (the combiner artifact pair)."""
    os.makedirs(directory, exist_ok=True)
    np.savez_compressed(os.path.join(directory, "combiner.npz"),
                        **{k: np.asarray(v, np.float32) for k, v in comb.params.items()})
    manifest = {
        "schema": SPATIAL_GATE_SCHEMA,
        "kind": SPATIAL_GATE_KIND,
        "member_labels": list(comb.member_labels),
        "band_names": list(comb.band_names),
        "width": int(comb.width),
        "use_lr": bool(comb.use_lr),
        "mix_space": comb.mix_space,
        "active_members": (None if comb.active_members is None
                           else [int(i) for i in comb.active_members]),
        "dilations": list(DILATIONS),
        "records_fp": comb.records_fp,
        "starfull": bool(comb.starfull),
        "val_l1": comb.val_l1,
        "fit_meta": comb.fit_meta,
    }
    with open(os.path.join(directory, "combiner.json"), "w") as handle:
        json.dump(manifest, handle, indent=2)


def load_spatial_gate(directory: str, *, member_labels: list[str] | None = None
                      ) -> SpatialGateCombiner | None:
    """The saved gate, or ``None`` when absent, stale for ``member_labels``,
    or written by an incompatible schema."""
    manifest_path = os.path.join(directory, "combiner.json")
    arrays_path = os.path.join(directory, "combiner.npz")
    if not (os.path.isfile(manifest_path) and os.path.isfile(arrays_path)):
        return None
    try:
        with open(manifest_path) as handle:
            manifest = json.load(handle)
        if (manifest.get("kind") != SPATIAL_GATE_KIND
                or int(manifest.get("schema", 0)) != SPATIAL_GATE_SCHEMA
                or list(manifest.get("dilations", [])) != list(DILATIONS)):
            return None
        labels = [str(v) for v in manifest["member_labels"]]
        if member_labels is not None and labels != [str(v) for v in member_labels]:
            return None
        with np.load(arrays_path) as arrays:
            params = {name: np.asarray(arrays[name], np.float32)
                      for name in PARAM_NAMES}
        active = manifest.get("active_members")
        return SpatialGateCombiner(
            member_labels=labels, params=params,
            width=int(manifest["width"]), use_lr=bool(manifest["use_lr"]),
            active_members=None if active is None else tuple(int(i) for i in active),
            band_names=tuple(manifest.get("band_names", BAND_NAMES)),
            records_fp=manifest.get("records_fp"),
            starfull=bool(manifest.get("starfull", True)),
            val_l1=manifest.get("val_l1"),
            fit_meta=manifest.get("fit_meta", {}),
            mix_space=str(manifest.get("mix_space", MIX_ASINH)))
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None
