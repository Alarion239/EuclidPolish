"""Shared helpers for the 257 grid test (run from the repo root).

Models, tile list, per-window grid statistics with AxisExcess (a93122dd).
"""
import glob
import os
import sys

import numpy as np
from scipy import ndimage

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))

import tensorflow as tf  # noqa: E402

from euclid_polish.training.grid_penalty import default_detector  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = "data/jwst_euclid_overlap/resolution_cache"
RECORDS = ("data/_fasrc_cache/n/netscratch/lconnor_lab/Lab/abelotserkovtsev/EuclidPolish/"
           "data/images/records_v2")
SRC = os.path.join(HERE, "sr_cache")          # real tiles: VIS + H float32 (2h, 2w, 2)
STATS = os.path.join(HERE, "stats")           # per-window statistics (all 4 bands)
BANDS = ("VIS", "Y_E", "J_E", "H_E")
W = 64
BORDER = 32
MODELS = ["milestone:member_257@15000000", "universal:member_238", "universal:member_227",
          "milestone:member_257@1000000", "milestone:member_257@8000000"]
NAMES = {"milestone:member_257@15000000": "257@15M", "universal:member_238": "238",
         "universal:member_227": "227", "milestone:member_257@1000000": "257@1M",
         "milestone:member_257@8000000": "257@8M", "truth": "truth", "jwst": "JWST F200W",
         "lr": "LR (cubic x2)", "white": "white noise"}
DET = default_detector()


def slug(spec):
    return spec.replace(":", "_").replace("@", "_")


_MODEL = {}


def load_model(spec):
    """One model at a time (frees the previous one)."""
    if spec in _MODEL:
        return _MODEL[spec]
    _MODEL.clear()
    tf.keras.backend.clear_session()
    from euclid_polish.ensemble import EnsembleModel
    from euclid_polish.web.helpers import milestones, model_catalog
    if spec.startswith("milestone:"):
        m = milestones.load(*milestones.parse_spec(spec))
    else:
        item = model_catalog.resolve_spec(spec)
        m = EnsembleModel(model_catalog.ensemble_dir(), labels=[item.reads[0]],
                          include_universal=True, starless=False)
    _MODEL[spec] = m
    return m


def run_sr(spec, lr):
    m = load_model(spec)
    return np.asarray(m.member_arrays(np.ascontiguousarray(lr, np.float32))[0], np.float32)


# ---------------------------------------------------------------- tiles


def clean_tiles():
    """NEXUS tiles with no blank (0) LR pixel in any band, VIS < 5e4 and
    NISP < 5e3 e- (the 10-17 'clean tile' definition)."""
    out = []
    for f in sorted(glob.glob(os.path.join(CACHE, "f200w-*.npz"))):
        with np.load(f) as d:
            lr = d["lr"]
        if not np.isfinite(lr).all() or (lr == 0).any():
            continue
        if lr[..., 0].max() >= 5e4 or lr[..., 1:].max() >= 5e3:
            continue
        out.append(os.path.basename(f)[:-4])
    return out


def tile(tid):
    with np.load(os.path.join(CACHE, f"{tid}.npz")) as d:
        return d["lr"].astype(np.float32), d["jwst"].astype(np.float32)


def noise_sigma(x):
    """Clipped MAD of a set of pixels (as scripts/jwst_resolution._noise_sigma)."""
    values = np.asarray(x, np.float64)
    values = values[np.isfinite(values)].ravel()
    sigma = 1.4826 * float(np.median(np.abs(values - np.median(values))))
    for _ in range(3):
        core = values[np.abs(values - np.median(values)) < 3.0 * sigma]
        sigma = 1.4826 * float(np.median(np.abs(core - np.median(core))))
    return sigma


# ---------------------------------------------------------------- detector


def highpass(img):
    """(H, W, C) -> high-passed (H, W, C) float32 (the detector's own, mirror-padded)."""
    return DET.highpass(img[None].astype(np.float32)).numpy()[0]


def window_stats(img, origins):
    """z, amplitude, score per window of the full image's high-pass.

    ``img`` (H, W, C); ``origins`` [(y0, x0)] -> dict of (N, C) arrays."""
    hp = highpass(img)
    wins = np.stack([hp[y:y + W, x:x + W, :] for y, x in origins])      # (N, w, w, C)
    wins = np.transpose(wins, (0, 3, 1, 2))                              # (N, C, w, w)
    on, pred = DET.ring_power(wins)
    excess, spread = DET._excess_and_spread(on, pred)
    excess, spread = excess.numpy(), spread.numpy()
    on_s = tf.reduce_sum(on, -1).numpy()
    pred_s = tf.reduce_sum(pred, -1).numpy()
    amp = np.sign(excess) * np.sqrt(np.abs(excess) / DET._norm)
    return {"z": excess / spread, "amp": amp, "score": on_s / (pred_s + 1e-30)}


def protocol_z(img, border):
    """The reconstructed protocol: drop ``border`` px, then the detector's own
    non-overlapping 64-px windows -> z (nh, nw, C)."""
    h, w = img.shape[:2]
    crop = img[border:h - border, border:w - border] if border else img
    return DET.zscore(crop[None].astype(np.float32)).numpy()[0]


def stride_origins(h, w, lo=0, step=32):
    """Window origins on a stride-``step`` grid with windows inside [lo, size - lo)."""
    ys = list(range(lo, h - lo - W + 1, step))
    xs = list(range(lo, w - lo - W + 1, step))
    return [(y, x) for y in ys for x in xs]
