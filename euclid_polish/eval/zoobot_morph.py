"""Pure helpers for the Zoobot before/after morphology evaluation.

These functions deliberately avoid any PyTorch / Zoobot import so they can be
unit-tested in the main (TensorFlow) environment. The model-driven half — which
runs Zoobot locally in its own isolated PyTorch env — lives in
``scripts/zoobot_morphology.py`` and calls into here for everything except
the forward pass.

The morphology evaluation compares the model's effect on the VIS plane:

  * **before** — the dirty LR VIS image (``original_stack.fits`` band 0)
  * **after**  — the super-resolved VIS image (``SR.fits`` band 0)
  * **hr**     — the HR ground truth VIS, when present (``HR.fits`` band 0),
    available for synthetic eval outputs

Each is rendered to an 8-bit PNG (what Zoobot's image loader expects), Zoobot
produces a per-image vector (a representation embedding by default, or Galaxy
Zoo vote fractions with a finetuned tree model), and we score how the vectors
move: ``before → after`` distance, and — when HR is available — whether SR moves
the vector *toward* the HR ground truth (the cleanest validation that SR
improves morphology recovery rather than merely changing it).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

# VIS plane index inside the multi-band FITS cubes the eval pipeline writes
# (LR_INPUT_BAND_NAMES = VIS, Y_E, J_E, H_E → band 0 is VIS).
_VIS_PLANE = 0


# --------------------------------------------------------------------------- #
# Object discovery
# --------------------------------------------------------------------------- #

def discover_objects(run_dir: str) -> List[Dict[str, Any]]:
    """Find per-object eval outputs under a run directory.

    A run directory (from ``scripts/fasrc_eval_catalog.py``) has one
    sub-directory per object containing ``SR.fits`` and ``original_stack.fits``
    (and optionally ``HR.fits`` for synthetic targets). Returns a list of
    ``{"id", "before", "after", "hr"}`` dicts (``hr`` is ``None`` when absent),
    sorted by id. Sub-directories without an ``SR.fits`` are skipped.
    """
    out: List[Dict[str, Any]] = []
    if not os.path.isdir(run_dir):
        return out
    for name in sorted(os.listdir(run_dir)):
        obj_dir = os.path.join(run_dir, name)
        if not os.path.isdir(obj_dir):
            continue
        sr = os.path.join(obj_dir, "SR.fits")
        if not os.path.isfile(sr):
            continue
        before = os.path.join(obj_dir, "original_stack.fits")
        hr = os.path.join(obj_dir, "HR.fits")
        out.append({
            "id":     name,
            "dir":    obj_dir,
            "after":  sr,
            "before": before if os.path.isfile(before) else None,
            "hr":     hr if os.path.isfile(hr) else None,
        })
    return out


# --------------------------------------------------------------------------- #
# Image preparation (FITS VIS plane → 8-bit PNG for Zoobot)
# --------------------------------------------------------------------------- #

def load_vis_plane(fits_path: str) -> np.ndarray:
    """Return the VIS plane (band 0) of a FITS cube as a 2-D float32 array."""
    from astropy.io import fits  # local import keeps this module light

    with fits.open(fits_path) as hdul:
        data = np.asarray(hdul[0].data, dtype=np.float32)
    if data.ndim == 3:
        return data[_VIS_PLANE]
    return data


def stretch_to_uint8(
    plane: np.ndarray,
    *,
    asinh_scale: float,
    pmin: float = 1.0,
    pmax: float = 99.5,
) -> np.ndarray:
    """asinh-stretch + percentile-normalise a 2-D image to ``uint8`` [0, 255].

    The asinh knee (``asinh_scale``, electrons) matches the rendering the rest
    of the pipeline uses; percentile clipping makes the contrast robust to a
    handful of bright pixels. Non-finite pixels are treated as the floor.
    """
    a = np.arcsinh(np.nan_to_num(plane, nan=0.0, posinf=0.0, neginf=0.0)
                   / float(asinh_scale))
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return np.zeros(plane.shape, dtype=np.uint8)
    lo, hi = np.percentile(finite, [pmin, pmax])
    if hi <= lo:
        hi = lo + 1e-6
    norm = np.clip((a - lo) / (hi - lo), 0.0, 1.0)
    return (norm * 255.0 + 0.5).astype(np.uint8)


def render_vis_png(
    fits_path: str,
    out_png: str,
    *,
    asinh_scale: float,
    size: int = 424,
) -> str:
    """Render a FITS VIS plane to a square 3-channel 8-bit PNG for Zoobot.

    Zoobot's loader reads ordinary image files and resizes internally, so the
    exact ``size`` only needs to be large enough to preserve structure; 424 px
    matches Galaxy-Zoo-style inputs. Returns ``out_png``.
    """
    from PIL import Image

    plane = load_vis_plane(fits_path)
    u8 = stretch_to_uint8(plane, asinh_scale=asinh_scale)
    img = Image.fromarray(u8, mode="L").convert("RGB")
    if size and (img.width != size or img.height != size):
        img = img.resize((size, size), Image.BILINEAR)
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    img.save(out_png)
    return out_png


# --------------------------------------------------------------------------- #
# Vector comparison metrics (work for representations OR vote fractions)
# --------------------------------------------------------------------------- #

def _l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(1.0 - np.dot(a, b) / (na * nb))


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def vector_deltas(
    before: Sequence[float],
    after: Sequence[float],
    ref: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """Compare ``before``/``after`` (and optionally an ``hr`` reference) vectors.

    Returns L2 distance, cosine distance and Pearson correlation between the
    before and after vectors. When ``ref`` (the HR ground-truth vector) is
    given, also returns ``l2_before_ref`` / ``l2_after_ref``, the signed
    ``ref_improvement`` (positive ⇒ SR moved the vector closer to HR) and a
    boolean ``closer_to_ref``.
    """
    b = np.asarray(before, dtype=np.float64)
    a = np.asarray(after, dtype=np.float64)
    out: Dict[str, Any] = {
        "l2_before_after":     _l2(b, a),
        "cosine_before_after": _cosine_distance(b, a),
        "pearson_before_after": _pearson(b, a),
    }
    if ref is not None:
        r = np.asarray(ref, dtype=np.float64)
        d_before = _l2(b, r)
        d_after = _l2(a, r)
        out.update({
            "l2_before_ref":  d_before,
            "l2_after_ref":   d_after,
            "ref_improvement": d_before - d_after,   # >0 ⇒ closer after SR
            "closer_to_ref":  bool(d_after < d_before),
        })
    return out


def write_morph_manifest(path: str, rows: List[Dict[str, Any]]) -> None:
    """Write morphology rows to CSV. Column set is the union of all row keys
    (``id`` first), so representation-mode and vote-mode runs both serialise
    cleanly."""
    import csv

    if not rows:
        # Still write a header-only file so consumers see an (empty) manifest.
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(["id"])
        return
    keys: List[str] = ["id"]
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# --------------------------------------------------------------------------- #
# Summary plot (run-level) — reads the manifest + raw predictions
# --------------------------------------------------------------------------- #

def _read_predictions(pred_path: str):
    """Read zoobot_predictions.csv → (ids, vectors). Non-``id_str`` columns are
    the per-image vector (representation feats or vote fractions)."""
    import csv

    import numpy as np
    with open(pred_path) as f:
        rdr = csv.reader(f)
        header = next(rdr)
        id_idx = header.index("id_str")
        vec_idx = [i for i, h in enumerate(header) if h != "id_str"]
        ids, vecs = [], []
        for row in rdr:
            ids.append(row[id_idx])
            vecs.append([float(row[i]) for i in vec_idx])
    return ids, np.asarray(vecs, dtype=np.float64)


def render_morphology_summary(run_dir: str, out_png: str) -> Optional[str]:
    """Render a run-level before/after morphology summary PNG → ``out_png``.

    Panels (drawn from what's available): (1) the distribution of the
    before↔after similarity (Pearson r) across the run; (2) a 2-D PCA "shift
    map" of the raw Zoobot vectors with an arrow per object from its LR (before)
    point to its SR (after) point; (3) the actual before|after images Zoobot
    classified for the most-changed object. Returns ``out_png`` or ``None`` when
    there's no ``morphology_manifest.csv`` yet.
    """
    import csv

    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mani_path = os.path.join(run_dir, "morphology_manifest.csv")
    if not os.path.isfile(mani_path):
        return None
    mani = [r for r in csv.DictReader(open(mani_path)) if r.get("id")]
    if not mani:
        return None

    def _col(name):
        return np.array([float(r[name]) for r in mani
                         if r.get(name) not in (None, "")])
    pear = _col("pearson_before_after")
    l2 = _col("l2_before_after")
    mode = mani[0].get("mode", "?")
    by_id = {r["id"]: r for r in mani}

    # Optional PCA embedding from the raw predictions.
    pred_path = os.path.join(run_dir, "zoobot_predictions.csv")
    emb = None
    if os.path.isfile(pred_path):
        ids, vecs = _read_predictions(pred_path)
        view = {}
        for s, v in zip(ids, vecs):
            obj, _, vw = s.rpartition("__")
            view.setdefault(obj, {})[vw] = v
        objs = [o for o, d in view.items() if "before" in d and "after" in d]
        if objs:
            Xb = np.array([view[o]["before"] for o in objs])
            Xa = np.array([view[o]["after"] for o in objs])
            X = np.vstack([Xb, Xa]); mu = X.mean(0)
            _, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
            P = Vt[:2]
            emb = {
                "objs": objs,
                "pb": (Xb - mu) @ P.T, "pa": (Xa - mu) @ P.T,
                "var": (S ** 2 / (S ** 2).sum())[:2] * 100,
            }

    # An example: the most-changed object whose before/after PNGs exist.
    example = None
    order = sorted(by_id, key=lambda i: -float(by_id[i].get("l2_before_after") or 0))
    for oid in order:
        bp = os.path.join(run_dir, oid, "zoobot", "before.png")
        ap = os.path.join(run_dir, oid, "zoobot", "after.png")
        if os.path.isfile(bp) and os.path.isfile(ap):
            example = (oid, bp, ap)
            break

    ncols = 1 + int(emb is not None) + int(example is not None)
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5.2), squeeze=False)
    ax = list(axes[0])
    k = 0

    # Panel 1 — similarity distribution.
    axes_hist = ax[k]; k += 1
    axes_hist.hist(pear, bins=min(15, max(4, len(pear))), color="#2a5db0", alpha=.85)
    axes_hist.axvline(pear.mean(), color="#b03a3a", ls="--",
                      label=f"mean = {pear.mean():.3f}")
    axes_hist.set_xlabel("Pearson r (before vs after)")
    axes_hist.set_ylabel("# objects")
    axes_hist.set_title("Morphology preserved?\n(1 = identical)")
    axes_hist.legend(fontsize=9)

    # Panel 2 — PCA shift map.
    if emb is not None:
        a = ax[k]; k += 1
        a.scatter(emb["pb"][:, 0], emb["pb"][:, 1], facecolors="none",
                  edgecolors="#888", s=55, label="before (LR)")
        a.scatter(emb["pa"][:, 0], emb["pa"][:, 1], color="#2a5db0", s=32,
                  label="after (SR)")
        for i in range(len(emb["objs"])):
            a.annotate("", xy=emb["pa"][i], xytext=emb["pb"][i],
                       arrowprops=dict(arrowstyle="->", color="#b03a3a",
                                       alpha=.55, lw=1.1))
        a.set_xlabel(f"PC1 ({emb['var'][0]:.0f}% var)")
        a.set_ylabel(f"PC2 ({emb['var'][1]:.0f}% var)")
        a.set_title("Shift in Zoobot feature space\n(arrow = LR → SR)")
        a.legend(fontsize=9)

    # Panel 3 — example before|after.
    if example is not None:
        from matplotlib import image as mpimg
        a = ax[k]; k += 1
        oid, bp, ap = example
        bimg, aimg = mpimg.imread(bp), mpimg.imread(ap)
        gap = np.ones((bimg.shape[0], 8) + bimg.shape[2:], dtype=bimg.dtype)
        a.imshow(np.concatenate([bimg, gap, aimg], axis=1))
        a.axis("off")
        d = by_id[oid].get("l2_before_after", "?")
        a.set_title(f"What Zoobot sees: before | after\n{oid[:18]}…  "
                    f"Δl2={float(d):.1f}", fontsize=10)

    fig.suptitle(f"Zoobot morphology before/after — {os.path.basename(run_dir)} "
                 f"({len(mani)} objects, {mode} mode)", fontsize=13)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_png
