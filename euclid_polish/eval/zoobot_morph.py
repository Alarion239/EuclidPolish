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


#: Stable colour per analysis group (the manifest's ``grade`` column).
GROUP_COLORS = {"A": "#2a5db0", "B": "#2e8b57", "C": "#b8860b",
                "syn-lens": "#b03a3a", "syn-gal": "#7a4fb0"}


def _group_color(g) -> str:
    return GROUP_COLORS.get(str(g), "#666666")


def read_grade_map(run_dir: str) -> Dict[str, str]:
    """``id → grade/group`` from a run's ``manifest.csv`` (empty if absent)."""
    import csv
    p = os.path.join(run_dir, "manifest.csv")
    if not os.path.isfile(p):
        return {}
    return {r["id"]: (r.get("grade") or "")
            for r in csv.DictReader(open(p)) if r.get("id")}


def _pad3(coords: np.ndarray) -> np.ndarray:
    """Return coordinates with exactly three columns, padding missing axes."""
    if coords.ndim != 2:
        coords = np.zeros((0, 3), dtype=np.float64)
    if coords.shape[1] >= 3:
        return coords[:, :3]
    return np.pad(coords, ((0, 0), (0, 3 - coords.shape[1])), mode="constant")


def _pca3(X: np.ndarray) -> tuple[np.ndarray, List[float]]:
    Xc = X - X.mean(0)
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    P = Vt[:min(3, Vt.shape[0])]
    coords = _pad3(Xc @ P.T)
    denom = float((S ** 2).sum())
    var = ((S ** 2 / denom)[:3] * 100.0).tolist() if denom > 0 else []
    return coords, (var + [0.0, 0.0, 0.0])[:3]


def _classical_mds3(X: np.ndarray) -> tuple[np.ndarray, List[float]]:
    if len(X) == 0:
        return np.zeros((0, 3), dtype=np.float64), [0.0, 0.0, 0.0]
    # Pairwise squared Euclidean distances via ‖xᵢ‖² + ‖xⱼ‖² − 2·xᵢ·xⱼ. The naive
    # X[:, None, :] - X[None, :, :] materializes an (N, N, D) tensor (≈5.6 GB at
    # N=1046, D=640, then doubled by the square) which spiked the embedding
    # endpoint to tens of GB; this form only needs the N×N Gram + distance
    # matrices (a few MB).
    sq = np.einsum("ij,ij->i", X, X)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
    np.maximum(d2, 0.0, out=d2)                  # clip tiny negatives from roundoff
    n = d2.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ d2 @ J
    vals, vecs = np.linalg.eigh(B)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    pos = np.maximum(vals[:3], 0.0)
    coords = _pad3(vecs[:, :3] * np.sqrt(pos))
    total = float(np.maximum(vals, 0.0).sum())
    var = (pos / total * 100.0).tolist() if total > 0 else [0.0, 0.0, 0.0]
    return coords, (var + [0.0, 0.0, 0.0])[:3]


def morphology_embedding_payload(run_dir: str) -> Optional[Dict[str, Any]]:
    """Build JSON-ready 3-D PCA and classical-MDS embeddings for a Zoobot run.

    The input is ``zoobot_predictions.csv``. Each row is a view vector named
    ``<object-id>__before``, ``<object-id>__after`` or ``<object-id>__hr``.
    PCA and MDS are fit over every available view vector so before/after/HR
    points share one coordinate system per method.
    """
    pred_path = os.path.join(run_dir, "zoobot_predictions.csv")
    if not os.path.isfile(pred_path):
        return None

    ids, X = _read_predictions(pred_path)
    if len(ids) == 0 or X.size == 0:
        return None

    grade_of = read_grade_map(run_dir)
    # P(lens) per object/recon (if the lens-finder has scored this run) drives
    # the PCA marker opacity. Imported lazily to avoid a module-load cycle.
    from euclid_polish.eval.lensfinder_eval import VIEW_TO_RECON, per_object_plens
    plens_by_obj = per_object_plens(run_dir)
    view_rank = {"before": 0, "after": 1, "hr": 2}
    rows = []
    for idx, raw in enumerate(ids):
        obj, sep, view = raw.rpartition("__")
        if not sep:
            obj, view = raw, "vector"
        group = grade_of.get(obj, "")
        recon = VIEW_TO_RECON.get(view)
        pl = plens_by_obj.get(obj, {}).get(recon) if recon else None
        if isinstance(pl, float) and pl != pl:           # NaN → null for JSON
            pl = None
        rows.append({
            "key": raw,
            "id": obj,
            "view": view,
            "group": group,
            "color": _group_color(group),
            "plens": pl,
            "_idx": idx,
        })
    rows.sort(key=lambda r: (r["id"], view_rank.get(r["view"], 99), r["view"]))
    order = [r["_idx"] for r in rows]
    X = X[order]

    points = [{k: v for k, v in r.items() if k != "_idx"} for r in rows]
    keyset = {p["key"] for p in points}
    object_views: Dict[str, set[str]] = {}
    for p in points:
        object_views.setdefault(p["id"], set()).add(p["view"])
    edges = []
    for obj in sorted(object_views):
        if f"{obj}__before" in keyset and f"{obj}__after" in keyset:
            edges.append({
                "from": f"{obj}__before",
                "to": f"{obj}__after",
                "kind": "before-after",
            })
        if f"{obj}__after" in keyset and f"{obj}__hr" in keyset:
            edges.append({
                "from": f"{obj}__after",
                "to": f"{obj}__hr",
                "kind": "after-hr",
            })

    def _method_payload(coords: np.ndarray, variance: List[float]):
        out = []
        for point, xyz in zip(points, coords):
            p = dict(point)
            p["xyz"] = [float(v) for v in xyz]
            out.append(p)
        return {"points": out, "variance_pct": [float(v) for v in variance]}

    def _subset_payload(views: set[str]):
        # PCA fit on just the named views so the axes capture that subset's own
        # variance (LR alone, or SR+HR together). ``points``/``X`` are row-
        # aligned, so selection is index-parallel. Lets the Morphology-space
        # panel compare how the explained-variance structure changes between the
        # LR inputs and the SR+HR outputs, independent of the LR→SR shift.
        idx = [i for i, p in enumerate(points) if p["view"] in views]
        if not idx:
            return {"points": [], "variance_pct": [0.0, 0.0, 0.0]}
        coords, var = _pca3(X[idx])
        out = []
        for point, xyz in zip((points[i] for i in idx), coords):
            p = dict(point)
            p["xyz"] = [float(v) for v in xyz]
            out.append(p)
        return {"points": out, "variance_pct": [float(v) for v in var]}

    pca, pca_var = _pca3(X)
    mds, mds_var = _classical_mds3(X)
    return {
        "run": os.path.basename(run_dir),
        "count": len(points),
        "points": points,
        "edges": edges,
        "embeddings": {
            "pca": _method_payload(pca, pca_var),
            "mds": _method_payload(mds, mds_var),
            "pca_lr": _subset_payload({"before"}),
            "pca_srhr": _subset_payload({"after", "hr"}),
        },
    }


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
    grade_of = read_grade_map(run_dir)
    groups = [g for g in dict.fromkeys(grade_of.get(r["id"], "") for r in mani)]
    multi = len([g for g in groups if g]) > 1

    # Optional 2-D embeddings from the raw predictions.
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
            # HR ground truth exists only for synthetic objects; fold it into the
            # PCA fit so the projection captures the LR→SR→HR geometry and the
            # truth point lands on the same axes as the before/after points.
            hr_objs = [o for o in objs if "hr" in view[o]]
            Xhr = (np.array([view[o]["hr"] for o in hr_objs])
                   if hr_objs else None)
            X = np.vstack([Xb, Xa] + ([Xhr] if Xhr is not None else []))
            n_obj = len(objs)

            def _split_coords(coords):
                coords = coords[:, :2]
                out = {
                    "pb": coords[:n_obj],
                    "pa": coords[n_obj:2 * n_obj],
                    "phr": None,
                }
                if Xhr is not None:
                    out["phr"] = coords[2 * n_obj:]
                return out

            pca_coords, pca_var = _pca3(X)
            mds_coords, mds_var = _classical_mds3(X)
            emb = {
                "objs": objs,
                "hr_objs": hr_objs,
                "pca": {**_split_coords(pca_coords), "var": pca_var[:2]},
                "mds": {**_split_coords(mds_coords), "var": mds_var[:2]},
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

    ncols = 1 + (2 * int(emb is not None)) + int(example is not None)
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5.2), squeeze=False)
    ax = list(axes[0])
    k = 0

    # Panel 1 — Pearson similarity (per-group strip when grouped, else hist).
    p1 = ax[k]; k += 1
    if multi:
        gs = [g for g in groups if g]
        for xi, g in enumerate(gs):
            ys = [float(r["pearson_before_after"]) for r in mani
                  if grade_of.get(r["id"], "") == g
                  and r.get("pearson_before_after") not in (None, "")]
            if not ys:
                continue
            xj = np.random.default_rng(0).normal(xi, 0.06, len(ys))
            p1.scatter(xj, ys, color=_group_color(g), s=26, alpha=.8)
            p1.scatter([xi], [np.mean(ys)], color=_group_color(g), s=240,
                       marker="_", lw=2.5)
        p1.set_xticks(range(len(gs))); p1.set_xticklabels(gs)
        p1.set_xlabel("group"); p1.set_ylabel("Pearson r (before vs after)")
        p1.set_title("Morphology preserved by group\n(1 = identical; bar = mean)")
    else:
        p1.hist(pear, bins=min(15, max(4, len(pear))), color="#2a5db0", alpha=.85)
        p1.axvline(pear.mean(), color="#b03a3a", ls="--",
                   label=f"mean = {pear.mean():.3f}")
        p1.set_xlabel("Pearson r (before vs after)"); p1.set_ylabel("# objects")
        p1.set_title("Morphology preserved?\n(1 = identical)"); p1.legend(fontsize=9)

    def _plot_embedding_shift(a, method, label, axis_prefix, var_word):
        gcol = [_group_color(grade_of.get(o, "")) for o in emb["objs"]]
        a.scatter(method["pb"][:, 0], method["pb"][:, 1], marker="s",
                  facecolors="none", edgecolors=gcol, s=52, linewidths=1.4)
        a.scatter(method["pa"][:, 0], method["pa"][:, 1], c=gcol, s=34)
        for i in range(len(emb["objs"])):
            a.annotate("", xy=method["pa"][i], xytext=method["pb"][i],
                       arrowprops=dict(arrowstyle="->", color=gcol[i],
                                       alpha=.5, lw=1.1))
        if method.get("phr") is not None:
            pos = {o: j for j, o in enumerate(emb["objs"])}
            for j, o in enumerate(emb["hr_objs"]):
                hcol = _group_color(grade_of.get(o, "synthetic"))
                a.plot([method["pa"][pos[o], 0], method["phr"][j, 0]],
                       [method["pa"][pos[o], 1], method["phr"][j, 1]],
                       color=hcol, ls=":", lw=1.0, alpha=.6)
            a.scatter(method["phr"][:, 0], method["phr"][:, 1],
                      c=[_group_color(grade_of.get(o, "synthetic"))
                         for o in emb["hr_objs"]],
                      marker="*", s=150, edgecolors="k", linewidths=.4)
        a.set_xlabel(f"{axis_prefix}1 ({method['var'][0]:.0f}% {var_word})")
        a.set_ylabel(f"{axis_prefix}2 ({method['var'][1]:.0f}% {var_word})")
        # Title carries the toward-truth tally (real metric is full-dim L2, read
        # from the manifest's closer_to_ref column, not the 2-D projection).
        hr_rows = [r for r in mani if str(r.get("has_hr", "")).lower() == "true"]
        toward = sum(1 for r in hr_rows
                     if str(r.get("closer_to_ref", "")).lower() == "true")
        sub = (f"\nSR → HR truth for {toward}/{len(hr_rows)}"
               if hr_rows else "")
        a.set_title(f"{label} shift in Zoobot feature space\n(□ before → ● after"
                    + (", ★ HR truth)" if method.get("phr") is not None else ")")
                    + sub)
        if multi or method.get("phr") is not None:
            import matplotlib.lines as mlines
            handles = [mlines.Line2D([], [], marker="o", ls="",
                                     color=_group_color(g), label=g)
                       for g in groups if g]
            if method.get("phr") is not None:
                handles.append(mlines.Line2D([], [], marker="*", ls="",
                                             color="#444", markersize=11,
                                             label="HR truth"))
            a.legend(handles=handles, fontsize=8, title="group")

    # Panels 2/3 — PCA and MDS shift maps: ○ before → ● after arrows coloured
    # by group, plus ★ HR-truth anchors (synthetic only) with dotted after→truth
    # ties so you can read whether SR moved each object toward the truth.
    if emb is not None:
        _plot_embedding_shift(ax[k], emb["pca"], "PCA", "PC", "var"); k += 1
        _plot_embedding_shift(ax[k], emb["mds"], "MDS", "MDS", "inertia"); k += 1

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


def _group_triptych(run_dir: str, obj_id: str, asinh: float, h: int = 110):
    """Build an RGB uint8 strip [LR | SR | HR?] for one object, panels resized
    to ``h`` px tall with thin separators. Returns the array or ``None``."""
    import numpy as np
    from PIL import Image
    parts = []
    for fn in ("original_stack.fits", "SR.fits", "HR.fits"):
        p = os.path.join(run_dir, obj_id, fn)
        if not os.path.isfile(p):
            continue
        u8 = stretch_to_uint8(load_vis_plane(p), asinh_scale=asinh)
        im = Image.fromarray(u8, mode="L").convert("RGB").resize((h, h),
                                                                 Image.BILINEAR)
        parts.append(np.asarray(im, dtype=np.uint8))
    if not parts:
        return None
    sep = np.full((h, 3, 3), 255, np.uint8)
    out = parts[0]
    for pt in parts[1:]:
        out = np.concatenate([out, sep, pt], axis=1)
    return out


def render_transformation_summary(run_dir: str, out_png: str) -> Optional[str]:
    """Render a run-level SR-transformation summary PNG → ``out_png``.

    Panels: (1) SR-vs-HR recovery for the synthetic group (PSNR LR↔HR vs SR↔HR;
    points above the diagonal mean SR is closer to truth); (2) flux ratio
    Σ SR/LR per group; (3) an example transformation strip per group
    (LR | SR for lenses, LR | SR | HR for synthetic). Returns ``out_png`` or
    ``None`` if there's no manifest.
    """
    import csv

    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mani_path = os.path.join(run_dir, "manifest.csv")
    if not os.path.isfile(mani_path):
        return None
    rows = [r for r in csv.DictReader(open(mani_path))
            if r.get("id") and str(r.get("ok", "")).lower() == "true"]
    if not rows:
        return None
    groups = [g for g in dict.fromkeys(r.get("grade", "") for r in rows)]

    def _f(r, k):
        v = r.get(k, "")
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    fig, ax = plt.subplots(1, 3, figsize=(18, 5.4))

    # Panel 1 — SR-vs-HR recovery (any group carrying HR truth: syn-lens/syn-gal).
    syn = [r for r in rows
           if _f(r, "psnr_lr_hr") is not None and _f(r, "psnr_sr_hr") is not None]
    if syn:
        x = [_f(r, "psnr_lr_hr") for r in syn]
        y = [_f(r, "psnr_sr_hr") for r in syn]
        ax[0].scatter(x, y, c=[_group_color(r.get("grade", "")) for r in syn], s=40)
        lo, hi = min(x + y), max(x + y)
        ax[0].plot([lo, hi], [lo, hi], color="#888", ls="--", label="no change")
        ax[0].set_xlabel("PSNR  LR vs HR (dB)")
        ax[0].set_ylabel("PSNR  SR vs HR (dB)")
        win = sum(b > a for a, b in zip(x, y))
        ax[0].set_title(f"SR vs HR recovery (synthetic)\nabove line = closer to "
                        f"truth: {win}/{len(syn)}")
        ax[0].legend(fontsize=9)
    else:
        ax[0].axis("off")
        ax[0].text(0.5, 0.5, "No synthetic group\n(SR-vs-HR needs HR truth)",
                   ha="center", va="center", color="#888")

    # Panel 2 — flux ratio Σ SR/LR per group.
    gs = [g for g in groups if g]
    for xi, g in enumerate(gs):
        ys = [_f(r, "flux_ratio_sr_over_lr") for r in rows
              if r.get("grade") == g and _f(r, "flux_ratio_sr_over_lr") is not None]
        if not ys:
            continue
        xj = np.random.default_rng(0).normal(xi, 0.06, len(ys))
        ax[1].scatter(xj, ys, color=_group_color(g), s=26, alpha=.8)
        ax[1].scatter([xi], [np.mean(ys)], color=_group_color(g), s=240,
                      marker="_", lw=2.5)
    ax[1].axhline(1.0, color="#888", ls=":", lw=1)
    ax[1].set_xticks(range(len(gs))); ax[1].set_xticklabels(gs)
    ax[1].set_xlabel("group"); ax[1].set_ylabel("flux Σ SR / Σ LR")
    ax[1].set_title("Flux conservation by group\n(1 = conserved; bar = mean)")

    # Panel 3 — example transformation strip per group.
    from euclid_polish.config import Config
    asinh = float(Config.STRETCH_SCALE_E)
    strips, labels = [], []
    width = 0
    for g in (gs or [""]):
        ex = next((r for r in rows if r.get("grade") == g), None)
        if ex is None:
            continue
        strip = _group_triptych(run_dir, ex["out_subdir"], asinh)
        if strip is None:
            continue
        strips.append(strip); labels.append(g or "?")
        width = max(width, strip.shape[1])
    if strips:
        padded = [np.pad(s, ((0, 0), (0, width - s.shape[1]), (0, 0)),
                         constant_values=255) for s in strips]
        gap = np.full((6, width, 3), 255, np.uint8)
        stacked = padded[0]
        ytick = [padded[0].shape[0] / 2]
        ycur = padded[0].shape[0]
        for s in padded[1:]:
            stacked = np.concatenate([stacked, gap, s], axis=0)
            ytick.append(ycur + gap.shape[0] + s.shape[0] / 2)
            ycur += gap.shape[0] + s.shape[0]
        ax[2].imshow(stacked)
        ax[2].set_yticks(ytick); ax[2].set_yticklabels(labels)
        ax[2].set_xticks([])
        ax[2].set_title("Example transform per group\nLR | SR | HR", fontsize=11)
    else:
        ax[2].axis("off")

    fig.suptitle(f"How the data transformed — {os.path.basename(run_dir)}",
                 fontsize=13)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_png
