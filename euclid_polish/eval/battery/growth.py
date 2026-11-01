"""Test 9 — curves of growth of real galaxies: where the SR puts the input's light.

For every MER galaxy of the NEXUS tiles (``scripts/flux_conservation.galaxies``:
VIS-detected, ``point_like_prob`` < 0.1, S/N ≥ 10) the cumulative flux F(<r)
inside circular apertures on the MER position, r = 0.05″ … 3″, in all four
bands, on the input and on each model's SR. Both are measured on the SR grid
(the input spread flux-conservingly, each LR pixel → 2×2 of ¼ its value), so
input and SR see the very same apertures and the very same mask: the pixels
inside another MER source's Kron ellipse that belong to it (smallest
elliptical r / r_Kron, r_Kron = KRON_RADIUS / 2.5) and the NISP padding
holes. A galaxy with more than :data:`MAX_MASKED` of its 3″ disc masked is
dropped; a galaxy on two tiles is measured once.

The table is the median per galaxy of F_SR(<r) / F_input(<r) per VIS magnitude
range: > 1 at small r is light moved inwards (the deconvolution), < 1 at 3″ is
light lost. The SR's blank-sky level differs from the input's, which over a 3″
disc outweighs a faint galaxy, so a sky-matched ratio also subtracts the tile's
mean SR − input on blank sky (> 4 r_Kron from every MER source) from each SR
pixel. Blank sky still holds undetected faint galaxies, so the truth sits
between the raw and the sky-matched ratio.

The geometry (rings, masks, input curves) is measured once (no model); each
model adds its SR curves and blank-sky offsets.
"""
from __future__ import annotations

import numpy as np
from astropy.wcs import WCS

from euclid_polish.eval.battery.common import (
    MER_CATALOGUE,
    SHORT,
    Battery,
    ModelInfo,
    nexus_entries,
    nexus_key,
    nexus_lr,
)
from euclid_polish.eval.battery.flux import boot_median, paired_median
from euclid_polish.photometry import uJy_to_ab_mag
from euclid_polish.training.inference import scaled_wcs_header
from euclid_polish.web.helpers import real_tiles as rt
from scripts import flux_conservation as fcs

SR_PIXEL_ARCSEC = 0.05
R_MAX = 60                                              # SR px: apertures out to 3″
RADII = np.arange(1, R_MAX + 1) * SR_PIXEL_ARCSEC       # aperture radii (″)
#: Drop a galaxy with more of its 3″ disc masked than this.
MAX_MASKED = 0.2
#: A pixel belongs to a source inside this r / r_Kron (its Kron ellipse).
OWNED = 2.5
#: Blank sky: farther than this r / r_Kron from every source.
BLANK = 4.0
MIN_BLANK_PX = 1000
MAG_RANGES = ((0.0, 22.0), (22.0, 23.0), (23.0, 24.0), (24.0, 99.0))
QUOTED_RADII = (0.5, 1.0, 3.0)
#: The Kron aperture in r_Kron (MER's KRON_RADIUS).
KRON_APERTURE = 2.5


def range_name(lo: float, hi: float) -> str:
    return f"<{hi:g}" if lo <= 0 else (f">{lo:g}" if hi >= 99 else f"{lo:g}-{hi:g}")


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #

def ownership(x, y, kron, ell, pa, shape) -> tuple[np.ndarray, np.ndarray]:
    """Per pixel: the source with the smallest elliptical r / r_Kron (``-1``
    for none within reach) and that value. ``kron`` in pixels of this grid."""
    h, w = shape
    owner = np.full(shape, -1, np.int32)
    best = np.full(shape, np.inf, np.float32)
    reach = BLANK * kron / np.maximum(1 - np.nan_to_num(ell), 0.05) + 2
    yy, xx = np.mgrid[0:h, 0:w]
    for i in np.flatnonzero((x > -reach) & (x < w + reach) & (y > -reach) & (y < h + reach)):
        r = int(np.ceil(reach[i]))
        x0, x1 = max(0, int(x[i]) - r), min(w, int(x[i]) + r + 2)
        y0, y1 = max(0, int(y[i]) - r), min(h, int(y[i]) + r + 2)
        if x0 >= x1 or y0 >= y1:
            continue
        dx, dy = xx[y0:y1, x0:x1] - x[i], yy[y0:y1, x0:x1] - y[i]
        e = ell[i] if np.isfinite(ell[i]) else 0.0
        th = np.deg2rad((pa[i] if np.isfinite(pa[i]) else 0.0) + 90)
        u = dx * np.cos(th) + dy * np.sin(th)
        v = -dx * np.sin(th) + dy * np.cos(th)
        rn = np.sqrt(u ** 2 + (v / max(1 - e, 0.05)) ** 2) / kron[i]
        better = rn < best[y0:y1, x0:x1]
        best[y0:y1, x0:x1][better] = rn[better]
        owner[y0:y1, x0:x1][better] = i
    return owner, best


def rings(owner, best, hole, index: int, x: float, y: float) -> tuple[np.ndarray, float]:
    """The ``(2 R_MAX + 1)²`` ring map of source ``index`` centred on the pixel
    nearest ``(x, y)``: ring k (1 … R_MAX) holds the pixels with k − 1 < r ≤ k
    (r from the exact centre, SR px; the central pixel is ring 1), masked or
    farther pixels ``R_MAX + 1``. Also the masked share of the disc."""
    cx, cy = int(round(x)), int(round(y))
    sl = (slice(cy - R_MAX, cy + R_MAX + 1), slice(cx - R_MAX, cx + R_MAX + 1))
    yy, xx = np.mgrid[-R_MAX:R_MAX + 1, -R_MAX:R_MAX + 1]
    r = np.hypot(xx + cx - x, yy + cy - y)
    bad = hole[sl] | ((owner[sl] != index) & (owner[sl] >= 0) & (best[sl] < OWNED))
    disc = r < R_MAX
    ring = np.maximum(np.ceil(r), 1).astype(np.int64)
    ring[bad | (ring > R_MAX)] = R_MAX + 1
    return ring.astype(np.uint8), float(bad[disc].mean())


def cumulative(ring: np.ndarray, image: np.ndarray) -> np.ndarray:
    """F(<r) for r = 1 … R_MAX SR px of ``image`` ``(n, n, C)`` (or the
    unmasked pixel count for a 2-D image of ones)."""
    flat = ring.ravel()
    planes = image.reshape(flat.size, -1)
    sums = np.stack([np.bincount(flat, weights=planes[:, c], minlength=R_MAX + 2)
                     for c in range(planes.shape[1])], -1)
    return np.cumsum(sums[1:R_MAX + 1], axis=0)


def centre(x: float, y: float) -> tuple[slice, slice]:
    cx, cy = int(round(x)), int(round(y))
    return slice(cy - R_MAX, cy + R_MAX + 1), slice(cx - R_MAX, cx + R_MAX + 1)


def spread(lr: np.ndarray) -> np.ndarray:
    """The input on the SR grid, flux-conserving."""
    return np.repeat(np.repeat(lr, 2, 0), 2, 1) / 4.0


def _float(table, name) -> np.ndarray:
    col = table[name]
    return np.asarray(col.filled(np.nan) if hasattr(col, "filled") else col, np.float64)


def geometry(bt: Battery) -> dict[str, np.ndarray]:
    """Per galaxy: tile, object id, SR-grid centre, VIS magnitude, r_Kron (″),
    masked share, ring map, unmasked area and the input's curves; per tile its
    blank-sky mask (packed bits)."""
    def compute():
        cat = fcs.fetch_catalogue(MER_CATALOGUE)
        object_id = np.asarray(cat["object_id"])
        is_galaxy = np.isin(object_id, np.asarray(fcs.galaxies(cat)["object_id"]))
        ra, dec = _float(cat, "right_ascension"), _float(cat, "declination")
        kron_lr = _float(cat, "kron_radius")
        kron = np.clip(np.where(kron_lr > 0, kron_lr, 3.75) / 2.5, 1.0, 25.0) * 2.0   # SR px
        ell, pa = _float(cat, "ellipticity"), _float(cat, "position_angle")
        mag = uJy_to_ab_mag(_float(cat, "flux_detection_total"))
        out = {k: [] for k in ("tile", "object_id", "x", "y", "mag", "kron_arcsec", "masked", "rings",
                               "area", "lr")}
        blank_tiles, blank = [], []
        seen: set = set()
        for entry in nexus_entries(bt.limits.tiles):
            lr = nexus_lr(entry)
            lr = spread(np.where(np.isfinite(lr), lr, 0.0).astype(np.float64))
            hole = (lr == 0).any(-1)
            x, y = WCS(scaled_wcs_header(rt.lr_header(entry), 2)).celestial.world_to_pixel_values(ra, dec)
            owner, best = ownership(x, y, kron, ell, pa, hole.shape)
            free = (best > BLANK) & ~hole
            blank_tiles.append(entry.id)
            blank.append(np.packbits(free.ravel()))
            h, w = hole.shape
            inside = is_galaxy & (x >= R_MAX) & (x < w - R_MAX - 1) & (y >= R_MAX) & (y < h - R_MAX - 1)
            for i in np.flatnonzero(inside):
                if object_id[i] in seen:
                    continue
                ring, masked = rings(owner, best, hole, i, x[i], y[i])
                if masked > MAX_MASKED:
                    continue
                seen.add(object_id[i])
                out["tile"].append(entry.id)
                out["object_id"].append(object_id[i])
                out["x"].append(x[i])
                out["y"].append(y[i])
                out["mag"].append(mag[i])
                out["kron_arcsec"].append(kron[i] * SR_PIXEL_ARCSEC)
                out["masked"].append(masked)
                out["rings"].append(ring)
                out["area"].append(cumulative(ring, np.ones(ring.shape + (1,)))[:, 0])
                out["lr"].append(cumulative(ring, lr[centre(x[i], y[i])]))
        arrays = {k: np.asarray(v) for k, v in out.items()}
        arrays["blank_tiles"] = np.asarray(blank_tiles)
        arrays["blank"] = np.asarray(blank)
        return {"n_galaxies": len(out["tile"]), "n_tiles": len(set(out["tile"]))}, arrays

    bt.stage(None, "growth_geometry", compute)
    return bt.store.arrays(None, "growth_geometry")


def run(bt: Battery, info: ModelInfo) -> dict:
    """The model's SR curves of every galaxy of :func:`geometry` and its
    tiles' blank-sky SR − input (e⁻ per SR px)."""
    geo = geometry(bt)

    def compute():
        entries = {e.id: e for e in nexus_entries(bt.limits.tiles)}
        blank = dict(zip(geo["blank_tiles"].tolist(), geo["blank"], strict=True))
        cog = np.full(geo["lr"].shape, np.nan)
        sky = np.full((len(geo["tile"]), cog.shape[-1]), np.nan)
        tiles = geo["tile"]
        for tid in dict.fromkeys(tiles.tolist()):
            raw = nexus_lr(entries[tid])
            sr = np.nan_to_num(np.asarray(bt.cache.get(info, nexus_key(tid), raw), np.float64))
            lr = spread(np.where(np.isfinite(raw), raw, 0.0).astype(np.float64))
            free = np.unpackbits(blank[tid])[:sr.shape[0] * sr.shape[1]].reshape(sr.shape[:2]).astype(bool)
            sel = np.flatnonzero(tiles == tid)
            if free.sum() >= MIN_BLANK_PX:
                sky[sel] = (sr[free] - lr[free]).mean(0)
            for i in sel:
                cog[i] = cumulative(geo["rings"][i], sr[centre(geo["x"][i], geo["y"][i])])
        return {"n_galaxies": len(tiles)}, {"cog": cog, "sky": sky}

    return bt.stage(info, "growth", compute)


# --------------------------------------------------------------------------- #
# summary
# --------------------------------------------------------------------------- #

def ratios(geo: dict, model: dict) -> tuple[np.ndarray, np.ndarray]:
    """Per galaxy, radius and band: F_SR(<r) / F_input(<r), raw and sky-matched
    (NaN where the input's F(<r) is not positive)."""
    lr = np.where(geo["lr"] > 0, geo["lr"], np.nan)
    matched = model["cog"] - np.nan_to_num(model["sky"])[:, None, :] * geo["area"][:, :, None]
    return model["cog"] / lr, matched / lr


def half_light(curve: np.ndarray) -> np.ndarray:
    """r50 (″) of each ``(N, R)`` curve, against its own F(<3″)."""
    out = np.full(len(curve), np.nan)
    for i, c in enumerate(curve):
        if not np.isfinite(c).all() or c[-1] <= 0:
            continue
        c = c / c[-1]
        k = int(np.argmax(c >= 0.5))
        out[i] = RADII[0] if k == 0 else np.interp(0.5, [c[k - 1], c[k]], [RADII[k - 1], RADII[k]])
    return out


def at_kron(ratio: np.ndarray, kron_arcsec: np.ndarray) -> np.ndarray:
    """``ratio`` ``(N, R)`` at the Kron aperture, NaN beyond 3″."""
    r = KRON_APERTURE * kron_arcsec
    out = np.array([np.interp(ri, RADII, row) for ri, row in zip(r, ratio, strict=True)])
    out[r > RADII[-1]] = np.nan
    return out


def _index(radius: float) -> int:
    return int(np.argmin(np.abs(RADII - radius)))


def summary(bt: Battery, infos: list[ModelInfo]) -> dict:
    geo = geometry(bt)
    mag = geo["mag"]
    selections = {"all": np.ones(len(mag), bool),
                  **{range_name(a, b): (mag >= a) & (mag < b) for a, b in MAG_RANGES}}
    out: dict = {"n_galaxies": int(len(mag)), "n_tiles": int(len(set(geo["tile"].tolist()))),
                 "ranges": list(selections), "radii": list(QUOTED_RADII), "models": {}, "vs_baseline": {}}
    r50_lr = [half_light(geo["lr"][:, :, b]) for b in range(len(SHORT))]
    raw3 = {}
    for info in infos:
        if not bt.store.read(info, "growth"):
            continue
        model = bt.store.arrays(info, "growth")
        raw, matched = ratios(geo, model)
        cell: dict = {"sky": [float(v) for v in np.nanmean(model["sky"], 0)], "bands": {}}
        for b, band in enumerate(SHORT):
            r50 = half_light(model["cog"][:, :, b] - np.nan_to_num(model["sky"][:, b])[:, None]
                             * geo["area"]) / r50_lr[b]
            kron = at_kron(raw[:, :, b], geo["kron_arcsec"])
            cell["bands"][band] = {name: {**{f"r{r:g}": boot_median(raw[sel, _index(r), b])
                                             for r in QUOTED_RADII},
                                          "r3_sky": boot_median(matched[sel, _index(3.0), b]),
                                          "kron": boot_median(kron[sel]),
                                          "r50": boot_median(r50[sel])}
                                   for name, sel in selections.items()}
        out["models"][info.spec] = cell
        raw3[info.spec] = raw[:, _index(3.0), :]
    if infos and infos[0].spec in raw3:
        base = raw3[infos[0].spec]
        for info in infos[1:]:
            if info.spec in raw3:
                d = raw3[info.spec] - base
                out["vs_baseline"][info.spec] = {
                    band: {name: paired_median(d[sel, b]) for name, sel in selections.items()}
                    for b, band in enumerate(SHORT)}
    return out


def median_curves(bt: Battery, infos: list[ModelInfo], sel: np.ndarray | None = None) -> dict:
    """VIS median curves for the figure: per model raw and sky-matched
    SR/input, and F(<r) / F_input(<3″) of the input and each model."""
    geo = geometry(bt)
    sel = np.ones(len(geo["mag"]), bool) if sel is None else sel
    norm = np.where(geo["lr"][sel, -1:, 0] > 0, geo["lr"][sel, -1:, 0], np.nan)
    out = {"input": np.nanmedian(geo["lr"][sel, :, 0] / norm, 0), "models": {}}
    for info in infos:
        if not bt.store.read(info, "growth"):
            continue
        model = bt.store.arrays(info, "growth")
        raw, matched = ratios(geo, model)
        out["models"][info.spec] = {"raw": np.nanmedian(raw[sel, :, 0], 0),
                                    "sky": np.nanmedian(matched[sel, :, 0], 0),
                                    "norm": np.nanmedian(model["cog"][sel, :, 0] / norm, 0)}
    return out
