"""Per-pixel features + z-scores on the archive grid (valid 236×236 region of each tile)."""
import json

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from scipy import ndimage

from common import BANDS, block_factors
from euclid_polish.eval import knee_attribution as ka

CAT = "data/jwst_euclid_overlap/flux_conservation/mer_nexus.fits"
TRIM = 10           # LR px trimmed by the VALID 41-px HR kernel
CLASSES = ("hole", "star core", "star wing", "galaxy core", "galaxy body", "galaxy outskirts",
           "faint-source halo", "background")


def col(t, name):
    c = t[name]
    return np.asarray(c.filled(np.nan) if hasattr(c, "filled") else c, np.float64)


def catalogue():
    t = Table.read(CAT)
    out = {k: col(t, k) for k in ("right_ascension", "declination", "point_like_prob", "kron_radius",
                                  "ellipticity", "position_angle", "fwhm", "flux_detection_total",
                                  "vis_det")}
    with np.errstate(divide="ignore", invalid="ignore"):
        out["mag"] = np.where(out["flux_detection_total"] > 0,
                              -2.5 * np.log10(out["flux_detection_total"]) + 23.9, np.nan)
    return out


def source_maps(cat, header, shape):
    """Per pixel (valid grid): normalised elliptical radius to the best source in r_Kron = kron_radius/2.5, pixel
    distance to it, its star flag and magnitude, and the crowding count (sources whose 1.5 Kron
    (Kron-aperture) ellipses cover the pixel)."""
    h, w = shape
    x, y = WCS(fits.Header.fromstring(header)).world_to_pixel_values(cat["right_ascension"],
                                                                     cat["declination"])
    x, y = x - TRIM, y - TRIM
    kron = np.clip(np.where(np.isfinite(cat["kron_radius"]) & (cat["kron_radius"] > 0),
                            cat["kron_radius"] / 2.5, 1.5), 1.0, 25.0)   # r_Kron, LR px
    reach = 4.0 * kron + 2
    near = np.flatnonzero((x > -reach) & (x < w + reach) & (y > -reach) & (y < h + reach))
    best = np.full(shape, np.inf)
    dist = np.full(shape, np.inf)
    star = np.zeros(shape, bool)
    mag = np.full(shape, np.nan)
    crowd = np.zeros(shape, np.int16)
    yy, xx = np.mgrid[0:h, 0:w]
    # brightest first so ties go to the brighter source
    order = near[np.argsort(np.nan_to_num(cat["mag"][near], nan=99))]
    for i in order:
        r = int(np.ceil(reach[i]))
        x0, x1 = max(0, int(x[i]) - r), min(w, int(x[i]) + r + 2)
        y0, y1 = max(0, int(y[i]) - r), min(h, int(y[i]) + r + 2)
        if x0 >= x1 or y0 >= y1:
            continue
        dx, dy = xx[y0:y1, x0:x1] - x[i], yy[y0:y1, x0:x1] - y[i]
        e = cat["ellipticity"][i] if np.isfinite(cat["ellipticity"][i]) else 0.0
        th = np.deg2rad((cat["position_angle"][i] if np.isfinite(cat["position_angle"][i]) else 0.0) + 90)
        u = dx * np.cos(th) + dy * np.sin(th)
        v = -dx * np.sin(th) + dy * np.cos(th)
        rn = np.sqrt(u ** 2 + (v / max(1 - e, 0.05)) ** 2) / kron[i]
        crowd[y0:y1, x0:x1] += (rn < 2.5)
        better = rn < best[y0:y1, x0:x1]
        best[y0:y1, x0:x1][better] = rn[better]
        dist[y0:y1, x0:x1][better] = np.hypot(dx, dy)[better]
        star[y0:y1, x0:x1][better] = cat["point_like_prob"][i] > 0.5
        mag[y0:y1, x0:x1][better] = cat["mag"][i]
    return best, dist, star, mag, crowd


def tile_features(lr, a, header, cat, factors):
    """Flat per-pixel records for one tile and one model (all 4 bands)."""
    lrv = lr[TRIM:-TRIM, TRIM:-TRIM]
    shape = lrv.shape[:2]
    sigma = ka.noise_sigma(lr)                        # per-pixel scatter, e⁻
    sky = sigma / factors[:, 0]                       # noise-model level
    level2 = sky ** 2 + np.clip(lrv, 0, None)
    diff = a - lrv
    z1 = diff / (factors[:, 0] * np.sqrt(level2))
    # 4×4 block z (0.4"), broadcast back to pixels
    hb = (shape[0] // 4) * 4
    def block(v):
        b = v[:hb, :hb].reshape(hb // 4, 4, hb // 4, 4, 4).mean((1, 3))
        return np.repeat(np.repeat(b, 4, 0), 4, 1)
    z4 = np.full_like(z1, np.nan)
    z4[:hb, :hb] = block(diff) / (factors[:, 2] * np.sqrt(block(level2)))
    best, dist, star, mag, crowd = source_maps(cat, header, shape)
    hole = (lrv == 0) | ~np.isfinite(lrv)
    near_hole = ndimage.binary_dilation(hole.any(-1), iterations=3)
    cls = np.full(shape, CLASSES.index("background"), np.int8)
    covered = best < 2.5
    cls[covered & ~star & (best >= 1.5)] = CLASSES.index("galaxy outskirts")
    cls[covered & ~star & (best < 1.5)] = CLASSES.index("galaxy body")
    cls[covered & ~star & (best < 0.5)] = CLASSES.index("galaxy core")
    cls[covered & star] = CLASSES.index("star wing")
    cls[covered & star & (dist < 2.5)] = CLASSES.index("star core")
    cls[(best >= 2.5) & (best < 4)] = CLASSES.index("faint-source halo")
    smooth = ndimage.uniform_filter(lrv, (3, 3, 1))
    curv = (lrv - smooth) / (factors[:, 0] * np.sqrt(level2))   # peakedness in σ
    feats = {
        "z1": z1, "z4": z4, "diff": diff, "lr": lrv, "a": a,
        "snr": lrv / (factors[:, 0] * np.sqrt(level2)),
        "snr_smooth": smooth / (factors[:, 0] * np.sqrt(level2) / 3.0),
        "curv": curv,
        "colour": np.log10(np.clip(smooth, 1e-3, None) / np.clip(smooth[..., :1], 1e-3, None)),
    }
    out = {k: v.reshape(-1, 4) for k, v in feats.items()}
    per_px = {"cls": cls, "rnorm": np.minimum(best, 9.0), "dist": np.minimum(dist, 99.0), "mag": mag,
              "crowd": crowd, "near_hole": near_hole}
    for k, v in per_px.items():
        out[k] = np.repeat(v.reshape(-1, 1), 4, 1)
    out["hole"] = hole.reshape(-1, 4)
    out["cls"] = np.where(out["hole"], CLASSES.index("hole"), out["cls"])
    return out
