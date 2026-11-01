"""Test 7 — galaxy outskirts: how much faint extended light a model keeps.

* **Injection.** 40 noise-free synthetic galaxies (the starless truth of test
  fields 0–39, ``RC``, re-observed by a fitted LR kernel ``K``) are added to a
  host and the model's response ``rebin(SR(host + K ⊛ RC)) − rebin(SR(host))``
  is summed in rings of r / r_Kron around each galaxy (host magnitude > 19.5),
  over the truth in the same rings: 1 = all injected light recovered. Hosts:
  real NEXUS tiles and other synthetic test fields (40–79 → fields 50–89)
  (``hosts``), and Gaussian noise at the real Euclid σ and lag-1 correlation or
  at the training (v5) ones (``deck``). The table quotes 2–2.5 r_Kron.
* **Real-sky gain profile.** On every tenth NEXUS tile: the backprop gain
  ∂ Σ A(SR) / ∂ LR (A = the SR re-observed through the mean ePSF) seen by a
  faint, slightly extended blob, and A_fit / LR with A_fit = K_fit ⊛ rebin(SR)
  (K_fit fitted per model), both per r / r_Kron ring of MER galaxies.

Exactly the 2026-10-28/29 pipeline (underflux investigation, tests257
outskirts): the network runs through :class:`KneeNet` (the served member
read the way the console serves it) inside one ``tf.function``. Inputs: ``K``
and the synthetic sources (``data/K_syn.npy``, ``data/sources_syn.npz``, small,
committed; :func:`regenerate_inputs` rebuilds them) and ``RC``, rebuilt from the
``clean_test`` records.
"""
from __future__ import annotations

import json
import os

import numpy as np
import tensorflow as tf
from astropy.convolution import convolve
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.table import Table
from astropy.wcs import WCS
from numpy.lib.stride_tricks import sliding_window_view
from photutils.segmentation import SourceCatalog, deblend_sources, detect_sources, make_2dgaussian_kernel
from scipy import ndimage

from euclid_polish.config import Config
from euclid_polish.eval import knee_attribution as ka
from euclid_polish.eval.battery.common import MER_CATALOGUE, Battery, ModelInfo, nexus_entries, records_dir
from euclid_polish.image.collection import ImageSet
from euclid_polish.image.tfio import tfrecord_path
from euclid_polish.training.brightness import BRIGHTNESS_KERNEL_PX, block_noise_factors, core_kernel
from euclid_polish.training.forward_onthefly import member_psf_sets
from euclid_polish.training.target_blur import blur_target_array
from euclid_polish.web.helpers import real_tiles as rt

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
BANDS = [Config.get_band(n) for n in Config.HR_TARGET_BAND_NAMES]
SCALES = (1, 2, 4, 8)
TRIM = 10                       # LR px the VALID 41-px HR re-observation kernel trims
R_EDGES = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 9.1])
MAG_EDGES = np.array([0, 18, 19.5, 21, 22.5, 24, 99])
N_HOSTS = 40
SYN_HOST_OFFSET = 50            # synthetic host of galaxy j: test field j + 50
#: Gaussian hosts: per-band σ (e⁻) and lag-1 correlation of real Euclid / training (v5) noise.
SIG = {"real": np.array([18.44, 1.972, 2.124, 2.095]), "syn": np.array([20.89, 2.101, 2.25, 2.165])}
LAG1 = {"real": np.array([0.196, 0.76, 0.762, 0.762]), "syn": np.array([0.243, 0.829, 0.832, 0.826])}
DECK_HOSTS = ("real-like noise", "training-like noise")
INJECT_HOSTS = ("real NEXUS tiles", "synthetic test fields")
GRAD_EVERY = 10
GAIN_LO, GAIN_HI = 40, 215
KERNEL_HALF = 10                # the fitted LR kernels are 21 × 21
CLASSES = ("hole", "star core", "star wing", "galaxy core", "galaxy body", "galaxy outskirts",
           "faint-source halo", "background")
N_BOOT = 4000
BOOT_SEED = 12345
#: The quoted ring (index into the r / r_Kron rings): 2–2.5 r_Kron.
QUOTED_RING = 4


def ring_names() -> list[str]:
    return [f"{R_EDGES[i]:g}-{R_EDGES[i + 1]:g}" for i in range(len(R_EDGES) - 1)]


# --------------------------------------------------------------------------- #
# the network and the re-observation
# --------------------------------------------------------------------------- #

def kernels() -> tf.Tensor:
    """The mean-ePSF 41 × 41 HR re-observation kernel per band ``(41, 41, 4, 1)``."""
    sets, _note = member_psf_sets(seed=0)
    k = np.stack([core_kernel(sets[b.name].mean().data, BRIGHTNESS_KERNEL_PX) for b in BANDS], -1)
    return tf.constant(k[..., None], tf.float32)


def block_factors() -> np.ndarray:
    return np.array([block_noise_factors(b, SCALES) for b in BANDS])


def reobserve(img, kernel):
    """``(N, H, W, 4)`` HR electrons → ``(N, H/2 − 20, W/2 − 20, 4)`` archive grid (VALID)."""
    blurred = tf.nn.depthwise_conv2d(img, kernel, [1, 1, 1, 1], "VALID")
    return tf.nn.avg_pool2d(blurred, 2, 2, "VALID") * 4.0


def make_fns(net, kernel):
    knees = net.knees

    @tf.function(reduce_retracing=True)
    def forward(x):
        s = ka.asinh_stretch_multi_knee(x, knees)
        sr = net.predict_electrons(s)
        return sr, reobserve(sr, kernel)

    @tf.function(reduce_retracing=True)
    def gain(x, band):
        """∂ Σ A(SR)[..., band] / ∂ x (all input bands)."""
        with tf.GradientTape() as tape:
            tape.watch(x)
            s = ka.asinh_stretch_multi_knee(x, knees)
            a = reobserve(net.predict_electrons(s), kernel)
            total = tf.reduce_sum(tf.gather(a, band, axis=-1))
        return tape.gradient(total, x)

    return forward, gain


def rebin(x: np.ndarray) -> np.ndarray:
    return x.reshape(255, 2, 255, 2, 4).sum((1, 3))


def _inner(v: np.ndarray) -> np.ndarray:
    """A valid-grid per-pixel vector (235²) cut to the gain profile's interior."""
    sl = (slice(GAIN_LO - TRIM, GAIN_HI - TRIM),) * 2
    return v.reshape(235, 235)[sl].ravel()


_KERNEL: dict = {}


def shared_kernel() -> tf.Tensor:
    """:func:`kernels`, built once per process."""
    if "k" not in _KERNEL:
        _KERNEL["k"] = kernels()
    return _KERNEL["k"]


def network(bt: Battery, info: ModelInfo):
    """``(forward, gain)`` of the model's served member."""
    net = ka.KneeNet.from_member(bt.runner.model(info).members[0])
    return make_fns(net, shared_kernel())


# --------------------------------------------------------------------------- #
# inputs
# --------------------------------------------------------------------------- #

def load_inputs() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """``(K (4, 21, 21), synthetic sources)`` — the committed inputs."""
    K = np.load(os.path.join(DATA, "K_syn.npy"))
    with np.load(os.path.join(DATA, "sources_syn.npz")) as data:
        src = {k: data[k] for k in data.files}
    return K, src


def rc_rows(bt: Battery, n: int) -> np.ndarray:
    """The starless targets of test fields 0 … n−1 at the target PSF, 2 × 2
    summed to the LR grid ``(n, 255, 255, 4)`` float32 (``RC_syn``)."""
    def compute():
        fwhm = float(Config.TARGET_PSF_FWHM_ARCSEC)
        rows = []
        for i, clean in enumerate(ImageSet.read(tfrecord_path(records_dir(), "clean_test"), num_images=n)):
            assert clean.index == i, (clean.index, i)
            t = blur_target_array(np.asarray(clean.data, np.float32), fwhm,
                                  pixel_scale_arcsec=clean.pixel_scale_arcsec)
            rows.append(rebin(np.asarray(t, np.float32)))
        return {"n": n}, {"rc": np.stack(rows).astype(np.float32)}

    bt.stage(None, f"outskirt_rc_{n}", compute)
    return bt.store.arrays(None, f"outskirt_rc_{n}")["rc"]


def synthetic_hosts(n: int) -> np.ndarray:
    """The LR inputs of test fields 50 … 50 + n − 1 ``(n, 255, 255, 4)`` float32."""
    want = SYN_HOST_OFFSET + n
    out = []
    for i, lr in enumerate(ImageSet.read(tfrecord_path(records_dir(), "dirty_test"), num_images=want)):
        assert lr.index == i, (lr.index, i)
        if i >= SYN_HOST_OFFSET:
            out.append(np.asarray(lr.data, np.float32))
    return np.stack(out)


def real_host_ids(n: int) -> list[str]:
    """40 NEXUS tiles evenly spread over the field (the injection's real hosts)."""
    entries = nexus_entries()
    pick = np.linspace(0, len(entries) - 1, N_HOSTS).astype(int)[:n]
    return [entries[i].id for i in pick]


def real_lr(tile_id: str) -> np.ndarray:
    return np.asarray(rt.get_tile("nexus", tile_id).lr_e, np.float32)


def maps(s: dict, i: int, shape=(235, 235)):
    """Per valid-grid pixel of field ``i``: r / r_Kron to its best source, the
    source's star flag and magnitude."""
    sel = np.flatnonzero(s["tile"] == i)
    h, w = shape
    best = np.full(shape, np.inf)
    star = np.zeros(shape, bool)
    mag = np.full(shape, np.nan)
    yy, xx = np.mgrid[0:h, 0:w]
    for j in sel[np.argsort(s["mag"][sel])]:
        kron = np.clip(s["kron_px"][j], 1.0, 25.0)
        reach = int(np.ceil(4 * kron + 2))
        x, y = s["x"][j] - TRIM, s["y"][j] - TRIM
        x0, x1 = max(0, int(x) - reach), min(w, int(x) + reach + 2)
        y0, y1 = max(0, int(y) - reach), min(h, int(y) + reach + 2)
        if x0 >= x1 or y0 >= y1:
            continue
        dx, dy = xx[y0:y1, x0:x1] - x, yy[y0:y1, x0:x1] - y
        th = s["theta"][j]
        u = dx * np.cos(th) + dy * np.sin(th)
        v = -dx * np.sin(th) + dy * np.cos(th)
        rn = np.sqrt(u ** 2 + (v / max(1 - s["ellipticity"][j], 0.05)) ** 2) / kron
        better = rn < best[y0:y1, x0:x1]
        best[y0:y1, x0:x1][better] = rn[better]
        star[y0:y1, x0:x1][better] = s["star"][j]
        mag[y0:y1, x0:x1][better] = s["mag"][j]
    return np.minimum(best, 9.0), star, mag


def bins(rn: np.ndarray, mag: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(host magnitude bin, ring)`` indices."""
    return (np.clip(np.digitize(mag, MAG_EDGES) - 1, 0, len(MAG_EDGES) - 2),
            np.clip(np.digitize(rn, R_EDGES) - 1, 0, len(R_EDGES) - 2))


def injected(rc: np.ndarray, K: np.ndarray) -> np.ndarray:
    return np.stack([ndimage.correlate(rc[..., b], K[b], mode="constant") for b in range(4)], -1)


# --------------------------------------------------------------------------- #
# injection
# --------------------------------------------------------------------------- #

def run_deck(bt: Battery, info: ModelInfo) -> dict:
    """Gaussian hosts: ``acc (2 hosts, 2 [recovered, truth], ring, band)`` and
    the per-field sums ``acc_f (N, 2, 2, ring, band)``, host m > 19.5."""
    def compute():
        n = bt.limits.hosts or N_HOSTS
        K, src = load_inputs()
        RC = rc_rows(bt, n)
        fwd, _ = network(bt, info)

        def sr(x):
            return rebin(fwd(tf.constant(x[None].astype(np.float32)))[0].numpy()[0])

        nr = len(R_EDGES) - 1
        acc = np.zeros((2, 2, nr, 4))
        acc_f = np.zeros((n, 2, 2, nr, 4))
        for j in range(n):
            rc = np.asarray(RC[j], np.float64)
            inj = injected(rc, K)
            rn, star, mag = maps(src, j)
            mi, ri = bins(rn, np.nan_to_num(mag, nan=99))
            sel = ~star & (rn < 9) & (mi >= 2)
            tru = rc[TRIM:-TRIM, TRIM:-TRIM]
            white = np.random.default_rng(j).standard_normal((255, 255, 4))
            for h, key in enumerate(("real", "syn")):
                base = np.empty((255, 255, 4))
                for b in range(4):
                    noise = ndimage.gaussian_filter(white[..., b], np.sqrt(1 / (4 * -np.log(LAG1[key][b]))),
                                                    mode="wrap")
                    base[..., b] = noise / noise.std() * SIG[key][b]
                rec = (sr(base + inj) - sr(base))[TRIM:-TRIM, TRIM:-TRIM]
                for b in range(4):
                    np.add.at(acc[h, 0, :, b], ri[sel], rec[..., b][sel])
                    np.add.at(acc[h, 1, :, b], ri[sel], tru[..., b][sel])
                    np.add.at(acc_f[j, h, 0, :, b], ri[sel], rec[..., b][sel])
                    np.add.at(acc_f[j, h, 1, :, b], ri[sel], tru[..., b][sel])
        return {"hosts": list(DECK_HOSTS), "n_galaxies": n}, {"acc": acc, "acc_f": acc_f, "r_edges": R_EDGES}

    return bt.stage(info, "outskirts_deck", compute)


def run_hosts(bt: Battery, info: ModelInfo) -> dict:
    """Real NEXUS tiles and synthetic test fields as hosts: ``<host>_rec`` /
    ``<host>_tru`` ``(mag bin, ring, band)`` and ``acc_f (N, 2, 2, ring, band)``
    (host m > 19.5)."""
    def compute():
        n = bt.limits.hosts or N_HOSTS
        K, src = load_inputs()
        RC = rc_rows(bt, n)
        syn_hosts = synthetic_hosts(n)
        real_ids = real_host_ids(n)
        fwd, _ = network(bt, info)

        def sr(x):
            return rebin(fwd(tf.constant(x[None].astype(np.float32)))[0].numpy()[0])

        nm, nr = len(MAG_EDGES) - 1, len(R_EDGES) - 1
        acc = {h: {k: np.zeros((nm, nr, 4)) for k in ("rec", "tru")} for h in ("real", "syn")}
        acc_f = np.zeros((n, 2, 2, nr, 4))
        for j in range(n):
            rc = np.asarray(RC[j], np.float64)
            inj = injected(rc, K)
            rn, star, mag = maps(src, j)
            mi, ri = bins(rn, np.nan_to_num(mag, nan=99))
            gal = ~star & (rn < 9)
            tru = rc[TRIM:-TRIM, TRIM:-TRIM]
            for h, (host, base) in enumerate((("real", np.asarray(real_lr(real_ids[j]), np.float64)),
                                              ("syn", np.asarray(syn_hosts[j], np.float64)))):
                ok_host = base != 0
                rec = (sr(np.where(ok_host, base + inj, 0.0)) - sr(base))[TRIM:-TRIM, TRIM:-TRIM]
                hole = ~ok_host[TRIM:-TRIM, TRIM:-TRIM]
                for b in range(4):
                    ok = gal & ~hole[..., b]
                    idx = (mi[ok], ri[ok])
                    np.add.at(acc[host]["rec"][..., b], idx, rec[..., b][ok])
                    np.add.at(acc[host]["tru"][..., b], idx, tru[..., b][ok])
                    f = ok & (mi >= 2)
                    np.add.at(acc_f[j, h, 0, :, b], ri[f], rec[..., b][f])
                    np.add.at(acc_f[j, h, 1, :, b], ri[f], tru[..., b][f])
        arrays = {"acc_f": acc_f, "r_edges": R_EDGES,
                  **{f"{h}_{k}": v for h, d in acc.items() for k, v in d.items()}}
        return {"hosts": list(INJECT_HOSTS), "n_galaxies": n, "real_tiles": real_ids}, arrays

    return bt.stage(info, "outskirts_hosts", compute)


# --------------------------------------------------------------------------- #
# the real-sky gain profile
# --------------------------------------------------------------------------- #

def _col(t: Table, name: str) -> np.ndarray:
    c = t[name]
    return np.asarray(c.filled(np.nan) if hasattr(c, "filled") else c, np.float64)


def catalogue() -> dict[str, np.ndarray]:
    t = Table.read(MER_CATALOGUE)
    out = {k: _col(t, k) for k in ("right_ascension", "declination", "point_like_prob", "kron_radius",
                                   "ellipticity", "position_angle", "fwhm", "flux_detection_total",
                                   "vis_det")}
    with np.errstate(divide="ignore", invalid="ignore"):
        out["mag"] = np.where(out["flux_detection_total"] > 0,
                              -2.5 * np.log10(out["flux_detection_total"]) + 23.9, np.nan)
    return out


def source_maps(cat: dict, header: str, shape):
    """Per valid-grid pixel: r / r_Kron to the best MER source (r_Kron =
    KRON_RADIUS / 2.5), the distance to it, its star flag and magnitude."""
    h, w = shape
    x, y = WCS(fits.Header.fromstring(header)).world_to_pixel_values(cat["right_ascension"],
                                                                     cat["declination"])
    x, y = x - TRIM, y - TRIM
    kron = np.clip(np.where(np.isfinite(cat["kron_radius"]) & (cat["kron_radius"] > 0),
                            cat["kron_radius"] / 2.5, 1.5), 1.0, 25.0)
    reach = 4.0 * kron + 2
    near = np.flatnonzero((x > -reach) & (x < w + reach) & (y > -reach) & (y < h + reach))
    best = np.full(shape, np.inf)
    dist = np.full(shape, np.inf)
    star = np.zeros(shape, bool)
    mag = np.full(shape, np.nan)
    yy, xx = np.mgrid[0:h, 0:w]
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
        better = rn < best[y0:y1, x0:x1]
        best[y0:y1, x0:x1][better] = rn[better]
        dist[y0:y1, x0:x1][better] = np.hypot(dx, dy)[better]
        star[y0:y1, x0:x1][better] = cat["point_like_prob"][i] > 0.5
        mag[y0:y1, x0:x1][better] = cat["mag"][i]
    return best, dist, star, mag


def tile_features(lr: np.ndarray, a: np.ndarray, header: str, cat: dict) -> dict[str, np.ndarray]:
    """Flat per-pixel records of one tile (all 4 bands): class, r / r_Kron,
    host magnitude, hole, LR and A − LR."""
    lrv = lr[TRIM:-TRIM, TRIM:-TRIM]
    shape = lrv.shape[:2]
    best, dist, star, mag = source_maps(cat, header, shape)
    hole = (lrv == 0) | ~np.isfinite(lrv)
    cls = np.full(shape, CLASSES.index("background"), np.int8)
    covered = best < 2.5
    cls[covered & ~star & (best >= 1.5)] = CLASSES.index("galaxy outskirts")
    cls[covered & ~star & (best < 1.5)] = CLASSES.index("galaxy body")
    cls[covered & ~star & (best < 0.5)] = CLASSES.index("galaxy core")
    cls[covered & star] = CLASSES.index("star wing")
    cls[covered & star & (dist < 2.5)] = CLASSES.index("star core")
    cls[(best >= 2.5) & (best < 4)] = CLASSES.index("faint-source halo")
    out = {"diff": (a - lrv).reshape(-1, 4), "lr": lrv.reshape(-1, 4), "hole": hole.reshape(-1, 4)}
    for k, v in (("cls", cls), ("rnorm", np.minimum(best, 9.0)), ("mag", mag)):
        out[k] = np.repeat(v.reshape(-1, 1), 4, 1)
    out["cls"] = np.where(out["hole"], CLASSES.index("hole"), out["cls"])
    return out


def fit_kernel(R: np.ndarray, lrs: list[np.ndarray], factors: np.ndarray) -> np.ndarray:
    """Unit-sum 21 × 21 LR kernels ``K (4, 21, 21)`` with ``LR ≈ K ⊛ R``
    (inverse-variance least squares, the centre substituted out)."""
    H = KERNEL_HALF
    p = (2 * H + 1) ** 2
    c = p // 2
    XtX = np.zeros((4, p - 1, p - 1))
    Xty = np.zeros((4, p - 1))
    for r, lr in zip(R, lrs, strict=True):
        lr = np.asarray(lr, np.float64)
        r = np.asarray(r, np.float64)
        sky = ka.noise_sigma(lr) / factors[:, 0]
        for b in range(4):
            win = sliding_window_view(r[..., b], (2 * H + 1, 2 * H + 1)).reshape(-1, p)
            y = lr[H:-H, H:-H, b].ravel()
            ok = np.isfinite(y) & (y != 0)
            w = 1.0 / (sky[b] ** 2 + np.clip(y, 0, None))
            X = (np.delete(win, c, 1) - win[:, c:c + 1])[ok]
            t = (y - win[:, c])[ok]
            XtX[b] += X.T @ (X * w[ok][:, None])
            Xty[b] += X.T @ (t * w[ok])
    K = np.zeros((4, 2 * H + 1, 2 * H + 1))
    for b in range(4):
        k = np.linalg.solve(XtX[b], Xty[b])
        K[b] = np.insert(k, c, 1.0 - k.sum()).reshape(2 * H + 1, 2 * H + 1)
    return K


def gain_tiles(limit: int | None = None) -> list:
    entries = nexus_entries()
    return [entries[i] for i in np.arange(0, len(entries), GRAD_EVERY)][:limit]


def run_gain(bt: Battery, info: ModelInfo) -> dict:
    """The model's real-sky gain profile (host m > 19.5 pooled, and per host
    magnitude bin); the gradient arrays live only while it is measured."""
    def compute():
        tiles = gain_tiles(bt.limits.grad_tiles)
        fwd, gain = network(bt, info)
        folder = bt.store.folder(info)
        os.makedirs(folder, exist_ok=True)
        g_path = os.path.join(folder, "gain_G.tmp.npy")
        G = np.lib.format.open_memmap(g_path, "w+", np.float32, (len(tiles), 4, 255, 255, 4))
        R = np.zeros((len(tiles), 255, 255, 4), np.float32)
        lrs = [np.asarray(rt.get_tile("nexus", e.id, entry=e).lr_e, np.float32) for e in tiles]
        for j, lr in enumerate(lrs):
            x = tf.constant(lr[None])
            sr, _a = fwd(x)
            R[j] = sr.numpy()[0].reshape(255, 2, 255, 2, 4).sum((1, 3))
            for b in range(4):
                G[j, b] = gain(x, tf.constant(b)).numpy()[0]
        G.flush()
        try:
            headers = [WCS(rt.lr_header(e)).celestial.to_header_string() for e in tiles]
            payload = gain_profile(G, R, lrs, headers)
        finally:
            del G
            os.remove(g_path)
        return {**payload, "tiles": [e.id for e in tiles]}, None

    return bt.stage(info, "outskirts_gain", compute)


def gain_profile(G: np.ndarray, R: np.ndarray, lrs: list[np.ndarray], headers: list[str]) -> dict:
    """Blob gain medians and A_fit / LR per (band, host magnitude bin, ring)."""
    cat = catalogue()
    factors = block_factors()
    k = np.pad(shared_kernel().numpy()[..., 0], ((0, 1), (0, 1), (0, 0)))
    psf_lr = k.reshape(21, 2, 21, 2, 4).sum((1, 3))
    psf_lr /= psf_lr.sum((0, 1))
    blob = np.stack([ndimage.gaussian_filter(np.pad(psf_lr[..., c], 12), 3.0) for c in range(4)], -1)
    blob /= blob.sum((0, 1))
    nm, nr = len(MAG_EDGES) - 1, len(R_EDGES) - 1
    star_cls = [CLASSES.index("star core"), CLASSES.index("star wing")]
    K = fit_kernel(R, lrs, factors)
    gain_vals = [[[[] for _ in range(nr)] for _ in range(nm)] for _ in range(4)]
    fsum = np.zeros((4, nm, nr, 2))
    for j, lr in enumerate(lrs):
        r = np.asarray(R[j], np.float64)
        a_fit = np.stack([ndimage.correlate(r[..., b], K[b], mode="constant") for b in range(4)],
                         -1)[10:-10, 10:-10]
        f = tile_features(np.asarray(lr), a_fit, headers[j], cat)
        rn, mag, hole = f["rnorm"][:, 0], f["mag"][:, 0], f["hole"]
        mi = np.where(np.isfinite(mag), np.clip(np.digitize(np.nan_to_num(mag), MAG_EDGES) - 1, 0, nm - 1),
                      -1)
        ri = np.clip(np.digitize(rn, R_EDGES) - 1, 0, nr - 1)
        for b in range(4):
            nonstar = ~np.isin(f["cls"][:, b], star_cls) & (mi >= 0)
            m = nonstar & ~hole[:, b] & np.isfinite(f["diff"][:, b]) & np.isfinite(f["lr"][:, b])
            np.add.at(fsum[b, :, :, 0], (mi[m], ri[m]), f["diff"][m, b])
            np.add.at(fsum[b, :, :, 1], (mi[m], ri[m]), f["lr"][m, b])
            g = np.asarray(G[j, b, :, :, b], np.float64)
            ext = ndimage.correlate(g, blob[..., b], mode="constant")[GAIN_LO:GAIN_HI, GAIN_LO:GAIN_HI]
            ext = ext.ravel()
            gm, gr, gs = (_inner(v) for v in (mi, ri, nonstar))
            for a_ in range(nm):
                for r_ in range(nr):
                    s = gs & (gm == a_) & (gr == r_)
                    if s.any():
                        gain_vals[b][a_][r_].append(ext[s].astype(np.float32))
    gmed = np.full((4, nm, nr), np.nan)
    gn = np.zeros((4, nm, nr), int)
    gmed_faint = np.full((4, nr), np.nan)
    for b in range(4):
        for r_ in range(nr):
            pooled = []
            for a_ in range(nm):
                v = np.concatenate(gain_vals[b][a_][r_]) if gain_vals[b][a_][r_] else np.zeros(0)
                gn[b, a_, r_] = v.size
                if v.size >= 30:
                    gmed[b, a_, r_] = np.median(v)
                if a_ >= 2:
                    pooled.append(v)
            v = np.concatenate(pooled) if pooled else np.zeros(0)
            if v.size >= 30:
                gmed_faint[b, r_] = np.median(v)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = 1 + fsum[..., 0] / np.where(fsum[..., 1] > 0, fsum[..., 1], np.nan)
        ff = fsum[:, 2:].sum(1)
        ratio_faint = 1 + ff[..., 0] / np.where(ff[..., 1] > 0, ff[..., 1], np.nan)
    return {"r_edges": R_EDGES.tolist(), "mag_edges": MAG_EDGES.tolist(), "gain_median": gmed, "gain_n": gn,
            "ratio": ratio, "gain_median_m_gt_19.5": gmed_faint, "ratio_m_gt_19.5": ratio_faint,
            "K_centre": K[:, KERNEL_HALF, KERNEL_HALF]}


# --------------------------------------------------------------------------- #
# summaries
# --------------------------------------------------------------------------- #

def deck_recovery(acc: np.ndarray) -> np.ndarray:
    """``(host, ring, band)`` recovered fraction of a deck ``acc``."""
    return acc[:, 0] / acc[:, 1]


def hosts_recovery(arrays: dict, host: str) -> np.ndarray:
    """``(ring, band)`` recovered fraction of an injection host, host m > 19.5."""
    return arrays[f"{host}_rec"][2:].sum(0) / arrays[f"{host}_tru"][2:].sum(0)


def boot_index(n: int) -> np.ndarray:
    """The field bootstrap's resamples (the same for every model: paired)."""
    return np.random.default_rng(BOOT_SEED).integers(0, n, size=(N_BOOT, n))


def boot_recovery(acc_f: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """``(B, host, ring)`` 4-band mean recovery of resampled per-field sums."""
    s = acc_f[idx].sum(1)
    return (s[:, :, 0] / s[:, :, 1]).mean(-1)


def _test_summary(bt: Battery, infos: list[ModelInfo], stage: str, hosts: tuple[str, ...],
                  recovery) -> dict:
    names = ring_names()
    data = {info.spec: bt.store.arrays(info, stage) for info in infos if bt.store.read(info, stage)}
    out: dict = {"rings": names, "hosts": list(hosts), "models": {}, "vs_baseline": {}}
    boots = {}
    for spec, d in data.items():
        rec = recovery(d)                                             # (host, ring, band)
        idx = boot_index(d["acc_f"].shape[0])
        boots[spec] = boot_recovery(d["acc_f"], idx)
        out["models"][spec] = {
            host: {"mean4": rec[h].mean(-1).tolist(), "per_band": rec[h].T.tolist(),
                   "boot68": {names[i]: np.percentile(boots[spec][:, h, i], [16, 50, 84]).tolist()
                              for i in range(len(names))}}
            for h, host in enumerate(hosts)}
    if infos and infos[0].spec in boots:
        base = boots[infos[0].spec]
        for info in infos[1:]:
            if info.spec not in boots or boots[info.spec].shape != base.shape:
                continue
            d = boots[info.spec] - base
            out["vs_baseline"][info.spec] = {
                host: {names[i]: np.percentile(d[:, h, i], [2.5, 50, 97.5]).tolist()
                       for i in range(len(names))}
                for h, host in enumerate(hosts)}
    return out


def summary(bt: Battery, infos: list[ModelInfo]) -> dict:
    deck = _test_summary(bt, infos, "outskirts_deck", DECK_HOSTS, lambda d: deck_recovery(d["acc"]))
    hosts = _test_summary(bt, infos, "outskirts_hosts", INJECT_HOSTS,
                          lambda d: np.stack([hosts_recovery(d, "real"), hosts_recovery(d, "syn")]))
    gain = {}
    for info in infos:
        res = bt.store.read(info, "outskirts_gain")
        if res is not None:
            g = np.array(res["gain_median_m_gt_19.5"], np.float64)
            q = np.array(res["ratio_m_gt_19.5"], np.float64)
            gain[info.spec] = {"gain_mean4": np.nanmean(g, 0).tolist(),
                               "ratio_mean4": np.nanmean(q, 0).tolist(),
                               "K_centre": res["K_centre"]}
    return {"deck": deck, "hosts": hosts, "gain": gain, "quoted_ring": ring_names()[QUOTED_RING]}


# --------------------------------------------------------------------------- #
# regenerating the committed inputs
# --------------------------------------------------------------------------- #

def _detect(vis: np.ndarray):
    hole = (vis == 0) | ~np.isfinite(vis)
    img = np.where(hole, 0.0, vis)
    sm = convolve(img, make_2dgaussian_kernel(2.0, size=5), mask=hole, normalize_kernel=True)
    _, med, std = sigma_clipped_stats(sm[~hole], sigma=3.0)
    seg = detect_sources(sm - med, 2.5 * std, npixels=10, mask=hole)
    if seg is None:
        return None, None
    seg = deblend_sources(sm - med, seg, npixels=10, nlevels=32, contrast=0.005, progress_bar=False)
    return SourceCatalog(img - med, seg, convolved_data=sm - med, mask=hole), seg


def _rows(cat) -> dict[str, np.ndarray]:
    kr = np.asarray(cat.kron_radius.value, float) * np.asarray(cat.semimajor_sigma.value, float)
    flux = np.asarray(cat.kron_flux, float)
    flux = np.where(np.isfinite(flux) & (flux > 0), flux, np.asarray(cat.segment_flux, float))
    return {"x": np.asarray(cat.xcentroid, float), "y": np.asarray(cat.ycentroid, float),
            "kron_px": np.where(np.isfinite(kr) & (kr > 0), kr, 1.5),
            "ellipticity": np.nan_to_num(np.asarray(cat.ellipticity.value, float)),
            "theta": np.deg2rad(np.nan_to_num(np.asarray(cat.orientation.value, float))),
            "flux": flux, "label": np.asarray(cat.labels)}


def regenerate_inputs(out_dir: str, log=print) -> dict[str, str]:
    """Rebuild ``K_syn.npy``, ``sources_syn.npz`` and ``outskirt_zp.json`` into
    ``out_dir`` from the test records, the NEXUS tiles and the MER catalogue
    (2026-10-28 ``synth_run.py fields`` + ``detect.py``; no model)."""
    os.makedirs(out_dir, exist_ok=True)
    rdir = records_dir()
    fwhm = float(Config.TARGET_PSF_FWHM_ARCSEC)
    factors = block_factors()
    streams = [ImageSet.read(tfrecord_path(rdir, k)) for k in ("dirty_test", "hr_test", "clean_test")]
    lr_syn, RT, RC = [], [], []
    for i, (lr, hr, clean) in enumerate(zip(*streams, strict=False)):
        assert lr.index == hr.index == clean.index == i
        lr_syn.append(np.asarray(lr.data, np.float32))
        for dst, im in ((RT, hr), (RC, clean)):
            t = blur_target_array(np.asarray(im.data, np.float32), fwhm,
                                  pixel_scale_arcsec=im.pixel_scale_arcsec)
            dst.append(rebin(np.asarray(t, np.float32)))
    K = fit_kernel(RT, lr_syn, factors)
    np.save(os.path.join(out_dir, "K_syn.npy"), K)
    log(f"K_syn centre {K[:, KERNEL_HALF, KERNEL_HALF].round(3)}")
    # the VIS e⁻ → AB zero point from the real MER cross-match
    t = Table.read(MER_CATALOGUE)
    mer = {k: _col(t, k) for k in ("right_ascension", "declination", "point_like_prob",
                                   "flux_detection_total")}
    with np.errstate(divide="ignore", invalid="ignore"):
        mer["mag"] = np.where(mer["flux_detection_total"] > 0,
                              -2.5 * np.log10(mer["flux_detection_total"]) + 23.9, np.nan)
    zp_terms = []
    for n, entry in enumerate(nexus_entries()):
        lr = np.asarray(rt.get_tile("nexus", entry.id, entry=entry).lr_e, np.float32)
        cat, _seg = _detect(np.asarray(lr[..., 0], np.float64))
        if cat is None:
            continue
        r = _rows(cat)
        header = WCS(rt.lr_header(entry)).celestial.to_header_string()
        mx, my = WCS(fits.Header.fromstring(header)).world_to_pixel_values(mer["right_ascension"],
                                                                            mer["declination"])
        inside = np.flatnonzero((mx > -3) & (mx < 258) & (my > -3) & (my < 258))
        for s in range(len(r["x"])):
            d = np.hypot(mx[inside] - r["x"][s], my[inside] - r["y"][s])
            if d.size and d.min() < 2.0:
                j = inside[np.argmin(d)]
                if 18 < mer["mag"][j] < 21 and r["flux"][s] > 0:
                    zp_terms.append(mer["mag"][j] + 2.5 * np.log10(r["flux"][s]))
        if n % 100 == 0:
            log(f"zero point: tile {n}")
    zp = float(np.median(zp_terms))
    syn: dict[str, list] = {}
    for i, lr in enumerate(lr_syn):
        cat, seg = _detect(np.asarray(lr[..., 0], np.float64))
        if cat is None:
            continue
        r = _rows(cat)
        tr, c = np.asarray(RT[i][..., 0], np.float64), np.asarray(RC[i][..., 0], np.float64)
        labels = seg.data
        tot = np.bincount(labels.ravel(), weights=tr.ravel(), minlength=labels.max() + 1)
        stars = np.bincount(labels.ravel(), weights=(tr - c).ravel(), minlength=labels.max() + 1)
        star = stars[r["label"]] > 0.5 * np.maximum(tot[r["label"]], 1e-9)
        for k, v in r.items():
            syn.setdefault(k, []).append(v)
        syn.setdefault("tile", []).append(np.full(len(r["x"]), i))
        syn.setdefault("star", []).append(star)
    out = {k: np.concatenate(v) for k, v in syn.items()}
    out["mag"] = zp - 2.5 * np.log10(np.clip(out["flux"], 1e-3, None))
    np.savez(os.path.join(out_dir, "sources_syn.npz"), **out)
    with open(os.path.join(out_dir, "outskirt_zp.json"), "w", encoding="utf-8") as fh:
        json.dump({"zp": zp}, fh)
    return {name: os.path.join(out_dir, name)
            for name in ("K_syn.npy", "sources_syn.npz", "outskirt_zp.json")}
