"""SR − input (and truth − input on synthetic) VIS flux in elliptical annuli
(semi-major in arcsec) around faint isolated galaxies: real NEXUS (MER
isolated set) or the synthetic test fields. Writes per-galaxy annulus sums."""
import json, sys, numpy as np
from photutils.aperture import EllipticalAperture, EllipticalAnnulus, aperture_photometry
from astropy.table import Table
from astropy.wcs import WCS
from euclid_polish.web.helpers import model_catalog as mc, real_tiles as rt, ensemble_viz as ev, universal_models as um
from euclid_polish.training.inference import scaled_wcs_header
from euclid_polish.photometry import uJy_to_electrons, electrons_to_ab_mag
from euclid_polish.config import Config
from photutils.segmentation import detect_sources, SourceCatalog

EDGES = [0.0, 0.2, 0.4, 0.7, 1.0, 1.5, 2.5, 4.0]          # semi-major, arcsec
mode, out = sys.argv[1], sys.argv[2]
specs = ["universal:member_249", "universal:member_227", "production", "milestone:member_257@8000000"]
items = {s: mc.resolve_spec(s) for s in specs}
runner = mc.EnsembleMemberRunner(labels=mc.needed_members(items.values()))
class Members:
    def __init__(self, lr): self.lr, self.m = lr, {}
    def get(self, l):
        if l not in self.m: self.m[l] = runner.predict(self.lr, l)
        return self.m[l]

def rings(img, x, y, q, th, pix):
    vals = []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        a_hi = hi / pix
        ap = (EllipticalAperture((x, y), a_hi, a_hi * q, th) if lo == 0 else
              EllipticalAnnulus((x, y), lo / pix, a_hi, a_hi * q, b_in=lo / pix * q, theta=th))
        vals.append(float(aperture_photometry(np.nan_to_num(img), ap, method="exact")["aperture_sum"][0]))
    return vals

def galaxy(lr, x, y, q, th, srs, truth=None):
    """Rings of input (on the SR grid), each SR, truth — SR pixel coords."""
    lr2 = np.repeat(np.repeat(lr[..., 0], 2, 0), 2, 1) / 4.0
    X, Y = 2 * x + 0.5, 2 * y + 0.5
    row = {"input": rings(lr2, X, Y, q, th, 0.05)}
    for s, sr in srs.items():
        row[s] = rings(sr[..., 0], X, Y, q, th, 0.05)
    if truth is not None:
        row["truth"] = rings(truth[..., 0], X, Y, q, th, 0.05)
    return row

reach = EDGES[-1] / 0.1 + 3
rows = []
if mode == "real":
    d = np.load("data/jwst_euclid_overlap/flux_conservation/report_fluxes.npz")
    t = Table.read("data/jwst_euclid_overlap/flux_conservation/mer_nexus.fits"); t.add_index("object_id")
    sub = t.loc[list(d["object_id"])]
    c = uJy_to_electrons(1, Config.get_band("VIS"))
    cat = np.asarray(sub["flux_detection_total"], float) * c; err = np.asarray(sub["fluxerr_detection_total"], float) * c
    iso = (np.abs(d["lr:kron_vis"] - cat) <= 0.5 * err) & (d["vis_mag"] > 22.0)
    sub = sub[iso]; mags = d["vis_mag"][iso]
    ra, dec = np.asarray(sub["right_ascension"], float), np.asarray(sub["declination"], float)
    done = set()
    for e in [e for e in rt.list_entries("nexus") if e.extras.get("jwst_file")]:
        x, y = WCS(rt.lr_header(e)).celestial.world_to_pixel_values(ra, dec)
        H = W = 255
        inside = [i for i in np.flatnonzero((x > reach) & (y > reach) & (x < W - reach) & (y < H - reach)) if i not in done]
        if not inside: continue
        lr = np.asarray(rt.get_tile("nexus", e.id, entry=e).lr_e, np.float32); mem = Members(lr)
        srs = {s: mc.predict(items[s], lr, mem) for s in specs}
        for i in inside:
            q = 1.0 - float(sub["ellipticity"][i]); th = np.deg2rad(float(sub["position_angle"][i]) + 90.0)
            rows.append({"mag": float(mags[i]), **galaxy(lr, x[i], y[i], q, th, srs)}); done.add(i)
        print(e.id, len(rows), flush=True)
else:
    ident = ev.universal_score_identity(); fwhm = ident.get("target_fwhm_arcsec") or Config.TARGET_PSF_FWHM_ARCSEC
    for n, (index, lr, target) in enumerate(um._test_pairs(ev._sky_records_local_dir(), ident["indices"])):
        truth = um._field_truth(target, fwhm)
        v = lr[..., 0]; med = np.median(v); sig = 1.4826 * np.median(np.abs(v - med))
        seg = detect_sources(v - med, 3 * sig, npixels=8)
        if seg is None: continue
        mem = Members(lr); srs = {s: mc.predict(items[s], lr, mem) for s in specs}
        Hh, Ww = v.shape
        for s in SourceCatalog(v - med, seg):
            k = s.kron_aperture
            if k is None or not np.isfinite(k.a) or k.a <= 0: continue
            x, y = float(s.xcentroid), float(s.ycentroid)
            if x < reach or y < reach or x > Ww - reach or y > Hh - reach: continue
            th = float(k.theta.to_value("rad")) if hasattr(k.theta, "to_value") else float(k.theta); q = float(k.b / k.a)
            iso = EllipticalAperture((x, y), reach - 3, (reach - 3) * q, th).to_mask(method="center").multiply(seg.data)
            labs = np.unique(iso); labs = labs[(labs > 0) & (labs != s.label)]
            if labs.size: continue
            flux = float(s.kron_flux)
            if not np.isfinite(flux) or flux <= 0: continue
            mag = float(electrons_to_ab_mag(flux, Config.get_band("VIS")))
            if mag <= 22.0: continue
            rows.append({"mag": mag, **galaxy(lr, x, y, q, th, srs, truth)})
        print("field", n + 1, len(rows), flush=True)
json.dump({"edges": EDGES, "rows": rows}, open(out, "w"))
