"""Kron VIS (and NISP) flux of synthetic test-field sources: SR vs truth vs input.

Copy of the pixel-input-knee session's syn_flux.py (which wrote syn_flux2.json);
only the model list changed: 257's final 15M milestone and production (238),
re-measured in the same run."""
import json, sys, numpy as np
from photutils.segmentation import detect_sources, SourceCatalog
from photutils.aperture import EllipticalAperture, aperture_photometry
from euclid_polish.web.helpers import ensemble_viz as ev, universal_models as um, model_catalog as mc
from euclid_polish.photometry import electrons_to_ab_mag
from euclid_polish.config import Config

specs = ["milestone:member_257@15000000", "universal:member_238"]
items = {s: mc.resolve_spec(s) for s in specs}
runner = mc.EnsembleMemberRunner(labels=mc.needed_members(items.values()))
class Members:
    def __init__(self, lr): self.lr, self.m = lr, {}
    def get(self, label):
        if label not in self.m: self.m[label] = runner.predict(self.lr, label)
        return self.m[label]
identity = ev.universal_score_identity(); fwhm = identity.get("target_fwhm_arcsec") or Config.TARGET_PSF_FWHM_ARCSEC
print("fwhm", fwhm, "fields", len(identity["indices"]), flush=True)
vis = Config.get_band("VIS")
rows = []
def kron(img, x, y, a, b, th):
    return float(aperture_photometry(img, EllipticalAperture((x, y), a, b, th), method="exact")["aperture_sum"][0])
for n, (index, lr, target) in enumerate(um._test_pairs(ev._sky_records_local_dir(), identity["indices"])):
    truth = um._field_truth(target, fwhm)
    srs = {s: mc.predict(items[s], lr, Members(lr)) for s in specs}
    v = lr[..., 0]; med = np.median(v); sig = 1.4826 * np.median(np.abs(v - med))
    seg = detect_sources(v - med, 3 * sig, npixels=8)
    if seg is None: continue
    cat = SourceCatalog(v - med, seg)
    H, W = v.shape
    for s in cat:
        a_k = float(s.kron_aperture.a) if s.kron_aperture is not None else np.nan
        if not np.isfinite(a_k) or a_k <= 0: continue
        b_k = float(s.kron_aperture.b); th = float(s.kron_aperture.theta.to_value("rad")) if hasattr(s.kron_aperture.theta, "to_value") else float(s.kron_aperture.theta)
        x, y = float(s.xcentroid), float(s.ycentroid)
        if x - a_k < 2 or y - a_k < 2 or x + a_k > W - 3 or y + a_k > H - 3: continue
        # isolated: no other segment inside the Kron ellipse
        ap = EllipticalAperture((x, y), a_k, b_k, th).to_mask(method="center")
        labs = np.unique(ap.multiply(seg.data)); labs = labs[(labs > 0) & (labs != s.label)]
        if labs.size: continue
        X, Y = 2 * x + 0.5, 2 * y + 0.5
        q = b_k / a_k; A = max(a_k, 18.0); B = A * q
        if x - A < 2 or y - A < 2 or x + A > W - 3 or y + A > H - 3: continue
        big = {"lr_big": kron(v, x, y, A, B, th), "truth_big": kron(truth[..., 0], 2 * x + 0.5, 2 * y + 0.5, 2 * A, 2 * B, th),
               **{sp + "_big": kron(sr[..., 0], 2 * x + 0.5, 2 * y + 0.5, 2 * A, 2 * B, th) for sp, sr in srs.items()}}
        row = {"field": int(index), "a": a_k, **big, "lr": kron(v, x, y, a_k, b_k, th),
               "truth": kron(truth[..., 0], X, Y, 2 * a_k, 2 * b_k, th),
               "peak_lr": float(v[int(round(y)), int(round(x))])}
        for sp, sr in srs.items():
            row[sp] = kron(sr[..., 0], X, Y, 2 * a_k, 2 * b_k, th)
        # flux error from the input's noise over the aperture area (white-noise approx)
        row["err"] = sig * np.sqrt(np.pi * a_k * b_k)
        rows.append(row)
    print(f"field {n + 1}: {len(rows)} sources", flush=True)
json.dump(rows, open(sys.argv[1], "w"))
