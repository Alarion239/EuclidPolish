"""Resolution of models against JWST/NIRCam F200W on the NEXUS deep field:
the Fourier ring correlation of each model's Euclid H with JWST, and the
period where it falls to ½ (see :mod:`euclid_polish.eval.jwst_resolution`).

    python scripts/jwst_resolution.py --models lr,production,universal:member_227 \
        --cache data/jwst_euclid_overlap/resolution_cache --out resolution.json

``lr`` is the Euclid H input itself, upsampled ×2 (cubic spline, flux kept):
Euclid's own resolution measured the same way. The per-tile JWST image,
resampled to the SR grid in Euclid-H electrons, is cached under ``--cache``
once; one field-wide sub-pixel shift (the median of the per-tile shifts of
the fully covered tiles, measured against the Euclid INPUT) registers every
tile; windows need full JWST coverage. Every model runs on the same tiles.

Results (JSON, per model) in three stretches of both images:
``linear`` (electrons: bright structure dominates), ``asinh_euclid`` (knee
at Euclid H's per-SR-pixel noise: structure counts by contrast wherever
Euclid can see it) and ``asinh_jwst`` (knee at JWST's noise: JWST's deep
faint sky too, most of which Euclid cannot detect — a ceiling, not a
target). ``knee_scan`` adds the ½-resolutions in asinh stretches at
``--knees`` multiples of Euclid's noise: which flux levels a model resolves.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
from astropy.wcs import WCS
from scipy.ndimage import zoom

from euclid_polish.ensemble import EnsembleModel
from euclid_polish.eval import jwst_resolution as jr
from euclid_polish.training.inference import scaled_wcs_header
from euclid_polish.web.helpers import model_catalog
from euclid_polish.web.helpers import real_tiles as rt

JWST_PIXEL_ARCSEC = 0.03
H = 3                       # Euclid H in the (VIS, Y, J, H) cubes


def _cached_tiles(cache: str, limit: int | None):
    """``(id, lr (h, w, 4) e⁻, jwst (2h, 2w) e⁻ on the SR grid, shift|None)``
    per NEXUS tile with any JWST coverage, from the cache or built into it
    (``shift``: the tile's own registration, fully covered tiles only)."""
    os.makedirs(cache, exist_ok=True)
    entries = [e for e in rt.list_entries("nexus") if e.extras.get("jwst_file")]
    for entry in entries[:limit]:
        path = os.path.join(cache, f"{entry.id}.npz")
        if not os.path.isfile(path):
            lr = np.asarray(rt.get_tile("nexus", entry.id, entry=entry).lr_e, np.float32)
            plane = rt.jwst_planes(entry)[0]
            sr_wcs = WCS(scaled_wcs_header(rt.lr_header(entry), 2)).celestial
            jwst = jr.jwst_on_grid(np.asarray(plane["data"], np.float64),
                                   WCS(plane["header"]).celestial, sr_wcs,
                                   (2 * lr.shape[0], 2 * lr.shape[1]), JWST_PIXEL_ARCSEC)
            shift = jr.registration_shift(lr[..., H], jwst)
            np.savez(path, lr=lr, jwst=jwst.astype(np.float32),
                     shift=np.asarray(shift if shift is not None else [], np.float64))
        with np.load(path) as data:
            if np.isfinite(data["jwst"]).any():
                yield (entry.id, data["lr"], data["jwst"],
                       tuple(data["shift"]) if data["shift"].size else None)


def _runner(spec: str):
    """``lr (h, w, 4) → SR H (2h, 2w)`` for a model spec, or the input itself."""
    if spec == "lr":
        return lambda lr: zoom(np.nan_to_num(lr[..., H].astype(np.float64)), 2, order=3) / 4.0
    item = model_catalog.resolve_spec(spec)
    if not item.available or len(item.reads) != 1:
        raise SystemExit(f"{spec}: not a runnable single model ({item.reason or 'a combiner'})")
    model = EnsembleModel(model_catalog.ensemble_dir(), labels=[item.reads[0]],
                          include_universal=True, starless=False)
    return lambda lr: np.asarray(model.member_arrays(np.ascontiguousarray(lr, np.float32))[0],
                                 np.float32)[..., H]


def _noise_sigma(images) -> float:
    """Per-pixel noise of a set of images: a clipped MAD of their pixels."""
    values = np.concatenate([x[np.isfinite(x)].ravel()[::7] for x in images])
    sigma = 1.4826 * float(np.median(np.abs(values - np.median(values))))
    for _ in range(3):
        core = values[np.abs(values - np.median(values)) < 3.0 * sigma]
        sigma = 1.4826 * float(np.median(np.abs(core - np.median(core))))
    return sigma


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--models", required=True, help="comma list of model specs (and/or lr)")
    p.add_argument("--cache", required=True, help="per-tile cache directory")
    p.add_argument("--out", required=True, help="JSON report (merged into when it exists)")
    p.add_argument("--tiles", type=int, default=None, help="at most this many tiles")
    p.add_argument("--knees", default="0.1,0.3,1,3,10,30,100",
                   help="asinh knees of the scan, in multiples of Euclid H's per-SR-pixel noise")
    args = p.parse_args()
    raw = list(_cached_tiles(args.cache, args.tiles))
    shifts = np.array([s for _t, _l, _j, s in raw if s is not None])
    if not len(shifts):
        raise SystemExit("no fully covered tile to measure the registration on")
    field_shift = tuple(float(v) for v in np.median(shifts, axis=0))
    tiles = [(tid, lr, jr.register(jwst, field_shift)) for tid, lr, jwst, _s in raw]
    sigma_j = _noise_sigma([j for _t, _l, j in tiles])
    # Euclid H's per-pixel noise of the input, referred to one SR pixel
    # (flux / 4 when one LR pixel becomes four).
    sigma_e = _noise_sigma([lr[..., H] for _t, lr, _j in tiles]) / 4.0
    report = json.load(open(args.out)) if os.path.isfile(args.out) else {"models": {}}
    report["meta"] = {
        "field": "NEXUS deep, JWST/NIRCam F200W vs Euclid H", "n_tiles": len(tiles),
        "jwst_noise_e_per_sr_px": round(sigma_j, 5), "euclid_h_noise_e_per_sr_px": round(sigma_e, 5),
        "registration": {"field_shift_px": [round(v, 3) for v in field_shift],
                         "tiles_measured": int(len(shifts)),
                         "spread_mad_px": [round(float(v), 3) for v in
                                           1.4826 * np.median(np.abs(shifts - np.median(shifts, axis=0)),
                                                              axis=0)]},
        "star_cut_e": jr.STAR_CUT_E, "window_px": jr.WINDOW, "pixel_arcsec": jr.SR_PIXEL_ARCSEC}
    print(f"{len(tiles)} tiles; noise per SR px: JWST {sigma_j:.4g}, Euclid H {sigma_e:.4g} e⁻; "
          f"registration {report['meta']['registration']}", flush=True)
    for spec in [s.strip() for s in args.models.split(",") if s.strip()]:
        started = time.monotonic()
        run = _runner(spec)
        accs = {"linear": jr.RingCorrelation(lambda x: x),
                "asinh_euclid": jr.RingCorrelation(lambda x, k=sigma_e: np.arcsinh(x / k)),
                "asinh_jwst": jr.RingCorrelation(lambda x, k=sigma_j: np.arcsinh(x / k))}
        multiples = [float(v) for v in args.knees.split(",") if v.strip()]
        scan = {m: jr.RingCorrelation(lambda x, k=m * sigma_e: np.arcsinh(x / k)) for m in multiples}
        for _tid, lr, jwst in tiles:
            sr = run(lr)
            for acc in (*accs.values(), *scan.values()):
                acc.add(sr, jwst)
        report["models"][spec] = {name: jr.summary(acc) for name, acc in accs.items()}
        report["models"][spec]["knee_scan"] = [
            {"multiple": m, "knee_e": round(m * sigma_e, 5),
             **{key: jr.summary(acc)[key] for key in ("resolution_coherence", "resolution_gain")}}
            for m, acc in scan.items()]
        line = ", ".join(
            f"{name} {(report['models'][spec][name]['resolution_coherence'] or {}).get('arcsec')}″"
            for name in accs)
        print(f"{spec}: {accs['linear'].windows} windows; ½-coherence {line} — "
              f"{time.monotonic() - started:.0f} s", flush=True)
        with open(args.out + ".tmp", "w") as handle:
            json.dump(report, handle, indent=1)
        os.replace(args.out + ".tmp", args.out)


if __name__ == "__main__":
    main()
