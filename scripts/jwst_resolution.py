"""Resolution of models against JWST/NIRCam on the NEXUS field: the Fourier
ring correlation of each model's Euclid band with the nearest NIRCam filter,
and the period where it falls to ½ (see
:mod:`euclid_polish.eval.jwst_resolution`).

    python scripts/jwst_resolution.py --band H_E --models lr,production \\
        --cache data/jwst_euclid_overlap/resolution_cache --out resolution_H_E.json

``--band`` picks the Euclid band and its NIRCam partner (VIS–F090W,
Y–F115W, J–F150W, H–F200W). H uses the NEXUS field's own per-tile F200W
images (quick release, deep epoch 5); the other bands — or any band with
``--mosaic`` — sample a NIRCam mosaic (default: the early-release wide
epoch-1 pointing 01, which covers the field), streamed row by row (tiles in
mosaic row order, so the gzip plane is decompressed once). ``lr`` is the
Euclid input band itself, upsampled ×2 (cubic spline, flux kept): Euclid's
own resolution measured the same way. ``milestone:member_NNN@STEP`` runs a
long run's milestone checkpoint (``euclid_polish/web/helpers/milestones.py``).

Per tile the JWST image is resampled to the SR grid in the band's electrons
and cached under ``--cache`` once; one field-wide sub-pixel shift (the median
of the per-tile shifts of the fully covered tiles, measured against the
Euclid INPUT) registers every tile; a window needs ``--min-coverage`` of
its pixels covered (the early-release mosaics have thin detector gaps; both
images share one apodized coverage mask).

Results (JSON, per model) in three stretches of both images: ``linear``
(electrons: bright structure dominates), ``asinh_euclid`` (knee at the
Euclid band's per-SR-pixel noise: structure counts by contrast wherever
Euclid can see it) and ``asinh_jwst`` (knee at JWST's noise: JWST's deep
faint sky too, most of which Euclid cannot detect — a ceiling, not a
target). ``knee_scan`` adds the ½-resolutions in asinh stretches at
``--knees`` multiples of Euclid's noise, each with its surface brightness:
which flux levels a model resolves.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
from astropy.wcs import WCS
from scipy.ndimage import zoom

from euclid_polish.config import Config
from euclid_polish.ensemble import EnsembleModel
from euclid_polish.eval import jwst_resolution as jr
from euclid_polish.training.inference import scaled_wcs_header
from euclid_polish.web.helpers import fits_plane, jwst_euclid, milestones, model_catalog
from euclid_polish.web.helpers import real_tiles as rt

JWST_PIXEL_ARCSEC = 0.03
BANDS = tuple(Config.LR_INPUT_BAND_NAMES)              # (VIS, Y_E, J_E, H_E)
#: The early-release wide epoch-1 pointing 01 mosaics (they cover the field).
EDR_MOSAIC = "nexus_mosaics/edr_wide_ep1_01/wide_ep1_01_{filter}_i2d_data.fits.gz"


def _entries(limit: int | None, ids: set[str] | None = None):
    return [e for e in rt.list_entries("nexus")
            if e.extras.get("jwst_file") and (ids is None or e.id in ids)][:limit]


def _lr(cache: str, entry) -> np.ndarray:
    """The tile's Euclid input cube (h, w, 4) e⁻ — from the base cache when it is there."""
    path = os.path.join(cache, f"{entry.id}.npz")
    if os.path.isfile(path):
        with np.load(path) as data:
            return data["lr"]
    return np.asarray(rt.get_tile("nexus", entry.id, entry=entry).lr_e, np.float32)


def _sr_wcs(entry):
    return WCS(scaled_wcs_header(rt.lr_header(entry), 2)).celestial


def _tile_jwst(cache: str, entries, band: str):
    """``{id: (lr, jwst on the SR grid, shift|None)}`` from the field's own
    per-tile F200W images (the base cache, built when missing)."""
    out = {}
    for entry in entries:
        path = os.path.join(cache, f"{entry.id}.npz")
        if not os.path.isfile(path):
            lr = _lr(cache, entry)
            plane = rt.jwst_planes(entry)[0]
            jwst = jr.jwst_on_grid(np.asarray(plane["data"], np.float64), WCS(plane["header"]).celestial,
                                   _sr_wcs(entry), (2 * lr.shape[0], 2 * lr.shape[1]), JWST_PIXEL_ARCSEC,
                                   to_e=jr.mjysr_to_e(band))
            shift = jr.registration_shift(lr[..., BANDS.index(band)], jwst, jr.EUCLID_FWHM_ARCSEC[band])
            np.savez(path, lr=lr, jwst=jwst.astype(np.float32),
                     shift=np.asarray(shift if shift is not None else [], np.float64))
        with np.load(path) as data:
            out[entry.id] = (data["lr"], data["jwst"], tuple(data["shift"]) if data["shift"].size else None)
    return out


def _mosaic_jwst(cache: str, entries, band: str, mosaic: str):
    """``{id: (lr, jwst on the SR grid, shift|None)}`` sampled from a NIRCam
    mosaic, tiles taken in mosaic row order (one forward pass of a gzip plane)."""
    tag = f"{band}-{os.path.basename(mosaic).split('_i2d')[0]}"
    folder = os.path.join(cache, tag)
    os.makedirs(folder, exist_ok=True)
    missing = [e for e in entries if not os.path.isfile(os.path.join(folder, f"{e.id}.npz"))]
    if missing:
        to_e = jr.mjysr_to_e(band)
        with fits_plane.open_plane(mosaic) as plane:
            ny, nx = plane.shape

            def bounds(entry):
                lr_shape = _lr(cache, entry).shape
                h, w = 2 * int(lr_shape[0]), 2 * int(lr_shape[1])
                ys, xs = np.array([0, 0, h - 1, h - 1]), np.array([0, w - 1, 0, w - 1])
                mx, my = plane.wcs.world_to_pixel_values(*_sr_wcs(entry).pixel_to_world_values(xs, ys))
                return (max(0, int(np.floor(my.min())) - 8), min(ny, int(np.ceil(my.max())) + 9),
                        max(0, int(np.floor(mx.min())) - 8), min(nx, int(np.ceil(mx.max())) + 9), (h, w))

            ordered = sorted(((bounds(e), e) for e in missing), key=lambda item: item[0][0])
            for n, ((y0, y1, x0, x1, shape), entry) in enumerate(ordered, 1):
                lr = _lr(cache, entry)
                if y1 <= y0 or x1 <= x0:
                    jwst = np.full(shape, np.nan, np.float32)
                else:
                    cut = plane[y0:y1, x0:x1]
                    jwst = jr.jwst_on_grid(np.asarray(cut, np.float64),
                                           plane.wcs.slice((slice(y0, y1), slice(x0, x1))),
                                           _sr_wcs(entry), shape, JWST_PIXEL_ARCSEC, to_e=to_e)
                shift = jr.registration_shift(lr[..., BANDS.index(band)], jwst,
                                              jr.EUCLID_FWHM_ARCSEC[band])
                np.savez(os.path.join(folder, f"{entry.id}.npz"), jwst=jwst.astype(np.float32),
                         shift=np.asarray(shift if shift is not None else [], np.float64))
                if n % 50 == 0:
                    print(f"  sampled {n}/{len(ordered)} tiles from {os.path.basename(mosaic)}", flush=True)
    out = {}
    for entry in entries:
        with np.load(os.path.join(folder, f"{entry.id}.npz")) as data:
            out[entry.id] = (_lr(cache, entry), data["jwst"],
                             tuple(data["shift"]) if data["shift"].size else None)
    return out


def _runner(spec: str, band: str):
    """``lr (h, w, 4) → SR band (2h, 2w)`` for a model spec, or the input itself."""
    b = BANDS.index(band)
    if spec == "lr":
        return lambda lr: zoom(np.nan_to_num(lr[..., b].astype(np.float64)), 2, order=3) / 4.0
    if spec.startswith("milestone:"):                 # a long run's kept state (helpers/milestones.py)
        model = milestones.load(*milestones.parse_spec(spec))
        return lambda lr: np.asarray(model.member_arrays(np.ascontiguousarray(lr, np.float32))[0],
                                     np.float32)[..., b]
    item = model_catalog.resolve_spec(spec)
    if not item.available or len(item.reads) != 1:
        raise SystemExit(f"{spec}: not a runnable single model ({item.reason or 'a combiner'})")
    model = EnsembleModel(model_catalog.ensemble_dir(), labels=[item.reads[0]],
                          include_universal=True, starless=False)
    return lambda lr: np.asarray(model.member_arrays(np.ascontiguousarray(lr, np.float32))[0],
                                 np.float32)[..., b]


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
    p.add_argument("--band", default="H_E", choices=BANDS, help="the Euclid band (default H_E)")
    p.add_argument("--mosaic", default=None,
                   help="a NIRCam mosaic to sample (default: H the field's own F200W tiles, "
                        "other bands the early-release wide epoch-1 mosaic of their filter)")
    p.add_argument("--cache", required=True, help="per-tile cache directory")
    p.add_argument("--out", required=True, help="JSON report (merged into when it exists)")
    p.add_argument("--tiles", type=int, default=None, help="at most this many tiles")
    p.add_argument("--tile-ids", default=None,
                   help="comma list of NEXUS tile ids to measure on (e.g. the tiles of chosen galaxies)")
    p.add_argument("--min-coverage", type=float, default=0.95,
                   help="fraction of a window both images must cover (default 0.95)")
    p.add_argument("--knees", default="0.1,0.3,1,3,10,30,100,300,1000",
                   help="asinh knees of the scan, in multiples of the Euclid band's per-SR-pixel noise")
    args = p.parse_args()
    band, b = args.band, BANDS.index(args.band)
    ids = {t.strip() for t in args.tile_ids.split(",") if t.strip()} if args.tile_ids else None
    entries = _entries(args.tiles, ids)
    mosaic = args.mosaic
    if mosaic is None and band != "H_E":
        mosaic = os.path.join(str(jwst_euclid.overlap_root()),
                              EDR_MOSAIC.format(filter=jr.JWST_FILTER[band].lower()))
    raw = (_mosaic_jwst(args.cache, entries, band, mosaic) if mosaic
           else _tile_jwst(args.cache, entries, band))
    raw = {tid: v for tid, v in raw.items() if np.isfinite(v[1]).any()}
    shifts = np.array([s for _l, _j, s in raw.values() if s is not None])
    if not len(shifts):
        raise SystemExit("no fully covered tile to measure the registration on")
    field_shift = tuple(float(v) for v in np.median(shifts, axis=0))
    tiles = [(tid, lr, jr.register(jwst, field_shift)) for tid, (lr, jwst, _s) in raw.items()]
    sigma_j = _noise_sigma([j for _t, _l, j in tiles])
    # The Euclid band's per-pixel noise of the input, referred to one SR
    # pixel (flux / 4 when one LR pixel becomes four).
    sigma_e = _noise_sigma([lr[..., b] for _t, lr, _j in tiles]) / 4.0
    report = json.load(open(args.out)) if os.path.isfile(args.out) else {"models": {}}
    report["meta"] = {
        "field": f"NEXUS, JWST/NIRCam {jr.JWST_FILTER[band]} vs Euclid {band}",
        "band": band, "jwst_filter": jr.JWST_FILTER[band],
        "jwst_source": os.path.basename(mosaic) if mosaic else "nexus field tiles (QDR deep ep05)",
        "mjysr_to_e": round(jr.mjysr_to_e(band), 4), "n_tiles": len(tiles),
        "jwst_noise_e_per_sr_px": round(sigma_j, 5), "euclid_noise_e_per_sr_px": round(sigma_e, 5),
        "euclid_noise_sb": round(jr.sb_of_knee(band, sigma_e), 3),
        "registration": {"field_shift_px": [round(v, 3) for v in field_shift],
                         "tiles_measured": int(len(shifts)),
                         "spread_mad_px": [round(float(v), 3) for v in
                                           1.4826 * np.median(np.abs(shifts - np.median(shifts, axis=0)),
                                                              axis=0)]},
        "star_cut_e": jr.STAR_CUT_E, "window_px": jr.WINDOW, "pixel_arcsec": jr.SR_PIXEL_ARCSEC,
        "min_coverage": args.min_coverage}
    print(f"{band}: {len(tiles)} tiles; noise per SR px: JWST {sigma_j:.4g}, Euclid {sigma_e:.4g} e⁻ "
          f"(μ {report['meta']['euclid_noise_sb']}); registration {report['meta']['registration']}",
          flush=True)
    multiples = [float(v) for v in args.knees.split(",") if v.strip()]
    for spec in [s.strip() for s in args.models.split(",") if s.strip()]:
        started = time.monotonic()
        run = _runner(spec, band)
        cover = args.min_coverage
        accs = {"linear": jr.RingCorrelation(lambda x: x, cover),
                "asinh_euclid": jr.RingCorrelation(lambda x, k=sigma_e: np.arcsinh(x / k), cover),
                "asinh_jwst": jr.RingCorrelation(lambda x, k=sigma_j: np.arcsinh(x / k), cover)}
        scan = {m: jr.RingCorrelation(lambda x, k=m * sigma_e: np.arcsinh(x / k), cover)
                for m in multiples}
        for _tid, lr, jwst in tiles:
            sr = run(lr)
            for acc in (*accs.values(), *scan.values()):
                acc.add(sr, jwst)
        report["models"][spec] = {name: jr.summary(acc) for name, acc in accs.items()}
        report["models"][spec]["knee_scan"] = [
            {"multiple": m, "knee_e": round(m * sigma_e, 5),
             "sb": round(jr.sb_of_knee(band, m * sigma_e), 3),
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
