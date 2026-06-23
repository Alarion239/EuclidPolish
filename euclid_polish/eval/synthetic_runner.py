"""Evaluate the SR model on the synthetic validation set (with HR truth).

Loads N triptychs from the cached ``*_validate.tfrecord`` records, runs the
model, and writes per-object ``original_stack.fits`` (LR) + ``SR.fits`` +
``HR.fits``. Unlike real lens cutouts, the synthetic set has ground truth, so
we also record SR-vs-HR and LR-vs-HR PSNR — the "did SR move toward the truth?"
signal. Returns manifest rows (the grouped runner aggregates them); no network.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional

from euclid_polish.config import Config


def default_records_dir() -> Optional[str]:
    """Local dir holding the validation TFRecords, or ``None`` if not present.

    Prefers ``Config.RECORDS_DIR_V2``; falls back to the FASRC rsync cache the
    sky/inference pages populate.
    """
    cand = [Config.RECORDS_DIR_V2]
    try:                                        # web-layer cache resolver
        from euclid_polish.web.helpers.paths import _sky_records_local_dir
        cand.append(_sky_records_local_dir())
    except Exception:                           # noqa: BLE001 — optional
        pass
    for d in cand:
        if d and os.path.isfile(os.path.join(d, "dirty_validate.tfrecord")):
            return d
    return None


def _psnr(a, b) -> Optional[float]:
    """PSNR (dB) of ``a`` vs ``b`` on the overlapping region, peak = mag-17
    star (Config.PSNR_PEAK_E). ``None`` if shapes are empty."""
    import numpy as np
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    h = min(a.shape[0], b.shape[0]); w = min(a.shape[1], b.shape[1])
    if h == 0 or w == 0:
        return None
    rmse = float(np.sqrt(np.mean((a[:h, :w] - b[:h, :w]) ** 2)) + 1e-9)
    return float(20.0 * np.log10(float(Config.PSNR_PEAK_E) / rmse))


def run_synthetic_eval(
    out_dir: str, n: int, *,
    model=None,
    records_dir: Optional[str] = None,
    checkpoint: Optional[str] = None,
    num_res_blocks: Optional[int] = None,
    asinh_scale: Optional[float] = None,
    seed: int = 0,
    on_progress: Optional[Callable[[int, int, str], None]] = None,
    log: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Reconstruct a seeded-random N of validation triptychs into ``out_dir``.

    Returns ``{"rows": [...], "n_ok": int, "n_skip": int}``; the rows carry
    ``grade="synthetic"`` + flux/PSNR metrics. Writes LR/SR/HR FITS per object.
    """
    import numpy as np
    from astropy.io import fits

    from euclid_polish.sky.tfrecord import read_multiband_skyimages, tfrecord_path
    from euclid_polish.training.inference import reconstruct
    from euclid_polish.eval.catalog_runner import load_eval_model

    def _emit(m): (log or print)(m)
    def _tick(i, total, lbl=""):
        if on_progress: on_progress(i, total, lbl)

    rdir = records_dir or default_records_dir()
    if rdir is None:
        raise FileNotFoundError(
            "validation records not found (dirty/hr_validate.tfrecord). "
            "Open the /inference page once to sync them from FASRC, or place "
            "them under Config.RECORDS_DIR_V2.")

    # Read a bounded window, then seeded-sample N indices present in both files.
    window = max(n * 5, 50)
    lr_recs = read_multiband_skyimages(tfrecord_path(rdir, "dirty_validate"),
                                       num_images=window)
    hr_recs = read_multiband_skyimages(tfrecord_path(rdir, "hr_validate"),
                                       num_images=window)
    hr_by = {h.index: h for h in hr_recs}
    lr_by = {r.index: r for r in lr_recs}
    common = sorted(set(lr_by) & set(hr_by))
    if not common:
        raise RuntimeError("no matching LR/HR validation indices found")
    rng = np.random.default_rng(seed)
    pick = sorted(rng.choice(common, size=min(n, len(common)), replace=False).tolist())
    _emit(f"synthetic: {len(pick)} of {len(common)} validation triptychs")

    if model is None:
        model = load_eval_model(checkpoint, num_res_blocks)
    scale_hdr = float(asinh_scale or Config.STRETCH_SCALE_E)
    bands = ",".join(Config.LR_INPUT_BAND_NAMES)

    rows: List[Dict[str, Any]] = []
    n_ok = n_skip = 0
    for j, idx in enumerate(pick):
        _tick(j, len(pick), f"synthetic idx {idx}")
        sub = f"synthetic_{idx:04d}"
        obj_dir = os.path.join(out_dir, sub)
        rec: Dict[str, Any] = {
            "id": sub, "ra": "", "dec": "", "grade": "synthetic",
            "ok": False, "error": "", "out_subdir": sub,
            "lr_total_e": "", "sr_total_e": "", "flux_ratio_sr_over_lr": "",
            "psnr_lr_hr": "", "psnr_sr_hr": "",
        }
        try:
            os.makedirs(obj_dir, exist_ok=True)
            lr_cube = np.asarray(lr_by[idx].data, dtype=np.float32)   # (H,W,4)
            hr_raw = np.asarray(hr_by[idx].data, dtype=np.float32)
            hr_vis = hr_raw[..., 0] if hr_raw.ndim == 3 else hr_raw   # (2H,2W)
            lr_vis, sr_data = reconstruct(model, lr_cube)
            sr_arr = np.asarray(sr_data, dtype=np.float32)
            sr_vis = sr_arr[..., 0] if sr_arr.ndim == 3 else sr_arr

            def _wr(path, arr, obj, extra=None):
                hdr = fits.Header()
                hdr["OBJECT"] = obj
                hdr["BUNIT"] = "electron"
                hdr["ASINH"] = (scale_hdr, "asinh knee for the local renderer")
                if extra:
                    for k, v in extra.items():
                        hdr[k] = v
                fits.PrimaryHDU(np.ascontiguousarray(arr), header=hdr).writeto(
                    path, overwrite=True, output_verify="silentfix")

            _wr(os.path.join(obj_dir, "original_stack.fits"),
                np.moveaxis(lr_cube, -1, 0), "Synthetic LR stack (electrons)",
                {"BANDS": (bands, "NAXIS3 plane order (band 0 = VIS)")})
            if sr_arr.ndim == 3:
                _wr(os.path.join(obj_dir, "SR.fits"),
                    np.moveaxis(sr_arr, -1, 0), "Synthetic SR (WDSR, 4-band)",
                    {"BANDS": (bands, "NAXIS3 plane order (band 0 = VIS)")})
            else:
                _wr(os.path.join(obj_dir, "SR.fits"), sr_arr,
                    "Synthetic SR (WDSR VIS)")
            _wr(os.path.join(obj_dir, "HR.fits"), hr_vis,
                "Synthetic HR truth (VIS)")

            lr_sum, sr_sum = float(np.sum(lr_vis)), float(np.sum(sr_vis))
            lr_up = np.repeat(np.repeat(lr_vis, 2, 0), 2, 1)   # nearest 2× → HR grid
            rec.update({
                "ok": True,
                "lr_total_e": lr_sum, "sr_total_e": sr_sum,
                "flux_ratio_sr_over_lr": (sr_sum / lr_sum) if lr_sum else "",
                "psnr_lr_hr": _psnr(lr_up, hr_vis),
                "psnr_sr_hr": _psnr(sr_vis, hr_vis),
            })
            n_ok += 1
        except Exception as e:  # noqa: BLE001 — one bad record must not kill the run
            rec["error"] = f"{type(e).__name__}: {e}"
            _emit(f"  ! {sub} skipped: {rec['error']}")
            n_skip += 1
        rows.append(rec)

    return {"rows": rows, "n_ok": n_ok, "n_skip": n_skip}
