"""Source-centered synthetic evaluation (syn-lens / syn-gal).

Synthetic validation fields have HR truth, so we can measure whether SR moved a
source *toward* the truth. Unlike real lens cutouts these fields are crowded, so
instead of scoring the whole field (out of distribution for Zoobot) we crop M×M
HR-pixel postage stamps centered on one known source per field — a lens
(``syn-lens``) or a field galaxy (``syn-gal``) — using the sidecar source
catalog written at generation time. Every stamp is then a centered single object,
comparable to the real A/B/C lens cutouts. No network; needs the cached
``*_validate.tfrecord`` records + ``sources_validate.csv``.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional

from euclid_polish.config import Config


def default_records_dir() -> Optional[str]:
    """Local dir holding the validation TFRecords, or ``None`` if not present."""
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
    """PSNR (dB) over the overlapping region, peak = Config.PSNR_PEAK_E."""
    import numpy as np
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    h = min(a.shape[0], b.shape[0]); w = min(a.shape[1], b.shape[1])
    if h == 0 or w == 0:
        return None
    rmse = float(np.sqrt(np.mean((a[:h, :w] - b[:h, :w]) ** 2)) + 1e-9)
    return float(20.0 * np.log10(float(Config.PSNR_PEAK_E) / rmse))


def select_central_source(sources, want_type: str, *, field: int, m: int):
    """Most-central source of ``want_type`` whose m×m box fits in a ``field``-px
    grid (>= m/2 from every edge), or ``None``. Distance to (field/2, field/2)."""
    half = m / 2.0
    c = field / 2.0
    best = None
    best_d = None
    for s in sources:
        if s.get("type") != want_type:
            continue
        x, y = float(s["x_pix"]), float(s["y_pix"])
        if x < half or x > field - half or y < half or y > field - half:
            continue
        d = (x - c) ** 2 + (y - c) ** 2
        if best_d is None or d < best_d:
            best, best_d = s, d
    return best


def crop_stamp(plane, *, cx: float, cy: float, m: int):
    """Crop an m×m stamp from a 2-D ``plane`` centered at (cx, cy) pixel coords."""
    import numpy as np
    x0 = int(round(cx)) - m // 2
    y0 = int(round(cy)) - m // 2
    return np.asarray(plane)[y0:y0 + m, x0:x0 + m]


#: Analysis subgroups: (manifest grade, source ``type`` to center on).
_SUBGROUPS = (("syn-lens", "lens"), ("syn-gal", "galaxy"))


def run_synthetic_eval(
    out_dir: str, n: int, *,
    model=None,
    records_dir: Optional[str] = None,
    checkpoint: Optional[str] = None,
    num_res_blocks: Optional[int] = None,
    asinh_scale: Optional[float] = None,
    stamp_m: int = 64,
    seed: int = 0,
    on_progress: Optional[Callable[[int, int, str], None]] = None,
    log: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Crop up to N syn-lens + N syn-gal source-centered stamps into ``out_dir``.

    Returns ``{"rows": [...], "n_ok", "n_skip", "groups": {...}}``. Requires the
    sidecar source catalog; if absent returns no rows and logs a clear message
    (the grouped runner then runs A/B/C only). Writes LR/SR/HR FITS per stamp.
    """
    import numpy as np
    from astropy.io import fits

    from euclid_polish.sky.tfrecord import read_multiband_skyimages, tfrecord_path
    from euclid_polish.sky.source_catalog import read_sources
    from euclid_polish.training.inference import reconstruct
    from euclid_polish.eval.catalog_runner import load_eval_model

    def _emit(m): (log or print)(m)
    def _tick(i, total, lbl=""):
        if on_progress: on_progress(i, total, lbl)

    m = int(stamp_m)
    if m % 2:
        m += 1                                  # even → integer LR half-crop

    rdir = records_dir or default_records_dir()
    if rdir is None:
        raise FileNotFoundError(
            "validation records not found (dirty/hr_validate.tfrecord). Open the "
            "/inference page once to sync them, or set Config.RECORDS_DIR_V2.")
    src_csv = os.path.join(rdir, "sources_validate.csv")
    by_field = read_sources(src_csv)
    if not by_field:
        _emit("source catalog not found (sources_validate.csv) — regenerate the "
              "validation set with metadata; skipping syn-lens/syn-gal.")
        return {"rows": [], "n_ok": 0, "n_skip": 0, "groups": {}}

    # Read a window of records (enough to find N of each subgroup).
    window = max(n * 12, 60)
    lr_recs = read_multiband_skyimages(tfrecord_path(rdir, "dirty_validate"),
                                       num_images=window)
    hr_recs = read_multiband_skyimages(tfrecord_path(rdir, "hr_validate"),
                                       num_images=window)
    if not hr_recs:
        _emit("no HR validation records read.")
        return {"rows": [], "n_ok": 0, "n_skip": 0, "groups": {}}
    hr_by = {h.index: h for h in hr_recs}
    lr_by = {r.index: r for r in lr_recs}
    common = sorted(set(lr_by) & set(hr_by) & set(by_field))
    if not common:
        _emit("no validation fields with matching LR/HR + source catalog.")
        return {"rows": [], "n_ok": 0, "n_skip": 0, "groups": {}}

    # HR field size (HR is 2× LR; sources carry HR-grid coords).
    field = int(np.asarray(hr_recs[0].data, np.float32).shape[0])

    # Assign fields to subgroups (each field used at most once). Seeded order.
    rng = np.random.default_rng(seed)
    order = list(common)
    rng.shuffle(order)
    plan: List[tuple] = []                      # (field_index, grade, source)
    used = set()
    for grade, stype in _SUBGROUPS:
        taken = 0
        for idx in order:
            if idx in used or taken >= n:
                continue
            pick = select_central_source(by_field[idx], stype, field=field, m=m)
            if pick is None:
                continue
            plan.append((idx, grade, pick))
            used.add(idx); taken += 1
        if taken < n:
            _emit(f"{grade}: only {taken}/{n} fields had a fitting {stype}.")

    if not plan:
        _emit("no fittable sources found in the window.")
        return {"rows": [], "n_ok": 0, "n_skip": 0, "groups": {}}

    if model is None:
        model = load_eval_model(checkpoint, num_res_blocks)
    scale_hdr = float(asinh_scale or Config.STRETCH_SCALE_E)
    bands = ",".join(Config.LR_INPUT_BAND_NAMES)

    rows: List[Dict[str, Any]] = []
    n_ok = n_skip = 0
    total = len(plan)
    for j, (idx, grade, src) in enumerate(plan):
        _tick(j, total, f"{grade} idx {idx}")
        sub = f"{grade}_{idx:04d}"
        obj_dir = os.path.join(out_dir, sub)
        rec: Dict[str, Any] = {
            "id": sub, "ra": "", "dec": "", "grade": grade,
            "ok": False, "error": "", "out_subdir": sub,
            "lr_total_e": "", "sr_total_e": "", "flux_ratio_sr_over_lr": "",
            "psnr_lr_hr": "", "psnr_sr_hr": "",
        }
        try:
            os.makedirs(obj_dir, exist_ok=True)
            lr_cube = np.asarray(lr_by[idx].data, dtype=np.float32)   # (H,W,4)
            hr_raw = np.asarray(hr_by[idx].data, dtype=np.float32)
            hr_vis = hr_raw[..., 0] if hr_raw.ndim == 3 else hr_raw   # (2H,2W)
            _, sr_data = reconstruct(model, lr_cube)
            sr_arr = np.asarray(sr_data, dtype=np.float32)            # (2H,2W,4)

            cx, cy = float(src["x_pix"]), float(src["y_pix"])
            # HR & SR live on the HR grid; the LR cube is half-resolution.
            hr_st = crop_stamp(hr_vis, cx=cx, cy=cy, m=m)
            if sr_arr.ndim == 3:
                sr_cube_st = np.stack(
                    [crop_stamp(sr_arr[..., b], cx=cx, cy=cy, m=m)
                     for b in range(sr_arr.shape[-1])], axis=-1)
            else:
                sr_cube_st = crop_stamp(sr_arr, cx=cx, cy=cy, m=m)
            lr_cube_st = np.stack(
                [crop_stamp(lr_cube[..., b], cx=cx / 2.0, cy=cy / 2.0, m=m // 2)
                 for b in range(lr_cube.shape[-1])], axis=-1)
            sr_vis_st = sr_cube_st[..., 0] if sr_cube_st.ndim == 3 else sr_cube_st
            lr_vis_st = lr_cube_st[..., 0]

            def _wr(path, arr, obj, extra=None):
                hdr = fits.Header()
                hdr["OBJECT"] = obj
                hdr["BUNIT"] = "electron"
                hdr["ASINH"] = (scale_hdr, "asinh knee for the local renderer")
                hdr["SRCX"] = (cx, "source x_pix in HR field")
                hdr["SRCY"] = (cy, "source y_pix in HR field")
                if extra:
                    for k, v in extra.items():
                        hdr[k] = v
                fits.PrimaryHDU(np.ascontiguousarray(arr), header=hdr).writeto(
                    path, overwrite=True, output_verify="silentfix")

            _wr(os.path.join(obj_dir, "original_stack.fits"),
                np.moveaxis(lr_cube_st, -1, 0), f"{grade} LR stamp (electrons)",
                {"BANDS": (bands, "NAXIS3 plane order (band 0 = VIS)")})
            _wr(os.path.join(obj_dir, "SR.fits"),
                np.moveaxis(sr_cube_st, -1, 0) if sr_cube_st.ndim == 3 else sr_cube_st,
                f"{grade} SR stamp (WDSR)",
                {"BANDS": (bands, "NAXIS3 plane order (band 0 = VIS)")})
            _wr(os.path.join(obj_dir, "HR.fits"), hr_st, f"{grade} HR truth (VIS)")

            lr_sum, sr_sum = float(np.sum(lr_vis_st)), float(np.sum(sr_vis_st))
            lr_up = np.repeat(np.repeat(lr_vis_st, 2, 0), 2, 1)  # nearest 2× → HR grid
            rec.update({
                "ok": True,
                "lr_total_e": lr_sum, "sr_total_e": sr_sum,
                "flux_ratio_sr_over_lr": (sr_sum / lr_sum) if lr_sum else "",
                "psnr_lr_hr": _psnr(lr_up, hr_st),
                "psnr_sr_hr": _psnr(sr_vis_st, hr_st),
            })
            n_ok += 1
        except Exception as e:  # noqa: BLE001 — one bad field must not kill the run
            rec["error"] = f"{type(e).__name__}: {e}"
            _emit(f"  ! {sub} skipped: {rec['error']}")
            n_skip += 1
        rows.append(rec)

    groups: Dict[str, int] = {}
    for r in rows:
        if r.get("ok"):
            groups[r["grade"]] = groups.get(r["grade"], 0) + 1
    return {"rows": rows, "n_ok": n_ok, "n_skip": n_skip, "groups": groups}
