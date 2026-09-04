"""Source-centered synthetic evaluation (syn-lens / syn-gal).

Synthetic validation fields have HR truth, so we can measure whether SR moved a
source *toward* the truth. Unlike real lens cutouts these fields are crowded, so
we crop M×M HR-pixel postage stamps centered on one known source per field — a lens
(``syn-lens``) or a field galaxy (``syn-gal``) — using the sidecar source
catalog written at generation time. Every stamp is then a centered single object,
comparable to the real A/B/C lens cutouts. No network; needs the cached
``*_validate.tfrecord`` records + ``sources_validate.csv``.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable
from typing import Any

from euclid_polish.config import Config
from euclid_polish.eval.catalog_runner import EVAL_HR_SIZE, EVAL_LR_SIZE, enforce_object_sizes
from euclid_polish.eval.disagreement import write_disagreement_cubes
from euclid_polish.eval.ensemble_infer import load_eval_ensemble
from euclid_polish.eval.stamp_geometry import crop_stamp  # re-export (back-compat)
from euclid_polish.sky.generation.source_catalog import source_is_off_field


def default_records_dir() -> str | None:
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


def _psnr(a, b) -> float | None:
    """PSNR (dB) over the overlapping region, peak = Config.PSNR_PEAK_E."""
    import numpy as np
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    h = min(a.shape[0], b.shape[0]); w = min(a.shape[1], b.shape[1])
    if h == 0 or w == 0:
        return None
    rmse = float(np.sqrt(np.mean((a[:h, :w] - b[:h, :w]) ** 2)) + 1e-9)
    return float(20.0 * np.log10(float(Config.PSNR_PEAK_E) / rmse))


def select_central_source(
    sources, want_type: str, *, field: int, m: int, prefer_bright: bool = False
):
    """Pick a fittable source of ``want_type`` in a ``field``-px grid.

    By default this preserves the original behavior: choose the source nearest
    the field center. For ``syn-gal`` we can instead prefer high VIS flux so the
    selected galaxy is visible above the noise floor, with centrality as the
    tie-breaker.
    """
    half = m / 2.0
    c = field / 2.0
    best = None
    best_key = None
    for s in sources:
        if s.get("type") != want_type or source_is_off_field(s):
            continue
        x, y = float(s["x_pix"]), float(s["y_pix"])
        if x < half or x > field - half or y < half or y > field - half:
            continue
        d = (x - c) ** 2 + (y - c) ** 2
        if prefer_bright:
            try:
                flux = float(s.get("flux_vis_e", 0.0))
            except (TypeError, ValueError):
                flux = 0.0
            key = (-flux, d)
        else:
            key = (d,)
        if best_key is None or key < best_key:
            best, best_key = s, key
    return best


def _fitting_sources(sources, want_type: str, *, field: int, m: int):
    """All sources of ``want_type`` whose ``m``-px stamp fits in the field,
    brightest (VIS flux) first.

    Crowded synthetic fields host many galaxies; returning every fittable one
    (not just the central pick) lets the planner take several per field to meet
    the target. A source is fittable when its center is ≥ ``m/2`` from each edge
    so the centered crop stays on-grid.
    """
    half = m / 2.0
    out = []
    for s in sources:
        if s.get("type") != want_type or source_is_off_field(s):
            continue
        x, y = float(s["x_pix"]), float(s["y_pix"])
        if x < half or x > field - half or y < half or y > field - half:
            continue
        try:
            flux = float(s.get("flux_vis_e", 0.0))
        except (TypeError, ValueError):
            flux = 0.0
        out.append((flux, s))
    out.sort(key=lambda t: -t[0])               # brightest first
    return [s for _, s in out]


#: Analysis subgroups: (manifest grade, source ``type`` to center on).
_SUBGROUPS = (("syn-lens", "lens"), ("syn-gal", "galaxy"))


def run_synthetic_eval(
    out_dir: str, n: int, *,
    model=None,
    records_dir: str | None = None,
    ensemble_dir: str | None = None,
    num_res_blocks: int | None = None,
    asinh_scale: float | None = None,
    stamp_m: int = EVAL_HR_SIZE,
    seed: int = 0,
    on_progress: Callable[[int, int, str], None] | None = None,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Crop up to N syn-lens + N syn-gal source-centered stamps into ``out_dir``.

    Returns ``{"rows": [...], "n_ok", "n_skip", "groups": {...}}``. Requires the
    sidecar source catalog; if absent returns no rows and logs a clear message
    (the grouped runner then runs A/B/C only). Writes LR/SR/raw-HR/BHR FITS per
    stamp, where BHR is the PSF-stabilised supervision target.
    """
    import numpy as np
    from astropy.io import fits

    from euclid_polish.eval.ensemble_cube_cache import (
        cached_member_labels,
        load_cached_member_stack,
    )
    from euclid_polish.eval.ensemble_infer import sr_from_model
    from euclid_polish.eval.subsets import eval_subset
    from euclid_polish.image.tfio import read_images, tfrecord_path
    from euclid_polish.sky.generation.source_catalog import read_sources
    from euclid_polish.training.target_blur import blur_target_array

    def _emit(m): (log or print)(m)
    if on_progress is None:                     # local/CLI run → visible bar
        from euclid_polish.eval.progress import tqdm_progress
        on_progress = tqdm_progress("synthetic")
    def _tick(i, total, lbl=""):
        if on_progress: on_progress(i, total, lbl)

    m = int(stamp_m)
    if m % 2:
        m += 1                                  # even → integer LR half-crop

    rdir = records_dir or default_records_dir()
    if rdir is None:
        raise FileNotFoundError(
            "eval records not found (dirty/hr_test.tfrecord). Open the "
            "/inference page once to sync them, or set Config.RECORDS_DIR_V2.")
    # Held-out test split when present, else validate (pre-test-split datasets).
    sub = eval_subset(rdir)
    field_subset = sub                          # preserve subset; `sub` is reused as the
                                                # per-object id inside the loop below
    src_csv = os.path.join(rdir, f"sources_{sub}.csv")
    by_field = read_sources(src_csv)
    if not by_field:
        _emit(f"source catalog not found (sources_{sub}.csv) — regenerate the "
              f"{sub} set with metadata; skipping syn-lens/syn-gal.")
        return {"rows": [], "n_ok": 0, "n_skip": 0, "groups": {}}

    # Read a window of records (enough to find N of each subgroup).
    window = max(n * 12, 60)
    lr_recs = read_images(tfrecord_path(rdir, f"dirty_{sub}"),
                                       num_images=window)
    hr_recs = read_images(tfrecord_path(rdir, f"hr_{sub}"),
                          num_images=window)
    raw_hr_by = {
        image.index: np.array(image.data, dtype=np.float32, copy=True)
        for image in hr_recs
        if image.index is not None
    }
    for image in hr_recs:
        image.data = blur_target_array(
            image.data,
            Config.TARGET_PSF_FWHM_ARCSEC,
            pixel_scale_arcsec=getattr(
                image, "pixel_scale_arcsec", Config.DEFAULT_PIXEL_SCALE),
        )
    if not hr_recs:
        _emit(f"no HR {sub} records read.")
        return {"rows": [], "n_ok": 0, "n_skip": 0, "groups": {}}
    hr_by = {h.index: h for h in hr_recs if h.index is not None}
    lr_by = {r.index: r for r in lr_recs if r.index is not None}
    common = sorted(set(lr_by) & set(hr_by) & set(by_field))
    if not common:
        _emit("no validation fields with matching LR/HR + source catalog.")
        return {"rows": [], "n_ok": 0, "n_skip": 0, "groups": {}}

    # HR field size (HR is 2× LR; sources carry HR-grid coords).
    field = int(np.asarray(hr_recs[0].data, np.float32).shape[0])

    # Plan up to N stamps per subgroup. To hit the target we take MULTIPLE
    # sources per field (crowded fields host many galaxies) — the brightest
    # first — distributed round-robin across fields, so on average ≈ N / #fields
    # come from each. Lens fields carry about one lens, so syn-lens is still
    # bounded by the number of lens-bearing fields.
    rng = np.random.default_rng(seed)
    order = list(common)
    rng.shuffle(order)
    plan: list[tuple] = []                      # (field_index, grade, source, rank)
    for grade, stype in _SUBGROUPS:
        by_idx = {idx: _fitting_sources(by_field[idx], stype, field=field, m=m)
                  for idx in order}
        by_idx = {idx: lst for idx, lst in by_idx.items() if lst}
        taken, rank = 0, 0
        while taken < n:
            advanced = False
            for idx in order:                   # one (the next-brightest) per field
                lst = by_idx.get(idx)
                if lst is not None and rank < len(lst):
                    plan.append((idx, grade, lst[rank], rank))
                    taken += 1
                    advanced = True
                    if taken >= n:
                        break
            if not advanced:
                break                           # every field exhausted
            rank += 1
        msg = (f"{grade}: {taken}/{n} stamps across {len(by_idx)} field(s)"
               + (" (brightest first)" if taken == n else
                  " — short of target (brightest taken first)"))
        _emit(msg)

    if not plan:
        _emit("no fittable sources found in the window.")
        return {"rows": [], "n_ok": 0, "n_skip": 0, "groups": {}}

    plan.sort(key=lambda t: (t[0], t[1], t[3]))  # group by field → reconstruct once

    if model is None:
        model = load_eval_ensemble(ensemble_dir, num_res_blocks, log=_emit)
    scale_hdr = float(asinh_scale or Config.STRETCH_SCALE_E)
    bands = ",".join(Config.LR_INPUT_BAND_NAMES)

    rows: list[dict[str, Any]] = []
    n_ok = n_skip = 0
    total = len(plan)
    cur_idx = None                              # SR is per-field; reconstruct once
    lr_cube = sr_arr = members_full = None
    member_labels: list[str] = []               # fingerprint for members.json
    for j, (idx, grade, src, rank) in enumerate(plan):
        _tick(j, total, f"{grade} idx {idx}")
        sub = f"{grade}_{idx:04d}_{rank}"       # unique per (field, brightness rank)
        obj_dir = os.path.join(out_dir, sub)
        rec: dict[str, Any] = {
            "id": sub, "ra": "", "dec": "", "grade": grade,
            "ok": False, "error": "", "out_subdir": sub,
            "lr_total_e": "", "sr_total_e": "", "flux_ratio_sr_over_lr": "",
            "psnr_lr_hr": "", "psnr_sr_hr": "",
        }
        try:
            os.makedirs(obj_dir, exist_ok=True)
            if idx != cur_idx:                  # same field → reuse per-field arrays
                lr_cube = np.asarray(lr_by[idx].data, dtype=np.float32)   # (H,W,4)
                cached = load_cached_member_stack(idx, subset=field_subset)
                if cached is not None:          # reuse the ensemble page's cubes
                    members_full = cached                              # (M,2H,2W,C)
                    member_labels = cached_member_labels() or []
                    sr_arr = members_full.mean(axis=0).astype(np.float32)
                    _emit(f"  field {idx}: reused ensemble cache "
                          f"({members_full.shape[0]} members)")
                    if members_full.shape[0] < 2:   # ensemble of 1 → no cubes
                        members_full = None
                else:                           # no cache → run inference on the field
                    _, sr_data, members_full = sr_from_model(model, lr_cube)
                    member_labels = list(model.member_labels)
                    sr_arr = np.asarray(sr_data, dtype=np.float32)     # (2H,2W,C)
                    _emit(f"  field {idx}: inference")
                cur_idx = idx
            if lr_cube is None or sr_arr is None:
                raise RuntimeError(f"field {idx} reconstruction arrays were not initialized")
            bhr_raw = np.asarray(hr_by[idx].data, dtype=np.float32)
            hr_raw = raw_hr_by[idx]

            cx, cy = float(src["x_pix"]), float(src["y_pix"])
            # HR & SR live on the HR grid; the LR cube is half-resolution.
            if bhr_raw.ndim == 3:
                hr_cube_st = np.stack(
                    [crop_stamp(bhr_raw[..., b], cx=cx, cy=cy, m=m)
                     for b in range(bhr_raw.shape[-1])], axis=-1)
                raw_hr_cube_st = np.stack(
                    [crop_stamp(hr_raw[..., b], cx=cx, cy=cy, m=m)
                     for b in range(hr_raw.shape[-1])], axis=-1)
                hr_vis_st = hr_cube_st[..., 0]
            else:                                   # legacy VIS-only HR record
                hr_vis_st = crop_stamp(bhr_raw, cx=cx, cy=cy, m=m)
                hr_cube_st = hr_vis_st[..., None]
                raw_hr_cube_st = crop_stamp(
                    hr_raw, cx=cx, cy=cy, m=m)[..., None]
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

            def _wr(path, arr, obj, extra=None, *, cx=cx, cy=cy):
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
            _wr(os.path.join(obj_dir, "HR.fits"),
                np.moveaxis(raw_hr_cube_st, -1, 0),
                f"{grade} HR truth (electrons)",
                {"BANDS": (bands, "NAXIS3 plane order (band 0 = VIS)")})
            _wr(os.path.join(obj_dir, "BHR.fits"),
                np.moveaxis(hr_cube_st, -1, 0),
                f"{grade} blurred HR target (electrons)",
                {"BANDS": (bands, "NAXIS3 plane order (band 0 = VIS)"),
                 "TARGFWH": (float(Config.TARGET_PSF_FWHM_ARCSEC),
                              "Gaussian target PSF FWHM [arcsec]")})

            # Ensemble disagreement cubes, cropped to the SAME source stamp as
            # SR.fits (crop_stamp at cx,cy) so the movie overlays exactly. No-op
            # for a single model. members_full is (M, 2H, 2W, C).
            if members_full is not None:
                try:
                    mem_st = np.stack([
                        np.stack([crop_stamp(mem[..., b], cx=cx, cy=cy, m=m)
                                  for b in range(mem.shape[-1])], axis=-1)
                        for mem in np.asarray(members_full, dtype=np.float32)
                    ], axis=0)                                   # (M, m, m, C)
                    write_disagreement_cubes(obj_dir, mem_st,
                                             member_labels=member_labels)
                except Exception as exc:  # noqa: BLE001
                    _emit(f"  [disagreement] {sub}: cubes not written: {exc}")

            # Hold the canonical eval geometry (LR EVAL_LR_SIZE², SR/HR
            # EVAL_HR_SIZE²); a stamp truncated below the target at a field edge
            # is dropped rather than carried at an off-size.
            if not enforce_object_sizes(obj_dir, log=_emit):
                shutil.rmtree(obj_dir, ignore_errors=True)
                rec["error"] = (f"stamp below {EVAL_LR_SIZE}×{EVAL_LR_SIZE} LR / "
                                f"{EVAL_HR_SIZE}×{EVAL_HR_SIZE} SR/HR")
                _emit(f"  ! {sub} dropped: {rec['error']}")
                n_skip += 1
                rows.append(rec)
                continue

            lr_sum, sr_sum = float(np.sum(lr_vis_st)), float(np.sum(sr_vis_st))
            lr_up = np.repeat(np.repeat(lr_vis_st, 2, 0), 2, 1)  # nearest 2× → HR grid
            rec.update({
                "ok": True,
                "lr_total_e": lr_sum, "sr_total_e": sr_sum,
                "flux_ratio_sr_over_lr": (sr_sum / lr_sum) if lr_sum else "",
                "psnr_lr_hr": _psnr(lr_up, hr_vis_st),
                "psnr_sr_hr": _psnr(sr_vis_st, hr_vis_st),
            })
            n_ok += 1
        except Exception as e:  # noqa: BLE001 — one bad field must not kill the run
            rec["error"] = f"{type(e).__name__}: {e}"
            _emit(f"  ! {sub} skipped: {rec['error']}")
            n_skip += 1
        rows.append(rec)

    groups: dict[str, int] = {}
    for r in rows:
        if r.get("ok"):
            groups[r["grade"]] = groups.get(r["grade"], 0) + 1
    return {"rows": rows, "n_ok": n_ok, "n_skip": n_skip, "groups": groups}
