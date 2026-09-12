"""Native exposure acquisition and same-band difference diagnostics."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import requests
from astropy.io import fits
from astropy.wcs import WCS
from scipy import sparse
from scipy.ndimage import map_coordinates

from euclid_polish.config import Config
from euclid_polish.photometry import adu_per_s_to_electrons_factor

from . import BANDS, SCHEMA_VERSION
from .archive import Archive, RemoteFits, assert_aligned, digest, read_json, save_json, utc_now
from .numerics import (
    PropagatedNoise,
    bilinear_operator,
    convolution_operator,
    difference_measurement,
    median_prediction,
)

VIS_DEFINITION = "https://euclid.esac.esa.int/dr/q1/dpdd/visdpd/dpcards/vis_calibratedquadframe.html"
NIR_DEFINITION = "https://euclid.esac.esa.int/dr/q1/dpdd/nirdpd/dpcards/nir_calibratedframe.html"


def cloud_url(row):
    cloud = json.loads(row["cloud_access"])["aws"]
    if cloud["bucket_name"] != "nasa-irsa-euclid-q1" or not cloud["key"].startswith("q1/"):
        raise ValueError("Exposure cloud metadata does not identify the Q1 archive")
    return "https://nasa-irsa-euclid-q1.s3.us-east-1.amazonaws.com/" + cloud["key"]


def native_conversion(primary, header, band):
    unit = str(header.get("BUNIT", primary.get("BUNIT", ""))).strip().upper()
    time = float(header.get("EXPTIME", primary.get("EXPTIME", np.nan)))
    if not np.isfinite(time) or time <= 0:
        raise ValueError("Exposure lacks a positive metadata exposure time")
    config = Config.get_band(band if band == "VIS" else band + "_E")
    if band == "VIS":
        if unit != "ADU":
            raise ValueError("VIS native science/RMS must be in ADU")
        zp = float(header.get("MAGZEROP", primary.get("MAGZEROP", np.nan)))
        factor = adu_per_s_to_electrons_factor(zp, config) / time
        definition = "MAGZEROP is an ADU/s zero point; divide by header EXPTIME"
    else:
        if unit not in ("ELECTRON", "ELECTRONS"):
            raise ValueError("NIR native science/RMS must be in electrons")
        zp = float(header.get("ZPAB", np.nan))
        factor = adu_per_s_to_electrons_factor(zp, config)
        definition = (
            "ZPAB calibrates integrated electrons and already includes exposure time and relative calibration"
        )
    if not np.isfinite(zp) or not np.isfinite(factor) or factor <= 0:
        raise ValueError("Exposure lacks a finite calibrated photometric zero point")
    return {
        "native_unit": unit,
        "exptime_s": time,
        "zeropoint": zp,
        "science_factor": factor,
        "rms_factor": factor,
        "variance_factor": factor * factor,
        "definition": definition,
        "unit": "reference-stack-equivalent electrons per native pixel",
        "source": VIS_DEFINITION if band == "VIS" else NIR_DEFINITION,
        "EXPDUR1": header.get("EXPDUR1"),
        "EXPDUR2": header.get("EXPDUR2"),
        "FLXSCALE": header.get("FLXSCALE"),
        "PHRELDT": header.get("PHRELDT"),
        "PHRELOB": primary.get("PHRELOB"),
        "PHRELEX": primary.get("PHRELEX"),
    }


def next_image_header(remote, header, offset):
    nbytes = header["NAXIS1"] * header["NAXIS2"] * abs(header["BITPIX"]) // 8
    return remote.header_at(offset + math.ceil(nbytes / 2880) * 2880)


def associated_extension(remote, name, detector_index, triplets):
    primary, _, first = remote.header_at(0)
    h, off, end = remote.header_at(first)
    if triplets:
        for _ in range(2):
            _, _, end = remote.header_at(end)
    if detector_index is not None:
        target, data_offset, _ = remote.header_at(first + detector_index * (end - first))
        if target.get("EXTNAME") in (name, name.removesuffix(".SCI")):
            return primary, target, data_offset
    for _, h, off in remote.headers():
        if h.get("EXTNAME") in (name, name.removesuffix(".SCI")):
            return primary, h, off
    raise ValueError(f"Associated product has no matching extension {name}")


def read_psf(archive, row, detector, native_header, primary, x0, y0, size, band):
    path, provenance = archive.cached(row["access_url"], suffix=".fits")
    with fits.open(path, memmap=False) as hdus:
        name = detector if band == "VIS" else detector + ".PSF"
        hdu = hdus[name]
        psf = np.asarray(hdu.data, float)
        header = {**dict(hdus[0].header), **dict(hdu.header)}
        if band == "VIS":
            if psf.shape != (189, 189) or float(header.get("OVERSAMP", 0)) != 1:
                raise ValueError("Unsupported VIS PSF grid layout")
            # Q1's documented PSF_SIZE=21 and PSFEx's 9x9 snapshot grid.
            # Grid coordinates are not present in the released image header.
            # Keep this limitation explicit; no certification at the 5% level.
            gx = int(np.clip((x0 + size / 2) / native_header["NAXIS1"] * 9, 0, 8))
            gy = int(np.clip((y0 + size / 2) / native_header["NAXIS2"] * 9, 0, 8))
            grid = psf.reshape(9, 21, 9, 21).transpose(0, 2, 1, 3)
            selected = grid[gy, gx]
            norms = grid.sum(axis=(-1, -2), keepdims=True)
            if np.any(norms <= 0):
                raise ValueError("Invalid supplied PSF snapshot normalization")
            variations = np.sqrt(np.mean((grid / norms - selected / selected.sum()) ** 2, axis=(-1, -2)))
            provenance.update(
                {
                    "grid_index_yx": [gy, gx],
                    "stamp_size": 21,
                    "position_mapping": "inferred equal-width 9x9 snapshot cells; coordinates absent from "
                    "released header",
                    "spatial_model_limit": "VIS local PSF grid coordinates unverified; difference results are"
                    " provisional",
                    "grid_max_rms_shape_variation": float(variations.max()),
                    "grid_layout_sources": [
                        "https://arxiv.org/html/2503.15303v2#A4",
                        "https://psfex.readthedocs.io/en/latest/GettingStarted.html",
                    ],
                }
            )
            psf = selected
            sampling = 1.0
        else:
            sampling = float(header.get("OVERSAMP", 0))
            if sampling <= 0 or header.get("POLNAXIS") != 0:
                raise ValueError("NIR PSF sampling missing or unsupported spatial context")
            if str(header.get("FILTER")) != band or int(header["OBS_ID"]) != int(primary["OBS_ID"]):
                raise ValueError("NIR PSF exposure identity mismatch")
            provenance["spatial_model_limit"] = (
                "Supplied constant detector PSF; within-detector/time variation unmodeled"
            )
        if not np.all(np.isfinite(psf)) or psf.sum() <= 0:
            raise ValueError("Invalid supplied PSF values")
        provenance["negative_flux_fraction"] = float(-psf[psf < 0].sum() / psf.sum())
        provenance["oversampling"] = sampling
        provenance["extension"] = name
        provenance["archive_metadata"] = row
        return psf / psf.sum(), provenance


def acquire_one_exposure(archive, patch, band, row, background, psf_row, size=128):
    remote = RemoteFits(archive, cloud_url(row))
    primary, index, header, offset, x0, y0, _ = remote.detector(patch["ra"], patch["dec"], size)
    if band == "VIS" and primary.get("DATASETR") != "Q1_R1":
        raise ValueError("VIS exposure release mismatch")
    if str(primary.get("OBS_ID")) != str(row["obs_id"]).split("_")[0]:
        raise ValueError("Native exposure observation identity mismatch")
    if band != "VIS" and primary.get("FILTER") != band:
        raise ValueError("Native exposure band identity mismatch")
    conversion = native_conversion(primary, header, band)
    detector = header["EXTNAME"].removesuffix(".SCI")
    science, shifted = remote.rectangle(header, offset, x0, y0, size)
    rms_header, rms_offset, _ = next_image_header(remote, header, offset)
    flag_header, flag_offset, _ = next_image_header(remote, rms_header, rms_offset)
    if rms_header.get("EXTNAME") != detector + ".RMS" or flag_header.get("EXTNAME") != detector + (
        ".FLG" if band == "VIS" else ".DQ"
    ):
        raise ValueError("Exposure RMS/flag detector identities do not match science")
    rms, rms_shifted = remote.rectangle(rms_header, rms_offset, x0, y0, size)
    flags, flags_shifted = remote.rectangle(flag_header, flag_offset, x0, y0, size)
    assert_aligned([shifted, rms_shifted, flags_shifted], [science.shape, rms.shape, flags.shape])
    bg_remote = RemoteFits(archive, cloud_url(background))
    _, bg_header, bg_offset = associated_extension(bg_remote, header["EXTNAME"], index, band != "VIS")
    bg, bg_shifted = bg_remote.rectangle(bg_header, bg_offset, x0, y0, size)
    background_alignment = {"method": "same native detector pixel indices", "wcs_agrees": True}
    try:
        assert_aligned([shifted, bg_shifted], [science.shape, bg.shape])
    except ValueError:
        if band == "VIS" or bg_header.get("DET_ID") != header.get("DET_ID"):
            raise
        # NIR backgrounds precede final astrometric fitting. Both products are
        # explicitly native, unresampled detector arrays (NIR DPDD). Applying
        # the background's old sky WCS would move values to different pixels.
        if (bg_header["NAXIS1"], bg_header["NAXIS2"]) != (header["NAXIS1"], header["NAXIS2"]):
            raise ValueError("Native NIR background detector dimensions differ") from None
        sky = WCS(header).celestial.pixel_to_world(x0 + size / 2, y0 + size / 2)
        bx, by = WCS(bg_header).celestial.world_to_pixel(sky)
        background_alignment.update(
            {
                "wcs_agrees": False,
                "header_offset_pixels": [float(bx - x0 - size / 2), float(by - y0 - size / 2)],
                "evidence": NIR_DEFINITION,
                "reason": "Associated background and SCI are native arrays on the same "
                "DET_ID; background header retains earlier astrometry",
                "limit": "Auxiliary sky WCS differs; native detector indexing is used, "
                "never sky interpolation",
            }
        )
    if bg_header.get("BUNIT", header.get("BUNIT")) != header.get("BUNIT"):
        raise ValueError("Background and native science units differ")
    psf, psf_provenance = read_psf(archive, psf_row, detector, header, primary, x0, y0, size, band)
    key = hashlib.sha256((row["access_url"] + patch["patch_id"]).encode()).hexdigest()[:24]
    path = archive.root / "exposures" / (key + ".npz")
    path.parent.mkdir(exist_ok=True)
    np.savez_compressed(
        path,
        science=science,
        rms=rms,
        flags=flags.astype(np.int64),
        background=bg,
        psf=psf,
        wcs_header=np.array(shifted.tostring(sep="\n")),
    )
    return {
        "status": "acquired",
        "archive_metadata": row,
        "input_exposure_ids": [row["access_url"]],
        "arrays": str(path.relative_to(archive.root)),
        "arrays_sha256": digest(path),
        "conversion": conversion,
        "primary_header": primary.tostring(sep="\n"),
        "science_header": header.tostring(sep="\n"),
        "release_evidence": "Q1 public cloud metadata"
        if band != "VIS"
        else "DATASETR header and Q1 cloud metadata",
        "detector": detector,
        "native_origin_xy": [x0, y0],
        "shape": [size, size],
        "science_retrieval": remote.records,
        "background_retrieval": bg_remote.records,
        "background_archive_metadata": background,
        "background_alignment": background_alignment,
        "psf": psf_provenance,
        "invalid_policy": "INVALID bit 0; all transformation footprints touching invalid pixels are excluded",
        "variance_kind": "total supplied RMS squared; background estimator covariance is not supplied",
        "background_decomposition": "unavailable: background model is not a background-only noise map",
    }


def acquire_exposures(root, pilot=False):
    root = Path(root)
    manifest = read_json(root / "manifest.json")
    archive = Archive(root)
    path = root / "exposures.json"
    records = (
        read_json(path)
        if path.exists()
        else {"schema_version": SCHEMA_VERSION, "pointings": {}, "pilot_verified": False}
    )
    if pilot:
        selection = [{"patch_id": manifest["patches"][0]["patch_id"], "stratum": "retrieval pilot"}]
    else:
        if not records["pilot_verified"]:
            raise ValueError("Verify exposure retrieval with --pilot before expanding")
        selection = manifest["validation_selection"]
    for selected in selection:
        patch = next(p for p in manifest["patches"] if p["patch_id"] == selected["patch_id"])
        pointing = records["pointings"].setdefault(patch["patch_id"], {"selection": selected, "bands": {}})
        rows_vis, query_vis = archive.sia(patch["ra"], patch["dec"], "euclid_DpdVisCalibratedQuadFrame")
        rows_nir, query_nir = archive.sia(patch["ra"], patch["dec"], "euclid_DpdNirCalibratedFrame")
        pointing["queries"] = [query_vis, query_nir]
        associations = {r["observationid"] for r in patch["ancestry"]}
        for band in BANDS:
            rows = [
                r
                for r in (rows_vis if band == "VIS" else rows_nir)
                if r["energy_bandpassname"] == band and r["obs_id"] in associations
            ]
            sciences = sorted(
                (r for r in rows if r["dataproduct_subtype"] == "science"), key=lambda r: r["access_url"]
            )
            band_records = pointing["bands"].setdefault(band, {})
            for row in sciences:
                key = row["obs_publisher_did"]
                if band_records.get(key, {}).get("status") == "acquired":
                    existing = band_records[key]
                    if digest(root / existing["arrays"]) != existing["arrays_sha256"]:
                        raise ValueError("Native exposure cache checksum mismatch")
                    continue
                background = [
                    r for r in rows if r["obs_publisher_did"] == key and ("-BKG" in r["access_url"])
                ]
                psfs = [
                    r
                    for r in rows
                    if ("GRD-PSF" in r["access_url"] and r["obs_id"] == row["obs_id"])
                    or ("PSF-I_" in r["access_url"] and r["obs_publisher_did"] == key)
                ]
                try:
                    if len(background) != 1 or len(psfs) != 1:
                        raise ValueError("Associated background or PSF missing/ambiguous in archive metadata")
                    band_records[key] = acquire_one_exposure(
                        archive, patch, band, row, background[0], psfs[0]
                    )
                except (ValueError, OSError, KeyError, requests.RequestException) as exc:
                    band_records[key] = {"status": "unavailable", "reason": str(exc), "archive_metadata": row}
                print(
                    f"Exposure {patch['patch_id']} {band} {key.split('?')[-1]}: {band_records[key]['status']}"
                    + (" " + band_records[key].get("reason", "")),
                    flush=True,
                )
                save_json(path, records)
            if not sciences:
                band_records["missing"] = {
                    "status": "unavailable",
                    "reason": "No calibrated frames with matching tile observation ancestry",
                }
            save_json(path, records)
    if pilot:
        records["pilot_verified"] = all(
            any(e.get("status") == "acquired" for e in band.values()) for band in pointing["bands"].values()
        )
    save_json(path, records)
    return records


def target_wcs(ra, dec, size, pixel_scale):
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [(size + 1) / 2, (size + 1) / 2]
    wcs.wcs.crval = [ra, dec]
    wcs.wcs.cdelt = [-pixel_scale / 3600, pixel_scale / 3600]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return wcs


def load_transformed(root, exposure, output_wcs, shape):
    path = Path(root) / exposure["arrays"]
    if digest(path) != exposure["arrays_sha256"]:
        raise ValueError("Exposure array checksum mismatch")
    with np.load(path) as arrays:
        science, rms, flags, bg = (arrays[k] for k in ("science", "rms", "flags", "background"))
        native_wcs = WCS(fits.Header.fromstring(str(arrays["wcs_header"]), sep="\n")).celestial
        yy, xx = np.indices(shape)
        x, y = native_wcs.world_to_pixel(output_wcs.pixel_to_world(xx, yy))
        valid = np.isfinite(science + bg + rms) & (rms > 0) & ((flags & 1) == 0)
        # Local Jacobian of the ACTUAL WCS transformation, including distortion.
        dxdy, dxdx = np.gradient(x)
        dydy, dydx = np.gradient(y)
        area = np.abs(dxdx * dydy - dxdy * dydx)
        matrix, covered = bilinear_operator(science.shape, x, y, valid)
        matrix = sparse.diags(area.ravel() * exposure["conversion"]["science_factor"]) @ matrix
        values = np.asarray(matrix @ np.where(valid, science - bg, 0).ravel()).reshape(shape)
        noise = PropagatedNoise(matrix.tocsr(), np.where(valid, rms * rms, 0), shape)
        psf = arrays["psf"].copy()
        sampling = exposure["psf"]["oversampling"]
        # Resample the supplied effective PSF with the local Jacobian. PSFEx
        # models already include detector response; do not integrate it twice.
        cy, cx = np.array(shape) // 2
        jacobian = np.array([[dxdx[cy, cx], dxdy[cy, cx]], [dydx[cy, cx], dydy[cy, cx]]])
        radius = (
            int(np.ceil(max(psf.shape) / sampling / np.linalg.svd(jacobian, compute_uv=False).min() / 2)) + 1
        )
        if radius >= min(cx, cy):
            raise ValueError("Output patch is too small for the supplied PSF support")
        ny, nx = np.indices(science.shape, dtype=float)
        px = (nx - x[cy, cx]) * sampling + (psf.shape[1] - 1) / 2
        py = (ny - y[cy, cx]) * sampling + (psf.shape[0] - 1) / 2
        native_psf = map_coordinates(psf, [py, px], order=1, mode="constant", cval=0, prefilter=False)
        geometric, _ = bilinear_operator(science.shape, x, y)
        response = (geometric @ native_psf.ravel()).reshape(shape) * area
        kernel = response[cy - radius : cy + radius + 1, cx - radius : cx + radius + 1].copy()
        if kernel.sum() <= 0:
            raise ValueError("PSF transformation has no positive normalization")
        kernel /= kernel.sum()
        return (
            values,
            noise,
            covered,
            kernel,
            {
                "pixel_area_factor_range": [float(area.min()), float(area.max())],
                "psf_resampling": "Supplied effective PSF sampled on native pixels, then propagated "
                "with actual bilinear weights and interpolation phase at patch "
                "center",
                "native_covariance": "Not supplied; diagonal native RMS model. Induced covariance "
                "retained exactly.",
            },
        )


def measure_exposures(root, draws=512, size=80):
    if not 48 <= size <= 96 or draws < 32:
        raise ValueError("Use 48–96 output pixels to bound covariance memory, and at least 32 MC draws")
    root = Path(root)
    manifest, exposures, summary = (
        read_json(root / name) for name in ("manifest.json", "exposures.json", "summary.json")
    )
    measurements, predictions = [], []
    code_hashes = {name: digest(Path(__file__).parent / name) for name in ("exposures.py", "numerics.py")}
    previous = {
        r.get("measurement_fingerprint"): r
        for r in summary.get("exposure_measurements", [])
        if r.get("status") == "measured"
    }
    selected_ids = {r["patch_id"] for r in manifest.get("validation_selection", [])}
    for patch_id, pointing in exposures["pointings"].items():
        if selected_ids and patch_id not in selected_ids:
            continue
        patch = next(p for p in manifest["patches"] if p["patch_id"] == patch_id)
        for band in BANDS:
            available = [e for e in pointing["bands"].get(band, {}).values() if e.get("status") == "acquired"]
            available.sort(key=lambda e: (-e["conversion"]["exptime_s"], e["input_exposure_ids"][0]))
            base = {
                "patch_id": patch_id,
                "sample_id": patch["sample_id"],
                "field": patch["field"],
                "band": band,
                "stratum": pointing["selection"]["stratum"],
                "native_exposure_count": len(available),
            }
            if len(available) < 2:
                measurements.append(
                    {
                        **base,
                        "status": "unavailable",
                        "reason": "Fewer than two fully acquired independent exposures",
                    }
                )
                continue
            scale = 0.1 if band == "VIS" else 0.3
            wcs = target_wcs(patch["ra"], patch["dec"], size, scale)
            # Disjoint adjacent pairs after sorting by time: no pair reuses a frame.
            for pair_index in range(len(available) // 2):
                ea, eb = available[2 * pair_index : 2 * pair_index + 2]
                record = {
                    **base,
                    "pair_index": pair_index,
                    "input_exposure_ids": ea["input_exposure_ids"] + eb["input_exposure_ids"],
                    "exptime_s": [ea["conversion"]["exptime_s"], eb["conversion"]["exptime_s"]],
                    "pixel_scale_arcsec": scale,
                    "pairing": "Disjoint pairs, descending exposure time then product URL",
                }
                fingerprint = hashlib.sha256(
                    json.dumps(
                        {
                            "inputs": [ea["arrays_sha256"], eb["arrays_sha256"]],
                            "size": size,
                            "conversions": [ea["conversion"], eb["conversion"]],
                            "code": code_hashes,
                        },
                        sort_keys=True,
                    ).encode()
                ).hexdigest()
                record["measurement_fingerprint"] = fingerprint
                if fingerprint in previous:
                    cached = previous[fingerprint]
                    files = [(cached["arrays"], cached["arrays_sha256"])] + list(
                        cached.get("operators", {}).items()
                    )
                    if all(
                        (root / path).exists() and digest(root / path) == checksum for path, checksum in files
                    ):
                        measurements.append(cached)
                        continue
                try:
                    a, na, va, pa, ma = load_transformed(root, ea, wcs, (size, size))
                    b, nb, vb, pb, mb = load_transformed(root, eb, wcs, (size, size))
                    ca, valid_a = convolution_operator((size, size), pb, va)
                    cb, valid_b = convolution_operator((size, size), pa, vb)
                    na = PropagatedNoise((ca @ na.operator).tocsr(), na.native_variance, (size, size))
                    nb = PropagatedNoise((cb @ nb.operator).tocsr(), nb.native_variance, (size, size))
                    aa, bb = (ca @ a.ravel()).reshape(size, size), (cb @ b.ravel()).reshape(size, size)
                    stats, arrays = difference_measurement(
                        aa,
                        bb,
                        na,
                        nb,
                        valid_a & valid_b,
                        ea["input_exposure_ids"],
                        eb["input_exposure_ids"],
                        block=16,
                    )
                    arrays.update(
                        {
                            "aligned_a": a,
                            "aligned_b": b,
                            "matched_a": aa,
                            "matched_b": bb,
                            "psf_a": pa,
                            "psf_b": pb,
                            "native_variance_a": na.native_variance,
                            "native_variance_b": nb.native_variance,
                        }
                    )
                    stem = f"{patch_id}_{band}_pair{pair_index}"
                    path = root / "arrays" / (stem + ".npz")
                    np.savez_compressed(path, **arrays)
                    sparse.save_npz(root / "arrays" / (stem + "_operator_a.npz"), na.operator)
                    sparse.save_npz(root / "arrays" / (stem + "_operator_b.npz"), nb.operator)
                    record.update(
                        {
                            "status": "measured" if stats["n_valid"] >= 64 else "unavailable",
                            "statistics": stats,
                            "arrays": str(path.relative_to(root)),
                            "arrays_sha256": digest(path),
                            "operators": {
                                str(p.relative_to(root)): digest(p)
                                for p in (
                                    root / "arrays" / (stem + "_operator_a.npz"),
                                    root / "arrays" / (stem + "_operator_b.npz"),
                                )
                            },
                            "transformations": [ma, mb],
                            "psf_limits": [
                                ea["psf"]["spatial_model_limit"],
                                eb["psf"]["spatial_model_limit"],
                            ],
                            "source_masking": (
                                "none; only documented invalid pixels and transformation borders excluded"
                            ),
                        }
                    )
                    if stats["n_valid"] < 64:
                        record["reason"] = stats["reason"]
                    if band == "VIS":
                        record["certification"] = "provisional: VIS PSF grid coordinates not verified"
                except (ValueError, KeyError, OSError) as exc:
                    record.update({"status": "unavailable", "reason": str(exc)})
                measurements.append(record)
                print(f"Difference {patch_id} {band} pair {pair_index}: {record['status']}", flush=True)
                summary["exposure_measurements"] = measurements
                save_json(root / "summary.json", summary)
            # An exact MER replica requires LayersStorage: association with a
            # tile alone does not establish which exposures survived MER cuts.
            mer_wcs = target_wcs(patch["ra"], patch["dec"], size, 0.1)
            try:
                transformed = [load_transformed(root, e, mer_wcs, (size, size)) for e in available]
                metadata, variance = median_prediction(
                    [t[1] for t in transformed],
                    [t[0] for t in transformed],
                    [t[2] for t in transformed],
                    draws=draws,
                )
                path = root / "arrays" / f"{patch_id}_{band}_median_prediction.npz"
                np.savez_compressed(
                    path,
                    predicted_variance=variance,
                    coverage_count=np.sum([t[2] for t in transformed], axis=0),
                )
                predictions.append(
                    {
                        **base,
                        **metadata,
                        "status": "conditional prediction",
                        "arrays": str(path.relative_to(root)),
                        "arrays_sha256": digest(path),
                        "input_exposure_ids": [e["input_exposure_ids"][0] for e in available],
                        "exptime_s": [e["conversion"]["exptime_s"] for e in available],
                        "limit": "Exact MER contributing layers/rejection metadata unavailable; "
                        "median of retrieved candidates, not a verified MER reproduction",
                        "signal_model": "Observed background-subtracted exposures held fixed in Monte "
                        "Carlo; noisy signal template limitation",
                    }
                )
            except ValueError as exc:
                predictions.append({**base, "status": "unavailable", "reason": str(exc)})
            summary["mer_predictions"] = predictions
            save_json(root / "summary.json", summary)
    summary["exposure_measured_at"] = utc_now()
    summary["exposure_processing_parameters"] = {
        "output_size": size,
        "draws": draws,
        "code_sha256": code_hashes,
        "block_size_pixels": 16,
    }
    summary["exposure_measurements"], summary["mer_predictions"] = measurements, predictions
    save_json(root / "summary.json", summary)
    return summary
