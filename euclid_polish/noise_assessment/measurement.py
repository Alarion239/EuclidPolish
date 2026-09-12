"""Archive measurements in explicit native and reference photometric units."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from euclid_polish.config import Config
from euclid_polish.photometry import adu_per_s_to_electrons_factor
from euclid_polish.web.helpers.vis_noise_calibration import (
    _robust_sigma,
    _source_masked_residual,
)

from . import BANDS, SCHEMA_VERSION
from .archive import DPDD, assert_aligned, digest, image_product, read_json, save_json, utc_now
from .numerics import clustered_interval


def mer_conversion(header, band):
    """Resolve Q1 MER units by header plus product definition, never pixel values."""
    if header.get("DATASETR") != "Q1_R1" or header.get("FILTER", "").replace("NIR_", "") != band:
        raise ValueError("Product release/filter is not supported by the unit definition")
    unit = str(header.get("BUNIT", "")).strip().lower().replace(" ", "")
    allowed = {"adu/s", "adu/sec", "adus-1"} if band == "VIS" else {"electron", "electrons", "e-"}
    if unit and unit not in allowed:
        raise ValueError(f"Unexpected {band} MER unit: {unit!r}")
    zp = float(header.get("MAGZERO", np.nan))
    if not np.isfinite(zp):
        raise ValueError("Missing finite MAGZERO; photometric scale cannot be inferred")
    config = Config.get_band(band if band == "VIS" else band + "_E")
    factor = adu_per_s_to_electrons_factor(zp, config)
    return {
        "native_unit": unit or ("ADU/s" if band == "VIS" else "electrons"),
        "native_unit_evidence": "BUNIT header" if unit else DPDD + " (Q1 MER band-specific definition)",
        "magzero": zp,
        "comparison_unit": "reference-stack-equivalent electrons/pixel",
        "reference_zeropoint": config.sim_zeropoint_e,
        "science_factor": factor,
        "rms_factor": abs(factor),
        "variance_factor": factor * factor,
        "conversion": "canonical MAGZERO to configured reference-stack photometric scale",
        "meaning": "Photometric comparison scale; not a claim about the mosaic's collected electron count",
    }


def crop_to_patch(patch, arrays, header):
    size = patch["size"]
    h, w = arrays[0].shape
    x0, y0 = (w - size) // 2, (h - size) // 2
    if patch.get("reference_wcs_header"):
        reference = fits.Header.fromstring(patch["reference_wcs_header"], sep="\n")
        sky = WCS(reference).celestial.pixel_to_world(0, 0)
        xx, yy = WCS(header).celestial.world_to_pixel(sky)
        if np.max(np.abs([xx - round(float(xx)), yy - round(float(yy))])) > 0.001:
            raise ValueError("Saved example and downloaded science do not share a pixel grid")
        x0, y0 = round(float(xx)), round(float(yy))
    elif patch.get("legacy_example"):
        example = patch["legacy_example"]
        source = Path(example["source_file"])
        with fits.open(source, memmap=True) as hdus:
            original = hdus["VIS"]
            oh, ow = original.shape
            ox = (ow - ow // 256 * 256) // 2 + 256 * example["patch_grid_column_zero_based"]
            oy = (oh - oh // 256 * 256) // 2 + 256 * example["patch_grid_row_zero_based"]
            sky = WCS(original.header).celestial.pixel_to_world(ox, oy)
            xx, yy = WCS(header).celestial.world_to_pixel(sky)
        if np.max(np.abs([xx - round(float(xx)), yy - round(float(yy))])) > 0.001:
            raise ValueError("Saved example and downloaded science do not share a pixel grid")
        x0, y0 = round(float(xx)), round(float(yy))
    if min(x0, y0) < 0 or x0 + size > w or y0 + size > h:
        raise ValueError("Archive cutout does not cover the exact requested patch")
    cropped = [a[y0 : y0 + size, x0 : x0 + size] for a in arrays]
    output_header = header.copy()
    output_header["CRPIX1"] -= x0
    output_header["CRPIX2"] -= y0
    output_header["NAXIS1"], output_header["NAXIS2"] = size, size
    return cropped, output_header


def legacy_measurement(science):
    # The existing implementation is the reference, not a reimplementation.
    residual, mask = _source_masked_residual(science)
    return float(_robust_sigma(residual[~mask])), residual, mask


def measure_mer(root):
    root = Path(root)
    manifest = read_json(root / "manifest.json")
    results = []
    array_dir = root / "arrays"
    array_dir.mkdir(exist_ok=True)
    for patch in manifest["patches"]:
        for band in BANDS:
            result = {
                "patch_id": patch["patch_id"],
                "sample_id": patch["sample_id"],
                "field": patch["field"],
                "kind": patch["kind"],
                "band": band,
                "ra": patch["ra"],
                "dec": patch["dec"],
                "ancestry": sorted({r["observationid"] for r in patch.get("ancestry", [])}),
                "official_uncertainty_kind": "total; includes source photon noise",
                "background_source_decomposition": "unavailable: separate variance components not supplied",
                "coverage": "exact per-pixel contributing exposure count unavailable in MER products",
            }
            product = patch.get("mer", {}).get(band, {})
            if product.get("status") != "verified":
                result["status"] = product.get("status", "unavailable: not acquired")
                results.append(result)
                continue
            try:
                raw, headers = zip(
                    *(image_product(root, product[role]) for role in ("science", "rms", "flags")), strict=True
                )
                assert_aligned(headers, [a.shape for a in raw])
                conversion = mer_conversion(headers[0], band)
                # RMS is in the associated science's native units, even when it omits BUNIT.
                if headers[1].get("BUNIT") and headers[1].get("BUNIT") != headers[0].get("BUNIT"):
                    raise ValueError("Science and RMS units conflict")
                (science, rms, flags), header = crop_to_patch(patch, raw, headers[0])
                flags = flags.astype(np.int64) & 0xFFFFFFFF
                valid = np.isfinite(science) & np.isfinite(rms) & (rms > 0) & ((flags & 1) == 0)
                if valid.sum() < 64:
                    raise ValueError("Insufficient documented valid coverage")
                factor = conversion["science_factor"]
                science_e, rms_e = science.astype(float) * factor, rms.astype(float) * factor
                sigma, residual, mask = legacy_measurement(science_e)
                rms_on_legacy = rms_e[valid & ~mask]
                result.update(
                    {
                        "status": "measured",
                        "conversion": conversion,
                        "legacy_mad": sigma,
                        "legacy_mask_fraction": float(mask.mean()),
                        "official_rms_median_native": float(np.median(rms[valid])),
                        "official_rms_median": float(np.median(rms_e[valid])),
                        "official_rms_rms": float(np.sqrt(np.mean(rms_e[valid] ** 2))),
                        "official_rms_median_on_legacy_pixels": float(np.median(rms_on_legacy))
                        if rms_on_legacy.size
                        else None,
                        "mad_to_total_rms_ratio": float(sigma / np.median(rms_e[valid])),
                        "ratio_caveat": "Different estimands: masked residual scatter versus total pixel "
                        "uncertainty",
                        "science_median_native": float(np.median(science[valid])),
                        "science_p99_native": float(np.quantile(science[valid], 0.99)),
                        "valid_fraction": float(valid.mean()),
                        "flagged_fraction": float((flags != 0).mean()),
                        "invalid_bit_policy": (
                            "Exclude documented INVALID bit 0; retain ordinary source flags"
                        ),
                        "flag_value_counts": {
                            str(k): int(v) for k, v in zip(*np.unique(flags, return_counts=True), strict=True)
                        },
                        "mer_combination": header.get("COMBINET"),
                        "mer_resampling": header.get("RESAMPT1"),
                    }
                )
                if band == "VIS" and patch.get("legacy_example"):
                    saved = patch["legacy_example"]["saved_sigma_e_per_pixel"]
                    reference_path = root / patch["example_array"]
                    if (
                        patch.get("example_array_sha256")
                        and digest(reference_path) != patch["example_array_sha256"]
                    ):
                        raise ValueError("Saved example reference checksum mismatch")
                    with np.load(reference_path) as old:
                        identical = np.allclose(
                            science_e, old["original_e"], rtol=1e-8, atol=1e-8, equal_nan=True
                        )
                    result["saved_example"] = {
                        "saved_mad": saved,
                        "recomputed_mad": sigma,
                        "relative_error": abs(sigma / saved - 1),
                        "pixels_match_saved": bool(identical),
                        "passed": bool(identical and np.isclose(sigma, saved, rtol=1e-8, atol=1e-8)),
                    }
                    if not result["saved_example"]["passed"]:
                        result["example_discrepancy"] = "Saved values not reproduced; discrepancy retained"
                array_path = array_dir / f"{patch['patch_id']}_{band}_mer.npz"
                np.savez_compressed(
                    array_path,
                    science_native=science,
                    rms_native=rms,
                    science=science_e,
                    official_rms=rms_e,
                    flags=flags,
                    valid=valid,
                    legacy_residual=residual,
                    legacy_mask=mask,
                    wcs_header=np.array(header.tostring(sep="\n")),
                )
                result["arrays"] = str(array_path.relative_to(root))
                result["arrays_sha256"] = digest(array_path)
            except (ValueError, KeyError, OSError) as exc:
                result["status"] = "unavailable: " + str(exc)
            results.append(result)
    summaries = {}
    for band in BANDS:
        rows = [
            r for r in results if r["band"] == band and r["kind"] == "central" and r["status"] == "measured"
        ]
        summaries[band] = {"measured_pointings": len(rows)}
        for field in ("EDF-N", "EDF-S", "EDF-F", "all"):
            selected = [r for r in rows if field == "all" or r["field"] == field]
            summaries[band][field] = {
                key: clustered_interval([r[key] for r in selected], [r["ancestry"] for r in selected])
                for key in ("legacy_mad", "official_rms_median", "mad_to_total_rms_ratio")
            }
    source = Path(__file__).resolve()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "measured_at": utc_now(),
        "manifest_sha256": digest(root / "manifest.json"),
        "code_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "diagnostic_amplitude_target": 0.05,
        "mer_measurements": results,
        "four_band_summary": summaries,
        "exposure_measurements": [],
        "mer_predictions": [],
        "assessment_complete": False,
    }
    save_json(root / "manifests" / (payload["manifest_sha256"] + ".json"), manifest)
    existing = root / "summary.json"
    if existing.exists():
        previous = read_json(existing)
        for key in (
            "exposure_measurements",
            "mer_predictions",
            "exposure_measured_at",
            "exposure_processing_parameters",
        ):
            if key in previous:
                payload[key] = previous[key]
    save_json(existing, payload)
    return payload


def select_validation(root):
    root = Path(root)
    manifest, summary = read_json(root / "manifest.json"), read_json(root / "summary.json")
    rows = [
        r
        for r in summary["mer_measurements"]
        if r["band"] == "VIS"
        and r["kind"] == "central"
        and r["status"] == "measured"
        and r["sample_id"] != 31
    ]
    selection = []
    for field in ("EDF-N", "EDF-S", "EDF-F"):
        candidates = sorted(
            (r for r in rows if r["field"] == field), key=lambda r: (r["official_rms_median"], r["sample_id"])
        )
        if len(candidates) < 3:
            raise ValueError(f"Need at least three verified distinct {field} pointings before selection")
        for label, row in zip(
            ("low", "median", "high"),
            (candidates[0], candidates[len(candidates) // 2], candidates[-1]),
            strict=True,
        ):
            selection.append(
                {
                    "patch_id": row["patch_id"],
                    "sample_id": row["sample_id"],
                    "field": field,
                    "stratum": label,
                    "selection_vis_rms": row["official_rms_median"],
                }
            )
    stress = next(p for p in manifest["patches"] if p["kind"] == "extreme")
    selection.append(
        {
            "patch_id": stress["patch_id"],
            "sample_id": stress["sample_id"],
            "field": stress["field"],
            "stratum": "bright-star stress case",
        }
    )
    manifest["validation_selection"] = selection
    manifest["validation_selection_rule"] = (
        "Official VIS RMS median in central patch; median uses upper "
        "middle order statistic; sample 31 reserved for separate stress "
        "case"
    )
    save_json(root / "manifest.json", manifest)
    return selection
