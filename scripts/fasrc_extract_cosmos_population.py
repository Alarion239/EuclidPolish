#!/usr/bin/env python3
"""Extract a compact, generation-ready population prior from COSMOS2025.

This script is intended to run on FASRC beside the 10-GB COSMOS2025 v1.1
master FITS.  It deliberately keeps two selections separate:

``population``
    Classified galaxies with a valid photo-z and total HST/F814W model
    magnitude.  This broad sample defines the *latent number counts*; it does
    not require a successful bulge+disk fit and retains blends.

``generator_ready``
    Population rows with clean flags, no blend flag, a viable B+D fit, and
    finite B+D photometry in the four Euclid proxy bands.  This is the subset
    from which morphology/colour conditionals may safely be learned.

The separation is important: applying morphology-fit or Euclid-detection cuts
to the number-count target would manufacture a faint-end turnover.

Outputs
-------
``cosmos2025_population_prior.npz``
    One row per population galaxy with photo-z, physical properties,
    four-band photometry, morphology, and selection flags.
``cosmos2025_number_counts.csv``
    Differential and cumulative F814W counts for the broad, clean, isolated,
    and generator-ready selections.  Counts use the supplied survey area and
    are raw catalog counts, not completeness corrected.
``cosmos2025_population_summary.json``
    Provenance, selection definitions, row counts, and robust percentiles.
``cosmos2025_schema.json``
    Full HDU/column inventory plus the columns consumed by this extraction.
``cosmos2025_population_diagnostics.png``
    Number-count, redshift, mass, size, and morphology diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "euclid_mpl_cache")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import Planck15
from astropy.io import fits

from euclid_polish.config import Config
from euclid_polish.sky.generation.cosmos2025 import (
    circularized_effective_radius_arcsec,
)

DEFAULT_AREA_DEG2 = 0.54
DEFAULT_OUTPUT_DIR = os.path.join(
    Config.DATA_DIR, "population_comparison", "cosmos2025"
)
MAG_BINS = np.arange(18.0, 31.0001, 0.25)
EUCLID_PROXY_COLUMNS = {
    "VIS": "hst-f814w",
    "Y_E": "uvista-y",
    "J_E": "uvista-j",
    "H_E": "uvista-h",
}
HDU_INDEX = {
    "photometry": 1,
    "lephare": 2,
    "cigale": 4,
    "ml_morphology": 5,
    "bulge_disk": 6,
    "galfitm": 7,
}


def _native(values: Any, dtype: np.dtype | type) -> np.ndarray:
    """Copy a FITS column into a native-endian numpy array."""
    # COSMOS uses extreme finite sentinels in a few floating columns. Casting
    # those to float32 intentionally turns them into +/-inf, which our
    # finite-value masks reject; suppress only that expected cast warning.
    with np.errstate(over="ignore", invalid="ignore"):
        return np.asarray(values, dtype=dtype)


def _optional(
    table: Any,
    name: str,
    *,
    dtype: np.dtype | type = np.float32,
    fill: float | int = np.nan,
) -> np.ndarray:
    """Read an optional FITS column or return a filled array."""
    if name in (table.names or ()):
        return _native(table[name], dtype)
    return np.full(len(table), fill, dtype=dtype)


def _valid_mag(values: np.ndarray) -> np.ndarray:
    return np.isfinite(values) & (values > 5.0) & (values < 50.0)


def _valid_range(
    values: np.ndarray, low: float, high: float
) -> np.ndarray:
    return np.isfinite(values) & (values > low) & (values < high)


def _nan_percentiles(
    values: np.ndarray, mask: np.ndarray | None = None
) -> dict[str, float | None]:
    arr = np.asarray(values, dtype=np.float64)
    good = np.isfinite(arr)
    if mask is not None:
        good &= mask
    if not np.any(good):
        return dict.fromkeys(("p05", "p16", "p50", "p84", "p95"))
    pct = np.percentile(arr[good], (5, 16, 50, 84, 95))
    return {
        key: float(value)
        for key, value in zip(
            ("p05", "p16", "p50", "p84", "p95"), pct, strict=True
        )
    }


def _mag_from_components(
    bulge_mag: np.ndarray, disk_mag: np.ndarray
) -> np.ndarray:
    """Total AB magnitude from independent bulge and disk magnitudes."""
    valid = _valid_mag(bulge_mag) & _valid_mag(disk_mag)
    result = np.full(bulge_mag.shape, np.nan, dtype=np.float32)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        flux = (
            np.power(10.0, -0.4 * bulge_mag.astype(np.float64))
            + np.power(10.0, -0.4 * disk_mag.astype(np.float64))
        )
        result[valid] = (-2.5 * np.log10(flux[valid])).astype(np.float32)
    return result


def _bulge_fraction(
    bulge_mag: np.ndarray, disk_mag: np.ndarray
) -> np.ndarray:
    valid = _valid_mag(bulge_mag) & _valid_mag(disk_mag)
    result = np.full(bulge_mag.shape, np.nan, dtype=np.float32)
    with np.errstate(over="ignore", invalid="ignore"):
        fb = np.power(10.0, -0.4 * bulge_mag.astype(np.float64))
        fd = np.power(10.0, -0.4 * disk_mag.astype(np.float64))
        result[valid] = (fb[valid] / (fb[valid] + fd[valid])).astype(np.float32)
    return result


def _physical_re_kpc(
    re_arcsec: np.ndarray, redshift: np.ndarray
) -> np.ndarray:
    """Fast Planck15 angular-size conversion using an interpolated grid."""
    result = np.full(re_arcsec.shape, np.nan, dtype=np.float32)
    good = (
        np.isfinite(re_arcsec)
        & (re_arcsec > 0.0)
        & np.isfinite(redshift)
        & (redshift > 0.0)
        & (redshift < 15.0)
    )
    if not np.any(good):
        return result
    z_grid = np.linspace(1.0e-4, 15.0, 16_384)
    kpc_per_arcsec = (
        Planck15.kpc_proper_per_arcmin(z_grid).value / 60.0
    )
    result[good] = (
        re_arcsec[good]
        * np.interp(redshift[good], z_grid, kpc_per_arcsec)
    ).astype(np.float32)
    return result


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(tmp, path)


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(tmp, path)


def _write_counts(
    path: Path,
    magnitudes: np.ndarray,
    selections: dict[str, np.ndarray],
    *,
    area_arcmin2: float,
    bins: np.ndarray = MAG_BINS,
) -> list[dict[str, float | int]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    centers = 0.5 * (bins[:-1] + bins[1:])
    width = np.diff(bins)
    counts = {
        name: np.histogram(magnitudes[mask], bins=bins)[0]
        for name, mask in selections.items()
    }
    cumulative = {
        name: np.cumsum(value) for name, value in counts.items()
    }
    rows: list[dict[str, float | int]] = []
    for index, center in enumerate(centers):
        row: dict[str, float | int] = {
            "mag_lo": float(bins[index]),
            "mag_hi": float(bins[index + 1]),
            "mag_center": float(center),
        }
        for name in selections:
            count = int(counts[name][index])
            density = count / area_arcmin2 / width[index]
            row[f"{name}_count"] = count
            row[f"{name}_density_per_mag_arcmin2"] = float(density)
            row[f"{name}_poisson_error_per_mag_arcmin2"] = float(
                math.sqrt(count) / area_arcmin2 / width[index]
            )
            row[f"{name}_cumulative_arcmin2"] = float(
                cumulative[name][index] / area_arcmin2
            )
        rows.append(row)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)
    return rows


def _make_plot(
    path: Path,
    arrays: dict[str, np.ndarray],
    selections: dict[str, np.ndarray],
    count_rows: list[dict[str, float | int]],
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.2))
    colors = {
        "population": "#1f77b4",
        "clean": "#008f5d",
        "isolated": "#e07a1f",
        "generator_ready": "#8b45a6",
    }

    ax = axes[0, 0]
    centers = np.asarray([row["mag_center"] for row in count_rows], float)
    for name, color in colors.items():
        density = np.asarray(
            [
                row[f"{name}_density_per_mag_arcmin2"]
                for row in count_rows
            ],
            float,
        )
        ax.plot(centers, density, label=name.replace("_", " "), color=color)
    ax.set_yscale("log")
    ax.set_xlabel("HST F814W total model magnitude [AB]")
    ax.set_ylabel(r"$dN/(dm\,dA)$ [arcmin$^{-2}$ mag$^{-1}$]")
    ax.set_title("Raw COSMOS2025 differential counts")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    pop = selections["population"]
    for lo, hi, color in (
        (20.0, 23.0, "#263b80"),
        (23.0, 25.0, "#008f5d"),
        (25.0, 27.0, "#e07a1f"),
        (27.0, 29.0, "#a13d63"),
    ):
        mask = pop & (arrays["mag_vis"] >= lo) & (arrays["mag_vis"] < hi)
        if np.any(mask):
            ax.hist(
                arrays["z_phot"][mask],
                bins=np.linspace(0.0, 8.0, 65),
                histtype="step",
                density=True,
                linewidth=1.5,
                color=color,
                label=f"{lo:g}–{hi:g}",
            )
    ax.set_xlabel("photometric redshift")
    ax.set_ylabel("normalized density")
    ax.set_title("Redshift by VIS-proxy magnitude")
    ax.legend(fontsize=8)

    ax = axes[0, 2]
    sample = (
        selections["clean"]
        & np.isfinite(arrays["logmass_lephare"])
        & np.isfinite(arrays["z_phot"])
    )
    indices = np.flatnonzero(sample)
    if indices.size > 120_000:
        indices = np.random.default_rng(73032).choice(
            indices, 120_000, replace=False
        )
    hexbin = ax.hexbin(
        arrays["z_phot"][indices],
        arrays["logmass_lephare"][indices],
        gridsize=80,
        bins="log",
        mincnt=1,
        cmap="viridis",
    )
    fig.colorbar(hexbin, ax=ax, label="log10 rows")
    ax.set_xlim(0.0, 8.0)
    ax.set_ylim(5.0, 12.5)
    ax.set_xlabel("photometric redshift")
    ax.set_ylabel(r"LePHARE $\log_{10}(M_\star/M_\odot)$")
    ax.set_title("Joint mass–redshift prior")

    ax = axes[1, 0]
    gen = selections["generator_ready"]
    for lo, hi, color in (
        (20.0, 23.0, "#263b80"),
        (23.0, 25.0, "#008f5d"),
        (25.0, 27.0, "#e07a1f"),
        (27.0, 29.0, "#a13d63"),
    ):
        mask = (
            gen
            & (arrays["mag_vis"] >= lo)
            & (arrays["mag_vis"] < hi)
            & np.isfinite(arrays["re_combined_arcsec"])
        )
        if np.any(mask):
            ax.hist(
                arrays["re_combined_arcsec"][mask],
                bins=np.geomspace(0.015, 3.0, 55),
                histtype="step",
                density=True,
                linewidth=1.5,
                color=color,
                label=f"{lo:g}–{hi:g}",
            )
    ax.set_xscale("log")
    ax.set_xlabel(r"combined circularized $R_e$ [arcsec]")
    ax.set_ylabel("normalized density")
    ax.set_title("Renderer-ready angular sizes")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    good = gen & np.isfinite(arrays["bulge_to_total_vis"])
    ax.hist(
        arrays["bulge_to_total_vis"][good],
        bins=np.linspace(0.0, 1.0, 51),
        color="#8b45a6",
        alpha=0.8,
    )
    ax.set_xlabel("VIS-proxy bulge / total flux")
    ax.set_ylabel("galaxies")
    ax.set_title("B+D light fractions")

    ax = axes[1, 2]
    morph_names = ("sph", "disk", "irr", "bd")
    morph_values = []
    for name in morph_names:
        key = f"morph_{name}_f150w"
        values = arrays[key][selections["clean"]]
        values = values[np.isfinite(values)]
        morph_values.append(float(np.median(values)) if values.size else 0.0)
    ax.bar(
        morph_names,
        morph_values,
        color=("#4c6fc1", "#008f5d", "#e07a1f", "#8b45a6"),
    )
    ax.set_ylim(0.0, max(1.0, max(morph_values, default=1.0) * 1.1))
    ax.set_ylabel("median ML probability")
    ax.set_title("F150W morphology probabilities")

    for axis in axes.flat:
        axis.grid(alpha=0.18)
    fig.suptitle(
        "COSMOS2025 population prior · raw catalog, no completeness correction",
        fontsize=14,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    fig.savefig(tmp, format="png", dpi=150)
    plt.close(fig)
    os.replace(tmp, path)


def extract_catalog(
    catalog_path: str,
    output_dir: str,
    *,
    area_deg2: float = DEFAULT_AREA_DEG2,
    max_bd_chi2: float = 10.0,
) -> dict[str, Any]:
    source = Path(catalog_path).resolve()
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if not source.is_file():
        raise FileNotFoundError(f"COSMOS2025 catalog not found: {source}")
    if area_deg2 <= 0.0:
        raise ValueError("area_deg2 must be positive")

    print(f"[cosmos-prior] opening {source} ({source.stat().st_size / 1e9:.2f} GB)")
    consumed: dict[str, list[str]] = {}
    with fits.open(source, memmap=True, lazy_load_hdus=False) as hdul:
        schema = {
            "source": str(source),
            "hdus": [
                {
                    "index": index,
                    "name": hdu.name,
                    "rows": int(len(hdu.data)) if hdu.data is not None else 0,
                    "columns": list(hdu.columns.names or ())
                    if hasattr(hdu, "columns")
                    else [],
                }
                for index, hdu in enumerate(hdul)
            ],
        }
        photo = hdul[HDU_INDEX["photometry"]].data
        lephare = hdul[HDU_INDEX["lephare"]].data
        cigale = hdul[HDU_INDEX["cigale"]].data
        ml = hdul[HDU_INDEX["ml_morphology"]].data
        bd = hdul[HDU_INDEX["bulge_disk"]].data
        galfitm = hdul[HDU_INDEX["galfitm"]].data
        row_count = len(photo)
        if not all(
            len(table) == row_count
            for table in (lephare, cigale, ml, bd, galfitm)
        ):
            raise ValueError("COSMOS2025 HDUs do not have aligned row counts")

        ids = _native(photo["id"], np.int64)
        ra = _native(photo["ra"], np.float64)
        dec = _native(photo["dec"], np.float64)
        flag_star = _native(photo["flag_star"], np.int16)
        flag_blend = _native(photo["flag_blend"], np.int16)
        warn_flag = _native(photo["warn_flag"], np.int32)
        model_mags = {
            band: _native(photo[f"mag_model_{column}"], np.float32)
            for band, column in EUCLID_PROXY_COLUMNS.items()
        }
        mag_vis = model_mags["VIS"]
        mag_auto_vis = _optional(
            photo, "mag_auto_hst-f814w", dtype=np.float32
        )
        consumed["photometry"] = [
            "id", "ra", "dec", "flag_star", "flag_blend", "warn_flag",
            "mag_auto_hst-f814w",
            *(f"mag_model_{column}" for column in EUCLID_PROXY_COLUMNS.values()),
        ]

        obj_type = _native(lephare["type"], np.int16)
        z_phot = _native(lephare["zfinal"], np.float32)
        z_med = _optional(lephare, "zpdf_med")
        z_l68 = _optional(lephare, "zpdf_l68")
        z_u68 = _optional(lephare, "zpdf_u68")
        logmass_lephare = _optional(lephare, "mass_med")
        logsfr_lephare = _optional(lephare, "sfr_med")
        logssfr_lephare = _optional(lephare, "ssfr_med")
        logage_lephare = _optional(lephare, "age_med")
        ebv_lephare = _optional(lephare, "ebv_minchi2")
        consumed["lephare"] = [
            "type", "zfinal", "zpdf_med", "zpdf_l68", "zpdf_u68",
            "mass_med", "sfr_med", "ssfr_med", "age_med", "ebv_minchi2",
        ]

        cigale_mass = _optional(cigale, "mass")
        with np.errstate(divide="ignore", invalid="ignore"):
            logmass_cigale = np.where(
                cigale_mass > 0.0, np.log10(cigale_mass), np.nan
            ).astype(np.float32)
        sfr_cigale = _optional(cigale, "sfr_inst")
        sfr100_cigale = _optional(cigale, "sfr_100myr")
        metallicity_cigale = _optional(cigale, "metallicity")
        ebv_cigale = _optional(cigale, "ebv_stars")
        cigale_chi2 = _optional(cigale, "chi2_red_best_fit")
        consumed["cigale"] = [
            "mass", "sfr_inst", "sfr_100myr", "metallicity", "ebv_stars",
            "chi2_red_best_fit",
        ]

        disk_re = _native(bd["disk_radius_deg"], np.float32) * 3600.0
        bulge_re = _native(bd["bulge_radius_deg"], np.float32) * 3600.0
        disk_q = _native(bd["disk_axratio"], np.float32)
        bulge_q = _native(bd["bulge_axratio"], np.float32)
        angle_deg = _native(bd["angle_bd"], np.float32)
        bd_chi2 = _native(bd["fmf_b+d_chi2"], np.float32)
        band_mags: dict[str, np.ndarray] = {}
        bulge_mags: dict[str, np.ndarray] = {}
        disk_mags: dict[str, np.ndarray] = {}
        consumed["bulge_disk"] = [
            "disk_radius_deg", "bulge_radius_deg", "disk_axratio",
            "bulge_axratio", "angle_bd", "fmf_b+d_chi2",
        ]
        for band, column in EUCLID_PROXY_COLUMNS.items():
            total_name = f"mag_model_bd_total_{column}"
            bulge_name = f"mag_model_bulge_{column}"
            disk_name = f"mag_model_disk_{column}"
            bulge_mag = _native(bd[bulge_name], np.float32)
            disk_mag = _native(bd[disk_name], np.float32)
            direct_total = _optional(bd, total_name)
            summed_total = _mag_from_components(bulge_mag, disk_mag)
            band_mags[band] = np.where(
                _valid_mag(direct_total), direct_total, summed_total
            ).astype(np.float32)
            bulge_mags[band] = bulge_mag
            disk_mags[band] = disk_mag
            consumed["bulge_disk"].extend(
                (total_name, bulge_name, disk_name)
            )

        morph: dict[str, np.ndarray] = {}
        for band in ("f150w", "f277w", "f444w"):
            for label in ("sph", "disk", "irr", "bd"):
                name = f"{label}_{band}_mean"
                morph[f"morph_{label}_{band}"] = _optional(ml, name)
                consumed.setdefault("ml_morphology", []).append(name)

        galfit_arrays = {}
        for name in (
            "rearc_f150w_sersic", "nsersic_f150w_sersic",
            "qratio_f150w_sersic", "asymmetry_f150w",
            "smoothness_f150w", "concentration_f150w",
            "gini_f150w", "m20_f150w",
        ):
            galfit_arrays[name] = _optional(galfitm, name)
            consumed.setdefault("galfitm", []).append(name)

    valid_z = _valid_range(z_phot, 0.0, 15.0)
    valid_vis = _valid_mag(mag_vis)
    population = (
        (obj_type == 0)
        & (flag_star == 0)
        & valid_z
        & valid_vis
    )
    clean = population & (warn_flag == 0)
    isolated = clean & (flag_blend == 0)
    finite_bd_photometry = np.ones(row_count, dtype=bool)
    for band in EUCLID_PROXY_COLUMNS:
        finite_bd_photometry &= (
            _valid_mag(bulge_mags[band])
            & _valid_mag(disk_mags[band])
            & _valid_mag(band_mags[band])
        )
    finite_geometry = (
        np.isfinite(disk_re)
        & np.isfinite(bulge_re)
        & np.isfinite(disk_q)
        & np.isfinite(bulge_q)
        & np.isfinite(angle_deg)
        & (disk_re > 0.0)
        & (bulge_re > 0.0)
        & (disk_q > 0.0)
        & (disk_q <= 1.0)
        & (bulge_q > 0.0)
        & (bulge_q <= 1.0)
    )
    viable_bd = (
        finite_geometry
        & np.isfinite(bd_chi2)
        & (bd_chi2 >= 0.0)
        & (bd_chi2 < max_bd_chi2)
        & finite_bd_photometry
    )
    generator_ready = isolated & viable_bd

    bulge_to_total = _bulge_fraction(
        bulge_mags["VIS"], disk_mags["VIS"]
    )
    re_combined = np.full(row_count, np.nan, dtype=np.float32)
    if np.any(viable_bd):
        re_combined[viable_bd] = circularized_effective_radius_arcsec(
            bulge_re[viable_bd],
            bulge_q[viable_bd],
            np.power(
                10.0, -0.4 * bulge_mags["VIS"][viable_bd].astype(np.float64)
            ),
            disk_re[viable_bd],
            disk_q[viable_bd],
            np.power(
                10.0, -0.4 * disk_mags["VIS"][viable_bd].astype(np.float64)
            ),
        ).astype(np.float32)
    re_kpc = _physical_re_kpc(re_combined, z_phot)
    mean_sb_vis = np.full(row_count, np.nan, dtype=np.float32)
    sb_good = _valid_mag(band_mags["VIS"]) & (re_combined > 0.0)
    mean_sb_vis[sb_good] = (
        band_mags["VIS"][sb_good]
        + 2.5 * np.log10(2.0 * np.pi * re_combined[sb_good] ** 2)
    ).astype(np.float32)

    selected = population
    arrays: dict[str, np.ndarray] = {
        "catalog_id": ids[selected],
        "ra_deg": ra[selected],
        "dec_deg": dec[selected],
        "object_type": obj_type[selected],
        "flag_star": flag_star[selected],
        "flag_blend": flag_blend[selected],
        "warn_flag": warn_flag[selected],
        "count_clean": clean[selected],
        "count_isolated": isolated[selected],
        "generator_ready": generator_ready[selected],
        "mag_vis": mag_vis[selected],
        "mag_auto_vis": mag_auto_vis[selected],
        "z_phot": z_phot[selected],
        "z_pdf_median": z_med[selected],
        "z_pdf_l68": z_l68[selected],
        "z_pdf_u68": z_u68[selected],
        "logmass_lephare": logmass_lephare[selected],
        "logsfr_lephare": logsfr_lephare[selected],
        "logssfr_lephare": logssfr_lephare[selected],
        "logage_lephare": logage_lephare[selected],
        "ebv_lephare": ebv_lephare[selected],
        "logmass_cigale": logmass_cigale[selected],
        "sfr_cigale": sfr_cigale[selected],
        "sfr100myr_cigale": sfr100_cigale[selected],
        "metallicity_cigale": metallicity_cigale[selected],
        "ebv_cigale": ebv_cigale[selected],
        "chi2_reduced_cigale": cigale_chi2[selected],
        "disk_re_arcsec": disk_re[selected],
        "bulge_re_arcsec": bulge_re[selected],
        "disk_axis_ratio": disk_q[selected],
        "bulge_axis_ratio": bulge_q[selected],
        "position_angle_deg": angle_deg[selected],
        "bd_chi2": bd_chi2[selected],
        "bulge_to_total_vis": bulge_to_total[selected],
        "re_combined_arcsec": re_combined[selected],
        "re_combined_kpc": re_kpc[selected],
        "mean_surface_brightness_vis": mean_sb_vis[selected],
    }
    for band in EUCLID_PROXY_COLUMNS:
        # Broad total profile-fit photometry is available for population
        # galaxies even when the stricter B+D decomposition failed.
        arrays[f"mag_{band}"] = model_mags[band][selected]
        arrays[f"mag_bd_{band}"] = band_mags[band][selected]
        arrays[f"mag_bulge_{band}"] = bulge_mags[band][selected]
        arrays[f"mag_disk_{band}"] = disk_mags[band][selected]
    arrays.update({name: value[selected] for name, value in morph.items()})
    arrays.update(
        {
            f"galfitm_{name}": value[selected]
            for name, value in galfit_arrays.items()
        }
    )

    selections_full = {
        "population": population,
        "clean": clean,
        "isolated": isolated,
        "generator_ready": generator_ready,
    }
    selections_selected = {
        "population": np.ones(int(population.sum()), dtype=bool),
        "clean": arrays["count_clean"],
        "isolated": arrays["count_isolated"],
        "generator_ready": arrays["generator_ready"],
    }
    count_rows = _write_counts(
        out / "cosmos2025_number_counts.csv",
        mag_vis,
        selections_full,
        area_arcmin2=area_deg2 * 3600.0,
    )

    clean_density = np.asarray(
        [
            row["clean_density_per_mag_arcmin2"]
            for row in count_rows
        ],
        dtype=np.float64,
    )
    count_centers = np.asarray(
        [row["mag_center"] for row in count_rows], dtype=np.float64
    )
    peak_index = int(np.argmax(clean_density))
    source_stat = source.stat()
    selection_counts = {
        name: int(mask.sum()) for name, mask in selections_full.items()
    }
    type_counts = {
        str(int(value)): int(count)
        for value, count in zip(
            *np.unique(obj_type, return_counts=True), strict=True
        )
    }
    summary: dict[str, Any] = {
        "created_utc": datetime.now(UTC).isoformat(),
        "source": {
            "path": str(source),
            "size_bytes": int(source_stat.st_size),
            "mtime_utc": datetime.fromtimestamp(
                source_stat.st_mtime, UTC
            ).isoformat(),
            "rows": int(row_count),
            "catalog": "COSMOS2025 v1.1 / COSMOS-Web",
        },
        "normalization": {
            "area_deg2": float(area_deg2),
            "area_arcmin2": float(area_deg2 * 3600.0),
            "completeness_correction": "none",
            "warning": (
                "Raw NIRCam-selected catalog counts. Treat a faint-end "
                "turnover as selection/completeness until injection-recovery "
                "completeness is supplied."
            ),
        },
        "selection_definitions": {
            "population": (
                "LePHARE type == 0, flag_star == 0, 0 < zfinal < 15, "
                "and 5 < mag_model_hst-f814w < 50; blends retained"
            ),
            "clean": "population and warn_flag == 0; blends retained",
            "isolated": "clean and flag_blend == 0",
            "generator_ready": (
                f"isolated plus viable B+D geometry, chi2 < {max_bd_chi2:g}, "
                "and finite bulge+disk photometry in F814W and UltraVISTA Y/J/H"
            ),
        },
        "counts": {
            "all_rows": int(row_count),
            "object_type_raw": type_counts,
            **selection_counts,
            "clean_density_arcmin2_all_valid_vis": float(
                clean.sum() / (area_deg2 * 3600.0)
            ),
            "raw_clean_differential_peak_mag": float(
                count_centers[peak_index]
            ),
        },
        "percentiles": {
            "population_mag_vis": _nan_percentiles(
                mag_vis, population
            ),
            "population_z_phot": _nan_percentiles(
                z_phot, population
            ),
            "population_logmass_lephare": _nan_percentiles(
                logmass_lephare, population
            ),
            "generator_re_arcsec": _nan_percentiles(
                re_combined, generator_ready
            ),
            "generator_re_kpc": _nan_percentiles(
                re_kpc, generator_ready
            ),
            "generator_bulge_to_total_vis": _nan_percentiles(
                bulge_to_total, generator_ready
            ),
            "generator_mean_surface_brightness_vis": _nan_percentiles(
                mean_sb_vis, generator_ready
            ),
        },
        "outputs": {
            "prior_npz": str(out / "cosmos2025_population_prior.npz"),
            "number_counts_csv": str(
                out / "cosmos2025_number_counts.csv"
            ),
            "summary_json": str(
                out / "cosmos2025_population_summary.json"
            ),
            "schema_json": str(out / "cosmos2025_schema.json"),
            "diagnostics_png": str(
                out / "cosmos2025_population_diagnostics.png"
            ),
        },
        "consumed_columns": consumed,
    }
    schema["consumed_columns"] = consumed

    _atomic_npz(out / "cosmos2025_population_prior.npz", arrays)
    _atomic_json(out / "cosmos2025_population_summary.json", summary)
    _atomic_json(out / "cosmos2025_schema.json", schema)
    _make_plot(
        out / "cosmos2025_population_diagnostics.png",
        arrays,
        selections_selected,
        count_rows,
    )
    print(
        "[cosmos-prior] "
        f"population={selection_counts['population']:,} "
        f"clean={selection_counts['clean']:,} "
        f"generator_ready={selection_counts['generator_ready']:,}"
    )
    print(
        "[cosmos-prior] raw clean count peak at "
        f"m_F814W={count_centers[peak_index]:.3g}; "
        "no completeness correction applied"
    )
    print(f"[cosmos-prior] outputs -> {out}")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--catalog",
        default=Config.COSMOS2025_CATALOG_PATH,
        help="COSMOS2025 master FITS path.",
    )
    parser.add_argument(
        "--out-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the compact prior and diagnostics.",
    )
    parser.add_argument(
        "--area-deg2",
        type=float,
        default=DEFAULT_AREA_DEG2,
        help="Effective catalog area for density normalization (default 0.54).",
    )
    parser.add_argument(
        "--max-bd-chi2",
        type=float,
        default=10.0,
        help="B+D chi-square ceiling for generator_ready rows.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    extract_catalog(
        args.catalog,
        args.out_dir,
        area_deg2=args.area_deg2,
        max_bd_chi2=args.max_bd_chi2,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
