"""Per-source sidecar catalog for synthetic fields.

``SkySimulator.simulate_field`` knows every galaxy/lens/star it places, but the
TFRecord schema stores only pixels. This module persists that source list as a
CSV next to the records (``sources_<subset>.csv``) so the evaluation can crop
postage stamps centered on a known lens or galaxy, and so the forward op can
re-inject a field's fixed stars, including their sampled four-band colours,
temperature, and extinction (the scene is stored STARLESS). One row per
galaxy, per lens, and per star.
"""

from __future__ import annotations

import csv
import math
import os
from typing import Any

TNG_RENDER_RECORD_VERSION = 3

SOURCE_COLS = ["field_index", "type", "render", "x_pix", "y_pix",
               "flux_vis_e", "flux_y_e", "flux_j_e", "flux_h_e",
               "z", "subhalo_id", "theta_E_arcsec",
               # Extra galaxy truth persisted for later analysis (empty for
               # lenses, and for whichever render path doesn't provide it):
               "re_arcsec", "logmass", "mass_scale",
               # Conditional empirical mass-sSFR rank transport provenance.
               # The proxy is selection-only; it never rescales flux or size.
               "native_tng_logmass", "native_tng_sfr",
               "native_tng_logssfr", "native_tng_zero_sfr",
               "morphology_proxy_logmass",
               "target_logmass", "target_logssfr",
               "target_mass_quantile", "target_ssfr_quantile",
               "tng_mass_quantile", "tng_ssfr_quantile",
               "morphology_mass_quantile_delta",
               "morphology_ssfr_quantile_delta",
               "morphology_selection_probability",
               "morphology_effective_donors",
               "morphology_kernel_bandwidth_quantile",
               "morphology_mass_kernel_bandwidth_quantile",
               "morphology_ssfr_kernel_bandwidth_quantile",
               "morphology_worker_use_count", "morphology_activity_class",
               "physical_model_fingerprint",
               # TNG population configuration saved with every rendered TNG
               # row so analysis can distinguish legacy and regenerated data.
               "tng_density_arcmin2", "tng_mf_alpha",
               "galaxy_density_arcmin2", "galaxy_prior_density_arcmin2",
               "galaxy_vis_magnitude_max", "population_prior",
               "mag_hst_f814w", "target_vis_mag",
               "target_re_arcsec", "achieved_re_arcsec",
               "nominal_re_arcsec", "native_halflight_px",
               "radius_scale_factor", "radius_rendering",
               "radius_renderer_fingerprint", "radius_remeasured",
               "render_support_clipped", "tng_render_record_version",
               "magnitude_fit_fingerprint",
               "target_vis_2fwhm_mag", "target_vis_2fwhm_flux_e",
               "achieved_vis_2fwhm_mag", "achieved_vis_2fwhm_flux_e",
               "aperture_psf_fwhm_arcsec", "aperture_radius_arcsec",
               "aperture_psf_source", "brightness_scale",
               "brightness_transfer",
               # Per-band stellar magnitudes (empty for galaxies/lenses); the
               # forward op re-injects fixed stars with their sampled colour.
               "mag_vis", "mag_y_e", "mag_j_e", "mag_h_e",
               "temperature_k", "extinction_av"]


def _flux_vis(src: dict[str, Any]):
    f = src.get("flux_e_per_band")
    return float(f[0]) if f else ""


def _flux_band(src: dict[str, Any], index: int):
    flux = src.get("flux_e_per_band")
    return float(flux[index]) if flux and len(flux) > index else ""


def _num(v: Any):
    """A finite float for the CSV, or '' for None/NaN/unparseable."""
    if v is None:
        return ""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return ""
    return "" if not math.isfinite(f) else f


def _z(src: dict[str, Any]):
    z = src.get("z_phot", src.get("z_lens", src.get("z")))
    if z is None:
        return ""
    z = float(z)
    return "" if math.isnan(z) else z


def _galaxy_row(field_index: int, g: dict[str, Any]) -> dict[str, Any]:
    one_pass_radius = bool(g.get("radius_rendering"))
    achieved_re = (
        None if one_pass_radius
        else g.get("achieved_re_arcsec", g.get("apparent_re_arcsec"))
    )
    return {
        "field_index": field_index, "type": "galaxy",
        "render": g.get("render", ""),
        "x_pix": float(g["x_pix"]), "y_pix": float(g["y_pix"]),
        "flux_vis_e": _flux_vis(g),
        "flux_y_e": _flux_band(g, 1),
        "flux_j_e": _flux_band(g, 2),
        "flux_h_e": _flux_band(g, 3),
        "z": _z(g),
        "subhalo_id": g.get("subhalo_id", ""), "theta_E_arcsec": "",
        # Sampled nominal Euclid Sersic R_e (arcsec), or the legacy apparent
        # radius for old records; log10 stellar mass (Msun) where known.
        "re_arcsec":  _num(g.get("re_arcsec", g.get("apparent_re_arcsec"))),
        "logmass":    _num(g.get("logmass")),
        "mass_scale": _num(g.get("mass_scale")),
        "native_tng_logmass": _num(g.get("native_tng_logmass")),
        "native_tng_sfr": _num(g.get("native_tng_sfr")),
        "native_tng_logssfr": _num(g.get("native_tng_logssfr")),
        "native_tng_zero_sfr": _num(g.get("native_tng_zero_sfr")),
        "morphology_proxy_logmass": _num(
            g.get("morphology_proxy_logmass")
        ),
        "target_logmass": _num(g.get("target_logmass")),
        "target_logssfr": _num(g.get("target_logssfr")),
        "target_mass_quantile": _num(g.get("target_mass_quantile")),
        "target_ssfr_quantile": _num(g.get("target_ssfr_quantile")),
        "tng_mass_quantile": _num(g.get("tng_mass_quantile")),
        "tng_ssfr_quantile": _num(g.get("tng_ssfr_quantile")),
        "morphology_mass_quantile_delta": _num(
            g.get("morphology_mass_quantile_delta")
        ),
        "morphology_ssfr_quantile_delta": _num(
            g.get("morphology_ssfr_quantile_delta")
        ),
        "morphology_selection_probability": _num(
            g.get("morphology_selection_probability")
        ),
        "morphology_effective_donors": _num(
            g.get("morphology_effective_donors")
        ),
        "morphology_kernel_bandwidth_quantile": _num(
            g.get("morphology_kernel_bandwidth_quantile")
        ),
        "morphology_mass_kernel_bandwidth_quantile": _num(
            g.get("morphology_mass_kernel_bandwidth_quantile")
        ),
        "morphology_ssfr_kernel_bandwidth_quantile": _num(
            g.get("morphology_ssfr_kernel_bandwidth_quantile")
        ),
        "morphology_worker_use_count": _num(
            g.get("morphology_worker_use_count")
        ),
        "morphology_activity_class": g.get(
            "morphology_activity_class", ""
        ),
        "physical_model_fingerprint": g.get(
            "physical_model_fingerprint", ""
        ),
        "tng_density_arcmin2": _num(g.get("tng_density_arcmin2")),
        "tng_mf_alpha": _num(g.get("tng_mf_alpha")),
        "galaxy_density_arcmin2": _num(g.get("galaxy_density_arcmin2")),
        "galaxy_prior_density_arcmin2": _num(
            g.get("galaxy_prior_density_arcmin2")
        ),
        "galaxy_vis_magnitude_max": _num(
            g.get("galaxy_vis_magnitude_max")
        ),
        "population_prior": g.get("population_prior", ""),
        "mag_hst_f814w": _num(g.get("mag_hst_f814w")),
        "target_vis_mag": _num(g.get("target_vis_mag")),
        "target_re_arcsec": _num(g.get("target_re_arcsec")),
        "achieved_re_arcsec": _num(achieved_re),
        "nominal_re_arcsec": _num(g.get("nominal_re_arcsec")),
        "native_halflight_px": _num(g.get("native_halflight_px")),
        "radius_scale_factor": _num(g.get("radius_scale_factor")),
        "radius_rendering": g.get("radius_rendering", ""),
        "radius_renderer_fingerprint": g.get(
            "radius_renderer_fingerprint", ""
        ),
        "radius_remeasured": int(bool(g.get("radius_remeasured", False))),
        "render_support_clipped": int(bool(
            g.get("render_support_clipped", False)
        )),
        "tng_render_record_version": (
            TNG_RENDER_RECORD_VERSION if g.get("render") == "tng" else ""
        ),
        "magnitude_fit_fingerprint": g.get(
            "magnitude_fit_fingerprint", ""
        ),
        "target_vis_2fwhm_mag": _num(g.get("target_vis_2fwhm_mag")),
        "target_vis_2fwhm_flux_e": _num(g.get("target_vis_2fwhm_flux_e")),
        "achieved_vis_2fwhm_mag": _num(g.get("achieved_vis_2fwhm_mag")),
        "achieved_vis_2fwhm_flux_e": _num(g.get("achieved_vis_2fwhm_flux_e")),
        "aperture_psf_fwhm_arcsec": _num(g.get("aperture_psf_fwhm_arcsec")),
        "aperture_radius_arcsec": _num(g.get("aperture_radius_arcsec")),
        "aperture_psf_source": g.get("aperture_psf_source", ""),
        "brightness_scale": _num(g.get("brightness_scale")),
        "brightness_transfer": g.get("brightness_transfer", ""),
    }


def _lens_row(field_index: int, lens: dict[str, Any]) -> dict[str, Any]:
    theta = lens.get("theta_E_arcsec")
    return {
        "field_index": field_index, "type": "lens", "render": "",
        "x_pix": float(lens["x_pix"]), "y_pix": float(lens["y_pix"]),
        "flux_vis_e": _flux_vis(lens), "z": _z(lens),
        "subhalo_id": lens.get("lens_subhalo_id", ""),
        "theta_E_arcsec": float(theta) if theta is not None else "",
        # Galaxy-truth columns are not meaningful for the lens row.
        "re_arcsec": "", "logmass": "", "mass_scale": "",
        "native_tng_logmass": "", "native_tng_sfr": "",
        "native_tng_logssfr": "", "native_tng_zero_sfr": "",
        "morphology_proxy_logmass": "",
        "target_logmass": "", "target_logssfr": "",
        "target_mass_quantile": "", "target_ssfr_quantile": "",
        "tng_mass_quantile": "", "tng_ssfr_quantile": "",
        "morphology_mass_quantile_delta": "",
        "morphology_ssfr_quantile_delta": "",
        "morphology_selection_probability": "",
        "morphology_effective_donors": "",
        "morphology_kernel_bandwidth_quantile": "",
        "morphology_mass_kernel_bandwidth_quantile": "",
        "morphology_ssfr_kernel_bandwidth_quantile": "",
        "morphology_worker_use_count": "",
        "morphology_activity_class": "",
        "physical_model_fingerprint": "",
        "tng_density_arcmin2": "", "tng_mf_alpha": "",
        "galaxy_density_arcmin2": "", "population_prior": "",
        "mag_hst_f814w": "", "target_vis_mag": "", "brightness_scale": "",
        "brightness_transfer": "",
    }


def _star_row(field_index: int, star: dict[str, Any]) -> dict[str, Any]:
    return {
        "field_index": field_index, "type": "star", "render": "",
        "x_pix": float(star["x_pix"]), "y_pix": float(star["y_pix"]),
        "flux_vis_e": "", "z": "", "subhalo_id": "", "theta_E_arcsec": "",
        "re_arcsec": "", "logmass": "", "mass_scale": "",
        "native_tng_logmass": "", "native_tng_sfr": "",
        "native_tng_logssfr": "", "native_tng_zero_sfr": "",
        "morphology_proxy_logmass": "",
        "target_logmass": "", "target_logssfr": "",
        "target_mass_quantile": "", "target_ssfr_quantile": "",
        "tng_mass_quantile": "", "tng_ssfr_quantile": "",
        "morphology_mass_quantile_delta": "",
        "morphology_ssfr_quantile_delta": "",
        "morphology_selection_probability": "",
        "morphology_effective_donors": "",
        "morphology_kernel_bandwidth_quantile": "",
        "morphology_mass_kernel_bandwidth_quantile": "",
        "morphology_ssfr_kernel_bandwidth_quantile": "",
        "morphology_worker_use_count": "",
        "morphology_activity_class": "",
        "physical_model_fingerprint": "",
        "tng_density_arcmin2": "", "tng_mf_alpha": "",
        "galaxy_density_arcmin2": "", "population_prior": "",
        "mag_hst_f814w": "", "target_vis_mag": "", "brightness_scale": "",
        "brightness_transfer": "",
        "mag_vis": _num(star.get("mag_vis")),
        "mag_y_e": _num(star.get("mag_y_e")),
        "mag_j_e": _num(star.get("mag_j_e")),
        "mag_h_e": _num(star.get("mag_h_e")),
        "temperature_k": _num(star.get("temperature_k")),
        "extinction_av": _num(star.get("extinction_av")),
    }


class SourceCatalogWriter:
    """Append galaxy/lens/star rows to ``path`` as fields are generated."""

    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._f = open(path, "w", newline="")
        self._w = csv.DictWriter(self._f, fieldnames=SOURCE_COLS)
        self._w.writeheader()

    def add_field(self, field_index: int, meta: dict[str, Any]) -> None:
        for g in meta.get("galaxies", []) or []:
            self._w.writerow(_galaxy_row(field_index, g))
        for lens in meta.get("lenses", []) or []:
            self._w.writerow(_lens_row(field_index, lens))
        for star in meta.get("stars", []) or []:
            self._w.writerow(_star_row(field_index, star))

    def close(self) -> None:
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _parse(row: dict[str, str]) -> dict[str, Any]:
    out: dict[str, Any] = {"type": row["type"], "render": row["render"],
                           "subhalo_id": row["subhalo_id"] or None}
    out["field_index"] = int(row["field_index"])
    for k in ("x_pix", "y_pix", "flux_vis_e", "flux_y_e", "flux_j_e",
              "flux_h_e", "z", "theta_E_arcsec",
              "re_arcsec", "logmass", "mass_scale", "mag_vis",
              "native_tng_logmass", "native_tng_sfr",
              "native_tng_logssfr", "native_tng_zero_sfr",
              "morphology_proxy_logmass",
              "target_logmass", "target_logssfr",
              "target_mass_quantile", "target_ssfr_quantile",
              "tng_mass_quantile", "tng_ssfr_quantile",
              "morphology_mass_quantile_delta",
              "morphology_ssfr_quantile_delta",
              "morphology_selection_probability",
              "morphology_effective_donors",
              "morphology_kernel_bandwidth_quantile",
              "morphology_mass_kernel_bandwidth_quantile",
              "morphology_ssfr_kernel_bandwidth_quantile",
              "morphology_worker_use_count",
              "tng_density_arcmin2", "tng_mf_alpha",
              "galaxy_density_arcmin2", "galaxy_prior_density_arcmin2",
              "galaxy_vis_magnitude_max",
              "mag_hst_f814w", "target_vis_mag",
              "target_re_arcsec", "achieved_re_arcsec",
              "nominal_re_arcsec", "native_halflight_px",
              "radius_scale_factor", "radius_remeasured",
              "render_support_clipped", "tng_render_record_version",
              "target_vis_2fwhm_mag", "target_vis_2fwhm_flux_e",
              "achieved_vis_2fwhm_mag", "achieved_vis_2fwhm_flux_e",
              "aperture_psf_fwhm_arcsec", "aperture_radius_arcsec",
              "brightness_scale",
              "mag_y_e", "mag_j_e", "mag_h_e", "temperature_k",
              "extinction_av"):
        v = row.get(k, "")
        out[k] = float(v) if v not in ("", None) else None
    out["population_prior"] = row.get("population_prior") or None
    out["brightness_transfer"] = row.get("brightness_transfer") or None
    out["magnitude_fit_fingerprint"] = (
        row.get("magnitude_fit_fingerprint") or None
    )
    out["radius_rendering"] = row.get("radius_rendering") or None
    out["radius_renderer_fingerprint"] = (
        row.get("radius_renderer_fingerprint") or None
    )
    for key in ("radius_remeasured", "render_support_clipped"):
        if out[key] is not None:
            out[key] = bool(out[key])
    out["aperture_psf_source"] = row.get("aperture_psf_source") or None
    out["morphology_activity_class"] = (
        row.get("morphology_activity_class") or None
    )
    out["physical_model_fingerprint"] = (
        row.get("physical_model_fingerprint") or None
    )
    return out


def read_sources(csv_path: str) -> dict[int, list[dict[str, Any]]]:
    """``field_index -> list[source dict]``; missing file -> ``{}``."""
    if not os.path.isfile(csv_path):
        return {}
    by_field: dict[int, list[dict[str, Any]]] = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            r = _parse(row)
            by_field.setdefault(r["field_index"], []).append(r)
    return by_field


def concat_source_csvs(part_paths: list[str], out_path: str) -> None:
    """Concatenate shard CSVs (in the given order) into one, single header.

    Atomic: build a sibling temp file then ``os.replace`` it into place, so a
    crash mid-merge never leaves a truncated ``sources_<subset>.csv`` that a
    resumed run would mistake for complete."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "w", newline="") as out:
        out.write(",".join(SOURCE_COLS) + "\r\n")
        for p in part_paths:
            if not os.path.isfile(p):
                continue
            with open(p, newline="") as f:
                next(f, None)                     # skip shard header
                for line in f:
                    out.write(line)
    os.replace(tmp_path, out_path)
