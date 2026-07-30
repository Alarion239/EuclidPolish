from __future__ import annotations

import csv
import json

import numpy as np
from astropy.io import fits

from scripts import fasrc_extract_cosmos_population as mod


def _table(columns: dict[str, np.ndarray], name: str) -> fits.BinTableHDU:
    cols = []
    for key, values in columns.items():
        values = np.asarray(values)
        fmt = "K" if values.dtype.kind in "iu" else "D"
        cols.append(fits.Column(name=key, format=fmt, array=values))
    return fits.BinTableHDU.from_columns(cols, name=name)


def _tiny_catalog(path) -> None:
    n = 5
    photo = _table(
        {
            "id": np.arange(n),
            "ra": np.linspace(150.0, 150.1, n),
            "dec": np.linspace(2.0, 2.1, n),
            "flag_star": [0, 0, 1, 0, 0],
            "flag_blend": [0, 1, 0, 0, 0],
            "warn_flag": [0, 0, 0, 1, 0],
            "mag_model_hst-f814w": [22.0, 24.0, 23.0, 25.0, 27.0],
            "mag_model_uvista-y": [21.8, 23.8, 22.8, 24.8, 26.8],
            "mag_model_uvista-j": [21.7, 23.7, 22.7, 24.7, 26.7],
            "mag_model_uvista-h": [21.6, 23.6, 22.6, 24.6, 26.6],
            "mag_auto_hst-f814w": [22.1, 24.1, 23.1, 25.1, 27.1],
        },
        "PHOTOMETRY HOTCOLD AND SE++",
    )
    lephare = _table(
        {
            "type": [0, 0, 1, 0, 0],
            "zfinal": [0.5, 1.0, 0.2, 1.5, 2.0],
            "zpdf_med": [0.5, 1.0, 0.2, 1.5, 2.0],
            "zpdf_l68": [0.4, 0.9, 0.1, 1.4, 1.9],
            "zpdf_u68": [0.6, 1.1, 0.3, 1.6, 2.1],
            "mass_med": [9.0, 9.5, 8.0, 10.0, 8.5],
            "sfr_med": [0.0, 0.2, -1.0, 0.4, -0.5],
            "ssfr_med": [-9.0, -9.3, -9.0, -9.6, -9.0],
            "age_med": [9.0, 9.1, 8.0, 9.3, 8.7],
            "ebv_minchi2": [0.1] * n,
        },
        "LEPHARE",
    )
    blank = _table({"unused": np.zeros(n)}, "SE++APER")
    cigale = _table(
        {
            "mass": 10.0 ** np.array([9.0, 9.5, 8.0, 10.0, 8.5]),
            "sfr_inst": np.ones(n),
            "sfr_100myr": np.ones(n),
            "metallicity": np.full(n, 0.014),
            "ebv_stars": np.full(n, 0.1),
            "chi2_red_best_fit": np.ones(n),
        },
        "CIGALE",
    )
    ml_columns = {}
    for band in ("f150w", "f277w", "f444w"):
        for label, value in (("sph", 0.2), ("disk", 0.5), ("irr", 0.1), ("bd", 0.2)):
            ml_columns[f"{label}_{band}_mean"] = np.full(n, value)
    ml = _table(ml_columns, "ML-MORPHO")
    bd_columns = {
        "disk_radius_deg": np.full(n, 0.3 / 3600.0),
        "bulge_radius_deg": np.full(n, 0.1 / 3600.0),
        "disk_axratio": np.full(n, 0.7),
        "bulge_axratio": np.full(n, 0.8),
        "angle_bd": np.linspace(0.0, 90.0, n),
        "fmf_b+d_chi2": [1.0, 1.0, 1.0, 1.0, 20.0],
    }
    for column in mod.EUCLID_PROXY_COLUMNS.values():
        bd_columns[f"mag_model_bd_total_{column}"] = np.linspace(22.0, 26.0, n)
        bd_columns[f"mag_model_bulge_{column}"] = np.linspace(23.0, 27.0, n)
        bd_columns[f"mag_model_disk_{column}"] = np.linspace(22.5, 26.5, n)
    bd = _table(bd_columns, "B+D")
    galfitm = _table(
        {
            "rearc_f150w_sersic": np.full(n, 0.2),
            "nsersic_f150w_sersic": np.full(n, 1.5),
            "qratio_f150w_sersic": np.full(n, 0.7),
            "asymmetry_f150w": np.full(n, 0.1),
            "smoothness_f150w": np.full(n, 0.1),
            "concentration_f150w": np.full(n, 3.0),
            "gini_f150w": np.full(n, 0.5),
            "m20_f150w": np.full(n, -1.5),
        },
        "GALFITM-MORPHO",
    )
    fits.HDUList(
        [fits.PrimaryHDU(), photo, lephare, blank, cigale, ml, bd, galfitm]
    ).writeto(path)


def test_extract_catalog_keeps_counts_separate_from_morphology(tmp_path):
    catalog = tmp_path / "tiny.fits"
    output = tmp_path / "out"
    _tiny_catalog(catalog)

    summary = mod.extract_catalog(
        str(catalog), str(output), area_deg2=1.0, max_bd_chi2=10.0
    )

    assert summary["counts"]["population"] == 4
    assert summary["counts"]["clean"] == 3
    assert summary["counts"]["isolated"] == 2
    assert summary["counts"]["generator_ready"] == 1
    with np.load(output / "cosmos2025_population_prior.npz") as prior:
        assert prior["catalog_id"].tolist() == [0, 1, 3, 4]
        assert prior["generator_ready"].tolist() == [True, False, False, False]
        assert np.isfinite(prior["re_combined_arcsec"][0])
        assert prior["mag_VIS"][0] == 22.0
        assert prior["mag_Y_E"][0] == 21.8
        assert prior["mag_bd_VIS"][0] < prior["mag_bulge_VIS"][0]

    with (output / "cosmos2025_number_counts.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    row_22 = next(row for row in rows if float(row["mag_lo"]) == 22.0)
    assert int(row_22["population_count"]) == 1
    assert int(row_22["generator_ready_count"]) == 1

    saved = json.loads(
        (output / "cosmos2025_population_summary.json").read_text()
    )
    assert saved["normalization"]["completeness_correction"] == "none"
    assert (output / "cosmos2025_schema.json").is_file()
    assert (output / "cosmos2025_population_diagnostics.png").is_file()


def test_component_magnitudes_and_bulge_fraction():
    bulge = np.array([23.0, 999.0])
    disk = np.array([22.0, 22.0])
    total = mod._mag_from_components(bulge, disk)
    bt = mod._bulge_fraction(bulge, disk)
    assert total[0] < 22.0
    assert 0.0 < bt[0] < 0.5
    assert np.isnan(total[1])
    assert np.isnan(bt[1])
