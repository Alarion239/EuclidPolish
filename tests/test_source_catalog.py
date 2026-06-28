import os

from euclid_polish.sky.generation import source_catalog as sc


def _meta():
    return {
        "galaxies": [
            {"type": "galaxy", "render": "sersic", "x_pix": 100.0, "y_pix": 120.0,
             "z_phot": 0.7, "catalog_id": 5, "flux_e_per_band": [3000.0, 1, 2, 3]},
            {"type": "galaxy", "render": "tng", "x_pix": 40.0, "y_pix": 200.0,
             "z": float("nan"), "subhalo_id": 99, "flux_e_per_band": [800.0, 1, 2, 3]},
        ],
        "lenses": [
            {"type": "lens", "x_pix": 128.0, "y_pix": 130.0, "z_lens": 0.5,
             "z_source": 2.0, "theta_E_arcsec": 1.3, "lens_subhalo_id": "g7",
             "flux_e_per_band": [5000.0, 1, 2, 3]},
        ],
    }


def test_writer_then_reader_roundtrip(tmp_path):
    p = str(tmp_path / "sources_validate.csv")
    w = sc.SourceCatalogWriter(p)
    w.add_field(0, _meta())
    w.add_field(1, {"galaxies": [], "lenses": []})  # empty field still ok
    w.close()

    by_field = sc.read_sources(p)
    assert set(by_field) == {0}                       # field 1 contributed no rows
    rows = by_field[0]
    assert len(rows) == 3
    sersic = next(r for r in rows if r["render"] == "sersic")
    assert sersic["type"] == "galaxy" and sersic["x_pix"] == 100.0
    assert sersic["flux_vis_e"] == 3000.0 and sersic["z"] == 0.7
    lens = next(r for r in rows if r["type"] == "lens")
    assert lens["theta_E_arcsec"] == 1.3 and lens["subhalo_id"] == "g7"
    tng = next(r for r in rows if r["render"] == "tng")
    assert tng["subhalo_id"] == "99" and tng["z"] is None   # NaN -> None


def test_read_sources_missing_file(tmp_path):
    assert sc.read_sources(str(tmp_path / "nope.csv")) == {}


def test_concat_source_csvs_preserves_order(tmp_path):
    a = str(tmp_path / "a.csv"); b = str(tmp_path / "b.csv")
    wa = sc.SourceCatalogWriter(a); wa.add_field(0, _meta()); wa.close()
    wb = sc.SourceCatalogWriter(b); wb.add_field(1, _meta()); wb.close()
    out = str(tmp_path / "sources_validate.csv")
    sc.concat_source_csvs([a, b], out)
    by_field = sc.read_sources(out)
    assert set(by_field) == {0, 1}
    with open(out) as f:
        assert sum(1 for ln in f if ln.startswith("field_index,")) == 1
