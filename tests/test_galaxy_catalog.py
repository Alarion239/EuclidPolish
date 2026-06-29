"""Unit tests for the real-galaxy eval catalog builder (archive mocked)."""
import csv
import math
import os

import pytest

from euclid_polish.config import Config
from euclid_polish.eval import galaxy_catalog as gc


def test_diam_to_area_px_matches_circle():
    # diameter 5" at 0.1"/px → radius 25 px → area = pi*25^2.
    area = gc._diam_to_area_px(5.0)
    r_px = (5.0 / 2.0) / Config.VIS_PIXEL_SCALE_ARCSEC
    assert area == pytest.approx(math.pi * r_px * r_px)


def test_galaxy_adql_has_cuts():
    q = gc.galaxy_adql(10.0, -5.0, 0.05)
    assert "catalogue.mer_catalogue" in q
    assert f"{gc._POINTLIKE_COL} = 0" in q          # extended, not a star
    assert f"{gc._SPURIOUS_COL} = 0" in q           # not an artifact
    assert f"{gc._QUALITY_COL} = 0" in q            # clean detection
    assert f"{gc._SIZE_COL} BETWEEN" in q           # size window
    assert "CIRCLE('ICRS', 10.0, -5.0, 0.05)" in q  # the cone


def test_candidates_parse_and_mag_floor():
    rows = [
        # flux 50 µJy → mag ~19.65 → kept
        {"object_id": 1, "right_ascension": 10.01, "declination": -5.0,
         "segmentation_area": 800.0, "flux_vis_psf": 50.0},
        # flux 0.01 µJy → mag ~28.9 → too faint → dropped
        {"object_id": 2, "right_ascension": 10.02, "declination": -5.0,
         "segmentation_area": 800.0, "flux_vis_psf": 0.01},
        # non-finite flux → dropped
        {"object_id": 3, "right_ascension": 10.03, "declination": -5.0,
         "segmentation_area": 800.0, "flux_vis_psf": float("nan")},
    ]
    cands = gc._candidates_from_results(rows)
    assert [c["id"] for c in cands] == ["gal_1"]
    assert cands[0]["ra"] == 10.01 and cands[0]["dec"] == -5.0


def test_candidates_handle_none_results():
    assert gc._candidates_from_results(None) == []


class _FakeJob:
    def __init__(self, rows):
        self._rows = rows

    def get_results(self):
        return self._rows


def _fake_launch(query):
    # Field at ra=10 returns one good galaxy + one sitting ON the lens (excluded);
    # field at ra=20 returns two good galaxies.
    if "10.0," in query:
        return _FakeJob([
            {"object_id": 1, "right_ascension": 10.01, "declination": -5.0,
             "segmentation_area": 800.0, "flux_vis_psf": 50.0},
            {"object_id": 2, "right_ascension": 10.0, "declination": -5.0,
             "segmentation_area": 800.0, "flux_vis_psf": 50.0},
        ])
    return _FakeJob([
        {"object_id": 3, "right_ascension": 20.02, "declination": 30.0,
         "segmentation_area": 800.0, "flux_vis_psf": 50.0},
        {"object_id": 4, "right_ascension": 20.03, "declination": 30.01,
         "segmentation_area": 800.0, "flux_vis_psf": 50.0},
    ])


def _lens_csv(tmp_path):
    p = tmp_path / "lenses.csv"
    p.write_text("id,ra,dec,grade\nL1,10.0,-5.0,A\nL2,20.0,30.0,A\n")
    return str(p)


def test_build_draws_3n_and_excludes_lenses(monkeypatch, tmp_path):
    monkeypatch.setattr(gc, "_login", lambda **k: True)
    monkeypatch.setattr(gc.Euclid, "launch_job", staticmethod(_fake_launch))
    out = tmp_path / "galaxies.csv"
    path, n = gc.build(str(out), n_galaxies=3, lens_catalog_path=_lens_csv(tmp_path),
                       seed=0, cone_radius_arcmin=3.0, oversample=4)
    rows = list(csv.DictReader(open(path)))
    ids = {r["id"] for r in rows}
    assert n == 3
    assert "gal_2" not in ids                       # within 10" of lens L1 → excluded
    assert ids == {"gal_1", "gal_3", "gal_4"}
    assert all(r["grade"] == "gal" for r in rows)


def test_build_requires_auth(monkeypatch, tmp_path):
    monkeypatch.setattr(gc, "_login", lambda **k: False)
    with pytest.raises(RuntimeError):
        gc.build(str(tmp_path / "g.csv"), n_galaxies=3,
                 lens_catalog_path=_lens_csv(tmp_path), seed=0)


def test_build_with_client_skips_env_login(monkeypatch, tmp_path):
    # The WebUI authenticates via euclid_session (Euclid.login on the global
    # singleton), not env vars. When an already-authenticated client is passed,
    # build must NOT re-demand EUCLID_USER/PASSWORD — it should query and write.
    monkeypatch.setattr(gc, "_login", lambda **k: False)   # env login unavailable
    monkeypatch.setattr(gc.Euclid, "launch_job", staticmethod(_fake_launch))
    out = tmp_path / "galaxies.csv"
    path, n = gc.build(str(out), n_galaxies=3,
                       lens_catalog_path=_lens_csv(tmp_path), seed=0,
                       client=object())   # truthy stand-in for an authed EuclidCatalog
    ids = {r["id"] for r in csv.DictReader(open(path))}
    assert n == 3 and ids == {"gal_1", "gal_3", "gal_4"}


def test_build_reuses_cache_without_requery(monkeypatch, tmp_path):
    out = tmp_path / "galaxies.csv"
    out.write_text("id,ra,dec,grade\n"
                   "gal_1,10.0,-5.0,gal\ngal_2,11.0,-5.0,gal\ngal_3,12.0,-5.0,gal\n")
    called = {"login": False}
    monkeypatch.setattr(gc, "_login",
                        lambda **k: called.__setitem__("login", True) or True)
    path, n = gc.build(str(out), n_galaxies=3,
                       lens_catalog_path="does-not-exist.csv", seed=0)
    assert n == 3 and called["login"] is False      # cache satisfied → no archive call


def test_build_seed_deterministic(monkeypatch, tmp_path):
    monkeypatch.setattr(gc, "_login", lambda **k: True)
    monkeypatch.setattr(gc.Euclid, "launch_job", staticmethod(_fake_launch))
    a, _ = gc.build(str(tmp_path / "a.csv"), n_galaxies=2,
                    lens_catalog_path=_lens_csv(tmp_path), seed=7)
    b, _ = gc.build(str(tmp_path / "b.csv"), n_galaxies=2,
                    lens_catalog_path=_lens_csv(tmp_path), seed=7)
    ids_a = [r["id"] for r in csv.DictReader(open(a))]
    ids_b = [r["id"] for r in csv.DictReader(open(b))]
    assert ids_a == ids_b


def test_cli_invokes_build(monkeypatch, tmp_path):
    import importlib
    cli = importlib.import_module("scripts.fetch_galaxy_catalog")
    captured = {}

    def fake_build(out_csv=None, *, n_galaxies, lens_catalog_path, seed=0, **kw):
        captured.update(n_galaxies=n_galaxies, lens=lens_catalog_path, out=out_csv)
        return (out_csv or "galaxies.csv"), n_galaxies

    monkeypatch.setattr(cli.galaxy_catalog, "build", fake_build)
    rc = cli.main(["--n", "6", "--lens", "lenses.csv", "--out", str(tmp_path / "g.csv")])
    assert rc == 0
    assert captured["n_galaxies"] == 6 and captured["lens"] == "lenses.csv"
