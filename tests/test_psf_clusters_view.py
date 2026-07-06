"""PSF-cluster inspection view (Euclid PSF page).

Builds a synthetic multi-extension ePSF FITS (cluster centroids in two
far-apart patches) and a small catalog, then checks the renderer produces a
figure, computes diameters, and 404s when the ePSF is missing — without
depending on any real data file.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from astropy.io import fits
from werkzeug.exceptions import NotFound

from euclid_polish.catalog.catalog_object import CatalogObject
from euclid_polish.config import Config
from euclid_polish.web.helpers import sky_render


def _write_psf_fits(path, centroids):
    hdus = [fits.PrimaryHDU(np.zeros((4, 4), dtype=np.float32))]
    hdus[0].header["NPSF"] = len(centroids)
    for i, (ra, dec) in enumerate(centroids):
        h = fits.ImageHDU(np.zeros((4, 4), dtype=np.float32), name=f"PSF{i:03d}")
        h.header["RA"] = ra
        h.header["DEC"] = dec
        hdus.append(h)
    fits.HDUList(hdus).writeto(path, overwrite=True)


# Two clusters near (150, 2) and two near (151, 2.5) → one region; plus one
# far away at (60, -45) → a second region.
_CENTROIDS = [(150.0, 2.0), (150.3, 2.1), (151.0, 2.5), (60.0, -45.0)]


@pytest.fixture
def psf_file(tmp_path):
    p = tmp_path / "euclid_psf_VIS.fits"
    _write_psf_fits(str(p), _CENTROIDS)
    return p


def _is_png(b):
    return b[:8] == b"\x89PNG\r\n\x1a\n"


def test_great_circle_arcmin_known_value():
    # 1° along a meridian = 60 arcmin
    assert abs(sky_render._great_circle_arcmin(10.0, 0.0, 10.0, 1.0) - 60.0) < 1e-6


def test_centroids_only_render(psf_file):
    # No catalog → centroids-only map still renders.
    assert _is_png(sky_render._render_psf_clusters_png(None, str(psf_file)))


def test_render_with_catalog_diameters(psf_file, tmp_path):
    rng = np.random.default_rng(0)
    bands = [b.name for b in Config.BANDS]
    objects = []
    sid = 0
    for ra, dec in _CENTROIDS:
        for _ in range(12):
            sid += 1
            o = CatalogObject(ra=ra + rng.normal(0, 0.02),
                              dec=dec + rng.normal(0, 0.02),
                              id=sid, magnitude=18.0)
            for bn in bands:
                o.set_valid(511, band=bn)
            objects.append(o)
    cat_dir = tmp_path / "cat"
    cat_dir.mkdir()
    CatalogObject.write(objects, os.path.join(str(cat_dir), Config.CATALOG_FILE))
    assert _is_png(sky_render._render_psf_clusters_png(str(cat_dir), str(psf_file)))


def test_404_when_psf_missing(tmp_path):
    # Missing ePSF path (not synced) or None → 404 so the page hides the panel.
    with pytest.raises(NotFound):
        sky_render._render_psf_clusters_png(None, str(tmp_path / "nope.fits"))
    with pytest.raises(NotFound):
        sky_render._render_psf_clusters_png(None, None)


def _write_clusters_json(path, centroids):
    import json

    with open(path, "w") as f:
        json.dump({"source": "euclid_psf_VIS.fits",
                   "n_clusters": len(centroids),
                   "clusters": [{"ra": ra, "dec": dec, "n_stars": 100}
                                for ra, dec in centroids]}, f)


def test_json_metadata_renders_without_fits(tmp_path):
    # The synced FASRC metadata alone (no local ePSF FITS) renders the map.
    p = tmp_path / "euclid_psf_clusters.json"
    _write_clusters_json(str(p), _CENTROIDS)
    assert _is_png(sky_render._render_psf_clusters_png(
        None, None, clusters_json=str(p)))


def test_json_metadata_preferred_over_fits(psf_file, tmp_path):
    # JSON carries MORE clusters than the local FITS → it wins (that's the
    # point: FASRC has far more clusters than the synced kernel stack).
    many = _CENTROIDS + [(150.0 + 0.1 * i, 2.0 + 0.05 * i) for i in range(20)]
    p = tmp_path / "euclid_psf_clusters.json"
    _write_clusters_json(str(p), many)
    png = sky_render._render_psf_clusters_png(
        None, str(psf_file), clusters_json=str(p))
    assert _is_png(png)


def test_corrupt_json_falls_back_to_fits(psf_file, tmp_path):
    p = tmp_path / "euclid_psf_clusters.json"
    p.write_text("{not json")
    assert _is_png(sky_render._render_psf_clusters_png(
        None, str(psf_file), clusters_json=str(p)))


def test_dense_region_renders_without_annotations(tmp_path):
    # >60 clusters in one region → annotation-free branch still renders.
    rng = np.random.default_rng(1)
    many = [(150.0 + rng.uniform(-0.5, 0.5), 2.0 + rng.uniform(-0.5, 0.5))
            for _ in range(80)]
    p = tmp_path / "euclid_psf_clusters.json"
    _write_clusters_json(str(p), many)
    assert _is_png(sky_render._render_psf_clusters_png(
        None, None, clusters_json=str(p)))


def test_remote_meta_cmd_contains_paths():
    from euclid_polish.web import fasrc_config
    from euclid_polish.web.routes.psfs import _clusters_meta_remote_cmd

    cfg = fasrc_config.load()
    cmd = _clusters_meta_remote_cmd(cfg)
    assert "euclid_psf_VIS.fits" in cmd
    assert "euclid_psf_clusters.json" in cmd
    assert "astropy.io" in cmd and "PYEOF" in cmd
    # headers only — the dump must never touch HDU data
    assert ".data" not in cmd
