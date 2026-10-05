"""The angular power spectrum scores the production SR against the record it
is trained to reproduce, and its cached curves know which record that was.

The sky SR cubes come from the STARFULL production ensemble, so HR is the
starfull ``hr_<subset>`` record (the scene with its stars), never the starless
``clean_<subset>`` scene. The cached JSON records that target and its content
fingerprint; the routes treat a cache measured against another record (or an
older copy of it) as not measured.
"""

from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from euclid_polish.config import Config
from euclid_polish.eval import power_spectrum
from euclid_polish.image import Image
from euclid_polish.image.tfio import tfrecord_path, write_images
from euclid_polish.training.target_blur import blur_target_array
from euclid_polish.web.app import create_app
from euclid_polish.web.helpers import paths, sky_records
from euclid_polish.web.routes import evaluation as evaluation_routes

N = 64
SAME_ORIGIN = {"Sec-Fetch-Site": "same-origin"}
CROSS_SITE = {"Sec-Fetch-Site": "cross-site"}


def _scene(seed: int) -> np.ndarray:
    """A smooth 4-band scene with structure across scales (no stars)."""
    rng = np.random.default_rng(seed)
    return np.stack([50.0 * gaussian_filter(rng.standard_normal((N, N)), 2.0) + 100.0
                     for _ in Config.LR_INPUT_BAND_NAMES], axis=-1).astype(np.float32)


def _with_stars(scene: np.ndarray) -> np.ndarray:
    """The same scene with a few bright point sources (the starfull target)."""
    out = scene.copy()
    for y, x in ((12, 20), (40, 50), (30, 9)):
        out[y, x, :] += 5.0e4
    return out


def _write(records_dir, kind: str, subset: str, data: np.ndarray) -> str:
    image = Image(data, float(Config.DEFAULT_PIXEL_SCALE), tuple(Config.LR_INPUT_BAND_NAMES),
                  kind == "clean", index=0)
    return write_images([image], f"{kind}_{subset}", records_dir=str(records_dir))


def _write_sr(subset: str, target: np.ndarray) -> None:
    """The SR cube a perfect starfull model makes: the blurred starfull target."""
    sr = blur_target_array(target, Config.TARGET_PSF_FWHM_ARCSEC,
                           pixel_scale_arcsec=float(Config.DEFAULT_PIXEL_SCALE))
    os.makedirs(sky_records.sky_sr_dir(), exist_ok=True)
    np.save(sky_records.sr_path(subset, 0), np.asarray(sr, np.float32))


@pytest.fixture
def records(tmp_path, monkeypatch):
    """A hermetic records dir (the synced test split) and SR-cube dir."""
    rdir = tmp_path / "records"
    rdir.mkdir()
    monkeypatch.setattr(paths, "_sky_records_local_dir", lambda: str(rdir))
    monkeypatch.setattr(evaluation_routes, "_sky_records_local_dir", lambda: str(rdir))
    monkeypatch.setattr(Config, "VIS_DIR", str(tmp_path / "vis"))
    monkeypatch.setattr(Config, "EVAL_RESULTS_DIR", str(tmp_path / "res"))
    os.makedirs(Config.EVAL_RESULTS_DIR)
    return rdir


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def _finite(values) -> np.ndarray:
    arr = np.asarray([np.nan if v is None else v for v in values], dtype=float)
    return arr[np.isfinite(arr)]


def test_the_starfull_sr_is_scored_against_the_starfull_hr_record(records):
    scene = _scene(0)
    target = _with_stars(scene)
    _write(records, "hr", "test", target)
    _write(records, "clean", "test", scene)          # the starless scene: not the target
    _write_sr("test", target)

    data = power_spectrum.power_spectrum_summary_data("test")

    assert data is not None and data["n_fields"] == 1
    # SR == the blurred starfull target → perfectly correlated, unit transfer.
    # Scored against the starless scene the stars would decorrelate it.
    for space in ("linear", "asinh"):
        curves = data["bands"]["VIS"][space]
        assert _finite(curves["r"]).size > 0
        np.testing.assert_allclose(_finite(curves["r"]), 1.0, atol=1e-6)
        np.testing.assert_allclose(_finite(curves["T"]), 1.0, atol=1e-6)


def test_the_summary_needs_only_the_starfull_record(records):
    target = _with_stars(_scene(1))
    _write(records, "hr", "test", target)            # no clean_test at all
    _write_sr("test", target)
    assert power_spectrum.power_spectrum_summary_data("test") is not None


def test_the_summary_records_its_target_and_its_fingerprint(records):
    target = _with_stars(_scene(2))
    path = _write(records, "hr", "test", target)
    _write_sr("test", target)

    data = power_spectrum.power_spectrum_summary_data("test")

    assert data["target"] == {"kind": "hr", "records": "hr_test.tfrecord",
                              "fingerprint": sky_records.records_fingerprint(path)}
    assert power_spectrum.spectrum_target_name("test") == "hr_test"
    # a regenerated target record is a different measurement
    _write(records, "hr", "test", _with_stars(_scene(3)))
    os.utime(path, ns=(1, 1))
    again = power_spectrum.power_spectrum_summary_data("test")
    assert again["target"]["fingerprint"] != data["target"]["fingerprint"]


def test_the_figure_title_names_the_split_it_measured(tmp_path, monkeypatch):
    titles: list[str] = []
    real_close = plt.close
    monkeypatch.setattr(plt, "close", lambda fig=None: (
        titles.append(fig._suptitle.get_text()) if fig is not None and fig._suptitle else None,
        real_close(fig))[-1])
    curves = {key: [1.0, 1.0] for key in ("T", "T_lo", "T_hi", "r", "r_lo", "r_hi", "count")}
    data = {"subset": "test", "n_fields": 3, "field_n": 64, "pixel_scale": 0.05, "lr_scale": 0.1,
            "theta_max": 2.0, "band_names": ["VIS"],
            "bands": {"VIS": {"psf_fwhm": 0.16, "linear": {"theta": [0.1, 0.5], **curves},
                              "asinh": {"theta": [0.1, 0.5], **curves}}}}
    power_spectrum.render_power_spectrum_figure(str(tmp_path / "aps.png"), data)
    assert titles and "test" in titles[0] and "validation" not in titles[0]


# --------------------------------------------------------------------------- #
# The cached curves (routes)                                                   #
# --------------------------------------------------------------------------- #
def _cache(payload: dict) -> str:
    path = os.path.join(Config.EVAL_RESULTS_DIR, "angular_power_spectrum.json")
    with open(path, "w") as handle:
        json.dump(payload, handle)
    with open(os.path.join(Config.EVAL_RESULTS_DIR, "angular_power_spectrum.png"), "wb") as handle:
        handle.write(b"\x89PNG cached")
    return path


def test_a_cache_scored_against_the_starless_scene_is_not_served(client, records):
    _write(records, "hr", "test", _with_stars(_scene(4)))
    _cache({"subset": "test", "n_fields": 1, "bands": {}})       # predates the target identity
    stale = client.get("/api/evaluation/angular-power-spectrum.json")
    assert stale.status_code == 404 and "measure it again" in stale.get_json()["error"]
    _cache({"subset": "test", "n_fields": 1, "bands": {},
            "target": {"kind": "clean", "records": "clean_test.tfrecord", "fingerprint": "x"}})
    assert client.get("/api/evaluation/angular-power-spectrum.json").status_code == 404


def test_a_cache_of_an_older_target_record_is_not_served(client, records):
    path = _write(records, "hr", "test", _with_stars(_scene(5)))
    current = {"kind": "hr", "records": "hr_test.tfrecord",
               "fingerprint": sky_records.records_fingerprint(path)}
    _cache({"subset": "test", "n_fields": 1, "bands": {}, "target": current})
    served = client.get("/api/evaluation/angular-power-spectrum.json")
    assert served.status_code == 200 and served.get_json()["target"] == current
    _write(records, "hr", "test", _with_stars(_scene(6)))         # the records were regenerated
    os.utime(path, ns=(1, 1))
    assert client.get("/api/evaluation/angular-power-spectrum.json").status_code == 404


def test_a_cache_is_kept_while_its_record_is_not_synced(client, records):
    _cache({"subset": "test", "n_fields": 1, "bands": {},
            "target": {"kind": "hr", "records": "hr_test.tfrecord", "fingerprint": "f"}})
    assert client.get("/api/evaluation/angular-power-spectrum.json").status_code == 200


def test_a_stale_figure_is_re_measured_on_a_same_origin_get_only(client, records, monkeypatch):
    _write(records, "hr", "test", _with_stars(_scene(7)))
    _cache({"subset": "test", "n_fields": 1, "bands": {}})       # stale (no target identity)
    renders: list[str] = []

    def render(out_png, *, out_json=None):
        renders.append(out_png)
        with open(out_png, "wb") as handle:
            handle.write(b"\x89PNG fresh")
        return out_png

    monkeypatch.setattr(power_spectrum, "render_power_spectrum_summary", render)
    cross = client.get("/api/evaluation/angular-power-spectrum", headers=CROSS_SITE)
    assert cross.data == b"\x89PNG cached" and renders == []      # a cross-site GET never renders
    same = client.get("/api/evaluation/angular-power-spectrum", headers=SAME_ORIGIN)
    assert same.data == b"\x89PNG fresh" and len(renders) == 1


def test_the_route_says_where_the_records_and_sr_come_from(client, records):
    missing = client.post("/api/evaluation/angular-power-spectrum")
    assert missing.status_code == 404
    error = missing.get_json()["error"]
    assert "Synthetic › Records" in error and "validation" not in error
    doc = client.application.view_functions["api_evaluation_angular_power_spectrum"].__doc__
    assert "Synthetic › Records" in doc and "/sky" not in doc
