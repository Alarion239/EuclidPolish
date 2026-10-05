"""The poster cutout: scripts/fasrc_poster_cutout.py and its console step.

The poster renders the training populations (activated galaxy-population
artifact + stellar prior, at their densities), carries the stars as the
forward's separate star plane, and reconstructs a field with the production
model (``load_eval_ensemble`` + ``sr_from_model``). Everything that would
touch TNG stamps, PSFs, the forward model or a network is faked.
"""

from __future__ import annotations

import json
import warnings
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits

import scripts.fasrc_poster_cutout as poster
from euclid_polish.config import Config
from euclid_polish.population.magnitude_law import StraightMagnitudeLaw
from euclid_polish.web import fasrc_config
from euclid_polish.web.fasrc_pipeline import REGISTRY

GALAXY_DENSITY = 151.25
STAR_DENSITY = 2.5


def _stellar_prior_payload(density: float | None = STAR_DENSITY) -> dict:
    law = StraightMagnitudeLaw(
        slope=0.15, intercept=-3.5, mag_bright=12.0, mag_faint=25.0,
        fit_bright=15.0, fit_faint=22.0,
        covariance=((1e-4, 0.0), (0.0, 1e-3)),
        r_squared=1.0, rms_log10_density=0.0, source="fixture",
    )
    population = {"magnitude_distribution": law.to_payload()}
    if density is not None:
        population["density_arcmin2"] = density
    return {
        "fingerprint": "s" * 64,
        "gaia": {"bp_rp_quantiles": [0.0, 2.0],
                 "temperature_quantiles_k": [7500.0, 4000.0]},
        "euclid_mapping": {
            "g_to_band_offset_coefficients": {
                key: [0.0, 0.0, 0.0]
                for key in ("mag_vis", "mag_y_e", "mag_j_e", "mag_h_e")},
            "residual_covariance": np.eye(4).tolist(),
        },
        "population": population,
        "color_model": {
            "kind": "gaia_euclid_latent_locus_v1",
            "bp_rp_edges": [0.0, 1.0, 2.0], "bp_rp_nodes": [0.5, 1.5],
            "temperature_nodes_k": [7000.0, 4200.0],
            "locus_colors": [[0.2, 0.1, 0.05], [0.8, 0.3, 0.1]],
            "intrinsic_color_covariance": (np.eye(3) * 0.01).tolist(),
            "magnitude_edges": [12.0, 18.5, 25.0],
            "magnitude_node_weights": [[0.5, 0.5], [0.5, 0.5]],
        },
    }


def _galaxy_payload() -> dict:
    return {"kind": "fixture-joint", "fingerprint": "j" * 64,
            "generation": {"surface_density_arcmin2": GALAXY_DENSITY}}


_STAR = {"type": "star", "x_pix": 10.0, "y_pix": 20.0, "mag_vis": 18.0,
         "mag_y_e": 17.5, "mag_j_e": 17.3, "mag_h_e": 17.1,
         "temperature_k": 5000.0, "extinction_av": 0.1}


class _FakeSky:
    """Stands in for SkySimulator: records its prior/config and returns a
    starless scene whose metadata records one star (as the generator does)."""

    instances: list[_FakeSky] = []

    def __init__(self, prior, cfg):
        self.prior, self.cfg = prior, cfg
        self.tng_atlas = [object()]
        _FakeSky.instances.append(self)

    def simulate_field(self, rng, **counts):
        n = self.cfg.image_size
        data = np.zeros((n, n, 4), np.float32)
        data[5, 5] = 7.0                                # one galaxy pixel
        # A field (no explicit counts): one galaxy and one star.
        n_gal = counts.get("n_galaxies", 1)
        n_star = counts.get("n_stars", 1)
        n_lens = counts.get("n_lenses", 0)
        meta = {
            "field_area_arcmin2": 0.1,
            "galaxy_density_arcmin2": self.cfg.galaxy_density_arcmin2,
            "star_density_arcmin2": self.cfg.star_density_arcmin2,
            "n_galaxies": n_gal, "n_stars": n_star, "n_lenses": n_lens,
            "galaxies": [{"render": "tng"}] * n_gal,
            "stars": [dict(_STAR)] * n_star,
            "lenses": [{"theta_E_arcsec": 1.0}] * n_lens,
        }
        return SimpleNamespace(data=data), meta


class _FakeForward:
    """Stands in for ObservationSimulator: a 2x sum-rebin, no noise."""

    calls: list[dict] = []

    def __init__(self, *, psf_sets_by_band, config):
        self.config = config

    def process(self, hr, rng=None, *, star_hr_4ch=None):
        data = np.asarray(hr.data, np.float32)
        if star_hr_4ch is not None:
            data = data + star_hr_4ch
        h, w, c = data.shape
        lr = data.reshape(h // 2, 2, w // 2, 2, c).sum(axis=(1, 3))
        _FakeForward.calls.append({"config": self.config, "hr": hr,
                                   "star_hr_4ch": star_hr_4ch})
        return SimpleNamespace(data=lr), SimpleNamespace(data=data)


class _FakeEnsemble:
    """The surface sr_from_model + eval_model_identity read (plain mean)."""

    member_labels = ["170·psnr", "171·psnr"]
    run_labels = ["170·psnr", "171·psnr"]
    combiner_kind = None
    n_members = n_run = 2
    label = "member mean (2 STARFULL models)"

    def member_arrays(self, lr):
        up = np.repeat(np.repeat(np.asarray(lr, np.float32), 2, 0), 2, 1)
        return np.stack([up, up + 1.0])

    def combine(self, members, lr):
        return members.mean(axis=0)


@pytest.fixture
def fakes(monkeypatch):
    """Fake the simulator, forward, PSFs, galaxy prior and ensemble loader."""
    _FakeSky.instances.clear()
    _FakeForward.calls.clear()
    loads: list[tuple] = []

    def fake_load(base_dir=None, num_res_blocks=None, *, log=None, **_kw):
        loads.append((base_dir, num_res_blocks))
        return _FakeEnsemble()

    monkeypatch.setattr(poster, "SkySimulator", _FakeSky)
    monkeypatch.setattr(poster, "ObservationSimulator", _FakeForward)
    monkeypatch.setattr(poster, "load_all_band_psf_sets", lambda **_kw: {})
    monkeypatch.setattr(poster, "population_prior_from_payload",
                        lambda payload: ("prior", payload["fingerprint"]))
    monkeypatch.setattr(poster, "load_eval_ensemble", fake_load)
    return SimpleNamespace(loads=loads)


# ---------------------------------------------------------------------------
# Stars actually appear
# ---------------------------------------------------------------------------

def test_star_mode_renders_the_star_without_the_tng_atlas(monkeypatch):
    def no_atlas(*_a, **_k):
        raise AssertionError("the star mode must not open the TNG atlas")

    monkeypatch.setattr(
        "euclid_polish.sky.generation.sky_simulator.TNGAtlas.open", no_atlas)
    scene, stars, rec, _seed = poster.generate_cutout(
        "star", seed=3, image_size=48,
        star_prior_payload=_stellar_prior_payload())

    assert not scene.any()                       # the scene itself is starless
    assert stars is not None and stars.shape == (48, 48, 4)
    peak = np.unravel_index(np.argmax(stars[..., 0]), stars.shape[:2])
    assert peak == (round(rec["y_pix"]), round(rec["x_pix"]))
    assert abs(peak[0] - 23.5) <= 1 and abs(peak[1] - 23.5) <= 1   # centred
    assert np.all(stars[peak] > 0.0)             # flux in all four bands


def test_star_mode_main_writes_a_nonzero_clean_cutout(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "euclid_polish.sky.generation.sky_simulator.TNGAtlas.open",
        lambda *_a, **_k: None)
    monkeypatch.setattr(poster.Config, "DATA_DIR", str(tmp_path))
    star_file = tmp_path / "star.json"
    star_file.write_text(json.dumps(_stellar_prior_payload()))

    assert poster.main(["--mode", "star", "--seed", "3", "--image-size", "48",
                        "--save", "--star-prior-file", str(star_file)]) == 0

    with fits.open(tmp_path / "_poster" / poster.FITS_NAME) as hdul:
        for band in poster.BAND_NAMES:
            assert float(hdul[band].data.sum()) > 0.0
    assert (tmp_path / "_poster" / poster.PNG_NAME).stat().st_size > 0


def test_star_plane_deposits_recorded_stars():
    assert poster.star_plane((32, 32, 4), []) is None
    plane = poster.star_plane((32, 32, 4), [dict(_STAR)])
    assert plane is not None
    assert np.count_nonzero(plane[..., 0]) == 1
    assert np.all(plane[20, 10] > 0.0)           # every band gets the star


# ---------------------------------------------------------------------------
# Training populations and densities
# ---------------------------------------------------------------------------

def test_field_mode_uses_the_activated_populations_and_densities(fakes):
    scene, stars, rec, _seed = poster.generate_cutout(
        "field", seed=1, image_size=32,
        star_prior_payload=_stellar_prior_payload(),
        galaxy_population_payload=_galaxy_payload())

    sim = _FakeSky.instances[-1]
    assert sim.prior == ("prior", "j" * 64)
    assert sim.cfg.galaxy_density_arcmin2 == GALAXY_DENSITY
    assert sim.cfg.star_density_arcmin2 == STAR_DENSITY
    assert sim.cfg.lens_density_arcmin2 == Config.LENS_DENSITY_ARCMIN2
    assert rec["galaxy_density_arcmin2"] == GALAXY_DENSITY
    assert rec["star_density_arcmin2"] == STAR_DENSITY
    assert scene[20, 10].sum() == 0.0            # star kept off the scene
    assert stars is not None and stars[20, 10, 0] > 0.0


def test_star_density_falls_back_to_the_generator_default():
    payload = _stellar_prior_payload(density=None)
    assert (poster.star_density_arcmin2(payload)
            == Config.DEFAULT_STAR_DENSITY_ARCMIN2)


@pytest.mark.parametrize("payload", [
    {}, {"generation": {}}, {"generation": {"surface_density_arcmin2": 0.0}},
    {"generation": {"surface_density_arcmin2": "nan"}},
])
def test_galaxy_density_needs_a_finite_artifact_density(payload):
    with pytest.raises(ValueError, match="surface density"):
        poster.galaxy_density_arcmin2(payload)


@pytest.mark.parametrize("mode", ["tng", "lens"])
def test_single_object_galaxy_modes_draw_from_the_galaxy_artifact(fakes, mode):
    _scene, stars, _rec, _seed = poster.generate_cutout(
        mode, seed=1, image_size=32,
        galaxy_population_payload=_galaxy_payload())

    sim = _FakeSky.instances[-1]
    assert sim.prior == ("prior", "j" * 64)
    assert sim.cfg.galaxy_density_arcmin2 == 0.0   # explicit counts instead
    assert sim.cfg.lens_density_arcmin2 == 0.0
    assert sim.cfg.lens_require_showable is (mode == "lens")
    assert stars is None


@pytest.mark.parametrize("mode, missing", [
    ("tng", "galaxy-population"), ("lens", "galaxy-population"),
    ("field", "galaxy-population"), ("star", "stellar prior"),
])
def test_modes_refuse_without_their_artifacts(mode, missing):
    kwargs = {"star_prior_payload": _stellar_prior_payload()} if mode == "field" else {}
    with pytest.raises(ValueError, match=missing):
        poster.generate_cutout(mode, seed=1, image_size=32, **kwargs)


# ---------------------------------------------------------------------------
# Field forward + production-model reconstruction
# ---------------------------------------------------------------------------

def test_field_forward_carries_the_star_plane_and_runs_the_production_model(fakes):
    scene = np.zeros((32, 32, 4), np.float32)
    stars = poster.star_plane(scene.shape, [dict(_STAR)])

    dirty, sr, identity, label = poster.forward_and_reconstruct(
        scene, stars, 5, psf_dir="unused", ensemble_dir="/x/ensemble",
        num_res_blocks=8)

    call = _FakeForward.calls[-1]
    assert call["star_hr_4ch"] is stars
    assert not np.asarray(call["hr"].data).any()        # starless HR scene
    config = call["config"]
    assert config.saturation_mask_prob == Config.TRAIN_SATURATION_MASK_PROB
    assert config.psf_warp_prob == Config.TRAIN_PSF_WARP_PROB
    assert config.psf_warp_alpha_max == Config.TRAIN_PSF_WARP_ALPHA_MAX
    assert fakes.loads == [("/x/ensemble", 8)]
    assert dirty.shape == (16, 16, 4) and dirty[10, 5, 0] > 0.0
    assert sr.shape == (32, 32, 4)
    assert identity["member_labels"] == ["170·psnr", "171·psnr"]
    assert identity["combiner_kind"] is None
    assert label == _FakeEnsemble.label


def test_model_cards_are_ascii_and_name_combiner_and_members():
    cards = {key: value for key, value, _ in poster.model_cards(
        {"member_labels": ["170·psnr", "171·psnr", "195·psnr"],
         "run_labels": ["170·psnr", "195·psnr"],
         "combiner_kind": "spatial_gate", "combiner_fingerprint": "f" * 64},
        "spatial gate over 2 of 3 STARFULL models")}
    assert cards["COMBINER"] == "spatial_gate"
    assert cards["MEMBERS"] == "170.psnr,171.psnr,195.psnr"
    assert cards["RUNMEMB"] == "170.psnr,195.psnr"
    assert (cards["NMEMBERS"], cards["NRUN"]) == (3, 2)
    assert cards["COMBFP"] == "f" * 64
    long_label = "spatial gate (convolutional, convex) over 18 of 26 STARFULL models"
    labels = [f"{i}·psnr" for i in range(26)]
    for identity in ({"member_labels": labels},
                     {"member_labels": labels, "combiner_kind": "spatial_gate",
                      "combiner_fingerprint": "0123456789abcdef" * 4}):
        header = fits.Header()
        with warnings.catch_warnings():
            warnings.simplefilter("error")       # no truncated-card warning
            for key, value, comment in poster.model_cards(identity, long_label):
                header[key] = (value, comment)
            header.tostring()
        assert header["MODEL"] == long_label
        assert header["MEMBERS"].split(",")[-1] == "25.psnr"
        assert header["COMBFP"] == identity.get("combiner_fingerprint", "")
        assert header["COMBINER"] == identity.get("combiner_kind", "member_mean")


def test_field_main_writes_clean_dirty_sr_with_model_identity(fakes, monkeypatch,
                                                              tmp_path):
    monkeypatch.setattr(poster.Config, "DATA_DIR", str(tmp_path))
    galaxy_file = tmp_path / "galaxy.json"
    galaxy_file.write_text(json.dumps(_galaxy_payload()))
    star_file = tmp_path / "star.json"
    star_file.write_text(json.dumps(_stellar_prior_payload()))

    assert poster.main([
        "--mode", "field", "--seed", "4", "--image-size", "32", "--save",
        "--joint-galaxy-population-file", str(galaxy_file),
        "--star-prior-file", str(star_file),
        "--ensemble-dir", "/x/ensemble",
    ]) == 0

    assert fakes.loads == [("/x/ensemble", Config.DEFAULT_NUM_RES_BLOCKS)]
    with fits.open(tmp_path / "_poster" / poster.FITS_NAME) as hdul:
        names = [hdu.name for hdu in hdul[1:]]
        assert names == ([f"CLEAN_{b}" for b in poster.BAND_NAMES]
                         + [f"DIRTY_{b}" for b in poster.BAND_NAMES]
                         + [f"SR_{b}" for b in poster.BAND_NAMES])
        clean_vis = hdul["CLEAN_VIS"].data
        assert clean_vis[20, 10] > 0.0 and clean_vis[5, 5] == 7.0  # star + galaxy
        assert hdul[0].header["STARDENS"] == STAR_DENSITY
        assert hdul[0].header["GALAXYDE"] == GALAXY_DENSITY
        sr = hdul["SR_VIS"].header
        assert "CKPT" not in sr
        assert sr["COMBINER"] == "member_mean"
        assert sr["MEMBERS"] == "170.psnr,171.psnr"
        assert sr["NRUN"] == 2
    assert (tmp_path / "_poster" / poster.PNG_NAME).stat().st_size > 0


def test_the_retired_checkpoint_flag_is_gone():
    with pytest.raises(SystemExit):
        poster.parse_args(["--mode", "field", "--ckpt-dir", "ckpt/wdsr"])
    args = poster.parse_args(["--mode", "field"])
    assert args.ensemble_dir is None


def test_inline_and_file_payloads_are_mutually_exclusive(tmp_path):
    path = tmp_path / "p.json"
    path.write_text('{"a": 1}')
    assert poster._read_payload("", str(path), label="x") == {"a": 1}
    assert poster._read_payload('{"b": 2}', "", label="x") == {"b": 2}
    assert poster._read_payload("", "", label="x") is None
    with pytest.raises(ValueError, match="not both"):
        poster._read_payload('{"b": 2}', str(path), label="x")


# ---------------------------------------------------------------------------
# Console step: artifacts frozen at submit, staged beside the job script
# ---------------------------------------------------------------------------

@pytest.fixture
def cfg():
    return fasrc_config.FasrcConfig(
        ssh_user="alice", repo_path="/n/foo/EuclidPolish",
        data_dir="/n/foo/data", ckpt_dir="/n/foo/ckpt",
        conda_env_path="/n/foo/conda",
    )


@pytest.fixture
def activated(monkeypatch):
    joint = {"version": 6, "fingerprint": "j" * 64,
             "generation": {"surface_density_arcmin2": GALAXY_DENSITY},
             "packed_forest": "x" * 200_000}
    stars = {"version": 6, "fingerprint": "s" * 64,
             "population": {"density_arcmin2": STAR_DENSITY},
             "large_locus": "y" * 140_000}
    state = {"active": joint, "candidate": joint, "is_active": True}
    monkeypatch.setattr(
        "euclid_polish.web.helpers.population_calibration.joint_galaxy_state",
        lambda: state)
    monkeypatch.setattr(
        "euclid_polish.web.helpers.population_calibration.active_star",
        lambda: stars)
    return SimpleNamespace(joint=joint, stars=stars, state=state)


@pytest.mark.parametrize("mode, galaxy, star", [
    ("field", True, True), ("tng", True, False),
    ("lens", True, False), ("star", False, True),
])
def test_poster_step_stages_the_artifacts_its_mode_needs(cfg, activated, mode,
                                                        galaxy, star):
    step = REGISTRY.get("poster_cutout")
    built = step.build_sbatch_body(params={"mode": mode}, resources=step.defaults,
                                   cfg=cfg, label="poster")
    body, params, files = built["body"], built["params"], built["payload_files"]

    assert ("--joint-galaxy-population-file" in body) is galaxy
    assert ("--star-prior-file" in body) is star
    assert "--joint-galaxy-population-json" not in body
    assert "--star-prior-json" not in body
    assert "x" * 1_000 not in body and "y" * 1_000 not in body
    assert len(body.encode("utf-8")) < 20_000
    assert "_joint_galaxy_population_json" not in params
    assert "_star_prior_json" not in params
    if galaxy:
        path = params["_joint_galaxy_population_file"]
        assert path.startswith("logs/pipeline/") and path in files
        assert json.loads(files[path]) == activated.joint
        assert params["_joint_galaxy_population_fingerprint"] == "j" * 64
        assert len(params["_joint_galaxy_population_sha256"]) == 64
    if star:
        path = params["_star_prior_file"]
        assert path in files and json.loads(files[path]) == activated.stars
        assert params["_star_prior_fingerprint"] == "s" * 64
    assert len(files) == int(galaxy) + int(star)


@pytest.mark.parametrize("mode", ["tng", "lens", "field"])
def test_poster_step_refuses_galaxy_modes_without_an_activated_fit(activated, mode):
    activated.state["is_active"] = False
    with pytest.raises(ValueError, match="galaxy fit"):
        REGISTRY.get("poster_cutout").prepare_params({"mode": mode})
    # The star mode does not need it.
    REGISTRY.get("poster_cutout").prepare_params({"mode": "star"})
