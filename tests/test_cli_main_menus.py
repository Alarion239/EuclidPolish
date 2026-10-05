"""Menu flows of the interactive CLI (:mod:`euclid_polish.cli.main`).

Each test scripts the questionary / ``input()`` answers of one menu action and
fakes every archive, model and subprocess call, so the flows run offline and
fast. They pin the on-disk layouts the CLI reads (per-band cutout dirs, the v2
records), the production-model path (ensemble + production combiner, never
the retired single checkpoint) and the forward step's record contract
(``clean_`` untouched, ``hr_`` + ``dirty_`` written with the recorded stars).
"""

from __future__ import annotations

import builtins
import importlib
import os
import subprocess
import sys

import numpy as np
import pytest
from astropy.io import fits

import euclid_polish.cli.main as cli_main
from euclid_polish.catalog import CatalogObject
from euclid_polish.cli.main import InteractiveCLI
from euclid_polish.config import Config
from euclid_polish.image import Image
from euclid_polish.image.tfio import open_writer, read_images, tfrecord_path
from euclid_polish.psf import PSF
from euclid_polish.sky.generation.source_catalog import SourceCatalogWriter, read_sources

BANDS = Config.LR_INPUT_BAND_NAMES
HR_SCALE = Config.DEFAULT_PIXEL_SCALE
LR_SCALE = Config.BAND_VIS.pixel_scale_lr_arcsec


class _Ask:
    def __init__(self, value):
        self._value = value

    def ask(self):
        return self._value


class Script:
    """Scripted answers for one menu flow. ``input()`` returns ``""`` (the
    prompt's default) once its answers run out; every questionary prompt and
    its choices are recorded."""

    def __init__(self, monkeypatch, *, inputs=(), selects=(), confirms=(), checkboxes=()):
        self._inputs = list(inputs)
        self._selects = list(selects)
        self._confirms = list(confirms)
        self._checkboxes = list(checkboxes)
        self.prompts: list[str] = []
        self.choices: list[list[dict]] = []
        monkeypatch.setattr(builtins, "input", self._input)
        monkeypatch.setattr(cli_main, "select", self._select)
        monkeypatch.setattr(cli_main, "confirm", self._confirm)
        monkeypatch.setattr(cli_main, "checkbox", self._checkbox)

    def _input(self, prompt=""):
        self.prompts.append(prompt)
        return self._inputs.pop(0) if self._inputs else ""

    def _select(self, message, choices=(), **_kw):
        self.prompts.append(message)
        self.choices.append(list(choices))
        return _Ask(self._selects.pop(0) if self._selects else None)

    def _confirm(self, message, **_kw):
        self.prompts.append(message)
        return _Ask(self._confirms.pop(0) if self._confirms else False)

    def _checkbox(self, message, choices=(), **_kw):
        self.prompts.append(message)
        self.choices.append(list(choices))
        return _Ask(self._checkboxes.pop(0) if self._checkboxes else None)


def _hr(i=0, side=8, value=1.0):
    return Image(data=np.full((side, side, 4), value, np.float32),
                 pixel_scale_arcsec=HR_SCALE, band_names=BANDS, is_clean=True,
                 index=i)


def _lr(i=0, side=4, value=1.0):
    return Image(data=np.full((side, side, 4), value, np.float32),
                 pixel_scale_arcsec=LR_SCALE, band_names=BANDS, is_clean=False,
                 index=i)


def _write_records(records_dir, name, images):
    with open_writer(name, records_dir=str(records_dir)) as w:
        for img in images:
            w.write(img, index=img.index)


def _write_cutout(path, value=1.0):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = np.arange(256, dtype=np.float32).reshape(16, 16) + value
    fits.PrimaryHDU(data).writeto(path, overwrite=True)


class _FakeEnsemble:
    """Production-model double: ``member_arrays`` gives two members and
    ``combine`` (the production gate) a result distinct from their mean."""

    def __init__(self, sr_value=5.0):
        self.sr_value = float(sr_value)
        self.label = "spatial gate over 2 STARFULL models"
        self.n_members = self.n_run = 2
        self.member_labels = self.run_labels = ["00·psnr", "01·psnr"]
        self.combine_calls = 0

    def member_arrays(self, lr):
        h, w = np.asarray(lr).shape[:2]
        return np.stack([np.zeros((2 * h, 2 * w, 4), np.float32),
                         np.full((2 * h, 2 * w, 4), 100.0, np.float32)])

    def combine(self, members, lr):
        self.combine_calls += 1
        return np.full(members.shape[1:], self.sr_value, np.float32)


@pytest.fixture
def records(tmp_path, monkeypatch):
    """Point the v2 records at tmp (and the retired v1 dir at an empty one)."""
    v2 = tmp_path / "records_v2"
    v2.mkdir()
    monkeypatch.setattr(Config, "RECORDS_DIR_V2", str(v2))
    monkeypatch.setattr(Config, "RECORDS_DIR", str(tmp_path / "records_v1_empty"))
    return v2


@pytest.fixture
def no_single_model(monkeypatch):
    """Any construction of the retired single-checkpoint model fails the test."""
    def _forbidden(*_a, **_k):
        raise AssertionError("the CLI must not build the retired single-checkpoint Model")
    monkeypatch.setattr(cli_main, "Model", _forbidden, raising=False)
    monkeypatch.setattr(cli_main, "load_model_from_checkpoint", _forbidden, raising=False)


@pytest.fixture
def fake_ensemble(monkeypatch, tmp_path):
    model = _FakeEnsemble()
    calls = []

    def _load(base_dir=None, num_res_blocks=None, *, log=None, combiner_dir=None):
        calls.append(base_dir)
        return model
    monkeypatch.setattr(cli_main, "load_eval_ensemble", _load, raising=False)
    monkeypatch.setattr(cli_main, "default_ensemble_dir",
                        lambda: str(tmp_path / "ensemble"), raising=False)
    model.load_calls = calls
    return model


# ---------------------------------------------------------------------------
# Euclid operations: PSF band picker, output size, cutout layout, download
# ---------------------------------------------------------------------------

def test_psf_band_picker_shows_archive_pixel_scale(monkeypatch):
    script = Script(monkeypatch, selects=[None])
    InteractiveCLI()._extract_psf()
    labels = [c["name"] for c in script.choices[0]]
    assert [c["value"] for c in script.choices[0]] == [b.name for b in Config.BANDS]
    assert all('0.10"/pix' in label for label in labels), labels
    assert not any("0.30" in label for label in labels), labels


def test_extract_psf_bumps_even_output_size_to_odd(tmp_path, monkeypatch, capsys):
    cutouts = tmp_path / "cutouts_vis"
    cutouts.mkdir()
    seen = {}

    class _Spy(cli_main.PSFExtractor):
        def __init__(self, config=None):
            super().__init__(config)
            seen["config"] = self.config

        def get_cutout_files(self, cutout_dir, cutout_size=None):
            return []
    monkeypatch.setattr(cli_main, "PSFExtractor", _Spy)
    Script(monkeypatch,
           selects=["VIS", str(cutouts), str(tmp_path / "psf")],
           inputs=["64", "", "", "1024"])
    InteractiveCLI()._extract_psf()
    assert seen["config"].effective_output_size == 1023
    assert "1023" in capsys.readouterr().out


def test_extract_psf_reports_invalid_config_instead_of_crashing(tmp_path, monkeypatch, capsys):
    cutouts = tmp_path / "cutouts_vis"
    cutouts.mkdir()

    def _reject(config=None):
        raise ValueError("Invalid PSFExtractionConfig: boom")
    monkeypatch.setattr(cli_main, "PSFExtractor", _reject)
    Script(monkeypatch, selects=["VIS", str(cutouts), str(tmp_path / "psf")],
           inputs=["64", "", "", "1024"])
    InteractiveCLI()._extract_psf()
    assert "boom" in capsys.readouterr().out


@pytest.mark.parametrize("answer", ["many", "0"])
def test_extract_psf_rejects_a_bad_star_count(tmp_path, monkeypatch, capsys, answer):
    cutouts = tmp_path / "cutouts_vis"
    cutouts.mkdir()
    monkeypatch.setattr(cli_main, "PSFExtractor",
                        lambda config=None: pytest.fail("no extraction for a bad star count"))
    Script(monkeypatch, selects=["VIS", str(cutouts), str(tmp_path / "psf")],
           inputs=["64", answer])
    InteractiveCLI()._extract_psf()
    out = capsys.readouterr().out
    assert "✗" in out and "number of stars" in out.lower()


def test_visualize_psf_reads_the_band_psf_file(tmp_path, monkeypatch):
    psf_dir = tmp_path / "euclid_psf"
    yy, xx = np.mgrid[-7:8, -7:8]
    kernel = np.exp(-(xx ** 2 + yy ** 2) / 8.0).astype(np.float32)
    PSF(kernel, pixel_scale=0.05, oversampling=2).save(
        str(psf_dir), filename=Config.get_band("Y_E").psf_fits_filename)
    monkeypatch.setattr(Config, "VIS_PSF_DIR", str(tmp_path / "vis_psf"))
    saved = []

    class _Viz:
        def __init__(self, *a, **k):
            pass

        def add_scale_panel(self, *a, **k):
            pass

        def add_statistics_panel(self, data, info):
            saved.append(float(np.sum(data)))

        def save_figure(self, path):
            saved.append(path)
    monkeypatch.setattr(cli_main, "BaseVisualizer", _Viz)
    Script(monkeypatch, selects=["custom", "Y_E"], inputs=[str(psf_dir)])
    InteractiveCLI()._visualize_psf()
    assert saved == [pytest.approx(1.0),
                     os.path.join(str(tmp_path / "vis_psf"), "euclid_psf_Y_E.png")]


def _catalog(output_dir, ids):
    objects = [CatalogObject(ra=150.0 + i * 0.01, dec=2.0, id=i, magnitude=18.0)
               for i in ids]
    CatalogObject.write(objects, os.path.join(str(output_dir), Config.CATALOG_FILE))


def _by_id(output_dir):
    return {o.id: o for o in CatalogObject.read(
        os.path.join(str(output_dir), Config.CATALOG_FILE))}


def test_check_integrity_scans_band_dirs_and_flags_that_band(tmp_path, monkeypatch, capsys):
    out = tmp_path / "stars"
    _catalog(out, [1, 2])
    root = out / Config.CUTOUTS_SUBDIR
    _write_cutout(str(root / "VIS" / "star_0002_64.fits"))
    bad = root / "Y_E" / "star_0001_64.fits"
    bad.parent.mkdir(parents=True)
    bad.write_bytes(b"not a fits file")
    Script(monkeypatch, selects=[str(out)])
    InteractiveCLI()._check_integrity()
    text = capsys.readouterr().out
    assert "Total files:      2" in text
    objs = _by_id(out)
    assert objs[1].is_corrupted(64, band="Y_E")
    assert not objs[1].is_corrupted(64, band="VIS")
    assert not objs[2].is_corrupted(64, band="VIS")


def test_visualize_cutouts_reads_the_band_dir(tmp_path, monkeypatch):
    out = tmp_path / "stars"
    _catalog(out, [1])
    _write_cutout(str(out / Config.CUTOUTS_SUBDIR / "Y_E" / "star_0001_64.fits"))
    monkeypatch.setattr(Config, "VIS_CUTOUTS_DIR", str(tmp_path / "vis_cutouts"))
    saved = []

    class _Viz:
        def __init__(self, *a, **k):
            pass

        def add_scale_panel(self, *a, **k):
            pass

        def add_statistics_panel(self, data, info):
            saved.append(info["stats"])

        def save_figure(self, path):
            saved.append(path)
    monkeypatch.setattr(cli_main, "BaseVisualizer", _Viz)
    Script(monkeypatch, selects=[str(out), "Y_E"], inputs=["1"])
    InteractiveCLI()._visualize_cutouts()
    assert saved and saved[-1] == os.path.join(str(tmp_path / "vis_cutouts"),
                                               "star_0001_Y_E.png")
    assert saved[0]["Band"] == "Y_E"


def test_download_summary_lists_rejected_ids(tmp_path, monkeypatch, capsys):
    out = tmp_path / "stars"
    _catalog(out, [7])

    class _Client:
        def download_cutouts(self, objects, output_dir, cfg, show_progress=True):
            return {"downloaded": 0, "valid": 0, "corrupted": 1, "failed": 0,
                    "rejected_ids": [7], "unmatched_ids": [],
                    "invalid_coordinate_ids": [], "cutout_size": 64, "band": cfg.band}

    class _Catalog:
        @staticmethod
        def _unauthenticated():
            return _Client()
    monkeypatch.setattr(cli_main, "EuclidCatalog", _Catalog)
    Script(monkeypatch, selects=[str(out)], checkboxes=[["VIS"]],
           inputs=["64", "1"], confirms=[True])
    InteractiveCLI()._download_cutouts()
    text = capsys.readouterr().out
    assert ("VIS: rejected (failed download/validation or saturated core) "
            "star ids = [7]") in text


# ---------------------------------------------------------------------------
# Sky generation: the forward step keeps clean_ and writes hr_ + dirty_
# ---------------------------------------------------------------------------

class _FakeForward:
    """``process`` returns an LR (2× rebinned) and the HR target = scene +
    stars, like the real forward model; records the star planes it got."""

    instances: list = []

    def __init__(self, *a, **k):
        self.star_planes = []
        _FakeForward.instances.append(self)

    def process(self, hr, rng=None, *, star_hr_4ch=None):
        self.star_planes.append(star_hr_4ch)
        data = hr.data + (star_hr_4ch if star_hr_4ch is not None else 0.0)
        lr = data.reshape(data.shape[0] // 2, 2, data.shape[1] // 2, 2, 4).sum(axis=(1, 3))
        return (Image(data=lr.astype(np.float32), pixel_scale_arcsec=LR_SCALE,
                      band_names=BANDS, is_clean=False),
                Image(data=data.astype(np.float32), pixel_scale_arcsec=HR_SCALE,
                      band_names=BANDS, is_clean=True))


def _star(x, y, mag=18.0):
    return {"type": "star", "x_pix": x, "y_pix": y, "mag_vis": mag,
            "mag_y_e": mag, "mag_j_e": mag, "mag_h_e": mag}


def test_convolve_keeps_clean_and_writes_starfull_hr(records, monkeypatch):
    _write_records(records, "clean_validate", [_hr(0), _hr(1)])
    clean_path = tfrecord_path(str(records), "clean_validate")
    clean_bytes = open(clean_path, "rb").read()
    with SourceCatalogWriter(os.path.join(str(records), "sources_validate.csv")) as w:
        w.add_field(1, {"stars": [_star(3.0, 4.0)]})
    _FakeForward.instances.clear()
    monkeypatch.setattr(cli_main, "ObservationSimulator", _FakeForward)
    monkeypatch.setattr(cli_main, "load_all_band_psf_sets", lambda **k: {})
    monkeypatch.setattr(cli_main, "psf_inventory", lambda **k: {})
    Script(monkeypatch, inputs=["", "n"], confirms=[True])

    InteractiveCLI()._convolve_hr_to_lr()

    assert open(clean_path, "rb").read() == clean_bytes      # never reopened for writing
    hr = read_images(tfrecord_path(str(records), "hr_validate"), num_images=10)
    lr = read_images(tfrecord_path(str(records), "dirty_validate"), num_images=10)
    assert [h.index for h in hr] == [0, 1] and [x.index for x in lr] == [0, 1]
    planes = _FakeForward.instances[0].star_planes
    assert planes[0] is None                                  # field 0 recorded no stars
    assert planes[1][4, 3, 0] > 0                             # field 1's star re-injected
    assert hr[1].data[4, 3, 0] > hr[0].data[4, 3, 0]          # and kept in the HR target


def test_generate_records_each_fields_stars(records, tmp_path, monkeypatch):
    catalog = tmp_path / "cosmos.fits"
    catalog.write_bytes(b"x")

    class _Sim:
        def __init__(self, *a, **k):
            pass

        def simulate_field(self, rng):
            return _hr(), {"stars": [_star(2.0, 3.0)]}
    monkeypatch.setattr(cli_main, "CosmosTngPrior", lambda path: [])
    monkeypatch.setattr(cli_main, "active_star", lambda: {"stub": True})
    monkeypatch.setattr(cli_main, "SkySimulator", _Sim)
    monkeypatch.setattr(cli_main, "SkySimulatorConfig", lambda **k: k)
    Script(monkeypatch, inputs=[str(catalog), "2", "0", "0", "", "12"], confirms=[True])

    InteractiveCLI()._generate_clean_data()

    stars = read_sources(os.path.join(str(records), "sources_train.csv"))
    assert sorted(stars) == [0, 1]
    assert stars[1][0]["type"] == "star" and stars[1][0]["x_pix"] == 2.0


# ---------------------------------------------------------------------------
# Training menu: the production ensemble, never the retired checkpoint
# ---------------------------------------------------------------------------

def test_cli_module_has_no_single_model_entry_points():
    assert not hasattr(cli_main, "Model")
    assert not hasattr(cli_main, "load_model_from_checkpoint")


def _touch_records(records, *names):
    for name in names:
        open(tfrecord_path(str(records), name), "wb").close()


def test_train_summary_lists_every_output_band(records, monkeypatch, capsys, fake_ensemble):
    _touch_records(records, "dirty_train", "hr_train", "clean_train")
    Script(monkeypatch, confirms=[False])
    InteractiveCLI()._train_model()
    out = capsys.readouterr().out
    assert "Output channels: 4 (VIS, Y_E, J_E, H_E)" in out


def test_train_adds_an_ensemble_member_via_train_ensemble(
        records, tmp_path, monkeypatch, capsys, no_single_model, fake_ensemble):
    _touch_records(records, "dirty_train", "hr_train", "clean_train")
    monkeypatch.setattr(cli_main, "next_member_names",
                        lambda base, k: ["member_07"], raising=False)
    runs = []
    monkeypatch.setattr(cli_main.subprocess, "run",
                        lambda cmd, **k: runs.append(cmd) or
                        subprocess.CompletedProcess(cmd, 0))
    scored = []
    monkeypatch.setattr(cli_main, "evaluate_member_on_records",
                        lambda mdir, rdir: scored.append((mdir, rdir)) or
                        {"subset": "test", "n_scored": 2, "psnr_stretched": 41.25},
                        raising=False)
    Script(monkeypatch, inputs=["16", "", "1000", "2", "100"], confirms=[True])

    InteractiveCLI()._train_model()

    ens = str(tmp_path / "ensemble")
    assert len(runs) == 1
    cmd = runs[0]
    assert cmd[0] == sys.executable
    assert cmd[2].endswith(os.path.join("scripts", "train_ensemble.py"))
    args = dict(zip(cmd[3::2], cmd[4::2], strict=True))
    assert args == {"--mode": "add", "--count": "1", "--member-names": "member_07",
                    "--base-dir": ens, "--records-dir": str(records),
                    "--steps": "1000", "--batch-size": "2",
                    "--evaluate-every": "100", "--num-res-blocks": "16"}
    assert scored == [(os.path.join(ens, "member_07"), str(records))]
    assert "41.250 dB" in capsys.readouterr().out

    # The argv is a valid train_ensemble.py add run: one STARFULL record-mode
    # member under that name and depth.
    train_ensemble = importlib.import_module("scripts.train_ensemble")
    (spec,) = train_ensemble.build_specs(train_ensemble.parse_args(cmd[3:]), ens)
    assert (spec.name, spec.op, spec.target_steps, spec.num_res_blocks) == (
        "member_07", "add", 1000, 16)
    assert not spec.starless and not spec.forward_onthefly


def test_train_reports_a_failed_member_run(records, monkeypatch, capsys,
                                           no_single_model, fake_ensemble):
    _touch_records(records, "dirty_train", "hr_train", "clean_train")
    monkeypatch.setattr(cli_main, "next_member_names",
                        lambda base, k: ["member_07"], raising=False)
    monkeypatch.setattr(cli_main.subprocess, "run",
                        lambda cmd, **k: subprocess.CompletedProcess(cmd, 2))
    monkeypatch.setattr(cli_main, "evaluate_member_on_records",
                        lambda *a: pytest.fail("no eval after a failed run"), raising=False)
    Script(monkeypatch, confirms=[True])
    InteractiveCLI()._train_model()
    assert "exited with status 2" in capsys.readouterr().out


def test_evaluate_scores_production_sr_on_the_starfull_target(
        records, monkeypatch, capsys, no_single_model, fake_ensemble):
    """PSNR of the production SR (the gate's ``combine``, 5 e⁻ everywhere)
    against ``hr_`` (6 and 7 e⁻), averaged over fields; ``clean_`` is unused."""
    monkeypatch.setattr(Config, "TARGET_PSF_FWHM_ARCSEC", 0.0)
    _write_records(records, "dirty_test", [_lr(0), _lr(1)])
    _write_records(records, "hr_test", [_hr(0, value=6.0), _hr(1, value=7.0)])
    _write_records(records, "clean_test", [_hr(0, value=1e4), _hr(1, value=1e4)])
    Script(monkeypatch, inputs=["", ""])

    InteractiveCLI()._evaluate_model()

    out = capsys.readouterr().out
    assert fake_ensemble.load_calls and fake_ensemble.combine_calls == 2
    peak, knee = float(Config.PSNR_PEAK_E), float(Config.STRETCH_SCALE_E)
    peak_str = float(Config.PSNR_PEAK_STRETCHED)
    raw = np.mean([10 * np.log10(peak ** 2 / (v - 5.0) ** 2) for v in (6.0, 7.0)])
    stretched = np.mean([
        10 * np.log10(peak_str ** 2 / (np.arcsinh(v / knee) - np.arcsinh(5.0 / knee)) ** 2)
        for v in (6.0, 7.0)])
    assert f"PSNR (raw e⁻):                 {raw:.3f} dB" in out
    assert f"PSNR (stretched, loss-aligned): {stretched:.3f} dB" in out
    assert "test set" in out and "2 fields" in out


def test_reconstruct_from_records_v2_with_production_model(
        records, monkeypatch, no_single_model, fake_ensemble):
    monkeypatch.setattr(Config, "TARGET_PSF_FWHM_ARCSEC", 0.0)
    _write_records(records, "dirty_validate", [_lr(0)])
    _write_records(records, "hr_validate", [_hr(0, value=3.0)])
    rendered = {}

    def _render(lr_images, model, out_dir, *, hr_images=None, **_k):
        rendered.update(lr=lr_images, model=model, hr=hr_images)
        return []
    monkeypatch.setattr(cli_main, "reconstruct_and_render", _render)
    Script(monkeypatch, selects=["tfrecord", "validate"], inputs=["1", ""])

    InteractiveCLI()._reconstruct_image()

    assert rendered["model"] is fake_ensemble
    assert [img.index for img in rendered["lr"]] == [0]
    assert float(rendered["hr"][0].data.mean()) == pytest.approx(3.0)


def test_reconstruct_file_uses_the_production_sr(
        tmp_path, monkeypatch, no_single_model, fake_ensemble):
    lr_path = tmp_path / "lr.npy"
    np.save(lr_path, np.ones((4, 4, 4), np.float32))
    monkeypatch.setattr(Config, "VIS_RECONSTRUCTION_DIR", str(tmp_path / "recon"))
    plotted = {}
    monkeypatch.setattr(cli_main, "plot_reconstruction",
                        lambda lr, sr, **k: plotted.update(lr=lr, sr=sr, **k))
    Script(monkeypatch, selects=["file"], inputs=[str(lr_path), "", "", ""])

    InteractiveCLI()._reconstruct_image()

    assert plotted["sr"].shape == (8, 8, 4)
    assert float(plotted["sr"].mean()) == pytest.approx(fake_ensemble.sr_value)
    assert plotted["output_path"].startswith(str(tmp_path / "recon"))


def test_fetch_and_superresolve_uses_production_ensemble(
        tmp_path, monkeypatch, no_single_model, fake_ensemble):
    fetched = {}
    monkeypatch.setattr(cli_main, "fetch_and_superresolve",
                        lambda **k: fetched.update(k) or ("SR.fits", "SR.png"))
    Script(monkeypatch, inputs=["10", "-5", "8", str(tmp_path / "ens")])

    InteractiveCLI()._fetch_and_superresolve()

    assert fake_ensemble.load_calls == [str(tmp_path / "ens")]
    assert fetched["model"] is fake_ensemble


def _member(ensemble_dir, name, step):
    d = ensemble_dir / name
    d.mkdir(parents=True)
    (d / "checkpoint").write_text(
        f'model_checkpoint_path: "ckpt-{step}"\n'
        f'all_model_checkpoint_paths: "ckpt-{step}"\n')
    (d / f"ckpt-{step}.index").write_bytes(b"")
    return d


def test_inspect_lists_each_active_member(tmp_path, monkeypatch, capsys, fake_ensemble):
    ens = tmp_path / "ensemble"
    _member(ens, "member_02", 5)
    _member(ens, "member_10", 7)
    Script(monkeypatch)
    InteractiveCLI()._inspect_checkpoints()
    out = capsys.readouterr().out
    assert "member_02: 1 checkpoint(s), latest ckpt-5" in out
    assert "member_10: 1 checkpoint(s), latest ckpt-7" in out
    assert "Total: 2 member(s)" in out


def test_plot_training_log_defaults_to_the_newest_member(tmp_path, monkeypatch, fake_ensemble):
    ens = tmp_path / "ensemble"
    _member(ens, "member_99", 5)
    newest = _member(ens, "member_100", 7)          # sorts before member_99 by name
    log = cli_main.default_log_path(str(newest))
    open(log, "w").close()
    plotted = []
    monkeypatch.setattr(cli_main, "plot_training_log",
                        lambda path, out, smooth_window=0: plotted.append(path) or (0, 0))
    Script(monkeypatch)
    InteractiveCLI()._plot_training_log()
    assert plotted == [log]


# ---------------------------------------------------------------------------
# Visualization: training data lives in the v2 records
# ---------------------------------------------------------------------------

def test_visualize_training_data_reads_records_v2(records, tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "VIS_CLEAN_DIR", str(tmp_path / "vis_clean"))
    _write_records(records, "clean_train", [_hr(0)])
    drawn = []
    monkeypatch.setattr(cli_main, "draw_clean_image",
                        lambda data, out, index=None, vmax=None: drawn.append(index))
    Script(monkeypatch, selects=["clean"], inputs=["1", ""])
    InteractiveCLI()._visualize_training_data()
    assert drawn == [0]


def test_visualize_pairs_hr_target_with_dirty(records, tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "TARGET_PSF_FWHM_ARCSEC", 0.0)
    monkeypatch.setattr(Config, "VIS_DIRTY_DIR", str(tmp_path / "vis_dirty"))
    _write_records(records, "dirty_test", [_lr(0)])
    _write_records(records, "hr_test", [_hr(0, value=2.0)])
    pairs = []
    monkeypatch.setattr(cli_main, "draw_clean_dirty_pair",
                        lambda hr, lr, path, index=None, vmax=None:
                        pairs.append((index, float(hr.mean()))))
    Script(monkeypatch, selects=["pair"], inputs=["1", ""])
    InteractiveCLI()._visualize_training_data()
    assert pairs == [(0, 2.0)]
