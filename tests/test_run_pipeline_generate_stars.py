"""Serial ``step_generate`` in scripts/run_pipeline.py: which splits draw fixed
stars, and the STEP 1 banner.

The simulator, writers, PSF loading and provenance are stubbed, so the step
runs its split loop without rendering or writing real records.
"""

from __future__ import annotations

import contextlib
import importlib.util
import os
from types import SimpleNamespace

import pytest


def _load_run_pipeline():
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "scripts", "run_pipeline.py",
    )
    spec = importlib.util.spec_from_file_location("run_pipeline_generate_stars", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rp = _load_run_pipeline()


class _FakeSim:
    """Tags each field with the ``n_stars`` it was drawn with."""

    def __init__(self, cat, cfg, vis_psf_set=None):
        pass

    def simulate_field(self, rng, n_stars=None):
        return SimpleNamespace(n_stars=n_stars), {"stars": []}


class _FakeSources:
    def __init__(self, path):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def add_field(self, i, meta):
        pass


@pytest.fixture
def n_stars_by_subset(monkeypatch):
    """``{subset: [n_stars of each written clean field]}``."""
    written: dict[str, list] = {}

    @contextlib.contextmanager
    def fake_open_writer(name, records_dir=None):
        fields = written.setdefault(name.removeprefix("clean_"), [])
        yield SimpleNamespace(path=name, count=0,
                              write=lambda sky, index=None: fields.append(sky.n_stars))

    monkeypatch.setattr(rp, "SkySimulator", _FakeSim)
    monkeypatch.setattr(rp, "open_writer", fake_open_writer)
    monkeypatch.setattr(rp, "SourceCatalogWriter", _FakeSources)
    monkeypatch.setattr(rp, "load_all_band_psf_sets", lambda **kw: {"VIS": None})
    monkeypatch.setattr(rp, "_generator_config_from_args", lambda args: object())
    monkeypatch.setattr(rp, "make_generation_context", lambda cfg, seed=None: None)
    monkeypatch.setattr(rp, "ResourceSampler",
                        lambda reporter: SimpleNamespace(start=lambda: None))
    return written


def _args(tmp_path, *, onthefly_train: bool):
    return SimpleNamespace(
        records_dir=str(tmp_path), psf_dir="/nonexistent", require_empirical_psf=False,
        ntrain=2, nvalid=1, ntest=1, image_size=64, galaxy_density_arcmin2=0.0,
        seed=7, force=False, onthefly_train=onthefly_train)


def test_record_mode_train_draws_fixed_stars(tmp_path, n_stars_by_subset):
    """Record mode forward-models train from the sources CSV, so its fields
    carry fixed stars exactly as on the parallel path."""
    rp.step_generate(_args(tmp_path, onthefly_train=False))
    assert n_stars_by_subset == {"train": [None, None], "validate": [None], "test": [None]}


def test_onthefly_train_is_starless_eval_splits_keep_stars(tmp_path, n_stars_by_subset):
    rp.step_generate(_args(tmp_path, onthefly_train=True))
    assert n_stars_by_subset == {"train": [0, 0], "validate": [None], "test": [None]}


def test_step1_banner_counts_every_split(tmp_path, n_stars_by_subset, capsys):
    rp.step_generate(_args(tmp_path, onthefly_train=False))
    banner = next(line for line in capsys.readouterr().out.splitlines() if "STEP 1" in line)
    assert "2 train + 1 valid + 1 test" in banner
