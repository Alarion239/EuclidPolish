"""Smoke tests: every CLI / pipeline entry point imports cleanly.

These don't *invoke* the interactive menu — they only verify that the module
graph is consistent (no circular imports, no missing dependencies, no
typos in symbol names).
"""

from __future__ import annotations


def test_cli_module_imports():
    import euclid_polish.cli.main  # noqa: F401


def test_run_pipeline_script_imports():
    import scripts.run_pipeline  # noqa: F401


def test_fasrc_generate_hst_tfrecords_script_imports():
    """Smoke guard for the analytic-A pair generator. A syntax error
    here would only fire when SLURM actually executed the script,
    wasting a real allocation."""
    import importlib.util
    import os
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "scripts", "fasrc_generate_hst_tfrecords.py",
    )
    spec = importlib.util.spec_from_file_location(
        "fasrc_generate_hst_tfrecords_under_test", path,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert callable(mod._init_worker)
    assert callable(mod._kernel_forward)
    assert callable(mod._is_stamp_too_bright)


def test_cli_class_constructs():
    """InteractiveCLI class is well-formed and instantiates."""
    from euclid_polish.cli.main import InteractiveCLI
    cli = InteractiveCLI()
    # Methods we touched in Phase 7 must exist.
    assert hasattr(cli, "_generate_clean_data")
    assert hasattr(cli, "_convolve_hr_to_lr")
    assert hasattr(cli, "_extract_psf")
    assert hasattr(cli, "_train_model")
    assert hasattr(cli, "_show_psf_inventory")


def test_psf_inventory_helper_works():
    """The CLI's PSF inventory helper returns a dict over all known bands."""
    from euclid_polish.config import Config
    from euclid_polish.psf.psf_library import psf_inventory
    inv = psf_inventory(psf_dir="/nonexistent_dir_for_test")
    assert set(inv.keys()) == {b.name for b in Config.BANDS}
    assert all(v is None for v in inv.values())


def test_multiband_pipeline_components_wired():
    """The three pipeline steps can be wired together end-to-end."""
    import numpy as np
    from euclid_polish.config import Config
    from euclid_polish.sky.generation.sky_simulator import (
        SkySimulatorConfig, SkySimulator,
    )
    from euclid_polish.sky.observation.observation_simulator import (
        ObservationSimulator, ObservationSimulatorConfig,
    )
    from euclid_polish.psf.psf_library import load_all_band_psfs
    from tests._tiny_catalog import TinyCosmosCatalog

    cat = TinyCosmosCatalog(n_galaxies=200, seed=0)
    sim = SkySimulator(cat, SkySimulatorConfig(image_size=96))
    psfs = load_all_band_psfs(psf_dir="/nonexistent_for_test")
    fwd = ObservationSimulator(psfs_by_band=psfs,
                           config=ObservationSimulatorConfig(add_noise=False))

    rng = np.random.default_rng(0)
    hr, _ = sim.simulate_field(rng, n_sersic=3, n_stars=1, n_lenses=0)
    lr, hr_target = fwd.process(hr, rng=rng)

    assert lr.shape[-1] == Config.NUM_LR_CHANNELS
    assert hr_target.shape[-1] == Config.NUM_HR_CHANNELS
    assert lr.pixel_scale_arcsec == Config.BAND_VIS.pixel_scale_lr_arcsec
