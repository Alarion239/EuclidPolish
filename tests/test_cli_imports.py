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


def test_fasrc_train_transition_model_script_imports():
    """REGRESSION — a syntax error in the new training script would
    only fire when SLURM actually executed it, wasting a real job
    allocation. Importing it from a unit test gives an immediate
    SyntaxError if anything is malformed."""
    import importlib.util
    import os
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "scripts", "fasrc_train_transition_model.py",
    )
    spec = importlib.util.spec_from_file_location(
        "fasrc_train_transition_model_under_test", path,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # Spot-check that the central function is callable (signature-only,
    # not invoked — invocation would need pair shards on disk).
    assert callable(mod._make_pair_dataset)
    assert callable(mod._probe_image_size)
    assert callable(mod._psf_identity_residual)


def _load_trainer_mod():
    import importlib.util
    import os as _os
    path = _os.path.join(
        _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
        "scripts", "fasrc_train_transition_model.py",
    )
    spec = importlib.util.spec_from_file_location(
        f"fasrc_trainer_probe_{id(path)}", path,
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_probe_image_size_missing_file(tmp_path):
    """REGRESSION — clear error when pair shard doesn't exist.

    Previous behaviour: opaque ``input_train.tfrecord is empty`` even
    though the file was actually absent. Now distinguishes between
    'file missing' (need to run step 3a) and 'file present but empty'
    (step 3a ran but produced 0 records, usually because crop_size
    was too large)."""
    import pytest
    import re
    mod = _load_trainer_mod()
    with pytest.raises(RuntimeError, match=re.compile(
            r"not found.*step 3a", re.DOTALL)):
        mod._probe_image_size(str(tmp_path), "train")


def test_probe_image_size_zero_record_shard(tmp_path):
    """REGRESSION — clear error when pair shard exists but has 0 records.

    This is the exact symptom of the bug we just hit: pair-gen ran
    with --crop-size larger than the synthetic scene side, every
    scene was skipped, the file was opened (creating 0-byte/empty
    output) and the trainer crashed on the empty stream."""
    import pytest
    # Create a 0-byte input_train.tfrecord — what the writer leaves
    # behind when nothing is written.
    (tmp_path / "input_train.tfrecord").write_bytes(b"")
    mod = _load_trainer_mod()
    with pytest.raises(RuntimeError, match="0 bytes"):
        mod._probe_image_size(str(tmp_path), "train")


def test_probe_image_size_short_but_non_empty_record(tmp_path):
    """REGRESSION — clear error when file has non-zero bytes but no
    parseable records (e.g. a header byte stub or a TFRecord whose
    body was truncated mid-write)."""
    import pytest
    # 1 byte that isn't a valid TFRecord. ``read_multiband_skyimages``
    # will fail or return [] depending on the parser; either way we
    # want the trainer to fail loudly and tell the user step 3a's
    # crop_size is most likely the culprit.
    (tmp_path / "input_train.tfrecord").write_bytes(b"\x00")
    mod = _load_trainer_mod()
    with pytest.raises(RuntimeError):
        mod._probe_image_size(str(tmp_path), "train")


def test_fasrc_generate_transition_pairs_script_imports():
    """Same regression guard for the pair-generation script."""
    import importlib.util
    import os
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "scripts", "fasrc_generate_transition_pairs.py",
    )
    spec = importlib.util.spec_from_file_location(
        "fasrc_generate_transition_pairs_under_test", path,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert callable(mod._convolve_pair)
    assert callable(mod._load_hst_psf_on_hr)


def test_fasrc_generate_hst_tfrecords_script_imports():
    """Same guard for the consumer script that loads A_θ or the analytic kernel."""
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
    assert callable(mod._apply_transition)


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
    from euclid_polish.euclid.psf_library import psf_inventory
    inv = psf_inventory(psf_dir="/nonexistent_dir_for_test")
    assert set(inv.keys()) == {b.name for b in Config.BANDS}
    assert all(v is None for v in inv.values())


def test_multiband_pipeline_components_wired():
    """The three pipeline steps can be wired together end-to-end."""
    import numpy as np
    from euclid_polish.config import Config
    from euclid_polish.sky.multiband_generator import (
        MultiBandGeneratorConfig, MultiBandSimulator,
    )
    from euclid_polish.sky.multiband_forward import (
        MultiBandForward, MultiBandForwardConfig,
    )
    from euclid_polish.euclid.psf_library import load_all_band_psfs
    from tests._tiny_catalog import TinyCosmosCatalog

    cat = TinyCosmosCatalog(n_galaxies=200, seed=0)
    sim = MultiBandSimulator(cat, MultiBandGeneratorConfig(image_size=96))
    psfs = load_all_band_psfs(psf_dir="/nonexistent_for_test")
    fwd = MultiBandForward(psfs_by_band=psfs,
                           config=MultiBandForwardConfig(add_noise=False))

    rng = np.random.default_rng(0)
    hr, _ = sim.simulate_field(rng, n_galaxies=3, n_stars=1, n_lenses=0)
    lr, hr_target = fwd.process(hr, rng=rng)

    assert lr.shape[-1] == Config.NUM_LR_CHANNELS
    assert hr_target.shape[-1] == Config.NUM_HR_CHANNELS
    assert lr.pixel_scale_arcsec == Config.BAND_VIS.pixel_scale_lr_arcsec
