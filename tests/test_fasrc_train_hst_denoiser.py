"""Tests for ``scripts/fasrc_train_hst_denoiser.py``.

Focused on the script's two failure-prone surfaces:
  * The on-the-fly noise-injection dataset (shape contract + that noise
    is actually present + scales with the configured ranges).
  * The end-to-end ``main()`` happy path on a tiny synthetic pair shard
    — verifies model build + dataset + a few training steps + save.

Long-running performance smoke tests live elsewhere.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

# Make the script importable as a module (it lives under scripts/).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPTS_DIR = os.path.join(_PROJECT_ROOT, "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)


def _load_script():
    import fasrc_train_hst_denoiser as mod
    return mod


def _write_synthetic_input_shard(records_dir, name, n_scenes, side):
    """Drop a minimal ``input_<subset>.tfrecord`` under records_dir."""
    from euclid_polish.config import Config
    from euclid_polish.sky.tfrecord import open_multiband_writer
    from euclid_polish.sky.types import MultiBandSkyImage
    rng = np.random.default_rng(0)
    with open_multiband_writer(name, records_dir=str(records_dir)) as w:
        for i in range(n_scenes):
            # Single-channel VIS-only HR scenes, electron-scale values
            # matching what the real input shards carry.
            data = rng.uniform(0, 500, size=(side, side, 1)).astype(np.float32)
            img = MultiBandSkyImage(
                data=data,
                pixel_scale_arcsec=Config.DEFAULT_PIXEL_SCALE,
                band_names=("VIS",), is_clean=True,
            )
            w.write(img, index=i)


class TestDenoiserDataset:

    def test_yields_correct_shape_and_dtype(self, tmp_path):
        mod = _load_script()
        _write_synthetic_input_shard(tmp_path, "input_train", n_scenes=4, side=32)
        ds = mod._make_denoiser_dataset(
            str(tmp_path), "train",
            batch_size=2, seed=0, shuffle=False,
            alpha_min=0.5, alpha_max=1.0,
            sigma_floor_min=8.0, sigma_floor_max=22.0,
        )
        for noisy_b, clean_b in ds.take(1):
            assert tuple(noisy_b.shape) == (2, 32, 32, 1)
            assert tuple(clean_b.shape) == (2, 32, 32, 1)
            assert noisy_b.dtype == tf.float32
            assert clean_b.dtype == tf.float32
            break

    def test_input_actually_differs_from_target(self, tmp_path):
        """Noise has to be added — input == clean would mean the
        denoiser has nothing to learn. Sanity check the dataset emits
        a measurable difference."""
        mod = _load_script()
        _write_synthetic_input_shard(tmp_path, "input_train", n_scenes=8, side=32)
        ds = mod._make_denoiser_dataset(
            str(tmp_path), "train",
            batch_size=4, seed=0, shuffle=False,
            alpha_min=0.5, alpha_max=1.0,
            sigma_floor_min=8.0, sigma_floor_max=22.0,
        )
        for noisy_b, clean_b in ds.take(1):
            diff = (noisy_b - clean_b).numpy()
            assert diff.std() > 1.0, (
                f"noise σ too small ({diff.std():.3f}); the dataset "
                f"should add visible HLSP-style noise to the input."
            )
            break

    def test_noise_sigma_scales_with_floor_range(self, tmp_path):
        """Higher sigma_floor → larger residual std. Lock both alpha
        endpoints to 0 so only the sky floor contributes — gives a
        clean linear relationship to measure."""
        mod = _load_script()
        _write_synthetic_input_shard(tmp_path, "input_train", n_scenes=32, side=32)
        # Low floor band
        ds_lo = mod._make_denoiser_dataset(
            str(tmp_path), "train",
            batch_size=8, seed=0, shuffle=False,
            alpha_min=0.0, alpha_max=0.0,
            sigma_floor_min=2.0, sigma_floor_max=2.0,
        )
        # High floor band
        ds_hi = mod._make_denoiser_dataset(
            str(tmp_path), "train",
            batch_size=8, seed=0, shuffle=False,
            alpha_min=0.0, alpha_max=0.0,
            sigma_floor_min=50.0, sigma_floor_max=50.0,
        )
        diffs_lo, diffs_hi = [], []
        for nb, cb in ds_lo.take(4):
            diffs_lo.append((nb - cb).numpy())
        for nb, cb in ds_hi.take(4):
            diffs_hi.append((nb - cb).numpy())
        sigma_lo = float(np.concatenate([d.ravel() for d in diffs_lo]).std())
        sigma_hi = float(np.concatenate([d.ravel() for d in diffs_hi]).std())
        # Ratio should be ~ 50/2 = 25 within sampling tolerance.
        assert 15 < sigma_hi / sigma_lo < 35, (
            f"σ_hi/σ_lo ≈ 25 expected, got {sigma_hi/sigma_lo:.2f} "
            f"(σ_lo={sigma_lo:.2f}, σ_hi={sigma_hi:.2f})"
        )

    def test_404_when_input_shard_missing(self, tmp_path):
        """No input_train.tfrecord → loud RuntimeError, not silent
        empty dataset that would confuse the trainer."""
        mod = _load_script()
        with pytest.raises(RuntimeError, match="not found"):
            mod._make_denoiser_dataset(
                str(tmp_path), "train",
                batch_size=2, seed=0, shuffle=False,
                alpha_min=0.5, alpha_max=1.0,
                sigma_floor_min=8.0, sigma_floor_max=22.0,
            )


class TestMainDryRun:

    def test_dry_run_succeeds_with_minimal_inputs(self, tmp_path, monkeypatch):
        mod = _load_script()
        synth_dir = tmp_path / "synth"
        synth_dir.mkdir()
        _write_synthetic_input_shard(synth_dir, "input_train", n_scenes=2, side=32)
        _write_synthetic_input_shard(synth_dir, "input_validate", n_scenes=1, side=32)

        # Write outputs to tmp_path, NOT the real DATA_DIR.
        out_weights = tmp_path / "hst_denoiser.weights.h5"
        out_summary = tmp_path / "hst_denoiser_summary.json"

        argv = [
            "fasrc_train_hst_denoiser.py",
            "--pairs-dir",       str(synth_dir),
            "--output",          str(out_weights),
            "--summary-output",  str(out_summary),
            "--steps",           "2",
            "--batch-size",      "1",
            "--validate-every",  "100",
            "--channels",        "4",
            "--n-inner-layers",  "1",
            "--kernel-size",     "3",   # cheap for the test
            "--max-params",      "10000",
            "--dry-run",
        ]
        monkeypatch.setattr("sys.argv", argv)
        rc = mod.main()
        assert rc == 0
        # Dry-run shouldn't write weights.
        assert not out_weights.exists()


class TestMainEndToEnd:

    def test_main_trains_and_saves(self, tmp_path, monkeypatch):
        """Few real training steps + checkpoint save. Loss numbers
        aren't tested (the model is tiny + 2 steps isn't enough to
        converge); just that the script completes and produces both
        the weights and summary JSON."""
        mod = _load_script()
        synth_dir = tmp_path / "synth"
        synth_dir.mkdir()
        _write_synthetic_input_shard(synth_dir, "input_train", n_scenes=4, side=32)
        _write_synthetic_input_shard(synth_dir, "input_validate", n_scenes=2, side=32)

        out_weights = tmp_path / "hst_denoiser.weights.h5"
        out_summary = tmp_path / "hst_denoiser_summary.json"

        argv = [
            "fasrc_train_hst_denoiser.py",
            "--pairs-dir",       str(synth_dir),
            "--output",          str(out_weights),
            "--summary-output",  str(out_summary),
            "--steps",           "4",
            "--batch-size",      "1",
            "--validate-every",  "2",
            "--channels",        "4",
            "--n-inner-layers",  "1",
            "--kernel-size",     "3",
            "--max-params",      "10000",
        ]
        monkeypatch.setattr("sys.argv", argv)
        rc = mod.main()
        assert rc == 0
        assert out_weights.exists(), "weights file should be saved"
        assert out_summary.exists(), "summary JSON should be saved"

        # Summary JSON shape sanity.
        import json
        with open(out_summary) as f:
            s = json.load(f)
        for k in ("model_path", "params", "channels", "n_inner_layers",
                  "kernel_size", "noise", "log"):
            assert k in s, f"summary missing key {k!r}"
        for nk in ("alpha_min", "alpha_max",
                   "sigma_floor_min", "sigma_floor_max"):
            assert nk in s["noise"], f"summary['noise'] missing {nk!r}"
