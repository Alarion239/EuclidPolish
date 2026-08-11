"""Raw HR and blurred BHR FITS persist 4-band (heavy deps monkeypatched)."""

from __future__ import annotations

import numpy as np
from astropy.io import fits

from euclid_polish.config import Config
from euclid_polish.eval import synthetic_runner as sr
from euclid_polish.eval.catalog_runner import EVAL_HR_SIZE, EVAL_LR_SIZE


class _Img:
    def __init__(self, index, data):
        self.index = index
        self.data = data


class _FakeEnsemble:
    """Ensemble-of-1 stub: member_arrays returns a fixed (1,H,W,C) stack."""

    def __init__(self, sr_field):
        self._sr = np.asarray(sr_field, np.float32)
        self.calls = 0

    n_members = 1
    member_labels = ["00·psnr"]

    def member_arrays(self, lr_array):
        self.calls += 1
        return self._sr[None, ...]


def test_hr_fits_written_four_band(tmp_path, monkeypatch):
    # dirty_* records are LR half-grid (64²×4); hr_* are HR-grid (128²×4) — large
    # enough to crop the canonical 53² LR / 106² SR·HR stamp centered at (64,64).
    def fake_read(path, num_images=0):
        if "dirty" in str(path):
            return [_Img(0, np.zeros((64, 64, 4), np.float32))]
        target = np.zeros((128, 128, 4), np.float32)
        target[64, 64, 0] = 1.0
        return [_Img(0, target)]

    monkeypatch.setattr("euclid_polish.image.tfio.read_images",
                        fake_read)
    monkeypatch.setattr("euclid_polish.sky.generation.source_catalog.read_sources",
                        lambda p: {0: [{"type": "lens", "x_pix": 64.0,
                                        "y_pix": 64.0, "flux_vis_e": 1.0}]})

    out_dir = str(tmp_path / "eval")
    res = sr.run_synthetic_eval(
        out_dir, n=1, model=_FakeEnsemble(np.ones((128, 128, 4), np.float32)),
        records_dir=str(tmp_path),
        on_progress=lambda *a: None, log=lambda *a: None)
    assert res["n_ok"] == 1

    base = f"{out_dir}/syn-lens_0000_0"
    # Canonical geometry: LR EVAL_LR_SIZE², SR/HR EVAL_HR_SIZE², all 4-band.
    with fits.open(f"{base}/HR.fits") as hdul:
        raw_hr = np.asarray(hdul[0].data)
        assert raw_hr.shape == (4, EVAL_HR_SIZE, EVAL_HR_SIZE)
        assert "VIS" in hdul[0].header.get("BANDS", "")
    with fits.open(f"{base}/BHR.fits") as hdul:
        blurred_hr = np.asarray(hdul[0].data)
        assert blurred_hr.shape == (4, EVAL_HR_SIZE, EVAL_HR_SIZE)
        assert hdul[0].header["TARGFWH"] == Config.TARGET_PSF_FWHM_ARCSEC
    center = EVAL_HR_SIZE // 2
    assert raw_hr[0, center, center] == 1.0
    assert blurred_hr[0, center, center] < raw_hr[0, center, center]
    with fits.open(f"{base}/SR.fits") as hdul:
        assert np.asarray(hdul[0].data).shape == (4, EVAL_HR_SIZE, EVAL_HR_SIZE)
    with fits.open(f"{base}/original_stack.fits") as hdul:
        assert np.asarray(hdul[0].data).shape == (4, EVAL_LR_SIZE, EVAL_LR_SIZE)


def test_multiple_galaxies_extracted_per_field(tmp_path, monkeypatch):
    """A single crowded field yields several syn-gal stamps (brightest first),
    reconstructing the field's SR only once, to meet the target."""
    def fake_read(path, num_images=0):
        if "dirty" in str(path):
            return [_Img(0, np.zeros((64, 64, 4), np.float32))]
        return [_Img(0, np.ones((128, 128, 4), np.float32))]

    monkeypatch.setattr("euclid_polish.image.tfio.read_images", fake_read)
    # One field, three galaxies at distinct in-window positions and fluxes.
    monkeypatch.setattr(
        "euclid_polish.sky.generation.source_catalog.read_sources",
        lambda p: {0: [
            {"type": "galaxy", "x_pix": 64.0, "y_pix": 64.0, "flux_vis_e": 10.0},
            {"type": "galaxy", "x_pix": 60.0, "y_pix": 70.0, "flux_vis_e": 100.0},
            {"type": "galaxy", "x_pix": 70.0, "y_pix": 60.0, "flux_vis_e": 50.0},
        ]})
    fake = _FakeEnsemble(np.ones((128, 128, 4), np.float32))

    out_dir = str(tmp_path / "eval")
    res = sr.run_synthetic_eval(
        out_dir, n=3, model=fake, records_dir=str(tmp_path),
        on_progress=lambda *a: None, log=lambda *a: None)

    assert res["n_ok"] == 3                       # all three drawn from the one field
    assert fake.calls == 1                        # SR reconstructed once per field
    for rank in (0, 1, 2):
        assert (tmp_path / "eval" / f"syn-gal_0000_{rank}" / "SR.fits").exists()


def test_reuses_cached_ensemble_cubes(tmp_path, monkeypatch):
    """When ensemble cube cache is present, sr_from_model is never called."""
    import json

    from euclid_polish.config import Config
    from euclid_polish.eval import synthetic_runner

    # Same LR/HR geometry as the other tests: LR 64²×4, HR 128²×4.
    # eval_subset returns "validate" because no dirty_test.tfrecord exists.
    subset = "validate"
    field_indices = [0]
    hr_field_shape = (128, 128, 4)

    def fake_read(path, num_images=0):
        if "dirty" in str(path):
            return [_Img(0, np.zeros((64, 64, 4), np.float32))]
        return [_Img(0, np.ones(hr_field_shape, np.float32))]

    monkeypatch.setattr("euclid_polish.image.tfio.read_images", fake_read)
    monkeypatch.setattr(
        "euclid_polish.sky.generation.source_catalog.read_sources",
        lambda p: {0: [{"type": "lens", "x_pix": 64.0,
                        "y_pix": 64.0, "flux_vis_e": 1.0}]})

    # Registry-active members whose labels the cache manifest must match:
    # member_00..03 with a bare checkpoint file each (psnr track only).
    ckpt_root = tmp_path / "ckpt"
    n_members = 4
    for i in range(n_members):
        d = ckpt_root / "ensemble" / f"member_{i:02d}"
        d.mkdir(parents=True)
        (d / "checkpoint").write_text("x")
        (d / "origin.json").write_text(json.dumps({"starless": True}))
    monkeypatch.setattr(Config, "DEFAULT_CHECKPOINT_DIR",
                        str(ckpt_root / "wdsr"))
    labels = [f"{i:02d}·psnr" for i in range(n_members)]

    # Stage the ensemble cube cache that synthetic_runner should reuse.
    vis_dir = tmp_path / "vis"
    cubes_dir = vis_dir / "ensemble" / "starless" / "cubes"
    cubes_dir.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for idx in field_indices:
        for i in range(n_members):
            np.save(cubes_dir / f"member{i}_{idx:05d}.npy",
                    rng.normal(10, 1, hr_field_shape).astype(np.float32))
    with open(cubes_dir / "viz_index.json", "w") as f:
        json.dump({"subset": subset, "indices": list(field_indices),
                   "pca_n": 3, "pca_amps": {},
                   "member_labels": labels}, f)
    monkeypatch.setattr(Config, "VIS_DIR", str(vis_dir))

    def _boom(*a, **k):
        raise AssertionError("sr_from_model called — cache reuse failed")
    monkeypatch.setattr("euclid_polish.eval.ensemble_infer.sr_from_model", _boom)

    out = synthetic_runner.run_synthetic_eval(
        str(tmp_path / "out"), n=1, model=object(),
        records_dir=str(tmp_path), seed=0,
        on_progress=lambda *a: None, log=lambda *a: None)

    ok_rows = [r for r in out["rows"] if r["ok"]]
    assert ok_rows, "no synthetic cutouts produced from cache"
    sub0 = ok_rows[0]["out_subdir"]
    for name in ("SR.fits", "std.fits", "pca0.fits", "members.json"):
        assert (tmp_path / "out" / sub0 / name).is_file()
