"""Synthetic real-data stores for the C9 tests (real tiles, experiments, sky
atlas): small FITS/NPY/JSON fixtures written under ``tmp_path`` and the
``Config`` paths pointed at them. Nothing here touches the network."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from euclid_polish.config import Config
from euclid_polish.eval.spatial_gate import MIX_LINEAR, SpatialGateCombiner, save_spatial_gate
from euclid_polish.eval.spatial_gate_fit import init_params
from euclid_polish.web.helpers import (
    experiments,
    jwst_euclid,
    model_catalog,
    real_tiles,
    sky_atlas,
)

NEXUS_RA, NEXUS_DEC = 268.4625, 65.19917
LABELS = ["1·psnr", "2·psnr", "3·psnr"]


def wcs_header(ra: float, dec: float, scale_arcsec: float, shape: tuple[int, int],
               *, bunit: str | None = None) -> fits.Header:
    """North-up TAN header whose grid centre is ``(ra, dec)``."""
    header = fits.Header()
    header["CTYPE1"], header["CTYPE2"] = "RA---TAN", "DEC--TAN"
    header["CRVAL1"], header["CRVAL2"] = ra, dec
    header["CRPIX1"], header["CRPIX2"] = (shape[1] + 1) / 2.0, (shape[0] + 1) / 2.0
    header["CD1_1"], header["CD1_2"] = -scale_arcsec / 3600.0, 0.0
    header["CD2_1"], header["CD2_2"] = 0.0, scale_arcsec / 3600.0
    if bunit:
        header["BUNIT"] = bunit
    return header


def scene(side: int, *, seed: int = 0, amplitude: float = 5.0e4) -> np.ndarray:
    """``(side, side, 4)`` electrons: background + noise + one Gaussian star."""
    rng = np.random.default_rng(seed)
    yy, xx = np.indices((side, side), dtype=np.float64)
    star = amplitude * np.exp(-((xx - side / 2) ** 2 + (yy - side / 2) ** 2) / (2 * 2.0 ** 2))
    plane = 20.0 + rng.normal(0, 5.0, (side, side)) + star
    return np.repeat(plane[..., None], 4, axis=-1).astype(np.float32)


def tile_position(dx: float, dy: float, side: int = 40) -> tuple[float, float]:
    """Sky position ``(dx, dy)`` tile widths east/north of the first NEXUS tile."""
    step = side * 0.1 / 3600.0
    return (NEXUS_RA + dx * step / np.cos(np.radians(NEXUS_DEC)), NEXUS_DEC + dy * step)


def make_nexus_field(n_tiles: int = 2, side: int = 40, *,
                     offsets: list[tuple[int, int]] | None = None) -> str:
    """A NEXUS field manifest with ``n_tiles`` four-band LR tiles + JWST, laid
    along RA (or at ``offsets`` in tile widths)."""
    identifier = jwst_euclid.nexus_field_id("F200W")
    root = jwst_euclid.nexus_field_root() / identifier
    (root / "tiles").mkdir(parents=True, exist_ok=True)
    tiles = []
    offsets = offsets or [(index, 0) for index in range(n_tiles)]
    for index, (dx, dy) in enumerate(offsets):
        ra, dec = tile_position(dx, dy, side)
        header = wcs_header(ra, dec, 0.1, (side, side), bunit="electron")
        lr = scene(side, seed=index)
        files = {}
        for band_index, band in enumerate(Config.LR_INPUT_BAND_NAMES):
            name = f"tiles/euclid_{band.lower()}_{index:04d}.fits"
            fits.PrimaryHDU(lr[..., band_index], header=header).writeto(root / name)
            files[band] = name
        fits.PrimaryHDU(np.moveaxis(lr, -1, 0), header=header).writeto(
            root / f"tiles/euclid_lr_vis_y_j_h_{index:04d}.fits")
        jwst_side = side * 10 // 3
        fits.PrimaryHDU(np.full((jwst_side, jwst_side), 0.5, np.float32),
                        header=wcs_header(ra, dec, 0.03, (jwst_side, jwst_side),
                                          bunit="MJy/sr")).writeto(
            root / f"tiles/jwst_{index:04d}.fits")
        tiles.append({
            "index": index, "source_index": index, "ra_deg": ra, "dec_deg": dec,
            "euclid_file": files["VIS"], "euclid_files": files,
            "lr_file": f"tiles/euclid_lr_vis_y_j_h_{index:04d}.fits",
            "jwst_file": f"tiles/jwst_{index:04d}.fits",
        })
    (root / "manifest.json").write_text(json.dumps({
        "field_id": identifier, "filter": "F200W", "count": len(tiles), "tiles": tiles,
    }), encoding="utf-8")
    return identifier


def make_pair(side: int = 30, *, identifier: str = "mast-102158584-jw01-30as") -> str:
    """A saved JWST × Euclid pair with its four-band LR input already built
    (``starfull_inference/euclid_lr_vis_y_j_h.fits``), so inference never
    downloads."""
    root = jwst_euclid.pair_root() / identifier
    (root / "starfull_inference").mkdir(parents=True, exist_ok=True)
    ra, dec = NEXUS_RA + 0.01, NEXUS_DEC
    header = wcs_header(ra, dec, 0.1, (side, side), bunit="electron")
    lr = scene(side, seed=11)
    fits.PrimaryHDU(lr[..., 0], header=header).writeto(root / "euclid_vis.fits")
    jwst_side = side * 10 // 3
    fits.PrimaryHDU(np.full((jwst_side, jwst_side), 0.5, np.float32),
                    header=wcs_header(ra, dec, 0.03, (jwst_side, jwst_side),
                                      bunit="MJy/sr")).writeto(root / "jwst_native.fits")
    for name in ("euclid_vis.png", "jwst_native.png"):
        (root / name).write_bytes(b"png")
    fits.PrimaryHDU(np.moveaxis(lr, -1, 0), header=header).writeto(
        root / "starfull_inference" / "euclid_lr_vis_y_j_h.fits")
    (root / "manifest.json").write_text(json.dumps({
        "version": 3, "field_id": identifier, "target_name": "Test pair",
        "ra_deg": ra, "dec_deg": dec, "size_arcsec": side * 0.1,
        "jwst_filters": "F200W",
        "jwst_bands": [{"key": "jwst0", "filter": "F200W", "file": "jwst_native.fits"}],
        "files": {"euclid": "euclid_vis.fits", "jwst_native": "jwst_native.fits",
                  "euclid_png": "euclid_vis.png", "jwst_png": "jwst_native.png"},
        "lr_input": {"file": "starfull_inference/euclid_lr_vis_y_j_h.fits"},
    }), encoding="utf-8")
    return identifier


def make_eval_store(side: int = 20) -> str:
    root = Path(Config.EVAL_RESULTS_DIR)
    sub = "102158584_OBJ1"
    (root / sub).mkdir(parents=True, exist_ok=True)
    ra, dec = 268.30, 65.10
    fits.PrimaryHDU(np.moveaxis(scene(side, seed=7), -1, 0),
                    header=wcs_header(ra, dec, 0.1, (side, side), bunit="electron")).writeto(
        root / sub / "original_stack.fits")
    with (root / "manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "id", "ra", "dec", "grade", "ok", "error", "out_subdir", "flux_ratio_sr_over_lr"])
        writer.writeheader()
        writer.writerow({"id": "OBJ1", "ra": ra, "dec": dec, "grade": "A", "ok": "True",
                         "error": "", "out_subdir": sub, "flux_ratio_sr_over_lr": "0.7"})
        writer.writerow({"id": "SYN", "ra": "", "dec": "", "grade": "syn", "ok": "True",
                         "error": "", "out_subdir": "syn_0", "flux_ratio_sr_over_lr": ""})
    return sub


def make_poster(directory: Path, side: int = 32) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    primary = fits.PrimaryHDU()
    primary.header["RA"], primary.header["DEC"] = 273.2308875, 68.3636556
    primary.header["PIXSCALE"] = 0.1
    primary.header["COMB_KIND"] = "raw_incremental_minmeanmax_rbf"
    primary.header["N_MEMBER"] = 20
    lr = scene(side, seed=3)
    hdus = [primary]
    for index, band in enumerate(Config.LR_INPUT_BAND_NAMES):
        hdus.append(fits.ImageHDU(lr[..., index], name=f"LR_{band}"))
    for index, band in enumerate(Config.LR_INPUT_BAND_NAMES):
        hdus.append(fits.ImageHDU(np.kron(lr[..., index], np.ones((2, 2))) / 4.0,
                                  name=f"SR_{band}"))
    path = directory / "target_181255_test_results.fits"
    fits.HDUList(hdus).writeto(path)
    return path


def make_real_field(side: int = 32, grid: int = 2) -> str:
    field_id = "ra0267.42290_decp064.88730"
    root = Path(Config.EUCLID_INFERENCE_DIR) / "real_fields" / field_id
    (root / "cubes").mkdir(parents=True, exist_ok=True)
    size = side * grid
    fits.PrimaryHDU(np.zeros((4, size, size), np.float32),
                    header=wcs_header(267.4229, 64.8873, 0.1, (size, size))).writeto(
        root / "original_stack.fits")
    for index in range(grid * grid):
        np.save(root / "cubes" / f"lr_{index:03d}.npy", scene(side, seed=index))
    (root / "manifest.json").write_text(json.dumps({
        "field_id": field_id, "ra": 267.4229, "dec": 64.8873, "count": grid * grid,
        "tile_size": side, "grid_side": grid, "field_size": size,
        "member_labels": ["145·psnr"], "combiner_kinds": [],
    }), encoding="utf-8")
    return field_id


def point_store(tmp_path: Path, monkeypatch) -> dict:
    """Point every real-data ``Config`` path at ``tmp_path`` and reset caches."""
    data = tmp_path / "data"
    monkeypatch.setattr(Config, "DATA_DIR", str(data))
    monkeypatch.setattr(Config, "EUCLID_INFERENCE_DIR", str(data / "euclid_inference"))
    monkeypatch.setattr(Config, "EVAL_RESULTS_DIR", str(data / "eval_results"))
    monkeypatch.setattr(Config, "EVAL_CATALOG_DIR", str(data / "eval_catalogs"))
    monkeypatch.setattr(Config, "EUCLID_SKY_DIR", str(data / "euclid_sky"))
    monkeypatch.setattr(Config, "EUCLID_PSF_DIR", str(data / "euclid_psf"))
    monkeypatch.setattr(real_tiles, "poster_root", lambda: tmp_path / "poster")
    monkeypatch.setattr(sky_atlas, "stars_path", lambda: None)
    monkeypatch.setattr(sky_atlas, "_cached_psf_clusters_json", lambda: None)
    # Disk-space guards must not depend on the machine running the tests.
    monkeypatch.setattr(experiments, "free_bytes", lambda _path: 10 ** 15)
    real_tiles.invalidate()
    sky_atlas.invalidate()
    jwst_euclid._NEXUS_POLYGONS.clear()
    return {"data": data, "poster": tmp_path / "poster"}


def uniform_gate(labels) -> SpatialGateCombiner:
    params = init_params(len(labels), 4, 8, False, [0, 0, 0, 0], seed=1)
    params["b_out"] = np.zeros_like(params["b_out"])
    return SpatialGateCombiner(list(labels), params, width=8, use_lr=False,
                               mix_space=MIX_LINEAR, fit_meta={"loss": "test"})


def stub_regime(tmp_path: Path, monkeypatch) -> dict:
    """Three fake STARFULL members, a production gate over them, one variant."""
    root = tmp_path / "regime" / "starfull"
    save_spatial_gate(uniform_gate(LABELS), str(root / "spatial_gate_combiner"))
    save_spatial_gate(uniform_gate(LABELS[:2]), str(root / "spatial_gate_two"))
    fingerprints = {label: f"ckpt-{i}:10:100" for i, label in enumerate(LABELS)}
    monkeypatch.setattr(model_catalog, "regime_dir", lambda: root)
    monkeypatch.setattr(model_catalog, "active_member_labels", lambda: list(LABELS))
    monkeypatch.setattr(model_catalog, "member_fingerprints",
                        lambda labels: {label: fingerprints.get(label) for label in labels})
    return {"root": root, "fingerprints": fingerprints}


class StubRunner:
    """Member ``k`` predicts the flux-conserving 2× upsample × ``k``."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def predict(self, lr, label):
        self.calls.append(label)
        k = float(label.split("·")[0])
        return (np.kron(np.asarray(lr, np.float32), np.ones((2, 2, 1))) / 4.0 * k).astype(np.float32)


def fake_cutout_writer(calls: list | None = None, *, magzero=None, nisp_scale: float = 0.3,
                       pad: int = 1):
    """A ``fetch_q1_cutout`` stand-in writing an archive-like ADU/s cutout
    (CD TAN grid centred on the request, side rounded up by ``pad`` like the
    service; NISP at ``nisp_scale`` ″/px — 0.1 for MER-grid mosaics)."""
    zeropoints = magzero or {"VIS": 24.6, "Y_E": 29.8, "J_E": 30.0, "H_E": 29.9}

    def fetch(*, ra, dec, band_name, output_file, cutout_size_vis_pixels, **_kwargs):
        if calls is not None:
            calls.append(band_name)
        scale = 0.1 if band_name == "VIS" else nisp_scale
        side = int(round(cutout_size_vis_pixels * 0.1 / scale)) + pad
        header = wcs_header(ra, dec, scale, (side, side))
        header["MAGZERO"] = zeropoints[band_name]
        header["BUNIT"] = "ADU/s"
        fits.PrimaryHDU(np.ones((side, side), np.float32), header=header).writeto(
            output_file, overwrite=True)
        return True, None

    return fetch


def served_wcs(header_json: str) -> WCS:
    return WCS(fits.Header(json.loads(header_json)))
