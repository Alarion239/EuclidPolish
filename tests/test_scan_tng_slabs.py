"""The TNG slab scan: flat faint light inside the kept circle is flagged,
a dotty particle halo is not."""
from __future__ import annotations

import csv

import numpy as np
from astropy.io import fits

from scripts.scan_tng_slabs import flat_area, main


def _donor(seed: int = 0, slab: bool = False) -> np.ndarray:
    """A bright core and a halo of isolated particles with zeros between them
    (what SKIRT frames look like), optionally with a flat sheet in the halo."""
    rng = np.random.default_rng(seed)
    yy, xx = np.indices((320, 320))
    r = np.hypot(yy - 160, xx - 160)
    image = np.where(r < 30, 1e4 * np.exp(-r / 6.0), 0.0)
    dots = (rng.random(r.shape) < 0.4 * np.exp(-r / 80.0)) & (r >= 20)
    image[dots] += rng.exponential(1.0, int(dots.sum()))
    if slab:
        image[60:110, 170:240] += 0.05
    return image


def test_a_dotty_halo_is_not_flat_and_a_sheet_is():
    clean = flat_area(_donor())
    sheet = flat_area(_donor(slab=True))
    assert clean["flat_area"] < 0.01
    assert sheet["flat_area"] > 0.03          # the sheet covers ~4 % of the kept faint area
    assert abs(sheet["keep_radius_px"] - clean["keep_radius_px"]) < 5


def test_the_scan_ranks_the_sheet_first(tmp_path):
    for subhalo, slab in (("101", False), ("202", True)):
        folder = tmp_path / "atlas" / subhalo
        folder.mkdir(parents=True)
        fits.writeto(folder / f"TNG{subhalo}_O1_Euclid_VIS.fits", _donor(int(subhalo), slab).astype(np.float32))
    out = tmp_path / "scan"
    assert main(["--data-dir", str(tmp_path / "atlas"), "--out-dir", str(out), "--workers", "1", "--sheet", "2"]) == 0
    rows = list(csv.DictReader(open(out / "tng_slab_scan.csv")))
    assert [r["subhalo_id"] for r in rows] == ["202", "101"]
    assert (out / "tng_slab_scan_top.png").exists()
