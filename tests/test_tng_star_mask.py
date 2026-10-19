"""The star mask: SKIRT's dusty frames carry dust-scattered light far beyond
the stars (a flat box filling the simulation volume around massive galaxies);
frames keep their pixels only where a dust-free ``_nodust`` twin has light."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from euclid_polish.config import Config
from euclid_polish.tng.atlas import _scan_complete_galaxies
from euclid_polish.tng.radius_manifest import ALGORITHM_VERSION, build_manifest
from euclid_polish.tng.types import TNGView, dust_free_twin

SIZE = 96
BANDS = ("VIS", "Y", "J", "H")


def _write_frame(path: Path, data: np.ndarray) -> None:
    hdu = fits.PrimaryHDU(np.asarray(data, dtype=">f4"))
    hdu.header["BUNIT"] = "MJy/sr"
    for key, value in (("CDELT1", 100.0), ("CUNIT1", "pc"), ("CDELT2", 100.0), ("CUNIT2", "pc")):
        hdu.header[key] = value
    hdu.writeto(path, overwrite=True)


def _stars() -> np.ndarray:
    """A disc of stars that ends at r = 20 px (exact zeros beyond, as in the
    dust-free frames)."""
    yy, xx = np.mgrid[:SIZE, :SIZE]
    r = np.hypot(yy - SIZE / 2, xx - SIZE / 2)
    return np.where(r < 20, np.exp(-r / 4.0), 0.0)


def _galaxy(root: Path, gid: str = "42", twins: bool = True, h_only_star: bool = False,
            box: float = 1e-3) -> Path:
    """Dusty frames = attenuated stars + a flat box of scattered light that
    fills the frame; dust-free twins = the stars alone."""
    folder = root / gid
    folder.mkdir(parents=True)
    stars = _stars()
    for orientation in range(1, 6):
        for b, band in enumerate(BANDS):
            dusty = 0.8 * stars * (b + 1) + box
            _write_frame(folder / f"TNG{gid}_O{orientation}_Euclid_{band}.fits", dusty)
            if twins:
                free = stars * (b + 1)
                if h_only_star and band == "H":
                    free = free.copy()
                    free[80, 80] = 1.0               # a star only the H twin sees
                _write_frame(folder / f"TNG{gid}_O{orientation}_Euclid_{band}_nodust.fits", free)
    (folder / Config.Tng.DONE_MARKER).touch()
    return folder


def _view(folder: Path, gid: str = "42") -> TNGView:
    return TNGView(galaxy_dir=folder, subhalo_id=gid, orientation=1, native_re_px=10.0)


def test_the_twin_of_a_dusty_frame():
    assert dust_free_twin(Path("/a/TNG42_O3_Euclid_VIS.fits")) == Path("/a/TNG42_O3_Euclid_VIS_nodust.fits")


def test_frames_keep_their_light_only_where_the_dust_free_twins_have_stars(tmp_path):
    image = _view(_galaxy(tmp_path)).load_surface_brightness().data
    stars = _stars() > 0
    assert np.all(image[~stars] == 0.0)                      # the box is gone in every band
    for b in range(4):
        np.testing.assert_allclose(image[..., b][stars], (0.8 * _stars() * (b + 1) + 1e-3)[stars], rtol=1e-6)


def test_one_mask_for_all_bands(tmp_path):
    image = _view(_galaxy(tmp_path, h_only_star=True)).load_surface_brightness().data
    assert np.all(image[80, 80] > 0)                         # kept in VIS too, not just H


def test_without_twins_frames_are_unmasked_unless_required(tmp_path, monkeypatch):
    folder = _galaxy(tmp_path, twins=False)
    image = _view(folder).load_surface_brightness().data
    assert np.all(image > 0)                                 # unmasked: the box is still there
    assert [g.subhalo_id for g in _scan_complete_galaxies(tmp_path)] == ["42"]
    monkeypatch.setattr(Config.Tng, "REQUIRE_DUST_FREE", True)
    with pytest.raises(FileNotFoundError):
        _view(folder).load_surface_brightness()
    assert _scan_complete_galaxies(tmp_path) == ()


def test_the_cache_identity_covers_the_twins(tmp_path):
    with_twins = _view(_galaxy(tmp_path / "a")).file_identity()
    without = _view(_galaxy(tmp_path / "b", twins=False)).file_identity()
    assert len(with_twins) == 8 and len(without) == 4


def test_the_radius_is_measured_on_the_masked_frame(tmp_path):
    properties = tmp_path / "properties.csv"
    properties.write_text("id,sfr,mass_stars,m_halo,reff\n42,1,1e10,1e12,2\n43,1,1e10,1e12,2\n")
    # A box holding ~30 % of the light, as around subhalo 117266.
    _galaxy(tmp_path / "atlas", "42", twins=True, box=4e-3)
    _galaxy(tmp_path / "atlas", "43", twins=False, box=4e-3)
    report = build_manifest(str(tmp_path / "atlas"), properties_path=str(properties),
                            output_path=str(tmp_path / "manifest.json"))
    assert report["algorithm_version"] == ALGORITHM_VERSION
    radius = {(e["subhalo_id"], e["orientation"]): e["native_re_px"] for e in report["entries"]}
    # The box fills the frame: unmasked, it drags the half-light radius far out.
    assert radius[("42", 1)] < 0.8 * radius[("43", 1)]
