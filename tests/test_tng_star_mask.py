"""The star mask: SKIRT's dusty frames carry dust-scattered light far beyond
the stars (a flat box filling the simulation volume around massive galaxies);
with ``Config.Tng.STAR_MASK`` frames keep their dusty light where a dust-free
``_nodust`` twin has light, and only a smooth glow of the scattered light just
around it. Off (the default since 2026-10-20) the dusty frames are used whole,
and the masked renderer identity is unchanged when it is switched back on."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from scipy.ndimage import distance_transform_edt, gaussian_filter

from euclid_polish.config import Config
from euclid_polish.tng import renderer
from euclid_polish.tng._image import GLOW_FADE_REACH, star_light_with_glow, start_view_read
from euclid_polish.tng.atlas import _scan_complete_galaxies
from euclid_polish.tng.radius_manifest import algorithm_version, build_manifest
from euclid_polish.tng.types import TNGView, dust_free_twin

SIZE = 96
BANDS = ("VIS", "Y", "J", "H")


@pytest.fixture(autouse=True)
def _star_mask_on(monkeypatch):
    """These tests are about the mask; the ones about whole frames switch it off."""
    monkeypatch.setattr(Config.Tng, "STAR_MASK", True)


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


def _radius() -> np.ndarray:
    yy, xx = np.mgrid[:SIZE, :SIZE]
    return np.hypot(yy - SIZE / 2, xx - SIZE / 2)


def test_stars_keep_their_dusty_light_and_the_box_becomes_a_thin_glow(tmp_path):
    image = _view(_galaxy(tmp_path)).load_surface_brightness().data
    stars = _stars() > 0
    for b in range(4):                                        # the stars are untouched in every band
        np.testing.assert_allclose(image[..., b][stars], (0.8 * _stars() * (b + 1) + 1e-3)[stars], rtol=1e-6)
    r = _radius()
    rim, far = image[..., 0][(r > 21) & (r < 23)], image[..., 0][r > 40]
    assert 0.0 < rim.mean() < 1e-3                            # a glow just outside the stars, below the box
    assert far.max() < 1e-3 * np.exp(-3)                      # gone a few fade lengths (0.5 kpc) out


def test_gaps_between_stars_get_the_full_glow_and_lone_stars_none():
    r = _radius()
    stars = (r < 20) & ~((r > 10) & (r < 11.5))              # a ring of gaps inside the galaxy
    stars[88, 88] = True                                       # a lone halo star far outside
    data = (np.where(stars, 1.0, 0.0) + 1e-3)[..., None].astype(np.float32)
    out = star_light_with_glow(data, stars, pixel_scale_pc=100.0)[..., 0]
    gap = ~stars & (r > 10) & (r < 11.5)
    np.testing.assert_allclose(out[gap], 1e-3, rtol=0.05)    # inside the outline: the full (blurred) glow
    assert out[88, 88] == pytest.approx(1.0 + 1e-3)           # the lone star keeps its light ...
    assert out[86:91, 86:91][~stars[86:91, 86:91]].max() < 1e-3 * np.exp(-6)   # ... and grows no glow


def test_the_glow_matches_a_plain_gaussian_blur():
    rng = np.random.default_rng(0)
    r = _radius()
    stars = r < 15
    sky = (1e-3 * (1 + np.linspace(0, 1, SIZE))[None, :] * (1 + 0.3 * rng.random((SIZE, SIZE)))).astype(np.float32)
    data = np.where(stars, 1.0, 0.0)[..., None].astype(np.float32) + sky[..., None]
    out = star_light_with_glow(data, stars, pixel_scale_pc=100.0)[..., 0]
    off = (~stars).astype(float)
    body = gaussian_filter(stars.astype(float), 10.0) > 0.5
    fade = np.exp(-distance_transform_edt(~body) / 5.0)
    glow = gaussian_filter(sky * off, 30.0, mode="constant") / gaussian_filter(off, 30.0, mode="constant")
    near = ~stars & (fade > 1e-2)
    np.testing.assert_allclose(out[near], (glow * fade)[near], rtol=0.03)


def test_the_glow_ends_at_zero_before_where_it_is_computed_stops():
    """No faint rectangle: the fade reaches zero at GLOW_FADE_REACH fade
    lengths, inside the region the glow is computed in (434357, 2026-10-20)."""
    size = 400
    yy, xx = np.mgrid[:size, :size]
    stars = np.hypot(yy - size / 2, xx - size / 2) < 20
    data = (np.where(stars, 1.0, 0.0) + 1e-3)[..., None].astype(np.float32)   # scattered light everywhere
    out = star_light_with_glow(data, stars, pixel_scale_pc=100.0)[..., 0]
    body = gaussian_filter(stars.astype(float), 10.0) > 0.5
    d = distance_transform_edt(~body)
    reach = GLOW_FADE_REACH * 5.0                                  # 0.5 kpc fade at 100 pc pixels
    assert np.all(out[~stars & (d >= reach)] == 0.0)
    inside = ~stars & (d < reach - 1)
    assert np.all(out[inside] > 0.0)                               # ... and fades there continuously
    assert out[~stars & (d > reach - 2) & (d < reach)].max() < 1e-3 * np.exp(-GLOW_FADE_REACH + 1)


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


def test_the_radius_is_measured_on_the_frame_the_renderer_uses(tmp_path):
    properties = tmp_path / "properties.csv"
    properties.write_text("id,sfr,mass_stars,m_halo,reff\n42,1,1e10,1e12,2\n43,1,1e10,1e12,2\n")
    # A box holding ~30 % of the light, as around subhalo 117266.
    _galaxy(tmp_path / "atlas", "42", twins=True, box=4e-3)
    _galaxy(tmp_path / "atlas", "43", twins=False, box=4e-3)
    report = build_manifest(str(tmp_path / "atlas"), properties_path=str(properties),
                            output_path=str(tmp_path / "manifest.json"))
    assert report["algorithm_version"] == algorithm_version(True)
    radius = {(e["subhalo_id"], e["orientation"]): e["native_re_px"] for e in report["entries"]}
    # The box fills the frame: unmasked, it drags the half-light radius far out.
    assert radius[("42", 1)] < 0.8 * radius[("43", 1)]


def test_without_the_mask_the_dusty_frames_are_used_whole(tmp_path, monkeypatch):
    monkeypatch.setattr(Config.Tng, "STAR_MASK", False)
    folder = _galaxy(tmp_path)
    image = _view(folder).load_surface_brightness().data
    for b in range(4):
        np.testing.assert_allclose(image[..., b], 0.8 * _stars() * (b + 1) + 1e-3, rtol=1e-6)
    paths = [folder / f"TNG42_O1_Euclid_{band}.fits" for band in BANDS]
    with ThreadPoolExecutor(2) as pool:
        assert start_view_read(paths, pool).lit is None            # the twins are not even read


def test_without_the_mask_the_radius_is_measured_on_the_whole_frame(tmp_path, monkeypatch):
    monkeypatch.setattr(Config.Tng, "STAR_MASK", False)
    properties = tmp_path / "properties.csv"
    properties.write_text("id,sfr,mass_stars,m_halo,reff\n42,1,1e10,1e12,2\n43,1,1e10,1e12,2\n")
    _galaxy(tmp_path / "atlas", "42", twins=True, box=4e-3)
    _galaxy(tmp_path / "atlas", "43", twins=False, box=4e-3)
    report = build_manifest(str(tmp_path / "atlas"), properties_path=str(properties),
                            output_path=str(tmp_path / "manifest.json"))
    assert report["algorithm_version"] == algorithm_version(False) != algorithm_version(True)
    radius = {(e["subhalo_id"], e["orientation"]): e["native_re_px"] for e in report["entries"]}
    assert radius[("42", 1)] == pytest.approx(radius[("43", 1)])


def test_the_masked_renderer_identity_is_unchanged_and_the_whole_frame_one_differs():
    assert renderer.rendering_name(True) == "euclid_sersic_shrink_only_v6"
    assert renderer.renderer_fingerprint(True) == (
        "c902961f8ef2c51ef9fb8d02b689fd14e0ed7e8a50a94cb34a8852edfaeff18a")
    assert algorithm_version(True) == "centered-vis-cog-v3-starglow-3000-500-1000"
    assert renderer.rendering_name(False) == "euclid_sersic_shrink_only_v7"
    assert renderer.renderer_fingerprint(False) != renderer.renderer_fingerprint(True)
