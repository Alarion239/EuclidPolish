"""The fast TNG donor loader gives bit for bit what the plain one gave.

Generation loads a donor view for nearly every galaxy it draws, so the
loader reads its eight FITS files concurrently, converts each pixel once,
blurs the star body on its bounding box only and runs the glow's distance
transform and per-band arithmetic in numba. Each test here pins one of those
to the straightforward numpy/SciPy expression it replaced (the 2026-10-19
recipe, with the fade ending at zero since 2026-10-20), on frames with
negative, non-finite and edge-touching pixels.
"""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import pytest
from astropy.io import fits
from scipy.ndimage import distance_transform_edt

from euclid_polish.config import Config
from euclid_polish.tng import _image as tng_image
from euclid_polish.tng.atlas import TNGGalaxy, _resolve_fits_path, _scan_complete_galaxies
from euclid_polish.tng.image import TNGSurfaceBrightnessImage
from euclid_polish.tng.radius_manifest import _file_identity, _inventory
from euclid_polish.tng.types import TNG_FITS_BANDS, TNG_MODEL_BANDS, TNGView, dust_free_twin


# --------------------------------------------------------------------------- #
# The replaced implementations, verbatim in substance.

def _reference_glow(data, stars, *, pixel_scale_pc):
    data = np.asarray(data, dtype=np.float32)
    stars = np.asarray(stars, dtype=bool)
    blur_px = Config.Tng.SCATTER_GLOW_BLUR_PC / pixel_scale_pc
    body_px = Config.Tng.STAR_BODY_SMOOTH_PC / pixel_scale_pc
    fade_px = Config.Tng.SCATTER_GLOW_FADE_PC / pixel_scale_pc
    out = data * stars[..., None]
    body = cv2.GaussianBlur(stars.astype(np.float32), (0, 0), sigmaX=body_px,
                            sigmaY=body_px, borderType=cv2.BORDER_CONSTANT) > 0.5
    if not body.any():
        return out
    rows, cols = np.nonzero(body)
    margin = int(np.ceil(tng_image.GLOW_FADE_REACH * fade_px + 3.0 * blur_px))
    ys = slice(max(rows.min() - margin, 0), min(rows.max() + margin + 1, data.shape[0]))
    xs = slice(max(cols.min() - margin, 0), min(cols.max() + margin + 1, data.shape[1]))
    sub_stars, sub_body = stars[ys, xs], body[ys, xs]
    fade = tng_image._glow_fade(distance_transform_edt(~sub_body), fade_px)
    off = (~sub_stars).astype(np.float32)
    factor = max(1, int(blur_px // 10.0))
    coverage = _reference_coarse_blur(off, blur_px, factor)
    for band in range(data.shape[-1]):
        light = np.nan_to_num(data[ys, xs, band]) * off
        glow = _reference_coarse_blur(light, blur_px, factor) / np.maximum(coverage, 1e-6)
        out[ys, xs, band] = np.where(sub_stars, data[ys, xs, band], glow * fade)
    return out


def _reference_load(paths):
    planes = []
    for path in paths:
        with fits.open(path) as hdul:
            arr = np.asarray(hdul[0].data, dtype=np.float32)
        planes.append(np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0))
    mask = None
    for path in paths:
        with fits.open(dust_free_twin(path)) as hdul:
            lit = np.nan_to_num(np.asarray(hdul[0].data, dtype=np.float64)) > 0.0
        mask = lit if mask is None else (mask | lit)
    return np.stack(planes, axis=-1), mask


def _reference_inventory(galaxies):
    entries = []
    for galaxy in galaxies:
        for orientation in range(1, 6):
            for band in TNG_FITS_BANDS:
                path = galaxy.fits_path(orientation, band)
                if path.is_file():
                    entries.append(_file_identity(str(path)))
                else:
                    entries.append({"path": os.path.abspath(path), "size": 0, "mtime_ns": 0})
                twin = dust_free_twin(path)
                if twin.is_file():
                    entries.append(_file_identity(str(twin)))
    entries.sort(key=lambda row: str(row["path"]))
    return entries


# --------------------------------------------------------------------------- #

def _same(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def _galaxy_frame(rng, size, centres, *, negative=True):
    yy, xx = np.mgrid[:size, :size]
    light = np.zeros((size, size), np.float64)
    stars = np.zeros((size, size), bool)
    for cy, cx, r in centres:
        rr = np.hypot(yy - cy, xx - cx)
        light += np.exp(-rr / max(r / 3.0, 1.0))
        stars |= rr < r
    data = np.stack([light * (b + 1) + rng.normal(0, 1e-3, light.shape)
                     for b in range(4)], axis=-1).astype(np.float32)
    if not negative:
        data = np.abs(data)
    stars |= rng.random(stars.shape) < 2e-4            # lone halo star particles
    return data, stars


@pytest.mark.parametrize("seed", range(6))
def test_glow_is_bitwise_the_plain_numpy_glow(seed):
    rng = np.random.default_rng(seed)
    size = int(rng.integers(300, 700))
    centres = [(int(rng.integers(0, size)), int(rng.integers(0, size)), int(rng.integers(8, 60)))
               for _ in range(int(rng.integers(1, 4)))]
    centres[0] = (size // 2, size // 2, 40)
    if seed == 5:
        centres.append((3, size - 4, 30))                # a body cut by the frame edge
    data, stars = _galaxy_frame(rng, size, centres)
    data.setflags(write=False)                          # as a loaded donor's pixels
    got = tng_image.star_light_with_glow(data, stars, pixel_scale_pc=100.0)
    assert _same(got, _reference_glow(data, stars, pixel_scale_pc=100.0))


def test_glow_keeps_non_finite_light_like_the_plain_glow():
    rng = np.random.default_rng(1)
    data, stars = _galaxy_frame(rng, 400, [(200, 200, 40)])
    data[180, 200, 1] = np.nan                           # on a star
    data[150, 150, 2] = np.inf                           # off the stars, in the glow
    got = tng_image.star_light_with_glow(data, stars, pixel_scale_pc=100.0)
    want = _reference_glow(data, stars, pixel_scale_pc=100.0)
    assert np.array_equal(got, want, equal_nan=True)


def test_glow_without_a_body_is_the_starlight_alone():
    data = np.ones((200, 200, 4), np.float32)
    stars = np.zeros((200, 200), bool)
    stars[5, 5] = True                                  # one lone particle: no outline
    got = tng_image.star_light_with_glow(data, stars, pixel_scale_pc=100.0)
    assert _same(got, _reference_glow(data, stars, pixel_scale_pc=100.0))
    empty = tng_image.star_light_with_glow(data, np.zeros_like(stars), pixel_scale_pc=100.0)
    assert not empty.any()


@pytest.mark.parametrize("seed", range(4))
def test_star_body_on_its_bounding_box_is_the_full_frame_blur(seed):
    rng = np.random.default_rng(seed)
    size = 500
    stars = np.zeros((size, size), bool)
    for _ in range(int(rng.integers(1, 5))):
        cy, cx, r = rng.integers(0, size), rng.integers(0, size), rng.integers(3, 40)
        yy, xx = np.ogrid[:size, :size]
        stars |= (yy - cy) ** 2 + (xx - cx) ** 2 < r * r
    stars |= rng.random(stars.shape) < 1e-4
    full = cv2.GaussianBlur(stars.astype(np.float32), (0, 0), sigmaX=10.0, sigmaY=10.0,
                            borderType=cv2.BORDER_CONSTANT) > 0.5
    assert _same(tng_image._star_body(stars, 10.0), full)


def test_distance_to_mask_is_bitwise_scipy():
    rng = np.random.default_rng(0)
    for trial in range(150):
        h, w = (int(v) for v in rng.integers(1, 70, size=2))
        mask = rng.random((h, w)) < rng.choice([0.002, 0.05, 0.5, 0.97])
        if trial % 5 == 0:
            mask[:] = False
            mask[rng.integers(0, h), rng.integers(0, w)] = True
        if not mask.any():
            continue
        assert _same(tng_image.distance_to_mask(mask), distance_transform_edt(~mask))
    assert np.isinf(tng_image.distance_to_mask(np.zeros((3, 4), bool))).all()


def _write(path: Path, data: np.ndarray) -> None:
    hdu = fits.PrimaryHDU(np.asarray(data, dtype=">f4"))
    hdu.header["BUNIT"] = "MJy/sr"
    for key, value in (("CDELT1", 100.0), ("CUNIT1", "pc"), ("CDELT2", 100.0), ("CUNIT2", "pc")):
        hdu.header[key] = value
    hdu.writeto(path, overwrite=True)


def _view_files(folder: Path, rng, *, twins: bool = True) -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for b, band in enumerate(TNG_FITS_BANDS):
        dusty = rng.normal(0.0, 1.0, (64, 64)).astype(np.float32)
        dusty[3, 4] = np.nan
        dusty[5, 6] = np.inf
        dusty[7, 8] = -np.inf
        path = folder / f"TNG7_O1_Euclid_{band}.fits"
        _write(path, dusty)
        if twins:
            free = rng.normal(0.0, 1.0, (64, 64)).astype(np.float32)
            free[1, b] = np.nan
            free[2, b] = np.inf
            free[3, b] = -np.inf
            free[4, b] = 1e-40                           # a subnormal is still light
            _write(dust_free_twin(path), free)
        paths.append(path)
    return paths


def test_concurrent_view_load_is_bitwise_the_per_plane_load(tmp_path):
    paths = _view_files(tmp_path / "7", np.random.default_rng(3))
    image, mask = tng_image.load_view_frames(paths, TNG_MODEL_BANDS)
    planes, reference_mask = _reference_load(paths)
    assert _same(image.data, planes)
    assert image.bands == TNG_MODEL_BANDS and image.pixel_scale_pc == 100.0
    assert _same(mask, reference_mask)
    assert _same(tng_image.load_star_mask(paths, required=True), reference_mask)
    single = tng_image._load_tng_plane(paths[2], "J")
    assert _same(single.data, planes[..., 2:3])


def test_view_load_without_twins_reports_no_mask(tmp_path):
    paths = _view_files(tmp_path / "7", np.random.default_rng(4), twins=False)
    image, mask = tng_image.load_view_frames(paths, TNG_MODEL_BANDS)
    assert mask is None
    planes = [np.nan_to_num(np.asarray(fits.getdata(path), np.float32), nan=0.0, posinf=0.0, neginf=0.0)
              for path in paths]
    assert _same(image.data, np.stack(planes, axis=-1))
    view = TNGView(galaxy_dir=tmp_path / "7", subhalo_id="7", orientation=1, native_re_px=5.0)
    if Config.Tng.REQUIRE_DUST_FREE:
        with pytest.raises(FileNotFoundError):
            view.load_surface_brightness()
    else:
        assert _same(view.load_surface_brightness().data, image.data)


def test_view_load_checks_the_frames_as_before(tmp_path):
    paths = _view_files(tmp_path / "7", np.random.default_rng(5))
    hdul = fits.open(paths[1])
    hdul[0].header["BUNIT"] = "erg/s"
    hdul.writeto(paths[1], overwrite=True)
    with pytest.raises(ValueError, match="SKIRT BUNIT must be 'MJy/sr'"):
        tng_image.load_view_frames(paths, TNG_MODEL_BANDS)


def test_adopted_image_owns_its_pixels_without_a_copy():
    values = np.ones((4, 5, 2), np.float32)
    image = TNGSurfaceBrightnessImage.adopt(values, ("VIS", "H"), 100.0)
    assert image.data is values and not values.flags.writeable
    assert image == TNGSurfaceBrightnessImage(data=np.ones((4, 5, 2)), bands=("VIS", "H"),
                                              pixel_scale_pc=100.0)
    strided = TNGSurfaceBrightnessImage.adopt(np.ones((4, 6, 2), np.float64)[:, ::2], ("VIS", "H"), 1.0)
    assert strided.data.dtype == np.float32 and strided.data.flags.c_contiguous
    with pytest.raises(ValueError, match="only finite values"):
        TNGSurfaceBrightnessImage.adopt(np.full((2, 2, 1), np.nan, np.float32), ("VIS",), 1.0)
    with pytest.raises(ValueError, match="bands has 1 entries"):
        TNGSurfaceBrightnessImage.adopt(np.ones((2, 2, 2), np.float32), ("VIS",), 1.0)
    with pytest.raises(ValueError, match="pixel_scale_pc"):
        TNGSurfaceBrightnessImage.adopt(np.ones((2, 2, 1), np.float32), ("VIS",), 0.0)


def _atlas(root: Path) -> None:
    """Galaxies with unpadded and zero-padded names, one missing a frame, one
    missing a twin, one without its done marker."""
    def galaxy(gid, name_id, *, skip=None, skip_twin=None, done=True):
        folder = root / gid
        folder.mkdir(parents=True)
        for orientation in range(1, 6):
            for band in TNG_FITS_BANDS:
                name = f"TNG{name_id}_O{orientation}_Euclid_{band}.fits"
                if name != skip:
                    (folder / name).write_bytes(b"x" * (orientation + len(band)))
                if name != skip_twin:
                    (folder / name.replace(".fits", "_nodust.fits")).write_bytes(b"yy")
        if done:
            (folder / Config.Tng.DONE_MARKER).write_text("40 fits with-nodust")
    galaxy("11", "11")
    galaxy("12", "000012")
    galaxy("13", "13", skip="TNG13_O2_Euclid_J.fits")
    galaxy("14", "14", skip_twin="TNG14_O5_Euclid_H.fits")
    galaxy("15", "15", done=False)
    (root / "16").mkdir()
    (root / "notes.txt").write_text("not a galaxy")


def test_atlas_scan_matches_per_file_checks(tmp_path, monkeypatch):
    _atlas(tmp_path)
    for required in (False, True):
        monkeypatch.setattr(Config.Tng, "REQUIRE_DUST_FREE", required)
        found = {g.subhalo_id for g in _scan_complete_galaxies(tmp_path, include_box_galaxies=True)}
        expected = set()
        for folder in tmp_path.iterdir():
            if not folder.is_dir() or not (folder / Config.Tng.DONE_MARKER).is_file():
                continue
            frames = [_resolve_fits_path(folder, folder.name, o, b)
                      for o in range(1, 6) for b in TNG_FITS_BANDS]
            if all(p.is_file() for p in frames) and (
                    not required or all(dust_free_twin(p).is_file() for p in frames)):
                expected.add(folder.name)
        assert found == expected
    assert found == {"11", "12"}                        # REQUIRE_DUST_FREE drops 14 too


def test_inventory_matches_per_file_stats(tmp_path):
    _atlas(tmp_path)
    galaxies = [TNGGalaxy(directory=tmp_path / gid, subhalo_id=gid) for gid in ("11", "12", "13", "14")]
    assert _inventory(galaxies) == _reference_inventory(galaxies)


def _reference_coarse_blur(image, sigma_px, factor):
    if factor <= 1:
        return tng_image._gaussian_blur(image, sigma_px)
    height, width = image.shape
    padded = np.pad(image, ((0, -height % factor), (0, -width % factor)))
    small = padded.reshape(padded.shape[0] // factor, factor,
                           padded.shape[1] // factor, factor).mean(axis=(1, 3))
    small = tng_image._gaussian_blur(small.astype(np.float32), sigma_px / factor)
    big = cv2.resize(small, (padded.shape[1], padded.shape[0]), interpolation=cv2.INTER_LINEAR)
    return big[:height, :width]


def test_coarse_blur_is_bitwise_the_padded_numpy_mean():
    """Including the shapes where numpy sums a block in another order (one
    block across the width, 8+ terms) and so keeps the numpy path."""
    rng = np.random.default_rng(2)
    shapes = [(int(h), int(w)) for h, w in rng.integers(1, 260, size=(150, 2))]
    shapes += [(30, 3), (31, 4), (7, 2), (90, 9), (64, 64)]
    for index, (height, width) in enumerate(shapes):
        factor = int(rng.integers(1, 10)) if index < 150 else 3
        image = (rng.normal(0.0, 1.0, (height, width)) * 10 ** rng.uniform(-8, 3)).astype(np.float32)
        if index % 4 == 0:
            image[rng.random(image.shape) < 0.6] = 0.0
        sigma = float(rng.uniform(1.0, 40.0))
        assert _same(tng_image._coarse_blur(image, sigma, factor),
                     _reference_coarse_blur(image, sigma, factor)), (height, width, factor)
