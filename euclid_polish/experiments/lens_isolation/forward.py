"""Target-aware live observation and crop selection."""

from __future__ import annotations

import threading
from collections.abc import Callable

import numpy as np

from euclid_polish.config import Config
from euclid_polish.image import Image
from euclid_polish.sky.generation.sky_simulator import inject_random_stars


class LensIsolationForward:
    """Forward-model a full scene, then crop input and lens target together."""

    def __init__(
        self,
        observation_simulator,
        *,
        seed: int | None = None,
        crops_per_field: int = 16,
        hr_crop_size: int = Config.DEFAULT_HR_CROP_SIZE,
        scale: int = Config.DEFAULT_REBIN_FACTOR,
        jitter_pixels: int | None = None,
        star_injector: Callable[[np.ndarray, np.random.Generator], None] | None = None,
        star_density_arcmin2: float = Config.DEFAULT_STAR_DENSITY_ARCMIN2,
    ) -> None:
        self.observation = observation_simulator
        self.crops_per_field = int(crops_per_field)
        self.hr_crop_size = int(hr_crop_size)
        self.scale = int(scale)
        self.jitter_pixels = self.hr_crop_size // 4 if jitter_pixels is None else int(jitter_pixels)
        self.star_density_arcmin2 = float(star_density_arcmin2)
        self.star_injector = star_injector or self._inject_random_stars
        if self.crops_per_field < 1:
            raise ValueError("crops_per_field must be >= 1")
        if self.hr_crop_size < 1 or self.hr_crop_size % self.scale:
            raise ValueError("hr_crop_size must be positive and divisible by scale")
        if self.jitter_pixels < 0:
            raise ValueError("jitter_pixels must be non-negative")
        self._sequence = np.random.SeedSequence(seed)
        self._lock = threading.Lock()

    def _rng(self) -> np.random.Generator:
        with self._lock:
            child = self._sequence.spawn(1)[0]
        return np.random.default_rng(child)

    def _inject_random_stars(self, canvas: np.ndarray, rng: np.random.Generator) -> None:
        if self.star_density_arcmin2 <= 0:
            return
        side_arcmin = canvas.shape[0] * Config.DEFAULT_PIXEL_SCALE / 60.0
        n_stars = int(rng.poisson(self.star_density_arcmin2 * side_arcmin**2))
        inject_random_stars(
            canvas,
            rng,
            n_stars=n_stars,
            mag_slope=Config.STAR_MAG_SLOPE,
            mag_bright=Config.STAR_MAG_BRIGHT,
            mag_faint=Config.STAR_MAG_FAINT,
        )

    @staticmethod
    def _centre(scene: np.ndarray, lens: np.ndarray) -> tuple[float, float]:
        lens_flux = np.maximum(lens, 0).sum(axis=-1)
        if float(lens_flux.sum()) > 0:
            yy, xx = np.indices(lens_flux.shape)
            total = float(lens_flux.sum())
            return float((yy * lens_flux).sum() / total), float((xx * lens_flux).sum() / total)
        galaxy_flux = np.maximum(scene, 0).sum(axis=-1)
        return tuple(float(v) for v in np.unravel_index(np.argmax(galaxy_flux), galaxy_flux.shape))

    def _offset(
        self,
        centre: tuple[float, float],
        shape: tuple[int, int],
        rng: np.random.Generator,
    ) -> tuple[int, int]:
        c, scale = self.hr_crop_size, self.scale
        jitter = (
            rng.integers(-self.jitter_pixels, self.jitter_pixels + 1, size=2)
            if self.jitter_pixels
            else (0, 0)
        )
        max_y, max_x = shape[0] - c, shape[1] - c
        y = int(round(centre[0] - c / 2 + int(jitter[0])))
        x = int(round(centre[1] - c / 2 + int(jitter[1])))
        y = min(max(y, 0), max_y) // scale * scale
        x = min(max(x, 0), max_x) // scale * scale
        return y, x

    def crops(self, scene: np.ndarray, lens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        scene = np.asarray(scene, np.float32)
        lens = np.asarray(lens, np.float32)
        if scene.shape != lens.shape or scene.ndim != 3:
            raise ValueError("scene and lens must have the same (H, W, C) shape")
        if scene.shape[-1] != len(Config.LR_INPUT_BAND_NAMES):
            raise ValueError("scene/lens must contain the four Euclid bands")
        if min(scene.shape[:2]) < self.hr_crop_size:
            raise ValueError("field is smaller than the requested HR crop")

        rng = self._rng()
        observed = scene.copy()
        self.star_injector(observed, rng)
        image = Image(
            data=observed,
            pixel_scale_arcsec=Config.DEFAULT_PIXEL_SCALE,
            band_names=Config.LR_INPUT_BAND_NAMES,
            is_clean=True,
        )
        lr_image, hr_observed = self.observation.process(image, rng)
        lr = np.asarray(lr_image.data, np.float32)
        height, width = hr_observed.data.shape[:2]
        scene = scene[:height, :width]
        lens = lens[:height, :width]
        centre = self._centre(scene, lens)

        c, scale = self.hr_crop_size, self.scale
        lr_crops = np.empty(
            (self.crops_per_field, c // scale, c // scale, scene.shape[-1]),
            np.float32,
        )
        targets = np.empty((self.crops_per_field, c, c, lens.shape[-1]), np.float32)
        for index in range(self.crops_per_field):
            y, x = self._offset(centre, (height, width), rng)
            targets[index] = lens[y : y + c, x : x + c]
            lr_crops[index] = lr[
                y // scale : (y + c) // scale,
                x // scale : (x + c) // scale,
            ]
        return lr_crops, targets
