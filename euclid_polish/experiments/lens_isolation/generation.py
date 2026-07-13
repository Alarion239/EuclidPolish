"""Generate physically separated full-scene and lens-only HR layers."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from euclid_polish.experiments.lens_isolation.config import SCHEMA_VERSION
from euclid_polish.image import Image, Role


class LensRenderError(RuntimeError):
    """An intended positive could not produce a crop-safe lens system."""


@dataclass(frozen=True)
class GeneratedExample:
    scene: Image
    lens: Image
    dirty: Image | None
    row: dict[str, Any]


class LensIsolationGenerator:
    """Compose ordinary-galaxy, complete-lens, and stellar layers.

    ``sky_simulator`` is an existing :class:`SkySimulator`; tests may inject a
    small object with the same ``simulate_field`` contract. The lens layer is
    rendered independently, so both deflector and lensed source survive while
    unrelated galaxies never enter the target.
    """

    def __init__(
        self,
        sky_simulator,
        observation_simulator,
        *,
        crop_size: int,
        max_lens_retries: int = 50,
    ) -> None:
        if int(crop_size) < 1:
            raise ValueError("crop_size must be >= 1")
        if int(max_lens_retries) < 1:
            raise ValueError("max_lens_retries must be >= 1")
        self.sky = sky_simulator
        self.observation = observation_simulator
        self.crop_size = int(crop_size)
        self.max_lens_retries = int(max_lens_retries)

    def _crop_safe(self, record: dict[str, Any], shape: tuple[int, int]) -> bool:
        half = self.crop_size / 2.0
        x = float(record.get("x_pix", float("nan")))
        y = float(record.get("y_pix", float("nan")))
        return bool(
            np.isfinite(x)
            and np.isfinite(y)
            and half <= x <= shape[1] - half
            and half <= y <= shape[0] - half
        )

    def _lens_layer(self, rng: np.random.Generator, shape: tuple[int, int]) -> tuple[Image, dict[str, Any]]:
        for _attempt in range(1, self.max_lens_retries + 1):
            image, metadata = self.sky.simulate_field(
                rng,
                n_sersic=0,
                n_tng=0,
                n_stars=0,
                n_lenses=1,
            )
            lenses = list(metadata.get("lenses") or [])
            if lenses and self._crop_safe(lenses[0], shape):
                return image, dict(lenses[0])
        raise LensRenderError(f"failed to render a crop-safe lens after {self.max_lens_retries} attempts")

    @staticmethod
    def _copy(template: Image, data: np.ndarray, *, metadata: dict | None = None) -> Image:
        return replace(
            template,
            data=np.asarray(data, dtype=np.float32),
            role=Role.HR,
            is_clean=True,
            metadata=dict(metadata or {}),
            stamp=None,
        )

    def generate_example(
        self,
        rng: np.random.Generator,
        *,
        label: int,
        fixed_dirty: bool,
    ) -> GeneratedExample:
        """Generate one balanced example; ``label`` must be 0 or 1."""
        if int(label) not in {0, 1}:
            raise ValueError("label must be 0 or 1")

        background, background_meta = self.sky.simulate_field(rng, n_stars=0, n_lenses=0)
        background_data = np.asarray(background.data, dtype=np.float32)
        lens_record: dict[str, Any] = {}
        if int(label) == 1:
            lens_image, lens_record = self._lens_layer(rng, background_data.shape[:2])
            lens_data = np.asarray(lens_image.data, dtype=np.float32)
        else:
            lens_data = np.zeros_like(background_data)

        scene_data = background_data + lens_data
        lens = self._copy(background, lens_data, metadata={"lens": lens_record})
        scene = self._copy(
            background,
            scene_data,
            metadata={"label": int(label), "lens": lens_record},
        )

        dirty = None
        n_stars = 0
        if fixed_dirty:
            stars, star_meta = self.sky.simulate_field(
                rng,
                n_sersic=0,
                n_tng=0,
                n_lenses=0,
                deposit_stars=True,
            )
            n_stars = int(star_meta.get("n_stars", len(star_meta.get("stars") or [])))
            observed_scene = self._copy(
                scene,
                scene_data + np.asarray(stars.data, dtype=np.float32),
            )
            dirty, _hr = self.observation.process(observed_scene, rng)

        galaxies = list(background_meta.get("galaxies") or [])
        row = {
            "schema_version": SCHEMA_VERSION,
            "label": int(label),
            "lens_x_pix": lens_record.get("x_pix", ""),
            "lens_y_pix": lens_record.get("y_pix", ""),
            "theta_E_arcsec": lens_record.get("theta_E_arcsec", ""),
            "n_galaxies": int(background_meta.get("n_galaxies", len(galaxies))),
            "n_stars": n_stars,
            "lens_json": json.dumps(lens_record, sort_keys=True),
        }
        return GeneratedExample(scene=scene, lens=lens, dirty=dirty, row=row)
