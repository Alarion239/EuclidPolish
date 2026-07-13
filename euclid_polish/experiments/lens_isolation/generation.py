"""Capture complete normal lens renders as aligned experiment targets."""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from euclid_polish.image import Image, Role
from euclid_polish.sky.generation.sky_simulator import _deposit_star

StarDepositor = Callable[[np.ndarray, dict[str, Any]], None]


def _deposit_fixed_star(canvas: np.ndarray, star: dict[str, Any]) -> None:
    """Use the production star primitive to restore recorded fixed stars."""
    _deposit_star(
        canvas,
        float(star["x_pix"]),
        float(star["y_pix"]),
        float(star["mag_vis"]),
    )


@dataclass(frozen=True)
class GeneratedExample:
    """One dirty normal-field input and its clean lens-system target."""

    dirty: Image
    lens: Image
    sources: dict[str, Any]


class LensCaptureAdapter:
    """Adapt one normal :class:`SkySimulator` draw without changing it.

    The adapter temporarily intercepts the existing private lens-addition
    callback for exactly one ``simulate_field`` call.  Each existing render is
    still executed once on the normal scene canvas; its before/after delta is
    accumulated in a second canvas.  That delta contains the complete lens
    system (deflector plus lensed source), but no ordinary galaxies or stars.
    """

    def __init__(
        self,
        sky_simulator,
        observation_simulator,
        *,
        star_depositor: StarDepositor = _deposit_fixed_star,
    ) -> None:
        self.sky = sky_simulator
        self.observation = observation_simulator
        self.star_depositor = star_depositor
        self._capture_lock = threading.RLock()

    @staticmethod
    def _lens_image(template: Image, data: np.ndarray) -> Image:
        return replace(
            template,
            data=np.asarray(data, dtype=np.float32),
            role=Role.HR,
            is_clean=True,
            metadata={"experiment": "lens_isolation", "target": "complete_lens_system"},
            stamp=None,
        )

    def _with_fixed_stars(self, scene: Image, sources: dict[str, Any]) -> Image:
        data = np.asarray(scene.data, dtype=np.float32).copy()
        for star in sources.get("stars", []):
            self.star_depositor(data, star)
        return replace(scene, data=data, stamp=None)

    @contextmanager
    def _capture_lens_deltas(self) -> Iterator[Callable[[Image], np.ndarray]]:
        """Scope temporary interception to one field and always restore it."""
        with self._capture_lock:
            original = self.sky._add_lens
            target: np.ndarray | None = None

            def capture(canvas: np.ndarray, rng: np.random.Generator):
                nonlocal target
                before = np.asarray(canvas, dtype=np.float32).copy()
                record = original(canvas, rng)
                if target is None:
                    target = np.zeros_like(canvas, dtype=np.float32)
                target += np.asarray(canvas, dtype=np.float32) - before
                return record

            self.sky._add_lens = capture
            try:
                yield (
                    lambda scene: (
                        np.zeros_like(scene.data, dtype=np.float32)
                        if target is None
                        else np.asarray(target, dtype=np.float32)
                    )
                )
            finally:
                self.sky._add_lens = original

    def generate_example(self, rng: np.random.Generator) -> GeneratedExample:
        """Generate one unbiased normal field and its aligned lens-only target."""
        with self._capture_lens_deltas() as target_for:
            scene, sources = self.sky.simulate_field(rng)
            target = target_for(scene)
        dirty, _ = self.observation.process(self._with_fixed_stars(scene, sources), rng)
        return GeneratedExample(
            dirty=dirty,
            lens=self._lens_image(scene, target),
            sources=dict(sources),
        )
