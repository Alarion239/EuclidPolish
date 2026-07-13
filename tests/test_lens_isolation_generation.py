from __future__ import annotations

import numpy as np
import pytest

from euclid_polish.experiments.lens_isolation.generation import LensCaptureAdapter
from euclid_polish.image import Image


def _image(data):
    return Image(
        data=np.asarray(data, dtype=np.float32),
        pixel_scale_arcsec=0.05,
        band_names=("VIS", "Y_E", "J_E", "H_E"),
        is_clean=True,
    )


class FakeNormalSky:
    """A normal field loop whose lens method is intercepted by the adapter."""

    def __init__(self, lens_deltas):
        self.background = np.ones((8, 8, 4), np.float32)
        self.lens_deltas = list(lens_deltas)
        self.lens_calls = 0

    def _add_lens(self, canvas, _rng):
        delta = self.lens_deltas[self.lens_calls]
        canvas += delta
        self.lens_calls += 1
        return {"type": "lens", "ordinal": self.lens_calls}

    def simulate_field(self, rng):
        canvas = self.background.copy()
        for _ in self.lens_deltas:
            self._add_lens(canvas, rng)
        return _image(canvas), {
            "galaxies": [{"type": "galaxy", "x_pix": 1.0, "y_pix": 1.0}],
            "stars": [{"type": "star", "x_pix": 0.0, "y_pix": 0.0, "mag_vis": 18.0}],
            "lenses": [{"type": "lens", "ordinal": index + 1} for index in range(len(self.lens_deltas))],
            "n_galaxies": 1,
            "n_stars": 1,
            "n_lenses": len(self.lens_deltas),
        }


class FakeObservation:
    def __init__(self):
        self.received = None

    def process(self, image, _rng):
        self.received = np.asarray(image.data).copy()
        return _image(image.data + 100.0), image


def _deposit_test_star(canvas, _star):
    canvas[0, 0, :] += 11.0


def _lens_delta(value, y, x):
    delta = np.zeros((8, 8, 4), np.float32)
    delta[y, x, :] = value
    return delta


def test_capture_adds_each_single_render_delta_to_scene_and_target():
    first = _lens_delta(5.0, 3, 3)  # foreground deflector
    second = _lens_delta(7.0, 4, 5)  # lensed source
    sky = FakeNormalSky([first, second])
    observation = FakeObservation()
    adapter = LensCaptureAdapter(sky, observation, star_depositor=_deposit_test_star)

    example = adapter.generate_example(np.random.default_rng(1))

    np.testing.assert_array_equal(example.lens.data, first + second)
    expected_observed = sky.background + first + second
    expected_observed[0, 0, :] += 11.0
    np.testing.assert_array_equal(observation.received, expected_observed)
    np.testing.assert_array_equal(example.dirty.data, observation.received + 100.0)
    assert example.sources["n_lenses"] == 2
    assert sky.lens_calls == 2


@pytest.mark.parametrize(
    "lens_deltas", [[], [_lens_delta(5.0, 3, 3)], [_lens_delta(5.0, 3, 3), _lens_delta(7.0, 4, 5)]]
)
def test_normal_lens_outcomes_are_not_relabelled_or_retried(lens_deltas):
    sky = FakeNormalSky(lens_deltas)
    adapter = LensCaptureAdapter(sky, FakeObservation(), star_depositor=_deposit_test_star)

    example = adapter.generate_example(np.random.default_rng(2))

    assert example.sources["n_lenses"] == len(lens_deltas)
    assert sky.lens_calls == len(lens_deltas)
    np.testing.assert_array_equal(
        example.lens.data,
        np.sum(lens_deltas, axis=0) if lens_deltas else np.zeros_like(sky.background),
    )


def test_capture_state_restores_original_lens_method_after_failure():
    sky = FakeNormalSky([_lens_delta(1.0, 3, 3)])
    original = sky._add_lens
    adapter = LensCaptureAdapter(sky, FakeObservation(), star_depositor=_deposit_test_star)

    adapter.generate_example(np.random.default_rng(3))

    assert sky._add_lens.__func__ is original.__func__
