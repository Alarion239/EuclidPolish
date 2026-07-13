from __future__ import annotations

import numpy as np

from euclid_polish.experiments.lens_isolation.forward import LensIsolationForward
from euclid_polish.image import Image


class Observation:
    def __init__(self):
        self.seen = None

    def process(self, image, _rng):
        self.seen = image.data.copy()
        lr = image.data.reshape(8, 2, 8, 2, 4).sum(axis=(1, 3))
        return Image(lr, 0.1, image.band_names, False), image


def test_forward_injects_stars_only_into_input_and_centres_positive_target():
    observation = Observation()

    def inject_stars(canvas, _rng):
        canvas[0, 0] += 7

    forward = LensIsolationForward(
        observation,
        seed=1,
        crops_per_field=1,
        hr_crop_size=8,
        scale=2,
        jitter_pixels=0,
        star_injector=inject_stars,
    )
    scene = np.zeros((16, 16, 4), np.float32)
    scene[8, 8] = 3
    lens = np.zeros_like(scene)
    lens[8, 8] = 2
    lr, target = forward.crops(scene, lens)
    assert observation.seen[0, 0, 0] == 7
    assert scene[0, 0, 0] == 0
    assert lr.shape == (1, 4, 4, 4)
    assert target.shape == (1, 8, 8, 4)
    assert target.sum() == 8


def test_negative_centres_on_brightest_plain_galaxy_and_target_stays_zero():
    observation = Observation()
    forward = LensIsolationForward(
        observation,
        seed=2,
        crops_per_field=1,
        hr_crop_size=8,
        scale=2,
        jitter_pixels=0,
        star_injector=lambda _canvas, _rng: None,
    )
    scene = np.zeros((16, 16, 4), np.float32)
    scene[4, 12] = 9
    lens = np.zeros_like(scene)
    _lr, target = forward.crops(scene, lens)
    assert target.sum() == 0
    assert observation.seen[4, 12, 0] == 9
