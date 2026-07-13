from __future__ import annotations

import numpy as np
import pytest

from euclid_polish.experiments.lens_isolation.generation import (
    LensIsolationGenerator,
    LensRenderError,
)
from euclid_polish.image import Image


def _image(data):
    return Image(
        data=np.asarray(data, dtype=np.float32),
        pixel_scale_arcsec=0.05,
        band_names=("VIS", "Y_E", "J_E", "H_E"),
        is_clean=True,
    )


class FakeSky:
    def __init__(self, background, lens, stars, *, valid_lens=True):
        self.background = _image(background)
        self.lens = _image(lens)
        self.stars = _image(stars)
        self.valid_lens = valid_lens
        self.calls = []

    def simulate_field(self, _rng, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("n_lenses") == 1:
            lenses = (
                [
                    {
                        "x_pix": 4.0 if self.valid_lens else 0.0,
                        "y_pix": 4.0 if self.valid_lens else 0.0,
                        "theta_E_arcsec": 1.2,
                        "lens_light_render": "deflector",
                        "source_render": "lensed-source",
                    }
                ]
                if self.valid_lens
                else []
            )
            return self.lens, {"lenses": lenses, "n_lenses": len(lenses)}
        if kwargs.get("deposit_stars"):
            return self.stars, {"stars": [{"mag_vis": 18.0}], "n_stars": 1}
        return self.background, {"galaxies": [{"type": "galaxy"}], "n_galaxies": 1}


class FakeObservation:
    def __init__(self):
        self.received = None

    def process(self, image, _rng):
        self.received = np.asarray(image.data).copy()
        dirty = _image(image.data + 100.0)
        return dirty, image


@pytest.fixture
def layers():
    background = np.ones((8, 8, 4), np.float32)
    lens = np.zeros_like(background)
    lens[3, 3, :] = 5.0  # foreground deflector light
    lens[4, 5, :] = 7.0  # lensed source light
    stars = np.zeros_like(background)
    stars[1, 1, :] = 11.0
    return background, lens, stars


def test_positive_keeps_complete_lens_layer_and_excludes_plain_galaxies(layers):
    background, lens, stars = layers
    sky = FakeSky(background, lens, stars)
    generator = LensIsolationGenerator(sky, FakeObservation(), crop_size=4, max_lens_retries=2)

    example = generator.generate_example(np.random.default_rng(1), label=1, fixed_dirty=False)

    np.testing.assert_array_equal(example.lens.data, lens)
    np.testing.assert_array_equal(example.scene.data, background + lens)
    assert example.lens.data[3, 3, 0] == 5.0
    assert example.lens.data[4, 5, 0] == 7.0
    assert example.row["label"] == 1
    assert example.row["theta_E_arcsec"] == pytest.approx(1.2)
    assert example.dirty is None


def test_negative_target_is_exactly_zero(layers):
    background, lens, stars = layers
    generator = LensIsolationGenerator(FakeSky(background, lens, stars), FakeObservation(), crop_size=4)

    example = generator.generate_example(np.random.default_rng(2), label=0, fixed_dirty=False)

    np.testing.assert_array_equal(example.scene.data, background)
    np.testing.assert_array_equal(example.lens.data, np.zeros_like(background))
    assert example.row["label"] == 0


def test_fixed_dirty_is_forwarded_from_scene_plus_stars(layers):
    background, lens, stars = layers
    observation = FakeObservation()
    generator = LensIsolationGenerator(FakeSky(background, lens, stars), observation, crop_size=4)

    example = generator.generate_example(np.random.default_rng(3), label=1, fixed_dirty=True)

    np.testing.assert_array_equal(observation.received, background + lens + stars)
    np.testing.assert_array_equal(example.dirty.data, background + lens + stars + 100.0)
    np.testing.assert_array_equal(example.lens.data, lens)
    assert example.row["n_stars"] == 1


def test_intended_positive_raises_after_bounded_lens_retries(layers):
    background, lens, stars = layers
    sky = FakeSky(background, lens, stars, valid_lens=False)
    generator = LensIsolationGenerator(sky, FakeObservation(), crop_size=4, max_lens_retries=3)

    with pytest.raises(LensRenderError, match="3"):
        generator.generate_example(np.random.default_rng(4), label=1, fixed_dirty=False)

    assert sum(call.get("n_lenses") == 1 for call in sky.calls) == 3
