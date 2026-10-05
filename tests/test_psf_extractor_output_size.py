"""The ePSF output-size contract: an even ``output_size`` is bumped down to odd.

``PSFExtractionConfig.output_size`` is documented (and promised by the CLI
prompt and ``extract_all_band_psfs.py --output-size``) to accept any positive
side, with an even value bumped down to the nearest odd one (``1024 → 1023``)
so the ePSF keeps a central pixel. The builder must receive that odd shape.
"""

from __future__ import annotations

import numpy as np
import pytest

from euclid_polish.psf.psf_extractor import (
    PSFExtractionConfig,
    PSFExtractor,
    odd_output_size,
)


@pytest.mark.parametrize("requested,expected", [
    (None, None), (1, 1), (2, 1), (511, 511), (1024, 1023),
])
def test_odd_output_size(requested, expected):
    assert odd_output_size(requested) == expected


@pytest.mark.parametrize("output_size", [1024, 256, 64])
def test_even_output_size_is_valid(output_size):
    ok, msg = PSFExtractionConfig(output_size=output_size).validate()
    assert ok, msg


def test_extractor_accepts_even_output_size():
    extractor = PSFExtractor(PSFExtractionConfig(output_size=1024, progress_bar=False))
    assert extractor.config.effective_output_size == 1023


@pytest.mark.parametrize("output_size", [0, -4])
def test_non_positive_output_size_still_rejected(output_size):
    with pytest.raises(ValueError, match="output_size must be positive"):
        PSFExtractor(PSFExtractionConfig(output_size=output_size))


@pytest.mark.parametrize("requested,shape", [(1024, (1023, 1023)), (511, (511, 511))])
def test_builder_receives_odd_shape(monkeypatch, requested, shape):
    """The builder is asked for the bumped odd shape (odd sizes pass through)."""
    seen = {}

    class _Builder:
        def __init__(self, **kwargs):
            seen.update(kwargs)

        def __call__(self, stars):
            return object(), stars

    extractor = PSFExtractor(PSFExtractionConfig(
        psf_size=5, output_size=requested, progress_bar=False))
    monkeypatch.setattr(extractor, "_epsf_builder_cls", lambda: _Builder)
    star = type("Star", (), {"data": np.ones((5, 5), np.float32)})()
    monkeypatch.setattr("euclid_polish.psf.psf_extractor.EPSFStars", lambda stars: stars)
    extractor.build_epsf_from_stars([star])
    assert seen["shape"] == shape
