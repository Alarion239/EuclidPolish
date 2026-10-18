"""Grid detector: signed axis excess of high-passed windows (score, z-score,
amplitude) and the gated training penalty."""
from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf
from scipy.ndimage import gaussian_filter

from euclid_polish.training.grid_penalty import AxisExcess, default_detector

W = 64


def _reference(img: np.ndarray) -> dict[str, np.ndarray]:
    """Plain NumPy re-implementation for one (H, W) band."""
    ky, kx = np.meshgrid(np.fft.fftfreq(W), np.fft.fftfreq(W), indexing="ij")
    kr = np.hypot(kx, ky)
    annulus = (kr >= 1 / 24) & (kr <= 1 / 3)
    on = annulus & ((np.abs(kx) <= 1.5 / W) | (np.abs(ky) <= 1.5 / W))
    off = annulus & ~on
    rings = np.floor(kr * W + 0.5).astype(int)
    yy, xx = np.mgrid[0:W, 0:W]
    rr = np.hypot(yy - (W - 1) / 2, xx - (W - 1) / 2)
    taper = np.where(rr < W / 2, 0.5 * (1 + np.cos(np.pi * rr / (W / 2))), 0.0)
    norm = W * W * (taper ** 2).sum()
    fluct = default_detector()._fluctuation
    hp = img - gaussian_filter(img.astype(np.float64), 3.0, mode="reflect", truncate=4.0)
    out = {"score": [], "z": [], "amp": []}
    for y in range(0, img.shape[0] - W + 1, W):
        for x in range(0, img.shape[1] - W + 1, W):
            p = np.abs(np.fft.fft2(hp[y:y + W, x:x + W] * taper)) ** 2
            s_on = s_pred = var = 0.0
            for r in sorted(set(rings[on].tolist())):
                mo, mf = on & (rings == r), off & (rings == r)
                if mo.any() and mf.any():
                    pred = mo.sum() * p[mf].mean()
                    s_on += p[mo].sum()
                    s_pred += pred
                    var += pred ** 2 / mo.sum()
            excess = s_on - s_pred
            out["score"].append(s_on / s_pred)
            out["z"].append(excess / np.sqrt(fluct * var))
            out["amp"].append(np.sign(excess) * np.sqrt(abs(excess) / norm))
    return {k: np.array(v) for k, v in out.items()}


def _stripes(shape, period, rms, diagonal=False):
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    phase = (xx + yy) / np.sqrt(2.0) if diagonal else xx
    return (rms * np.sqrt(2.0) * np.cos(2 * np.pi * phase / period)).astype(np.float32)


def _red_field(n, alpha, rng):
    fy, fx = np.meshgrid(np.fft.fftfreq(n), np.fft.fftfreq(n), indexing="ij")
    k = np.hypot(fx, fy)
    k[0, 0] = 1.0
    amp = k ** (-alpha / 2.0)
    amp[0, 0] = 0.0
    field = np.real(np.fft.ifft2(np.fft.fft2(rng.normal(size=(n, n))) * amp))
    return (field / field.std()).astype(np.float32)


def test_matches_a_numpy_reference():
    rng = np.random.default_rng(3)
    img = (rng.normal(size=(1, 128, 192, 2)) + _stripes((128, 192), 6.0, 0.3)[None, ..., None]).astype(np.float32)
    det = default_detector()
    got = {"score": det.score(img).numpy(), "z": det.zscore(img).numpy(), "amp": det.amplitude(img).numpy()}
    for c in range(2):
        ref = _reference(img[0, ..., c])
        for k in ref:
            np.testing.assert_allclose(got[k][0, ..., c].ravel(), ref[k], rtol=2e-4, atol=1e-5)


@pytest.mark.parametrize("alpha", [0.0, 3.0])
def test_isotropic_fields_score_zero_with_unit_spread(alpha):
    rng = np.random.default_rng(11)
    img = np.stack([_red_field(512, alpha, rng) if alpha else rng.normal(size=(512, 512)).astype(np.float32)
                    for _ in range(2)])[..., None]
    z = default_detector().zscore(img).numpy().ravel()          # 128 windows
    assert abs(z.mean()) < 0.35
    assert 0.7 < z.std() < 1.3


def test_axis_aligned_stripes_are_caught_and_diagonal_ones_are_not():
    rng = np.random.default_rng(5)
    noise = rng.normal(size=(256, 256)).astype(np.float32)
    det = default_detector()
    axis = det.zscore((noise + _stripes(noise.shape, 5.0, 0.3))[None, ..., None]).numpy()
    diag = det.zscore((noise + _stripes(noise.shape, 5.0, 0.3, diagonal=True))[None, ..., None]).numpy()
    plain = det.zscore(noise[None, ..., None]).numpy()
    assert np.all(axis > 4.0)
    assert diag.mean() < plain.mean() + 1.0
    amp = det.amplitude((noise + _stripes(noise.shape, 5.0, 0.3))[None, ..., None]).numpy()
    assert 0.2 < np.median(amp) < 0.45                           # the injected RMS, 0.3


def test_penalty_ignores_isotropic_noise_and_its_gradient_removes_only_the_grid():
    rng = np.random.default_rng(8)
    noise = rng.normal(size=(1, 256, 256, 1)).astype(np.float32)
    grid = _stripes((256, 256), 7.0, 0.4)[None, ..., None]
    det = default_detector()
    assert float(det.penalty(noise, 1.0)) < 0.05
    x = tf.Variable(noise + grid)
    assert float(det.penalty(x, 1.0)) > 5.0
    for _ in range(40):
        with tf.GradientTape() as tape:
            loss = det.penalty(x, 1.0)
        x.assign_sub(3.0 * tape.gradient(loss, x))
    change = x.numpy() - (noise + grid)
    assert float(det.penalty(x, 1.0)) < 0.05
    # The change is itself on the axes: it takes the grid out, it does not
    # repaint the noise.
    assert float(np.median(det.score(change).numpy())) > 5.0
    assert float(np.sum(change * grid)) < 0.0


def test_penalty_is_in_fluctuation_units_down_to_the_floor():
    rng = np.random.default_rng(9)
    img = (rng.normal(size=(1, 256, 256, 1)) + _stripes((256, 256), 7.0, 0.4)[None, ..., None]).astype(np.float32)
    det = default_detector()
    full = float(det.penalty(img, 1.0))
    np.testing.assert_allclose(float(det.penalty(7.0 * img, 7.0)), full, rtol=1e-4)
    # The same pattern far below 1 % of the noise costs (almost) nothing.
    assert float(det.penalty(1e-4 * img, 1.0)) < 1e-3 * full


def test_images_smaller_than_a_window_are_rejected():
    with pytest.raises(ValueError):
        default_detector().zscore(np.zeros((1, 32, 128, 1), np.float32))
    with pytest.raises(ValueError):
        AxisExcess(window=8)
