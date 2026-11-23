"""Blank (exactly zero) NISP pixels of a MER mosaic.

A MER mosaic pixel is blank where **every exposure covering it is flagged
invalid**. Each NISP exposure carries an invalid mask on its own 0.30"
detector grid:

* **persistence** (NIR flag bit 13) — the afterglow of earlier grism
  exposures: single pixels and small clusters, plus long bands along the
  dispersion direction (the spectra of bright stars);
* **cosmic rays** (bit 16) in the exposure itself, each masked with its
  neighbours;
* **other** invalid pixels (bad-pixel clusters).

MER resamples every exposure bilinearly onto the 0.10" tile grid, so one
invalid detector pixel removes the 6x6 mosaic pixels whose 2x2 interpolation
stencil touches it. With one covering exposure a DR1 field loses 5-9% of its
NISP pixels, with two about 0.9%, with three about 0.13%.

Rates were fitted to single-exposure regions of real mosaics (2026-11-22,
``output/holes_model``): ``"dr1"`` masks persistence from five hours of
grism exposures, ``"q1"`` from one exposure, about a quarter as much.

**Star cores.** A bright star's core is invalid in an exposure (saturated,
or taken for a cosmic ray) once its peak passes a level, and the pipeline
masks the pixel's neighbours with it. Half of the real cores are blank near
15 ke- (Q1) and 23 ke- (DR1) of predicted peak in the NISP bands and near
1.8 Me- in VIS, in a round patch about an arcsecond across — not a box.
Every exposure must lose the core, so marginal stars survive more often
under more exposures. Smooth galaxy light is recorded well above those
levels and only loses its brightest pixels. VIS gets star cores only: its
other blanks are rare (0.1% under two exposures).

Y, J and H of one dither are taken at one pointing: they share the bands
and the other invalid pixels, part of the persistence clusters, and none of
the cosmic rays.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation, label, maximum_filter

#: Bands of a blank mask, in the order of :meth:`BlankPixelModel.blank_mask`
#: (bit ``k`` of the per-exposure invalid bits is band ``k``).
BAND_NAMES = ("VIS", "Y_E", "J_E", "H_E")
#: Bands that get persistence, cosmic-ray and bad-pixel blanks.
BLANK_BAND_NAMES = ("Y_E", "J_E", "H_E")
#: Detector pixel over mosaic pixel (0.30" / 0.10").
DETECTOR_FACTOR = 3
#: Detector edges crossing a field, per mosaic pixel of its diagonal. Of real
#: DR1 fields, 54% of 25.5" crops and 19% of 64" windows hold a single
#: exposure count: a Poisson count of mean 0.62 and 1.66 for diagonals of 360
#: and 905 pixels.
SEAM_RATE_PER_PIXEL = 1.75e-3

_BIT = {name: np.uint8(1 << k) for k, name in enumerate(BAND_NAMES)}
_ALL = np.uint8((1 << len(BAND_NAMES)) - 1)
_NISP = np.uint8(sum(int(_BIT[name]) for name in BLANK_BAND_NAMES))
# A core is traced from this share of its blanking level, so a source's
# scatter can still lift it over the level.
_CORE_CANDIDATE_SHARE = 0.25
_PLUS = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
# Persistence masks resemble each other along J -> H -> Y.
_CHAIN = ("J_E", "H_E", "Y_E")
_PARAMS_PATH = Path(__file__).with_name("blank_pixel_params.json")
_STAMPS_PER_MIX = 512
_NEIGHBOURS_4 = ((0, 1), (1, 0), (0, -1), (-1, 0))
_NEIGHBOURS_8 = tuple((a, b) for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0))


@lru_cache(maxsize=1)
def _all_params() -> dict:
    with open(_PARAMS_PATH, encoding="utf-8") as handle:
        return json.load(handle)


def blank_pixel_releases() -> tuple[str, ...]:
    """Releases with fitted blank-pixel rates."""
    return tuple(key for key in _all_params() if not key.startswith("_"))


class _Stamps:
    """A fixed library of invalid-pixel clusters drawn from a shape mixture.

    A shape is a connected cluster of ``cells`` detector pixels grown by its
    four or eight neighbours (how cosmic rays are masked). Picking a library
    entry uniformly samples the mixture.
    """

    def __init__(self, shapes: list, seed: int) -> None:
        rng = np.random.default_rng(seed)
        weights = np.array([row[2] for row in shapes], dtype=np.float64)
        picks = rng.choice(len(shapes), size=_STAMPS_PER_MIX, p=weights / weights.sum())
        clusters = [self._cluster(int(shapes[i][0]), int(shapes[i][1]), rng) for i in picks]
        self.counts = np.array([len(c) for c in clusters], dtype=np.int32)
        self.offsets = np.zeros((_STAMPS_PER_MIX, int(self.counts.max()), 2), dtype=np.int32)
        for k, cells in enumerate(clusters):
            self.offsets[k, :len(cells)] = cells

    @staticmethod
    def _cluster(cells: int, growth: int, rng: np.random.Generator) -> list:
        core = [(0, 0)]
        while len(core) < cells:
            y, x = core[int(rng.integers(len(core)))]
            dy, dx = _NEIGHBOURS_4[int(rng.integers(4))]
            if (y + dy, x + dx) not in core:
                core.append((y + dy, x + dx))
        grown = set(core)
        if growth:
            neighbours = _NEIGHBOURS_4 if growth == 1 else _NEIGHBOURS_8
            grown |= {(y + a, x + b) for y, x in core for a, b in neighbours}
        return sorted(grown)


@dataclass(frozen=True)
class _Clusters:
    """Library indices and detector positions of drawn clusters."""

    index: np.ndarray
    y: np.ndarray
    x: np.ndarray

    def kept(self, keep: np.ndarray) -> _Clusters:
        return _Clusters(self.index[keep], self.y[keep], self.x[keep])

    def joined(self, other: _Clusters) -> _Clusters:
        return _Clusters(np.concatenate([self.index, other.index]),
                         np.concatenate([self.y, other.y]),
                         np.concatenate([self.x, other.x]))


def _draw_clusters(side: int, density: float, rng: np.random.Generator) -> _Clusters:
    count = int(rng.poisson(density * side * side))
    return _Clusters(rng.integers(_STAMPS_PER_MIX, size=count),
                     rng.integers(side, size=count), rng.integers(side, size=count))


def _stamp(det: np.ndarray, stamps: _Stamps, clusters: _Clusters, bits: np.uint8) -> None:
    """OR ``bits`` into ``det`` over every pixel of ``clusters``."""
    if clusters.index.size == 0:
        return
    side = det.shape[0]
    offsets = stamps.offsets[clusters.index]
    ys = clusters.y[:, None] + offsets[..., 0]
    xs = clusters.x[:, None] + offsets[..., 1]
    ok = ((np.arange(offsets.shape[1])[None, :] < stamps.counts[clusters.index][:, None])
          & (ys >= 0) & (ys < side) & (xs >= 0) & (xs < side))
    det[ys[ok], xs[ok]] |= bits


@lru_cache(maxsize=8)
def _centred_grid(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Pixel coordinates of ``shape`` relative to its centre (read-only)."""
    height, width = shape
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    y -= (height - 1) / 2
    x -= (width - 1) / 2
    y.setflags(write=False)
    x.setflags(write=False)
    return y, x


def _detector_coordinates(
    shape: tuple[int, int], side: int, theta_deg: float, offset: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Detector coordinates ``(u, v)`` of every mosaic pixel centre: the
    detector is rotated by ``theta_deg`` about the mosaic centre and shifted
    by ``offset`` detector pixels (the dither's sub-pixel phase). Detector
    pixel centres sit at integer coordinates."""
    y, x = _centred_grid((int(shape[0]), int(shape[1])))
    cos, sin = np.cos(np.radians(theta_deg)), np.sin(np.radians(theta_deg))
    u = (cos * x + sin * y) / DETECTOR_FACTOR + (side / 2 + float(offset[0]))
    v = (-sin * x + cos * y) / DETECTOR_FACTOR + (side / 2 + float(offset[1]))
    return u, v


def _detector_side(shape: tuple[int, int]) -> int:
    return int(np.ceil(np.hypot(*shape) / DETECTOR_FACTOR)) + 6


def stencil_to_mosaic(
    det: np.ndarray,
    shape: tuple[int, int],
    *,
    theta_deg: float,
    offset: tuple[float, float],
) -> np.ndarray:
    """A detector mask on the mosaic grid, as bilinear resampling spreads it.

    ``det`` holds one bit per band on the 0.30" detector grid. A mosaic pixel
    takes the OR of the four detector pixels of its interpolation stencil
    (:func:`_detector_coordinates`).
    """
    side = det.shape[0]
    u, v = _detector_coordinates(shape, side, theta_deg, offset)
    iu = np.clip(np.floor(u).astype(np.intp), 0, side - 2)
    iv = np.clip(np.floor(v).astype(np.intp), 0, side - 2)
    return det[iv, iu] | det[iv, iu + 1] | det[iv + 1, iu] | det[iv + 1, iu + 1]


def blank_where_all_invalid(exposures, coverage: np.ndarray) -> np.ndarray:
    """Band bits that are invalid in every exposure covering a pixel.

    ``exposures`` yields per-exposure invalid bits on the mosaic grid;
    exposure ``k`` covers the pixels with ``coverage > k``. Pixels without
    coverage are left alone: off-footprint sky is not what this models.
    """
    coverage = np.asarray(coverage)
    blank = np.where(coverage > 0, _ALL, np.uint8(0)).astype(np.uint8)
    for k, bits in enumerate(exposures):
        blank &= np.where(coverage > k, np.asarray(bits, dtype=np.uint8), _ALL)
    return blank


def draw_coverage_map(
    shape: tuple[int, int],
    n_exposures: int,
    rng: np.random.Generator,
    *,
    theta_deg: float,
) -> np.ndarray:
    """Exposure count per mosaic pixel of a field with ``n_exposures`` typically.

    Detector edges cross a field as straight seams parallel to a detector
    axis (:data:`SEAM_RATE_PER_PIXEL`); the count is one higher or one lower
    on a seam's far side, so the field's typical count stays the sky's.
    Never below one exposure.
    """
    y, x = _centred_grid((int(shape[0]), int(shape[1])))
    coverage = np.full(y.shape, int(n_exposures), dtype=np.int16)
    diagonal = float(np.hypot(*y.shape))
    for _ in range(int(rng.poisson(SEAM_RATE_PER_PIXEL * diagonal))):
        angle = np.radians(theta_deg + 90.0 * int(rng.integers(2)))
        across = -np.sin(angle) * x + np.cos(angle) * y
        side = 1.0 if rng.random() < 0.5 else -1.0
        step = 1 if rng.random() < 0.5 else -1
        coverage += step * (side * (across - rng.uniform(-0.5, 0.5) * diagonal) > 0).astype(np.int16)
    return np.maximum(coverage, 1).astype(np.int8)


class BlankPixelModel:
    """Draws the blank NISP pixels of a field for one release's rates."""

    def __init__(self, release: str) -> None:
        params = _all_params()
        if release not in blank_pixel_releases():
            raise ValueError(
                f"unknown blank-pixel release {release!r}; "
                f"choose from {blank_pixel_releases()}"
            )
        self.release = str(release)
        self.params = params[release]
        p = self.params
        # Fixed seeds: the libraries are part of the model, not of a draw.
        self._persistence = _Stamps(p["persistence_cluster_shapes"], seed=20261122)
        self._cosmic = _Stamps(p["cosmic_shapes"], seed=20261123)
        self._other = _Stamps(p["other_shapes"], seed=20261124)
        self._band_widths = np.asarray(p["band_widths"], dtype=np.float64)
        self._tilts = np.asarray(p["band_tilts_deg"], dtype=np.float64)
        self._tilt_p = np.asarray(p["band_tilt_probabilities"], dtype=np.float64)
        self._count_cdf = np.cumsum(p["exposure_count_shares"]) / np.sum(p["exposure_count_shares"])
        # Half the cores are blank at the level under four exposures: the
        # faintest of four exposure draws sits one sigma below the median.
        # A 0.30" detector pixel averages the star's peak down by
        # ``nisp_detector_peak_share``.
        dim = np.exp(-float(p["core_sigma_exposure"]))
        self._core_threshold = {
            band: float(p["star_core_level_e"][band]) * dim
            * (float(p["nisp_detector_peak_share"]) if band in BLANK_BAND_NAMES else 1.0)
            for band in BAND_NAMES
        }
        self._vis_growth = self._disc(int(p["vis_core_growth_px"]))
        sigma, shared = float(p["persistence_activity_sigma"]), float(p["persistence_activity_shared"])
        self._sigma_field = sigma * np.sqrt(shared)
        self._sigma_exposure = sigma * np.sqrt(1.0 - shared)

    # ------------------------------------------------------------------ #
    def exposure_count(self, depth_quantile: float) -> int:
        """NISP exposures over a scene at ``depth_quantile`` of the sky, from
        the shallowest (0) to the deepest (1): the release's measured area
        shares of each count, read as a quantile function."""
        q = min(max(float(depth_quantile), 0.0), 1.0)
        return int(min(np.searchsorted(self._count_cdf, q, side="right"), len(self._count_cdf) - 1)) + 1

    def star_core_level_e(self, band: str) -> float:
        """Peak (electrons on the 0.1" stack) at which half the star cores of
        ``band`` are blank under four exposures."""
        return float(self.params["star_core_level_e"][band])

    def extended_level_e(self, band: str) -> float:
        """Level above which smooth galaxy light of ``band`` is blank."""
        return float(self.params["extended_level_e"][band])

    @staticmethod
    def _disc(radius: int) -> np.ndarray:
        y, x = np.mgrid[-radius:radius + 1, -radius:radius + 1]
        return (y * y + x * x) <= radius * radius

    def core_exposure_bits(
        self,
        trigger: np.ndarray,
        star_trigger: np.ndarray | None,
        n_exposures: int,
        rng: np.random.Generator,
        *,
        theta_deg: float = 0.0,
    ) -> list:
        """Per-exposure invalid bits of the bright cores of a scene.

        ``trigger`` is the noise-free signal ``(H, W, 4)`` on the 0.1" stack
        (bands as :data:`BAND_NAMES`), ``star_trigger`` the stars' share of
        it; without it all light counts as stars.

        A NISP exposure sees the scene through its own 0.30" detector grid,
        rotated by ``theta_deg`` and at its dither's sub-pixel phase: a
        detector pixel measures the mean of the mosaic pixels it holds. Where
        stars hold at least half the light, a
        source's peak detector pixel is invalid once its value, times that
        source's scatter and that exposure's scatter, passes the star-core
        level, and its four neighbours go with it; other pixels join only
        ``core_saturation_factor`` times higher, which is how real cores
        grow from 80 to 170 mosaic pixels over a factor ten in brightness.
        A pixel of galaxy light is invalid above the extended level, alone
        and without scatter. The bilinear
        stencil spreads each invalid detector pixel over the mosaic. VIS
        cores, on their own 0.1" pixels, grow by a small disc instead.
        """
        trigger = np.asarray(trigger, dtype=np.float32)
        stars = None if star_trigger is None else np.asarray(star_trigger, dtype=np.float32)
        shape = trigger.shape[:2]
        empty = [np.zeros(shape, dtype=np.uint8) for _ in range(int(n_exposures))]
        bright = []
        candidate = np.zeros(shape, dtype=bool)
        for k, band in enumerate(BAND_NAMES):
            level = min(_CORE_CANDIDATE_SHARE * self._core_threshold[band], self.extended_level_e(band))
            bright.append(trigger[..., k] >= level)
            candidate |= bright[-1]
        if not candidate.any():
            return empty
        sources, count = label(candidate)
        p = self.params
        side = _detector_side(shape)
        source_scatter = np.exp(rng.normal(0.0, float(p["core_sigma_source"]), count + 1))
        out = []
        for bits in empty:
            exposure_scatter = np.exp(rng.normal(0.0, float(p["core_sigma_exposure"]), count + 1))
            scatter = (source_scatter * exposure_scatter)[sources].astype(np.float32)
            offset = rng.uniform(-1.0, 1.0, 2)
            u, v = _detector_coordinates(shape, side, theta_deg, offset)
            # The detector pixel holding each mosaic pixel, and how many it holds.
            held = (np.clip(np.rint(v).astype(np.intp), 0, side - 1) * side
                    + np.clip(np.rint(u).astype(np.intp), 0, side - 1)).ravel()
            holds = np.maximum(np.bincount(held, minlength=side * side), 1)

            def seen_by_detector(plane: np.ndarray, held=held, holds=holds) -> np.ndarray:
                total = np.bincount(held, weights=plane.ravel(), minlength=side * side)
                return (total / holds).reshape(side, side)

            for k, band in enumerate(BAND_NAMES):
                if not bright[k].any():
                    continue
                plane = trigger[..., k]
                if band in BLANK_BAND_NAMES:
                    seen = seen_by_detector(plane)
                    dominated = (np.ones(seen.shape, dtype=bool) if stars is None
                                 else seen_by_detector(stars[..., k]) >= 0.5 * seen)
                    scaled = np.where(dominated, seen_by_detector(plane * scatter), 0.0)
                    level = self._core_threshold[band]
                    # The star's peak pixel goes first; its neighbours join it as
                    # pixels of their own only once they are truly saturated.
                    star = ((scaled >= level) & (scaled >= maximum_filter(scaled, size=3))) | (
                        scaled >= float(p["core_saturation_factor"]) * level)
                    galaxy = ~dominated & (seen >= self.extended_level_e(band))
                    det = binary_dilation(star, structure=_PLUS) | galaxy
                    invalid = stencil_to_mosaic(det, shape, theta_deg=theta_deg, offset=offset)
                else:
                    dominated = np.ones(shape, dtype=bool) if stars is None else stars[..., k] >= 0.5 * plane
                    star = dominated & (plane * scatter >= self._core_threshold[band])
                    invalid = (binary_dilation(star, structure=self._vis_growth)
                               | (~dominated & (plane >= self.extended_level_e(band))))
                bits[invalid] |= _BIT[band]
            out.append(bits)
        return out

    def _tilt(self, rng: np.random.Generator) -> float:
        return float(rng.choice(self._tilts, p=self._tilt_p)
                     + rng.normal(0.0, float(self.params["band_tilt_jitter_deg"])))

    def _band_count(self, mean: float, rng: np.random.Generator) -> int:
        k = float(self.params["band_gamma_k"])
        return int(rng.poisson(mean * rng.gamma(k, 1.0 / k)))

    def _detector_bits(self, side: int, activity: float, rng: np.random.Generator) -> np.ndarray:
        """Invalid bits of one dither (its Y, J and H exposures) on a detector patch."""
        p = self.params
        det = np.zeros((side, side), dtype=np.uint8)
        # Persistence clusters: from band to band some drop out and new ones appear.
        density = float(p["persistence_cluster_density"]) * activity
        clusters = _draw_clusters(side, density, rng)
        for k, band in enumerate(_CHAIN):
            if k:
                keep = float(p["persistence_keep_j_to_h"] if k == 1 else p["persistence_keep_h_to_y"])
                clusters = clusters.kept(rng.random(clusters.index.size) < keep).joined(
                    _draw_clusters(side, density * (1.0 - keep), rng))
            _stamp(det, self._persistence, clusters, _BIT[band])
        for band in BLANK_BAND_NAMES:
            _stamp(det, self._cosmic,
                   _draw_clusters(side, float(p["cosmic_density"][band]), rng), _BIT[band])
        _stamp(det, self._other, _draw_clusters(side, float(p["other_density"]), rng), _NISP)
        # Persistence bands of sources elsewhere on the detector: random on the sky.
        length = float(p["band_length"])
        rate = float(p["band_rate"]) * (1.0 - float(p["sky_band_share"]))
        count = self._band_count(rate * side * (side + length) / length, rng)
        if count:
            v, u = _centred_grid((side, side))
            for _ in range(count):
                tilt = np.radians(self._tilt(rng))
                du = u - rng.uniform(-(side + length) / 2, (side + length) / 2)
                dv = v - rng.uniform(-side / 2, side / 2)
                along = du * np.cos(tilt) + dv * np.sin(tilt)
                across = -du * np.sin(tilt) + dv * np.cos(tilt)
                width = float(rng.choice(self._band_widths))
                det[(np.abs(across) <= width / 2) & (np.abs(along) <= length / 2)] |= _NISP
        return det

    def _sky_bands(self, shape: tuple[int, int], theta_deg: float, rng: np.random.Generator) -> list:
        """Bands whose source sits at a fixed place on the sky: every exposure
        lays its own band through that anchor, so they overlap in the mosaic
        and leave the long blank slivers seen under three or more exposures.
        Returns ``(x, y, width)`` in mosaic pixels relative to the centre."""
        p = self.params
        extent = float(np.hypot(*shape))
        length = float(p["band_length"]) * DETECTOR_FACTOR
        rate = float(p["band_rate"]) / DETECTOR_FACTOR * float(p["sky_band_share"])
        cos, sin = np.cos(np.radians(theta_deg)), np.sin(np.radians(theta_deg))
        anchors = []
        for _ in range(self._band_count(rate * extent * (extent + length) / length, rng)):
            along = rng.uniform(-(extent + length) / 2, (extent + length) / 2)
            across = rng.uniform(-extent / 2, extent / 2)
            width = DETECTOR_FACTOR * float(rng.choice(self._band_widths)) + DETECTOR_FACTOR
            anchors.append((along * cos - across * sin, along * sin + across * cos, width))
        return anchors

    def _sky_band_mask(self, shape, theta_deg: float, anchors: list,
                       rng: np.random.Generator) -> np.ndarray | None:
        if not anchors:
            return None
        y, x = _centred_grid(shape)
        length = float(self.params["band_length"]) * DETECTOR_FACTOR
        mask = np.zeros(shape, dtype=bool)
        for ax, ay, width in anchors:
            angle = np.radians(theta_deg + self._tilt(rng))
            # The trace lies to one side of its source.
            shift = (0.3 if rng.random() < 0.5 else -0.3) * length
            dx, dy = x - ax, y - ay
            along = dx * np.cos(angle) + dy * np.sin(angle) - shift
            across = -dx * np.sin(angle) + dy * np.cos(angle)
            mask |= (np.abs(across) <= width / 2) & (np.abs(along) <= length / 2)
        return mask

    def exposure_bits(
        self,
        shape: tuple[int, int],
        rng: np.random.Generator,
        *,
        theta_deg: float,
        activity: float = 1.0,
        anchors: list | None = None,
    ) -> np.ndarray:
        """Invalid bits (one per band of :data:`BLANK_BAND_NAMES`) of one dither
        on the mosaic grid."""
        shape = (int(shape[0]), int(shape[1]))
        det = self._detector_bits(_detector_side(shape), activity, rng)
        bits = stencil_to_mosaic(det, shape, theta_deg=theta_deg, offset=rng.uniform(-1.0, 1.0, 2))
        sky = self._sky_band_mask(shape, theta_deg, anchors or [], rng)
        if sky is not None:
            bits[sky] |= _NISP
        return bits

    def blank_mask(
        self,
        coverage: np.ndarray,
        rng: np.random.Generator,
        *,
        theta_deg: float | None = None,
        trigger: np.ndarray | None = None,
        star_trigger: np.ndarray | None = None,
    ) -> np.ndarray:
        """Blank pixels ``(H, W, 4)`` (bool, bands as :data:`BAND_NAMES`) of a
        field whose pixels are covered by ``coverage`` exposures. With the
        scene's ``trigger`` (and its ``star_trigger`` share, see
        :meth:`core_exposure_bits`) bright cores are blanked too."""
        coverage = np.asarray(coverage)
        shape = (int(coverage.shape[0]), int(coverage.shape[1]))
        count = int(coverage.max())
        theta = float(rng.uniform(0.0, 360.0)) if theta_deg is None else float(theta_deg)
        field_activity = float(np.exp(rng.normal(0.0, self._sigma_field)))
        anchors = self._sky_bands(shape, theta, rng)
        cores = (self.core_exposure_bits(trigger, star_trigger, count, rng, theta_deg=theta)
                 if trigger is not None else [None] * count)
        exposures = (
            self.exposure_bits(
                shape, rng, theta_deg=theta, anchors=anchors,
                activity=field_activity * float(np.exp(rng.normal(0.0, self._sigma_exposure))),
            ) | (0 if core is None else core)
            for core in cores
        )
        bits = blank_where_all_invalid(exposures, coverage)
        return np.stack([(bits & _BIT[band]) > 0 for band in BAND_NAMES], axis=-1)

    def draw(
        self,
        shape: tuple[int, int],
        depth_quantile: float,
        rng: np.random.Generator,
        *,
        trigger: np.ndarray | None = None,
        star_trigger: np.ndarray | None = None,
    ) -> np.ndarray:
        """Blank pixels ``(H, W, 4)`` of a scene at ``depth_quantile`` of the
        sky (0 shallowest, 1 deepest): its exposure count
        (:meth:`exposure_count`), the detector seams that step the count
        across the field, and the blanks of that coverage — with the bright
        cores of the scene when its ``trigger`` is given."""
        theta = float(rng.uniform(0.0, 360.0))
        coverage = draw_coverage_map(shape, self.exposure_count(depth_quantile), rng, theta_deg=theta)
        return self.blank_mask(coverage, rng, theta_deg=theta, trigger=trigger, star_trigger=star_trigger)
