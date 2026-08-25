"""Data-derived normalization diagnostics for the synthetic TNG population."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np

from euclid_polish.config import Config

DETECTION_SIGMA = 4.0
DETECTION_NPIXELS = 8
DETECTION_BOX_SIZE = 32
DETECTION_DEBLEND_LEVELS = 24
DETECTION_DEBLEND_CONTRAST = 0.003
TRUTH_MATCH_RADIUS_LR_PIX = 4.0


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _segment_centroids(
    plane: np.ndarray,
) -> tuple[list[tuple[float, float]], int]:
    """Detect positive sources and matched-significance negative islands."""
    from astropy.stats import SigmaClip
    from photutils.background import Background2D, MedianBackground
    from photutils.segmentation import (
        SourceCatalog,
        deblend_sources,
        detect_sources,
    )

    data = np.asarray(plane, dtype=np.float64)
    background = Background2D(
        data,
        (DETECTION_BOX_SIZE, DETECTION_BOX_SIZE),
        filter_size=(3, 3),
        # Photutils leaves this parameter unannotated and Pyright infers the
        # private default-sentinel type instead of the documented SigmaClip.
        sigma_clip=cast(Any, SigmaClip(sigma=3.0, maxiters=5)),
        bkg_estimator=MedianBackground(),
    )
    residual = data - background.background
    threshold = DETECTION_SIGMA * background.background_rms

    def segments(sign: int):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            segmentation = detect_sources(
                sign * residual,
                threshold,
                npixels=DETECTION_NPIXELS,
            )
            if segmentation is None:
                return None
            return deblend_sources(
                sign * residual,
                segmentation,
                npixels=DETECTION_NPIXELS,
                nlevels=DETECTION_DEBLEND_LEVELS,
                contrast=DETECTION_DEBLEND_CONTRAST,
                progress_bar=False,
            )

    positive = segments(1)
    negative = segments(-1)
    if positive is None:
        return [], 0 if negative is None else int(negative.nlabels)
    catalog = SourceCatalog(residual, positive)
    centroids = [
        (float(x), float(y))
        for x, y in zip(catalog.xcentroid, catalog.ycentroid, strict=True)
        if np.isfinite(x) and np.isfinite(y)
    ]
    return centroids, 0 if negative is None else int(negative.nlabels)


def _match_truth(
    centroids: list[tuple[float, float]],
    truth: list[dict[str, Any]],
    shape: tuple[int, int],
) -> tuple[int, int, int]:
    """Greedily match unique detections to unique HR source positions."""
    scale = float(Config.DEFAULT_PIXEL_SCALE / Config.VIS_PIXEL_SCALE_ARCSEC)
    height, width = shape
    candidates: list[tuple[float, int, int]] = []
    valid_truth: list[dict[str, Any]] = []
    for row in truth:
        x = _finite(row.get("x_pix"))
        y = _finite(row.get("y_pix"))
        if x is None or y is None:
            continue
        x *= scale
        y *= scale
        if not (0 <= x < width and 0 <= y < height):
            continue
        truth_index = len(valid_truth)
        valid_truth.append(row)
        for detection_index, (det_x, det_y) in enumerate(centroids):
            distance = float(np.hypot(det_x - x, det_y - y))
            if distance <= TRUTH_MATCH_RADIUS_LR_PIX:
                candidates.append((distance, detection_index, truth_index))

    used_detections: set[int] = set()
    used_truth: set[int] = set()
    matched_galaxies = 0
    matched_stars = 0
    for _distance, detection_index, truth_index in sorted(candidates):
        if detection_index in used_detections or truth_index in used_truth:
            continue
        used_detections.add(detection_index)
        used_truth.add(truth_index)
        kind = str(valid_truth[truth_index].get("type", "unknown"))
        if kind == "star":
            matched_stars += 1
        elif kind in {"galaxy", "lens"}:
            matched_galaxies += 1
    truth_galaxies = sum(
        str(row.get("type", "unknown")) in {"galaxy", "lens"}
        for row in valid_truth
    )
    return matched_galaxies, matched_stars, truth_galaxies


@dataclass
class DetectionAccumulator:
    """Streaming per-field VIS detections, optionally matched to truth."""

    positive: list[int] = field(default_factory=list)
    negative: list[int] = field(default_factory=list)
    matched_galaxies: list[int] = field(default_factory=list)
    matched_stars: list[int] = field(default_factory=list)
    truth_galaxies: list[int] = field(default_factory=list)

    def add(
        self,
        vis_plane: np.ndarray,
        truth: list[dict[str, Any]] | None = None,
    ) -> None:
        centroids, negative = _segment_centroids(vis_plane)
        self.positive.append(len(centroids))
        self.negative.append(negative)
        if truth is None:
            self.matched_galaxies.append(0)
            self.matched_stars.append(0)
            self.truth_galaxies.append(0)
            return
        galaxies, stars, truth_count = _match_truth(
            centroids, truth, np.asarray(vis_plane).shape
        )
        self.matched_galaxies.append(galaxies)
        self.matched_stars.append(stars)
        self.truth_galaxies.append(truth_count)

    def payload(self) -> dict[str, Any]:
        return {
            "positive": self.positive,
            "negative": self.negative,
            "matched_galaxies": self.matched_galaxies,
            "matched_stars": self.matched_stars,
            "truth_galaxies": self.truth_galaxies,
        }


def detection_payload(
    synthetic: DetectionAccumulator,
    real: DetectionAccumulator,
) -> dict[str, Any]:
    return {
        "settings": {
            "band": "VIS",
            "threshold_sigma": DETECTION_SIGMA,
            "minimum_connected_pixels": DETECTION_NPIXELS,
            "background_box_pixels": DETECTION_BOX_SIZE,
            "deblend_levels": DETECTION_DEBLEND_LEVELS,
            "deblend_contrast": DETECTION_DEBLEND_CONTRAST,
            "negative_image_correction": True,
            "truth_match_radius_lr_pixels": TRUTH_MATCH_RADIUS_LR_PIX,
        },
        "synthetic": synthetic.payload(),
        "real": real.payload(),
    }
