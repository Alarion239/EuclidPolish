"""Field-scale LR variations absent from compact empirical PSF stamps.

The empirical ePSFs describe ordinary point sources well, but two effects live
on scales larger than a training cutout:

* very bright stars thousands of pixels outside a field can leave a long,
  nearly straight diffraction wing across most or all of the cutout;
* overlapping pointings can have visibly different background-noise levels.

Both helpers here are deterministic for a supplied NumPy generator.  They
operate on the delivered 0.10-arcsec LR grid and never modify the clean HR
target.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DistantStarWing:
    """One field-spanning wing cast by a star far outside the LR cutout."""

    angle_rad: float
    offset_lr_pix: float
    source_distance_lr_pix: float
    amplitude_sigma: float
    width_lr_pix: float


def draw_distant_star_wings(
    shape: tuple[int, int],
    rng: np.random.Generator,
    *,
    probability: float,
    source_distance_min_lr_pix: float,
    source_distance_max_lr_pix: float,
    amplitude_sigma_min: float,
    amplitude_sigma_max: float,
    width_min_lr_pix: float,
    width_max_lr_pix: float,
) -> tuple[DistantStarWing, ...]:
    """Possibly draw one wing whose parent star lies far beyond the cutout."""
    if rng.random() >= float(probability):
        return ()
    height, width = (int(shape[0]), int(shape[1]))
    angle = float(rng.uniform(0.0, np.pi))
    half_extent = 0.5 * (
        abs(np.sin(angle)) * width + abs(np.cos(angle)) * height
    )
    return (DistantStarWing(
        angle_rad=angle,
        offset_lr_pix=float(rng.uniform(-0.9, 0.9) * half_extent),
        source_distance_lr_pix=float(rng.uniform(
            source_distance_min_lr_pix, source_distance_max_lr_pix,
        )),
        amplitude_sigma=float(rng.uniform(
            amplitude_sigma_min, amplitude_sigma_max,
        )),
        width_lr_pix=float(rng.uniform(
            width_min_lr_pix, width_max_lr_pix,
        )),
    ),)


def add_distant_star_wings(
    image_e: np.ndarray,
    wings: tuple[DistantStarWing, ...],
    *,
    local_sigma_e: float,
) -> np.ndarray:
    """Add off-field-star wings to one delivered LR band.

    Across each wing the profile is Gaussian. Along it, brightness follows
    inverse distance to a parent star 1,000--5,000 pixels away, and therefore
    changes only slightly over one cutout.
    """
    if not wings or local_sigma_e <= 0.0:
        return image_e
    out = np.asarray(image_e, dtype=np.float32).copy()
    height, width = out.shape
    yy, xx = np.indices((height, width), dtype=np.float32)
    cy = (height - 1) / 2.0
    cx = (width - 1) / 2.0

    for wing in wings:
        sin_a = float(np.sin(wing.angle_rad))
        cos_a = float(np.cos(wing.angle_rad))
        dx = xx - cx
        dy = yy - cy
        across = -sin_a * dx + cos_a * dy - wing.offset_lr_pix
        along = cos_a * dx + sin_a * dy
        cross_profile = np.exp(
            -0.5 * (across / wing.width_lr_pix) ** 2
        )
        radial_distance = np.maximum(
            wing.source_distance_lr_pix - along, 1.0,
        )
        long_profile = wing.source_distance_lr_pix / radial_distance
        out += np.float32(local_sigma_e * wing.amplitude_sigma) * (
            cross_profile * long_profile
        ).astype(np.float32)
    return out


def draw_noise_scale_map(
    shape: tuple[int, int],
    rng: np.random.Generator,
    *,
    global_scale_min: float,
    global_scale_max: float,
    region_probability: float,
    region_fraction_min: float,
    region_fraction_max: float,
    region_scale_min: float,
    region_scale_max: float,
) -> np.ndarray:
    """Draw a field-wide noise scale plus an oversized rotated rectangle.

    The rectangle is much longer than the cutout and therefore appears as one
    pointing/intersection region with a straight rotated boundary.  Rejection
    sampling keeps its visible coverage between the requested fractions.
    """
    height, width = (int(shape[0]), int(shape[1]))
    if height < 1 or width < 1:
        raise ValueError(f"shape must be positive, got {shape}")
    global_scale = float(rng.uniform(global_scale_min, global_scale_max))
    scale_map = np.full((height, width), global_scale, dtype=np.float32)
    if rng.random() >= float(region_probability):
        return scale_map

    yy, xx = np.indices((height, width), dtype=np.float32)
    cy0 = (height - 1) / 2.0
    cx0 = (width - 1) / 2.0
    diagonal = float(np.hypot(height, width))
    target_fraction = float(
        rng.uniform(region_fraction_min, region_fraction_max)
    )
    best_mask = None
    best_distance = np.inf

    for _ in range(64):
        angle = float(rng.uniform(0.0, np.pi))
        sin_a = float(np.sin(angle))
        cos_a = float(np.cos(angle))
        # An offset long strip gives one pointing-like region, commonly with
        # only one boundary crossing the cutout.
        offset = float(rng.uniform(-0.45, 0.45) * diagonal)
        centre_x = cx0 - sin_a * offset
        centre_y = cy0 + cos_a * offset
        half_width = float(rng.uniform(0.12, 0.42) * min(height, width))
        dx = xx - centre_x
        dy = yy - centre_y
        along = cos_a * dx + sin_a * dy
        across = -sin_a * dx + cos_a * dy
        mask = (
            (np.abs(along) <= 1.5 * diagonal)
            & (np.abs(across) <= half_width)
        )
        fraction = float(mask.mean())
        distance = abs(fraction - target_fraction)
        if distance < best_distance:
            best_mask = mask
            best_distance = distance
        if region_fraction_min <= fraction <= region_fraction_max:
            best_mask = mask
            break

    if best_mask is not None and np.any(best_mask):
        region_scale = float(rng.uniform(region_scale_min, region_scale_max))
        scale_map[best_mask] *= np.float32(region_scale)
    return scale_map
