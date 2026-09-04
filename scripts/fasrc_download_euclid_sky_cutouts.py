#!/usr/bin/env python
"""Download multi-band Euclid sky cutouts for round-trip training.

The HST path (``scripts/fasrc_generate_hst_tfrecords.py``) gives us
forward-modelled HST→Euclid pairs with HR ground truth. Round-trip
training adds a self-supervised signal on *real* Euclid observations:
``loss = |Conv(M(LR_real)) - LR_real|`` where ``Conv`` is the
deterministic Euclid forward operator (PSF + rebin, no noise).

This script handles the data-acquisition half of that path:

  1. Generate ``N`` random sky positions inside a circular footprint
     (default: 2° radius around RA=270°, Dec=66° — a deep Euclid
     coverage region). Positions outside Euclid coverage are filtered
     downstream when the cutout service can't find a covering mosaic
     tile, so we don't need an explicit footprint mask.
  2. Write the positions to ``$output_dir/sky_positions.csv`` (columns
     ``id, ra, dec``).
  3. For each position, fetch all four Euclid bands (VIS + NISP Y/J/H)
     via :func:`euclid_polish.catalog.downloader.fetch_cutout_at` and
     bundle them into a single multi-HDU FITS at
     ``$output_dir/cutouts/sky_NNNN.fits`` with one ``ImageHDU`` per
     band (``EXTNAME`` in ``VIS``, ``Y_E``, ``J_E``, ``H_E``).

A position is only written to disk if **all four bands** succeed —
half-bundled files would force the downstream TFRecord step to grep
for coverage across four directories. Better to lose a position than
to silently corrupt the 4-band invariant.

Migration notes — the on-disk layout used to be
``$output_dir/cutouts/<band>/star_NNNN_<size>.fits`` with a catalogue
called ``stars.csv``. To re-download cleanly under the new layout,
remove the old artefacts first::

    rm -rf $DATA_DIR/euclid_sky/cutouts/{VIS,Y_E,J_E,H_E}
    rm -f  $DATA_DIR/euclid_sky/stars.csv

Then re-run this script.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import os
import re
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord, SkyOffsetFrame
from astropy.io import fits
from astropy.io.fits.verify import VerifyError
from astropy.wcs import WCS

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import contextlib

from euclid_polish.catalog.downloader import (
    DownloadConfig,
    download_one_cutout,
    fetch_cutout_at,
    query_mosaic_tiles,
)
from euclid_polish.config import Config
from euclid_polish.observability.reporter import Reporter
from euclid_polish.web.helpers.archive_fields import (
    compute_collection_fingerprint,
    compute_plan_fingerprint,
)

# Sky-catalog location, catalogue filename, and cutouts subdir now live in
# Config — see Config.EUCLID_SKY_DIR, Config.EuclidSky.SKY_CATALOG_FILENAME,
# and Config.EuclidSky.CUTOUTS_SUBDIR.

SAMPLING_MANIFEST_VERSION = 1
SAMPLING_MANIFEST_NAME = "vis_noise_sampling_manifest.json"
VIS_NOISE_SAMPLING_SUBDIR = "vis_noise_samples"
VIS_NOISE_DEFAULT_WORKERS = 1
VIS_NOISE_DOWNLOAD_ATTEMPTS = 3
VIS_NOISE_DOWNLOAD_RETRY_DELAYS_SECONDS = (2.0, 5.0)
VIS_NOISE_MAX_OVERSIZE_FRACTION = 0.01
ARCHIVE_FIELDS_MANIFEST_VERSION = 1
ARCHIVE_FIELDS_MANIFEST_KIND = "euclid_archive_fields"
ARCHIVE_FIELDS_SOURCE_KIND = "euclid_vis_noise_sampling"
ARCHIVE_FIELDS_POSITIONS_PER_PARENT = 5
ARCHIVE_FIELDS_VIS_PIXELS = 256
ARCHIVE_FIELDS_OFFSET_ARCSEC = 80.0
# Five tiles need 2 * (80" centre offset + 12.8" tile half-side) =
# 185.6".  Request 192.0" to leave 3.2" of rounding/WCS safety per edge
# while transferring only 56.25% as many NISP pixels as a 256.0" source field.
ARCHIVE_FIELDS_PARENT_DOWNLOAD_VIS_PIXELS = 1920
ARCHIVE_FIELDS_MINIMUM_PARENT_VIS_PIXELS = 1856
ARCHIVE_FIELDS_WCS_TOLERANCE_PIXELS = 1.0e-3
ARCHIVE_FIELDS_DOWNLOAD_ATTEMPTS = 3
ARCHIVE_FIELDS_DOWNLOAD_RETRY_DELAYS_SECONDS = (2.0, 5.0)
ARCHIVE_FIELDS_PATTERN = (
    ("center", 0.0, 0.0),
    ("southwest", -ARCHIVE_FIELDS_OFFSET_ARCSEC, -ARCHIVE_FIELDS_OFFSET_ARCSEC),
    ("southeast", ARCHIVE_FIELDS_OFFSET_ARCSEC, -ARCHIVE_FIELDS_OFFSET_ARCSEC),
    ("northwest", -ARCHIVE_FIELDS_OFFSET_ARCSEC, ARCHIVE_FIELDS_OFFSET_ARCSEC),
    ("northeast", ARCHIVE_FIELDS_OFFSET_ARCSEC, ARCHIVE_FIELDS_OFFSET_ARCSEC),
)
Q1_SUPPORT_REGIONS = (
    ("EDF-N", 269.733, 66.018, 6.0),
    ("EDF-F", 61.241, -48.423, 6.0),
    ("EDF-S", 52.932, -28.088, 6.0),
)
_SPHERE_AREA_DEG2 = 4.0 * math.pi * (180.0 / math.pi) ** 2
_FOV_POLYGON_RE = re.compile(
    r"^\s*POLYGON(?:\s+ICRS)?\s+(.+?)\s*$", re.IGNORECASE,
)
_FOV_POLYGON_CALL_RE = re.compile(
    r"^\s*POLYGON\s*\(\s*['\"]?ICRS['\"]?\s*,\s*(.+?)\s*\)\s*$",
    re.IGNORECASE,
)
_FOV_COORDINATE_TUPLE_RE = re.compile(
    r"^\s*\(\s*(.+?)\s*\)\s*$",
)


def _worker_count(sampling_mode: str, requested: int | None) -> int:
    """Resolve a safe mode-specific download concurrency.

    The archive's cutout endpoint proved unreliable when several large VIS
    requests arrived together, while the exact same requests succeeded
    serially.  Keep the legacy four-band downloader at eight workers, but make
    independent VIS-noise sampling serial unless a caller explicitly opts in.
    """
    if requested is None:
        return (
            VIS_NOISE_DEFAULT_WORKERS
            if sampling_mode in {"star-support", "archive-fields"}
            else 8
        )
    return max(1, int(requested))


def _vis_noise_shape_matches(
    actual_shape: Iterable[int], requested_pixels: int,
) -> bool:
    """Accept only tiny undersize or WCS-driven oversize SODA rasters.

    MER cutouts are requested by angular radius, so a nominal 2560-pixel VIS
    field can contain a few more detector pixels where the local mosaic WCS is
    finer than 0.10 arcsec/pixel.  A one-percent oversize is harmless because
    the fitter centrally crops to its complete 256-pixel tile grid.  Undersize
    remains limited to the repository's stricter generic tolerance because it
    represents missing requested sky rather than extra sky.
    """
    try:
        dimensions = tuple(int(value) for value in actual_shape)
        requested = int(requested_pixels)
    except (TypeError, ValueError):
        return False
    if len(dimensions) != 2 or requested <= 0:
        return False
    undersize_tolerance = int(Config.Matching.DOWNLOAD_SIZE_TOL_PIXELS)
    oversize_tolerance = max(
        undersize_tolerance,
        int(math.ceil(requested * VIS_NOISE_MAX_OVERSIZE_FRACTION)),
    )
    return all(
        -undersize_tolerance <= actual - requested <= oversize_tolerance
        for actual in dimensions
    )


def _unit_vectors(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    """Return ICRS unit vectors for matching arrays of angles in degrees."""
    ra = np.deg2rad(np.asarray(ra_deg, dtype=np.float64))
    dec = np.deg2rad(np.asarray(dec_deg, dtype=np.float64))
    cos_dec = np.cos(dec)
    return np.column_stack((cos_dec * np.cos(ra), cos_dec * np.sin(ra), np.sin(dec)))


def _angles_from_unit(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vectors = np.asarray(vectors, dtype=np.float64)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    ra = np.rad2deg(np.arctan2(vectors[:, 1], vectors[:, 0])) % 360.0
    dec = np.rad2deg(np.arcsin(np.clip(vectors[:, 2], -1.0, 1.0)))
    return ra, dec


def _angular_separation_deg(
    ra_deg: float,
    dec_deg: float,
    other_ra_deg: np.ndarray,
    other_dec_deg: np.ndarray,
) -> np.ndarray:
    centre = _unit_vectors(np.asarray([ra_deg]), np.asarray([dec_deg]))[0]
    other = _unit_vectors(other_ra_deg, other_dec_deg)
    return np.rad2deg(np.arccos(np.clip(other @ centre, -1.0, 1.0)))


def _load_star_support(stars_csv: str | os.PathLike[str]) -> pd.DataFrame:
    """Read unique finite star coordinates used only as coverage support.

    Bright-star density is deliberately discarded downstream: occupied
    equal-solid-angle cells each contribute one support point, irrespective of
    how many calibration stars happened to land in the cell.
    """
    rows: list[tuple[float, float]] = []
    seen: set[tuple[float, float]] = set()
    with open(stars_csv, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                ra = float(row["ra"]) % 360.0
                dec = float(row["dec"])
            except (KeyError, TypeError, ValueError):
                continue
            if not (np.isfinite(ra) and np.isfinite(dec) and -90.0 <= dec <= 90.0):
                continue
            key = (ra, dec)
            if key in seen:
                continue
            seen.add(key)
            rows.append(key)
    if not rows:
        raise ValueError(f"No finite star coordinates in {stars_csv}")
    return pd.DataFrame(rows, columns=["ra", "dec"])


def _assign_q1_regions(stars: pd.DataFrame) -> pd.DataFrame:
    """Keep support in the three released Q1 deep fields and label it."""
    ra = stars["ra"].to_numpy(dtype=np.float64)
    dec = stars["dec"].to_numpy(dtype=np.float64)
    distance = np.full(len(stars), np.inf, dtype=np.float64)
    labels = np.full(len(stars), "", dtype=object)
    for name, centre_ra, centre_dec, radius_deg in Q1_SUPPORT_REGIONS:
        candidate = _angular_separation_deg(centre_ra, centre_dec, ra, dec)
        use = (candidate < distance) & (candidate <= radius_deg)
        labels[use] = name
        distance[use] = candidate[use]
    keep = labels != ""
    if not np.any(keep):
        raise ValueError("Star catalogue has no support in the Q1 deep fields")
    out = stars.loc[keep].copy().reset_index(drop=True)
    out["field"] = labels[keep]
    return out


def _equal_area_support(
    stars: pd.DataFrame,
    *,
    cell_area_deg2: float = 0.04,
) -> pd.DataFrame:
    """Collapse stars into occupied cylindrical equal-area sky cells.

    The grid is uniform in RA and ``sin(dec)``.  Every occupied cell has one
    vote in clustering, preventing calibration-star surface density from
    becoming a sky-area weight.
    """
    if not np.isfinite(cell_area_deg2) or cell_area_deg2 <= 0.0:
        raise ValueError("cell_area_deg2 must be finite and positive")
    n_total = max(8, int(round(_SPHERE_AREA_DEG2 / float(cell_area_deg2))))
    n_lat = max(2, int(round(math.sqrt(n_total / 2.0))))
    n_lon = 2 * n_lat
    ra = stars["ra"].to_numpy(dtype=np.float64) % 360.0
    dec = stars["dec"].to_numpy(dtype=np.float64)
    lon_bin = np.minimum(n_lon - 1, np.floor(ra / 360.0 * n_lon).astype(int))
    equal_area_y = (np.sin(np.deg2rad(dec)) + 1.0) / 2.0
    lat_bin = np.minimum(n_lat - 1, np.floor(equal_area_y * n_lat).astype(int))
    grouped: dict[tuple[str, int, int], list[int]] = {}
    for index, (field, iy, ix) in enumerate(
        zip(stars["field"], lat_bin, lon_bin, strict=True)
    ):
        grouped.setdefault((str(field), int(iy), int(ix)), []).append(index)
    records: list[dict[str, Any]] = []
    vectors = _unit_vectors(ra, dec)
    for (field, iy, ix), indices in sorted(grouped.items()):
        vector = np.mean(vectors[indices], axis=0)
        vector /= np.linalg.norm(vector)
        cell_ra, cell_dec = _angles_from_unit(vector[None, :])
        records.append({
            "field": field,
            "lat_bin": iy,
            "lon_bin": ix,
            "ra": float(cell_ra[0]),
            "dec": float(cell_dec[0]),
            "star_count": len(indices),
        })
    return pd.DataFrame.from_records(records)


def _spherical_kmeans(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    n_clusters: int,
    *,
    seed: int,
    max_iter: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic K-means++ on unit vectors using angular chord distance."""
    vectors = _unit_vectors(ra_deg, dec_deg)
    count = len(vectors)
    if not 1 <= int(n_clusters) <= count:
        raise ValueError(f"n_clusters must be in [1, {count}]")
    rng = np.random.default_rng(int(seed))
    chosen = [int(rng.integers(0, count))]
    closest = np.maximum(0.0, 2.0 - 2.0 * (vectors @ vectors[chosen[0]]))
    while len(chosen) < int(n_clusters):
        weights = closest.copy()
        weights[chosen] = 0.0
        total = float(weights.sum())
        if total <= 0.0:
            candidate = next(i for i in range(count) if i not in chosen)
        else:
            candidate = int(rng.choice(count, p=weights / total))
        chosen.append(candidate)
        distance = np.maximum(0.0, 2.0 - 2.0 * (vectors @ vectors[candidate]))
        closest = np.minimum(closest, distance)
    centres = vectors[chosen].copy()
    labels = np.full(count, -1, dtype=np.int64)
    for _ in range(max(1, int(max_iter))):
        updated_labels = np.argmax(vectors @ centres.T, axis=1)
        if np.array_equal(updated_labels, labels):
            break
        labels = updated_labels
        for cluster in range(int(n_clusters)):
            members = vectors[labels == cluster]
            if not len(members):
                nearest_dot = np.max(vectors @ centres.T, axis=1)
                centres[cluster] = vectors[int(np.argmin(nearest_dot))]
                continue
            centre = np.mean(members, axis=0)
            norm = float(np.linalg.norm(centre))
            centres[cluster] = centre / norm if norm > 0.0 else members[0]
    return centres, labels


def _allocate_clusters(counts: Mapping[str, int], total: int) -> dict[str, int]:
    nonempty = {name: int(count) for name, count in counts.items() if count > 0}
    if total < len(nonempty):
        raise ValueError("n_clusters must cover every non-empty Q1 field")
    allocation = dict.fromkeys(nonempty, 1)
    remaining = int(total) - len(nonempty)
    weight_total = sum(nonempty.values())
    quotas = {name: remaining * count / weight_total for name, count in nonempty.items()}
    for name, quota in quotas.items():
        add = min(nonempty[name] - 1, int(math.floor(quota)))
        allocation[name] += add
    while sum(allocation.values()) < int(total):
        choices = [name for name in nonempty if allocation[name] < nonempty[name]]
        if not choices:
            break
        name = max(
            choices,
            key=lambda item: (quotas[item] - math.floor(quotas[item]), nonempty[item], item),
        )
        allocation[name] += 1
        quotas[name] = math.floor(quotas[name])
    if sum(allocation.values()) != int(total):
        raise ValueError("Requested more clusters than occupied support cells")
    return allocation


def build_star_support_anchors(
    stars_csv: str | os.PathLike[str],
    *,
    n_clusters: int = 44,
    cell_area_deg2: float = 0.04,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build area-weighted spherical medoid anchors from saved star support."""
    stars = _assign_q1_regions(_load_star_support(stars_csv))
    support = _equal_area_support(stars, cell_area_deg2=cell_area_deg2)
    cell_counts = Counter(str(value) for value in support["field"])
    allocation = _allocate_clusters(cell_counts, int(n_clusters))
    anchors: list[dict[str, Any]] = []
    anchor_index = 0
    for field, *_unused in Q1_SUPPORT_REGIONS:
        cells = support[support["field"] == field].reset_index(drop=True)
        if cells.empty:
            continue
        k = allocation[field]
        centres, labels = _spherical_kmeans(
            cells["ra"].to_numpy(), cells["dec"].to_numpy(), k,
            seed=int(seed) + 104729 * (anchor_index + 1),
        )
        cell_vectors = _unit_vectors(cells["ra"].to_numpy(), cells["dec"].to_numpy())
        for local_index in range(k):
            member_indices = np.flatnonzero(labels == local_index)
            dots = cell_vectors[member_indices] @ centres[local_index]
            medoid_index = int(member_indices[int(np.argmax(dots))])
            medoid = cells.iloc[medoid_index]
            members = cells.iloc[member_indices]
            anchors.append({
                "anchor_id": f"{field}-{local_index:03d}",
                "field": field,
                "ra": float(medoid["ra"]),
                "dec": float(medoid["dec"]),
                "support_cell_count": int(len(member_indices)),
                "support_star_count": int(members["star_count"].sum()),
            })
        anchor_index += 1
    return pd.DataFrame.from_records(anchors), stars


def _uniform_spherical_disk_one(
    ra_deg: float,
    dec_deg: float,
    radius_deg: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """One exactly equal-solid-angle random draw in a spherical cap."""
    if not 0.0 <= float(radius_deg) < 180.0:
        raise ValueError("radius_deg must be in [0, 180)")
    centre = _unit_vectors(np.asarray([ra_deg]), np.asarray([dec_deg]))[0]
    ra = math.radians(float(ra_deg))
    dec = math.radians(float(dec_deg))
    east = np.asarray([-math.sin(ra), math.cos(ra), 0.0])
    north = np.asarray([
        -math.sin(dec) * math.cos(ra),
        -math.sin(dec) * math.sin(ra),
        math.cos(dec),
    ])
    cos_radius = math.cos(math.radians(float(radius_deg)))
    cos_theta = 1.0 - float(rng.random()) * (1.0 - cos_radius)
    sin_theta = math.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))
    bearing = 2.0 * math.pi * float(rng.random())
    vector = (
        cos_theta * centre
        + sin_theta * (math.cos(bearing) * north + math.sin(bearing) * east)
    )
    out_ra, out_dec = _angles_from_unit(vector[None, :])
    return float(out_ra[0]), float(out_dec[0])


def propose_star_support_samples(
    anchors: pd.DataFrame,
    stars: pd.DataFrame,
    *,
    samples_per_anchor: int = 1,
    candidates_per_sample: int = 24,
    jitter_radius_deg: float = 0.15,
    avoid_star_arcsec: float = 30.0,
    minimum_sample_separation_arcmin: float = 6.5,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate deterministic spherical candidates away from exact stars."""
    if samples_per_anchor < 1 or candidates_per_sample < 1:
        raise ValueError("sample and candidate counts must be positive")
    rng = np.random.default_rng(int(seed))
    star_ra = stars["ra"].to_numpy(dtype=np.float64)
    star_dec = stars["dec"].to_numpy(dtype=np.float64)
    accepted_centres: list[tuple[float, float]] = []
    records: list[dict[str, Any]] = []
    for anchor in anchors.to_dict("records"):
        for slot in range(int(samples_per_anchor)):
            emitted = 0
            attempts = 0
            while emitted < int(candidates_per_sample):
                attempts += 1
                if attempts > max(1000, 100 * int(candidates_per_sample)):
                    raise ValueError(
                        f"Could not draw candidates away from stars near {anchor['anchor_id']}"
                    )
                ra, dec = _uniform_spherical_disk_one(
                    float(anchor["ra"]), float(anchor["dec"]),
                    float(jitter_radius_deg), rng,
                )
                nearest_star = float(np.min(
                    _angular_separation_deg(ra, dec, star_ra, star_dec)
                )) * 3600.0
                if nearest_star < float(avoid_star_arcsec):
                    continue
                if accepted_centres:
                    previous_ra = np.asarray([item[0] for item in accepted_centres])
                    previous_dec = np.asarray([item[1] for item in accepted_centres])
                    nearest_sample = float(np.min(
                        _angular_separation_deg(ra, dec, previous_ra, previous_dec)
                    )) * 60.0
                    if nearest_sample < float(minimum_sample_separation_arcmin):
                        continue
                records.append({
                    "anchor_id": str(anchor["anchor_id"]),
                    "field": str(anchor["field"]),
                    "slot": slot,
                    "candidate_rank": emitted,
                    "ra": ra,
                    "dec": dec,
                    "nearest_support_star_arcsec": nearest_star,
                })
                # Only final samples need global non-overlap. Candidate variants
                # for one slot may overlap, so do not reserve them here.
                emitted += 1
            if records:
                first = next(
                    row for row in records
                    if row["anchor_id"] == anchor["anchor_id"] and row["slot"] == slot
                )
                accepted_centres.append((float(first["ra"]), float(first["dec"])))
    return pd.DataFrame.from_records(records)


def _row_value(row: Any, name: str, default: Any = None) -> Any:
    for candidate in (name, name.lower(), name.upper()):
        try:
            value = row[candidate]
        except (KeyError, IndexError, TypeError):
            continue
        if value is not None and not np.ma.is_masked(value):
            return np.asarray(value).item() if np.asarray(value).ndim == 0 else value
    return default


def _text(value: Any) -> str:
    if value is None or np.ma.is_masked(value):
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    return str(value).strip()


def _optional_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _polygon_from_fov(value: Any) -> list[tuple[float, float]] | None:
    """Parse a polygon returned by the Euclid TAP service.

    Depending on the TAP serialization path, ``sedm.mosaic_product.fov`` is
    returned either as STC-S (``POLYGON ICRS ...``) or as a bare
    parenthesized coordinate tuple (``(ra, dec, ...)``).  Both forms describe
    the same polygon and must pass through the same validation below.
    """
    text = _text(value)
    match = (
        _FOV_POLYGON_CALL_RE.match(text)
        or _FOV_POLYGON_RE.match(text)
        or _FOV_COORDINATE_TUPLE_RE.match(text)
    )
    if match is None:
        return None
    tokens = match.group(1).replace(",", " ").split()
    try:
        numbers = [float(token) for token in tokens]
    except ValueError:
        return None
    if len(numbers) < 6 or len(numbers) % 2:
        return None
    points = list(zip(numbers[::2], numbers[1::2], strict=True))
    if any(
        not math.isfinite(ra)
        or not math.isfinite(dec)
        or not -90.0 <= dec <= 90.0
        for ra, dec in points
    ):
        return None
    if len(points) > 3:
        first = _unit_vectors(np.asarray([points[0][0]]), np.asarray([points[0][1]]))[0]
        last = _unit_vectors(np.asarray([points[-1][0]]), np.asarray([points[-1][1]]))[0]
        if float(first @ last) > 1.0 - 1e-14:
            points.pop()
    return points if len(points) >= 3 else None


def _circle_clearance_in_convex_fov_deg(
    fov: Any,
    ra: float,
    dec: float,
) -> float | None:
    """Return spherical clearance to a convex FOV boundary, or ``None``.

    The known-working TAP predicate only establishes an intersection.  This
    second, client-side test treats each polygon edge as a great-circle
    half-space.  It both proves that the requested centre is inside every
    half-space and rejects non-convex/unparseable footprints rather than
    silently accepting a boundary cutout.
    """
    polygon = _polygon_from_fov(fov)
    if polygon is None:
        return None
    vertices = _unit_vectors(
        np.asarray([point[0] for point in polygon]),
        np.asarray([point[1] for point in polygon]),
    )
    centre = _unit_vectors(np.asarray([ra]), np.asarray([dec]))[0]
    normals: list[np.ndarray] = []
    centre_signs: list[float] = []
    for start, end in zip(vertices, np.roll(vertices, -1, axis=0), strict=True):
        normal = np.cross(start, end)
        norm = float(np.linalg.norm(normal))
        if not math.isfinite(norm) or norm <= 1e-14:
            return None
        normal /= norm
        normals.append(normal)
        centre_signs.append(float(centre @ normal))
    signs = np.asarray(centre_signs)
    tolerance = 1e-12
    if np.all(signs >= -tolerance):
        orientation = 1.0
    elif np.all(signs <= tolerance):
        orientation = -1.0
    else:
        return None
    # Confirm the polygon itself is the intersection of these half-spaces.
    # This is true for the convex MER mosaic footprints; rejecting anything
    # else is safer than claiming exact full-cutout coverage.
    for normal in normals:
        if np.any(vertices @ (orientation * normal) < -tolerance):
            return None
    signed_sines = np.clip(orientation * signs, 0.0, 1.0)
    return math.degrees(float(np.min(np.arcsin(signed_sines))))


def exact_vis_parent_query(
    ra: float,
    dec: float,
    cutout_radius_deg: float,
    source_release: str = "Q1_R1",
) -> str:
    """ADQL prefilter for exact-release VIS parents near the cutout.

    ``INTERSECTS(fov, CIRCLE)`` is the predicate already exercised by the
    repository's Euclid/JWST overlap query.  Full-circle containment is then
    established from the returned ``fov`` polygon in :func:`exact_vis_parents`.
    """
    release = str(source_release).strip()
    if not release:
        raise ValueError("source_release must be non-empty")
    escaped_release = release.replace("'", "''")
    return (
        "SELECT mosaic_product_oid, release_name, product_type, fov, "
        "file_path, file_name, tile_index, instrument_name, filter_name, "
        "technique, ra, dec FROM sedm.mosaic_product "
        "WHERE instrument_name = 'VIS' AND technique = 'IMAGE' "
        "AND product_type = 'DpdMerBksMosaic' "
        f"AND release_name = '{escaped_release}' "
        "AND INTERSECTS(mosaic_product.fov, CIRCLE('ICRS', "
        f"{float(ra):.10f}, {float(dec):.10f}, {float(cutout_radius_deg):.10f})) = 1"
    )


def exact_vis_parents(
    ra: float,
    dec: float,
    cutout_radius_deg: float,
    *,
    source_release: str = "Q1_R1",
    query_runner: Callable[..., tuple[Any, str]] = query_mosaic_tiles,
) -> tuple[list[dict[str, Any]], str]:
    """Resolve all exact VIS parent mosaics for a proposed cutout."""
    requested_release = str(source_release).strip()
    query = exact_vis_parent_query(
        ra, dec, cutout_radius_deg, source_release=requested_release,
    )
    rows, error = query_runner(query)
    if rows is None:
        return [], error or "archive parent query failed"
    parents: list[dict[str, Any]] = []
    rejected_fov = 0
    for row in rows:
        tile_index = str(_row_value(row, "tile_index", "")).strip()
        file_path = str(_row_value(row, "file_path", "")).strip()
        file_name = str(_row_value(row, "file_name", "")).strip()
        product_oid = _text(_row_value(row, "mosaic_product_oid", ""))
        release_name = _text(_row_value(row, "release_name", ""))
        product_type = _text(_row_value(row, "product_type", ""))
        fov = _text(_row_value(row, "fov", ""))
        if (
            not tile_index
            or not file_path
            or not file_name
            or not product_oid
            or release_name != requested_release
            or product_type != "DpdMerBksMosaic"
        ):
            continue
        clearance_deg = _circle_clearance_in_convex_fov_deg(fov, ra, dec)
        if clearance_deg is None or clearance_deg < float(cutout_radius_deg):
            rejected_fov += 1
            continue
        full_path = f"{file_path.rstrip('/')}/{file_name}"
        # Product OID is the physical parent identity.  File aliases for the
        # same OID must not masquerade as independent samples; release_name is
        # included because archive freezes can repeat logical products.
        parent_key = f"VIS:{release_name}:{product_oid}"
        parents.append({
            "parent_id": hashlib.sha256(parent_key.encode("utf-8")).hexdigest()[:20],
            "mosaic_product_oid": product_oid,
            "release_name": release_name,
            "product_type": product_type,
            "tile_index": tile_index,
            "file_path": full_path,
            "file_name": file_name,
            "instrument_name": str(_row_value(row, "instrument_name", "VIS")),
            "filter_name": str(_row_value(row, "filter_name", "VIS")),
            "technique": str(_row_value(row, "technique", "IMAGE")),
            "mosaic_ra": _optional_float(_row_value(row, "ra")),
            "mosaic_dec": _optional_float(_row_value(row, "dec")),
            "fov": fov,
            "coverage_clearance_deg": clearance_deg,
            "coverage_method": (
                "INTERSECTS TAP prefilter plus convex spherical FOV "
                "half-space containment"
            ),
        })
    parents.sort(key=lambda item: (
        item["release_name"], item["mosaic_product_oid"],
        item["tile_index"], item["file_path"],
    ))
    if parents:
        return parents, ""
    if rejected_fov:
        return [], "intersecting parents did not fully contain the requested cutout"
    return [], f"no {requested_release} DpdMerBksMosaic parent intersected the request"


def exact_band_parent_query(
    ra: float,
    dec: float,
    cutout_radius_deg: float,
    band_name: str,
    source_release: str = "Q1_R1",
) -> str:
    """ADQL prefilter for an exact-release parent in one Euclid band."""
    band = Config.get_band(band_name)
    release = str(source_release).strip()
    if not release:
        raise ValueError("source_release must be non-empty")
    escaped_release = release.replace("'", "''")
    where = [
        f"instrument_name = '{band.archive_instrument}'",
        "technique = 'IMAGE'",
        "product_type = 'DpdMerBksMosaic'",
        f"release_name = '{escaped_release}'",
    ]
    if band.archive_filter:
        escaped_filter = str(band.archive_filter).replace("'", "''")
        where.append(f"filter_name = '{escaped_filter}'")
    return (
        "SELECT mosaic_product_oid, release_name, product_type, fov, "
        "file_path, file_name, tile_index, instrument_name, filter_name, "
        "technique, ra, dec FROM sedm.mosaic_product WHERE "
        + " AND ".join(where)
        + " AND INTERSECTS(mosaic_product.fov, CIRCLE('ICRS', "
        + f"{float(ra):.10f}, {float(dec):.10f}, "
        + f"{float(cutout_radius_deg):.10f})) = 1"
    )


def exact_band_parents(
    ra: float,
    dec: float,
    cutout_radius_deg: float,
    band_name: str,
    *,
    source_release: str = "Q1_R1",
    query_runner: Callable[..., tuple[Any, str]] = query_mosaic_tiles,
) -> tuple[list[dict[str, Any]], str]:
    """Resolve fully containing, release-frozen mosaic parents for a band."""
    band = Config.get_band(band_name)
    requested_release = str(source_release).strip()
    rows, error = query_runner(exact_band_parent_query(
        ra,
        dec,
        cutout_radius_deg,
        band_name,
        source_release=requested_release,
    ))
    if rows is None:
        return [], error or "archive parent query failed"
    parents: list[dict[str, Any]] = []
    rejected_fov = 0
    for row in rows:
        tile_index = _text(_row_value(row, "tile_index", ""))
        file_path = _text(_row_value(row, "file_path", ""))
        file_name = _text(_row_value(row, "file_name", ""))
        product_oid = _text(_row_value(row, "mosaic_product_oid", ""))
        release_name = _text(_row_value(row, "release_name", ""))
        product_type = _text(_row_value(row, "product_type", ""))
        instrument_name = _text(_row_value(row, "instrument_name", ""))
        filter_name = _text(_row_value(row, "filter_name", ""))
        technique = _text(_row_value(row, "technique", ""))
        fov = _text(_row_value(row, "fov", ""))
        if (
            not tile_index
            or not file_path
            or not file_name
            or not product_oid
            or release_name != requested_release
            or product_type != "DpdMerBksMosaic"
            or instrument_name != band.archive_instrument
            or technique != "IMAGE"
            or (band.archive_filter and filter_name != band.archive_filter)
        ):
            continue
        clearance_deg = _circle_clearance_in_convex_fov_deg(fov, ra, dec)
        if clearance_deg is None or clearance_deg < float(cutout_radius_deg):
            rejected_fov += 1
            continue
        full_path = f"{file_path.rstrip('/')}/{file_name}"
        parent_key = f"{band_name}:{release_name}:{product_oid}"
        parents.append({
            "parent_id": hashlib.sha256(parent_key.encode("utf-8")).hexdigest()[:20],
            "mosaic_product_oid": product_oid,
            "release_name": release_name,
            "product_type": product_type,
            "tile_index": tile_index,
            "file_path": full_path,
            "file_name": file_name,
            "instrument_name": instrument_name,
            "filter_name": filter_name,
            "technique": technique,
            "mosaic_ra": _optional_float(_row_value(row, "ra")),
            "mosaic_dec": _optional_float(_row_value(row, "dec")),
            "fov": fov,
            "coverage_clearance_deg": clearance_deg,
            "coverage_method": (
                "INTERSECTS TAP prefilter plus convex spherical FOV "
                "half-space containment"
            ),
        })
    parents.sort(key=lambda item: (
        item["release_name"], item["mosaic_product_oid"],
        item["tile_index"], item["file_path"],
    ))
    if parents:
        return parents, ""
    if rejected_fov:
        return [], "intersecting parents did not fully contain the requested cutout"
    return [], (
        f"no {requested_release} {band_name} DpdMerBksMosaic parent "
        "intersected the request"
    )


def assign_unique_parents(
    candidates: pd.DataFrame,
    *,
    samples_per_anchor: int,
    cutout_radius_deg: float,
    minimum_sample_separation_arcmin: float = 6.5,
    parent_resolver: Callable[[float, float, float], tuple[list[dict[str, Any]], str]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Choose one deterministic candidate per slot and unique parent mosaic."""
    if int(samples_per_anchor) < 1:
        raise ValueError("samples_per_anchor must be positive")
    selected: list[dict[str, Any]] = []
    used_parents: set[str] = set()
    rejection_counts: Counter[str] = Counter()
    grouped = candidates.groupby(["anchor_id", "slot"], sort=False)
    for (anchor_id, slot), rows in grouped:
        chosen: dict[str, Any] | None = None
        for candidate in rows.sort_values("candidate_rank").to_dict("records"):
            if selected:
                prior_ra = np.asarray([float(row["ra"]) for row in selected])
                prior_dec = np.asarray([float(row["dec"]) for row in selected])
                separation = float(np.min(_angular_separation_deg(
                    float(candidate["ra"]), float(candidate["dec"]),
                    prior_ra, prior_dec,
                ))) * 60.0
                if separation < float(minimum_sample_separation_arcmin):
                    rejection_counts["selected field would overlap"] += 1
                    continue
            parents, error = parent_resolver(
                float(candidate["ra"]), float(candidate["dec"]),
                float(cutout_radius_deg),
            )
            if not parents:
                rejection_counts[error or "outside exact coverage"] += 1
                continue
            available = [p for p in parents if p["parent_id"] not in used_parents]
            if not available:
                rejection_counts["parent mosaic already selected"] += 1
                continue
            parent = available[0]
            chosen = {
                **candidate,
                "sample_id": len(selected),
                "parent": parent,
                "parent_id": parent["parent_id"],
                "status": "planned",
                "error": None,
            }
            used_parents.add(parent["parent_id"])
            selected.append(chosen)
            break
        if chosen is None:
            rejection_counts[f"no usable candidate for {anchor_id}/{slot}"] += 1
    expected = len(candidates[["anchor_id", "slot"]].drop_duplicates())
    if expected != len(selected):
        rejection_counts["unfilled sample slots"] += expected - len(selected)
    return selected, dict(rejection_counts)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _star_support_plan(args: argparse.Namespace) -> dict[str, Any]:
    source_path = Path(args.star_support_csv).resolve()
    return {
        "star_support_csv": str(source_path),
        "stars_csv_sha256": _sha256_file(source_path),
        "source_release": str(args.source_release),
        "seed": int(args.seed),
        "n_clusters": int(args.n_clusters),
        "samples_per_cluster": int(args.samples_per_cluster),
        "support_cell_area_deg2": float(args.support_cell_area_deg2),
        "jitter_radius_deg": float(args.jitter_radius_deg),
        "avoid_star_arcsec": float(args.avoid_star_arcsec),
        "minimum_sample_separation_arcmin": float(args.minimum_separation_arcmin),
        "candidates_per_sample": int(args.candidates_per_sample),
        "cutout_size_vis_pixels": int(args.vis_pixels),
    }


def _plan_fingerprint(plan: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        plan, allow_nan=False, separators=(",", ":"), sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _uniform_disk_positions(
    ra_centre_deg:  float,
    dec_centre_deg: float,
    radius_deg:     float,
    n_positions:    int,
    *,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Uniform random (RA, Dec) inside a small spherical disk.

    Uses a flat-sky rejection sample on a 2R × 2R square — fine for
    radii ≲ a few degrees, where the deviation from a true geodesic
    disk is sub-arcsec. The RA offset is divided by ``cos(dec_centre)``
    so the area density on the sphere stays uniform at high declination
    (at dec=66° this is a ~2.5× horizontal stretch compared to a naive
    flat sample, which would otherwise clump positions toward the poles
    of the local tangent plane).

    Returns a DataFrame with ``(id, ra, dec, magnitude)`` columns.
    ``magnitude`` is NaN — irrelevant for sky cutouts, kept for legacy
    test compatibility.
    """
    cos_dec = float(np.cos(np.deg2rad(dec_centre_deg)))
    # cos(90°) is ~6e-17 (positive due to float rounding), so a bare
    # ``cos_dec <= 0`` check would let pole-centred calls through and
    # blow up RA by 1e16×. Reject anything within ~3° of the poles.
    if cos_dec < 1e-3:
        raise ValueError(
            f"dec_centre_deg={dec_centre_deg} too close to a pole "
            "for the flat-sky approximation"
        )

    accepted: list = []
    # Acceptance rate of disk-in-square is π/4 ≈ 0.785, so 1.4× over-
    # sampling clears the budget in one round with high probability.
    while len(accepted) < n_positions:
        n_try = max(8, int((n_positions - len(accepted)) * 1.4))
        dx = rng.uniform(-radius_deg, radius_deg, n_try)
        dy = rng.uniform(-radius_deg, radius_deg, n_try)
        keep = (dx ** 2 + dy ** 2) <= radius_deg ** 2
        for x, y in zip(dx[keep], dy[keep], strict=False):
            ra  = (ra_centre_deg + x / cos_dec) % 360.0
            dec = dec_centre_deg + y
            accepted.append((ra, dec))
            if len(accepted) >= n_positions:
                break

    df = pd.DataFrame(accepted, columns=["ra", "dec"])
    df.insert(0, "id", range(len(df)))
    df["magnitude"] = np.nan
    return df


def bundle_path_for_id(output_dir: str, pos_id: int) -> str:
    """Return the absolute path of the bundled FITS for a given position id."""
    return os.path.join(output_dir, Config.EuclidSky.CUTOUTS_SUBDIR, f"sky_{pos_id:04d}.fits")


def default_vis_noise_output_dir() -> str:
    """Dedicated root that cannot collide with round-trip sky cutouts."""
    return os.path.join(Config.EUCLID_SKY_DIR, VIS_NOISE_SAMPLING_SUBDIR)


def default_archive_fields_output_dir() -> str:
    """Dedicated derived root for compact matched four-band fields."""
    return str(Config.EUCLID_ARCHIVE_FIELDS_DIR)


def default_archive_fields_source_manifest() -> str:
    """Return the immutable VIS-parent sampling manifest used as anchors."""
    return os.path.join(
        Config.EUCLID_SKY_DIR,
        VIS_NOISE_SAMPLING_SUBDIR,
        SAMPLING_MANIFEST_NAME,
    )


def archive_field_bundle_path(output_dir: str | os.PathLike[str], sample_id: int) -> str:
    """Stable output path for one compact four-band archive sample."""
    return os.path.join(
        os.fspath(output_dir), "cutouts", f"field_{int(sample_id):04d}.fits",
    )


def _archive_source_manifest(
    source_path: str | os.PathLike[str],
) -> tuple[dict[str, Any], str]:
    """Load and strictly validate the frozen 44-parent VIS sampling plan."""
    path = Path(source_path)
    try:
        raw = path.read_bytes()
        source = json.loads(raw)
    except OSError as exc:
        raise ValueError(f"Cannot read source VIS sampling manifest {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Source VIS sampling manifest is invalid JSON: {path}") from exc
    if not isinstance(source, dict):
        raise ValueError("Source VIS sampling manifest must be a JSON object")
    if source.get("kind") != ARCHIVE_FIELDS_SOURCE_KIND:
        raise ValueError(
            f"Source manifest kind must be {ARCHIVE_FIELDS_SOURCE_KIND!r}"
        )
    if source.get("version") != SAMPLING_MANIFEST_VERSION:
        raise ValueError(
            f"Source manifest version must be {SAMPLING_MANIFEST_VERSION}"
        )
    release = str(source.get("source_release") or "").strip()
    if not release:
        raise ValueError("Source VIS sampling manifest has no source_release")
    plan = source.get("plan")
    if not isinstance(plan, Mapping):
        raise ValueError("Source VIS sampling manifest has no acquisition plan")
    source_plan_fingerprint = str(source.get("plan_fingerprint") or "")
    if source_plan_fingerprint != _plan_fingerprint(plan):
        raise ValueError("Source VIS sampling plan fingerprint is invalid")
    samples = source.get("samples")
    if not isinstance(samples, list) or len(samples) != 44:
        raise ValueError("Archive-field acquisition requires exactly 44 VIS parent samples")
    complete_statuses = {"written", "cached", "complete", "completed"}
    parent_ids: set[str] = set()
    for expected_id, sample in enumerate(samples):
        if not isinstance(sample, Mapping):
            raise ValueError(f"Source VIS sample {expected_id} is not an object")
        try:
            sample_id = int(sample["sample_id"])
            ra = float(sample["ra"])
            dec = float(sample["dec"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Source VIS sample {expected_id} has invalid identity") from exc
        if sample_id != expected_id or not (
            math.isfinite(ra) and math.isfinite(dec) and -90.0 <= dec <= 90.0
        ):
            raise ValueError(f"Source VIS sample {expected_id} has invalid identity")
        if str(sample.get("status") or "").lower() not in complete_statuses:
            raise ValueError(f"Source VIS sample {sample_id} is not complete")
        parent_id = str(sample.get("parent_id") or "").strip()
        parent = sample.get("parent")
        if not parent_id or not isinstance(parent, Mapping):
            raise ValueError(f"Source VIS sample {sample_id} has no exact parent")
        if parent_id in parent_ids:
            raise ValueError(f"Source VIS parent {parent_id!r} is not independent")
        parent_ids.add(parent_id)
        if str(parent.get("release_name") or "") != release:
            raise ValueError(f"Source VIS sample {sample_id} release is inconsistent")
        if str(parent.get("product_type") or "") != "DpdMerBksMosaic":
            raise ValueError(f"Source VIS sample {sample_id} is not a MER mosaic")
        if not str(sample.get("anchor_id") or "").strip():
            raise ValueError(f"Source VIS sample {sample_id} has no anchor_id")
        if str(sample.get("field") or "") not in {"EDF-N", "EDF-F", "EDF-S"}:
            raise ValueError(f"Source VIS sample {sample_id} has no known Q1 field")
    return source, hashlib.sha256(raw).hexdigest()


def _archive_field_plan(
    source: Mapping[str, Any], source_manifest_sha256: str,
) -> dict[str, Any]:
    """Build the immutable, relocatable 44-by-5 acquisition plan."""
    source_plan = source["plan"]
    assert isinstance(source_plan, Mapping)
    return {
        "source_manifest_sha256": str(source_manifest_sha256),
        "source_plan_fingerprint": str(source["plan_fingerprint"]),
        "source_release": str(source["source_release"]),
        "source_sample_count": len(source["samples"]),
        "positions_per_parent": ARCHIVE_FIELDS_POSITIONS_PER_PARENT,
        "cutout_size_vis_pixels": ARCHIVE_FIELDS_VIS_PIXELS,
        "source_vis_size_vis_pixels": int(source_plan["cutout_size_vis_pixels"]),
        "minimum_parent_download_size_vis_pixels": (
            ARCHIVE_FIELDS_MINIMUM_PARENT_VIS_PIXELS
        ),
        "parent_download_size_vis_pixels": (
            ARCHIVE_FIELDS_PARENT_DOWNLOAD_VIS_PIXELS
        ),
        "parent_download_safety_margin_arcsec_per_edge": 3.2,
        "bands": list(Config.LR_INPUT_BAND_NAMES),
        "offset_pattern_arcsec": [
            {
                "position_index": index,
                "name": name,
                "east": east,
                "north": north,
            }
            for index, (name, east, north) in enumerate(ARCHIVE_FIELDS_PATTERN)
        ],
        "registration_method": (
            "exact celestial-WCS integer translation and common crop; no interpolation"
        ),
        "wcs_tolerance_pixels": ARCHIVE_FIELDS_WCS_TOLERANCE_PIXELS,
    }


def _offset_coordinate(
    ra: float, dec: float, east_arcsec: float, north_arcsec: float,
) -> tuple[float, float]:
    """Apply a deterministic ICRS tangent-plane offset."""
    origin = SkyCoord(float(ra) * u.deg, float(dec) * u.deg, frame="icrs")
    frame = SkyOffsetFrame(origin=origin)
    offset = SkyCoord(
        lon=float(east_arcsec) * u.arcsec,
        lat=float(north_arcsec) * u.arcsec,
        frame=frame,
    ).icrs
    return float(offset.ra.deg % 360.0), float(offset.dec.deg)


def _new_archive_fields_manifest(
    source: Mapping[str, Any], source_manifest_sha256: str,
) -> dict[str, Any]:
    """Derive 220 deterministic positions without changing the source plan."""
    plan = _archive_field_plan(source, source_manifest_sha256)
    source_plan_fingerprint = str(source["plan_fingerprint"])
    source_release = str(source["source_release"])
    samples: list[dict[str, Any]] = []
    for source_sample in source["samples"]:
        source_sample_id = int(source_sample["sample_id"])
        source_ra = float(source_sample["ra"])
        source_dec = float(source_sample["dec"])
        for position_index, (position_name, east, north) in enumerate(
            ARCHIVE_FIELDS_PATTERN
        ):
            sample_id = (
                source_sample_id * ARCHIVE_FIELDS_POSITIONS_PER_PARENT
                + position_index
            )
            ra, dec = _offset_coordinate(source_ra, source_dec, east, north)
            samples.append({
                "sample_id": sample_id,
                "field_id": sample_id,
                "source_sample_id": source_sample_id,
                "anchor_id": str(source_sample["anchor_id"]),
                "source_slot": int(source_sample.get("slot", 0)),
                "parent_id": str(source_sample["parent_id"]),
                "field": str(source_sample["field"]),
                "position_index": position_index,
                "position_name": position_name,
                "offset_east_arcsec": east,
                "offset_north_arcsec": north,
                "source_ra": source_ra,
                "source_dec": source_dec,
                "ra": ra,
                "dec": dec,
                "source_release": source_release,
                "source_plan_fingerprint": source_plan_fingerprint,
                "archive_parents": {"VIS": dict(source_sample["parent"])},
                "status": "planned",
                "error": None,
                "output_path": None,
                "bands": {},
            })
    plan_fingerprint = compute_plan_fingerprint(plan)
    for sample in samples:
        sample["plan_fingerprint"] = plan_fingerprint
    manifest: dict[str, Any] = {
        "version": ARCHIVE_FIELDS_MANIFEST_VERSION,
        "kind": ARCHIVE_FIELDS_MANIFEST_KIND,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_release": source_release,
        "source": {
            "manifest_kind": str(source["kind"]),
            "manifest_version": int(source["version"]),
            "manifest_sha256": str(source_manifest_sha256),
            "plan_fingerprint": source_plan_fingerprint,
        },
        "plan": plan,
        "plan_fingerprint": plan_fingerprint,
        "samples": samples,
    }
    manifest["collection_fingerprint"] = compute_collection_fingerprint(manifest)
    return manifest


def _cached_planned_bundle_matches(
    target_path: str,
    sample: Mapping[str, Any],
    *,
    vis_pixels: int,
    source_release: str,
) -> bool:
    """True only when an existing bundle belongs to this exact sample plan."""
    if not os.path.isfile(target_path) or os.path.getsize(target_path) <= 0:
        return False
    try:
        with fits.open(target_path, memmap=True) as hdul:
            hdul.verify("exception")
            header = hdul[0].header
            image_hdu = hdul["VIS"]
            image_shape = None if image_hdu.data is None else image_hdu.data.shape
    except (OSError, KeyError, IndexError, ValueError, VerifyError):
        return False
    parent = sample.get("parent") or {}
    actual_release = str(parent.get("release_name") or source_release)
    try:
        return bool(
            actual_release == str(source_release)
            and int(header.get("POS_ID", -1)) == int(sample["sample_id"])
            and int(header.get("VIS_PIX", -1)) == int(vis_pixels)
            and str(header.get("PARENT", "")) == str(sample["parent_id"])
            and str(header.get("RELEASE", "")) == actual_release
            and str(header.get("MOSAIC_PRODUCT_OID", ""))
            == str(parent.get("mosaic_product_oid") or "")
            and str(header.get("PRODTYPE", ""))
            == str(parent.get("product_type") or "")
            and image_shape is not None
            and _vis_noise_shape_matches(image_shape, int(vis_pixels))
            and math.isclose(
                float(header.get("RA")), float(sample["ra"]), abs_tol=1e-8,
            )
            and math.isclose(
                float(header.get("DEC")), float(sample["dec"]), abs_tol=1e-8,
            )
        )
    except (KeyError, TypeError, ValueError):
        return False


def _write_bundle(
    target_path: str,
    *,
    pos_id: int,
    ra: float,
    dec: float,
    vis_pixels: int,
    arcsec_side: float,
    band_files: dict,
    band_names: Iterable[str] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> None:
    """Combine per-band tempfile FITS into a single multi-HDU bundle.

    ``band_files`` is a ``{band_name: tempfile_path}`` map; one entry
    per :data:`Config.LR_INPUT_BAND_NAMES` band. The output has a data-
    less ``PrimaryHDU`` carrying position metadata in its header plus
    one ``ImageHDU`` per band (``EXTNAME = band_name``).
    """
    primary_hdr = fits.Header()
    primary_hdr["POS_ID"]  = (int(pos_id),         "Sky position id (matches sky_positions.csv)")
    primary_hdr["RA"]      = (float(ra),           "ICRS right ascension (deg)")
    primary_hdr["DEC"]     = (float(dec),          "ICRS declination (deg)")
    primary_hdr["VIS_PIX"] = (int(vis_pixels),     "Cutout side in VIS pixels (0.10\"/pix)")
    primary_hdr["ARCSEC"]  = (float(arcsec_side),  "Cutout side on sky (arcsec)")
    if provenance:
        if provenance.get("tile_index") is not None:
            primary_hdr["TILEIDX"] = (str(provenance["tile_index"]), "Parent mosaic tile")
        if provenance.get("parent_id") is not None:
            primary_hdr["PARENT"] = (str(provenance["parent_id"]), "Sampling parent id")
        actual_release = provenance.get("release_name") or provenance.get("source_release")
        if actual_release is not None:
            primary_hdr["RELEASE"] = (str(actual_release), "Archive release_name")
        if provenance.get("mosaic_product_oid") is not None:
            primary_hdr["HIERARCH MOSAIC_PRODUCT_OID"] = (
                str(provenance["mosaic_product_oid"]), "Parent archive product OID",
            )
        if provenance.get("product_type") is not None:
            primary_hdr["PRODTYPE"] = (str(provenance["product_type"]), "Archive product type")

    hdul = fits.HDUList([fits.PrimaryHDU(header=primary_hdr)])
    selected_bands = list(band_names or Config.LR_INPUT_BAND_NAMES)
    for band_name in selected_bands:
        src = band_files[band_name]
        with fits.open(src, memmap=False) as src_hdul:
            data = np.asarray(src_hdul[0].data)
            band_hdr = src_hdul[0].header.copy(strip=True)
        band_hdr["EXTNAME"] = band_name
        hdul.append(fits.ImageHDU(data=data, header=band_hdr, name=band_name))

    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    temporary_path = target_path + ".tmp"
    try:
        hdul.writeto(temporary_path, overwrite=True, output_verify="exception")
        with fits.open(temporary_path, memmap=True) as written:
            written.verify("exception")
        os.replace(temporary_path, target_path)
    finally:
        with contextlib.suppress(OSError):
            os.remove(temporary_path)


def _fetch_position_bundle(
    *,
    pos_id: int,
    ra: float,
    dec: float,
    vis_pixels: int,
    arcsec_side: float,
    output_dir: str,
) -> dict:
    """Fetch all 4 bands for one position and write the bundled FITS.

    Returns a dict with ``{"status": "written" | "cached" | "failed",
    "id": pos_id, "errors": [...]}`` so the caller can tally outcomes.
    Per-band tempfiles are removed in a ``finally`` block so a crash
    mid-write doesn't strand them on disk.
    """
    target = bundle_path_for_id(output_dir, pos_id)
    if os.path.isfile(target) and os.path.getsize(target) > 0:
        return {"status": "cached", "id": pos_id, "errors": []}

    band_names: list[str] = list(Config.LR_INPUT_BAND_NAMES)
    band_files: dict = {}
    errors: list = []
    tmp_dir = tempfile.mkdtemp(prefix=f"euclid_sky_{pos_id:04d}_")
    try:
        for band_name in band_names:
            tmp_path = os.path.join(tmp_dir, f"{band_name}.fits")
            ok, err = fetch_cutout_at(
                ra=ra, dec=dec, band_name=band_name,
                output_file=tmp_path,
                cutout_size_vis_pixels=vis_pixels,
            )
            if not ok:
                errors.append(f"{band_name}: {err}")
                return {"status": "failed", "id": pos_id, "errors": errors}
            band_files[band_name] = tmp_path

        _write_bundle(
            target,
            pos_id=pos_id, ra=ra, dec=dec,
            vis_pixels=vis_pixels, arcsec_side=arcsec_side,
            band_files=band_files,
        )
        return {"status": "written", "id": pos_id, "errors": []}
    finally:
        for p in band_files.values():
            with contextlib.suppress(OSError):
                os.remove(p)
        with contextlib.suppress(OSError):
            os.rmdir(tmp_dir)


def _fetch_planned_vis_bundle(
    sample: Mapping[str, Any],
    *,
    vis_pixels: int,
    arcsec_side: float,
    output_dir: str,
    source_release: str,
) -> dict[str, Any]:
    """Download one planned VIS sample from its already-resolved parent."""
    pos_id = int(sample["sample_id"])
    target = bundle_path_for_id(output_dir, pos_id)
    if _cached_planned_bundle_matches(
        target, sample, vis_pixels=int(vis_pixels), source_release=source_release,
    ):
        with fits.open(target, memmap=True) as hdul:
            actual_shape = list(hdul["VIS"].shape)
        return {
            "status": "cached", "id": pos_id, "errors": [],
            "actual_shape": actual_shape,
        }
    parent = dict(sample["parent"])
    config = DownloadConfig.for_band(
        "VIS", cutout_size_vis_pixels=int(vis_pixels), saturation_core_size=0,
    )
    cutout_radius_arcmin = (arcsec_side / 2.0) / 60.0
    temp_dir = tempfile.mkdtemp(prefix=f"vis_noise_{pos_id:04d}_")
    raw_path = os.path.join(temp_dir, "VIS.fits")
    try:
        last_download_error = "VIS parent cutout download failed"
        actual_shape: list[int] | None = None
        for attempt in range(VIS_NOISE_DOWNLOAD_ATTEMPTS):
            # ``Euclid.get_cutout`` swallows HTTP failures and can leave a
            # partial output behind.  Never let that partial file make the
            # next attempt look successful.
            with contextlib.suppress(OSError):
                os.remove(raw_path)
            ok = download_one_cutout(
                float(sample["ra"]), float(sample["dec"]), config,
                cutout_radius_arcmin, raw_path, parent,
            )
            if ok:
                try:
                    with fits.open(raw_path, memmap=True) as hdul:
                        # Q1 raw products can carry a non-string EXTNAME in
                        # HDU 0.  The science array is still readable, and
                        # `_write_bundle` replaces EXTNAME before strictly
                        # verifying the normalized output.  Do not reject the
                        # usable archive payload at this pre-normalization
                        # boundary.
                        image_hdu = next(
                            (
                                candidate for candidate in hdul
                                if candidate.data is not None
                            ),
                            None,
                        )
                        if image_hdu is None:
                            last_download_error = (
                                "VIS parent cutout has no image data"
                            )
                        else:
                            actual_shape = [int(value) for value in image_hdu.shape]
                except (OSError, ValueError, VerifyError) as exc:
                    last_download_error = (
                        "VIS parent cutout is unreadable: "
                        f"{type(exc).__name__}: {exc}"
                    )
                if actual_shape is not None:
                    break
            if attempt < VIS_NOISE_DOWNLOAD_ATTEMPTS - 1:
                delay_index = min(
                    attempt,
                    len(VIS_NOISE_DOWNLOAD_RETRY_DELAYS_SECONDS) - 1,
                )
                time.sleep(VIS_NOISE_DOWNLOAD_RETRY_DELAYS_SECONDS[delay_index])
        if actual_shape is None:
            return {
                "status": "failed", "id": pos_id,
                "errors": [
                    f"{last_download_error} after "
                    f"{VIS_NOISE_DOWNLOAD_ATTEMPTS} attempts"
                ],
            }
        if not _vis_noise_shape_matches(actual_shape, int(vis_pixels)):
            undersize_tolerance = int(Config.Matching.DOWNLOAD_SIZE_TOL_PIXELS)
            oversize_tolerance = max(
                undersize_tolerance,
                int(math.ceil(
                    int(vis_pixels) * VIS_NOISE_MAX_OVERSIZE_FRACTION
                )),
            )
            return {
                "status": "failed", "id": pos_id,
                "errors": [
                    f"VIS cutout shape {tuple(actual_shape)} differs from "
                    f"{vis_pixels}; allowed undersize is {undersize_tolerance}px "
                    f"and oversize is {oversize_tolerance}px"
                ],
            }
        _write_bundle(
            target,
            pos_id=pos_id,
            ra=float(sample["ra"]),
            dec=float(sample["dec"]),
            vis_pixels=int(vis_pixels),
            arcsec_side=float(arcsec_side),
            band_files={"VIS": raw_path},
            band_names=["VIS"],
            provenance={
                **parent,
                "source_release": source_release,
            },
        )
        return {
            "status": "written", "id": pos_id, "errors": [],
            "actual_shape": actual_shape,
        }
    finally:
        with contextlib.suppress(OSError):
            os.remove(raw_path)
        with contextlib.suppress(OSError):
            os.rmdir(temp_dir)


def _source_vis_bundle_path(
    source_manifest_path: str | os.PathLike[str],
    source_sample: Mapping[str, Any],
) -> Path:
    """Resolve a source bundle after remote/local manifest relocation."""
    recorded = Path(str(source_sample.get("output_path") or ""))
    if recorded.is_file():
        return recorded
    return (
        Path(source_manifest_path).parent
        / "cutouts"
        / f"sky_{int(source_sample['sample_id']):04d}.fits"
    )


def _read_archive_image(
    path: str | os.PathLike[str], band_name: str,
) -> tuple[np.ndarray, fits.Header]:
    """Read one unmodified archive-rate image and its full science header."""
    target = Path(path)
    try:
        with fits.open(target, memmap=False) as hdul:
            # Q1 SODA products can carry a numeric EXTNAME in the raw primary
            # HDU.  The science array and WCS remain valid; the final compact
            # bundle replaces EXTNAME with the canonical band name and is
            # strictly verified after writing.  Do not reject usable archive
            # pixels at this pre-normalization boundary.
            try:
                candidate = hdul[band_name]
            except (KeyError, IndexError):
                candidate = next((hdu for hdu in hdul if hdu.data is not None), None)
            if candidate is None or candidate.data is None:
                raise ValueError(f"{target} has no image data for {band_name}")
            data = np.array(candidate.data, copy=True)
            header = candidate.header.copy(strip=True)
    except (OSError, ValueError, VerifyError) as exc:
        raise ValueError(f"Cannot read {band_name} archive cutout {target}: {exc}") from exc
    if data.ndim != 2 or min(data.shape) < ARCHIVE_FIELDS_VIS_PIXELS:
        raise ValueError(
            f"{band_name} archive cutout has unusable shape {data.shape}"
        )
    try:
        zeropoint = float(header["MAGZERO"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{band_name} archive cutout has no finite MAGZERO") from exc
    if not math.isfinite(zeropoint):
        raise ValueError(f"{band_name} archive cutout has no finite MAGZERO")
    unit = "".join(str(header.get("BUNIT") or "").strip().lower().split())
    archive_rate_units = {
        "", "adu/s", "adu/sec", "count/s", "count/sec", "counts/s",
        "counts/sec", "electron/s", "electron/sec", "electrons/s",
        "electrons/sec", "e-/s", "e-/sec",
    }
    if unit not in archive_rate_units:
        raise ValueError(f"{band_name} archive cutout has unsupported BUNIT {unit!r}")
    wcs = WCS(header).celestial
    if not wcs.has_celestial or wcs.pixel_n_dim != 2 or wcs.world_n_dim != 2:
        raise ValueError(f"{band_name} archive cutout has no two-axis celestial WCS")
    return data, header


def _validate_source_vis_bundle(
    path: Path,
    source_sample: Mapping[str, Any],
    source_release: str,
) -> tuple[np.ndarray, fits.Header]:
    """Bind the reused full-size VIS image to its immutable source record."""
    if not path.is_file():
        raise ValueError(f"Source VIS bundle is unavailable: {path}")
    try:
        with fits.open(path, memmap=False) as hdul:
            primary = hdul[0].header
            if int(primary.get("POS_ID", -1)) != int(source_sample["sample_id"]):
                raise ValueError("POS_ID disagrees with source manifest")
            if str(primary.get("PARENT") or "") != str(source_sample["parent_id"]):
                raise ValueError("PARENT disagrees with source manifest")
            if str(primary.get("RELEASE") or "").strip() != str(source_release):
                raise ValueError("RELEASE disagrees with source manifest")
            if not math.isclose(
                float(primary.get("RA")), float(source_sample["ra"]), abs_tol=1e-8,
            ) or not math.isclose(
                float(primary.get("DEC")), float(source_sample["dec"]), abs_tol=1e-8,
            ):
                raise ValueError("RA/DEC disagree with source manifest")
    except (OSError, TypeError, ValueError, VerifyError) as exc:
        raise ValueError(f"Invalid source VIS bundle {path}: {exc}") from exc
    return _read_archive_image(path, "VIS")


def _download_parent_band(
    *,
    band_name: str,
    ra: float,
    dec: float,
    parent_download_vis_pixels: int,
    parent: Mapping[str, Any],
    output_path: str,
) -> None:
    """Download one full parent-centred NISP cutout with bounded retries."""
    config = DownloadConfig.for_band(
        band_name,
        cutout_size_vis_pixels=int(parent_download_vis_pixels),
        saturation_core_size=0,
    )
    arcsec_side = (
        int(parent_download_vis_pixels) * Config.BAND_VIS.pixel_scale_lr_arcsec
    )
    radius_arcmin = (arcsec_side / 2.0) / 60.0
    last_error = f"{band_name} parent cutout download failed"
    for attempt in range(ARCHIVE_FIELDS_DOWNLOAD_ATTEMPTS):
        with contextlib.suppress(OSError):
            os.remove(output_path)
        if download_one_cutout(
            float(ra), float(dec), config, radius_arcmin, output_path, dict(parent),
        ):
            try:
                _read_archive_image(output_path, band_name)
                return
            except ValueError as exc:
                last_error = str(exc)
        if attempt < ARCHIVE_FIELDS_DOWNLOAD_ATTEMPTS - 1:
            delay_index = min(
                attempt, len(ARCHIVE_FIELDS_DOWNLOAD_RETRY_DELAYS_SECONDS) - 1,
            )
            time.sleep(ARCHIVE_FIELDS_DOWNLOAD_RETRY_DELAYS_SECONDS[delay_index])
    raise RuntimeError(
        f"{last_error} after {ARCHIVE_FIELDS_DOWNLOAD_ATTEMPTS} attempts"
    )


def _registration_control_pixels(size: int) -> tuple[np.ndarray, np.ndarray]:
    """Corners, edge centres, and centre constrain the complete output grid."""
    edge = float(size - 1)
    middle = edge / 2.0
    points = (
        (0.0, 0.0), (middle, 0.0), (edge, 0.0),
        (0.0, middle), (middle, middle), (edge, middle),
        (0.0, edge), (middle, edge), (edge, edge),
    )
    return (
        np.asarray([point[0] for point in points], dtype=np.float64),
        np.asarray([point[1] for point in points], dtype=np.float64),
    )


def _aligned_archive_crops(
    raw_bands: Mapping[str, tuple[np.ndarray, fits.Header]],
    *,
    ra: float,
    dec: float,
    output_size: int = ARCHIVE_FIELDS_VIS_PIXELS,
) -> dict[str, tuple[np.ndarray, fits.Header]]:
    """Crop common-grid tiles and reject any resampling requirement."""
    expected_bands = list(Config.LR_INPUT_BAND_NAMES)
    if list(raw_bands) != expected_bands:
        raise ValueError(f"Raw bands must be ordered as {expected_bands!r}")
    _reference_data, reference_header = raw_bands["VIS"]
    reference_wcs = WCS(reference_header).celestial
    target = SkyCoord(float(ra) * u.deg, float(dec) * u.deg, frame="icrs")
    target_x, target_y = reference_wcs.world_to_pixel(target)
    if not (math.isfinite(float(target_x)) and math.isfinite(float(target_y))):
        raise ValueError("Requested field centre is outside the VIS WCS")
    centre = (int(output_size) - 1) / 2.0
    reference_x0 = int(math.floor(float(target_x) - centre + 0.5))
    reference_y0 = int(math.floor(float(target_y) - centre + 0.5))
    control_x, control_y = _registration_control_pixels(int(output_size))
    reference_world = reference_wcs.pixel_to_world(
        control_x + reference_x0, control_y + reference_y0,
    )
    crops: dict[str, tuple[np.ndarray, fits.Header]] = {}
    for band_name in expected_bands:
        data, source_header = raw_bands[band_name]
        source_wcs = WCS(source_header).celestial
        mapped_x, mapped_y = source_wcs.world_to_pixel(reference_world)
        if not (
            np.all(np.isfinite(mapped_x)) and np.all(np.isfinite(mapped_y))
        ):
            raise ValueError(f"{band_name} WCS cannot map the VIS output grid")
        delta_x = np.asarray(mapped_x, dtype=np.float64) - control_x
        delta_y = np.asarray(mapped_y, dtype=np.float64) - control_y
        x0 = int(np.rint(np.median(delta_x)))
        y0 = int(np.rint(np.median(delta_y)))
        residual = max(
            float(np.max(np.abs(delta_x - x0))),
            float(np.max(np.abs(delta_y - y0))),
        )
        if residual > ARCHIVE_FIELDS_WCS_TOLERANCE_PIXELS:
            raise ValueError(
                f"{band_name} is not on the VIS pixel grid: maximum WCS "
                f"residual {residual:.6g}px"
            )
        y1 = y0 + int(output_size)
        x1 = x0 + int(output_size)
        if x0 < 0 or y0 < 0 or y1 > data.shape[0] or x1 > data.shape[1]:
            raise ValueError(
                f"{band_name} does not cover the requested {output_size}px crop "
                f"at source origin ({x0}, {y0}) in shape {data.shape}"
            )
        tile = np.array(data[y0:y1, x0:x1], copy=True)
        if tile.shape != (int(output_size), int(output_size)):
            raise ValueError(f"{band_name} crop has unexpected shape {tile.shape}")
        if not np.all(np.isfinite(tile)):
            raise ValueError(f"{band_name} crop contains non-finite pixels")
        output_header = source_header.copy(strip=True)
        try:
            output_header["CRPIX1"] = float(source_header["CRPIX1"]) - x0
            output_header["CRPIX2"] = float(source_header["CRPIX2"]) - y0
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{band_name} cannot adjust its crop WCS") from exc
        output_header["EXTNAME"] = band_name
        if not str(output_header.get("BUNIT") or "").strip():
            output_header["BUNIT"] = ("adu/s", "Archive mosaic rate units")
        output_header["SRCNX"] = (int(data.shape[1]), "Source cutout width")
        output_header["SRCNY"] = (int(data.shape[0]), "Source cutout height")
        output_header["CROPX0"] = (x0, "Zero-based source crop x origin")
        output_header["CROPY0"] = (y0, "Zero-based source crop y origin")
        output_wcs = WCS(output_header).celestial
        check_x, check_y = output_wcs.world_to_pixel(reference_world)
        check_residual = max(
            float(np.max(np.abs(np.asarray(check_x) - control_x))),
            float(np.max(np.abs(np.asarray(check_y) - control_y))),
        )
        if check_residual > ARCHIVE_FIELDS_WCS_TOLERANCE_PIXELS:
            raise ValueError(
                f"{band_name} crop header loses alignment by "
                f"{check_residual:.6g}px"
            )
        crops[band_name] = (tile, output_header)
    return crops


_ARCHIVE_HEADER_MANIFEST_KEYS = (
    "BUNIT", "MAGZERO", "FILTER", "INSTRUME", "RADESYS", "EQUINOX",
    "CTYPE1", "CTYPE2", "CUNIT1", "CUNIT2", "CRVAL1", "CRVAL2",
    "CRPIX1", "CRPIX2", "CD1_1", "CD1_2", "CD2_1", "CD2_2",
    "PC1_1", "PC1_2", "PC2_1", "PC2_2", "CDELT1", "CDELT2",
    "SRCNX", "SRCNY", "CROPX0", "CROPY0",
)


def _json_header_value(value: Any) -> str | int | float | bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value)
    return str(value)


def _archive_band_metadata(
    hdu: fits.ImageHDU | fits.CompImageHDU,
    archive_parent: Mapping[str, Any],
) -> dict[str, Any]:
    header = hdu.header
    header_text = header.tostring(sep="\n", endcard=True, padding=False)
    assert hdu.data is not None
    return {
        "shape": [int(side) for side in hdu.data.shape],
        "source_shape": [int(header["SRCNY"]), int(header["SRCNX"])],
        "crop_origin_xy": [int(header["CROPX0"]), int(header["CROPY0"])],
        "header": {
            key: _json_header_value(header[key])
            for key in _ARCHIVE_HEADER_MANIFEST_KEYS
            if key in header
        },
        "header_sha256": hashlib.sha256(header_text.encode("utf-8")).hexdigest(),
        "archive_parent_id": str(archive_parent["parent_id"]),
        "archive_parent_oid": str(archive_parent["mosaic_product_oid"]),
    }


def _validate_archive_field_bundle(
    path: str | os.PathLike[str],
    sample: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate identity, pixels, and exact four-band WCS alignment."""
    target = Path(path)
    expected_primary = {
        "SAMPLEID": int(sample["sample_id"]),
        "SRC_ID": int(sample["source_sample_id"]),
        "PARENT": str(sample["parent_id"]),
        "Q1FIELD": str(sample["field"]),
        "RELEASE": str(sample["source_release"]),
        "SRCPLAN": str(sample["source_plan_fingerprint"]),
        "PLANHASH": str(sample["plan_fingerprint"]),
    }
    band_metadata: dict[str, Any] = {}
    try:
        with fits.open(target, memmap=False) as hdul:
            hdul.verify("exception")
            if len(hdul) != len(Config.LR_INPUT_BAND_NAMES) + 1:
                raise ValueError("expected one primary plus four image HDUs")
            if hdul[0].data is not None:
                raise ValueError("primary HDU must be dataless")
            primary = hdul[0].header
            for key, wanted in expected_primary.items():
                if str(primary.get(key)) != str(wanted):
                    raise ValueError(f"{key} disagrees with the acquisition plan")
            for key, sample_key in (("RA", "ra"), ("DEC", "dec")):
                if not math.isclose(
                    float(primary.get(key)), float(sample[sample_key]), abs_tol=1e-8,
                ):
                    raise ValueError(f"{key} disagrees with the acquisition plan")
            names = [str(hdu.name) for hdu in hdul[1:]]
            if names != list(Config.LR_INPUT_BAND_NAMES):
                raise ValueError(
                    f"image HDUs must be ordered as {list(Config.LR_INPUT_BAND_NAMES)!r}"
                )
            control_x, control_y = _registration_control_pixels(
                ARCHIVE_FIELDS_VIS_PIXELS
            )
            reference_world: SkyCoord | None = None
            archive_parents = sample["archive_parents"]
            for band_name, hdu in zip(
                Config.LR_INPUT_BAND_NAMES, hdul[1:], strict=True,
            ):
                if hdu.data is None or hdu.data.shape != (
                    ARCHIVE_FIELDS_VIS_PIXELS, ARCHIVE_FIELDS_VIS_PIXELS,
                ):
                    raise ValueError(f"{band_name} does not have a 256x256 image")
                if not np.all(np.isfinite(hdu.data)):
                    raise ValueError(f"{band_name} contains non-finite pixels")
                if not math.isfinite(float(hdu.header["MAGZERO"])):
                    raise ValueError(f"{band_name} has no finite MAGZERO")
                band_wcs = WCS(hdu.header).celestial
                if reference_world is None:
                    reference_world = band_wcs.pixel_to_world(control_x, control_y)
                else:
                    mapped_x, mapped_y = band_wcs.world_to_pixel(reference_world)
                    residual = max(
                        float(np.max(np.abs(np.asarray(mapped_x) - control_x))),
                        float(np.max(np.abs(np.asarray(mapped_y) - control_y))),
                    )
                    if residual > ARCHIVE_FIELDS_WCS_TOLERANCE_PIXELS:
                        raise ValueError(
                            f"{band_name} output WCS differs from VIS by {residual:.6g}px"
                        )
                band_metadata[band_name] = _archive_band_metadata(
                    hdu, archive_parents[band_name],
                )
    except (OSError, KeyError, TypeError, ValueError, VerifyError) as exc:
        raise ValueError(f"Invalid archive-field bundle {target}: {exc}") from exc
    return {
        "bands": band_metadata,
        "bundle_sha256": _sha256_file(target),
    }


def _write_archive_field_bundle(
    target_path: str,
    *,
    sample: Mapping[str, Any],
    crops: Mapping[str, tuple[np.ndarray, fits.Header]],
) -> dict[str, Any]:
    """Atomically write and re-open one compact matched four-band bundle."""
    primary = fits.Header()
    primary["SAMPLEID"] = (int(sample["sample_id"]), "Archive-field sample id")
    primary["FIELD_ID"] = (int(sample["sample_id"]), "Alias of SAMPLEID")
    primary["SRC_ID"] = (int(sample["source_sample_id"]), "Source VIS sample id")
    primary["PARENT"] = (str(sample["parent_id"]), "Independent VIS parent id")
    primary["Q1FIELD"] = (str(sample["field"]), "Euclid Q1 deep field")
    primary["ANCHOR"] = (str(sample["anchor_id"]), "Source k-means anchor id")
    primary["POSINDEX"] = (int(sample["position_index"]), "Position within parent")
    primary["POSNAME"] = (str(sample["position_name"]), "Position pattern name")
    primary["RA"] = (float(sample["ra"]), "ICRS right ascension (deg)")
    primary["DEC"] = (float(sample["dec"]), "ICRS declination (deg)")
    primary["RELEASE"] = (str(sample["source_release"]), "Archive release_name")
    primary["SRCPLAN"] = str(sample["source_plan_fingerprint"])
    primary["PLANHASH"] = str(sample["plan_fingerprint"])
    hdul = fits.HDUList([fits.PrimaryHDU(header=primary)])
    for band_name in Config.LR_INPUT_BAND_NAMES:
        tile, header = crops[band_name]
        hdul.append(fits.ImageHDU(data=tile, header=header, name=band_name))
    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    try:
        hdul.writeto(
            temporary, overwrite=True, output_verify="exception", checksum=True,
        )
        _validate_archive_field_bundle(temporary, sample)
        os.replace(temporary, target)
        return _validate_archive_field_bundle(target, sample)
    finally:
        with contextlib.suppress(OSError):
            temporary.unlink()


def _resolve_archive_parents(
    source_sample: Mapping[str, Any],
    *,
    source_release: str,
    parent_download_vis_pixels: int,
) -> dict[str, dict[str, Any]]:
    """Resolve one fully containing parent per band for a complete batch."""
    half_side_deg = (
        int(parent_download_vis_pixels)
        * Config.BAND_VIS.pixel_scale_lr_arcsec
        / 2.0
        / 3600.0
    )
    required_radius_deg = math.sqrt(2.0) * half_side_deg
    parents: dict[str, dict[str, Any]] = {"VIS": dict(source_sample["parent"])}
    for band_name in Config.LR_INPUT_BAND_NAMES[1:]:
        candidates, error = exact_band_parents(
            float(source_sample["ra"]),
            float(source_sample["dec"]),
            required_radius_deg,
            band_name,
            source_release=source_release,
        )
        if not candidates:
            raise RuntimeError(f"{band_name} exact parent: {error}")
        parents[band_name] = dict(candidates[0])
    return parents


def _archive_manifest_summary(manifest: Mapping[str, Any]) -> dict[str, Any]:
    samples = list(manifest.get("samples") or [])
    complete = {"written", "cached", "complete", "completed"}
    return {
        "planned_samples": len(samples),
        "completed_samples": sum(
            str(sample.get("status") or "").lower() in complete
            for sample in samples
        ),
        "failed_samples": sum(sample.get("status") == "failed" for sample in samples),
        "independent_parent_count": len({
            str(sample.get("parent_id"))
            for sample in samples
            if str(sample.get("status") or "").lower() in complete
        }),
        "collection_fingerprint": manifest.get("collection_fingerprint"),
    }


_ARCHIVE_FIELD_REDOWNLOAD_MARKER = "redownload_required"


def _reset_archive_sample_for_redownload(sample: dict[str, Any]) -> None:
    """Mark one bundle for replacement without deleting its last good file.

    The marker is persisted before any network work starts.  A later ordinary
    resume therefore keeps downloading this sample instead of mistaking the
    still-present, pre-refresh bundle for completed work.  The old FITS file
    remains readable until :func:`_write_archive_field_bundle` atomically
    replaces it.
    """
    sample["status"] = "planned"
    sample["error"] = None
    sample["output_path"] = None
    sample["bands"] = {}
    sample.pop("bundle_sha256", None)
    sample[_ARCHIVE_FIELD_REDOWNLOAD_MARKER] = True


def _prepare_archive_samples_for_download(
    samples: list[dict[str, Any]],
    *,
    output_dir: str,
    plan_fingerprint: str,
    force_redownload: bool,
) -> None:
    """Validate reusable bundles or persist a resumable forced refresh."""
    if force_redownload:
        for sample in samples:
            _reset_archive_sample_for_redownload(sample)

    complete_statuses = {"written", "cached", "complete", "completed"}
    for sample in samples:
        sample["plan_fingerprint"] = plan_fingerprint
        # A forced refresh is resumable.  Until a replacement is fully written
        # and validated, do not accept the previous generation that remains at
        # the stable target path.
        if sample.get(_ARCHIVE_FIELD_REDOWNLOAD_MARKER) is True:
            _reset_archive_sample_for_redownload(sample)
            continue
        target = archive_field_bundle_path(output_dir, int(sample["sample_id"]))
        try:
            metadata = _validate_archive_field_bundle(target, sample)
            recorded_sha = str(sample.get("bundle_sha256") or "")
            if recorded_sha and recorded_sha != metadata["bundle_sha256"]:
                raise ValueError("bundle SHA256 disagrees with manifest")
            sample.update(metadata)
            sample["status"] = "cached"
            sample["error"] = None
            sample["output_path"] = os.path.abspath(target)
        except (KeyError, TypeError, ValueError):
            if str(sample.get("status") or "").lower() in complete_statuses:
                sample["error"] = "cached output missing, corrupt, or stale"
            sample["status"] = "planned"
            sample.pop("bundle_sha256", None)


def _run_archive_fields(args: argparse.Namespace, reporter: Reporter) -> int:
    """Build compact 44-by-5 fields using only three new downloads per parent."""
    source_path = str(
        args.source_sampling_manifest or default_archive_fields_source_manifest()
    )
    source, source_sha256 = _archive_source_manifest(source_path)
    if str(args.source_release) != str(source["source_release"]):
        raise ValueError(
            f"Source manifest is frozen to {source['source_release']!r}, not "
            f"{args.source_release!r}"
        )
    output_dir = str(args.output_dir)
    manifest_path = Path(
        args.sampling_manifest
        or os.path.join(output_dir, Config.EuclidSky.ARCHIVE_FIELDS_MANIFEST_FILENAME)
    )
    expected_plan = _archive_field_plan(source, source_sha256)
    expected_plan_fingerprint = compute_plan_fingerprint(expected_plan)
    if manifest_path.is_file() and not args.regenerate_catalog:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot reuse archive-field manifest: {exc}") from exc
        if (
            manifest.get("kind") != ARCHIVE_FIELDS_MANIFEST_KIND
            or manifest.get("version") != ARCHIVE_FIELDS_MANIFEST_VERSION
            or manifest.get("plan") != expected_plan
            or manifest.get("plan_fingerprint") != expected_plan_fingerprint
            or len(manifest.get("samples") or [])
            != len(source["samples"]) * ARCHIVE_FIELDS_POSITIONS_PER_PARENT
        ):
            raise ValueError(
                "Existing archive-field manifest does not match the immutable "
                "source and 44-by-5 acquisition plan; pass --regenerate-catalog"
            )
        reporter.set_stage("reusing matched archive-field manifest")
    else:
        manifest = _new_archive_fields_manifest(source, source_sha256)
        if not args.dry_run:
            _atomic_write_json(manifest_path, manifest)

    if args.dry_run:
        print(json.dumps(_archive_manifest_summary(manifest), indent=2, sort_keys=True))
        print(f"source manifest = {source_path}")
        print(f"output manifest = {manifest_path}")
        return 0

    source_by_id = {
        int(sample["sample_id"]): sample for sample in source["samples"]
    }
    samples = list(manifest["samples"])
    _prepare_archive_samples_for_download(
        samples,
        output_dir=output_dir,
        plan_fingerprint=expected_plan_fingerprint,
        force_redownload=bool(getattr(args, "force_redownload", False)),
    )
    manifest["collection_fingerprint"] = compute_collection_fingerprint(manifest)
    _atomic_write_json(manifest_path, manifest)

    done = sum(sample["status"] == "cached" for sample in samples)
    reporter.set_step(done, len(samples), "validated cached archive fields")
    reporter.set_stage("downloading parent-batched matched archive fields")
    parent_download_pixels = int(expected_plan["parent_download_size_vis_pixels"])
    for source_sample_id in range(len(source_by_id)):
        group = [
            sample for sample in samples
            if int(sample["source_sample_id"]) == source_sample_id
        ]
        pending = [sample for sample in group if sample["status"] != "cached"]
        if not pending:
            continue
        source_sample = source_by_id[source_sample_id]
        temp_dir = tempfile.mkdtemp(prefix=f"archive_parent_{source_sample_id:04d}_")
        try:
            stored_parents = group[0].get("archive_parents")
            if not isinstance(stored_parents, Mapping) or set(stored_parents) != set(
                Config.LR_INPUT_BAND_NAMES
            ):
                stored_parents = _resolve_archive_parents(
                    source_sample,
                    source_release=str(source["source_release"]),
                    parent_download_vis_pixels=parent_download_pixels,
                )
            archive_parents = {
                band_name: dict(stored_parents[band_name])
                for band_name in Config.LR_INPUT_BAND_NAMES
            }
            for sample in group:
                sample["archive_parents"] = archive_parents

            vis_path = _source_vis_bundle_path(source_path, source_sample)
            raw_bands: dict[str, tuple[np.ndarray, fits.Header]] = {
                "VIS": _validate_source_vis_bundle(
                    vis_path, source_sample, str(source["source_release"]),
                ),
            }
            for band_name in Config.LR_INPUT_BAND_NAMES[1:]:
                raw_path = os.path.join(temp_dir, f"{band_name}.fits")
                _download_parent_band(
                    band_name=band_name,
                    ra=float(source_sample["ra"]),
                    dec=float(source_sample["dec"]),
                    parent_download_vis_pixels=parent_download_pixels,
                    parent=archive_parents[band_name],
                    output_path=raw_path,
                )
                raw_bands[band_name] = _read_archive_image(raw_path, band_name)

            prepared: dict[int, dict[str, tuple[np.ndarray, fits.Header]]] = {}
            for sample in pending:
                prepared[int(sample["sample_id"])] = _aligned_archive_crops(
                    raw_bands, ra=float(sample["ra"]), dec=float(sample["dec"]),
                )
            for sample in pending:
                target = archive_field_bundle_path(output_dir, int(sample["sample_id"]))
                metadata = _write_archive_field_bundle(
                    target,
                    sample=sample,
                    crops=prepared[int(sample["sample_id"])],
                )
                sample.update(metadata)
                sample["status"] = "written"
                sample["error"] = None
                sample["output_path"] = os.path.abspath(target)
                sample.pop(_ARCHIVE_FIELD_REDOWNLOAD_MARKER, None)
                done += 1
                reporter.set_step(
                    done, len(samples), f"archive field {int(sample['sample_id']):04d}",
                )
        except Exception as exc:  # noqa: BLE001 - persisted for resumable runs
            error = f"{type(exc).__name__}: {exc}"
            reporter.warn(f"source VIS sample {source_sample_id}: {error}")
            for sample in pending:
                if sample.get("status") != "written":
                    sample["status"] = "failed"
                    sample["error"] = error
        finally:
            with contextlib.suppress(OSError):
                for child in Path(temp_dir).iterdir():
                    child.unlink()
            with contextlib.suppress(OSError):
                Path(temp_dir).rmdir()
            manifest["collection_fingerprint"] = compute_collection_fingerprint(manifest)
            _atomic_write_json(manifest_path, manifest)

    if _sha256_file(Path(source_path)) != source_sha256:
        raise RuntimeError("Source VIS sampling manifest changed during acquisition")
    summary = _archive_manifest_summary(manifest)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"manifest = {manifest_path}")
    return 0 if summary["completed_samples"] == summary["planned_samples"] else 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--sampling-mode", choices=("single-disk", "star-support", "archive-fields"),
        default="single-disk",
        help=(
            "single-disk preserves the round-trip downloader; star-support "
            "builds equal-area spherical anchors over the saved Q1 star "
            "footprint and downloads one VIS field per unique parent mosaic; "
            "archive-fields derives five compact matched four-band fields "
            "from each frozen VIS parent"
        ),
    )
    p.add_argument("--output-dir", default=None,
                   help="Root for the sky catalogue CSV and bundled "
                        "FITS cutouts. Default: data/euclid_sky for "
                        "single-disk mode, vis_noise_samples for star-support, "
                        "and archive_fields for archive-fields.")
    p.add_argument("--n-positions", type=int, default=100,
                   help="Number of random sky positions to generate. "
                        "After per-position 4-band download some will "
                        "drop out (off-coverage, tile boundary, NISP "
                        "missing) — over-sample by ~2× the count of "
                        "fully-multi-band positions you need.")
    p.add_argument("--ra-centre", type=float, default=270.0,
                   help="Disk centre RA (deg). Default 270.")
    p.add_argument("--dec-centre", type=float, default=66.0,
                   help="Disk centre Dec (deg). Default 66 — a deep "
                        "Euclid coverage region near the NEP.")
    p.add_argument("--radius-deg", type=float, default=2.0,
                   help="Disk radius (deg). Default 2.")
    p.add_argument("--vis-pixels", type=int, default=512,
                   help="Cutout size in VIS pixels (= 0.10\"/pix). "
                        "Default 512 (= 51.2\") gives plenty of room "
                        "for the TFRecord chopper to extract many "
                        "training stamps per position; the same "
                        "angular extent gets requested from each band "
                        "at its own native pixel scale.")
    p.add_argument(
        "--workers", type=int, default=None,
        help=(
            "Parallel positions in flight. Defaults to 1 for independent "
            "VIS-noise sampling (the archive cutout service is unreliable "
            "under concurrent large requests) and 8 for four-band sky fields."
        ),
    )
    p.add_argument("--regenerate-catalog", action="store_true",
                   help="Overwrite the existing sky-positions CSV. "
                        "Without this flag, an existing catalogue is "
                        "reused (positions stay fixed across runs so "
                        "the bundle ids line up).")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed for position generation. Only "
                        "matters on the first run (or with "
                        "--regenerate-catalog).")
    p.add_argument(
        "--force-redownload",
        action="store_true",
        help=(
            "Archive-fields mode only: ignore reusable four-band bundles, "
            "download every NISP parent again, and atomically replace all "
            "220 matched bundles. The forced refresh remains resumable if "
            "the job is interrupted."
        ),
    )
    p.add_argument(
        "--star-support-csv",
        default=os.path.join(Config.DEFAULT_OUTPUT_DIR, Config.CATALOG_FILE),
        help="Saved calibration stars used only as occupied-sky support.",
    )
    p.add_argument("--n-clusters", type=int, default=44,
                   help="Area-weighted spherical support anchors.")
    p.add_argument("--samples-per-cluster", type=int, default=1,
                   help="Requested primary VIS samples per support anchor.")
    p.add_argument("--support-cell-area-deg2", type=float, default=0.04,
                   help="Equal-solid-angle occupancy cell area.")
    p.add_argument("--jitter-radius-deg", type=float, default=0.15,
                   help="Spherical random-draw radius around each anchor.")
    p.add_argument("--avoid-star-arcsec", type=float, default=30.0,
                   help="Minimum distance from every saved calibration star.")
    p.add_argument("--minimum-separation-arcmin", type=float, default=6.5,
                   help="Minimum separation between selected field centres.")
    p.add_argument("--candidates-per-sample", type=int, default=24,
                   help="Coverage-resolver retries prepared per sample slot.")
    p.add_argument("--source-release", default="Q1_R1",
                   help="Exact sedm.mosaic_product.release_name value.")
    p.add_argument("--sampling-manifest", default=None,
                   help="Output manifest override for a manifest-driven mode.")
    p.add_argument(
        "--source-sampling-manifest",
        default=None,
        help=(
            "Immutable 44-parent VIS sampling manifest used by archive-fields "
            "(default data/euclid_sky/vis_noise_samples/"
            "vis_noise_sampling_manifest.json)."
        ),
    )
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would be done and exit.")
    return p.parse_args()


def _dir_size_bytes(path: str) -> int:
    """Sum of file sizes under ``path``; returns 0 if the dir is absent."""
    total = 0
    if not os.path.isdir(path):
        return 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                continue
    return total


def _sampling_manifest_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    samples = list(payload.get("samples") or [])
    completed = [
        sample for sample in samples
        if sample.get("status") in {"written", "cached"}
    ]
    return {
        "planned_samples": len(samples),
        "completed_samples": len(completed),
        "failed_samples": sum(sample.get("status") == "failed" for sample in samples),
        "independent_parent_count": len({
            str(sample.get("parent_id")) for sample in completed
            if sample.get("parent_id")
        }),
    }


def _write_positions_from_manifest(
    manifest: Mapping[str, Any],
    catalog_path: str,
) -> None:
    rows = [{
        "id": int(sample["sample_id"]),
        "ra": float(sample["ra"]),
        "dec": float(sample["dec"]),
        "magnitude": np.nan,
        "field": str(sample["field"]),
        "anchor_id": str(sample["anchor_id"]),
        "parent_id": str(sample["parent_id"]),
        "tile_index": str(sample["parent"]["tile_index"]),
    } for sample in manifest.get("samples", [])]
    pd.DataFrame.from_records(rows).to_csv(catalog_path, index=False)


def _new_star_support_manifest(args: argparse.Namespace) -> dict[str, Any]:
    plan = _star_support_plan(args)
    anchors, stars = build_star_support_anchors(
        args.star_support_csv,
        n_clusters=int(args.n_clusters),
        cell_area_deg2=float(args.support_cell_area_deg2),
        seed=int(args.seed),
    )
    candidates = propose_star_support_samples(
        anchors,
        stars,
        samples_per_anchor=int(args.samples_per_cluster),
        candidates_per_sample=int(args.candidates_per_sample),
        jitter_radius_deg=float(args.jitter_radius_deg),
        avoid_star_arcsec=float(args.avoid_star_arcsec),
        minimum_sample_separation_arcmin=float(args.minimum_separation_arcmin),
        seed=int(args.seed),
    )
    arcsec_side = int(args.vis_pixels) * Config.BAND_VIS.pixel_scale_lr_arcsec
    cutout_half_side_deg = (arcsec_side / 2.0) / 3600.0
    # The cutout service's radius sets the half-width of a square image.  Use
    # its circumscribed circle for parent selection so even the four corners
    # are inside the same mosaic footprint.
    cutout_radius_deg = math.sqrt(2.0) * cutout_half_side_deg

    def resolve(ra: float, dec: float, radius: float):
        return exact_vis_parents(
            ra, dec, radius, source_release=str(args.source_release),
        )

    samples, rejections = assign_unique_parents(
        candidates,
        samples_per_anchor=int(args.samples_per_cluster),
        cutout_radius_deg=cutout_radius_deg,
        minimum_sample_separation_arcmin=float(args.minimum_separation_arcmin),
        parent_resolver=resolve,
    )
    source_path = Path(args.star_support_csv)
    source_digest = str(plan["stars_csv_sha256"])
    return {
        "version": SAMPLING_MANIFEST_VERSION,
        "kind": "euclid_vis_noise_sampling",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": int(args.seed),
        "source_release": str(args.source_release),
        "plan": plan,
        "plan_fingerprint": _plan_fingerprint(plan),
        "archive": {
            "table": "sedm.mosaic_product",
            "instrument_name": "VIS",
            "technique": "IMAGE",
            "product_type": "DpdMerBksMosaic",
            "requested_release_name": str(args.source_release),
            "coverage_predicate": (
                "INTERSECTS(mosaic_product.fov, request_circle) TAP prefilter; "
                "client-side convex spherical half-space clearance proves "
                "the circumscribed cutout circle is contained"
            ),
            "one_primary_sample_per_parent": True,
            "auxiliary_products": {
                "science_image": True,
                "rms_map": False,
                "flag_map": False,
                "weight_map": False,
            },
            "auxiliary_caveat": (
                "Science-image-only MVP; source masks and robust statistics "
                "are data-derived, not official RMS/flag/weight products"
            ),
        },
        "support": {
            "stars_csv": str(source_path.resolve()),
            "stars_csv_sha256": source_digest,
            "finite_q1_star_positions": int(len(stars)),
            "role": "binary occupied-sky support; never a density weight",
            "cell_area_deg2": float(args.support_cell_area_deg2),
            "q1_fields": [row[0] for row in Q1_SUPPORT_REGIONS],
        },
        "selection": {
            "method": "field-stratified equal-area spherical k-means medoids plus spherical jitter",
            "n_clusters": int(args.n_clusters),
            "samples_per_cluster": int(args.samples_per_cluster),
            "jitter_radius_deg": float(args.jitter_radius_deg),
            "avoid_star_arcsec": float(args.avoid_star_arcsec),
            "minimum_sample_separation_arcmin": float(args.minimum_separation_arcmin),
            "cutout_size_vis_pixels": int(args.vis_pixels),
            "archive_request_half_side_deg": cutout_half_side_deg,
            "required_coverage_radius_deg": cutout_radius_deg,
            "rejections": rejections,
        },
        "anchors": anchors.to_dict("records"),
        "samples": samples,
    }


def _run_star_support_sampling(
    args: argparse.Namespace,
    reporter: Reporter,
) -> int:
    output_dir = str(args.output_dir)
    manifest_path = Path(
        args.sampling_manifest or os.path.join(output_dir, SAMPLING_MANIFEST_NAME)
    )
    if manifest_path.is_file() and not args.regenerate_catalog:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("version") != SAMPLING_MANIFEST_VERSION:
            raise ValueError(
                f"Unsupported sampling manifest version {manifest.get('version')!r}"
            )
        current_plan = _star_support_plan(args)
        if (
            manifest.get("plan") != current_plan
            or manifest.get("plan_fingerprint") != _plan_fingerprint(current_plan)
        ):
            raise ValueError(
                "Existing VIS-noise manifest does not match the current "
                "stars.csv or frozen sampling inputs; pass "
                "--regenerate-catalog to build a new sampling plan"
            )
        manifest_release = str(manifest.get("source_release") or "")
        if manifest_release != str(args.source_release):
            raise ValueError(
                "Existing VIS-noise manifest is frozen to release_name "
                f"{manifest_release!r}, not {args.source_release!r}; pass "
                "--regenerate-catalog to build a new sampling plan"
            )
        manifest_pixels = int(
            (manifest.get("selection") or {}).get("cutout_size_vis_pixels", -1)
        )
        if manifest_pixels != int(args.vis_pixels):
            raise ValueError(
                "Existing VIS-noise manifest uses "
                f"{manifest_pixels} VIS pixels, not {args.vis_pixels}; pass "
                "--regenerate-catalog to build a new sampling plan"
            )
        reporter.set_stage("reusing VIS-noise sampling manifest")
    else:
        if args.dry_run:
            anchors, stars = build_star_support_anchors(
                args.star_support_csv,
                n_clusters=int(args.n_clusters),
                cell_area_deg2=float(args.support_cell_area_deg2),
                seed=int(args.seed),
            )
            print(
                f"DRY RUN — {len(stars)} Q1 support stars collapse to "
                f"{len(anchors)} equal-area spherical anchors; no archive query run"
            )
            return 0
        reporter.set_stage("resolving exact VIS mosaic parents")
        manifest = _new_star_support_manifest(args)
        _atomic_write_json(manifest_path, manifest)

    if args.dry_run:
        print(json.dumps(_sampling_manifest_summary(manifest), indent=2))
        return 0
    os.makedirs(output_dir, exist_ok=True)
    catalog_path = os.path.join(output_dir, Config.EuclidSky.SKY_CATALOG_FILENAME)
    _write_positions_from_manifest(manifest, catalog_path)

    samples = list(manifest.get("samples") or [])
    if not samples:
        raise RuntimeError("No exact-coverage, unique-parent VIS samples were planned")
    invalidated = 0
    manifest_release = str(manifest.get("source_release") or args.source_release)
    for sample in samples:
        if sample.get("status") not in {"written", "cached"}:
            continue
        target = bundle_path_for_id(output_dir, int(sample["sample_id"]))
        if _cached_planned_bundle_matches(
            target,
            sample,
            vis_pixels=int(args.vis_pixels),
            source_release=manifest_release,
        ):
            sample["status"] = "cached"
            sample["error"] = None
            sample["output_path"] = os.path.abspath(target)
            with fits.open(target, memmap=True) as hdul:
                sample["actual_shape"] = list(hdul["VIS"].shape)
            continue
        sample["status"] = "planned"
        sample["error"] = "cached output missing, corrupt, or stale; scheduled again"
        invalidated += 1
    if invalidated:
        reporter.warn(f"revalidating and re-downloading {invalidated} stale VIS samples")
        _atomic_write_json(manifest_path, manifest)
    arcsec_side = int(args.vis_pixels) * Config.BAND_VIS.pixel_scale_lr_arcsec
    reporter.set_stage("downloading independent VIS parent samples")
    sample_by_id = {int(sample["sample_id"]): sample for sample in samples}
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max(1, int(args.workers)),
    ) as pool:
        future_to_id = {
            pool.submit(
                _fetch_planned_vis_bundle,
                sample,
                vis_pixels=int(args.vis_pixels),
                arcsec_side=arcsec_side,
                output_dir=output_dir,
                source_release=str(manifest.get("source_release") or args.source_release),
            ): int(sample["sample_id"])
            for sample in samples
            if sample.get("status") not in {"written", "cached"}
        }
        done = sum(
            sample.get("status") in {"written", "cached"} for sample in samples
        )
        for future in concurrent.futures.as_completed(future_to_id):
            sample_id = future_to_id[future]
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001 - persisted in manifest
                result = {
                    "status": "failed",
                    "errors": [f"{type(exc).__name__}: {exc}"],
                }
            sample = sample_by_id[sample_id]
            sample["status"] = str(result["status"])
            sample["error"] = "; ".join(result.get("errors") or []) or None
            sample["output_path"] = os.path.abspath(
                bundle_path_for_id(output_dir, sample_id)
            )
            if result.get("actual_shape") is not None:
                sample["actual_shape"] = [
                    int(value) for value in result["actual_shape"]
                ]
            if sample["status"] in {"written", "cached"}:
                done += 1
            reporter.set_step(done, len(samples), f"VIS sample {sample_id}")
            _atomic_write_json(manifest_path, manifest)
    summary = _sampling_manifest_summary(manifest)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"manifest = {manifest_path}")
    return 0


def main() -> int:
    args = parse_args()
    if args.force_redownload and args.sampling_mode != "archive-fields":
        raise ValueError("--force-redownload requires --sampling-mode archive-fields")
    args.workers = _worker_count(args.sampling_mode, args.workers)
    args.source_release = str(args.source_release).strip()
    if not args.source_release:
        raise ValueError("--source-release must be non-empty")
    if args.output_dir is None:
        if args.sampling_mode == "star-support":
            args.output_dir = default_vis_noise_output_dir()
        elif args.sampling_mode == "archive-fields":
            args.output_dir = default_archive_fields_output_dir()
        else:
            args.output_dir = Config.EUCLID_SKY_DIR
    reporter = Reporter.from_env()
    if args.sampling_mode == "star-support":
        return _run_star_support_sampling(args, reporter)
    if args.sampling_mode == "archive-fields":
        return _run_archive_fields(args, reporter)
    arcsec_side = args.vis_pixels * Config.BAND_VIS.pixel_scale_lr_arcsec

    print("=" * 64)
    print("  Euclid sky cutout download (for round-trip training)")
    print("=" * 64)
    print(f"  output dir       = {args.output_dir}")
    print(f"  sky disk         = (RA={args.ra_centre:.3f}°, "
          f"Dec={args.dec_centre:.3f}°, r={args.radius_deg:.2f}°)")
    print(f"  n_positions      = {args.n_positions}")
    print(f"  cutout size      = {args.vis_pixels} VIS px "
          f"(= {arcsec_side:.1f}\")")
    print(f"  bands (bundled)  = {list(Config.LR_INPUT_BAND_NAMES)}")
    print(f"  workers          = {args.workers}")
    print()

    t0 = time.perf_counter()

    # ---- 1. Sky catalogue: generate or reuse ----
    os.makedirs(args.output_dir, exist_ok=True)
    catalog_path = os.path.join(args.output_dir, Config.EuclidSky.SKY_CATALOG_FILENAME)
    if os.path.isfile(catalog_path) and not args.regenerate_catalog:
        positions = pd.read_csv(catalog_path)
        reporter.set_stage("reusing sky catalog")
        print(f"[1/2] reusing sky catalogue: {len(positions)} positions "
              f"in {catalog_path}")
    else:
        rng = np.random.default_rng(args.seed)
        positions = _uniform_disk_positions(
            args.ra_centre, args.dec_centre, args.radius_deg,
            args.n_positions, rng=rng,
        )
        if args.dry_run:
            print(f"[1/2] DRY RUN — would generate {len(positions)} positions "
                  f"and write to {catalog_path}")
        else:
            reporter.set_stage("generating sky catalog")
            positions.to_csv(catalog_path, index=False)
            print(f"[1/2] generated sky catalogue: {len(positions)} positions → "
                  f"{catalog_path}")
            for _, row in positions.head(3).iterrows():
                print(f"        id={int(row['id']):03d}  "
                      f"RA={row['ra']:9.5f}°  Dec={row['dec']:+9.5f}°")

    if args.dry_run:
        print()
        print(f"  DRY RUN — would fetch {len(Config.LR_INPUT_BAND_NAMES)} bands × "
              f"{len(positions)} positions at "
              f"{args.vis_pixels} VIS px each, bundling per position.")
        runtime = time.perf_counter() - t0
        print(f"\nRUNTIME_SECONDS={runtime:.1f}")
        return 0

    # ---- 2. Per-position bundled download ----
    reporter.set_stage("downloading bundles")
    cutouts_dir = os.path.join(args.output_dir, Config.EuclidSky.CUTOUTS_SUBDIR)
    os.makedirs(cutouts_dir, exist_ok=True)

    n_positions = len(positions)
    n_written  = 0
    n_cached   = 0
    n_failed   = 0

    # Materialise the (id, ra, dec) work-list up front so the
    # ThreadPoolExecutor can dispatch from a simple iterable.
    work = [
        (int(row["id"]), float(row["ra"]), float(row["dec"]))
        for _, row in positions.iterrows()
    ]

    completed_i = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        future_to_id = {
            pool.submit(
                _fetch_position_bundle,
                pos_id=pid, ra=ra, dec=dec,
                vis_pixels=args.vis_pixels, arcsec_side=arcsec_side,
                output_dir=args.output_dir,
            ): pid
            for (pid, ra, dec) in work
        }
        for fut in concurrent.futures.as_completed(future_to_id):
            pid = future_to_id[fut]
            completed_i += 1
            reporter.set_step(completed_i, n_positions, f"sky_{pid:04d}")
            try:
                result = fut.result()
            except Exception as e:
                n_failed += 1
                reporter.warn(f"position {pid} crashed: {type(e).__name__}: {e}")
                continue
            status = result["status"]
            if status == "written":
                n_written += 1
            elif status == "cached":
                n_cached += 1
            else:
                n_failed += 1
                for err in result["errors"]:
                    reporter.warn(f"position {pid} failed: {err}")

    # ---- 3. Summary ----
    total_bytes = _dir_size_bytes(cutouts_dir)
    print()
    print("=" * 64)
    print(f"Summary  ({(time.perf_counter() - t0) / 60:.1f} min total):")
    print(f"  positions written  = {n_written}")
    print(f"  positions cached   = {n_cached}  (already on disk; skipped)")
    print(f"  positions failed   = {n_failed}  (couldn't get all 4 bands)")
    print(f"  bundle dir size    = {total_bytes / 1e9:.2f} GB  ({cutouts_dir})")

    runtime = time.perf_counter() - t0
    print(f"\nRUNTIME_SECONDS={runtime:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
