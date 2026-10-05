"""Read-only facade over a validated local TNG SKIRT atlas."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from euclid_polish.config import Config
from euclid_polish.tng.catalog import TNGPropertyCatalog
from euclid_polish.tng.types import TNG_FITS_BANDS, TNGView

if TYPE_CHECKING:
    from euclid_polish.tng.radius_manifest import TNGRadiusManifest


_ORIENTATIONS = range(1, 6)


def _resolve_fits_path(
    directory: Path,
    subhalo_id: str,
    orientation: int,
    band: str,
) -> Path:
    if isinstance(orientation, bool) or not isinstance(orientation, int):
        raise TypeError("orientation must be an integer in 1..5")
    if orientation not in _ORIENTATIONS:
        raise ValueError(f"orientation must be in 1..5, got {orientation!r}")
    fits_band = str(band).strip().upper()
    if fits_band not in TNG_FITS_BANDS:
        raise ValueError(f"unknown TNG FITS band {band!r}")

    unpadded = directory / (
        f"TNG{subhalo_id}_O{orientation}_Euclid_{fits_band}.fits"
    )
    if unpadded.is_file():
        return unpadded
    try:
        padded_id = f"{int(subhalo_id):06d}"
    except ValueError:
        return unpadded
    padded = directory / (
        f"TNG{padded_id}_O{orientation}_Euclid_{fits_band}.fits"
    )
    return padded if padded.is_file() else unpadded


@dataclass(frozen=True, slots=True)
class TNGGalaxy:
    """One completed galaxy directory in the local SKIRT atlas."""

    directory: Path
    subhalo_id: str

    def __post_init__(self) -> None:
        subhalo_id = str(self.subhalo_id).strip()
        if not subhalo_id:
            raise ValueError("subhalo_id must be non-empty")
        object.__setattr__(self, "directory", Path(self.directory))
        object.__setattr__(self, "subhalo_id", subhalo_id)

    def fits_path(self, orientation: int, band: str) -> Path:
        """Resolve one orientation/band, including zero-padded filenames."""
        return _resolve_fits_path(
            self.directory, self.subhalo_id, orientation, band
        )

    @classmethod
    def discover(cls, root: str | Path) -> tuple[TNGGalaxy, ...]:
        """Return every complete galaxy stored under an atlas root."""
        return _scan_complete_galaxies(Path(root))


def _galaxy_sort_key(galaxy: TNGGalaxy) -> tuple[int, int | str]:
    try:
        return (0, int(galaxy.subhalo_id))
    except ValueError:
        return (1, galaxy.subhalo_id)


def _scan_complete_galaxies(root: Path) -> tuple[TNGGalaxy, ...]:
    if not root.is_dir():
        return ()
    galaxies: list[TNGGalaxy] = []
    for directory in root.iterdir():
        if not directory.is_dir():
            continue
        if not (directory / Config.Tng.DONE_MARKER).is_file():
            continue
        galaxy = TNGGalaxy(directory=directory, subhalo_id=directory.name)
        if all(
            galaxy.fits_path(orientation, band).is_file()
            for orientation in _ORIENTATIONS
            for band in TNG_FITS_BANDS
        ):
            galaxies.append(galaxy)
    return tuple(sorted(galaxies, key=_galaxy_sort_key))


def _positive_finite(value: float, name: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and positive, got {value!r}")
    return number


@dataclass(frozen=True, slots=True, repr=False)
class TNGAtlas:
    """Validated, immutable inventory/catalog/radius facade.

    Opening an atlas is deliberately read-only. Manifest construction and
    repair remain explicit preparation operations in :mod:`radius_manifest`.
    Galaxy membership and each galaxy's largest native R_e are precomputed at
    construction, because generation asks for donor eligibility on every
    field-galaxy draw.
    """

    root: Path
    galaxies: tuple[TNGGalaxy, ...]
    properties: TNGPropertyCatalog
    radii: TNGRadiusManifest
    _members: frozenset[TNGGalaxy] = field(init=False, repr=False, compare=False)
    #: ``radii.max_radius`` per galaxy, in ``galaxies`` order (read-only).
    _max_native_re_px: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        from euclid_polish.tng.radius_manifest import TNGRadiusManifest

        object.__setattr__(self, "root", Path(self.root))
        object.__setattr__(self, "galaxies", tuple(self.galaxies))
        if not all(isinstance(galaxy, TNGGalaxy) for galaxy in self.galaxies):
            raise TypeError("galaxies must contain only TNGGalaxy values")
        subhalo_ids = tuple(galaxy.subhalo_id for galaxy in self.galaxies)
        if len(set(subhalo_ids)) != len(subhalo_ids):
            raise ValueError("TNG atlas contains duplicate subhalo ids")
        if not isinstance(self.properties, TNGPropertyCatalog):
            raise TypeError("properties must be a TNGPropertyCatalog")
        if not isinstance(self.radii, TNGRadiusManifest):
            raise TypeError("radii must be a TNGRadiusManifest")
        expected_views = {
            (subhalo_id, orientation)
            for subhalo_id in subhalo_ids
            for orientation in _ORIENTATIONS
        }
        if set(self.radii) != expected_views:
            raise ValueError(
                "TNG atlas galaxies and radius manifest must describe the "
                "same complete views"
            )
        object.__setattr__(self, "_members", frozenset(self.galaxies))
        max_native_re_px = np.fromiter(
            (self.radii.max_radius(subhalo_id) for subhalo_id in subhalo_ids),
            dtype=np.float64,
            count=len(subhalo_ids),
        )
        max_native_re_px.setflags(write=False)
        object.__setattr__(self, "_max_native_re_px", max_native_re_px)

    @classmethod
    def open(
        cls,
        root: str | Path,
        properties_path: str | Path | None = None,
        manifest_path: str | Path | None = None,
    ) -> TNGAtlas:
        """Open an existing submit-ready atlas without repairing artifacts."""
        from euclid_polish.tng.radius_manifest import (
            TNGRadiusManifest,
            validate_manifest,
        )
        from euclid_polish.tng.radius_manifest import (
            manifest_path as resolve_manifest_path,
        )

        atlas_root = Path(root).expanduser().resolve()
        properties_source = (
            Path(properties_path).expanduser().resolve()
            if properties_path is not None
            else Path(
                Config.DATA_DIR,
                "_tng_infographics",
                "tng_properties.csv",
            ).resolve()
        )
        manifest_source = Path(
            resolve_manifest_path(
                str(Path(manifest_path).expanduser().resolve())
                if manifest_path is not None
                else None,
            )
        )
        status = validate_manifest(
            str(atlas_root),
            properties_path=str(properties_source),
            manifest_path_value=str(manifest_source),
        )
        if not status.get("valid"):
            reasons = list(status.get("reasons") or [])
            if not reasons and status.get("reason"):
                reasons.append(str(status["reason"]))
            detail = "; ".join(reasons) or "unknown validation failure"
            raise ValueError(f"TNG atlas manifest is not valid: {detail}")

        radii = TNGRadiusManifest.read(manifest_source)
        properties = TNGPropertyCatalog.read(properties_source)
        galaxies = _scan_complete_galaxies(atlas_root)
        expected_views = {
            (galaxy.subhalo_id, orientation)
            for galaxy in galaxies
            for orientation in _ORIENTATIONS
        }
        radius_views = set(radii)
        if expected_views != radius_views:
            missing = len(expected_views - radius_views)
            incomplete = len(radius_views - expected_views)
            raise ValueError(
                "TNG atlas inventory and radius manifest disagree "
                f"({missing} missing radii, {incomplete} incomplete-file views)"
            )
        return cls(
            root=atlas_root,
            galaxies=galaxies,
            properties=properties,
            radii=radii,
        )

    def __iter__(self) -> Iterator[TNGGalaxy]:
        return iter(self.galaxies)

    def __len__(self) -> int:
        return len(self.galaxies)

    def __repr__(self) -> str:
        return (
            f"TNGAtlas(root={str(self.root)!r}, galaxies={len(self)}, "
            f"fingerprint={self.fingerprint!r})"
        )

    @property
    def fingerprint(self) -> str:
        """Return the validated inventory/radius-manifest fingerprint."""
        return self.radii.fingerprint

    def _require_galaxy(self, galaxy: TNGGalaxy) -> TNGGalaxy:
        if not isinstance(galaxy, TNGGalaxy):
            raise TypeError("galaxy must be a TNGGalaxy")
        if galaxy not in self._members:
            raise ValueError(
                f"TNG{galaxy.subhalo_id} is not a complete galaxy in this atlas"
            )
        return galaxy

    def max_native_re_px(self, galaxy: TNGGalaxy) -> float:
        """Return the largest manifest radius over the galaxy's five views."""
        selected = self._require_galaxy(galaxy)
        return self.radii.max_radius(selected.subhalo_id)

    def eligible_indices(
        self,
        target_re_arcsec: float,
        pixel_scale_arcsec: float,
    ) -> np.ndarray:
        """Return ascending ``galaxies`` indices with a shrink-only view.

        A galaxy qualifies when its largest native R_e reaches
        ``target / pixel_scale`` pixels; one vectorised comparison over the
        precomputed maxima.
        """
        target = _positive_finite(target_re_arcsec, "target_re_arcsec")
        pixel_scale = _positive_finite(
            pixel_scale_arcsec, "pixel_scale_arcsec"
        )
        minimum_native_re_px = target / pixel_scale
        return np.flatnonzero(self._max_native_re_px >= minimum_native_re_px)

    def eligible_galaxies(
        self,
        target_re_arcsec: float,
        pixel_scale_arcsec: float,
    ) -> tuple[TNGGalaxy, ...]:
        """Return galaxies having at least one shrink-only eligible view."""
        return tuple(
            self.galaxies[index]
            for index in self.eligible_indices(
                target_re_arcsec, pixel_scale_arcsec,
            )
        )

    def view(self, galaxy: TNGGalaxy, orientation: int) -> TNGView:
        """Materialize one typed view from validated atlas metadata."""
        selected = self._require_galaxy(galaxy)
        if isinstance(orientation, bool) or not isinstance(orientation, int):
            raise TypeError("orientation must be an integer in 1..5")
        if orientation not in _ORIENTATIONS:
            raise ValueError(f"orientation must be in 1..5, got {orientation!r}")
        native_re_px = self.radii.radius(selected.subhalo_id, orientation)
        return TNGView(
            galaxy_dir=selected.directory,
            subhalo_id=selected.subhalo_id,
            orientation=orientation,
            native_re_px=native_re_px,
            radius_manifest_fingerprint=self.radii.fingerprint,
        )

    def eligible_views(
        self,
        galaxy: TNGGalaxy,
        target_re_arcsec: float,
        pixel_scale_arcsec: float,
    ) -> tuple[TNGView, ...]:
        """Return this galaxy's orientations that need no enlargement."""
        selected = self._require_galaxy(galaxy)
        target = _positive_finite(target_re_arcsec, "target_re_arcsec")
        pixel_scale = _positive_finite(
            pixel_scale_arcsec, "pixel_scale_arcsec"
        )
        return tuple(
            view
            for orientation in _ORIENTATIONS
            if (
                view := self.view(selected, orientation)
            ).native_re_px * pixel_scale >= target
        )
