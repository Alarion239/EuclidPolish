"""Atlas file naming: the leaf that ``types``, ``atlas`` and ``_image`` all
import, so none of them has to defer an import to break a cycle."""
from __future__ import annotations

from pathlib import Path

TNG_FITS_BANDS: tuple[str, ...] = ("VIS", "Y", "J", "H")
ORIENTATIONS = range(1, 6)


def dust_free_twin(path: str | Path) -> Path:
    """The atlas's dust-free twin of a dusty frame: ``…_<band>_nodust.fits``."""
    path = Path(path)
    return path.with_name(f"{path.stem}_nodust{path.suffix}")


def resolve_fits_path(
    directory: Path,
    subhalo_id: str,
    orientation: int,
    band: str,
) -> Path:
    if isinstance(orientation, bool) or not isinstance(orientation, int):
        raise TypeError("orientation must be an integer in 1..5")
    if orientation not in ORIENTATIONS:
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
