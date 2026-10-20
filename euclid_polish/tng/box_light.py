"""Which TNG galaxies carry SKIRT's dust box: the table the atlas scan reads.

SKIRT re-gridded the TNG50 dust onto an octree of cubes (Baes et al. 2024,
§2.4.2). Far from a galaxy the cubes are large and evenly dusty, so their
scattered light lands as flat facets whose straight edges run along the
simulation axes, with radial shadow rays and a box at the edge of the grid
(2026-10-19: 36 % of the VIS light of subhalo 117266). The star mask removes
it off the stars, but in such a galaxy it also lies on the faint outskirt
stars as a flat plateau. It is bimodal: 338 of 342 galaxies have under 0.5 %
of their VIS light where no star is, 4 have 3-36 %. A galaxy above
``Config.Tng.MAX_OUTSIDE_STAR_FRACTION`` in any orientation is no donor
(``euclid_polish.tng.atlas._box_galaxies``).

The table ``<atlas>/<Config.Tng.BOX_LIGHT_TABLE>`` holds each galaxy's share
per orientation: VIS light where its dust-free twin is dark over all VIS
light. :func:`refresh` measures only galaxies that are new or whose frames
changed.

    python -m euclid_polish.tng.box_light [--root DIR] [--workers N]
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

from euclid_polish.config import Config
from euclid_polish.tng.atlas import TNGGalaxy, _scan_complete_galaxies
from euclid_polish.tng.types import dust_free_twin

TABLE_VERSION = 1
_ORIENTATIONS = range(1, 6)


def _frames(galaxy: TNGGalaxy) -> list[tuple[Path, Path]]:
    return [(galaxy.fits_path(o, "VIS"), dust_free_twin(galaxy.fits_path(o, "VIS"))) for o in _ORIENTATIONS]


def _identity(galaxy: TNGGalaxy) -> list[list[Any]]:
    out = []
    for pair in _frames(galaxy):
        for path in pair:
            status = path.stat()
            out.append([path.name, int(status.st_size), int(status.st_mtime_ns)])
    return out


def outside_star_fractions(galaxy: TNGGalaxy) -> list[float]:
    """Per orientation, the share of the dusty VIS light where the dust-free
    VIS twin is dark (its support is that of every band's twin: each star
    particle shines in all of them)."""
    shares = []
    for dusty_path, twin_path in _frames(galaxy):
        dusty = np.nan_to_num(np.asarray(fits.getdata(dusty_path), dtype=np.float64))
        stars = np.nan_to_num(np.asarray(fits.getdata(twin_path), dtype=np.float64)) > 0.0
        total = float(dusty.sum())
        shares.append(float(dusty[~stars].sum() / total) if total > 0.0 else 0.0)
    return shares


def _read(path: Path) -> dict[str, Any]:
    try:
        table = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return table.get("galaxies", {}) if table.get("version") == TABLE_VERSION else {}


def refresh(root: str | Path, *, workers: int = 8) -> dict[str, Any]:
    """Measure the galaxies that are new or changed since the table was
    written, keep the rest, and rewrite the table. Galaxies without dust-free
    twins are not measured (they are not star-masked either)."""
    root = Path(root)
    path = root / Config.Tng.BOX_LIGHT_TABLE
    previous = _read(path)

    def twinned_identity(galaxy: TNGGalaxy) -> list[list[Any]] | None:
        if not all(twin.is_file() for _, twin in _frames(galaxy)):
            return None
        return _identity(galaxy)

    # ~20 stat calls per galaxy: concurrent, as each is a network round trip.
    scanned = _scan_complete_galaxies(root, include_box_galaxies=True)
    with ThreadPoolExecutor(max_workers=16) as pool:
        identities = list(pool.map(twinned_identity, scanned))
    entries: dict[str, Any] = {}
    todo = []
    for galaxy, identity in zip(scanned, identities, strict=True):
        if identity is None:
            continue
        old = previous.get(galaxy.subhalo_id)
        if old is not None and old.get("files") == identity:
            entries[galaxy.subhalo_id] = old
        else:
            todo.append((galaxy, identity))
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        shares = list(pool.map(lambda job: outside_star_fractions(job[0]), todo))
    for (galaxy, identity), fractions in zip(todo, shares, strict=True):
        entries[galaxy.subhalo_id] = {"fractions": [round(f, 6) for f in fractions], "files": identity}
    limit = Config.Tng.MAX_OUTSIDE_STAR_FRACTION
    boxed = sorted((gid for gid, e in entries.items() if max(e["fractions"]) > limit), key=lambda g: (len(g), g))
    table = {"version": TABLE_VERSION,
             "rule": "share of the dusty VIS light where the dust-free VIS twin is dark, per orientation",
             "limit": limit, "galaxies": dict(sorted(entries.items()))}
    # Written only when there is something to record and it changed (an atlas
    # without dust-free twins, e.g. a local one, gets no table).
    if (entries or path.exists()) and (todo or entries.keys() != previous.keys()):
        temporary = path.with_name(f".{path.name}.part")
        temporary.write_text(json.dumps(table, indent=1) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    return {"galaxies": len(entries), "measured": len(todo), "reused": len(entries) - len(todo),
            "left_out": boxed}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure which TNG galaxies carry SKIRT's dust box (light where no star is).")
    parser.add_argument("--root", default=Config.TNG_SKIRT_DIR)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    result = refresh(args.root, workers=args.workers)
    print(f"TNG box light: {result['galaxies']} galaxies ({result['measured']} measured, {result['reused']} reused); "
          f"{len(result['left_out'])} left out above {100 * Config.Tng.MAX_OUTSIDE_STAR_FRACTION:g} %: "
          f"{', '.join(result['left_out']) or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["outside_star_fractions", "refresh"]
