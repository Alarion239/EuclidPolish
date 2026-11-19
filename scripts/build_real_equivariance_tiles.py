#!/usr/bin/env python
"""Build the real Euclid LR tiles for the real-sky equivariance penalty.

Draws random positions inside the Q1 MER tiles of the chosen fields (default
EDF-F and EDF-S: every real-sky battery test — NEXUS, the poster galaxy, the
grid / holes / flux / growth / JWST-resolution tests — lies in EDF-N, so a
member trained on these never sees a test field), caches each as a 25.6″
four-band tile (:func:`euclid_polish.web.helpers.real_tiles.cache_tile`: VIS
cropped to 256², Y/J/H registered onto it, electrons via MAGZERO; positions
without full four-band coverage are skipped) and packs them into one
``(N, 256, 256, 4)`` float32 ``.npy`` with a JSON manifest next to it.

    python scripts/build_real_equivariance_tiles.py --n 4000 --workers 16 \\
        --store-dir $DATA/real_equivariance/store \\
        --out $DATA/real_equivariance/edf_fs_4000.npy

Positions are drawn up front from ``--seed``, so a re-run asks for the same
places; tiles already in the store are reused (a killed run resumes). The
packed set keeps the first ``--n`` successes in draw order, so it does not
depend on the order downloads finish in.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from euclid_polish.config import Config  # noqa: E402
from euclid_polish.sky.observation import q1_mer_tiles  # noqa: E402
from euclid_polish.sky.observation.q1_fields import q1_field_for  # noqa: E402
from euclid_polish.web.helpers import jwst_euclid, real_tiles  # noqa: E402

#: Fields the battery's real-sky tests never use.
DEFAULT_FIELDS = ("EDF-F", "EDF-S")
#: Keep a drawn centre this far inside its MER tile (the 25.6″ tile's half
#: diagonal plus the registration padding), so the cutout never straddles an edge.
EDGE_MARGIN_DEG = 20.0 / 3600.0


def candidate_tiles(fields) -> list[q1_mer_tiles.Q1Tile]:
    """The usable Q1 MER tiles of ``fields`` (rejected tiles dropped)."""
    wanted = {str(f) for f in fields}
    return [t for t in q1_mer_tiles.load_tiles() if t.field in wanted and not t.rejected]


def draw_positions(tiles, n: int, seed: int, *, fields) -> list[tuple[float, float, str]]:
    """``n`` random ``(ra, dec, tile)`` centres: a uniform tile, then a uniform
    point in its bounding box kept when inside the polygon by
    :data:`EDGE_MARGIN_DEG` and inside one of ``fields``."""
    if not tiles:
        raise ValueError("no usable Q1 tiles in the requested fields")
    rng = np.random.default_rng(seed)
    wanted = {str(f) for f in fields}
    out: list[tuple[float, float, str]] = []
    while len(out) < n:
        tile = tiles[int(rng.integers(len(tiles)))]
        ras = [p[0] for p in tile.polygon]
        decs = [p[1] for p in tile.polygon]
        ra = float(rng.uniform(min(ras), max(ras)))
        dec = float(rng.uniform(min(decs), max(decs)))
        if not q1_mer_tiles.point_in_polygon(ra, dec, tile.polygon):
            continue
        if q1_mer_tiles.polygon_margin_deg(ra, dec, tile.polygon) < EDGE_MARGIN_DEG:
            continue
        if q1_field_for(ra, dec) not in wanted:
            continue
        out.append((ra, dec, tile.tile))
    return out


def fetch(position) -> tuple[str | None, str | None]:
    """Cache one tile → ``(tile id, None)`` or ``(None, reason)``."""
    ra, dec, _tile = position
    identifier = real_tiles.real_tile_id(ra, dec)
    if (real_tiles.tiles_root() / identifier / "lr_e.npy").is_file():
        return identifier, None
    try:
        real_tiles.cache_tile(ra, dec, group="real-equivariance")
    except (RuntimeError, ValueError, OSError) as exc:
        return None, f"{type(exc).__name__}: {str(exc)[:120]}"
    return identifier, None


def pack(identifiers, out_path: str) -> np.ndarray:
    """Stack the tiles' ``lr_e.npy`` into one float32 ``.npy`` at ``out_path``."""
    side = real_tiles.TILE_SIDE
    out = np.lib.format.open_memmap(out_path + ".partial.npy", mode="w+", dtype=np.float32,
                                    shape=(len(identifiers), side, side, 4))
    for i, identifier in enumerate(identifiers):
        cube = np.load(real_tiles.tiles_root() / identifier / "lr_e.npy")
        if cube.shape != (side, side, 4) or not np.all(np.isfinite(cube)):
            raise ValueError(f"tile {identifier} is {cube.shape} or not finite")
        out[i] = cube
    out.flush()
    del out
    os.replace(out_path + ".partial.npy", out_path)
    return np.load(out_path, mmap_mode="r")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=4000, help="Tiles to pack (default 4000).")
    ap.add_argument("--fields", default=",".join(DEFAULT_FIELDS),
                    help="Q1 fields to draw from (default EDF-F,EDF-S).")
    ap.add_argument("--seed", type=int, default=20261119)
    ap.add_argument("--workers", type=int, default=8, help="Parallel downloads.")
    ap.add_argument("--oversample", type=float, default=1.5,
                    help="Draw this many × --n positions (some lack coverage).")
    ap.add_argument("--irsa-only", action="store_true",
                    help="Skip the ESA archive (each failed ESA attempt costs ~10 s).")
    ap.add_argument("--store-dir", required=True,
                    help="Where the per-tile caches go (its real_tiles/ subfolder).")
    ap.add_argument("--out", required=True, help="The packed .npy.")
    args = ap.parse_args(argv)

    fields = tuple(f.strip() for f in args.fields.split(",") if f.strip())
    if "EDF-N" in fields:
        print("✗ EDF-N holds every real-sky battery test; it may not be a training field")
        return 2
    Config.EUCLID_INFERENCE_DIR = os.path.abspath(args.store_dir)
    if args.irsa_only:
        jwst_euclid.Q1_CUTOUT_SOURCES = ("irsa",)
    real_tiles.tiles_root().mkdir(parents=True, exist_ok=True)
    tiles = candidate_tiles(fields)
    positions = draw_positions(tiles, int(np.ceil(args.n * args.oversample)), args.seed,
                               fields=fields)
    print(f"{len(tiles)} usable MER tiles in {','.join(fields)}; "
          f"{len(positions)} positions drawn (seed {args.seed}); store {real_tiles.tiles_root()}",
          flush=True)

    results: list[tuple[str | None, str | None]] = [(None, "not tried")] * len(positions)
    done = 0
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = {pool.submit(fetch, p): i for i, p in enumerate(positions)}
        for future, i in futures.items():
            results[i] = future.result()
            done += 1
            if done % 100 == 0:
                ok = sum(1 for r in results if r[0])
                print(f"  {done}/{len(positions)} tried, {ok} tiles", flush=True)

    kept = [(i, r[0]) for i, r in enumerate(results) if r[0]][: args.n]
    failures = Counter(r[1].split(":")[0] for r in results if not r[0])
    if len(kept) < args.n:
        print(f"✗ only {len(kept)} of {args.n} tiles ({dict(failures)}); raise --oversample")
        return 3
    cube = pack([identifier for _i, identifier in kept], args.out)
    by_field = Counter(q1_field_for(positions[i][0], positions[i][1]) for i, _ in kept)
    manifest = {
        "kind": "real_equivariance_tiles", "created": datetime.now(UTC).isoformat(),
        "shape": list(cube.shape), "dtype": "float32", "unit": "e-",
        "bands": list(real_tiles.BAND_NAMES), "fields": list(fields), "seed": args.seed,
        "by_field": dict(by_field), "tried": len(positions), "failures": dict(failures),
        "tiles": [{"id": identifier, "ra": positions[i][0], "dec": positions[i][1],
                   "mer_tile": positions[i][2]} for i, identifier in kept],
        "median_e_per_band": [float(np.median(cube[: min(200, len(cube)), ..., b]))
                              for b in range(cube.shape[-1])],
    }
    with open(os.path.splitext(args.out)[0] + ".json", "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"✓ {len(kept)} tiles → {args.out} ({cube.nbytes / 1e9:.2f} GB); by field "
          f"{dict(by_field)}; skipped {dict(failures)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
