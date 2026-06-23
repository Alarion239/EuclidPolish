"""Per-source sidecar catalog for synthetic fields.

``MultiBandSimulator.simulate_field`` knows every galaxy/lens it places, but the
TFRecord schema stores only pixels. This module persists that source list as a
CSV next to the records (``sources_<subset>.csv``) so the evaluation can crop
postage stamps centered on a known lens or galaxy. One row per galaxy and per
lens; stars are not recorded (we never center morphology on them).
"""

from __future__ import annotations

import csv
import math
import os
from typing import Any, Dict, List

SOURCE_COLS = ["field_index", "type", "render", "x_pix", "y_pix",
               "flux_vis_e", "z", "subhalo_id", "theta_E_arcsec"]


def _flux_vis(src: Dict[str, Any]):
    f = src.get("flux_e_per_band")
    return float(f[0]) if f else ""


def _z(src: Dict[str, Any]):
    z = src.get("z_phot", src.get("z_lens", src.get("z")))
    if z is None:
        return ""
    z = float(z)
    return "" if math.isnan(z) else z


def _galaxy_row(field_index: int, g: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "field_index": field_index, "type": "galaxy",
        "render": g.get("render", ""),
        "x_pix": float(g["x_pix"]), "y_pix": float(g["y_pix"]),
        "flux_vis_e": _flux_vis(g), "z": _z(g),
        "subhalo_id": g.get("subhalo_id", ""), "theta_E_arcsec": "",
    }


def _lens_row(field_index: int, lens: Dict[str, Any]) -> Dict[str, Any]:
    theta = lens.get("theta_E_arcsec")
    return {
        "field_index": field_index, "type": "lens", "render": "",
        "x_pix": float(lens["x_pix"]), "y_pix": float(lens["y_pix"]),
        "flux_vis_e": _flux_vis(lens), "z": _z(lens),
        "subhalo_id": lens.get("lens_subhalo_id", ""),
        "theta_E_arcsec": float(theta) if theta is not None else "",
    }


class SourceCatalogWriter:
    """Append galaxy/lens rows to ``path`` as fields are generated."""

    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._f = open(path, "w", newline="")
        self._w = csv.DictWriter(self._f, fieldnames=SOURCE_COLS)
        self._w.writeheader()

    def add_field(self, field_index: int, meta: Dict[str, Any]) -> None:
        for g in meta.get("galaxies", []) or []:
            self._w.writerow(_galaxy_row(field_index, g))
        for lens in meta.get("lenses", []) or []:
            self._w.writerow(_lens_row(field_index, lens))

    def close(self) -> None:
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _parse(row: Dict[str, str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"type": row["type"], "render": row["render"],
                           "subhalo_id": row["subhalo_id"] or None}
    out["field_index"] = int(row["field_index"])
    for k in ("x_pix", "y_pix", "flux_vis_e", "z", "theta_E_arcsec"):
        v = row.get(k, "")
        out[k] = float(v) if v not in ("", None) else None
    return out


def read_sources(csv_path: str) -> Dict[int, List[Dict[str, Any]]]:
    """``field_index -> list[source dict]``; missing file -> ``{}``."""
    if not os.path.isfile(csv_path):
        return {}
    by_field: Dict[int, List[Dict[str, Any]]] = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            r = _parse(row)
            by_field.setdefault(r["field_index"], []).append(r)
    return by_field


def concat_source_csvs(part_paths: List[str], out_path: str) -> None:
    """Concatenate shard CSVs (in the given order) into one, single header.

    Atomic: build a sibling temp file then ``os.replace`` it into place, so a
    crash mid-merge never leaves a truncated ``sources_<subset>.csv`` that a
    resumed run would mistake for complete (see the resume design doc)."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "w", newline="") as out:
        out.write(",".join(SOURCE_COLS) + "\r\n")
        for p in part_paths:
            if not os.path.isfile(p):
                continue
            with open(p, newline="") as f:
                next(f, None)                     # skip shard header
                for line in f:
                    out.write(line)
    os.replace(tmp_path, out_path)
