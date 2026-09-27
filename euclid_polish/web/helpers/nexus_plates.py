"""NEXUS × Euclid comparison plates: Euclid LR | SR of one model | NEXUS.

The renderer behind ``scripts/render_nexus_comparisons.py`` and the Figures
› Plates job (``POST /api/figures/nexus-plates``). Every panel is read
through the C9 ``real`` viewer collection (``source=nexus``), so a plate
shows exactly the arrays the viewer shows: the registered four-band Euclid
LR (``lr``), the SR of the chosen model spec (``m:<spec>``: production,
mean, rbf, member:…, gate:…; the RBF-era cached SRs are ``m:rbf``) and the
native NEXUS image (``jwst``). All panels cover the same 25.5″ tile.

``band`` = one Euclid band (every panel gets its own asinh display stretch)
or ``temp`` (Euclid LR and SR in the viewer's "Temp" colour —
:func:`eye_rgb` at the viewer's default knee — while NEXUS stays native
grey). NEXUS is an external morphological reference in a different band and
unit, not a photometric truth for the SR.

A run is one output directory ``<root>/<tag>/`` holding per-tile PNGs
``nexus_tile<NNN>_<band>__<model slug>.png``, one contact sheet
``nexus_tiles_<band>__<model slug>.png`` per (band, model) and
``plates.json`` (every render's provenance: field, model identity and state,
tile positions). Runs written by the pre-W-Figures script (names without the
model slug, one ``provenance.json``) are listed too.
"""
from __future__ import annotations

import contextlib
import json
import math
import os
import re
import shutil
import threading
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image as PILImage

from euclid_polish.config import Config
from euclid_polish.visualization.color import eye_rgb
from euclid_polish.web.helpers import model_catalog, real_tiles, viewer_data
from euclid_polish.web.helpers.paths import REPO_ROOT

SOURCE = "nexus"
BANDS = ("VIS", "Y_E", "J_E", "H_E", "temp")
DEFAULT_TILES = (40, 42, 70, 178)
DEFAULT_BAND = "VIS"
MAX_TILES = 24
MANIFEST = "plates.json"
LEGACY_PROVENANCE = "provenance.json"
THUMB_MAX_SIDE = 1600
#: contact sheet geometry: one 3-panel row is SHEET_ROW_IN tall; the dpi drops
#: for long sheets so a 24-tile run stays within SHEET_MAX_PIXELS (~40 MB of
#: Agg RGBA) on the shared machine instead of a ~36 Mpx canvas.
SHEET_WIDTH_IN = 10.5
SHEET_ROW_IN = 3.6
SHEET_DPI = 200
SHEET_MIN_DPI = 72
SHEET_MAX_PIXELS = 10_000_000
_THUMB_CACHE_MAX = 64

_TAG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_BAND_ALT = "|".join(re.escape(band) for band in BANDS)
_TILE_FILE = re.compile(rf"^nexus_tile(\d{{1,5}})_({_BAND_ALT})(?:__([a-z0-9._+-]{{1,80}}))?\.png$")
_SHEET_FILE = re.compile(rf"^nexus_tiles_({_BAND_ALT})(?:__([a-z0-9._+-]{{1,80}}))?\.png$")

#: tier, title, pixel scale, spine colour (the figures are black plates).
PANELS = (
    ("lr", "Euclid {band} (LR)", "0.10″/pix", "#1F6FB2"),
    ("sr", "Super-resolved {band} · {model}", "0.05″/pix", "#2E8B57"),
    ("jwst", "NEXUS {filter}", "0.03″/pix", "#D9760A"),
)

_THUMBS: OrderedDict[tuple[str, int, int, int], bytes] = OrderedDict()
_THUMB_LOCK = threading.Lock()


class PlateError(Exception):
    """A client-visible plate request error (``code`` = HTTP status)."""

    def __init__(self, code: int, message: str, **extra: Any):
        super().__init__(message)
        self.code = int(code)
        self.extra = extra


@dataclass(frozen=True)
class PlateTile:
    """One resolved NEXUS tile of a plate request."""

    index: int          # position in the ``real`` collection (viewer index)
    id: str             # real-tile id, e.g. ``f200w-0040``
    source_index: int   # the NEXUS tile number shown on the plate
    ra: float | None
    dec: float | None
    field_id: str | None
    model_state: str | None


def plates_root() -> Path:
    """``output/nexus_comparisons`` in the repo (``EUCLID_POLISH_NEXUS_PLATES_DIR``
    overrides it; resolved per call so tests can retarget it)."""
    explicit = os.environ.get("EUCLID_POLISH_NEXUS_PLATES_DIR")
    return Path(explicit).expanduser() if explicit else REPO_ROOT / "output" / "nexus_comparisons"


def check_tag(tag: str) -> str:
    text = str(tag or "").strip()
    if not _TAG.fullmatch(text):
        raise PlateError(400, "tag must be 1–64 letters, digits, '.', '_' or '-' (no leading dot)")
    return text


def check_band(band: str) -> str:
    text = str(band or "").strip()
    match = next((item for item in BANDS if item.lower() == text.lower()), None)
    if match is None:
        raise PlateError(400, f"band must be one of {', '.join(BANDS)}")
    return match


def model_short(spec: str) -> str:
    """Compact plate title of a spec: ``member:member_196`` → ``member 196``."""
    if spec == "production":
        return "production gate"
    if spec == "mean":
        return "member mean"
    if spec == "rbf":
        return "RBF"
    if spec.startswith("member:"):
        return "member " + spec.split(":", 1)[1].removeprefix("member_")
    if spec.startswith("gate:"):
        return "gate " + spec.split(":", 1)[1]
    return spec


def _params(spec: str) -> dict[str, str]:
    return {"source": SOURCE, "models": spec}


def resolve_request(tiles: str | Sequence[Any], model: str) -> tuple[str, list[PlateTile], dict[str, Any]]:
    """Validate a plate request: ``(canonical spec, tiles, meta)``.

    ``tiles`` = a comma list (or sequence) of NEXUS tile numbers (``40``),
    real-tile ids (``f200w-0040``) or refs (``nexus/f200w-0040``). Every tile
    must hold an output of the spec (never a silent partial plate): 400 with
    ``missing`` listing the tiles without one."""
    try:
        spec = model_catalog.canonical_spec(model or "production")
    except ValueError as exc:
        raise PlateError(400, str(exc)) from exc
    raw_items = tiles.split(",") if isinstance(tiles, str) else list(tiles)
    items = [str(item).strip() for item in raw_items if str(item).strip()]
    if not items:
        raise PlateError(400, "pick at least one NEXUS tile")
    if len(items) > MAX_TILES:
        raise PlateError(400, f"a plate run renders at most {MAX_TILES} tiles")
    try:
        meta = viewer_data.get_meta("real", _params(spec))
    except viewer_data.ViewerError as exc:
        raise PlateError(exc.code, str(exc)) from exc
    objects = meta.get("objects") or []
    by_id: dict[str, int] = {}
    by_number: dict[int, int] = {}
    for position, obj in enumerate(objects):
        identifier = str(obj.get("id") or "")
        by_id[identifier] = position
        match = re.search(r"(\d+)$", identifier)
        if match:
            by_number.setdefault(int(match.group(1)), position)
    resolved: list[PlateTile] = []
    missing: list[str] = []
    for item in items:
        key = item.split("/", 1)[1] if item.startswith(f"{SOURCE}/") else item
        position = by_id.get(key)
        if position is None and key.isdigit():
            position = by_number.get(int(key))
        if position is None:
            raise PlateError(404, f"unknown NEXUS tile {item!r}")
        if any(tile.index == position for tile in resolved):
            continue
        obj = objects[position]
        identifier = str(obj.get("id"))
        number = re.search(r"(\d+)$", identifier)
        states = obj.get("model_states") or {}
        if spec not in states:
            missing.append(identifier)
        resolved.append(PlateTile(
            index=position, id=identifier,
            source_index=int(number.group(1)) if number else position,
            ra=_finite(obj.get("ra")), dec=_finite(obj.get("dec")),
            field_id=_field_id(identifier),
            model_state=states.get(spec),
        ))
    if missing:
        raise PlateError(
            400,
            f"{spec} has not been run on {', '.join(missing[:6])}"
            + (f" and {len(missing) - 6} more" if len(missing) > 6 else "")
            + " — run it in Sky › Experiments first",
            missing=missing,
        )
    return spec, resolved, meta


def _field_id(identifier: str) -> str | None:
    """The cached NEXUS field (mosaic) a tile belongs to, for provenance."""
    try:
        return str(real_tiles.get_entry(SOURCE, identifier).extras.get("field_id") or "") or None
    except real_tiles.RealTileError:
        return None


def _finite(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and np.isfinite(value) else None


def _asinh_display(data: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(np.asarray(data, dtype=np.float32), nan=0.0)
    values = np.clip(values, 0.0, None)
    positive = values[values > 0]
    scale = float(np.percentile(positive, 90.0)) if positive.size else 1.0
    stretched = np.arcsinh(values / max(scale, 1e-12))
    lo, hi = np.percentile(stretched, [0.5, 99.5])
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((stretched - lo) / (hi - lo), 0.0, 1.0)


def _plane(cube: np.ndarray, info: Mapping[str, Any], band: str) -> np.ndarray:
    bands = [str(name) for name in info.get("bands", [])]
    index = bands.index(band) if band in bands else 0
    return cube[..., index]


def panel_image(cube: np.ndarray, info: Mapping[str, Any], band: str) -> tuple[np.ndarray, dict[str, Any]]:
    """A panel image plus its imshow keywords for the requested band."""
    bands = tuple(str(name) for name in info.get("bands", []))
    if band == "temp" and bands == tuple(Config.LR_INPUT_BAND_NAMES):
        # The viewer's Temp default: knee = the tier's asinh, white = 30×.
        knee = float(info.get("asinh", Config.STRETCH_SCALE_E))
        return eye_rgb(np.nan_to_num(cube), bands, asinh_scale_e=knee), {}
    plane_band = "VIS" if band == "temp" else band
    return (_asinh_display(_plane(cube, info, plane_band)),
            {"cmap": "gray", "vmin": 0.0, "vmax": 1.0})


def _load_panels(tile: PlateTile, spec: str) -> list[tuple[np.ndarray, dict[str, Any]]]:
    params = _params(spec)
    out = []
    for tier in ("lr", f"m:{spec}", "jwst"):
        try:
            out.append(viewer_data.get_cube("real", tile.index, tier, params))
        except viewer_data.ViewerError as exc:
            raise PlateError(exc.code, f"{tile.id} {tier}: {exc}") from exc
    return out


def _draw_row(axes_row, images, tile: PlateTile, band: str, model: str,
              filter_name: str, *, titles: bool) -> None:
    """Draw one 3-panel row from display-ready ``(image, imshow style)``."""
    for ax, (image, style), (_tier, title, scale, color) in zip(axes_row, images, PANELS, strict=True):
        # Viewer orientation: row 0 at the top, as in the exported figures.
        ax.imshow(image, origin="upper", interpolation="nearest", **style)
        if titles:
            label = "temperature" if band == "temp" else band
            ax.set_title(f"{title.format(band=label, filter=filter_name, model=model)}\n{scale}",
                         color="white", fontsize=12, fontweight="bold", pad=8)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(color)
            spine.set_linewidth(2.5)
    axes_row[0].set_ylabel(f"tile {tile.source_index}", color="white", fontsize=12,
                           fontweight="bold")


def _save(figure: Figure, path: Path) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    figure.savefig(tmp, facecolor="black", format="png")
    os.replace(tmp, path)


def sheet_dpi(rows: int) -> int:
    """The contact sheet's dpi: SHEET_DPI unless the canvas would exceed
    SHEET_MAX_PIXELS (never below SHEET_MIN_DPI)."""
    area_in2 = SHEET_WIDTH_IN * SHEET_ROW_IN * max(1, int(rows))
    return int(max(SHEET_MIN_DPI, min(SHEET_DPI, math.floor(math.sqrt(SHEET_MAX_PIXELS / area_in2)))))


def sheet_panel(image: np.ndarray, style: Mapping[str, Any],
                max_side: int) -> tuple[np.ndarray, dict[str, Any]]:
    """A display-ready ``[0, 1]`` panel reduced for the contact sheet: uint8,
    at most ``max_side`` px per side (the panel's slot on the sheet), so a
    run keeps ~1 MB per tile for the sheet instead of the raw cubes."""
    unit = np.clip(np.nan_to_num(np.asarray(image, dtype=np.float32)), 0.0, 1.0)
    data = np.rint(unit * 255.0).astype(np.uint8)
    height, width = data.shape[:2]
    side = max(1, int(max_side))
    if max(height, width) > side:
        scale = side / max(height, width)
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        data = np.asarray(PILImage.fromarray(data).resize(size, PILImage.Resampling.LANCZOS))
    out_style = dict(style)
    if data.ndim == 2 and "vmax" in out_style:
        out_style.update(vmin=0, vmax=255)
    return data, out_style


def tile_file(source_index: int, band: str, spec: str) -> str:
    return f"nexus_tile{source_index:03d}_{band}__{model_catalog.spec_slug(spec)}.png"


def sheet_file(band: str, spec: str) -> str:
    return f"nexus_tiles_{band}__{model_catalog.spec_slug(spec)}.png"


def default_tag(spec: str, now: datetime | None = None) -> str:
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%d")
    return f"{model_catalog.spec_slug(spec)}-{stamp}"


def render_plates(tiles: str | Sequence[Any], *, band: str = DEFAULT_BAND, model: str = "production",
                  tag: str | None = None,
                  progress: Callable[[int, int, str], None] | None = None,
                  out_root: Path | None = None) -> dict[str, Any]:
    """Render one tile PNG per tile plus a contact sheet into ``<root>/<tag>/``
    and record the render in ``plates.json``; returns that render record.
    ``out_root`` overrides :func:`plates_root` (the CLI's ``--out-dir``)."""
    band = check_band(band)
    spec, resolved, _meta = resolve_request(tiles, model)
    tag = check_tag(tag) if tag else default_tag(spec)
    out_dir = (out_root or plates_root()) / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    catalog = {item.spec: item for item in model_catalog.list_specs()}
    spec_info = catalog.get(spec)
    short = model_short(spec)
    total = len(resolved) + 1
    # the sheet keeps only reduced uint8 panels (never the raw cubes)
    rows: list[tuple[PlateTile, list[tuple[np.ndarray, dict[str, Any]]]]] = []
    dpi = sheet_dpi(len(resolved))
    slot_px = math.ceil(SHEET_WIDTH_IN / 3 * dpi)
    filter_name = "F200W"
    tile_records = []
    for step, tile in enumerate(resolved):
        if progress:
            progress(step, total, f"tile {tile.source_index}")
        panels = _load_panels(tile, spec)
        jwst_bands = panels[2][1].get("bands") or []
        filter_name = str(jwst_bands[0]) if jwst_bands else filter_name
        sr_info = dict(panels[1][1])
        images = [panel_image(cube, info, band) for cube, info in panels]
        del panels
        figure = Figure(figsize=(13.5, 4.9), dpi=200, facecolor="black")
        FigureCanvasAgg(figure)
        axes = figure.subplots(1, 3)
        _draw_row(axes, images, tile, band, short, filter_name, titles=True)
        figure.subplots_adjust(left=0.03, right=0.99, bottom=0.02, top=0.88, wspace=0.03)
        name = tile_file(tile.source_index, band, spec)
        _save(figure, out_dir / name)
        figure.clear()
        rows.append((tile, [sheet_panel(image, style, slot_px) for image, style in images]))
        del images
        tile_records.append({
            "index": tile.source_index, "id": tile.id, "ref": f"{SOURCE}/{tile.id}",
            "ra_deg": tile.ra, "dec_deg": tile.dec, "file": name,
            "model_state": sr_info.get("model_state", tile.model_state),
            "legacy": bool(sr_info.get("legacy")), "sr_label": sr_info.get("label"),
        })
    if progress:
        progress(len(resolved), total, "contact sheet")
    figure = Figure(figsize=(SHEET_WIDTH_IN, SHEET_ROW_IN * len(rows)), dpi=dpi, facecolor="black")
    FigureCanvasAgg(figure)
    axes = np.atleast_2d(figure.subplots(len(rows), 3, squeeze=False))
    for row, (tile, panels) in enumerate(rows):
        _draw_row(axes[row], panels, tile, band, short, filter_name, titles=row == 0)
    figure.subplots_adjust(left=0.05, right=0.99, bottom=0.01, top=0.95, wspace=0.03, hspace=0.04)
    sheet = sheet_file(band, spec)
    _save(figure, out_dir / sheet)
    figure.clear()
    rows.clear()

    record = {
        "band": band, "model": spec, "model_label": spec_info.label if spec_info else spec,
        "model_short": short, "model_fingerprint": spec_info.fingerprint if spec_info else None,
        "model_available": bool(spec_info and spec_info.available),
        "field_id": next((tile.field_id for tile in resolved if tile.field_id), None),
        "filter": filter_name, "created": datetime.now(UTC).isoformat(),
        "sheet": sheet, "tiles": tile_records,
        "source": SOURCE, "collection": "real",
    }
    _merge_manifest(out_dir, record)
    if progress:
        progress(total, total, "done")
    return {"tag": tag, **record}


def _merge_manifest(out_dir: Path, record: Mapping[str, Any]) -> None:
    path = out_dir / MANIFEST
    try:
        current = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        current = {}
    renders = [item for item in (current.get("renders") or []) if isinstance(item, Mapping)
               and not (item.get("band") == record["band"] and item.get("model") == record["model"])]
    renders.append(dict(record))
    tmp = path.with_name(f".{MANIFEST}.tmp")
    tmp.write_text(json.dumps({"version": 1, "renders": renders}, indent=2), encoding="utf-8")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# listing, files, thumbnails, delete
# ---------------------------------------------------------------------------

def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, UTC).isoformat()


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _real_tile_ids(field_id: Any) -> dict[int, str]:
    """NEXUS tile number → real-tile id, for legacy runs that recorded only
    the number. A tile of the run's own field wins over another field's."""
    try:
        entries = real_tiles.list_entries(SOURCE)
    except (real_tiles.RealTileError, OSError, ValueError):
        return {}
    out: dict[int, str] = {}
    exact: set[int] = set()
    for entry in entries:
        number = entry.extras.get("source_index")
        if not isinstance(number, int) or number in exact:
            continue
        if field_id and entry.extras.get("field_id") == field_id:
            out[number] = entry.id
            exact.add(number)
        else:
            out.setdefault(number, entry.id)
    return out


def _legacy_renders(directory: Path, pngs: list[str]) -> list[dict[str, Any]]:
    """Renders of a pre-W-Figures run: grouped by band from the file names;
    ``provenance.json`` (the last band it wrote) adds positions and identity."""
    provenance = _read_json(directory / LEGACY_PROVENANCE) or {}
    positions = {int(item.get("index")): item for item in provenance.get("tiles") or []
                 if isinstance(item, Mapping) and isinstance(item.get("index"), int)}
    tile_ids = _real_tile_ids(provenance.get("field_id"))
    by_band: dict[str, dict[str, Any]] = {}
    for name in pngs:
        tile = _TILE_FILE.fullmatch(name)
        sheet = _SHEET_FILE.fullmatch(name)
        if tile and not tile.group(3):
            band = tile.group(2)
            entry = by_band.setdefault(band, {"band": band, "tiles": [], "sheet": None})
            index = int(tile.group(1))
            info = positions.get(index, {})
            inference = info.get("inference") if isinstance(info.get("inference"), Mapping) else {}
            real_id = tile_ids.get(index)
            entry["tiles"].append({
                "index": index, "id": real_id, "file": name,
                "ref": f"{SOURCE}/{real_id}" if real_id else None,
                "ra_deg": info.get("ra_deg"), "dec_deg": info.get("dec_deg"),
                "model_state": None, "legacy": True,
                "sr_label": inference.get("combiner_label"),
            })
        elif sheet and not sheet.group(2):
            entry = by_band.setdefault(sheet.group(1), {"band": sheet.group(1), "tiles": [], "sheet": None})
            entry["sheet"] = name
    first_inference = next((item.get("inference") for item in positions.values()
                            if isinstance(item.get("inference"), Mapping)), None) or {}
    out = []
    for band in BANDS:
        entry = by_band.get(band)
        if entry is None:
            continue
        entry["tiles"].sort(key=lambda item: item["index"])
        out.append({
            **entry, "model": None, "legacy": True,
            "model_label": first_inference.get("combiner_label") or "legacy SR (RBF era)",
            "model_short": "legacy SR", "field_id": provenance.get("field_id"),
            "created": None, "filter": None,
        })
    return out


def list_runs() -> dict[str, Any]:
    """Every run under the plates root, newest first: ``{root, runs:[{tag,
    updated, renders:[…], files:[{name, size, kind, band, tile_index}]}]}``."""
    root = plates_root()
    runs = []
    if root.is_dir():
        for directory in sorted(root.iterdir(), key=lambda path: path.name):
            if not directory.is_dir() or directory.is_symlink() or not _TAG.fullmatch(directory.name):
                continue
            files = []
            newest = 0.0
            for path in sorted(directory.iterdir(), key=lambda item: item.name):
                if not path.is_file() or path.name.startswith("."):
                    continue
                with contextlib.suppress(OSError):
                    stat = path.stat()
                    newest = max(newest, stat.st_mtime)
                    tile = _TILE_FILE.fullmatch(path.name)
                    sheet = _SHEET_FILE.fullmatch(path.name)
                    if tile or sheet:
                        files.append({
                            "name": path.name, "size": stat.st_size,
                            "kind": "tile" if tile else "sheet",
                            "band": tile.group(2) if tile else sheet.group(1),
                            "tile_index": int(tile.group(1)) if tile else None,
                            "model_slug": (tile.group(3) if tile else sheet.group(2)) or None,
                        })
            if not files:
                continue
            manifest = _read_json(directory / MANIFEST) or {}
            renders = [dict(item) for item in manifest.get("renders") or [] if isinstance(item, Mapping)]
            legacy_names = [item["name"] for item in files if not item["model_slug"]]
            renders.extend(_legacy_renders(directory, legacy_names) if legacy_names else [])
            present = {item["name"] for item in files}
            for render in renders:
                render["tiles"] = [tile for tile in render.get("tiles") or [] if tile.get("file") in present]
                if render.get("sheet") not in present:
                    render["sheet"] = None
            renders = [render for render in renders if render["tiles"] or render.get("sheet")]
            runs.append({"tag": directory.name, "updated": _iso(newest) if newest else None,
                         "renders": renders, "files": files})
    runs.sort(key=lambda run: str(run.get("updated") or ""), reverse=True)
    return {"root": _display_root(root), "runs": runs, "bands": list(BANDS),
            "defaults": {"tiles": list(DEFAULT_TILES), "band": DEFAULT_BAND, "max_tiles": MAX_TILES}}


def _display_root(root: Path) -> str:
    try:
        return os.path.relpath(root, REPO_ROOT) if root.is_relative_to(REPO_ROOT) else os.fspath(root)
    except ValueError:
        return os.fspath(root)


def run_file(tag: str, name: str) -> Path:
    """A plate PNG of one run, jailed to the run directory."""
    directory = plates_root() / check_tag(tag)
    if not (_TILE_FILE.fullmatch(name or "") or _SHEET_FILE.fullmatch(name or "")):
        raise PlateError(404, "not a plate file")
    path = directory / name
    try:
        resolved = path.resolve(strict=True)
        root = plates_root().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise PlateError(404, "plate not found") from exc
    if directory.is_symlink() or resolved.parent != root / directory.name or not resolved.is_file():
        raise PlateError(404, "plate not found")
    return resolved


def thumbnail(path: Path, max_side: int = 640) -> bytes:
    """A JPEG preview (longest side ≤ ``max_side``), memoised per file state."""
    side = max(64, min(int(max_side), THUMB_MAX_SIDE))
    stat = path.stat()
    key = (os.fspath(path), int(stat.st_mtime_ns), int(stat.st_size), side)
    with _THUMB_LOCK:
        cached = _THUMBS.get(key)
        if cached is not None:
            _THUMBS.move_to_end(key)
            return cached
    with PILImage.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail((side, side), PILImage.Resampling.LANCZOS)
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=85, optimize=True)
    body = buffer.getvalue()
    with _THUMB_LOCK:
        _THUMBS[key] = body
        while len(_THUMBS) > _THUMB_CACHE_MAX:
            _THUMBS.popitem(last=False)
    return body


def delete_run(tag: str) -> str:
    """Remove one run directory (its PNGs and manifests)."""
    directory = plates_root() / check_tag(tag)
    if directory.is_symlink() or not directory.is_dir():
        raise PlateError(404, "plate run not found")
    shutil.rmtree(directory)
    return directory.name
