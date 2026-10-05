"""Attached study fields: pack one product at a time, upload it to holylabs,
verify the remote sha256, delete the local temp; fetch into the local cache.

A field's products (``<name>.npz``, key ``data``, float32 — lossless
``numpy.savez_compressed``; the hole mask is uint8):

* ``member_<N>`` — every member's SR, in the study's member order;
* ``mean`` — the plain mean of the members;
* ``gate`` — the production gate (the baked test cube, else the gate applied
  to the member stack; a real tile's current stored production SR, else the
  gate applied to its cached member SRs);
* ``lr`` — the LR input (a blackout field's is the stamped LR);
* ``hr`` — the raw target record (synthetic fields; the metrics' blurred
  target is ``blur_target_array(hr, target.fwhm_arcsec)``);
* ``mask`` — a blackout field's hole mask (:func:`spatial_gate_compare.hole_masks`);

plus ``truth.json`` (the record's source catalogue; empty for real tiles)
and ``field.json`` (identity, member labels, grids, pixel scales, WCS, and
the sha256 of every product) written last. The local disk never holds more
than the one product being uploaded (:func:`pack_field`).
"""

from __future__ import annotations

import contextlib
import json
import os
import shlex
import shutil
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from euclid_polish.config import Config
from euclid_polish.eval import spatial_gate_compare as sgc
from euclid_polish.eval.ensemble_cube_cache import (
    BLACKOUT_INDEX,
    VIZ_INDEX,
    bucket_member_paths,
    load_cached_field_lr,
    manifest_member_labels,
    read_bucket_manifest,
)
from euclid_polish.eval.spatial_gate import SPATIAL_GATE_KIND
from euclid_polish.eval.spatial_gate_fit import GateField
from euclid_polish.image.tfio import tfrecord_path
from euclid_polish.studies import candidates
from euclid_polish.studies.cache import (
    CORE_PRODUCTS,
    FieldCache,
    claim,
    member_product,
    product_file,
)
from euclid_polish.studies.store import sha256_file
from euclid_polish.web.helpers import ensemble_viz as ev
from euclid_polish.web.helpers import experiments, model_catalog, real_tiles, sky_records
from euclid_polish.web.helpers.viewer_data import celestial_wcs_keywords, scaled_wcs_keywords

FIELD_SCHEMA = 1
SR_FACTOR = int(Config.DEFAULT_REBIN_FACTOR)

ProductFn = Callable[[str, int, int], None]


class UploadError(RuntimeError):
    """A product did not arrive intact (sha256 mismatch or transfer error)."""


def product_names(kind: str, labels: Sequence[str]) -> list[str]:
    names = [member_product(label) for label in labels] + ["mean", "gate", "lr"]
    if kind in ("test", "blackout"):
        names.append("hr")
    if kind == "blackout":
        names.append("mask")
    return names


def write_npz(path: Path, array: np.ndarray) -> dict[str, Any]:
    """One product → ``path`` (compressed, lossless); ``{sha256, bytes, shape, dtype}``."""
    data = np.asarray(array)
    if data.dtype != np.uint8:
        data = data.astype(np.float32, copy=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        np.savez_compressed(handle, data=data)
    return {"sha256": sha256_file(path), "bytes": int(path.stat().st_size),
            "shape": [int(v) for v in data.shape], "dtype": str(data.dtype)}


class _DiskMembers:
    """:class:`model_catalog.MemberSource` over a tile's member-SR cache that
    never keeps more than the arrays a caller holds (no memo)."""

    def __init__(self, source: str, identifier: str) -> None:
        self.directory = experiments.member_cache_dir(source, identifier)

    def get(self, label: str) -> np.ndarray:
        return np.load(self.directory / f"{model_catalog.member_name(label)}.npy").astype(
            np.float32)


@dataclass
class FieldSource:
    """One field to pack: yields its products one at a time."""

    fid: str
    kind: str
    ref: str
    starless: bool
    labels: list[str]
    label: str = ""
    meta: dict[str, Any] = field(default_factory=dict)
    truth: dict[str, Any] = field(default_factory=dict)
    _thumb: np.ndarray | None = field(default=None, repr=False)

    def products(self) -> Iterator[tuple[str, np.ndarray]]:
        if self.kind == "test":
            yield from self._test_products()
        elif self.kind == "blackout":
            yield from self._blackout_products()
        else:
            yield from self._real_products()

    @property
    def thumbnail_cube(self) -> np.ndarray | None:
        return self._thumb

    # --------------------------------------------------------------- test
    def _regime(self) -> Path:
        return Path(ev._regime_dir_ro(self.starless))

    def _target(self, rec: int) -> np.ndarray:
        rdir = ev._sky_records_local_dir()
        subset = self.meta["target"]["subset"]
        name = "clean" if self.starless else "hr"
        image = sky_records.read_record(tfrecord_path(rdir, f"{name}_{subset}"), rec)
        if image.index is not None and int(image.index) != rec:
            raise RuntimeError(f"target record {rec} holds index {image.index}")
        self.meta["target"]["pixel_scale_arcsec"] = float(image.pixel_scale_arcsec)
        return np.asarray(image.data, np.float32)

    def _gate_on(self, stack_paths: Sequence[Path], lr: np.ndarray | None) -> np.ndarray | None:
        gate, positions = ev._combiner_for_stack(str(self._regime()), list(self.labels),
                                                 SPATIAL_GATE_KIND)
        if gate is None:
            self.meta["gate"] = {"available": False,
                                 "reason": "the production gate does not apply to these members"}
            return None
        stack = np.stack([np.load(stack_paths[p]) for p in positions])
        self.meta["gate"] = {"available": True, "baked": False, "reads": len(positions)}
        combiner: Any = gate
        return np.asarray(combiner.apply_field(stack, lr=lr), np.float32)

    def _member_paths(self, directory: Path, manifest_name: str, rec: int) -> list[Path]:
        """The study members' cubes of one field in a cube bucket (its
        manifest names the file of each member)."""
        manifest = read_bucket_manifest(str(directory), manifest_name)
        paths = bucket_member_paths(manifest, str(directory), self.labels, rec)
        absent = [label for label, path in zip(self.labels, paths, strict=True) if path is None]
        if absent:
            raise RuntimeError(f"{directory.name} holds no cubes of {', '.join(absent)}")
        return [Path(str(path)) for path in paths]

    def _test_products(self) -> Iterator[tuple[str, np.ndarray]]:
        rec = int(self.ref)
        tag = f"{rec:05d}"
        cubes = self._regime() / "cubes"
        paths = self._member_paths(cubes, VIZ_INDEX, rec)
        lr = load_cached_field_lr(str(cubes), rec, records_dir=ev._sky_records_local_dir(),
                                  subset=self.meta["target"]["subset"])
        if lr is None:                      # refuse before anything is uploaded
            raise RuntimeError(f"no LR input for test field {rec}")
        for label, path in zip(self.labels, paths, strict=True):
            yield member_product(label), np.load(path)
        yield "mean", np.load(cubes / f"sr_{tag}.npy")
        baked = cubes / f"{candidates.GATE_PREFIX}_{tag}.npy"
        if baked.is_file():
            self.meta["gate"] = {"available": True, "baked": True}
            gate = np.load(baked)
        else:
            gate = self._gate_on(paths, lr)
        if gate is not None:
            self._thumb = gate
            yield "gate", gate
        yield "lr", lr
        yield "hr", self._target(rec)

    def _blackout_products(self) -> Iterator[tuple[str, np.ndarray]]:
        rec = int(self.ref)
        tag = f"{rec:05d}"
        directory = self._regime() / "cubes_blackout"
        paths = self._member_paths(directory, BLACKOUT_INDEX, rec)
        total = None
        for label, path in zip(self.labels, paths, strict=True):
            member = np.load(path)
            total = member.astype(np.float64) if total is None else total + member
            yield member_product(label), member
        assert total is not None
        mean = (total / len(paths)).astype(np.float32)
        yield "mean", mean
        stamped = np.load(directory / f"lr_{tag}.npy")
        gate = self._gate_on(paths, stamped)
        self._thumb = gate if gate is not None else mean
        if gate is not None:
            yield "gate", gate
        yield "lr", stamped
        hr = self._target(rec)
        yield "hr", hr
        source_lr = load_cached_field_lr(str(self._regime() / "cubes"), rec,
                                         records_dir=ev._sky_records_local_dir(),
                                         subset=self.meta["target"]["subset"])
        if source_lr is None:
            raise RuntimeError(f"no unstamped LR for blackout field {rec}")
        holes = sgc.hole_masks(GateField(rec, [str(p) for p in paths], hr, stamped),
                               source_lr)
        yield "mask", holes.astype(np.uint8)

    # --------------------------------------------------------------- real
    def _real_products(self) -> Iterator[tuple[str, np.ndarray]]:
        source, _, identifier = self.ref.partition("/")
        entry = real_tiles.get_entry(source, identifier)
        tile = real_tiles.get_tile(source, identifier, entry=entry)
        lr = np.asarray(tile.lr_e, np.float32)
        lr_input = np.where(np.isfinite(lr), lr, 0.0).astype(np.float32)
        lr_sha = model_catalog.array_sha(lr_input)
        missing, fingerprints = candidates.missing_real_members(entry, self.labels)
        if missing:
            raise RuntimeError(f"{self.ref}: no current cached SR of " + ", ".join(missing))
        members = _DiskMembers(source, identifier)
        total = None
        for label in self.labels:
            member = members.get(label)
            total = member.astype(np.float64) if total is None else total + member
            yield member_product(label), member
            del member
        assert total is not None
        mean = (total / len(self.labels)).astype(np.float32)
        del total
        yield "mean", mean
        gate = self._real_gate(entry, lr_input, lr_sha, members)
        self._thumb = gate if gate is not None else mean
        if gate is not None:
            yield "gate", gate
        yield "lr", lr
        header = tile.wcs_header
        lr_wcs = celestial_wcs_keywords(header)
        self.meta["wcs"] = {"lr": lr_wcs, "sr": scaled_wcs_keywords(lr_wcs, SR_FACTOR)}
        self.meta["pixscale"] = {"lr": float(entry.pixscale),
                                 "sr": float(entry.pixscale) / SR_FACTOR}
        self.meta["lr_sha"] = lr_sha
        self.meta["member_fingerprints"] = fingerprints

    def _real_gate(self, entry, lr_input: np.ndarray, lr_sha: str,
                   members: _DiskMembers) -> np.ndarray | None:
        """The tile's current stored production SR, else the gate applied to
        the cached member SRs. Recomputing holds the stack of every member the
        gate reads in memory: ``reads × 2H × 2W × 4 × 4`` bytes — for the
        poster-size tile (1024² LR, 64 MB per member SR) about 2.4 GB at 37
        reads, plus the gate's own feature maps."""
        try:
            spec = model_catalog.resolve_spec(model_catalog.SPEC_PRODUCTION)
        except KeyError:
            spec = None
        if spec is None or not spec.available:
            self.meta["gate"] = {"available": False,
                                 "reason": getattr(spec, "reason", "no production gate")}
            return None
        meta = real_tiles.tile_outputs(entry, {spec.spec: spec.fingerprint}).get(spec.spec)
        if (meta is not None and meta.get("fingerprint") == spec.fingerprint
                and meta.get("lr_sha") == lr_sha):
            cube, _header, _meta = real_tiles.load_output(entry, spec.spec,
                                                          current=spec.fingerprint)
            self.meta["gate"] = {"available": True, "baked": True,
                                 "fingerprint": spec.fingerprint}
            return np.asarray(cube, np.float32)
        self.meta["gate"] = {"available": True, "baked": False,
                             "fingerprint": spec.fingerprint}
        return model_catalog.predict(spec, lr_input, members)


def open_field(fid: str, *, starless: bool, labels: Sequence[str]) -> FieldSource:
    """The packer of one candidate field for the member order ``labels``."""
    kind, ref = candidates.parse_field_id(fid)
    labels = list(labels)
    regime = Path(ev._regime_dir_ro(starless))
    meta: dict[str, Any] = {"schema": FIELD_SCHEMA, "fid": fid, "kind": kind, "ref": ref,
                            "regime": candidates.regime_slug(starless), "labels": labels}
    truth: dict[str, Any] = {"present": False, "sources": [],
                             "reason": "real tile: no truth catalogue"}
    label = ref
    if kind in ("test", "blackout"):
        manifest = json.loads((regime / "cubes" / "viz_index.json").read_text("utf-8"))
        subset = str(manifest.get("subset") or "test")
        if kind == "test" and [str(v) for v in manifest.get("member_labels") or []] != labels:
            raise RuntimeError("the test cubes are not the study's membership")
        if kind == "blackout":
            index = json.loads((regime / "cubes_blackout" / BLACKOUT_INDEX)
                               .read_text("utf-8"))
            if manifest_member_labels(index) != labels:
                raise RuntimeError("the blackout cubes are not the study's membership")
            meta["blackout"] = {**(index.get("identity") or {}), "member_labels": labels}
        meta["target"] = {"kind": "clean" if starless else "hr", "subset": subset,
                          "fwhm_arcsec": float(manifest.get("target_psf_fwhm_arcsec")
                                               or Config.TARGET_PSF_FWHM_ARCSEC),
                          "records_fp": manifest.get("records_fp")}
        meta["pixscale"] = {"lr": float(Config.VIS_PIXEL_SCALE_ARCSEC),
                            "sr": float(Config.DEFAULT_PIXEL_SCALE)}
        meta["wcs"] = {"lr": None, "sr": None}
        rdir = ev._sky_records_local_dir()
        if rdir:
            truth = sky_records.record_sources(rdir, subset, int(ref))
        label = f"{kind} · {subset} · idx {int(ref)}"
    else:
        meta["wcs"] = {"lr": None, "sr": None}
        label = f"real · {ref}"
    meta["label"] = label
    return FieldSource(fid=fid, kind=kind, ref=ref, starless=starless, labels=labels,
                       label=label, meta=meta, truth=truth)


# ---------------------------------------------------------------------------
# upload / verify
# ---------------------------------------------------------------------------

def remote_sha256(ssh: Any, remote_path: str) -> str | None:
    rc, out, _err = ssh.run(f"sha256sum {shlex.quote(remote_path)}", timeout=300)
    if rc != 0 or not str(out).strip():
        return None
    return str(out).split()[0]


def upload_verified(ssh: Any, local_file: Path, remote_dir: str, sha256: str) -> None:
    """Push the one file in ``local_file``'s directory to ``remote_dir`` and
    check the remote sha256 (:class:`UploadError` on any mismatch)."""
    try:
        rc, _out, err = ssh.rsync_push(str(local_file.parent), remote_dir, timeout=1800)
    except Exception as exc:  # noqa: BLE001 - SSHError, timeouts…
        raise UploadError(f"{local_file.name}: upload failed ({type(exc).__name__}: {exc})") from exc
    if rc != 0:
        raise UploadError(f"{local_file.name}: rsync exit {rc}: {str(err).strip()[:300]}")
    remote = remote_sha256(ssh, f"{remote_dir.rstrip('/')}/{local_file.name}")
    if remote != sha256:
        raise UploadError(f"{local_file.name}: remote sha256 {remote or 'missing'} "
                          f"≠ local {sha256}")


def pack_field(ssh: Any, source: FieldSource, remote_dir: str, *, staging: Path,
               on_product: ProductFn | None = None,
               check: Callable[[], None] | None = None) -> dict[str, Any]:
    """Pack, upload and verify every product of ``source`` into ``remote_dir``
    (one product on the local disk at a time). Returns the field record."""
    names = product_names(source.kind, source.labels)
    total = len(names) + 2
    work = Path(staging) / source.fid
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    products: dict[str, Any] = {}

    def ship(name: str, filename: str, write: Callable[[Path], dict[str, Any]],
             nbytes: int = 1024 ** 2) -> None:
        if check is not None:
            check()
        path = work / filename
        # The disk margin is checked (and the bytes reserved against the other
        # study jobs) before EVERY product is written.
        with claim(work, nbytes):
            try:
                info = write(path)
                upload_verified(ssh, path, remote_dir, info["sha256"])
            finally:
                with contextlib.suppress(FileNotFoundError):
                    path.unlink()
        products[name] = info
        if on_product is not None:
            on_product(name, len(products), total)

    try:
        for name, array in source.products():
            ship(name, f"{name}.npz", lambda path, a=array: write_npz(path, a),
                 int(np.asarray(array).nbytes))
            del array

        def write_json(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
            path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", "utf-8")
            return {"sha256": sha256_file(path), "bytes": int(path.stat().st_size)}

        ship("truth.json", "truth.json", lambda path: write_json(path, source.truth))
        meta = {**source.meta, "products": dict(products)}
        ship("field.json", "field.json", lambda path: write_json(path, meta))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return {"fid": source.fid, "kind": source.kind, "ref": source.ref, "label": source.label,
            "state": "uploaded", "remote_dir": remote_dir, "products": products,
            "bytes": sum(int(v["bytes"]) for v in products.values()),
            "gate": source.meta.get("gate")}


# ---------------------------------------------------------------------------
# fetch
# ---------------------------------------------------------------------------

#: One fetch at a time: the budget, the disk margin and the LRU accounting
#: hold for the whole admit → evict → pull → verify → record sequence.
_FETCH_LOCK = threading.Lock()


def core_products(record: Mapping[str, Any]) -> list[str]:
    """The products "Fetch field" brings (those this field has)."""
    products = record.get("products") or {}
    return [name for name in CORE_PRODUCTS if name in products]


def member_products(record: Mapping[str, Any]) -> list[str]:
    return [name for name in (record.get("products") or {}) if name.startswith("member_")]


def fetch_request(record: Mapping[str, Any], products: str | Sequence[str] | None
                  ) -> tuple[list[str], bool]:
    """``(product names, fit)`` of a fetch request: ``None``/``"core"`` → the
    core products; ``"members"`` → every member SR that fits (``fit``);
    else explicit product names (:class:`ValueError` for unknown ones)."""
    if products is None or products == "core":
        return core_products(record), False
    if products == "members":
        return member_products(record), True
    names = [str(p).strip() for p in ([products] if isinstance(products, str) else products)
             if str(p).strip()]
    unknown = [name for name in names if name not in (record.get("products") or {})]
    if unknown or not names:
        raise ValueError(f"{record.get('fid')} has no product(s) {', '.join(unknown) or '(none)'}")
    return names, False


def _pull_verified(ssh: Any, fid: str, name: str, info: Mapping[str, Any], remote_dir: str,
                   partial: Path, target: Path) -> None:
    """Pull one product file, check its sha256, move it into the cache and
    stamp it used NOW (rsync keeps the holylabs mtime, the LRU needs local use)."""
    shutil.rmtree(partial, ignore_errors=True)
    filename = product_file(name)
    rc, _out, err = ssh.rsync_pull(f"{remote_dir.rstrip('/')}/{filename}", str(partial),
                                   timeout=3600)
    pulled = partial / filename
    if rc != 0 or not pulled.is_file():
        raise UploadError(f"{fid}: fetching {name} failed "
                          f"(rsync exit {rc}: {str(err).strip()[:300]})")
    if sha256_file(pulled) != info.get("sha256"):
        raise UploadError(f"{fid}: {name} does not match the study's sha256")
    os.replace(pulled, target)
    os.utime(target)


def fetch_field(ssh: Any, study_id: str, record: dict[str, Any], remote_dir: str,
                cache: FieldCache, *, products: str | Sequence[str] | None = None,
                check: Callable[[], None] | None = None,
                progress: Callable[[int, int, str], None] | None = None) -> dict[str, Any]:
    """Copy products of an attached field from holylabs into ``cache``, one
    file at a time, each verified against the manifest's sha256 (see
    :func:`fetch_request`). Cached products are only marked recently used.
    Least recently used products of OTHER fields are evicted to fit the
    budget; a product that cannot fit (budget or the 5 GiB disk margin)
    raises :class:`experiments.DiskSpaceError` — or, for ``"members"``,
    ends the fetch with the rest listed as ``skipped``."""
    fid = str(record["fid"])
    manifest_products = record.get("products") or {}
    names, fit = fetch_request(record, products)
    fetched: list[str] = []
    skipped: list[str] = []
    with _FETCH_LOCK:
        directory = cache.field_dir(study_id, fid)
        directory.mkdir(parents=True, exist_ok=True)
        partial = directory.parent / f".{fid}.partial"
        try:
            for position, name in enumerate(names):
                if check is not None:
                    check()
                if progress is not None:
                    progress(position, len(names), f"{fid} · {name}")
                if name in cache.cached_products(study_id, fid, manifest_products):
                    cache.touch(study_id, fid, name)
                    continue
                info = manifest_products[name]
                need = int(info.get("bytes") or 0)
                own = [cache.product_path(study_id, fid, n)
                       for n in cache.cached_products(study_id, fid)]
                if not (need <= cache.budget and cache.make_room(need, keep=own)):
                    if fit:
                        skipped = names[position:]
                        break
                    raise experiments.DiskSpaceError(
                        f"cannot fetch {fid} · {name} ({need / 1024 ** 2:.0f} MB): the field "
                        f"cache holds at most {cache.budget / 1024 ** 3:.1f} GiB and this "
                        "field's own products stay", needed=need,
                        free=experiments.free_bytes(cache.root))
                try:
                    # Only the margin check raises DiskSpaceError here.
                    with claim(cache.root, need, min_free=cache.min_free):
                        _pull_verified(ssh, fid, name, info, remote_dir, partial,
                                       cache.product_path(study_id, fid, name))
                except experiments.DiskSpaceError:
                    if fit:
                        skipped = names[position:]
                        break
                    raise
                cache.record(study_id, fid, name, info)
                fetched.append(name)
        finally:
            shutil.rmtree(partial, ignore_errors=True)
    if progress is not None:
        progress(len(names), len(names), f"{fid} fetched")
    return {"fid": fid, "fetched": fetched, "skipped": skipped,
            "cached": sorted(cache.cached_products(study_id, fid, manifest_products)),
            "bytes": sum(int(manifest_products[n].get("bytes") or 0) for n in fetched)}


__all__ = [
    "FieldSource",
    "UploadError",
    "core_products",
    "fetch_field",
    "fetch_request",
    "member_products",
    "member_product",
    "open_field",
    "pack_field",
    "product_names",
    "remote_sha256",
    "upload_verified",
    "write_npz",
]
