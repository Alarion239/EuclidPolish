"""Resumable, provenance-preserving access to public Euclid Q1 products."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import requests
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

from . import BANDS, SCHEMA_VERSION

IRSA = "https://irsa.ipac.caltech.edu"
DPDD = "https://euclid.esac.esa.int/dr/q1/dpdd/merdpd/dpcards/mer_bksmosaic.html"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def utc_now():
    return datetime.now(UTC).isoformat()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


class Archive:
    def __init__(self, root):
        self.root = Path(root)
        self.cache = self.root / "cache"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "EuclidPolish-noise-assessment/1 (public science diagnostics)"

    def get(self, url, *, params=None, headers=None, max_bytes=128_000_000):
        # Only read public scientific products. No credentials are needed or logged.
        if urlparse(url).hostname not in {
            "irsa.ipac.caltech.edu",
            "nasa-irsa-euclid-q1.s3.us-east-1.amazonaws.com",
        }:
            raise ValueError("Unrecognized archive host")
        for attempt in range(3):
            try:
                with self.session.get(
                    url, params=params, headers=headers, timeout=(15, 60), stream=True
                ) as response:
                    response.raise_for_status()
                    if int(response.headers.get("Content-Length", 0)) > max_bytes:
                        raise ValueError("Archive response exceeds bounded retrieval size")
                    chunks, total = [], 0
                    for chunk in response.iter_content(1 << 20):
                        total += len(chunk)
                        if total > max_bytes:
                            raise ValueError("Archive response exceeds bounded retrieval size")
                        chunks.append(chunk)
                    return b"".join(chunks), dict(response.headers), response.status_code
            except (requests.ConnectionError, requests.Timeout):
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)
        raise RuntimeError("Unreachable retrieval state")

    def cached(self, url, *, params=None, headers=None, suffix=".bin", max_bytes=128_000_000):
        request = {"url": url, "parameters": params or {}, "headers": headers or {}}
        key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
        path = self.cache / (key + suffix)
        provenance = path.with_suffix(path.suffix + ".json")
        if path.exists() and provenance.exists():
            record = read_json(provenance)
            if digest(path) != record["sha256"]:
                raise ValueError(f"Cached product checksum mismatch: {path}")
            return path, record
        content, response_headers, status = self.get(url, params=params, headers=headers, max_bytes=max_bytes)
        temporary = path.with_suffix(path.suffix + ".part")
        temporary.write_bytes(content)
        temporary.replace(path)
        record = {
            **request,
            "retrieved_at": utc_now(),
            "http_status": status,
            "bytes": len(content),
            "sha256": digest(path),
            "local_path": str(path.relative_to(self.root)),
            "etag": response_headers.get("ETag"),
            "last_modified": response_headers.get("Last-Modified"),
            "content_range": response_headers.get("Content-Range"),
            "checksum_scope": "SHA256 of retrieved bytes; verified on every cache reuse",
            "upstream_checksum": "unavailable; multipart S3 ETag is not a file checksum",
        }
        save_json(provenance, record)
        return path, record

    def sia(self, ra, dec, collection):
        path, provenance = self.cached(
            IRSA + "/SIA",
            params={"COLLECTION": collection, "POS": f"CIRCLE {ra:.12f} {dec:.12f} 0.00001"},
            suffix=".xml",
        )
        table = Table.read(path, format="votable", use_names_over_ids=True)
        rows = []
        for row in table:
            record = {}
            for key in table.colnames:
                value = row[key]
                if np.ma.is_masked(value):
                    record[key] = None
                elif isinstance(value, np.generic):
                    record[key] = value.item()
                else:
                    record[key] = value
            rows.append(record)
        return rows, provenance

    def ancestry(self, tile):
        if not str(tile).isdigit():
            raise ValueError("Tile must be a numeric archive identity")
        path, provenance = self.cached(
            IRSA + "/TAP/sync",
            params={
                "REQUEST": "doQuery",
                "LANG": "ADQL",
                "FORMAT": "csv",
                "QUERY": f"SELECT * FROM euclid.tileid_association_q1 WHERE tileid={tile}",
            },
            suffix=".csv",
        )
        rows = list(csv.DictReader(io.StringIO(path.read_text())))
        if rows and "observationid" not in rows[0]:
            raise ValueError("Archive ancestry query did not return observation identities")
        return rows, provenance

    def cutout(self, row, ra, dec, size):
        path, provenance = self.cached(
            row["access_url"],
            params={"center": f"{ra:.12f},{dec:.12f}", "size": f"{size * 0.1 / 3600:.12f}", "gzip": "false"},
            suffix=".fits",
        )
        with fits.open(path, memmap=False) as hdus:
            hdus.verify("exception")
            hdu = next((h for h in hdus if h.data is not None and h.data.ndim == 2), None)
            if hdu is None:
                raise ValueError("No two-dimensional image in archive cutout")
            if "CHECKSUM" in hdu.header and hdu.verify_checksum() != 1:
                raise ValueError("FITS CHECKSUM verification failed")
            if "DATASUM" in hdu.header and hdu.verify_datasum() != 1:
                raise ValueError("FITS DATASUM verification failed")
            if hdu.header.get("DATASETR") != "Q1_R1":
                raise ValueError("Unverified or different release in product header")
            if max(abs(n - size) for n in hdu.data.shape) > 10:
                raise ValueError("Unexpected cutout dimensions")
            provenance = {
                **provenance,
                "header": hdu.header.tostring(sep="\n"),
                "shape": list(hdu.data.shape),
                "release": hdu.header["DATASETR"],
                "parent_dimensions": [row.get("s_xel2"), row.get("s_xel1")],
                "archive_metadata": row,
                "status": "available",
            }
        return provenance


def image_product(root, product):
    path = Path(root) / product["local_path"]
    if digest(path) != product["sha256"]:
        raise ValueError(f"Product checksum mismatch: {path}")
    with fits.open(path, memmap=False) as hdus:
        hdu = next(h for h in hdus if h.data is not None and h.data.ndim == 2)
        return hdu.data.copy(), hdu.header.copy()


def assert_aligned(headers, shapes, tolerance=1e-3):
    if len({tuple(s) for s in shapes}) != 1:
        raise ValueError("Science and auxiliary dimensions differ")
    h, w = shapes[0]
    x, y = np.array([0, w - 1, 0, w - 1, (w - 1) / 2]), np.array([0, 0, h - 1, h - 1, (h - 1) / 2])
    sky = WCS(headers[0]).celestial.pixel_to_world(x, y)
    for header in headers[1:]:
        ax, ay = WCS(header).celestial.world_to_pixel(sky)
        if not np.all(np.isfinite(ax + ay)) or np.max(np.hypot(ax - x, ay - y)) > tolerance:
            raise ValueError("Science and auxiliary WCS are not aligned")


def initialize(root, source_manifest, parents_manifest, example_provenance, size=256):
    root = Path(root)
    target = root / "manifest.json"
    if target.exists():
        manifest = read_json(target)
        freeze_example_references(root, manifest)
        return manifest
    source, parents = read_json(source_manifest), read_json(parents_manifest)
    by_id = {
        s["source_sample_id"]: s["archive_parents"] for s in parents["samples"] if s["position_index"] == 0
    }
    samples = [s for s in source["samples"] if s.get("status") == "written"]
    if len(samples) != 44 or source.get("source_release", source.get("release", "Q1_R1")) != "Q1_R1":
        raise ValueError("Expected the 44 saved Q1 sampling positions")
    patches = []
    for sample in samples:
        sid = sample["sample_id"]
        patches.append(
            {
                "patch_id": f"pointing_{sid:04d}",
                "sample_id": sid,
                "field": sample["field"],
                "ra": sample["ra"],
                "dec": sample["dec"],
                "size": size,
                "kind": "central",
                "parents": {b: by_id[sid][b if b == "VIS" else b + "_E"] for b in BANDS},
            }
        )
    examples = read_json(example_provenance)
    for example in examples["examples"]:
        sid = example["sample_id"]
        local = Path(source_manifest).parent / "cutouts" / Path(example["source_file"]).name
        if digest(local) != example["source_sha256"]:
            raise ValueError("Saved example source checksum changed")
        with fits.open(local, memmap=True) as hdus:
            hdu = hdus["VIS"]
            height, width = hdu.shape
            x = (width - width // 256 * 256) // 2 + 256 * example["patch_grid_column_zero_based"] + 127.5
            y = (height - height // 256 * 256) // 2 + 256 * example["patch_grid_row_zero_based"] + 127.5
            ra, dec = WCS(hdu.header).celestial.pixel_to_world_values(x, y)
        base = next(p for p in patches if p["sample_id"] == sid and p["kind"] == "central")
        patches.append(
            {
                **base,
                "patch_id": f"example_{sid:04d}_{example['patch_index_zero_based']}",
                "ra": float(ra),
                "dec": float(dec),
                "size": 256,
                "kind": example["kind"],
                "legacy_example": example,
                "example_array": str((Path(example_provenance).parent / example["array_file"]).resolve()),
            }
        )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "release": "Q1_R1",
        "diagnostic_amplitude_target": 0.05,
        "source_manifest": str(Path(source_manifest).resolve()),
        "source_manifest_sha256": digest(source_manifest),
        "parents_manifest_sha256": digest(parents_manifest),
        "examples_provenance_sha256": digest(example_provenance),
        "patches": patches,
        "pilot_verified": False,
        "sampling": {
            "central_size_pixels": size,
            "pointings": 44,
            "additional_example_patches": 4,
            "rule": "One central patch per saved pointing, plus the four exact previously displayed patches",
            "exposure_validation": "Per field: low, median, high central-patch official VIS RMS, then"
            " separate bright-star stress case",
        },
        "definitions": {
            "official_rms": "Total uncertainty including source photon noise; not background-only",
            "legacy_mad": "Existing source-masked, plane-subtracted MAD; diagnostic application in each band",
            "background_decomposition": "Unavailable unless individual product metadata explicitly "
            "supplies components",
        },
    }
    freeze_example_references(root, manifest)
    return manifest


def freeze_example_references(root, manifest):
    """Make saved example validation portable with the diagnostic directory."""
    root = Path(root)
    for patch in manifest["patches"]:
        if "legacy_example" not in patch or "reference_wcs_header" in patch:
            continue
        example = patch["legacy_example"]
        source = Path(example["source_file"])
        if not source.exists():
            source = Path(manifest["source_manifest"]).parent / "cutouts" / source.name
        if digest(source) != example["source_sha256"]:
            raise ValueError("Example source checksum does not match saved provenance")
        with fits.open(source, memmap=True) as hdus:
            hdu = hdus["VIS"]
            height, width = hdu.shape
            x0 = (width - width // 256 * 256) // 2 + 256 * example["patch_grid_column_zero_based"]
            y0 = (height - height // 256 * 256) // 2 + 256 * example["patch_grid_row_zero_based"]
            header = hdu.header.copy()
        header["CRPIX1"] -= x0
        header["CRPIX2"] -= y0
        header["NAXIS1"], header["NAXIS2"] = 256, 256
        patch["reference_wcs_header"] = header.tostring(sep="\n")
        target = root / "references" / (patch["patch_id"] + ".npz")
        target.parent.mkdir(exist_ok=True)
        original = Path(patch["example_array"])
        shutil.copyfile(original, target)
        patch["example_array"] = str(target.relative_to(root))
        patch["example_array_sha256"] = digest(target)
    save_json(root / "manifest.json", manifest)


def acquire_mer(root, pilot=False):
    root = Path(root)
    manifest = read_json(root / "manifest.json")
    if not pilot and not manifest["pilot_verified"]:
        raise ValueError("Run acquisition with --pilot successfully before expanding")
    archive = Archive(root)
    for patch in manifest["patches"][:1] if pilot else manifest["patches"]:
        products = patch.setdefault("mer", {})
        rows, query = archive.sia(patch["ra"], patch["dec"], "euclid_DpdMerBksMosaic")
        patch["mer_query"] = query
        tile = patch["parents"]["VIS"]["tile_index"]
        ancestry, ancestry_query = archive.ancestry(tile)
        patch["ancestry"] = ancestry
        patch["ancestry_query"] = ancestry_query
        patch["ancestry_scope"] = (
            "Archive tile-to-observation associations; exact per-pixel contributing layers unavailable"
        )
        for band in BANDS:
            parent = patch["parents"][band]
            candidates = [
                r
                for r in rows
                if r["energy_bandpassname"] == band
                and r["obs_id"] == f"{parent['tile_index']}_{parent['instrument_name']}"
            ]
            science = [
                r for r in candidates if Path(urlparse(r["access_url"]).path).name == parent["file_name"]
            ]
            record = products.setdefault(band, {})
            if len(science) != 1:
                record["status"] = "unavailable: exact saved parent science identity not found"
                continue
            did = science[0]["obs_publisher_did"]
            roles = {
                "science": science,
                "rms": [r for r in candidates if r["dataproduct_subtype"] == "noise"],
                "flags": [r for r in candidates if "-FLAG_" in r["access_url"]],
            }
            for role, choices in roles.items():
                choices = [r for r in choices if r["obs_publisher_did"] == did]
                if len(choices) != 1:
                    record[role] = {
                        "status": "unavailable",
                        "reason": "Missing or ambiguous archive association",
                    }
                    continue
                if record.get(role, {}).get("status") == "available":
                    if digest(root / record[role]["local_path"]) != record[role]["sha256"]:
                        raise ValueError("Existing product checksum mismatch")
                    continue
                try:
                    record[role] = archive.cutout(choices[0], patch["ra"], patch["dec"], patch["size"])
                except (ValueError, OSError, requests.RequestException) as exc:
                    record[role] = {
                        "status": "unavailable",
                        "reason": str(exc),
                        "archive_metadata": choices[0],
                    }
                save_json(root / "manifest.json", manifest)
            if all(record.get(role, {}).get("status") == "available" for role in roles):
                arrays, headers = zip(*(image_product(root, record[role]) for role in roles), strict=True)
                try:
                    assert_aligned(headers, [a.shape for a in arrays])
                    if len({h.get("PPOID") for h in headers}) != 1:
                        raise ValueError("Science and auxiliary pipeline parent identities differ")
                    record["status"] = "verified"
                    record["alignment_tolerance_pixels"] = 0.001
                except ValueError as exc:
                    record["status"] = "unavailable: " + str(exc)
            else:
                record["status"] = "unavailable: missing auxiliary product"
            print(f"MER {patch['patch_id']} {band}: {record['status']}", flush=True)
            save_json(root / "manifest.json", manifest)
    if pilot:
        manifest["pilot_verified"] = all(
            r.get("status") == "verified" for r in manifest["patches"][0]["mer"].values()
        )
    save_json(root / "manifest.json", manifest)
    return manifest


class RemoteFits:
    """Read only FITS headers and rectangular strips, cache each byte range once.

    No compressed-image or arbitrary table decoding is guessed. Range responses
    must be exact 206 responses; a server returning the whole file is rejected.
    """

    def __init__(self, archive, url):
        self.archive, self.url = archive, url
        self.records = []
        self.source_etag = None

    def header_at(self, offset):
        """Fetch several header blocks per request; ignore bytes after END."""
        blocks = bytearray()
        while True:
            chunk = self.read(offset + len(blocks), 11520)
            for start in range(0, len(chunk), 2880):
                block = chunk[start : start + 2880]
                blocks.extend(block)
                if any(block[i : i + 8] == b"END     " for i in range(0, 2880, 80)):
                    header = fits.Header.fromstring(bytes(blocks).decode("ascii"), sep="")
                    dimensions = [header[f"NAXIS{i + 1}"] for i in range(header["NAXIS"])]
                    count = math.prod(dimensions) if dimensions else 0
                    nbytes = (
                        (count + header.get("PCOUNT", 0))
                        * abs(header["BITPIX"])
                        // 8
                        * header.get("GCOUNT", 1)
                    )
                    data_start = offset + len(blocks)
                    return header, data_start, data_start + math.ceil(nbytes / 2880) * 2880

    def detector(self, ra, dec, size, triplets=True):
        """Locate a detector using native WCS, verifying predicted HDU identities.

        Euclid MEFs repeat fixed detector layouts. A stride is learned from the
        first detector and every accessed header is checked; varying layouts
        fall back to sequential FITS traversal. No data offset is used unchecked.
        """
        primary, _, first = self.header_at(0)
        header, offset, next_offset = self.header_at(first)
        triplet = [(header, offset)]
        if triplets:
            for _ in range(2):
                hh, oo, next_offset = self.header_at(next_offset)
                triplet.append((hh, oo))
        stride = next_offset - first
        count = 144 if str(primary.get("INSTRUME", "")).startswith("VIS") else 16
        order = list(range(count))
        # Prioritize detector candidates using a few actual WCS headers. This
        # changes search order only: every accepted detector still passes its
        # full native WCS coverage check. It avoids scanning a 7 GB VIS MEF's
        # hundred-plus preceding extensions through individual range requests.
        basis_indices = (1, 3, 4, 24) if count == 144 else (1, 4)
        base_wcs = WCS(header).celestial
        base_center = np.array([header["NAXIS1"] / 2, header["NAXIS2"] / 2])
        centers = {}
        for index in basis_indices:
            candidate, _, _ = self.header_at(first + index * stride)
            if not str(candidate.get("EXTNAME", "")).endswith(".SCI"):
                break
            sky = WCS(candidate).celestial.pixel_to_world(candidate["NAXIS1"] / 2, candidate["NAXIS2"] / 2)
            centers[index] = np.array(base_wcs.world_to_pixel(sky)) - base_center
        if len(centers) == len(basis_indices):
            projected = np.array(base_wcs.world_to_pixel_values(ra, dec)) - base_center

            def distance(index):
                if count == 144:
                    ccd, quadrant = divmod(index, 4)
                    row, column = divmod(ccd, 6)
                    qshift = (0, centers[1], centers[1] + centers[3], centers[3])[quadrant]
                    center = column * centers[4] + row * centers[24] + qshift
                else:
                    row, column = divmod(index, 4)
                    center = column * centers[1] + row * centers[4]
                return float(np.linalg.norm(projected - center))

            order.sort(key=distance)
        for detector in order:
            start = first + detector * stride
            hh, oo, _ = self.header_at(start) if detector else (header, offset, next_offset)
            if not str(hh.get("EXTNAME", "")).endswith(".SCI"):
                break
            x, y = WCS(hh).celestial.world_to_pixel_values(ra, dec)
            x0, y0 = int(np.floor(x - size / 2)), int(np.floor(y - size / 2))
            if 0 <= x < hh["NAXIS1"] and 0 <= y < hh["NAXIS2"]:
                x0 = int(np.clip(x0, 0, hh["NAXIS1"] - size))
                y0 = int(np.clip(y0, 0, hh["NAXIS2"] - size))
                return primary, detector, hh, oo, x0, y0, stride
        else:
            raise ValueError("No native detector fully covers the patch and transformation margin")
        for _, hh, oo in self.headers():
            if not str(hh.get("EXTNAME", "")).endswith(".SCI"):
                continue
            x, y = WCS(hh).celestial.world_to_pixel_values(ra, dec)
            x0, y0 = int(np.floor(x - size / 2)), int(np.floor(y - size / 2))
            if 0 <= x < hh["NAXIS1"] and 0 <= y < hh["NAXIS2"]:
                x0 = int(np.clip(x0, 0, hh["NAXIS1"] - size))
                y0 = int(np.clip(y0, 0, hh["NAXIS2"] - size))
                return primary, None, hh, oo, x0, y0, None
        raise ValueError("No native detector fully covers requested sky position")

    def read(self, start, count):
        path, record = self.archive.cached(
            self.url, headers={"Range": f"bytes={start}-{start + count - 1}"}, max_bytes=count + 1
        )
        content_range = record.get("content_range") or ""
        if record["http_status"] != 206 or not content_range.startswith(
            f"bytes {start}-{start + count - 1}/"
        ):
            raise ValueError("Archive did not honor exact FITS byte range")
        if self.source_etag is not None and record.get("etag") != self.source_etag:
            raise ValueError("Source ETag changed between FITS byte ranges")
        self.source_etag = record.get("etag")
        self.records.append(record)
        return path.read_bytes()

    def headers(self):
        offset = 0
        for index in range(600):
            blocks = bytearray()
            while True:
                block = self.read(offset + len(blocks), 2880)
                blocks.extend(block)
                if any(block[i : i + 8] == b"END     " for i in range(0, 2880, 80)):
                    break
            header = fits.Header.fromstring(bytes(blocks).decode("ascii"), sep="")
            dimensions = [header[f"NAXIS{i + 1}"] for i in range(header["NAXIS"])]
            count = math.prod(dimensions) if dimensions else 0
            nbytes = (count + header.get("PCOUNT", 0)) * abs(header["BITPIX"]) // 8 * header.get("GCOUNT", 1)
            data_start = offset + len(blocks)
            yield index, header, data_start
            offset = data_start + math.ceil(nbytes / 2880) * 2880
            total = int(self.records[-1]["content_range"].split("/")[-1])
            if offset >= total:
                break

    def rectangle(self, header, offset, x0, y0, size):
        h, w = header["NAXIS2"], header["NAXIS1"]
        if header.get("ZIMAGE") or header["NAXIS"] != 2 or x0 < 0 or y0 < 0 or x0 + size > w or y0 + size > h:
            raise ValueError("Unsupported image layout or missing native detector coverage")
        bitpix = header["BITPIX"]
        dtype = {8: ">u1", 16: ">i2", 32: ">i4", 64: ">i8", -32: ">f4", -64: ">f8"}[bitpix]
        bytes_per_pixel = abs(bitpix) // 8
        content = self.read(offset + y0 * w * bytes_per_pixel, size * w * bytes_per_pixel)
        values = np.frombuffer(content, dtype=dtype).reshape(size, w)[:, x0 : x0 + size].astype(float)
        values = values * header.get("BSCALE", 1) + header.get("BZERO", 0)
        shifted = header.copy()
        shifted["CRPIX1"] -= x0
        shifted["CRPIX2"] -= y0
        shifted["NAXIS1"], shifted["NAXIS2"] = size, size
        return values, shifted
