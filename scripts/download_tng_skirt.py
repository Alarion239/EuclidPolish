#!/usr/bin/env python
"""Download one galaxy from the IllustrisTNG TNG50-1 SKIRT atlas.

The IllustrisTNG public API needs your *personal* API key
(https://www.tng-project.org/users/profile/ → "API Token"). It is NOT
committed; supply it at run time via the ``TNG_API_KEY`` env var or
``--api-key``.

Two-step usage (the ``files/skirt_atlas/`` listing format isn't documented,
so look first, then grab one):

    # 1) See what's available under files/skirt_atlas/
    TNG_API_KEY=xxxxx python scripts/download_tng_skirt.py --list

    # 2) Download one entry — by its name from the listing, or by index
    TNG_API_KEY=xxxxx python scripts/download_tng_skirt.py --name <file>
    TNG_API_KEY=xxxxx python scripts/download_tng_skirt.py --index 0

Bonus — a single galaxy's per-subhalo SKIRT broadband image (a different,
fully-documented endpoint), if that's what you actually want:

    TNG_API_KEY=xxxxx python scripts/download_tng_skirt.py \
        --subhalo 63871 --snapshot 99 --survey sdss

Files stream to ``--out-dir`` (default ``data/tng_skirt/``) so multi-GB
atlas files don't sit in RAM. Running with no selection defaults to --list.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from urllib.parse import urljoin

import requests

BASE = "https://www.tng-project.org/api/"
SKIRT_ATLAS = BASE + "TNG50-1/files/skirt_atlas/"


def _key(args) -> str:
    k = args.api_key or os.environ.get("TNG_API_KEY", "")
    if not k:
        sys.exit("No API key. Set TNG_API_KEY or pass --api-key "
                 "(get it from https://www.tng-project.org/users/profile/).")
    return k


def _request(url: str, key: str, *, stream: bool = False) -> requests.Response:
    r = requests.get(url, headers={"api-key": key}, stream=stream, timeout=120)
    if r.status_code in (401, 403):
        sys.exit(f"HTTP {r.status_code} — API key rejected/forbidden for {url}")
    r.raise_for_status()
    return r


def _candidate_files(listing):
    """Best-effort extraction of (name, url, size) entries from whatever shape
    the skirt_atlas listing comes back as (dict-of-urls, {'files':[...]}, …)."""
    out = []
    if isinstance(listing, dict):
        if isinstance(listing.get("files"), (list, dict)):
            return _candidate_files(listing["files"])
        for k, v in listing.items():
            if isinstance(v, str):
                out.append((str(k), v, None))
            elif isinstance(v, dict):
                out.append((str(k), v.get("url"),
                            v.get("size") or v.get("bytes")))
    elif isinstance(listing, list):
        for v in listing:
            if isinstance(v, str):
                # The skirt_atlas listing is a flat list of full file URLs.
                if v.startswith(("http://", "https://")):
                    out.append((_name_from_url(v), v, None))
                else:
                    out.append((v, None, None))
            elif isinstance(v, dict):
                name = (v.get("name") or v.get("file") or v.get("filename")
                        or v.get("path"))
                out.append((name, v.get("url"),
                            v.get("size") or v.get("bytes")))
    return out


def _name_from_url(url: str) -> str:
    """Unique, readable filename for a skirt_atlas URL, e.g.
    .../subhalos/665976/skirt/skirt_atlas.tar.gz → ``665976.tar.gz``.
    (Each subhalo's file is literally named ``skirt_atlas.tar.gz`` on the API
    side, so we key on the subhalo id to avoid collisions.)"""
    m = re.search(r"/subhalos/(\d+)/", url)
    last = url.rstrip("/").split("/")[-1] or "download"
    if m:
        ext = last.split(".", 1)[1] if "." in last else "tar.gz"
        return f"{m.group(1)}.{ext}"
    return last


def _list_atlas(key: str):
    """Return ``(name, url, size)`` entries exposed under ``skirt_atlas``.

    The endpoint is Django-REST-framework's *browsable API* (HTML), the same
    page a recursive downloader would crawl. We try JSON first (cleaner), then
    fall back to scraping archive links from the HTML — so we can pick one
    subhalo instead of downloading the whole atlas.
    """
    # 1) JSON if the server will give it.
    r = _request(SKIRT_ATLAS + "?format=json", key)
    if "application/json" in r.headers.get("content-type", ""):
        try:
            files = _candidate_files(r.json())
            if files:
                return files
        except ValueError:
            pass
    # 2) Scrape the browsable-API HTML for legacy .hdf5 listing links.
    html = _request(SKIRT_ATLAS, key).text
    seen, out = set(), []
    for href in re.findall(r'href=["\']([^"\']+\.hdf5)["\']', html):
        name = href.rstrip("/").split("/")[-1]
        if name in seen:
            continue
        seen.add(name)
        out.append((name, urljoin(SKIRT_ATLAS, href), None))
    return out


_TAR_SUFFIXES = (".tar.gz", ".tgz", ".tar.bz2", ".tar.xz", ".tar")
# Keep only the dusty Euclid-band renders: TNG<id>_O<orient>_Euclid_<band>.fits
# (band = VIS/Y/J/H — no underscore, so the *_nodust twins are excluded).
# Everything else in the atlas tarball — the _nodust files and the
# 2MASS / SDSS / GALEX / dustmass FITS — is dropped.
_KEEP_RE = re.compile(r"(?:^|/)TNG\d+_O\d+_Euclid_[^/_]+\.fits$", re.IGNORECASE)


def _extract(archive: str, *, remove: bool, euclid_only: bool = True) -> None:
    """Unpack a tar archive into a sibling dir named after it; safe filter.

    With ``euclid_only`` (default) only the dusty ``TNG*_O?_Euclid_<band>.fits``
    members are written — the _nodust twins and the other-survey FITS are never
    extracted, so a ~2 GB galaxy tarball lands as ~20 Euclid frames instead of
    240 files."""
    import tarfile
    if not archive.endswith(_TAR_SUFFIXES):
        return
    dest = archive
    for suf in _TAR_SUFFIXES:
        if dest.endswith(suf):
            dest = dest[: -len(suf)]
            break
    os.makedirs(dest, exist_ok=True)
    with tarfile.open(archive) as tf:
        members = None
        if euclid_only:
            members = [m for m in tf.getmembers()
                       if m.isfile() and _KEEP_RE.search(m.name)]
            if not members:
                print("  ⚠ no TNG*_O?_Euclid_*.fits members matched — "
                      "extracting everything instead")
                members = None
        try:
            tf.extractall(dest, members=members, filter="data")  # py3.12 safe
        except TypeError:
            tf.extractall(dest, members=members)                 # older pythons
    n = sum(len(fs) for _dp, _d, fs in os.walk(dest))
    kept = "Euclid FITS" if euclid_only else "files"
    print(f"  ✓ unpacked {n} {kept} → {dest}/")
    if remove:
        try:
            os.remove(archive)
            print(f"  ✓ removed {os.path.basename(archive)}")
        except OSError:
            pass


def _download(url: str, key: str, out_dir: str, *,
              extract: bool = True, keep_archive: bool = False,
              euclid_only: bool = True) -> str:
    r = _request(url, key, stream=True)         # follows redirects → data host
    cd = r.headers.get("content-disposition", "")
    name = cd.split("filename=")[1].strip().strip('";') if "filename=" in cd else _name_from_url(url)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    total = int(r.headers.get("content-length", 0))
    done = 0
    print(f"downloading → {path}")
    with open(path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 20):
            if not chunk:
                continue
            f.write(chunk)
            done += len(chunk)
            bar = (f"{done/1e6:.0f}/{total/1e6:.0f} MB"
                   if total else f"{done/1e6:.0f} MB")
            print(f"\r  {bar}", end="", flush=True)
    print(f"\n  ✓ saved {path} ({done:,} bytes)")
    if extract:
        _extract(path, remove=not keep_archive, euclid_only=euclid_only)
    return path


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--api-key", default="", help="TNG API key (or $TNG_API_KEY)")
    p.add_argument("--out-dir", default="data/tng_skirt",
                   help="download directory (default: data/tng_skirt)")
    p.add_argument("--list", action="store_true",
                   help="list the files under files/skirt_atlas/ and exit")
    p.add_argument("--name", default="",
                   help="download this entry name from the skirt_atlas listing")
    p.add_argument("--index", type=int, default=None,
                   help="download the Nth entry from the listing (0-based)")
    # Per-subhalo SKIRT broadband image (separate documented endpoint).
    p.add_argument("--subhalo", type=int, default=None,
                   help="download a single galaxy's SKIRT broadband image")
    p.add_argument("--snapshot", type=int, default=99,
                   help="snapshot number for --subhalo (default 99 = z=0)")
    p.add_argument("--survey", default="sdss",
                   help="survey for --subhalo broadband (sdss | pogs)")
    p.add_argument("--no-extract", action="store_true",
                   help="don't auto-unpack a downloaded .tar.gz")
    p.add_argument("--keep-archive", action="store_true",
                   help="keep the .tar.gz after unpacking (default: delete it)")
    p.add_argument("--all-bands", action="store_true",
                   help="extract every FITS, not just TNG*_O?_Euclid_*.fits")
    args = p.parse_args()
    key = _key(args)
    dl = {"extract": not args.no_extract, "keep_archive": args.keep_archive,
              "euclid_only": not args.all_bands}

    # --- per-subhalo broadband image (documented endpoint) ---
    if args.subhalo is not None:
        url = (f"{BASE}TNG50-1/snapshots/{args.snapshot}/subhalos/"
               f"{args.subhalo}/skirt/broadband_{args.survey}.fits")
        _download(url, key, args.out_dir, **dl)
        return 0

    # --- skirt_atlas listing (HTML browsable API or JSON) ---
    files = _list_atlas(key)

    if args.list or (not args.name and args.index is None):
        print(f"files/skirt_atlas/ — {len(files)} entries:\n")
        for i, (name, _url, size) in enumerate(files):
            sz = f"  ({int(size)/1e6:.0f} MB)" if size else ""
            print(f"  [{i}] {name}{sz}")
        if not files:
            print("  (no entries parsed — open the URL in a browser to inspect.)")
        print("\nRe-run with --index N (or --name <file>) to download just one.")
        return 0

    # --- resolve the chosen entry → download ---
    chosen = None
    if args.name:
        chosen = next((t for t in files if t[0] == args.name), None)
        if chosen is None:
            sys.exit(f"'{args.name}' not in the listing. Run --list to see names.")
    elif args.index is not None:
        if not (0 <= args.index < len(files)):
            sys.exit(f"--index {args.index} out of range (0..{len(files)-1}).")
        chosen = files[args.index]

    name, url, _size = chosen
    if not url:                      # listing gave only a name → build the URL
        url = SKIRT_ATLAS + name
    _download(url, key, args.out_dir, **dl)
    return 0


if __name__ == "__main__":
    sys.exit(main())
