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
                out.append((v, None, None))
            elif isinstance(v, dict):
                name = (v.get("name") or v.get("file") or v.get("filename")
                        or v.get("path"))
                out.append((name, v.get("url"),
                            v.get("size") or v.get("bytes")))
    return out


def _list_atlas(key: str):
    """Return [(name, url, size)] of the .hdf5 galaxy files under skirt_atlas.

    The endpoint is Django-REST-framework's *browsable API* (HTML), the same
    page ``wget -r -A hdf5`` crawls. We try JSON first (cleaner), then fall
    back to scraping ``href="...hdf5"`` links out of the HTML — so we can
    pick ONE file instead of downloading the whole atlas.
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
    # 2) Scrape the browsable-API HTML for .hdf5 links.
    html = _request(SKIRT_ATLAS, key).text
    seen, out = set(), []
    for href in re.findall(r'href=["\']([^"\']+\.hdf5)["\']', html):
        name = href.rstrip("/").split("/")[-1]
        if name in seen:
            continue
        seen.add(name)
        out.append((name, urljoin(SKIRT_ATLAS, href), None))
    return out


def _download(url: str, key: str, out_dir: str) -> str:
    r = _request(url, key, stream=True)
    cd = r.headers.get("content-disposition", "")
    if "filename=" in cd:
        name = cd.split("filename=")[1].strip().strip('";')
    else:
        name = url.rstrip("/").split("/")[-1] or "tng_download.bin"
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
    args = p.parse_args()
    key = _key(args)

    # --- per-subhalo broadband image (documented endpoint) ---
    if args.subhalo is not None:
        url = (f"{BASE}TNG50-1/snapshots/{args.snapshot}/subhalos/"
               f"{args.subhalo}/skirt/broadband_{args.survey}.fits")
        _download(url, key, args.out_dir)
        return 0

    # --- skirt_atlas listing (HTML browsable API or JSON) ---
    files = _list_atlas(key)

    if args.list or (not args.name and args.index is None):
        print(f"files/skirt_atlas/ — {len(files)} .hdf5 files:\n")
        for i, (name, url, size) in enumerate(files):
            sz = f"  ({int(size)/1e6:.0f} MB)" if size else ""
            print(f"  [{i}] {name}{sz}")
        if not files:
            print("  (no .hdf5 links found — the listing may need a different\n"
                  "   format; open the URL in a browser to inspect.)")
        print("\nRe-run with --name <file> or --index N to download just one.")
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
    if not url:                      # listing gave only names → build the URL
        url = SKIRT_ATLAS + name
    _download(url, key, args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
