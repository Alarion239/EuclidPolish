#!/usr/bin/env python3
"""Fail-closed submit-time check for a remote TNG radius manifest."""

from __future__ import annotations

import argparse
import json
import sys

from euclid_polish.config import Config
from euclid_polish.tng.radius_manifest import validate_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tng-dir", default=Config.TNG_SKIRT_DIR)
    parser.add_argument("--properties", default="")
    parser.add_argument("--manifest", default="")
    args = parser.parse_args(argv)
    result = validate_manifest(
        args.tng_dir,
        properties_path=args.properties or None,
        manifest_path_value=args.manifest or None,
    )
    print(json.dumps(result, sort_keys=True))
    if result.get("valid"):
        return 0
    for reason in result.get("reasons", [result.get("reason", "invalid")]):
        print(reason, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
