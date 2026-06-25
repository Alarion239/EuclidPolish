#!/usr/bin/env python
"""Thin entry point for the provenance CLI.

    python scripts/prov.py ancestors <id>
    python scripts/prov.py stale --model <id>
"""

from __future__ import annotations

import sys

from euclid_polish.provenance.cli import main

if __name__ == "__main__":
    sys.exit(main())
