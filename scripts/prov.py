#!/usr/bin/env python
"""Thin entry point for the provenance CLI.

    python scripts/prov.py ancestors <id>
    python scripts/prov.py stale --model <id>
"""

from __future__ import annotations

import os
import sys

# Make ``euclid_polish`` importable when running this file directly.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from euclid_polish.provenance.cli import main

if __name__ == "__main__":
    sys.exit(main())
