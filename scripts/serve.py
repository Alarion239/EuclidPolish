#!/usr/bin/env python
"""Run the EuclidPolish localhost web UI.

Thin entrypoint that imports the Flask app factory and starts the server.

Usage:
    python scripts/serve.py
    python scripts/serve.py --port 9000 --debug
"""

from __future__ import annotations

import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from euclid_polish.web.app import main


if __name__ == "__main__":
    main()
