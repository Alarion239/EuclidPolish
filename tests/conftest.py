"""Shared pytest fixtures and config for the EuclidPolish test suite."""

from __future__ import annotations

import os
import sys

# Make ``euclid_polish`` importable even when pytest is run from /tests.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
