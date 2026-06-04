"""Shared constants for the web layer."""
import re

# Regex guard for cutout filenames: alphanumerics, underscores, dashes,
# dots. Rejects any path-traversal sequences ("..", "/").
_CUTOUT_FNAME_RE = re.compile(r"^[A-Za-z0-9_.\-]+\.fits$")
