"""scripts/infer_euclid_cutout.py with the Q1 cutout fetcher that falls back to IRSA's mirror
(euclid_polish.web.helpers.jwst_euclid.fetch_q1_cutout) while the ESA archive is down. Same arguments."""
import sys

sys.path.insert(0, "/Users/alarion239/Desktop/EuclidPolish")
sys.path.insert(0, "/Users/alarion239/Desktop/EuclidPolish/scripts")
import infer_euclid_cutout as inf
import os

from euclid_polish.web.helpers import jwst_euclid
from euclid_polish.web.helpers.jwst_euclid import fetch_q1_cutout

if os.environ.get("TALK_SKIP_ESA"):
    # the ESA archive answers HTTP 500 (2026-11-18); go straight to IRSA's mirror
    jwst_euclid._esa_q1_cutout = lambda *a, **k: (False, "skipped (ESA down)")

inf.fetch_cutout_at = fetch_q1_cutout
sys.exit(inf.main(sys.argv[1:]))
