"""scripts/infer_euclid_cutout.py with the Q1 cutout fetcher that falls back to IRSA's mirror
(euclid_polish.web.helpers.jwst_euclid.fetch_q1_cutout) while the ESA archive is down. Same arguments."""
import sys

sys.path.insert(0, "/Users/alarion239/Desktop/EuclidPolish")
sys.path.insert(0, "/Users/alarion239/Desktop/EuclidPolish/scripts")
import infer_euclid_cutout as inf
from euclid_polish.web.helpers.jwst_euclid import fetch_q1_cutout

inf.fetch_cutout_at = fetch_q1_cutout
sys.exit(inf.main(sys.argv[1:]))
