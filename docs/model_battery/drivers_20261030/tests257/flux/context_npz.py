"""Read-only on data/: a small npz in this folder holding only the old reports'
257@8M (report_fluxes.npz) and 258@950k Kron VIS fluxes, so the plot can show
them as context without duplicating the old 'production' (= 238) curve."""
import sys
import numpy as np
D = "/Users/alarion239/Desktop/EuclidPolish/data/jwst_euclid_overlap/flux_conservation/"
a = np.load(D + "report_fluxes.npz"); b = np.load(D + "report_258_950k_fluxes.npz")
assert np.array_equal(a["object_id"], b["object_id"])
out = {"object_id": a["object_id"], "vis_mag": a["vis_mag"], "lr:kron_vis": a["lr:kron_vis"],
       "milestone:member_257@8000000:kron_vis": a["milestone:member_257@8000000:kron_vis"],
       "milestone:member_258@950000:kron_vis": b["milestone:member_258@950000:kron_vis"]}
np.savez_compressed(sys.argv[1], **out)
print("wrote", sys.argv[1], {k: v.shape for k, v in out.items()})
