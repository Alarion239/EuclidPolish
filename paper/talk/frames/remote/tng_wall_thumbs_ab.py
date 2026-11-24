"""Run on FASRC: realistic-colour (11-23) re-render of the talk's TNG50-SKIRT atlas wall thumbnails.
Same galaxies and crops as tng_wall_thumbs.py (read from its meta.json: id, fov_kpc), orientation 1.
The atlas images are MJy/sr (f_nu), so the AB balance of the talk (ab_colour.py: a flat-f_nu source is
white) is equal channel weights: R = H, G = (Y + J) / 2, B = VIS, each through the same asinh stretch
(white = 99.7th pct of the intensity, knee = white / 20). The old render white-balanced on the wall's
median galaxy, which made every galaxy blue-white.
usage: python tng_wall_thumbs_ab.py <atlas root> <meta.json> <out dir>"""
import json
import os
import sys

import numpy as np
from astropy.io import fits
from PIL import Image

root, meta_path, out = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(out, exist_ok=True)
for m in json.load(open(meta_path)):
    sid = int(m["id"])
    planes = [np.asarray(fits.getdata(os.path.join(root, str(sid), f"TNG{sid:06d}_O1_Euclid_{b}.fits")), np.float64)
              for b in ("VIS", "Y", "J", "H")]
    h, w = planes[0].shape
    hw = int(round(m["fov_kpc"] / 0.1 / 2))
    cy, cx = h // 2, w // 2
    cube = np.stack([p[cy - hw:cy + hw, cx - hw:cx + hw] for p in planes], -1)
    c = np.stack([np.asarray(Image.fromarray(cube[..., k].astype(np.float32)).resize((360, 360), Image.BOX))
                  for k in range(4)], -1)
    R, G, B = c[..., 3], 0.5 * (c[..., 1] + c[..., 2]), c[..., 0]
    white = np.percentile((R + G + B) / 3, 99.7)
    knee = white / 20
    rgb = np.stack([np.clip(np.arcsinh(np.maximum(x, 0) / knee) / np.arcsinh(white / knee), 0, 1) for x in (R, G, B)], -1)
    Image.fromarray((rgb * 255).astype(np.uint8)[::-1]).save(os.path.join(out, f"tng_{sid:06d}.png"))
    print(sid, hw, flush=True)
