"""Run on FASRC: deep-stretch VIS views of atlas galaxies that show what the talk calls the plain downsides
(dust-box facets, periodic-box cuts, particle grain). Writes <id>_O<k>.png (800 px, full 160 kpc frame)."""
import os, sys
import numpy as np
from astropy.io import fits
from PIL import Image
sys.path.insert(0, os.getcwd())
from euclid_polish.tng.atlas import TNGGalaxy

root, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
for spec in sys.argv[3:]:
    sid, o = spec.split(":")
    g = TNGGalaxy(os.path.join(root, sid), sid)
    try:
        img = np.asarray(fits.getdata(g.fits_path(int(o), "VIS")), np.float64)
    except Exception as e:
        print(spec, "missing", e); continue
    pos = img[img > 0]
    white = np.percentile(pos, 99.5); soft = np.percentile(pos, 20)
    f = np.clip(np.arcsinh(np.maximum(img, 0) / soft) / np.arcsinh(white / soft), 0, 1)
    Image.fromarray((f * 255).astype(np.uint8)[::-1]).resize((800, 800), Image.LANCZOS).save(
        os.path.join(out, f"{int(sid):06d}_O{o}.png"))
    print(spec, img.shape, "ok")
