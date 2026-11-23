"""Wide, coarse ground-survey views around the HUDF spiral for the proportional-area ladder (box side ∝
sqrt(survey area), fixed by JWST = 12"): DSS2 red 55.3' (whole sky), Pan-STARRS1 g/r/i 47.2' (30,000 deg²,
a grid of fitscut cutouts at 1"/px, one per projection cell), Legacy Surveys DR10 g/r/i/z 19.3' at 0.5"/px
(DES 5,000 deg²). Into data/hudf_multi/{dsswide_red, ps1wide_<band>_<k>, lswide_griz}.fits."""
import os
import subprocess

import numpy as np

OUT = "/Users/alarion239/Desktop/EuclidPolish/paper/talk/data/hudf_multi"
GRA, GDEC = 53.17468, -27.799422


def get(url, path):
    if os.path.exists(path) and os.path.getsize(path) > 10000:
        return
    subprocess.run(["curl", "-s", "-m", "1800", "-o", path, url], check=True)
    print("wrote", os.path.basename(path), os.path.getsize(path), flush=True)


get(f"https://archive.stsci.edu/cgi-bin/dss_search?v=poss2ukstu_red&r={GRA}&d={GDEC}&e=J2000&h=57&w=57&f=fits",
    os.path.join(OUT, "dsswide_red.fits"))
get(f"https://www.legacysurvey.org/viewer/fits-cutout?ra={GRA}&dec={GDEC}&layer=ls-dr10&pixscale=1.0&size=1200&bands=griz",
    os.path.join(OUT, "lswide_griz.fits"))
cosd = np.cos(np.radians(GDEC))
k = 0
for dy in np.arange(-0.4, 0.41, 0.2):
    for dx in np.arange(-0.4, 0.41, 0.2):
        ra, dec = GRA + dx / cosd, GDEC + dy
        lst = subprocess.run(["curl", "-s", "-m", "120",
                              f"https://ps1images.stsci.edu/cgi-bin/ps1filenames.py?ra={ra}&dec={dec}&filters=gri"],
                             capture_output=True, text=True, check=True).stdout.split("\n")[1:]
        for line in lst:
            f = line.split()
            if len(f) < 8:
                continue
            band, fname = f[4], f[7]
            get(f"https://ps1images.stsci.edu/cgi-bin/fitscut.cgi?ra={ra}&dec={dec}&size=3200&output_size=800"
                f"&format=fits&red={fname}", os.path.join(OUT, f"ps1wide_{band}_{k:02d}.fits"))
        k += 1
print("done")
