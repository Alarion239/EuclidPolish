import numpy as np, sys
from astropy.table import Table
from astropy.wcs import WCS
from photutils.aperture import CircularAperture, aperture_photometry
from euclid_polish.web.helpers import real_tiles as rt
from euclid_polish.photometry import uJy_to_electrons
from euclid_polish.config import Config
t = Table.read('data/jwst_euclid_overlap/flux_conservation/mer_nexus.fits')
bands = ['vis','y','j','h']
cfg = None

entries=[e for e in rt.list_entries('nexus') if e.extras.get('jwst_file')][::12]
D = np.arange(0.1, 2.61, 0.05)
rows = {b: [] for b in bands}
for e in entries:
    lr = np.asarray(rt.get_tile('nexus', e.id, entry=e).lr_e, np.float64)
    w = WCS(rt.lr_header(e)).celestial
    x, y = w.world_to_pixel_values(np.asarray(t['right_ascension']), np.asarray(t['declination']))
    ok = (x > 30) & (x < lr.shape[1]-31) & (y > 30) & (y < lr.shape[0]-31)
    for i in np.flatnonzero(ok):
        pos = (x[i], y[i])
        for bi, b in enumerate(bands):
            f = t[f'flux_{b}_2fwhm_aper'][i]; s = t[f'fluxerr_{b}_2fwhm_aper'][i]
            if not (np.isfinite(f) and np.isfinite(s) and f > 30*s): continue
            sums = [aperture_photometry(lr[..., bi], CircularAperture(pos, r=d/0.1/2), method='exact')['aperture_sum'][0] for d in D]
            cat = [t[f'flux_{b}_{n}fwhm_aper'][i] for n in (1,2,3,4)]
            rows[b].append((np.array(sums), np.array(cat, float)))
for bi, b in enumerate(bands):
    band = Config.LR_INPUT_BAND_NAMES[bi]
    S = np.array([r[0] for r in rows[b]]); C = np.array([r[1] for r in rows[b]])
    conv = uJy_to_electrons(1.0, Config.get_band(band))
    print(b, len(S), 'expected e-/uJy', round(conv,2))
    for n in range(4):
        ratio = S / C[:, n:n+1]   # e- per uJy
        # best D: where the scatter of ratio across objects is smallest relative to its median
        med = np.nanmedian(ratio/conv,0); k = int(np.argmin(np.abs(med-1)))
        spread = np.nanmedian(np.abs(ratio/conv-1),0)
        print(f'  n={n+1}: D(match)={D[k]:.2f}" spread {spread[k]:.4f} median ratio {med[k]:.3f}; also at D=0.4..1.2: {np.round(med[[6,10,14,18,22]],3)}')
