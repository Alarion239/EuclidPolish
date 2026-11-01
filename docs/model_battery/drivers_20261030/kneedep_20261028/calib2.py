import numpy as np
from astropy.table import Table
from astropy.wcs import WCS
from photutils.aperture import CircularAperture, aperture_photometry
from euclid_polish.web.helpers import real_tiles as rt
from euclid_polish.photometry import uJy_to_electrons
from euclid_polish.config import Config
t = Table.read('data/jwst_euclid_overlap/flux_conservation/mer_nexus.fits')
print('fwhm col', np.nanpercentile(np.asarray(t['fwhm'],float),[5,50,95]), t['fwhm'].unit)
bands=['vis','y','j','h']; names=Config.LR_INPUT_BAND_NAMES
entries=[e for e in rt.list_entries('nexus') if e.extras.get('jwst_file')][::12]
scales=[0.05,0.1]   # fwhm in arcsec? → pixels via /0.1 ; if in pixels: D_px = n*fwhm
res={(b,n,s):[] for b in bands for n in (1,2,3,4) for s in ('arcsec','pix')}
for e in entries:
    lr=np.asarray(rt.get_tile('nexus',e.id,entry=e).lr_e,np.float64); w=WCS(rt.lr_header(e)).celestial
    x,y=w.world_to_pixel_values(np.asarray(t['right_ascension']),np.asarray(t['declination']))
    ok=(x>40)&(x<lr.shape[1]-41)&(y>40)&(y<lr.shape[0]-41)
    for i in np.flatnonzero(ok):
        fw=float(t['fwhm'][i])
        if not np.isfinite(fw) or fw<=0: continue
        for bi,b in enumerate(bands):
            f2=t[f'flux_{b}_2fwhm_aper'][i]; s2=t[f'fluxerr_{b}_2fwhm_aper'][i]
            if not(np.isfinite(f2) and f2>30*s2): continue
            conv=uJy_to_electrons(1.0,Config.get_band(names[bi]))
            for n in (1,2,3,4):
                c=t[f'flux_{b}_{n}fwhm_aper'][i]*conv
                for unit,dpx in (('arcsec',n*fw/0.1),('pix',n*fw)):
                    if dpx/2>38: continue
                    s=aperture_photometry(lr[...,bi],CircularAperture((x[i],y[i]),r=dpx/2),method='exact')['aperture_sum'][0]
                    res[(b,n,unit)].append(s/c)
for k,v in res.items():
    if v: print(k,len(v),'median',round(float(np.median(v)),4),'MAD',round(float(np.median(np.abs(np.array(v)-np.median(v)))),4))
