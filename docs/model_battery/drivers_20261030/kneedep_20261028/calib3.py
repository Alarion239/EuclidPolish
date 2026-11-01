import numpy as np
from astropy.table import Table
from astropy.wcs import WCS
from scipy.ndimage import gaussian_filter
from photutils.aperture import CircularAperture, aperture_photometry
from euclid_polish.web.helpers import real_tiles as rt
from euclid_polish.photometry import uJy_to_electrons
from euclid_polish.config import Config
t = Table.read('data/jwst_euclid_overlap/flux_conservation/mer_nexus.fits')
bands=['vis','y','j','h']; names=Config.LR_INPUT_BAND_NAMES
native={'vis':0.16,'y':0.46,'j':0.48,'h':0.51}
entries=[e for e in rt.list_entries('nexus') if e.extras.get('jwst_file')][::12]
res={(b,n,m):[] for b in bands for n in (1,2,3,4) for m in ('native','matched')}
pull={(b,n):[] for b in bands for n in (1,2,3,4)}
for e in entries:
    lr=np.asarray(rt.get_tile('nexus',e.id,entry=e).lr_e,np.float64); w=WCS(rt.lr_header(e)).celestial
    x,y=w.world_to_pixel_values(np.asarray(t['right_ascension']),np.asarray(t['declination']))
    ok=(x>45)&(x<lr.shape[1]-46)&(y>45)&(y<lr.shape[0]-46)
    F=float(np.nanmedian(np.asarray(t['fwhm'],float)[ok])) if ok.any() else 1.2
    for bi,b in enumerate(bands):
        sig=np.sqrt(max(F**2-native[b]**2,0))/2.3548/0.1
        im={'native':lr[...,bi],'matched':gaussian_filter(np.nan_to_num(lr[...,bi]),sig,mode='nearest')}
        conv=uJy_to_electrons(1.0,Config.get_band(names[bi]))
        for i in np.flatnonzero(ok):
            fw=float(t['fwhm'][i]); f2=t[f'flux_{b}_2fwhm_aper'][i]; s2=t[f'fluxerr_{b}_2fwhm_aper'][i]
            if not(np.isfinite(f2) and f2>20*s2 and np.isfinite(fw)): continue
            for n in (1,2,3,4):
                c=t[f'flux_{b}_{n}fwhm_aper'][i]*conv; ce=t[f'fluxerr_{b}_{n}fwhm_aper'][i]*conv
                for m,img in im.items():
                    s=aperture_photometry(img,CircularAperture((x[i],y[i]),r=n*fw/0.1/2),method='exact')['aperture_sum'][0]
                    res[(b,n,m)].append(s/c)
                    if m=='matched': pull[(b,n)].append((s-c)/ce)
for k,v in res.items():
    v=np.array(v); print(k,len(v),'median',round(float(np.median(v)),4),'MAD',round(float(np.median(np.abs(v-np.median(v)))),4), ('pull med/MAD %.2f %.2f'%(np.median(pull[k[:2]]),np.median(np.abs(np.array(pull[k[:2]])-np.median(pull[k[:2]]))))) if k[2]=='matched' else '')
