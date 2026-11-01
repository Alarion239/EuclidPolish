import numpy as np
from astropy.table import Table
from astropy.wcs import WCS
from photutils.aperture import EllipticalAperture, aperture_photometry
from euclid_polish.web.helpers import real_tiles as rt
from euclid_polish.photometry import uJy_to_electrons
from euclid_polish.config import Config
t = Table.read('data/jwst_euclid_overlap/flux_conservation/mer_nexus.fits')
print('pa range', np.nanpercentile(np.asarray(t['position_angle'],float),[1,50,99]), 'kron', np.nanpercentile(np.asarray(t['kron_radius'],float),[5,50,95]),'ell',np.nanpercentile(np.asarray(t['ellipticity'],float),[5,50,95]))
conv=uJy_to_electrons(1.0,Config.get_band('VIS'))
entries=[e for e in rt.list_entries('nexus') if e.extras.get('jwst_file')][::12]
out={k:[] for k in ('pa','pa+90','-pa','90-pa')}; pulls={k:[] for k in out}; seg=[]
for e in entries:
    lr=np.asarray(rt.get_tile('nexus',e.id,entry=e).lr_e,np.float64)[...,0]; w=WCS(rt.lr_header(e)).celestial
    x,y=w.world_to_pixel_values(np.asarray(t['right_ascension']),np.asarray(t['declination']))
    ok=(x>45)&(x<lr.shape[1]-46)&(y>45)&(y<lr.shape[0]-46)
    for i in np.flatnonzero(ok):
        f=t['flux_detection_total'][i]; s=t['fluxerr_detection_total'][i]; a=float(t['kron_radius'][i]); el=float(t['ellipticity'][i]); pa=float(t['position_angle'][i])
        if not(np.isfinite(f) and f>20*s and np.isfinite(a) and a>0 and a<40 and t['vis_det'][i]==1): continue
        for k,ang in (('pa',pa),('pa+90',pa+90),('-pa',-pa),('90-pa',90-pa)):
            m=aperture_photometry(lr,EllipticalAperture((x[i],y[i]),a,a*(1-el),np.deg2rad(ang)),method='exact')['aperture_sum'][0]
            out[k].append(m/(f*conv)); pulls[k].append((m-f*conv)/(s*conv))
for k,v in out.items():
    v=np.array(v); p=np.array(pulls[k]); print(k,len(v),'ratio med %.4f MAD %.4f  pull med %.2f MAD %.2f'%(np.median(v),np.median(np.abs(v-np.median(v))),np.median(p),np.median(np.abs(p-np.median(p)))))
