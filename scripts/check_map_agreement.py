"""Do the denoised retardance / azimuth maps agree with a low-noise estimate? (one acquisition)

Reference for directions: the 16-frame average, lightly smoothed (Gaussian sigma 1.5 on the Mueller matrix), restricted
to strongly retarding tissue (reference R > 20 deg) where the azimuth is well defined. Also applies each filter to the
16-frame average: if a filter gives the same retardance on 1 frame as on 16 frames, the single-frame result is not
limited by noise. Caveat: the smoothed reference slightly favours smoothing methods.

Usage: python scripts/check_map_agreement.py --config configs/compare_levels_calf.yaml --acq <acquisition>
"""
import argparse
import sys, numpy as np, yaml
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from denoising_mueller import pipeline as pl
from denoising_mueller.data.io import find_acquisition
from denoising_mueller.polarimetry.reconstruction import load_calibration, reconstruct_mm
from denoising_mueller.polarimetry.lu_chipman import polarimetric_maps
from denoising_mueller.methods.classical import denoise
from denoising_mueller.methods.vst import gat_forward, gat_inverse
from denoising_mueller.data.io import load_paths
from denoising_mueller.polarimetry.reconstruction import find_calibration
ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); ap.add_argument('--acq',required=True); args=ap.parse_args()
cfg=yaml.safe_load(open(args.config)); paths=load_paths(cfg.get('paths','configs/paths.yaml')); name=args.acq
acq=find_acquisition(paths['data_root'],name,cfg['wavelength'])
cal=load_calibration(find_calibration(acq.path,paths['calib_root']),cfg['wavelength'])
p=pl.prepare(name, acq.load_frames(), cal)
I16=np.moveaxis(p.frames.mean(0),0,-1)
full=reconstruct_mm(I16,p.A,p.W,p.dark); m16=polarimetric_maps(full['nM'])
tissue=full['M11']>0.1*np.median(full['M11'])
# low-noise reference for direction: 16-frame MM lightly smoothed (sigma 1.5) -> maps
Ms,_=denoise(full['M'],'gaussian',mask=tissue,sigma=1.5); ms=polarimetric_maps(Ms/Ms[...,:1])
strong=tissue & (ms['linR']>20)
print(f'tissue px {tissue.sum()}, strongly retarding px (smoothed 16-frame R>20 deg): {strong.sum()} ({100*strong.sum()/tissue.sum():.0f}% of tissue)')
def azdiff(a,b):
    d=np.abs(a-b)%180; d=np.minimum(d,180-d); return d
def report(lab,mp):
    ok=strong&np.isfinite(mp['azimuth'])&np.isfinite(ms['azimuth'])
    d=azdiff(mp['azimuth'][ok],ms['azimuth'][ok])
    print(f'{lab:28s} retardance median: tissue {np.nanmedian(mp["linR"][tissue]):5.1f}  strong {np.nanmedian(mp["linR"][strong]):5.1f} | azimuth vs ref (strong px): median diff {np.median(d):5.1f} deg, within 15 deg: {100*np.mean(d<15):3.0f}%')
report('16-frame avg (raw)',m16); report('16-frame avg, light smooth',ms)
_,mm,mp=pl.baseline_none(p); report('single frame',mp)
_,mm,mp=pl.baseline_average(p,8); report('average of 8',mp)
a,b=p.noise
for meth,lev in [('gaussian','intensity'),('nlm','intensity'),('tv','intensity'),('bm3d','mm')]:
    prm=dict(cfg['methods'][meth][lev])
    if lev=='intensity': _,mm,mp,_=pl.intensity_path(p,meth,prm,vst=True)
    else: _,mm,mp,_=pl.mm_path(p,meth,prm)
    report(f'{meth} ({lev}) on 1 frame',mp)
    # same filter on the 16-frame average (noise model scaled: variance /16)
    if lev=='intensity':
        z=gat_forward(I16,a/16,b/16); y,_=denoise(z,meth,mask=p.mask,noise_sigma=np.ones(16),**prm); I=gat_inverse(y,a/16,b/16)
        mp2=polarimetric_maps(reconstruct_mm(I,p.A,p.W,p.dark)['nM'])
    else:
        M2,_=denoise(full['M'],meth,mask=p.mask,**prm); mp2=polarimetric_maps(M2/M2[...,:1])
    report(f'{meth} ({lev}) on 16-frame avg',mp2)
