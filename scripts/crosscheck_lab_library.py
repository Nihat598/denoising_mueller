"""Cross-check our NumPy reconstruction + Lu-Chipman against the lab's compiled C library.

Result on 2026-10-15 (Linux build of fastMMPT-full-display/src/mmpt/C-libs): all differences
<= 3e-13 (M, nM, totD, linR, totP, azimuth). Rerun after any change to polarimetry/.

Usage:
  # 1) compile the lab library once (Linux / Spark):
  #    cd <fastMMPT>/src/mmpt/C-libs && for i in *.c; do gcc -c -O2 -fpic -fopenmp $i; done \
  #    && gcc -fopenmp -shared -o libmpMuelMat.so *.o
  # 2) python scripts/crosscheck_lab_library.py --mmpt <fastMMPT>/src/mmpt --lib <.../libmpMuelMat.so>
  (needs: opencv-python-headless, tqdm for the lab module's imports)
"""
import argparse, importlib.util, numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("--mmpt", required=True, help="folder containing the lab's libmpMuelMat.py")
ap.add_argument("--lib", required=True, help="compiled libmpMuelMat.so / .dll")
args = ap.parse_args()
spec = importlib.util.spec_from_file_location("libmp", f"{args.mmpt}/libmpMuelMat.py")
lab = importlib.util.module_from_spec(spec); spec.loader.exec_module(lab)
lab._get_CLib_path = lambda: args.lib

from denoising_mueller.polarimetry.reconstruction import reconstruct_mm, intensities_from_mm, mat_to_vec
from denoising_mueller.polarimetry.lu_chipman import polarimetric_maps

rng = np.random.default_rng(0)
H, Wd = 64, 80
def retarder(delta, theta):
    c2, s2, cd, sd = np.cos(2*theta), np.sin(2*theta), np.cos(delta), np.sin(delta)
    R = np.zeros(delta.shape + (4, 4)); R[..., 0, 0] = 1
    R[..., 1, 1] = c2**2 + s2**2*cd; R[..., 1, 2] = c2*s2*(1-cd); R[..., 1, 3] = -s2*sd
    R[..., 2, 1] = c2*s2*(1-cd); R[..., 2, 2] = s2**2 + c2**2*cd; R[..., 2, 3] = c2*sd
    R[..., 3, 1] = s2*sd; R[..., 3, 2] = -c2*sd; R[..., 3, 3] = cd
    return R
yy, xx = np.mgrid[0:H, 0:Wd] / 64
delta = np.radians(10 + 70*xx); theta = np.radians(180*yy) % np.pi
p = 0.3 + 0.6*yy                                     # depolarization power varies
Dm = np.zeros((H, Wd, 4, 4)); Dm[..., 0, 0] = 1
for k in (1, 2, 3): Dm[..., k, k] = 1 - p
d = 0.08*xx                                          # weak linear diattenuation along x
MD = np.zeros((H, Wd, 4, 4)); MD[..., 0, 0] = 1; MD[..., 0, 1] = MD[..., 1, 0] = d
MD[..., 1, 1] = 1; s = np.sqrt(1-d**2); MD[..., 2, 2] = MD[..., 3, 3] = s
Mtrue = 120 * (Dm @ retarder(delta, theta) @ MD)     # textbook matrices, m11 = 120 counts
Mvec = mat_to_vec(Mtrue)

# realistic-ish per-pixel calibration: fixed PSG/PSA + small spatial variation
base_A = np.array([[1, .9, .3, .2], [1, -.4, .8, .3], [1, -.3, -.6, .7], [1, .2, -.4, -.85]])
base_W = np.array([[1, 1, 1, 1], [.85, -.45, -.35, .25], [.35, .75, -.65, -.4], [.25, .3, .55, -.8]])
A = mat_to_vec(base_A + 0.03*rng.standard_normal((H, Wd, 4, 4)))
W = mat_to_vec(base_W + 0.03*rng.standard_normal((H, Wd, 4, 4)))
I = intensities_from_mm(Mvec, A, W)

ours = reconstruct_mm(I, A, W)
M_c, nM_c = lab.compute_MM_AIW(A.copy(), I.copy(), W.copy())
print("max |M_ours - M_true|     :", np.abs(ours["M"] - Mvec).max())
print("max |M_ours - M_labC|     :", np.abs(ours["M"] - M_c).max())
print("max |nM_ours - nM_labC|   :", np.abs(ours["nM"] - nM_c).max())

maps = polarimetric_maps(ours["nM"])
MDc, MRc, Mdc = lab.compute_MM_polar_LuChipman(nM_c.copy())
P = lab.compute_MM_polarim_Params(MDc, MRc, Mdc)
for k in ("totD", "linR", "totP"):
    print(f"max |{k} ours - labC| :", np.nanmax(np.abs(maps[k] - P[k])))
daz = np.abs((maps["azimuth"] - P["azimuth"] + 90) % 180 - 90)
print("max |azimuth ours - labC| (deg, axial):", np.nanmax(daz))
print("truth check: linR", maps["linR"].min().round(2), "-", maps["linR"].max().round(2), "deg (true 10-96);",
      "totP", maps["totP"].min().round(3), "-", maps["totP"].max().round(3), "(true 0.30-0.89)")
