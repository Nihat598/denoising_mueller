"""Create synthetic IMPv2-like acquisitions (lab folder layout) with known ground truth.

Writes, under --out:
  data/<date>_F_FF_HORAO_9999_001/To_Process/630_Image_Number_<k>.npy   tissue-like sample
  data/<date>_F_FF_HORAO_9999_002/To_Process/630_Image_Number_<k>.npy   quarter-wave plate
  data/<acq>/MetaData.json  (names the calibration folder), data/<acq>/ground_truth.npz
  calibration/<date>/Processed_images/630nm/{A.npy, W.npy, 630_Noise.npy}

Noise: Poisson-Gaussian, var = a*I + b on raw counts (default: the 0000_098 fit), dark offset,
8-bit rounding and saturation at 255. Use it to test the pipeline end to end:

  python scripts/make_synthetic_data.py --out synthetic
  # paths.yaml: data_root: synthetic/data, calib_root: synthetic/calibration
  python scripts/compare_levels.py --acq 2026-01-01_120000_F_FF_HORAO_9999_001 --methods gaussian --crop 256
  python scripts/check_waveplate.py --acq 2026-01-01_120500_F_FF_HORAO_9999_002
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from denoising_mueller.polarimetry.reconstruction import intensities_from_mm, mat_to_vec

CALIB = "2026-01-01_115500"
TISSUE = "2026-01-01_120000_F_FF_HORAO_9999_001"
QWP = "2026-01-01_120500_F_FF_HORAO_9999_002"


def linear_retarder(delta_rad, theta_rad):
    c2, s2, cd, sd = np.cos(2 * theta_rad), np.sin(2 * theta_rad), np.cos(delta_rad), np.sin(delta_rad)
    R = np.zeros(np.shape(delta_rad) + (4, 4))
    R[..., 0, 0] = 1
    R[..., 1, 1] = c2 ** 2 + s2 ** 2 * cd; R[..., 1, 2] = c2 * s2 * (1 - cd); R[..., 1, 3] = -s2 * sd
    R[..., 2, 1] = c2 * s2 * (1 - cd); R[..., 2, 2] = s2 ** 2 + c2 ** 2 * cd; R[..., 2, 3] = c2 * sd
    R[..., 3, 1] = s2 * sd; R[..., 3, 2] = -c2 * sd; R[..., 3, 3] = cd
    return R


def calibration(rng, H, W):
    base_A = np.array([[1, .9, .3, .2], [1, -.4, .8, .3], [1, -.3, -.6, .7], [1, .2, -.4, -.85]])
    base_W = np.array([[1, 1, 1, 1], [.85, -.45, -.35, .25], [.35, .75, -.65, -.4], [.25, .3, .55, -.8]])
    smooth = lambda: gaussian_filter(rng.standard_normal((H, W, 4, 4)), (40, 40, 0, 0)) * 3
    A = mat_to_vec((base_A + 0.03 * smooth()) * 0.5)
    W = mat_to_vec((base_W + 0.03 * smooth()) * 0.5)
    return A, W


def tissue_mm(rng, H, W, level):
    sm = lambda s: gaussian_filter(rng.standard_normal((H, W)), s)
    norm = lambda x: (x - x.min()) / (x.max() - x.min())
    delta = np.radians(5 + 60 * norm(sm(25)))                 # linear retardance 5-65 deg
    theta = np.pi * norm(sm(40))                               # azimuth field
    p = 0.15 + 0.7 * norm(sm(15))                              # depolarization 0.15-0.85
    D = np.zeros((H, W, 4, 4)); D[..., 0, 0] = 1
    for k in (1, 2, 3):
        D[..., k, k] = 1 - p
    m11 = level * (0.5 + norm(sm(8)))                         # intensity texture
    return m11[..., None, None] * (D @ linear_retarder(delta, theta))


def acquire(rng, M_true, A, Wc, dark, n, a, b):
    I_clean = intensities_from_mm(mat_to_vec(M_true), A, Wc) + dark   # expected raw counts
    frames = []
    for _ in range(n):
        y = a * rng.poisson(np.maximum(I_clean, 0) / a) + rng.normal(0, np.sqrt(b), I_clean.shape)
        frames.append(np.clip(np.round(y), 0, 255).astype(np.uint8))
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="synthetic")
    ap.add_argument("--size", type=int, nargs=2, default=[256, 304])
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--level", type=float, default=40.0, help="typical m11 in counts")
    ap.add_argument("--noise", type=float, nargs=2, default=[0.034, 0.18], metavar=("A", "B"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    H, W = args.size
    out = Path(args.out)

    A, Wc = calibration(rng, H, W)
    dark = np.full((H, W, 16), 2.0)
    cdir = out / "calibration" / CALIB / "Processed_images" / "630nm"
    cdir.mkdir(parents=True, exist_ok=True)
    np.save(cdir / "A.npy", A); np.save(cdir / "W.npy", Wc); np.save(cdir / "630_Noise.npy", dark)

    qwp_delta, qwp_theta = 90.0, 30.0
    M_qwp = args.level * 1.5 * linear_retarder(np.full((H, W), np.radians(qwp_delta)),
                                               np.full((H, W), np.radians(qwp_theta)))
    for name, M in ((TISSUE, tissue_mm(rng, H, W, args.level)), (QWP, M_qwp)):
        d = out / "data" / name / "To_Process"
        d.mkdir(parents=True, exist_ok=True)
        for k, f in enumerate(acquire(rng, M, A, Wc, dark, args.frames, *args.noise), start=1):
            np.save(d / f"630_Image_Number_{k}.npy", f)
        (out / "data" / name / "MetaData.json").write_text(json.dumps(
            {"Calibration Folder": f"C:\\synthetic\\Calibration\\{CALIB}\\Processed_images\\630nm",
             "Exposure_time_ms": 1.0, "synthetic": True}))
        np.savez_compressed(out / "data" / name / "ground_truth.npz", M=mat_to_vec(M))
    (out / "data" / QWP / "waveplate.json").write_text(json.dumps({"retardance_deg": qwp_delta,
                                                                    "azimuth_deg": qwp_theta}))
    print(f"Wrote {out}/data/{{{TISSUE},{QWP}}} and {out}/calibration/{CALIB}")


if __name__ == "__main__":
    main()
