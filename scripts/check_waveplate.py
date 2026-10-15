"""Check calibration + MM reconstruction on a quarter-wave plate (or any uniform linear retarder).

A good QWP measurement should give, inside the plate:
  linear retardance ~ nominal (90 deg at its design wavelength), depolarization ~ 0,
  diattenuation ~ 0, uniform azimuth, and a Mueller matrix close to the ideal retarder.

The MM is computed from the mean of ALL frames (dark-subtracted). The plate region is the
central --roi fraction of the image unless --roi-box y0 y1 x0 x1 is given.

Usage
-----
  python scripts/check_waveplate.py --acq <qwp acquisition folder name> [--retardance 90] [--roi 0.5]
Outputs results/waveplate/<acq>/: report.txt, maps.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from denoising_mueller.data.io import find_acquisition, load_paths
from denoising_mueller.data.registration import register
from denoising_mueller.polarimetry.lu_chipman import normalize, physical_mask, polarimetric_maps, to_matrix
from denoising_mueller.polarimetry.reconstruction import find_calibration, load_calibration, reconstruct_mm


def ideal_retarder(delta_deg, theta_deg):
    d, t = np.radians(delta_deg), np.radians(theta_deg)
    c2, s2, cd, sd = np.cos(2 * t), np.sin(2 * t), np.cos(d), np.sin(d)
    return np.array([[1, 0, 0, 0],
                     [0, c2 ** 2 + s2 ** 2 * cd, c2 * s2 * (1 - cd), -s2 * sd],
                     [0, c2 * s2 * (1 - cd), s2 ** 2 + c2 ** 2 * cd, c2 * sd],
                     [0, s2 * sd, -c2 * sd, cd]])


def axial_mean(deg):
    """Mean of an axial angle (period 180 deg) and its circular spread (deg)."""
    z = np.exp(2j * np.radians(deg))
    m = z.mean()
    return float(np.degrees(np.angle(m)) / 2 % 180), float(np.degrees(np.sqrt(-2 * np.log(abs(m)))) / 2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--acq", required=True)
    ap.add_argument("--config", default="configs/paths.yaml")
    ap.add_argument("--wavelength", type=int, default=630)
    ap.add_argument("--retardance", type=float, default=90.0, help="nominal retardance (deg)")
    ap.add_argument("--roi", type=float, default=0.5, help="central fraction of the image used as the plate")
    ap.add_argument("--roi-box", type=int, nargs=4, metavar=("Y0", "Y1", "X0", "X1"))
    ap.add_argument("--no-dark", action="store_true", help="do not subtract <wl>_Noise.npy")
    args = ap.parse_args()

    paths = load_paths(args.config)
    acq = find_acquisition(paths["data_root"], args.acq, args.wavelength)
    calib_dir = find_calibration(acq.path, paths["calib_root"])
    cal = load_calibration(calib_dir, args.wavelength)
    frames = acq.load_frames()
    frames, valid, shifts = register(frames)
    I = np.moveaxis(frames.mean(0), 0, -1)
    mm = reconstruct_mm(I, cal["A"], cal["W"], None if args.no_dark else cal["dark"])
    maps = polarimetric_maps(mm["nM"])

    H, W = I.shape[:2]
    if args.roi_box:
        y0, y1, x0, x1 = args.roi_box
    else:
        fy, fx = int(H * (1 - args.roi) / 2), int(W * (1 - args.roi) / 2)
        y0, y1, x0, x1 = fy, H - fy, fx, W - fx
    roi = np.zeros((H, W), bool); roi[y0:y1, x0:x1] = True
    roi &= valid & (frames.max(axis=(0, 1)) < 255)

    linR, totP, totD, az = (maps[k][roi] for k in ("linR", "totP", "totD", "azimuth"))
    az_mean, az_spread = axial_mean(az)
    M = normalize(to_matrix(mm["nM"]))[roi]
    M_ideal = ideal_retarder(args.retardance, az_mean)
    # the lab's azimuth convention may differ from the textbook one: fit the ideal matrix's angle directly
    thetas = np.arange(0, 180, 0.25)
    errs = [np.sqrt(np.mean((M - ideal_retarder(args.retardance, t)) ** 2)) for t in thetas]
    t_best = float(thetas[int(np.argmin(errs))])
    rms_best = float(min(errs))

    lines = [
        f"Acquisition: {acq.name}",
        f"Calibration: {calib_dir}  (dark subtracted: {not args.no_dark and cal['dark'] is not None})",
        f"Frames: {acq.n_frames} (max registration shift {int(np.abs(shifts).max())} px)",
        f"ROI: y {y0}-{y1}, x {x0}-{x1}  ({roi.sum()} px)",
        "",
        f"Linear retardance : median {np.median(linR):6.2f} deg  (IQR {np.percentile(linR, 25):.2f}-{np.percentile(linR, 75):.2f}),"
        f"  nominal {args.retardance:.1f}",
        f"Depolarization    : median {np.median(totP):6.4f}  (ideal 0)",
        f"Diattenuation     : median {np.median(totD):6.4f}  (ideal 0)",
        f"Azimuth (maps)    : mean {az_mean:6.2f} deg, circular spread {az_spread:.2f} deg",
        f"Physical pixels   : {physical_mask(normalize(to_matrix(mm['nM'])), tol=0.01)[roi].mean():.3f}"
        f"  (Cloude eigenvalues >= -0.01; a pure retarder sits on the boundary)",
        "",
        f"Element-wise RMS vs ideal retarder ({args.retardance:.0f} deg) at best-fit axis {t_best:.2f} deg: {rms_best:.4f}",
        f"(at the map azimuth {az_mean:.2f} deg: {np.sqrt(np.mean((M - M_ideal) ** 2)):.4f};"
        " a large difference only means the two angle conventions differ)",
        "",
        "Mean normalized Mueller matrix in the ROI (textbook orientation):",
        np.array2string(M.mean(0), precision=3, suppress_small=True),
    ]
    gt = acq.path / "waveplate.json"
    if gt.exists():
        g = json.loads(gt.read_text())
        lines += ["", f"Synthetic ground truth: retardance {g['retardance_deg']} deg, azimuth {g['azimuth_deg']} deg"]
    report = "\n".join(lines)
    print(report)

    out = Path(paths.get("results_root", "results")) / "waveplate" / acq.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.txt").write_text(report)

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2))
    for ax, (k, vmin, vmax, cmap) in zip(axes, [("linR", 0, 180, "viridis"), ("azimuth", 0, 180, "twilight"),
                                               ("totP", 0, 0.3, "viridis"), ("totD", 0, 0.3, "viridis")]):
        im = ax.imshow(maps[k], vmin=vmin, vmax=vmax, cmap=cmap)
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="w", lw=1))
        ax.set_title(k, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(acq.name, fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "maps.png", dpi=130)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
