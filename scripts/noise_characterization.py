"""Noise characterization of one acquisition: variance vs. mean after registration.

Usage: python scripts/noise_characterization.py --acq <folder name> [--config configs/paths.yaml]
Writes results/noise/<acq>_shifts.csv, _var_vs_mean.png, _fit.txt
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from denoising_mueller.data.io import find_acquisition, load_paths
from denoising_mueller.data.registration import register


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acq", required=True)
    ap.add_argument("--config", default="configs/paths.yaml")
    args = ap.parse_args()
    cfg = load_paths(args.config)

    acq = find_acquisition(cfg["data_root"], args.acq, cfg.get("wavelength", 630))
    print(f"Found {acq.path} ({acq.n_frames} frames)")
    if acq.n_frames < 2:
        raise SystemExit(f"{acq.name}: needs at least 2 frames, found {acq.n_frames}")
    raw = acq.load_frames()                       # (N, 16, H, W)
    frames, valid, shifts = register(raw)
    print("Shifts (dy, dx) per frame:\n", np.round(shifts, 2))

    # Pixels usable for statistics: inside the FOV for all frames, never saturated
    ok = valid[None] & (raw.max(axis=0) < 255)    # (16, H, W)
    mean = frames.mean(axis=0)[ok]
    var = frames.var(axis=0, ddof=1)[ok]

    # Robust fit: median variance in quantile bins of the mean, then a line through the
    # bin medians. A least-squares fit on all pixels is pulled up by edges and residual motion.
    edges = np.unique(np.round(np.percentile(mean, np.linspace(0.5, 99.5, 41)), 1))
    xs, ys, ns = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (mean >= lo) & (mean < hi)
        if sel.sum() >= 2000:
            xs.append(np.median(mean[sel])); ys.append(np.median(var[sel])); ns.append(int(sel.sum()))
    xs, ys = np.array(xs), np.array(ys)
    a, b = np.polyfit(xs, ys, 1)
    print(f"Robust fit: variance = {a:.4f} * mean + {b:.3f}  (N pixels = {mean.size}, {len(xs)} bins)")

    out = Path(cfg.get("results_root", "results")) / "noise"
    out.mkdir(parents=True, exist_ok=True)
    np.savetxt(out / f"{acq.name}_shifts.csv", shifts, delimiter=",", header="dy,dx", comments="")
    (out / f"{acq.name}_fit.txt").write_text(f"a={a}\nb={b}\nn_frames={acq.n_frames}\nfit=binned_median\n")
    np.savetxt(out / f"{acq.name}_bins.csv", np.c_[xs, ys, ns], delimiter=",",
               header="mean_median,var_median,n_pixels", comments="")

    fig, ax = plt.subplots(figsize=(5, 4))
    xmax = float(np.percentile(mean, 99.5)) * 1.05
    ymax = float(np.percentile(var, 99)) * 1.05
    ax.hist2d(mean, var, bins=[120, 120], range=[[0, xmax], [0, ymax]], cmap="Greys", cmin=1)
    ax.plot(xs, ys, "o", ms=3.5, color="C0", label="median per bin")
    x = np.linspace(0, xmax, 100)
    ax.plot(x, a * x + b, color="C3", lw=1.5, label=f"var = {a:.3f}·mean + {b:.2f}")
    ax.set_xlim(0, xmax); ax.set_ylim(0, ymax)
    ax.set_xlabel("Per-pixel mean (counts)")
    ax.set_ylabel("Per-pixel variance (counts²)")
    ax.set_title(acq.name, fontsize=8)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / f"{acq.name}_var_vs_mean.png", dpi=200)
    print(f"Saved results to {out}")


if __name__ == "__main__":
    main()
