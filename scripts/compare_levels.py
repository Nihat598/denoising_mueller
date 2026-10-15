"""Intensity-level vs MM-level classical denoising, on calibrated repeated acquisitions.

For each acquisition (same input frame and same reference for every method and both levels):
  baselines : frame 1 as is ("none"), plain averages of the first 2/4/8 frames
  intensity : filter the 16 raw intensities (Anscombe domain) -> MM -> maps
  mm        : MM of frame 1 -> filter the 16 MM elements -> maps
All scored against the reference (mean of the second half of the frames) on one mask.

Usage
-----
  python scripts/compare_levels.py --split val --sweep          # tune each method at each level
  python scripts/compare_levels.py --split test                 # evaluate once with fixed parameters
  python scripts/compare_levels.py --acq <name> --methods gaussian bm3d --crop 256   # quick look

Paths: configs/paths.yaml must define data_root and calib_root (folder holding the
calibration folders, e.g. .../Calibration/2025-06-23_103720/Processed_images/630nm/A.npy).
Outputs in results/compare_levels/<timestamp>_<split>/: metrics.csv, summary.csv, figures/.
"""
from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from denoising_mueller import pipeline as pl
from denoising_mueller.data.io import find_acquisition, load_paths
from denoising_mueller.methods.classical import PARAM_GRIDS
from denoising_mueller.polarimetry.reconstruction import find_calibration, load_calibration

KEY_METRICS = ["I_PSNR", "nM_RMSE", "nM_RMSE_corr", "nM_SSIM", "totP_RMSE", "linR_RMSE", "azimuth_MAE", "totD_RMSE",
               "physical_frac"]


def figure(path, title, cols, mask):
    import matplotlib.pyplot as plt
    rows = [("totP", "Depolarization", 0, 1, "viridis"), ("linR", "Linear retardance (deg)", 0, 90, "viridis"),
            ("azimuth", "Azimuth (deg)", 0, 180, "twilight"), ("totD", "Diattenuation", 0, 0.3, "viridis")]
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(2.2 * len(cols), 2.0 * len(rows)), squeeze=False)
    for i, (key, label, vmin, vmax, cmap) in enumerate(rows):
        for j, (name, maps) in enumerate(cols):
            cm = plt.get_cmap(cmap).copy(); cm.set_bad("0.85")
            im = axes[i, j].imshow(np.ma.masked_where(~mask, maps[key]), vmin=vmin, vmax=vmax, cmap=cm,
                                   interpolation="nearest")
            axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
            if i == 0:
                axes[i, j].set_title(name, fontsize=8)
            if j == 0:
                axes[i, j].set_ylabel(label, fontsize=8)
        fig.colorbar(im, ax=axes[i, -1], fraction=0.046, pad=0.04)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/compare_levels.yaml")
    ap.add_argument("--split", choices=["val", "test", "all"], default=None)
    ap.add_argument("--acq", nargs="*")
    ap.add_argument("--methods", nargs="*")
    ap.add_argument("--levels", nargs="*", default=["intensity", "mm"], choices=["intensity", "mm"])
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--crop", type=int, default=0)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    paths = load_paths(cfg.get("paths", "configs/paths.yaml"))
    if "calib_root" not in paths:
        raise SystemExit("configs/paths.yaml needs 'calib_root' (folder containing the calibration folders)")
    if args.acq:
        names, tag = args.acq, "custom"
    else:
        split = args.split or "val"
        names = cfg["split"]["val"] + cfg["split"]["test"] if split == "all" else cfg["split"][split]
        tag = split
    if not names:
        raise SystemExit("No acquisitions: fill split.val / split.test in the config, or pass --acq.")
    if args.split == "test" and args.sweep:
        raise SystemExit("Refusing to sweep parameters on the test split: tune on val only.")
    methods = args.methods or list(cfg["methods"])

    out = Path(paths.get("results_root", "results")) / "compare_levels" / \
        f"{datetime.now():%Y-%m-%d_%H%M%S}_{tag}{'_sweep' if args.sweep else ''}"
    (out / "figures").mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, out / "config.yaml")
    print(f"Writing to {out}")

    rows = []
    for name in names:
        acq = find_acquisition(paths["data_root"], name, cfg["wavelength"])
        if acq.n_frames < cfg["min_frames"]:
            print(f"[skip] {name}: {acq.n_frames} frames < {cfg['min_frames']}"); continue
        calib_dir = find_calibration(acq.path, paths["calib_root"])
        calib = load_calibration(calib_dir, cfg["wavelength"])
        if cfg.get("subtract_dark", True) and calib["dark"] is None:
            print(f"[warn] {name}: no {cfg['wavelength']}_Noise.npy in {calib_dir}; continuing without dark subtraction")
        gt = acq.path / "ground_truth.npz"                  # only in synthetic data
        p = pl.prepare(name, acq.load_frames(), calib, register_frames=cfg.get("register", True),
                       subtract_dark=cfg.get("subtract_dark", True), sat=cfg.get("saturation", 255), crop=args.crop,
                       truth_M=np.load(gt)["M"] if gt.exists() else None,
                       physical_tol=cfg.get("physical_tol", "auto"))
        print(f"\n{name}  calib={calib_dir.name}  frames={p.info['n_frames']} (ref {p.info['ref_frames']})  "
              f"shift<={p.info['max_shift']}px  ref physical={p.info['ref_physical']:.2f}  "
              f"noise var={p.noise[0]:.4f}*I+{p.noise[1]:.3f}  eval px={p.info['eval_px']}")
        if p.info["ref_physical"] < cfg.get("min_reference_physical", 0.8):
            print("[skip] reference MM mostly non-physical: check calibration (zoom/focus, date)"); continue
        base = dict(acq=name, setting=acq.setting, calibration=calib_dir.name, **p.info)

        def add(level, method, pname, value, I, mm, maps, dt):
            r = {**base, "level": level, "method": method, "param": pname, "value": value, "time_s": dt,
                 **pl.score(p, I, mm["nM"], mm["M11"], maps)}
            rows.append(r)
            print(f"   {level:9s} {method:9s} {pname}={value!s:<5} I_PSNR={r['I_PSNR']:.2f}  nM_RMSE={r['nM_RMSE']:.4f}  "
                  f"linR_RMSE={r['linR_RMSE']:.2f}  az_MAE={r['azimuth_MAE']:.2f}  totP_RMSE={r['totP_RMSE']:.4f}  "
                  f"phys={r['physical_frac']:.2f}  [{dt:.0f}s]")
            return maps

        in_maps = add("baseline", "none", "", np.nan, *pl.baseline_none(p), 0.0)
        for m in cfg.get("averaging_baselines", []):
            if m <= p.info["n_frames"] // 2:
                add("baseline", "average", "frames", m, *pl.baseline_average(p, m), 0.0)

        fig_maps = {}
        for method in methods:
            pname, grid = PARAM_GRIDS[method]
            for level in args.levels:
                fixed = dict(cfg["methods"].get(method, {}).get(level, {}))
                for v in (grid if args.sweep else [fixed.get(pname)]):
                    params = {**fixed, pname: v} if v is not None else fixed
                    if level == "intensity":
                        I, mm, maps, dt = pl.intensity_path(p, method, params, vst=cfg.get("vst", True))
                    else:
                        I, mm, maps, dt = pl.mm_path(p, method, params)
                    add(level, method, pname, v, I, mm, maps, dt)
                    if not args.sweep:
                        fig_maps[(method, level)] = (f"{method} @ {level} ({pname}={v})", maps)
            pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)

        if not args.sweep and not args.no_figures:
            for method in cfg.get("figure_methods", []):
                if method not in methods:
                    continue
                cols = [("input (frame 1)", in_maps)]
                cols += [fig_maps[(method, lv)] for lv in args.levels if (method, lv) in fig_maps]
                cols.append((f"reference (mean {p.info['ref_frames']})", p.ref_maps))
                figure(out / "figures" / f"{name}__{method}.png", f"{name}: {method}", cols, p.mask)

    df = pd.DataFrame(rows)
    df.to_csv(out / "metrics.csv", index=False)
    if df.empty:
        print("No results."); return
    keys = KEY_METRICS + [c for c in df.columns if c.startswith("truth_")]
    summary = df.groupby(["method", "level", "param", "value"], dropna=False)[keys].mean()
    summary = summary.sort_values("nM_RMSE")
    summary.to_csv(out / "summary.csv")
    with pd.option_context("display.width", 220, "display.precision", 4, "display.max_rows", 200):
        print("\nMean over acquisitions (sorted by nM_RMSE):\n", summary)
    if args.sweep:
        d = df[df.level != "baseline"]
        best = d.groupby(["method", "level", "param", "value"])["nM_RMSE"].mean().reset_index()
        best = best.loc[best.groupby(["method", "level"])["nM_RMSE"].idxmin()]
        print("\nBest value per method and level on this split (copy into configs/compare_levels.yaml):")
        for _, r in best.iterrows():
            print(f"  {r.method:9s} {r.level:9s} {r.param} = {r.value}")


if __name__ == "__main__":
    main()
