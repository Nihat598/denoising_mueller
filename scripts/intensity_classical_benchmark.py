"""Intensity-level classical denoising benchmark on the raw repeated frames.

For each acquisition with >= min_frames repeated frames:
  frames are registered onto frame 1 (integer shifts),
  input      = frame 1 (single shot),
  reference  = mean of the second half of the frames (never overlaps the input),
  baselines  = plain average of the first m frames (m = 2, 4, ...), same reference.
Each classical method filters the 16 raw intensities (after a generalized Anscombe transform
with a noise model fitted on the REFERENCE frames only), and is scored in counts.

Mueller matrices / maps from the denoised intensities need the calibration (A, W, Noise) and
are added once those files are available.

Usage
-----
  python scripts/intensity_classical_benchmark.py --split val --sweep
  python scripts/intensity_classical_benchmark.py --split test
  python scripts/intensity_classical_benchmark.py --acq 2026-05-27_152402_F_FF_HORAO_0000_098 --methods gaussian bm3d --crop 256
"""
from __future__ import annotations

import argparse
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from denoising_mueller.data.io import find_acquisition, load_paths
from denoising_mueller.data.registration import register
from denoising_mueller.evaluation.metrics import psnr, rmse, ssim_masked
from denoising_mueller.methods.classical import PARAM_GRIDS, denoise
from denoising_mueller.methods.vst import fit_noise_model, gat_forward, gat_inverse


def intensity_metrics(pred, ref, mask, data_range=255.0):
    """pred, ref: (16, H, W) counts."""
    r, p, s = [], [], []
    for c in range(pred.shape[0]):
        r.append(rmse(pred[c], ref[c], mask))
        p.append(psnr(pred[c], ref[c], mask, data_range))
        s.append(ssim_masked(pred[c], ref[c], mask, data_range))
    # PSNR of the 16-state mean image: noise is already averaged out there, so smoothing mostly
    # costs detail. Reported for completeness; per-state I_PSNR / I_SSIM are the main metrics.
    sum_pred, sum_ref = pred.mean(0), ref.mean(0)
    return {"I_RMSE": float(np.nanmean(r)), "I_PSNR": float(np.nanmean(p)), "I_SSIM": float(np.nanmean(s)),
            "Imean_PSNR": psnr(sum_pred, sum_ref, mask, data_range)}


def crop_center(x, n):
    if not n:
        return x
    h, w = x.shape[-2:]
    y0, x0 = max((h - n) // 2, 0), max((w - n) // 2, 0)
    return x[..., y0:y0 + n, x0:x0 + n]


def figure(path, title, columns, mask, channel=5):
    import matplotlib.pyplot as plt
    ref = columns[-1][1]
    lo, hi = np.percentile(ref.mean(0)[mask], [1, 99])
    clo, chi = np.percentile(ref[channel][mask], [1, 99])
    fig, axes = plt.subplots(2, len(columns), figsize=(2.1 * len(columns), 4.3), squeeze=False)
    for j, (name, I) in enumerate(columns):
        axes[0, j].imshow(I.mean(0), vmin=lo, vmax=hi, cmap="gray")
        axes[1, j].imshow(I[channel], vmin=clo, vmax=chi, cmap="gray")
        axes[0, j].set_title(name, fontsize=8)
        for i in range(2):
            axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
    axes[0, 0].set_ylabel("mean of 16 states", fontsize=8)
    axes[1, 0].set_ylabel(f"state {channel + 1}", fontsize=8)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/intensity_classical.yaml")
    ap.add_argument("--split", choices=["val", "test", "all"], default=None)
    ap.add_argument("--acq", nargs="*")
    ap.add_argument("--methods", nargs="*")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--crop", type=int, default=0)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    paths = load_paths(cfg.get("paths", "configs/paths.yaml"))
    if args.acq:
        names, tag = args.acq, "custom"
    else:
        split = args.split or "val"
        names = cfg["split"]["val"] + cfg["split"]["test"] if split == "all" else cfg["split"][split]
        tag = split
    if args.split == "test" and args.sweep:
        raise SystemExit("Refusing to sweep parameters on the test split: tune on val only.")
    methods = args.methods or list(cfg["methods"])
    sat = cfg.get("saturation", 255)

    out = Path(paths.get("results_root", "results")) / "intensity_classical" / \
        f"{datetime.now():%Y-%m-%d_%H%M%S}_{tag}{'_sweep' if args.sweep else ''}"
    (out / "figures").mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, out / "config.yaml")
    print(f"Writing to {out}")

    rows = []
    for name in names:
        acq = find_acquisition(paths["data_root"], name, cfg["wavelength"])
        if acq.n_frames < cfg["min_frames"]:
            print(f"[skip] {name}: {acq.n_frames} frames < {cfg['min_frames']}"); continue
        frames = acq.load_frames()                                   # (N, 16, H, W)
        shifts = np.zeros((len(frames), 2), int)
        valid = np.ones(frames.shape[-2:], bool)
        if cfg.get("register", True):
            frames, valid, shifts = register(frames)
        frames, valid = crop_center(frames, args.crop), crop_center(valid, args.crop)
        n = len(frames)
        half = n // 2
        inp, ref_frames = frames[0], frames[half:]
        ref = ref_frames.mean(0)
        not_sat = (frames.max(axis=0) < sat).all(axis=0)              # (H, W): no state saturated in any frame
        mask = valid & not_sat
        b = 8
        mask[:b] = mask[-b:] = False
        mask[:, :b] = mask[:, -b:] = False

        a_n, b_n = fit_noise_model(ref_frames, valid & not_sat, sat) if cfg.get("vst", True) else (np.nan, np.nan)
        max_shift = int(np.abs(shifts).max())
        print(f"\n{name}  frames={n} (input 1, reference {half + 1}-{n})  max shift={max_shift}px  "
              f"noise var={a_n:.4f}*I+{b_n:.3f}  eval px={mask.sum()}")
        base = dict(acq=name, setting=acq.setting, n_frames=n, ref_frames=n - half, max_shift=max_shift,
                    noise_a=a_n, noise_b=b_n, eval_px=int(mask.sum()))

        def add(method, pname, value, pred, dt):
            r = {**base, "method": method, "param": pname, "value": value, "time_s": dt,
                 **intensity_metrics(pred, ref, mask)}
            rows.append(r)
            print(f"   {method:9s} {pname}={value!s:<6} I_PSNR={r['I_PSNR']:.2f} dB  I_SSIM={r['I_SSIM']:.3f}  [{dt:.0f}s]")

        add("none", "", np.nan, inp, 0.0)
        for m in cfg.get("averaging_baselines", []):
            if m <= half:
                add(f"average", "frames", m, frames[:m].mean(0), 0.0)

        x = np.moveaxis(inp, 0, -1)                                    # (H, W, 16)
        if cfg.get("vst", True):
            z = gat_forward(x, a_n, b_n)
            sigma = np.ones(x.shape[-1])                               # unit noise by construction
        else:
            z, sigma = x, None
        fig_cols = [("input (frame 1)", inp)]
        for method in methods:
            fixed = dict(cfg["methods"].get(method, {}))
            pname, grid = PARAM_GRIDS[method]
            for v in (grid if args.sweep else [fixed.get(pname)]):
                params = {**fixed, pname: v} if v is not None else fixed
                t = time.time()
                y, _ = denoise(z, method, mask=mask, noise_sigma=sigma, **params)
                if cfg.get("vst", True):
                    y = gat_inverse(y, a_n, b_n)
                pred = np.moveaxis(y, -1, 0)
                add(method, pname, v, pred, time.time() - t)
                if not args.sweep:
                    fig_cols.append((f"{method} {pname}={v}", pred))
            pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
        if not args.sweep and not args.no_figures:
            for m in cfg.get("averaging_baselines", []):
                if m <= half:
                    fig_cols.append((f"average of {m}", frames[:m].mean(0)))
            fig_cols.append((f"reference (mean {half + 1}-{n})", ref))
            figure(out / "figures" / f"{name}.png", name, fig_cols, mask)

    df = pd.DataFrame(rows)
    df.to_csv(out / "metrics.csv", index=False)
    if df.empty:
        print("No results."); return
    summary = (df.groupby(["method", "param", "value"], dropna=False)[["I_PSNR", "I_SSIM", "I_RMSE", "Imean_PSNR", "time_s"]]
                 .mean().sort_values("I_PSNR", ascending=False))
    summary.to_csv(out / "summary.csv")
    with pd.option_context("display.width", 200, "display.precision", 3):
        print("\nMean over acquisitions (sorted by I_PSNR):\n", summary)
    if args.sweep:
        d = df[~df.method.isin(["none", "average"])]
        best = d.groupby(["method", "param", "value"])["I_PSNR"].mean().reset_index()
        best = best.loc[best.groupby("method")["I_PSNR"].idxmax()]
        print("\nBest value per method on this split (copy into configs/intensity_classical.yaml):")
        for _, r in best.iterrows():
            print(f"  {r.method}: {r.param} = {r.value}")


if __name__ == "__main__":
    main()
