"""MM-level classical denoising benchmark on the lab's precomputed MM.npz files.

For each acquisition:  input = LQ MM (single shot),  reference = SHQ (or HQ) MM.
Each method denoises the 16 unnormalized MM elements (M = nM * M11), then the maps are
recomputed with the validated NumPy Lu-Chipman and scored against the reference on ONE
shared mask.

Usage
-----
  # 1) tune: sweep each method's parameter on the validation acquisitions
  python scripts/mm_classical_benchmark.py --split val --sweep
  # 2) put the best values in configs/mm_classical.yaml, then evaluate once on test
  python scripts/mm_classical_benchmark.py --split test
  # quick check on one acquisition, centre crop, two methods
  python scripts/mm_classical_benchmark.py --acq 2025-08-13_124640_F_FF_HORAO_2006_004 --methods gaussian median --crop 256

Outputs in results/mm_classical/<timestamp>_<split>/: metrics.csv, summary.csv, figures/*.png, config.yaml
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
from denoising_mueller.data.registration import estimate_shifts
from denoising_mueller.evaluation.metrics import evaluation_mask, map_metrics, mm_metrics
from denoising_mueller.methods.classical import PARAM_GRIDS, denoise
from denoising_mueller.polarimetry import from_matrix, polarimetric_maps, to_matrix
from denoising_mueller.polarimetry.lu_chipman import physical_mask
from denoising_mueller.polarimetry.mm_io import available_references, load_mm_npz


def unnormalized(nM, M11):
    return nM * M11[..., None]


def renormalize(M16):
    M11 = M16[..., 0].copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        nM = M16 / M11[..., None]
    return nM, M11


def crop_center(arr, n):
    if not n:
        return arr
    h, w = arr.shape[:2]
    y0, x0 = max((h - n) // 2, 0), max((w - n) // 2, 0)
    return arr[y0:y0 + n, x0:x0 + n]


def physical_fraction(nM, mask):
    M = to_matrix(nM)
    with np.errstate(divide="ignore", invalid="ignore"):
        M = M / M[..., :1, :1]
    return float(physical_mask(M)[mask].mean())


def load_pair(acq_path, cfg, crop):
    refs = [r for r in cfg["reference_preference"] if r in available_references(acq_path, cfg["wavelength"])]
    if not refs:
        return None
    ref_name = refs[0]
    inp = load_mm_npz(acq_path, "LQ", cfg["wavelength"])
    ref = load_mm_npz(acq_path, ref_name, cfg["wavelength"])
    shift = (0, 0)
    if cfg.get("register_reference", True):
        stack = np.stack([inp["M11"], ref["M11"]])[:, None]           # (2, 1, H, W)
        shift = tuple(int(s) for s in estimate_shifts(stack, reference=0)[1])
        if shift != (0, 0):
            ref["nM"] = np.roll(ref["nM"], shift, axis=(0, 1))
            ref["M11"] = np.roll(ref["M11"], shift, axis=(0, 1))
    for d in (inp, ref):
        d["nM"], d["M11"] = crop_center(d["nM"], crop), crop_center(d["M11"], crop)
    border = max(8, abs(shift[0]) + 2, abs(shift[1]) + 2)
    return inp, ref, ref_name, shift, border


def run_method(M16_in, method, params, mask_fg):
    t = time.time()
    M16_out, sigma = denoise(M16_in, method, mask=mask_fg, **params)
    nM_out, M11_out = renormalize(M16_out)
    maps = polarimetric_maps(nM_out)
    return nM_out, M11_out, maps, time.time() - t


def score(nM, M11, maps, ref_nM, ref_M11, ref_maps, mask):
    row = {}
    row.update(mm_metrics(nM, ref_nM, mask))
    row.update(map_metrics(maps, ref_maps, mask, M11, ref_M11))
    row["physical_frac"] = physical_fraction(nM, mask)
    return row


def figure(path, title, columns, mask):
    import matplotlib.pyplot as plt
    rows = [("totP", "Depolarization", 0, 1, "viridis"), ("linR", "Linear retardance (deg)", 0, 90, "viridis"),
            ("azimuth", "Azimuth (deg)", 0, 180, "twilight"), ("totD", "Diattenuation", 0, 0.3, "viridis")]
    fig, axes = plt.subplots(len(rows), len(columns), figsize=(2.1 * len(columns), 2.0 * len(rows)), squeeze=False)
    for i, (key, label, vmin, vmax, cmap) in enumerate(rows):
        for j, (name, maps) in enumerate(columns):
            ax = axes[i, j]
            img = np.ma.masked_where(~mask, maps[key])
            cm = plt.get_cmap(cmap).copy()
            cm.set_bad("0.85")
            im = ax.imshow(img, vmin=vmin, vmax=vmax, cmap=cm, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(name, fontsize=8)
            if j == 0:
                ax.set_ylabel(label, fontsize=8)
        fig.colorbar(im, ax=axes[i, -1], fraction=0.046, pad=0.04)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/mm_classical.yaml")
    ap.add_argument("--split", choices=["val", "test", "all"], default=None)
    ap.add_argument("--acq", nargs="*", help="acquisition folder names (overrides --split)")
    ap.add_argument("--methods", nargs="*", help="subset of methods (default: all in config)")
    ap.add_argument("--sweep", action="store_true", help="sweep each method's main parameter (PARAM_GRIDS)")
    ap.add_argument("--crop", type=int, default=0, help="centre crop size for quick tests (0 = full image)")
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

    out = Path(paths.get("results_root", "results")) / "mm_classical" / \
        f"{datetime.now():%Y-%m-%d_%H%M%S}_{tag}{'_sweep' if args.sweep else ''}"
    (out / "figures").mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, out / "config.yaml")
    print(f"Writing to {out}")

    rows = []
    for name in names:
        acq = find_acquisition(paths["data_root"], name, cfg["wavelength"])
        pair = load_pair(acq.path, cfg, args.crop)
        if pair is None:
            print(f"[skip] {name}: no reference MM.npz"); continue
        inp, ref, ref_name, shift, border = pair
        ref_maps = polarimetric_maps(ref["nM"])
        mask = evaluation_mask(ref_maps["valid"], ref["M11"], border=border)
        ref_phys = float(ref_maps["valid"].mean())
        print(f"\n{name}  ref={ref_name}  shift={shift}  ref physical={ref_phys:.2f}  eval px={mask.sum()}")
        if ref_phys < cfg["min_reference_physical"]:
            print(f"[skip] reference only {ref_phys:.0%} physical (< {cfg['min_reference_physical']:.0%})"); continue

        base = dict(acq=name, setting=acq.setting, reference=ref_name, shift_dy=shift[0], shift_dx=shift[1],
                    ref_physical=ref_phys, eval_px=int(mask.sum()))
        in_maps = polarimetric_maps(inp["nM"])
        rows.append({**base, "method": "none", "param": "", "value": np.nan, "time_s": 0.0,
                     **score(inp["nM"], inp["M11"], in_maps, ref["nM"], ref["M11"], ref_maps, mask)})
        print(f"   {'none':9s}            nM_RMSE={rows[-1]['nM_RMSE']:.4f}  linR_RMSE={rows[-1]['linR_RMSE']:.2f}  az_MAE={rows[-1]['azimuth_MAE']:.2f}")

        M16_in = unnormalized(inp["nM"], inp["M11"])
        fg = np.isfinite(inp["M11"]) & (inp["M11"] < 250)
        fig_cols = [("input (LQ)", in_maps)]
        for method in methods:
            fixed = dict(cfg["methods"].get(method, {}))
            pname, grid = PARAM_GRIDS[method]
            values = grid if args.sweep else [fixed.get(pname)]
            for v in values:
                params = {**fixed, pname: v} if v is not None else fixed
                nM, M11, maps, dt = run_method(M16_in, method, params, fg)
                r = {**base, "method": method, "param": pname, "value": v, "time_s": dt,
                     **score(nM, M11, maps, ref["nM"], ref["M11"], ref_maps, mask)}
                rows.append(r)
                print(f"   {method:9s} {pname}={v!s:<6} nM_RMSE={r['nM_RMSE']:.4f}  linR_RMSE={r['linR_RMSE']:.2f}  "
                      f"az_MAE={r['azimuth_MAE']:.2f}  phys={r['physical_frac']:.2f}  [{dt:.0f}s]")
                if not args.sweep:
                    fig_cols.append((f"{method} {pname}={v}", maps))
            pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)       # save as we go
        if not args.sweep and not args.no_figures:
            fig_cols.append((f"reference ({ref_name})", ref_maps))
            figure(out / "figures" / f"{name}.png", name, fig_cols, mask)

    df = pd.DataFrame(rows)
    df.to_csv(out / "metrics.csv", index=False)
    if df.empty:
        print("No results."); return
    keys = ["nM_RMSE", "nM_SSIM", "M11_PSNR", "totP_RMSE", "linR_RMSE", "azimuth_MAE", "totD_RMSE", "physical_frac"]
    summary = df.groupby(["method", "param", "value"], dropna=False)[keys].mean().sort_values("nM_RMSE")
    summary.to_csv(out / "summary.csv")
    with pd.option_context("display.width", 200, "display.precision", 4):
        print("\nMean over acquisitions (sorted by nM_RMSE):\n", summary)
    if args.sweep:
        best = df[df.method != "none"].groupby(["method", "param", "value"])["nM_RMSE"].mean().reset_index()
        best = best.loc[best.groupby("method")["nM_RMSE"].idxmin()]
        print("\nBest value per method on this split (copy into configs/mm_classical.yaml):")
        for _, b in best.iterrows():
            print(f"  {b.method}: {b.param} = {b.value}")


if __name__ == "__main__":
    main()
