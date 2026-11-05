"""Conference / paper figures from a finished test run of scripts/compare_levels.py.

Figures (PNG at 300 dpi + vector PDF) written to --out:
  fig1_maps.(png|pdf)          diagnostic maps of one acquisition: single frame, 8-frame average, each
                               selected filter at its better level, and the 16-frame average (best estimate)
  fig2_scores.(png|pdf)        error per filter, intensity vs MM level, per group, with averaging baselines
  fig3_quality_vs_time.(...)   error vs run time (frame -> maps), one point per filter and level
  fig4_noise_bias.(...)        how noise shifts the tissue medians of depolarization and retardance

Usage
-----
  python scripts/make_paper_figures.py --config configs/compare_levels_calf.yaml \
      --metrics results/compare_levels/<run>_test/metrics.csv \
      --acq 2026-11-03_152714_F_FF_HORAO_0000_012 [--inset y0 y1 x0 x1] [--methods nlm bm3d wavelet tv]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                                   # noqa: E402
import numpy as np                                                                # noqa: E402
import pandas as pd                                                               # noqa: E402
import yaml                                                                       # noqa: E402
from matplotlib.patches import Rectangle                                          # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from denoising_mueller import pipeline as pl                                      # noqa: E402
from denoising_mueller.data.io import find_acquisition, load_paths               # noqa: E402
from denoising_mueller.polarimetry.lu_chipman import polarimetric_maps           # noqa: E402
from denoising_mueller.polarimetry.reconstruction import (find_calibration,      # noqa: E402
                                                          load_calibration, reconstruct_mm)

# categorical slots 1-2 of the reference palette (validated); baselines in neutral ink
C_INT, C_MM = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#1f1f1e", "#5f5e58", "#d9d8d2"
NAMES = {"median": "Median", "gaussian": "Gaussian", "nlm": "NLM", "wavelet": "Wavelet", "tv": "TV",
         "bm3d": "BM3D"}
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 300, "font.family": "DejaVu Sans"})


def save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}.png / .pdf")


def best_levels(m):
    d = m[m.level != "baseline"].groupby(["method", "level"])["nM_RMSE_corr"].median().unstack()
    return {meth: d.loc[meth].idxmin() for meth in d.index}


# ---------------------------------------------------------------- fig 2 and 3: from metrics.csv
def fig_scores(m, groups, out, score="nM_RMSE_corr", agg="median"):
    methods = [k for k in NAMES if k in set(m.method)]
    fig, axes = plt.subplots(1, len(groups), figsize=(max(5.4, 4.6 * len(groups)), 3.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (gname, acqs) in zip(axes, groups.items()):
        g = m[m.acq.isin(acqs)]
        x = np.arange(len(methods))
        for off, lev, col, lab in ((-0.19, "intensity", C_INT, "intensity level"), (0.19, "mm", C_MM, "MM level")):
            v = [g[(g.method == k) & (g.level == lev)][score].agg(agg) for k in methods]
            ax.bar(x + off, v, width=0.36, color=col, label=lab, edgecolor="white", linewidth=1)
        base = g[g.level == "baseline"]
        for (meth, val), style in (((("none", np.nan)), (1.4, "-")), ((("average", 2.0)), (1.0, ":")),
                                   ((("average", 4.0)), (1.0, "--")), ((("average", 8.0)), (1.4, "--"))):
            sel = base[(base.method == meth) & ((base.value == val) | (np.isnan(val) & base.value.isna()))]
            if sel.empty:
                continue
            y = sel[score].agg(agg)
            ax.axhline(y, color=INK2, lw=style[0], ls=style[1], zorder=0)
            lab = "single frame" if meth == "none" else f"average of {int(val)}"
            ax.text(1.01, y, lab, transform=ax.get_yaxis_transform(), color=INK2, fontsize=7, va="center")
        ax.set_xticks(x, [NAMES[k] for k in methods], fontsize=8)
        ax.set_title(f"{gname} (n={g.acq.nunique()})", fontsize=9, color=INK)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel(f"Mueller-matrix error ({score}, {agg} over acq.)")
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=8, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.05, 1, 1), w_pad=4)
    save(fig, out, "fig2_scores")


def fig_quality_time(m, out, score="nM_RMSE_corr", agg="median"):
    d = m[m.level != "baseline"].groupby(["method", "level"])[[score, "time_pipeline_s"]].agg(agg)
    d = d.rename(columns={score: "nM_RMSE_corr"})
    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    for (meth, lev), r in d.iterrows():
        col = C_INT if lev == "intensity" else C_MM
        ax.scatter(r.time_pipeline_s, r.nM_RMSE_corr, s=46, color=col, edgecolor="white", linewidth=1.5, zorder=3,
                   marker="o" if lev == "intensity" else "s")
        off, ha = ((6, 3), "left") if lev == "intensity" else ((6, -9), "left")
        ax.annotate(NAMES.get(meth, meth), (r.time_pipeline_s, r.nM_RMSE_corr), xytext=off, ha=ha,
                    textcoords="offset points", fontsize=7, color=INK)
    base = m[m.level == "baseline"].groupby(["method", "value"], dropna=False)[score].agg(agg)
    for (meth, val), y in base.items():
        lab = "single frame" if meth == "none" else f"average of {int(val)}"
        ax.axhline(y, color=INK2, lw=0.8, ls="--" if meth == "average" else "-", zorder=0)
        ax.text(1.02, y, lab, transform=ax.get_yaxis_transform(), fontsize=7,
                color=INK2, va="center")
    ax.set_xscale("log")
    ax.set_xlabel("run time, one frame to maps (s)")
    ax.set_ylabel(f"Mueller-matrix error ({score}, {agg})")
    ax.scatter([], [], color=C_INT, marker="o", label="intensity level")
    ax.scatter([], [], color=C_MM, marker="s", label="MM level")
    ax.legend(frameon=False, fontsize=8, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    save(fig, out, "fig3_quality_vs_time")


# ---------------------------------------------------------------- fig 1 and 4: recompute maps
def masked(maps, M11, thr, min_linR=5.0):
    out = {k: np.where(M11 > thr, maps[k], np.nan) for k in ("totP", "linR", "azimuth", "totD")}
    out["azimuth"] = np.where(out["linR"] >= min_linR, out["azimuth"], np.nan)
    return out


def fig_maps(cols, inset, out, title):
    rows = [("totP", "Depolarization", "viridis", (0.6, 1.0)), ("linR", "Linear retardance (°)", "viridis", (0, 45)),
            ("azimuth", "Azimuth (°)", "twilight", (0, 180)), ("totD", "Diattenuation", "viridis", (0, 0.2))]
    n = len(cols)
    fig, axes = plt.subplots(len(rows), n, figsize=(1.75 * n + 0.6, 1.55 * len(rows) + 0.4), squeeze=False)
    cmaps = {}
    for i, (key, lab, cmap, (lo, hi)) in enumerate(rows):
        cm = plt.get_cmap(cmap).copy()
        cm.set_bad("#e6e5e0")
        cmaps[key] = cm
        for j, (name, mp) in enumerate(cols):
            ax = axes[i, j]
            im = ax.imshow(mp[key], cmap=cm, vmin=lo, vmax=hi, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if i == 0:
                ax.set_title(name, fontsize=8, color=INK)
            if j == 0:
                ax.set_ylabel(lab, fontsize=8, color=INK)
            if inset is not None and i == 0:
                y0, y1, x0, x1 = inset
                ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="white", lw=0.8))
        cb = fig.colorbar(im, ax=axes[i, :].tolist(), fraction=0.012, pad=0.01)
        cb.ax.tick_params(labelsize=7)
    fig.suptitle(title, fontsize=9, color=INK)
    save(fig, out, "fig1_maps")
    if inset is not None:
        y0, y1, x0, x1 = inset
        fig, axes = plt.subplots(2, n, figsize=(1.75 * n + 0.6, 3.6), squeeze=False)
        for i, (key, lab, cmap, (lo, hi)) in enumerate(rows[:2]):
            for j, (name, mp) in enumerate(cols):
                ax = axes[i, j]
                im = ax.imshow(mp[key][y0:y1, x0:x1], cmap=cmaps[key], vmin=lo, vmax=hi, interpolation="nearest")
                ax.set_xticks([])
                ax.set_yticks([])
                if i == 0:
                    ax.set_title(name, fontsize=8, color=INK)
                if j == 0:
                    ax.set_ylabel(lab, fontsize=8, color=INK)
            cb = fig.colorbar(im, ax=axes[i, :].tolist(), fraction=0.012, pad=0.01)
            cb.ax.tick_params(labelsize=7)
        fig.suptitle(title + " (zoom)", fontsize=9, color=INK)
        save(fig, out, "fig1b_maps_zoom")


def fig_bias(stats, out, title):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 0.28 * len(stats) + 1.0), sharey=True)
    names = [s[0] for s in stats]
    y = np.arange(len(names))[::-1]
    best = dict(stats)[names[-1]]
    for ax, key, lab in ((axes[0], "totP", "median depolarization (tissue)"),
                         (axes[1], "linR", "median linear retardance (°, tissue)")):
        v = [s[1][key] for s in stats]
        ax.axvline(best[key], color=INK2, lw=1, ls="--", zorder=0)
        cols = [C_INT if "(int.)" in n else C_MM if "(MM)" in n else INK2 for n in names]
        ax.scatter(v, y, s=40, color=cols,
                   edgecolor="white", linewidth=1.5, zorder=3)
        ax.set_xlabel(lab)
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_yticks(y, names)
    fig.suptitle(title + "  (dashed: 16-frame average; blue = intensity level, orange = MM level)",
                 fontsize=9, color=INK)
    fig.tight_layout()
    save(fig, out, "fig4_noise_bias")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--acq", required=True, help="acquisition for the map figures")
    ap.add_argument("--methods", nargs="*", default=["nlm", "bm3d", "wavelet", "tv"])
    ap.add_argument("--inset", nargs=4, type=int, default=None, metavar=("Y0", "Y1", "X0", "X1"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--score", default="nM_RMSE_corr", help="metrics column for figs 2-3")
    ap.add_argument("--agg", default="median", choices=["median", "mean"],
                    help="aggregate over acquisitions (median is robust to single-acquisition blow-ups)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    paths = load_paths(cfg.get("paths", "configs/paths.yaml"))
    m = pd.read_csv(args.metrics)
    out = Path(args.out or Path(args.metrics).parent / "paper_figures")
    out.mkdir(parents=True, exist_ok=True)
    print(f"Writing to {out}")
    groups = cfg.get("groups") or {"test": sorted(m.acq.unique())}
    best = best_levels(m)
    print("better level per method (mean over test):", best)

    fig_scores(m, groups, out, args.score, args.agg)
    fig_quality_time(m, out, args.score, args.agg)
    if os.environ.get("FIGS_ONLY_SCORES"):
        return

    acq = find_acquisition(paths["data_root"], args.acq, cfg["wavelength"])
    calib = load_calibration(find_calibration(acq.path, paths["calib_root"]), cfg["wavelength"])
    p = pl.prepare(args.acq, acq.load_frames(), calib, register_frames=cfg.get("register", True),
                   subtract_dark=cfg.get("subtract_dark", True), sat=cfg.get("saturation", 255))
    full = reconstruct_mm(np.moveaxis(p.frames.mean(0), 0, -1), p.A, p.W, p.dark)
    thr = 0.1 * float(np.nanmedian(full["M11"]))
    tissue = full["M11"] > thr

    cols, stats = [], []
    _, mm, maps = pl.baseline_none(p)
    cols.append(("single frame", masked(maps, full["M11"], thr)))
    stats.append(("single frame", maps))
    for k in (2, 4, 8):
        _, mm, maps = pl.baseline_average(p, k)
        stats.append((f"average of {k}", maps))
        if k == 8:
            cols.append(("average of 8", masked(maps, full["M11"], thr)))
    for meth in args.methods:
        lev = best[meth]
        params = dict(cfg["methods"][meth][lev])
        if lev == "intensity":
            _, mm, maps, _ = pl.intensity_path(p, meth, params, vst=cfg.get("vst", True))
        else:
            _, mm, maps, _ = pl.mm_path(p, meth, params)
        label = f"{NAMES[meth]} ({'int.' if lev == 'intensity' else 'MM'})"
        cols.append((label, masked(maps, full["M11"], thr)))
        stats.append((label, maps))
        print(f"  computed {label}")
    fmaps = polarimetric_maps(full["nM"])
    cols.append(("16-frame average", masked(fmaps, full["M11"], thr)))
    stats.append(("16-frame average", fmaps))

    title = f"{args.acq}: one frame vs filters vs averaging"
    fig_maps(cols, args.inset, out, title)
    med = [(n, {k: float(np.nanmedian(mp[k][tissue & np.isfinite(mp[k])])) for k in ("totP", "linR")})
           for n, mp in stats]
    pd.DataFrame([{"what": n, **v} for n, v in med]).to_csv(out / "fig4_noise_bias.csv", index=False)
    fig_bias(med, out, args.acq)


if __name__ == "__main__":
    main()
