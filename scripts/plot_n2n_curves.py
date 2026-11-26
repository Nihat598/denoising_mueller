"""Learning curves of the Noise2Noise runs (scripts/train_n2n.py), for slides and the paper.

  python scripts/plot_n2n_curves.py                       # all runs in runs/n2n
  python scripts/plot_n2n_curves.py --runs runs/n2n --out runs/n2n/figures

Reads runs/n2n/<fold>_<level>/log.csv and writes (PNG 300 dpi + PDF):
  n2n_curves.(png|pdf)        top: training loss, bottom: validation error (nM_RMSE_corr), one column per level,
                              one line per fold; dashed line = the same validation acquisitions without denoising
  n2n_curves_<run>.(png|pdf)  the same for each run alone (training loss and validation error on one figure)

Reading the curves: the training loss does NOT go to zero in Noise2Noise (the target is itself a noisy frame, so the
loss stays near the noise level); the validation error is what shows progress. The training loss is in the scaled
units of each network (divided by its data scale), so compare its shape, not its value, between levels.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

FOLD_STYLE = {"slices_test": ("#2a6f97", "trained on cavity piece + objects"),
              "cavity_test": ("#c8553d", "trained on slices + objects"),
              "objects_only": ("#6a994e", "trained on objects only")}
LEVEL_NAME = {"intensity": "intensity level", "mm": "Mueller-matrix level"}
INK, INK2, GRID = "#1f2328", "#57606a", "#e6e8eb"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 300, "font.family": "DejaVu Sans"})


def read_log(path):
    d = pd.read_csv(path)
    d = d.drop_duplicates("iteration", keep="last").sort_values("iteration")      # a resumed run appends
    val = next(c for c in d.columns if c.startswith("val_nM") and "raw" not in c)
    raw = next(c for c in d.columns if c.startswith("val_raw"))
    return d.rename(columns={val: "val", raw: "raw"})


def save(fig, out, name):
    fig.savefig(out / f"{name}.png", bbox_inches="tight")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  {out / name}.png/.pdf")


def style(ax, ylabel):
    ax.set_ylabel(ylabel)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs/n2n")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    runs = sorted(p.parent for p in Path(args.runs).glob("*/log.csv"))
    if not runs:
        raise SystemExit(f"no */log.csv under {args.runs}")
    out = Path(args.out or Path(args.runs) / "figures")
    out.mkdir(parents=True, exist_ok=True)
    print(f"Writing to {out}")

    logs = {}
    for r in runs:
        level = "mm" if r.name.endswith("_mm") else "intensity"
        fold = r.name[: -len(level) - 1]
        logs[(fold, level)] = read_log(r / "log.csv")

    # one figure per run
    for (fold, level), d in logs.items():
        fig, ax = plt.subplots(figsize=(5.2, 3.2))
        col = FOLD_STYLE.get(fold, ("#444444", fold))[0]
        ax.plot(d.iteration, d.val, color=col, lw=1.8, label="validation error (denoised)")
        ax.axhline(d.raw.iloc[0], color=INK2, ls="--", lw=1, label="validation error (raw frame)")
        best = d.loc[d.val.idxmin()]
        ax.scatter([best.iteration], [best.val], color=col, zorder=3, s=25)
        ax.annotate(f"best {best.val:.4f}", (best.iteration, best.val), xytext=(-6, 8), textcoords="offset points",
                    ha="right", fontsize=7, color=INK)
        style(ax, "nM_RMSE_corr (validation)")
        ax.set_xlabel("training iteration")
        ax2 = ax.twinx()
        ax2.plot(d.iteration, d.train_loss, color=INK2, lw=1, alpha=0.6, label="training loss")
        ax2.set_ylabel("training loss (scaled units)", color=INK2)
        ax2.spines["top"].set_visible(False)
        h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=7, loc="center right")
        ax.set_title(f"Noise2Noise, {LEVEL_NAME[level]}, {FOLD_STYLE.get(fold, ('', fold))[1]}", fontsize=9)
        save(fig, out, f"n2n_curves_{fold}_{level}")

    # overview: columns = levels, rows = training loss / validation error
    levels = [lv for lv in ("intensity", "mm") if any(k[1] == lv for k in logs)]
    fig, axes = plt.subplots(2, len(levels), figsize=(4.6 * len(levels), 5.4), squeeze=False, sharex=True)
    for j, lv in enumerate(levels):
        raws = []
        for (fold, level), d in logs.items():
            if level != lv:
                continue
            col, lab = FOLD_STYLE.get(fold, ("#444444", fold))
            axes[0, j].plot(d.iteration, d.train_loss, color=col, lw=1.4, label=lab)
            axes[1, j].plot(d.iteration, d.val, color=col, lw=1.6, label=lab)
            axes[1, j].axhline(d.raw.iloc[0], color=col, ls="--", lw=0.8, alpha=0.7)
            raws.append(d.raw.iloc[0])
        axes[0, j].set_title(LEVEL_NAME[lv], fontsize=10, color=INK)
        style(axes[0, j], "training loss (scaled)")
        style(axes[1, j], "validation nM_RMSE_corr")
        axes[1, j].set_xlabel("training iteration")
    h, lab = axes[1, 0].get_legend_handles_labels()
    h.append(plt.Line2D([], [], color=INK2, ls="--", lw=0.8)); lab.append("raw frame, no denoising (validation)")
    fig.legend(h, lab, frameon=False, fontsize=8, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, out, "n2n_curves")


if __name__ == "__main__":
    main()
