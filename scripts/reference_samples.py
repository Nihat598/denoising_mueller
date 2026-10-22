"""Bias check of the denoisers on uniform reference samples (air, polarizer, waveplate, tape, mirror).

On a uniform sample every pixel should give the same value, and for some samples the true value is
known (air: identity; polarizer: diattenuation 1; QWP in double pass: retardance 180 deg). Noise
does not only scatter the maps, it also biases them: diattenuation and depolarization of a single
noisy frame are pushed away from 0, retardance near 180 deg folds below 180. This script measures,
inside a central ROI, for frame 1 (no denoising), plain averages, and every method at both levels:
  - median of each map (compared with the nominal value and with the reference = mean of frames N/2+1..N)
  - spread (IQR) of each map
  - element-wise RMS of nM vs the reference

Usage
-----
  python scripts/reference_samples.py --config configs/compare_levels.yaml --samples configs/reference_samples.yaml
Outputs results/reference_samples/<timestamp>/: metrics.csv, summary.txt
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from denoising_mueller import pipeline as pl                                    # noqa: E402
from denoising_mueller.data.io import find_acquisition, load_paths             # noqa: E402
from denoising_mueller.polarimetry.reconstruction import find_calibration, load_calibration  # noqa: E402

MAPS = ("linR", "totD", "totP")


def roi_stats(maps, nM, ref_nM, roi):
    out = {}
    for k in MAPS:
        v = maps[k][roi]
        v = v[np.isfinite(v)]
        q1, med, q3 = np.percentile(v, [25, 50, 75]) if v.size else (np.nan,) * 3
        out[f"{k}_median"], out[f"{k}_IQR"] = float(med), float(q3 - q1)
    d = (nM[..., 1:] - ref_nM[..., 1:])[roi]
    out["nM_RMS_vs_ref"] = float(np.sqrt(np.nanmean(d ** 2)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/compare_levels.yaml", help="method parameters + paths")
    ap.add_argument("--samples", default="configs/reference_samples.yaml")
    ap.add_argument("--methods", nargs="*")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    scfg = yaml.safe_load(open(args.samples))
    paths = load_paths(cfg.get("paths", "configs/paths.yaml"))
    methods = args.methods or list(cfg["methods"])
    out = Path(paths.get("results_root", "results")) / "reference_samples" / f"{datetime.now():%Y-%m-%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"Writing to {out}")

    rows = []
    for s in scfg["samples"]:
        acq = find_acquisition(paths["data_root"], s["acq"], cfg["wavelength"])
        calib = load_calibration(find_calibration(acq.path, paths["calib_root"]), cfg["wavelength"])
        p = pl.prepare(s["acq"], acq.load_frames(), calib, register_frames=cfg.get("register", True),
                       subtract_dark=cfg.get("subtract_dark", True), sat=cfg.get("saturation", 255),
                       physical_tol=1.0)
        y0, y1, x0, x1 = s.get("roi", scfg.get("roi"))
        roi = np.zeros(p.mask.shape, bool)
        roi[y0:y1, x0:x1] = True
        base = dict(acq=s["acq"], sample=s["name"], **{f"nominal_{k}": s.get("nominal", {}).get(k, np.nan) for k in MAPS})
        ref = roi_stats(p.ref_maps, p.ref["nM"], p.ref["nM"], roi)
        base.update({f"ref_{k}_median": ref[f"{k}_median"] for k in MAPS})
        print(f"\n{s['name']} ({s['acq']}): reference medians " +
              "  ".join(f"{k}={ref[f'{k}_median']:.3f}" for k in MAPS))

        def add(level, method, value, mm, maps, dt):
            r = {**base, "level": level, "method": method, "value": value, "time_s": dt,
                 **roi_stats(maps, mm["nM"], p.ref["nM"], roi)}
            rows.append(r)
            print(f"   {level:9s} {method:9s} {value!s:<5} " +
                  "  ".join(f"{k} {r[f'{k}_median']:7.3f} (IQR {r[f'{k}_IQR']:.3f})" for k in MAPS) +
                  f"  nM_RMS {r['nM_RMS_vs_ref']:.4f}")

        _, mm, maps = pl.baseline_none(p)
        add("baseline", "none", np.nan, mm, maps, 0.0)
        for m in cfg.get("averaging_baselines", []):
            if m <= p.info["n_frames"] // 2:
                _, mm, maps = pl.baseline_average(p, m)
                add("baseline", "average", m, mm, maps, 0.0)
        for method in methods:
            for level in ("intensity", "mm"):
                params = dict(cfg["methods"][method][level])
                if level == "intensity":
                    _, mm, maps, dt = pl.intensity_path(p, method, params, vst=cfg.get("vst", True))
                else:
                    _, mm, maps, dt = pl.mm_path(p, method, params)
                add(level, method, list(params.values())[0], mm, maps, dt)
        pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)

    df = pd.DataFrame(rows)
    # bias = |median - target|, target = nominal where known, else the reference median
    for k in MAPS:
        tgt = df[f"nominal_{k}"].where(df[f"nominal_{k}"].notna(), df[f"ref_{k}_median"])
        df[f"{k}_bias"] = (df[f"{k}_median"] - tgt).abs()
    df.to_csv(out / "metrics.csv", index=False)
    cols = [f"{k}_{s}" for k in MAPS for s in ("bias", "IQR")] + ["nM_RMS_vs_ref"]
    summary = df.groupby(["method", "level"], dropna=False)[cols].mean().sort_values("nM_RMS_vs_ref")
    with pd.option_context("display.width", 220, "display.precision", 4):
        txt = "Mean over reference samples (bias = |median - nominal| or |median - reference median|):\n" + str(summary)
    print("\n" + txt)
    (out / "summary.txt").write_text(txt + "\n")


if __name__ == "__main__":
    main()
