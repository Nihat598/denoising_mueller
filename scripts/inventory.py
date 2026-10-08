"""List all acquisitions with frame counts, exposure, signal level and saturation.

Usage: python scripts/inventory.py [--config configs/paths.yaml]
Writes results/inventory.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from denoising_mueller.data.io import list_acquisitions, load_paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/paths.yaml")
    args = ap.parse_args()
    cfg = load_paths(args.config)

    rows = []
    for acq in list_acquisitions(cfg["data_root"], cfg.get("wavelength", 630)):
        row = dict(name=acq.name, setting=acq.setting, n_frames=acq.n_frames,
                   exposure_ms=acq.metadata.get("Exposure_time_ms"))
        if acq.n_frames:
            f1 = np.load(acq.frame_files[0])
            row.update(mean_counts=round(float(f1.mean()), 1),
                       saturated_pct=round(100 * float((f1 == 255).mean()), 2))
        rows.append(row)
        print(row)

    out = Path(cfg.get("results_root", "results")) / "inventory.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
