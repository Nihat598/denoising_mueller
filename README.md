# denoising_mueller

Systematic comparison of image denoising strategies for biomedical Mueller polarimetric imaging:
classical filters, supervised networks (PDDN, U-Net) and self-supervised learning (Noise2Noise),
applied at the raw-intensity level and at the Mueller-matrix (MM) level.

Work for SPIE Photonics West 2027 — LPICM, CNRS, École polytechnique, IP Paris.

## Layout

```
configs/                 YAML configs (paths, method parameters)
scripts/                 Entry points run from the command line
src/denoising_mueller/
    data/                Loading acquisitions, frame registration, splits
    polarimetry/         MM reconstruction and Lu-Chipman decomposition (wrappers)
    methods/             Denoisers: classical filters, PDDN, U-Net, Noise2Noise
    evaluation/          Metrics at intensity, MM and diagnostic-map levels
notebooks/               Exploration only; nothing the results depend on
results/                 Outputs (not committed, except small summary tables)
tests/                   Quick checks
docs/                    Notes, figures for the paper
```

## Data

Raw data is **not** stored in this repository. Point `configs/paths.yaml` to the folder
containing the acquisitions (`<date>_<time>_F_<OR|FF>_HORAO_<id>/To_Process/*.npy`).

Each frame file is `uint8`, shaped `(512, 608, 16)` = (height, width, polarization state), 630 nm.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

## Benchmarks

```bash
# MM level (precomputed MM.npz: LQ input, SHQ/HQ reference; no calibration needed)
python scripts/mm_classical_benchmark.py --split val --sweep      # tune on validation
python scripts/mm_classical_benchmark.py --split test             # evaluate once

# Intensity level (raw repeated frames: frame 1 input, mean of second half as reference)
python scripts/intensity_classical_benchmark.py --split val --sweep
python scripts/intensity_classical_benchmark.py --split test
```
Splits and fixed parameters: `configs/mm_classical.yaml`, `configs/intensity_classical.yaml`.

## First scripts

```bash
python scripts/inventory.py        # list acquisitions, frame counts, saturation
python scripts/noise_characterization.py --acq <acquisition folder name>
```
