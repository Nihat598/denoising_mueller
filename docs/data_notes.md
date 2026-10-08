# Data notes

Checked 2026-10-08.

## Format
- `To_Process/630_Image_Number_<k>.npy`: one frame, `uint8`, `(512, 608, 16)`, 630 nm, 2x2 binning.
- `LQ` = frame 1. `SHQ` = floor(mean of 16 frames) (verified exactly). `HQ` = 8-frame average,
  but it does **not** match any subset of the saved frames: frames used are unknown. Do not use HQ.
- `Raw_Images/` (May 2026 FF only): unbinned camera images, 2 cameras x 4 PSG states x 16 frames.
- `*.csv` timing log: one 16-state acquisition takes about 1 s.

## Datasets
| Set | Folder | Frames | Motion | Use |
|---|---|---|---|---|
| FF, May 2026 | `Data_for_denoising/2026-05-27_*` (2 acq.) | 16 | 098: none; 089: slow drift, -2 px | main benchmark |
| FF, Aug 2025 | `OR_measurements/*_F_FF_*` (5 acq.) | 8-16 | small | benchmark |
| OR, Aug 2025 | `OR_measurements/*_F_OR_*` | 1-8 | up to ~8 px | intraoperative test, needs registration |

OR = operating room (in vivo). Several OR acquisitions have 20-40% saturated pixels.

## Noise (robust binned fit, scripts/noise_characterization.py)
| Acquisition | var = a*mean + b | Notes |
|---|---|---|
| FF 0000_098 (1 ms) | a = 0.030, b = 0.13 | static; clean shot-noise line (~33 e-/count) |
| FF 0000_089 (1 ms) | a ~ 0.10-0.15 (inflated) | drifts -2 px vertically over 16 frames; sub-pixel residual inflates variance |
| OR 2007_052 (12 ms) | curved, ~0.05*mean + 1.5 | not a straight line: non-rigid tissue change, not only noise |

Use the FF 0000_098 model for the variance-stabilizing transform.

## Precomputed MM.npz (checked 2026-10-08)
- `nM` is stored transposed (textbook M = nM.reshape(4,4).T) and in float16 ("light" mode).
- `Msk` is all False for IMPv2 data (saturation test uses the wrong camera type): do not use it.
- Our NumPy Lu-Chipman (`polarimetry/lu_chipman.py`) reproduces the stored totD, linR, totP and
  azimuth to <= 1e-3 (median), so all maps can be recomputed off-Windows.
- Physical realizability of the stored reference MMs (Cloude test):
  | Data | physical | elements abs > 1 |
  |---|---|---|
  | Aug 2025 FF and OR (13 acquisitions with LQ + HQ/SHQ) | 97-99% | 0% |
  | May 2026 FF 0000_089 (zoom 12000) | ~57% | 38% |
  | May 2026 FF 0000_098 (zoom 10000, focus 6500) | 0% | 97% |
  Likely cause for 0000_098: zoom/focus differ from the calibration (pixel-wise A, W no longer aligned).
  Raw intensity frames are unaffected; only the MMs are wrong. Excluded from MM-level benchmarks.
- HQ references (Aug 2025) are 8-frame averages of unknown frames; LQ (= frame 1) may be part of
  them, which would make metrics slightly optimistic for all methods alike.

## Rules
- Build references from raw frames in float; register OR frames first.
- An input frame is never part of its own reference.
