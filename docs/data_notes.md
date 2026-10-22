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
Refit with mean (not median) of squared frame differences, frames 5-8 of 0000_098:
var = 0.034*I + 0.18 (the median-based fit above underestimates the variance).
The intensity benchmark fits this model per acquisition on its reference frames only.
Frame 1 of 0000_098 is ~0.12 counts darker than frames 5-8 (lamp drift), shared by all methods.

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

## MM reconstruction (2026-10-15)
- Ported from the lab's C code (`fastMMPT-full-display/src/mmpt/C-libs`, compute_M_AIW.c +
  mp_comp_MM_AIW.c): 16-vectors are column-major (mat = vec.reshape(4,4).T), M = A^-1 I W^-1.
- Cross-check vs the compiled lab library (scripts/crosscheck_lab_library.py, synthetic data):
  M, nM, totD, linR, totP, azimuth all agree to <= 3e-13.
- Dark image: the IMPv2 calibration subtracts <wl>_Noise.npy from all calibration images; the
  acquisition GUI subtracts it from measurements, the offline batch pipeline does NOT (so stored
  MM.npz are not dark-subtracted). We subtract it by default (`subtract_dark`).
- Real-data check with the lab's calibration 2026-02-20_153510 (its own B0/P0/P90/L30 images,
  dark-subtracted): air -> identity (3 decimals), P0/P90 -> ideal polarizers (P90 ~1-2 deg off axis),
  L30 -> linear retardance 89.8 deg, depolarization ~0. Without dark subtraction elements move by
  up to 0.05. Confirms the reconstruction convention on real data.
- Lab azimuth convention = textbook axis + 90 deg (seen on the synthetic QWP: axis 30 -> map 120).
- Reference noise matters: with an 8-frame reference, NLM scored nM_RMSE 0.0236 vs the reference
  but 0.0062 vs ground truth (synthetic). `nM_RMSE_corr` = sqrt(MSE_vs_ref - var_ref), var_ref from
  two halves of the reference frames, matched the true error within 0.0002 for all methods.
- Physicality: a pure retarder sits on the Cloude boundary, so noise makes ~half its pixels
  "non-physical" at any fixed small tolerance. The evaluation mask uses tol = 3 x reference noise.

## Rules
- Build references from raw frames in float; register OR frames first.
- An input frame is never part of its own reference.

## Measurement_15102026 (Théotim, checked 2026-10-22)
- 12 acquisitions, 16 frames each, calibration 2026-10-14_164941 (A, W, 630_Noise.npy present; named in MetaData).
  Frames are now **float16** holding integer counts (old data: uint8). Digital gain 2, exposure 1 ms.
- Folder names do not always match the content; use MetaData "Name" (e.g. Linear_polarizer/0000_006 = "QWP 0°",
  Plastic card/0000_010 = "Tap 0°").
- Reconstruction check (mean of 16 frames, dark-subtracted, central ROI): air and metal plate -> identity,
  linear polarizer -> ideal polarizer (m12 = m21 = -1), tape -> ~90° linear retarder.
  QWP 0°/90° -> linR 177°/175°, m44 = -1: a QWP passed twice (reflection geometry) acts as a half-wave plate.
  RMS vs ideal 180° retarder 0.105. Use `check_waveplate.py --retardance 180` for these.
- Noise (static samples, frames 9-16): var ~ 0.15-0.24 x I (old May data: 0.034 x I). Skin signal is low
  (mean 13-17 counts), so single frames are very noisy.
- Tissue = in vivo skin only (Finger 0000_002, Scar 0000_003). Both move: finger up to 3 px, scar drifts
  steadily up to 10 px over 16 frames. Motion inflates the fitted noise (scar: 0.87 x I) and makes the
  "mean of frames 9-16" reference unreliable -> not suitable for the main benchmark; static ex vivo
  samples (formalin-fixed brain) are still needed.
