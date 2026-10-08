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
| FF, May 2026 | `Data_for_denoising/2026-05-27_*` (2 acq.) | 16 | none | main benchmark |
| FF, Aug 2025 | `OR_measurements/*_F_FF_*` (5 acq.) | 8-16 | small | benchmark |
| OR, Aug 2025 | `OR_measurements/*_F_OR_*` | 1-8 | up to ~8 px | intraoperative test, needs registration |

OR = operating room (in vivo). Several OR acquisitions have 20-40% saturated pixels.

## Noise
FF May 2026 (1 ms exposure): variance from consecutive-frame differences rises linearly
with mean, ~0.45 at 5 counts to ~3.9 at 100 counts (shot + read noise).

## Rules
- Build references from raw frames in float; register OR frames first.
- An input frame is never part of its own reference.
