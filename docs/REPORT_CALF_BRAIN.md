# Classical denoising on calf brain (measured 2026-11-03) — report (5 Nov 2026)

What was done, what the results say, where the files are, and how to reproduce everything.
All numbers come from `results/2026-11-03_calf_brain/` in the GitHub repo.
Method details shared with the October run are in `docs/REPORT_2026-10-22.md` (sections 3 and 8).

---

## 1. Short version

- **On brain tissue, a single denoised frame clearly beats averaging 8 frames.** 11 of the 12 filter/level
  combinations (all except median at the intensity level) reach a lower error than the 8-frame average, on both the
  cavity piece and on slices never used for tuning.
- **Best group: BM3D (MM level), NLM (intensity level), BM3D (intensity level), TV (both levels).** Their estimated
  errors (0.0035–0.006) are ~5–9× lower than the 8-frame average (0.031) and ~20–35× lower than one raw frame (0.129).
  The differences *inside* this group are too small to rank reliably with our reference (section 4).
- **Intensity vs MM level:** no single winner. BM3D and median are better at the MM level; NLM, Gaussian and
  (on slices) wavelet are better at the intensity level; TV is equal. Median at the intensity level and wavelet at the
  MM level are the two poor/unstable combinations.
- **Noise strongly biases the maps.** In the tissue of slice 011, the median retardance is 53° in one raw frame,
  24° for 8 frames and 19° for 16 frames, still falling: the "reference" retardance is itself inflated by noise.
  Depolarization is pulled down by noise (0.847 → 0.890) and every good filter brings it back to the 16-frame value.
- **Brain needs strong smoothing** (e.g. Gaussian σ 3 vs 0.5 on the October plastic card). The error score favours
  smoothing large uniform regions; the zoomed maps show what fine detail it costs (section 6).
- **Speed:** unchanged from the October timing (Gaussian/NLM/wavelet ≤ 2 s, TV/median 7–9 s, BM3D ~150 s alone).

---

## 2. Data

12 acquisitions (16 frames each), calibration `2026-11-03_095543` (same day), 630 nm, 1 ms, digital gain 2.
Two specimens (Théotim): a **brain piece with a cavity** (001–008, morning; zoom 9000 for 001–005, 3000 after) and
**slices** (009–012, afternoon). Overview of all 12: `figures/overview_all_acquisitions.png`.

Checks: no saturation (≤ 0.4% of pixels), drift ≤ 1–2 px (corrected by registration), reference matrices 76–99%
physically valid (lower values = more background around the tissue).

| Role | Acquisitions |
|---|---|
| tuning (val) | 001 (cavity piece, zoom 9000), 006 (cavity piece, zoom 3000) |
| test, same specimen | 002, 003, 004, 005, 007, 008 (other views of the cavity piece) |
| test, unseen specimen | 009, 010, 011, 012 (slices) |

---

## 3. Changes since the October run

1. **Dark pixels excluded.** The cavity is dark (signal ≈ 0). There, M/m11 explodes in a single frame and dominated
   every score. Pixels below **10% of the median signal** are now excluded from scoring (about 5% of pixels), in both
   the evaluation mask and the reference-noise estimate. October results were computed with the old rule (2nd
   percentile only); re-running them would change them slightly.
2. **Median over acquisitions** in the summary tables and figures (instead of the mean), because wavelet at the MM
   level blows up on 3 acquisitions and would distort means.
3. **Wider parameter ranges.** The brain kept preferring the strongest setting, so ranges were extended upward until
   the scores flattened (Gaussian to σ 4, median to 11×11, TV to 8, NLM to 2.0, wavelet to 4, BM3D to 2.0).
4. **Tuning used the plain score `nM_RMSE`**, not the corrected one (a bug in the reference-noise estimate during the
   tuning run, fixed before the test). Within one acquisition the correction is almost the same offset for every
   filter, so the chosen strengths would barely change.
5. **Consistency re-check.** The upper strengths (added in step 3) were scored after the reference-noise fix, which
   also changes which pixels count as physically valid; the lower ones before it. All strengths were therefore
   re-scored on 001 + 006 with the final code (`val_sweep/recheck_metrics.csv`). The chosen strengths stay best or
   within ~1% of the best (e.g. NLM at the MM level: h 1.0 → 0.0431, h 1.5 → 0.0433), so the test results stand.
   BM3D: see `val_sweep/recheck_bm3d.txt`.
6. Acquisitions 007 and 009 were first skipped by the safety check "reference ≥ 80% physical" (they are at 76% and
   79% because of background). The threshold is 70% for this dataset; the scoring mask excludes invalid pixels anyway.

**Tuned strengths** (mean `nM_RMSE` over 001 + 006):

| Filter | Intensity level | MM level | Note |
|---|---|---|---|
| Gaussian | σ 3.0 | σ 3.0 | σ 2–4 within 1% |
| Median | 11×11 | 11×11 | still improving slightly at 11 |
| NLM | h 1.5 | h 1.5 | h 0.8–2.0 within 1% |
| Wavelet | scale 3.0 | scale 3.0 | clearly worse at 4 |
| TV | weight 8.0 | weight 4.0 | 2–8 within 2% |
| BM3D | scale 2.0 | scale 2.0 | small gains above 1.0 |

---

## 4. Main comparison (10 test acquisitions)

Score = `nM_RMSE_corr` (Mueller-matrix element error, corrected for the reference's own noise), **median** over
acquisitions; lower is better. Last column: the uncorrected `nM_RMSE`, which includes the reference noise (~0.052).

| Method | Level | Cavity piece (6) | Slices, unseen (4) | All (10) | Uncorrected |
|---|---|---|---|---|---|
| BM3D | MM | **0.004** | **0.003** | **0.0035** | 0.048 |
| NLM | intensity | **0.004** | **0.003** | 0.004 | 0.051 |
| BM3D | intensity | 0.007 | **0.003** | 0.004 | 0.052 |
| TV | intensity | 0.005 | 0.006 | 0.005 | 0.052 |
| TV | MM | 0.005 | 0.006 | 0.005 | 0.049 |
| NLM | MM | 0.007 | 0.009 | 0.007 | 0.051 |
| Gaussian | intensity | 0.008 | 0.008 | 0.008 | 0.053 |
| median | MM | 0.008 | 0.010 | 0.008 | 0.051 |
| Gaussian | MM | 0.013 | 0.010 | 0.012 | 0.054 |
| wavelet | intensity | 0.022 | 0.018 | 0.022 | 0.058 |
| wavelet | MM | 0.017 | 0.033 | 0.025 | 0.061 (blows up on 007, 008, 009) |
| **average of 8 frames** | — | 0.033 | 0.027 | 0.031 | 0.062 |
| median | intensity | 0.035 | 0.042 | 0.039 | 0.067 |
| average of 4 frames | — | 0.050 | 0.051 | 0.050 | 0.076 |
| average of 2 frames | — | 0.078 | 0.084 | 0.081 | 0.100 |
| single frame | — | 0.126 | 0.132 | 0.129 | 0.139 |

How to read it:

1. **Generalization:** the slices were never used for tuning and give the same picture (often slightly better), so
   the tuned strengths are not specific to the tuning piece.
2. **Resolution limit:** the reference (mean of frames 9–16) carries ~0.052 of noise. The best filters are at
   0.048–0.052 uncorrected, i.e. at that floor. The corrected values (0.003–0.006) say their true error is small, but
   differences of a few thousandths are within the uncertainty of the correction. Report the top group, not a strict
   order. A less noisy reference (e.g. 32+ frames) would be needed to separate them.
3. **Level:** BM3D about −35% at the MM level (cavity piece), NLM about −40% and Gaussian about −30% at the intensity
   level, median about −80% at the MM level (an 11×11 median on intensities distorts the polarization states). Which level
   is better depends on the filter.
4. Compared with October (plastic cards, where no filter beat 8-frame averaging): tissue has large smooth regions,
   which spatial filters exploit well; plastic cards have pixel-scale grain, which they cannot.

Figure: `paper_figures/fig2_scores.(png|pdf)` (both groups, both levels, averaging lines).

---

## 5. Noise bias of the diagnostic maps (slice 011, tissue pixels)

| | Median depolarization | Median linear retardance |
|---|---|---|
| single frame | 0.847 | 52.5° |
| average of 2 / 4 / 8 | 0.867 / 0.879 / 0.886 | 41.3° / 31.5° / 24.2° |
| 16-frame average | 0.890 | 19.0° |
| Gaussian (int.) | 0.889 | 10.8° |
| NLM (int.) | 0.890 | 11.8° |
| TV (MM) | 0.890 | 9.4° |
| BM3D (MM) | 0.897 | 5.4° |

- **Depolarization:** noise lowers it; averaging converges to ~0.890 and the filters land on that value (BM3D a bit
  higher). Consistent and easy to explain.
- **Retardance:** noise inflates it strongly (retardance is the size of a vector; noise can only make it bigger), and
  even 16 frames have not converged. The filters give 5–12°, below the 16-frame value. Two effects mix here and we
  cannot separate them with this data: (a) the 16-frame value is still too high (noise bias), and (b) strong spatial
  smoothing averages fibre directions together, which lowers retardance for real. **This is the question to expect at
  the conference**; the honest answer is that the true value lies below 19° and the filters bound it from below.
  A long acquisition (e.g. 64 frames) of one slice would settle it.

Figure: `paper_figures/fig4_noise_bias.(png|pdf)` (numbers in `fig4_noise_bias.csv`).

---

## 6. Maps (what the score does not show)

`paper_figures/fig1_maps.(png|pdf)`: slice 011, rows depolarization / retardance / azimuth / diattenuation;
columns single frame, 8-frame average, Gaussian, NLM, TV, BM3D (each at its better level), 16-frame average.
`fig1b_maps_zoom` shows the white box enlarged.

- All filters recover the grey/white-matter pattern of the gyri that is barely visible in one frame, and the azimuth
  (fibre orientation) becomes readable in the white matter, which even 16 frames do not achieve.
- In the zoom, the cost of strong smoothing is visible: Gaussian (σ 3) makes blobs, NLM leaves a faint grid-like
  texture in the retardance, BM3D and TV give the cleanest maps but also the smoothest. Thin structures smaller than
  ~5 px are lost by all of them.
- So the "best" strength by error score is on the strong side for fine detail. For the talk: show the zoom and say
  that the score and visual detail do not fully agree.

---

## 7. Run time

The test runs mostly ran two jobs at once on 2 CPU cores, so their times are inflated (BM3D ~190 s instead of
~150 s). Use the clean October timing (`docs/REPORT_2026-10-22.md`, section 6) and re-time on the DGX Spark for the
paper. `paper_figures/fig3_quality_vs_time` uses the test-run times and is therefore only indicative.

---

## 8. Files

```
results/2026-11-03_calf_brain/
  val_sweep/metrics.csv, log.txt     tuning on 001 + 006 (every strength tried, both levels)
  test/metrics.csv                   one row per acquisition x method x level (10 acquisitions)
  test/summary.csv, config.yaml, log.txt
  paper_figures/                     fig1 maps, fig1b zoom, fig2 scores, fig3 quality vs time, fig4 noise bias (PNG 300 dpi + PDF)
  figures/                           working figures for 002 and 011, overview of all 12 acquisitions
```

Columns of `test/metrics.csv`: see `docs/REPORT_2026-10-22.md` section 8.

---

## 9. Reproduce on your PC (Windows, Anaconda Prompt)

```bat
cd C:\Users\Nihat_AHMADLI\Desktop\SPIE\denoising_mueller
conda activate denoising
git pull
pip install -r requirements.txt
pip install -e .
```

`configs\paths.yaml`:

```yaml
data_root: "C:/Users/Nihat_AHMADLI/Desktop/SPIE/Data_for_denoising/CALF_BRAIN"
calib_root: "C:/Users/Nihat_AHMADLI/Desktop/SPIE/Data_for_denoising/CALF_BRAIN/Calibration"
results_root: "results"
wavelength: 630
```

Then (each step writes a new time-stamped folder under `results\`):

```bat
:: 1. tuning on the two tuning views (slow: ~1-2 h because of BM3D; ranges as used here)
python scripts/compare_levels.py --config configs/compare_levels_calf.yaml --split val --sweep --grid gaussian=0.5,1.0,2.0,3.0,4.0 median=3,5,7,9,11 nlm=0.8,1.2,1.5,2.0 wavelet=1.0,2.0,3.0,4.0 tv=1.0,2.0,4.0,8.0 bm3d=1.0,1.5,2.0

:: 2. test with the strengths already written in configs/compare_levels_calf.yaml (~1.5 h)
python scripts/compare_levels.py --config configs/compare_levels_calf.yaml --split test

:: 3. conference figures from the test metrics (replace <run> by the folder name from step 2)
python scripts/make_paper_figures.py --config configs/compare_levels_calf.yaml --metrics results/compare_levels/<run>_test/metrics.csv --acq 2026-11-03_152503_F_FF_HORAO_0000_011 --methods gaussian nlm tv bm3d --inset 150 300 230 430
```

To show another acquisition in the map figure, change `--acq` and the `--inset y0 y1 x0 x1` box (pixels).
Run one job at a time: two full-size jobs together need more than 6 GB of memory.

**Housekeeping:** `denoising_mueller\results\_transfer_calf\` (~600 MB) on your PC was only used for the transfer;
you can delete it.

---

## 10. Next steps

1. **PDDN** (inference only) on the same 10 test acquisitions and the same scores, timed on the DGX Spark GPU;
   add it to fig2 and fig3. Confirm the training patch size and training acquisitions with Théotim first.
2. **Re-time** all classical methods on the DGX Spark CPU (one clean run).
3. Optional, settles the retardance question (section 5): one 64-frame acquisition of a slice.
4. Agree the scope and the main message with Tatiana; suggested message: *"one denoised frame beats an 8-frame
   average on brain tissue; filtering level matters per filter; noise biases retardance and depolarization, and
   denoising corrects most of it."*
