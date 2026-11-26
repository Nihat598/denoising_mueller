# Noise2Noise for Mueller polarimetry: training and testing

Code: `src/denoising_mueller/methods/n2n.py` (network, Mueller algebra, inference),
`src/denoising_mueller/data/n2n_data.py` (splits, data, pairs, validation), `scripts/train_n2n.py`,
`scripts/compare_levels.py --methods n2n`, configs `configs/n2n.yaml` and `configs/compare_levels_newdata.yaml`.
Tests: `tests/test_n2n.py`.

## 1. The idea in short

Noise2Noise (Lehtinen et al., ICML 2018) trains a denoiser without clean images: the network sees one noisy
frame as input and **another noisy frame of the same scene** as target. With a squared-error loss, the best it can
do is predict the average of the possible targets, i.e. the clean image, provided that the noise of the target
(1) has zero mean around the true signal and (2) is independent of the noise of the input.

Our acquisitions give this for free: 16 repeated frames of a static sample. Frame noise (shot noise + read noise)
is independent from frame to frame and averages to the true signal. Every pair of frames (i, j), i ≠ j, is a
training pair (240 ordered pairs per acquisition), and every 128 × 128 crop of them is a training patch.

## 2. What is denoised: two levels, as for the classical filters

**Level `intensity`.** Input and target are one raw frame: the 16 intensities in counts, dark not subtracted,
exactly the input of the classical intensity path and of PDDN. Raw counts average to the true intensity (plus a
constant dark offset), so this is a valid Noise2Noise target.

**Level `mm`.** Input and target are the **unnormalized** Mueller matrix of one frame, M = A⁻¹ (I − dark) W⁻¹
(16 elements, lab layout). M is **linear** in I, so its noise also averages to the true M and stays independent
between frames: also a valid target.

Never used as targets: the normalized matrix M/m11 and the maps (retardance, depolarization, diattenuation,
azimuth). They are nonlinear in the data, so their noise does not average to the truth (we measured this: noise
inflates retardance and lowers depolarization). A network trained on them would learn the bias. This is the same
reason Lehtinen et al. could not tone-map their targets.

After denoising, both levels go through the same steps as every other method: normalization by m11, Lu-Chipman
maps, the same mask and the same scores (`nM_RMSE_corr` etc.).

## 3. Choices and why

- **Network**: the U-Net of the Noise2Noise paper (5 poolings, 48 features, leaky ReLU, no normalization layers),
  16 channels in and out, about 1 million parameters, with a residual connection (it predicts the correction to
  its input). Images of any size are padded to a multiple of 32 at inference.
- **Loss**: mean squared error (L2), which recovers the mean, the right statistic for shot/read noise. Averaged
  over valid pixels only.
- **Excluded pixels**: a pixel is left out of the loss when the input frame or a target frame is saturated in any
  channel (clipping removes part of the noise distribution, so the target no longer averages to the truth;
  Lehtinen et al. §3.2), and in the 1–2 px border strips created by registration.
- **Registration**: frames are aligned onto frame 1 with the same integer-shift registration as the evaluation
  (no interpolation, which would correlate noise between neighbouring pixels). Shifts are cached in
  `runs/n2n/shift_cache.json`.
- **No flips or rotations**: rotating or mirroring a polarimetric image changes what the 16 channels mean
  (orientation of polarizers vs sample axes), so augmented patches would not be physical scenes. Training uses
  random crops of the real frames only.
- **Scale**: inputs and targets are divided by the 99th percentile of |values| of the training data (stored in the
  checkpoint and reused at test time). A fixed linear scale keeps the Noise2Noise property.
- **Targets**: `target_frames: 1` = pure Noise2Noise (frame i → frame j). Setting k > 1 uses the mean of k other
  frames as target (still noisy and unbiased, lower variance). Rep2Rep (MRI, 2025) found no difference between
  1→1 and averaged targets, so 1 is the default.
- **Brain share**: when a fold contains brain views, 30% of training patches are drawn from them (`brain_fraction`),
  so the few brain views are not drowned by the ~35 object acquisitions.
- **Checkpoint selection**: every 1000 iterations the network is run on the validation acquisitions with exactly
  the evaluation protocol (input = frame 1, reference = mean of frames N/2+1..N, evaluation mask, `nM_RMSE_corr`).
  `best.pt` is the checkpoint with the lowest validation score. Test acquisitions are never looked at during
  training.
- **Optimizer**: Adam (β2 = 0.99 as in the paper), learning rate 3·10⁻⁴ with 500 warm-up steps and cosine decay,
  batch 16, 50 000 iterations.

## 4. Data split (configs/n2n.yaml)

- Excluded: 018, 031 (almost no light), 050 (frames change between shots), 059 (moves ~5 px).
- **New data, split by object**: all views of one object stay together. Test objects: stressed plastic
  (016–017), 3D-printed holder (026–028), leaf (040–043), skin (038, 047, 048), playing card (065), plastic
  flower (070): 14 acquisitions. Validation objects: transparent sheet (029–030), plastic film (036), woven fabric
  (060–061): 5. Training: the other 35. **The groups were assigned from the images**: if you know that two
  groups are the same object (or that 038 is not skin), edit `object_groups`.
- **Brain, two folds by specimen**:
  - `slices_test`: trains on the cavity piece (001–005, 007, 008; 006 kept for validation), tested on the slices
    009–012;
  - `cavity_test`: trains on the slices (009, 011, 012; 010 kept for validation), tested on the cavity piece
    001–008;
  - `objects_only`: never sees brain; tested on all brain acquisitions.
- Leak guards: `build_split` refuses overlapping splits, a group or specimen split across train/test, or a brain
  validation view from a test specimen. Each checkpoint stores its split; `compare_levels.py` picks, for every
  acquisition, a fold that has it in its **test** set and refuses to evaluate anything the network was trained or
  validated on.

## 5. Running it on the DGX Spark

### 5.1 Copy the new data (once, from PowerShell on your PC)

Only the frames, metadata and calibration are needed (~9 GB instead of 24 GB):

```powershell
robocopy "C:\Users\Nihat_AHMADLI\Desktop\SPIE\Data_for_denoising\new_data" "C:\Users\Nihat_AHMADLI\Desktop\SPIE\new_data_small" MetaData.json 630_Image_Number_*.npy A.npy W.npy 630_Noise.npy /S /XF 630_Image_Number_HQ.npy 630_Image_Number_LQ.npy 630_Image_Number_SHQ.npy
scp -r "C:\Users\Nihat_AHMADLI\Desktop\SPIE\new_data_small" aopolarimetry@spark-6c3f:~/data/new_data
```

The calf brain is already on the Spark in `~/data/CALF_BRAIN` (with its `Calibration` folder).

### 5.2 Paths (on the Spark)

```bash
ssh aopolarimetry@spark-6c3f
tmux new -s n2n                    # or: tmux attach -t n2n
cd ~/denoising_mueller
source .venv/bin/activate
git fetch origin
git checkout noise2noise          # the Noise2Noise work is on this branch until it is merged into main
git pull
nano configs/paths.yaml
```

`data_root` and `calib_root` must be the folder that contains **both** `CALF_BRAIN` and `new_data` (they are
searched recursively). Keep your PDDN lines as they are:

```yaml
data_root: "/home/aopolarimetry/data"
calib_root: "/home/aopolarimetry/data"
results_root: "results"
wavelength: 630
```

### 5.3 Check, then a 2-minute dry run

```bash
python -m pytest -q tests
python scripts/train_n2n.py --fold slices_test --level intensity --iterations 300 --dry-run --out runs/n2n_dry
```

The dry run loads 3 training and 2 validation acquisitions. It should print the data scale, the validation score
of the raw frame, then a line every 1000 iterations (here only at 300) with the validation score going down.

### 5.4 Train the six networks (3 folds × 2 levels)

```bash
mkdir -p runs
for f in slices_test cavity_test objects_only; do
  for l in intensity mm; do
    python scripts/train_n2n.py --fold $f --level $l 2>&1 | tee runs/train_${f}_${l}.log
  done
done
```

The first run registers all acquisitions (a few minutes, then cached). The time per run is printed at each
validation line (`[.. min]`). If a run is interrupted, add `--resume` to the same command. If the GPU runs out of
memory, add `--data-device cpu` (frames stay in RAM, slightly slower).

How to read a validation line:

```
it  12000/50000  lr 2.6e-04  train loss 0.0123  val nM_RMSE_corr 0.0210 (raw 0.1100)  ...  * best
```

`raw` is the score of the noisy input frame; `val` is the network's score on the same frames; lower is better and
`* best` marks a new best checkpoint. The **train loss does not go to zero**: the target is itself noisy, so the
loss stays at about the noise level. This is expected in Noise2Noise; follow the validation score instead.

### 5.5 Test

```bash
# brain: two-fold (each acquisition denoised by the network that never saw its specimen)
python scripts/compare_levels.py --config configs/compare_levels_calf.yaml --split test --methods n2n --n2n-folds cavity_test slices_test
# brain: network trained on objects only
python scripts/compare_levels.py --config configs/compare_levels_calf.yaml --split test --methods n2n --n2n-folds objects_only
# new data: held-out objects
python scripts/compare_levels.py --config configs/compare_levels_newdata.yaml --split test --methods n2n --n2n-folds objects_only
```

Each run writes `results/compare_levels/<time>_test/metrics.csv` in the same format as the classical runs (the
column `n2n_fold` says which network denoised each acquisition). Time is measured on the GPU with
synchronization, like PDDN.

### 5.6 Learning curves

```bash
python scripts/plot_n2n_curves.py          # -> runs/n2n/figures/n2n_curves*.png/.pdf
```

One overview figure (training loss and validation error, one column per level, one line per fold, raw-frame
level dashed) and one figure per run.

### 5.7 Figures with the classical filters

```bash
python scripts/make_paper_figures.py --config configs/compare_levels_calf.yaml \
  --metrics results/compare_levels/<classical_run>_test/metrics.csv results/compare_levels/<n2n_run>_test/metrics.csv \
  --acq 2026-11-03_152503_F_FF_HORAO_0000_011 --methods gaussian nlm tv bm3d n2n \
  --n2n-folds cavity_test slices_test --inset 150 300 230 430 --cmap jet
```

## 6. Checks done before handing over (26 Nov 2026)

- Mueller reconstruction in torch = the numpy pipeline (relative difference ~3·10⁻⁷) on real brain 011 and object
  058 with their own calibrations; the training validation score equals the evaluation's `nM_RMSE_corr` exactly
  for the raw frame (brain 011: 0.12732; object 058: 0.09618).
- Pair sampling: input and target frames always differ; saturated pixels of either are masked (unit test).
- Synthetic data with known ground truth (3 small training acquisitions, 8000 CPU iterations, intensity level):
  true error of the normalized matrix 0.0087 for Noise2Noise vs 0.0205 for the 8-frame average, 0.0164 for a
  Gaussian filter and 0.0060 for NLM (the synthetic sample is very smooth, the ideal case for NLM); the corrected
  score `nM_RMSE_corr` equals the true error. The real training (more data, 50 000 iterations) runs on the GPU.
