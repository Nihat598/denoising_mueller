"""Train a Noise2Noise denoiser on repeated polarimetric frames (one fold, one level).

  python scripts/train_n2n.py --fold slices_test --level intensity
  python scripts/train_n2n.py --fold slices_test --level mm
  python scripts/train_n2n.py --fold cavity_test --level intensity      # etc. (folds and levels in configs/n2n.yaml)
  python scripts/train_n2n.py --fold slices_test --level intensity --iterations 300 --dry-run   # quick check

Writes runs/n2n/<fold>_<level>/: best.pt (lowest validation error), last.pt, log.csv, split.json, config.yaml.
Each checkpoint stores the split; scripts/compare_levels.py --methods n2n uses it to pick, for every test
acquisition, the fold that never saw it, and refuses acquisitions used in training or validation.

Validation (every val_every iterations) uses exactly the evaluation's protocol, mask and score: input frame 1 of
each validation acquisition, reference = mean of frames N/2+1..N, nM_RMSE_corr (normalized Mueller-matrix error
corrected for the reference noise). It is printed next to the same score for the raw input frame (no denoising).
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from denoising_mueller.data.io import load_paths
from denoising_mueller.data.n2n_data import (PairSampler, Store, build_split, estimate_scale, load_acq, masked_mse,
                                             validation_score)
from denoising_mueller.methods.n2n import LEVELS, UNet, pad_to_multiple


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/n2n.yaml")
    ap.add_argument("--fold", required=True)
    ap.add_argument("--level", required=True, choices=LEVELS)
    ap.add_argument("--iterations", type=int, default=None, help="override training.iterations")
    ap.add_argument("--out", default="runs/n2n")
    ap.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    ap.add_argument("--data-device", default=None, help="where the frames are kept (default: same as --device)")
    ap.add_argument("--resume", action="store_true", help="continue from last.pt in the run folder")
    ap.add_argument("--dry-run", action="store_true", help="load only 3 training and 2 validation acquisitions")
    args = ap.parse_args()

    import torch
    cfg = yaml.safe_load(open(args.config))
    tr = dict(cfg["training"])
    if args.iterations:
        tr["iterations"] = args.iterations
    if tr["patch"] % 32:
        raise SystemExit("training.patch must be a multiple of 32")
    paths = load_paths(cfg.get("paths", "configs/paths.yaml"))
    split = build_split(cfg, args.fold)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    data_device = torch.device(args.data_device or device)
    torch.manual_seed(tr["seed"]); np.random.seed(tr["seed"])

    run = Path(args.out) / f"{args.fold}_{args.level}"
    run.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, run / "config.yaml")
    (run / "split.json").write_text(json.dumps({k: split[k] for k in ("fold", "train", "val", "test")}, indent=1))
    print(f"fold {args.fold}, level {args.level}: {len(split['train'])} train, {len(split['val'])} val, "
          f"{len(split['test'])} test acquisitions -> {run}   device {device} (data on {data_device})")

    train_names, val_names = split["train"], split["val"]
    if args.dry_run:
        train_names = [n for n in train_names if n in split["brain"]][:1] + \
                      [n for n in train_names if n not in split["brain"]][:2]
        val_names = val_names[:2]
    brain = set(split["brain"])
    cache = Path(args.out) / "shift_cache.json"

    def load(names, what):
        out = []
        for k, n in enumerate(names, 1):
            t0 = time.perf_counter()
            a = load_acq(n, paths, cfg, brain, cache, validation=(what == "val"))
            print(f"  [{what} {k}/{len(names)}] {n}  frames={a.frames.shape[0]}  shift<={a.max_shift}px  "
                  f"saturated={a.sat.any(0).mean():.1%}  ({time.perf_counter() - t0:.0f}s)", flush=True)
            out.append(a)
        return out

    print("Loading training data (registration shifts are cached after the first run)")
    tr_store = Store(load(train_names, "train"), data_device)
    va_store = Store(load(val_names, "val"), data_device)

    scale = estimate_scale(tr_store, args.level, tr["scale_quantile"], seed=tr["seed"])
    print(f"data scale ({tr['scale_quantile']}th percentile of |values| at level {args.level}): {scale:.3f}")
    sampler = PairSampler(tr_store, args.level, tr["patch"], tr.get("brain_fraction"), tr.get("target_frames", 1),
                          tr.get("min_valid", 0.5), seed=tr["seed"])

    net = UNet(16, tr["width"], tr["residual"]).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=tr["lr"], betas=(0.9, 0.99))
    n_it, warm = tr["iterations"], tr.get("warmup", 0)

    def lr_at(it):
        if it < warm:
            return tr["lr"] * (it + 1) / warm
        q = (it - warm) / max(1, n_it - warm)
        return tr["lr_min"] + 0.5 * (tr["lr"] - tr["lr_min"]) * (1 + math.cos(math.pi * q))

    def ckpt(it, val, path):
        torch.save({"state_dict": net.state_dict(), "level": args.level, "scale": scale, "width": tr["width"],
                    "residual": tr["residual"], "iteration": it, "val_nM_RMSE_corr": val,
                    "split": {k: split[k] for k in ("fold", "train", "val", "test")},
                    "meta": {"training": tr, "commit": git_commit(), "config": args.config,
                             "time": time.strftime("%Y-%m-%d %H:%M:%S")},
                    "optimizer": opt.state_dict()}, path)

    start, best = 0, float("inf")
    if args.resume and (run / "last.pt").exists():
        ck = torch.load(run / "last.pt", map_location="cpu", weights_only=False)
        net.load_state_dict(ck["state_dict"]); opt.load_state_dict(ck["optimizer"])
        start, scale = ck["iteration"], ck["scale"]
        if (run / "best.pt").exists():
            best = torch.load(run / "best.pt", map_location="cpu", weights_only=False)["val_nM_RMSE_corr"]
        print(f"resumed at iteration {start} (best val so far {best:.4f})")

    def net_fn(X):                                                       # data units in and out (validation)
        Xp, (h, w) = pad_to_multiple(X.to(device) / scale, 32)
        return (net(Xp)[..., :h, :w] * scale).to(X.device)

    def validate(it):
        net.eval()
        v, per, raw = validation_score(net_fn, va_store, args.level)
        net.train()
        return v, per, raw

    log = open(run / "log.csv", "a")
    if start == 0:
        log.write("iteration,lr,train_loss,val_nM_RMSE_corr,val_raw_nM_RMSE_corr,seconds\n")
        v0, per0, raw0 = validate(0)
        print("validation, raw input frame (no denoising): " +
              "  ".join(f"{n[-3:]}={raw0[n]:.4f}" for n in raw0) + f"   mean={np.mean(list(raw0.values())):.4f}")
    amp = bool(tr.get("amp")) and device.type == "cuda"
    net.train()
    t0, run_loss, n_loss = time.perf_counter(), 0.0, 0
    for it in range(start, n_it):
        for g in opt.param_groups:
            g["lr"] = lr_at(it)
        x, y, m = sampler.batch(tr["batch"], scale)
        x, y, m = x.to(device, non_blocking=True), y.to(device, non_blocking=True), m.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            pred = net(x)
        loss = masked_mse(pred.float(), y, m)
        if not torch.isfinite(loss):
            raise SystemExit(f"non-finite loss at iteration {it}: check the data (saturation, calibration)")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        run_loss += float(loss.detach()); n_loss += 1
        if (it + 1) % tr["val_every"] == 0 or it + 1 == n_it:
            v, per, raw = validate(it + 1)
            el = time.perf_counter() - t0
            mark = ""
            if v < best:
                best, mark = v, "  * best"
                ckpt(it + 1, v, run / "best.pt")
            ckpt(it + 1, v, run / "last.pt")
            print(f"it {it + 1:6d}/{n_it}  lr {lr_at(it):.2e}  train loss {run_loss / n_loss:.5f}  "
                  f"val nM_RMSE_corr {v:.4f} (raw {np.mean(list(raw.values())):.4f})  "
                  + "  ".join(f"{n[-3:]}={per[n]:.4f}" for n in per) + f"  [{el / 60:.1f} min]{mark}", flush=True)
            log.write(f"{it + 1},{lr_at(it):.3e},{run_loss / n_loss:.6f},{v:.6f},"
                      f"{np.mean(list(raw.values())):.6f},{el:.1f}\n"); log.flush()
            run_loss, n_loss = 0.0, 0
    log.close()
    print(f"done. best val nM_RMSE_corr {best:.4f} -> {run / 'best.pt'}")


if __name__ == "__main__":
    sys.exit(main())
