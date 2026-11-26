"""Data for Noise2Noise training: splits with leak checks, in-memory acquisitions, pair sampling, validation.

Split rules (configs/n2n.yaml)
- Objects (new data) are split BY OBJECT: all views of one object are in the same group, and a group is either
  train, val or test. A random split by acquisition would put a second view of a test object into training.
- Brain is split BY SPECIMEN (two-fold): a fold trains on one specimen (minus one view kept for validation)
  and is tested on the other specimen. The network never sees a test specimen, not even its noisy frames.
- `build_split` checks that train, val and test do not overlap and that no group or brain specimen is split
  across them; the checkpoint stores the split, and evaluation refuses any acquisition in train/val.

Training pairs: two different frames i != j of the same registered acquisition (Noise2Noise), or optionally
frame i vs the mean of k other frames (still a noisy, unbiased target; lower variance). Loss mask: registration
border strips and pixels saturated in the input or in any target frame are excluded.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .io import find_acquisition
from .registration import apply_shifts, estimate_shifts
from ..polarimetry.reconstruction import find_calibration, load_calibration


# ------------------------------------------------------------------------------------------------- splits
def build_split(cfg: dict, fold: str) -> dict:
    """Return {"train": [...], "val": [...], "test": [...], "brain": set, ...} of acquisition names for a fold."""
    excluded = set(cfg.get("exclude", []))
    groups = {g: list(v) for g, v in cfg["object_groups"].items()}
    specimens = {s: list(v) for s, v in cfg["brain_specimens"].items()}
    # every acquisition belongs to exactly one group / specimen
    seen = {}
    for kind, d in (("object group", groups), ("brain specimen", specimens)):
        for g, names in d.items():
            for n in names:
                if n in seen:
                    raise ValueError(f"{n} is in {seen[n]} and in {kind} {g}")
                seen[n] = f"{kind} {g}"
    f = cfg["folds"][fold]
    test_groups, val_groups = set(cfg["objects_test"]), set(cfg["objects_val"])
    if test_groups & val_groups:
        raise ValueError(f"object groups both val and test: {test_groups & val_groups}")
    for g in test_groups | val_groups:
        if g not in groups:
            raise ValueError(f"unknown object group {g}")
    obj_test = [n for g in groups if g in test_groups for n in groups[g]]
    obj_val = [n for g in groups if g in val_groups for n in groups[g]]
    obj_train = [n for g in groups if g not in test_groups | val_groups for n in groups[g]]

    train_spec, test_spec = f.get("brain_train", []), f.get("brain_test", [])
    if set(train_spec) & set(test_spec):
        raise ValueError(f"fold {fold}: brain specimen both train and test")
    brain_val = list(f.get("brain_val", []))
    for n in brain_val:                                    # a val view must come from a TRAINING specimen
        if not any(n in specimens[s] for s in train_spec):
            raise ValueError(f"fold {fold}: brain_val {n} is not a view of a training specimen")
    brain_train = [n for s in train_spec for n in specimens[s] if n not in brain_val]
    brain_test = [n for s in test_spec for n in specimens[s]]

    split = {"train": [n for n in obj_train + brain_train if n not in excluded],
             "val": [n for n in obj_val + brain_val if n not in excluded],
             "test": [n for n in obj_test + brain_test if n not in excluded]}
    a, b, c = (set(split[k]) for k in ("train", "val", "test"))
    if a & b or a & c or b & c:
        raise ValueError(f"fold {fold}: overlapping splits {(a & b) | (a & c) | (b & c)}")
    split["brain"] = sorted(n for v in specimens.values() for n in v)
    split["fold"] = fold
    return split


# ------------------------------------------------------------------------------------------- acquisitions
@dataclass
class N2NAcq:
    name: str
    frames: np.ndarray        # (N, 16, H, W) float16 raw counts, registered onto frame 1
    valid: np.ndarray         # (H, W) bool, False in registration wrap-around strips
    sat: np.ndarray           # (N, H, W) bool, True where a frame has any saturated channel
    Ainv: np.ndarray          # (4, 4, H, W) float32
    Winv: np.ndarray          # (4, 4, H, W) float32
    dark: np.ndarray          # (16, H, W) float32 (zeros if no dark image or subtract_dark is False)
    is_brain: bool
    max_shift: int
    eval_mask: np.ndarray | None = None     # validation only: the evaluation mask of pipeline.prepare
    ref_nM: np.ndarray | None = None        # validation only: (16, H, W) reference nM (mean of frames N/2+1..N)
    ref_nM_var: np.ndarray | None = None    # validation only: (15,) noise variance of the reference nM elements


def _shift_cache(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else {}


def load_acq(name: str, paths: dict, cfg: dict, brain: set, cache_file: str | Path | None = None,
             validation: bool = False) -> N2NAcq:
    """Load frames, register them onto frame 1 (same registration as the evaluation pipeline), saturation masks,
    calibration inverses and dark image. Shifts are cached in cache_file (registration is the slow part).
    validation=True also runs the evaluation's own preparation (pipeline.prepare) to get its mask, reference nM and
    reference noise, so checkpoints are selected with exactly the score used at test time."""
    from ..methods.n2n import inverse_mats
    wl = cfg.get("wavelength", 630)
    acq = find_acquisition(paths["data_root"], name, wl)
    frames = acq.load_frames(np.float32)                                        # (N, 16, H, W)
    cache = _shift_cache(cache_file) if cache_file else {}
    if not cfg.get("register", True):
        shifts = np.zeros((len(frames), 2), int)
    elif name in cache and len(cache[name]) == len(frames):
        shifts = np.asarray(cache[name], int)
    else:
        shifts = estimate_shifts(frames)
        if cache_file:
            cache = _shift_cache(cache_file)
            cache[name] = shifts.tolist()
            Path(cache_file).parent.mkdir(parents=True, exist_ok=True)
            Path(cache_file).write_text(json.dumps(cache, indent=0))
    frames, valid = apply_shifts(frames, shifts)
    sat = frames.max(axis=1) >= cfg.get("saturation", 255)
    calib = load_calibration(find_calibration(acq.path, paths["calib_root"]), wl)
    dark = calib["dark"] if cfg.get("subtract_dark", True) else None
    if dark is None:
        dark = np.zeros(frames.shape[1:], np.float32)
    else:
        dark = np.asarray(dark, np.float32)
        dark = np.broadcast_to(dark[None] if dark.ndim == 2 else np.moveaxis(dark, -1, 0), frames.shape[1:])
    out = N2NAcq(name, frames.astype(np.float16), valid, sat, inverse_mats(calib["A"]), inverse_mats(calib["W"]),
                 np.ascontiguousarray(dark, np.float32), name in brain, int(np.abs(shifts).max()))
    if validation:
        from .. import pipeline as pl
        p = pl.prepare(name, frames, calib, register_frames=False, subtract_dark=cfg.get("subtract_dark", True),
                       sat=cfg.get("saturation", 255), physical_tol=cfg.get("physical_tol", "auto"))
        out.eval_mask = p.mask & valid
        out.ref_nM = np.ascontiguousarray(np.moveaxis(p.ref["nM"], -1, 0), np.float32)
        out.ref_nM_var = np.asarray(p.ref_nM_var, np.float32)
    return out


# ---------------------------------------------------------------------------------------- torch side
class Store:
    """All acquisitions of one split as torch tensors on one device (GPU if it fits)."""

    def __init__(self, acqs: list[N2NAcq], device):
        import torch
        self.names = [a.name for a in acqs]
        self.is_brain = np.array([a.is_brain for a in acqs])
        t = lambda x, dt=None: torch.from_numpy(np.ascontiguousarray(x)).to(device, dt)
        self.frames = [t(a.frames) for a in acqs]
        self.valid = [t(a.valid) for a in acqs]
        self.sat = [t(a.sat) for a in acqs]
        self.Ainv = [t(a.Ainv) for a in acqs]
        self.Winv = [t(a.Winv) for a in acqs]
        self.dark = [t(a.dark) for a in acqs]
        self.eval_mask = [t(a.eval_mask) if a.eval_mask is not None else None for a in acqs]
        self.ref_nM = [t(a.ref_nM) if a.ref_nM is not None else None for a in acqs]
        self.ref_nM_var = [t(a.ref_nM_var) if a.ref_nM_var is not None else None for a in acqs]
        self.device = device

    def to_level(self, I, k, level, ys=slice(None), xs=slice(None)):
        """Raw intensities (B, 16, h, w) of acquisition k -> network domain (intensity: unchanged; mm: M)."""
        if level == "intensity":
            return I
        from ..methods.n2n import mm_from_intensities_t
        b = I.shape[0]
        exp = lambda x: x[..., ys, xs].unsqueeze(0).expand(b, *x[..., ys, xs].shape)
        return mm_from_intensities_t(I, exp(self.Ainv[k]), exp(self.Winv[k]), exp(self.dark[k]))


class PairSampler:
    """Random Noise2Noise training patches: input frame i, target = frame j != i (or mean of k frames != i)."""

    def __init__(self, store: Store, level: str, patch: int, brain_fraction: float | None = None,
                 target_frames: int = 1, min_valid: float = 0.5, seed: int = 0):
        import torch
        self.s, self.level, self.patch, self.k = store, level, patch, int(target_frames)
        self.min_valid = min_valid
        self.rng = np.random.default_rng(seed)
        n = len(store.names)
        w = np.ones(n)
        if brain_fraction is not None and store.is_brain.any() and (~store.is_brain).any():
            w[store.is_brain] = brain_fraction / store.is_brain.sum()
            w[~store.is_brain] = (1 - brain_fraction) / (~store.is_brain).sum()
        self.p = w / w.sum()
        for k, f in enumerate(store.frames):
            if f.shape[0] < self.k + 1:
                raise ValueError(f"{store.names[k]}: {f.shape[0]} frames, need >= {self.k + 1}")
            if min(f.shape[-2:]) < patch:
                raise ValueError(f"{store.names[k]}: image smaller than the patch size {patch}")
        self.torch = torch

    def _one(self):
        torch = self.torch
        for _ in range(20):
            k = self.rng.choice(len(self.p), p=self.p)
            fr = self.s.frames[k]
            n, _, H, W = fr.shape
            idx = self.rng.permutation(n)[: self.k + 1]
            i, js = int(idx[0]), [int(j) for j in idx[1:]]                       # i never in js
            y0, x0 = self.rng.integers(0, H - self.patch + 1), self.rng.integers(0, W - self.patch + 1)
            ys, xs = slice(y0, y0 + self.patch), slice(x0, x0 + self.patch)
            m = self.s.valid[k][ys, xs] & ~self.s.sat[k][i, ys, xs]
            for j in js:
                m = m & ~self.s.sat[k][j, ys, xs]
            if m.float().mean() >= self.min_valid:
                break                                                            # else redraw (mostly saturated)
        x = fr[i, :, ys, xs].float()
        y = torch.stack([fr[j, :, ys, xs].float() for j in js]).mean(0)
        return k, x, y, m, ys, xs                                                # the mask handles what is left

    def batch(self, size: int, scale: float):
        torch = self.torch
        xs_, ys_, ms_ = [], [], []
        for _ in range(size):
            k, x, y, m, ys, xs = self._one()
            xy = self.s.to_level(torch.stack([x, y]), k, self.level, ys, xs)
            xs_.append(xy[0]); ys_.append(xy[1]); ms_.append(m)
        return torch.stack(xs_) / scale, torch.stack(ys_) / scale, torch.stack(ms_)[:, None].float()


def masked_mse(pred, target, mask):
    """Mean squared error over valid pixels and all 16 channels. mask: (B, 1, h, w) in {0, 1}."""
    d = (pred - target) ** 2 * mask
    return d.sum() / (mask.sum() * pred.shape[1]).clamp_min(1.0)


def estimate_scale(store: Store, level: str, q: float = 99.0, n: int = 64, seed: int = 0) -> float:
    """Data scale for the network input: q-th percentile of |values| over random frames of the training data."""
    import torch
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        k = rng.integers(len(store.names))
        i = rng.integers(store.frames[k].shape[0])
        I = store.frames[k][i:i + 1].float()
        X = store.to_level(I, k, level)[0]
        ok = store.valid[k] & ~store.sat[k][i]
        v = X[:, ok].abs().flatten()
        vals.append(v[torch.randperm(v.numel(), device=v.device)[:20000]].cpu().numpy())
    return float(np.percentile(np.concatenate(vals), q))


def validation_score(net_fn, store: Store, level: str):
    """Exactly the evaluation protocol and score (scripts/compare_levels.py): input = frame 1, reference = mean of
    frames N/2+1..N, mask and reference noise from pipeline.prepare (store loaded with validation=True),
    score = nM_RMSE_corr = mean over the 15 normalized elements of sqrt(max(MSE vs reference - reference noise
    variance, 0)). Returns (mean over acquisitions, {name: score}, {name: same score for the raw input frame})."""
    import torch
    from ..methods.n2n import mm_from_intensities_t, normalized_t
    per, raw = {}, {}
    for k, name in enumerate(store.names):
        if store.eval_mask[k] is None:
            raise ValueError(f"{name}: validation data must be loaded with load_acq(..., validation=True)")
        x_raw = store.frames[k][:1].float()
        A, Wi, D = store.Ainv[k][None], store.Winv[k][None], store.dark[k][None]
        X = store.to_level(x_raw, k, level)
        with torch.no_grad():
            Y = net_fn(X)
        M_pred = mm_from_intensities_t(Y, A, Wi, D) if level == "intensity" else Y
        M_raw = mm_from_intensities_t(x_raw, A, Wi, D)
        mask, ref, var = store.eval_mask[k], store.ref_nM[k][1:], store.ref_nM_var[k]

        def score(M):
            d = (normalized_t(M)[0, 1:] - ref)[:, mask].double()               # (15, n_px)
            ok = torch.isfinite(d)
            mse = torch.where(ok, d, torch.zeros_like(d)).pow(2).sum(1) / ok.sum(1).clamp_min(1)
            return float(torch.sqrt(torch.clamp(mse - var.double(), min=0.0)).mean())
        per[name], raw[name] = score(M_pred), score(M_raw)
    return float(np.mean(list(per.values()))), per, raw
