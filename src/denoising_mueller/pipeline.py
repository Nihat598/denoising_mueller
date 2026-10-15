"""Shared steps for the intensity-level vs MM-level comparison (scripts/compare_levels.py).

One acquisition = N repeated raw frames (N, 16, H, W) + its calibration (A, W, dark).
  input      : frame 1 (single shot, raw counts)
  reference  : mean of the second half of the frames -> reference MM -> reference maps
  intensity path: VST -> filter 16 raw intensities -> inverse VST -> MM -> maps
  MM path       : MM of frame 1 -> filter 16 unnormalized MM elements -> renormalize -> maps
Both paths are scored at intensity, MM-element and map level against the same reference,
on one mask defined by the reference only.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .data.registration import register
from .evaluation.metrics import evaluation_mask, map_metrics, mm_metrics, psnr, rmse, ssim_masked
from .methods.classical import denoise
from .methods.vst import fit_noise_model, gat_forward, gat_inverse
from .polarimetry.lu_chipman import physical_mask, polarimetric_maps, to_matrix
from .polarimetry.reconstruction import intensities_from_mm, reconstruct_mm


@dataclass
class Prepared:
    name: str
    frames: np.ndarray                 # (N, 16, H, W) raw counts, registered
    inp: np.ndarray                    # (H, W, 16) frame 1
    ref_I: np.ndarray                  # (H, W, 16) mean of reference frames
    A: np.ndarray
    W: np.ndarray
    dark: np.ndarray | None
    ref: dict                          # reconstructed reference: M, nM, M11
    ref_maps: dict
    mask: np.ndarray                   # (H, W) evaluation mask
    noise: tuple[float, float]         # var = a*I + b on raw counts
    info: dict = field(default_factory=dict)
    truth: dict | None = None
    ref_nM_var: np.ndarray | None = None   # (15,) noise variance of the reference nM elements          # synthetic data only: {'nM', 'maps'}


def _crop(x, n, axes=(-2, -1)):
    if not n:
        return x
    h, w = x.shape[axes[0]], x.shape[axes[1]]
    y0, x0 = max((h - n) // 2, 0), max((w - n) // 2, 0)
    sl = [slice(None)] * x.ndim
    sl[axes[0]] = slice(y0, y0 + n)
    sl[axes[1]] = slice(x0, x0 + n)
    return x[tuple(sl)]


def _physical(nM, tol):
    M = to_matrix(nM)
    with np.errstate(divide="ignore", invalid="ignore"):
        M = M / M[..., :1, :1]
    return physical_mask(M, tol=tol)


def prepare(name, frames, calib, *, register_frames=True, subtract_dark=True, sat=255.0,
            crop=0, border=8, truth_M=None, physical_tol="auto") -> Prepared:
    """physical_tol: Cloude eigenvalues down to -physical_tol count as physical. Non-depolarizing
    samples (waveplates) sit on the boundary (eigenvalues 1, 0, 0, 0), so any fixed small tolerance
    rejects about half of their pixels once noise is present. "auto" = 3 x the measured noise rms of
    the reference nM elements (min 0.01): noise is tolerated, grossly wrong matrices are not."""
    frames = np.asarray(frames, dtype=np.float64)
    shifts = np.zeros((len(frames), 2), int)
    valid = np.ones(frames.shape[-2:], bool)
    if register_frames:
        frames, valid, shifts = register(frames)
    A, W, dark = calib["A"], calib["W"], calib.get("dark")
    frames, valid = _crop(frames, crop), _crop(valid, crop)
    A, W = _crop(A, crop, (0, 1)), _crop(W, crop, (0, 1))
    if dark is not None:
        dark = _crop(dark, crop, (0, 1))
    if not subtract_dark:
        dark = None

    n = len(frames)
    half = n // 2
    inp = np.moveaxis(frames[0], 0, -1)
    ref_frames = frames[half:]
    ref_I = np.moveaxis(ref_frames.mean(0), 0, -1)
    ref = reconstruct_mm(ref_I, A, W, dark)
    ref_maps = polarimetric_maps(ref["nM"])
    not_sat = (frames.max(axis=0) < sat).all(axis=0)
    noise = fit_noise_model(ref_frames, valid & not_sat, sat)

    # Noise variance of the reference itself, from two disjoint halves of the reference frames:
    # ref = (h1 + h2) / 2, so var(ref noise) = E[(h1 - h2)^2] / 4. Used to correct the scores:
    # MSE(pred, truth) ~ MSE(pred, ref) - var(ref noise) (input and reference noise independent).
    q = len(ref_frames) // 2
    h1 = reconstruct_mm(np.moveaxis(ref_frames[:q].mean(0), 0, -1), A, W, dark)["nM"]
    h2 = reconstruct_mm(np.moveaxis(ref_frames[q:2 * q].mean(0), 0, -1), A, W, dark)["nM"]
    region = valid & not_sat & (ref["M11"] > np.nanpercentile(ref["M11"], 2))
    d = ((h1 - h2)[..., 1:][region]) ** 2 / 4.0                     # (n_px, 15)
    ref_nM_var = np.nanmean(d, axis=0) * (2 * q / len(ref_frames))  # per element; scale if count is odd
    if physical_tol == "auto":
        physical_tol = max(0.01, 3.0 * float(np.sqrt(np.mean(ref_nM_var))))
    ref_phys = _physical(ref["nM"], physical_tol)
    mask = evaluation_mask(ref_phys & valid & not_sat, ref["M11"], sat_level=np.inf, border=border)
    info = dict(n_frames=n, ref_frames=f"{half + 1}-{n}", max_shift=int(np.abs(shifts).max()),
                ref_nM_noise_rms=float(np.mean(np.sqrt(ref_nM_var))),
                ref_physical=float(ref_phys[valid].mean()), physical_tol=physical_tol, eval_px=int(mask.sum()),
                noise_a=noise[0], noise_b=noise[1])
    truth = None
    if truth_M is not None:
        tM = _crop(np.asarray(truth_M, dtype=np.float64), crop, (0, 1))
        with np.errstate(divide="ignore", invalid="ignore"):
            t_nM = tM / tM[..., :1]
        truth = {"nM": t_nM, "maps": polarimetric_maps(t_nM)}
        info["ref_vs_truth_nM_RMSE"] = rmse(ref["nM"][..., 1:], t_nM[..., 1:], mask[..., None].repeat(15, -1))
    return Prepared(name, frames, inp, ref_I, A, W, dark, ref, ref_maps, mask, noise, info, truth, ref_nM_var)


def score(p: Prepared, I_pred, nM, M11, maps) -> dict:
    """All metrics for one prediction (intensities in raw counts, (H, W, 16))."""
    out = {}
    r, s_, q = [], [], []
    for c in range(16):
        r.append(rmse(I_pred[..., c], p.ref_I[..., c], p.mask))
        q.append(psnr(I_pred[..., c], p.ref_I[..., c], p.mask, 255.0))
        s_.append(ssim_masked(I_pred[..., c], p.ref_I[..., c], p.mask, 255.0))
    out.update(I_RMSE=float(np.nanmean(r)), I_PSNR=float(np.nanmean(q)), I_SSIM=float(np.nanmean(s_)))
    out.update(mm_metrics(nM, p.ref["nM"], p.mask))
    # reference-noise-corrected estimate of the error vs the (unknown) truth, per element
    mse = np.array([np.nanmean((nM[..., k] - p.ref["nM"][..., k])[p.mask] ** 2) for k in range(1, 16)])
    out["nM_RMSE_corr"] = float(np.mean(np.sqrt(np.maximum(mse - p.ref_nM_var, 0.0))))
    out.update(map_metrics(maps, p.ref_maps, p.mask, M11, p.ref["M11"]))
    out["physical_frac"] = float(_physical(nM, p.info["physical_tol"])[p.mask].mean())
    if p.truth is not None:                                  # synthetic data: error vs ground truth
        t = map_metrics(maps, p.truth["maps"], p.mask)
        out["truth_nM_RMSE"] = mm_metrics(nM, p.truth["nM"], p.mask)["nM_RMSE"]
        out["truth_linR_RMSE"], out["truth_azimuth_MAE"] = t["linR_RMSE"], t["azimuth_MAE"]
        out["truth_totP_RMSE"] = t["totP_RMSE"]
    return out


def from_intensities(p: Prepared, I):
    mm = reconstruct_mm(I, p.A, p.W, p.dark)
    return mm, polarimetric_maps(mm["nM"])


def baseline_none(p: Prepared):
    mm, maps = from_intensities(p, p.inp)
    return p.inp, mm, maps


def baseline_average(p: Prepared, m: int):
    I = np.moveaxis(p.frames[:m].mean(0), 0, -1)
    mm, maps = from_intensities(p, I)
    return I, mm, maps


def intensity_path(p: Prepared, method: str, params: dict, vst: bool = True):
    """Filter the 16 raw intensities, then compute MM and maps."""
    t = time.time()
    a, b = p.noise
    fg = p.mask
    if vst:
        z = gat_forward(p.inp, a, b)
        y, _ = denoise(z, method, mask=fg, noise_sigma=np.ones(16), **params)
        I = gat_inverse(y, a, b)
    else:
        I, _ = denoise(p.inp, method, mask=fg, **params)
    mm, maps = from_intensities(p, I)
    return I, mm, maps, time.time() - t


def mm_path(p: Prepared, method: str, params: dict):
    """MM of frame 1, filter the 16 unnormalized elements, renormalize, maps.
    The denoised MM is also mapped back to intensities (A M W + dark) for intensity metrics."""
    t = time.time()
    mm0 = reconstruct_mm(p.inp, p.A, p.W, p.dark)
    M_den, _ = denoise(mm0["M"], method, mask=p.mask, **params)
    M11 = M_den[..., 0].copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        nM = M_den / M11[..., None]
    maps = polarimetric_maps(nM)
    I = intensities_from_mm(M_den, p.A, p.W)
    if p.dark is not None:
        I = I + (p.dark[..., None] if p.dark.ndim == 2 else p.dark)
    return I, {"M": M_den, "nM": nM, "M11": M11}, maps, time.time() - t
