"""Metrics at MM-element and diagnostic-map levels.

Fixes relative to the May 2026 GUI metrics:
- ONE evaluation mask per acquisition, built from the reference only, used for every method
  (previously each method was scored on its own physical pixels);
- azimuth is compared as an axial angle (period 180 deg) and only where the reference
  retardance is high enough for an orientation to exist;
- SSIM is computed on the full image and averaged inside the mask (no fill values).
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_erosion
from skimage.metrics import structural_similarity

# Display ranges used as SSIM/PSNR data ranges.
MAP_RANGES = {"totD": 1.0, "totP": 1.0, "linR": 180.0, "M11": 255.0}


def evaluation_mask(ref_valid: np.ndarray, ref_M11: np.ndarray, sat_level: float = 250.0,
                    min_M11: float | None = None, border: int = 8) -> np.ndarray:
    """Pixels used for all metrics: physical in the reference, not saturated, above a minimum
    signal (default: 2nd percentile of M11), and away from the image border."""
    if min_M11 is None:
        min_M11 = float(np.nanpercentile(ref_M11, 2))
    m = ref_valid & np.isfinite(ref_M11) & (ref_M11 < sat_level) & (ref_M11 > min_M11)
    if border:
        m = binary_erosion(m, iterations=1, border_value=0)
        m[:border] = m[-border:] = False
        m[:, :border] = m[:, -border:] = False
    return m


def rmse(a, b, mask):
    d = (a - b)[mask]
    d = d[np.isfinite(d)]
    return float(np.sqrt(np.mean(d ** 2))) if d.size else np.nan


def psnr(a, b, mask, data_range):
    e = rmse(a, b, mask)
    return float(20 * np.log10(data_range / e)) if e and np.isfinite(e) and e > 0 else np.nan


def ssim_masked(a, b, mask, data_range):
    a = np.where(np.isfinite(a), a, 0.0)
    b = np.where(np.isfinite(b), b, 0.0)
    _, smap = structural_similarity(a, b, data_range=data_range, full=True, gaussian_weights=True,
                                    sigma=1.5, use_sample_covariance=False)
    return float(np.mean(smap[mask])) if mask.any() else np.nan


def azimuth_error(a_deg, b_deg, mask):
    """Mean and median absolute axial difference (degrees, in [0, 90])."""
    d = np.abs((a_deg - b_deg + 90.0) % 180.0 - 90.0)[mask]
    d = d[np.isfinite(d)]
    return (float(np.mean(d)), float(np.median(d))) if d.size else (np.nan, np.nan)


def mm_metrics(nM_pred, nM_ref, mask) -> dict:
    """Over the 15 normalized elements m12..m44 (m11 == 1 is excluded)."""
    r, p, s = [], [], []
    for k in range(1, 16):
        r.append(rmse(nM_pred[..., k], nM_ref[..., k], mask))
        p.append(psnr(nM_pred[..., k], nM_ref[..., k], mask, 2.0))
        s.append(ssim_masked(nM_pred[..., k], nM_ref[..., k], mask, 2.0))
    return {"nM_RMSE": float(np.nanmean(r)), "nM_PSNR": float(np.nanmean(p)), "nM_SSIM": float(np.nanmean(s))}


def map_metrics(maps_pred: dict, maps_ref: dict, mask, M11_pred=None, M11_ref=None,
                min_linR_for_azimuth: float = 10.0) -> dict:
    out = {}
    if M11_pred is not None and M11_ref is not None:
        out["M11_PSNR"] = psnr(M11_pred, M11_ref, mask, MAP_RANGES["M11"])
        out["M11_SSIM"] = ssim_masked(M11_pred, M11_ref, mask, MAP_RANGES["M11"])
    for k in ("totD", "totP", "linR"):
        out[f"{k}_RMSE"] = rmse(maps_pred[k], maps_ref[k], mask)
        out[f"{k}_SSIM"] = ssim_masked(maps_pred[k], maps_ref[k], mask, MAP_RANGES[k])
    az_mask = mask & (maps_ref["linR"] >= min_linR_for_azimuth)
    out["azimuth_MAE"], out["azimuth_MedAE"] = azimuth_error(maps_pred["azimuth"], maps_ref["azimuth"], az_mask)
    out["azimuth_n_px"] = int(az_mask.sum())
    return out
