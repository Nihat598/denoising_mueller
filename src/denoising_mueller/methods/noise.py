"""Noise-level estimation."""
from __future__ import annotations

import numpy as np


def sigma_haar_mad(img: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Donoho-Johnstone estimate of the noise std of a 2-D image.

    Uses the finest Haar diagonal detail HH = (a - b - c + d) / 2 on 2x2 blocks, which cancels
    locally linear signal; sigma = median(|HH|) / 0.6745. (Ported from intensity_denoise.py.)
    `mask` (same shape as img) restricts the estimate to blocks fully inside the mask.
    """
    h, w = (img.shape[0] // 2) * 2, (img.shape[1] // 2) * 2
    x = img[:h, :w].astype(np.float64)
    a, b, c, d = x[0::2, 0::2], x[0::2, 1::2], x[1::2, 0::2], x[1::2, 1::2]
    hh = (a - b - c + d) / 2.0
    sel = np.isfinite(hh)
    if mask is not None:
        m = mask[:h, :w]
        sel &= m[0::2, 0::2] & m[0::2, 1::2] & m[1::2, 0::2] & m[1::2, 1::2]
    return float(np.median(np.abs(hh[sel])) / 0.6745)


def sigma_per_channel(x: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """x: (H, W, C) -> (C,) noise std per channel."""
    return np.array([sigma_haar_mad(x[..., c], mask) for c in range(x.shape[-1])])
