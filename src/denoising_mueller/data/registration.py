"""Rigid (translation) registration of repeated frames.

OR acquisitions move between repeats (breathing, pulsation), so frames
must be aligned before averaging or building Noise2Noise pairs.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import shift as nd_shift
from skimage.filters import window
from skimage.registration import phase_cross_correlation


def estimate_shifts(frames: np.ndarray, reference: int = 0, upsample: int = 10) -> np.ndarray:
    """frames: (N, 16, H, W). Returns (N, 2) sub-pixel shifts (dy, dx) to apply to each frame.

    Shifts are estimated on the mean over the 16 polarization states.
    """
    intensity = frames.mean(axis=1)
    win = window("hann", intensity.shape[1:])   # suppresses field-of-view edge effects
    prep = lambda im: (im - im.mean()) * win
    ref = prep(intensity[reference])
    shifts = np.zeros((len(frames), 2))
    for k in range(len(frames)):
        if k == reference:
            continue
        s, _, _ = phase_cross_correlation(ref, prep(intensity[k]), upsample_factor=upsample)
        shifts[k] = s
    return shifts


def apply_shifts(frames: np.ndarray, shifts: np.ndarray, order: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Shift each frame; returns (aligned_frames, valid_mask (H, W)).

    The mask is False where any frame was shifted in from outside the field of view.
    """
    n, c, h, w = frames.shape
    out = np.empty_like(frames)
    valid = np.ones((h, w), bool)
    for k in range(n):
        for ch in range(c):
            out[k, ch] = nd_shift(frames[k, ch], shifts[k], order=order, mode="constant", cval=np.nan)
        valid &= ~np.isnan(out[k, 0])
    out = np.nan_to_num(out)
    return out, valid


def register(frames: np.ndarray, reference: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shifts = estimate_shifts(frames, reference)
    aligned, valid = apply_shifts(frames, shifts)
    return aligned, valid, shifts
