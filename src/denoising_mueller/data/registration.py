"""Rigid (translation) registration of repeated frames.

Some acquisitions move between repeats (OR: breathing, pulsation; FF: slow drift of
~2 px over 16 frames was measured on 0000_089), so frames are aligned before
averaging or building Noise2Noise pairs.

Design choices (2026-10-08, after checking on real data):
- Shifts are estimated on the state-averaged intensity after light smoothing, by
  minimizing the mean squared mismatch over integer shifts. Sub-pixel phase
  correlation on raw low-signal frames returned false shifts of up to ~2 px on a
  static acquisition.
- Default is integer shifts (np.roll, no interpolation): interpolation averages
  neighboring pixels and lowers the measured noise, which biases noise statistics
  and the input/reference noise levels.
- A shift is only kept if it reduces the mismatch by `min_gain` versus no shift.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter


def _mismatch(a: np.ndarray, b: np.ndarray, dy: int, dx: int, margin: int) -> float:
    d = a - np.roll(b, (dy, dx), axis=(0, 1))
    return float(np.mean(d[margin:-margin, margin:-margin] ** 2))


def estimate_shifts(frames: np.ndarray, reference: int = 0, max_shift: int = 10,
                    smooth: int = 5, min_gain: float = 0.05) -> np.ndarray:
    """frames: (N, 16, H, W). Returns (N, 2) integer shifts (dy, dx) to apply to each frame."""
    intensity = uniform_filter(frames.mean(axis=1), size=(1, smooth, smooth))
    ref = intensity[reference]
    margin = max_shift + smooth
    shifts = np.zeros((len(frames), 2), int)
    for k in range(len(frames)):
        if k == reference:
            continue
        # coarse-to-fine is unnecessary at these sizes; exhaustive search over a small window
        best, best_err = (0, 0), _mismatch(ref, intensity[k], 0, 0, margin)
        err0 = best_err
        for dy in range(-max_shift, max_shift + 1):
            for dx in range(-max_shift, max_shift + 1):
                e = _mismatch(ref, intensity[k], dy, dx, margin)
                if e < best_err:
                    best, best_err = (dy, dx), e
        if best_err < (1 - min_gain) * err0:
            shifts[k] = best
    return shifts


def apply_shifts(frames: np.ndarray, shifts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Integer shift of each frame; returns (aligned_frames, valid_mask (H, W)).

    valid is False in the border strips that wrapped around for any frame.
    """
    n, c, h, w = frames.shape
    out = np.empty_like(frames)
    valid = np.ones((h, w), bool)
    for k in range(n):
        dy, dx = (int(s) for s in shifts[k])
        out[k] = np.roll(frames[k], (dy, dx), axis=(1, 2))
        if dy > 0:
            valid[:dy] = False
        elif dy < 0:
            valid[dy:] = False
        if dx > 0:
            valid[:, :dx] = False
        elif dx < 0:
            valid[:, dx:] = False
    return out, valid


def register(frames: np.ndarray, reference: int = 0, **kw) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shifts = estimate_shifts(frames, reference, **kw)
    aligned, valid = apply_shifts(frames, shifts)
    return aligned, valid, shifts
