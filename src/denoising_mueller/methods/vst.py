"""Variance-stabilizing transform for raw camera counts.

Noise model (measured, see docs/data_notes.md):  var(y) = a * E[y] + b   (counts^2)
Generalized Anscombe transform (Starck/Murtagh form) for that model:

    f(y) = (2 / a) * sqrt(a * y + 3/8 * a^2 + b)        -> noise std ~ 1

Valid on RAW INTENSITIES only (Poisson-Gaussian counts), never on MM elements.
The inverse used here is the algebraic one; at the low counts of the 1 ms data it is slightly
biased, which is why the evaluation always compares in the original count domain.
"""
from __future__ import annotations

import numpy as np


def gat_forward(y, a, b):
    return (2.0 / a) * np.sqrt(np.maximum(a * y + 0.375 * a * a + b, 0.0))


def gat_inverse(z, a, b):
    return ((a * z / 2.0) ** 2 - 0.375 * a * a - b) / a


def fit_noise_model(frames: np.ndarray, valid: np.ndarray | None = None, sat: float = 255.0,
                    min_px_per_bin: int = 2000) -> tuple[float, float]:
    """Fit var = a*mean + b from repeated, aligned frames (N, C, H, W), N >= 2.

    Uses half the variance of consecutive-frame differences (insensitive to the scene),
    medians in quantile bins of the mean, then a straight line through the bin medians.
    """
    if len(frames) < 2:
        raise ValueError("need at least 2 frames")
    diffs = frames[1:] - frames[:-1]
    means = 0.5 * (frames[1:] + frames[:-1])
    ok = np.isfinite(diffs) & (np.maximum(frames[1:], frames[:-1]) < sat)
    if valid is not None:
        ok &= valid[None, None]
    m = means[ok].ravel()
    d2 = 0.5 * diffs[ok].ravel() ** 2                  # E[d^2]/2 = var for independent frames
    edges = np.unique(np.percentile(m, np.linspace(0.5, 99.5, 41)))
    xs, ys = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (m >= lo) & (m < hi)
        if s.sum() >= min_px_per_bin:
            xs.append(np.median(m[s]))
            ys.append(np.mean(d2[s]))                  # mean, not median: d^2 is chi-square, median is biased
    a, b = np.polyfit(np.array(xs), np.array(ys), 1)
    return float(max(a, 1e-6)), float(max(b, 0.0))
