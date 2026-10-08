"""Build input / reference pairs from repeated frames.

Rules (see docs/data_notes.md):
- References are float averages of raw frames, never the stored uint8 HQ.
- An input frame is never part of its own reference.
"""
from __future__ import annotations

import numpy as np


def reference_excluding(frames: np.ndarray, k: int) -> np.ndarray:
    """Average of all frames except frame k. frames: (N, 16, H, W)."""
    n = len(frames)
    return (frames.sum(axis=0) - frames[k]) / (n - 1)


def split_inputs_reference(frames: np.ndarray, n_inputs: int | None = None):
    """Disjoint split: first half = candidate inputs, second half = reference.

    Returns (inputs (n_inputs, 16, H, W), reference (16, H, W)).
    """
    n = len(frames)
    n_inputs = n // 2 if n_inputs is None else n_inputs
    if not 0 < n_inputs < n:
        raise ValueError(f"n_inputs must be in 1..{n - 1}")
    return frames[:n_inputs], frames[n_inputs:].mean(axis=0)


def average_of_first(frames: np.ndarray, m: int) -> np.ndarray:
    """Plain averaging of the first m frames (the 1/2/4/8-frame baseline)."""
    return frames[:m].mean(axis=0)
