import numpy as np
from scipy.ndimage import gaussian_filter

from denoising_mueller.data.registration import register


def _tissue(rng, h=128, w=128):
    t = gaussian_filter(rng.random((h, w)), 3) * 400 + 20
    return np.stack([t * (0.5 + 0.03 * c) for c in range(16)])


def test_recovers_known_shift_with_noise():
    rng = np.random.default_rng(0)
    base = _tissue(rng)
    moved = np.roll(base, (3, -5), axis=(1, 2))
    frames = rng.poisson(np.stack([base, moved])).astype(float)
    _, _, shifts = register(frames)
    assert tuple(shifts[1]) == (-3, 5)


def test_static_noisy_frames_get_no_shift():
    rng = np.random.default_rng(1)
    base = _tissue(rng) * 0.05            # low signal, like 1 ms acquisitions
    frames = rng.poisson(np.stack([base] * 4)).astype(float)
    _, _, shifts = register(frames)
    assert not shifts.any()
