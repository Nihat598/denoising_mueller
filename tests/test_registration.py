import numpy as np
from scipy.ndimage import shift as nd_shift

from denoising_mueller.data.registration import register


def test_recovers_known_shift():
    rng = np.random.default_rng(0)
    from scipy.ndimage import gaussian_filter
    tissue = gaussian_filter(rng.random((128, 128)), 3) * 100     # structure shared by all states
    base = np.stack([tissue * (0.5 + 0.03 * c) for c in range(16)]).astype(np.float32)
    moved = np.stack([nd_shift(c, (3, -5), order=1) for c in base])
    _, _, shifts = register(np.stack([base, moved]))
    assert np.allclose(shifts[1], (-3, 5), atol=0.3)
