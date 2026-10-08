import numpy as np

from denoising_mueller.evaluation.metrics import azimuth_error, rmse, ssim_masked


def test_azimuth_is_axial():
    a = np.array([[179.0, 1.0]])
    b = np.array([[1.0, 179.0]])
    mae, _ = azimuth_error(a, b, np.ones_like(a, bool))
    assert np.isclose(mae, 2.0)               # 179 vs 1 deg is a 2 deg error, not 178


def test_masked_metrics_ignore_outside():
    rng = np.random.default_rng(0)
    ref = rng.random((64, 64))
    pred = ref.copy()
    pred[:, :32] += 10                         # garbage outside the mask
    mask = np.zeros_like(ref, bool)
    mask[:, 40:] = True
    assert rmse(pred, ref, mask) == 0
    assert ssim_masked(pred, ref, mask, 1.0) > 0.99
