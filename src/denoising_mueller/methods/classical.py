"""Classical denoisers for multi-channel images (16 intensities or 16 MM elements).

All filters work on (H, W, C) float arrays already scaled to unit noise per channel
(see `denoise`), so every parameter is expressed in units of the noise std.

Changes from the May 2026 intensity_denoise.py / notebook, on purpose:
- no 2-98 percentile clipping of inputs or [0, 1] clamping of outputs (it cut real MM tails);
- no Anscombe transform here: it is only valid on raw photon counts, so it belongs to the
  intensity-level pipeline (with the measured noise model), never to MM elements;
- BM3D runs channel by channel, sequentially (its C++ thread pool crashes when nested).
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.ndimage import gaussian_filter, median_filter

from .noise import sigma_per_channel


def _median(x, size=3):
    return median_filter(x, size=(int(size), int(size), 1), mode="reflect")


def _gaussian(x, sigma=1.0):
    return gaussian_filter(x, sigma=(float(sigma), float(sigma), 0), mode="reflect")


def _nlm(x, h=0.8, patch_size=5, patch_distance=6):
    """Joint NLM: patch distances computed over all channels together.
    h is relative to the (unit) noise std, as in skimage's convention h ~ 0.6-1.0 * sigma."""
    from skimage.restoration import denoise_nl_means
    return denoise_nl_means(x, h=float(h), sigma=1.0, patch_size=int(patch_size),
                            patch_distance=int(patch_distance), fast_mode=True, channel_axis=-1)


def _wavelet(x, scale=1.0, wavelet="db2", mode="soft"):
    """BayesShrink wavelet thresholding; `scale` multiplies the noise std given to the thresholding."""
    from skimage.restoration import denoise_wavelet
    out = np.empty_like(x)
    for c in range(x.shape[-1]):
        out[..., c] = denoise_wavelet(x[..., c], sigma=float(scale), wavelet=wavelet, mode=mode,
                                      method="BayesShrink", rescale_sigma=False)
    return out


def _tv(x, weight=0.5):
    """Chambolle total variation, channel by channel. weight in units of the noise std."""
    from skimage.restoration import denoise_tv_chambolle
    out = np.empty_like(x)
    for c in range(x.shape[-1]):
        out[..., c] = denoise_tv_chambolle(x[..., c], weight=float(weight))
    return out


def _bm3d(x, scale=1.0, profile="np"):
    """BM3D channel by channel (sequential), sigma_psd = scale * 1 (unit noise)."""
    import bm3d
    prof = {"np": bm3d.BM3DProfile(), "high": bm3d.BM3DProfileHigh(),
            "refilter": bm3d.BM3DProfileRefilter()}[profile]
    out = np.empty_like(x)
    for c in range(x.shape[-1]):
        out[..., c] = bm3d.bm3d(x[..., c], sigma_psd=float(scale), profile=prof,
                                stage_arg=bm3d.BM3DStages.ALL_STAGES)
    return out


FILTERS: dict[str, Callable] = {
    "median": _median,
    "gaussian": _gaussian,
    "nlm": _nlm,
    "wavelet": _wavelet,
    "tv": _tv,
    "bm3d": _bm3d,
}

# Main parameter swept for each method (name, candidate values). Tune on validation data only.
PARAM_GRIDS: dict[str, tuple[str, list]] = {
    "median": ("size", [3, 5, 7]),
    "gaussian": ("sigma", [0.5, 1.0, 1.5, 2.0, 3.0]),
    "nlm": ("h", [0.4, 0.6, 0.8, 1.0, 1.2]),
    "wavelet": ("scale", [0.5, 0.75, 1.0, 1.5, 2.0]),
    "tv": ("weight", [0.1, 0.25, 0.5, 1.0, 2.0]),
    "bm3d": ("scale", [0.5, 0.75, 1.0, 1.5, 2.0]),
}


def denoise(x: np.ndarray, method: str, mask: np.ndarray | None = None,
            sigma: np.ndarray | None = None, **params) -> tuple[np.ndarray, np.ndarray]:
    """Denoise (H, W, C) data with one classical method.

    Each channel is scaled to unit noise (sigma from the Haar-MAD estimator unless given),
    filtered, and scaled back. Returns (denoised, sigma_per_channel).
    """
    if method not in FILTERS:
        raise ValueError(f"Unknown method '{method}'. Available: {sorted(FILTERS)}")
    x = np.asarray(x, dtype=np.float64)
    if sigma is None:
        sigma = sigma_per_channel(x, mask)
    sigma = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-12)
    finite = np.isfinite(x)
    xin = np.where(finite, x, 0.0) / sigma
    y = FILTERS[method](xin, **params) * sigma
    return y, sigma
