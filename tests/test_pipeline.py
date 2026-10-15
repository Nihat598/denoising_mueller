"""End-to-end check of the intensity-vs-MM pipeline on a small synthetic acquisition."""
import numpy as np
from scipy.ndimage import gaussian_filter

from denoising_mueller import pipeline as pl
from denoising_mueller.polarimetry.reconstruction import intensities_from_mm, mat_to_vec


def _acquisition(rng, H=64, W=72, n=8, a=0.034, b=0.18):
    base_A = np.array([[1, .9, .3, .2], [1, -.4, .8, .3], [1, -.3, -.6, .7], [1, .2, -.4, -.85]]) * .5
    base_W = np.array([[1, 1, 1, 1], [.85, -.45, -.35, .25], [.35, .75, -.65, -.4], [.25, .3, .55, -.8]]) * .5
    A = mat_to_vec(np.broadcast_to(base_A, (H, W, 4, 4)).copy())
    Wc = mat_to_vec(np.broadcast_to(base_W, (H, W, 4, 4)).copy())
    p = 0.2 + 0.6 * (gaussian_filter(rng.random((H, W)), 6) - 0.4) / 0.2
    M = np.zeros((H, W, 4, 4)); M[..., 0, 0] = 1
    for k in (1, 2, 3):
        M[..., k, k] = np.clip(1 - p, 0.05, 0.95)
    M *= 40
    dark = np.full((H, W, 16), 2.0)
    clean = intensities_from_mm(mat_to_vec(M), A, Wc) + dark
    frames = np.stack([np.clip(np.round(a * rng.poisson(clean / a) + rng.normal(0, np.sqrt(b), clean.shape)), 0, 255)
                       for _ in range(n)]).transpose(0, 3, 1, 2)
    return frames, {"A": A, "W": Wc, "dark": dark}, mat_to_vec(M)


def test_pipeline_runs_and_filters_help():
    rng = np.random.default_rng(0)
    frames, calib, M = _acquisition(rng)
    p = pl.prepare("synthetic", frames, calib, truth_M=M, border=4)
    assert p.mask.sum() > 1000
    I0, mm0, maps0 = pl.baseline_none(p)
    none = pl.score(p, I0, mm0["nM"], mm0["M11"], maps0)
    I, mm, maps, _ = pl.intensity_path(p, "gaussian", {"sigma": 1.0})
    s_int = pl.score(p, I, mm["nM"], mm["M11"], maps)
    I, mm, maps, _ = pl.mm_path(p, "gaussian", {"sigma": 1.0})
    s_mm = pl.score(p, I, mm["nM"], mm["M11"], maps)
    for s in (s_int, s_mm):
        assert s["truth_nM_RMSE"] < 0.7 * none["truth_nM_RMSE"]      # filtering helps vs ground truth
    # the reference-noise correction tracks the true error
    for s in (none, s_int, s_mm):
        assert abs(s["nM_RMSE_corr"] - s["truth_nM_RMSE"]) < 0.25 * s["truth_nM_RMSE"] + 0.003
