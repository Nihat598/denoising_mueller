"""Noise2Noise: Mueller algebra in torch, network shapes, split rules, pair sampling, validation score, leak guard."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from denoising_mueller import pipeline as pl                                               # noqa: E402
from denoising_mueller.data.io import find_acquisition                                      # noqa: E402
from denoising_mueller.data.n2n_data import (N2NAcq, PairSampler, Store, build_split, load_acq,  # noqa: E402
                                             validation_score)
from denoising_mueller.methods.n2n import (N2NModel, UNet, inverse_mats, mm_from_intensities_t,  # noqa: E402
                                           pad_to_multiple)
from denoising_mueller.polarimetry.reconstruction import (find_calibration, load_calibration,  # noqa: E402
                                                          reconstruct_mm)

ROOT = Path(__file__).resolve().parents[1]


def _calib(rng, H, W):
    base = np.eye(4) * 2 + 0.3 * rng.normal(size=(4, 4))
    vec = lambda m: m.T.reshape(16)                                    # lab layout: column-major
    A = np.broadcast_to(vec(base), (H, W, 16)) + 0.01 * rng.normal(size=(H, W, 16))
    Wc = np.broadcast_to(vec(base.T), (H, W, 16)) + 0.01 * rng.normal(size=(H, W, 16))
    return A, Wc


def test_mm_torch_matches_numpy():
    rng = np.random.default_rng(0)
    H, W = 12, 10
    A, Wc = _calib(rng, H, W)
    I = rng.uniform(5, 200, (H, W, 16))
    dark = np.full((H, W, 16), 2.0)
    ref = reconstruct_mm(I, A, Wc, dark)["M"]
    t = lambda a: torch.from_numpy(np.moveaxis(a, -1, 0)[None].astype(np.float64))
    M = mm_from_intensities_t(t(I), torch.from_numpy(inverse_mats(A)).double()[None],
                              torch.from_numpy(inverse_mats(Wc)).double()[None], t(dark))
    assert np.allclose(np.moveaxis(M[0].numpy(), 0, -1), ref, rtol=1e-5, atol=1e-5 * np.abs(ref).max())


def test_unet_shapes_and_padding():
    net = UNet(16, 16)
    for h, w in ((64, 96), (50, 70)):
        x = torch.randn(1, 16, h, w)
        xp, (h0, w0) = pad_to_multiple(x, 32)
        assert xp.shape[-2] % 32 == 0 and xp.shape[-1] % 32 == 0
        assert net(xp)[..., :h0, :w0].shape == x.shape


def _cfg():
    return {"exclude": ["x9"],
            "object_groups": {"a": ["o1", "o2"], "b": ["o3"], "c": ["o4"], "d": ["o5", "x9"]},
            "objects_test": ["a"], "objects_val": ["b"],
            "brain_specimens": {"s1": ["b1", "b2", "b3"], "s2": ["b4", "b5"]},
            "folds": {"f": {"brain_train": ["s1"], "brain_val": ["b2"], "brain_test": ["s2"]}}}


def test_split_rules():
    s = build_split(_cfg(), "f")
    assert set(s["train"]) == {"o4", "o5", "b1", "b3"}            # excluded x9 dropped, b2 is val
    assert set(s["val"]) == {"o3", "b2"} and set(s["test"]) == {"o1", "o2", "b4", "b5"}
    bad = _cfg(); bad["folds"]["f"]["brain_val"] = ["b4"]          # val view from the TEST specimen
    with pytest.raises(ValueError):
        build_split(bad, "f")
    bad = _cfg(); bad["object_groups"]["b"].append("o1")           # one acquisition in two groups
    with pytest.raises(ValueError):
        build_split(bad, "f")
    bad = _cfg(); bad["folds"]["f"]["brain_train"] = ["s1", "s2"]  # specimen both train and test
    with pytest.raises(ValueError):
        build_split(bad, "f")


def _acq(name, n=6, H=64, W=64, sat_frame=None):
    frames = np.stack([np.full((16, H, W), 2.0 ** k) for k in range(n)]).astype(np.float16)   # frame k holds 2^k
    frames[:, 1] = np.arange(W)[None, None, :]                                               # channel 1 = column
    sat = np.zeros((n, H, W), bool)
    if sat_frame is not None:
        sat[sat_frame, :, : W // 2] = True                                                   # left half saturated
    eye = np.broadcast_to(np.eye(4)[:, :, None, None], (4, 4, H, W)).astype(np.float32).copy()
    return N2NAcq(name, frames, np.ones((H, W), bool), sat, eye, eye.copy(), np.zeros((16, H, W), np.float32),
                  False, 0)


def test_pairs_use_different_frames_and_mask_saturation():
    st = Store([_acq("a", sat_frame=2)], "cpu")
    n_checked = 0
    for k in (1, 3):
        smp = PairSampler(st, "intensity", 32, target_frames=k, min_valid=0.0, seed=1)
        for _ in range(30):
            x, y, m = smp.batch(4, 1.0)
            for b in range(4):
                i = int(round(np.log2(float(x[b, 0, 0, 0]))))
                bits = int(round(float(y[b, 0, 0, 0]) * k))        # sum of 2^j over the target frames
                js = [j for j in range(6) if bits >> j & 1]
                assert len(js) == k and i not in js                 # k distinct target frames, never the input
                cols = x[b, 1, 0].numpy()                           # image column of each patch column
                expected = np.where((cols < 32) & (2 in [i] + js), 0.0, 1.0)
                assert np.array_equal(m[b, 0, 0].numpy(), expected)
                n_checked += int(2 in [i] + js and (cols < 32).any())
    assert n_checked > 10                                           # the saturated case was actually exercised


def _synthetic(tmp_path, extra=2):
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_synthetic_data.py"), "--out", str(tmp_path),
                    "--size", "64", "96", "--extra-tissue", str(extra)], check=True, capture_output=True)
    return {"data_root": str(tmp_path / "data"), "calib_root": str(tmp_path / "calibration")}


def test_validation_score_equals_evaluation_score(tmp_path):
    paths = _synthetic(tmp_path, extra=0)
    name = "2026-01-01_120000_F_FF_HORAO_9999_001"
    cfg = {"register": True, "subtract_dark": True, "saturation": 255}
    st = Store([load_acq(name, paths, cfg, set(), validation=True)], "cpu")
    v, per, raw = validation_score(lambda X: X, st, "intensity")      # identity network = raw frame
    acq = find_acquisition(paths["data_root"], name)
    p = pl.prepare(name, acq.load_frames(), load_calibration(find_calibration(acq.path, paths["calib_root"])))
    I0, mm0, maps0 = pl.baseline_none(p)
    s = pl.score(p, I0, mm0["nM"], mm0["M11"], maps0)
    assert abs(v - s["nM_RMSE_corr"]) < 1e-6 and abs(raw[name] - s["nM_RMSE_corr"]) < 1e-6


def test_checkpoint_roundtrip_and_leak_guard(tmp_path):
    net = UNet(16, 16)
    split = {"fold": "f", "train": ["t1"], "val": ["v1"], "test": ["x1"]}
    ck = tmp_path / "best.pt"
    torch.save({"state_dict": net.state_dict(), "level": "intensity", "scale": 50.0, "width": 16, "residual": True,
                "iteration": 1, "split": split}, ck)
    m = N2NModel(ck, "cpu")
    y, dt = m.denoise(np.random.default_rng(0).uniform(0, 100, (40, 72, 16)))
    assert y.shape == (40, 72, 16) and dt >= 0
    m.check_not_trained_on("x1")
    for name in ("t1", "v1"):
        with pytest.raises(RuntimeError):
            m.check_not_trained_on(name)
