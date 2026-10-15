import json

import numpy as np

from denoising_mueller.polarimetry.reconstruction import (
    find_calibration, intensities_from_mm, load_calibration, mat_to_vec, reconstruct_mm, vec_to_mat,
)


def _calib(rng, shape=(8, 9)):
    base_A = np.array([[1, .9, .3, .2], [1, -.4, .8, .3], [1, -.3, -.6, .7], [1, .2, -.4, -.85]])
    base_W = np.array([[1, 1, 1, 1], [.85, -.45, -.35, .25], [.35, .75, -.65, -.4], [.25, .3, .55, -.8]])
    A = mat_to_vec(base_A + 0.03 * rng.standard_normal(shape + (4, 4)))
    W = mat_to_vec(base_W + 0.03 * rng.standard_normal(shape + (4, 4)))
    return A, W


def test_layout_is_column_major():
    v = np.arange(16.0)
    m = vec_to_mat(v)
    assert m[1, 0] == 1 and m[0, 1] == 4          # mat[r, c] = vec[r + 4c], as in mp_comp_MM_AIW.c
    assert np.array_equal(mat_to_vec(m), v)


def test_round_trip_recovers_mm():
    rng = np.random.default_rng(0)
    A, W = _calib(rng)
    M = mat_to_vec(np.eye(4) * 50 + rng.normal(0, 5, (8, 9, 4, 4)))
    I = intensities_from_mm(M, A, W)
    out = reconstruct_mm(I, A, W)
    assert np.allclose(out["M"], M, atol=1e-10)
    assert np.allclose(out["nM"][..., 0], 1.0)
    assert np.allclose(out["M11"], M[..., 0])


def test_dark_is_subtracted_before_inversion():
    rng = np.random.default_rng(1)
    A, W = _calib(rng)
    M = mat_to_vec(np.tile(np.diag([80., 40, 40, 40]), (8, 9, 1, 1)))
    dark = np.full((8, 9, 16), 3.0)
    I_raw = intensities_from_mm(M, A, W) + dark
    assert np.allclose(reconstruct_mm(I_raw, A, W, dark)["M"], M, atol=1e-10)
    assert not np.allclose(reconstruct_mm(I_raw, A, W, dark, subtract_dark=False)["M"], M, atol=1e-3)


def test_calibration_files_and_lookup(tmp_path):
    rng = np.random.default_rng(2)
    A, W = _calib(rng)
    for name in ("2025-06-23_090000", "2025-06-23_103720"):
        d = tmp_path / "calib" / name / "Processed_images" / "630nm"
        d.mkdir(parents=True)
        np.save(d / "A.npy", A); np.save(d / "W.npy", W); np.save(d / "630_Noise.npy", np.ones((8, 9, 16)))
    acq = tmp_path / "data" / "2025-06-23_104621_F_FF_HORAO_0001_001"
    acq.mkdir(parents=True)
    # no metadata -> most recent calibration before the acquisition
    assert find_calibration(acq, tmp_path / "calib").name == "2025-06-23_103720"
    # metadata naming an older calibration wins
    (acq / "MetaData.json").write_text(json.dumps({"Calibration Folder": r"C:\x\Calibration\2025-06-23_090000\Processed_images\630nm"}))
    path = find_calibration(acq, tmp_path / "calib")
    assert path.name == "2025-06-23_090000"
    cal = load_calibration(path)
    assert cal["A"].shape == (8, 9, 16) and cal["dark"] is not None
